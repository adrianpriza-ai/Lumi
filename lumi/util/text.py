"""Small text helpers shared by the bot and the CLI."""

from __future__ import annotations

import re

#: Telegram rejects Bot API messages longer than this.
TELEGRAM_LIMIT = 4096
#: Leave room for a "…" marker when we cut.
_HEADROOM = 8

_FENCE = "```"


def _preferred_cut(window: str, limit: int) -> int:
    """Last good break point in *window*: paragraph, then line, then word."""
    for pattern in ("\n\n", "\n", ". ", " "):
        cut = window.rfind(pattern)
        if cut > limit // 4:
            return cut + 1
    return limit


#: Tags that must stay balanced for Telegram's HTML parser to accept a message:
#: every formatting tag except anchors (an ``<a>`` cannot be reopened across a
#: chunk without its ``href``, so links are simply not balanced). The pattern
#: matches the exact attribute-free shapes that survive :func:`sanitize_html` —
#: a ``<b onclick=…>`` is escaped into visible text and must not count here.
_BALANCE_TAG = r"(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote)"
_BALANCE_TAG_RE = re.compile(rf"<(/?)({_BALANCE_TAG})\s*(/?)>", re.IGNORECASE)


def _open_tags(text: str) -> list[str]:
    """Tags opened but not yet closed in *text*, in opening order.

    The returned stack is what a chunk boundary has to close and then reopen on
    the next chunk, so each piece parses on its own. Sloppy nesting is handled
    by popping the nearest matching opener rather than requiring perfect LIFO
    order — model output is not always tidy, and the goal is parseable, not
    semantically pure.
    """
    stack: list[str] = []
    for match in _BALANCE_TAG_RE.finditer(text):
        closing, tag, selfclose = match.groups()
        tag = tag.lower()
        if closing:
            if tag in stack:
                stack.reverse()
                stack.remove(tag)
                stack.reverse()
        elif not selfclose:
            stack.append(tag)
    return stack


def split_message(text: str, limit: int = TELEGRAM_LIMIT - _HEADROOM) -> list[str]:
    """Split *text* into chunks no longer than *limit*.

    Prefers paragraph breaks, then line breaks, then word boundaries, and only
    hard-splits as a last resort so a giant unbroken token still gets through.

    Markup is balanced across chunks, because a piece with an odd number of ```
    markers or an unclosed ``<pre>`` is either rejected by Telegram or rendered
    with literal junk. When a cut would land inside a block, either the block is
    allowed to finish on this chunk (fences), or the open tags are closed here
    and reopened on the next one. Either way every chunk parses on its own.
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        cut = _preferred_cut(remaining[:limit], limit)
        suffix = ""
        prefix = ""

        if remaining.count(_FENCE, 0, cut) % 2 == 1:
            # The cut lands inside a fenced block.
            closing = remaining.find(_FENCE, cut)
            if closing != -1 and closing + len(_FENCE) <= limit:
                cut = closing + len(_FENCE)  # let the block finish on this chunk
            else:
                # Too big to close naturally: break it, closing here and
                # reopening next chunk. Reserve room so the chunk still fits.
                suffix = f"\n{_FENCE}"
                prefix = f"{_FENCE}\n"
                cut = max(1, min(limit - len(suffix), len(remaining)))

        # HTML tags get the close-and-reopen treatment regardless of what the
        # fence logic did: a chunk that ends inside <pre> or <b> must still
        # parse. The cut shrinks to make room for the closers, but never below
        # the point where the reopened next chunk would grow instead of shrink
        # — that would loop forever. The tag stack is recomputed at the final
        # cut, because shrinking can move the boundary back across openers. If
        # even the floor leaves no room (deeply nested markup), balancing is
        # skipped and the sender's fallback degrades that chunk to plain text.
        def _closes(tags: list[str]) -> str:
            return "".join(f"</{tag}>" for tag in reversed(tags))

        def _reopens(tags: list[str]) -> str:
            return "".join(f"<{tag}>" for tag in tags)

        open_tags = _open_tags(remaining[:cut])
        if open_tags:
            room = limit - len(suffix) - len(_closes(open_tags))
            if cut > room and room >= len(_reopens(open_tags)) + 1:
                cut = min(cut, room)
                open_tags = _open_tags(remaining[:cut])
            closers = _closes(open_tags)
            if open_tags and cut + len(closers) <= limit:
                suffix += closers
                prefix = _reopens(open_tags) + prefix

        chunk = (remaining[:cut].rstrip() + suffix).strip()
        if chunk:
            chunks.append(chunk)
        remaining = prefix + remaining[cut:].lstrip("\n")

    if remaining.strip():
        tail = remaining.strip()
        # Balance the last chunk too, but never at the cost of exceeding the
        # limit: fitting in a Telegram message matters more than tidy markup.
        if tail.count(_FENCE) % 2 == 1 and len(tail) + len(_FENCE) + 1 <= limit:
            tail += f"\n{_FENCE}"
        open_tags = _open_tags(tail)
        closes = "".join(f"</{tag}>" for tag in reversed(open_tags))
        if closes and len(tail) + len(closes) <= limit:
            tail += closes
        chunks.append(tail)

    return [c for c in chunks if c]


def truncate(text: str, max_chars: int, marker: str = "\n… [truncated]") -> str:
    """Hard-cap *text*, keeping the head and telling the reader it was cut."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker


def truncate_middle(text: str, max_chars: int, marker: str = "\n…\n") -> str:
    """Cap *text* keeping both ends — better for logs and stack traces."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, (max_chars - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]


def escape_markdown(text: str) -> str:
    """Escape the legacy Markdown subset Telegram's ``Markdown`` mode chokes on."""
    out = []
    for ch in text:
        if ch in r"_*`[":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Telegram HTML
# --------------------------------------------------------------------------- #

#: The tags Telegram's HTML parse mode accepts for formatting, written as the
#: exact shapes Telegram accepts: no attributes (the one exception, ``<a
#: href="…">``, is matched separately so nothing else can ride in on it), and
#: everything else in a message — stray angle brackets included — is escaped
#: into literal text.
_HTML_TAG_RE = re.compile(
    r"(</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|tg-spoiler)\s*/?>"
    r"|<blockquote\s*/?>|</blockquote>"
    r"|<a\s+href=\"[^\"<>]*\"\s*/?>"
    r"|</a>)",
    re.IGNORECASE,
)

#: ``&`` only starts an entity when it is followed by something entity-shaped;
#: a bare ``&`` (``R&D``, ``a && b``) must become ``&amp;`` or Telegram rejects
#: the whole message. Anything already entity-shaped is left alone, so text
#: that arrives pre-escaped is not double-escaped.
_BARE_AMP_RE = re.compile(r"&(?!(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]{1,31});)")


def escape_html(text: str) -> str:
    """Escape the three characters Telegram's HTML mode reserves."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def sanitize_html(text: str) -> str:
    """Make arbitrary text safe for Telegram's HTML parse mode.

    Bot-authored strings and model output both reach the chat through this:
    the formatting tags Telegram accepts are kept exactly as written, and
    everything around them — stray angle brackets, bare ampersands, markup the
    parser would reject — is escaped into literal text. Unlike the legacy
    Markdown modes, no underscore or asterisk anywhere can break the message:
    they are not special in HTML, so dynamic content (paths, shell output,
    model prose) rides along untouched.

    The result always parses, so callers can send with ``ParseMode.HTML`` and
    treat the plain-text fallback as pure paranoia.
    """

    def _escape_gap(gap: str) -> str:
        gap = _BARE_AMP_RE.sub("&amp;", gap)
        return gap.replace("<", "&lt;").replace(">", "&gt;")

    out: list[str] = []
    last = 0
    # Walk the whitelisted tags explicitly: between them sits plain text, and
    # anything in there that looks like a tag is escaped, never preserved.
    # (Splitting on the pattern cannot do this: an unmatched segment can also
    # start with "<" and end with ">", e.g. an escaped <script> would be
    # indistinguishable from a kept tag.)
    for match in _HTML_TAG_RE.finditer(text):
        if match.start() > last:
            out.append(_escape_gap(text[last : match.start()]))
        out.append(match.group(0))
        last = match.end()
    if last < len(text):
        out.append(_escape_gap(text[last:]))
    return "".join(out)


#: The two Markdown constructs models fall back to even when told to write
#: Telegram HTML: fenced code blocks (with or without a language line) and
#: inline ``code`` spans. Matched in one pass so insertion order cannot
#: double-convert; a lone unpaired backtick matches nothing and stays literal.
_MD_CODE_RE = re.compile(
    r"```[^\n`]*\n[\s\S]*?```"  # fenced block, optional language on the first line
    r"|```[\s\S]*?```"  # fenced block with no language line
    r"|`[^`\n]+`"  # inline span
)


def markdown_code_to_html(text: str) -> str:
    """Convert Markdown code blocks and spans in *text* to Telegram HTML.

    A model coached to write ``<pre>``/``<code>`` still slips into plain
    Markdown now and then, and under an HTML parse mode a literal ``
    fence renders as backticks instead of a code block. This bridges the gap:
    fenced blocks become ``<pre>`` (language tag dropped, contents escaped),
    inline spans become ``<code>``. Everything else is left for the caller's
    sanitizer, so prose formatting the model should not have used is simply
    shown as written rather than guessed at.
    """

    def _convert(match: re.Match[str]) -> str:
        snippet = match.group(0)
        if snippet.startswith("```"):
            body = snippet[3:-3]
            if "\n" in body:
                body = body.split("\n", 1)[1]  # drop the language line
            return f"<pre>{escape_html(body.strip(chr(10)))}</pre>"
        return f"<code>{escape_html(snippet[1:-1])}</code>"

    return _MD_CODE_RE.sub(_convert, text)


def format_error(exc: BaseException) -> str:
    """One-line, human-readable error description."""
    name = type(exc).__name__
    message = str(exc).strip() or name
    if len(message) > 300:
        message = message[:300] + "…"
    return f"{name}: {message}" if name not in message else message
