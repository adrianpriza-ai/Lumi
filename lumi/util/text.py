"""Small text helpers shared by the bot and the CLI."""

from __future__ import annotations

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


def split_message(text: str, limit: int = TELEGRAM_LIMIT - _HEADROOM) -> list[str]:
    """Split *text* into chunks no longer than *limit*.

    Prefers paragraph breaks, then line breaks, then word boundaries, and only
    hard-splits as a last resort so a giant unbroken token still gets through.

    Fenced code blocks are handled explicitly, because a chunk with an odd number
    of ``` markers is either rejected by Telegram or rendered with literal
    backticks. When a cut would land inside a block, either close the block on
    this chunk or — if it is too big to close — close it artificially and reopen
    it on the next one. Either way every chunk comes out balanced.
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
            # The cut lands inside a block.
            closing = remaining.find(_FENCE, cut)
            if closing != -1 and closing + len(_FENCE) <= limit:
                cut = closing + len(_FENCE)  # let the block finish on this chunk
            else:
                # Too big to close naturally: break it, closing here and
                # reopening next chunk. Reserve room so the chunk still fits.
                suffix = f"\n{_FENCE}"
                prefix = f"{_FENCE}\n"
                cut = max(1, min(limit - len(suffix), len(remaining)))

        chunk = (remaining[:cut].rstrip() + suffix).strip()
        if chunk:
            chunks.append(chunk)
        remaining = prefix + remaining[cut:].lstrip("\n")

    if remaining.strip():
        tail = remaining.strip()
        # Balance the last chunk too, but never at the cost of exceeding the
        # limit: fitting in a Telegram message matters more than tidy fences.
        if tail.count(_FENCE) % 2 == 1 and len(tail) + len(_FENCE) + 1 <= limit:
            tail += f"\n{_FENCE}"
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


def format_error(exc: BaseException) -> str:
    """One-line, human-readable error description."""
    name = type(exc).__name__
    message = str(exc).strip() or name
    if len(message) > 300:
        message = message[:300] + "…"
    return f"{name}: {message}" if name not in message else message
