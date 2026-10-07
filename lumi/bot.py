"""The Telegram layer.

Responsibilities stop at the edges: authenticate the sender, translate a message
into one :meth:`lumi.agent.Agent.handle` call, translate the result back into
Telegram (splitting long text, streaming a typing indicator, drawing the
Confirm/Cancel keyboard). All the intelligence lives in the agent, which is why
the same agent also drives ``lumi chat`` in a terminal.

Owner gating is applied twice on purpose: as a handler filter so strangers cost
nothing, and as an explicit check in each privileged handler so a routing change
cannot silently expose the shell.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from telegram import Chat, InlineKeyboardButton, InlineKeyboardMarkup, MessageEntity, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import Agent, PendingAction, TurnResult
from .artifacts import Artifact, ArtifactError, ArtifactStore
from .config import Config, ConfigError, validate
from .doctor import platform_line, run_checks
from .memory import History, MemoryFile
from .personality import Personality
from .util.log import get_logger
from .util.text import (
    escape_html,
    format_error,
    markdown_code_to_html,
    sanitize_html,
    split_message,
    truncate,
)

log = get_logger(__name__)

#: Callback data prefix; keeps our buttons from colliding with anything else.
CB_OK = "lumi:ok:"
CB_NO = "lumi:no:"
#: Second segment is "show" or "hide"; the rest is a token from :class:`TraceStore`.
CB_THINK = "lumi:think:"

TYPING_INTERVAL = 4.0

#: Pool size for the getUpdates request object. PTB sizes it at 1 connection;
#: that is fine except with a non-empty bot.proxy_url, where httpx with
#: ``max_connections=1`` is documented to misbehave (PTB wiki: "Working with
#: proxies"). One extra connection is harmless without a proxy.
GET_UPDATES_POOL_SIZE = 2

#: How long a collapsed thinking trace stays expandable. Long enough to come
#: back to a message you scrolled past, short enough that scrolling back through
#: old turns does not resurrect a model's entire reasoning.
TRACE_TTL = 30 * 60.0
#: Ceiling on stored traces, so a busy day cannot grow without bound.
TRACE_CACHE_MAX = 64

# All bot-authored strings are Telegram HTML: <b>…</b> for headers, <code>…</code>
# around commands and paths, <i>…</i> for asides. HTML only reserves < > & —
# and every dynamic value goes through escape_html/sanitize_html — so a path
# with underscores or a model name with asterisks can never break a message the
# way the legacy Markdown modes did.
HELP_TEXT = """\
<b>what i can do</b>
just talk to me. i remember you between sessions, look things up on the web,
read and write files in this project, and run shell commands.

send me a photo and i'll look at it. add a caption to tell me what to look for.
send me a document and i'll read it — it lands in workspace/uploads/ and i can
work with it from there. ask me to make a file and i'll send it back as one.

if i'm running on a reasoning model i'll think first, then say a small grey line
above my answer. tap <i>show thinking</i> if you want to see how i got there.

<b>slash commands</b>
<code>/help</code> — this list
<code>/ask &lt;question&gt;</code> — same as just typing
<code>/run &lt;command&gt;</code> — run a shell command directly
<code>/search &lt;query&gt;</code> — web search
<code>/fetch &lt;url&gt;</code> — read a page
<code>/reasoning</code> — reasoning setup, or <code>on</code>/<code>off</code> to show or hide traces
<code>/memory</code> — show what i remember
<code>/remember &lt;fact&gt;</code> — save a fact
<code>/forget [n]</code> — drop the last n saved facts
<code>/personality</code> — show the personality file
<code>/tools</code> — list my tools
<code>/context</code> — what i'm holding in my context window right now
<code>/status</code> — model, tools and provider health
<code>/doctor</code> — diagnose the whole setup
<code>/config</code> — the resolved configuration this process runs on (owner only)
<code>/env</code> — which credentials are set, values masked (owner only)
<code>/reload</code> — re-read personality, memory and config
<code>/reset</code> — forget this conversation (keeps long-term memory)
<code>/whitelist</code> — show whitelisted users and groups (owner only)
<code>/whitelist_add_user &lt;id&gt;</code> — add a user to the whitelist (owner only)
<code>/whitelist_remove_user &lt;id&gt;</code> — remove a user from the whitelist (owner only)
<code>/whitelist_add_group &lt;id&gt;</code> — add a group to the whitelist (owner only)
<code>/whitelist_remove_group &lt;id&gt;</code> — remove a group from the whitelist (owner only)

dangerous commands ask for confirmation first. i can't delete that, on purpose.
"""

NOT_AUTHORISED = "this bot is private. your id is not whitelisted."
NOT_OWNER = "only the owner can manage the whitelist."
NOT_OWNER_SENSITIVE = "only the owner can see that."


def _mask(value: str | None) -> str:
    """A credential safe to show in chat.

    Stricter than :func:`lumi.llm.keypool.mask`, which is tuned for log lines:
    only the first two and last four characters survive, and anything shorter
    than 12 characters is described by length instead. Full values are never
    sent to Telegram — a chat lives on Telegram's servers and on every device
    logged into the account, so even an owner-only command keeps its secrets.
    """
    if not value:
        return ""
    if len(value) < 12:
        return f"(len {len(value)})"
    return f"{value[:2]}…{value[-4:]}"


def _render_env(config: Config) -> str:
    """The ``/env`` body: which credentials exist, masked, never in full."""
    lines = [
        "<b>env</b> — credential status (masked)",
        "<i>full values are never sent to this chat.</i>",
        "",
    ]

    token = config.telegram_token
    lines.append(
        f"<code>TELEGRAM_BOT_TOKEN</code> = <code>{_mask(token)}</code>"
        if token
        else "<code>TELEGRAM_BOT_TOKEN</code> — unset (the bot cannot start)"
    )

    # One line per key-bearing variable, in web tier order for the providers.
    # unset is a normal state for tavily (keyless tier) and context7 (hidden
    # until set), so the role says what an unset means rather than alarming.
    specs = (
        (
            config.llm.api_key_env,
            config.llm.api_keys(),
            config.llm.strategy_of(),
            "the model endpoint",
        ),
        (
            config.tools.web.firecrawl.api_key_env,
            config.tools.web.firecrawl.api_keys(),
            config.tools.web.firecrawl.key_strategy_of(),
            "web tier 1: firecrawl",
        ),
        (
            config.tools.web.exa.api_key_env,
            config.tools.web.exa.api_keys(),
            config.tools.web.exa.key_strategy_of(),
            "web tier 2: exa",
        ),
        (
            config.tools.web.tavily.api_key_env,
            config.tools.web.tavily.api_keys(),
            config.tools.web.tavily.key_strategy_of(),
            "web tier 3: tavily (free keyless tier when unset)",
        ),
        (
            config.tools.context7.api_key_env,
            config.tools.context7.api_keys(),
            config.tools.context7.key_strategy_of(),
            "context7 (hidden from the model until set)",
        ),
    )
    for name, keys, strategy, role in specs:
        name = escape_html(name)
        if keys:
            pool = f" ({len(keys)} keys, {escape_html(strategy)})" if len(keys) > 1 else ""
            lines.append(f"<code>{name}</code> = <code>{_mask(keys[0])}</code>{pool} — {role}")
        else:
            lines.append(f"<code>{name}</code> — unset — {role}")

    overrides = sorted(key for key in os.environ if key.startswith("LUMI__"))
    if overrides:
        lines += ["", "<b>LUMI__ overrides:</b>"]
        for name in overrides:
            lines.append(
                f"- <code>{escape_html(name)}={escape_html(truncate(os.environ[name], 60))}</code>"
            )

    lines += [
        "",
        "<i>keys are read at startup: edit .env, then restart. /reload does not re-read credentials.</i>",
    ]
    return "\n".join(lines)


def _render_config(config: Config, memory_limit: int) -> str:
    """The ``/config`` body: the resolved configuration this process runs on.

    Same values ``lumi config`` prints in a terminal, shaped for chat. Reads
    only the config tree — no bot_data — so it stays a pure function and the
    live view is exactly what the running handlers closed over.

    *memory_limit* is ``Agent.memory_limit()`` — the cap actually applied to
    the system prompt — rather than a re-derivation, so the display cannot
    drift from behaviour.
    """
    llm = config.llm
    web = config.tools.web
    shell = config.tools.shell
    files = config.tools.files
    bot = config.bot
    temperature = "provider default" if llm.temperature is None else str(llm.temperature)
    # Every one of these reports "provider default" or "no cap" at 0, because
    # 0 is the state most of them ship in. Memory is the exception: 0 means the
    # file is bounded by its share of the window rather than sent whole.
    ceiling = f"{llm.max_tokens:,}" if llm.max_tokens > 0 else "provider default (no cap sent)"
    memory_cap = f"{memory_limit:,} chars"
    if llm.max_memory_chars <= 0:
        memory_cap += " (an eighth of the window)"
    web_content = f"{web.max_content_chars:,} chars" if web.max_content_chars > 0 else "uncapped"
    shell_output = f"{shell.max_output_chars:,} chars" if shell.max_output_chars > 0 else "uncapped"
    return "\n".join(
        [
            "<b>config</b> — resolved, live",
            "",
            "<b>model</b>",
            f"- <code>{escape_html(llm.model_of())}</code> via {escape_html(llm.base_url_of())} (from {escape_html(llm.where_from())})"
            + (
                f"\n- vision: <code>{escape_html(llm.vision_model_of())}</code> for image turns"
                if llm.vision_model_of()
                else "\n- vision: unset — photos go to the default model"
            ),
            f"- temperature: {escape_html(temperature)} · max_tokens: {escape_html(ceiling)} · keys: {escape_html(llm.strategy_of())}",
            f"- reasoning: {'on at ' + escape_html(llm.effort_of() or 'provider default') if llm.reasoning else 'off'}"
            f" · trace in chat: {'on' if llm.show_reasoning else 'off'}",
            f"- context: {llm.context_window:,} token window, {llm.context_headroom:,} held back"
            f" · keep {llm.context_keep_recent} recent"
            f" · compaction {'on' if llm.compaction else 'off'}"
            f" · {llm.max_conversations} chats in memory",
            f"- agent: {llm.max_tool_iterations} tool iterations"
            f" · history: {'as much as fits' if not llm.history_turns else f'at least {llm.history_turns} turns'}"
            f" · memory {memory_cap}",
            "",            f"<b>web</b> — {'enabled' if web.enabled else 'DISABLED'}",
            f"- order: {' → '.join(escape_html(p) for p in web.provider_order)}",
            f"- results: {web.max_results or 'unlimited'} per search · floor {web.min_results}",
            f"- firecrawl: scrape top {web.firecrawl.auto_scrape_top_n} · {escape_html(web.firecrawl.key_strategy_of())}",
            f"- exa: type {escape_html(web.exa.search_type)}"
            + (f", category {escape_html(web.exa.category)}" if web.exa.category else ""),
            f"- tavily: depth {escape_html(web.tavily.search_depth)}, topic {escape_html(web.tavily.topic)}",
            f"- mcp: {escape_html(str(web.mcp.config_file))} · timeout {web.mcp.timeout_seconds}s",
            f"- call timeout {web.timeout_seconds}s · page content {web_content}",
            "",
            f"<b>shell</b> — cwd {escape_html(str(shell.cwd))} · timeout {shell.timeout_seconds}s"
            f" · output {shell_output}",
            f"- safety: {'asks before risky' if shell.ask_before_risky else 'hard block, no prompts'}"
            f" · home: {escape_html(str(shell.home))} · writes: {escape_html(', '.join(shell.writable)) or 'whole project'}",
            "",
            "<b>files</b>",
            f"- writable: {escape_html(', '.join(files.writable)) or 'none'}"
            f" · read {'no cap' if files.max_read_chars <= 0 else f'{files.max_read_chars:,}'}"
            f" · write cap {files.max_write_chars:,}",
            "",
            "<b>bot</b>",
            f"- group mode: {escape_html(bot.group_reply_mode)} · whitelisted users: {len(bot.whitelisted_users)}"
            f" · groups: {len(bot.whitelisted_groups)}",
            f"- file delivery: documents up to {bot.max_upload_mb} MB"
            f" · files tool: {'uploads on' if files.uploads else 'uploads off'}",
            f"- logging: {escape_html(config.logging.level)} → {escape_html(str(config.logging.file))}",
            f"- config file: {escape_html(str(config.config_path)) if config.config_path else 'none (built-in defaults)'}",
        ]
    )


# --------------------------------------------------------------------------- #
# Group / private chat routing
# --------------------------------------------------------------------------- #


def _should_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, config: Config) -> bool:
    """Whether the bot should respond to this update.

    Private chats always respond — same as a normal 1:1 chat. Group and
    supergroup chats follow ``bot.group_reply_mode``:

    - ``"mention"`` (default): only respond when the bot is mentioned
      (``@Lumi_a_bot``), replied to, or sent a slash command targeting this bot
      (``/help@Lumi_a_bot``). Plain ``/help`` in a group is left alone because
      Telegram routes commands without ``@botname`` to whichever bot claims
      them first.
    - ``"always"``: respond to every owner message regardless of chat type.
    - ``"off"``: never respond in groups; private chats are unaffected.

    ``bot.always_reply_chats`` is consulted first and overrides the mode, so a
    dedicated "Lumi lab" group can opt in by id without flipping the global
    setting.
    """
    chat = update.effective_chat
    if chat is None:
        return True
    if str(chat.id) in config.bot.always_reply_chats:
        return True
    if str(chat.id) in config.bot.whitelisted_groups:
        return True
    if chat.type == Chat.PRIVATE:
        return True
    if chat.type not in (Chat.GROUP, Chat.SUPERGROUP):
        # Channels are a corner case; stay quiet by default — there is no
        # good way to address a bot in a channel without an inline query.
        return False

    mode = (config.bot.group_reply_mode or "mention").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    return _is_addressed(update, context)


def _is_addressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Whether this message is aimed at *this* bot in a group context.

    Three positive signals, any one of which is enough:

    1. The text starts with a slash command that names this bot
       (e.g. ``/help@Lumi_a_bot``). Plain ``/help`` is *not* enough — Telegram
       routes that to whichever bot owns the command, so acting on it would
       be presumptuous.
    2. The message is a reply to a message sent by this bot.
    3. The message text contains a ``@Lumi_a_bot`` mention, either as a plain
       ``mention`` entity or as a ``text_mention`` entity pointing at the bot.
    """
    message = update.effective_message
    if message is None:
        return False
    bot_username = (context.bot.username or "").lower()
    if not bot_username:
        # The bot has no username (rare) — fall back to "always reply" rather
        # than silently swallowing everything.
        return True

    text = message.text or message.caption or ""

    # 1. Slash command targeting this bot.
    if text.startswith("/"):
        cmd = text.split(maxsplit=1)[0]
        if "@" in cmd:
            target = cmd.split("@", 1)[1].lower().rstrip(",.;:!?")
            if target == bot_username:
                return True
        return False  # command in a group with no @botname: assume different bot

    # 2. Reply to a message from this bot.
    reply = message.reply_to_message
    if (
        reply is not None
        and reply.from_user is not None
        and reply.from_user.is_bot
        and (reply.from_user.username or "").lower() == bot_username
    ):
        return True

    # 3. Mention entity anywhere in the message (text or caption).
    entities = list(message.entities or []) + list(message.caption_entities or [])
    for entity in entities:
        if entity.type == MessageEntity.MENTION:
            snippet = text[entity.offset : entity.offset + entity.length].lower().lstrip("@")
            if snippet == bot_username:
                return True
        elif entity.type == MessageEntity.TEXT_MENTION and entity.user is not None:
            if (entity.user.username or "").lower() == bot_username:
                return True

    return False


# --------------------------------------------------------------------------- #
# Outbound helpers
# --------------------------------------------------------------------------- #


async def reply(update: Update, text: str, **kwargs: Any) -> None:
    """Send *text*, splitting it and degrading gracefully on bad HTML.

    *text* is arbitrary — model prose, shell output, web results — so it goes
    through :func:`sanitize_html` first: Telegram's formatting tags survive,
    everything else (stray angle brackets, a bare ``&`` in ``R&D``) is escaped
    into literal text. That is the whole reason the bot speaks HTML rather than
    legacy Markdown: an underscore in a filename cannot fail the parse, and the
    fallback below is paranoia rather than a load-bearing path.
    """
    message = update.effective_message
    if message is None or not text.strip():
        return
    # Models are coached to write Telegram HTML, but they slip into Markdown
    # code fences and spans out of habit. Bridge those to HTML first; the
    # sanitizer then keeps Telegram's tags and escapes everything else, so the
    # send cannot fail on stray angle brackets or a bare ampersand.
    for chunk in split_message(markdown_code_to_html(text)):
        try:
            await message.reply_text(
                sanitize_html(chunk),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                **kwargs,
            )
        except BadRequest:
            # Unreachable when the text is sanitised, kept as a safety net:
            # send it raw and, failing that, stripped of all markup.
            try:
                await message.reply_text(chunk, disable_web_page_preview=True, **kwargs)
            except TelegramError as exc:
                log.warning("could not deliver a chunk: %s", exc)
        except TelegramError as exc:
            log.warning("could not deliver a chunk: %s", exc)


async def reply_html(update: Update, text: str, **kwargs: Any) -> None:
    """Send our own UI strings, where the HTML tags are ours and therefore exact.

    Unlike :func:`reply` this does not sanitise — the markup was written by
    hand — but the fallback still escapes everything, so a caller bug degrades
    to a plain message instead of a silently dropped one.
    """
    message = update.effective_message
    if message is None:
        return
    for chunk in split_message(text):
        try:
            await message.reply_text(
                chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True, **kwargs
            )
        except (BadRequest, TelegramError) as exc:
            log.warning("could not deliver a chunk: %s", exc)
            with contextlib.suppress(TelegramError):
                await message.reply_text(
                    escape_html(chunk), disable_web_page_preview=True, **kwargs
                )


class _Typing:
    """Keeps the typing indicator alive for as long as the agent is working."""

    def __init__(self, bot: Any, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _Typing:
        # Send the first action inline rather than waiting for the task to be
        # scheduled: a fast turn would otherwise finish before the indicator
        # ever appeared.
        await self._send()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            # Best effort: swallow the cancellation and anything the loop raised,
            # because this runs in a finally-ish path.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _send(self) -> None:
        """Best effort: a failed chat action must never break the turn."""
        with contextlib.suppress(Forbidden, TelegramError, asyncio.CancelledError):
            await self._bot.send_chat_action(self._chat_id, ChatAction.TYPING)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=TYPING_INTERVAL)
            except TimeoutError:
                await self._send()


# --------------------------------------------------------------------------- #
# Approval UI
# --------------------------------------------------------------------------- #


def approval_keyboard(action: PendingAction) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("run it", callback_data=CB_OK + action.id),
                InlineKeyboardButton("cancel", callback_data=CB_NO + action.id),
            ]
        ]
    )


def render_approval(action: PendingAction) -> str:
    detail = truncate(action.preview, 1200, marker="\n…")
    return (
        f"<b>needs your approval</b>\n"
        f"<code>{escape_html(action.tool)}</code> — {escape_html(action.reason)}\n\n"
        f"<pre>{escape_html(detail)}</pre>"
    )


# --------------------------------------------------------------------------- #
# Thinking traces
# --------------------------------------------------------------------------- #


class TraceStore:
    """Thinking traces waiting behind a collapsed message.

    Telegram has no way to hide text, so the collapsed line carries only a
    summary and the trace itself lives here, keyed by a short token that fits in
    the button's callback data. That is the whole reason this exists: a model
    can think for a long time and produce more than fits in one message, and
    none of it should sit in the chat until someone asks for it.

    Evicted by age and by count, so a tap that never comes cannot keep a
    conversation's reasoning alive for the life of the process.
    """

    def __init__(self, ttl: float = TRACE_TTL, limit: int = TRACE_CACHE_MAX) -> None:
        self._ttl = ttl
        self._limit = limit
        self._traces: dict[str, tuple[float, str]] = {}

    def put(self, trace: str) -> str:
        """Store *trace* and return the token that retrieves it."""
        self._prune()
        token = uuid.uuid4().hex[:12]
        self._traces[token] = (time.monotonic(), trace)
        while len(self._traces) > self._limit:
            self._traces.pop(next(iter(self._traces)))
        return token

    def get(self, token: str) -> str | None:
        entry = self._traces.get(token)
        if entry is None:
            return None
        stored_at, trace = entry
        if time.monotonic() - stored_at > self._ttl:
            self._traces.pop(token, None)
            return None
        return trace

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for token in [t for t, (at, _) in self._traces.items() if at < cutoff]:
            self._traces.pop(token, None)

    def __len__(self) -> int:
        return len(self._traces)


def thinking_stub(result: TurnResult) -> str:
    """The collapsed one-liner shown above an answer.

    Deliberately says how long the model thought rather than what about, so
    reading the chat stays cheap and the button is the only way in. The duration
    is the model's own time, not the turn's: on a turn that spent most of itself
    waiting on a web search, "thought for 20s" would be a lie.
    """
    seconds = result.thinking_seconds or result.elapsed
    parts = [f"thought for {seconds:.0f}s"] if seconds else []
    if result.reasoning_tokens:
        parts.append(f"{result.reasoning_tokens:,} reasoning tokens")
    parts.append(f"{len(result.reasoning):,} chars")
    return "<i>🧠 " + " · ".join(parts) + "</i>"


def thinking_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("show thinking", callback_data=f"{CB_THINK}show:{token}")]]
    )


def thinking_keyboard_hide() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("hide", callback_data=f"{CB_THINK}hide:")]]
    )


async def _toast(query: Any, text: str = "", **kwargs: Any) -> None:
    """Answer a callback query, suppressing errors from stale queries.

    Telegram expires callback queries quickly, and PTB processes updates
    sequentially, so a tap that queues behind a long agent turn can easily be
    answered late. ``BadRequest`` from an expired query must not propagate —
    the follow-up action (edit, resolve) does not depend on the toast.
    """
    with contextlib.suppress(BadRequest, TelegramError):
        await query.answer(text, **kwargs)


async def _edit_thinking(
    query: Any, text: str, *, keyboard: InlineKeyboardMarkup | None = None
) -> None:
    """Replace the collapsed stub in place, degrading on bad HTML.

    Same contract as :func:`reply`: our own markup first, escaped text if the
    Bot API will not take it, and silence if even that fails — a failed edit
    must never surface as an error to someone who just wanted to read a reply.
    """
    for kwargs in (
        {"parse_mode": ParseMode.HTML, "reply_markup": keyboard},
        {"reply_markup": keyboard},
    ):
        try:
            await query.edit_message_text(text, **kwargs)
            return
        except BadRequest:
            continue
        except TelegramError as exc:
            log.warning("could not edit the thinking message: %s", exc)
            return
    log.warning("the thinking message was rejected as HTML and as plain text")


def render_thinking(trace: str) -> str:
    """The expanded trace, wrapped in ``<pre>`` so its shape survives.

    The trace is model output, so it is escaped before wrapping: a stray ``<``
    in a thought must become visible text, not a parse error or a swallowed
    tag. Truncated to what one message can hold — an expanded trace is a
    curiosity, and a 40,000-character wall is not, so the rest is dropped
    rather than split across a dozen messages nobody asked for.
    """
    return f"🧠 <b>thinking</b>\n\n<pre>{escape_html(truncate(trace.strip(), 3600))}</pre>"


# --------------------------------------------------------------------------- #
# Artifacts (file delivery)
# --------------------------------------------------------------------------- #


async def send_artifact(update: Update, artifact: Artifact) -> None:
    """Send one artifact as a document, degrading to a path when it fails.

    The caption is deliberately plain text (no parse mode): filenames with
    underscores would fight legacy Markdown parsing, and the file is the
    payload — the caption only has to identify it.
    """
    message = update.effective_message
    if message is None:
        return
    try:
        with artifact.absolute.open("rb") as handle:
            await message.reply_document(
                document=handle,
                filename=Path(artifact.path).name,
                caption=artifact.caption(),
            )
    except (BadRequest, TelegramError, OSError) as exc:
        log.warning("could not send artifact %s: %s", artifact.path, exc)
        await reply(
            update,
            f"📎 couldn't attach {escape_html(artifact.path)} ({artifact.size} bytes) — "
            f"it's on disk at <code>{escape_html(artifact.path)}</code>",
        )


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #


async def deliver(
    update: Update, result: TurnResult, config: Config, traces: TraceStore | None = None
) -> None:
    """Turn an agent result into messages, documents included."""
    if result.error and not result.text:
        await reply(update, f"that didn't work: {escape_html(result.error)}")
        # A crashed turn can still have staged files (e.g. the model wrote with
        # upload: true and a later iteration hit the cap). Deliver what exists.
        for artifact in result.artifacts:
            await send_artifact(update, artifact)
        return

    # Before the answer, because that is the order it happened in: the model
    # thought, then it spoke. The line stays collapsed either way, so the answer
    # below is still what you read.
    if result.has_reasoning and config.llm.show_reasoning and traces is not None:
        token = traces.put(result.reasoning)
        await reply(update, thinking_stub(result), reply_markup=thinking_keyboard(token))

    if result.text:
        await reply(update, result.text)

    # Deliver artifacts after the text — the chat reads the explanation first,
    # then gets the file.
    for artifact in result.artifacts:
        await send_artifact(update, artifact)

    for action in result.pending:
        await reply_html(update, render_approval(action), reply_markup=approval_keyboard(action))

    if result.error and result.text:
        note = f"<i>note: {escape_html(result.error)}</i>"
        if result.tools_used:
            note += (
                f"\n<i>tools used: {escape_html(', '.join(dict.fromkeys(result.tools_used)))}</i>"
            )
        await reply(update, note)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _is_allowed(user: Any, config: Config) -> bool:
    """Whether *user* is authorised to use the bot (owner or whitelisted)."""
    owner = config.owner_id
    if owner is not None and user.id == owner:
        return True
    return str(user.id) in config.bot.whitelisted_users


def _is_owner(user: Any, config: Config) -> bool:
    """Whether *user* is the owner (not just whitelisted)."""
    owner = config.owner_id
    return owner is not None and user.id == owner


def _toml_value(value: Any) -> str:
    """Serialize a value to TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return f'"{value}"'


def _write_toml(f: Any, data: dict[str, Any], prefix: str = "") -> None:
    """Write a dict as TOML."""
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        f.write(f"{key} = {_toml_value(value)}\n")
    for key, value in data.items():
        if isinstance(value, dict):
            section = f"{prefix}{key}"
            f.write(f"\n[{section}]\n")
            _write_toml(f, value, prefix=f"{section}.")


def build_handlers(config: Config) -> list[Any]:
    class OwnerFilter(filters.UpdateFilter):
        """Filter that checks the live config for authorisation.

        Must subclass :class:`telegram.ext.filters.UpdateFilter`, not
        ``BaseFilter``: ``BaseFilter.check_update`` only tests that the update
        *contains a message* and never calls ``filter()``, which would silently
        turn this gate into a no-op (every handler re-checks internally, so the
        damage is that ``on_stranger`` never fires and strangers walk into each
        handler before being rejected there).
        """

        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg

        def filter(self, update: Update) -> bool:
            user = update.effective_user
            return user is not None and _is_allowed(user, self.cfg)

    owner_filter = OwnerFilter(config)

    async def authorised(update: Update) -> bool:
        user = update.effective_user
        if user is not None and _is_allowed(user, config):
            return True
        await reply(update, NOT_AUTHORISED)
        log.warning("rejected %s from user %s", update.effective_message and "message", user and user.id)
        return False

    def gated(handler):
        """Apply the group/mention gating to a command handler.

        Each command is wrapped so a group-chat message that is not addressed
        to the bot is dropped before the handler runs. Private chats and
        whitelisted chats bypass the gate.
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not await authorised(update):
                return
            if not _should_reply(update, context, config):
                return
            await handler(update, context)

        return wrapper

    def agent_of(context: ContextTypes.DEFAULT_TYPE) -> Agent:
        agent: Agent = context.application.bot_data["agent"]
        return agent

    def chat_of(update: Update) -> int | str:
        chat = update.effective_chat
        return chat.id if chat else 0

    def traces_of(context: ContextTypes.DEFAULT_TYPE) -> TraceStore:
        """The store behind the collapsed thinking messages.

        Read from bot_data rather than captured, so a test can build an
        application without a real one and still exercise the whole path.
        """
        store: TraceStore = context.application.bot_data["traces"]
        return store

    def artifacts_of(context: ContextTypes.DEFAULT_TYPE) -> ArtifactStore:
        """The store that stages incoming document uploads."""
        store: ArtifactStore = context.application.bot_data["artifacts"]
        return store

    def history_of(context: ContextTypes.DEFAULT_TYPE) -> History:
        """The transcript store, for ``/context``."""
        store: History = context.application.bot_data["history"]
        return store

    # -- basic commands ---------------------------------------------------- #

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        name = escape_html(user.first_name) if user else "there"
        log.info("start from %s (%s)", name, user and user.id)
        await reply(
            update,
            f"hey {name}. i'm up.\n\n{HELP_TEXT}\n\n"
            f"model: <code>{escape_html(config.llm.model_of())}</code> via {escape_html(config.llm.base_url_of())}\n"
            f"tools: {escape_html(', '.join(agent_of(context).registry.names()) or 'none')}",
        )

    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        await reply(update, HELP_TEXT)

    async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        text = " ".join(context.args).strip() if context.args else ""
        if not text:
            await reply(update, "usage: <code>/ask &lt;question&gt;</code>")
            return
        await run_agent(update, context, text)

    # -- direct tool invocations ------------------------------------------- #

    async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        command = " ".join(context.args).strip() if context.args else ""
        if not command:
            await reply(
                update,
                "usage: <code>/run &lt;command&gt;</code>\ne.g. <code>/run git status --short</code>",
            )
            return
        await run_tool(update, context, "run_shell", {"command": command})

    async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        query = " ".join(context.args).strip() if context.args else ""
        if not query:
            await reply(update, "usage: <code>/search &lt;query&gt;</code>")
            return
        await run_tool(update, context, "web", {"action": "search", "query": query})

    async def fetch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        url = " ".join(context.args).strip() if context.args else ""
        if not url:
            await reply(update, "usage: <code>/fetch &lt;url&gt;</code>")
            return
        await run_tool(update, context, "web", {"action": "fetch", "url": url})

    # -- memory ------------------------------------------------------------ #

    async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        memory: MemoryFile = context.application.bot_data["memory"]
        memory.ensure_loaded()
        stats = memory.stats()
        # The chat view is a window onto the file, so it shows what the model
        # actually gets — including the part that did not fit, which is the one
        # worth knowing about.
        limit = agent_of(context).memory_limit()
        size = memory.prompt_size(limit)
        body = memory.for_prompt(limit)
        lines = [
            f"<b>memory</b> — {stats['managed_count']} saved fact(s), {stats['chars']} chars",
            f"<code>{escape_html(stats['path'])}</code>",
        ]
        if size["over"]:
            lines.append(
                f"showing {size['sent_chars']:,} of {size['chars']:,} chars "
                f"(budget {limit:,}); the oldest {size['dropped']} fact(s) are on disk "
                f"but not being sent — raise <code>llm.max_memory_chars</code>"
            )
        lines += [
            "",
            sanitize_html(body),
            "",
            "add one with <code>/remember &lt;fact&gt;</code>, drop the newest with <code>/forget</code>.",
        ]
        await reply(update, "\n".join(lines))

    async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        fact = " ".join(context.args).strip() if context.args else ""
        if not fact:
            await reply(update, "usage: `/remember <fact>`")
            return
        memory: MemoryFile = context.application.bot_data["memory"]
        await reply(update, memory.remember(fact, source="chat"))

    async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        count = 1
        if context.args:
            try:
                count = max(1, int(context.args[0]))
            except ValueError:
                await reply(
                    update, "usage: <code>/forget [count]</code> — count has to be a number"
                )
                return
        memory: MemoryFile = context.application.bot_data["memory"]
        removed = memory.forget(count)
        if not removed:
            await reply(update, "nothing to forget — there are no saved facts.")
            return
        listed = "\n".join(f"- {escape_html(item)}" for item in removed)
        await reply(update, f"forgot {len(removed)}:\n{listed}")

    # -- introspection ----------------------------------------------------- #

    async def personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        personality: Personality = context.application.bot_data["personality"]
        await reply(
            update,
            f"<b>PERSONALITY.md</b> — {len(personality.text)} chars\n\n"
            + sanitize_html(personality.text),
        )

    async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        registry = agent_of(context).registry
        await reply(update, f"<b>tools</b>\n\n{registry.describe(html=True)}")

    async def reasoning_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Report the reasoning setup, and toggle the visible trace.

        ``/reasoning`` with no argument only reports. ``on`` and ``off`` flip
        ``llm.show_reasoning`` for this process, which is the same scope as
        every other config change here: a restart restores config.toml, and that
        is the right time to make it stick.
        """
        if not await authorised(update):
            return
        llm = config.llm
        argument = (context.args[0].strip().lower() if context.args else "")

        if argument in ("on", "off"):
            llm.show_reasoning = argument == "on"
            log.info("thinking traces turned %s by chat %s", argument, chat_of(update))
        elif argument:
            await reply(update, "usage: <code>/reasoning</code> or <code>/reasoning on|off</code>")
            return

        effort = llm.effort_of() or "provider default"
        budget = (
            f"{llm.reasoning_tokens:,} tokens on top of {llm.max_tokens:,}"
            if llm.max_tokens > 0
            else "none — no output ceiling is sent, the provider decides"
        )
        await reply(
            update,
            "<b>reasoning</b>\n"
            f"model mode: {'on' if llm.reasoning else 'off'}\n"
            f"effort: {escape_html(effort)}\n"
            f"thinking budget: {escape_html(budget)}\n"
            f"temperature: {'omitted' if llm.reasoning and llm.temperature is not None else llm.temperature}\n"
            f"trace in chat: {'on' if llm.show_reasoning else 'off'}",
        )

    async def context_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """What the model is actually holding for this chat right now.

        A bot that forgets is hard to argue with, so the state behind that is
        inspectable: the window, the budget, how much of it is in use, and what
        has been condensed to make room. The record itself is quoted when there
        is one, because that text is now the only version of the old
        conversation the model has.
        """
        if not await authorised(update):
            return
        agent = agent_of(context)
        chat_id = chat_of(update)
        report = agent.context_report(chat_id)
        lines = ["<b>context</b>", *report.lines()]

        conv = agent.conversation_state(chat_id)
        if conv is not None and conv.summary:
            lines += ["", "<b>condensed record</b>", sanitize_html(conv.summary)]
        transcript = history_of(context).stats(chat_id)
        lines += [
            "",
            f"transcript: {truncate(str(transcript['path']), 60, '…')}"
            f" · {transcript['bytes']:,} bytes"
            + (" (replay reads the tail only)" if transcript["truncated"] else ""),
            f"in memory: {agent.conversations_in_memory()} chat(s),"
            f" each bounded by its {report.window:,}-token window",
        ]
        await reply(update, "\n".join(lines))

    async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        agent = agent_of(context)
        uptime = time.monotonic() - STARTED
        lines = [
            "<b>status</b>",
            f"uptime: {uptime:.0f}s",
            f"model: <code>{escape_html(config.llm.model_of())}</code>",
        ]
        vision = config.llm.vision_model_of()
        if vision:
            lines.append(
                f"vision model: <code>{escape_html(vision)}</code> (for photos)"
            )
        lines += [
            f"endpoint: {escape_html(config.llm.base_url_of())} (from {escape_html(config.llm.where_from())})",
            f"reasoning: {'on' if config.llm.reasoning else 'off'}"
            + (f" at {escape_html(config.llm.effort_of())}" if config.llm.reasoning else ""),
            f"tools: {escape_html(', '.join(agent.registry.names())) or 'none'}",
        ]
        for tool in agent.registry.all():
            ok, reason = tool.available()
            if not ok:
                lines.append(f"- <code>{tool.name}</code>: {escape_html(reason)}")
        await reply(update, "\n".join(lines))

    async def doctor_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        checks = run_checks(config, agent_of(context).registry)
        await reply(update, "<b>doctor</b>\n" + sanitize_html("\n".join(check.render() for check in checks)))

    async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Owner-only view of the resolved configuration.

        Shows what this process is actually running with — config.toml merged
        with config.local.toml and LUMI__ env overrides. Nothing here is secret
        (keys are ``/env``'s job), but it is owner-only anyway: a whitelisted
        user in a shared group does not need to see the endpoint or the layout.
        """
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_SENSITIVE)
            return
        await reply(update, _render_config(config, agent_of(context).memory_limit()))

    async def env_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Owner-only credential status, masked.

        Deliberately never prints a full key: the double gate (owner filter on
        the handler, explicit check inside) is the same defence as ``/run``,
        but the payload here would be credentials, so the values themselves are
        masked too. See :func:`_mask` for why.
        """
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_SENSITIVE)
            return
        await reply(update, _render_env(config))

    async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        notes = agent_of(context).reload()
        await reply(update, "reloaded:\n" + "\n".join(f"- {note}" for note in notes))

    async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        agent = agent_of(context)
        agent.reset(chat_of(update))
        history: History = context.application.bot_data["history"]
        if history.clear(chat_of(update)):
            await reply(update, "cleared this conversation and its transcript. long-term memory is untouched.")
        else:
            await reply(update, "cleared this conversation. long-term memory is untouched.")

    # -- approvals --------------------------------------------------------- #

    async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        await _resolve(update, context, approved=True)

    async def deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        await _resolve(update, context, approved=False)

    # -- whitelist management (owner only) --------------------------------- #

    async def whitelist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER)
            return
        lines = ["<b>whitelist</b>"]
        lines.append(f"owner: <code>{config.owner_id}</code>")
        lines.append("")
        lines.append("<b>whitelisted users:</b>")
        if config.bot.whitelisted_users:
            for uid in config.bot.whitelisted_users:
                lines.append(f"- <code>{escape_html(uid)}</code>")
        else:
            lines.append("- <i>(none)</i>")
        lines.append("")
        lines.append("<b>whitelisted groups:</b>")
        if config.bot.whitelisted_groups:
            for gid in config.bot.whitelisted_groups:
                lines.append(f"- <code>{escape_html(gid)}</code>")
        else:
            lines.append("- <i>(none)</i>")
        await reply(update, "\n".join(lines))

    async def whitelist_add_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER)
            return
        if not context.args:
            await reply(update, "usage: <code>/whitelist_add_user &lt;id&gt;</code>")
            return
        new_id = context.args[0].strip()
        try:
            int(new_id)
        except ValueError:
            await reply(update, f"<code>{escape_html(new_id)}</code> is not a valid id")
            return
        if new_id not in config.bot.whitelisted_users:
            config.bot.whitelisted_users.append(new_id)
            _save_whitelist(config)
            await reply(update, f"added <code>{escape_html(new_id)}</code> to the user whitelist.")
        else:
            await reply(update, f"<code>{escape_html(new_id)}</code> is already whitelisted.")

    async def whitelist_remove_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER)
            return
        if not context.args:
            await reply(update, "usage: <code>/whitelist_remove_user &lt;id&gt;</code>")
            return
        remove_id = context.args[0].strip()
        if remove_id in config.bot.whitelisted_users:
            config.bot.whitelisted_users.remove(remove_id)
            _save_whitelist(config)
            await reply(update, f"removed <code>{escape_html(remove_id)}</code> from the user whitelist.")
        else:
            await reply(update, f"<code>{escape_html(remove_id)}</code> is not in the user whitelist.")

    async def whitelist_add_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER)
            return
        if not context.args:
            await reply(update, "usage: <code>/whitelist_add_group &lt;id&gt;</code>")
            return
        new_id = context.args[0].strip()
        try:
            int(new_id)
        except ValueError:
            await reply(update, f"<code>{escape_html(new_id)}</code> is not a valid id")
            return
        if new_id not in config.bot.whitelisted_groups:
            config.bot.whitelisted_groups.append(new_id)
            _save_whitelist(config)
            await reply(update, f"added <code>{escape_html(new_id)}</code> to the group whitelist.")
        else:
            await reply(update, f"<code>{escape_html(new_id)}</code> is already whitelisted.")

    async def whitelist_remove_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER)
            return
        if not context.args:
            await reply(update, "usage: <code>/whitelist_remove_group &lt;id&gt;</code>")
            return
        remove_id = context.args[0].strip()
        if remove_id in config.bot.whitelisted_groups:
            config.bot.whitelisted_groups.remove(remove_id)
            _save_whitelist(config)
            await reply(update, f"removed <code>{escape_html(remove_id)}</code> from the group whitelist.")
        else:
            await reply(update, f"<code>{escape_html(remove_id)}</code> is not in the group whitelist.")

    async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        user = update.effective_user
        if user is None or not _is_allowed(user, config):
            await _toast(query, "not authorised", show_alert=True)
            return
        data = query.data or ""
        if data.startswith(CB_THINK):
            await _expand_thinking(update, context, data[len(CB_THINK) :])
            return
        approved = data.startswith(CB_OK)
        action_id = data[len(CB_OK) :] if approved else data[len(CB_NO) :]
        if not action_id:
            await _toast(query, "malformed button", show_alert=True)
            return
        await _toast(query, "ok")
        await _resolve(update, context, approved=approved, action_id=action_id)

    # -- the main text path ------------------------------------------------ #

    async def fetch_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
        """Download the largest photo size and return it as base64.

        Telegram sends one message per resolution; the last entry is the
        biggest. Returns None (after explaining itself) if the download fails,
        so a bad image never turns into a model call with a broken payload.
        """
        message = update.effective_message
        if message is None or not message.photo:
            return None
        try:
            file = await context.bot.get_file(message.photo[-1].file_id)
            buffer = await file.download_as_bytearray()
        except (TelegramError, BadRequest, OSError) as exc:
            log.warning("could not download photo: %s", exc)
            await reply(update, "i couldn't download that image — try sending it again.")
            return None
        return base64.b64encode(bytes(buffer)).decode("ascii")

    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return
        if not await authorised(update):
            return
        if not _should_reply(update, context, config):
            return
        await run_agent(update, context, message.text)

    async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """An incoming file: store it in the project, then let the model at it.

        Documents arrive through the non-text handler (a document with a caption
        has ``text`` unset, so it never reaches ``on_text`` — the caption rides
        along in ``message.caption``). The file lands under
        ``workspace/uploads/<chat_id>/`` and the model gets a prompt naming the
        path and the file's facts, so it can read, rename, summarise or process
        it like any other project file.
        """
        message = update.effective_message
        if message is None or message.document is None:
            return
        chat_id = chat_of(update)

        try:
            file = await context.bot.get_file(message.document.file_id)
            buffer = await file.download_as_bytearray()
        except (TelegramError, BadRequest, OSError) as exc:
            log.warning("could not download document: %s", exc)
            await reply(update, "i couldn't download that file — try sending it again.")
            return

        store = artifacts_of(context)
        name = message.document.file_name or "upload.bin"
        try:
            artifact = store.ingest(bytes(buffer), name, chat_id, config.root)
        except ArtifactError as exc:
            await reply(update, f"i can't take that file: {escape_html(str(exc))}")
            return
        except OSError as exc:
            log.warning("could not store document %s: %s", name, exc)
            await reply(update, "i couldn't store that file — check the logs.")
            return

        log.info("document %s (%d bytes) stored for chat %s", artifact.path, artifact.size, chat_id)
        await reply(update, f"📎 saved <code>{escape_html(artifact.path)}</code> ({artifact.size:,} bytes)")

        caption = (message.caption or "").strip()
        prompt = (
            f"[The owner sent a file: {artifact.path} ({artifact.size} bytes, {artifact.mime}). "
            "It is in the project; read it with the `files` tool before acting on it."
            f"]\n\n{caption or 'What is this file, and what should I do with it?'}"
        )
        await run_agent(update, context, prompt)

    async def on_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        if not _should_reply(update, context, config):
            return
        message = update.effective_message
        if message is None:
            return

        # A document (with or without a caption): store it in the project so
        # the model can work on it like any other file, then hand the model a
        # prompt that says where it landed.
        if message.document:
            await on_document(update, context)
            return

        # A photo: use the caption if present, otherwise ask the obvious question.
        if message.photo:
            image_b64 = await fetch_photo(update, context)
            if image_b64 is None:
                return
            prompt = (message.caption or "").strip() or "what's in this image?"
            await run_agent(update, context, prompt, image_b64=image_b64)
            return

        await reply(update, "i read text and images only — send a message or a photo.")

    async def on_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        if not _should_reply(update, context, config):
            return
        message = update.effective_message
        text = (message.text if message else "") or ""
        # context.args is None when a command was sent with no arguments, so the
        # name has to come from the text itself.
        command = text.split(maxsplit=1)[0] if text.strip() else "that"
        await reply(update, f"i don't know <code>{escape_html(command)}</code>. try /help for what i can do.")

    async def on_stranger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Anyone who is not the owner gets one clear sentence and nothing else.

        Deliberately does not enumerate tools, models, or the filesystem: a
        stranger should not be able to learn what this bot can reach.
        """
        log.warning("rejected a message from non-owner %s", update.effective_user and update.effective_user.id)
        await reply(update, NOT_AUTHORISED)

    # -- shared runners ---------------------------------------------------- #

    async def run_agent(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
        *,
        image_b64: str | None = None,
        image_mime: str = "image/jpeg",
    ) -> None:
        agent = agent_of(context)
        chat_id = chat_of(update)
        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.handle(chat_id, text, image_b64=image_b64, image_mime=image_mime)
            except Exception as exc:  # noqa: BLE001 - a crash must not kill the bot
                log.exception("agent turn failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config, traces_of(context))

    async def run_tool(
        update: Update, context: ContextTypes.DEFAULT_TYPE, tool: str, arguments: dict[str, Any]
    ) -> None:
        agent = agent_of(context)
        chat_id = chat_of(update)
        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.run_tool(chat_id, tool, arguments)
            except Exception as exc:  # noqa: BLE001
                log.exception("tool %s failed", tool)
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config, traces_of(context))

    async def _expand_thinking(
        update: Update, context: ContextTypes.DEFAULT_TYPE, payload: str
    ) -> None:
        """Show or hide a collapsed thinking trace, in place.

        Toggles rather than expanding once, because a chat that is being
        scrolled back through is exactly where you want to close it again.
        """
        query = update.callback_query
        if query is None:
            return
        action, _, token = payload.partition(":")

        if action == "hide":
            # No token needed: collapsing does not read the trace, so it works
            # even after the store has forgotten it.
            await _toast(query, "hidden")
            await _edit_thinking(query, "<i>thinking hidden</i>")
            return

        if action != "show" or not token:
            await _toast(query, "malformed button", show_alert=True)
            return

        traces: TraceStore = context.application.bot_data["traces"]
        trace = traces.get(token)
        if trace is None:
            # Expired or evicted. Say so in the toast rather than silently
            # editing the message, which would look like a broken button.
            await _toast(query, "that trace has expired — ask again to get a new one", show_alert=True)
            return

        await _toast(query, "ok")
        await _edit_thinking(query, render_thinking(trace), keyboard=thinking_keyboard_hide())

    async def _resolve(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        approved: bool,
        action_id: str | None = None,
    ) -> None:
        agent = agent_of(context)
        chat_id = chat_of(update)
        if action_id is None:
            pending = agent.conversation(chat_id).pending
            if not pending:
                await reply(update, "nothing is waiting for approval.")
                return
            action_id = pending[0].id

        verb = "running" if approved else "cancelling"
        if update.callback_query is not None:
            with contextlib.suppress(BadRequest, TelegramError):
                await update.callback_query.edit_message_text(
                    f"{verb}…", parse_mode=ParseMode.HTML
                )

        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.resolve(chat_id, action_id, approved)
            except Exception as exc:  # noqa: BLE001
                log.exception("resolving approval failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config, traces_of(context))

    # -- whitelist persistence --------------------------------------------- #

    def _save_whitelist(cfg: Config) -> None:
        """Persist the whitelist to data/whitelist.json.

        Uses a dedicated JSON file rather than rewriting ``config.local.toml``
        to avoid corrupting the user's personal config with a hand-rolled TOML
        writer that cannot round-trip all value types.
        """
        path = cfg.root / "data" / "whitelist.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "whitelisted_users": cfg.bot.whitelisted_users,
            "whitelisted_groups": cfg.bot.whitelisted_groups,
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)

    def _load_whitelist(cfg: Config) -> None:
        """Load the whitelist from data/whitelist.json into the config.

        Called during ``build_handlers`` so that whitelist changes take effect
        immediately without requiring a process restart.
        """
        path = cfg.root / "data" / "whitelist.json"
        if not path.is_file():
            return
        try:
            with path.open("r") as f:
                data = json.load(f)
            cfg.bot.whitelisted_users = data.get("whitelisted_users", [])
            cfg.bot.whitelisted_groups = data.get("whitelisted_groups", [])
        except Exception:
            log.warning("could not read whitelist file %s", path, exc_info=True)

    # -- assembly ---------------------------------------------------------- #

    _load_whitelist(config)

    return [
        # Owner-gated commands, all before the catch-all text handler.
        # Every command is wrapped in ``gated`` so the group/mention rule
        # applies uniformly: in a group, only addressed commands run.
        CommandHandler("start", gated(start), filters=owner_filter),
        CommandHandler("help", gated(help_command), filters=owner_filter),
        CommandHandler("ask", gated(ask), filters=owner_filter),
        CommandHandler("run", gated(run_command), filters=owner_filter),
        CommandHandler("search", gated(search_command), filters=owner_filter),
        CommandHandler("fetch", gated(fetch_command), filters=owner_filter),
        CommandHandler("memory", gated(memory_command), filters=owner_filter),
        CommandHandler("remember", gated(remember_command), filters=owner_filter),
        CommandHandler("forget", gated(forget_command), filters=owner_filter),
        CommandHandler("personality", gated(personality_command), filters=owner_filter),
        CommandHandler("tools", gated(tools_command), filters=owner_filter),
        CommandHandler("reasoning", gated(reasoning_command), filters=owner_filter),
        CommandHandler("context", gated(context_command), filters=owner_filter),
        CommandHandler("status", gated(status_command), filters=owner_filter),
        CommandHandler("doctor", gated(doctor_command), filters=owner_filter),
        # Owner only, checked inside each handler on top of the filter: these
        # expose the backend (endpoint, layout, credential inventory).
        CommandHandler("config", gated(config_command), filters=owner_filter),
        CommandHandler("env", gated(env_command), filters=owner_filter),
        CommandHandler("reload", gated(reload_command), filters=owner_filter),
        CommandHandler("reset", gated(reset_command), filters=owner_filter),
        CommandHandler(["approve", "yes", "y"], gated(approve), filters=owner_filter),
        CommandHandler(["deny", "no", "n"], gated(deny), filters=owner_filter),
        # Whitelist management — owner only (checked inside each handler).
        CommandHandler("whitelist", gated(whitelist_command), filters=owner_filter),
        CommandHandler("whitelist_add_user", gated(whitelist_add_user), filters=owner_filter),
        CommandHandler("whitelist_remove_user", gated(whitelist_remove_user), filters=owner_filter),
        CommandHandler("whitelist_add_group", gated(whitelist_add_group), filters=owner_filter),
        CommandHandler("whitelist_remove_group", gated(whitelist_remove_group), filters=owner_filter),
        # Inline buttons: the approval Confirm/Cancel pair and the "show
        # thinking" toggle. One handler for both, because both are the same
        # shape — a callback_data prefix that says which kind of button it is.
        # Not wrapped in ``gated``: a tap is never a message in a group, so the
        # mention rule does not apply, and ``button`` checks the sender itself.
        # (CallbackQueryHandler takes a ``pattern``, not ``filters``, in v22.)
        CallbackQueryHandler(button),
        # Keyword arguments on purpose: MessageHandler's parameter order changed
        # between python-telegram-bot v20 and v22, and the keyword form is stable.
        MessageHandler(
            callback=on_text, filters=filters.TEXT & ~filters.COMMAND & owner_filter
        ),
        MessageHandler(
            callback=on_non_text, filters=~filters.TEXT & ~filters.COMMAND & owner_filter
        ),
        # An unrecognised command from the owner still deserves an answer.
        MessageHandler(
            callback=on_unknown_command, filters=filters.COMMAND & owner_filter
        ),
        # Everyone else. Last, so an owner's message is never caught here.
        MessageHandler(callback=on_stranger, filters=~owner_filter),
    ]


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

STARTED = time.monotonic()


def _apply_network_settings(builder: ApplicationBuilder, config: Config) -> None:
    """Apply the network settings from ``[bot]`` to *builder*.

    The connect timeout is deliberately generous: the first thing PTB does is a
    TCP connect to api.telegram.org, and on a slow or censored network the
    library default of 5s is not enough for even that handshake — the bot then
    dies in bootstrap before printing a single line. Skipped entirely when a
    test injected its own Bot object, because the request objects were already
    built and there is nothing to tune.
    """
    if config.bot.proxy_url:
        builder.proxy(config.bot.proxy_url).get_updates_proxy(config.bot.proxy_url)
    builder.connect_timeout(config.bot.connect_timeout)
    builder.get_updates_connect_timeout(config.bot.connect_timeout)
    builder.get_updates_connection_pool_size(GET_UPDATES_POOL_SIZE)


def build_application(config: Config, agent: Agent | None = None, *, bot: Any = None) -> Application:
    """Assemble the python-telegram-bot Application.

    *bot* is an injection point for tests: pass a Bot subclass that records what
    it was asked to send. When omitted, a real bot is built from the token.
    """
    token = config.telegram_token
    if bot is None and not token:
        raise ConfigError("TELEGRAM_BOT_TOKEN is not set. Add it to .env and try again.")

    problems = validate(config)
    if problems:
        log.error("configuration problems:\n  - %s", "\n  - ".join(problems))

    memory = agent.memory if agent else MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    personality = agent.personality if agent else Personality.load(config.personality_file)
    history = agent.history if agent else History(config.history_dir)

    builder: ApplicationBuilder = (
        Application.builder().bot(bot) if bot is not None else Application.builder().token(token)
    )
    if bot is None:
        # A test-injected Bot was already built with its own request objects;
        # the builder's request settings only apply to a bot built here.
        _apply_network_settings(builder, config)
    application = builder.post_init(_post_init).post_shutdown(_post_shutdown).build()

    application.bot_data["config"] = config
    application.bot_data["memory"] = memory
    application.bot_data["personality"] = personality
    application.bot_data["history"] = history
    application.bot_data["traces"] = TraceStore()
    application.bot_data["artifacts"] = ArtifactStore(
        config.root, max_bytes=max(1, config.bot.max_upload_mb) * 1024 * 1024
    )
    application.bot_data["agent"] = agent or Agent(config, personality, memory, history)

    for handler in build_handlers(config):
        application.add_handler(handler)
    application.add_error_handler(_error_handler)

    return application


async def _post_init(application: Application) -> None:
    config: Config = application.bot_data["config"]
    agent: Agent = application.bot_data["agent"]
    me = await application.bot.get_me()
    log.info("telegram: @%s (%s) ready", me.username, me.id)
    log.info("model: %s via %s", config.llm.model_of(), config.llm.base_url_of())
    log.info("tools: %s", ", ".join(agent.registry.names()) or "none")
    log.info("memory: %s", agent.memory.stats())
    if config.bot.startup_chat_id:
        try:
            await application.bot.send_message(
                chat_id=int(config.bot.startup_chat_id),
                text=f"lumi is up on {platform_line()} as @{me.username}.",
            )
        except (TelegramError, ValueError) as exc:
            log.warning("could not send the startup ping: %s", exc)


async def _post_shutdown(application: Application) -> None:
    log.info("shutting down")


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("unhandled error while processing an update", exc_info=context.error)


def run(config: Config) -> None:
    """Start long polling. Blocks until interrupted.

    Bootstrap retries follow PTB's contract: the counter counts consecutive
    failures and the loop aborts on the first success, so a bot that got online
    once has no residual budget to lose. ``-1`` means retry startup forever —
    matching the polling loop's own unlimited retries once the bot is up — so
    the process waits out an outage instead of exiting while the owner sleeps.
    """
    application = build_application(config)
    retries = config.bot.bootstrap_retries
    log.info(
        "polling for updates — ctrl-c to stop (connect timeout %ss, %s bootstrap attempt(s)%s)",
        config.bot.connect_timeout,
        "unlimited" if retries < 0 else f"up to {retries + 1}",
        f" via {config.bot.proxy_url}" if config.bot.proxy_url else "",
    )
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            close_loop=False,
            # PTB's default is 0: one TCP connect failure to api.telegram.org
            # aborts the whole startup. -1 keeps retrying startup forever,
            # matching the polling loop's own unlimited retries once running.
            bootstrap_retries=retries,
        )
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        # Best effort on the way out: a failure here must not mask the exit reason.
        with contextlib.suppress(Exception):
            application.shutdown()


__all__ = ["build_application", "build_handlers", "run", "reply", "deliver", "HELP_TEXT"]
