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
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from telegram import Chat, InlineKeyboardButton, InlineKeyboardMarkup, MessageEntity, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import Agent, PendingAction, TurnHooks, TurnResult
from .artifacts import Artifact, ArtifactError, ArtifactStore
from .chatlog import CHATLOG, entry_from
from .config import Config, ConfigError, validate
from .doctor import platform_line, run_checks
from .memory import History, MemoryFile, MemoryStore
from .personality import Personality
from .util.log import get_logger
from .util.text import (
    escape_html,
    format_error,
    markdown_bold_to_html,
    markdown_code_to_html,
    markdown_table_to_html,
    sanitize_html,
    split_message,
    truncate,
)

log = get_logger(__name__)

#: Callback data prefix; keeps our buttons from colliding with anything else.
CB_OK = "lumi:ok:"
CB_NO = "lumi:no:"

TYPING_INTERVAL = 4.0

#: Pool size for the getUpdates request object. PTB sizes it at 1 connection;
#: that is fine except with a non-empty bot.proxy_url, where httpx with
#: ``max_connections=1`` is documented to misbehave (PTB wiki: "Working with
#: proxies"). One extra connection is harmless without a proxy.
GET_UPDATES_POOL_SIZE = 2

#: How many step lines the live progress message keeps visible before it
#: starts collapsing the oldest under an "… earlier" marker. A long turn can
#: call twenty tools; the list is a glance at what is happening, not a log.
STEP_MAX_LINES = 10

# All bot-authored strings are Telegram HTML: <b>…</b> for headers, <code>…</code>
# around commands and paths, <i>…</i> for asides. HTML only reserves < > & —
# and every dynamic value goes through escape_html/sanitize_html — so a path
# with underscores or a model name with asterisks can never break a message the
# way the legacy Markdown modes did.
HELP_TEXT = """\
<b>what i can do</b>
just talk to me. i remember you between sessions, look things up on the web,
read and write files in this project, and run shell commands.

send a photo (the caption tells me what to look for) or a document — it lands
in workspace/uploads/ and i work with it from there. ask me to make a file and
i'll send it back as one.

while i'm working you'll see a small live note above my answer — thinking, then
each thing i run. the thinking itself stays in my head; the note only shows the
steps.

<b>slash commands</b>
<code>/memory</code> — show what i remember in this chat
<code>/remember &lt;fact&gt;</code> — save a fact
<code>/personality</code> — show the personality file
<code>/reasoning</code> — reasoning setup, or <code>on</code>/<code>off</code> to show or hide the live step list
<code>/context</code> — what i'm holding in my context window right now
<code>/status</code> — model, tools and provider health
<code>/doctor</code> — diagnose the whole setup

dangerous commands ask for confirmation first. i can't delete that, on purpose.
dropping saved facts, clearing this conversation and reloading are owner-only —
they live in <code>/own_help</code> with the rest of the owner commands.
"""

#: The owner-only commands, listed by ``/own_help`` and nowhere else. Kept
#: out of :data:`HELP_TEXT` so ``/start`` stays a glance, not a manual — and
#: so a whitelisted user in a shared group never sees the backend inventory.
OWNER_HELP_TEXT = """\
<b>owner commands</b>
<code>/config</code> — the resolved configuration this process runs on
<code>/env</code> — which credentials are set, values masked
<code>/forget [n]</code> — drop the newest n saved facts (erasing memory is a human decision, and yours)
<code>/reload</code> — re-read personality, memory and config
<code>/reset</code> — forget this conversation (keeps long-term memory)
<code>/whitelist</code> — show whitelisted users and groups
<code>/whitelist_add_user &lt;id&gt;</code> — add a user to the whitelist
<code>/whitelist_remove_user &lt;id&gt;</code> — remove a user from the whitelist
<code>/whitelist_add_group &lt;id&gt;</code> — add a group to the whitelist
<code>/whitelist_remove_group &lt;id&gt;</code> — remove a group from the whitelist
"""

NOT_AUTHORISED = "this bot is private. your id is not whitelisted."
NOT_OWNER = "only the owner can manage the whitelist."
NOT_OWNER_SENSITIVE = "only the owner can see that."
NOT_OWNER_APPROVAL = "only the owner can approve or deny a pending action."
#: /forget, /reset and /reload. These change what the bot remembers — a chat's
#: saved facts, its transcript, its loaded state — and the policy is that only
#: the owner does that, matching docs/policy/policy.md ("only the account ID
#: registered by the person running the instance may … change stored notes").
NOT_OWNER_ACTION = "only the owner can change what this bot remembers there."


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
        memory_cap += " (an eighth of the effective context target)"
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
            f" · live steps in chat: {'on' if llm.show_reasoning else 'off'}",
            f"- context: {llm.context_window:,} token window, {llm.context_headroom:,} held back"
            f" · keep {llm.context_keep_recent} recent"
            f" · compaction {'on' if llm.compaction else 'off'}"
            f" · {llm.max_conversations} chats in memory"
            + (
                f"\n- fill target: {llm.context_target:.0%} of the usable budget"
                " — older messages trimmed back to it; system prompt and memory untouched"
                if 0.0 < llm.context_target < 1.0
                else ""
            ),
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
            "- mcp: "
            + escape_html(
                " + ".join(v for v in (web.mcp.user_config_file, web.mcp.config_file) if v)
                or "none"
            )
            + f" · timeout {web.mcp.timeout_seconds}s",
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
            f" · groups: {len(bot.whitelisted_groups)}"
            f" · always-reply chats: {len(bot.always_reply_chats)}",
            f"- file delivery: documents up to {bot.max_upload_mb} MB"
            f" · files tool: {'uploads on' if files.uploads else 'uploads off'}",
            f"- logging: {escape_html(config.logging.level)} → {escape_html(str(config.logging.file))}",
            f"- config file: {escape_html(str(config.config_path)) if config.config_path else 'none (built-in defaults)'}",
        ]
    )


# --------------------------------------------------------------------------- #
# Group / private chat routing
# --------------------------------------------------------------------------- #

#: Message attributes that carry something a human actually sent. A message
#: with none of them is a Telegram service notice — "X added the bot", "Y left",
#: "the chat was renamed" — and there is nothing in it to answer. Telegram
#: delivers the "the bot was added" notice as an ordinary message, so without
#: this check the bot greets its own arrival with "i read text and images only".
CONTENT_FIELDS = (
    "text",
    "caption",
    "photo",
    "document",
    "video",
    "audio",
    "voice",
    "animation",
    "video_note",
    "sticker",
    "contact",
    "location",
    "venue",
    "poll",
    "dice",
    "game",
    "paid_media",
    "story",
)


def _is_service_message(message: Any) -> bool:
    """Whether *message* is a Telegram service notice rather than content.

    Deliberately content-based instead of field-based: the Bot API keeps
    adding service fields (``new_chat_members``, ``new_chat_participants``,
    ``left_chat_member``, ``migrate_to_chat_id`` …) and a list of the known
    ones would go stale. "Carries nothing a person wrote or sent" is the
    property that actually matters.
    """
    if message is None:
        return True
    return not any(getattr(message, field, None) for field in CONTENT_FIELDS)


def _is_group(chat: Any) -> bool:
    """Whether *chat* is a group or supergroup (as opposed to a private chat)."""
    return chat is not None and chat.type in (Chat.GROUP, Chat.SUPERGROUP)


def _should_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, config: Config) -> bool:
    """Whether the bot should respond to this update.

    Private chats always respond — same as a normal 1:1 chat. Group and
    supergroup chats follow ``bot.group_reply_mode``:

    - ``"mention"`` (default): only respond when the bot is mentioned
      (``@Lumi_a_bot``), replied to, or sent a slash command targeting this bot
      (``/start@Lumi_a_bot``). Plain ``/start`` in a group is left alone because
      Telegram routes commands without ``@botname`` to whichever bot claims
      them first.
    - ``"always"``: respond to every owner message regardless of chat type.
    - ``"off"``: never respond in groups; private chats are unaffected.

    ``bot.always_reply_chats`` is consulted first and overrides the mode, so a
    dedicated "Lumi lab" group can opt in by id without flipping the global
    setting.

    Note what is *not* here: ``bot.whitelisted_groups``. Being whitelisted
    decides *who* the bot talks to (see :func:`_is_allowed`), never *when* —
    a whitelisted group is still a group, so the bot only answers when it is
    addressed. The one exception is ``always_reply_chats``, which is exactly
    the "answer me in this room without the mention ritual" switch.
    """
    chat = update.effective_chat
    if chat is None:
        return True
    if str(chat.id) in config.bot.always_reply_chats:
        return True
    if chat.type == Chat.PRIVATE:
        return True
    if not _is_group(chat):
        # Channels are a corner case; stay quiet by default — there is no
        # good way to address a bot in a channel without an inline query.
        return False

    mode = (config.bot.group_reply_mode or "mention").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    return _is_addressed(update, context)


def _entity_text(text: str, entity: MessageEntity) -> str:
    """The slice of *text* that *entity* covers.

    Telegram counts entity offsets in UTF-16 code units while Python counts
    characters, and the two disagree as soon as an emoji sits in front of the
    mention. Slicing the UTF-16 view and decoding it back is exact; slicing
    the string directly would read the wrong characters.
    """
    units = text.encode("utf-16-le")
    start = entity.offset * 2
    end = (entity.offset + entity.length) * 2
    return units[start:end].decode("utf-16-le", errors="ignore")


def _is_addressed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Whether this message is aimed at *this* bot in a group context.

    Three positive signals, any one of which is enough:

    1. The message is a reply to a message sent by *this* bot. A reply to
       somebody else's message — a human's, or another bot's — is not
       addressed to the bot and is ignored.
    2. The text starts with a slash command that names this bot
       (e.g. ``/start@Lumi_a_bot``). Plain ``/start`` is *not* enough — Telegram
       routes commands without ``@botname`` to whichever bot claims them first.
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

    # 1. Reply to a message from this bot. Checked before the command rule on
    #    purpose: replying to the bot *is* addressing the bot, so a command
    #    sent as a reply ("/start" quoted under the bot's answer) counts even
    #    without the @botname suffix. The reply has to be one of ours — the
    #    username has to match, so a reply to another bot in the room (or to
    #    a person) leaves the bot alone.
    reply = message.reply_to_message
    if (
        reply is not None
        and reply.from_user is not None
        and reply.from_user.is_bot
        and (reply.from_user.username or "").lower() == bot_username
    ):
        return True

    # 2. Slash command targeting this bot.
    if text.startswith("/"):
        cmd = text.split(maxsplit=1)[0]
        if "@" in cmd:
            target = cmd.split("@", 1)[1].lower().rstrip(",.;:!?")
            if target == bot_username:
                return True
        return False  # command in a group with no @botname: assume different bot

    # 3. Mention entity anywhere in the message (text or caption).
    entities = list(message.entities or []) + list(message.caption_entities or [])
    for entity in entities:
        if entity.type == MessageEntity.MENTION:
            snippet = _entity_text(text, entity).lower().lstrip("@")
            if snippet == bot_username:
                return True
        elif entity.type == MessageEntity.TEXT_MENTION and entity.user is not None:
            if (entity.user.username or "").lower() == bot_username:
                return True

    return False


#: How many messages a reply chain may quote into the prompt. Telegram puts
#: no bound on the chain — a pinned message can be answered for weeks — and
#: every link is another block in front of the model's question, so the walk
#: stops here and says so rather than letting the prefix grow unbounded.
REPLY_CHAIN_MAX = 5


def _quote_body(quoted: Any, excerpt: str = "") -> tuple[str, str]:
    """``(sender, text)`` for one replied-to message.

    A message with no text of its own — a photo, a sticker, a poll — still
    gets named, because the reply points at it: dropping it would leave the
    model holding a pointer to nothing. Telegram can also provide only an
    inaccessible reply target plus a selected quote excerpt, so *excerpt* is
    used as the best available text in that case.
    """
    sender = getattr(quoted, "from_user", None)
    who = "someone" if sender is None else (sender.first_name or sender.username or "someone")
    body = (getattr(quoted, "text", None) or getattr(quoted, "caption", None) or "").strip()
    excerpt = excerpt.strip()
    if not body:
        body = excerpt
        if not body:
            kind = next((field for field in CONTENT_FIELDS if getattr(quoted, field, None)), "")
            body = f"(a {kind} with no text)" if kind else "(an inaccessible or empty message)"
    return who, body


def _quoted_prefix(message: Any) -> str:
    """The message *message* replies to, shaped as a prompt prefix.

    A reply carries its meaning in the thing it quotes: ``@lumi try this`` is
    unanswerable without the message above it. But Telegram delivers only the
    new text, and nothing else fills the gap either — the quoted message was
    addressed to a person (or is older than the turns this conversation kept),
    so it never became a turn here and is not in the transcript. Unless it is
    folded into the prompt at this point, the model is handed a word like
    "this" with nothing for it to point at.

    Replies chain in Telegram (a reply to a reply to a reply…), and one link
    is rarely the whole story — "no it isn't" answers the message *above the
    quote*. So the chain is walked too, up to :data:`REPLY_CHAIN_MAX` links,
    rendered oldest first so the prompt reads in the order things were said.
    A single link keeps the one-message format; deeper chains get numbered
    blocks. Returns ``""`` for a message that is not a reply, so the ordinary
    path is byte-for-byte what it was.
    """
    quoted = getattr(message, "reply_to_message", None)
    if quoted is None:
        return ""

    # Telegram may include the manually selected part of a reply separately
    # from the replied-to message. It is especially important when the target
    # is inaccessible, because that excerpt may be the only readable content.
    reply_quote = getattr(message, "quote", None)
    excerpt = (getattr(reply_quote, "text", None) or "").strip()

    # chain[0] is the direct quote, chain[-1] the oldest ancestor reachable.
    # The seen-set is paranoia: Telegram has no cycles, but one would spin
    # this loop to the cap and back if it ever did.
    chain: list[Any] = []
    seen: set[int] = set()
    while quoted is not None and len(chain) < REPLY_CHAIN_MAX:
        chain.append(quoted)
        marker = getattr(quoted, "message_id", None)
        if marker is not None:
            if marker in seen:
                break
            seen.add(marker)
        quoted = getattr(quoted, "reply_to_message", None)

    if len(chain) == 1:
        who, body = _quote_body(chain[0], excerpt)
        return f'[the message this replies to, from {who}:\n"""\n{body}\n"""\n]\n\n'

    blocks: list[str] = []
    for index, link in enumerate(reversed(chain)):
        who, body = _quote_body(link, excerpt if index == len(chain) - 1 else "")
        blocks.append(f'(from {who}):\n"""\n{body}\n"""')
    note = " — chain cut at the cap" if quoted is not None else ""
    head = f"[reply chain: the {len(chain)} message(s) this replies to, oldest first{note}:"
    return f"{head}\n" + "\n\n".join(blocks) + "\n]\n\n"


def _record_room(update: Update) -> None:
    """Keep a group message the bot is about to drop.

    Called from the paths that stay silent — the mention rule and the
    stranger gate — because those messages are exactly the ones the model
    will never see any other way: not addressed to it, so never a turn of its
    conversation. The log is what ``more_context`` reads back later.

    Private chats are skipped on purpose: there every message becomes a turn,
    so the log would only duplicate what the transcript already holds.
    """
    chat = update.effective_chat
    message = update.effective_message
    if message is None or not _is_group(chat):
        return
    sender = message.from_user
    if sender is not None and sender.is_bot:
        return  # never another bot's lines — and never our own
    entry = entry_from(message)
    if entry is not None:
        CHATLOG.record(chat.id, entry)


# --------------------------------------------------------------------------- #
# Outbound helpers
# --------------------------------------------------------------------------- #


async def reply(update: Update, text: str, *, strict: bool = False, **kwargs: Any) -> None:
    """Send *text*, splitting it and degrading gracefully on bad HTML.

    *text* is arbitrary — model prose, shell output, web results — so it goes
    through :func:`sanitize_html` first: Telegram's formatting tags survive,
    everything else (stray angle brackets, a bare ``&`` in ``R&D``) is escaped
    into literal text. That is the whole reason the bot speaks HTML rather than
    legacy Markdown: an underscore in a filename cannot fail the parse, and the
    fallback below is paranoia rather than a load-bearing path.

    With ``strict=True`` a chunk that cannot be delivered after the raw-text
    fallback re-raises, so the caller (the live ``on_message`` hook) can hand
    the failure back to the agent: the agent then appends the message to
    ``TurnResult.messages`` and the turn-tail flush re-sends it instead of the
    message vanishing into a log line.
    """
    if not text.strip():
        return

    message = update.effective_message
    chat = update.effective_chat
    bot = getattr(update, "_bot", None)
    if bot is None and hasattr(update, "get_bot"):
        with contextlib.suppress(Exception):
            bot = update.get_bot()

    if message is None and chat is None:
        if strict:
            raise TelegramError("cannot deliver reply: effective_message and effective_chat are None")
        return

    async def _send_chunk(chunk: str, *, formatted: bool) -> None:
        parse_mode = ParseMode.HTML if formatted else None
        body = sanitize_html(chunk) if formatted else chunk
        if message is not None:
            await message.reply_text(
                body,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
                **kwargs,
            )
        elif chat is not None and bot is not None:
            await bot.send_message(
                chat_id=chat.id,
                text=body,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
                **kwargs,
            )
        else:
            raise TelegramError("no target to deliver chunk to")

    # Order matters: code fences first (their <pre> is protected from every
    # later pass), then tables — so bold markers inside cells are stripped by
    # the table renderer instead of turning into literal <b> tags — and bold
    # last, skipping every <pre>/<code> the first two passes created.
    for chunk in split_message(
        markdown_bold_to_html(markdown_table_to_html(markdown_code_to_html(text)))
    ):
        try:
            await _send_chunk(chunk, formatted=True)
        except BadRequest:
            # First try raw text if markup was rejected
            try:
                await _send_chunk(chunk, formatted=False)
            except BadRequest:
                # If replying to the message failed (e.g. original message was deleted),
                # fall back to sending directly to the chat without reply_to_message_id.
                if chat is not None and bot is not None and message is not None:
                    try:
                        await bot.send_message(
                            chat_id=chat.id,
                            text=chunk,
                            disable_web_page_preview=True,
                            **kwargs,
                        )
                    except (TelegramError, OSError) as exc:
                        log.warning("could not deliver a chunk to chat: %s", exc)
                        if strict:
                            raise
                else:
                    log.warning("could not deliver a chunk: rejected by Telegram")
                    if strict:
                        raise
            except (TelegramError, OSError) as exc:
                log.warning("could not deliver a chunk: %s", exc)
                if strict:
                    raise
        except (TelegramError, OSError) as exc:
            log.warning("could not deliver a chunk: %s", exc)
            if strict:
                raise


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
# Step feed (live progress)
# --------------------------------------------------------------------------- #

#: How each step kind reads in the chat. The thinking line is the header —
#: it is there from the first model call and stays while the steps stack
#: under it. An unknown tool falls back to a wrench and its raw name, so a
#: newly added tool still shows up rather than silently not.
STEP_LABELS = {
    "thinking": "🧠 Thinking",
    "run_shell": "💻 Run Bash",
    "files": "📁 Files",
    "read_file": "📖 Read File",
    "list_files": "📂 List Files",
    "grep": "🔍 Search",
    "web": "🌐 Web",
    # The web step says which action ran: a search reads one way in the feed,
    # a page fetch another. Keys come from Tool.step_detail.
    "web.search": "🌐 Web Search",
    "web.fetch": "🌐 Web Fetch",
    "memory": "📌 Memory",
    "more_context": "🕒 More Context",
    "context7": "📚 Context7",
    "send": "✉️ Send",
}


def step_label(kind: str, detail: str = "") -> str:
    """One step of a turn as a line of chat: ``🧠 Thinking``, ``💻 Run Bash``…"""
    if kind == "thinking":
        return STEP_LABELS["thinking"]
    if kind == "tool":
        return STEP_LABELS.get(detail, f"🔧 {detail}")
    return f"· {kind}{f' {detail}' if detail else ''}"


class StepFeed:
    """The live progress message: one message, edited in place as the turn runs.

    This replaces the old collapsed-thinking button. The thinking trace itself
    stays internal — never shown, not behind a button — and what the chat sees
    is the steps the turn actually took:

        🧠 Thinking
        💻 Run Bash
        🌐 Web Search

    Sent on the first step and edited after that, so a turn reads as one
    growing note above the answer. Every failure degrades the same way: if the
    message cannot be edited (or a test's Bot stub returns nothing to edit),
    each step falls back to a fresh message carrying the list so far — noisy
    but never wrong, and never an exception into the turn.
    """

    def __init__(self, bot: Any, chat_id: int | str) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._steps: list[str] = []
        self._message: Any = None

    @property
    def steps(self) -> list[str]:
        return list(self._steps)

    async def step(self, kind: str, detail: str = "") -> None:
        """Append one step (deduplicated against the previous line) and render."""
        label = step_label(kind, detail)
        if kind == "thinking" and not self._steps:
            # The header. Not worth a message of its own: a turn that never
            # touches a tool just answers, as it always did. It goes on screen
            # the moment there is a step to sit above.
            self._steps.append(label)
            return
        if self._steps and self._steps[-1] == label:
            # A second model call after a tool result would otherwise stack
            # identical lines.
            return
        self._steps.append(label)
        await self._render()

    def _body(self) -> str:
        if len(self._steps) <= STEP_MAX_LINES:
            return "\n".join(self._steps)
        hidden = len(self._steps) - (STEP_MAX_LINES - 1)
        return f"… (+{hidden} earlier)\n" + "\n".join(self._steps[-(STEP_MAX_LINES - 1) :])

    async def _render(self) -> None:
        text = sanitize_html(self._body())
        if self._message is not None:
            try:
                await self._bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=self._message.message_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                )
                return
            except (BadRequest, TelegramError) as exc:
                log.debug("could not edit the step message, sending a fresh one: %s", exc)
        sent = await self._send(text)
        if sent is not None:
            self._message = sent

    async def _send(self, text: str) -> Any:
        """Best effort: a progress message must never fail the turn it reports."""
        try:
            return await self._bot.send_message(
                chat_id=self._chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except (Forbidden, TelegramError) as exc:
            log.debug("could not send a step message: %s", exc)
            return None


async def _toast(query: Any, text: str = "", **kwargs: Any) -> None:
    """Answer a callback query, suppressing errors from stale queries.

    Telegram expires callback queries quickly, and PTB processes updates
    sequentially, so a tap that queues behind a long agent turn can easily be
    answered late. ``BadRequest`` from an expired query must not propagate —
    the follow-up action (edit, resolve) does not depend on the toast.
    """
    with contextlib.suppress(BadRequest, TelegramError):
        await query.answer(text, **kwargs)


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


async def deliver(update: Update, result: TurnResult, config: Config) -> None:
    """Turn an agent result into messages, documents included.

    The live step message is already on screen by the time this runs — the
    runner owns its lifecycle — so this is only the tail: messages the model
    sent mid-turn but that found no channel, the answer, files, approvals.
    """
    if result.error and not result.text:
        await reply(update, f"that didn't work: {escape_html(result.error)}")
        # A crashed turn can still have staged files (e.g. the model wrote with
        # upload: true and a later iteration hit the cap). Deliver what exists.
        for artifact in result.artifacts:
            await send_artifact(update, artifact)
        return

    # Mid-turn messages (the `send` tool) normally go out live through the
    # hooks; anything still sitting on the result got there because the live
    # channel failed, so flush it here rather than dropping it.
    for message in result.messages:
        await reply(update, message)

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


def _is_allowed(user: Any, config: Config, chat: Any = None) -> bool:
    """Whether *user* may talk to the bot in *chat*.

    Three ways in:

    1. the owner,
    2. a user id on ``bot.whitelisted_users``,
    3. anybody at all — in a group that is itself on ``bot.whitelisted_groups``.

    The third one is what the group whitelist is for: it names the rooms where
    the bot answers the *room* rather than one person. It grants nothing in a
    private chat, and nothing outside the whitelisted group.
    """
    owner = config.owner_id
    if owner is not None and user.id == owner:
        return True
    if str(user.id) in config.bot.whitelisted_users:
        return True
    # A whitelisted group opens the bot to every member of that room.
    return _is_group(chat) and str(chat.id) in config.bot.whitelisted_groups


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


def _toml_key(key: Any) -> str:
    """Render a dict key so it round-trips as *that key*.

    A bare TOML key cannot contain a dot, and writing one bare would make the
    parser read it as a nested table — silently restructuring the file on the
    next load. Anything that is not a bare key is quoted instead.
    """
    text = str(key)
    if text and all(c.isascii() and (c.isalnum() or c in "_-") for c in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_toml(f: Any, data: dict[str, Any], prefix: str = "") -> None:
    """Write a dict as TOML that ``tomllib`` reads back unchanged.

    Scalar keys are emitted before any table header (after a ``[table]`` line
    a bare key would be read as belonging to that table), and every key goes
    through :func:`_toml_key`, so a name containing a dot stays one key.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        f.write(f"{_toml_key(key)} = {_toml_value(value)}\n")
    for key, value in data.items():
        if isinstance(value, dict):
            section = f"{prefix}{_toml_key(key)}"
            f.write(f"\n[{section}]\n")
            _write_toml(f, value, prefix=f"{section}.")


def build_handlers(config: Config) -> list[Any]:
    class AllowedFilter(filters.UpdateFilter):
        """Filter that checks the live config for authorisation.

        This admits *allowed* principals, not the owner: the owner, users on
        ``bot.whitelisted_users``, and every member of a whitelisted group (see
        :func:`_is_allowed`). The name says so on purpose. An earlier version
        called this ``OwnerFilter`` while returning ``_is_allowed``, which read
        as a much narrower gate than it was — and it is how the owner-only
        checks in this module were easy to miss.

        Owner-only handlers do *not* rely on this filter: it lets whitelisted
        principals through. They re-check :func:`_is_owner` inline, so the
        filter answers "may this person talk to the bot?" and the handler
        answers "may this person do *that*?".

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
            return user is not None and _is_allowed(user, self.cfg, update.effective_chat)

    allowed_filter = AllowedFilter(config)

    async def authorised(update: Update) -> bool:
        user = update.effective_user
        if user is not None and _is_allowed(user, config, update.effective_chat):
            return True
        await reply(update, NOT_AUTHORISED)
        log.warning("rejected %s from user %s", update.effective_message and "message", user and user.id)
        return False

    def gated(handler):
        """Apply the group/mention gating to a command handler.

        Each command is wrapped so a group-chat message that is not addressed
        to the bot is dropped before the handler runs — and before the
        authorisation check, so an unaddressed command from a stranger stays
        silent instead of announcing the bot's existence in the room.
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not _should_reply(update, context, config):
                return
            if not await authorised(update):
                return
            await handler(update, context)

        return wrapper

    def agent_of(context: ContextTypes.DEFAULT_TYPE) -> Agent:
        agent: Agent = context.application.bot_data["agent"]
        return agent

    def chat_of(update: Update) -> int | str:
        chat = update.effective_chat
        return chat.id if chat else 0

    def memory_of(update: Update, context: ContextTypes.DEFAULT_TYPE) -> MemoryFile:
        """The long-term memory scoped to the chat this update arrived in.

        A private chat resolves to that user's own memory file and a group to
        the group's — the same file the model's prompt and the `memory` tool
        use for this chat, so /memory, /remember and /forget act on exactly
        what the bot remembers *here*.
        """
        store: MemoryStore = context.application.bot_data["memory"]
        cfg: Config = context.application.bot_data.get("config", config)
        return store.get(cfg.memory_scope_of(chat_of(update) or None))

    def hooks_for(update: Update, context: ContextTypes.DEFAULT_TYPE) -> TurnHooks:
        """Wire one turn to the chat: mid-turn messages reply live, and the
        step feed (unless the owner turned traces off) grows above the answer.

        Built per turn because both close over *update* — the reply target —
        and because a turn that never touches a tool never creates a message.
        """
        feed = StepFeed(context.bot, chat_of(update)) if config.llm.show_reasoning else None

        async def _on_message(text: str) -> None:
            await reply(update, text, strict=True)

        return TurnHooks(
            # strict: a failed live send must surface to the agent, which then
            # appends the message to TurnResult.messages so the turn-tail flush
            # in deliver() re-sends it. Without strict the error is swallowed
            # here and the message is lost.
            on_message=_on_message,
            on_step=feed.step if feed is not None else None,
        )

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
        await reply(update, f"hey {name}. i'm up.\n\n{HELP_TEXT}")

    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """``/help`` is no longer the list — it points at ``/start`` instead.

        The command itself stays one hop wide on purpose: anyone who types it
        (muscle memory, an old message, a client with it in the menu) gets a
        one-line redirect rather than "i don't know /help".
        """
        if not await authorised(update):
            return
        await reply(update, "the list lives on <code>/start</code> — send that.")

    async def own_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """The owner-only commands, for the owner only.

        Double-gated like ``/config`` and ``/env``: the handler filter admits
        owner *and* whitelisted users, so the explicit ``_is_owner`` check
        inside is what makes this owner-only. A whitelisted user in a shared
        group must not collect the backend's command inventory from a help
        text.
        """
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_SENSITIVE)
            return
        await reply(update, OWNER_HELP_TEXT)

    # -- memory ------------------------------------------------------------ #

    async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        memory = memory_of(update, context)
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
        # by="human": this fact was typed by a person, so it is injected as
        # ordinary memory rather than fenced as model-authored data. The model
        # writing memories through its own tool takes the other path.
        await reply(update, memory_of(update, context).remember(fact, source="chat", by="human"))

    async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_ACTION)
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
        removed = memory_of(update, context).forget(count)
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

    async def reasoning_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Report the reasoning setup, and toggle the live step list.

        ``/reasoning`` with no argument only reports. ``on`` and ``off`` flip
        ``llm.show_reasoning`` for this process, which is the same scope as
        every other config change here: a restart restores config.toml, and that
        is the right time to make it stick. The thinking trace itself is never
        shown either way — the toggle only controls the step list above the
        answer (🧠 Thinking / 💻 Run Bash …).
        """
        if not await authorised(update):
            return
        llm = config.llm
        argument = (context.args[0].strip().lower() if context.args else "")

        if argument in ("on", "off"):
            llm.show_reasoning = argument == "on"
            log.info("the step feed turned %s by chat %s", argument, chat_of(update))
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
            f"live steps in chat: {'on' if llm.show_reasoning else 'off'}"
            f" (the thinking trace itself is never shown)",
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
        the handler, explicit check inside) is the same defence as ``/own_help``,
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
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_ACTION)
            return
        notes = agent_of(context).reload()
        await reply(update, "reloaded:\n" + "\n".join(f"- {note}" for note in notes))

    async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_ACTION)
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
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_APPROVAL)
            return
        await _resolve(update, context, approved=True)

    async def deny(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        if user is None or not _is_owner(user, config):
            await reply(update, NOT_OWNER_APPROVAL)
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
        if user is None or not _is_allowed(user, config, update.effective_chat):
            await _toast(query, "not authorised", show_alert=True)
            return
        # ``_is_allowed`` decides who may *talk* to the bot (whitelisted users
        # and every member of a whitelisted group included). Approval is a
        # separate, owner-reserved decision, so it re-checks ``_is_owner`` —
        # otherwise the confirm tap gives no separation of duty at all.
        if not _is_owner(user, config):
            await _toast(query, NOT_OWNER_APPROVAL, show_alert=True)
            return
        data = query.data or ""
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
        if not _should_reply(update, context, config):
            # The bot stays silent; the room does not. Keep what it said so
            # `more_context` can show the model a conversation it was never
            # addressed in (see lumi.chatlog).
            _record_room(update)
            return
        if not await authorised(update):
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
            "It is in the project; read it with the `files` tool before acting on it. "
            "For a PDF, the `ocr` tool extracts its text (scanned pages included)."
            f"]\n\n{caption or 'What is this file, and what should I do with it?'}"
        )
        await run_agent(update, context, prompt)

    async def on_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return
        if _is_service_message(message):
            # "X added the bot to the group", "Y left", "the chat was renamed":
            # Telegram service notices. Nothing was said to the bot and there
            # is nothing to answer — the bot stays silent instead of replying
            # "i read text and images only" to its own arrival.
            log.debug("ignoring a service message in chat %s", chat_of(update))
            return
        if not _should_reply(update, context, config):
            # Same as on_text: a dropped message is the only kind worth
            # logging — anything the bot answers is already in its history.
            _record_room(update)
            return
        if not await authorised(update):
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
        if not _should_reply(update, context, config):
            return
        if not await authorised(update):
            return
        message = update.effective_message
        text = (message.text if message else "") or ""
        # context.args is None when a command was sent with no arguments, so the
        # name has to come from the text itself.
        command = text.split(maxsplit=1)[0] if text.strip() else "that"
        await reply(update, f"i don't know <code>{escape_html(command)}</code>. try /start for what i can do.")

    async def on_stranger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Anyone who is not allowed gets one clear sentence and nothing else.

        Deliberately does not enumerate tools, models, or the filesystem: a
        stranger should not be able to learn what this bot can reach.

        In a group the sentence is only for a message that was *addressed* to
        the bot. Answering a stranger's ordinary chatter would make the bot
        interrupt every conversation in the room and announce that it is
        there — the opposite of "only reply when mentioned".
        """
        if _is_group(update.effective_chat) and not _is_addressed(update, context):
            # A stranger talking to the room, not to the bot. Silent, but not
            # invisible: it goes to the log `more_context` reads.
            _record_room(update)
            log.debug(
                "ignoring an unaddressed group message from user %s",
                update.effective_user and update.effective_user.id,
            )
            return
        log.warning("rejected a message from non-owner %s", update.effective_user and update.effective_user.id)
        await reply(update, NOT_AUTHORISED)

    async def on_my_chat_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """The bot was added to — or removed from — a chat. Say nothing.

        Telegram reports the bot's own membership change, and the room shows a
        service message for it as well. Neither is a question, so the bot
        stays silent: no greeting, no error, nothing in the chat. The event is
        only written to the log, where the operator can see who added it.
        """
        member = update.my_chat_member
        if member is None:
            return
        log.info(
            "membership in chat %s changed: %s -> %s (by user %s)",
            member.chat.id,
            member.old_chat_member.status,
            member.new_chat_member.status,
            member.from_user.id if member.from_user else "?",
        )

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
        hooks = hooks_for(update, context)
        # If this message is a reply, the message it quotes goes in front of
        # the text — see _quoted_prefix: it is the only place that quote can
        # still be recovered, and without it "try this" has no referent.
        text = _quoted_prefix(update.effective_message) + text
        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.handle(
                    chat_id,
                    text,
                    image_b64=image_b64,
                    image_mime=image_mime,
                    hooks=hooks,
                )
            except Exception as exc:  # noqa: BLE001 - a crash must not kill the bot
                log.exception("agent turn failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config)

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
                result = await agent.resolve(
                    chat_id, action_id, approved, hooks=hooks_for(update, context)
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("resolving approval failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config)

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
        # Commands, all before the catch-all text handler. Every command is
        # wrapped in ``gated`` so the group/mention rule applies uniformly: in a
        # group, only addressed commands run. ``allowed_filter`` admits
        # whitelisted principals as well as the owner, so each handler decides
        # for itself whether the caller needs to be the owner.
        CommandHandler("start", gated(start), filters=allowed_filter),
        # /help used to be the command list; it now points at /start.
        CommandHandler("help", gated(help_command), filters=allowed_filter),
        CommandHandler("memory", gated(memory_command), filters=allowed_filter),
        CommandHandler("remember", gated(remember_command), filters=allowed_filter),
        CommandHandler("forget", gated(forget_command), filters=allowed_filter),
        CommandHandler("personality", gated(personality_command), filters=allowed_filter),
        CommandHandler("reasoning", gated(reasoning_command), filters=allowed_filter),
        CommandHandler("context", gated(context_command), filters=allowed_filter),
        CommandHandler("status", gated(status_command), filters=allowed_filter),
        CommandHandler("doctor", gated(doctor_command), filters=allowed_filter),
        # Owner only, checked inside each handler on top of the filter: these
        # expose the backend (endpoint, layout, credential inventory).
        # Owner-only help: the backend commands, for the owner alone.
        # ``/own-help`` is also accepted, but only via a regex match below:
        # Telegram commands allow letters, digits and underscores only
        # (clients autocomplete ``/own_help``), and PTB's CommandHandler
        # rejects a hyphen outright with a ValueError at build time.
        CommandHandler("own_help", gated(own_help_command), filters=allowed_filter),
        MessageHandler(
            callback=gated(own_help_command),
            filters=filters.Regex(r"^/own-help(?:\s|$|@)") & allowed_filter,
        ),
        CommandHandler("config", gated(config_command), filters=allowed_filter),
        CommandHandler("env", gated(env_command), filters=allowed_filter),
        CommandHandler("reload", gated(reload_command), filters=allowed_filter),
        CommandHandler("reset", gated(reset_command), filters=allowed_filter),
        # Approvals — owner only (checked inside each handler), unlike the
        # whitelisted-principal commands: the tap decides, not the room.
        CommandHandler(["approve", "yes", "y"], gated(approve), filters=allowed_filter),
        CommandHandler(["deny", "no", "n"], gated(deny), filters=allowed_filter),
        # Whitelist management — owner only (checked inside each handler).
        CommandHandler("whitelist", gated(whitelist_command), filters=allowed_filter),
        CommandHandler("whitelist_add_user", gated(whitelist_add_user), filters=allowed_filter),
        CommandHandler("whitelist_remove_user", gated(whitelist_remove_user), filters=allowed_filter),
        CommandHandler("whitelist_add_group", gated(whitelist_add_group), filters=allowed_filter),
        CommandHandler("whitelist_remove_group", gated(whitelist_remove_group), filters=allowed_filter),
        # Inline buttons: the approval Confirm/Cancel pair. Taps are never
        # messages in a group, so the mention rule does not apply, and
        # ``button`` checks the sender itself.
        # (CallbackQueryHandler takes a ``pattern``, not ``filters``, in v22.)
        CallbackQueryHandler(button),
        # The bot's own membership changes ("the bot was added to this group").
        # Handled so the event is logged instead of falling through to a
        # handler that would answer it — the bot stays silent on arrival.
        ChatMemberHandler(on_my_chat_member),
        # Keyword arguments on purpose: MessageHandler's parameter order changed
        # between python-telegram-bot v20 and v22, and the keyword form is stable.
        MessageHandler(
            callback=on_text, filters=filters.TEXT & ~filters.COMMAND & allowed_filter
        ),
        MessageHandler(
            callback=on_non_text, filters=~filters.TEXT & ~filters.COMMAND & allowed_filter
        ),
        # An unrecognised command from the owner still deserves an answer.
        MessageHandler(
            callback=on_unknown_command, filters=filters.COMMAND & allowed_filter
        ),
        # Everyone else. Last, so an owner's message is never caught here.
        MessageHandler(callback=on_stranger, filters=~allowed_filter),
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

    store = agent.memory if agent else config.build_memory_store()
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
    application.bot_data["memory"] = store
    application.bot_data["personality"] = personality
    application.bot_data["history"] = history
    application.bot_data["artifacts"] = ArtifactStore(
        config.root, max_bytes=max(1, config.bot.max_upload_mb) * 1024 * 1024
    )
    application.bot_data["agent"] = agent or Agent(config, personality, store, history)

    for handler in build_handlers(config):
        application.add_handler(handler)
    application.add_error_handler(_error_handler)

    return application


def stop_signal_handler(
    application: Application, force_exit: Callable[[], None]
) -> Callable[[], None]:
    """The callback installed for SIGINT/SIGTERM: stop cleanly, or die on demand.

    PTB's own stop handling raises ``SystemExit`` inside the event loop. That
    works exactly once: a second press lands mid-cleanup — inside
    ``run_until_complete(updater.stop())`` — and aborts the graceful chain, so
    ``post_shutdown`` never runs and the process leaves a traceback behind. It
    also offers no escape from the wait itself: while a turn is in flight
    (an LLM call retrying through its backoff, a slow search), the first
    ctrl-c looks like the bot ignored it.

    This handler replaces PTB's: the first signal stops the loop cleanly (the
    graceful chain in :meth:`Application.run_polling` then drains whatever is
    in flight), and a second one — the user has said "now" twice — exits the
    process immediately. When the application is not running yet (a signal
    during bootstrap), it raises :exc:`KeyboardInterrupt`, which PTB's
    ``__run`` handles the same as its own ``SystemExit``.
    """
    stopping = False

    def handler() -> None:
        nonlocal stopping
        if application.running:
            if stopping:
                log.warning("second stop signal — exiting immediately")
                force_exit()
                return
            stopping = True
            log.info("stop signal — finishing in-flight work; send it again to stop now")
            application.stop_running()
        else:
            raise KeyboardInterrupt

    return handler


def _install_stop_handlers(application: Application) -> None:
    """Wire :func:`stop_signal_handler` to SIGINT/SIGTERM on the running loop.

    Best effort: an event loop without ``add_signal_handler`` (Windows'
    ProactorEventLoop) keeps PTB's default handling — the KeyboardInterrupt
    path still shuts down gracefully there, just without the force-exit
    escape.
    """
    try:
        loop = asyncio.get_running_loop()
        handler = stop_signal_handler(application, lambda: os._exit(130))
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handler)
    except (NotImplementedError, RuntimeError):
        log.debug("no signal handler support on this loop; using PTB's default stop handling")


async def _post_init(application: Application) -> None:
    config: Config = application.bot_data["config"]
    agent: Agent = application.bot_data["agent"]
    _install_stop_handlers(application)
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
    agent: Agent | None = application.bot_data.get("agent")
    if agent is not None:
        # Cached HTTP clients (context7) close here rather than being left to
        # the GC; the registry swallows per-tool failures, and shutdown is
        # best-effort on this path anyway.
        await agent.registry.aclose()
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

    Stopping is PTB's graceful chain (drain the polling loop, stop handlers,
    ``post_shutdown``), triggered by the first SIGINT/SIGTERM via the handler
    :func:`_post_init` installs; a second signal exits immediately. No cleanup
    is done here: ``run_polling`` owns the whole shutdown, and an extra
    ``application.shutdown()`` on this side would only ever be a coroutine
    nobody awaited.
    """
    application = build_application(config)
    retries = config.bot.bootstrap_retries
    log.info(
        "polling for updates — ctrl-c to stop (connect timeout %ss, %s bootstrap attempt(s)%s)",
        config.bot.connect_timeout,
        "unlimited" if retries < 0 else f"up to {retries + 1}",
        f" via {config.bot.proxy_url}" if config.bot.proxy_url else "",
    )
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
        # PTB's default is 0: one TCP connect failure to api.telegram.org
        # aborts the whole startup. -1 keeps retrying startup forever,
        # matching the polling loop's own unlimited retries once running.
        bootstrap_retries=retries,
    )


__all__ = ["build_application", "build_handlers", "run", "reply", "deliver", "HELP_TEXT"]
