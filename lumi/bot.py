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
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import Agent, PendingAction, TurnResult
from .config import Config, ConfigError, validate
from .doctor import platform_line, run_checks
from .memory import History, MemoryFile
from .personality import Personality
from .util.log import get_logger
from .util.text import escape_markdown, format_error, split_message, truncate

log = get_logger(__name__)

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]

#: Callback data prefix; keeps our buttons from colliding with anything else.
CB_OK = "lumi:ok:"
CB_NO = "lumi:no:"

TYPING_INTERVAL = 4.0

HELP_TEXT = """\
*what i can do*
just talk to me. i remember you between sessions, look things up on the web,
read and write files in this project, and run shell commands.

*slash commands*
`/help` — this list
`/ask <question>` — same as just typing
`/run <command>` — run a shell command directly
`/search <query>` — web search
`/fetch <url>` — read a page
`/memory` — show what i remember
`/remember <fact>` — save a fact
`/forget [n]` — drop the last n saved facts
`/personality` — show the personality file
`/tools` — list my tools
`/status` — model, tools and provider health
`/doctor` — diagnose the whole setup
`/reload` — re-read personality, memory and config
`/reset` — forget this conversation (keeps long-term memory)

dangerous commands ask for confirmation first. i can't delete that, on purpose.
"""

NOT_AUTHORISED = "this bot is private. your id is not the owner."


# --------------------------------------------------------------------------- #
# Outbound helpers
# --------------------------------------------------------------------------- #


async def reply(update: Update, text: str, **kwargs: Any) -> None:
    """Send *text*, splitting it and degrading gracefully on bad markdown.

    The model writes Telegram-flavoured markdown, which is not valid in every
    message (an unbalanced ``_`` is enough). Rather than sanitising its prose, try
    the pretty version and fall back to plain text for the chunk that failed.
    """
    message = update.effective_message
    if message is None or not text.strip():
        return
    for chunk in split_message(text):
        try:
            await message.reply_text(
                chunk,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
                **kwargs,
            )
        except BadRequest:
            try:
                await message.reply_text(chunk, disable_web_page_preview=True, **kwargs)
            except TelegramError as exc:
                log.warning("could not deliver a chunk: %s", exc)
        except TelegramError as exc:
            log.warning("could not deliver a chunk: %s", exc)


async def reply_html(update: Update, text: str, **kwargs: Any) -> None:
    """For our own UI strings, where the markdown is ours and therefore correct."""
    message = update.effective_message
    if message is None:
        return
    for chunk in split_message(text):
        try:
            await message.reply_text(
                chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True, **kwargs
            )
        except (BadRequest, TelegramError) as exc:
            log.warning("could not deliver a chunk: %s", exc)
            with contextlib.suppress(TelegramError):
                await message.reply_text(
                    escape_markdown(chunk), disable_web_page_preview=True, **kwargs
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
        f"*needs your approval*\n"
        f"`{action.tool}` — {action.reason}\n\n"
        f"```\n{detail}\n```"
    )


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #


async def deliver(update: Update, result: TurnResult, config: Config) -> None:
    """Turn an agent result into messages, approval prompts included."""
    if result.error and not result.text:
        await reply(update, f"that didn't work: {result.error}")
        return

    if result.text:
        await reply(update, result.text)

    for action in result.pending:
        await reply_html(update, render_approval(action), reply_markup=approval_keyboard(action))

    if result.error and result.text:
        note = f"_note: {result.error}_"
        if result.tools_used:
            note += f"\n_tools used: {', '.join(dict.fromkeys(result.tools_used))}_"
        await reply(update, note)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def build_handlers(config: Config) -> list[Any]:
    owner = config.owner_id

    async def authorised(update: Update) -> bool:
        user = update.effective_user
        if owner is not None and user is not None and user.id == owner:
            return True
        await reply(update, NOT_AUTHORISED)
        log.warning("rejected %s from user %s", update.effective_message and "message", user and user.id)
        return False

    def agent_of(context: ContextTypes.DEFAULT_TYPE) -> Agent:
        agent: Agent = context.application.bot_data["agent"]
        return agent

    def chat_of(update: Update) -> int | str:
        chat = update.effective_chat
        return chat.id if chat else 0

    # -- basic commands ---------------------------------------------------- #

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        user = update.effective_user
        name = user.first_name if user else "there"
        log.info("start from %s (%s)", name, user and user.id)
        await reply(
            update,
            f"hey {name}. i'm up.\n\n{HELP_TEXT}\n\n"
            f"model: `{config.llm.model}`\n"
            f"tools: {', '.join(agent_of(context).registry.names()) or 'none'}",
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
            await reply(update, "usage: `/ask <question>`")
            return
        await run_agent(update, context, text)

    # -- direct tool invocations ------------------------------------------- #

    async def run_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        command = " ".join(context.args).strip() if context.args else ""
        if not command:
            await reply(update, "usage: `/run <command>`\ne.g. `/run git status --short`")
            return
        await run_tool(update, context, "run_shell", {"command": command})

    async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        query = " ".join(context.args).strip() if context.args else ""
        if not query:
            await reply(update, "usage: `/search <query>`")
            return
        await run_tool(update, context, "web", {"action": "search", "query": query})

    async def fetch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        url = " ".join(context.args).strip() if context.args else ""
        if not url:
            await reply(update, "usage: `/fetch <url>`")
            return
        await run_tool(update, context, "web", {"action": "fetch", "url": url})

    # -- memory ------------------------------------------------------------ #

    async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        memory: MemoryFile = context.application.bot_data["memory"]
        memory.ensure_loaded()
        stats = memory.stats()
        body = memory.for_prompt(limit=3000)
        await reply(
            update,
            f"*memory* — {stats['managed_count']} saved fact(s), {stats['chars']} chars\n"
            f"`{stats['path']}`\n\n{truncate(body, 3000)}\n\n"
            f"add one with `/remember <fact>`, drop the newest with `/forget`.",
        )

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
                await reply(update, "usage: `/forget [count]` — count has to be a number")
                return
        memory: MemoryFile = context.application.bot_data["memory"]
        removed = memory.forget(count)
        if not removed:
            await reply(update, "nothing to forget — there are no saved facts.")
            return
        listed = "\n".join(f"- {item}" for item in removed)
        await reply(update, f"forgot {len(removed)}:\n{listed}")

    # -- introspection ----------------------------------------------------- #

    async def personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        personality: Personality = context.application.bot_data["personality"]
        await reply(update, f"*PERSONALITY.md* — {len(personality.text)} chars\n\n" + personality.text)

    async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        registry = agent_of(context).registry
        await reply(update, f"*tools*\n\n{registry.describe()}")

    async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        agent = agent_of(context)
        uptime = time.monotonic() - STARTED
        lines = [
            "*status*",
            f"uptime: {uptime:.0f}s",
            f"model: `{config.llm.model}` via {config.llm.base_url}",
            f"tools: {', '.join(agent.registry.names()) or 'none'}",
        ]
        for tool in agent.registry.all():
            ok, reason = tool.available()
            if not ok:
                lines.append(f"- `{tool.name}`: {reason}")
        await reply(update, "\n".join(lines))

    async def doctor_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        checks = run_checks(config, agent_of(context).registry)
        await reply(update, "*doctor*\n" + "\n".join(check.render() for check in checks))

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

    async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        owner = config.owner_id
        user = update.effective_user
        if owner is not None and (user is None or user.id != owner):
            await query.answer("not the owner", show_alert=True)
            return
        data = query.data or ""
        approved = data.startswith(CB_OK)
        action_id = data[len(CB_OK) :] if approved else data[len(CB_NO) :]
        if not action_id:
            await query.answer("malformed button", show_alert=True)
            return
        await query.answer("ok")
        await _resolve(update, context, approved=approved, action_id=action_id)

    # -- the main text path ------------------------------------------------ #

    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not message.text:
            return
        if not await authorised(update):
            return
        await run_agent(update, context, message.text)

    async def on_non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        await reply(update, "i read text only right now — send a message instead.")

    async def on_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await authorised(update):
            return
        message = update.effective_message
        text = (message.text if message else "") or ""
        # context.args is None when a command was sent with no arguments, so the
        # name has to come from the text itself.
        command = text.split(maxsplit=1)[0] if text.strip() else "that"
        await reply(update, f"i don't know `{command}`. try /help for what i can do.")

    async def on_stranger(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Anyone who is not the owner gets one clear sentence and nothing else.

        Deliberately does not enumerate tools, models, or the filesystem: a
        stranger should not be able to learn what this bot can reach.
        """
        log.warning("rejected a message from non-owner %s", update.effective_user and update.effective_user.id)
        await reply(update, NOT_AUTHORISED)

    # -- shared runners ---------------------------------------------------- #

    async def run_agent(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        agent = agent_of(context)
        chat_id = chat_of(update)
        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.handle(chat_id, text)
            except Exception as exc:  # noqa: BLE001 - a crash must not kill the bot
                log.exception("agent turn failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config)

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
                    f"{verb}…", parse_mode=ParseMode.MARKDOWN
                )

        async with _Typing(context.bot, chat_id):
            try:
                result = await agent.resolve(chat_id, action_id, approved)
            except Exception as exc:  # noqa: BLE001
                log.exception("resolving approval failed")
                await reply(update, f"i broke on that: {format_error(exc)}")
                return
        await deliver(update, result, config)

    # -- assembly ---------------------------------------------------------- #

    owner_filter = filters.User(user_id=owner) if owner is not None else filters.User(user_id=0)

    return [
        # Owner-gated commands, all before the catch-all text handler.
        CommandHandler("start", start, filters=owner_filter),
        CommandHandler("help", help_command, filters=owner_filter),
        CommandHandler("ask", ask, filters=owner_filter),
        CommandHandler("run", run_command, filters=owner_filter),
        CommandHandler("search", search_command, filters=owner_filter),
        CommandHandler("fetch", fetch_command, filters=owner_filter),
        CommandHandler("memory", memory_command, filters=owner_filter),
        CommandHandler("remember", remember_command, filters=owner_filter),
        CommandHandler("forget", forget_command, filters=owner_filter),
        CommandHandler("personality", personality_command, filters=owner_filter),
        CommandHandler("tools", tools_command, filters=owner_filter),
        CommandHandler("status", status_command, filters=owner_filter),
        CommandHandler("doctor", doctor_command, filters=owner_filter),
        CommandHandler("reload", reload_command, filters=owner_filter),
        CommandHandler("reset", reset_command, filters=owner_filter),
        CommandHandler(["approve", "yes", "y"], approve, filters=owner_filter),
        CommandHandler(["deny", "no", "n"], deny, filters=owner_filter),
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
    application = builder.post_init(_post_init).post_shutdown(_post_shutdown).build()

    application.bot_data["config"] = config
    application.bot_data["memory"] = memory
    application.bot_data["personality"] = personality
    application.bot_data["history"] = history
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
    log.info("model: %s", config.llm.model)
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
    """Start long polling. Blocks until interrupted."""
    application = build_application(config)
    log.info("polling for updates — ctrl-c to stop")
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            close_loop=False,
        )
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        # Best effort on the way out: a failure here must not mask the exit reason.
        with contextlib.suppress(Exception):
            application.shutdown()


__all__ = ["build_application", "build_handlers", "run", "reply", "deliver", "HELP_TEXT"]
