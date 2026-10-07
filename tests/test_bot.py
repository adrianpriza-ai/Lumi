"""The Telegram layer, end to end.

A fake Bot records what would have been sent, and the scripted LLM stands in for
the model. That makes it possible to assert on the whole path: a message arrives,
is authorised, reaches the agent, and comes back as Telegram messages — including
the Confirm/Cancel keyboard for a risky command.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest
from conftest import FakeLLM, make_reply
from telegram import Bot, Update, User

from lumi.agent import Agent, TurnResult
from lumi.bot import CB_NO, CB_OK, CB_THINK, TraceStore, build_application, build_handlers
from lumi.memory import History, MemoryFile
from lumi.personality import Personality
from lumi.tools import build_registry

CHAT = 42
OWNER = 42
STRANGER = 999


class _FakeFile:
    """A minimal file object that supports download_as_bytearray."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def download_as_bytearray(self, **kwargs: Any) -> bytearray:
        return bytearray(self._data)


class FakeBot(Bot):
    """A real Bot with the network stubbed out, recording what it was asked to send.

    TelegramObject freezes attribute assignment once constructed, so the record
    lists are private with read-only properties in front of them.
    """

    def __init__(self) -> None:
        self._sent: list[dict[str, Any]] = []
        self._edits: list[dict[str, Any]] = []
        self._toasts: list[dict[str, Any]] = []
        self._actions: list[tuple[Any, ...]] = []
        super().__init__(token="123456:TESTTOKENFORLUMI")

    @property
    def sent(self) -> list[dict[str, Any]]:
        return self._sent

    @property
    def edits(self) -> list[dict[str, Any]]:
        return self._edits

    @property
    def toasts(self) -> list[dict[str, Any]]:
        """Every callback-query answer, i.e. what the button reported back."""
        return self._toasts

    @property
    def actions(self) -> list[tuple[Any, ...]]:
        return self._actions

    async def initialize(self) -> Bot:
        # Skip the real get_me round trip that Bot.initialize() performs, but set
        # the same cache it would so `bot.bot` / `bot.id` keep working.
        self._initialized = True
        self._bot_user = await self.get_me()
        return self

    async def get_me(self) -> User:
        return User(id=12345, is_bot=True, first_name="Lumi", username="lumi_test_bot")

    async def send_message(self, chat_id: Any, text: str, **kwargs: Any) -> Any:
        self._sent.append({"chat_id": chat_id, "text": text, **kwargs})
        return None

    async def edit_message_text(self, text: str, **kwargs: Any) -> Any:
        self._edits.append({"text": text, **kwargs})
        return None

    async def answer_callback_query(self, *args: Any, **kwargs: Any) -> bool:
        self._toasts.append({"args": args, **kwargs})
        return True

    async def send_chat_action(self, chat_id: Any, action: Any, **kwargs: Any) -> bool:
        self._actions.append((chat_id, action))
        return True

    async def get_file(self, file_id: str, **kwargs: Any) -> Any:
        return _FakeFile(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    async def send_document(self, chat_id: Any, document: Any = None, **kwargs: Any) -> Any:
        """Record a document send; read the payload if it is a real file handle."""
        payload = None
        if hasattr(document, "read"):
            payload = document.read()
            document.seek(0) if hasattr(document, "seek") else None
        elif isinstance(document, bytes):
            payload = document
        self._sent.append({"chat_id": chat_id, "document": payload, **kwargs})
        return None


class StaleToastBot(FakeBot):
    """A FakeBot whose callback toasts always fail the way Telegram's do.

    Telegram expires callback queries quickly, and PTB processes updates
    sequentially, so a tap that queues behind a long agent turn can be answered
    late: ``query.answer()`` then raises ``BadRequest: Query is too old and
    response timeout expired...``. The follow-up (approval resolve, message
    edit) does not depend on the toast, so it must happen anyway.
    """

    async def answer_callback_query(self, *args: Any, **kwargs: Any) -> bool:
        from telegram.error import BadRequest

        raise BadRequest(
            "Query is too old and response timeout expired or query id is invalid"
        )


def make_update(bot: Bot, text: str | None = None, *, chat_id: int = CHAT, user_id: int = OWNER,
                message_id: int = 1, chat_type: str = "private",
                chat_username: str = "lab_group",
                entities: list[dict[str, Any]] | None = None,
                reply_to_message: dict[str, Any] | None = None) -> Update:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": chat_type, "title": "Lab"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner", "username": "owner"},
    }
    if text is not None:
        message["text"] = text
        # Real Telegram always sends a bot_command entity alongside the text, and
        # filters.COMMAND keys off the entity, not off the leading slash. Without
        # it every command would fall through to the catch-all text handler.
        if text.startswith("/") and entities is None:
            command = text.split(maxsplit=1)[0]
            message["entities"] = [
                {"type": "bot_command", "offset": 0, "length": len(command)}
            ]
        if entities is not None:
            message["entities"] = entities
    if reply_to_message is not None:
        message["reply_to_message"] = reply_to_message
    return Update.de_json({"update_id": message_id, "message": message}, bot)


def make_document_update(bot: Bot, *, caption: str | None = None, chat_id: int = CHAT,
                         chat_type: str = "private", user_id: int = OWNER,
                         file_name: str = "notes.txt",
                         file_size: int = 11) -> Update:
    """An update carrying a document message (with or without a caption)."""
    message: dict[str, Any] = {
        "message_id": 5,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": chat_type, "title": "Lab"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner", "username": "owner"},
        "document": {
            "file_id": "doc1",
            "file_unique_id": "du1",
            "file_name": file_name,
            "mime_type": "text/plain",
            "file_size": file_size,
        },
    }
    if caption is not None:
        message["caption"] = caption
    return Update.de_json({"update_id": 5, "message": message}, bot)


def make_callback(bot: Bot, data: str, *, chat_id: int = CHAT, user_id: int = OWNER) -> Update:
    """An update for a tapped inline button."""
    return Update.de_json(
        {
            "update_id": 1,
            "callback_query": {
                "id": "q1",
                "from": {"id": user_id, "is_bot": False, "first_name": "Owner"},
                "chat_instance": "ci",
                "data": data,
                "message": {
                    "message_id": 7,
                    "date": int(time.time()),
                    "chat": {"id": chat_id, "type": "private"},
                    "text": "stub",
                },
            },
        },
        bot,
    )


def build(config, replies, bot: FakeBot | None = None) -> tuple[Any, FakeBot, Agent]:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(replies),
    )
    if bot is None:
        bot = FakeBot()
    application = build_application(config, agent, bot=bot)
    return application, bot, agent


async def send(config, replies, text: str, **kwargs) -> tuple[FakeBot, Agent]:
    application, bot, agent = build(config, replies)
    await application.initialize()
    try:
        await application.process_update(make_update(bot, text, **kwargs))
    finally:
        await application.shutdown()
    return bot, agent


def texts(bot: FakeBot) -> str:
    return "\n".join(message["text"] for message in bot.sent if "text" in message)


def documents(bot: FakeBot) -> list[dict[str, Any]]:
    """Every document the bot tried to send (payload bytes recorded)."""
    return [message for message in bot.sent if "document" in message]


# --------------------------------------------------------------------------- #
# owner gating
# --------------------------------------------------------------------------- #


async def test_stranger_is_rejected(config) -> None:
    bot, _ = await send(config, [make_reply("should never be sent")], "hi", user_id=STRANGER)
    assert "not whitelisted" in texts(bot).lower()
    assert "should never be sent" not in texts(bot)


async def test_owner_is_allowed(config) -> None:
    bot, _ = await send(config, [make_reply("hello owner")], "hi", user_id=OWNER)
    assert "hello owner" in texts(bot)


async def test_owner_filter_blocks_before_the_agent(config) -> None:
    """A stranger must not even cost a model call."""
    _, agent = await send(config, [make_reply("nope")], "hi", user_id=STRANGER)
    assert agent.llm.calls == []


async def test_tools_are_invisible_to_strangers(config) -> None:
    bot, _ = await send(config, [], "/run rm -rf /", user_id=STRANGER)
    assert "not whitelisted" in texts(bot).lower()


def test_the_handler_gate_is_a_live_view_of_the_whitelist(config) -> None:
    """The filter in front of every handler must read the *live* config.

    A snapshot of the id list taken when handlers were built kept rejecting a
    user added via ``/whitelist_add_user`` until the process restarted. Every
    handler also re-checks internally, which hides a gate that never runs at
    all — so the gate itself has to be asserted, not inferred from replies.
    """
    from telegram.ext import MessageHandler

    handlers = build_handlers(config)
    text_gate = next(
        h for h in handlers
        if isinstance(h, MessageHandler) and getattr(h.callback, "__name__", "") == "on_text"
    )
    stranger_gate = next(
        h for h in handlers
        if isinstance(h, MessageHandler) and getattr(h.callback, "__name__", "") == "on_stranger"
    )
    bot = FakeBot()

    # Owner passes, a stranger does not — and the stranger routes to on_stranger.
    assert text_gate.filters.check_update(make_update(bot, "hi", user_id=OWNER))
    assert not text_gate.filters.check_update(make_update(bot, "hi", user_id=STRANGER))
    assert stranger_gate.filters.check_update(make_update(bot, "hi", user_id=STRANGER))
    assert not stranger_gate.filters.check_update(make_update(bot, "hi", user_id=OWNER))

    # A user added *after* the handlers were built must pass without a restart.
    config.bot.whitelisted_users.append("777")
    assert text_gate.filters.check_update(make_update(bot, "hi", user_id=777))
    assert not stranger_gate.filters.check_update(make_update(bot, "hi", user_id=777))


# --------------------------------------------------------------------------- #
# conversation
# --------------------------------------------------------------------------- #


async def test_plain_text_gets_a_reply(config) -> None:
    bot, _ = await send(config, [make_reply("the answer is 42")], "what is the answer?")
    assert "the answer is 42" in texts(bot)


async def test_start_greets_and_lists_tools(config) -> None:
    bot, _ = await send(config, [], "/start")
    body = texts(bot)
    assert "i'm up" in body
    assert "run_shell" in body
    assert "test-model" in body


async def test_help_lists_commands(config) -> None:
    bot, _ = await send(config, [], "/help")
    for command in ("/run", "/search", "/memory", "/doctor", "/reset"):
        assert command in texts(bot)


async def test_help_advertises_the_approval_flow(config) -> None:
    bot, _ = await send(config, [], "/help")
    assert "confirmation" in texts(bot).lower()


async def test_long_answer_is_split(config) -> None:
    long_text = "word " * 3000  # comfortably over 4096 characters
    assert len(long_text) > 4096
    bot, _ = await send(config, [make_reply(long_text)], "say a lot")
    assert len(bot.sent) > 1
    assert all(len(message["text"]) <= 4096 for message in bot.sent)


async def test_typing_indicator_is_sent(config) -> None:
    bot, _ = await send(config, [make_reply("done")], "hi")
    assert bot.actions, "the bot should show as typing while the agent works"


# --------------------------------------------------------------------------- #
# direct tool commands
# --------------------------------------------------------------------------- #


async def test_run_command_executes(config) -> None:
    bot, _ = await send(config, [make_reply("it printed hello")], "/run echo hello")
    assert "it printed hello" in texts(bot)


async def test_run_without_arguments_explains_usage(config) -> None:
    bot, _ = await send(config, [], "/run")
    assert "usage" in texts(bot).lower()


async def test_search_command(config) -> None:
    bot, _ = await send(config, [make_reply("no provider was available")], "/search python")
    assert "no provider" in texts(bot)


async def test_fetch_command_requires_a_url(config) -> None:
    bot, _ = await send(config, [], "/fetch")
    assert "usage" in texts(bot).lower()


# --------------------------------------------------------------------------- #
# approvals
# --------------------------------------------------------------------------- #


async def test_risky_command_renders_a_keyboard(config) -> None:
    bot, _ = await send(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
        "/run rm -rf workspace/tmp",
    )
    body = texts(bot)
    assert "needs your approval" in body
    assert "rm -rf workspace/tmp" in body

    keyboards = [m.get("reply_markup") for m in bot.sent if m.get("reply_markup")]
    assert keyboards, "an approval prompt must carry inline buttons"
    inline = keyboards[0].inline_keyboard
    assert len(inline[0]) == 2
    labels = [button.text for button in inline[0]]
    assert labels == ["run it", "cancel"]


async def test_approval_button_data_is_well_formed(config) -> None:
    bot, _ = await send(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
        "/run rm -rf workspace/tmp",
    )
    keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
    approve, cancel = keyboard.inline_keyboard[0]
    assert approve.callback_data.startswith(CB_OK)
    assert cancel.callback_data.startswith(CB_NO)
    assert len(approve.callback_data) <= 64  # Telegram's hard limit
    assert len(cancel.callback_data) <= 64


async def test_tapping_run_it_actually_runs_the_command(config) -> None:
    """The whole approval flow, through the button rather than the slash command.

    A handler that is defined but never registered is invisible to every other
    test here: the keyboard is drawn and the callback data is well formed, and
    the tap still goes nowhere. Only driving a real callback query catches it.

    Note the script: ``/run`` invokes its tool directly and never calls the
    model, so the first scripted reply is the model's turn *after* the approval
    rather than the one that asked for it.
    """
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo done"})]),
            make_reply("it printed approved"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/run rm -rf workspace/tmp"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(make_callback(bot, approve.callback_data))
    finally:
        await application.shutdown()

    assert "it printed approved" in texts(bot)
    assert not agent.conversation(42).pending  # the queue was drained


async def test_tapping_cancel_declines_the_command(config) -> None:
    """A refusal runs nothing, so the model's very next turn is its first."""
    application, bot, agent = build(
        config, [make_reply("understood, i left it alone")]
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/run rm -rf workspace/tmp"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        cancel = keyboard.inline_keyboard[0][1]
        await application.process_update(make_callback(bot, cancel.callback_data))
    finally:
        await application.shutdown()

    assert "left it alone" in texts(bot)
    assert agent.llm.index == 1


async def test_a_stale_approval_tap_still_resolves(config) -> None:
    """A late ``query.answer()`` must not eat the tap.

    Telegram expires callback queries quickly and a tap can queue behind a long
    agent turn, so the toast raises ``BadRequest: Query is too old...``. The
    approval resolution does not depend on the toast — it must still run.
    """
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo done"})]),
            make_reply("it printed approved"),
        ],
        bot=StaleToastBot(),
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/run rm -rf workspace/tmp"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(make_callback(bot, approve.callback_data))
    finally:
        await application.shutdown()

    assert "it printed approved" in texts(bot)
    assert not agent.conversation(CHAT).pending  # the tap still resolved it


async def test_model_asks_for_approval_from_plain_text(config) -> None:
    bot, _ = await send(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
        "please clean up the workspace",
    )
    assert "needs your approval" in texts(bot)


# --------------------------------------------------------------------------- #
# memory commands
# --------------------------------------------------------------------------- #


async def test_memory_shows_the_file(config) -> None:
    bot, _ = await send(config, [], "/memory")
    body = texts(bot)
    assert "MEMORY.md" in body
    assert "saved fact" in body


async def test_remember_saves_a_fact(config) -> None:
    bot, _ = await send(config, [], "/remember the owner prefers dark mode")
    assert "remembered" in texts(bot).lower()

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    assert "the owner prefers dark mode" in memory.managed()


async def test_remember_requires_an_argument(config) -> None:
    bot, _ = await send(config, [], "/remember")
    assert "usage" in texts(bot).lower()


async def test_forget_removes_the_newest(config) -> None:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    memory.remember("first fact")
    memory.remember("second fact")

    bot, _ = await send(config, [], "/forget")
    body = texts(bot)
    assert "forgot 1" in body
    assert "second fact" in body
    assert memory.managed() == ["first fact"]


async def test_forget_rejects_a_non_number(config) -> None:
    bot, _ = await send(config, [], "/forget banana")
    assert "usage" in texts(bot).lower()


async def test_forget_with_nothing_to_forget(config) -> None:
    bot, _ = await send(config, [], "/forget")
    assert "nothing to forget" in texts(bot).lower()


# --------------------------------------------------------------------------- #
# introspection commands
# --------------------------------------------------------------------------- #


async def test_personality_command_shows_the_file(config) -> None:
    bot, _ = await send(config, [], "/personality")
    assert "You are Lumi" in texts(bot)


async def test_tools_command(config) -> None:
    bot, _ = await send(config, [], "/tools")
    body = texts(bot)
    assert "run_shell" in body
    assert "files" in body


async def test_status_command(config) -> None:
    bot, _ = await send(config, [], "/status")
    body = texts(bot)
    assert "uptime" in body
    assert "test-model" in body


async def test_context_command_reports_the_window(config) -> None:
    """A bot that forgets is hard to argue with, so what it is holding has to
    be inspectable rather than a matter of trust."""
    bot, _ = await send(config, [], "/context")
    body = texts(bot)
    assert "window" in body
    assert "in use" in body
    assert "condensed" in body
    assert "transcript" in body


async def test_context_shows_the_record_once_there_is_one(config) -> None:
    history = History(config.history_dir)
    history.append(CHAT, "user", "how do we deploy?")
    history.append(CHAT, "assistant", "through the release script")
    history.append(CHAT, "summary", "they work on a service called lumi and deploy it with a script")
    bot, _ = await send(config, [], "/context")
    body = texts(bot)
    assert "condensed record" in body
    assert "deploy it with a script" in body


async def test_doctor_command_runs_checks(config) -> None:
    bot, _ = await send(config, [], "/doctor")
    body = texts(bot)
    assert "project root" in body
    assert "shell policy" in body


async def test_doctor_reports_unset_context7_key(config, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    bot, _ = await send(config, [], "/doctor")
    body = texts(bot)
    # The env-var line for context7 should be present and flagged as unset.
    assert "CONTEXT7_API_KEY" in body


async def test_doctor_reports_a_set_context7_key(config, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    bot, _ = await send(config, [], "/doctor")
    body = texts(bot)
    assert "CONTEXT7_API_KEY" in body
    assert "set" in body.lower()


async def test_reload_command(config) -> None:
    (config.personality_file).write_text("You are a lighthouse keeper.\n", encoding="utf-8")
    bot, _ = await send(config, [], "/reload")
    assert "reloaded" in texts(bot)


async def test_reload_picks_up_the_new_personality(config) -> None:
    await send(config, [], "/reload")
    personality = Personality.load(config.personality_file)
    assert "lighthouse keeper" not in personality.text  # the file was restored by the fixture


async def test_reset_clears_the_transcript(config) -> None:
    bot, agent = await send(config, [make_reply("one")], "hello")
    assert History(config.history_dir).read(CHAT)

    bot, _ = await send(config, [make_reply("two")], "/reset", message_id=2)
    assert "cleared this conversation" in texts(bot)


async def test_unknown_command_is_explained(config) -> None:
    bot, _ = await send(config, [], "/nonsense")
    body = texts(bot)
    assert "don't know" in body
    assert "/help" in body


async def test_stranger_gets_one_clear_sentence(config) -> None:
    """A stranger learns nothing about what this bot can reach."""
    bot, _ = await send(config, [], "/tools", user_id=STRANGER)
    body = texts(bot)
    assert "not whitelisted" in body.lower()
    for leak in ("run_shell", "test-model", "workspace", "MEMORY.md"):
        assert leak not in body, f"{leak!r} leaked to a stranger"


async def test_unsupported_media_is_explained(config) -> None:
    """A sticker is neither text nor an image the model can read."""
    application, bot, _ = build(config, [])
    update = Update.de_json(
        {
            "update_id": 7,
            "message": {
                "message_id": 7,
                "date": int(time.time()),
                "chat": {"id": CHAT, "type": "private"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "Owner"},
                "sticker": {
                    "file_id": "abc", "file_unique_id": "u1",
                    "width": 1, "height": 1, "type": "regular", "is_animated": False,
                    "is_video": False,
                },
            },
        },
        bot,
    )
    await application.initialize()
    try:
        await application.process_update(update)
    finally:
        await application.shutdown()
    assert "read text and images only" in texts(bot).lower()


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


def test_application_registers_every_handler(config) -> None:
    application = build_application(config)
    assert application.bot_data["agent"] is not None
    assert application.bot_data["memory"] is not None
    assert application.bot_data["personality"] is not None
    assert application.bot_data["history"] is not None
    # handlers is a dict of {group: [handlers]}
    assert len(application.handlers[0]) >= 20


def test_owner_filter_is_applied_when_an_owner_is_set(config) -> None:
    from telegram.ext import CommandHandler as CH

    application = build_application(config)
    handlers = application.handlers[0]
    # CommandHandler.commands is a frozenset in python-telegram-bot v22.
    names = {next(iter(h.commands)) for h in handlers if isinstance(h, CH) and h.commands}
    assert {"start", "help", "run", "search", "memory", "doctor"} <= names


def test_the_stranger_handler_is_registered_last(config) -> None:
    """Whoever is not the owner must be handled after every owner handler."""
    from telegram.ext import MessageHandler as MH

    handlers = build_application(config).handlers[0]
    last = handlers[-1]
    assert isinstance(last, MH)
    assert "on_stranger" in repr(last)


def test_builder_needs_a_token(config, monkeypatch) -> None:
    from lumi.config import ConfigError

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        build_application(config)


# --------------------------------------------------------------------------- #
# network settings (connect timeout, proxy, bootstrap retries)
# --------------------------------------------------------------------------- #


def _request_kwargs(application) -> list[dict]:
    """The httpx kwargs of both request objects.

    Bot._request is (get_updates_request, regular_request); the getUpdates
    request is what bootstrap and polling go through. Both are configured
    identically, so the tests do not care about the order.
    """
    return [req._client_kwargs for req in application.bot._request]


def test_connect_timeout_is_applied_to_both_requests(config) -> None:
    """PTB's stock 5s connect timeout aborts startup on a slow network."""
    config.bot.connect_timeout = 20.0
    for kwargs in _request_kwargs(build_application(config)):
        assert kwargs["timeout"].connect == 20.0


def test_proxy_is_applied_to_both_requests(config) -> None:
    config.bot.proxy_url = "socks5://127.0.0.1:9050"
    for kwargs in _request_kwargs(build_application(config)):
        assert kwargs["proxy"] == "socks5://127.0.0.1:9050"


def test_no_proxy_by_default(config) -> None:
    for kwargs in _request_kwargs(build_application(config)):
        assert kwargs["proxy"] is None


def test_a_test_injected_bot_keeps_its_own_request_objects(config) -> None:
    """The builder's network settings must not touch an injected Bot."""
    config.bot.connect_timeout = 20.0
    for kwargs in _request_kwargs(build_application(config, bot=FakeBot())):
        assert kwargs["timeout"].connect == 5.0  # PTB default, untouched


# --------------------------------------------------------------------------- #
# group vs private chat routing
# --------------------------------------------------------------------------- #
#
# The bot username comes from ``FakeBot.get_me``: "lumi_test_bot". All the
# group tests use a supergroup with chat id GROUP_CHAT and check whether the
# owner message reaches the agent (or stays silent under mention mode).
GROUP_CHAT = -1001


def _group_text_update(bot: Bot, text: str, **kwargs: Any) -> Update:
    return make_update(bot, text, chat_id=GROUP_CHAT, chat_type="supergroup", **kwargs)


async def test_private_chat_replies_to_every_message(config) -> None:
    """Private chats are unaffected — current behaviour."""
    bot, _ = await send(config, [make_reply("hi back")], "hi")
    assert "hi back" in texts(bot)


async def test_group_chat_silent_without_mention(config) -> None:
    """Group chat, plain text, no mention → the bot does not reply at all."""
    bot, agent = await send(
        config,
        [make_reply("this should never be sent")],
        "hello group",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
    )
    assert bot.sent == []
    # The model was never called — gating happens before the agent loop.
    assert agent.llm.calls == []


async def test_group_chat_replies_when_mentioned(config) -> None:
    """An @botname mention in the text opens the gate."""
    mention_text = "hi @lumi_test_bot what's the weather"
    entities = [
        {"type": "mention", "offset": 3, "length": len("@lumi_test_bot")},
    ]
    bot, agent = await send(
        config,
        [make_reply("sunny")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=entities,
    )
    assert "sunny" in texts(bot)
    assert agent.llm.calls


async def test_group_chat_replies_to_a_reply_to_the_bot(config) -> None:
    """Replying to a message from the bot is treated as addressing it."""
    reply_to_bot = {
        "message_id": 99,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {
            "id": 12345,
            "is_bot": True,
            "first_name": "Lumi",
            "username": "lumi_test_bot",
        },
        "text": "an earlier bot message",
    }
    bot, _ = await send(
        config,
        [make_reply("got it")],
        "thanks!",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_bot,
    )
    assert "got it" in texts(bot)


async def test_group_chat_does_not_reply_to_a_reply_to_a_human(config) -> None:
    """Replying to a non-bot message is not enough — the bot would spam the group."""
    reply_to_human = {
        "message_id": 99,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
        "text": "an earlier human message",
    }
    bot, _ = await send(
        config,
        [make_reply("should never fire")],
        "ok",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_human,
    )
    assert bot.sent == []


async def test_group_chat_slash_command_targeting_the_bot(config) -> None:
    """/help@lumi_test_bot in a group runs the command — it names the bot."""
    bot, _ = await send(config, [], "/help@lumi_test_bot", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert "confirmation" in texts(bot).lower()


async def test_group_chat_plain_slash_command_is_silent(config) -> None:
    """Plain /help in a group is dropped — Telegram routes that elsewhere."""
    bot, _ = await send(config, [], "/help", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert bot.sent == []


async def test_group_chat_slash_command_targeting_another_bot_is_silent(config) -> None:
    """/help@some_other_bot must not fire this bot's handlers."""
    bot, _ = await send(config, [], "/help@some_other_bot", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert bot.sent == []


async def test_group_chat_unknown_command_without_address_is_silent(config) -> None:
    """Unknown slash commands in a group are dropped too."""
    bot, _ = await send(config, [], "/random_typo", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert bot.sent == []


async def test_group_chat_always_mode_replies_without_mention(config) -> None:
    """With ``group_reply_mode = \"always\"`` every owner message goes through."""
    config.bot.group_reply_mode = "always"
    bot, _ = await send(
        config,
        [make_reply("always on")],
        "hi group",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
    )
    assert "always on" in texts(bot)


async def test_group_chat_off_mode_is_silent_even_with_mention(config) -> None:
    """With ``group_reply_mode = \"off\"`` nothing reaches the bot, ever."""
    config.bot.group_reply_mode = "off"
    mention_text = "hi @lumi_test_bot"
    entities = [{"type": "mention", "offset": 3, "length": len("@lumi_test_bot")}]
    bot, _ = await send(
        config,
        [make_reply("should not fire")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=entities,
    )
    assert bot.sent == []


async def test_always_reply_chats_overrides_the_mode(config) -> None:
    """A chat listed in ``always_reply_chats`` answers even under ``off`` mode."""
    config.bot.group_reply_mode = "off"
    config.bot.always_reply_chats = [str(GROUP_CHAT)]
    bot, _ = await send(
        config,
        [make_reply("lab here")],
        "hi",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
    )
    assert "lab here" in texts(bot)


async def test_private_chat_ignores_always_reply_chats(config) -> None:
    """Private chats always reply, with or without the whitelist."""
    config.bot.group_reply_mode = "off"
    config.bot.always_reply_chats = ["-1009999"]  # not this chat
    bot, _ = await send(config, [make_reply("private reply")], "hi")
    assert "private reply" in texts(bot)


async def test_unknown_command_in_private_chat_still_works(config) -> None:
    """The gating must not break existing behaviour in private chats."""
    bot, _ = await send(config, [], "/typo_here")
    assert "i don't know" in texts(bot).lower()


async def test_non_text_in_group_is_silent(config) -> None:
    """A photo in a group without a mention does not even trigger 'i read text only'."""
    bot, _ = await send_photo_in_group(config)
    assert bot.sent == []


async def test_photo_in_group_with_mention_reaches_the_agent(config) -> None:
    """A captioned photo that mentions the bot is sent to the model."""
    bot, agent = await send_photo_in_group(config, mention_text="@lumi_test_bot")
    assert agent.llm.calls, "a mentioned photo in a group should reach the model"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


async def send_photo_in_group(config, *, mention_text: str | None = None) -> tuple[Any, Agent]:
    """Send a non-text (photo) message into the group chat.

    ``mention_text`` is set as the photo's caption and a matching ``mention``
    entity is added — that is how a real photo-with-caption arrives when
    someone tags the bot.
    """
    application, bot, agent = build(config, [make_reply("a photo of something")])
    await application.initialize()
    caption = mention_text or ""
    caption_entities = (
        [{"type": "mention", "offset": 0, "length": len(caption)}] if caption else None
    )
    update = Update.de_json(
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "date": int(time.time()),
                "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "Owner", "username": "owner"},
                "photo": [{"file_id": "p1", "file_unique_id": "u1", "width": 1, "height": 1}],
                "caption": caption or None,
                "caption_entities": caption_entities,
            },
        },
        bot,
    )
    try:
        await application.process_update(update)
    finally:
        await application.shutdown()
    return bot, agent


# --------------------------------------------------------------------------- #
# whitelist system
# --------------------------------------------------------------------------- #

WHITELISTED_USER = 555
WHITELISTED_GROUP = -2002


async def test_whitelisted_user_can_use_the_bot(config) -> None:
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [make_reply("hello friend")], "hi", user_id=WHITELISTED_USER)
    assert "hello friend" in texts(bot)


async def test_whitelisted_user_cannot_manage_whitelist(config) -> None:
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [], "/whitelist", user_id=WHITELISTED_USER)
    assert "only the owner" in texts(bot).lower()


async def test_non_whitelisted_user_is_still_rejected(config) -> None:
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [make_reply("should never be sent")], "hi", user_id=STRANGER)
    assert "not whitelisted" in texts(bot).lower()
    assert "should never be sent" not in texts(bot)


async def test_whitelisted_group_replies_without_mention(config) -> None:
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    bot, _ = await send(
        config,
        [make_reply("group reply")],
        "hello group",
        chat_id=WHITELISTED_GROUP,
        chat_type="supergroup",
    )
    assert "group reply" in texts(bot)


async def test_non_whitelisted_group_still_requires_mention(config) -> None:
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    bot, agent = await send(
        config,
        [make_reply("should not fire")],
        "hello group",
        chat_id=-9999,
        chat_type="supergroup",
    )
    assert bot.sent == []
    assert agent.llm.calls == []


async def test_owner_can_add_user_to_whitelist(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_user 777")
    assert "added" in texts(bot).lower()
    assert "777" in texts(bot)


async def test_owner_can_remove_user_from_whitelist(config) -> None:
    config.bot.whitelisted_users = ["777"]
    bot, _ = await send(config, [], "/whitelist_remove_user 777")
    assert "removed" in texts(bot).lower()


async def test_owner_can_add_group_to_whitelist(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_group -100123")
    assert "added" in texts(bot).lower()
    assert "-100123" in texts(bot)


async def test_owner_can_remove_group_from_whitelist(config) -> None:
    config.bot.whitelisted_groups = ["-100123"]
    bot, _ = await send(config, [], "/whitelist_remove_group -100123")
    assert "removed" in texts(bot).lower()


async def test_whitelist_command_shows_current_lists(config) -> None:
    config.bot.whitelisted_users = ["555"]
    config.bot.whitelisted_groups = ["-2002"]
    bot, _ = await send(config, [], "/whitelist")
    body = texts(bot)
    assert "whitelist" in body.lower()
    assert "555" in body
    assert "-2002" in body


async def test_whitelist_add_user_requires_id(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_user")
    assert "usage" in texts(bot).lower()


async def test_whitelist_add_group_requires_id(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_group")
    assert "usage" in texts(bot).lower()


async def test_whitelist_add_user_rejects_non_integer(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_user abc")
    assert "not a valid id" in texts(bot).lower()


async def test_whitelist_add_group_rejects_non_integer(config) -> None:
    bot, _ = await send(config, [], "/whitelist_add_group xyz")
    assert "not a valid id" in texts(bot).lower()


async def test_a_new_user_is_allowed_without_a_restart(config) -> None:
    """``/whitelist_add_user`` must take effect on the *running* application.

    The authorisation filter used to be a snapshot of the id list taken once
    when handlers were built, so the command reported "added" while the very
    next message from that user still fell through to "this bot is private" —
    until the process was restarted.
    """
    application, bot, agent = build(config, [make_reply("hello new user")])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/whitelist_add_user 777"))
        assert "added" in texts(bot).lower()
        # Same application, no restart — the new id must pass the filter now.
        await application.process_update(make_update(bot, "hi lumi", user_id=777))
    finally:
        await application.shutdown()

    assert "not whitelisted" not in texts(bot).lower()
    assert "hello new user" in texts(bot)
    assert agent.llm.calls, "the newly whitelisted user should reach the model"


async def test_an_added_whitelist_entry_survives_a_restart(config) -> None:
    """The whitelist must be persisted where the next process reads it back.

    A fresh process has none of the in-memory state, so the add is only real
    if ``build_handlers`` repopulates the config from disk.
    """
    application, bot, _ = build(config, [])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/whitelist_add_user 777"))
    finally:
        await application.shutdown()
    assert "added" in texts(bot).lower()

    # Simulate the restart: no in-memory whitelist, fresh handlers.
    config.bot.whitelisted_users = []
    application, bot, agent = build(config, [make_reply("hello again")])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "hi again", user_id=777))
    finally:
        await application.shutdown()

    assert "not whitelisted" not in texts(bot).lower()
    assert "hello again" in texts(bot)


async def test_saving_the_whitelist_leaves_config_local_toml_alone(config) -> None:
    """Persisting the whitelist must not rewrite the user's config.

    It used to round-trip ``config.local.toml`` through a hand-rolled TOML
    writer that cannot escape newlines: one multi-line value became invalid
    TOML, and the next boot refused to start with a configuration error.
    """
    from lumi.config import load_config

    local = config.root / "config.local.toml"
    local.write_text(
        '[bot]\nstartup_chat_id = """\n42\n"""\n[llm]\ncontext_window = 128000\n',
        encoding="utf-8",
    )
    before = local.read_bytes()

    application, bot, _ = build(config, [])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/whitelist_add_user 777"))
    finally:
        await application.shutdown()

    assert "added" in texts(bot).lower()
    assert local.read_bytes() == before, "config.local.toml must be left alone"
    saved = json.loads(
        (config.root / "data" / "whitelist.json").read_text(encoding="utf-8")
    )
    assert "777" in saved["whitelisted_users"]
    # The file still parses — a ConfigError here means boot is broken.
    assert load_config(config.root).llm.context_window == 128_000


# --------------------------------------------------------------------------- #
# vision support
# --------------------------------------------------------------------------- #

class _FakeFile:
    """A minimal file object that supports download_as_bytearray."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def download_as_bytearray(self, **kwargs: Any) -> bytearray:
        return bytearray(self._data)


class FakePhotoBot(FakeBot):
    """A FakeBot that can serve a real image for photo downloads."""

    def __init__(self, image_bytes: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) -> None:
        super().__init__()
        self._image_bytes = image_bytes

    async def get_file(self, file_id: str, **kwargs: Any) -> Any:
        return _FakeFile(self._image_bytes)


def _make_photo_update(bot: Bot, *, caption: str | None = None, chat_id: int = CHAT,
                       chat_type: str = "private",
                       user_id: int = OWNER, message_id: int = 1) -> Update:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": chat_type, "title": "Lab"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner", "username": "owner"},
        "photo": [
            {"file_id": "small", "file_unique_id": "s1", "width": 90, "height": 90},
            {"file_id": "large", "file_unique_id": "l1", "width": 800, "height": 600},
        ],
    }
    if caption is not None:
        message["caption"] = caption
        # Real Telegram sends a `mention` entity alongside the @handle in the
        # caption; group routing keys off the entity, not the raw text.
        if caption.startswith("@"):
            handle = caption.split(maxsplit=1)[0]
            message["caption_entities"] = [
                {"type": "mention", "offset": 0, "length": len(handle)}
            ]
    return Update.de_json({"update_id": message_id, "message": message}, bot)


async def test_photo_is_sent_to_the_model_as_base64(config) -> None:
    """A photo message triggers a multimodal LLM call with the image embedded."""
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    bot = FakePhotoBot(image_bytes)
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("i see a cat")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(_make_photo_update(bot, caption="what is this?"))
    finally:
        await application.shutdown()

    assert "i see a cat" in texts(bot)
    # The LLM should have received a multimodal message with image content.
    assert agent.llm.calls, "the model should have been called"
    user_messages = [m for m in agent.llm.calls[0] if m.get("role") == "user"]
    assert user_messages, "there should be a user message"
    last_user = user_messages[-1]
    content = last_user["content"]
    assert isinstance(content, list), "content should be a multimodal array"
    types = [part["type"] for part in content]
    assert "text" in types
    assert "image_url" in types
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_photo_without_caption_gets_default_prompt(config) -> None:
    """A photo with no caption gets a default 'what's in this image?' prompt."""
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    bot = FakePhotoBot(image_bytes)
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("looks like a diagram")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(_make_photo_update(bot))
    finally:
        await application.shutdown()

    assert "looks like a diagram" in texts(bot)
    user_messages = [m for m in agent.llm.calls[0] if m.get("role") == "user"]
    last_user = user_messages[-1]
    content = last_user["content"]
    text_part = next(p for p in content if p["type"] == "text")
    assert "what" in text_part["text"].lower()


async def test_photo_caption_reaches_the_model(config) -> None:
    """The caption *is* the question (README: "The caption becomes your question").

    A photo has ``text = None``, so it never matches the text filter and lands
    in the non-text handler — a caption handled only in the text path is
    silently dropped and the model gets the canned default instead.
    """
    bot = FakePhotoBot(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("it is a stack trace")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(
            _make_photo_update(bot, caption="what does this error mean?")
        )
    finally:
        await application.shutdown()

    assert "it is a stack trace" in texts(bot)
    user_messages = [m for m in agent.llm.calls[0] if m.get("role") == "user"]
    content = user_messages[-1]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    assert "what does this error mean?" in text_part["text"], (
        "caption never reached the model"
    )
    assert "what's in this image?" not in text_part["text"]


async def test_photo_download_failure_gracefully(config) -> None:
    """If the photo download fails, the bot says so instead of crashing."""

    class FailingPhotoBot(FakePhotoBot):
        async def get_file(self, file_id: str, **kwargs: Any) -> Any:
            from telegram.error import TelegramError
            raise TelegramError("file unavailable")

    bot = FailingPhotoBot()
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("should never be sent")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(_make_photo_update(bot, caption="what is this?"))
    finally:
        await application.shutdown()

    assert "couldn't download" in texts(bot).lower()
    assert agent.llm.calls == [], "the model should not have been called"


async def test_photo_in_group_with_mention_replies(config) -> None:
    """A photo with a mention in a group triggers a vision reply."""
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    bot = FakePhotoBot(image_bytes)
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("that's a screenshot of code")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(
            _make_photo_update(bot, caption="@lumi_test_bot what is this?",
                             chat_id=GROUP_CHAT, chat_type="supergroup")
        )
    finally:
        await application.shutdown()

    assert "screenshot" in texts(bot)


async def test_photo_in_group_without_mention_is_silent(config) -> None:
    """A photo without a mention in a group is ignored (mention mode)."""
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    bot = FakePhotoBot(image_bytes)
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("should not fire")]),
    )
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(
            _make_photo_update(bot, caption="what is this?",
                               chat_id=GROUP_CHAT, chat_type="supergroup")
        )
    finally:
        await application.shutdown()

    assert bot.sent == []
    assert agent.llm.calls == []


# --------------------------------------------------------------------------- #
# file delivery and document intake
# --------------------------------------------------------------------------- #


async def test_a_model_written_file_is_delivered_as_a_document(config) -> None:
    """The full out path: the model writes with upload:true, the chat gets a file."""
    application, bot, agent = build(
        config,
        [
            make_reply(
                "",
                [
                    (
                        "c1",
                        "files",
                        {
                            "action": "write",
                            "path": "workspace/report.md",
                            "content": "# Report\n\ncontents here",
                            "upload": True,
                        },
                    )
                ],
            ),
            make_reply("here is the report"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "make me a report"))
    finally:
        await application.shutdown()

    assert "here is the report" in texts(bot)
    docs = documents(bot)
    assert len(docs) == 1
    assert docs[0]["document"] == b"# Report\n\ncontents here"
    assert "report.md" in docs[0]["caption"]


async def test_an_upload_action_result_is_delivered(config) -> None:
    """The files.upload action, driven by the model, reaches the chat."""
    (config.shell_cwd / "shell-made.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "files", {"action": "upload", "path": "workspace/shell-made.csv"})]),
            make_reply("sent it"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "send me that csv"))
    finally:
        await application.shutdown()

    docs = documents(bot)
    assert len(docs) == 1
    assert docs[0]["document"] == b"a,b\n1,2\n"


async def test_a_plain_turn_sends_no_document(config) -> None:
    bot, _ = await send(config, [make_reply("just words")], "hi")
    assert documents(bot) == []


async def test_the_document_caption_is_plain_and_identifies_the_file(config) -> None:
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "files", {"action": "upload", "path": "workspace/notes.md"})]),
            make_reply("ok"),
        ],
    )
    (config.shell_cwd / "notes.md").write_text("note", encoding="utf-8")
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "send notes"))
    finally:
        await application.shutdown()

    docs = documents(bot)
    assert docs, "the file must be delivered"
    caption = docs[0]["caption"]
    assert "notes.md" in caption
    assert "from files" in caption, "the origin names the tool that produced it"


async def test_a_failed_delivery_degrades_to_a_path(config, monkeypatch) -> None:
    """If reply_document fails, the owner still learns where the file lives."""
    from telegram.error import TelegramError

    class NoSendBot(FakeBot):
        async def send_document(self, chat_id: Any, document: Any = None, **kwargs: Any) -> Any:
            raise TelegramError("attachment rejected")

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(
            [
                make_reply("", [("c1", "files", {"action": "upload", "path": "workspace/doomed.md"})]),
                make_reply("here you go"),
            ]
        ),
    )
    bot = NoSendBot()
    application = build_application(config, agent, bot=bot)
    (config.shell_cwd / "doomed.md").write_text("x", encoding="utf-8")
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "send it"))
    finally:
        await application.shutdown()

    body = texts(bot)
    assert "couldn't attach" in body
    assert "workspace/doomed.md" in body
    assert "here you go" in body, "the turn itself still succeeded"


class FakeDocBot(FakeBot):
    """A FakeBot whose get_file serves real text bytes for document downloads."""

    def __init__(self, data: bytes = b"hello world") -> None:
        super().__init__()
        self._data = data

    async def get_file(self, file_id: str, **kwargs: Any) -> Any:
        return _FakeFile(self._data)


async def test_a_document_message_is_stored_and_announced(config) -> None:
    """The in path: an owner uploads a file, it lands in workspace/uploads/."""
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("it's a shopping list")]),
    )
    bot = FakeDocBot(b"hello world")
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(
            make_document_update(bot, caption="what did i write here?")
        )
    finally:
        await application.shutdown()

    body = texts(bot)
    assert "workspace/uploads/42/notes.txt" in body
    assert "11 bytes" in body

    stored = config.root / "workspace" / "uploads" / "42" / "notes.txt"
    assert stored.is_file()
    assert stored.read_bytes() == b"hello world"

    # The model got a prompt naming the file and the caption.
    assert agent.llm.calls, "the model should be asked about the document"
    last_user = [m for m in agent.llm.calls[0] if m["role"] == "user"][-1]
    prompt = last_user["content"]
    assert isinstance(prompt, str)
    assert "workspace/uploads/42/notes.txt" in prompt
    assert "what did i write here?" in prompt


async def test_a_document_without_a_caption_gets_a_default_prompt(config) -> None:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM([make_reply("it's a list")]),
    )
    bot = FakeDocBot()
    application = build_application(config, agent, bot=bot)
    await application.initialize()
    try:
        await application.process_update(make_document_update(bot))
    finally:
        await application.shutdown()

    last_user = [m for m in agent.llm.calls[0] if m["role"] == "user"][-1]
    assert "what is this file" in last_user["content"].lower()


async def test_a_disallowed_document_is_refused_politely(config) -> None:
    application, bot, agent = build(config, [make_reply("never")])
    await application.initialize()
    try:
        await application.process_update(
            make_document_update(bot, file_name="program.exe")
        )
    finally:
        await application.shutdown()

    assert "can't take that file" in texts(bot)
    assert agent.llm.calls == []


async def test_a_document_download_failure_explains_itself(config) -> None:
    from telegram.error import TelegramError

    application, bot, agent = build(config, [make_reply("never")])
    await application.initialize()
    try:
        async def fail_get_file(self, file_id: str, **kwargs: Any) -> Any:
            raise TelegramError("file gone")

        original = FakeBot.get_file
        FakeBot.get_file = fail_get_file
        try:
            await application.process_update(make_document_update(bot))
        finally:
            FakeBot.get_file = original
    finally:
        await application.shutdown()

    assert "couldn't download that file" in texts(bot)
    assert agent.llm.calls == []


async def test_document_intake_in_a_group_without_mention_is_silent(config) -> None:
    """Group rules apply to documents exactly as to text and photos."""
    application, bot, agent = build(config, [make_reply("never")])
    await application.initialize()
    try:
        await application.process_update(
            make_document_update(bot, chat_id=GROUP_CHAT, chat_type="supergroup")
        )
    finally:
        await application.shutdown()

    assert bot.sent == []
    assert agent.llm.calls == []


async def test_document_intake_respects_the_whitelist(config) -> None:
    application, bot, agent = build(config, [make_reply("never")])
    await application.initialize()
    try:
        await application.process_update(
            make_document_update(bot, user_id=STRANGER)
        )
    finally:
        await application.shutdown()

    assert "not whitelisted" in texts(bot).lower()
    assert agent.llm.calls == []
    stored = config.root / "workspace" / "uploads"
    assert not stored.exists(), "a stranger's file must not be stored"


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


THINKING = "first I checked the spelling, then the plural, so four"


async def test_the_trace_is_not_spelled_out_in_the_chat(config) -> None:
    """The whole point of collapsing it: the answer reads the same whether or
    not anyone ever opens the trace."""
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert THINKING not in texts(bot)
    assert "four" in texts(bot)


async def test_the_collapsed_line_comes_before_the_answer(config) -> None:
    """The order it happened in: the model thought, then it spoke."""
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert "🧠" in texts(bot)
    assert "🧠" in bot.sent[0]["text"]
    assert bot.sent[1]["text"] == "four"


async def test_the_collapsed_line_carries_a_button_that_fits_callback_data(config) -> None:
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
    button = keyboard.inline_keyboard[0][0]
    assert button.text == "show thinking"
    assert button.callback_data.startswith(CB_THINK)
    assert len(button.callback_data) <= 64  # Telegram's hard limit


async def test_the_collapsed_line_reports_the_effort(config) -> None:
    bot, _ = await send(
        config,
        [make_reply("four", reasoning=THINKING, reasoning_tokens=1200)],
        "how many r's?",
    )
    assert "1,200 reasoning tokens" in texts(bot)


async def test_no_button_when_the_model_did_not_reason(config) -> None:
    bot, _ = await send(config, [make_reply("four")], "how many r's?")
    assert "🧠" not in texts(bot)
    assert not [m for m in bot.sent if m.get("reply_markup")]


async def test_traces_can_be_turned_off(config) -> None:
    config.llm.show_reasoning = False
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert "🧠" not in texts(bot)


async def test_tapping_show_edits_the_message_in_place(config) -> None:
    """In place, not as a new message: a chat scrolled back to an old turn
    should not gain a second copy of the trace below it."""
    application, bot, _ = build(config, [make_reply("four", reasoning=THINKING)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        posted = len(bot.sent)  # the answer and the collapsed stub
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token))
    finally:
        await application.shutdown()

    assert len(bot.sent) == posted  # tapping added no new message
    assert THINKING in bot.edits[0]["text"]
    assert "thinking" in bot.edits[0]["text"].lower()


async def test_a_stale_thinking_tap_still_expands(config) -> None:
    """Same staleness, other button: the expired toast must not prevent the
    trace from being edited into the message."""
    application, bot, _ = build(
        config, [make_reply("four", reasoning=THINKING)], bot=StaleToastBot()
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token))
    finally:
        await application.shutdown()

    assert THINKING in bot.edits[0]["text"], "the expired toast ate the tap"


async def test_the_expanded_trace_offers_a_way_back(config) -> None:
    application, bot, _ = build(config, [make_reply("four", reasoning=THINKING)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token))
        keyboard = bot.edits[0]["reply_markup"]
        assert keyboard.inline_keyboard[0][0].text == "hide"
    finally:
        await application.shutdown()


async def test_tapping_hide_collapses_it_again(config) -> None:
    application, bot, _ = build(config, [make_reply("four", reasoning=THINKING)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token))
        await application.process_update(make_callback(bot, f"{CB_THINK}hide:"))
    finally:
        await application.shutdown()

    assert THINKING not in bot.edits[-1]["text"]


async def test_an_expired_trace_does_not_edit_the_message(config) -> None:
    """Editing it to something meaningless looks like a broken button; saying so
    in the toast is honest and leaves the message alone."""
    application, bot, _ = build(config, [make_reply("four", reasoning=THINKING)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        await application.process_update(make_callback(bot, f"{CB_THINK}show:deadbeef1234"))
    finally:
        await application.shutdown()

    assert bot.edits == []


async def test_a_stranger_cannot_read_someone_elses_trace(config) -> None:
    application, bot, _ = build(config, [make_reply("four", reasoning=THINKING)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token, user_id=STRANGER))
    finally:
        await application.shutdown()

    assert THINKING not in texts(bot)
    assert bot.edits == []


async def test_a_malformed_think_button_is_ignored(config) -> None:
    application, bot, _ = build(config, [])
    await application.initialize()
    try:
        await application.process_update(make_callback(bot, CB_THINK))
    finally:
        await application.shutdown()
    assert bot.edits == []


async def test_a_tool_turn_still_shows_one_collapsed_line(config) -> None:
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})], reasoning="run it"),
            make_reply("it said hi", reasoning="it worked"),
        ],
        "run echo hi",
    )
    assert texts(bot).count("🧠") == 1


async def test_the_stub_reports_the_models_time_not_the_turns(config) -> None:
    """On a turn that spent most of itself waiting on a tool, "thought for Ns"
    would be a lie — the number has to be the model's."""
    from lumi.bot import thinking_stub

    result = TurnResult(text="done", reasoning="x" * 10)
    result.elapsed = 30.0
    result.thinking_seconds = 4.0
    assert "thought for 4s" in thinking_stub(result)

    # And it falls back to the turn total when nothing timed the model.
    result.thinking_seconds = 0.0
    assert "thought for 30s" in thinking_stub(result)


async def test_a_turn_paused_for_approval_still_offers_its_trace(config) -> None:
    """Both keyboards on the same turn, and neither shadowing the other."""
    bot, _ = await send(
        config,
        [
            make_reply(
                "",
                [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})],
                reasoning="this deletes things",
            )
        ],
        "clean up",
    )
    assert "needs your approval" in texts(bot)
    assert "🧠" in texts(bot)
    assert len([m for m in bot.sent if m.get("reply_markup")]) == 2


async def test_a_huge_trace_is_truncated_rather_than_spamming_the_chat(config) -> None:
    application, bot, _ = build(config, [make_reply("four", reasoning="x" * 40_000)])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "how many r's?"))
        stub = next(m for m in bot.sent if m.get("reply_markup"))
        token = stub["reply_markup"].inline_keyboard[0][0].callback_data
        await application.process_update(make_callback(bot, token))
    finally:
        await application.shutdown()

    assert len(bot.edits) == 1
    assert len(bot.edits[0]["text"]) < 4096  # still one Telegram message
    assert "truncated" in bot.edits[0]["text"]


# --------------------------------------------------------------------------- #
# the store behind the button
# --------------------------------------------------------------------------- #


def test_a_stored_trace_comes_back() -> None:
    store = TraceStore()
    assert store.get(store.put("the thinking")) == "the thinking"


def test_an_unknown_token_is_simply_absent() -> None:
    assert TraceStore().get("nope") is None


def test_tokens_are_short_enough_for_callback_data() -> None:
    assert len(TraceStore().put("x")) <= 64


def test_an_expired_trace_is_gone() -> None:
    store = TraceStore(ttl=0.0)
    token = store.put("the thinking")
    assert store.get(token) is None


def test_the_store_is_bounded() -> None:
    """A tap that never comes must not keep a model's reasoning alive forever."""
    store = TraceStore(limit=3)
    tokens = [store.put(f"trace {i}") for i in range(10)]
    assert len(store) == 3
    assert store.get(tokens[-1]) == "trace 9"
    assert store.get(tokens[0]) is None  # the oldest went first


def test_stale_traces_are_pruned_on_the_way_in() -> None:
    store = TraceStore(ttl=60.0)
    token = store.put("old")
    store._ttl = 0.0  # as if the entry had aged out
    store.put("new")
    assert store.get(token) is None
    assert len(store) == 1


# --------------------------------------------------------------------------- #
# /reasoning
# --------------------------------------------------------------------------- #


async def test_reasoning_reports_the_setup(config) -> None:
    bot, _ = await send(config, [], "/reasoning")
    body = texts(bot)
    assert "reasoning" in body
    assert "off" in body


async def test_reasoning_off_hides_the_trace(config) -> None:
    bot, _ = await send(config, [], "/reasoning off")
    assert "trace in chat: off" in texts(bot)

    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert "🧠" not in texts(bot)


async def test_reasoning_on_brings_the_trace_back(config) -> None:
    application, bot, _ = build(config, [])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/reasoning off"))
        bot._sent.clear()
        await application.process_update(make_update(bot, "/reasoning on"))
    finally:
        await application.shutdown()

    assert "trace in chat: on" in texts(bot)


async def test_reasoning_rejects_nonsense(config) -> None:
    bot, _ = await send(config, [], "/reasoning maybe")
    assert "usage" in texts(bot).lower()


async def test_status_mentions_reasoning(config) -> None:
    bot, _ = await send(config, [], "/status")
    assert "reasoning: off" in texts(bot)


async def test_help_mentions_the_thinking_toggle(config) -> None:
    bot, _ = await send(config, [], "/help")
    assert "/reasoning" in texts(bot)
    assert "show thinking" in texts(bot)


# --------------------------------------------------------------------------- #
# /config and /env (owner-only introspection)
# --------------------------------------------------------------------------- #


async def test_config_shows_the_resolved_settings(config) -> None:
    bot, _ = await send(config, [], "/config")
    body = texts(bot)
    assert "test-model" in body
    assert "firecrawl → exa → tavily → mcp" in body
    assert "group mode: mention" in body
    assert "workspace" in body


async def test_config_reflects_live_toggles(config) -> None:
    """The view reads the same config object the handlers mutate."""
    config.llm.show_reasoning = False
    config.bot.group_reply_mode = "always"
    bot, _ = await send(config, [], "/config")
    body = texts(bot)
    assert "trace in chat: off" in body
    assert "group mode: always" in body


async def test_config_memory_cap_is_the_limit_the_agent_uses(config) -> None:
    """The displayed memory cap must be ``Agent.memory_limit()``, verbatim.

    Re-deriving it in the renderer let display and behaviour drift (it used to
    show 3/8 of the window while the prompt applied an eighth of the budget
    times chars_per_token).
    """
    bot, agent = await send(config, [], "/config")
    assert f"{agent.memory_limit():,} chars" in texts(bot)


async def test_env_masks_long_keys(config, monkeypatch) -> None:
    secret = "sk-proj-abcdef1234567890abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    bot, _ = await send(config, [], "/env")
    body = texts(bot)
    assert "sk…cdef" in body
    assert secret not in body
    assert "OPENAI_API_KEY" in body


async def test_env_describes_short_keys_by_length(config) -> None:
    """A short key must not be half-revealed by the mask."""
    bot, _ = await send(config, [], "/env")
    body = texts(bot)
    assert "(len 8)" in body  # conftest sets OPENAI_API_KEY=test-key
    assert "test-key" not in body


async def test_env_reports_a_key_pool(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key-one, key-two, key-three")
    bot, _ = await send(config, [], "/env")
    body = texts(bot)
    assert "3 keys" in body
    assert "fallback" in body
    assert "key-two" not in body


async def test_env_reports_unset_variables_honestly(config, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    bot, _ = await send(config, [], "/env")
    body = texts(bot)
    assert "TAVILY_API_KEY</code> — unset" in body
    assert "keyless" in body  # unset tavily is normal, not an error
    assert "EXA_API_KEY</code> — unset" in body


async def test_env_lists_lumi_overrides(config, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__MODEL", "override-model")
    bot, _ = await send(config, [], "/env")
    body = texts(bot)
    assert "LUMI__LLM__MODEL" in body
    assert "override-model" in body


async def test_env_says_keys_need_a_restart(config) -> None:
    bot, _ = await send(config, [], "/env")
    assert "restart" in texts(bot).lower()


async def test_help_advertises_the_owner_introspection_commands(config) -> None:
    bot, _ = await send(config, [], "/help")
    body = texts(bot)
    assert "/config" in body
    assert "/env" in body
    assert "masked" in body


async def test_env_and_config_are_owner_only(config) -> None:
    """A whitelisted user is allowed the bot but not the backend view."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [], "/config", user_id=WHITELISTED_USER)
    assert "only the owner can see that" in texts(bot).lower()

    bot, _ = await send(config, [], "/env", user_id=WHITELISTED_USER, message_id=2)
    assert "only the owner can see that" in texts(bot).lower()


async def test_env_leaks_nothing_to_a_stranger(config) -> None:
    bot, _ = await send(config, [], "/env", user_id=STRANGER)
    body = texts(bot)
    assert "not whitelisted" in body.lower()
    for leak in ("OPENAI_API_KEY", "test-key", "sk…"):
        assert leak not in body, f"{leak!r} leaked to a stranger"
