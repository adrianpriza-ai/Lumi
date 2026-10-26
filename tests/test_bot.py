"""The Telegram layer, end to end.

A fake Bot records what would have been sent, and the scripted LLM stands in for
the model. That makes it possible to assert on the whole path: a message arrives,
is authorised, reaches the agent, and comes back as Telegram messages — including
the Confirm/Cancel keyboard for a risky command.
"""

from __future__ import annotations

import base64
import io
import json
import time
import tomllib
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import FakeLLM, backdate, make_reply, make_store
from telegram import Bot, Chat, InaccessibleMessage, Message, TextQuote, Update, User
from telegram.error import TelegramError

from lumi.agent import Agent
from lumi.bot import (
    CB_NO,
    CB_OK,
    StepFeed,
    _quoted_prefix,
    _write_toml,
    build_application,
    build_handlers,
    stop_signal_handler,
)
from lumi.chatlog import CHATLOG
from lumi.memory import History
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
        # A real Bot returns the sent Message; the step feed needs its id to
        # edit in place, so the stub has to hand one back too.
        message = Message(
            message_id=len(self._sent),
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type="private"),
            text=text,
        )
        message.set_bot(self)
        return message

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


class FlakySendBot(FakeBot):
    """A FakeBot whose first send of a marked message fails like a transient
    Telegram error, then succeeds — a retryable outage, not a dead chat.

    The regression this exists for: the live ``on_message`` hook swallowed
    ``TelegramError`` inside ``reply()``, so a failed live send disappeared
    before the agent's ``TurnResult.messages`` fallback could see it. With
    strict delivery the failure surfaces, the agent keeps the message, and the
    turn-tail flush retries it.
    """

    def __init__(self, needle: str) -> None:
        super().__init__()
        self._needle = needle
        self._failed = False

    async def send_message(self, chat_id: Any, text: str, **kwargs: Any) -> Any:
        if not self._failed and self._needle in text:
            self._failed = True
            raise TelegramError("simulate: Telegram hiccup")
        return await super().send_message(chat_id, text, **kwargs)


class DeadBot(FakeBot):
    """A FakeBot where every send fails — the chat is unreachable."""

    async def send_message(self, chat_id: Any, text: str, **kwargs: Any) -> Any:
        raise TelegramError("chat is gone")


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


def make_callback(bot: Bot, data: str, *, chat_id: int = CHAT, user_id: int = OWNER,
                  chat_type: str = "private") -> Update:
    """An update for a tapped inline button.

    ``chat_type`` matters for authorisation: ``_is_allowed`` admits *any* member
    of a whitelisted group, but only when the callback's chat really is a group
    — a tap defaulting to "private" would silently never exercise that path.
    """
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
                    "chat": {"id": chat_id, "type": chat_type},
                    "text": "stub",
                },
            },
        },
        bot,
    )


def build(config, replies, bot: FakeBot | None = None) -> tuple[Any, FakeBot, Agent]:
    memory = make_store(config)
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


def prompts(agent: Agent) -> str:
    """Every message the agent handed the model, flattened for assertions."""
    return "\n".join(
        str(message.get("content", "")) for call in agent.llm.calls for message in call
    )


def documents(bot: FakeBot) -> list[dict[str, Any]]:
    """Every document the bot tried to send (payload bytes recorded)."""
    return [message for message in bot.sent if "document" in message]


@pytest.fixture(autouse=True)
def empty_room():
    """The room log is process-wide; every test starts with it empty."""
    CHATLOG.clear()
    yield
    CHATLOG.clear()


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


async def test_commands_are_invisible_to_strangers(config) -> None:
    bot, _ = await send(config, [], "/status", user_id=STRANGER)
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


async def test_start_greets_without_the_backend_details(config) -> None:
    """``/start`` is a glance: greeting + commands — no model or tool dump."""
    bot, _ = await send(config, [], "/start")
    body = texts(bot)
    assert "i'm up" in body
    assert "/memory" in body
    assert "test-model" not in body  # the model line is gone
    assert "run_shell" not in body  # so is the tool inventory


async def test_start_lists_the_commands(config) -> None:
    bot, _ = await send(config, [], "/start")
    body = texts(bot)
    for command in ("/memory", "/doctor", "/status"):
        assert command in body
    for gone in ("/run", "/search", "/fetch", "/ask", "/tools", "/help"):
        assert gone not in body
    # Owner commands live behind /own_help, not in the greeting.
    assert "/config" not in body
    assert "/whitelist" not in body
    for owner_only in ("/forget", "/reload", "/reset"):
        assert owner_only not in body


async def test_start_advertises_the_approval_flow(config) -> None:
    bot, _ = await send(config, [], "/start")
    assert "confirmation" in texts(bot).lower()


async def test_help_points_at_start(config) -> None:
    """/help is no longer the list — it redirects instead of erroring."""
    bot, _ = await send(config, [], "/help")
    body = texts(bot)
    assert "/start" in body
    assert "/config" not in body  # it is not the old dump


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
# removed commands (they fall through to the unknown-command reply)
# --------------------------------------------------------------------------- #


async def test_removed_commands_are_gone(config) -> None:
    """``/run``, ``/search``, ``/fetch``, ``/ask`` and ``/tools`` no longer
    exist: each gets the unknown-command reply pointing at ``/start``, and no
    model call is made along the way."""
    for command in (
        "/run echo hi",
        "/search python",
        "/fetch https://example.com",
        "/ask what is the answer",
        "/tools",
    ):
        bot, agent = await send(config, [], command)
        body = texts(bot)
        assert "don't know" in body, command
        assert "/start" in body, command
        assert agent.llm.calls == [], command


# --------------------------------------------------------------------------- #
# approvals
# --------------------------------------------------------------------------- #


async def test_risky_command_renders_a_keyboard(config) -> None:
    bot, _ = await send(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
        "clean up the workspace",
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
        "clean up the workspace",
    )
    keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
    approve, cancel = keyboard.inline_keyboard[0]
    assert approve.callback_data.startswith(CB_OK)
    assert cancel.callback_data.startswith(CB_NO)
    assert len(approve.callback_data) <= 64  # Telegram's hard limit
    assert len(cancel.callback_data) <= 64


async def test_tapping_run_it_actually_runs_the_command(config) -> None:
    """The whole approval flow, through the button rather than plain prose.

    A handler that is defined but never registered is invisible to every other
    test here: the keyboard is drawn and the callback data is well formed, and
    the tap still goes nowhere. Only driving a real callback query catches it.

    Scripted: the first reply is the model asking for the risky run — the turn
    that ends in the keyboard — and the second is its narration *after* the
    approval, because the tool itself runs on the tap, outside any model turn.
    """
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})]),
            make_reply("it printed approved"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "clean up the workspace"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(make_callback(bot, approve.callback_data))
    finally:
        await application.shutdown()

    assert "it printed approved" in texts(bot)
    assert not agent.conversation(42).pending  # the queue was drained


async def test_tapping_cancel_declines_the_command(config) -> None:
    """A refusal runs nothing — the model's next turn only narrates that."""
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})]),
            make_reply("understood, i left it alone"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "clean up the workspace"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        cancel = keyboard.inline_keyboard[0][1]
        await application.process_update(make_callback(bot, cancel.callback_data))
    finally:
        await application.shutdown()

    assert "left it alone" in texts(bot)
    assert agent.llm.index == 2  # one turn asked, one narrated the refusal


async def test_a_stale_approval_tap_still_resolves(config) -> None:
    """A late ``query.answer()`` must not eat the tap.

    Telegram expires callback queries quickly and a tap can queue behind a long
    agent turn, so the toast raises ``BadRequest: Query is too old...``. The
    approval resolution does not depend on the toast — it must still run.
    """
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})]),
            make_reply("it printed approved"),
        ],
        bot=StaleToastBot(),
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "clean up the workspace"))
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
# approval authority (owner only)
# --------------------------------------------------------------------------- #
#
# ``_is_allowed`` answers "may this person talk to the bot" — it lets in the
# owner, whitelisted users and every member of a whitelisted group. Approval is
# a separate, owner-reserved decision (lumi/agent.py, docs/policy/policy.md),
# so every way of resolving a pending action re-checks ``_is_owner`` itself.

RISKY_TURN = ("c1", "run_shell", {"command": "rm -rf workspace/tmp"})


async def test_a_group_member_cannot_tap_run_it(config) -> None:
    """A whitelisted room's member taps Confirm: nothing may run.

    The regression this exists for: the button handler checked only
    ``_is_allowed``, so in a whitelisted group any member could approve a
    pending ``run_shell`` that the owner was asked to review.
    """
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    config.bot.always_reply_chats = [str(WHITELISTED_GROUP)]
    application, bot, agent = build(
        config,
        [make_reply("", [RISKY_TURN]), make_reply("should never narrate an approval")],
    )
    await application.initialize()
    try:
        await application.process_update(
            make_update(
                bot, "clean up the workspace",
                chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER,
            )
        )
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(
            make_callback(
                bot, approve.callback_data,
                chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER,
            )
        )
    finally:
        await application.shutdown()

    assert agent.conversation(WHITELISTED_GROUP).pending, "the tap must not resolve the action"
    assert "should never narrate an approval" not in texts(bot)
    # PTB forwards the toast text as a keyword, not a positional arg.
    toast = " ".join(
        str(t["args"][0]) if t["args"] else str(t.get("text", "")) for t in bot.toasts
    )
    assert "only the owner" in toast


async def test_a_whitelisted_user_cannot_tap_run_it_in_their_own_chat(config) -> None:
    """Same gate for a whitelisted user's private chat, not just groups."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    application, bot, agent = build(
        config,
        [make_reply("", [RISKY_TURN]), make_reply("should never narrate an approval")],
    )
    await application.initialize()
    try:
        await application.process_update(
            make_update(bot, "clean up the workspace", chat_id=WHITELISTED_USER,
                        user_id=WHITELISTED_USER)
        )
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(
            make_callback(bot, approve.callback_data,
                          chat_id=WHITELISTED_USER, user_id=WHITELISTED_USER)
        )
    finally:
        await application.shutdown()

    assert agent.conversation(WHITELISTED_USER).pending
    assert "should never narrate an approval" not in texts(bot)


async def test_a_group_member_cannot_approve_or_deny_by_command(config) -> None:
    """``/approve`` and ``/deny`` are owner-only, same as the button."""
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    config.bot.always_reply_chats = [str(WHITELISTED_GROUP)]
    application, bot, agent = build(
        config,
        [make_reply("", [RISKY_TURN]), make_reply("should never narrate an approval")],
    )
    await application.initialize()
    try:
        await application.process_update(
            make_update(
                bot, "clean up the workspace",
                chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER,
            )
        )
        for command, message_id in (("/approve", 8), ("/deny", 9)):
            await application.process_update(
                make_update(
                    bot, command, message_id=message_id,
                    chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER,
                )
            )
    finally:
        await application.shutdown()

    body = texts(bot).lower()
    assert body.count("only the owner") == 2, body
    assert agent.conversation(WHITELISTED_GROUP).pending
    assert "should never narrate an approval" not in texts(bot)


async def test_the_owner_still_approves_a_pending_action_in_the_group(config) -> None:
    """Positive control: the gate must not be so wide it locks the owner out.

    A member raises the action; the owner resolves it — approval stays a
    review of someone else's request, which is the whole point of the gate.
    """
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    config.bot.always_reply_chats = [str(WHITELISTED_GROUP)]
    application, bot, agent = build(
        config,
        [make_reply("", [RISKY_TURN]), make_reply("it is gone")],
    )
    await application.initialize()
    try:
        await application.process_update(
            make_update(
                bot, "clean up the workspace",
                chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER,
            )
        )
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        await application.process_update(
            make_callback(
                bot, approve.callback_data,
                chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=OWNER,
            )
        )
    finally:
        await application.shutdown()

    assert "it is gone" in texts(bot)
    assert not agent.conversation(WHITELISTED_GROUP).pending


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

    memory = make_store(config).get(CHAT)  # the private chat's own scope
    assert "the owner prefers dark mode" in memory.managed()


async def test_remember_requires_an_argument(config) -> None:
    bot, _ = await send(config, [], "/remember")
    assert "usage" in texts(bot).lower()


async def test_forget_removes_the_newest(config) -> None:
    memory = make_store(config).get(CHAT)
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


async def test_forget_is_owner_only(config) -> None:
    """A whitelisted user may use the bot, but erasing its memory is the
    owner's call — /forget is the human side of the memory tool's asymmetry."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    memory = make_store(config).get(CHAT)
    memory.remember("first fact")

    bot, _ = await send(config, [], "/forget", user_id=WHITELISTED_USER)
    assert "only the owner" in texts(bot).lower()
    assert memory.managed() == ["first fact"]  # nothing was erased


async def test_reset_is_owner_only(config) -> None:
    """Clearing a shared conversation destroys other people's transcript too,
    so it is not a whitelisted user's button to press."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    await send(config, [make_reply("one")], "hello", user_id=WHITELISTED_USER)

    bot, _ = await send(config, [], "/reset", user_id=WHITELISTED_USER, message_id=2)
    assert "only the owner" in texts(bot).lower()
    assert History(config.history_dir).read(CHAT)  # the transcript survives


async def test_reload_is_owner_only(config) -> None:
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [], "/reload", user_id=WHITELISTED_USER)
    assert "only the owner" in texts(bot).lower()


async def test_a_whitelisted_user_can_still_remember(config) -> None:
    """Saving a fact stays open to allowed users — and is stored as what it is,
    a human-typed fact, not fenced as model-authored data."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [], "/remember likes quiet rooms", user_id=WHITELISTED_USER)
    assert "remembered" in texts(bot).lower()

    memory = make_store(config).get(CHAT)
    assert "likes quiet rooms" in memory.managed()
    assert "lumi:model" not in memory.text
    assert "lumi:model-notes" not in memory.for_prompt()


# --------------------------------------------------------------------------- #
# introspection commands
# --------------------------------------------------------------------------- #


async def test_personality_command_shows_the_file(config) -> None:
    bot, _ = await send(config, [], "/personality")
    assert "You are Lumi" in texts(bot)


async def test_status_command(config) -> None:
    bot, _ = await send(config, [], "/status")
    body = texts(bot)
    assert "uptime" in body
    assert "test-model" in body
    assert "run_shell" in body  # tool inventory lives here now that /tools is gone


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
    assert "/start" in body


async def test_stranger_gets_one_clear_sentence(config) -> None:
    """A stranger learns nothing about what this bot can reach."""
    bot, _ = await send(config, [], "/status", user_id=STRANGER)
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
    assert {"start", "help", "own_help", "memory", "doctor"} <= names
    # The removed direct-invocation commands must not be registered either.
    assert not ({"run", "search", "fetch", "ask", "tools"} & names)


def test_the_gate_filter_admits_allowed_principals_not_just_the_owner(config) -> None:
    """The filter in front of every handler answers "may this person talk to
    the bot?", not "is this person the owner?".

    It used to be called ``OwnerFilter`` while returning ``_is_allowed``,
    which read as a much narrower gate than it was — and is how the owner-only
    checks in this module were easy to miss. The name now says what it does,
    and the behaviour is asserted here so the two cannot drift apart again.
    """
    from telegram.ext import MessageHandler

    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    handlers = build_handlers(config)
    text_gate = next(
        h for h in handlers
        if isinstance(h, MessageHandler) and getattr(h.callback, "__name__", "") == "on_text"
    )
    bot = FakeBot()

    # The owner passes.
    assert text_gate.filters.check_update(make_update(bot, "hi", user_id=OWNER))
    # A member of a whitelisted group passes, in that group.
    assert text_gate.filters.check_update(
        make_update(
            bot, "hi", chat_id=WHITELISTED_GROUP, chat_type="supergroup", user_id=STRANGER
        )
    )
    # The same stranger in a private chat does not.
    assert not text_gate.filters.check_update(make_update(bot, "hi", user_id=STRANGER))

    # And the filter is not named after a narrower gate than it implements.
    # The filter class is local to build_handlers, so it is found by walking
    # the composed filter tree rather than by importing it.
    def filter_names(f) -> set[str]:
        found = {type(f).__name__}
        for part in getattr(f, "filters", ()) or ():
            found |= filter_names(part)
        return found

    names = set()
    for handler in handlers:
        gate = getattr(handler, "filters", None)
        if gate is not None:
            names |= filter_names(gate)
    assert "AllowedFilter" in names
    assert "OwnerFilter" not in names


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


async def test_a_mentioned_reply_carries_the_quoted_message_to_the_model(config) -> None:
    """``@bot`` under a reply to someone else: the quote must reach the model.

    The mention opens the gate, but the message being replied to never became
    a turn of this conversation — without folding it into the prompt the model
    is handed "try this" with nothing to point at.
    """
    reply_to_human = {
        "message_id": 99,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
        "text": "the failing test is in tests/test_web.py",
    }
    mention_text = "@lumi_test_bot try to fix this"
    bot, agent = await send(
        config,
        [make_reply("on it")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=[{"type": "mention", "offset": 0, "length": len("@lumi_test_bot")}],
        reply_to_message=reply_to_human,
    )
    assert "on it" in texts(bot)
    assert "the failing test is in tests/test_web.py" in prompts(agent)
    assert "try to fix this" in prompts(agent)


async def test_a_reply_quoting_a_media_message_says_what_it_quoted(config) -> None:
    """A quote with no text (a photo) is named, not silently dropped."""
    reply_to_photo = {
        "message_id": 98,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
        "photo": [{"file_id": "p1", "file_unique_id": "pu1", "width": 90, "height": 60}],
    }
    mention_text = "@lumi_test_bot what is in this"
    _, agent = await send(
        config,
        [make_reply("no idea")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=[{"type": "mention", "offset": 0, "length": len("@lumi_test_bot")}],
        reply_to_message=reply_to_photo,
    )
    assert "photo with no text" in prompts(agent)


def test_an_inaccessible_reply_uses_telegram_quote_excerpt() -> None:
    """A selected quote remains readable when Telegram hides the source message."""
    source = InaccessibleMessage(Chat(id=CHAT, type="private"), message_id=98)
    message = Message(
        message_id=99,
        date=datetime.now(UTC),
        chat=Chat(id=CHAT, type="private"),
        text="please handle this",
        reply_to_message=source,
        quote=TextQuote(text="the important line", position=0, is_manual=True),
    )

    prompt = _quoted_prefix(message)

    assert "the important line" in prompt
    assert "inaccessible" not in prompt


async def test_a_plain_message_gets_no_quote_prefix(config) -> None:
    """Not a reply → the model sees the text exactly as it was sent."""
    _, agent = await send(config, [make_reply("ok")], "just saying hi")
    assert any(message.get("content") == "just saying hi" for message in agent.llm.calls[0])


async def test_a_reply_chain_reaches_the_prompt_oldest_first(config) -> None:
    """A reply to a reply carries every link, in the order they were said.

    "no it isn't" answers the message *above* the quote, so one link is not
    enough — the chain is walked and rendered oldest first.
    """
    oldest = {
        "message_id": 96,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
        "text": "the deploy token is expired",
    }
    middle = {
        "message_id": 97,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": OWNER, "is_bot": False, "first_name": "Owner", "username": "owner"},
        "text": "which token though",
        "reply_to_message": oldest,
    }
    mention_text = "@lumi_test_bot fix it"
    _, agent = await send(
        config,
        [make_reply("on it")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=[{"type": "mention", "offset": 0, "length": len("@lumi_test_bot")}],
        reply_to_message=middle,
    )
    prompt = prompts(agent)
    assert "reply chain" in prompt
    assert prompt.index("the deploy token is expired") < prompt.index("which token though")
    assert "fix it" in prompt


async def test_the_reply_chain_stops_at_five_links(config) -> None:
    """Telegram chains are unbounded; the prompt stops at five of them."""
    chat = {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"}
    node: dict[str, Any] | None = None
    for i in range(7):
        node = {
            "message_id": 200 + i,
            "date": int(time.time()),
            "chat": chat,
            "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
            "text": f"chain link {i}",
            **({"reply_to_message": node} if node is not None else {}),
        }
    mention_text = "@lumi_test_bot and this"
    _, agent = await send(
        config,
        [make_reply("ok")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=[{"type": "mention", "offset": 0, "length": len("@lumi_test_bot")}],
        reply_to_message=node,
    )
    prompt = prompts(agent)
    assert "chain link 6" in prompt  # the direct quote
    assert "chain link 2" in prompt  # the fifth link back
    assert "chain link 1" not in prompt
    assert "chain link 0" not in prompt
    assert "chain cut at the cap" in prompt


async def test_an_unaddressed_group_message_is_kept_for_the_model(config) -> None:
    """The bot stays silent, but the room's line goes to the log.

    This is what `more_context` exists for: a dropped message is one the model
    will never see any other way.
    """
    bot, agent = await send(
        config,
        [make_reply("never sent")],
        "the deploy is broken again",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
    )
    assert bot.sent == []
    assert agent.llm.calls == []
    assert [entry.text for entry in CHATLOG.entries(GROUP_CHAT)] == [
        "the deploy is broken again"
    ]


async def test_a_group_document_is_not_logged(config) -> None:
    """Files are skipped even with a caption: the log is conversation, not an
    inbox the model can recall."""
    application, bot, _ = build(config, [make_reply("never sent")])
    await application.initialize()
    try:
        await application.process_update(
            make_document_update(
                bot,
                caption="see attached",
                chat_id=GROUP_CHAT,
                chat_type="supergroup",
            )
        )
    finally:
        await application.shutdown()
    assert bot.sent == []
    assert CHATLOG.entries(GROUP_CHAT) == []


async def test_a_group_photo_is_not_logged(config) -> None:
    """A captioned photo is media, not room chatter."""
    application, bot, _ = build(config, [make_reply("never sent")])
    await application.initialize()
    try:
        await application.process_update(
            _make_photo_update(
                bot,
                caption="look at this",
                chat_id=GROUP_CHAT,
                chat_type="supergroup",
            )
        )
    finally:
        await application.shutdown()
    assert bot.sent == []
    assert CHATLOG.entries(GROUP_CHAT) == []


async def test_a_mentioned_group_message_is_not_logged(config) -> None:
    """An addressed message is already in the conversation — logging it would
    show the model the same line twice."""
    mention_text = "@lumi_test_bot hello"
    await send(
        config,
        [make_reply("hi")],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=[{"type": "mention", "offset": 0, "length": len("@lumi_test_bot")}],
    )
    assert CHATLOG.entries(GROUP_CHAT) == []


async def test_a_strangers_chatter_in_a_group_is_logged(config) -> None:
    """Silent for the stranger, visible in the log: the not-whitelisted path
    records too, so a room the bot refuses still reads as a room."""
    bot, agent = await send(
        config,
        [make_reply("never sent")],
        "anyone seen the release notes?",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        user_id=STRANGER,
    )
    assert bot.sent == []
    assert agent.llm.calls == []
    assert [entry.text for entry in CHATLOG.entries(GROUP_CHAT)] == [
        "anyone seen the release notes?"
    ]
    assert CHATLOG.entries(GROUP_CHAT)[0].author  # a name the model can quote


async def test_a_private_chat_is_not_logged(config) -> None:
    """Every private message becomes a turn, so logging it would only
    duplicate the transcript."""
    bot, _ = await send(config, [make_reply("hi back")], "hello")
    assert "hi back" in texts(bot)
    assert CHATLOG.entries(CHAT) == []


async def test_group_chat_does_not_reply_to_a_reply_to_another_bot(config) -> None:
    """A reply to *another* bot in the room is not a reply to this bot.

    "is a bot" is not enough: the reply has to be one of ours, matched by
    username. Otherwise every bot in a group would answer every reply.
    """
    reply_to_other_bot = {
        "message_id": 99,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {
            "id": 54321,
            "is_bot": True,
            "first_name": "Some Other Bot",
            "username": "some_other_bot",
        },
        "text": "an earlier message from another bot",
    }
    bot, agent = await send(
        config,
        [make_reply("should never fire")],
        "ok",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_other_bot,
    )
    assert bot.sent == []
    assert agent.llm.calls == []


async def test_group_chat_runs_a_command_sent_as_a_reply_to_the_bot(config) -> None:
    """Quoting a command under the bot's own answer addresses the bot.

    Telegram hands a replied-to command to the bot that wrote the message it
    replies to, so the ``@botname`` suffix is not needed here.
    """
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
        [],
        "/start",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_bot,
    )
    assert "confirmation" in texts(bot).lower()


async def test_a_stranger_replying_to_the_bot_is_told_it_is_private(config) -> None:
    """A reply to the bot counts as addressing it — so it gets the refusal."""
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
        [],
        "are you there?",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_bot,
        user_id=STRANGER,
    )
    assert "not whitelisted" in texts(bot).lower()


async def test_a_stranger_replying_to_another_stranger_is_ignored(config) -> None:
    """A stranger talking to another stranger never reaches the bot at all."""
    reply_to_stranger = {
        "message_id": 99,
        "date": int(time.time()),
        "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
        "from": {"id": 777, "is_bot": False, "first_name": "Other", "username": "other"},
        "text": "an earlier human message",
    }
    bot, agent = await send(
        config,
        [make_reply("should never fire")],
        "same here",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        reply_to_message=reply_to_stranger,
        user_id=STRANGER,
    )
    assert bot.sent == []
    assert agent.llm.calls == []


async def test_group_chat_slash_command_targeting_the_bot(config) -> None:
    """/start@lumi_test_bot in a group runs the command — it names the bot."""
    bot, _ = await send(config, [], "/start@lumi_test_bot", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert "confirmation" in texts(bot).lower()


async def test_group_chat_plain_slash_command_is_silent(config) -> None:
    """Plain /start in a group is dropped — Telegram routes that elsewhere."""
    bot, _ = await send(config, [], "/start", chat_id=GROUP_CHAT, chat_type="supergroup")
    assert bot.sent == []


async def test_group_chat_slash_command_targeting_another_bot_is_silent(config) -> None:
    """/start@some_other_bot must not fire this bot's handlers."""
    bot, _ = await send(config, [], "/start@some_other_bot", chat_id=GROUP_CHAT, chat_type="supergroup")
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


async def test_a_stranger_in_a_group_is_ignored_until_they_address_the_bot(config) -> None:
    """An unaddressed group message from a stranger gets no answer at all.

    Answering every stranger would make the bot announce itself in every
    conversation the room happens to have — the opposite of "only reply when
    mentioned".
    """
    bot, agent = await send(
        config,
        [make_reply("should never be sent")],
        "just chatting",
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        user_id=STRANGER,
    )
    assert bot.sent == []
    assert agent.llm.calls == []


async def test_a_stranger_who_mentions_the_bot_is_told_it_is_private(config) -> None:
    """Addressing the bot does get an answer — the refusal, not silence."""
    mention_text = "hi @lumi_test_bot"
    entities = [{"type": "mention", "offset": 3, "length": len("@lumi_test_bot")}]
    bot, _ = await send(
        config,
        [],
        mention_text,
        chat_id=GROUP_CHAT,
        chat_type="supergroup",
        entities=entities,
        user_id=STRANGER,
    )
    assert "not whitelisted" in texts(bot).lower()


# --------------------------------------------------------------------------- #
# arriving in a group
# --------------------------------------------------------------------------- #


async def test_the_bot_is_silent_when_it_is_added_to_a_group(config) -> None:
    """Telegram announces the bot's arrival as a service message.

    There is nothing to answer there: the bot must not reply "i read text and
    images only" to its own arrival, and must not greet the room either.
    """
    application, bot, agent = build(config, [make_reply("should never be sent")])
    await application.initialize()
    update = Update.de_json(
        {
            "update_id": 11,
            "message": {
                "message_id": 12,
                "date": int(time.time()),
                "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "Owner", "username": "owner"},
                # The Bot API's newer name for the same service message.
                "new_chat_participants": [
                    {
                        "id": 12345,
                        "is_bot": True,
                        "first_name": "Lumi",
                        "username": "lumi_test_bot",
                    }
                ],
            },
        },
        bot,
    )
    try:
        await application.process_update(update)
    finally:
        await application.shutdown()

    assert bot.sent == []
    assert agent.llm.calls == []


async def test_the_membership_update_itself_is_silent(config) -> None:
    """The ``my_chat_member`` event is logged, never answered."""
    application, bot, agent = build(config, [make_reply("should never be sent")])
    await application.initialize()
    update = Update.de_json(
        {
            "update_id": 13,
            "my_chat_member": {
                "chat": {"id": GROUP_CHAT, "type": "supergroup", "title": "Lab"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "Owner", "username": "owner"},
                "date": int(time.time()),
                "old_chat_member": {
                    "user": {
                        "id": 12345,
                        "is_bot": True,
                        "first_name": "Lumi",
                        "username": "lumi_test_bot",
                    },
                    "status": "left",
                },
                "new_chat_member": {
                    "user": {
                        "id": 12345,
                        "is_bot": True,
                        "first_name": "Lumi",
                        "username": "lumi_test_bot",
                    },
                    "status": "member",
                },
            },
        },
        bot,
    )
    try:
        await application.process_update(update)
    finally:
        await application.shutdown()

    assert bot.sent == []
    assert agent.llm.calls == []


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


async def test_whitelisted_group_still_requires_a_mention(config) -> None:
    """A whitelisted group is still a group: no mention, no answer.

    The whitelist says *who* the bot talks to, never *when* — a whitelisted
    room is still answered only when the bot is addressed.
    """
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    bot, agent = await send(
        config,
        [make_reply("should not fire")],
        "hello group",
        chat_id=WHITELISTED_GROUP,
        chat_type="supergroup",
    )
    assert bot.sent == []
    assert agent.llm.calls == []


async def test_whitelisted_group_answers_any_member_that_mentions_the_bot(config) -> None:
    """In a whitelisted group the bot answers the room, not just the owner."""
    config.bot.whitelisted_groups = [str(WHITELISTED_GROUP)]
    mention_text = "hi @lumi_test_bot"
    entities = [{"type": "mention", "offset": 3, "length": len("@lumi_test_bot")}]
    bot, agent = await send(
        config,
        [make_reply("hello stranger")],
        mention_text,
        chat_id=WHITELISTED_GROUP,
        chat_type="supergroup",
        entities=entities,
        user_id=STRANGER,
    )
    assert "hello stranger" in texts(bot)
    assert agent.llm.calls


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
    memory = make_store(config)
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
    memory = make_store(config)
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
    memory = make_store(config)
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
    memory = make_store(config)
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
    memory = make_store(config)
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
    memory = make_store(config)
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
    # Pre-existing and empty, so the session-freshness gate (finding 3b) is not
    # what this test is exercising — it would stop the turn for an owner tap.
    report = config.shell_cwd / "report.md"
    report.write_text("", encoding="utf-8")
    backdate(report)
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

    memory = make_store(config)
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
    memory = make_store(config)
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
    memory = make_store(config)
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
# reasoning / the step feed
# --------------------------------------------------------------------------- #


THINKING = "first I checked the spelling, then the plural, so four"


async def test_the_thinking_trace_is_never_spelled_out_in_the_chat(config) -> None:
    """The trace stays internal — not shown, not behind a button. What the
    chat sees is the answer and, on tool turns, the step list."""
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert THINKING not in texts(bot)
    assert "four" in texts(bot)


async def test_a_tool_turn_shows_the_steps_it_took(config) -> None:
    """The live note reads like a receipt of the turn: thinking, then each
    tool — and it sits above the answer, in the order it happened."""
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})]),
            make_reply("it said hi"),
        ],
        "run echo hi",
    )
    body = texts(bot)
    step_message = next(m for m in bot.sent if "Run Bash" in m["text"])
    assert step_message["text"].splitlines() == ["🧠 Thinking", "💻 Run Bash"]
    assert body.index("Run Bash") < body.index("it said hi")


async def test_the_step_list_is_edited_in_place(config) -> None:
    """One message that grows, not a new message per step: a chat scrolled
    back to an old turn must not fill with progress chatter."""
    application, bot, _ = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})]),
            make_reply("it said hi"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "run echo hi"))
    finally:
        await application.shutdown()

    step_messages = [m for m in bot.sent if "Thinking" in m["text"]]
    assert len(step_messages) == 1
    assert bot.edits, "subsequent steps must edit that message, not re-send"
    assert "🧠 Thinking" in bot.edits[0]["text"]
    assert "💻 Run Bash" in bot.edits[0]["text"]


async def test_a_turn_without_tools_posts_no_step_message(config) -> None:
    """A plain answer is just the answer — the step note earns its place by
    reporting work, not by being there."""
    bot, _ = await send(config, [make_reply("four")], "how many r's?")
    assert "🧠" not in texts(bot)
    assert not [m for m in bot.sent if m.get("reply_markup")]


async def test_repeated_model_calls_do_not_stack_thinking_lines(config) -> None:
    """Every iteration starts with a model call; the header belongs on screen
    once."""
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})], reasoning="run it"),
            make_reply("it said hi", reasoning="it worked"),
        ],
        "run echo hi",
    )
    assert texts(bot).count("🧠 Thinking") == 1


async def test_steps_can_be_turned_off(config) -> None:
    config.llm.show_reasoning = False
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})]),
            make_reply("it said hi"),
        ],
        "run echo hi",
    )
    assert "🧠" not in texts(bot)
    assert "it said hi" in texts(bot)


async def test_step_feed_caps_rendered_body_at_ten_lines() -> None:
    bot = FakeBot()
    feed = StepFeed(bot, CHAT)

    for index in range(11):
        await feed.step("tool", f"tool-{index}")

    rendered = bot.edits[-1]["text"]
    assert rendered.splitlines() == [
        "… (+2 earlier)",
        *(f"🔧 tool-{index}" for index in range(2, 11)),
    ]
    assert len(rendered.splitlines()) == 10


def test_step_labels_read_the_way_the_owner_sees_them() -> None:
    """The feed names capabilities, not internals: a content search is a
    Search, and the web step says which action ran."""
    from lumi.bot import step_label

    assert step_label("tool", "grep") == "🔍 Search"
    assert step_label("tool", "web.search") == "🌐 Web Search"
    assert step_label("tool", "web.fetch") == "🌐 Web Fetch"
    assert step_label("tool", "web") == "🌐 Web"  # action unknown
    assert step_label("tool", "brand_new_tool") == "🔧 brand_new_tool"
    assert step_label("thinking") == "🧠 Thinking"


async def test_the_step_message_never_carries_a_button(config) -> None:
    bot, _ = await send(config, [make_reply("four", reasoning=THINKING)], "how many r's?")
    assert not [m for m in bot.sent if m.get("reply_markup")]


async def test_a_turn_paused_for_approval_keeps_its_step_list(config) -> None:
    bot, _ = await send(
        config,
        [
            make_reply(
                "",
                [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})],
            )
        ],
        "clean up",
    )
    assert "needs your approval" in texts(bot)
    assert "🧠 Thinking" in texts(bot)
    # Exactly one keyboard on the turn: the Confirm/Cancel pair, not a
    # second button riding on the step message.
    assert len([m for m in bot.sent if m.get("reply_markup")]) == 1


async def test_the_send_tool_delivers_before_the_final_answer(config) -> None:
    """One turn, several messages: the mid-turn one arrives while the turn is
    still running, and the final answer still lands after it."""
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "send", {"message": "phase 1 done"})]),
            make_reply("all done"),
        ],
        "do the whole thing",
    )
    body = texts(bot)
    assert "phase 1 done" in body
    assert body.index("phase 1 done") < body.index("all done")
    # The tool's own receipt is for the model, not the chat.
    assert "Sent to the owner" not in body


async def test_the_send_step_shows_up_in_the_step_list(config) -> None:
    bot, _ = await send(
        config,
        [
            make_reply("", [("c1", "send", {"message": "halfway"})]),
            make_reply("done"),
        ],
        "go",
    )
    assert "✉️ Send" in texts(bot)


async def test_a_failed_live_send_is_flushed_after_the_turn(config) -> None:
    """A transient Telegram failure on a mid-turn send must not swallow the
    message. The live hook surfaces the error, the agent keeps the message on
    the result, and the turn-tail flush delivers it — before the final answer.
    """
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "send", {"message": "phase 1 done"})]),
            make_reply("all done"),
        ],
        bot=FlakySendBot("phase 1 done"),
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "do the whole thing"))
    finally:
        await application.shutdown()

    assert bot._failed, "the live send must actually have failed once"
    body = texts(bot)
    assert "phase 1 done" in body, "the failed live send must not vanish"
    assert body.index("phase 1 done") < body.index("all done")


async def test_a_permanently_dead_chat_still_finishes_the_turn(config) -> None:
    """When even the flush cannot deliver, the turn must still complete: the
    message is lost to the log, but the final answer is not."""
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "send", {"message": "into the void"})]),
            make_reply("all done"),
        ],
        bot=DeadBot(),
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "do the whole thing"))
    finally:
        await application.shutdown()

    # DeadBot rejects every send, so nothing lands in the chat — the point is
    # the update was processed without raising: the model still got its send
    # receipt and produced the final answer.
    conv = agent.conversation(CHAT)
    assert any(
        m.get("role") == "tool" and "Sent to the owner" in str(m.get("content", ""))
        for m in conv.messages
    ), "the turn must continue past a dead delivery channel"
    assert conv.messages[-1]["role"] == "assistant"  # the answer was produced


async def test_reply_strict_raises_when_no_target_available() -> None:
    """When an update has no message and no chat, strict delivery raises."""
    from telegram import Update
    from telegram.error import TelegramError

    from lumi.bot import reply

    update = Update(update_id=1)
    with pytest.raises(TelegramError, match="effective_message and effective_chat are None"):
        await reply(update, "important message", strict=True)


async def test_reply_falls_back_to_bot_send_message_when_effective_message_is_none() -> None:
    """When effective_message is missing, reply sends directly to effective_chat."""
    from telegram import Chat, Update

    from lumi.bot import reply

    bot = FakeBot()
    update = Update(update_id=1)
    update.set_bot(bot)
    object.__setattr__(update, "_effective_chat", Chat(id=42, type="private"))

    await reply(update, "sent to chat directly", strict=True)
    assert len(bot.sent) == 1
    assert bot.sent[0]["chat_id"] == 42
    assert bot.sent[0]["text"] == "sent to chat directly"


async def test_reply_falls_back_to_bot_send_message_when_message_reply_raises_bad_request() -> None:
    """When replying to a deleted message raises BadRequest, reply falls back to direct chat send."""
    from telegram.error import BadRequest

    from lumi.bot import reply

    class DeletedMessageBot(FakeBot):
        async def send_message(self, chat_id: Any, text: str, **kwargs: Any) -> Any:
            if kwargs.get("reply_to_message_id") is not None:
                raise BadRequest("Message to be replied not found")
            return await super().send_message(chat_id, text, **kwargs)

    bot = DeletedMessageBot()
    update = make_update(bot, "hello")

    await reply(update, "still delivered to chat", strict=True)
    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "still delivered to chat"


async def test_an_approved_run_shows_up_in_the_step_list(config) -> None:
    """The tap on "run it" runs the tool after the turn ended — its step
    belongs to the resolve turn's feed, not the vanished one."""
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})]),
            make_reply("it is gone"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "clean up the workspace"))
        keyboard = next(m["reply_markup"] for m in bot.sent if m.get("reply_markup"))
        approve = keyboard.inline_keyboard[0][0]
        bot._sent.clear()  # only the resolve turn's feed is under test
        await application.process_update(make_callback(bot, approve.callback_data))
    finally:
        await application.shutdown()

    body = texts(bot)
    assert "💻 Run Bash" in body
    assert "it is gone" in body


async def test_slash_approve_command_shows_up_in_the_step_list(config) -> None:
    """Approving via `/approve` text command executes the pending action and
    renders the step feed."""
    application, bot, agent = build(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})]),
            make_reply("cleaned up"),
        ],
    )
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "clean up the workspace"))
        bot._sent.clear()
        await application.process_update(make_update(bot, "/approve"))
    finally:
        await application.shutdown()

    body = texts(bot)
    assert "💻 Run Bash" in body
    assert "cleaned up" in body


async def test_double_asterisk_bold_renders_as_bold(config) -> None:
    """Markdown **bold** is not formatting in Telegram's HTML mode — it used
    to show up as literal asterisks. Now it always renders."""
    bot, _ = await send(config, [make_reply("this is **bold** text")], "hi")
    answer = next(m for m in bot.sent if "bold" in m.get("text", ""))
    assert "<b>bold</b>" in answer["text"]
    assert "**" not in answer["text"]


async def test_bold_conversion_leaves_code_alone(config) -> None:
    """Inside <code>/<pre>, asterisks are frequently real (globs, pointers) —
    the bold pass must not touch them."""
    bot, _ = await send(config, [make_reply("run `ls **/*.py` first")], "hi")
    answer = next(m for m in bot.sent if "ls" in m.get("text", ""))
    assert "<code>ls **/*.py</code>" in answer["text"]


# --------------------------------------------------------------------------- #
# /reasoning
# --------------------------------------------------------------------------- #


async def test_reasoning_reports_the_setup(config) -> None:
    bot, _ = await send(config, [], "/reasoning")
    body = texts(bot)
    assert "reasoning" in body
    assert "off" in body


async def test_reasoning_off_hides_the_step_list(config) -> None:
    bot, _ = await send(config, [], "/reasoning off")
    assert "live steps in chat: off" in texts(bot)

    bot, _ = await send(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "echo hi"})]), make_reply("four")],
        "how many r's?",
    )
    assert "🧠" not in texts(bot)


async def test_reasoning_on_brings_the_step_list_back(config) -> None:
    application, bot, _ = build(config, [])
    await application.initialize()
    try:
        await application.process_update(make_update(bot, "/reasoning off"))
        bot._sent.clear()
        await application.process_update(make_update(bot, "/reasoning on"))
    finally:
        await application.shutdown()

    assert "live steps in chat: on" in texts(bot)


async def test_reasoning_rejects_nonsense(config) -> None:
    bot, _ = await send(config, [], "/reasoning maybe")
    assert "usage" in texts(bot).lower()


async def test_status_mentions_reasoning(config) -> None:
    bot, _ = await send(config, [], "/status")
    assert "reasoning: off" in texts(bot)


async def test_start_mentions_the_step_list(config) -> None:
    bot, _ = await send(config, [], "/start")
    assert "/reasoning" in texts(bot)
    assert "live note" in texts(bot)


# --------------------------------------------------------------------------- #
# /config and /env (owner-only introspection)
# --------------------------------------------------------------------------- #


async def test_config_shows_the_resolved_settings(config) -> None:
    bot, _ = await send(config, [], "/config")
    body = texts(bot)
    assert "test-model" in body
    assert "firecrawl → exa → tavily → you → mcp" in body
    assert "group mode: mention" in body
    assert "workspace" in body


async def test_config_reflects_live_toggles(config) -> None:
    """The view reads the same config object the handlers mutate."""
    config.llm.show_reasoning = False
    config.bot.group_reply_mode = "always"
    bot, _ = await send(config, [], "/config")
    body = texts(bot)
    assert "live steps in chat: off" in body
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


async def test_own_help_lists_the_owner_introspection_commands(config) -> None:
    bot, _ = await send(config, [], "/own_help")
    body = texts(bot)
    assert "/config" in body
    assert "/env" in body
    assert "masked" in body
    # Memory- and conversation-changing commands are the owner's alone.
    assert "/forget" in body
    assert "/reload" in body
    assert "/reset" in body


async def test_own_help_accepts_the_hyphen_spelling(config) -> None:
    """Telegram clients only autocomplete ``/own_help``, but ``/own-help`` is
    the spelling from the docs — it must work when typed too."""
    bot, _ = await send(config, [], "/own-help")
    assert "/config" in texts(bot)


async def test_own_help_is_owner_only(config) -> None:
    """Whitelisted users may use the bot, but not the owner's inventory."""
    config.bot.whitelisted_users = [str(WHITELISTED_USER)]
    bot, _ = await send(config, [], "/own_help", user_id=WHITELISTED_USER)
    body = texts(bot)
    assert "only the owner can see that" in body.lower()
    assert "/config" not in body


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


# --------------------------------------------------------------------------- #
# config writing (_write_toml)
# --------------------------------------------------------------------------- #


def test_write_toml_round_trips_dotted_keys() -> None:
    """A key containing a dot must stay one key. Written bare, TOML reads it
    as a nested table, so every rewrite would restructure the file."""
    data = {
        "a.b": 1,
        "plain": True,
        "bot": {"motd": "hi", "sub": {"x": 2}},
    }
    buf = io.StringIO()
    _write_toml(buf, data)
    text = buf.getvalue()

    assert tomllib.loads(text) == data
    # Quoted, not bare — `a.b = 1` would have parsed as nesting.
    assert '"a.b" = 1' in text


def test_write_toml_puts_top_level_keys_before_any_table() -> None:
    """After a [table] header a bare key belongs to that table, so the scalar
    pass must always come first — at every nesting level."""
    buf = io.StringIO()
    _write_toml(buf, {"z": 1, "a": {"y": 2, "b": {"x": 3}}})
    text = buf.getvalue()

    assert text.index("z = 1") < text.index("[a]")
    assert text.index("y = 2") < text.index("[a.b]")
    assert tomllib.loads(text) == {"z": 1, "a": {"y": 2, "b": {"x": 3}}}


# --------------------------------------------------------------------------- #
# stop-signal handling
# --------------------------------------------------------------------------- #


class _StopApp:
    """The slice of Application the stop handler touches."""

    def __init__(self) -> None:
        self.running = False
        self.stopped = False

    def stop_running(self) -> None:
        self.stopped = True


def test_first_stop_signal_stops_the_application_gracefully() -> None:
    app = _StopApp()
    app.running = True
    exits: list[int] = []
    handler = stop_signal_handler(app, lambda: exits.append(130))

    handler()

    assert app.stopped
    assert exits == []


def test_second_stop_signal_force_exits() -> None:
    """A second press means 'now': no more waiting on an in-flight turn."""
    app = _StopApp()
    app.running = True
    exits: list[int] = []
    handler = stop_signal_handler(app, lambda: exits.append(130))

    handler()
    handler()

    assert exits == [130]


def test_stop_signal_during_bootstrap_interrupts_cleanly() -> None:
    """Before the application is running (e.g. an endless bootstrap retry),
    a signal must abort the wait the same way PTB's own handler did."""
    app = _StopApp()
    handler = stop_signal_handler(app, lambda: None)

    with pytest.raises(KeyboardInterrupt):
        handler()


def test_install_stop_handlers_wires_a_real_signal() -> None:
    """A real SIGINT on the running loop reaches the handler: the application
    is asked to stop, and nothing exits the process."""
    import asyncio
    import os
    import signal as signal_module

    from lumi.bot import _install_stop_handlers

    async def scenario() -> None:
        app = _StopApp()
        app.running = True
        _install_stop_handlers(app)
        await asyncio.sleep(0)
        os.kill(os.getpid(), signal_module.SIGINT)
        await asyncio.sleep(0.1)  # let the loop run the handler callback
        assert app.stopped

    asyncio.run(scenario())
