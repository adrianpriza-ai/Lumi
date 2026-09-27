"""The Telegram layer, end to end.

A fake Bot records what would have been sent, and the scripted LLM stands in for
the model. That makes it possible to assert on the whole path: a message arrives,
is authorised, reaches the agent, and comes back as Telegram messages — including
the Confirm/Cancel keyboard for a risky command.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from conftest import FakeLLM, make_reply
from telegram import Bot, Update, User

from lumi.agent import Agent
from lumi.bot import CB_NO, CB_OK, build_application
from lumi.memory import History, MemoryFile
from lumi.personality import Personality
from lumi.tools import build_registry

CHAT = 42
OWNER = 42
STRANGER = 999


class FakeBot(Bot):
    """A real Bot with the network stubbed out, recording what it was asked to send.

    TelegramObject freezes attribute assignment once constructed, so the record
    lists are private with read-only properties in front of them.
    """

    def __init__(self) -> None:
        self._sent: list[dict[str, Any]] = []
        self._actions: list[tuple[Any, ...]] = []
        super().__init__(token="123456:TESTTOKENFORLUMI")

    @property
    def sent(self) -> list[dict[str, Any]]:
        return self._sent

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

    async def send_chat_action(self, chat_id: Any, action: Any, **kwargs: Any) -> bool:
        self._actions.append((chat_id, action))
        return True


def make_update(bot: Bot, text: str | None = None, *, chat_id: int = CHAT, user_id: int = OWNER,
                message_id: int = 1) -> Update:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(time.time()),
        "chat": {"id": chat_id, "type": "private", "first_name": "Chat"},
        "from": {"id": user_id, "is_bot": False, "first_name": "Owner", "username": "owner"},
    }
    if text is not None:
        message["text"] = text
        # Real Telegram always sends a bot_command entity alongside the text, and
        # filters.COMMAND keys off the entity, not off the leading slash. Without
        # it every command would fall through to the catch-all text handler.
        if text.startswith("/"):
            command = text.split(maxsplit=1)[0]
            message["entities"] = [
                {"type": "bot_command", "offset": 0, "length": len(command)}
            ]
    return Update.de_json({"update_id": message_id, "message": message}, bot)


def build(config, replies) -> tuple[Any, FakeBot, Agent]:
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
    return "\n".join(message["text"] for message in bot.sent)


# --------------------------------------------------------------------------- #
# owner gating
# --------------------------------------------------------------------------- #


async def test_stranger_is_rejected(config) -> None:
    bot, _ = await send(config, [make_reply("should never be sent")], "hi", user_id=STRANGER)
    assert "not the owner" in texts(bot).lower()
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
    assert "not the owner" in texts(bot).lower()


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
    assert "not the owner" in body.lower()
    for leak in ("run_shell", "test-model", "workspace", "MEMORY.md"):
        assert leak not in body, f"{leak!r} leaked to a stranger"


async def test_non_text_message_is_explained(config) -> None:
    application, bot, _ = build(config, [])
    update = Update.de_json(
        {
            "update_id": 7,
            "message": {
                "message_id": 7,
                "date": int(time.time()),
                "chat": {"id": CHAT, "type": "private"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "Owner"},
                "photo": [{"file_id": "abc", "file_unique_id": "u1", "width": 1, "height": 1}],
            },
        },
        bot,
    )
    await application.initialize()
    try:
        await application.process_update(update)
    finally:
        await application.shutdown()
    assert "read text only" in texts(bot).lower()


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
