"""The room log: how much it keeps and what it refuses to keep."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from lumi.chatlog import CHATLOG_MAXLEN, ChatLog, LogEntry, entry_from


def message(**fields: Any) -> SimpleNamespace:
    """A Telegram-shaped message with sane defaults for the fields under test."""
    defaults: dict[str, Any] = {
        "text": "hello",
        "caption": None,
        "from_user": SimpleNamespace(first_name="Other", username="other", is_bot=False),
        "date": datetime.now(UTC),
        "message_id": 1,
    }
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_the_log_keeps_only_the_newest_messages() -> None:
    log = ChatLog()
    for i in range(CHATLOG_MAXLEN + 5):
        log.record(1, LogEntry(message_id=i, author="Other", text=f"m{i}", clock="12:00"))

    assert [entry.text for entry in log.entries(1)] == [
        f"m{i}" for i in range(5, CHATLOG_MAXLEN + 5)
    ]


def test_a_text_message_is_loggable() -> None:
    entry = entry_from(message(text="the deploy is broken"))

    assert entry is not None
    assert entry.text == "the deploy is broken"


def test_a_photo_with_a_caption_is_not_loggable() -> None:
    entry = entry_from(message(text=None, caption="look at this", photo=[object()]))

    assert entry is None


def test_a_file_with_a_caption_is_not_loggable() -> None:
    entry = entry_from(message(text=None, caption="see attached", document=object()))

    assert entry is None
