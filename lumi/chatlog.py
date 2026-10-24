"""The room log: what the chat said while the bot was not being talked to.

In a group the bot only joins the conversation when it is addressed — that is
what mention mode buys, silence instead of a bot interrupting every thread.
The flip side is that the model then has no idea what the room is talking
about: those messages are never turns of its conversation and never reach it
any other way, so "as they said above" has nothing behind it.

This log is the missing half. Every text message the bot *dropped* is kept
here — in memory, per chat — and the ``more_context`` tool reads it back on
demand. Deliberately not the transcript: :class:`~lumi.memory.History` on disk
stays the bot's own conversation, and a restart starts the log fresh rather
than replaying a room the owner may consider settled.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

#: How many dropped messages are kept per chat. Deliberately short: the log
#: answers "what is happening right now", not "what was said last month" —
#: the transcript and MEMORY.md own the long view.
CHATLOG_MAXLEN = 10


@dataclass(slots=True)
class LogEntry:
    """One message the bot chose not to answer."""

    #: Telegram's per-chat message id — order of arrival, useful to the model
    #: when it stitches this log together with its own history.
    message_id: int
    #: Display name of whoever sent it. Never a credential: Telegram gives the
    #: first name, and that is what the room calls them too.
    author: str
    text: str
    #: Local wall-clock time of the message, ``HH:MM``. Chatter without a
    #: timestamp leaves the model guessing whether "the build is broken" is
    #: from this morning or from the message it is answering now.
    clock: str

    def render(self, index: int) -> str:
        """One line of the tool's output."""
        return f"[{index}] {self.clock} {self.author}: {self.text}"


class ChatLog:
    """Per-chat ring buffers of dropped messages, oldest entry first."""

    def __init__(self, maxlen: int = CHATLOG_MAXLEN) -> None:
        self._maxlen = max(1, int(maxlen))
        self._logs: dict[int | str, deque[LogEntry]] = {}

    def record(self, chat_id: int | str, entry: LogEntry) -> None:
        """Append one message for *chat_id*, dropping the oldest past the cap."""
        log = self._logs.get(chat_id)
        if log is None:
            log = deque(maxlen=self._maxlen)
            self._logs[chat_id] = log
        log.append(entry)

    def entries(self, chat_id: int | str | None) -> list[LogEntry]:
        """Everything recorded for *chat_id*, oldest first. Empty if none."""
        if chat_id is None:
            return []
        log = self._logs.get(chat_id)
        return list(log) if log else []

    def count(self, chat_id: int | str | None) -> int:
        if chat_id is None:
            return 0
        log = self._logs.get(chat_id)
        return len(log) if log else 0

    def clear(self, chat_id: int | str | None = None) -> None:
        """Forget one chat, or every chat when *chat_id* is ``None``."""
        if chat_id is None:
            self._logs.clear()
        else:
            self._logs.pop(chat_id, None)


def entry_from(message: Any) -> LogEntry | None:
    """Build a :class:`LogEntry` from a Telegram message, or ``None``.

    ``None`` means "nothing a person typed into the room": a photo, file,
    sticker or other media (caption or not), a service notice, or a message
    with no sender or usable clock. Callers have already decided the message
    is group chatter the bot dropped; this only decides whether it is
    loggable. Media is skipped on purpose: this log is the room's conversation,
    and an attachment is not a line of it.
    """
    text = (getattr(message, "text", None) or "").strip()
    if not text:
        return None
    sender = getattr(message, "from_user", None)
    author = "someone"
    if sender is not None:
        author = sender.first_name or sender.username or "someone"
    date = getattr(message, "date", None)
    clock = date.astimezone().strftime("%H:%M") if date is not None else "??:??"
    return LogEntry(
        message_id=int(getattr(message, "message_id", 0) or 0),
        author=author,
        text=text,
        clock=clock,
    )


#: The process-wide log. One bot process, one room per chat id, so a module
#: singleton is the whole wiring: the Telegram layer records into it and the
#: ``more_context`` tool reads from it, with no plumbing between them to get
#: out of sync. Tests build their own :class:`ChatLog` or clear this one.
CHATLOG = ChatLog()
