"""The ``more_context`` tool: the room talking when the bot was not being talked to.

Mention mode means the model only ever sees messages addressed to it, so in a
group its history is a series of interruptions with the conversation between
them missing entirely. When the owner says "as they said above" or "what is
everyone arguing about", there is nothing above — this tool is how the model
asks for it.

The log it reads comes from :mod:`lumi.chatlog`, filled by the Telegram layer
for exactly the messages the bot dropped. Everything here is about handing
that back safely: a hard ceiling on messages per call, a hard ceiling on
tokens, and refusals that name the remedy rather than a truncated answer that
looks complete — the same rule :class:`~lumi.tools.explore.ReadFileTool` uses
for oversized reads.
"""

from __future__ import annotations

from typing import Any

from ..chatlog import CHATLOG, CHATLOG_MAXLEN, ChatLog, LogEntry
from ..config import Config
from ..util.text import fence_untrusted
from .base import Tool, ToolContext, ToolResult

#: Rough chars-per-token for the output cap. Same divisor as ``read_file``;
#: the cap is a guardrail, not a budget, so a stable estimate beats a
#: dependency on the context window's live calibration.
CHARS_PER_TOKEN = 4

#: Defaults for the two caps below. The instance values come from
#: ``bot.chatlog_max_messages`` (or the log's own maxlen when the tool is built
#: without a config), so the setting raises them without touching this module.
#:
#: How far back from the newest message ``offset`` may reach: the whole
#: retained log, so nothing the bot kept is unreachable by paging.
MAX_OFFSET = CHATLOG_MAXLEN - 1

#: How many messages one call may return: the whole retained log in one call,
#: so recalling the room is not a paging exercise.
MAX_MESSAGES = CHATLOG_MAXLEN

#: One call may return at most this many tokens. Over it is refused with the
#: remedy (a smaller call), never truncated into something that reads whole.
MAX_OUTPUT_TOKENS = 30_000


def _as_int(value: Any, default: int) -> int | None:
    """Coerce a JSON argument to int, or ``None`` when it is not a number.

    ``None`` (rather than ``default``) for a non-numeric value: silently
    reading ``"offset": "two"`` as 0 would answer a call the model did not
    make, and the model learns nothing from that.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry_matches(entry: LogEntry, needles: list[str]) -> bool:
    """Grep's rule, applied to one log entry: every needle, any order.

    The author is part of the haystack so "what did Dana say?" is a query
    rather than a scan of the whole page — the same reason the rendered line
    shows the author at all.
    """
    hay = f"{entry.author} {entry.text}".casefold()
    return all(needle in hay for needle in needles)


def _description(max_messages: int, max_offset: int) -> str:
    """The model-facing description, sized to the configured room log.

    The caps in the text are not constants: ``bot.chatlog_max_messages`` can
    make the log bigger, and a description that still said "10" would teach
    the model to stop paging through messages the log still holds.
    """
    return f"""\
Recent messages from this chat that are NOT part of your conversation.

In a group you only see messages addressed to you — everything the room says
between those mentions never reaches you. Call this when the owner points at
something you cannot see ("as they said above", "what is everyone talking
about?"), or when you need the talk around the message you are answering.

Parameters:
- `query` (optional): only messages containing every word in it, in either
  order, case-insensitive — the same AND rule as `grep`. Searches the author
  and text of each message, so `query: "Dana deploy"` finds Dana's messages
  about the deploy. Omit it to read the room unfiltered.
- `offset` (0-{max_offset}): skip this many of the newest messages before
  reading. Raise it to walk further back; 0 (default) starts at the newest.
- `limit` (1-{max_messages}): how many messages to return, oldest first.
  Default {max_messages}.

`offset` and `limit` page the messages that match `query`, not the whole log.

Capped at {max_messages} messages and ~30k tokens per call: a result over the cap is
refused with the remedy (the exact `limit` that fits) instead of being
truncated into something that looks complete. When older messages are left
out, the head line names the exact `offset` to read them, so paging back never
depends on guesswork. Only messages the bot did *not* answer are logged —
anything addressed to you is already in your conversation, and private chats
log nothing because there your conversation holds everything. Photos and
files sent to the room are logged too (with their captions): the attachment
is saved under `workspace/uploads/` and the log line names its path, so you
can read it with `read_file` — which handles PDFs and images. The log lives
in memory, so it starts fresh on a restart."""


def _parameters(max_messages: int, max_offset: int) -> dict[str, Any]:
    """The argument schema, with the caps of the log this tool reads."""
    return {
        "type": "object",
        "properties": {
            "offset": {
                "type": "integer",
                "description": (
                    "How many of the newest messages to skip before reading. "
                    "0 (default) starts at the newest."
                ),
                "minimum": 0,
                "maximum": max_offset,
            },
            "query": {
                "type": "string",
                "description": (
                    "Only messages containing every word here, any order, "
                    "case-insensitive (author and text are searched). Omit for the "
                    "whole room."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"How many messages to return, oldest first. Defaults to {max_messages}."
                ),
                "minimum": 1,
                "maximum": max_messages,
            },
        },
        "additionalProperties": False,
    }


class MoreContextTool(Tool):
    name = "more_context"
    # Defaults for the class-level spec; ``__init__`` re-renders both against
    # the configured log size, which ``bot.chatlog_max_messages`` may have
    # raised past ``CHATLOG_MAXLEN``.
    description = _description(MAX_MESSAGES, MAX_OFFSET)
    parameters = _parameters(MAX_MESSAGES, MAX_OFFSET)

    def __init__(self, config: Config | None = None, log: ChatLog | None = None) -> None:
        # Both optional, matching send.py's habit of taking them now so the
        # registry's factory signatures stay uniform: *config* for future
        # knobs, *log* so tests can drive an isolated buffer instead of the
        # process-wide one.
        self._config = config
        self._log = log if log is not None else CHATLOG
        # Paging follows the configuration, not the module constant: a log
        # configured to keep 50 messages must be readable past offset 9 or the
        # setting would only cost memory. The log is the fallback for a tool
        # built without a config (tests), where its own capacity is the best
        # answer available.
        size = config.bot.chatlog_max_messages if config is not None else self._log.maxlen
        self._max_messages = max(1, int(size))
        self._max_offset = self._max_messages - 1
        if (self._max_messages, self._max_offset) != (MAX_MESSAGES, MAX_OFFSET):
            self.description = _description(self._max_messages, self._max_offset)
            self.parameters = _parameters(self._max_messages, self._max_offset)

    def summary_line(self) -> str:
        return (
            f"`{self.name}` — messages this chat never addressed to me, keyword-searchable "
            f"({self._max_messages} max per call, {MAX_OUTPUT_TOKENS:,}-token cap)"
        )

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        offset = _as_int(arguments.get("offset"), 0)
        if offset is None:
            return ToolResult.failure(
                f"`offset` must be a whole number 0-{self._max_offset}."
            )
        if not 0 <= offset <= self._max_offset:
            return ToolResult.failure(
                f"`offset` must be between 0 and {self._max_offset} — got {offset}. "
                f"Use offset={max(0, min(offset, self._max_offset))} and read again."
            )

        limit = _as_int(arguments.get("limit"), self._max_messages)
        if limit is None:
            return ToolResult.failure(
                f"`limit` must be a whole number 1-{self._max_messages}."
            )
        if not 1 <= limit <= self._max_messages:
            return ToolResult.failure(
                f"`limit` must be between 1 and {self._max_messages} — got {limit}. "
                f"Call again with limit={max(1, min(limit, self._max_messages))}."
            )

        logged = self._log.entries(ctx.chat_id)
        if not logged:
            return ToolResult.failure(
                "no messages are logged for this chat. Either nothing has been said "
                "since I arrived, or this is a private chat — there my conversation "
                "already holds everything, so there is nothing to fetch."
            )

        query = str(arguments.get("query") or "").strip()
        needles = [word.casefold() for word in query.split()]
        entries = [entry for entry in logged if _entry_matches(entry, needles)] if needles else logged
        if not entries:
            return ToolResult.failure(
                f"none of the {len(logged)} logged message(s) in this chat match "
                f"{query!r}. Every word must appear in a message (or its author), "
                "in any order and case-insensitive — try fewer words, or call again "
                "without `query` to see the whole room."
            )

        end = len(entries) - offset
        if end <= 0:
            scope = "match" if needles else "are logged for this chat"
            return ToolResult.failure(
                f"offset {offset} is past the end — only {len(entries)} message(s) "
                f"{scope}. Call again with offset=0."
            )
        start = max(0, end - limit)

        def render(window_start: int, window_end: int) -> str:
            """The full result text for ``entries[window_start:window_end]``.

            The head line carries counts and *actionable* paging: to read the
            messages before this window the model must raise ``offset`` by
            exactly the number of messages newer than them, not by one — a
            vague "raise offset" gets it an overlapping page.
            """
            window = entries[window_start:window_end]
            hints = []
            if offset:
                hints.append(f"{offset} newer skipped — offset=0 starts at the newest")
            if window_start:
                next_offset = len(entries) - window_start
                hints.append(
                    f"{window_start} older not shown — offset={next_offset} reads them "
                    f"(limit={min(window_start, self._max_messages)})"
                )
            if needles:
                head = (
                    f"{len(window)} of {len(entries)} matching message(s) for "
                    f"{query!r}, oldest first"
                )
            else:
                head = f"{len(window)} message(s) from this chat, oldest first"
            if hints:
                head += " — " + "; ".join(hints)
            body = "\n".join(entry.render(i) for i, entry in enumerate(window, start=1))
            # Room talk is third-party content the model was not addressed in,
            # so the rendered log travels as data, not instructions — same
            # fence the web and read tools wrap their payloads in. The head
            # line (counts and paging hints) is ours and stays outside.
            return f"{head}:\n{fence_untrusted(body)}"

        text = render(start, end)
        tokens = len(text) // CHARS_PER_TOKEN + 1
        if tokens > MAX_OUTPUT_TOKENS:
            # Unreachable through Telegram alone (5 × 4096 chars ≈ 5k tokens),
            # but the cap is the contract: if the log ever holds something
            # bigger, the model hears "make the call smaller", not a wall.
            # Find the largest limit that actually fits so it need not guess.
            fit = None
            for candidate in range(min(limit, end) - 1, 0, -1):
                candidate_text = render(max(0, end - candidate), end)
                if len(candidate_text) // CHARS_PER_TOKEN + 1 <= MAX_OUTPUT_TOKENS:
                    fit = candidate
                    break
            if fit is not None:
                remedy = f"call again with limit={fit} and offset={offset}"
            else:
                remedy = (
                    "even one message in this window is over the cap — raise "
                    "`offset` to step past it"
                )
            return ToolResult.failure(
                f"this call would be about {tokens:,} tokens — over the "
                f"{MAX_OUTPUT_TOKENS:,}-token limit. Change the call so it fits: "
                f"{remedy}, so the result stays under "
                f"{MAX_OUTPUT_TOKENS:,} tokens."
            )

        return ToolResult(
            text=text,
            data={
                "count": end - start,
                "offset": offset,
                "limit": limit,
                "logged": len(logged),
                "matched": len(entries),
                "query": query or None,
                "tokens": tokens,
                "older_left": start,
                "next_offset": len(entries) - start if start else None,
            },
            summary=f"showed {end - start} message(s) ({tokens:,} tokens)",
        )
