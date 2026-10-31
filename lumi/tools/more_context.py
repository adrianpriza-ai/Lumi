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

from ..chatlog import CHATLOG, CHATLOG_MAXLEN, ChatLog
from ..config import Config
from .base import Tool, ToolContext, ToolResult

#: Rough chars-per-token for the output cap. Same divisor as ``read_file``;
#: the cap is a guardrail, not a budget, so a stable estimate beats a
#: dependency on the context window's live calibration.
CHARS_PER_TOKEN = 4

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


class MoreContextTool(Tool):
    name = "more_context"
    description = f"""\
Recent messages from this chat that are NOT part of your conversation.

In a group you only see messages addressed to you — everything the room says
between those mentions never reaches you. Call this when the owner points at
something you cannot see ("as they said above", "what is everyone talking
about?"), or when you need the talk around the message you are answering.

Parameters:
- `offset` (0-{MAX_OFFSET}): skip this many of the newest messages before
  reading. Raise it to walk further back; 0 (default) starts at the newest.
- `limit` (1-{MAX_MESSAGES}): how many messages to return, oldest first.
  Default {MAX_MESSAGES}.

Capped at {MAX_MESSAGES} messages and ~30k tokens per call: a result over the cap is
refused with the remedy (re-issue with a smaller `limit`) instead of being
truncated into something that looks complete. Only messages the bot did *not*
answer are logged — anything addressed to you is already in your
conversation, and private chats log nothing because there your conversation
holds everything. Photos and files sent to the room are logged too (with
their captions): the attachment is saved under `workspace/uploads/` and the
log line names its path, so you can read it with `read_file` — which handles
PDFs and images. The log lives in memory, so it starts fresh on a restart."""
    parameters = {
        "type": "object",
        "properties": {
            "offset": {
                "type": "integer",
                "description": (
                    "How many of the newest messages to skip before reading. "
                    "0 (default) starts at the newest."
                ),
                "minimum": 0,
                "maximum": MAX_OFFSET,
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"How many messages to return, oldest first. Defaults to {MAX_MESSAGES}."
                ),
                "minimum": 1,
                "maximum": MAX_MESSAGES,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, config: Config | None = None, log: ChatLog | None = None) -> None:
        # Both optional, matching send.py's habit of taking them now so the
        # registry's factory signatures stay uniform: *config* for future
        # knobs, *log* so tests can drive an isolated buffer instead of the
        # process-wide one.
        self._config = config
        self._log = log if log is not None else CHATLOG

    def summary_line(self) -> str:
        return (
            f"`{self.name}` — messages this chat never addressed to me "
            f"({MAX_MESSAGES} max per call, {MAX_OUTPUT_TOKENS:,}-token cap)"
        )

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        offset = _as_int(arguments.get("offset"), 0)
        if offset is None:
            return ToolResult.failure(f"`offset` must be a whole number 0-{MAX_OFFSET}.")
        if not 0 <= offset <= MAX_OFFSET:
            return ToolResult.failure(
                f"`offset` must be between 0 and {MAX_OFFSET} — got {offset}. "
                f"Use offset={max(0, min(offset, MAX_OFFSET))} and read again."
            )

        limit = _as_int(arguments.get("limit"), MAX_MESSAGES)
        if limit is None:
            return ToolResult.failure(f"`limit` must be a whole number 1-{MAX_MESSAGES}.")
        if not 1 <= limit <= MAX_MESSAGES:
            return ToolResult.failure(
                f"`limit` must be between 1 and {MAX_MESSAGES} — got {limit}. "
                f"Call again with limit={max(1, min(limit, MAX_MESSAGES))}."
            )

        entries = self._log.entries(ctx.chat_id)
        if not entries:
            return ToolResult.failure(
                "no messages are logged for this chat. Either nothing has been said "
                "since I arrived, or this is a private chat — there my conversation "
                "already holds everything, so there is nothing to fetch."
            )

        end = len(entries) - offset
        if end <= 0:
            return ToolResult.failure(
                f"offset {offset} is past the end — only {len(entries)} message(s) "
                "are logged for this chat. Call again with offset=0."
            )
        start = max(0, end - limit)
        window = entries[start:end]

        skipped_newer = offset
        older_left = start
        more = []
        if skipped_newer:
            more.append(f"{skipped_newer} newer skipped by offset")
        if older_left:
            more.append(
                f"{older_left} older not shown — raise offset/limit (max {MAX_OFFSET}) to walk back"
            )
        head = f"{len(window)} message(s) from this chat, oldest first"
        if more:
            head += " — " + "; ".join(more)
        body = "\n".join(entry.render(i) for i, entry in enumerate(window, start=1))
        text = f"{head}:\n{body}"

        tokens = len(text) // CHARS_PER_TOKEN + 1
        if tokens > MAX_OUTPUT_TOKENS:
            # Unreachable through Telegram alone (5 × 4096 chars ≈ 5k tokens),
            # but the cap is the contract: if the log ever holds something
            # bigger, the model hears "make the call smaller", not a wall.
            return ToolResult.failure(
                f"this call would be about {tokens:,} tokens — over the "
                f"{MAX_OUTPUT_TOKENS:,}-token limit. Change the call so it fits: "
                f"call again with a smaller limit (1-{MAX_MESSAGES}, one message at a "
                f"time if needed) and a higher offset, so each result stays under "
                f"{MAX_OUTPUT_TOKENS:,} tokens."
            )

        return ToolResult(
            text=text,
            data={
                "count": len(window),
                "offset": offset,
                "limit": limit,
                "logged": len(entries),
                "tokens": tokens,
            },
            summary=f"showed {len(window)} message(s) ({tokens:,} tokens)",
        )
