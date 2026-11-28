"""The ``send`` tool: let the model say more than one thing per turn.

The agent loop delivers one final answer per turn, which is right for most
replies and wrong for long work: the owner would rather hear "phase 1 done,
starting phase 2" when it happens than read it three minutes later in a wall
of text. This tool is how the model says that out loud mid-turn.

The tool itself is deliberately dumb — it returns the text on
:attr:`~lumi.tools.base.ToolResult.messages` and the agent loop delivers it
through the live channel (:attr:`~lumi.tools.base.ToolContext.emit`). No tool
ever talks to Telegram directly.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..util.text import truncate
from .base import Tool, ToolContext, ToolResult

#: One Telegram message is capped at 4096 chars; leave room for the sanitizer
#: and the split threshold in ``split_message``. Longer text is truncated with
#: a marker rather than refused — a slightly short update beats a failed one.
MAX_MESSAGE_CHARS = 3500


class SendTool(Tool):
    name = "send"
    description = (
        "Tell the owner what you are doing, right now, while you keep working. "
        "The message lands in the chat immediately; it is not your final answer, "
        "which still arrives when the turn ends. Use it whenever a job takes "
        "more than a few steps: one line when you start, one when a phase "
        "finishes or the plan changes, one when something slow is running, and "
        "to deliver a long answer in parts. A sentence or two each; a quick task "
        "gets none. Never use it to ask a question — approval prompts do that."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send to the owner, as you would say it in chat.",
            }
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config | None = None) -> None:
        # Reserved for future knobs (rate caps, per-chat limits); taking it
        # now keeps the registry's factory signatures uniform.
        self._config = config

    def step_detail(self, arguments: dict[str, Any]) -> str:
        """The step feed reads ``✉️ <first words of the message>`` rather than a
        bare ``✉️ Send``: the owner watching the feed is the one the message is
        for, and a label that says what was sent is one less tap."""
        text = str(arguments.get("message", "")).strip()
        if not text:
            return self.name
        return f"{self.name} {truncate(text, 60, marker='…')}"

    async def invoke(self, arguments: dict[str, object], ctx: ToolContext) -> ToolResult:
        text = str(arguments.get("message", "")).strip()
        if not text:
            return ToolResult.failure("`message` must not be empty.")
        clipped = truncate(text, MAX_MESSAGE_CHARS, marker=" …[truncated]")
        return ToolResult(
            text="Sent to the owner. Continue working.",
            summary=truncate(clipped, 80, marker="…"),
            messages=[clipped],
        )
