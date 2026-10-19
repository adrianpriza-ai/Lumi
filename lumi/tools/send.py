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
        "Send a message to the owner right now, mid-turn, and keep working. "
        "Use for progress updates during long work (\"done with phase 1, "
        "starting phase 2\") or to deliver the answer in parts across several "
        "messages. One call sends one message. Do NOT use it for your final "
        "answer — that is your normal reply text — and keep it to a handful "
        "of messages per turn."
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
