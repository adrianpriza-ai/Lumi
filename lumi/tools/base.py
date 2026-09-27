"""Tool abstractions.

A tool is anything the model can call. Each one declares a name, a description,
and a JSON Schema for its arguments; that is the entire contract with the LLM.

To add your own, subclass :class:`Tool`, put it in ``lumi/tools/``, and register
it in :func:`lumi.tools.build_registry`. Nothing else in the codebase needs to
change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util.text import truncate


class ToolError(RuntimeError):
    """A tool failed in a way the model should hear about and can act on."""


class NeedsApproval(Exception):
    """Raised by a tool that wants the owner to confirm before it acts.

    Carries everything the UI needs to render a Confirm/Cancel prompt and, later,
    to re-run the call for real.
    """

    def __init__(self, tool: str, arguments: dict[str, Any], reason: str, preview: str = "") -> None:
        super().__init__(reason)
        self.tool = tool
        self.arguments = arguments
        self.reason = reason
        self.preview = preview

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "reason": self.reason,
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeedsApproval:
        return cls(
            tool=data["tool"],
            arguments=data.get("arguments", {}),
            reason=data.get("reason", ""),
            preview=data.get("preview", ""),
        )


@dataclass(slots=True)
class ToolContext:
    """Per-invocation state that is not part of the model's arguments."""

    #: True when the owner has explicitly approved this specific call.
    confirmed: bool = False
    #: Working directory for shell-like tools.
    cwd: Path = field(default_factory=lambda: Path.cwd())
    #: Who asked: "model", "chat", or "cli".
    source: str = "model"
    chat_id: int | str | None = None

    def approve(self) -> ToolContext:
        return ToolContext(
            confirmed=True, cwd=self.cwd, source=self.source, chat_id=self.chat_id
        )


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back to the agent loop."""

    text: str
    ok: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    #: Short line echoed into the transcript so the log shows what was used.
    summary: str = ""

    @classmethod
    def failure(cls, text: str, **data: Any) -> ToolResult:
        return cls(text=text, ok=False, data=data, summary=text[:120])

    def for_model(self, max_chars: int = 8000) -> str:
        return truncate(self.text, max_chars)


class Tool(abc.ABC):
    """Base class for every capability exposed to the model."""

    #: Must match the tool-function name sent to the LLM.
    name: str = ""
    #: Written for the model, not for humans. Say when to use it and when not to.
    description: str = ""
    #: JSON Schema object (with "properties" and "required").
    parameters: dict[str, Any] = {}
    def available(self) -> tuple[bool, str]:
        """Whether this tool can run right now, plus a reason for the log."""
        return True, ""

    @abc.abstractmethod
    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Do the work. Raise :class:`ToolError` or :class:`NeedsApproval`."""

    def spec(self) -> dict[str, Any]:
        """The OpenAI function-calling envelope for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

    def summary_line(self) -> str:
        return f"`{self.name}`"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Tool {self.name}>"
