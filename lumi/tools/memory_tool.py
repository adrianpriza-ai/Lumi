"""The ``memory`` tool: let the model read and append to MEMORY.md.

Deliberately asymmetric. The model can ``remember`` and ``recall``, but it can
never ``forget``: erasing something the owner told it is a human decision, made
with ``/forget`` in the chat. A prompt injection that convinces the model to
"forget the security rules" should not be able to rewrite the operator's notes.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..memory import MemoryFile
from ..util.log import get_logger
from ..util.text import truncate
from .base import Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)


class MemoryTool(Tool):
    name = "memory"
    description = """
Your long-term memory, stored as a markdown file you can also read and edit by hand.

Actions:
- `remember`: save one short, durable fact about the owner or their work. Phrase
  it as a standalone statement, not a transcript. Do not save secrets, and do
  not save anything the owner asked you to forget.
- `recall`: search existing memories. Useful when the owner refers to something
  you do not have in the conversation.
- `show`: print the whole file.

Saving is cheap and useful, but be selective: memories are injected into every
future conversation, so trivia crowds out signal. One fact per call.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall", "show"],
                "description": "remember = save a fact; recall = search; show = print everything.",
            },
            "fact": {
                "type": "string",
                "description": "The fact to save. Required for remember. One short declarative sentence.",
            },
            "query": {
                "type": "string",
                "description": "Keywords to search for. Required for recall.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config, memory: MemoryFile) -> None:
        self.config = config
        self.settings = config.tools.memory
        self.memory = memory

    def available(self) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "disabled in config (tools.memory.enabled = false)"
        return True, ""

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        self.memory.ensure_loaded()

        if action == "remember":
            if not self.settings.auto_remember:
                return ToolResult.failure(
                    "auto_remember is off in config, so I cannot save memories right now. "
                    "The owner can add one with /remember."
                )
            fact = arguments.get("fact")
            if not isinstance(fact, str) or not fact.strip():
                raise ToolError("fact is required when action is 'remember'")
            message = self.memory.remember(fact, source=ctx.source)
            return ToolResult(text=message, data={"fact": fact}, summary=truncate(message, 100, "…"))

        if action == "recall":
            query = str(arguments.get("query") or "").strip()
            if not query:
                raise ToolError("query is required when action is 'recall'")
            hits = self.memory.search(query, limit=15)
            if not hits:
                return ToolResult(text=f"no memories match {query!r}", data={"hits": []})
            body = "\n".join(f"- {hit}" for hit in hits)
            return ToolResult(
                text=f"{len(hits)} memories matching {query!r}:\n{body}",
                data={"hits": hits},
                summary=f"recalled {len(hits)} memories for {query!r}",
            )

        if action == "show":
            managed = self.memory.managed()
            body = self.memory.for_prompt()
            return ToolResult(
                text=body,
                data={"managed": managed, "stats": self.memory.stats()},
                summary=f"showed memory ({len(managed)} managed facts)",
            )

        raise ToolError(f"unknown action {action!r}; use remember, recall or show")

    def summary_line(self) -> str:
        mode = "read + append" if self.settings.auto_remember else "read only"
        return f"`{self.name}` — long-term memory in MEMORY.md ({mode}; forget is human-only)"
