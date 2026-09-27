"""Tool registry: name -> instance, plus the JSON Schemas the LLM sees."""

from __future__ import annotations

import json
from typing import Any

from ..util.log import get_logger
from ..util.text import format_error
from .base import NeedsApproval, Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- construction ------------------------------------------------------ #

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if not tool.description.strip():
            raise ValueError(f"tool {tool.name} has no description")
        self._tools[tool.name] = tool
        log.debug("registered tool %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    # -- LLM surface ------------------------------------------------------- #

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.all() if tool.available()[0]]

    def describe(self) -> str:
        if not self._tools:
            return "_no tools enabled_"
        lines = []
        for tool in self.all():
            ok, reason = tool.available()
            mark = "" if ok else f" _(unavailable: {reason})_"
            first = tool.description.strip().splitlines()[0] if tool.description.strip() else ""
            lines.append(f"- `{tool.name}`{mark} — {first}")
        return "\n".join(lines)

    # -- dispatch ---------------------------------------------------------- #

    async def invoke(
        self, name: str, arguments: dict[str, Any] | str, ctx: ToolContext
    ) -> ToolResult:
        """Run a tool. Never raises except :class:`NeedsApproval`.

        Errors are returned as failed results so the model can read them and
        retry; an exception here would kill the whole turn.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return ToolResult.failure(
                    f"Could not parse the arguments for {name}: {exc}. "
                    "Send a JSON object matching the schema."
                )
        if not isinstance(arguments, dict):
            return ToolResult.failure(f"Arguments for {name} must be a JSON object.")

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                f"Unknown tool {name!r}. Available tools: {', '.join(self.names()) or 'none'}."
            )

        ok, reason = tool.available()
        if not ok:
            return ToolResult.failure(f"Tool {name!r} is not available right now: {reason}")

        log.info("tool call: %s(%s)", name, _preview_args(arguments))
        try:
            result = await tool.invoke(arguments, ctx)
        except NeedsApproval:
            raise
        except ToolError as exc:
            log.warning("tool %s failed: %s", name, exc)
            return ToolResult.failure(f"{name} failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the bot
            log.exception("tool %s raised %s", name, type(exc).__name__)
            return ToolResult.failure(f"{name} crashed: {format_error(exc)}")

        if result.summary:
            log.info("tool %s -> %s", name, result.summary[:160])
        return result

    async def invoke_approved(
        self, pending: NeedsApproval, ctx: ToolContext
    ) -> ToolResult:
        """Re-run a previously blocked call, this time flagged as approved."""
        return await self.invoke(pending.tool, pending.arguments, ctx.approve())


def _preview_args(arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if len(rendered) > 80:
            rendered = rendered[:80] + "…"
        parts.append(f"{key}={rendered!r}")
    return ", ".join(parts) or "no args"
