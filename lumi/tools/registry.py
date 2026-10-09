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

    def describe(self, available_only: bool = False, *, html: bool = False) -> str:
        """A human-readable summary of every registered tool.

        With ``available_only=True`` the list is filtered down to tools that
        can actually run right now — this is what the system prompt uses, so
        the model never sees a tool it cannot call. The default (everything)
        is what ``/tools`` shows, so the owner can see *why* something is off.

        With ``html=True`` the same list comes back as Telegram HTML —
        ``<code>`` around tool names, ``&lt;unavailable: …&gt;`` annotations —
        so it can be sent to the chat without an unbalanced underscore in a
        tool description breaking the whole message.
        """
        if not self._tools:
            return "<i>no tools enabled</i>" if html else "_no tools enabled_"
        if html:
            from ..util.text import escape_html, sanitize_html

            lines = []
            for tool in self.all():
                ok, reason = tool.available()
                if available_only and not ok:
                    continue
                mark = "" if ok else f" <i>(unavailable: {escape_html(reason)})</i>"
                first = tool.description.strip().splitlines()[0] if tool.description.strip() else ""
                lines.append(f"- <code>{tool.name}</code>{mark} — {sanitize_html(first)}")
            return "\n".join(lines)
        lines = []
        for tool in self.all():
            ok, reason = tool.available()
            if available_only and not ok:
                continue
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

    async def aclose(self) -> None:
        """Release every tool's held resources. Called once at shutdown, by both
        the bot and the CLI.

        A tool that holds anything across calls — an HTTP client pool, open
        files, a subprocess — implements ``async def aclose(self)``; the tools
        that do not are simply skipped. Best effort, like any shutdown path:
        one tool failing to close must neither stop the others from closing nor
        mask the exit reason.
        """
        for tool in self.all():
            close = getattr(tool, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                log.warning("tool %s failed to close: %s", tool.name, format_error(exc))


def _preview_args(arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if len(rendered) > 80:
            rendered = rendered[:80] + "…"
        parts.append(f"{key}={rendered!r}")
    return ", ".join(parts) or "no args"
