"""Tool registry and the tools that ship with Lumi.

To add a tool:

1. subclass :class:`~lumi.tools.base.Tool` in a new file here,
2. register it in :func:`build_registry`.

It appears in the model's tool list immediately, with no other changes. The
prompt in :mod:`lumi.agent` refers to capabilities by name, so keep the
description field honest about when *not* to use the tool.
"""

from __future__ import annotations

from ..config import Config
from ..memory import MemoryStore
from ..util.log import get_logger
from .base import NeedsApproval, Tool, ToolContext, ToolError, ToolResult
from .context7 import Context7Tool
from .explore import GrepTool, ListFilesTool, ReadFileTool
from .files import FilesTool
from .mcp_tool import MCPTool
from .memory_tool import MemoryTool
from .more_context import MoreContextTool
from .ocr import OCRTool
from .registry import ToolRegistry
from .send import SendTool
from .shell import ShellTool
from .skills import SkillsTool
from .web import WebTool

log = get_logger(__name__)


def build_registry(config: Config, memory: MemoryStore) -> ToolRegistry:
    """Construct the registry, honouring ``tools.enabled`` and the per-tool flags.

    Tools that fail ``available()`` (e.g. ``context7`` without an API key) are
    still registered, so introspection commands like ``/doctor`` (and
    ``/tools`` in ``lumi chat``) can show why they are off — but ``ToolRegistry.specs()`` excludes them, so
    the model never sees a tool it cannot actually call. The system prompt
    follows the same rule via :meth:`ToolRegistry.describe` with
    ``available_only=True``.
    """
    registry = ToolRegistry()
    candidates = {
        "shell": lambda: ShellTool(config),
        "files": lambda: FilesTool(config),
        "read_file": lambda: ReadFileTool(config),
        "list_files": lambda: ListFilesTool(config),
        "grep": lambda: GrepTool(config),
        "memory": lambda: MemoryTool(config, memory),
        "more_context": lambda: MoreContextTool(config),
        "web": lambda: WebTool(config),
        "mcp": lambda: MCPTool(config),
        "context7": lambda: Context7Tool(config),
        "ocr": lambda: OCRTool(config),
        "send": lambda: SendTool(config),
        "skills": lambda: SkillsTool(config),
    }

    for name, factory in candidates.items():
        if not config.tools.is_enabled(name):
            log.info("tool %r disabled by tools.enabled", name)
            continue
        try:
            tool = factory()
        except Exception as exc:  # noqa: BLE001 - one bad tool must not stop the bot
            log.exception("could not build tool %r: %s", name, exc)
            continue
        ok, reason = tool.available()
        if not ok:
            log.warning("tool %r registered but unavailable: %s", name, reason)
        registry.register(tool)

    log.info("registry ready with %d tool(s): %s", len(registry), ", ".join(registry.names()))
    return registry


__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolError",
    "NeedsApproval",
    "ToolRegistry",
    "ShellTool",
    "FilesTool",
    "ReadFileTool",
    "ListFilesTool",
    "GrepTool",
    "MemoryTool",
    "MoreContextTool",
    "WebTool",
    "MCPTool",
    "Context7Tool",
    "OCRTool",
    "SendTool",
    "SkillsTool",
    "build_registry",
]
