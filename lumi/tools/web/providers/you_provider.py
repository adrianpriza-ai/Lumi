"""You.com search, through its hosted MCP endpoint.

``https://api.you.com/mcp?profile=free`` is You.com's free, keyless MCP
profile — no account, no API key, a low rate limit. Because it speaks MCP,
this provider is the generic MCP client pointed at one fixed server: same
session reuse, same tool discovery, same payload parsing, no config file to
write. ``YOU_API_KEY``, when set, rides along as an ``X-API-Key`` header.

So a fresh clone has a working search without editing anything. Adding the
same server to ``.mcp.json`` still works — the ``mcp`` provider reads it as
before; this just means you do not have to.
"""

from __future__ import annotations

from pathlib import Path

from ....config import MCPConfig, YouConfig
from .mcp_provider import MCPProvider, ServerSpec


class YouProvider(MCPProvider):
    """The MCP stack with the config file swapped for one built-in server."""

    name = "you"

    def __init__(self, config: YouConfig, project_root: Path) -> None:
        # Empty config paths: no file is read, ``servers()`` below is the only
        # source. ``enabled`` still follows YouConfig.
        super().__init__(
            MCPConfig(enabled=config.enabled, config_file="", user_config_file=""),
            project_root,
        )
        self.you = config
        self._spec = ServerSpec(name="you", kind="remote", url=config.url, raw=[config.url])
        key = config.api_key()
        if key:
            self._spec.headers["X-API-Key"] = key

    @property
    def sources(self) -> list[str]:
        """Human-facing source name for messages; there is no file to name."""
        return ["you.com"]

    def servers(self) -> list[ServerSpec]:
        if not self.you.enabled or not self.you.url:
            return []
        return [self._spec]


__all__ = ["YouProvider"]
