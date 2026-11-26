"""MCP support in Lumi.

The public surface is one tool: :class:`~lumi.tools.mcp.codemode.MCPTool`
(``mcp``), which exposes every configured server as a generated code API and
runs model-written scripts in a sandbox. The rest of the package is the
machinery behind it:

* :mod:`~lumi.tools.mcp.servers` — config files, transports, sessions, discovery.
* :mod:`~lumi.tools.mcp.trust` — owner approval for project-file entries.
* :mod:`~lumi.tools.mcp.oauth` — browser login and token storage.
* :mod:`~lumi.tools.mcp.sandbox` — the isolated interpreter that runs scripts.
"""

from .codemode import CheckOutcome, MCPTool
from .servers import DiscoveredTool, MCPServers, ServerSpec

__all__ = [
    "MCPTool",
    "CheckOutcome",
    "MCPServers",
    "ServerSpec",
    "DiscoveredTool",
]
