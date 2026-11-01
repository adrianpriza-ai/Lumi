"""Owner approval for the MCP server entries a *project* config file names.

``~/.agents/mcp.json`` is written by the owner's own hand, so its entries are
trusted as authored. The project ``.mcp.json`` is not: it lands in the
directory by ``git clone`` and ``git pull`` exactly as it does by the owner
editing it, and the two are indistinguishable to a file read. An entry in that
file can name a local command the provider will spawn or a remote URL it will
connect to with credentials-expanded headers — so before either happens, the
owner approves that exact entry, once, and the approval is remembered.

The memory is this module: a JSON file under ``data/`` mapping each approved
server name to the hash of the entry as it stood when approved. The hash —
not the name — is what gates: editing an entry changes its hash, so a pulled
branch that rewrites a command, a URL or a header asks for approval again,
while a rotated ``${VAR}`` value does not (the hash covers the entry as
written, placeholders intact). A missing or corrupt file reads as "nothing
approved", which fails closed.

The file is also a privileged write target — writing one is self-approval —
so it is listed in ``PROTECTED_PROJECT_PATHS``; see ``lumi.tools.safety``.
Approvals belong to the owner, through the chat's ``/mcp approve`` command.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ....util.log import get_logger

log = get_logger(__name__)

#: Where the approvals live, relative to the project root. ``data/`` is
#: already gitignored runtime state — the same shelf as the whitelist and the
#: OAuth tokens — and never travels with a clone.
TRUST_RELATIVE = ("data", "mcp-trust.json")


def trust_path(root: Path) -> Path:
    """The approval file for the project rooted at *root*."""
    return Path(root).joinpath(*TRUST_RELATIVE)


def entry_digest(raw: dict[str, Any]) -> str:
    """A stable hash of one config entry, exactly as the file wrote it.

    Canonical JSON (sorted keys, no whitespace) so re-saving the file with
    different indentation keeps an approval, while any change to a value the
    provider will act on — ``command``, ``args``, ``url``, ``headers``, env —
    breaks it. ``${VAR}`` placeholders hash as written: rotating a variable's
    value in the environment is not a config change, and must not re-prompt.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MCPTrustError(RuntimeError):
    """A project-config MCP entry tried to connect without the owner's approval.

    Raised by the provider's open gate, never caught into a retry: the remedy
    is a human decision (``/mcp approve <name>``), not a second attempt.
    """


class MCPTrustStore:
    """The approved project entries, on disk, keyed by name and content hash.

    One JSON file holds every approval: ``{"version": 1, "servers": {name:
    {"digest": …, "approved_at": …}}}``. Reading is strict about shape and
    lax about failure — anything missing, corrupt or of the wrong shape means
    "nothing approved", the safe reading. Writes are atomic and mode 0600,
    after :class:`~lumi.tools.web.providers.mcp_oauth.FileTokenStorage`: the
    file names which commands this bot will run, which makes it sensitive
    even though it holds no secret itself.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        servers = data.get("servers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            return {}
        return {name: entry for name, entry in servers.items() if isinstance(entry, dict)}

    def _write(self, servers: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps({"version": 1, "servers": servers}, ensure_ascii=False, indent=2)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, self.path)

    def is_trusted(self, name: str, digest: str) -> bool:
        """Whether the owner approved *name* with exactly this entry content."""
        entry = self._read().get(name)
        if not entry or not digest:
            return False
        return entry.get("digest") == digest

    def approve(self, name: str, digest: str) -> None:
        """Record the owner's approval of *name* as it stands at this digest."""
        servers = self._read()
        servers[name] = {
            "digest": digest,
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._write(servers)
        log.info("mcp server %r approved for connection (digest %s…)", name, digest[:12])

    def revoke(self, name: str) -> bool:
        """Withdraw the approval for *name*. True when there was one."""
        servers = self._read()
        if name not in servers:
            return False
        del servers[name]
        self._write(servers)
        log.info("mcp server %r approval revoked", name)
        return True


__all__ = ["MCPTrustError", "MCPTrustStore", "entry_digest", "trust_path"]
