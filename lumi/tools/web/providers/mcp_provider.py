"""Any MCP server as a web provider.

This is what makes ``.mcp.json`` meaningful for Lumi. Two files are read: the
project's ``.mcp.json``, and the user-level ``~/.agents/mcp.json`` for servers
configured once for every project. ``${VAR}`` placeholders are expanded from the
environment, and the server is spoken to over MCP proper — so a Tavily or
Firecrawl MCP server dropped into either file works immediately, with no code
change. A locally spawned server (``npx -y some-mcp-server``) works the same
way. When a server name appears in both files, the project file wins.

One asymmetry between the two files: the user-level one is written by the
owner's own hand, but the project one arrives by ``git clone`` exactly as it
does by editing, so its entries connect only after the owner approved that
exact entry — see :mod:`~lumi.tools.web.providers.mcp_trust`. The gate sits in
:meth:`MCPProvider._open`, which every trigger flows through, so a fresh entry
costs a ``/mcp approve`` before ``mcp list``, a web search or a call spawns or
dials it.

Two transport kinds are supported, under either spelling of ``"type"``:

* ``"type": "remote"`` (or ``"http"``) — streamable HTTP, the shape opencode and
  Claude Code use for hosted servers like ``https://mcp.tavily.com/mcp/``.
* ``"type": "local"`` (or ``"stdio"``) — a subprocess over stdio, spawned with
  ``command``/``args``. A config with a ``command`` and no ``type`` is local too,
  so the two common file formats both work as written.

A local server runs in the project root — or in its own ``cwd`` entry, when
the config names one — so a relative path in ``args`` means what it looks like
relative to the project that configured it, rather than to wherever Lumi
ehappened to be launched from.

Tool names are not hardcoded. The provider asks each server what it offers and
picks the first tool matching the configured preference list, then reads that
tool's own input schema to work out whether it wants ``query``, ``q`` or
something else. That is what lets one implementation front an arbitrary server.

The provider spans the whole ``mcp`` package line the dependency allows
(``mcp>=1.2``): the places where the SDK renamed things between major
versions — the HTTP transport and its headers, ``list_tools`` pagination,
the read-timeout type, the error class, and the result field spellings —
each resolve against the installed SDK at use time rather than assuming one
version. The stub sessions in the tests accept anything, so an unadapted
shape only ever failed against a real server.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from ....config import MCPConfig
from ....paths import resolve
from ....util.env import scrubbed_env
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider, final_url, recent_cutoff
from .mcp_trust import MCPTrustError, MCPTrustStore, entry_digest, trust_path

log = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Alternate spellings of ``"type"``, normalised as the config is parsed, so a
#: file written against the MCP spec (``"stdio"``) and one written against this
#: project (``"local"``) describe the same transport everywhere downstream.
_KIND_ALIASES = {"stdio": "local", "http": "remote"}

#: How long a discovered tool list stays valid before we re-ask the server.
TOOL_CACHE_TTL = 300.0

#: Cap on the cursor walk in :meth:`MCPProvider._list_tools`. Servers paginate
#: at their own discretion, and one that hands the same cursor back forever
#: must not be able to keep us paging.
LIST_PAGES_MAX = 10


try:  # mcp is optional: the provider imports without it, only usage needs it
    from mcp.shared.exceptions import McpError as _McpErrorV1
except ImportError:
    _McpErrorV1 = None  # type: ignore[assignment]
try:  # mcp 2.0 renamed the class; the old spelling is gone, not aliased
    from mcp.shared.exceptions import MCPError as _McpErrorV2
except ImportError:
    _McpErrorV2 = None  # type: ignore[assignment]
try:
    from mcp.exceptions import McpError as _McpErrorOld
except ImportError:
    _McpErrorOld = None  # type: ignore[assignment]

#: The error code the SDK hands pending requests when the transport's receive
#: stream ends — its tombstone for "the wire died under you", not the server's
#: word. Reserved by the JSON-RPC/MCP spec, so the fallback is exact even if a
#: future SDK moves the constant.
try:
    from mcp.types import CONNECTION_CLOSED as _CONNECTION_CLOSED
except ImportError:  # pragma: no cover - mcp always ships this constant
    _CONNECTION_CLOSED = -32000

#: Every class the installed SDK answers an MCP error with. Empty only when
#: the mcp package is absent at all — each spelling the installed version
#: lacks raises ImportError and contributes nothing. Importing only the 1.x
#: name used to degrade silently to ``None`` on 2.x, which made every
#: server-side *reply* look like a dead transport (see
#: :func:`_session_died`).
MCP_ERRORS: tuple[type[BaseException], ...] = tuple(
    cls for cls in (_McpErrorV1, _McpErrorV2, _McpErrorOld) if cls is not None
)


def _error_code(exc: BaseException) -> int | None:
    """The MCP error code *exc* carries, when it is an MCP error at all."""
    code = getattr(getattr(exc, "error", None), "code", None)
    try:
        return int(code)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _session_died(exc: BaseException) -> bool:
    """Whether *exc* means the session is dead (drop it) or the server answered.

    A trust refusal is neither death nor reply — it never came from the wire,
    and the remedy is the owner's ``/mcp approve``, not a respawn — so it
    returns False with the timeout. A timeout itself means the request went
    out and may already have been acted on, so it is never a death either.

    The subtle case is an MCP error whose code is ``CONNECTION_CLOSED``: it
    *looks* like a reply, but it is the SDK's own tombstone — when the wire
    dies while requests are in flight, pending requests each receive an MCP
    error with that code rather than a transport exception. Validated against
    the real stdio and streamable-HTTP transports: a server that acts on a
    request and dies before replying surfaces exactly this way. Every other
    MCP error is the server replying, however unhappy — nothing to reconnect
    to. Any non-MCP exception is a dead transport.
    """
    if isinstance(exc, (MCPTrustError, TimeoutError)):  # >= 3.11: asyncio's is this
        return False
    if MCP_ERRORS:
        if isinstance(exc, MCP_ERRORS):
            return _error_code(exc) == _CONNECTION_CLOSED
        return True
    return type(exc).__name__ not in {"McpError", "MCPError"}


#: Decided once from the installed SDK's own signature: does
#: ``read_timeout_seconds`` want a ``timedelta`` (1.x) or a float of seconds
#: (2.x)? ``None`` until :func:`_read_timeout` first runs.
_READ_TIMEOUT_WANTS_TD: bool | None = None


def _sdk_timeout_is_timedelta() -> bool:
    """Whether the installed ``ClientSession.call_tool`` wants a timedelta.

    Falls back to the 1.x shape when the signature cannot be read, so a
    future SDK surprise costs one timeout type, not a crash.
    """
    import inspect

    try:
        from mcp import ClientSession

        annotation = inspect.signature(ClientSession.call_tool).parameters[
            "read_timeout_seconds"
        ].annotation
    except Exception:  # noqa: BLE001 - mcp absent or unreadable: 1.x shape it is
        return True
    return "timedelta" in str(annotation)


def _read_timeout(seconds: float | int) -> timedelta | float:
    """The ``read_timeout_seconds`` the installed MCP SDK actually accepts.

    The SDK reversed the type between major versions: 1.x typed it
    ``timedelta | None`` and called ``.total_seconds()`` on whatever it was
    handed (a bare float raised ``AttributeError`` the moment the request went
    out), while 2.x spends the budget as arithmetic on a float of seconds (a
    ``timedelta`` raises there). The stub sessions in the tests accept
    anything, which is how each wrong shape survived against one SDK: it only
    ever failed against a real server. So the installed SDK's own signature
    decides — the same evidence a real server would force.
    """
    global _READ_TIMEOUT_WANTS_TD
    if _READ_TIMEOUT_WANTS_TD is None:
        _READ_TIMEOUT_WANTS_TD = _sdk_timeout_is_timedelta()
    return timedelta(seconds=float(seconds)) if _READ_TIMEOUT_WANTS_TD else float(seconds)


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` from the process environment."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or (default or "")

    return _ENV_PATTERN.sub(replace, value)


@dataclass(slots=True)
class ServerSpec:
    """One entry from an MCP config file (``.mcp.json`` or ``~/.agents/mcp.json``)."""

    name: str
    kind: str = "remote"  # "remote" | "local"
    url: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    #: Directory a local server is spawned in (``cwd`` in the config). Empty
    #: means the project root — never "wherever the bot was launched", which
    #: would make a relative ``args`` path depend on the shell it started in.
    cwd: str = ""
    #: The unexpanded strings, kept so we can still tell "unset variable" apart
    #: from "variable set to the empty string" after expansion has run.
    raw: list[str] = field(default_factory=list)
    #: ``"auth": "oauth"`` in the server's entry: the server wants the *owner's*
    #: authorization, negotiated per :mod:`~lumi.tools.web.providers.mcp_oauth`
    #: (browser login, stored tokens) rather than a static header token.
    oauth: bool = False
    #: Scopes the authorization request asks for (``"scopes"`` in the dict
    #: form of ``auth``), space-joined. Most servers grant everything the
    #: account can reach without being told anything, but some — Atlassian
    #: above all — only mint a *refresh* token when the request names scopes,
    #: which is the difference between logging in once and once an hour.
    oauth_scope: str = ""
    #: Where this entry was read from: ``"project"`` (``.mcp.json``, which
    #: arrives by clone as easily as by hand and so needs the owner's
    #: approval before it may connect) or ``"user"`` (``~/.agents/mcp.json``,
    #: written by the owner's own hand, trusted as authored).
    origin: str = "user"
    #: Hash of the entry exactly as the config file wrote it — the key the
    #: trust store approves against. Empty for a spec built in code rather
    #: than parsed from a file, which reads as "not a project entry".
    digest: str = ""

    @property
    def is_local(self) -> bool:
        return self.kind == "local" or bool(self.command)

    def unresolved(self) -> str | None:
        """Return the first variable this server needs but the environment lacks."""
        for candidate in self.raw or [self.url, *self.args, *self.env.values(), *self.headers.values()]:
            match = _ENV_PATTERN.search(str(candidate))
            if match and match.group(2) is None and not os.environ.get(match.group(1)):
                return match.group(1)
        return None

    @classmethod
    def from_json(cls, name: str, raw: dict[str, Any], origin: str = "user") -> ServerSpec:
        command = raw.get("command")
        if isinstance(command, list):  # opencode style: ["npx", "-y", "pkg"]
            command_parts = [str(part) for part in command]
        elif command:
            command_parts = [str(command)]
        else:
            command_parts = []
        raw_args = raw.get("args", [])
        args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
        raw_url = str(raw.get("url") or "")
        raw_env_value = raw.get("env") or {}
        raw_headers_value = raw.get("headers") or {}
        raw_cwd = str(raw.get("cwd") or "")
        raw_auth = raw.get("auth")
        # "oauth" as a string or as {"type": "oauth"} — the dict shape is also
        # where per-flow options live, currently "scopes" (string or list).
        oauth = raw_auth == "oauth" or (
            isinstance(raw_auth, dict) and raw_auth.get("type") == "oauth"
        )
        raw_scopes = ""
        if isinstance(raw_auth, dict):
            scopes = raw_auth.get("scopes")
            if isinstance(scopes, str):
                raw_scopes = scopes
            elif isinstance(scopes, list):
                raw_scopes = " ".join(str(item) for item in scopes)
        raw_env = (
            {str(k): str(v) for k, v in raw_env_value.items()}
            if isinstance(raw_env_value, dict)
            else {}
        )
        raw_headers = (
            {str(k): str(v) for k, v in raw_headers_value.items()}
            if isinstance(raw_headers_value, dict)
            else {}
        )

        return cls(
            name=name,
            # "stdio" is the spelling the MCP spec and Claude Code use for a
            # spawned server; "local" is this project's word for the same thing.
            # Normalising keeps `is_local` — and every message that reports a
            # transport — true whatever the file called it.
            kind=_KIND_ALIASES.get(
                str(raw.get("type") or "").strip().lower(),
                str(raw.get("type") or ("local" if command_parts else "remote")),
            ),
            url=expand_env(raw_url),
            command=command_parts[0] if command_parts else "",
            args=[expand_env(a) for a in command_parts[1:] + args],
            env={k: expand_env(v) for k, v in raw_env.items()},
            headers={k: expand_env(v) for k, v in raw_headers.items()},
            cwd=expand_env(raw_cwd),
            raw=[
                raw_url,
                *command_parts,
                *args,
                *raw_env.values(),
                *raw_headers.values(),
                raw_cwd,
                raw_scopes,
            ],
            oauth=oauth,
            oauth_scope=expand_env(raw_scopes),
            origin=origin,
            # The hash covers the entry as *written* — placeholders intact —
            # so an edit to anything the provider will act on asks for
            # approval again, while a rotated variable's value does not.
            digest=entry_digest(raw),
        )


def _http_transport(spec: ServerSpec, root: Path):
    """The streamable-HTTP async context manager for *spec*, headers and auth.

    The transport was renamed twice and the way custom headers are passed
    moved with it: ``streamablehttp_client(url, headers=…)`` through 1.25,
    ``streamable_http_client(url, http_client=…)`` from 1.26, and only from
    2.3 a factory (``create_mcp_http_client``) to build that client with
    headers. Whichever shape the installed SDK has, this returns one context
    manager with *spec*'s headers applied — or one clear error when the
    installed version genuinely cannot carry them, instead of an ImportError
    that would read as "server unreachable".

    An OAuth server (``"auth": "oauth"``) rides the same two shapes: the
    factory takes the auth handler directly (``auth=``), the legacy
    transport takes it as its own kwarg. Both factories exist across the
    supported range, so the handler is built once in
    :func:`~lumi.tools.web.providers.mcp_oauth.oauth_provider_for` and handed
    to whichever transport the installed SDK answers with.
    """
    from mcp.client import streamable_http

    auth = None
    if spec.oauth:
        from .mcp_oauth import oauth_provider_for

        auth = oauth_provider_for(spec, root)

    factory = getattr(streamable_http, "create_mcp_http_client", None)
    transport = getattr(streamable_http, "streamable_http_client", None)
    if transport is not None and (factory is not None or not (spec.headers or auth)):
        if factory is not None and (spec.headers or auth):
            http_client = factory(headers=spec.headers or None, auth=auth)
        else:
            http_client = None
        return transport(spec.url, http_client=http_client)
    legacy = getattr(streamable_http, "streamablehttp_client", None)
    if legacy is not None:  # pre-2.0: headers and auth are direct kwargs
        return legacy(spec.url, headers=spec.headers or None, auth=auth)
    raise RuntimeError(
        f"server {spec.name!r} declares custom headers, but the installed mcp "
        "package cannot carry them on streamable HTTP (needs mcp >= 2.3, or "
        "the pre-2.0 transport); upgrade the mcp package or use stdio"
    )


def _stdio_env(spec: ServerSpec) -> dict[str, str]:
    """The environment a local MCP server is spawned with.

    A server is a program the owner configured, but it is also a program whose
    output the model reads, so it gets the same allowlist scrub as
    ``run_shell`` rather than the bot's whole environment. Variables the owner
    names in the server's ``env`` block are explicit and still pass through.
    """
    return scrubbed_env(overrides=spec.env)


def _stdio_parameters(spec: ServerSpec, cwd: Path):
    """The SDK's spawn parameters for *spec*, running in *cwd*.

    ``cwd`` is the whole point of a local server's directory: without it the
    child inherits Lumi's working directory, so a relative ``args`` path (and
    every file the server opens itself) resolves against wherever the bot was
    launched instead of against the project that configured it.

    It is passed conditionally because ``StdioServerParameters`` is not the
    same shape across SDK versions — an older one takes these as a dataclass
    and would raise on an unknown keyword. The field is only omitted when the
    installed version genuinely lacks it, and the log line says so rather than
    leaving the wrong directory unexplained.
    """
    from mcp import StdioServerParameters

    kwargs: dict[str, Any] = {
        "command": spec.command,
        "args": spec.args,
        "env": _stdio_env(spec),
    }
    known = getattr(StdioServerParameters, "__annotations__", None) or {}
    if "cwd" in known:
        kwargs["cwd"] = str(cwd)
    else:  # pragma: no cover - only an SDK older than the cwd field lands here
        log.warning(
            "the installed mcp package has no StdioServerParameters.cwd; "
            "server %r runs in %s instead of %s",
            spec.name,
            Path.cwd(),
            cwd,
        )
    return StdioServerParameters(**kwargs)


@dataclass(slots=True)
class DiscoveredTool:
    name: str
    description: str
    properties: dict[str, Any]
    #: Which parameters the server marks required — the ``mcp`` tool's
    #: ``list`` action shows it, so a call to a six-argument tool is built
    #: right the first time instead of by trial and error.
    required: list[str] = field(default_factory=list)

    def arg_for(self, candidates: tuple[str, ...]) -> str | None:
        """Which of *candidates* this tool actually accepts as a parameter."""
        lowered = {key.lower(): key for key in self.properties}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        # Fall back to any single string-ish parameter so unknown servers work.
        strings = [
            key
            for key, spec in self.properties.items()
            if spec.get("type") in {"string", None}
            and key.lower() not in {"max_results", "limit", "topic", "depth", "format"}
        ]
        return strings[0] if len(strings) == 1 else None


class MCPProvider(WebProvider):
    name = "mcp"

    def __init__(self, config: MCPConfig, project_root: Path) -> None:
        self.config = config
        self.root = project_root
        #: Owner approvals for project-file entries — the gate :meth:`_open`
        #: enforces. See :mod:`~lumi.tools.web.providers.mcp_trust`.
        self.trust = MCPTrustStore(trust_path(project_root))
        self._tool_cache: dict[str, tuple[float, list[DiscoveredTool]]] = {}
        #: Live session per server name: (stack owning the transport, session).
        #: Discovery plus a call used to pay for two spawns — and a local
        #: ``npx`` server spends most of that booting.
        self._sessions: dict[str, tuple[AsyncExitStack, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        #: mtime fingerprint of the config files at the moment the caches were
        #: filled — :meth:`_expire_stale_caches` compares against it.
        self._config_stamp: tuple[tuple[str, int], ...] | None = None

    # -- configuration ----------------------------------------------------- #

    @property
    def config_path(self) -> Path:
        """The project-level file (messages that name exactly one file use it)."""
        return resolve(self.config.config_file, base=self.root)

    @property
    def config_paths(self) -> list[Path]:
        """Every configured file that will be read, user-level first.

        Deduplicated, so pointing ``user_config_file`` at ``config_file`` reads
        it once. Relative paths resolve against the project root, ``~`` against
        the home directory.
        """
        paths: list[Path] = []
        for configured in (self.config.user_config_file, self.config.config_file):
            if not configured:
                continue
            path = resolve(configured, base=self.root)
            if path not in paths:
                paths.append(path)
        return paths

    @property
    def sources(self) -> list[str]:
        """The configured file names as written, for human-facing messages."""
        return list(
            dict.fromkeys(
                c for c in (self.config.user_config_file, self.config.config_file) if c
            )
        )

    def _read_servers(self, path: Path, origin: str = "user") -> dict[str, ServerSpec]:
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", path, exc)
            return {}
        if not isinstance(raw, dict):
            log.warning("%s has no usable object at the top level", path)
            return {}
        block = raw.get("mcpServers") or raw.get("servers") or {}
        if not isinstance(block, dict):
            log.warning("%s has no usable mcpServers object", path)
            return {}
        servers: dict[str, ServerSpec] = {}
        for name, cfg in block.items():
            if not isinstance(name, str) or not isinstance(cfg, dict):
                log.warning("%s contains an invalid MCP server entry %r", path, name)
                continue
            try:
                servers[name] = ServerSpec.from_json(name, cfg, origin=origin)
            except (TypeError, ValueError) as exc:
                log.warning("could not parse MCP server %r from %s: %s", name, path, exc)
        return servers

    def servers(self) -> list[ServerSpec]:
        """Servers from every configured file, merged.

        The files are read in :attr:`config_paths` order, so the project file
        (last) both overrides a duplicate name and leads the list, shadowing a
        user-level entry everywhere the list is walked. Each entry remembers
        which file it came from: the project file's entries are the ones the
        trust gate holds, since that file arrives by clone as easily as by
        hand, while the user-level one is the owner's own writing.
        """
        project = self.config_path
        tables = [
            self._read_servers(path, origin="project" if path == project else "user")
            for path in self.config_paths
        ]
        if not tables:
            return []
        merged: dict[str, ServerSpec] = {}
        for table in tables:
            merged.update(table)
        order: dict[str, None] = {}
        for table in reversed(tables):
            for name in table:
                order.setdefault(name, None)
        return [merged[name] for name in order]

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        try:
            import mcp  # noqa: F401
        except ImportError:
            return False, "the mcp package is not installed"
        files = ", ".join(self.sources) or "the configured files"
        servers = self.servers()
        if not servers:
            return False, f"no servers in {files}"
        missing = [f"{s.name} (needs {s.unresolved()})" for s in servers if s.unresolved()]
        if missing and len(missing) == len(servers):
            return False, f"unset variables in {files}: {', '.join(missing)}"
        return True, ""

    # -- session plumbing -------------------------------------------------- #

    def _cwd_for(self, spec: ServerSpec) -> Path:
        """The directory a local server is spawned in.

        The project root by default: ``args`` in ``.mcp.json`` are written by
        someone thinking about the project, and whatever started Lumi is not
        part of that picture — a systemd unit, a wrapper script or a stray
        shell must not change what ``server.py`` resolves to. An explicit
        ``cwd`` in the config wins, resolved against the project root the same
        way ``config_file`` is, so ``~`` and relatives behave predictably.
        """
        if spec.cwd:
            return resolve(spec.cwd, base=self.root)
        return self.root

    def _trust_refusal(self, spec: ServerSpec) -> str | None:
        """Why *spec* may not connect yet, or ``None`` when it may.

        The invariant: an entry in the project config file — which a clone or
        a pull delivers just as an owner's edit does — causes no connection,
        spawn or HTTP, until the owner approved that exact entry. User-level
        entries are the owner's own writing and skip the gate; so does a spec
        built in code without a digest (the tests' stubs, mostly). The check
        lives here rather than at each caller because every trigger — the web
        tool's ``_pick``, the ``mcp`` tool's ``list`` and ``call``, ``check``
        — funnels through :meth:`_open`.
        """
        if spec.origin != "project" or not spec.digest:
            return None
        if self.trust.is_trusted(spec.name, spec.digest):
            return None
        return (
            f"server {spec.name!r} comes from the project config and has not been "
            f"approved to connect — the owner can approve it once with "
            f"`/mcp approve {spec.name}`"
        )

    async def _open(self, stack: AsyncExitStack, spec: ServerSpec):
        """Start a session against *spec* and return it, already initialised.

        The trust gate runs before anything is spawned or dialed: a refused
        entry costs no subprocess, no HTTP request and no header — not even
        the ones with expanded secrets in them.
        """
        if refusal := self._trust_refusal(spec):
            raise MCPTrustError(refusal)

        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        if spec.is_local:
            if not spec.command:
                raise RuntimeError(f"server {spec.name} has no command")
            params = _stdio_parameters(spec, self._cwd_for(spec))
            streams = await stack.enter_async_context(stdio_client(params))
        else:
            if not spec.url:
                raise RuntimeError(f"server {spec.name} has no url")
            transport = await stack.enter_async_context(_http_transport(spec, self.root))
            # An older stdio/HTTP transport yields 3 entries (the third is the
            # session id); only the first two are the session's read/write
            # channels, and the current one already yields exactly those two.
            streams = transport[:2]

        read_stream, write_stream = streams[0], streams[1]
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    def _lock_for(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[name] = lock
        return lock

    async def _session_for(self, spec: ServerSpec) -> Any:
        """The live session for *spec*, opened once and reused across calls.

        One spawn per server rather than one per step. The entry's
        :class:`AsyncExitStack` owns the transport, so the session and the
        subprocess behind it live exactly as long as the cache entry.
        """
        entry = self._sessions.get(spec.name)
        if entry is not None:
            return entry[1]
        async with self._lock_for(spec.name):
            entry = self._sessions.get(spec.name)
            if entry is None:
                stack = AsyncExitStack()
                try:
                    session = await self._open(stack, spec)
                except BaseException:
                    await stack.aclose()
                    raise
                self._sessions[spec.name] = (stack, session)
                log.debug("mcp session opened for server %r", spec.name)
                return session
        return self._sessions[spec.name][1]

    async def _drop_session(self, name: str) -> None:
        """Close and forget the cached session for *name*, if there is one.

        Every session opened *after* it is closed too, newest first. Each
        transport entered an anyio scope in this task, and those scopes unwind
        last-in-first-out: closing an older stack while a newer one is still
        open corrupts the task's cancel-scope bookkeeping, and the *next* await
        dies with ``Cancelled via cancel scope`` rather than doing any work.
        That is not hypothetical — it is what ``/mcp check`` did whenever
        another server was already connected (and what a dead session in
        ``_reuse`` would do too).

        The extra closes cost one re-spawn each, paid on demand the next time
        those servers are used.
        """
        names = list(self._sessions)
        if name not in self._sessions:
            return
        for later in reversed(names[names.index(name) :]):
            entry = self._sessions.pop(later, None)
            if entry is None:
                continue
            stack, _session = entry
            with suppress(Exception):  # shutdown is best-effort
                await stack.aclose()

    async def _reuse(self, spec: ServerSpec, op: Any, *, repeatable: bool) -> Any:
        """Run *op(session)* on the cached session; handle a dead transport.

        What a dead transport earns depends on what *op* is, so the caller
        says:

        * ``repeatable=True`` — *op* re-runs safely, which is tool discovery:
          a re-sent ``tools/list`` cannot repeat anything the server did. The
          dead session is dropped and *op* re-sent once on a fresh one, so a
          server that went away between queries costs one extra launch, not
          every later call.
        * ``repeatable=False`` — *op* is a tool *call*. The request may
          already have been acted on by the time the transport dies, and
          which exception a mid-flight disconnect produces is the SDK's
          business (the ``CONNECTION_CLOSED`` tombstone, a transport error,
          an ``httpx.RemoteProtocolError``) — none of them proves the server
          did *not* act. So the dead session is dropped and the error raised
          as-is: this call fails, the next one spawns fresh. Recovery is
          preserved across calls; the write is never risked twice.

        A timeout is not a death in either mode — the session may be slow,
        not gone — and a server reply is never a death either.
        """
        session = await self._session_for(spec)
        try:
            return await op(session)
        except Exception as exc:
            if not _session_died(exc):
                raise
            log.debug("mcp session for %r died; dropping it", spec.name)
            await self._drop_session(spec.name)
            if not repeatable:
                raise
            try:
                return await op(await self._session_for(spec))
            except Exception as retry_exc:
                if _session_died(retry_exc):
                    await self._drop_session(spec.name)
                raise

    async def aclose(self) -> None:
        """Close every cached session. Called at shutdown via the web tool."""
        # Reverse open order: each transport entered an anyio scope in this
        # task, and anyio requires scopes to unwind last-in-first-out.
        for name in reversed(list(self._sessions)):
            await self._drop_session(name)

    async def _expire_stale_caches(self) -> None:
        """Drop the tool cache and every session when a config file changed.

        The registry — and therefore this provider — lives across ``/reload``,
        which re-reads only personality and memory. Without this check an
        owner who edits ``.mcp.json`` waited out :data:`TOOL_CACHE_TTL` with
        sessions still pinned to servers the file no longer describes, and a
        renamed ``command`` kept respawning the old process. A ``stat()`` per
        discovery costs nothing next to an MCP round trip, and the first call
        only *records* the stamp — there is nothing stale before the first use.
        """
        stamp = tuple(
            (str(path), path.stat().st_mtime_ns if path.exists() else -1)
            for path in self.config_paths
        )
        if stamp == self._config_stamp:
            return
        first_use = self._config_stamp is None
        self._config_stamp = stamp
        if first_use:
            return
        log.info("mcp config file(s) changed on disk; dropping cached tools and sessions")
        self._tool_cache.clear()
        for name in reversed(list(self._sessions)):
            await self._drop_session(name)

    async def _list_tools(self, spec: ServerSpec) -> list[DiscoveredTool]:
        await self._expire_stale_caches()
        cached = self._tool_cache.get(spec.name)
        now = asyncio.get_running_loop().time()
        if cached and now - cached[0] < TOOL_CACHE_TTL:
            return cached[1]

        async def walk(session: Any) -> list[Any]:
            """Every tool across pages, stopping at the last or a stuck cursor."""
            found: list[Any] = []
            cursor: str | None = None
            seen: set[str] = set()
            for _ in range(LIST_PAGES_MAX):
                if cursor is not None:
                    if cursor in seen:
                        log.warning(
                            "mcp server %r repeated the tools cursor %r; stopping",
                            spec.name, cursor,
                        )
                        return found
                    seen.add(cursor)
                response = await _list_page(session, cursor)
                found.extend(_page_tools(response))
                cursor = _next_cursor(response)
                if not cursor:
                    return found
            log.warning(
                "mcp server %r still paginating after %d page(s); using %d tool(s) so far",
                spec.name, LIST_PAGES_MAX, len(found),
            )
            return found

        # Discovery is idempotent: re-sending a tools/list can repeat nothing
        # the server did, so a dead transport earns its one re-send here.
        raw_tools = await self._reuse(spec, walk, repeatable=True)

        tools = []
        for tool in raw_tools:
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            required = schema.get("required") if isinstance(schema, dict) else None
            tools.append(
                DiscoveredTool(
                    name=getattr(tool, "name", ""),
                    description=str(getattr(tool, "description", "") or ""),
                    properties=properties if isinstance(properties, dict) else {},
                    required=[str(r) for r in required] if isinstance(required, list) else [],
                )
            )
        self._tool_cache[spec.name] = (now, tools)
        log.info("mcp server %r advertises %d tool(s): %s", spec.name, len(tools), [t.name for t in tools])
        return tools

    async def _pick(
        self, preferences: list[str], predicate
    ) -> tuple[ServerSpec, DiscoveredTool] | None:
        """Find the first (server, tool) pair matching the preference order."""
        for wanted in preferences:
            lowered = wanted.lower()
            for spec in self.servers():
                if spec.unresolved():
                    log.debug("skipping mcp server %r: unresolved variables", spec.name)
                    continue
                try:
                    tools = await self._list_tools(spec)
                except MCPTrustError as exc:
                    # Distinct from unreachable: the server may be perfectly
                    # healthy, the owner just has not approved the entry. The
                    # search still fails closed — this server is skipped.
                    log.info("mcp server %r awaits approval: %s", spec.name, exc)
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.info("mcp server %r is unreachable: %s", spec.name, exc)
                    continue
                for tool in tools:
                    if tool.name.lower() == lowered or predicate(tool.name.lower()):
                        return spec, tool
        return None

    async def _call(
        self,
        spec: ServerSpec,
        tool: DiscoveredTool,
        arguments: dict[str, Any],
        timeout: float,  # noqa: ASYNC109 - a configured budget, not a per-call API knob
    ) -> str:
        """Call *tool* and flatten the result to text."""
        result = await self._reuse(
            spec,
            lambda session: session.call_tool(
                tool.name, arguments, read_timeout_seconds=_read_timeout(timeout)
            ),
            repeatable=False,  # a call may already have run; never re-send it
        )
        return _flatten(result)

    # -- WebProvider ------------------------------------------------------- #

    async def search(
        self, query: str, max_results: int, days: int | None = None
    ) -> SearchResult:
        result = SearchResult(query=query, provider=self.name)
        chosen = await self._pick(
            self.config.search_tools, lambda name: "search" in name
        )
        if chosen is None:
            result.error = (
                f"no search tool found in {', '.join(self.sources)} "
                f"(looked for {', '.join(self.config.search_tools)} or any tool containing 'search')"
            )
            return result

        spec, tool = chosen
        key = tool.arg_for(("query", "q", "search", "question")) or "query"
        arguments: dict[str, Any] = {key: query}
        if "max_results" in tool.properties:
            arguments["max_results"] = max_results
        elif "limit" in tool.properties:
            arguments["limit"] = max_results
        elif "count" in tool.properties:
            arguments["count"] = max_results
        # Tavily-shaped servers understand a coarse bucket; anything that takes
        # an ISO date gets the exact cutoff instead. Servers that accept
        # neither simply ignore both keys.
        if days and days > 0:
            if "freshness" in tool.properties:
                arguments["freshness"] = _mcp_freshness(days, tool.properties["freshness"])
            elif "time_range" in tool.properties:
                arguments["time_range"] = _mcp_time_range(days)
            elif "start_published_date" in tool.properties:
                arguments["start_published_date"] = recent_cutoff(days)

        try:
            text = await self._call(spec, tool, arguments, float(self.config.timeout_seconds))
        except Exception as exc:  # noqa: BLE001
            log.warning("mcp search via %s.%s failed: %s", spec.name, tool.name, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        _parse_search_payload(text, result)
        log.info("mcp %s/%s returned %d hit(s)", spec.name, tool.name, len(result.hits))
        return result

    async def fetch(self, url: str) -> Page:
        chosen = await self._pick(
            self.config.fetch_tools, lambda name: any(k in name for k in ("scrape", "extract", "fetch"))
        )
        if chosen is None:
            return Page(
                url=url,
                markdown=f"mcp fetch failed: no fetch tool found in {', '.join(self.sources)}",
            )

        spec, tool = chosen
        key = tool.arg_for(("url", "urls", "link")) or "url"
        arguments: dict[str, Any] = {key: url}

        try:
            text = await self._call(spec, tool, arguments, float(self.config.timeout_seconds))
        except Exception as exc:  # noqa: BLE001
            return Page(url=url, markdown=f"mcp fetch failed: {type(exc).__name__}: {exc}")

        payload = _try_json(text)
        if isinstance(payload, dict):
            for item in payload.get("results") or []:
                item = item if isinstance(item, dict) else {}
                if item.get("raw_content") or item.get("markdown") or item.get("content"):
                    reported_url = final_url(item.get("url"), requested="")
                    if not reported_url:
                        return Page(
                            url=url,
                            markdown="mcp fetch failed: server returned content without a final URL",
                        )
                    return Page(
                        # A tavily-shaped server reports the page it landed on
                        # after its own redirects; that URL is what the tool
                        # layer re-checks against the private-host rules.
                        url=reported_url,
                        title=str(item.get("title") or ""),
                        markdown=str(
                            item.get("raw_content") or item.get("markdown") or item.get("content")
                        ),
                    )
            # Some servers answer with the page at the top level. The text
            # below is whatever the server flattened; the URL is upgraded only
            # when the payload names one.
            reported_url = final_url(payload.get("url"), payload.get("sourceURL"), requested="")
            if not reported_url:
                return Page(
                    url=url,
                    markdown="mcp fetch failed: server returned content without a final URL",
                )
            return Page(
                url=reported_url,
                markdown=text,
            )
        return Page(
            url=url,
            markdown="mcp fetch failed: server returned content without a final URL",
        )

    def summary_line(self) -> str:
        ok, reason = self.available()
        if not ok:
            return f"- `{self.name}` _(unavailable: {reason})_"
        names = [s.name for s in self.servers() if not s.unresolved()]
        sources = " + ".join(self.sources)
        return f"- `{self.name}` — {len(names)} server(s) from {sources}: {', '.join(names)}"


# --------------------------------------------------------------------------- #
# Result flattening
# --------------------------------------------------------------------------- #


def _mcp_time_range(days: int) -> str:
    """Coarse bucket for servers that mirror Tavily's ``time_range`` shape."""
    for ceiling, name in ((1, "day"), (7, "week"), (31, "month"), (365, "year")):
        if days <= ceiling:
            return name
    return "year"


def _mcp_freshness(days: int, schema: Any) -> str:
    """Format *days* for MCP tools that expose You.com's ``freshness`` field."""
    if isinstance(schema, dict):
        pattern = str(schema.get("pattern") or "").lower()
        description = str(schema.get("description") or "").lower()
        if "yyyy-mm-ddtoyyyy-mm-dd" in description or "date range" in description:
            return f"{recent_cutoff(days)}to{recent_cutoff(0)}"
        if "yyyy-mm-dd" in pattern and "to" in pattern:
            return f"{recent_cutoff(days)}to{recent_cutoff(0)}"
    return _mcp_time_range(days)


async def _list_page(session: Any, cursor: str | None) -> Any:
    """One ``tools/list`` page, on whichever calling convention the SDK has.

    mcp 2.0 replaced ``list_tools(cursor=…)`` with
    ``list_tools(params=PaginatedRequestParams(cursor=…))``; calling the old
    shape on a new SDK raises ``TypeError`` on the very first page, which
    killed discovery entirely. The installed session's own signature decides
    — and the test stubs, which declare ``cursor=None`` (or take ``**kwargs``),
    keep the 1.x shape.
    """
    if cursor is None:
        return await session.list_tools()
    if _takes_cursor_kwarg(session):
        return await session.list_tools(cursor=cursor)
    from mcp.types import PaginatedRequestParams

    return await session.list_tools(params=PaginatedRequestParams(cursor=cursor))


def _takes_cursor_kwarg(session: Any) -> bool:
    """Whether *session*'s ``list_tools`` takes 1.x's ``cursor=`` kwarg."""
    import inspect

    try:
        parameters = inspect.signature(session.list_tools).parameters
    except (TypeError, ValueError):  # pragma: no cover - an opaque callable
        return True
    return "cursor" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _page_tools(response: Any) -> list[Any]:
    """The tools on one ``ListToolsResult`` page, in object or dict shape."""
    tools = response.get("tools") if isinstance(response, dict) else getattr(response, "tools", None)
    return list(tools) if isinstance(tools, list) else []


def _next_cursor(response: Any) -> str | None:
    """The cursor for the next tool page, if the server named one.

    Both field spellings: 1.x called it ``nextCursor``, 2.x renamed it
    ``next_cursor`` — and a raw server dict can carry either. Reading only
    the camel spelling made a paginated server silently lose every tool
    after its first page.
    """
    if isinstance(response, dict):
        cursor = response.get("nextCursor") or response.get("next_cursor")
    else:
        cursor = getattr(response, "nextCursor", None) or getattr(
            response, "next_cursor", None
        )
    return str(cursor) if cursor else None


def _structured(result: Any) -> Any:
    """The ``structuredContent`` payload, if any, in object or dict shape.

    Both field spellings: 1.x called it ``structuredContent``, 2.x renamed
    it ``structured_content`` — missing the installed one would render a
    perfectly good structured-only result as nothing.
    """
    if isinstance(result, dict):
        value = result.get("structuredContent")
        return value if value is not None else result.get("structured_content")
    value = getattr(result, "structuredContent", None)
    return value if value is not None else getattr(result, "structured_content", None)


def _flatten(result: Any) -> str:
    """Turn a CallToolResult into plain text.

    MCP servers return structured content blocks, sometimes with JSON inside a
    text block, sometimes bare text. Handle both, and fall back to
    ``structuredContent`` when a tool answers with data and no text block at
    all — otherwise a perfectly good result would read as nothing.
    """
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    parts: list[str] = []
    if content is not None:
        for block in content if isinstance(content, list) else [content]:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text is None:
                # Resource or image blocks: describe rather than ignore.
                kind = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                parts.append(f"[{kind} content omitted]")
            else:
                parts.append(str(text))
    text = "\n".join(parts)
    if text.strip():
        return text

    structured = _structured(result)
    if structured is not None and structured != {} and structured != []:
        if isinstance(structured, str):
            return structured
        try:
            return json.dumps(structured, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(structured)

    if content is None:
        return "" if result is None else str(result)
    return text


def _try_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _highlights(item: dict[str, Any]) -> str:
    """You.com's page text: a list under ``contents.highlights``."""
    contents = item.get("contents")
    if not isinstance(contents, dict):
        return ""
    highlights = contents.get("highlights")
    if isinstance(highlights, list):
        return "\n".join(str(h) for h in highlights if h)
    return str(highlights or "")


def _parse_search_payload(text: str, result: SearchResult) -> None:
    """Best-effort normalisation of whatever shape the server returned."""
    payload = _try_json(text)

    if isinstance(payload, list):
        payload = {"results": payload}

    if isinstance(payload, dict):
        result.answer = str(payload.get("answer") or payload.get("summary") or "")
        items = payload.get("results") or payload.get("data") or []
        # Grouped by source rather than a flat list — You.com answers with
        # ``{"results": {"web": [...]}}`` — so flatten the groups first.
        if isinstance(items, dict):
            items = [
                item
                for group in items.values()
                if isinstance(group, list)
                for item in group
            ]
        elif not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                result.hits.append(SearchHit(title="", url="", content=str(item)))
                continue
            url = str(item.get("url") or item.get("link") or "")
            if not url:
                continue
            # Tavily puts the snippet in "content" and the full page in
            # "raw_content"; firecrawl puts markdown in "markdown"; You.com
            # nests highlights under "contents". Accept all.
            body = str(
                item.get("raw_content")
                or item.get("markdown")
                or item.get("content")
                or _highlights(item)
            )
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or item.get("description") or item.get("snippet") or ""),
                    content=body,
                    score=item.get("score"),
                    published=str(
                        item.get("published_date")
                        or item.get("publishedDate")
                        or item.get("page_age")
                        or ""
                    ),
                )
            )
        if result.hits or result.answer:
            return

    if text.strip():
        # A plain-text answer is still useful; surface it as one synthetic hit.
        result.hits.append(SearchHit(title=f"result for {result.query!r}", url="", content=text))


__all__ = ["MCPProvider", "ServerSpec", "DiscoveredTool", "MCPTrustError", "expand_env"]
