"""Any MCP server as a web provider.

This is what makes ``.mcp.json`` meaningful for Lumi. The file is read from the
project root, ``${VAR}`` placeholders are expanded from the environment, and the
server is spoken to over MCP proper — so a Tavily or Firecrawl MCP server
dropped into that file works immediately, with no code change. A locally spawned
server (``npx -y some-mcp-server``) works the same way.

Two transport kinds are supported:

* ``"type": "remote"`` — streamable HTTP, the shape opencode and Claude Code use
  for hosted servers like ``https://mcp.tavily.com/mcp/``.
* ``"type": "local"``  — a subprocess over stdio, spawned with ``command``/``args``.

Tool names are not hardcoded. The provider asks each server what it offers and
picks the first tool matching the configured preference list, then reads that
tool's own input schema to work out whether it wants ``query``, ``q`` or
something else. That is what lets one implementation front an arbitrary server.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ....config import MCPConfig
from ....paths import resolve
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider, recent_cutoff

log = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: How long a discovered tool list stays valid before we re-ask the server.
TOOL_CACHE_TTL = 300.0


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` from the process environment."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or (default or "")

    return _ENV_PATTERN.sub(replace, value)


@dataclass(slots=True)
class ServerSpec:
    """One entry from ``.mcp.json``."""

    name: str
    kind: str = "remote"  # "remote" | "local"
    url: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    #: The unexpanded strings, kept so we can still tell "unset variable" apart
    #: from "variable set to the empty string" after expansion has run.
    raw: list[str] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.kind == "local" or bool(self.command)

    def unresolved(self) -> str | None:
        """Return the first variable this server needs but the environment lacks."""
        for candidate in self.raw or [self.url, *self.args, *self.env.values(), *self.headers.values()]:
            match = _ENV_PATTERN.search(str(candidate))
            if match and not os.environ.get(match.group(1)):
                return match.group(1)
        return None

    @classmethod
    def from_json(cls, name: str, raw: dict[str, Any]) -> ServerSpec:
        command = raw.get("command")
        if isinstance(command, list):  # opencode style: ["npx", "-y", "pkg"]
            command_parts = [str(part) for part in command]
        elif command:
            command_parts = [str(command)]
        else:
            command_parts = []
        args = [str(a) for a in raw.get("args", [])]
        raw_url = str(raw.get("url") or "")
        raw_env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        raw_headers = {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}

        return cls(
            name=name,
            kind=str(raw.get("type") or ("local" if command_parts else "remote")),
            url=expand_env(raw_url),
            command=command_parts[0] if command_parts else "",
            args=[expand_env(a) for a in command_parts[1:] + args],
            env={k: expand_env(v) for k, v in raw_env.items()},
            headers={k: expand_env(v) for k, v in raw_headers.items()},
            raw=[raw_url, *command_parts, *args, *raw_env.values(), *raw_headers.values()],
        )


@dataclass(slots=True)
class DiscoveredTool:
    name: str
    description: str
    properties: dict[str, Any]

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
        self._tool_cache: dict[str, tuple[float, list[DiscoveredTool]]] = {}
        #: Live session per server name: (stack owning the transport, session).
        #: Discovery plus a call used to pay for two spawns — and a local
        #: ``npx`` server spends most of that booting.
        self._sessions: dict[str, tuple[AsyncExitStack, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- configuration ----------------------------------------------------- #

    @property
    def config_path(self) -> Path:
        return resolve(self.config.config_file, base=self.root)

    def servers(self) -> list[ServerSpec]:
        path = self.config_path
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", path, exc)
            return []
        block = raw.get("mcpServers") or raw.get("servers") or {}
        if not isinstance(block, dict):
            log.warning("%s has no usable mcpServers object", path)
            return []
        return [ServerSpec.from_json(name, cfg) for name, cfg in block.items() if isinstance(cfg, dict)]

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        try:
            import mcp  # noqa: F401
        except ImportError:
            return False, "the mcp package is not installed"
        servers = self.servers()
        if not servers:
            return False, f"no servers in {self.config_path}"
        missing = [f"{s.name} (needs {s.unresolved()})" for s in servers if s.unresolved()]
        if missing and len(missing) == len(servers):
            return False, f"unset variables in {self.config_path}: {', '.join(missing)}"
        return True, ""

    # -- session plumbing -------------------------------------------------- #

    async def _open(self, stack: AsyncExitStack, spec: ServerSpec):
        """Start a session against *spec* and return it, already initialised."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if spec.is_local:
            if not spec.command:
                raise RuntimeError(f"server {spec.name} has no command")
            params = StdioServerParameters(
                command=spec.command, args=spec.args, env={**os.environ, **spec.env}
            )
            streams = await stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

            if not spec.url:
                raise RuntimeError(f"server {spec.name} has no url")
            http_client = None
            if spec.headers:
                http_client = create_mcp_http_client(headers=spec.headers)
            transport = await stack.enter_async_context(
                streamable_http_client(spec.url, http_client=http_client)
            )
            # stdio yields 3 streams, streamable HTTP yields 2; only the first two
            # are the session's read/write channels.
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
        """Close and forget the cached session for *name*, if there is one."""
        entry = self._sessions.pop(name, None)
        if entry is None:
            return
        stack, _session = entry
        with suppress(Exception):  # shutdown is best-effort
            await stack.aclose()

    async def _reuse(self, spec: ServerSpec, op: Any) -> Any:
        """Run *op(session)* on the cached session, re-spawning once if it died.

        A server that went away between queries must not poison every later
        call: the dead session is dropped and the operation retried on a fresh
        one, so the worst a dead server costs is one extra launch.
        """
        try:
            return await op(await self._session_for(spec))
        except Exception:
            log.debug("mcp session for %r failed; re-spawning once", spec.name)
            await self._drop_session(spec.name)
            try:
                return await op(await self._session_for(spec))
            except Exception:
                await self._drop_session(spec.name)
                raise

    async def aclose(self) -> None:
        """Close every cached session. Called at shutdown via the web tool."""
        for name in list(self._sessions):
            await self._drop_session(name)

    async def _list_tools(self, spec: ServerSpec) -> list[DiscoveredTool]:
        cached = self._tool_cache.get(spec.name)
        now = asyncio.get_running_loop().time()
        if cached and now - cached[0] < TOOL_CACHE_TTL:
            return cached[1]

        response = await self._reuse(spec, lambda session: session.list_tools())

        tools = []
        for tool in getattr(response, "tools", []) or []:
            schema = getattr(tool, "inputSchema", None) or {}
            properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
            tools.append(
                DiscoveredTool(
                    name=getattr(tool, "name", ""),
                    description=str(getattr(tool, "description", "") or ""),
                    properties=properties if isinstance(properties, dict) else {},
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
                tool.name, arguments, read_timeout_seconds=timeout
            ),
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
                f"no search tool found in {self.config_path} "
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
        # Tavily-shaped servers understand a coarse bucket; anything that takes
        # an ISO date gets the exact cutoff instead. Servers that accept
        # neither simply ignore both keys.
        if days and days > 0:
            if "time_range" in tool.properties:
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
            return Page(url=url, markdown=f"no fetch tool found in {self.config_path}")

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
                    return Page(
                        url=url,
                        title=str(item.get("title") or ""),
                        markdown=str(
                            item.get("raw_content") or item.get("markdown") or item.get("content")
                        ),
                    )
        return Page(url=url, markdown=text)

    def summary_line(self) -> str:
        ok, reason = self.available()
        if not ok:
            return f"- `{self.name}` _(unavailable: {reason})_"
        names = [s.name for s in self.servers() if not s.unresolved()]
        return f"- `{self.name}` — {len(names)} server(s) from {self.config_path.name}: {', '.join(names)}"


# --------------------------------------------------------------------------- #
# Result flattening
# --------------------------------------------------------------------------- #


def _mcp_time_range(days: int) -> str:
    """Coarse bucket for servers that mirror Tavily's ``time_range`` shape."""
    for ceiling, name in ((1, "day"), (7, "week"), (31, "month"), (365, "year")):
        if days <= ceiling:
            return name
    return "year"


def _flatten(result: Any) -> str:
    """Turn a CallToolResult into plain text.

    MCP servers return structured content blocks, sometimes with JSON inside a
    text block, sometimes bare text. Handle both.
    """
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if content is None:
        return "" if result is None else str(result)

    parts: list[str] = []
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
    return "\n".join(parts)


def _try_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _parse_search_payload(text: str, result: SearchResult) -> None:
    """Best-effort normalisation of whatever shape the server returned."""
    payload = _try_json(text)

    if isinstance(payload, list):
        payload = {"results": payload}

    if isinstance(payload, dict):
        result.answer = str(payload.get("answer") or payload.get("summary") or "")
        for item in payload.get("results") or payload.get("data") or []:
            if not isinstance(item, dict):
                result.hits.append(SearchHit(title="", url="", content=str(item)))
                continue
            url = str(item.get("url") or item.get("link") or "")
            if not url:
                continue
            # Tavily puts the snippet in "content" and the full page in
            # "raw_content"; firecrawl puts markdown in "markdown". Accept all.
            body = str(item.get("raw_content") or item.get("markdown") or item.get("content") or "")
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or item.get("description") or item.get("snippet") or ""),
                    content=body,
                    score=item.get("score"),
                    published=str(item.get("published_date") or item.get("publishedDate") or ""),
                )
            )
        if result.hits or result.answer:
            return

    if text.strip():
        # A plain-text answer is still useful; surface it as one synthetic hit.
        result.hits.append(SearchHit(title=f"result for {result.query!r}", url="", content=text))


__all__ = ["MCPProvider", "ServerSpec", "DiscoveredTool", "expand_env"]
