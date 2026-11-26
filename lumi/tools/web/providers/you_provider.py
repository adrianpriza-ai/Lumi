"""You.com search, through its hosted MCP endpoint.

``https://api.you.com/mcp?profile=free`` is You.com's free, keyless MCP
profile — no account, no API key, a low rate limit. Because it speaks MCP,
this provider points the shared MCP client (:class:`~lumi.tools.mcp.servers.MCPServers`)
at one fixed server: same session reuse, same tool discovery, same transports,
no config file to write. ``YOU_API_KEY``, when set, rides along as an
``X-API-Key`` header.

So a fresh clone has a working search without editing anything. Servers in
``.mcp.json`` are *not* search backends any more — they are reached through the
``mcp`` tool's Code Mode — but You.com keeps its own provider because the
endpoint is built in, not configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....config import MCPConfig, YouConfig
from ...mcp.servers import MCPServers, ServerSpec, _try_json
from .base import Page, SearchHit, SearchResult, WebProvider, final_url, recent_cutoff

#: Tool-name candidates, in order, for the endpoint's search tool. The
#: generic ``arg_for`` below still reads the tool's own schema for the query
#: parameter, so a renamed server keeps working.
SEARCH_TOOLS: tuple[str, ...] = ("search", "you_search")

#: Which tool counts as a page read when the endpoint offers one.
FETCH_HINTS: tuple[str, ...] = ("scrape", "extract", "fetch")


class YouProvider(WebProvider):
    """The MCP stack with the config file swapped for one built-in server."""

    name = "you"

    def __init__(self, config: YouConfig, project_root: Path) -> None:
        # Empty config paths: no file is read, ``servers()`` below is the only
        # source. ``enabled`` still follows YouConfig.
        self.client = MCPServers(
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

    def available(self) -> tuple[bool, str]:
        """Check the built-in endpoint directly: there is no config file to read.

        Delegating to ``MCPServers.available()`` used to report "no servers in
        the configured files" — the client was built with empty config paths on
        purpose, so the provider's own ``servers()`` is the only source.
        """
        if not self.you.enabled:
            return False, "disabled in config"
        if not self.you.url:
            return False, "no url configured"
        try:
            import mcp  # noqa: F401
        except ImportError:
            return False, "the mcp package is not installed"
        return True, ""

    async def aclose(self) -> None:
        await self.client.aclose()

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        result = SearchResult(query=query, provider=self.name)
        tool = await self._pick_tool(SEARCH_TOOLS, lambda name: "search" in name)
        if tool is None:
            result.error = "you.com advertises no search tool"
            return result

        key = tool.arg_for(("query", "q", "search", "question")) or "query"
        arguments: dict[str, Any] = {key: query}
        if "max_results" in tool.properties:
            arguments["max_results"] = max_results
        elif "limit" in tool.properties:
            arguments["limit"] = max_results
        elif "count" in tool.properties:
            arguments["count"] = max_results
        if days and days > 0:
            if "freshness" in tool.properties:
                arguments["freshness"] = _freshness(days, tool.properties["freshness"])
            elif "time_range" in tool.properties:
                arguments["time_range"] = _time_range(days)
            elif "start_published_date" in tool.properties:
                arguments["start_published_date"] = recent_cutoff(days)

        try:
            text = await self.client.call_tool(self._spec, tool, arguments)
        except Exception as exc:  # noqa: BLE001 - the tool layer reports the failure
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        _parse_search_payload(text, result)
        return result

    async def fetch(self, url: str) -> Page:
        tool = await self._pick_tool((), lambda name: any(k in name for k in FETCH_HINTS))
        if tool is None:
            return Page(url=url, markdown="you.com advertises no fetch tool")

        key = tool.arg_for(("url", "urls", "link")) or "url"
        try:
            text = await self.client.call_tool(self._spec, tool, {key: url})
        except Exception as exc:  # noqa: BLE001
            return Page(url=url, markdown=f"you fetch failed: {type(exc).__name__}: {exc}")

        payload = _try_json(text)
        if isinstance(payload, dict):
            for item in payload.get("results") or []:
                item = item if isinstance(item, dict) else {}
                body = item.get("raw_content") or item.get("markdown") or item.get("content")
                if body:
                    reported_url = final_url(item.get("url"), requested="")
                    if not reported_url:
                        return Page(
                            url=url,
                            markdown="you fetch failed: server returned content without a final URL",
                        )
                    return Page(
                        url=reported_url,
                        title=str(item.get("title") or ""),
                        markdown=str(body),
                    )
            reported_url = final_url(payload.get("url"), payload.get("sourceURL"), requested="")
            if not reported_url:
                return Page(
                    url=url,
                    markdown="you fetch failed: server returned content without a final URL",
                )
            return Page(url=reported_url, markdown=text)
        return Page(
            url=url,
            markdown="you fetch failed: server returned content without a final URL",
        )

    async def _pick_tool(self, names: tuple[str, ...], predicate):
        """The endpoint's first tool matching *names*, else *predicate*."""
        try:
            tools = await self.client._list_tools(self._spec)
        except Exception:  # noqa: BLE001 - a dead endpoint is reported by the caller
            return None
        lowered = {name.lower() for name in names}
        for tool in tools:
            if tool.name.lower() in lowered:
                return tool
        for tool in tools:
            if predicate(tool.name.lower()):
                return tool
        return None


# --------------------------------------------------------------------------- #
# Payload normalisation
# --------------------------------------------------------------------------- #


def _time_range(days: int) -> str:
    """Coarse bucket for the endpoint's ``time_range`` field."""
    for ceiling, name in ((1, "day"), (7, "week"), (31, "month"), (365, "year")):
        if days <= ceiling:
            return name
    return "year"


def _freshness(days: int, schema: Any) -> str:
    """Format *days* for the endpoint's ``freshness`` field, when it has one."""
    if isinstance(schema, dict):
        pattern = str(schema.get("pattern") or "").lower()
        description = str(schema.get("description") or "").lower()
        if "date range" in description or "yyyy-mm-ddtoyyyy-mm-dd" in description:
            return f"{recent_cutoff(days)}to{recent_cutoff(0)}"
        if "yyyy-mm-dd" in pattern and "to" in pattern:
            return f"{recent_cutoff(days)}to{recent_cutoff(0)}"
    return _time_range(days)


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
    """Best-effort normalisation of whatever shape the endpoint returned."""
    payload = _try_json(text)

    if isinstance(payload, list):
        payload = {"results": payload}

    if isinstance(payload, dict):
        result.answer = str(payload.get("answer") or payload.get("summary") or "")
        items = payload.get("results") or payload.get("data") or []
        # Grouped by source rather than a flat list — You.com answers with
        # ``{"results": {"web": [...]}}`` — so flatten the groups first.
        if isinstance(items, dict):
            items = [item for group in items.values() if isinstance(group, list) for item in group]
        elif not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                result.hits.append(SearchHit(title="", url="", content=str(item)))
                continue
            url = str(item.get("url") or item.get("link") or "")
            if not url:
                continue
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
                    snippet=str(
                        item.get("content") or item.get("description") or item.get("snippet") or ""
                    ),
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


__all__ = ["YouProvider"]
