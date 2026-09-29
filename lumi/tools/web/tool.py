"""The ``web`` tool: search and fetch, across whichever provider works.

The provider order comes from ``tools.web.provider_order``. Each provider is
asked whether it is available, and the first one that answers runs. A provider
that fails at call time falls through to the next, so a dead API key or an
unreachable MCP server degrades the bot instead of breaking it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...config import Config
from ...util.log import get_logger
from ...util.text import truncate
from ..base import Tool, ToolContext, ToolError, ToolResult
from .providers.base import Page, SearchResult, WebProvider
from .providers.exa_provider import ExaProvider
from .providers.firecrawl_provider import FirecrawlProvider
from .providers.mcp_provider import MCPProvider
from .providers.tavily_provider import TavilyProvider

log = get_logger(__name__)


def build_providers(config: Config) -> dict[str, WebProvider]:
    """Instantiate every provider named in the configured order."""
    web = config.tools.web
    built: dict[str, WebProvider] = {}
    for name in web.provider_order:
        try:
            if name == "tavily":
                built[name] = TavilyProvider(web.tavily)
            elif name == "firecrawl":
                built[name] = FirecrawlProvider(web.firecrawl)
            elif name == "exa":
                built[name] = ExaProvider(web.exa)
            elif name == "mcp":
                built[name] = MCPProvider(web.mcp, config.root)
            else:  # pragma: no cover - validate() catches this at startup
                log.warning("unknown web provider %r, skipping", name)
        except Exception as exc:  # noqa: BLE001 - a broken provider must not break startup
            log.warning("could not build web provider %r: %s", name, exc)
    return built


class WebTool(Tool):
    name = "web"
    description = """
Search the web, and read a specific page as clean markdown.

Use `search` for anything that depends on current facts — releases, prices,
news, changelogs, documentation you have not seen. Put today's date or year
into the `query` itself (e.g. "best graphics card September 2026"): search
engines rank pages mentioning the current period above old ones, and results
state the date they were retrieved on so stale hits are visible. For anything
time-sensitive also set `recency` to limit results to pages published in the
last N days.

Search more than once. One query rarely covers a question: run a second search
with different keywords or a different angle, and use `fetch` on the most
promising URL when snippets are thin. Results come with numbered links, so you
can cite them as [1], [2] and tell the owner where a claim came from. Quote
from the results rather than from memory when the topic is recent or
unfamiliar.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "fetch"],
                "description": "search the web, or fetch one URL as markdown.",
            },
            "query": {
                "type": "string",
                "description": "Search terms. Required for search. Be specific; natural language works well.",
            },
            "url": {
                "type": "string",
                "description": "The page to read. Required for fetch. Must be http or https.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return (default from config).",
                "minimum": 1,
                "maximum": 20,
            },
            "min_results": {
                "type": "integer",
                "description": (
                    "Fewer hits than this tops the result up from the next "
                    "provider before answering (default from config)."
                ),
                "minimum": 1,
                "maximum": 20,
            },
            "recency": {
                "type": "integer",
                "description": (
                    "Only pages published in the last N days. Use for "
                    "time-sensitive questions: 1 for today's news, 7 for this "
                    "week, 30 for this month, 365 for the last year. Omit for "
                    "timeless topics (docs, definitions, history)."
                ),
                "minimum": 1,
                "maximum": 3650,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.settings = config.tools.web
        self.providers = build_providers(config)

    def available(self) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "disabled in config (tools.web.enabled = false)"
        usable = [name for name, p in self.providers.items() if p.available()[0]]
        if not usable:
            reasons = "; ".join(f"{n}: {p.available()[1]}" for n, p in self.providers.items())
            return False, f"no provider is usable ({reasons})"
        return True, ""

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        if action == "search":
            return await self._search(arguments)
        if action == "fetch":
            return await self._fetch(arguments)
        raise ToolError(f"unknown action {action!r}; use search or fetch")

    def _max_results(self, arguments: dict[str, Any]) -> int:
        try:
            requested = int(arguments.get("max_results") or self.settings.max_results)
        except (TypeError, ValueError):
            requested = self.settings.max_results
        return max(1, min(requested, 20))

    def _recency_days(self, arguments: dict[str, Any]) -> int | None:
        """The ``recency`` argument as a day count; anything odd means unfiltered."""
        try:
            days = int(arguments.get("recency"))
        except (TypeError, ValueError):
            return None
        return max(1, min(days, 3650))

    def _min_results(self, arguments: dict[str, Any]) -> int:
        """The acceptable-result floor: the model's ask, else the config value."""
        try:
            requested = int(arguments.get("min_results"))
        except (TypeError, ValueError):
            requested = self.settings.min_results
        return max(1, min(requested, 20))

    async def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("query is required for a search")
        max_results = self._max_results(arguments)
        min_results = self._min_results(arguments)
        days = self._recency_days(arguments)

        attempts: list[str] = []
        merged: SearchResult | None = None
        for name, provider in self.providers.items():
            # A result below the floor is a thin answer, not a good one: the
            # next provider tops it up (deduped by URL) before it goes back.
            if merged is not None and len(merged.hits) >= min_results:
                break
            ok, reason = provider.available()
            if not ok:
                attempts.append(f"{name}: {reason}")
                continue
            try:
                result = await asyncio.wait_for(
                    provider.search(query, max_results, days),
                    timeout=self.settings.timeout_seconds,
                )
            except TimeoutError:
                attempts.append(f"{name}: timed out after {self.settings.timeout_seconds}s")
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if not result.ok or not result.hits:
                attempts.append(f"{name}: {result.error or 'no results'}")
                continue

            result.recency_days = days
            if merged is None:
                merged = result
            else:
                seen = {hit.url for hit in merged.hits}
                merged.hits.extend(h for h in result.hits if h.url not in seen)
                merged.answer = merged.answer or result.answer
                merged.provider = f"{merged.provider}+{name}"

        if merged is not None:
            return self._render_search(merged, attempts)

        return ToolResult.failure(
            "Every configured web provider failed. Attempts:\n"
            + "\n".join(f"- {line}" for line in attempts)
            + "\n\nAnswer from what you already know, and say plainly that you could not "
            "verify it online."
        )

    def _render_search(self, result: SearchResult, attempts: list[str]) -> ToolResult:
        body = result.render(self.settings.max_content_chars)
        if attempts:
            # Only useful when a provider was skipped, not on every happy path.
            body += "\n\n(skipped: " + "; ".join(attempts) + ")"
        return ToolResult(
            text=body,
            data={
                "query": result.query,
                "provider": result.provider,
                "hits": len(result.hits),
                "recency_days": result.recency_days,
            },
            summary=f"{result.provider}: {len(result.hits)} hit(s) for {result.query!r}",
        )

    async def _fetch(self, arguments: dict[str, Any]) -> ToolResult:
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise ToolError("url is required for a fetch")
        if not url.startswith(("http://", "https://")):
            raise ToolError("url must start with http:// or https://")

        attempts: list[str] = []
        for name, provider in self.providers.items():
            ok, reason = provider.available()
            if not ok:
                attempts.append(f"{name}: {reason}")
                continue
            try:
                page = await asyncio.wait_for(
                    provider.fetch(url), timeout=self.settings.timeout_seconds
                )
            except TimeoutError:
                attempts.append(f"{name}: timed out")
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if page.markdown.strip() and not page.markdown.startswith(("extract failed", "scrape failed", "mcp fetch failed")):
                return ToolResult(
                    text=page.render(self.settings.max_content_chars),
                    data={"url": url, "provider": name, "chars": len(page.markdown)},
                    summary=f"{name}: read {truncate(url, 60, '…')} ({len(page.markdown)} chars)",
                )
            attempts.append(f"{name}: {truncate(page.markdown, 120, '…')}")

        return ToolResult.failure(
            f"Could not read {url}. Attempts:\n"
            + "\n".join(f"- {line}" for line in attempts)
        )

    def summary_line(self) -> str:
        usable = [name for name, p in self.providers.items() if p.available()[0]]
        return f"`{self.name}` — search and page reads via {', '.join(usable) or 'no provider'}"


__all__ = ["WebTool", "build_providers", "Page", "SearchResult"]
