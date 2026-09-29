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

Use `search` for anything that depends on current facts, releases, prices, news,
or documentation you have not seen. Use `fetch` when you already have a URL and
want what is actually on the page.

Results come with numbered links, so you can cite them as [1], [2] and tell the
owner where a claim came from. Quote from the results rather than from memory
when the topic is recent or unfamiliar.
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

    async def _search(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("query is required for a search")
        max_results = self._max_results(arguments)

        attempts: list[str] = []
        for name, provider in self.providers.items():
            ok, reason = provider.available()
            if not ok:
                attempts.append(f"{name}: {reason}")
                continue
            try:
                result = await asyncio.wait_for(
                    provider.search(query, max_results), timeout=self.settings.timeout_seconds
                )
            except TimeoutError:
                attempts.append(f"{name}: timed out after {self.settings.timeout_seconds}s")
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"{name}: {type(exc).__name__}: {exc}")
                continue

            if result.ok and result.hits:
                return self._render_search(result, attempts)
            attempts.append(f"{name}: {result.error or 'no results'}")

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
            data={"query": result.query, "provider": result.provider, "hits": len(result.hits)},
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
