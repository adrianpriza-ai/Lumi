"""Firecrawl, through the official v4 Python SDK.

Firecrawl is the better of the two for *reading* a specific page: it renders
JavaScript and returns clean markdown. Its search endpoint returns titles and
descriptions, and body text only when ``scrape_options`` asks for it, so this
provider tops up the top hits with a scrape when the snippets come back empty.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ....config import FirecrawlConfig
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider

log = get_logger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    """Pydantic model, dataclass, or plain dict -> dict."""
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:  # pragma: no cover - defensive
                continue
    return {k: v for k, v in vars(value).items() if not k.startswith("_")} if hasattr(value, "__dict__") else {}


class FirecrawlProvider(WebProvider):
    name = "firecrawl"

    def __init__(self, config: FirecrawlConfig) -> None:
        self.config = config
        self._client: Any = None

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        if not self.config.api_key():
            return False, f"{self.config.api_key_env} is not set"
        try:
            import firecrawl  # noqa: F401
        except ImportError:
            return False, "firecrawl is not installed"
        return True, ""

    def _get_client(self) -> Any:
        if self._client is None:
            from firecrawl import AsyncFirecrawl

            self._client = AsyncFirecrawl(
                api_key=self.config.api_key(), api_url=self.config.base_url
            )
            log.debug("firecrawl client created")
        return self._client

    async def search(self, query: str, max_results: int) -> SearchResult:
        result = SearchResult(query=query, provider=self.name)
        try:
            raw = await self._get_client().search(
                query,
                limit=max_results,
                scrape_options={"formats": ["markdown"]},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("firecrawl search failed: %s", exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        data = _as_dict(raw)
        web = data.get("web") or []
        for item in web:
            item = _as_dict(item)
            url = str(item.get("url") or "")
            if not url:
                continue
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("description") or ""),
                    content=str(item.get("markdown") or item.get("content") or ""),
                )
            )
        news = data.get("news") or []
        for item in news:
            item = _as_dict(item)
            url = str(item.get("url") or "")
            if url and not any(h.url == url for h in result.hits):
                result.hits.append(
                    SearchHit(
                        title=str(item.get("title") or ""),
                        url=url,
                        snippet=str(item.get("description") or ""),
                    )
                )

        await self._top_up(result)
        log.info("firecrawl returned %d result(s) for %r", len(result.hits), query)
        return result

    async def _top_up(self, result: SearchResult) -> None:
        """Fill in missing bodies for the top hits with a real scrape."""
        thin = [hit for hit in result.hits if not hit.content.strip()][: self.config.auto_scrape_top_n]
        if not thin:
            return
        log.debug("scraping %d firecrawl hit(s) for body text", len(thin))
        pages = await asyncio.gather(
            *(self.scrape_markdown(hit.url) for hit in thin), return_exceptions=True
        )
        for hit, page in zip(thin, pages, strict=False):
            if isinstance(page, str) and page.strip():
                hit.content = page
                hit.title = hit.title or page[:80]

    async def scrape_markdown(self, url: str) -> str:
        try:
            doc = await self._get_client().scrape(
                url, formats=["markdown"], only_main_content=self.config.only_main_content
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("firecrawl scrape failed for %s: %s", url, exc)
            return ""
        return str(_as_dict(doc).get("markdown") or "")

    async def fetch(self, url: str) -> Page:
        try:
            doc = await self._get_client().scrape(
                url, formats=["markdown"], only_main_content=self.config.only_main_content
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("firecrawl scrape failed for %s: %s", url, exc)
            return Page(url=url, markdown=f"scrape failed: {type(exc).__name__}: {exc}")

        data = _as_dict(doc)
        metadata = _as_dict(data.get("metadata"))
        return Page(
            url=url,
            title=str(metadata.get("title") or metadata.get("ogTitle") or ""),
            markdown=str(data.get("markdown") or "(no content)"),
        )


__all__ = ["FirecrawlProvider"]
