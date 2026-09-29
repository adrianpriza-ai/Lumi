"""Firecrawl, through the official v4 Python SDK.

Firecrawl is the better of the two for *reading* a specific page: it renders
JavaScript and returns clean markdown. Its search endpoint returns titles and
descriptions, and body text only when ``scrape_options`` asks for it, so this
provider tops up the top hits with a scrape when the snippets come back empty.

Several keys are supported. A comma-separated list is a rotation pool — the
same shape ``OPENAI_API_KEY`` already uses — and a failed key is parked for a
minute before it is retried, so a 429 that clears itself recovers without a
restart. Firecrawl does not offer a keyless tier; an unset env var makes the
provider unavailable, which the web tool surfaces honestly to the model.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ....config import FirecrawlConfig
from ....llm.keypool import KeyPool, mask
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider, recent_cutoff

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
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())
        #: one ``AsyncFirecrawl`` client per key, so a parked key keeps its own session
        self._clients: dict[str, Any] = {}

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

    def _get_client(self, key: str) -> Any:
        """Build (and cache) an ``AsyncFirecrawl`` for *key*."""
        client = self._clients.get(key)
        if client is None:
            from firecrawl import AsyncFirecrawl

            client = AsyncFirecrawl(api_key=key, api_url=self.config.base_url)
            self._clients[key] = client
            log.debug("firecrawl client created for %s", mask(key))
        return client

    # -- public surface ---------------------------------------------------- #

    async def search(
        self, query: str, max_results: int, days: int | None = None
    ) -> SearchResult:
        """Search via Firecrawl, retrying the pool on failure.

        Unlike Tavily there is no keyless tier: the ``available()`` check
        guarantees at least one key exists. Still defensive about it — an
        empty pool is a single-attempt call rather than a None-crash.
        """
        result = SearchResult(query=query, provider=self.name)
        if not self._pool:
            result.error = f"{self.config.api_key_env} is not set"
            return result

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            client = self._get_client(key)
            try:
                raw = await client.search(
                    query,
                    limit=max_results,
                    tbs=_tbs(days),
                    scrape_options={"formats": ["markdown"]},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("firecrawl search failed on %s: %s", mask(key), exc)
                self._pool.report(key, ok=False, retryable=True)
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            self._pool.report(key, ok=True)
            self._populate_search(raw, result)
            await self._top_up(result)
            log.info("firecrawl returned %d result(s) for %r", len(result.hits), query)
            return result

        result.error = last_error or "every firecrawl key failed"
        return result

    async def scrape_markdown(self, url: str) -> str:
        try:
            doc = await self._scrape_client(url).scrape(
                url, formats=["markdown"], only_main_content=self.config.only_main_content
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("firecrawl scrape failed for %s: %s", url, exc)
            return ""
        return str(_as_dict(doc).get("markdown") or "")

    async def fetch(self, url: str) -> Page:
        """Scrape one page as markdown, retrying the pool on failure."""
        if not self._pool:
            return Page(url=url, markdown=f"{self.config.api_key_env} is not set")

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            client = self._get_client(key)
            try:
                doc = await client.scrape(
                    url, formats=["markdown"], only_main_content=self.config.only_main_content
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("firecrawl scrape failed on %s: %s", mask(key), exc)
                self._pool.report(key, ok=False, retryable=True)
                last_error = f"scrape failed: {type(exc).__name__}: {exc}"
                continue

            self._pool.report(key, ok=True)
            data = _as_dict(doc)
            metadata = _as_dict(data.get("metadata"))
            return Page(
                url=url,
                title=str(metadata.get("title") or metadata.get("ogTitle") or ""),
                markdown=str(data.get("markdown") or "(no content)"),
            )

        return Page(url=url, markdown=last_error or "every firecrawl key failed")

    # -- per-key helpers --------------------------------------------------- #

    def _scrape_client(self, url: str) -> Any:
        """Pick any healthy client for the top-up scrape (no rotation)."""
        # ``_top_up`` calls ``scrape_markdown`` once per thin hit; cycling the
        # whole pool for each scrape would be wasteful, so we pick a single
        # client. A failure here only costs a single missing body.
        if self._pool:
            key = self._pool.pick()
            if key is not None:
                return self._get_client(key)
        # Fall back to the primary key (or whatever the SDK accepts).
        return self._get_client(self.config.api_key() or "")

    def _populate_search(self, raw: Any, result: SearchResult) -> None:
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
                    published=str(item.get("publishedDate") or ""),
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
                        published=str(item.get("publishedDate") or ""),
                    )
                )

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


def _tbs(days: int | None) -> str | None:
    """Google ``tbs`` time filter for a day window; ``None`` means unfiltered."""
    if not days or days <= 0:
        return None
    if days <= 1:
        return "qdr:d"
    if days <= 7:
        return "qdr:w"
    if days <= 31:
        return "qdr:m"
    if days <= 365:
        return "qdr:y"
    # No coarser bucket exists; the explicit ISO cutoff still ranks freshness.
    return f"cdr:1,cd_min:{recent_cutoff(days)}"


__all__ = ["FirecrawlProvider"]