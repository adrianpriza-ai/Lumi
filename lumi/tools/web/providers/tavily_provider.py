"""Tavily, through the official Python SDK.

Tavily is the best fit for agentic search: it returns cleaned, LLM-ready
snippets rather than raw HTML, and can bundle an answer with the results.

Keyless mode is supported — if ``TAVILY_API_KEY`` is unset, the SDK runs in
its free tier (low rate limit; ``search`` and ``extract`` only). The provider
is therefore *available* without a key, so the web tool can fall through to
Tavily on a fresh clone and try something before giving up.

Several keys are supported. A comma-separated list is a rotation pool — the
same shape ``OPENAI_API_KEY`` already uses — and a failed key is parked for a
minute before it is retried, so a 429 that clears itself recovers without a
restart. The rotation lives in :class:`lumi.llm.keypool.KeyPool`; this module
just describes which keys are ``available`` and which client to build per key.
"""

from __future__ import annotations

from typing import Any

from ....config import TavilyConfig
from ....llm.keypool import KeyPool, mask
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider

log = get_logger(__name__)

#: Day-window ceilings mapped to Tavily's coarse ``time_range`` buckets.
#: A bucket must fully contain the window, so "last 3 days" does not silently
#: widen to a month.
TAVILY_TIME_RANGES: tuple[tuple[int, str], ...] = (
    (1, "day"),
    (7, "week"),
    (31, "month"),
    (365, "year"),
)


class TavilyProvider(WebProvider):
    name = "tavily"

    def __init__(self, config: TavilyConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())
        #: one ``AsyncTavilyClient`` per key, so a parked key keeps its own session
        self._auth_clients: dict[str, Any] = {}
        #: a single keyless client when the env var is empty
        self._keyless_client: Any = None

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        try:
            import tavily  # noqa: F401
        except ImportError:
            return False, "tavily-python is not installed"
        # Note: Tavily's SDK supports a keyless tier; the provider stays
        # available when no key is set so the web tool can try it before
        # reporting failure.
        return True, ""

    def _get_client(self, key: str | None) -> Any:
        """Build (and cache) the right SDK client for *key*.

        ``key=None`` means keyless mode; ``AsyncTavilyClient`` reads its own
        env var when ``api_key=None``, so the only way to opt into keyless is
        to pass ``None`` explicitly when ``TAVILY_API_KEY`` is unset.
        """
        if key is None:
            if self._keyless_client is None:
                from tavily import AsyncTavilyClient

                self._keyless_client = AsyncTavilyClient(
                    api_key=None,
                    api_base_url=self.config.base_url,
                )
                log.debug("tavily keyless client created")
            return self._keyless_client

        client = self._auth_clients.get(key)
        if client is None:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(
                api_key=key,
                api_base_url=self.config.base_url,
            )
            self._auth_clients[key] = client
            log.debug("tavily auth client created for %s", mask(key))
        return client

    # -- public surface ---------------------------------------------------- #

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        if not self._pool:
            # Keyless mode: single attempt, no rotation. The SDK's keyless
            # path returns 429 quickly when the rate limit trips, and there
            # is nothing else to fall back to — surface the error plainly.
            return await self._search_with(None, query, max_results, days)

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            result = await self._search_with(key, query, max_results, days)
            if result.ok:
                # The call itself succeeded — even an empty result is the
                # query's fault, not the key's. Mark healthy and return.
                self._pool.report(key, ok=True)
                return result
            last_error = result.error
            self._pool.report(key, ok=False, retryable=True)
        return SearchResult(
            query=query, provider=self.name, error=last_error or "every key failed"
        )

    async def fetch(self, url: str) -> Page:
        if not self._pool:
            return await self._fetch_with(None, url)

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            page = await self._fetch_with(key, url)
            if page.markdown.strip() and not page.markdown.startswith("extract failed"):
                self._pool.report(key, ok=True)
                return page
            last_error = page.markdown
            self._pool.report(key, ok=False, retryable=bool(page.markdown))
        return Page(url=url, markdown=last_error or "every tavily key failed")

    # -- single-call helpers ---------------------------------------------- #

    async def _search_with(
        self, key: str | None, query: str, max_results: int, days: int | None
    ) -> SearchResult:
        result = SearchResult(query=query, provider=self.name)
        try:
            client = self._get_client(key)
            raw = await client.search(
                query,
                search_depth=self.config.search_depth,
                topic=self.config.topic,
                max_results=max_results,
                include_answer=self.config.include_answer,
                include_raw_content="markdown" if self.config.include_raw_content else None,
                # A recency window makes the index rank recent pages instead of
                # the most-linked ones, which is how year-old answers win.
                time_range=_time_range(days),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed result
            log.warning("tavily search failed: %s", exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        data = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {}) or {}
        for item in data.get("results") or []:
            item = item if isinstance(item, dict) else getattr(item, "__dict__", {})
            url = str(item.get("url") or "")
            if not url:
                continue
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or ""),
                    content=str(item.get("raw_content") or item.get("content") or ""),
                    score=item.get("score"),
                    published=str(item.get("published_date") or ""),
                )
            )
        answer = data.get("answer")
        result.answer = str(answer) if answer else ""
        log.info("tavily returned %d result(s) for %r", len(result.hits), query)
        return result

    async def _fetch_with(self, key: str | None, url: str) -> Page:
        try:
            client = self._get_client(key)
            raw = await client.extract([url], extract_depth="basic", format="markdown")
        except Exception as exc:  # noqa: BLE001
            log.warning("tavily extract failed for %s: %s", url, exc)
            return Page(url=url, markdown=f"extract failed: {type(exc).__name__}: {exc}")

        data = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {}) or {}
        for item in data.get("results") or []:
            item = item if isinstance(item, dict) else getattr(item, "__dict__", {})
            if str(item.get("url") or "").rstrip("/") == url.rstrip("/"):
                return Page(url=url, markdown=str(item.get("raw_content") or ""))
        failed = data.get("failed_results") or []
        if failed:
            return Page(url=url, markdown=f"extraction failed: {failed}")
        return Page(url=url, markdown="(no content returned)")


def _time_range(days: int | None) -> str | None:
    """Map a day window to Tavily's coarse buckets; ``None`` means unfiltered."""
    if not days or days <= 0:
        return None
    for bucket in TAVILY_TIME_RANGES:
        if days <= bucket[0]:
            return bucket[1]
    return "month"


__all__ = ["TavilyProvider"]