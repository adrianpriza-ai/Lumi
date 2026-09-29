"""Exa, through the official ``exa-py`` SDK.

Exa (https://exa.ai) is a neural/keyword hybrid search engine built for agents:
natural-language queries in, semantically ranked pages out, with cleaned text or
highlights instead of raw HTML. Highlights are the Exa-recommended default for
agentic search — they are tuned to be token-efficient — so the search path asks
for highlights and falls back to full text only when a hit carries none.

Freshness: Exa supports filtering on the publish date directly. A recency
window is sent as ``start_published_date``, so year-old pages stop outranking
this week's news on time-sensitive queries.

Like Firecrawl, Exa has no keyless tier: an unset ``EXA_API_KEY`` makes the
provider unavailable, which the web tool surfaces honestly and falls through
from. Several keys are supported. A comma-separated list is a rotation pool —
the same shape ``OPENAI_API_KEY`` already uses — and a failed key is parked for a
minute before it is retried, so a 429 that clears itself recovers without a
restart. The rotation lives in :class:`lumi.llm.keypool.KeyPool`; this module
just describes which keys are ``available`` and which client to build per key.
"""

from __future__ import annotations

from typing import Any

from ....config import ExaConfig
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


class ExaProvider(WebProvider):
    name = "exa"

    def __init__(self, config: ExaConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())
        #: one ``AsyncExa`` client per key, so a parked key keeps its own session
        self._clients: dict[str, Any] = {}

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        if not self.config.api_key():
            return False, f"{self.config.api_key_env} is not set"
        try:
            import exa_py  # noqa: F401
        except ImportError:
            return False, "exa-py is not installed"
        return True, ""

    def _get_client(self, key: str) -> Any:
        """Build (and cache) an ``AsyncExa`` for *key*."""
        client = self._clients.get(key)
        if client is None:
            from exa_py import AsyncExa

            client = AsyncExa(api_key=key, api_base=self.config.base_url)
            self._clients[key] = client
            log.debug("exa client created for %s", mask(key))
        return client

    # -- public surface ---------------------------------------------------- #

    async def search(
        self, query: str, max_results: int, days: int | None = None
    ) -> SearchResult:
        """Search via Exa, retrying the pool on failure.

        No keyless tier: ``available()`` guarantees at least one key exists,
        but stay defensive about an empty pool — a single-attempt failure is
        friendlier than a None-crash.
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
                    num_results=max_results,
                    type=self.config.search_type,
                    category=self.config.category or None,
                    contents={"text": True},
                    start_published_date=recent_cutoff(days) if days else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("exa search failed on %s: %s", mask(key), exc)
                self._pool.report(key, ok=False, retryable=True)
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            self._pool.report(key, ok=True)
            self._populate(raw, result)
            log.info("exa returned %d result(s) for %r", len(result.hits), query)
            return result

        result.error = last_error or "every exa key failed"
        return result

    async def fetch(self, url: str) -> Page:
        """Read one page as text via ``get_contents``, retrying the pool."""
        if not self._pool:
            return Page(url=url, markdown=f"{self.config.api_key_env} is not set")

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            client = self._get_client(key)
            try:
                raw = await client.get_contents([url], text=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("exa fetch failed on %s: %s", mask(key), exc)
                self._pool.report(key, ok=False, retryable=True)
                last_error = f"fetch failed: {type(exc).__name__}: {exc}"
                continue

            self._pool.report(key, ok=True)
            page = self._page_from(raw, url)
            if page.markdown.strip() and not page.markdown.startswith(("extraction failed", "no content")):
                return page
            # The API call itself succeeded, so the miss is the URL's fault,
            # not the key's: leave the key in rotation and move on.
            last_error = page.markdown
            continue

        return Page(url=url, markdown=last_error or "every exa key failed")

    # -- response shaping -------------------------------------------------- #

    def _populate(self, raw: Any, result: SearchResult) -> None:
        """Flatten an ``ExaResults`` (or dict) into :class:`SearchHit` entries."""
        data = _as_dict(raw)
        results = data.get("results") or []

        for item in results:
            item = _as_dict(item)
            url = str(item.get("url") or "")
            if not url:
                continue
            text = str(item.get("text") or "")
            highlights = item.get("highlights") or []
            joined = " ".join(str(h) for h in highlights)
            # Highlights are Exa's token-efficient default; fall back to the
            # full text for the snippet when a hit carries none.
            snippet = (joined or text)[:400]
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=snippet,
                    content=text or joined,
                    score=item.get("score"),
                    published=str(item.get("publishedDate") or ""),
                )
            )

    def _page_from(self, raw: Any, url: str) -> Page:
        """Turn a ``get_contents`` response into a :class:`Page`."""
        data = _as_dict(raw)
        results = data.get("results") or []
        for item in results:
            item = _as_dict(item)
            if str(item.get("url") or "").rstrip("/") == url.rstrip("/"):
                return Page(
                    url=url,
                    title=str(item.get("title") or ""),
                    markdown=str(item.get("text") or ""),
                )
        # Either an empty response or the URL came back under a normalised
        # address; both read to the caller as "nothing here".
        return Page(url=url, markdown="(no content returned)")


__all__ = ["ExaProvider"]
