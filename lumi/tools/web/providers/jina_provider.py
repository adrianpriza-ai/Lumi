"""Jina Reader (https://r.jina.ai) and Search (https://s.jina.ai).

Reader reads a URL and returns clean markdown; the whole API is the URL::

    curl "https://r.jina.ai/https://www.example.com"

No account and no key, so a page that a search provider cannot read still has
a last resort. ``JINA_API_KEY``, when set, goes out as a bearer token and lifts
the keyless rate limit; comma-separated keys rotate like the other providers'.

Search is the same shape (``https://s.jina.ai/<query>``) but the endpoint
requires a key, unlike Reader. So the provider only *offers* search while a
key is configured (``supports_search``), and without one the tool skips it
instead of spending a failed attempt on every search. Search results arrive
already read: each entry carries the page's title, URL, description and full
content, which is why it makes a useful last-resort search fallback.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import date
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    import httpx

from ....config import JinaConfig
from ....llm.keypool import KeyPool, is_retryable
from ....util.log import get_logger
from ....util.text import format_error, truncate
from .base import Page, SearchHit, SearchResult, WebProvider, final_url, recent_cutoff

log = get_logger(__name__)

#: Characters of an error body quoted back to the model.
ERROR_BODY_CHARS = 300


class JinaRequestFailed(RuntimeError):
    """A Reader or Search request produced no usable body.

    ``retryable`` means the key is implicated (401/403/429/5xx), so the pool
    can park it and try the next one; a 400 (bad URL) or an empty/odd body
    leaves the key fit for the next call. One exception for both endpoints
    keeps the two pool loops identical.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def _published_within(published: str, cutoff: str) -> bool:
    """Whether a hit's date is unknown or on/after *cutoff* (an ISO date).

    Search has no recency parameter, so the window is applied here to the
    dates the hits actually report. An undated hit is kept: a missing date is
    not evidence that the page is old, and dropping those would leave searches
    that ask for recency with nothing at all.
    """
    day = published.strip()[:10]
    try:
        return date.fromisoformat(day) >= date.fromisoformat(cutoff)
    except ValueError:
        return True


class JinaProvider(WebProvider):
    name = "jina"

    def __init__(self, config: JinaConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())

    @property
    def supports_search(self) -> bool:
        # s.jina.ai answers 401 without a key, so keyless is fetch-only.
        return bool(self._pool)

    def status(self) -> str:
        """One line for ``/status``: what Reader and Search are running on."""
        ok, reason = self.available()
        if not ok:
            return f"off — {reason}" if reason.startswith("disabled") else f"unavailable — {reason}"
        keys = len(self.config.api_keys())
        if keys:
            plural = "key" if keys == 1 else "keys"
            return f"reader + search ({keys} {plural})"
        return "reader keyless · search off (JINA_API_KEY unset)"

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        # Lazy, like every raw-httpx provider: a missing httpx disables this
        # provider, it does not crash startup.
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "httpx is not installed; the jina provider needs it"
        # Keyless is the normal case: no key never makes this unavailable.
        return True, ""

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        """Search the web via s.jina.ai; needs a key.

        The endpoint has no recency parameter, so *days* is applied to the
        publication dates the hits report (undated hits are kept) rather than
        being silently ignored while the tool advertises a window.
        """
        if not self._pool:
            return SearchResult(
                query=query,
                provider=self.name,
                error="jina search needs a JINA_API_KEY (fetch works keyless)",
            )

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            try:
                hits = await self._search_with(key, query, max_results, days)
            except JinaRequestFailed as exc:
                last_error = f"jina search failed: {exc}"
                self._pool.report(key, ok=False, retryable=exc.retryable)
                continue
            self._pool.report(key, ok=True)
            return SearchResult(query=query, provider=self.name, hits=hits)
        return SearchResult(
            query=query, provider=self.name, error=last_error or "every jina key failed"
        )

    async def _search_with(
        self, key: str, query: str, max_results: int, days: int | None
    ) -> list[SearchHit]:
        base = self.config.search_base_url.strip().rstrip("/") or "https://s.jina.ai"
        target = base + "/" + quote(query.strip(), safe="")
        # ``num`` asks for exactly the hits the tool would keep; the API caps
        # it at 20, and the slice below keeps the result list honest anyway.
        params = {"num": str(max(1, min(max_results, 20)))}
        response = await self._get(key, target, params=params)

        body: Any = None
        with suppress(Exception):
            body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            # A SERP rendered as markdown cannot be turned back into hits with
            # their metadata intact, so this is a failure, not a result list.
            raise JinaRequestFailed(
                "no JSON result list returned (does JINA_API_KEY have credits?)",
                retryable=False,
            )

        cutoff = recent_cutoff(days) if days and days > 0 else ""
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("finalUrl") or "").strip()
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            published = str(item.get("publishedTime") or "")
            if cutoff and not _published_within(published, cutoff):
                continue
            seen.add(url)
            hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("description") or ""),
                    content=str(item.get("content") or ""),
                    published=published,
                )
            )
        log.info("jina search %r returned %d hits", truncate(query, 80, "…"), len(hits))
        return hits[:max_results]

    async def _get(
        self, key: str, target: str, *, params: dict[str, str] | None = None
    ) -> httpx.Response:
        """GET *target*; one place for the timeout, key headers and errors."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - available() catches it first
            raise JinaRequestFailed(
                "httpx is not installed; the jina provider needs it", retryable=False
            ) from exc

        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds, headers=headers
            ) as http:
                response = await http.get(target, params=params)
        except httpx.HTTPError as exc:
            raise JinaRequestFailed(f"could not reach {target}: {format_error(exc)}") from exc
        if response.status_code >= 400:
            raise JinaRequestFailed(
                f"{target} returned HTTP {response.status_code}: "
                f"{truncate(response.text, ERROR_BODY_CHARS)}",
                retryable=is_retryable(RuntimeError(f"HTTP {response.status_code}")),
            )
        return response

    async def fetch(self, url: str) -> Page:
        if not self._pool:
            try:
                return await self._fetch_with("", url)
            except JinaRequestFailed as exc:  # keyless: one attempt, no rotation
                return Page(url=url, markdown=f"scrape failed: {exc}")

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            try:
                page = await self._fetch_with(key, url)
            except JinaRequestFailed as exc:
                last_error = f"scrape failed: {exc}"
                self._pool.report(key, ok=False, retryable=exc.retryable)
                continue
            self._pool.report(key, ok=True)
            return page
        return Page(url=url, markdown=last_error or "every jina key failed")

    async def _fetch_with(self, key: str, url: str) -> Page:
        target = self.config.base_url.strip().rstrip("/") + "/" + url
        response = await self._get(key, target)

        body: Any = None
        # Plain-text markdown is a valid answer; only a parsed dict is JSON.
        with suppress(Exception):
            body = response.json()

        if isinstance(body, dict):
            data = body.get("data") if isinstance(body.get("data"), dict) else body
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            resolved = final_url(data.get("url"), requested=url)
        else:
            # Reader answers markdown directly when it cannot honour the JSON
            # accept header; that text is still the page.
            content = response.text
            title = ""
            resolved = url

        if not content.strip():
            raise JinaRequestFailed("no content returned", retryable=False)
        log.info("jina read %s (%d chars)", truncate(url, 80, "…"), len(content))
        return Page(url=resolved, title=title, markdown=content)


__all__ = ["JinaProvider", "JinaRequestFailed"]
