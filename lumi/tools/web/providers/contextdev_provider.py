"""Context.dev (https://context.dev) — web search built for AI agents.

Context.dev answers a search with ranked hits that carry a relevance label,
query-relevant highlight passages, and optionally full-page markdown — the
highlights are what make a snippet worth reading, and they are the reason to
pick this provider over a plain search API.

There is no official Python SDK worth pinning, so this provider speaks the
REST API directly over ``httpx`` — the same posture as the OCR and image
clients: one request, one clear failure, no borrowed retry policy.

.. code-block:: bash

    curl https://api.context.dev/v1/web/search \\
      -H "Authorization: Bearer $CONTEXT_DEV_API_KEY" \\
      -d '{"query": "how does hybrid search improve RAG retrieval?",
           "numResults": 10}'

Keyless mode: the hosted API has a free tier but still wants a key, so an
unset ``CONTEXT_DEV_API_KEY`` makes the provider unavailable there and the
web tool falls through. An overridden ``base_url`` is taken at its word —
the owner's endpoint, which may take no auth — and runs with one
credential-free attempt per call.

Several keys are supported. A comma-separated list is a rotation pool — the
same shape ``OPENAI_API_KEY`` already uses — and a failed key is parked for
a minute before it is retried, so a 429 that clears itself recovers without
a restart. The rotation lives in :class:`lumi.llm.keypool.KeyPool`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....config import ContextDevConfig
from ....llm.keypool import KeyPool
from ....util.log import get_logger
from ....util.text import format_error, truncate
from .base import Page, SearchHit, SearchResult, WebProvider, final_url

if TYPE_CHECKING:  # pragma: no cover - annotation only, imported lazily at call time
    import httpx

log = get_logger(__name__)

#: Characters of a failed response body quoted back to the model. Enough to
#: name the problem ("USAGE_EXCEEDED", "RATE_LIMITED"), short enough to not
#: eat the context.
ERROR_BODY_CHARS = 300

#: The search API caps the query length; anything longer is truncated with a
#: debug log rather than silently.
QUERY_MAX_CHARS = 500


class ContextScrapeFailed(RuntimeError):
    """A scrape request answered 200 but produced no markdown.

    Raised by :meth:`_fetch_with` so the pool loop can rotate keys without
    matching on the error text; the caller renders the ``scrape failed:``
    page the web tool already recognizes.
    """


#: Context.dev ranks every hit ``high`` / ``medium`` / ``low``; the other
#: providers carry a numeric score, so the label becomes one — a rank is a
#: score that admits only three values.
RELEVANCE_SCORES = {"high": 1.0, "medium": 0.5, "low": 0.0}

#: Day-window ceilings mapped to Context.dev's ``freshness`` buckets. A
#: bucket must fully contain the window, so "last 3 days" does not silently
#: widen to a month — the same rule Tavily's ``time_range`` mapping follows.
FRESHNESS: tuple[tuple[int, str], ...] = (
    (1, "last_24_hours"),
    (7, "last_week"),
    (31, "last_month"),
    (365, "last_year"),
)


def _freshness(days: int | None) -> str | None:
    """Map a day window to a ``freshness`` bucket; ``None`` means unfiltered."""
    if not days or days <= 0:
        return None
    for ceiling, bucket in FRESHNESS:
        if days <= ceiling:
            return bucket
    return None


class ContextDevProvider(WebProvider):
    name = "contextdev"

    def __init__(self, config: ContextDevConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        # The hosted API bills per call, so it needs a key; an overridden
        # base_url is the owner's endpoint and may take no auth at all.
        if not self.config.api_key() and self.config.needs_key():
            return False, f"{self.config.api_key_env} is not set"
        # Imported here, not at module scope: this provider is part of the web
        # tool's fixed import graph, so a module-level import made a missing
        # httpx a startup crash for the whole bot instead of one provider
        # deactivating. Same rule the other raw-httpx clients already follow.
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "httpx is not installed; the context.dev provider needs it"
        return True, ""

    # -- HTTP --------------------------------------------------------------- #

    def _headers(self, key: str) -> dict[str, str]:
        """Auth header when there is a key; nothing when the endpoint is keyless."""
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def _post(self, key: str, path: str, payload: dict[str, Any]) -> httpx.Response:
        """One POST to the API. Raises :class:`RuntimeError` on any failure.

        A failure carries the endpoint, status and a snippet of the
        provider's own response body — Context.dev fails in specific ways
        (``USAGE_EXCEEDED``, ``RATE_LIMITED``) and the model needs the exact
        words, not a guess.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - available() catches it first
            raise RuntimeError(
                "httpx is not installed; the context.dev provider needs it (pip install httpx)"
            ) from exc
        url = self.config.base_url.strip().rstrip("/") + path
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=self._headers(key),
            ) as http:
                response = await http.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach {url}: {format_error(exc)}") from exc
        if response.status_code >= 400:
            raise RuntimeError(
                f"{url} returned HTTP {response.status_code}: "
                f"{truncate(response.text, ERROR_BODY_CHARS)}"
            )
        return response

    # -- public surface ----------------------------------------------------- #

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        """Search via Context.dev, retrying the pool on failure.

        Keyless mode (owner-run endpoint) makes a single attempt, the same
        shape Tavily's keyless tier takes.
        """
        result = SearchResult(query=query, provider=self.name)
        if not self._pool:
            return await self._search_with("", query, max_results, days, result)

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            await self._search_with(key, query, max_results, days, result)
            if result.ok:
                self._pool.report(key, ok=True)
                return result
            last_error = result.error
            self._pool.report(key, ok=False, retryable=True)
        result.error = last_error or "every context.dev key failed"
        return result

    async def _search_with(
        self, key: str, query: str, max_results: int, days: int | None, result: SearchResult
    ) -> SearchResult:
        # Each pool attempt starts clean: a previous key's failure must not
        # stick to a result this key made healthy.
        result.error = ""
        if len(query) > QUERY_MAX_CHARS:
            log.debug(
                "context.dev query truncated from %d to %d chars",
                len(query),
                QUERY_MAX_CHARS,
            )
        payload: dict[str, Any] = {
            "query": query[:QUERY_MAX_CHARS],
            "numResults": max_results,
        }
        if freshness := _freshness(days):
            payload["freshness"] = freshness
        # Highlights cost one credit per ten results and are the point of
        # this provider; markdown costs one per page and is off by default —
        # the firecrawl top-up covers thin hits more cheaply.
        if self.config.highlights:
            payload["highlightsOptions"] = {"enabled": True}
        if self.config.include_markdown:
            payload["markdownOptions"] = {"enabled": True}

        try:
            response = await self._post(key, "/web/search", payload)
            data = response.json()
        except Exception as exc:
            # Any failure — HTTP, network, a 200 with a non-JSON body, an
            # unexpected shape — is a failed *attempt*: the pool rotates to
            # the next key, it is not a provider crash.
            result.error = str(exc)
            return result

        # A 200 can still be an error: some compatible servers answer with
        # ``{"error_code": "RATE_LIMITED"}`` and no results. Without this
        # check that reads as "success with 0 hits" and the pool never
        # learns the key is throttled.
        if error := _search_error_of(data):
            result.error = error
            return result

        for item in data.get("results") or []:
            item = item if isinstance(item, dict) else {}
            url = str(item.get("url") or "")
            if not url:
                continue
            highlights = _highlights_of(item)
            markdown = _markdown_of(item)
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("description") or ""),
                    # The body the model reads: highlight passages first —
                    # they are query-relevant — else the page markdown when
                    # it was asked for.
                    content=highlights or markdown,
                    score=RELEVANCE_SCORES.get(str(item.get("relevance") or "")),
                )
            )
        log.info("context.dev returned %d result(s) for %r", len(result.hits), query)
        return result

    async def fetch(self, url: str) -> Page:
        """Scrape one page as markdown, retrying the pool on failure."""
        if not self._pool:
            try:
                return await self._fetch_with("", url)
            except ContextScrapeFailed as exc:  # keyless: one attempt, no rotation
                return Page(url=url, markdown=f"scrape failed: {exc}")

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            try:
                page = await self._fetch_with(key, url)
            except ContextScrapeFailed as exc:
                # The endpoint answered, but the scrape itself failed — the
                # key is innocent (a 429 would have raised as HTTP 4xx), yet
                # the attempt failed, so rotate and record why.
                last_error = f"scrape failed: {exc}"
                self._pool.report(key, ok=False, retryable=False)
                continue
            self._pool.report(key, ok=True)
            return page
        return Page(url=url, markdown=last_error or "every context.dev key failed")

    async def _fetch_with(self, key: str, url: str) -> Page:
        payload = {"url": url, "formats": {"markdown": True}}
        try:
            response = await self._post(key, "/web/scrape", payload)
            body = response.json()
        except Exception as exc:
            # Same as search: any failure is a failed attempt for the pool
            # to rotate on, not a provider crash.
            raise ContextScrapeFailed(str(exc)) from exc

        # Every output key is present with a ``success`` flag; check the one
        # we asked for rather than trusting the 200.
        markdown = body.get("markdown") or {}
        if isinstance(markdown, dict):
            if not markdown.get("success"):
                reason = markdown.get("error_code") or markdown.get("message") or "no content"
                raise ContextScrapeFailed(str(reason))
            text = str(markdown.get("data") or "")
        else:  # defensive: some compatible servers inline the string
            text = str(markdown or "")

        metadata = body.get("metadata") or {}
        title = str(metadata.get("title") or "") if isinstance(metadata, dict) else ""
        return Page(
            url=final_url(
                metadata.get("url") if isinstance(metadata, dict) else None, requested=url
            ),
            title=title,
            markdown=text or "(no content)",
        )


def _search_error_of(data: Any) -> str:
    """The provider's own error words from a 200 body that reports failure.

    Some compatible servers answer an exhausted quota or a throttled key with
    an HTTP 200 and ``{"error_code": "RATE_LIMITED"}`` (or ``error``, or
    ``success: false``) and no results. Without this check that reads as
    "success with 0 hits" and the pool never learns the key is unhealthy.
    A body that also carries results is trusted — the hits win.
    """
    if not isinstance(data, dict) or data.get("results"):
        return ""
    if data.get("success") is False:
        return str(data.get("error_code") or data.get("message") or "search failed")
    code = data.get("error_code") or data.get("error")
    return str(code) if code else ""


def _highlights_of(item: dict[str, Any]) -> str:
    """The highlight passages of one search hit, joined; empty when none."""
    block = item.get("highlights")
    if not isinstance(block, dict):
        return ""
    passages = block.get("highlights") or []
    return "\n\n".join(str(p) for p in passages if str(p).strip())


def _markdown_of(item: dict[str, Any]) -> str:
    """The page markdown of one search hit, when it was requested and succeeded."""
    block = item.get("markdown")
    if not isinstance(block, dict) or not block.get("success"):
        return ""
    return str(block.get("markdown") or "")


__all__ = ["ContextDevProvider", "ContextScrapeFailed", "FRESHNESS"]
