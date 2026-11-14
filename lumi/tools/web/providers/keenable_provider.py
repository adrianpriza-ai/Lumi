"""Keenable (https://keenable.ai) — web search and page fetch for AI agents.

Keenable runs its own index and answers with ranked pages whose text is
already extracted, so a hit arrives as something the model can read rather
than a title it has to take on trust. Fetch returns one page as clean
markdown.

Keyless is the default: ``/search/public`` and ``/fetch/public`` need no
account and no key, only an ``X-Keenable-Title`` header naming the calling
app, so the provider is *available* on a fresh clone. A ``KEENABLE_API_KEY``
lifts the per-IP hourly cap and puts the calls on the account's free monthly
allowance; the same requests then go to ``/search`` and ``/fetch`` with an
``X-API-Key`` header.

There is no need to pin the SDK for this provider: the wire format is two
endpoints, and speaking it directly over ``httpx`` keeps the web tool's
dependency graph unchanged — the same posture as Context.dev.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....config import KeenableConfig
from ....llm.keypool import KeyPool, is_retryable
from ....util.log import get_logger
from ....util.text import format_error, truncate
from .base import Page, SearchHit, SearchResult, WebProvider, final_url

if TYPE_CHECKING:  # pragma: no cover - annotation only, imported lazily at call time
    import httpx

log = get_logger(__name__)

#: Characters of a failed response body quoted back to the model. Enough to
#: name the problem, short enough to not eat the context.
ERROR_BODY_CHARS = 300

#: The API caps ``max_results`` at 50 and rejects anything outside 1–50.
MAX_RESULTS = 50

#: ``snippet_max_length`` must be at least this; the API answers 400 below it.
MIN_SNIPPET_CHARS = 180
MAX_SNIPPET_CHARS = 10000

#: Search modes the API accepts. Anything else is left to the server default.
SEARCH_MODES = ("pro", "realtime")


class KeenableFetchFailed(RuntimeError):
    """A fetch request produced no content, at the HTTP layer or in the body.

    Raised by :meth:`_fetch_with` so the pool loop can rotate keys without
    matching on the error text; the caller renders the ``scrape failed:``
    page the web tool already recognizes. ``retryable`` says whether the
    *key* is implicated: a 401, 403, 429 or 5xx parks it for the cooldown,
    while a 400, a 404, or a page whose extraction failed is the request's
    or the site's fault, and the key stays fit for the next URL.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class KeenableProvider(WebProvider):
    name = "keenable"

    def __init__(self, config: KeenableConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.key_strategy_of())

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        # Imported here, not at module scope: this provider is part of the web
        # tool's fixed import graph, so a module-level import made a missing
        # httpx a startup crash for the whole bot instead of one provider
        # deactivating. Same rule the other raw-httpx clients follow.
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "httpx is not installed; the keenable provider needs it"
        # Keyless is a supported tier, so no key never makes this unavailable.
        return True, ""

    def status(self) -> str:
        """One line for ``/status``: the tier it runs on, or why it is off."""
        ok, reason = self.available()
        if not ok:
            return f"off — {reason}" if reason.startswith("disabled") else f"unavailable — {reason}"
        mode = self.config.mode if self.config.mode in SEARCH_MODES else "server default"
        keys = self.config.api_keys()
        if keys:
            plural = "key" if len(keys) == 1 else "keys"
            return f"keyed ({len(keys)} {plural}) · mode {mode}"
        return f"keyless (public endpoints as {self.config.app_title!r}) · mode {mode}"

    # -- HTTP --------------------------------------------------------------- #

    def _headers(self, key: str) -> dict[str, str]:
        """The key header when there is a key, else the public app identifier."""
        if key:
            return {"X-API-Key": key}
        # The public endpoints answer 400 "Missing app identifier" without it.
        return {"X-Keenable-Title": self.config.app_title or "Lumi"}

    def _path(self, endpoint: str, key: str) -> str:
        """The keyed endpoint, or its keyless ``/public`` twin."""
        return f"/{endpoint}" if key else f"/{endpoint}/public"

    async def _request(
        self,
        method: str,
        key: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """One request to the API. Raises :class:`RuntimeError` on any failure.

        A failure carries the endpoint, status and a snippet of the
        provider's own response body: Keenable names the exact problem
        (``insufficient_credits``, ``invalid_api_key``), and the model needs
        those words rather than a guess.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - available() catches it first
            raise RuntimeError(
                "httpx is not installed; the keenable provider needs it (pip install httpx)"
            ) from exc
        url = self.config.base_url.strip().rstrip("/") + self._path(endpoint, key)
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=self._headers(key),
            ) as http:
                if method == "GET":
                    response = await http.get(url, params=params)
                else:
                    response = await http.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not reach {url}: {format_error(exc)}") from exc
        if response.status_code >= 400:
            raise RuntimeError(
                f"{url} returned HTTP {response.status_code}: "
                f"{truncate(_error_message(response), ERROR_BODY_CHARS)}"
            )
        return response

    # -- public surface ----------------------------------------------------- #

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        """Search via Keenable, retrying the pool on failure.

        Keyless mode makes a single attempt, the same shape Tavily's keyless
        tier takes: there is no second credential to rotate to.
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
            self._pool.report(key, ok=False, retryable=is_retryable(RuntimeError(last_error)))
        result.error = last_error or "every keenable key failed"
        return result

    async def _search_with(
        self, key: str, query: str, max_results: int, days: int | None, result: SearchResult
    ) -> SearchResult:
        # Each pool attempt starts clean: a previous key's failure must not
        # stick to a result this key made healthy.
        result.error = ""
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(int(max_results), MAX_RESULTS)),
        }
        if self.config.mode in SEARCH_MODES:
            payload["mode"] = self.config.mode
        # The API takes relative deltas ("7d"), so the window goes out as one
        # rather than as a cutoff computed here.
        if days and days > 0:
            payload["published_after"] = f"{days}d"
        if self.config.snippet_max_length:
            payload["snippet_max_length"] = max(
                MIN_SNIPPET_CHARS, min(int(self.config.snippet_max_length), MAX_SNIPPET_CHARS)
            )

        try:
            response = await self._request("POST", key, "search", payload=payload)
            data = response.json()
        except Exception as exc:
            # Any failure — HTTP, network, a 200 with a non-JSON body, an
            # unexpected shape — is a failed *attempt*: the pool rotates to
            # the next key, it is not a provider crash.
            result.error = str(exc)
            return result

        if not isinstance(data, dict):
            result.error = "unexpected response shape"
            return result
        # A 200 can still be an error on a compatible server; no results plus
        # an error word is a failed attempt, not an empty search.
        if error := _search_error_of(data):
            result.error = error
            return result

        for item in data.get("results") or []:
            item = item if isinstance(item, dict) else {}
            url = str(item.get("url") or "")
            if not url:
                continue
            snippet = str(item.get("snippet") or "")
            result.hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=snippet,
                    # The snippet is page text chosen for the query, already
                    # markdown: it is the body the model should read.
                    content=snippet,
                    published=str(item.get("published_at") or ""),
                )
            )
        log.info("keenable returned %d result(s) for %r", len(result.hits), query)
        return result

    async def fetch(self, url: str) -> Page:
        """Fetch one page as markdown, retrying the pool on failure."""
        if not self._pool:
            try:
                return await self._fetch_with("", url)
            except KeenableFetchFailed as exc:  # keyless: one attempt, no rotation
                return Page(url=url, markdown=f"scrape failed: {exc}")

        tried: list[str] = []
        last_error = ""
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            try:
                page = await self._fetch_with(key, url)
            except KeenableFetchFailed as exc:
                # The key is parked only when the failure is its fault —
                # decided inside _fetch_with, by the same fault rule the
                # search loop applies.
                last_error = f"scrape failed: {exc}"
                self._pool.report(key, ok=False, retryable=exc.retryable)
                continue
            self._pool.report(key, ok=True)
            return page
        return Page(url=url, markdown=last_error or "every keenable key failed")

    async def _fetch_with(self, key: str, url: str) -> Page:
        try:
            response = await self._request("GET", key, "fetch", params={"url": url})
            body = response.json()
        except Exception as exc:
            # Same as search: any failure is a failed attempt for the pool
            # to rotate on, not a provider crash.
            raise KeenableFetchFailed(str(exc), retryable=is_retryable(exc)) from exc

        if not isinstance(body, dict):
            raise KeenableFetchFailed("unexpected response shape")
        content = str(body.get("content") or "")
        if not content.strip():
            # The request succeeded; the page had nothing to extract. The key
            # stays fit for the next URL.
            raise KeenableFetchFailed("no content returned", retryable=False)

        return Page(
            url=final_url(body.get("url"), requested=url),
            title=str(body.get("title") or ""),
            markdown=content,
        )


def _error_message(response: httpx.Response) -> str:
    """The provider's own words from an error body, else the raw text."""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON body is reported as-is
        return response.text
    if isinstance(body, dict):
        message = body.get("message") or body.get("error") or body.get("code")
        if message:
            return str(message)
    return response.text


def _search_error_of(data: Any) -> str:
    """The provider's own error words from a 200 body that reports failure.

    A body that also carries results is trusted — the hits win.
    """
    if not isinstance(data, dict) or data.get("results"):
        return ""
    if data.get("success") is False:
        return str(data.get("message") or data.get("error") or data.get("code") or "search failed")
    code = data.get("error") or data.get("code")
    return str(code) if code else ""


__all__ = ["KeenableProvider", "KeenableFetchFailed"]
