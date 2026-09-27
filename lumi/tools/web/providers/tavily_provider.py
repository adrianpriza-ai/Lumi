"""Tavily, through the official Python SDK.

Tavily is the best fit for agentic search: it returns cleaned, LLM-ready
snippets rather than raw HTML, and can bundle an answer with the results. Keyless
usage is supported by the API with a much lower rate limit, so a missing key is
reported as "degraded" rather than fatal.
"""

from __future__ import annotations

from typing import Any

from ....config import TavilyConfig
from ....util.log import get_logger
from .base import Page, SearchHit, SearchResult, WebProvider

log = get_logger(__name__)


class TavilyProvider(WebProvider):
    name = "tavily"

    def __init__(self, config: TavilyConfig) -> None:
        self.config = config
        self._client: Any = None

    def available(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "disabled in config"
        if not self.config.api_key():
            return False, f"{self.config.api_key_env} is not set"
        try:
            import tavily  # noqa: F401
        except ImportError:
            return False, "tavily-python is not installed"
        return True, ""

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import AsyncTavilyClient

            self._client = AsyncTavilyClient(
                api_key=self.config.api_key(),
                api_base_url=self.config.base_url,
            )
            log.debug("tavily client created")
        return self._client

    async def search(self, query: str, max_results: int) -> SearchResult:
        result = SearchResult(query=query, provider=self.name)
        try:
            raw = await self._get_client().search(
                query,
                search_depth=self.config.search_depth,
                topic=self.config.topic,
                max_results=max_results,
                include_answer=self.config.include_answer,
                include_raw_content="markdown" if self.config.include_raw_content else None,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
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
                )
            )
        answer = data.get("answer")
        result.answer = str(answer) if answer else ""
        log.info("tavily returned %d result(s) for %r", len(result.hits), query)
        return result

    async def fetch(self, url: str) -> Page:
        try:
            raw = await self._get_client().extract(
                [url], extract_depth="basic", format="markdown"
            )
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


__all__ = ["TavilyProvider"]
