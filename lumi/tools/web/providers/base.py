"""Web provider contract.

Three implementations ship: the Tavily SDK, the Firecrawl SDK, and a generic MCP
client that reads ``.mcp.json``. They all normalise to :class:`SearchHit` and
:class:`Page` so the tool layer never has to care which one answered.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from ....util.text import truncate


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    score: float | None = None

    def render(self, index: int, max_chars: int) -> str:
        head = f"[{index}] {self.title or self.url}\n{self.url}"
        body = self.content.strip() or self.snippet.strip()
        return f"{head}\n{truncate(body, max_chars)}" if body else head


@dataclass(slots=True)
class SearchResult:
    query: str
    provider: str
    hits: list[SearchHit] = field(default_factory=list)
    answer: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def render(self, max_chars: int) -> str:
        if self.error:
            return f"{self.provider} failed: {self.error}"
        lines = [f"Results for {self.query!r} (via {self.provider}):"]
        if self.answer:
            lines += ["", "Summary:", self.answer]
        lines.append("")
        for index, hit in enumerate(self.hits, start=1):
            lines.append(hit.render(index, max_chars))
            lines.append("")
        if not self.hits:
            lines.append("(no results)")
        return "\n".join(lines)


@dataclass(slots=True)
class Page:
    url: str
    title: str = ""
    markdown: str = ""

    def render(self, max_chars: int) -> str:
        head = f"--- {self.title or self.url} ---\n{self.url}\n"
        return head + truncate(self.markdown.strip(), max_chars)


class WebProvider(abc.ABC):
    """One way of reaching the web."""

    name: str = "unnamed"

    @abc.abstractmethod
    def available(self) -> tuple[bool, str]:
        """Whether this provider can be used right now, plus a reason."""

    @abc.abstractmethod
    async def search(self, query: str, max_results: int) -> SearchResult:
        ...

    @abc.abstractmethod
    async def fetch(self, url: str) -> Page:
        ...

    def summary_line(self) -> str:
        ok, reason = self.available()
        mark = "" if ok else f" _(unavailable: {reason})_"
        return f"- `{self.name}`{mark}"


__all__ = ["SearchHit", "SearchResult", "Page", "WebProvider"]
