"""Web provider contract.

Four implementations ship: the Firecrawl SDK, the Exa SDK, the Tavily SDK, and
a generic MCP client that reads ``.mcp.json``. They all normalise to
:class:`SearchHit` and :class:`Page` so the tool layer never has to care which
one answered.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ....util.text import truncate

#: Body-text floor for a hit when a search returns few results. The result's
#: budget is shared between the hits, but with five or fewer — the default
#: ``max_results`` — each hit gets at least this many characters of substance,
#: so snippets are readable instead of teaser-sized. Total render size stays
#: under the agent's ``for_model`` cap (8000 chars) in every case.
PER_HIT_CONTENT_CHARS = 1400

#: Hit count up to which the per-hit floor applies. Above this, sharing the
#: budget evenly is the only way to keep the render bounded.
PER_HIT_FLOOR_MAX_HITS = 5


@dataclass(slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    score: float | None = None
    #: Publish date when the provider reports one, rendered with the hit so a
    #: stale result can be spotted (and re-searched) before it is quoted.
    published: str = ""

    def render(self, index: int, max_chars: int) -> str:
        head = f"[{index}] {self.title or self.url}\n{self.url}"
        if self.published:
            head += f"\npublished {self.published}"
        body = self.content.strip() or self.snippet.strip()
        return f"{head}\n{truncate(body, max_chars)}" if body else head


@dataclass(slots=True)
class SearchResult:
    query: str
    provider: str
    hits: list[SearchHit] = field(default_factory=list)
    answer: str = ""
    error: str = ""
    #: Recency window the caller asked for, in days. ``None`` means the search
    #: was unfiltered. Stamped by the tool layer after the provider answers so
    #: the rendered text can say so — no shared mutable state, safe across
    #: concurrent chats.
    recency_days: int | None = None

    @property
    def ok(self) -> bool:
        return not self.error

    def render(self, max_chars: int) -> str:
        if self.error:
            return f"{self.provider} failed: {self.error}"
        # The search date rides along with every result. Providers rank well-
        # linked pages over fresh ones, and a result set that reads as timeless
        # invites answering from stale hits; this makes the age unmissable.
        today = datetime.now(UTC).strftime("%Y-%m-%d (%A)")
        window = f", last {self.recency_days} day(s) only" if self.recency_days else ""
        lines = [f"Results for {self.query!r} (via {self.provider}, searched on {today}{window}):"]
        if self.answer:
            lines += ["", "Summary:", self.answer]
        lines.append("")
        # Share the budget out, with a floor for small result sets: a lone hit
        # gets the whole cap, and up to five hits each get enough to be worth
        # reading. Larger sets share evenly so the render cannot balloon past
        # the tool-message cap.
        share = max_chars // max(len(self.hits), 1)
        per_hit = (
            max(PER_HIT_CONTENT_CHARS, share)
            if len(self.hits) <= PER_HIT_FLOOR_MAX_HITS
            else share
        )
        for index, hit in enumerate(self.hits, start=1):
            lines.append(hit.render(index, per_hit))
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
    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        """Search *query*, optionally limited to pages from the last *days*."""

    @abc.abstractmethod
    async def fetch(self, url: str) -> Page:
        ...

    def summary_line(self) -> str:
        ok, reason = self.available()
        mark = "" if ok else f" _(unavailable: {reason})_"
        return f"- `{self.name}`{mark}"


def recent_cutoff(days: int) -> str:
    """ISO date *days* back from now, for provider recency filters.

    Every provider takes a different shape — a ``time_range``, a ``tbs``, an
    ISO ``start_published_date`` — but they all mean the same thing: only
    pages from after this moment. Computed per call, never cached, so a bot
    that runs for months does not go stale.
    """
    return (datetime.now(UTC).date() - timedelta(days=max(days, 0))).isoformat()


__all__ = [
    "SearchHit",
    "SearchResult",
    "Page",
    "WebProvider",
    "recent_cutoff",
    "PER_HIT_CONTENT_CHARS",
    "PER_HIT_FLOOR_MAX_HITS",
]
