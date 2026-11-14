"""Tavily + Firecrawl + Exa provider tests.

The SDK clients are injected through the providers' private ``_get_client``
method (or the keyless slot), so each test owns the response shape and never
touches the network. The contract under test:

- **Tavily keyless** — the provider is *available* without an API key and
  builds its client with ``api_key=None`` when none is configured.
- **Tavily multi-key** — a 401 on one key falls through to the next; the
  first healthy key wins; an unknown strategy falls back to ``fallback``.
- **Firecrawl multi-key** — same rotation rules; the provider is *not*
  available without a key.
- **Exa multi-key** — same rotation rules; the provider is *not* available
  without a key; search highlights fall back to full text when missing.
- **Context.dev** — raw httpx (no SDK): availability follows the key on the
  hosted endpoint, an overridden base_url runs keyless, freshness maps to
  its day buckets, highlights become the hit body, and the pool rotates.
- **Keenable** — raw httpx (no SDK) and keyless by default: the public
  endpoints carry an app name instead of a key, a configured key switches to
  the keyed paths, the config knobs map to the API's own fields (clamped to
  the ranges it accepts), and a 400 does not park the key.
- **Jina** — Reader is the whole API as a URL (`<base>/<url>`) and works
  keyless; Search needs a key, so the provider offers it only then and the
  web tool skips it otherwise. Its recency window is applied to the publish
  dates the hits report, because the endpoint takes no date parameter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from lumi.config import (
    ContextDevConfig,
    ExaConfig,
    FirecrawlConfig,
    JinaConfig,
    KeenableConfig,
    MCPConfig,
    TavilyConfig,
)
from lumi.tools.web.providers.base import Page, SearchHit, SearchResult, WebProvider
from lumi.tools.web.providers.contextdev_provider import ContextDevProvider
from lumi.tools.web.providers.exa_provider import ExaProvider
from lumi.tools.web.providers.firecrawl_provider import FirecrawlProvider
from lumi.tools.web.providers.jina_provider import JinaProvider
from lumi.tools.web.providers.keenable_provider import KeenableProvider
from lumi.tools.web.providers.mcp_provider import DiscoveredTool, MCPProvider
from lumi.tools.web.providers.tavily_provider import TavilyProvider

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


@dataclass
class FakeTavilyClient:
    """Mimics ``AsyncTavilyClient`` for search and extract."""

    search_reply: Any = None
    extract_reply: Any = None
    search_error: Exception | None = None
    extract_error: Exception | None = None
    search_calls: int = 0
    extract_calls: int = 0
    last_search_kwargs: dict[str, Any] | None = None
    last_api_key: str | None | object = "__unset__"  # distinguish None from unset

    async def search(self, *_args, **kwargs):
        self.search_calls += 1
        self.last_search_kwargs = kwargs
        if self.search_error:
            raise self.search_error
        return self.search_reply

    async def extract(self, *_args, **_kwargs):
        self.extract_calls += 1
        if self.extract_error:
            raise self.extract_error
        return self.extract_reply


@dataclass
class FakeFirecrawlClient:
    """Mimics ``AsyncFirecrawl`` for search and scrape."""

    search_reply: Any = None
    scrape_reply: Any = None
    search_error: Exception | None = None
    scrape_error: Exception | None = None
    search_calls: int = 0
    scrape_calls: int = 0
    last_search_kwargs: dict[str, Any] | None = None

    async def search(self, *_args, **kwargs):
        self.search_calls += 1
        self.last_search_kwargs = kwargs
        if self.search_error:
            raise self.search_error
        return self.search_reply

    async def scrape(self, *_args, **_kwargs):
        self.scrape_calls += 1
        if self.scrape_error:
            raise self.scrape_error
        return self.scrape_reply


# --------------------------------------------------------------------------- #
# Tavily — keyless
# --------------------------------------------------------------------------- #


def test_tavily_is_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider(TavilyConfig())
    ok, reason = provider.available()
    assert ok, reason


def test_tavily_keyless_uses_api_key_none(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider(TavilyConfig())
    fake = FakeTavilyClient(search_reply={"results": [], "answer": ""})

    captured: dict[str, Any] = {}
    original = provider._get_client

    def capture(key):
        if key is None and "keyless_args" not in captured:
            captured["keyless_args"] = True
        return original(key)

    provider._get_client = capture  # type: ignore[method-assign]
    provider._keyless_client = fake

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok is True
    assert fake.search_calls == 1


def test_tavily_keyless_does_not_rotate(monkeypatch) -> None:
    """With one keyless client, a failure is final — no rotation."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider(TavilyConfig())
    fake = FakeTavilyClient(search_error=RuntimeError("keyless 429"))
    provider._keyless_client = fake

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert fake.search_calls == 1
    assert not result.ok
    assert "keyless 429" in result.error


# --------------------------------------------------------------------------- #
# Tavily — multi-key
# --------------------------------------------------------------------------- #


def test_tavily_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92")
    provider = TavilyProvider(TavilyConfig())

    bad = FakeTavilyClient(search_error=RuntimeError("ratelimit"))
    good = FakeTavilyClient(search_reply={"results": [{"url": "https://ok", "title": "OK"}]})

    # Park in order: tvly-91 is bad, tvly-92 is good.
    provider._auth_clients = {"tvly-91": bad, "tvly-92": good}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok
    assert len(result.hits) == 1
    assert result.hits[0].url == "https://ok"
    assert bad.search_calls == 1
    assert good.search_calls == 1


def test_tavily_pool_returns_last_error_when_every_key_fails(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92")
    provider = TavilyProvider(TavilyConfig())

    bad1 = FakeTavilyClient(search_error=RuntimeError("e1"))
    bad2 = FakeTavilyClient(search_error=RuntimeError("e2"))
    provider._auth_clients = {"tvly-91": bad1, "tvly-92": bad2}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert not result.ok
    assert "e2" in result.error  # last error surfaces
    assert bad1.search_calls == 1
    assert bad2.search_calls == 1


def test_tavily_pool_skips_a_parked_key(monkeypatch) -> None:
    """After one key fails it is parked; the next call uses the surviving one."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92")
    provider = TavilyProvider(TavilyConfig())

    bad = FakeTavilyClient(search_error=RuntimeError("boom"))
    good = FakeTavilyClient(search_reply={"results": [{"url": "u", "title": "t"}]})
    provider._auth_clients = {"tvly-91": bad, "tvly-92": good}

    import asyncio

    asyncio.run(provider.search("hi", 5))
    asyncio.run(provider.search("hi", 5))

    # tvly-91 was tried once and parked; tvly-92 should answer every call after.
    assert bad.search_calls == 1
    assert good.search_calls == 2


def test_tavily_strategy_round_robin(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92, tvly-93")
    provider = TavilyProvider(TavilyConfig(key_strategy="round_robin"))

    replies = {"results": [{"url": "u", "title": "t"}]}
    clients = {k: FakeTavilyClient(search_reply=replies) for k in ("tvly-91", "tvly-92", "tvly-93")}
    provider._auth_clients = clients

    import asyncio

    # Four calls — with round_robin, the first key (tvly-91) is parked after
    # one failure only if we trigger a failure. Here every call succeeds, so
    # the cursor cycles through 91 → 92 → 93 → 91.
    keys_seen: list[str] = []
    original_pick = provider._pool.pick

    def spy(exclude=()):
        key = original_pick(exclude=exclude)
        if key is not None:
            keys_seen.append(key)
        return key

    provider._pool.pick = spy  # type: ignore[method-assign]

    for _ in range(4):
        asyncio.run(provider.search("hi", 5))

    assert keys_seen == ["tvly-91", "tvly-92", "tvly-93", "tvly-91"]


def test_tavily_strategy_unknown_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92")
    provider = TavilyProvider(TavilyConfig(key_strategy="telepathy"))
    assert provider._pool.strategy == "fallback"


# --------------------------------------------------------------------------- #
# Firecrawl — availability
# --------------------------------------------------------------------------- #


def test_firecrawl_is_not_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    provider = FirecrawlProvider(FirecrawlConfig())
    ok, reason = provider.available()
    assert not ok
    assert "FIRECRAWL_API_KEY" in reason


def test_firecrawl_is_available_with_a_key(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    ok, _ = provider.available()
    assert ok


def test_firecrawl_self_hosted_is_available_without_a_key(monkeypatch) -> None:
    """Firecrawl is open source: a self-hosted base_url may take no auth."""
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    provider = FirecrawlProvider(FirecrawlConfig(base_url="http://localhost:9/"))
    assert provider.available() == (True, "")


async def test_firecrawl_keyless_search_makes_one_attempt(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    provider = FirecrawlProvider(FirecrawlConfig(base_url="http://localhost:9/"))
    fake = FakeFirecrawlClient(
        search_reply={"web": [{"url": "https://ok", "title": "OK", "description": "d"}], "news": []}
    )
    provider._keyless = fake

    result = await provider.search("hi", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert fake.search_calls == 1
    assert not provider._clients  # the keyed pool was never touched


async def test_firecrawl_keyless_fetch_shapes_the_page(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    provider = FirecrawlProvider(FirecrawlConfig(base_url="http://localhost:9/"))
    good = FakeFirecrawlClient(scrape_reply={"metadata": {"title": "Hello"}, "markdown": "# body"})
    provider._keyless = good

    page = await provider.fetch("https://example.com")

    assert page.title == "Hello"
    assert "# body" in page.markdown
    assert good.scrape_calls == 1


async def test_firecrawl_keyless_failure_is_surfaced(monkeypatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    provider = FirecrawlProvider(FirecrawlConfig(base_url="http://localhost:9/"))
    provider._keyless = FakeFirecrawlClient(search_error=RuntimeError("connection refused"))

    result = await provider.search("hi", 5)

    assert not result.ok
    assert "connection refused" in result.error


# --------------------------------------------------------------------------- #
# Firecrawl — multi-key
# --------------------------------------------------------------------------- #


def test_firecrawl_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91, fc-92")
    provider = FirecrawlProvider(FirecrawlConfig())

    bad = FakeFirecrawlClient(search_error=RuntimeError("ratelimit"))
    good_reply = {
        "web": [{"url": "https://ok", "title": "OK", "description": "d"}],
        "news": [],
    }
    good = FakeFirecrawlClient(search_reply=good_reply)
    provider._clients = {"fc-91": bad, "fc-92": good}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok
    assert len(result.hits) == 1
    assert bad.search_calls == 1
    assert good.search_calls == 1


def test_firecrawl_fetch_rotates(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91, fc-92")
    provider = FirecrawlProvider(FirecrawlConfig())

    bad = FakeFirecrawlClient(scrape_error=RuntimeError("boom"))
    good_reply = {"metadata": {"title": "Hello"}, "markdown": "# body"}
    good = FakeFirecrawlClient(scrape_reply=good_reply)
    provider._clients = {"fc-91": bad, "fc-92": good}

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))
    assert page.title == "Hello"
    assert "# body" in page.markdown
    assert bad.scrape_calls == 1
    assert good.scrape_calls == 1


def test_firecrawl_strategy_unknown_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig(key_strategy="telepathy"))
    assert provider._pool.strategy == "fallback"


# --------------------------------------------------------------------------- #
# Exa — availability
# --------------------------------------------------------------------------- #


@dataclass
class FakeExaClient:
    """Mimics ``AsyncExa`` for search and get_contents."""

    search_reply: Any = None
    contents_reply: Any = None
    search_error: Exception | None = None
    contents_error: Exception | None = None
    search_calls: int = 0
    contents_calls: int = 0
    last_search_kwargs: dict[str, Any] | None = None

    async def search(self, *_args, **kwargs):
        self.search_calls += 1
        self.last_search_kwargs = kwargs
        if self.search_error:
            raise self.search_error
        return self.search_reply

    async def get_contents(self, *_args, **_kwargs):
        self.contents_calls += 1
        if self.contents_error:
            raise self.contents_error
        return self.contents_reply


def test_exa_is_not_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider = ExaProvider(ExaConfig())
    ok, reason = provider.available()
    assert not ok
    assert "EXA_API_KEY" in reason


def test_exa_is_available_with_a_key(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    assert provider.available()[0]


def test_exa_overridden_base_url_is_available_without_a_key(monkeypatch) -> None:
    """An overridden base_url is the owner's endpoint; the key is optional there."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider = ExaProvider(ExaConfig(base_url="http://localhost:9/"))
    assert provider.available() == (True, "")


async def test_exa_keyless_search_makes_one_attempt(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider = ExaProvider(ExaConfig(base_url="http://localhost:9/"))
    fake = FakeExaClient(search_reply={"results": [{"url": "https://ok", "text": "body"}]})
    provider._keyless = fake

    result = await provider.search("hi", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert fake.search_calls == 1
    assert not provider._clients  # the keyed pool was never touched


async def test_exa_keyless_failure_is_surfaced(monkeypatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    provider = ExaProvider(ExaConfig(base_url="http://localhost:9/"))
    provider._keyless = FakeExaClient(search_error=RuntimeError("connection refused"))

    result = await provider.search("hi", 5)

    assert not result.ok
    assert "connection refused" in result.error


# --------------------------------------------------------------------------- #
# Exa — search
# --------------------------------------------------------------------------- #


def test_exa_search_normalises_results(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {
        "results": [
            {"title": "One", "url": "https://one.test", "text": "full body", "score": 0.91},
            {"title": "Two", "url": "https://two.test", "highlights": ["h1", "h2"]},
        ]
    }
    provider._clients = {"exa-91": FakeExaClient(search_reply=reply)}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok
    assert [h.url for h in result.hits] == ["https://one.test", "https://two.test"]
    assert result.hits[0].content == "full body"
    # No text? Fall back to the highlights.
    assert result.hits[1].content == "h1 h2"
    assert result.hits[0].score == 0.91


def test_exa_search_forwards_type_and_category(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig(search_type="neural", category="news"))
    fake = FakeExaClient(search_reply={"results": []})
    provider._clients = {"exa-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 3))
    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("type") == "neural"
    assert fake.last_search_kwargs.get("category") == "news"
    assert fake.last_search_kwargs.get("num_results") == 3


def test_exa_search_omits_an_empty_category(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    fake = FakeExaClient(search_reply={"results": []})
    provider._clients = {"exa-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 3))
    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("category") is None


def test_exa_search_reports_an_empty_result_set(monkeypatch) -> None:
    """A successful search with no hits is the query's fault: ok, no error."""
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    provider._clients = {"exa-91": FakeExaClient(search_reply={"results": []})}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok
    assert result.hits == []


# --------------------------------------------------------------------------- #
# Exa — multi-key rotation and fetch
# --------------------------------------------------------------------------- #


def test_exa_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91, exa-92")
    provider = ExaProvider(ExaConfig())

    bad = FakeExaClient(search_error=RuntimeError("ratelimit"))
    good = FakeExaClient(search_reply={"results": [{"url": "https://ok", "title": "OK"}]})
    provider._clients = {"exa-91": bad, "exa-92": good}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.ok
    assert result.hits[0].url == "https://ok"
    assert bad.search_calls == 1
    assert good.search_calls == 1


def test_exa_fetch_rotates(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91, exa-92")
    provider = ExaProvider(ExaConfig())

    bad = FakeExaClient(contents_error=RuntimeError("boom"))
    good = FakeExaClient(
        contents_reply={"results": [{"url": "https://x", "title": "T", "text": "body"}]}
    )
    provider._clients = {"exa-91": bad, "exa-92": good}

    import asyncio

    page = asyncio.run(provider.fetch("https://x"))
    assert page.title == "T"
    assert page.markdown == "body"
    assert bad.contents_calls == 1
    assert good.contents_calls == 1


def test_exa_fetch_matches_the_url_not_just_the_first_result(monkeypatch) -> None:
    # A single result for one requested URL is the page itself, however its
    # address changed on the way (the redirect propagation above); several
    # results naming none of the requested URLs read as "nothing here".
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {"results": [{"url": "https://other", "text": "wrong page"}]}
    provider._clients = {"exa-91": FakeExaClient(contents_reply=reply)}

    import asyncio

    page = asyncio.run(provider.fetch("https://wanted"))
    assert page.markdown == "wrong page"
    assert page.url == "https://other"


def test_exa_strategy_unknown_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig(key_strategy="telepathy"))
    assert provider._pool.strategy == "fallback"


def test_exa_api_keys_returns_a_pool(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91, exa-92")
    cfg = ExaConfig()
    assert cfg.api_keys() == ["exa-91", "exa-92"]
    assert cfg.api_key() == "exa-91"


# --------------------------------------------------------------------------- #
# Shared config: api_keys()
# --------------------------------------------------------------------------- #


def test_tavily_api_keys_returns_a_pool(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91, tvly-92")
    cfg = TavilyConfig()
    assert cfg.api_keys() == ["tvly-91", "tvly-92"]
    assert cfg.api_key() == "tvly-91"


def test_firecrawl_api_keys_returns_a_pool(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91, fc-92")
    cfg = FirecrawlConfig()
    assert cfg.api_keys() == ["fc-91", "fc-92"]
    assert cfg.api_key() == "fc-91"


# --------------------------------------------------------------------------- #
# Top-up scrape: only one client is used per call (no rotation), to avoid
# serialising one call per key during the post-search body fill.
# --------------------------------------------------------------------------- #


def test_firecrawl_top_up_does_not_rotate(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91, fc-92")
    provider = FirecrawlProvider(FirecrawlConfig())

    # Make the search succeed with thin hits so the top-up runs.
    search_reply = {"web": [{"url": "https://x", "title": "X"}], "news": []}
    scrape_reply = {"metadata": {"title": "X"}, "markdown": "body"}
    c1 = FakeFirecrawlClient(search_reply=search_reply, scrape_reply=scrape_reply)
    c2 = FakeFirecrawlClient(search_reply=search_reply, scrape_reply=scrape_reply)
    provider._clients = {"fc-91": c1, "fc-92": c2}

    import asyncio

    asyncio.run(provider.search("x", 5))
    # Exactly one scrape — not two.
    assert c1.scrape_calls + c2.scrape_calls == 1


def test_firecrawl_top_up_skips_urls_the_tool_would_refuse(monkeypatch) -> None:
    """A search index can name a URL web.fetch would refuse; the provider must
    not scrape it behind the tool's back."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())

    search_reply = {
        "web": [{"url": "http://169.254.169.254/latest/meta-data", "title": "meta"}],
        "news": [],
    }
    scrape_reply = {"metadata": {"title": "meta"}, "markdown": "instance secret"}
    fake = FakeFirecrawlClient(search_reply=search_reply, scrape_reply=scrape_reply)
    provider._clients = {"fc-91": fake}

    import asyncio

    result = asyncio.run(provider.search("x", 5))

    assert result.ok
    assert fake.scrape_calls == 0
    assert result.hits[0].content == ""


# --------------------------------------------------------------------------- #
# Recency: every provider must turn a day window into its API's filter shape.
# --------------------------------------------------------------------------- #


def test_tavily_recency_maps_to_time_range(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91")
    provider = TavilyProvider(TavilyConfig())
    # A key is set, so the pool path runs: inject the fake per key.
    fake = FakeTavilyClient(search_reply={"results": [], "answer": ""})
    provider._auth_clients["tvly-91"] = fake

    import asyncio

    asyncio.run(provider.search("hi", 5, days=7))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("time_range") == "week"


def test_tavily_without_recency_sends_no_time_range(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91")
    provider = TavilyProvider(TavilyConfig())
    fake = FakeTavilyClient(search_reply={"results": [], "answer": ""})
    provider._auth_clients["tvly-91"] = fake

    import asyncio

    asyncio.run(provider.search("hi", 5))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("time_range") is None


def test_tavily_parses_published_date(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91")
    provider = TavilyProvider(TavilyConfig())
    reply = {
        "results": [{"url": "https://x", "title": "X", "published_date": "2026-02-01"}],
        "answer": "",
    }
    provider._auth_clients["tvly-91"] = FakeTavilyClient(search_reply=reply)

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.hits[0].published == "2026-02-01"


def test_firecrawl_recency_maps_to_tbs(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    fake = FakeFirecrawlClient(search_reply={"web": [], "news": []})
    provider._clients = {"fc-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 5, days=3))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("tbs") == "qdr:w"


def test_firecrawl_without_recency_sends_no_tbs(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    fake = FakeFirecrawlClient(search_reply={"web": [], "news": []})
    provider._clients = {"fc-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 5))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("tbs") is None


def test_firecrawl_long_recency_uses_an_explicit_cutoff(monkeypatch) -> None:
    """Past a year there is no coarse bucket; the ISO cutoff carries the window."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    fake = FakeFirecrawlClient(search_reply={"web": [], "news": []})
    provider._clients = {"fc-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 5, days=800))

    assert fake.last_search_kwargs is not None
    tbs = fake.last_search_kwargs.get("tbs") or ""
    assert tbs.startswith("cdr:1,cd_min:")


def test_firecrawl_parses_published_dates(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    reply = {
        "web": [{"url": "https://x", "title": "X", "publishedDate": "2026-03-10"}],
        "news": [],
    }
    provider._clients = {"fc-91": FakeFirecrawlClient(search_reply=reply)}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.hits[0].published == "2026-03-10"


def test_exa_recency_sets_start_published_date(monkeypatch) -> None:
    from lumi.tools.web.providers.base import recent_cutoff as _cutoff

    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    fake = FakeExaClient(search_reply={"results": []})
    provider._clients = {"exa-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 5, days=30))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("start_published_date") == _cutoff(30)


def test_exa_without_recency_omits_start_published_date(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    fake = FakeExaClient(search_reply={"results": []})
    provider._clients = {"exa-91": fake}

    import asyncio

    asyncio.run(provider.search("hi", 5))

    assert fake.last_search_kwargs is not None
    assert fake.last_search_kwargs.get("start_published_date") is None


def test_exa_parses_published_date(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {"results": [{"url": "https://x", "title": "X", "publishedDate": "2026-04-02"}]}
    provider._clients = {"exa-91": FakeExaClient(search_reply=reply)}

    import asyncio

    result = asyncio.run(provider.search("hi", 5))
    assert result.hits[0].published == "2026-04-02"


# --------------------------------------------------------------------------- #
# Redirect URL propagation (finding E / R3)
# --------------------------------------------------------------------------- #
# The web tool re-checks the URL a fetch ended on against the private-host
# rules, but only when the provider hands over that URL: every provider used
# to stamp Page.url with the *requested* address, which made the re-check dead
# code. These tests pin the propagation itself, per provider.


def test_firecrawl_fetch_propagates_the_final_url(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    reply = {
        "metadata": {"url": "http://169.254.169.254/latest/meta-data", "title": "T"},
        "markdown": "# secret",
    }
    provider._clients = {"fc-91": FakeFirecrawlClient(scrape_reply=reply)}

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "http://169.254.169.254/latest/meta-data"


def test_firecrawl_fetch_keeps_the_requested_url_without_one_in_metadata(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-91")
    provider = FirecrawlProvider(FirecrawlConfig())
    provider._clients = {
        "fc-91": FakeFirecrawlClient(
            scrape_reply={"metadata": {"title": "T"}, "markdown": "# body"}
        )
    }

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "https://example.com"


def test_exa_fetch_propagates_the_final_url(monkeypatch) -> None:
    """Exa answers under the address its fetch landed on; one result for one
    requested URL is unambiguous even when the addresses differ."""
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {"results": [{"url": "http://169.254.169.254/latest", "title": "T", "text": "secret"}]}
    provider._clients = {"exa-91": FakeExaClient(contents_reply=reply)}

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "http://169.254.169.254/latest"
    assert page.markdown == "secret"


def test_exa_fetch_still_matches_the_requested_url_when_given(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {
        "results": [
            {"url": "https://example.com", "text": "body"},
            {"url": "https://other.test", "text": "noise"},
        ]
    }
    provider._clients = {"exa-91": FakeExaClient(contents_reply=reply)}

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com/"))

    assert page.markdown == "body"


def test_exa_fetch_stays_empty_for_an_unmatched_multi_result_reply(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-91")
    provider = ExaProvider(ExaConfig())
    reply = {
        "results": [
            {"url": "https://a.test", "text": "one"},
            {"url": "https://b.test", "text": "two"},
        ]
    }
    provider._clients = {"exa-91": FakeExaClient(contents_reply=reply)}

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "https://example.com"
    assert page.markdown == "(no content returned)"


def test_tavily_extract_propagates_the_final_url(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-91")
    provider = TavilyProvider(TavilyConfig())
    provider._auth_clients = {
        "tvly-91": FakeTavilyClient(
            extract_reply={
                "results": [{"url": "http://169.254.169.254/latest", "raw_content": "s"}]
            }
        )
    }

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "http://169.254.169.254/latest"
    assert page.markdown == "s"


def test_mcp_fetch_propagates_the_result_url() -> None:
    """The MCP fetch path is the one that reaches the bot's own network, so
    a tavily-shaped server's answer URL must reach the tool layer."""
    provider = MCPProvider.__new__(MCPProvider)
    provider.name = "mcp"
    provider.config = MCPConfig()

    async def fake_pick(*_args, **_kwargs):
        return object(), DiscoveredTool(name="fetch", description="", properties={})

    async def fake_call(*_args, **_kwargs):
        import json

        return json.dumps(
            {"results": [{"url": "http://169.254.169.254/latest", "raw_content": "secret"}]}
        )

    provider._pick = fake_pick  # type: ignore[method-assign]
    provider._call = fake_call  # type: ignore[method-assign]

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "http://169.254.169.254/latest"
    assert page.markdown == "secret"


def test_mcp_fetch_fails_closed_without_a_result_url() -> None:
    """Content without a landing URL cannot pass the tool-layer SSRF re-check."""
    provider = MCPProvider.__new__(MCPProvider)
    provider.name = "mcp"
    provider.config = MCPConfig()

    async def fake_pick(*_args, **_kwargs):
        return object(), DiscoveredTool(name="fetch", description="", properties={})

    async def fake_call(*_args, **_kwargs):
        return "page fetched after an undisclosed redirect"

    provider._pick = fake_pick  # type: ignore[method-assign]
    provider._call = fake_call  # type: ignore[method-assign]

    import asyncio

    page = asyncio.run(provider.fetch("https://example.com"))

    assert page.url == "https://example.com"
    assert page.markdown.startswith("mcp fetch failed:")
    assert "without a final URL" in page.markdown


# --------------------------------------------------------------------------- #
# Context.dev — raw httpx, no SDK
# --------------------------------------------------------------------------- #


#: The real client class, saved before any test patches ``httpx.AsyncClient``
#: — install_http may run several times in one test, and each patch must
#: wrap the pristine class, not the previous test's factory.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def install_http(monkeypatch, respond) -> list[dict]:
    """Route the provider's httpx calls through a MockTransport; return the
    list the parsed JSON request bodies land in."""
    bodies: list[dict] = []

    def respond_and_record(request: httpx.Request) -> httpx.Response:
        if request.content:
            bodies.append(json.loads(request.content))
        return respond(request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(respond_and_record), **kwargs
        ),
    )
    return bodies


def search_response(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"results": results})


def a_hit(url: str, relevance: str = "high", highlights: list[str] | None = None) -> dict:
    return {
        "url": url,
        "title": f"title of {url}",
        "description": "a hit",
        "relevance": relevance,
        "highlights": {"highlights": highlights, "code": "SUCCESS"},
        "markdown": {"markdown": None, "code": "NOT_REQUESTED"},
    }


def test_contextdev_is_not_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    ok, reason = ContextDevProvider(ContextDevConfig()).available()
    assert not ok
    assert "CONTEXT_DEV_API_KEY" in reason


def test_contextdev_overridden_base_url_is_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    provider = ContextDevProvider(ContextDevConfig(base_url="http://localhost:9/v1"))
    assert provider.available() == (True, "")


def test_contextdev_imports_httpx_lazily() -> None:
    """This provider is in the web tool's fixed import graph: a module-level
    httpx import would turn a missing package into a startup crash for the
    whole bot instead of one provider switching itself off."""
    from lumi.tools.web.providers import contextdev_provider as provider_module

    assert "httpx" not in vars(provider_module)


def test_contextdev_is_unavailable_without_httpx(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    monkeypatch.setattr(builtins, "__import__", no_httpx)

    ok, reason = ContextDevProvider(ContextDevConfig()).available()

    assert not ok
    assert "httpx" in reason


async def test_contextdev_search_maps_results(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    bodies = install_http(
        monkeypatch,
        lambda request: search_response(
            [
                a_hit("https://a.example/1", "high", ["passage one", "passage two"]),
                a_hit("https://a.example/2", "low", None),
            ]
        ),
    )
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("rag retrieval", 5)

    assert result.ok
    first, second = result.hits
    assert (first.url, first.title, first.snippet) == (
        "https://a.example/1",
        "title of https://a.example/1",
        "a hit",
    )
    assert first.content == "passage one\n\npassage two"
    assert first.score == 1.0
    assert second.content == ""  # no highlights, markdown not requested
    assert second.score == 0.0

    (payload,) = bodies
    assert payload["query"] == "rag retrieval"
    assert payload["numResults"] == 5
    assert payload["highlightsOptions"] == {"enabled": True}
    assert "markdownOptions" not in payload  # off by default: credits
    assert "freshness" not in payload  # days=None


async def test_contextdev_maps_freshness_to_buckets(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    # A bucket must fully contain the window, so 400 days has no bucket and
    # the search goes out unfiltered rather than silently widened to a year.
    for days, expected in ((1, "last_24_hours"), (3, "last_week"), (31, "last_month"), (400, None)):
        bodies = install_http(monkeypatch, lambda request: search_response([]))
        provider = ContextDevProvider(ContextDevConfig())

        await provider.search("q", 5, days=days)

        if expected is None:
            assert "freshness" not in bodies[0], f"days={days}"
        else:
            assert bodies[0]["freshness"] == expected, f"days={days}"


async def test_contextdev_keyless_sends_no_auth_header(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    client_kwargs: list[dict] = []

    def factory(**kwargs):
        client_kwargs.append(kwargs)
        return _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(lambda request: search_response([])), **kwargs
        )

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    provider = ContextDevProvider(ContextDevConfig(base_url="http://localhost:9/v1"))

    result = await provider.search("q", 5)

    assert result.ok
    assert "Authorization" not in (client_kwargs[0].get("headers") or {})


async def test_contextdev_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(500, json={"error_code": "INTERNAL_ERROR"})
        return search_response([a_hit("https://a.example/1")])

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert state["n"] == 2


async def test_contextdev_fetch_scrapes_markdown(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    bodies = install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": True, "data": "# body"},
                "metadata": {"title": "Hello", "url": "https://landed.example/"},
            },
        ),
    )
    provider = ContextDevProvider(ContextDevConfig())

    page = await provider.fetch("https://example.com")

    assert page.title == "Hello"
    assert page.url == "https://landed.example/"  # the redirect the API followed
    assert page.markdown == "# body"
    assert bodies[0]["formats"] == {"markdown": True}


async def test_contextdev_fetch_reports_a_failed_output(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": False, "error_code": "WEBSITE_BLOCKED"},
                "metadata": {},
            },
        ),
    )
    provider = ContextDevProvider(ContextDevConfig())

    page = await provider.fetch("https://example.com")

    assert page.markdown.startswith("scrape failed:")
    assert "WEBSITE_BLOCKED" in page.markdown


async def test_contextdev_non_json_body_rotates_instead_of_crashing(monkeypatch) -> None:
    """A 200 with a non-JSON body is a failed attempt: the next key answers."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, text="<html>not json</html>")
        return search_response([a_hit("https://a.example/1")])

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert state["n"] == 2


async def test_contextdev_a_200_error_body_rotates_instead_of_reading_as_zero_hits(
    monkeypatch,
) -> None:
    """Some compatible servers answer a throttled key with HTTP 200 and
    ``{"error_code": "RATE_LIMITED"}`` — that is a failed attempt the pool
    rotates on, not a successful search with no hits."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json={"error_code": "RATE_LIMITED"})
        return search_response([a_hit("https://a.example/1")])

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert state["n"] == 2


async def test_contextdev_a_200_error_body_names_the_reason_when_every_key_fails(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    install_http(
        monkeypatch, lambda request: httpx.Response(200, json={"error_code": "RATE_LIMITED"})
    )
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert not result.ok
    assert "RATE_LIMITED" in result.error


async def test_contextdev_an_unexpected_exception_rotates_instead_of_crashing(monkeypatch) -> None:
    """Anything escaping the HTTP layer — a shape the parser did not expect,
    an SDK-version surprise — is a failed attempt, the way Tavily and Exa
    already treat it, not a provider crash that skips the remaining keys."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    install_http(monkeypatch, lambda request: search_response([a_hit("https://a.example/1")]))
    provider = ContextDevProvider(ContextDevConfig())
    real_post = provider._post
    calls = {"n": 0}

    async def flaky_post(key, path, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("unexpected shape")
        return await real_post(key, path, payload)

    monkeypatch.setattr(provider, "_post", flaky_post)

    result = await provider.search("q", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert calls["n"] == 2


async def test_contextdev_search_truncates_an_overlong_query(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91")
    bodies = install_http(monkeypatch, lambda request: search_response([]))
    provider = ContextDevProvider(ContextDevConfig())

    await provider.search("x" * 600, 5)

    assert bodies[0]["query"] == "x" * 500


async def test_contextdev_fetch_rotates_keys_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(500, json={"error_code": "INTERNAL_ERROR"})
        return httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": True, "data": "# body"},
                "metadata": {"title": "Hello", "url": "https://landed.example/"},
            },
        )

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    page = await provider.fetch("https://example.com")

    assert page.markdown == "# body"
    assert state["n"] == 2
    # An HTTP-level failure is the key's problem: it is parked for the cooldown.
    assert "cd-91" in provider._pool._parked_at


async def test_contextdev_fetch_rotation_reports_the_reason_when_every_key_fails(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": False, "error_code": "WEBSITE_BLOCKED"},
                "metadata": {},
            },
        ),
    )
    provider = ContextDevProvider(ContextDevConfig())

    page = await provider.fetch("https://example.com")

    assert page.markdown.startswith("scrape failed:")
    assert "WEBSITE_BLOCKED" in page.markdown
    # A refused scrape is the site's fault, not the keys': neither is parked.
    assert provider._pool._parked_at == {}


async def test_contextdev_keyless_fetch_failure_still_renders_the_error_page(monkeypatch) -> None:
    """Keyless mode has no pool to rotate on: one attempt, and its failure
    must still come back as the ``scrape failed:`` page the web tool
    recognizes, not as an exception."""
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    install_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": False, "error_code": "WEBSITE_BLOCKED"},
                "metadata": {},
            },
        ),
    )
    provider = ContextDevProvider(ContextDevConfig(base_url="http://localhost:9/v1"))

    page = await provider.fetch("https://example.com")

    assert page.markdown.startswith("scrape failed:")
    assert "WEBSITE_BLOCKED" in page.markdown


async def test_contextdev_a_400_does_not_park_the_key(monkeypatch) -> None:
    """A 400 is the request's fault, not the key's — the pool still rotates
    within the call, but the key stays fit for the next one. The same rule
    the LLM keypool applies, via its message-based ``is_retryable``."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(400, json={"error_code": "BAD_QUERY"})
        return search_response([a_hit("https://a.example/1")])

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert state["n"] == 2
    assert provider._pool._parked_at == {}


async def test_contextdev_a_non_object_json_body_is_a_failed_attempt(monkeypatch) -> None:
    """A 200 whose JSON parses but is not an object (a bare list) must not
    crash the provider with an AttributeError — it is a failed attempt the
    pool rotates on, like any other malformed answer."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=["not", "an", "object"])
        return search_response([a_hit("https://a.example/1")])

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert state["n"] == 2


async def test_contextdev_a_non_object_scrape_body_is_a_failed_attempt(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "cd-91, cd-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=["not", "an", "object"])
        return httpx.Response(
            200,
            json={
                "markdown": {"requested": True, "success": True, "data": "# body"},
                "metadata": {},
            },
        )

    install_http(monkeypatch, respond)
    provider = ContextDevProvider(ContextDevConfig())

    page = await provider.fetch("https://example.com")

    assert page.markdown == "# body"
    assert state["n"] == 2


# --------------------------------------------------------------------------- #
# Keenable — raw httpx, keyless by default
# --------------------------------------------------------------------------- #


def install_recording_http(monkeypatch, respond) -> list[httpx.Request]:
    """Route the provider's httpx calls through a MockTransport; return the
    requests so tests can check paths, headers and query parameters."""
    requests: list[httpx.Request] = []

    def respond_and_record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: _REAL_ASYNC_CLIENT(
            transport=httpx.MockTransport(respond_and_record), **kwargs
        ),
    )
    return requests


def test_keenable_is_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    ok, reason = KeenableProvider(KeenableConfig()).available()
    assert ok, reason


async def test_keenable_keyless_search_uses_the_public_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "query": "q",
                "mode": "pro",
                "results": [
                    {
                        "title": "One",
                        "url": "https://one.test",
                        "snippet": "page text",
                        "published_at": "2026-10-30T00:00:00Z",
                    }
                ],
            },
        ),
    )
    provider = KeenableProvider(KeenableConfig())

    result = await provider.search("q", 5, days=7)

    assert result.ok
    hit = result.hits[0]
    assert (hit.url, hit.title) == ("https://one.test", "One")
    assert hit.content == "page text"
    assert hit.published == "2026-10-30T00:00:00Z"
    (request,) = requests
    assert request.url.path == "/v1/search/public"
    assert request.headers["X-Keenable-Title"] == "Lumi"
    assert "X-API-Key" not in request.headers
    body = json.loads(request.content)
    assert body["max_results"] == 5
    assert body["mode"] == "pro"
    assert body["published_after"] == "7d"
    assert "snippet_max_length" not in body  # 0 leaves the server default


async def test_keenable_keyed_search_sends_the_key(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    requests = install_recording_http(
        monkeypatch, lambda request: httpx.Response(200, json={"results": []})
    )
    provider = KeenableProvider(KeenableConfig())

    result = await provider.search("q", 3)

    assert result.ok
    (request,) = requests
    assert request.url.path == "/v1/search"
    assert request.headers["X-API-Key"] == "keen-91"
    assert "published_after" not in json.loads(request.content)


async def test_keenable_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91, keen-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, json={"error": "rate_limited", "message": "slow down"})
        return httpx.Response(200, json={"results": [{"url": "https://ok", "title": "OK"}]})

    install_recording_http(monkeypatch, respond)
    provider = KeenableProvider(KeenableConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert len(result.hits) == 1
    assert state["n"] == 2


async def test_keenable_fetch_returns_markdown_and_the_final_url(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "url": "https://landed.test/page",
                "title": "Hello",
                "content": "# body",
            },
        ),
    )
    provider = KeenableProvider(KeenableConfig())

    page = await provider.fetch("https://example.com")

    assert page.title == "Hello"
    assert page.url == "https://landed.test/page"
    assert page.markdown == "# body"
    (request,) = requests
    assert request.url.path == "/v1/fetch"
    assert request.url.params["url"] == "https://example.com"


async def test_keenable_fetch_failure_renders_the_error_page(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            422, json={"error": "Unprocessable", "message": "content could not be extracted"}
        ),
    )
    provider = KeenableProvider(KeenableConfig())

    page = await provider.fetch("https://example.com")

    assert page.markdown.startswith("scrape failed:")
    assert "content could not be extracted" in page.markdown


def test_keenable_strategy_unknown_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    provider = KeenableProvider(KeenableConfig(key_strategy="telepathy"))
    assert provider._pool.strategy == "fallback"


# --------------------------------------------------------------------------- #
# Jina Reader — fetch only, keyless by default
# --------------------------------------------------------------------------- #


def test_jina_is_available_without_a_key(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    ok, reason = JinaProvider(JinaConfig()).available()
    assert ok, reason


async def test_jina_fetch_builds_the_reader_url(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "title": "Example Domain",
                    "url": "https://www.example.com/",
                    "content": "# Example\n\nbody",
                },
            },
        ),
    )
    page = await JinaProvider(JinaConfig()).fetch("https://www.example.com")

    assert page.title == "Example Domain"
    assert page.url == "https://www.example.com/"
    assert page.markdown == "# Example\n\nbody"
    (request,) = requests
    assert request.url.path == "/https://www.example.com"
    assert request.headers["Accept"] == "application/json"
    assert "Authorization" not in request.headers


async def test_jina_key_goes_out_as_a_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": {"content": "text"}}),
    )
    page = await JinaProvider(JinaConfig()).fetch("https://example.com")

    assert page.markdown == "text"
    (request,) = requests
    assert request.headers["Authorization"] == "Bearer jina-91"


async def test_jina_plain_text_body_is_still_the_page(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, text="# Markdown\n\ntext", headers={"content-type": "text/plain"}
        ),
    )
    page = await JinaProvider(JinaConfig()).fetch("https://example.com")
    assert "Markdown" in page.markdown


async def test_jina_http_error_becomes_a_scrape_failure(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(404, text="Target URL returned a 404"),
    )
    page = await JinaProvider(JinaConfig()).fetch("https://example.com/missing")

    assert page.markdown.startswith("scrape failed:")
    assert "HTTP 404" in page.markdown


async def test_jina_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91, jina-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"data": {"content": "text"}})

    install_recording_http(monkeypatch, respond)
    page = await JinaProvider(JinaConfig()).fetch("https://example.com")

    assert page.markdown == "text"
    assert state["n"] == 2


async def test_web_search_skips_jina_without_a_key(config, monkeypatch) -> None:
    """Keyless Jina can only fetch; it must not cost a failed search attempt."""
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    from lumi.tools.web.tool import WebTool

    class Stub(WebProvider):
        name = "stub"

        def available(self) -> tuple[bool, str]:
            return True, ""

        async def search(
            self, query: str, max_results: int, days: int | None = None
        ) -> SearchResult:
            return SearchResult(
                query=query,
                provider=self.name,
                hits=[SearchHit(title="T", url="https://hit.test")],
            )

        async def fetch(self, url: str) -> Page:
            return Page(url=url)

    tool = WebTool(config)
    tool.providers = {"jina": JinaProvider(JinaConfig()), "stub": Stub()}

    result = await tool._search({"query": "q", "max_results": 3})

    assert result.data["provider"] == "stub"
    assert "jina" not in result.text


def test_jina_search_needs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    provider = JinaProvider(JinaConfig())
    assert provider.supports_search is False

    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    assert JinaProvider(JinaConfig()).supports_search is True


async def test_jina_keyless_search_says_it_needs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    result = await JinaProvider(JinaConfig()).search("lumi", 5)

    assert not result.ok
    assert "JINA_API_KEY" in result.error


async def test_jina_search_maps_the_json_result_list(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {
                        "title": "First",
                        "url": "https://first.test",
                        "description": "the snippet",
                        "content": "# First\n\nfull text",
                        "publishedTime": "2026-01-02",
                    },
                    {"title": "Second", "url": "https://second.test"},
                    {"title": "No URL"},
                ],
            },
        ),
    )
    result = await JinaProvider(JinaConfig()).search("capital of france", 3)

    assert result.ok
    assert [hit.title for hit in result.hits] == ["First", "Second"]
    first, second = result.hits
    assert first.url == "https://first.test"
    assert first.snippet == "the snippet"
    assert first.content == "# First\n\nfull text"
    assert first.published == "2026-01-02"
    assert second.url == "https://second.test"

    (request,) = requests
    assert str(request.url).startswith("https://s.jina.ai/capital%20of%20france")
    assert request.url.params["num"] == "3"
    assert request.headers["Authorization"] == "Bearer jina-91"
    assert request.headers["Accept"] == "application/json"


async def test_jina_search_trims_to_the_requested_count(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {"title": "A", "url": "https://a.test"},
                    {"title": "B", "url": "https://b.test"},
                ]
            },
        ),
    )
    result = await JinaProvider(JinaConfig()).search("q", 1)
    assert [hit.url for hit in result.hits] == ["https://a.test"]


async def test_jina_search_pool_rotates_on_failure(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91, jina-92")
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"data": [{"title": "A", "url": "https://a.test"}]})

    install_recording_http(monkeypatch, respond)
    result = await JinaProvider(JinaConfig()).search("q", 5)

    assert result.ok
    assert seen == ["Bearer jina-91", "Bearer jina-92"]


async def test_jina_search_http_error_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(401, text="Authentication required"),
    )
    result = await JinaProvider(JinaConfig()).search("q", 5)

    assert not result.ok
    assert "HTTP 401" in result.error
    assert "Authentication required" in result.error


async def test_jina_search_without_a_json_list_is_a_failure(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(200, text="a markdown SERP"),
    )
    result = await JinaProvider(JinaConfig()).search("q", 5)

    assert not result.ok
    assert "no JSON result list" in result.error


async def test_web_search_falls_back_to_keyed_jina(config, monkeypatch) -> None:
    from lumi.tools.web.tool import WebTool

    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"data": [{"title": "Jina hit", "url": "https://jina.test"}]},
        ),
    )

    class Down(WebProvider):
        name = "down"

        def available(self) -> tuple[bool, str]:
            return True, ""

        async def search(
            self, query: str, max_results: int, days: int | None = None
        ) -> SearchResult:
            return SearchResult(query=query, provider=self.name, error="offline")

        async def fetch(self, url: str) -> Page:
            return Page(url=url)

    tool = WebTool(config)
    tool.providers = {"down": Down(), "jina": JinaProvider(JinaConfig())}

    result = await tool._search({"query": "q", "max_results": 3})

    assert result.data["provider"] == "jina"
    assert "jina.test" in result.text


def test_jina_status_says_what_reader_and_search_run_on(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    assert "search off" in JinaProvider(JinaConfig()).status()

    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    assert JinaProvider(JinaConfig()).status() == "reader + search (1 key)"

    monkeypatch.setenv("JINA_API_KEY", "jina-91, jina-92")
    assert JinaProvider(JinaConfig()).status() == "reader + search (2 keys)"

    assert JinaProvider(JinaConfig(enabled=False)).status().startswith("off —")


async def test_jina_search_applies_the_recency_window_to_dated_hits(monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    today = datetime.now(UTC).date()
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {
                        "title": "old",
                        "url": "https://old.test",
                        "publishedTime": (today - timedelta(days=30)).isoformat(),
                    },
                    {"title": "new", "url": "https://new.test", "publishedTime": today.isoformat()},
                    {"title": "undated", "url": "https://undated.test"},
                ]
            },
        ),
    )
    result = await JinaProvider(JinaConfig()).search("q", 5, days=7)

    assert [hit.url for hit in result.hits] == [
        "https://new.test",
        "https://undated.test",
    ]


async def test_jina_search_drops_duplicate_urls(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "data": [
                    {"title": "A", "url": "https://a.test"},
                    {"title": "A again", "url": "https://a.test"},
                ]
            },
        ),
    )
    result = await JinaProvider(JinaConfig()).search("q", 5)
    assert [hit.title for hit in result.hits] == ["A"]


def test_web_status_lists_jina_tier(config, monkeypatch) -> None:
    from lumi.tools.web.tool import WebTool

    monkeypatch.delenv("JINA_API_KEY", raising=False)
    lines = WebTool(config).status_lines()
    assert "jina: reader keyless · search off (JINA_API_KEY unset)" in lines


def test_web_status_gives_every_provider_one_line_in_order(config) -> None:
    from lumi.tools.web.tool import WebTool

    tool = WebTool(config)
    lines = tool.status_lines()
    # After the two header lines, one line per provider, same order as the chain.
    assert [line.split(":")[0] for line in lines[2:]] == list(tool.providers)
    # Providers without a status() fall back to available(): tavily is keyless
    # in the fixture, firecrawl has no key, so each lands on its own side.
    assert "tavily: on" in lines
    assert "firecrawl: off (FIRECRAWL_API_KEY is not set)" in lines


def test_web_status_only_lists_providers_in_the_order(config) -> None:
    from lumi.tools.web.tool import WebTool

    tool = WebTool(config)
    tool.providers.pop("keenable")
    tool.providers.pop("jina")
    lines = tool.status_lines()
    assert not any(line.startswith(("keenable:", "jina:")) for line in lines)


def test_keenable_status_names_the_tier(monkeypatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    assert KeenableProvider(KeenableConfig()).status() == (
        "keyless (public endpoints as 'Lumi') · mode pro"
    )

    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91, keen-92")
    assert KeenableProvider(KeenableConfig()).status() == "keyed (2 keys) · mode pro"

    assert KeenableProvider(KeenableConfig(enabled=False)).status().startswith("off —")


def test_keenable_and_jina_are_unavailable_without_httpx(monkeypatch) -> None:
    """Both sit in the web tool's fixed import graph: a missing httpx must
    switch the provider off, not crash startup for the whole bot."""
    import builtins

    from lumi.tools.web.providers import jina_provider, keenable_provider

    assert "httpx" not in vars(keenable_provider)
    assert "httpx" not in vars(jina_provider)

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)

    for provider in (KeenableProvider(KeenableConfig()), JinaProvider(JinaConfig())):
        ok, reason = provider.available()
        assert not ok
        assert "httpx" in reason


async def test_keenable_keyless_fetch_uses_the_public_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    requests = install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"url": "https://example.com/", "title": "Example", "content": "# body"},
        ),
    )

    page = await KeenableProvider(KeenableConfig()).fetch("https://example.com")

    assert (page.url, page.markdown) == ("https://example.com/", "# body")
    (request,) = requests
    assert request.url.path == "/v1/fetch/public"
    assert request.headers["X-Keenable-Title"] == "Lumi"
    assert "X-API-Key" not in request.headers


async def test_keenable_clamps_snippet_length_to_the_api_floor(monkeypatch) -> None:
    """The API answers 400 below 180 characters; the request is raised to the
    floor it accepts instead of failing the search over a config typo."""
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    requests = install_recording_http(
        monkeypatch, lambda request: httpx.Response(200, json={"results": []})
    )

    await KeenableProvider(KeenableConfig(snippet_max_length=100)).search("q", 5)

    assert json.loads(requests[0].content)["snippet_max_length"] == 180


async def test_keenable_a_400_does_not_park_the_key(monkeypatch) -> None:
    """A rejected parameter is the request's fault: the pool still rotates
    within the call, but the key stays fit for the next one."""
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91, keen-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(400, json={"message": 'Unknown parameter(s): "nope"'})
        return httpx.Response(200, json={"results": [{"url": "https://ok.test"}]})

    install_recording_http(monkeypatch, respond)
    provider = KeenableProvider(KeenableConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert state["n"] == 2
    assert provider._pool._parked_at == {}


async def test_keenable_a_non_object_json_body_is_a_failed_attempt(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91, keen-92")
    state = {"n": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, json=["not", "an", "object"])
        return httpx.Response(200, json={"results": [{"url": "https://ok.test"}]})

    install_recording_http(monkeypatch, respond)
    provider = KeenableProvider(KeenableConfig())

    result = await provider.search("q", 5)

    assert result.ok
    assert state["n"] == 2


async def test_keenable_names_the_reason_when_every_key_fails(monkeypatch) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "keen-91")
    install_recording_http(
        monkeypatch,
        lambda request: httpx.Response(402, json={"message": "insufficient_credits"}),
    )

    result = await KeenableProvider(KeenableConfig()).search("q", 5)

    assert not result.ok
    assert "HTTP 402" in result.error
    assert "insufficient_credits" in result.error
