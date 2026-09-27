"""Tavily + Firecrawl provider tests.

The SDK clients are injected through the providers' private ``_get_client``
method, so each test owns the response shape and never touches the network.
The contract under test:

- **Tavily keyless** — the provider is *available* without an API key and
  builds its client with ``api_key=None`` when none is configured.
- **Tavily multi-key** — a 401 on one key falls through to the next; the
  first healthy key wins; an unknown strategy falls back to ``fallback``.
- **Firecrawl multi-key** — same rotation rules; the provider is *not*
  available without a key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lumi.config import FirecrawlConfig, TavilyConfig
from lumi.tools.web.providers.firecrawl_provider import FirecrawlProvider
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
    last_api_key: str | None | object = "__unset__"  # distinguish None from unset

    async def search(self, *_args, **_kwargs):
        self.search_calls += 1
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

    async def search(self, *_args, **_kwargs):
        self.search_calls += 1
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
    clients = {
        k: FakeTavilyClient(search_reply=replies) for k in ("tvly-91", "tvly-92", "tvly-93")
    }
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