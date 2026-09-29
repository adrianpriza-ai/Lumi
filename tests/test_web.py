"""The web tool: provider selection, fallback, and the MCP adapter.

Network calls are stubbed. What is under test is the part that actually breaks in
practice: which provider gets asked, what happens when one fails, and whether the
MCP adapter can drive a server whose tool names and parameter names it has never
seen before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.tools.base import ToolError
from lumi.tools.web import WebTool, build_providers
from lumi.tools.web.providers.base import (
    PER_HIT_CONTENT_CHARS,
    Page,
    SearchHit,
    SearchResult,
    WebProvider,
    recent_cutoff,
)
from lumi.tools.web.providers.mcp_provider import (
    DiscoveredTool,
    MCPProvider,
    _flatten,
    _parse_search_payload,
    expand_env,
)


class StubProvider(WebProvider):
    def __init__(self, name: str, *, ok: bool = True, reason: str = "",
                 search_result: SearchResult | None = None, page: Page | None = None,
                 raises: Exception | None = None) -> None:
        self.name = name
        self._ok = ok
        self._reason = reason
        self._search_result = search_result
        self._page = page
        self._raises = raises
        self.search_calls: list[tuple[str, int]] = []
        self.fetch_calls: list[str] = []

    def available(self) -> tuple[bool, str]:
        return self._ok, self._reason

    async def search(self, query: str, max_results: int, days: int | None = None) -> SearchResult:
        self.search_calls.append((query, max_results, days))
        if self._raises:
            raise self._raises
        return self._search_result or SearchResult(query=query, provider=self.name, hits=[
            SearchHit(title=f"{self.name} hit", url=f"https://{self.name}.test/1", snippet="body")
        ])

    async def fetch(self, url: str) -> Page:
        self.fetch_calls.append(url)
        if self._raises:
            raise self._raises
        return self._page or Page(url=url, title="stub", markdown="stub markdown")


def make_tool(config, *providers: WebProvider) -> WebTool:
    tool = WebTool.__new__(WebTool)
    tool.config = config
    tool.settings = config.tools.web
    tool.providers = {p.name: p for p in providers}
    return tool


# --------------------------------------------------------------------------- #
# provider selection and fallback
# --------------------------------------------------------------------------- #


async def test_uses_the_first_available_provider(config) -> None:
    first = StubProvider("tavily", ok=False, reason="no key")
    second = StubProvider("firecrawl")
    tool = make_tool(config, first, second)

    result = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert result.ok
    assert "firecrawl" in result.text
    assert first.search_calls == []
    assert second.search_calls == [("hi", 4, None)]


async def test_falls_through_when_a_provider_raises(config) -> None:
    first = StubProvider("tavily", raises=RuntimeError("502 from upstream"))
    second = StubProvider("firecrawl")
    tool = make_tool(config, first, second)

    result = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert result.ok
    assert "502 from upstream" in result.text  # the failure is reported, not hidden


async def test_falls_through_on_an_empty_result(config) -> None:
    first = StubProvider("tavily", search_result=SearchResult(query="hi", provider="tavily", hits=[]))
    second = StubProvider("firecrawl")
    tool = make_tool(config, first, second)

    await tool.invoke({"action": "search", "query": "hi"}, None)
    assert second.search_calls


async def test_all_providers_failing_is_reported_to_the_model(config) -> None:
    tool = make_tool(
        config,
        StubProvider("tavily", ok=False, reason="TAVILY_API_KEY is not set"),
        StubProvider("firecrawl", raises=RuntimeError("boom")),
    )
    result = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert not result.ok
    assert "Every configured web provider failed" in result.text
    assert "could not verify" in result.text  # tells the model how to behave


async def test_max_results_is_clamped(config) -> None:
    provider = StubProvider("tavily")
    tool = make_tool(config, provider)
    await tool.invoke({"action": "search", "query": "hi", "max_results": 999}, None)
    assert provider.search_calls == [("hi", 20, None)]


# --------------------------------------------------------------------------- #
# recency
# --------------------------------------------------------------------------- #


async def test_recency_is_passed_to_the_provider(config) -> None:
    provider = StubProvider("tavily")
    tool = make_tool(config, provider)

    await tool.invoke({"action": "search", "query": "hi", "recency": 7}, None)

    assert provider.search_calls == [("hi", 4, 7)]


async def test_recency_defaults_to_unfiltered(config) -> None:
    provider = StubProvider("tavily")
    tool = make_tool(config, provider)

    await tool.invoke({"action": "search", "query": "hi"}, None)

    assert provider.search_calls == [("hi", 4, None)]


async def test_recency_is_clamped(config) -> None:
    provider = StubProvider("tavily")
    tool = make_tool(config, provider)

    await tool.invoke({"action": "search", "query": "hi", "recency": 99_999}, None)
    await tool.invoke({"action": "search", "query": "hi", "recency": 0}, None)
    await tool.invoke({"action": "search", "query": "hi", "recency": "nonsense"}, None)

    assert provider.search_calls == [
        ("hi", 4, 3650),
        ("hi", 4, 1),
        ("hi", 4, None),
    ]


async def test_recency_is_reported_in_the_result(config) -> None:
    tool = make_tool(config, StubProvider("tavily"))

    filtered = await tool.invoke({"action": "search", "query": "hi", "recency": 30}, None)
    plain = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert filtered.data["recency_days"] == 30
    assert plain.data["recency_days"] is None
    assert "last 30 day(s) only" in filtered.text
    assert "day(s) only" not in plain.text


# --------------------------------------------------------------------------- #
# min_results: a thin result is topped up from the next provider
# --------------------------------------------------------------------------- #


def _hits(*urls: str) -> list[SearchHit]:
    return [SearchHit(title=f"h {u}", url=u, snippet="body") for u in urls]


async def test_thin_results_are_topped_up_from_the_next_provider(config) -> None:
    first = StubProvider(
        "firecrawl",
        search_result=SearchResult(query="hi", provider="firecrawl", hits=_hits("https://a")),
    )
    second = StubProvider(
        "tavily",
        search_result=SearchResult(
            query="hi", provider="tavily", hits=_hits("https://a", "https://b", "https://c")
        ),
    )
    tool = make_tool(config, first, second)

    result = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert result.ok
    assert result.data["hits"] == 3  # dup "https://a" dropped
    assert "firecrawl+tavily" in result.text
    assert first.search_calls and second.search_calls


async def test_results_at_the_floor_skip_the_next_provider(config) -> None:
    first = StubProvider(
        "firecrawl",
        search_result=SearchResult(
            query="hi", provider="firecrawl", hits=_hits("https://a", "https://b", "https://c")
        ),
    )
    second = StubProvider("tavily")
    tool = make_tool(config, first, second)

    await tool.invoke({"action": "search", "query": "hi"}, None)

    assert first.search_calls
    assert second.search_calls == []


async def test_min_results_argument_overrides_config(config) -> None:
    config.tools.web.min_results = 3
    first = StubProvider(
        "firecrawl",
        search_result=SearchResult(query="hi", provider="firecrawl", hits=_hits("https://a")),
    )
    second = StubProvider("tavily")
    tool = make_tool(config, first, second)

    await tool.invoke({"action": "search", "query": "hi", "min_results": 1}, None)

    assert second.search_calls == []


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


async def test_search_render_carries_the_search_date(config) -> None:
    tool = make_tool(config, StubProvider("tavily"))

    result = await tool.invoke({"action": "search", "query": "hi"}, None)

    assert "searched on" in result.text


def test_recent_cutoff_counts_back_from_today() -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC).date()
    assert recent_cutoff(0) == now.isoformat()
    assert recent_cutoff(10) == (now - timedelta(days=10)).isoformat()


def test_per_hit_budget_keeps_snippets_substantial() -> None:
    """A default-sized result set gets the per-hit floor, not a thin slice."""
    body = "x" * 5000
    result = SearchResult(
        query="q",
        provider="tavily",
        hits=[SearchHit(title=f"h{i}", url=f"https://{i}.test", content=body) for i in range(5)],
    )
    text = result.render(6000)

    # 6000 // 5 = 1200 would teaser-cut every hit; the floor (1400, minus
    # truncate's marker room) wins instead.
    assert text.count("x" * (PER_HIT_CONTENT_CHARS - 100)) == 5


def test_published_dates_render_when_present() -> None:
    result = SearchResult(
        query="q",
        provider="tavily",
        hits=[
            SearchHit(title="Dated", url="https://a.test", content="b", published="2026-01-05"),
            SearchHit(title="Undated", url="https://b.test", content="b"),
        ],
    )
    text = result.render(2000)

    assert "published 2026-01-05" in text
    assert text.count("published") == 1  # undated hits stay clean


async def test_search_requires_a_query(config) -> None:
    tool = make_tool(config, StubProvider("tavily"))
    with pytest.raises(ToolError, match="query is required"):
        await tool.invoke({"action": "search"}, None)


async def test_fetch_requires_a_url(config) -> None:
    tool = make_tool(config, StubProvider("tavily"))
    with pytest.raises(ToolError, match="url is required"):
        await tool.invoke({"action": "fetch"}, None)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "javascript:alert(1)"])
async def test_fetch_rejects_non_http_urls(config, url: str) -> None:
    tool = make_tool(config, StubProvider("tavily"))
    with pytest.raises(ToolError, match="http"):
        await tool.invoke({"action": "fetch", "url": url}, None)


async def test_fetch_works(config) -> None:
    provider = StubProvider("tavily", page=Page(url="https://x.test", title="T", markdown="# body"))
    tool = make_tool(config, provider)

    result = await tool.invoke({"action": "fetch", "url": "https://x.test"}, None)
    assert "# body" in result.text


async def test_fetch_skips_a_provider_that_failed_softly(config) -> None:
    first = StubProvider("tavily", page=Page(url="https://x.test", markdown="scrape failed: 403"))
    second = StubProvider("firecrawl", page=Page(url="https://x.test", markdown="real content"))
    tool = make_tool(config, first, second)

    result = await tool.invoke({"action": "fetch", "url": "https://x.test"}, None)
    assert "real content" in result.text
    assert second.fetch_calls == ["https://x.test"]


async def test_unknown_action(config) -> None:
    tool = make_tool(config, StubProvider("tavily"))
    with pytest.raises(ToolError, match="unknown action"):
        await tool.invoke({"action": "browse"}, None)


def test_availability_reports_why_every_provider_is_unusable(config) -> None:
    tool = make_tool(config, StubProvider("tavily", ok=False, reason="no key"))
    ok, reason = tool.available()
    assert not ok
    assert "no key" in reason


# --------------------------------------------------------------------------- #
# build_providers
# --------------------------------------------------------------------------- #


def test_build_providers_follows_the_configured_order(config) -> None:
    config.tools.web.provider_order = ["firecrawl", "tavily", "mcp"]
    assert list(build_providers(config)) == ["firecrawl", "tavily", "mcp"]


def test_build_providers_skips_unknown_names(config) -> None:
    config.tools.web.provider_order = ["tavily", "askjeeves"]
    assert list(build_providers(config)) == ["tavily"]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_search_rendering_numbers_the_sources() -> None:
    result = SearchResult(
        query="lumi",
        provider="tavily",
        answer="a short answer",
        hits=[
            SearchHit(title="First", url="https://a.test", content="body one"),
            SearchHit(title="Second", url="https://b.test", content="body two"),
        ],
    )
    text = result.render(1000)
    assert "[1] First" in text
    assert "[2] Second" in text
    assert "a short answer" in text
    assert text.index("[1]") < text.index("[2]")


def test_search_rendering_of_nothing() -> None:
    text = SearchResult(query="x", provider="tavily").render(500)
    assert "no results" in text


def test_page_rendering() -> None:
    text = Page(url="https://x.test", title="Title", markdown="# hi").render(500)
    assert "Title" in text and "# hi" in text


# --------------------------------------------------------------------------- #
# MCP: .mcp.json parsing
# --------------------------------------------------------------------------- #


def test_expand_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "abc123")
    assert expand_env("https://x/${MY_KEY}/v2") == "https://x/abc123/v2"


def test_expand_env_default_value(monkeypatch) -> None:
    monkeypatch.delenv("ABSENT", raising=False)
    assert expand_env("${ABSENT:-fallback}") == "fallback"


def test_expand_env_leaves_unknown_placeholders_empty(monkeypatch) -> None:
    monkeypatch.delenv("ABSENT", raising=False)
    assert expand_env("https://x/${ABSENT}/y") == "https://x//y"


def test_parse_remote_server(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "tavily": {"type": "remote", "url": "https://mcp.tavily.com/mcp/?key=K"}
        }
    }), encoding="utf-8")
    provider = MCPProvider(config.tools.web.mcp, config.root)
    servers = provider.servers()

    assert [s.name for s in servers] == ["tavily"]
    assert servers[0].is_local is False
    assert servers[0].url == "https://mcp.tavily.com/mcp/?key=K"


def test_parse_local_server_with_opencode_style_command(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"ctx7": {"type": "local", "command": ["npx", "-y", "@upstash/context7-mcp"]}}
    }), encoding="utf-8")
    server = MCPProvider(config.tools.web.mcp, config.root).servers()[0]

    assert server.is_local is True
    assert server.command == "npx"
    assert server.args == ["-y", "@upstash/context7-mcp"]


def test_parse_local_server_with_split_command_and_args(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"mem": {"command": "npx", "args": ["-y", "server-memory"]}}
    }), encoding="utf-8")
    server = MCPProvider(config.tools.web.mcp, config.root).servers()[0]
    assert server.command == "npx"
    assert server.args == ["-y", "server-memory"]


def test_servers_accepts_the_alternate_key(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({"servers": {"a": {"url": "https://a"}}}))
    assert [s.name for s in MCPProvider(config.tools.web.mcp, config.root).servers()] == ["a"]


def test_missing_mcp_file_is_not_fatal(config) -> None:
    provider = MCPProvider(config.tools.web.mcp, config.root)
    assert provider.servers() == []
    ok, reason = provider.available()
    assert not ok
    assert "no servers" in reason


def test_malformed_mcp_file_is_not_fatal(config) -> None:
    (config.root / ".mcp.json").write_text("{not json", encoding="utf-8")
    assert MCPProvider(config.tools.web.mcp, config.root).servers() == []


def test_unresolved_variables_are_reported(config: Path, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_KEY", raising=False)
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"tavily": {"url": "https://mcp.tavily.com/?key=${TAVILY_KEY}"}}
    }), encoding="utf-8")
    provider = MCPProvider(config.tools.web.mcp, config.root)

    ok, reason = provider.available()
    assert not ok
    assert "TAVILY_KEY" in reason
    assert provider.servers()[0].unresolved() == "TAVILY_KEY"


def test_resolved_variables_clear_the_check(config: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_KEY", "real")
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"tavily": {"url": "https://mcp.tavily.com/?key=${TAVILY_KEY}"}}
    }), encoding="utf-8")
    provider = MCPProvider(config.tools.web.mcp, config.root)
    assert provider.available()[0] is True
    assert provider.servers()[0].unresolved() is None


# --------------------------------------------------------------------------- #
# MCP: tool discovery and argument mapping
# --------------------------------------------------------------------------- #


def test_arg_for_prefers_the_named_parameter() -> None:
    tool = DiscoveredTool("tavily_search", "", {"query": {"type": "string"}, "topic": {"type": "string"}})
    assert tool.arg_for(("query", "q")) == "query"


def test_arg_for_falls_back_to_another_name() -> None:
    tool = DiscoveredTool("search", "", {"q": {"type": "string"}})
    assert tool.arg_for(("query", "q")) == "q"


def test_arg_for_uses_a_lone_string_parameter() -> None:
    tool = DiscoveredTool("mystery", "", {"terms": {"type": "string"}})
    assert tool.arg_for(("query", "q")) == "terms"


def test_arg_for_refuses_when_ambiguous() -> None:
    tool = DiscoveredTool("mystery", "", {"a": {"type": "string"}, "b": {"type": "string"}})
    assert tool.arg_for(("query", "q")) is None


def test_arg_for_ignores_known_knobs() -> None:
    tool = DiscoveredTool("mystery", "", {"max_results": {"type": "integer"}, "topic": {"type": "string"}})
    assert tool.arg_for(("query", "q")) is None


# --------------------------------------------------------------------------- #
# MCP: result flattening and normalisation
# --------------------------------------------------------------------------- #


class Block:
    def __init__(self, text: str | None = None, type: str = "text") -> None:
        self.text = text
        self.type = type


class Result:
    def __init__(self, content: list) -> None:
        self.content = content


def test_flatten_joins_text_blocks() -> None:
    assert _flatten(Result([Block("a"), Block("b")])) == "a\nb"


def test_flatten_handles_dict_blocks() -> None:
    assert _flatten({"content": [{"text": "hello", "type": "text"}]}) == "hello"


def test_flatten_describes_non_text_blocks() -> None:
    assert "image content omitted" in _flatten(Result([Block(None, "image")]))


def test_parse_tavily_shaped_payload() -> None:
    payload = json.dumps({
        "answer": "the answer",
        "results": [
            {"title": "One", "url": "https://one.test", "content": "snippet", "score": 0.9},
        ],
    })
    result = SearchResult(query="q", provider="mcp")
    _parse_search_payload(payload, result)

    assert result.answer == "the answer"
    assert result.hits[0].url == "https://one.test"
    assert result.hits[0].content == "snippet"


def test_parse_firecrawl_shaped_payload() -> None:
    payload = json.dumps({"data": [{"title": "One", "url": "https://one.test", "markdown": "body"}]})
    result = SearchResult(query="q", provider="mcp")
    _parse_search_payload(payload, result)
    assert result.hits[0].content == "body"


def test_parse_bare_list_payload() -> None:
    payload = json.dumps([{"url": "https://one.test", "description": "d"}])
    result = SearchResult(query="q", provider="mcp")
    _parse_search_payload(payload, result)
    assert result.hits[0].snippet == "d"


def test_parse_plain_text_payload() -> None:
    result = SearchResult(query="q", provider="mcp")
    _parse_search_payload("just some prose", result)
    assert result.hits[0].content == "just some prose"
    assert result.hits[0].url == ""


def test_parse_empty_payload() -> None:
    result = SearchResult(query="q", provider="mcp")
    _parse_search_payload("", result)
    assert result.hits == []


# --------------------------------------------------------------------------- #
# MCP: behaviour when no tool matches
# --------------------------------------------------------------------------- #


async def test_search_reports_when_no_tool_matches(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"empty": {"url": "https://x.test"}}
    }), encoding="utf-8")
    provider = MCPProvider(config.tools.web.mcp, config.root)
    # Bypass discovery: pretend every server failed to connect.
    provider._list_tools = lambda spec: _raise(RuntimeError("unreachable"))  # type: ignore[assignment]

    result = await provider.search("q", 5)
    assert not result.ok
    assert "no search tool found" in result.error


async def test_fetch_reports_when_no_tool_matches(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({
        "mcpServers": {"empty": {"url": "https://x.test"}}
    }), encoding="utf-8")
    provider = MCPProvider(config.tools.web.mcp, config.root)
    provider._list_tools = lambda spec: _raise(RuntimeError("unreachable"))  # type: ignore[assignment]

    page = await provider.fetch("https://x.test")
    assert "no fetch tool found" in page.markdown


async def _raise(exc: Exception):
    raise exc


# --------------------------------------------------------------------------- #
# end-to-end through the tool
# --------------------------------------------------------------------------- #


async def test_web_tool_picks_the_mcp_provider_when_configured(config) -> None:
    config.tools.web.provider_order = ["mcp"]
    tool = WebTool(config)
    assert "mcp" in tool.providers
    ok, reason = tool.available()
    # No .mcp.json in the fixture project, so it is correctly unavailable.
    assert not ok
    assert "no servers" in reason
