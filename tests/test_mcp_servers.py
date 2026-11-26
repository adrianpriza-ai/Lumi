"""The MCP server client: config parsing, transports, sessions, pagination, trust.

The tool surface lives in tests/test_mcp_codemode.py; this module is the
machinery Code Mode stands on — two config files merged, stdio and streamable
HTTP transports, session reuse, and the owner-approval gate for project-file
entries.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from lumi.tools.mcp.servers import (
    DiscoveredTool,
    MCPServers,
    ServerSpec,
    _flatten,
    _read_timeout,
    _sdk_timeout_is_timedelta,
    _stdio_env,
    expand_env,
)


def provider_of(config) -> MCPServers:
    return MCPServers(config.tools.web.mcp, config.root)


# --------------------------------------------------------------------------- #
# MCP: .mcp.json parsing
# --------------------------------------------------------------------------- #


def test_sources_names_both_files(config) -> None:
    assert config.tools.web.mcp.user_config_file == ""  # pinned by conftest
    assert provider_of(config).sources == [".mcp.json"]
    config.tools.web.mcp.user_config_file = "~/.agents/mcp.json"
    assert provider_of(config).sources == [
        "~/.agents/mcp.json",
        ".mcp.json",
    ]


def test_expand_env(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "abc123")
    assert expand_env("https://x/${MY_KEY}/v2") == "https://x/abc123/v2"


def test_expand_env_default_value(monkeypatch) -> None:
    monkeypatch.delenv("ABSENT", raising=False)
    assert expand_env("${ABSENT:-fallback}") == "fallback"


def test_expand_env_leaves_unknown_placeholders_empty(monkeypatch) -> None:
    monkeypatch.delenv("ABSENT", raising=False)
    assert expand_env("https://x/${ABSENT}/y") == "https://x//y"


def test_local_mcp_servers_do_not_inherit_bot_secrets(monkeypatch) -> None:
    """The provider spawns a program the model talks to, so it gets the same
    allowlist as run_shell rather than the bot's whole environment. A secret
    the owner writes into the server's own env block still passes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    spec = ServerSpec(name="local", command="npx", env={"TAVILY_API_KEY": "explicit-owner-value"})

    env = _stdio_env(spec)

    assert "OPENAI_API_KEY" not in env
    assert env["LANG"] == "C.UTF-8"
    assert env["TAVILY_API_KEY"] == "explicit-owner-value"


def test_parse_remote_server(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tavily": {"type": "remote", "url": "https://mcp.tavily.com/mcp/?key=K"}
                }
            }
        ),
        encoding="utf-8",
    )
    provider = provider_of(config)
    servers = provider.servers()

    assert [s.name for s in servers] == ["tavily"]
    assert servers[0].is_local is False
    assert servers[0].url == "https://mcp.tavily.com/mcp/?key=K"


def test_parse_local_server_with_opencode_style_command(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ctx7": {"type": "local", "command": ["npx", "-y", "@upstash/context7-mcp"]}
                }
            }
        ),
        encoding="utf-8",
    )
    server = provider_of(config).servers()[0]

    assert server.is_local is True
    assert server.command == "npx"
    assert server.args == ["-y", "@upstash/context7-mcp"]


def test_parse_local_server_with_split_command_and_args(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"mem": {"command": "npx", "args": ["-y", "server-memory"]}}}),
        encoding="utf-8",
    )
    server = provider_of(config).servers()[0]
    assert server.command == "npx"
    assert server.args == ["-y", "server-memory"]


def test_parse_stdio_server_uses_the_specs_own_type_name(config) -> None:
    """The MCP spec and Claude Code write ``"stdio"`` where this project writes
    ``"local"``. Both must land on the spawned transport — a file copied from
    either world works without being rewritten.
    """
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"gh": {"type": "stdio", "command": "npx", "args": ["-y", "pkg"]}}}
        ),
        encoding="utf-8",
    )
    server = provider_of(config).servers()[0]

    assert server.kind == "local"
    assert server.is_local is True


def test_parse_stdio_server_without_a_type_is_still_local(config) -> None:
    """A bare ``command`` entry is the Claude Desktop shape; no ``type`` at all
    must not turn it into an HTTP server with no URL."""
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"mem": {"command": "python", "args": ["-m", "memory"]}}}),
        encoding="utf-8",
    )
    server = provider_of(config).servers()[0]

    assert server.is_local is True


def test_local_server_defaults_to_the_project_directory(config) -> None:
    """A relative ``args`` path must not mean "relative to whatever directory
    started the bot" — the child runs in the project root unless told otherwise.
    """
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"mem": {"command": "python", "args": ["server.py"]}}}),
        encoding="utf-8",
    )
    provider = provider_of(config)

    spec = provider.servers()[0]
    assert spec.cwd == ""
    assert provider._cwd_for(spec) == config.root


def test_local_server_cwd_resolves_against_the_project(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "dev": {"command": "python", "args": ["server.py"], "cwd": "servers/dev"}
                }
            }
        ),
        encoding="utf-8",
    )
    provider = provider_of(config)

    spec = provider.servers()[0]
    assert spec.cwd == "servers/dev"
    assert provider._cwd_for(spec) == (config.root / "servers/dev").resolve()


def test_local_server_cwd_expands_env_and_flags_it_when_unset(config) -> None:
    """``cwd`` joins every other field under ``${VAR}`` expansion — and an
    unset one is reported rather than silently resolved to a path that exists."""
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"dev": {"command": "python", "cwd": "${LUMI_TEST_UNSET_CWD}"}}}),
        encoding="utf-8",
    )
    spec = provider_of(config).servers()[0]

    assert spec.unresolved() == "LUMI_TEST_UNSET_CWD"


def test_servers_accepts_the_alternate_key(config) -> None:
    (config.root / ".mcp.json").write_text(json.dumps({"servers": {"a": {"url": "https://a"}}}))
    assert [s.name for s in provider_of(config).servers()] == ["a"]


def test_missing_mcp_file_is_not_fatal(config) -> None:
    provider = provider_of(config)
    assert provider.servers() == []
    ok, reason = provider.available()
    assert not ok
    assert "no servers" in reason


def test_malformed_mcp_file_is_not_fatal(config) -> None:
    (config.root / ".mcp.json").write_text("{not json", encoding="utf-8")
    assert provider_of(config).servers() == []


def test_non_object_mcp_file_is_not_fatal(config) -> None:
    (config.root / ".mcp.json").write_text("[]", encoding="utf-8")
    assert provider_of(config).servers() == []


def test_malformed_server_fields_are_tolerated(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "valid": {"url": "https://valid.test"},
                    "bad-args": {"url": "https://bad.test", "args": "not-a-list"},
                    "bad-env": {"url": "https://bad-env.test", "env": "not-a-map"},
                }
            }
        ),
        encoding="utf-8",
    )
    servers = provider_of(config).servers()
    assert [server.name for server in servers] == ["valid", "bad-args", "bad-env"]
    assert servers[1].args == []
    assert servers[2].env == {}


def test_unresolved_variables_are_reported(config: Path, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_KEY", raising=False)
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"tavily": {"url": "https://mcp.tavily.com/?key=${TAVILY_KEY}"}}}
        ),
        encoding="utf-8",
    )
    provider = provider_of(config)

    ok, reason = provider.available()
    assert not ok
    assert "TAVILY_KEY" in reason
    assert provider.servers()[0].unresolved() == "TAVILY_KEY"


def test_resolved_variables_clear_the_check(config: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_KEY", "real")
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"tavily": {"url": "https://mcp.tavily.com/?key=${TAVILY_KEY}"}}}
        ),
        encoding="utf-8",
    )
    provider = provider_of(config)
    assert provider.available()[0] is True
    assert provider.servers()[0].unresolved() is None


def test_default_env_values_are_not_reported_as_unresolved(config: Path, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_KEY", raising=False)
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"tavily": {"url": "https://mcp.test/?key=${TAVILY_KEY:-fallback}"}}}
        ),
        encoding="utf-8",
    )
    provider = provider_of(config)
    assert provider.servers()[0].url.endswith("key=fallback")
    assert provider.servers()[0].unresolved() is None


# --------------------------------------------------------------------------- #
# MCP: the user-level ~/.agents/mcp.json
# --------------------------------------------------------------------------- #


def _user_file(config, monkeypatch, tmp_path: Path, servers: dict) -> Path:
    """Point the user-level setting at a throwaway HOME and write it there."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config.tools.web.mcp.user_config_file = "~/.agents/mcp.json"
    agents = tmp_path / ".agents"
    agents.mkdir(exist_ok=True)
    path = agents / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def test_the_user_level_file_is_read_too(config, tmp_path, monkeypatch) -> None:
    _user_file(config, monkeypatch, tmp_path, {"shared": {"url": "https://shared.test"}})
    provider = provider_of(config)

    assert [s.name for s in provider.servers()] == ["shared"]
    assert provider.available()[0] is True  # no .mcp.json in the project at all


def test_the_project_file_wins_and_leads_over_the_user_file(config, tmp_path, monkeypatch) -> None:
    _user_file(
        config,
        monkeypatch,
        tmp_path,
        {
            "tavily": {"url": "https://user-file.test"},
            "only-user": {"url": "https://only-user.test"},
        },
    )
    (config.root / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tavily": {"url": "https://project-file.test"},
                    "only-project": {"url": "https://only-project.test"},
                }
            }
        ),
        encoding="utf-8",
    )

    servers = provider_of(config).servers()
    by_name = {s.name: s for s in servers}

    assert by_name["tavily"].url == "https://project-file.test"  # project wins
    assert [s.name for s in servers] == ["tavily", "only-project", "only-user"]  # and leads


def test_one_file_spelled_twice_is_read_once(config) -> None:
    config.tools.web.mcp.user_config_file = ".mcp.json"
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"a": {"url": "https://a.test"}}}), encoding="utf-8"
    )
    assert [s.name for s in provider_of(config).servers()] == ["a"]


def test_an_unreadable_user_file_is_not_fatal(config, tmp_path, monkeypatch) -> None:
    _user_file(config, monkeypatch, tmp_path, {})
    (tmp_path / ".agents" / "mcp.json").write_text("{not json", encoding="utf-8")
    provider = provider_of(config)

    assert provider.servers() == []
    assert provider.available()[0] is False


def test_an_empty_user_config_file_setting_turns_the_user_file_off(
    config, tmp_path, monkeypatch
) -> None:
    # The conftest fixture pins the setting to ""; a file at the default
    # ~/.agents/mcp.json path must then not be read.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".agents").mkdir()
    (tmp_path / ".agents" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"home": {"url": "https://home.test"}}}), encoding="utf-8"
    )
    assert provider_of(config).servers() == []


# --------------------------------------------------------------------------- #
# MCP: tool discovery and argument mapping
# --------------------------------------------------------------------------- #


def _discovered(properties: dict) -> DiscoveredTool:
    return DiscoveredTool(name="search", description="", properties=properties)


def test_arg_for_prefers_a_named_parameter_ignoring_case() -> None:
    """A server that spells the parameter ``Query`` still answers a ``query``
    candidate — and the first candidate wins when both are present."""
    tool = _discovered({"q": {"type": "string"}, "Query": {"type": "string"}})
    assert tool.arg_for(("query", "q")) == "Query"


def test_arg_for_falls_back_to_a_lone_string_parameter() -> None:
    tool = _discovered({"text": {"type": "string"}})
    assert tool.arg_for(("query", "q")) == "text"


def test_arg_for_accepts_a_property_without_a_declared_type() -> None:
    """Some servers omit ``type`` in their schema; that still reads as a
    possible text argument rather than disqualifying the only candidate."""
    tool = _discovered({"term": {}})
    assert tool.arg_for(("query",)) == "term"


def test_arg_for_refuses_when_several_parameters_could_fit() -> None:
    """Ambiguity is a refusal, not a coin flip — guessing would call a server
    with a value meant for another parameter."""
    tool = _discovered({"term": {"type": "string"}, "text": {"type": "string"}})
    assert tool.arg_for(("query",)) is None


def test_arg_for_ignores_known_knobs() -> None:
    """Pagination and formatting knobs are not a search term, even when they
    are the only string-shaped parameters the server advertises."""
    knobs = {
        name: {"type": "string"}
        for name in (
            "max_results",
            "limit",
            "topic",
            "depth",
            "format",
        )
    }
    tool = _discovered(dict(knobs))
    assert tool.arg_for(("query",)) is None

    tool.properties["query"] = {"type": "string"}
    assert tool.arg_for(("query",)) == "query"


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


def test_flatten_uses_structured_content_when_there_are_no_text_blocks() -> None:
    """Some servers answer with data only; an empty content list must not
    read as an empty answer."""

    class Structured:
        content: list = []
        structuredContent = {"issues": [{"id": "ABC-1"}]}

    assert json.loads(_flatten(Structured()))["issues"][0]["id"] == "ABC-1"


def test_flatten_uses_structured_content_on_the_dict_shape() -> None:
    assert json.loads(_flatten({"content": [], "structuredContent": {"count": 2}})) == {"count": 2}


def test_flatten_prefers_text_over_structured_content() -> None:
    class Both:
        content = [Block("readable")]
        structuredContent = {"raw": True}

    assert _flatten(Both()) == "readable"


# --------------------------------------------------------------------------- #
# MCP: session reuse
# --------------------------------------------------------------------------- #


class FakeSession:
    """Enough of a ClientSession for the provider's two call sites."""

    def __init__(self, search_payload: str, properties: dict | None = None) -> None:
        self._payload = search_payload
        self._properties = properties or {"query": {"type": "string"}}
        self.listed = 0
        self.called = 0
        self.call_arguments: list[dict] = []
        self.read_timeouts: list[timedelta | float | None] = []

    async def list_tools(self, cursor=None):
        self.listed += 1
        tool = type(
            "T",
            (),
            {
                "name": "tavily_search",
                "description": "",
                "input_schema": {"properties": self._properties},
            },
        )()
        return type("R", (), {"tools": [tool]})()

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.called += 1
        self.call_arguments.append(arguments)
        self.read_timeouts.append(read_timeout_seconds)
        return Result([Block(self._payload)])


def _mcp_project(config) -> str:
    """One remote search server, and the payload its calls return."""
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"tavily": {"url": "https://mcp.tavily.test/mcp/"}}}),
        encoding="utf-8",
    )
    return json.dumps({"results": [{"url": "https://one.test", "content": "x"}]})


async def test_mcp_lists_tools_across_every_page(config) -> None:
    """Servers may paginate list_tools; dropping nextCursor would hide tools
    that the `mcp` tool's list action is supposed to show."""
    _mcp_project(config)
    provider = provider_of(config)
    pages = {
        None: ("page-2", ["one", "two"]),
        "page-2": (None, ["three"]),
    }
    cursors: list[str | None] = []

    class Paged:
        async def list_tools(self, cursor=None):
            cursors.append(cursor)
            next_cursor, names = pages[cursor]
            tools = [
                type(
                    "T",
                    (),
                    {
                        "name": name,
                        "description": "",
                        "input_schema": {"properties": {}},
                    },
                )()
                for name in names
            ]
            return type("R", (), {"tools": tools, "nextCursor": next_cursor})()

    async def fake_open(stack, spec):
        return Paged()

    provider._open = fake_open  # type: ignore[method-assign]

    tools = await provider._list_tools(provider.servers()[0])

    assert [t.name for t in tools] == ["one", "two", "three"]
    assert cursors == [None, "page-2"]
    await provider.aclose()


async def test_mcp_stops_when_a_server_repeats_a_cursor(config) -> None:
    _mcp_project(config)
    provider = provider_of(config)
    cursors: list[str | None] = []

    class Stuck:
        async def list_tools(self, cursor=None):
            cursors.append(cursor)
            tool = type(
                "T",
                (),
                {
                    "name": "same",
                    "description": "",
                    "input_schema": {"properties": {}},
                },
            )()
            return type("R", (), {"tools": [tool], "nextCursor": "again"})()

    async def fake_open(stack, spec):
        return Stuck()

    provider._open = fake_open  # type: ignore[method-assign]

    tools = await provider._list_tools(provider.servers()[0])

    assert [t.name for t in tools] == ["same", "same"]
    assert cursors == [None, "again"]  # stops rather than paging forever
    await provider.aclose()


async def test_mcp_stops_after_the_page_cap(config) -> None:
    from lumi.tools.mcp.servers import LIST_PAGES_MAX

    _mcp_project(config)
    provider = provider_of(config)
    cursors: list[str | None] = []

    class Endless:
        async def list_tools(self, cursor=None):
            cursors.append(cursor)
            step = 0 if cursor is None else int(cursor.removeprefix("page-"))
            tool = type(
                "T",
                (),
                {
                    "name": f"t{step}",
                    "description": "",
                    "input_schema": {"properties": {}},
                },
            )()
            return type("R", (), {"tools": [tool], "nextCursor": f"page-{step + 1}"})()

    async def fake_open(stack, spec):
        return Endless()

    provider._open = fake_open  # type: ignore[method-assign]

    tools = await provider._list_tools(provider.servers()[0])

    assert len(cursors) == LIST_PAGES_MAX
    assert len(tools) == LIST_PAGES_MAX
    await provider.aclose()


# --------------------------------------------------------------------------- #
# sessions: reuse, recovery, shutdown
# --------------------------------------------------------------------------- #


async def test_discovery_and_calls_reuse_one_session(config) -> None:
    payload = _mcp_project(config)
    provider = provider_of(config)
    session = FakeSession(payload)
    opened: list[str] = []

    async def fake_open(stack, spec):
        opened.append(spec.name)
        return session

    provider._open = fake_open  # type: ignore[method-assign]
    spec = provider.servers()[0]
    tool = (await provider._list_tools(spec))[0]

    await provider.call_tool(spec, tool, {"query": "one"})
    await provider.call_tool(spec, tool, {"query": "two"})

    assert opened == ["tavily"]  # one spawn for discovery and both calls
    assert session.listed == 1  # the tool catalog stays cached
    assert session.called == 2
    # The SDK reversed read_timeout_seconds' type between 1.x and 2.x. The
    # session stubs accept either, so only an explicit shape check catches a
    # regression that would fail against a real server.
    expected = _read_timeout(config.tools.web.mcp.timeout_seconds)
    assert session.read_timeouts == [expected, expected]
    assert isinstance(expected, timedelta if _sdk_timeout_is_timedelta() else float)
    await provider.aclose()


async def test_a_dead_session_during_discovery_is_replaced_once(config, monkeypatch) -> None:
    import lumi.tools.mcp.servers as servers_module

    monkeypatch.setattr(servers_module, "DISCOVERY_RETRY_SECONDS", 0)
    payload = _mcp_project(config)
    provider = provider_of(config)
    opened: list[str] = []

    class Flaky(FakeSession):
        async def list_tools(self, cursor=None):
            if len(opened) == 1:
                raise RuntimeError("connection lost")
            return await super().list_tools(cursor=cursor)

    async def fake_open(stack, spec):
        opened.append(spec.name)
        return Flaky(payload)

    provider._open = fake_open  # type: ignore[method-assign]

    tools = await provider._list_tools(provider.servers()[0])

    assert [t.name for t in tools] == ["tavily_search"]
    assert opened == ["tavily", "tavily"]  # discovery is safe to re-send
    await provider.aclose()


async def test_a_dead_transport_during_a_call_is_not_re_sent(config) -> None:
    """A call may already have run when the wire dies, so it fails instead of
    being re-sent; the dead session is dropped so the next call recovers."""
    payload = _mcp_project(config)
    provider = provider_of(config)
    opened: list[str] = []

    class OnceDead(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            raise RuntimeError("connection lost")

    async def fake_open(stack, spec):
        opened.append(spec.name)
        return OnceDead(payload)

    provider._open = fake_open  # type: ignore[method-assign]
    spec = provider.servers()[0]
    tool = (await provider._list_tools(spec))[0]

    with pytest.raises(RuntimeError):
        await provider.call_tool(spec, tool, {})

    assert opened == ["tavily"]
    assert spec.name not in provider._sessions
    await provider.aclose()


async def test_a_timeout_is_not_retried(config) -> None:
    payload = _mcp_project(config)
    provider = provider_of(config)
    opened: list[str] = []

    class Slow(FakeSession):
        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            raise TimeoutError("read timed out")

    async def fake_open(stack, spec):
        opened.append(spec.name)
        return Slow(payload)

    provider._open = fake_open  # type: ignore[method-assign]
    spec = provider.servers()[0]
    tool = (await provider._list_tools(spec))[0]

    with pytest.raises(TimeoutError):
        await provider.call_tool(spec, tool, {})

    assert opened == ["tavily"]
    await provider.aclose()


async def test_aclose_releases_every_cached_session(config) -> None:
    payload = _mcp_project(config)
    provider = provider_of(config)
    closed: list[str] = []

    async def fake_open(stack, spec):
        async def close():
            closed.append(spec.name)

        stack.aclose = close  # stand in for the transport's own shutdown
        return FakeSession(payload)

    provider._open = fake_open  # type: ignore[method-assign]
    spec = provider.servers()[0]
    tool = (await provider._list_tools(spec))[0]
    await provider.call_tool(spec, tool, {})
    assert closed == []

    await provider.aclose()

    assert closed == ["tavily"]
    assert provider._sessions == {}


async def test_the_tool_cache_is_bypassed_when_the_config_changes(config) -> None:
    """The registry (and this client) lives across /reload, so an edited
    .mcp.json must invalidate discovery by mtime, not by the TTL alone."""
    import os
    from types import SimpleNamespace

    _mcp_project(config)
    provider = provider_of(config)
    calls = {"n": 0}

    class Session:
        async def list_tools(self, cursor=None, **kwargs):
            calls["n"] += 1
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name=f"tool_{calls['n']}",
                        description="",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]
            )

    async def _reuse(spec, op, *, repeatable=True):
        return await op(Session())

    provider._reuse = _reuse  # type: ignore[method-assign]
    spec = provider.servers()[0]

    first = await provider._list_tools(spec)
    assert [t.name for t in first] == ["tool_1"]
    assert [t.name for t in await provider._list_tools(spec)] == ["tool_1"]
    assert calls["n"] == 1

    path = provider.config_path
    mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(mtime + 1_000_000_000,) * 2)

    second = await provider._list_tools(spec)
    assert [t.name for t in second] == ["tool_2"]
    assert calls["n"] == 2


async def test_editing_the_config_closes_pinned_sessions(config) -> None:
    """Sessions opened against the old file must not be respawned after it
    changed — a renamed ``command`` would keep launching the old process."""
    import os

    _mcp_project(config)
    provider = provider_of(config)
    provider._tool_cache["tavily"] = (0.0, [])
    closed: list[str] = []

    class Stack:
        async def aclose(self) -> None:
            closed.append("tavily")

    provider._sessions["tavily"] = (Stack(), object())

    await provider._expire_stale_caches()
    path = provider.config_path
    mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(mtime + 1_000_000_000,) * 2)

    await provider._expire_stale_caches()

    assert closed == ["tavily"]
    assert "tavily" not in provider._tool_cache
