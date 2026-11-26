"""The ``mcp`` tool's Code Mode surface: search, execute, approval, sandbox.

The servers are stubbed at the client boundary, so these tests pin the part
the model actually touches — the generated helper names, the approval prompt,
what a script can and cannot reach, and how output is fenced — without a
subprocess per server.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lumi.tools.base import NeedsApproval, ToolContext, ToolError
from lumi.tools.mcp.codemode import OUTPUT_MAX_CHARS, SEARCH_MAX_CHARS, MCPTool
from lumi.tools.mcp.sandbox import run_code
from lumi.tools.mcp.servers import DiscoveredTool, ServerSpec
from lumi.util.text import UNTRUSTED_BEGIN, UNTRUSTED_END


def spec(name: str = "github") -> ServerSpec:
    return ServerSpec(name=name, kind="remote", url="https://mcp.test")


def tool_of(*pairs, reasons: list[str] | None = None):
    """A tool with discovery pinned to *pairs*; returns (tool, calls)."""
    calls: list[tuple[str, str, dict]] = []

    def build(config) -> MCPTool:
        tool = MCPTool(config)
        tool._replies = {}  # type: ignore[attr-defined]
        tool._reasons = []  # type: ignore[attr-defined]
        tool._reply_func = None  # type: ignore[attr-defined]
        found = list(pairs)

        async def discover(*, reasons: list[str] | None = None):
            if reasons is not None and tool._reasons:  # type: ignore[attr-defined]
                reasons.extend(tool._reasons)  # type: ignore[attr-defined]
            return found

        async def call_tool(spec, discovered, arguments):
            calls.append((spec.name, discovered.name, arguments))
            if tool._reply_func is not None:  # type: ignore[attr-defined]
                return tool._reply_func(discovered.name, arguments)  # type: ignore[attr-defined]
            return tool._replies.get(discovered.name, "")  # type: ignore[attr-defined]

        tool.provider.discover = discover  # type: ignore[method-assign]
        tool.provider.call_tool = call_tool  # type: ignore[method-assign]
        return tool

    return build, calls


def issue_tool() -> DiscoveredTool:
    return DiscoveredTool(
        name="create_issue",
        description="Create an issue in a repository.",
        properties={
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
        },
        required=["owner", "repo", "title"],
    )


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


async def test_search_shows_generated_helpers_and_their_parameters(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search"}, ToolContext())

    assert result.ok
    assert (
        "codemode.github_create_issue(owner: string, repo: string, title: string, body: string = None)"
        in result.text
    )
    assert "Create an issue in a repository." in result.text
    assert result.data["tools"] == 1
    # Server-authored text is data: one fence, with the payload inside it.
    assert result.text.count(UNTRUSTED_BEGIN) == 1
    assert result.text.index(UNTRUSTED_BEGIN) < result.text.index("codemode.github_create_issue")
    assert result.text.index("codemode.github_create_issue") < result.text.index(UNTRUSTED_END)


async def test_search_filters_by_query_and_server(config) -> None:
    build, _ = tool_of(
        (spec("github"), issue_tool()),
        (
            spec("linear"),
            DiscoveredTool(name="list_tickets", description="List tickets", properties={}),
        ),
    )
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search", "query": "ticket"}, ToolContext())

    assert result.ok
    assert "linear_list_tickets" in result.text
    assert "github_create_issue" not in result.text

    by_server = await tool.invoke({"action": "search", "server": "github"}, ToolContext())
    assert "github_create_issue" in by_server.text
    assert "linear_list_tickets" not in by_server.text


async def test_search_with_no_match_says_how_to_widen_it(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search", "query": "zzz"}, ToolContext())

    assert not result.ok
    assert "No MCP tool matches 'zzz'" in result.text
    assert "search with no query" in result.text


async def test_search_reports_skipped_servers_as_data(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}
    tool._reasons = ["broken: unreachable (connection refused)"]

    result = await tool.invoke({"action": "search"}, ToolContext())

    assert "skipped: broken: unreachable" in result.text
    assert result.text.count(UNTRUSTED_BEGIN) == 1
    assert result.text.index(UNTRUSTED_BEGIN) < result.text.index("skipped: broken")
    assert result.text.index("skipped: broken") < result.text.index(UNTRUSTED_END)


async def test_search_stops_when_discovery_runs_past_the_budget(config) -> None:
    """Discovery has no session read timeout of its own: without the whole-op
    bound one slow spawn would hang the turn instead of reporting a failure."""
    config.tools.web.mcp.timeout_seconds = 0.02
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    async def slow_discover(*, reasons=None):
        await asyncio.sleep(0.5)
        return []

    tool.provider.discover = slow_discover  # type: ignore[method-assign]

    with pytest.raises(ToolError, match="did not answer within 0.02s"):
        await tool.invoke({"action": "search"}, ToolContext())


async def test_search_with_an_unknown_server_names_the_configured_ones(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"github": {"url": "https://mcp.test"}}}),
        encoding="utf-8",
    )
    tool = MCPTool(config)

    with pytest.raises(ToolError, match="no server named 'nope'") as caught:
        await tool.invoke({"action": "search", "server": "nope"}, ToolContext())

    assert "github" in str(caught.value)


async def test_search_reports_an_unset_variable_in_its_discovery_reason(
    config, monkeypatch
) -> None:
    monkeypatch.delenv("LUMI_TEST_MCP_UNSET", raising=False)
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"svc": {"url": "https://x.test/k=${LUMI_TEST_MCP_UNSET}"}}}),
        encoding="utf-8",
    )
    tool = MCPTool(config)

    result = await tool.invoke({"action": "search"}, ToolContext())

    assert not result.ok
    assert "needs ${LUMI_TEST_MCP_UNSET} set in the environment" in result.text
    assert result.text.index(UNTRUSTED_BEGIN) < result.text.index("needs ${")
    assert result.text.index("needs ${") < result.text.index(UNTRUSTED_END)


async def test_search_caps_a_huge_catalog(config) -> None:
    pairs = [
        (
            spec("srv"),
            DiscoveredTool(name=f"tool_{number}", description="d" * 400, properties={}),
        )
        for number in range(100)
    ]
    build, _ = tool_of(*pairs)
    tool = build(config)

    result = await tool.invoke({"action": "search"}, ToolContext())

    assert result.ok
    assert "truncated" in result.text
    assert len(result.text) < SEARCH_MAX_CHARS + 1000


# --------------------------------------------------------------------------- #
# execute
# --------------------------------------------------------------------------- #


async def test_execute_calls_a_helper_and_returns_printed_output(config) -> None:
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {
        "create_issue": json.dumps({"number": 7, "url": "https://github.test/issues/7"})
    }

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "issue = codemode.github_create_issue(\n"
                "    owner='acme', repo='widgets', title='Crash on save'\n"
                ")\n"
                "print(issue['number'], issue['url'])\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert "7 https://github.test/issues/7" in result.text
    assert calls == [
        (
            "github",
            "create_issue",
            {"owner": "acme", "repo": "widgets", "title": "Crash on save"},
        )
    ]
    assert result.data["calls"] == 1
    assert result.text.count(UNTRUSTED_BEGIN) == 1
    assert result.text.index(UNTRUSTED_BEGIN) < result.text.index("7 https://github.test/issues/7")
    assert result.text.index("7 https://github.test/issues/7") < result.text.index(UNTRUSTED_END)


async def test_execute_keeps_chained_work_inside_the_script(config) -> None:
    """The whole point of Code Mode: a loop and arithmetic in the sandbox,
    with only the final value in the conversation."""
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)

    def reply(name, arguments):
        return json.dumps({"number": len(arguments.get("title", ""))})

    tool._reply_func = reply  # type: ignore[attr-defined]

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "total = 0\n"
                "for title in ['a', 'bb', 'ccc']:\n"
                "    total += codemode.github_create_issue(\n"
                "        owner='o', repo='r', title=title\n"
                "    )['number']\n"
                "print('total', total)\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert "total 6" in result.text
    assert len(calls) == 3


async def test_execute_asks_the_owner_before_the_script_runs(config) -> None:
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {
                "action": "execute",
                "code": "print(codemode.github_create_issue(owner='o', repo='r', title='t'))",
            },
            ToolContext(),
        )

    assert "runs a Python script" in caught.value.reason
    assert "configured MCP servers" in caught.value.reason
    assert "codemode.github_create_issue" in caught.value.preview
    assert calls == []  # nothing connected or ran


async def test_execute_skips_the_prompt_when_ask_before_call_is_off(config) -> None:
    config.tools.web.mcp.ask_before_call = False
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {"create_issue": json.dumps({"number": 9})}

    result = await tool.invoke(
        {
            "action": "execute",
            "code": "print(codemode.github_create_issue(owner='o', repo='r', title='t')['number'])",
        },
        ToolContext(),  # deliberately unconfirmed
    )

    assert result.ok
    assert "9" in result.text
    assert len(calls) == 1


async def test_execute_does_not_prompt_when_every_server_is_refused(config) -> None:
    """A trust refusal is not something the owner can grant by confirming a
    script. The `/mcp approve` remedy must come before the prompt — and no
    discovery or spawn may run."""
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"hostile": {"url": "https://hostile.test"}}}),
        encoding="utf-8",
    )
    tool = MCPTool(config)

    async def never_discover(*, reasons=None):
        raise AssertionError("discovery must not run for a refused server")

    tool.provider.discover = never_discover  # type: ignore[method-assign]

    result = await tool.invoke(
        {"action": "execute", "code": "print(1)"},
        ToolContext(),  # unconfirmed
    )

    assert not result.ok
    assert "/mcp approve hostile" in result.text
    assert result.text.count(UNTRUSTED_BEGIN) == 1


async def test_execute_stops_when_discovery_runs_past_the_budget(config) -> None:
    config.tools.web.mcp.timeout_seconds = 0.02
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    async def slow_discover(*, reasons=None):
        await asyncio.sleep(0.5)
        return []

    tool.provider.discover = slow_discover  # type: ignore[method-assign]

    with pytest.raises(ToolError, match="did not answer within 0.02s"):
        await tool.invoke({"action": "execute", "code": "print(1)"}, ToolContext(confirmed=True))


async def test_execute_refuses_a_private_url_for_a_fetch_helper(config) -> None:
    """An approved fetch/scrape server runs inside the deployment, so a script
    must not be able to point it at the metadata endpoint or a private host."""
    fetch = DiscoveredTool(
        name="fetch", description="Fetch a URL", properties={"url": {"type": "string"}}
    )
    build, calls = tool_of((spec("web"), fetch))
    tool = build(config)
    tool._replies = {"fetch": "page"}

    script = (
        "try:\n"
        "    codemode.web_fetch(url='http://169.254.169.254/latest/meta-data/')\n"
        "except codemode.ToolError as exc:\n"
        "    print('blocked:', 'non-routable' in str(exc))\n"
        "print('public:', codemode.web_fetch(url='https://example.com/page'))\n"
    )
    result = await tool.invoke({"action": "execute", "code": script}, ToolContext(confirmed=True))

    assert result.ok
    assert "blocked: True" in result.text
    assert "public: page" in result.text
    assert calls == [("web", "fetch", {"url": "https://example.com/page"})]


async def test_execute_turns_a_server_error_into_a_catchable_toolerror(config) -> None:
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)

    class ErrorReply:
        content = [SimpleNamespace(text="Forbidden: missing scope")]
        isError = True

    async def _reuse(_spec, op, *, repeatable=True):
        return ErrorReply()

    tool.provider._reuse = _reuse  # type: ignore[method-assign]

    async def call_tool(server_spec, discovered, arguments):
        calls.append((server_spec.name, discovered.name, arguments))
        return await tool.provider._call(server_spec, discovered, arguments, 1.0)

    tool.provider.call_tool = call_tool  # type: ignore[method-assign]

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "try:\n"
                "    codemode.github_create_issue(owner='o', repo='r', title='t')\n"
                "except codemode.ToolError as exc:\n"
                "    print('caught:', 'missing scope' in str(exc))\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert "caught: True" in result.text
    assert len(calls) == 1


async def test_execute_caps_a_chatty_script(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    result = await tool.invoke(
        {"action": "execute", "code": "print('x' * 50000)"}, ToolContext(confirmed=True)
    )

    assert result.ok
    assert "truncated" in result.text
    assert len(result.text) < OUTPUT_MAX_CHARS + 1000


async def test_execute_caps_output_shown_with_a_script_error(config, monkeypatch) -> None:
    monkeypatch.setattr("lumi.tools.mcp.sandbox.CHILD_OUTPUT_LIMIT", 200)
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    result = await tool.invoke(
        {"action": "execute", "code": "print('x' * 5000)\n1 / 0"},
        ToolContext(confirmed=True),
    )

    assert not result.ok
    assert "[output truncated]" in result.text
    assert "ZeroDivisionError" in result.text


async def test_execute_with_no_available_tool_explains_why(config) -> None:
    build, _ = tool_of()
    tool = build(config)
    tool._replies = {}
    tool._reasons = ["hostile: awaiting approval (/mcp approve hostile)"]

    result = await tool.invoke(
        {"action": "execute", "code": "print(1)"}, ToolContext(confirmed=True)
    )

    assert not result.ok
    assert "No MCP tool is available" in result.text
    assert "/mcp approve hostile" in result.text


async def test_execute_surfaces_a_script_error_with_its_traceback(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke(
        {"action": "execute", "code": "print(1 / 0)"},
        ToolContext(confirmed=True),
    )

    assert not result.ok
    assert "ZeroDivisionError" in result.text
    assert "Output before it stopped" not in result.text  # nothing printed


async def test_execute_cannot_call_a_tool_it_did_not_discover(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "try:\n"
                "    codemode.call('github', 'delete_repo', {'repo': 'r'})\n"
                "except codemode.ToolError as exc:\n"
                "    print('refused:', 'no discovered tool' in str(exc))\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert "refused: True" in result.text


async def test_execute_enforces_the_call_budget(config) -> None:
    config.tools.web.mcp.max_code_calls = 1
    build, calls = tool_of((spec(), issue_tool()))
    tool = build(config)
    tool._replies = {"create_issue": json.dumps({"number": 1})}

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "for title in ['a', 'b']:\n"
                "    try:\n"
                "        codemode.github_create_issue(owner='o', repo='r', title=title)\n"
                "    except codemode.ToolError as exc:\n"
                "        print('stopped:', 'budget' in str(exc))\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert "stopped: True" in result.text
    assert len(calls) == 1


async def test_execute_requires_code(config) -> None:
    tool = MCPTool(config)
    with pytest.raises(ToolError, match="code"):
        await tool.invoke({"action": "execute"}, ToolContext(confirmed=True))


# --------------------------------------------------------------------------- #
# the sandbox boundary
# --------------------------------------------------------------------------- #


async def test_the_sandbox_has_no_filesystem_or_network() -> None:
    async def never(*_args):
        raise AssertionError("no tool call should happen")

    files = await run_code(
        code="open('/etc/passwd').read()", tools=[], on_call=never, timeout=10, max_calls=1
    )
    assert not files.ok
    assert "NameError" in files.error

    net = await run_code(
        code="import socket\nsocket.socket()",
        tools=[],
        on_call=never,
        timeout=10,
        max_calls=1,
    )
    assert not net.ok
    assert "not available in the MCP sandbox" in net.error


async def test_the_sandbox_returns_printed_output_only() -> None:
    async def echo(server, tool, arguments):
        return {"echo": arguments.get("text")}

    run = await run_code(
        code=(
            "value = codemode.echo(text='hi')\n"
            "print(value['echo'].upper())\n"
            "hidden = 'not printed'\n"
        ),
        tools=[{"helper": "echo", "server": "demo", "tool": "echo", "description": ""}],
        on_call=echo,
        timeout=10,
        max_calls=2,
    )

    assert run.ok
    assert run.output.strip() == "HI"
    assert "not printed" not in run.output


async def test_helper_names_are_sanitised_and_unique(config) -> None:
    build, _ = tool_of(
        (
            spec("my server"),
            DiscoveredTool(name="do-it", description="", properties={}),
        ),
        (
            spec("my_server"),
            DiscoveredTool(name="do_it", description="", properties={}),
        ),
    )
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search"}, ToolContext())

    assert result.ok
    assert "codemode.my_server_do_it(" in result.text
    assert "codemode.my_server_do_it_2(" in result.text


async def test_summary_line_calls_it_code_mode(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"github": {"url": "https://mcp.test"}}}),
        encoding="utf-8",
    )
    tool = MCPTool(config)
    assert "code mode" in tool.summary_line()
