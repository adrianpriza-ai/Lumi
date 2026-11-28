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
from lumi.tools.mcp.codemode import (
    OUTPUT_MAX_CHARS,
    PREVIEW_MAX_CHARS,
    SEARCH_MAX_CHARS,
    MCPTool,
    _url_targets,
)
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


async def test_search_falls_back_to_any_term_match(config) -> None:
    build, _ = tool_of(
        (spec("github"), issue_tool()),
        (
            spec("linear"),
            DiscoveredTool(name="list_tickets", description="List tickets", properties={}),
        ),
    )
    tool = build(config)
    tool._replies = {}

    # Neither tool has both terms, so AND would return an empty catalog.
    result = await tool.invoke({"action": "search", "query": "tickets issue"}, ToolContext())

    assert result.ok
    assert "linear_list_tickets" in result.text
    assert "github_create_issue" in result.text
    assert "no tool matched every term" in result.text


async def test_search_ranks_a_name_match_above_a_description_match(config) -> None:
    build, _ = tool_of(
        (
            spec("notes"),
            DiscoveredTool(name="list_notes", description="Read the issue tracker.", properties={}),
        ),
        (spec("github"), issue_tool()),
    )
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search", "query": "issue"}, ToolContext())

    assert result.ok
    # `create_issue` has the word in its name; `list_notes` only mentions it.
    assert result.text.index("github_create_issue") < result.text.index("notes_list_notes")


async def test_search_limit_caps_the_catalog_and_says_how_many_matched(config) -> None:
    build, _ = tool_of(
        (spec("github"), issue_tool()),
        (
            spec("linear"),
            DiscoveredTool(name="list_tickets", description="List tickets", properties={}),
        ),
        (
            spec("sentry"),
            DiscoveredTool(name="list_alerts", description="List alerts", properties={}),
        ),
    )
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search", "limit": 2}, ToolContext())

    assert result.ok
    assert result.data["tools"] == 2
    assert result.data["matched"] == 3
    assert "2 of 3 MCP tool(s)" in result.text
    # No query means no ranking: the note must not call discovery order relevance.
    assert "showing the first 2 in discovery order" in result.text
    assert "sentry_list_alerts" not in result.text

    ranked = await tool.invoke({"action": "search", "query": "list", "limit": 1}, ToolContext())
    assert ranked.ok
    assert "showing the first 1 by relevance" in ranked.text


async def test_search_helper_names_stay_stable_under_a_filtered_ranking(config) -> None:
    build, _ = tool_of(
        (spec("a_b"), DiscoveredTool(name="c", description="plain", properties={})),
        (spec("a"), DiscoveredTool(name="b_c", description="special", properties={})),
    )
    tool = build(config)
    tool._replies = {}

    result = await tool.invoke({"action": "search", "query": "special"}, ToolContext())

    assert result.ok
    # Both bases sanitise to `a_b_c`; discovery order gives the second one the
    # `_2` suffix, and a filtered, re-ranked listing must not rename it.
    assert "codemode.a_b_c_2(" in result.text
    assert "codemode.a_b_c(" not in result.text


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
# describe
# --------------------------------------------------------------------------- #


def rich_tool() -> DiscoveredTool:
    return DiscoveredTool(
        name="create_issue",
        description=(
            "Create an issue in a repository.\n\n"
            "Pass `labels` as an array of label names; the server creates any "
            "that do not exist yet."
        ),
        properties={
            "owner": {"type": "string", "description": "Repository owner."},
            "repo": {"type": "string"},
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "high"]},
            "labels": {"type": "array", "items": {"type": "string"}},
        },
        required=["owner", "repo", "title"],
        schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "high"]},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["owner", "repo", "title"],
            "additionalProperties": False,
        },
    )


async def test_describe_returns_the_full_description_and_schema(config) -> None:
    build, _ = tool_of((spec(), rich_tool()))
    tool = build(config)

    result = await tool.invoke({"action": "describe", "tool": "github_create_issue"}, ToolContext())

    assert result.ok
    # The whole description survives — search keeps only its first line.
    assert "the server creates any" in result.text
    assert "input schema:" in result.text
    assert '"enum"' in result.text and '"low"' in result.text
    assert '"items"' in result.text
    # Top-level schema keys survive too: nothing is re-rendered lossily.
    assert '"additionalProperties": false' in result.text
    assert result.data["helper"] == "github_create_issue"
    assert result.text.count(UNTRUSTED_BEGIN) == 1


async def test_describe_accepts_the_helper_as_search_prints_it(config) -> None:
    """`search` shows `` `codemode.github_create_issue(...)` ``; the model may
    pass that spelling back verbatim."""
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    result = await tool.invoke(
        {"action": "describe", "tool": "`codemode.github_create_issue(owner='o')`"},
        ToolContext(),
    )

    assert result.ok
    assert result.data["helper"] == "github_create_issue"


async def test_describe_accepts_the_server_dot_tool_form(config) -> None:
    build, _ = tool_of((spec(), rich_tool()))
    tool = build(config)

    result = await tool.invoke({"action": "describe", "tool": "github.create_issue"}, ToolContext())

    assert result.ok
    assert "codemode.github_create_issue" in result.text


async def test_describe_rebuilds_a_schema_for_tools_without_one_stored(config) -> None:
    """A DiscoveredTool built by hand has no raw ``schema``; describe still
    shows the object schema its properties and required fields imply."""
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    result = await tool.invoke({"action": "describe", "tool": "github_create_issue"}, ToolContext())

    assert result.ok
    assert '"owner"' in result.text
    assert '"required"' in result.text


async def test_describe_names_search_when_the_tool_is_unknown(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    with pytest.raises(ToolError, match="no discovered tool named 'nope'"):
        await tool.invoke({"action": "describe", "tool": "nope"}, ToolContext())


async def test_describe_requires_a_tool_name(config) -> None:
    tool = MCPTool(config)
    with pytest.raises(ToolError, match="tool"):
        await tool.invoke({"action": "describe"}, ToolContext())


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


async def test_execute_reports_only_the_servers_it_actually_called(config) -> None:
    tickets = DiscoveredTool(name="list_tickets", description="List tickets", properties={})
    build, _ = tool_of((spec("github"), issue_tool()), (spec("linear"), tickets))
    tool = build(config)
    tool._replies = {"create_issue": "{}"}

    result = await tool.invoke(
        {
            "action": "execute",
            "code": "codemode.github_create_issue(owner='o', repo='r', title='t')\n",
        },
        ToolContext(confirmed=True),
    )

    assert result.ok, result.text
    assert result.data["servers"] == ["github"]

    failed = await tool.invoke(
        {"action": "execute", "code": "1 / 0\n"},
        ToolContext(confirmed=True),
    )
    assert not failed.ok
    assert failed.data["servers"] == []


async def test_execute_keeps_host_exception_text_out_of_the_script(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    def boom(name, arguments):
        raise OSError("/home/owner/.config/lumi/token")

    tool._reply_func = boom

    result = await tool.invoke(
        {
            "action": "execute",
            "code": (
                "try:\n"
                "    codemode.github_create_issue(owner='o', repo='r', title='t')\n"
                "except codemode.ToolError as exc:\n"
                "    print(type(exc).__name__, exc)\n"
            ),
        },
        ToolContext(confirmed=True),
    )

    assert result.ok, result.text
    assert "OSError" in result.text
    assert "the tool call failed" in result.text
    assert "/home/owner" not in result.text


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


async def test_approval_preview_reaches_a_call_past_the_old_cutoff(config) -> None:
    """The preview is a structural summary, not the first 2000 characters: a
    call padded out of sight used to be approved without ever being shown."""
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    padding = "# " + "padding " * 600 + "\n"
    code = (
        padding
        + "codemode.github_create_issue(owner='acme', repo='widgets', title='Crash on save')"
    )

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke({"action": "execute", "code": code}, ToolContext())

    preview = caught.value.preview
    assert (
        "codemode.github_create_issue(owner='acme', repo='widgets', title='Crash on save')"
        in preview
    )
    assert "padding" not in preview  # a summary, not the script
    assert "1 MCP call site(s)" in preview


async def test_approval_preview_lists_calls_in_order_and_counts_what_it_omits(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    body = "\n".join(
        f"codemode.github_create_issue(owner='acme', repo='widgets', title='issue {index}')"
        for index in range(30)
    )
    code = "x = 1\nif x:\n    print(x)\n" + body

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke({"action": "execute", "code": code}, ToolContext())

    preview = caught.value.preview
    assert "30 MCP call site(s)" in preview
    assert "title='issue 0'" in preview and "title='issue 1'" in preview
    assert preview.index("title='issue 0'") < preview.index("title='issue 1'")
    assert "more call site(s), not shown" in preview
    assert "1 if statement(s)" in preview
    assert len(preview) <= PREVIEW_MAX_CHARS


async def test_approval_preview_renders_a_call_by_name(config) -> None:
    """`codemode.call(server, tool, {...})` bypasses the helpers, so it must be
    listed just as plainly."""
    fetch = DiscoveredTool(name="fetch", description="", properties={})
    build, _ = tool_of((spec("web"), fetch))
    tool = build(config)

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {
                "action": "execute",
                "code": "codemode.call('web', 'fetch', {'url': 'https://example.com'})",
            },
            ToolContext(),
        )

    assert "codemode.call('web', 'fetch', …)" in caught.value.preview


async def test_approval_preview_follows_an_alias_of_codemode(config) -> None:
    """`cm = codemode; cm.delete_repo()` used to read as "no MCP call".

    The card told the owner the script *only computes and prints* while it
    went on to call a tool, which is the one thing the summary exists to
    prevent. The documented `except codemode.ToolError` must not look like a
    hidden call either.
    """
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    code = (
        "cm = codemode\n"
        "try:\n"
        "    cm.github_create_issue(owner='acme', repo='widgets', title='Crash')\n"
        "except codemode.ToolError as exc:\n"
        "    print('failed', exc)\n"
    )

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke({"action": "execute", "code": code}, ToolContext())

    preview = caught.value.preview
    assert "1 MCP call site(s)" in preview
    assert "codemode.github_create_issue(owner='acme', repo='widgets', title='Crash')" in preview
    assert "only computes and prints" not in preview
    assert "not listed above" not in preview  # ToolError is not a hidden call


async def test_approval_preview_names_a_literal_getattr_call(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {
                "action": "execute",
                "code": "getattr(codemode, 'github_create_issue')(owner='acme', repo='w')",
            },
            ToolContext(),
        )

    assert "codemode.github_create_issue(owner='acme', repo='w')" in caught.value.preview


async def test_approval_preview_does_not_claim_a_computed_call_is_absent(config) -> None:
    """When the API is reached by a name the summary cannot follow, say so.

    The all-clear is reserved for scripts that really only compute and print:
    a computed `getattr` or a script that passes `codemode` into a function
    reports that the API is reached indirectly instead.
    """
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)
    code = "name = 'github_create_issue'\ngetattr(codemode, name)(owner='acme')\n"

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke({"action": "execute", "code": code}, ToolContext())

    preview = caught.value.preview
    assert "reached indirectly" in preview
    assert "only computes and prints" not in preview


async def test_approval_preview_does_not_claim_a_bare_vars_call_is_absent(config) -> None:
    """`vars()` returns the script namespace, which holds `codemode`.

    `vars()['codemode'].delete_repo(...)` reaches a tool with no `codemode`
    name in the source, so an all-clear would be false; `vars(obj)` inspects a
    named object and stays uncounted.
    """
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {
                "action": "execute",
                "code": "vars()['codemode'].github_create_issue(owner='acme', repo='w')\n",
            },
            ToolContext(),
        )

    preview = caught.value.preview
    assert "reached indirectly" in preview
    assert "only computes and prints" not in preview

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {"action": "execute", "code": "t = {}\nprint(vars(t))\n"},
            ToolContext(),
        )

    assert "only computes and prints" in caught.value.preview


async def test_approval_preview_of_an_unparseable_script_says_so(config) -> None:
    build, _ = tool_of((spec(), issue_tool()))
    tool = build(config)

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke(
            {"action": "execute", "code": "codemode.github_create_issue(owner='o', repo="},
            ToolContext(),
        )

    assert "does not parse" in caught.value.preview

    with pytest.raises(NeedsApproval) as caught:
        await tool.invoke({"action": "execute", "code": "print(1)\x00"}, ToolContext())

    # Which exception the interpreter raises for a null byte varies by
    # version, so only the report itself is pinned, not the branch.
    assert "parse" in caught.value.preview
    assert "null bytes" in caught.value.preview


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
    fetching = ServerSpec(name="web", kind="remote", url="https://mcp.test", fetches=True)
    build, calls = tool_of((fetching, fetch))
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


async def test_execute_guards_every_tool_on_a_server_declared_to_fetch(config) -> None:
    """The guard is a property of the server, not of the tool name: `get_page`
    never matched the old fetch|scrape|extract|url regex, and a URL embedded in
    a larger argument still reaches the fetcher."""
    page = DiscoveredTool(
        name="get_page", description="Read a page", properties={"target": {"type": "string"}}
    )
    fetching = ServerSpec(name="browser", kind="remote", url="https://mcp.test", fetches=True)
    build, calls = tool_of((fetching, page))
    tool = build(config)
    tool._replies = {"get_page": "page"}

    script = (
        "try:\n"
        "    codemode.browser_get_page(target='open http://169.254.169.254/latest/meta-data/')\n"
        "except codemode.ToolError as exc:\n"
        "    print('blocked:', 'non-routable' in str(exc))\n"
    )
    result = await tool.invoke({"action": "execute", "code": script}, ToolContext(confirmed=True))

    assert result.ok
    assert "blocked: True" in result.text
    assert calls == []


async def test_execute_does_not_guard_a_server_declared_not_to_fetch(config) -> None:
    """A tool *called* `fetch` on a server whose capability says it does not
    read URLs gets no SSRF check: URL-shaped strings there are data."""
    fetch = DiscoveredTool(
        name="fetch", description="Fetch a record", properties={"body": {"type": "string"}}
    )
    storing = ServerSpec(name="notes", kind="remote", url="https://mcp.test", fetches=False)
    build, calls = tool_of((storing, fetch))
    tool = build(config)
    tool._replies = {"fetch": "ok"}

    result = await tool.invoke(
        {
            "action": "execute",
            "code": "print(codemode.notes_fetch(body='see http://192.168.1.5/x'))",
        },
        ToolContext(confirmed=True),
    )

    assert result.ok
    assert calls == [("notes", "fetch", {"body": "see http://192.168.1.5/x"})]


def test_fetch_capability_defaults_to_the_transport() -> None:
    """MCP declares no fetch capability, so an entry that does not say is read
    from its transport: a local server runs beside the metadata endpoint, a
    remote one fetches from its own network. A declared value wins either way,
    and an endpoint that is itself loopback counts as local."""
    local = ServerSpec.from_json("local", {"type": "local", "command": "uvx mcp-server-fetch"})
    remote = ServerSpec.from_json("remote", {"url": "https://mcp.test"})
    local_opt_out = ServerSpec.from_json("off", {"type": "local", "command": "x", "fetches": False})
    remote_opt_in = ServerSpec.from_json("on", {"url": "https://mcp.test", "fetches": True})
    gateway = ServerSpec.from_json("gateway", {"url": "http://127.0.0.1:3000/mcp"})

    assert local.fetches_urls and not remote.fetches_urls
    assert not local_opt_out.fetches_urls and remote_opt_in.fetches_urls
    assert gateway.fetches_urls

    # The relaxing declaration is shown where an owner approves an entry.
    from lumi.bot import _mcp_transport

    assert "URL arguments unchecked" in _mcp_transport(local_opt_out)
    assert "URL arguments unchecked" not in _mcp_transport(remote_opt_in)


def test_url_targets_finds_urls_embedded_in_larger_strings() -> None:
    arguments = {
        "body": "see https://169.254.169.254/latest/meta-data/ first; also file:///etc/passwd",
        "path": "/var/log/app.log",
        "nested": [["a//b", "//evil.test/x", "https://169.254.169.254/latest/meta-data/"]],
    }

    assert _url_targets(arguments) == [
        "https://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "//evil.test/x",
    ]


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


async def test_helper_suffix_cannot_shadow_a_literal_base_name(config) -> None:
    """A generated suffix must not collide with another tool's real base name.

    Bases ``srv_x_2``, ``srv_x``, ``srv_x`` used to produce helpers
    ``srv_x_2``, ``srv_x``, ``srv_x_2``: the last `setattr` in the bootstrap
    won, so the first tool was unreachable and search advertised one name for
    two different tools.
    """
    pairs = [
        (spec("srv"), DiscoveredTool(name="x_2", description="first", properties={})),
        (spec("srv"), DiscoveredTool(name="x", description="second", properties={})),
        (spec("srv"), DiscoveredTool(name="x", description="third", properties={})),
    ]
    build, _ = tool_of(*pairs)
    tool = build(config)

    helpers = [item["helper"] for item in tool._api(pairs)]
    assert helpers == ["srv_x_2", "srv_x", "srv_x_3"]

    result = await tool.invoke({"action": "search"}, ToolContext())
    assert result.ok
    assert result.text.count("codemode.srv_x_2(") == 1
    assert result.text.count("codemode.srv_x(") == 1
    assert result.text.count("codemode.srv_x_3(") == 1


async def test_summary_line_calls_it_code_mode(config) -> None:
    (config.root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"github": {"url": "https://mcp.test"}}}),
        encoding="utf-8",
    )
    tool = MCPTool(config)
    assert "code mode" in tool.summary_line()
