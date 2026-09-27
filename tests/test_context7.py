"""The ``context7`` tool: availability, action dispatch, key rotation, rendering.

HTTP is stubbed at the ``httpx.AsyncClient`` boundary so the tests run offline.
What is under test is what actually breaks in practice: which action gets
called, what happens when a key is rejected, and that the system prompt drops
the tool cleanly when no key is set.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from lumi.config import Context7Config
from lumi.tools.base import ToolError
from lumi.tools.context7 import Context7Client, Context7Tool
from lumi.tools.registry import ToolRegistry


def ctx() -> Any:
    from lumi.tools.base import ToolContext

    return ToolContext()


def _full_config(**overrides: Any):
    """A minimal ``Config`` carrying only the ``tools.context7`` block.

    The ``Context7Tool`` constructor expects a top-level ``Config`` (the same
    shape :func:`lumi.tools.build_registry` passes). Tests reach in with
    ``config.tools.context7`` to tweak the inner dataclass without poking
    sibling sections.
    """

    ctx7 = Context7Config(**overrides)
    tools = type("ToolsStub", (), {"context7": ctx7})()
    return type("ConfigStub", (), {"tools": tools})()


# --------------------------------------------------------------------------- #
# Stub transport — feed it (status, body) tuples and it returns them in order.
# --------------------------------------------------------------------------- #


class StubTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[tuple[int, dict[str, Any] | str]]) -> None:
        self.responses = list(responses)
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError("more requests than scripted responses")
        status, body = self.responses.pop(0)
        if isinstance(body, (dict, list)):
            content = json.dumps(body).encode("utf-8")
            headers = {"content-type": "application/json"}
        else:
            content = body.encode("utf-8")
            headers = {"content-type": "text/plain"}
        return httpx.Response(status, content=content, headers=headers, request=request)


def transport(*responses: tuple[int, Any]) -> StubTransport:
    return StubTransport(list(responses))


@asynccontextmanager
async def _stub_client(
    client: Context7Client, *responses: tuple[int, Any]
):
    """Replace the client's httpx stack with the stub transport.

    The real ``_client_for`` is synchronous (``httpx.AsyncClient()`` construction
    is sync, only its methods are async), so the stub matches that — the async
    part is just the request itself, which the test driver awaits. We also
    build a single ``AsyncClient`` once and reuse it across keys, so ``httpx``
    has no connection pool to flush on the way out.
    """
    stub = transport(*responses)
    real = client._clients

    def _factory(_key: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=stub,
            headers={"Authorization": "Bearer test"},
        )

    cleanup_client = _factory("__cleanup__")
    client._clients = {}
    client._client_for = _factory  # type: ignore[method-assign]
    try:
        yield stub
    finally:
        client._clients = real
        await cleanup_client.aclose()


# --------------------------------------------------------------------------- #
# Config layer
# --------------------------------------------------------------------------- #


def test_api_key_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    assert Context7Config().api_key() is None


def test_api_keys_returns_a_pool(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    keys = Context7Config().api_keys()
    assert keys == ["ctx7sk-91", "ctx7sk-92"]
    assert Context7Config().api_key() == "ctx7sk-91"


def test_key_strategy_defaults_to_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    assert Context7Config().key_strategy_of() == "fallback"


@pytest.mark.parametrize("strategy", ["fallback", "round_robin", "random"])
def test_every_documented_strategy_is_accepted(monkeypatch, strategy: str) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    cfg = Context7Config(key_strategy=strategy)
    assert cfg.key_strategy_of() == strategy


def test_an_unknown_strategy_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    cfg = Context7Config(key_strategy="telepathy")
    assert cfg.key_strategy_of() == "fallback"


# --------------------------------------------------------------------------- #
# Availability — this is the auto-validation contract.
# --------------------------------------------------------------------------- #


def test_tool_is_unavailable_when_no_key_is_set(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    tool = Context7Tool(_full_config())
    ok, reason = tool.available()
    assert not ok
    assert "CONTEXT7_API_KEY" in reason


def test_tool_is_available_when_a_key_is_set(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    ok, _ = tool.available()
    assert ok


def test_disabled_in_config_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    config = _full_config()
    config.tools.context7.enabled = False
    tool = Context7Tool(config)
    ok, reason = tool.available()
    assert not ok
    assert "disabled" in reason


def test_summary_line_shows_key_count(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    tool = Context7Tool(_full_config())
    line = tool.summary_line()
    assert "2 keys" in line
    assert "fallback" in line


# --------------------------------------------------------------------------- #
# Action: resolve_library_id
# --------------------------------------------------------------------------- #


async def test_resolve_returns_library_matches(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    payload = {
        "results": [
            {
                "id": "/facebook/react",
                "title": "React",
                "description": "A JS library for building UIs",
                "score": 0.97,
            },
            {
                "id": "/remix-run/react-router",
                "title": "React Router",
                "description": "Declarative routing for React",
                "score": 0.71,
            },
        ]
    }
    async with _stub_client(tool._client, (200, payload)):
        result = await tool.invoke(
            {"action": "resolve_library_id", "library_name": "react", "query": "hooks"},
            ctx(),
        )

    assert result.ok
    assert "/facebook/react" in result.text
    assert "React" in result.text
    assert result.summary.startswith("context7:")


async def test_resolve_requires_a_library_name(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    with pytest.raises(ToolError, match="library_name is required"):
        await tool.invoke({"action": "resolve_library_id"}, ctx())


async def test_resolve_handles_no_matches(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    async with _stub_client(tool._client, (200, {"results": []})):
        result = await tool.invoke(
            {"action": "resolve_library_id", "library_name": "nope"}, ctx()
        )
    assert result.ok
    assert "no libraries" in result.text


# --------------------------------------------------------------------------- #
# Action: query_docs
# --------------------------------------------------------------------------- #


async def test_query_docs_returns_snippets(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    payload = {
        "codeSnippets": [
            {
                "libraryId": "/facebook/react",
                "codeTitle": "useState",
                "codeList": [{"code": "const [x, setX] = useState(0);", "language": "tsx"}],
            }
        ],
        "infoSnippets": [
            {"title": "Hooks rules", "content": "Only call hooks at the top level."},
        ],
    }
    async with _stub_client(tool._client, (200, payload)):
        result = await tool.invoke(
            {
                "action": "query_docs",
                "library_id": "/facebook/react",
                "query": "useState",
            },
            ctx(),
        )

    assert result.ok
    assert "useState" in result.text
    assert "top level" in result.text
    assert "context7 docs for" in result.text
    assert result.data["snippet_count"] == 2
    assert result.data["code_count"] == 1
    assert result.data["info_count"] == 1


async def test_query_docs_rejects_a_bare_id(monkeypatch) -> None:
    """``react`` is a name, not an ID — Context7 would misinterpret it."""
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    with pytest.raises(ToolError, match="must start with '/'"):
        await tool.invoke(
            {"action": "query_docs", "library_id": "react"}, ctx()
        )


async def test_query_docs_requires_an_id(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    with pytest.raises(ToolError, match="library_id is required"):
        await tool.invoke({"action": "query_docs"}, ctx())


# --------------------------------------------------------------------------- #
# Key rotation — the contract that lets a dead key stay dead.
# --------------------------------------------------------------------------- #


async def test_a_401_falls_through_to_the_next_key(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    tool = Context7Tool(_full_config())
    payload = {"results": [{"id": "/x/y", "title": "Y"}]}

    # First key: 401. Second key: 200.
    async with _stub_client(tool._client, (401, "unauthorized"), (200, payload)):
        result = await tool.invoke(
            {"action": "resolve_library_id", "library_name": "y"}, ctx()
        )

    assert result.ok
    assert len(tool._client._clients) == 0  # both calls used the same stub


async def test_a_400_does_not_burn_the_pool(monkeypatch) -> None:
    """The request is wrong, not the key — do not retry on a different key."""
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    tool = Context7Tool(_full_config())

    async with _stub_client(tool._client, (400, "bad libraryName")):
        with pytest.raises(ToolError, match="400"):
            await tool.invoke(
                {"action": "resolve_library_id", "library_name": "x"}, ctx()
            )


async def test_every_key_failing_reports_the_pool(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91, ctx7sk-92")
    tool = Context7Tool(_full_config())

    async with _stub_client(
        tool._client, (401, "u1"), (403, "f2"), (429, "rl"),
    ):
        with pytest.raises(ToolError) as caught:
            await tool.invoke(
                {"action": "resolve_library_id", "library_name": "x"}, ctx()
            )
    assert "all 2 keys" in str(caught.value)


# --------------------------------------------------------------------------- #
# Unknown action
# --------------------------------------------------------------------------- #


async def test_unknown_action_raises(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    tool = Context7Tool(_full_config())
    with pytest.raises(ToolError, match="unknown action"):
        await tool.invoke({"action": "yolo"}, ctx())


# --------------------------------------------------------------------------- #
# Registry integration — this is what the system prompt depends on.
# --------------------------------------------------------------------------- #


def test_unavailable_tool_is_not_in_the_specs(config, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    registry = ToolRegistry()
    tool = Context7Tool(_full_config())
    registry.register(tool)

    assert tool.name in registry.names()  # registered for introspection
    assert all(s["function"]["name"] != "context7" for s in registry.specs())  # not for the model
    assert "context7" not in registry.describe(available_only=True)
    assert "context7" in registry.describe()  # default still shows it


def test_available_tool_is_in_the_specs(config, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    registry = ToolRegistry()
    tool = Context7Tool(_full_config())
    registry.register(tool)

    assert "context7" in registry.describe(available_only=True)
    assert any(s["function"]["name"] == "context7" for s in registry.specs())


def test_describe_available_only_hides_every_disabled_tool(config, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    # Register two tools: one available, one not.
    registry = ToolRegistry()
    available_tool = Context7Tool.__new__(Context7Tool)
    available_tool.name = "always-on"
    available_tool.description = "Always on."

    unavailable_tool = Context7Tool.__new__(Context7Tool)
    unavailable_tool.name = "needs-key"
    unavailable_tool.description = "Needs a key."

    def available_ok():
        return True, ""

    def unavailable():
        return False, "no key"

    available_tool.available = available_ok  # type: ignore[method-assign]
    unavailable_tool.available = unavailable  # type: ignore[method-assign]
    registry.register(available_tool)
    registry.register(unavailable_tool)

    description = registry.describe(available_only=True)
    assert "always-on" in description
    assert "needs-key" not in description