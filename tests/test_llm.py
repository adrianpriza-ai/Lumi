"""The OpenAI-compatible adapter: request shape and response parsing.

The SDK is never called — a stub client captures the request and returns
constructed objects, which is enough to pin down the two things that break in
practice: what we send, and how we read the reply.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lumi.llm import build_llm
from lumi.llm.base import LLMError
from lumi.llm.openai_compat import OpenAICompatClient


class StubCompletions:
    def __init__(self, reply=None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[dict] = []

    async def create(self, **params):
        self.requests.append(params)
        if self.error:
            raise self.error
        return self.reply


class StubClient:
    def __init__(self, completions: StubCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def client_with(config, completions: StubCompletions) -> OpenAICompatClient:
    client = OpenAICompatClient(config.llm)
    client._client = StubClient(completions)
    return client


def reply(text: str = "", tool_calls: list | None = None, finish_reason: str = "stop", usage=True):
    function_calls = [
        SimpleNamespace(
            id=cid, function=SimpleNamespace(name=name, arguments=arguments)
        )
        for cid, name, arguments in (tool_calls or [])
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=function_calls or None),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7) if usage else None,
    )


# --------------------------------------------------------------------------- #
# request shape
# --------------------------------------------------------------------------- #


async def test_sends_the_model_and_messages(config) -> None:
    completions = StubCompletions(reply("hi"))
    client = client_with(config, completions)
    await client.complete([{"role": "user", "content": "hello"}])

    request = completions.requests[0]
    assert request["model"] == "test-model"
    assert request["messages"] == [{"role": "user", "content": "hello"}]
    assert request["max_tokens"] == 2000
    assert "tools" not in request


async def test_sends_tools_and_auto_choice(config) -> None:
    completions = StubCompletions(reply("hi"))
    client = client_with(config, completions)
    tools = [{"type": "function", "function": {"name": "x", "description": "y", "parameters": {}}}]
    await client.complete([{"role": "user", "content": "hi"}], tools=tools)

    request = completions.requests[0]
    assert request["tools"] == tools
    assert request["tool_choice"] == "auto"


async def test_temperature_is_included_when_set(config) -> None:
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "hi"}])
    assert completions.requests[0]["temperature"] == 0.7


async def test_null_temperature_is_omitted(config, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__TEMPERATURE", "none")
    from lumi.config import load_config

    reloaded = load_config(config.root)
    completions = StubCompletions(reply("hi"))
    await client_with(reloaded, completions).complete([{"role": "user", "content": "hi"}])
    assert "temperature" not in completions.requests[0]


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #


async def test_parses_plain_text(config) -> None:
    completions = StubCompletions(reply("the answer"))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    assert result.text == "the answer"
    assert result.tool_calls == []
    assert result.wants_tools is False
    assert result.usage == {"prompt": 11, "completion": 7}


async def test_parses_tool_calls_with_json_arguments(config) -> None:
    completions = StubCompletions(
        reply("", [("call_1", "run_shell", '{"command": "ls"}')], finish_reason="tool_calls")
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    assert result.wants_tools is True
    call = result.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "run_shell"
    assert call.arguments == {"command": "ls"}
    assert json.loads(call.arguments_json) == {"command": "ls"}


async def test_tool_arguments_are_parsed_even_with_preamble(config) -> None:
    completions = StubCompletions(
        reply("", [("c1", "files", '{"action": "read", "path": "a.md"}')])
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.tool_calls[0].arguments == {"action": "read", "path": "a.md"}


async def test_unparseable_tool_arguments_do_not_crash(config) -> None:
    completions = StubCompletions(reply("", [("c1", "run_shell", "{not json")]))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.tool_calls[0].arguments == {}


async def test_null_tool_arguments_become_an_empty_dict(config) -> None:
    completions = StubCompletions(reply("", [("c1", "memory", None)]))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.tool_calls[0].arguments == {}


async def test_multiple_tool_calls_are_all_kept(config) -> None:
    completions = StubCompletions(
        reply("", [("c1", "files", '{"action":"list"}'), ("c2", "run_shell", '{"command":"ls"}')])
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert [c.id for c in result.tool_calls] == ["c1", "c2"]


async def test_text_alongside_tool_calls_is_kept(config) -> None:
    completions = StubCompletions(
        reply("let me look", [("c1", "run_shell", '{"command":"ls"}')])
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.text == "let me look"
    assert result.wants_tools is True


async def test_missing_usage_is_tolerated(config) -> None:
    completions = StubCompletions(reply("hi", usage=False))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.usage == {}


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #


async def test_no_choices_is_an_error(config) -> None:
    completions = StubCompletions(SimpleNamespace(choices=[], usage=None))
    with pytest.raises(LLMError, match="no choices"):
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])


async def test_sdk_errors_become_llm_errors(config) -> None:
    completions = StubCompletions(error=RuntimeError("503 upstream"))
    with pytest.raises(LLMError, match="503 upstream"):
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])


async def test_missing_api_key_is_explained(config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatClient(config.llm)
    client._client = None
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        await client.complete([{"role": "user", "content": "q"}])


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_every_documented_provider_builds_the_same_client(config) -> None:
    for name in ("openai", "openai-compatible", "ollama", "lmstudio", "vllm", "groq", "openrouter"):
        config.llm.provider = name
        assert isinstance(build_llm(config), OpenAICompatClient)


def test_unknown_provider_is_rejected(config) -> None:
    config.llm.provider = "skynet"
    with pytest.raises(LLMError, match="unknown llm.provider"):
        build_llm(config)


def test_describe_names_the_model_and_endpoint(config) -> None:
    assert "test-model" in OpenAICompatClient(config.llm).describe()
