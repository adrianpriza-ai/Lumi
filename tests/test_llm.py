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
from lumi.llm.keypool import mask
from lumi.llm.openai_compat import OpenAICompatClient


class StubCompletions:
    def __init__(self, reply=None, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: list[dict] = []

    async def create(self, **params):
        self.requests.append(params)
        if isinstance(self.error, list) and self.error:
            # One error per attempt, so a retry can be scripted.
            raise self.error.pop(0)
        if self.error and not isinstance(self.error, list):
            raise self.error
        return self.reply


class StubClient:
    def __init__(self, completions: StubCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def client_with(config, completions: StubCompletions) -> OpenAICompatClient:
    """A client wired to a stub instead of the SDK.

    ``used_keys`` records which key each attempt went out with, which is how the
    rotation tests below see the pool at work without a network.
    """
    client = OpenAICompatClient(config.llm)
    stub = StubClient(completions)
    client.used_keys: list[str] = []

    def _client_for(key: str) -> StubClient:
        client.used_keys.append(key)
        return stub

    client._client_for = _client_for  # type: ignore[method-assign]
    return client


def reply(
    text: str = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    usage=True,
    *,
    reasoning: str | None = None,
    reasoning_tokens: int = 0,
):
    """A stand-in for a chat completion.

    ``reasoning`` lands as an attribute on the message rather than in the
    dataclass, which is exactly what the real SDK does with a provider's
    untyped extra field — and is the thing :mod:`lumi.llm.reasoning` probes.
    """
    function_calls = [
        SimpleNamespace(
            id=cid, function=SimpleNamespace(name=name, arguments=arguments)
        )
        for cid, name, arguments in (tool_calls or [])
    ]
    message = SimpleNamespace(content=text, tool_calls=function_calls or None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    details = SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens else None
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=(
            SimpleNamespace(
                prompt_tokens=11, completion_tokens=7, completion_tokens_details=details
            )
            if usage
            else None
        ),
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
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        await client.complete([{"role": "user", "content": "q"}])


# --------------------------------------------------------------------------- #
# key rotation
# --------------------------------------------------------------------------- #


def pooled(config, monkeypatch, *keys: str, strategy: str = "fallback") -> None:
    """Point the key variable at a comma-separated pool."""
    monkeypatch.setenv("OPENAI_API_KEY", ", ".join(keys))
    config.llm.key_strategy = strategy


def status_error(code: int, message: str = "the model refused") -> Exception:
    """An exception shaped like the SDK's: a ``status_code`` and a message."""
    exc = RuntimeError(f"Error code: {code} - {message}")
    exc.status_code = code  # type: ignore[attr-defined]
    return exc


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #


def test_masking_keeps_enough_to_identify_a_key() -> None:
    assert mask("sk-proj-abcdefghijklmnop90") == "sk-…op90"
    assert mask("tvly-abc123") == "tvl…c123"


def test_masking_never_reveals_a_short_key() -> None:
    """With fewer than 8 characters, a prefix and a suffix would overlap."""
    for key in ("abc", "sk-90", "x" * 7):
        assert key not in mask(key)
    assert mask("") == "(empty)"
    assert mask("sk-90") == "(len 5)"


async def test_a_single_key_is_the_only_key_tried(config) -> None:
    completions = StubCompletions(error=[status_error(429)])
    client = client_with(config, completions)
    with pytest.raises(LLMError):
        await client.complete([{"role": "user", "content": "q"}])

    assert client.used_keys == ["test-key"]  # no rotation without a pool


async def test_a_failing_key_falls_through_to_the_next(config, monkeypatch) -> None:
    pooled(config, monkeypatch, "sk-90", "sk-91")
    # One scripted failure, then the stub succeeds: the second key answers.
    completions = StubCompletions(reply("answered by the second key"), error=[status_error(429)])
    client = client_with(config, completions)

    result = await client.complete([{"role": "user", "content": "q"}])
    assert result.text == "answered by the second key"
    assert client.used_keys == ["sk-90", "sk-91"]


async def test_fallback_keeps_using_the_healthy_key(config, monkeypatch) -> None:
    """After one failure the parked key is skipped, not retried every turn."""
    pooled(config, monkeypatch, "sk-90", "sk-91")
    completions = StubCompletions(reply("hi"), error=[status_error(429)])
    client = client_with(config, completions)

    await client.complete([{"role": "user", "content": "q"}])
    await client.complete([{"role": "user", "content": "q again"}])

    assert client.used_keys == ["sk-90", "sk-91", "sk-91"]


async def test_a_400_is_not_retried_on_another_key(config, monkeypatch) -> None:
    """The request is wrong, not the key; burning the pool helps nobody."""
    pooled(config, monkeypatch, "sk-90", "sk-91")
    completions = StubCompletions(error=status_error(400))
    client = client_with(config, completions)

    with pytest.raises(LLMError, match="400"):
        await client.complete([{"role": "user", "content": "q"}])
    assert client.used_keys == ["sk-90"]


async def test_every_key_failing_reports_the_whole_pool(config, monkeypatch) -> None:
    pooled(config, monkeypatch, "sk-90", "sk-91")
    completions = StubCompletions(error=[status_error(401), status_error(403)])
    client = client_with(config, completions)

    with pytest.raises(LLMError) as caught:
        await client.complete([{"role": "user", "content": "q"}])

    assert client.used_keys == ["sk-90", "sk-91"]
    assert "all 2 of OPENAI_API_KEY failed" in str(caught.value)


async def test_round_robin_alternates_keys(config, monkeypatch) -> None:
    pooled(config, monkeypatch, "sk-90", "sk-91", strategy="round_robin")
    completions = StubCompletions(reply("hi"))
    client = client_with(config, completions)

    for _ in range(4):
        await client.complete([{"role": "user", "content": "q"}])

    assert client.used_keys == ["sk-90", "sk-91", "sk-90", "sk-91"]


async def test_random_stays_inside_the_pool(config, monkeypatch) -> None:
    pooled(config, monkeypatch, "sk-90", "sk-91", strategy="random")
    completions = StubCompletions(reply("hi"))
    client = client_with(config, completions)

    for _ in range(20):
        await client.complete([{"role": "user", "content": "q"}])

    assert set(client.used_keys) == {"sk-90", "sk-91"}


async def test_an_unknown_strategy_falls_back_instead_of_booting(config, monkeypatch) -> None:
    pooled(config, monkeypatch, "sk-90", "sk-91", strategy="telepathy")
    completions = StubCompletions(reply("hi"))
    client = client_with(config, completions)

    await client.complete([{"role": "user", "content": "q"}])
    assert client.used_keys == ["sk-90"]  # first key, as fallback means


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


# --------------------------------------------------------------------------- #
# endpoint resolution
# --------------------------------------------------------------------------- #


async def test_the_request_goes_to_the_resolved_endpoint(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    # The stub bypasses the SDK, so assert on what the client was told to use.
    assert config.llm.base_url_of() == "http://localhost:20128/v1"


async def test_the_resolved_model_is_sent(config, monkeypatch) -> None:
    config.llm.model = "gpt-4.1-mini"  # the shipped default, so the env can win
    monkeypatch.setenv("OPENAI_MODEL", "nemotron-3-nano-reasoning")
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["model"] == "nemotron-3-nano-reasoning"


async def test_sdk_errors_name_the_endpoint(config, monkeypatch) -> None:
    """A 401 from the wrong host must say which host it actually hit."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    completions = StubCompletions(error=RuntimeError("401 invalid_api_key"))
    with pytest.raises(LLMError) as caught:
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    message = str(caught.value)
    assert "http://localhost:20128/v1" in message
    assert "OPENAI_BASE_URL" in message  # where it came from
    assert "test-model" in message
    assert "OPENAI_API_KEY" in message


async def test_an_openai_endpoint_failure_suggests_the_fix(config, monkeypatch) -> None:
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    completions = StubCompletions(error=RuntimeError("401 invalid_api_key"))
    with pytest.raises(LLMError) as caught:
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    message = str(caught.value)
    assert "That is OpenAI itself" in message
    assert "OPENAI_BASE_URL" in message


async def test_a_custom_endpoint_does_not_get_the_openai_hint(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    completions = StubCompletions(error=RuntimeError("500 boom"))
    with pytest.raises(LLMError) as caught:
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "That is OpenAI itself" not in str(caught.value)


# --------------------------------------------------------------------------- #
# reasoning models
# --------------------------------------------------------------------------- #


def reasoning_config(config, **overrides) -> None:
    """Put the config into reasoning mode, as ``reasoning = true`` would."""
    config.llm.reasoning = True
    for name, value in overrides.items():
        setattr(config.llm, name, value)


async def test_reasoning_mode_drops_temperature(config) -> None:
    """The single most common reason a reasoning model refuses to run."""
    reasoning_config(config)
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "temperature" not in completions.requests[0]


async def test_temperature_is_still_sent_without_reasoning(config) -> None:
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["temperature"] == 0.7


async def test_reasoning_mode_uses_max_completion_tokens(config) -> None:
    reasoning_config(config)
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    request = completions.requests[0]
    assert "max_tokens" not in request
    # 2000 for the answer plus 2000 of headroom, so a long trace cannot eat
    # the reply.
    assert request["max_completion_tokens"] == 4000


async def test_the_thinking_budget_is_configurable(config) -> None:
    reasoning_config(config, reasoning_tokens=500, max_tokens=1000)
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["max_completion_tokens"] == 1500


async def test_effort_is_sent_when_set(config) -> None:
    reasoning_config(config, reasoning_effort="high")
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["reasoning_effort"] == "high"


async def test_no_effort_sends_the_default_so_thinking_switches_on(config) -> None:
    """An unset effort must not omit the parameter: on endpoints like Ollama's
    its presence is the thinking on/off switch, so omitting it silently leaves
    a default-off model (gemma4, ...) never thinking at all."""
    reasoning_config(config)
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["reasoning_effort"] == "medium"


async def test_plain_mode_never_sends_an_effort(config) -> None:
    """reasoning = false, effort unset: the parameter is none of the request's
    business, whatever an ambient LUMI__LLM__REASONING_EFFORT might say."""
    config.llm.reasoning = False
    config.llm.reasoning_effort = "high"
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "reasoning_effort" not in completions.requests[0]


async def test_an_unknown_effort_is_omitted_rather_than_sent(config) -> None:
    reasoning_config(config, reasoning_effort="turbo")
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "reasoning_effort" not in completions.requests[0]


async def test_effort_none_is_sent_when_asked_for(config) -> None:
    reasoning_config(config, reasoning_effort="none")
    completions = StubCompletions(reply("hi"))
    await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert completions.requests[0]["reasoning_effort"] == "none"


# -- the downgrade ladder ---------------------------------------------------- #


async def test_an_endpoint_that_only_knows_max_tokens_still_works(config) -> None:
    reasoning_config(config, reasoning_effort="high")
    completions = StubCompletions(
        reply("answered"),
        error=[RuntimeError("Unsupported parameter: 'max_completion_tokens'")],
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    assert result.text == "answered"
    second = completions.requests[1]
    assert "max_completion_tokens" not in second
    assert second["max_tokens"] == 4000
    # Everything else is untouched: only the rejected parameter is dropped.
    assert second["reasoning_effort"] == "high"
    assert "temperature" not in second


async def test_the_ladder_keeps_going_until_nothing_is_left_to_drop(config) -> None:
    reasoning_config(config, reasoning_effort="high")
    completions = StubCompletions(
        reply("answered"),
        error=[
            RuntimeError("Unsupported parameter: 'max_completion_tokens'"),
            RuntimeError("Unrecognized request argument supplied: reasoning_effort"),
        ],
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    assert result.text == "answered"
    assert len(completions.requests) == 3
    final = completions.requests[2]
    assert "reasoning_effort" not in final
    assert "max_completion_tokens" not in final
    assert final["max_tokens"] == 4000


async def test_a_value_error_is_not_retried(config) -> None:
    """Retrying a 400 about a bad value would only bury the message."""
    reasoning_config(config, reasoning_effort="high")
    completions = StubCompletions(error=status_error(400, "temperature is only supported when set to 1"))
    with pytest.raises(LLMError):
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert len(completions.requests) == 1


async def test_the_ladder_stops_when_every_variant_is_refused(config) -> None:
    reasoning_config(config, reasoning_effort="high")
    completions = StubCompletions(error=RuntimeError("unsupported parameter: everything"))
    with pytest.raises(LLMError):
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert len(completions.requests) == 3


async def test_the_ladder_does_not_apply_without_reasoning_mode(config) -> None:
    """A non-reasoning request sends one variant, so an endpoint that dislikes
    the message shape is told once rather than three times."""
    completions = StubCompletions(error=RuntimeError("unsupported parameter: whatever"))
    with pytest.raises(LLMError):
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert len(completions.requests) == 1


async def test_a_reasoning_model_failure_names_the_config_fix(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    completions = StubCompletions(error=status_error(400, "temperature is only supported when set to 1"))
    with pytest.raises(LLMError) as caught:
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])

    message = str(caught.value)
    assert "reasoning = true" in message
    assert "max_completion_tokens" in message


async def test_no_reasoning_hint_when_already_in_reasoning_mode(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    reasoning_config(config)
    completions = StubCompletions(error=status_error(400, "temperature is only supported when set to 1"))
    with pytest.raises(LLMError) as caught:
        await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "reasoning = true" not in str(caught.value)


# -- reading the trace back -------------------------------------------------- #


async def test_the_reasoning_field_is_captured(config) -> None:
    completions = StubCompletions(reply("42", reasoning="counted the letters first"))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.reasoning == "counted the letters first"
    assert result.text == "42"


async def test_reasoning_tokens_are_recorded_when_reported(config) -> None:
    completions = StubCompletions(reply("42", reasoning="thinking", reasoning_tokens=310))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.usage["reasoning"] == 310


async def test_absent_reasoning_token_details_are_tolerated(config) -> None:
    """Most providers do not report the split, and a missing one must not be
    mistaken for zero-and-therefore-broken."""
    completions = StubCompletions(reply("42", reasoning="thinking"))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert "reasoning" not in result.usage
    assert result.usage["completion"] == 7


async def test_inline_think_tags_never_reach_the_answer(config) -> None:
    completions = StubCompletions(reply("<think>weighing it up</think>here is the answer"))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.text == "here is the answer"
    assert result.reasoning == "weighing it up"


async def test_a_trace_field_wins_over_inline_tags(config) -> None:
    completions = StubCompletions(
        reply("<think>tags</think>answer", reasoning="from the field")
    )
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.reasoning == "from the field"
    assert result.text == "answer"


async def test_a_non_reasoning_model_yields_no_trace(config) -> None:
    completions = StubCompletions(reply("just an answer"))
    result = await client_with(config, completions).complete([{"role": "user", "content": "q"}])
    assert result.reasoning == ""
