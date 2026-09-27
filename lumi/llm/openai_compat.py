"""The OpenAI-compatible adapter.

One client, every provider that speaks the chat-completions API: OpenAI itself,
Ollama, LM Studio, vLLM, llama.cpp's server, Groq, OpenRouter, Together, Nous.
Switching between them is two lines in ``config.toml``:

.. code-block:: toml

    [llm]
    model = "qwen3:8b"
    base_url = "http://localhost:11434/v1"
    api_key_env = "OLLAMA_API_KEY"   # local servers usually ignore the value

``temperature = null`` omits the parameter, which some reasoning models require.

The key variable may hold a pool of keys rather than one (see
:mod:`lumi.llm.keypool`). One SDK client is built per key and the call is
retried on the next key when the failure looks like a key problem, so a 429
mid-conversation is invisible to the user.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import OPENAI_DEFAULT_BASE_URL, LLMConfig
from ..util.log import get_logger
from ..util.text import format_error
from .base import LLMClient, LLMError, LLMReply, ToolCall
from .keypool import KeyPool, is_retryable, mask

log = get_logger(__name__)


class OpenAICompatClient(LLMClient):
    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.strategy_of())
        self._clients: dict[str, Any] = {}

    def _client_for(self, key: str) -> Any:
        """The SDK client bound to *key*, built on first use.

        One client per key rather than one per call: the SDK pools connections
        and holds a session, so rebuilding it per request would be wasteful, and
        a cached client is also what keeps a key's state stable while it is in
        rotation.
        """
        client = self._clients.get(key)
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("the openai package is not installed") from exc

            # Local servers ignore the key but the SDK insists on one being set.
            client = AsyncOpenAI(
                api_key=key,
                base_url=self.config.base_url_of(),
                max_retries=2,
                timeout=120.0,
            )
            self._clients[key] = client
            log.info(
                "llm ready: model=%s endpoint=%s (from %s) keys=%s",
                self.config.model_of(),
                self.config.base_url_of(),
                self.config.where_from(),
                self._pool.describe(),
            )
        return client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMReply:
        if not self._pool:
            raise LLMError(
                f"{self.config.api_key_env} is not set. Put it in .env, or point "
                f"llm.base_url at a local server and set llm.api_key_env to a dummy value."
            )

        params: dict[str, Any] = {"model": self.config.model_of(), "messages": messages}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        tried: list[str] = []
        failure: BaseException | None = None
        # One attempt per key, never more: a single key must behave exactly as
        # it did before pools existed.
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            try:
                completion = await self._client_for(key).chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001 - normalised for the owner
                failure = exc
                retryable = is_retryable(exc)
                self._pool.report(key, ok=False, retryable=retryable)
                log.error(
                    "llm call to %s failed on key %s: %s",
                    self.config.base_url_of(), mask(key), format_error(exc),
                )
                if not retryable:
                    break
                log.warning(
                    "retrying on the next key (%d/%d tried)", len(tried), len(self._pool)
                )
                continue
            self._pool.report(key, ok=True)
            return self._parse(completion)

        raise LLMError(self._explain(failure, tried))

    def _explain(self, failure: BaseException | None, tried: list[str]) -> str:
        """The message the owner sees. Names the endpoint, model and key source.

        Always name the endpoint: "401 invalid_api_key" against the wrong host is
        the most confusing failure this program can produce.
        """
        endpoint = self.config.base_url_of()
        if failure is None:  # pragma: no cover - only if the pool goes empty mid-call
            return (
                f"no usable key in {self.config.api_key_env}.\n\n"
                f"Endpoint: {endpoint} (from {self.config.where_from()})\n"
                f"Model: {self.config.model_of()}"
            )

        hint = ""
        if endpoint == OPENAI_DEFAULT_BASE_URL:
            hint = (
                "\n\nThat is OpenAI itself. If you meant a different provider, set "
                "OPENAI_BASE_URL in .env, or base_url in the [llm] block of config.toml."
            )

        pool = ""
        if len(self._pool) > 1:
            pool = (
                f"\nKeys: all {len(tried)} of {self.config.api_key_env} failed "
                f"({self._pool.strategy})"
            )
        return (
            f"{format_error(failure)}\n\n"
            f"Endpoint: {endpoint} (from {self.config.where_from()})\n"
            f"Model: {self.config.model_of()}\n"
            f"Key: {self.config.api_key_env}{pool}{hint}"
        )

    def _parse(self, completion: Any) -> LLMReply:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError("the model returned no choices")
        message = choices[0].message
        text = (getattr(message, "content", None) or "").strip()

        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            function = getattr(raw, "function", None)
            name = getattr(function, "name", "") or ""
            raw_args = getattr(function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            except json.JSONDecodeError:
                log.warning("tool call %s had unparseable arguments: %r", name, raw_args)
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(ToolCall(id=getattr(raw, "id", "") or name, name=name, arguments=arguments))

        usage = {}
        if getattr(completion, "usage", None):
            usage = {
                "prompt": getattr(completion.usage, "prompt_tokens", 0) or 0,
                "completion": getattr(completion.usage, "completion_tokens", 0) or 0,
            }

        return LLMReply(
            text=text,
            tool_calls=calls,
            finish_reason=getattr(choices[0], "finish_reason", "") or "",
            usage=usage,
        )

    def describe(self) -> str:
        return f"{self.config.model_of()} via {self.config.base_url_of()}"


__all__ = ["OpenAICompatClient"]
