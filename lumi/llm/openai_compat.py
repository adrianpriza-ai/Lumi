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
"""

from __future__ import annotations

import json
from typing import Any

from ..config import OPENAI_DEFAULT_BASE_URL, LLMConfig
from ..util.log import get_logger
from ..util.text import format_error
from .base import LLMClient, LLMError, LLMReply, ToolCall

log = get_logger(__name__)


class OpenAICompatClient(LLMClient):
    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("the openai package is not installed") from exc

            api_key = self.config.api_key()
            if not api_key:
                raise LLMError(
                    f"{self.config.api_key_env} is not set. Put it in .env, or point "
                    f"llm.base_url at a local server and set llm.api_key_env to a dummy value."
                )
            # Local servers ignore the key but the SDK insists on one being set.
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url_of(),
                max_retries=2,
                timeout=120.0,
            )
            log.info(
                "llm ready: model=%s endpoint=%s (from %s)",
                self.config.model_of(),
                self.config.base_url_of(),
                self.config.where_from(),
            )
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMReply:
        client = self._get_client()
        endpoint = self.config.base_url_of()

        params: dict[str, Any] = {"model": self.config.model_of(), "messages": messages}
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            completion = await client.chat.completions.create(**params)
        except Exception as exc:  # noqa: BLE001 - normalised for the owner
            log.error("llm call to %s failed: %s", endpoint, format_error(exc))
            # Always name the endpoint. "401 invalid_api_key" against the wrong
            # host is the most confusing failure this program can produce.
            hint = ""
            if endpoint == OPENAI_DEFAULT_BASE_URL:
                hint = (
                    "\n\nThat is OpenAI itself. If you meant a different provider, set "
                    "OPENAI_BASE_URL in .env, or base_url in the [llm] block of config.toml."
                )
            raise LLMError(
                f"{format_error(exc)}\n\n"
                f"Endpoint: {endpoint} (from {self.config.where_from()})\n"
                f"Model: {self.config.model_of()}\n"
                f"Key: {self.config.api_key_env}{hint}"
            ) from exc

        return self._parse(completion)

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
