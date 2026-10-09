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
Setting ``reasoning = true`` in the ``[llm]`` block does that and the rest of
what a reasoning model needs automatically; see :mod:`lumi.llm.reasoning`.

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
from .base import (
    USAGE_COMPLETION,
    USAGE_PROMPT,
    USAGE_REASONING,
    LLMClient,
    LLMError,
    LLMReply,
    ToolCall,
)
from .keypool import KeyPool, is_retryable, mask
from .reasoning import (
    completion_ceiling,
    is_context_overflow,
    rejects_param,
    reported_context_limit,
    split_think,
    trace_of,
)

log = get_logger(__name__)


def has_image_part(messages: list[dict[str, Any]]) -> bool:
    """Whether any message in *messages* carries an image part.

    OpenAI-compatible providers send a photo as a ``content`` array holding an
    ``image_url`` item; plain turns keep ``content`` a string. Both shapes are
    checked defensively — some proxies normalise in one direction or the other —
    and anything unrecognised simply does not count as an image.
    """
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and "image_url" in part:
                return True
    return False


def request_model(messages: list[dict[str, Any]], config: LLMConfig) -> str:
    """The model id for this request: the vision model when images are present.

    The vision model is consulted for the whole request, not just the newest
    turn: replayed history can carry an image from an earlier photo, and a
    text-only default model would reject the transcript on the next turn for a
    reason nobody could see. An unset vision model routes everything — images
    included — to the default, which is the correct behaviour when the default
    is itself multimodal.
    """
    if has_image_part(messages):
        return config.vision_model_of() or config.model_of()
    return config.model_of()


class OpenAICompatClient(LLMClient):
    name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._pool = KeyPool(config.api_keys(), config.strategy_of())
        self._clients: dict[str, Any] = {}
        #: (model id, carried an image) for the call in flight, for error text.
        self._last_request: tuple[str, bool] = (config.model_of(), False)

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

    def _params(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """The request body, built for the configured model.

        Two deliberate departures from the plain case, both driven by
        ``llm.reasoning``:

        - ``temperature`` is omitted. Reasoning models reject any value but
          their own default, and a 400 on every turn is worse than a missing
          knob. It stays available for everyone else.
        - the token ceiling moves to ``max_completion_tokens`` and grows by
          ``llm.reasoning_tokens``, because the thinking trace is billed as
          output and would otherwise eat the answer. With both at their default
          of 0 no ceiling is sent at all and the provider's own limit stands.
        - ``reasoning_effort`` is never absent in this mode: an unset effort
          resolves to :data:`lumi.config.DEFAULT_REASONING_EFFORT` in
          :meth:`LLMConfig.effort_of`, because on endpoints like Ollama's the
          parameter's presence is the thinking on/off switch and omitting it
          would leave a default-off model silent.

        The model id itself is per request: turns whose context carries an
        image (the latest photo, or one still in replayed history) go to
        ``llm.vision_model`` when one is set. See :func:`request_model`.
        """
        config = self.config
        params: dict[str, Any] = {"model": request_model(messages, config), "messages": messages}

        if config.temperature is not None and not config.reasoning:
            params["temperature"] = config.temperature

        ceiling = (
            completion_ceiling(config.max_tokens, config.reasoning_tokens)
            if config.reasoning
            else max(0, config.max_tokens)
        )
        # 0 means unset: the parameter is left out entirely rather than sent as
        # zero, and the provider's own output limit applies.
        if ceiling:
            params["max_completion_tokens" if config.reasoning else "max_tokens"] = ceiling

        if config.reasoning:
            effort = config.effort_of()
            if effort:
                params["reasoning_effort"] = effort

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        return params

    def _variants(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """*params* plus progressively plainer versions of it.

        Reasoning support is a moving target across providers: one wants
        ``max_completion_tokens`` and another only knows ``max_tokens``; one
        accepts ``reasoning_effort`` and another 400s on it. Rather than making
        the owner discover which combination their endpoint wants, try the
        richest set and step down one parameter at a time when the endpoint says
        it does not recognise something. At most two extra requests, only on the
        failure path, and each step down is a strict subset of the last, so a
        request that succeeds is never repeated.
        """
        variants = [params]
        if "max_completion_tokens" in params:
            legacy = {k: v for k, v in params.items() if k != "max_completion_tokens"}
            legacy["max_tokens"] = params["max_completion_tokens"]
            variants.append(legacy)
        if "reasoning_effort" in params:
            variants.append({k: v for k, v in variants[-1].items() if k != "reasoning_effort"})
        return variants

    async def _try(
        self, key: str, variants: list[dict[str, Any]]
    ) -> tuple[Any | None, BaseException | None, bool]:
        """Call the endpoint on one key, stepping down *variants* as it refuses.

        Returns ``(completion, failure, retryable)``. ``completion`` is set only
        on success; ``failure`` carries the last error, which is the most
        informative one because it names the parameter that finally stuck.
        """
        for index, params in enumerate(variants):
            try:
                completion = await self._client_for(key).chat.completions.create(**params)
            except Exception as exc:  # noqa: BLE001 - normalised for the owner
                if rejects_param(exc) and index + 1 < len(variants):
                    log.warning(
                        "%s rejected this request: %s. Retrying without %s",
                        self.config.base_url_of(), format_error(exc), _dropped(params),
                    )
                    continue
                return None, exc, is_retryable(exc)
            if index:
                log.info("endpoint accepted the request once %s was dropped", _dropped(params))
            return completion, None, False
        raise LLMError("no parameter variant left to try")  # pragma: no cover

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

        variants = self._variants(self._params(messages, tools))
        # Remember which model this call went out under so a failure can name
        # it and point at the config line that fixes the common mistakes.
        self._last_request = (
            request_model(messages, self.config),
            has_image_part(messages),
        )
        tried: list[str] = []
        failure: BaseException | None = None
        # One attempt per key, never more: a single key must behave exactly as
        # it did before pools existed.
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            completion, error, retryable = await self._try(key, variants)
            if error is not None:
                failure = error
                self._pool.report(key, ok=False, retryable=retryable)
                log.error(
                    "llm call to %s failed on key %s: %s",
                    self.config.base_url_of(), mask(key), format_error(error),
                )
                if not retryable:
                    break
                log.warning(
                    "retrying on the next key (%d/%d tried)", len(tried), len(self._pool)
                )
                continue
            self._pool.report(key, ok=True)
            assert completion is not None  # noqa: S101 - guaranteed by the tuple contract
            return self._parse(completion)

        raise LLMError(self._explain(failure, tried))

    def _explain(self, failure: BaseException | None, tried: list[str]) -> str:
        """The message the owner sees. Names the endpoint, model and key source.

        Always name the endpoint: "401 invalid_api_key" against the wrong host is
        the most confusing failure this program can produce.
        """
        endpoint = self.config.base_url_of()
        model, with_images = self._last_request
        if failure is None:  # pragma: no cover - only if the pool goes empty mid-call
            return (
                f"no usable key in {self.config.api_key_env}.\n\n"
                f"Endpoint: {endpoint} (from {self.config.where_from()})\n"
                f"Model: {model}"
            )

        hint = ""
        if is_context_overflow(failure):
            # The one 400 with an exact fix, and the provider almost always
            # names its own limit in the message — so quote it back and point
            # at the line that has to change. Without this the only visible
            # symptom is that the model starts refusing, with nothing in the
            # conversation to explain why.
            actual = reported_context_limit(failure)
            window = self.config.context_window
            hint = (
                f"\n\nThis request did not fit the model's context window. "
                f"{self.config.model_of()} reports a limit of {actual:,} tokens"
                if actual
                else "\n\nThis request did not fit the model's context window. "
            )
            hint += (
                f", while llm.context_window is set to {window:,}. "
                "Set it in the [llm] block of config.toml (or LUMI__LLM__CONTEXT_WINDOW)"
                + (
                    f" — {actual:,} is what this endpoint just told you it accepts."
                    if actual and window != actual
                    else "."
                )
            )
            hint += (
                " llm.context_headroom is "
                f"{self.config.context_headroom:,} tokens, held back so the reply fits;"
                " history is packed to whatever is left."
            )
        elif with_images and not self.config.vision_model_of():
            # The request carried an image and no vision model is configured,
            # so the likeliest cause is a text-only default. Name the fix.
            hint = (
                f"\n\nThis request carried an image. If {model} cannot read images, "
                "add a multimodal model to the [llm] block of config.toml: "
                '`vision_model = "gpt-4o"` (or any vision-capable id your endpoint serves).'
            )
        elif endpoint == OPENAI_DEFAULT_BASE_URL:
            hint = (
                "\n\nThat is OpenAI itself. If you meant a different provider, set "
                "OPENAI_BASE_URL in .env, or base_url in the [llm] block of config.toml."
            )
        elif self._looks_like_a_reasoning_model(failure):
            # The value rejections reasoning models produce are specific and
            # confusing on their own ("temperature is only supported when set to
            # 1"), so name the config line that fixes the whole class of them.
            hint = (
                f"\n\n{self.config.model_of()} looks like a reasoning model. Add "
                "`reasoning = true` to the [llm] block of config.toml: it drops "
                "temperature, moves the token limit to max_completion_tokens and "
                "reserves reasoning_tokens for the thinking trace."
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
            f"Model: {model}\n"
            f"Key: {self.config.api_key_env}{pool}{hint}"
        )

    def _looks_like_a_reasoning_model(self, failure: BaseException) -> bool:
        """Whether *failure* is the shape of complaint a reasoning model makes.

        Not a model-name guess: the model id in use is frequently a local one
        (``qwen3:8b``, ``gpt-oss:20b``) that no list would be able to keep up
        with. Instead this matches the two things reasoning models reliably
        refuse — a temperature other than their own, and the old max_tokens name.
        """
        if self.config.reasoning:
            return False  # already in reasoning mode, so this is something else
        message = str(failure).lower()
        return "temperature" in message or "max_completion_tokens" in message

    def _parse(self, completion: Any) -> LLMReply:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError("the model returned no choices")
        message = choices[0].message

        # A dedicated trace field first; only fall back to unpicking inline
        # <think> tags, which some local templates emit instead.
        trace = trace_of(message)
        text, inline = split_think((getattr(message, "content", None) or "").strip())
        if not trace:
            trace = inline
            if inline:
                log.debug("extracted %d chars of inline thinking from content", len(inline))

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
            # Under the USAGE_* names the rest of the package reads them by.
            usage = {
                USAGE_PROMPT: getattr(completion.usage, "prompt_tokens", 0) or 0,
                USAGE_COMPLETION: getattr(completion.usage, "completion_tokens", 0) or 0,
            }
            # Not every provider reports the split, and a missing one is simply
            # absent rather than zero, so read it defensively.
            details = getattr(completion.usage, "completion_tokens_details", None)
            thinking = getattr(details, "reasoning_tokens", 0) or 0
            if thinking:
                usage[USAGE_REASONING] = int(thinking)

        return LLMReply(
            text=text,
            reasoning=trace,
            tool_calls=calls,
            finish_reason=getattr(choices[0], "finish_reason", "") or "",
            usage=usage,
        )

    def describe(self) -> str:
        return f"{self.config.model_of()} via {self.config.base_url_of()}"


def _dropped(params: dict[str, Any]) -> str:
    """A short description of a variant, for the log line about stepping down."""
    optional = [k for k in ("reasoning_effort", "temperature", "max_completion_tokens") if k in params]
    return ", ".join(optional) or "the optional parameters"


__all__ = ["OpenAICompatClient"]
