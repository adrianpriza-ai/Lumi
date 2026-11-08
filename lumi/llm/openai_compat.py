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
mid-conversation is invisible to the user. A successful but empty completion is
also retried briefly on the same key, which helps local servers such as Ollama
recover from a reasoning-only response without replaying any tools.

Passing ``on_delta`` to :meth:`OpenAICompatClient.complete` switches the same
call to ``stream=True`` and forwards each piece as it lands. The ladder above
keeps working because of where it bites: a parameter rejection and a key
failure both happen before the first token, and once a piece of answer text is
out the attempt is committed — see :class:`_StreamState` — so no retry can
text a viewer has already seen.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from ..config import OPENAI_DEFAULT_BASE_URL, LLMConfig
from ..util.log import get_logger
from ..util.text import format_error
from .base import (
    DELTA_TEXT,
    DELTA_THINKING,
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
    TRACE_FIELDS,
    completion_ceiling,
    is_context_overflow,
    rejects_param,
    reported_context_limit,
    split_think,
    trace_of,
)

log = get_logger(__name__)

# Ollama Cloud and other OpenAI-compatible endpoints occasionally return HTTP
# 200 with only a thinking trace and no final message. The retry budget lives in
# LLMConfig so it is visible and tunable rather than hidden in this adapter.
MAX_RETRY_DELAY_SECONDS = 8.0


class _StreamBroken(LLMError):
    """The stream died after answer text had already gone out.

    Another attempt would deliver those pieces again, so the call fails
    instead: no retry, no key rotation — and the key is not at fault, because
    the request itself succeeded. Raised only by :meth:`_stream` once
    :attr:`_StreamState.sent_text` is set.
    """


class _StreamState:
    """What one :meth:`OpenAICompatClient.complete` call has put on the wire.

    Streaming shares its entry point with the retry ladder, so the ladder has
    to know what a live viewer has already seen:

    - :attr:`sent_text` **commits** the call: an empty-completion retry, a key
      rotation and a parameter step-down are all pre-first-token events, and
      none may run once a piece of the answer has been delivered.
    - :attr:`sent_thinking` makes the reasoning trace single-delivery: the
      retry after a thinking-only completion (the shape some local servers
      return) re-thinks, and replaying that would show the same thought twice.
      The *reply* still carries the attempt that succeeded's trace; only the
      wire is silenced.
    """

    __slots__ = ("sent_text", "sent_thinking")

    def __init__(self) -> None:
        self.sent_text = False
        self.sent_thinking = False


class _Pieces:
    """One streamed attempt, assembled into the shape :meth:`_parse` reads.

    Reusing ``_parse`` is the point: a streamed reply and a whole one go
    through the same ``split_think``, tool-argument decoding and usage mapping,
    so the two can never disagree about what the model said.
    """

    __slots__ = ("text", "reasoning", "tools", "finish", "usage")

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        #: Stream index → [id, name, argument chunks]; tool calls arrive in
        #: pieces, and the id and name only appear in the first of them.
        self.tools: dict[int, list[Any]] = {}
        self.finish = ""
        self.usage: Any = None

    def add_tool(self, raw: Any) -> None:
        index = getattr(raw, "index", None)
        if not isinstance(index, int):
            index = len(self.tools)
        slot = self.tools.setdefault(index, ["", "", []])
        call_id = getattr(raw, "id", "") or ""
        if call_id and not slot[0]:
            slot[0] = call_id
        function = getattr(raw, "function", None)
        name = getattr(function, "name", "") or ""
        if name and not slot[1]:
            slot[1] = name
        arguments = getattr(function, "arguments", "") or ""
        if arguments:
            slot[2].append(arguments)

    def completion(self) -> Any:
        calls = None
        if self.tools:
            calls = [
                SimpleNamespace(
                    id=slot[0],
                    function=SimpleNamespace(name=slot[1], arguments="".join(slot[2])),
                )
                for _, slot in sorted(self.tools.items())
            ]
        message = SimpleNamespace(content="".join(self.text), tool_calls=calls)
        reasoning = "".join(self.reasoning).strip()
        if reasoning:
            # ``trace_of`` in :meth:`_parse` probes exactly these names.
            message.reasoning_content = reasoning
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish)],
            usage=self.usage,
        )


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

    def __init__(self, config: LLMConfig, vision: OpenAICompatClient | None = None) -> None:
        self.config = config
        #: A second provider for image-bearing requests, wired by
        #: :func:`lumi.llm.build_llm` from ``[llm.vision]`` — its own
        #: endpoint, key pool and model. None when the route is unset, in
        #: which case this client answers every request itself.
        self.vision = vision
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
                # The adapter owns the retry budget so empty responses and SDK
                # transport errors follow the same policy instead of multiplying
                # two independent retry loops.
                max_retries=0,
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
        self,
        key: str,
        variants: list[dict[str, Any]],
        on_delta: Callable[[str, str], Awaitable[None]] | None = None,
        state: _StreamState | None = None,
    ) -> tuple[Any | None, BaseException | None, bool]:
        """Call the endpoint on one key, stepping down *variants* as it refuses.

        Returns ``(completion, failure, retryable)``. ``completion`` is set only
        on success; ``failure`` carries the last error, which is the most
        informative one because it names the parameter that finally stuck.

        With *on_delta* each attempt streams instead. A rejection still
        arrives at ``create`` — before a single token exists — so the ladder
        behaves exactly as it does unstreamed; a failure *after* text went out
        is not a request failure at all and leaves as :class:`_StreamBroken`.
        """
        for index, params in enumerate(variants):
            try:
                if on_delta is None:
                    completion = await self._client_for(key).chat.completions.create(**params)
                else:
                    completion = await self._stream(
                        key, params, on_delta, state if state is not None else _StreamState()
                    )
            except _StreamBroken:
                raise
            except Exception as exc:  # noqa: BLE001 - normalised for the owner
                if rejects_param(exc) and index + 1 < len(variants):
                    log.warning(
                        "%s rejected this request: %s. Retrying without %s",
                        self.config.base_url_of(),
                        format_error(exc),
                        _dropped(params),
                    )
                    continue
                return None, exc, is_retryable(exc)
            if index:
                log.info("endpoint accepted the request once %s was dropped", _dropped(params))
            return completion, None, False
        raise LLMError("no parameter variant left to try")  # pragma: no cover

    async def _stream(
        self,
        key: str,
        params: dict[str, Any],
        on_delta: Callable[[str, str], Awaitable[None]],
        state: _StreamState,
    ) -> Any:
        """One streaming attempt: ``stream=True``, pieces forwarded as they land.

        Returns a completion-shaped object (see :class:`_Pieces`), so the reply
        is parsed by the same code an unstreamed one is.

        Failure handling splits on what the viewer has seen. Before the first
        piece of answer text, nothing has been delivered: the exception
        propagates to :meth:`_try`, which steps the parameter ladder and
        rotates keys as always. After it, a replay would show the answer
        twice, so the failure leaves as :class:`_StreamBroken` — terminal, and
        not the key's fault. A reasoning trace alone does not commit: that is
        exactly the thinking-only response the empty-completion retry exists
        for, and the retry must still be able to run (the trace it re-thinks
        with stays off the wire — see :attr:`_StreamState.sent_thinking`).
        """
        show_thinking = not state.sent_thinking
        stream = await self._client_for(key).chat.completions.create(**params, stream=True)
        pieces = _Pieces()
        try:
            async for chunk in stream:
                await self._feed(chunk, pieces, on_delta, state, show_thinking)
        except _StreamBroken:
            raise
        except Exception as exc:  # noqa: BLE001 - mid-stream transport failure
            if state.sent_text:
                raise _StreamBroken(str(exc)) from exc
            raise
        return pieces.completion()

    async def _feed(
        self,
        chunk: Any,
        pieces: _Pieces,
        on_delta: Callable[[str, str], Awaitable[None]],
        state: _StreamState,
        show_thinking: bool,
    ) -> None:
        """Fold one stream chunk into *pieces*, forwarding what a viewer should see.

        ``usage`` is read from any chunk that carries it (the SDK only attaches
        one when ``stream_options`` asked for it, which not every endpoint
        accepts — a streamed reply therefore usually reaches the context
        window as estimates rather than exact counts, same as a provider that
        never reports usage).
        """
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            pieces.usage = usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish:
            pieces.finish = finish
        delta = getattr(choice, "delta", None)
        if delta is None:
            return

        # The trace, under whichever name this provider invented for it —
        # the same fields ``trace_of`` probes on a whole message. Accumulated
        # unstripped (a fragment's whitespace is real); forwarded only when
        # no earlier attempt of this call already sent one.
        for name in TRACE_FIELDS:
            fragment = getattr(delta, name, None)
            if isinstance(fragment, str) and fragment:
                pieces.reasoning.append(fragment)
                if show_thinking:
                    state.sent_thinking = True
                    await on_delta(DELTA_THINKING, fragment)
                break

        text = getattr(delta, "content", None)
        if isinstance(text, str) and text:
            pieces.text.append(text)
            # Committed before the forward: the text exists for the reply
            # either way, so a hook that raises must still stop the ladder.
            state.sent_text = True
            await on_delta(DELTA_TEXT, text)

        for raw in getattr(delta, "tool_calls", None) or []:
            pieces.add_tool(raw)

    def _retry_same_key(self, error: BaseException) -> bool:
        """Whether a retry should wait and use this key again.

        Authentication failures should rotate to another configured key rather
        than retrying the same bad credential. A pooled 429 likewise moves to
        the next key immediately; a single-key Ollama Cloud setup waits and
        retries because there is nowhere else to go.
        """
        if not is_retryable(error):
            return False
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status in {401, 403}:
            return False
        name = type(error).__name__
        if "Authentication" in name or "Permission" in name:
            return False
        return not (status == 429 and len(self._pool) > 1)

    def _retry_delay(self, attempt: int) -> float:
        """Exponential delay before retry number *attempt* (1-based)."""
        return min(
            max(0.0, self.config.inference_retry_delay) * (2 ** (attempt - 1)),
            MAX_RETRY_DELAY_SECONDS,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> LLMReply:
        if not self._pool:
            raise LLMError(
                f"{self.config.api_key_env} is not set. Put it in .env, or point "
                f"llm.base_url at a local server and set llm.api_key_env to a dummy value."
            )

        if self.vision is not None and has_image_part(messages):
            # [llm.vision] owns this request: separate provider, endpoint and
            # key. The sub-client does its own retries and error explanation,
            # naming its own endpoint — which is the point of the route.
            return await self.vision.complete(messages, tools=tools, on_delta=on_delta)

        variants = self._variants(self._params(messages, tools))
        # Remember which model this call went out under so a failure can name
        # it and point at the config line that fixes the common mistakes.
        self._last_request = (
            request_model(messages, self.config),
            has_image_part(messages),
        )
        # One state for the whole call: what has been delivered is what every
        # later attempt (and its retry decision) has to be measured against.
        state = _StreamState() if on_delta is not None else None
        tried: list[str] = []
        failure: BaseException | None = None
        # The adapter owns one bounded retry budget for transient provider
        # failures and empty completions. No request here can have emitted a
        # tool call yet, so retrying inference cannot duplicate a side effect.
        while (key := self._pool.pick(exclude=tried)) is not None:
            tried.append(key)
            retry_number = 0
            stop_after_key = False
            while True:
                try:
                    completion, raw_error, retryable = await self._try(
                        key, variants, on_delta, state
                    )
                except _StreamBroken as exc:
                    # Text is already on its way out, so the ladder stops
                    # here — and the key did nothing wrong, so it is neither
                    # parked nor rotated. Name the endpoint anyway: it is the
                    # one fact that makes this message actionable.
                    raise LLMError(
                        "the connection dropped while the answer was streaming: "
                        f"{format_error(exc)}\n\n"
                        f"Endpoint: {self.config.base_url_of()} "
                        f"(from {self.config.where_from()})\n"
                        f"Model: {self.config.model_of()}"
                    ) from exc
                empty = False
                if raw_error is None:
                    if completion is None:
                        error = LLMError("the model returned no completion")
                    else:
                        parsed = self._parse(completion)
                        if parsed.wants_tools or parsed.text.strip():
                            self._pool.report(key, ok=True)
                            return parsed

                        # A successful HTTP response is not necessarily a usable
                        # model response. Reasoning-capable Ollama Cloud models
                        # can spend the whole response budget thinking and leave
                        # content empty. Treat it as transient, but never rotate
                        # the key.
                        error = LLMError("the model returned an empty response")
                    empty = True
                    retryable = True
                else:
                    error = raw_error

                failure = error
                # An answer that already went out commits the call: an "empty"
                # this late means the stream was whitespace or inline think
                # tags (split_think stripped them), and replaying it would
                # show the viewer the same pieces again.
                committed = state is not None and state.sent_text
                can_retry = (empty or self._retry_same_key(error)) and not committed
                if can_retry and retry_number < self.config.inference_retries:
                    retry_number += 1
                    delay = self._retry_delay(retry_number)
                    log.warning(
                        "%s from %s; retrying inference in %.1fs (%d/%d)",
                        "empty response" if empty else format_error(error),
                        self.config.base_url_of(),
                        delay,
                        retry_number,
                        self.config.inference_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                if empty:
                    failure = LLMError(
                        f"the model returned an empty response after {retry_number + 1} attempts"
                    )
                    # An empty completion is not evidence that the credential is
                    # bad, so do not park the key or rotate a pool for it.
                    self._pool.report(key, ok=True)
                    stop_after_key = True
                    break

                self._pool.report(key, ok=False, retryable=retryable)
                log.error(
                    "llm call to %s failed on key %s: %s",
                    self.config.base_url_of(),
                    mask(key),
                    format_error(error),
                )
                if not retryable:
                    stop_after_key = True
                    break
                log.warning("retrying on the next key (%d/%d tried)", len(tried), len(self._pool))
                break
            if stop_after_key:
                break

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
                arguments = (
                    json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                )
            except json.JSONDecodeError:
                log.warning("tool call %s had unparseable arguments: %r", name, raw_args)
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(id=getattr(raw, "id", "") or name, name=name, arguments=arguments)
            )

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
    optional = [
        k for k in ("reasoning_effort", "temperature", "max_completion_tokens") if k in params
    ]
    return ", ".join(optional) or "the optional parameters"


__all__ = ["OpenAICompatClient"]
