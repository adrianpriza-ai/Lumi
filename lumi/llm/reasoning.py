"""What a reasoning model does differently, in one place.

A reasoning model spends tokens before it answers, and that phase breaks three
conventions the rest of this program relies on:

1. **It rejects sampling parameters.** ``temperature`` is the one that bites
   hardest: o-series, gpt-5, DeepSeek's reasoner and gpt-oss all answer 400 to
   anything but their own default. ``llm.reasoning = true`` therefore omits it
   rather than asking the owner to remember which models are fussy.
2. **The thinking is billed as output.** It draws from the same ceiling as the
   answer, so a 2000-token budget with a long trace leaves nothing to say the
   answer in. ``llm.reasoning_tokens`` adds headroom on top of
   ``llm.max_tokens`` for exactly this.
3. **The trace comes back in a field nobody standardised.** There is no
   ``reasoning`` member on OpenAI's ``ChatCompletionMessage``, so every provider
   invented one and the SDK keeps it as an untyped extra. :func:`trace_of`
   probes the names in use.

The one thing they all agree on is that the trace does not belong in
``content`` — with the exception of the models that ignore that convention
entirely and inline ``<think>`` tags, which :func:`split_think` catches. Missing
those is how a chat ends up rendering raw XML to a human.
"""

from __future__ import annotations

import re
from typing import Any

#: Response fields that have carried a thinking trace, in probe order.
#:
#: ``reasoning_content`` is DeepSeek's and what vLLM and SGLang emit for Qwen and
#: R1 derivatives; ``reasoning`` is OpenRouter's and what Ollama's
#: OpenAI-compatible endpoint returns; ``thinking`` is Ollama's own name, which
#: some proxies pass through untranslated. Non-string values (OpenRouter also
#: offers a structured ``reasoning_details`` list) are ignored rather than
#: stringified, because a repr of a list is worse than no trace at all.
TRACE_FIELDS = ("reasoning_content", "reasoning", "thinking")

#: Tag names the open-weight chat templates use when a model inlines its
#: thinking instead of using a separate field.
_THINK_TAGS = ("think", "thinking", "reasoning")

_CLOSED_THINK_RE = re.compile(
    rf"</({'|'.join(_THINK_TAGS)})\s*>", re.IGNORECASE
)
_OPEN_THINK_RE = re.compile(
    rf"<({'|'.join(_THINK_TAGS)})(\s[^>]*)?>", re.IGNORECASE
)

#: Phrasings that mean "I have never heard of that field", as opposed to "that
#: field holds a bad value". Only the first kind is worth retrying without the
#: parameter; the second is a real error the owner has to fix.
_PARAM_REJECTION_HINTS = (
    "unsupported parameter",
    "unsupported_parameter",
    "unrecognized request argument",
    "unknown parameter",
    "unexpected keyword argument",
    "not supported with this model",
    "is not supported with this model",
)


def trace_of(message: Any) -> str:
    """The thinking trace attached to *message*, or ``""`` if there is none.

    Accepts either an SDK model object or a plain dict, because that is what the
    two ways of reaching a provider hand back.
    """
    for name in TRACE_FIELDS:
        value = message.get(name) if isinstance(message, dict) else getattr(message, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def split_think(text: str) -> tuple[str, str]:
    """Pull an inline thinking trace out of *text*. Returns ``(answer, trace)``.

    Handles both closed ``<think>…</think>`` blocks anywhere in the reply and a
    trailing unclosed one, which is what a model produces when it runs out of
    tokens mid-thought. Text with no tags comes back unchanged with an empty
    trace, so this is safe to run on every single reply.
    """
    if not text or "<" not in text:
        return text, ""

    traces: list[str] = []
    remainder = text

    while True:
        opening = _OPEN_THINK_RE.search(remainder)
        if opening is None:
            break
        closing = _CLOSED_THINK_RE.search(remainder, opening.end())
        if closing is None:
            # Unclosed: everything from the tag onwards was thinking, and the
            # model never got to an answer.
            traces.append(remainder[opening.end():])
            remainder = remainder[: opening.start()]
            break
        traces.append(remainder[opening.end() : closing.start()])
        remainder = remainder[: opening.start()] + remainder[closing.end():]

    if not traces:
        return text, ""

    trace = "\n\n".join(t.strip() for t in traces if t.strip())
    return remainder.strip(), trace


def completion_ceiling(max_tokens: int, reasoning_tokens: int) -> int:
    """How many output tokens to ask for, thinking included.

    Reasoning tokens come out of the same budget as the answer, so a limit set
    for a non-reasoning model silently truncates the *answer* rather than the
    thinking when a trace is long. Negative values are floored at zero rather
    than propagated, because a negative ceiling is rejected by every provider.
    """
    return max(0, max_tokens) + max(0, reasoning_tokens)


def rejects_param(exc: BaseException) -> bool:
    """Whether *exc* reads as "this endpoint does not accept that parameter".

    Deliberately narrow: a 400 complaining about the *value* of a parameter is a
    real problem and retrying without it would only bury the message. Duck-typed
    for the same reason :func:`lumi.llm.keypool.is_retryable` is — the SDK's
    exception classes are not imported here, so this works for any HTTP client
    an adapter might use and stays testable without a network.
    """
    message = str(exc).lower()
    if not any(word in message for word in ("parameter", "argument", "keyword", "support")):
        return False
    return any(hint in message for hint in _PARAM_REJECTION_HINTS)


__all__ = [
    "TRACE_FIELDS",
    "completion_ceiling",
    "rejects_param",
    "split_think",
    "trace_of",
]
