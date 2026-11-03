"""LLM client contract.

Kept deliberately small: one method, plain dicts in and out. The OpenAI-compatible
adapter in :mod:`lumi.llm.openai_compat` is the only implementation that ships,
but anything speaking the OpenAI chat-completions API works by changing two
config values.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """The model call failed in a way worth showing the owner."""


#: Keys of ``LLMReply.usage`` — the client contract, not one adapter's habit.
#: Adapters fill them from the provider's usage block; the agent loop and the
#: context window read them back by these names. ``USAGE_PROMPT`` in particular
#: feeds :meth:`lumi.context.ContextWindow.observe`'s calibration, which a
#: rename on either side would silently disable.
USAGE_PROMPT = "prompt"
USAGE_COMPLETION = "completion"
USAGE_REASONING = "reasoning"

#: The two kinds :meth:`LLMClient.complete`'s ``on_delta`` is called with —
#: answer text and reasoning trace. They double as the protocol event names in
#: :mod:`lumi.tui`, so one vocabulary covers the client and the wire.
DELTA_TEXT = "delta"
DELTA_THINKING = "thinking_delta"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def arguments_json(self) -> str:
        import json

        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass(slots=True)
class LLMReply:
    text: str = ""
    #: The thinking trace, when the model returned one. Separate from ``text``
    #: because that is how every provider that supports reasoning actually sends
    #: it; see :mod:`lumi.llm.reasoning`. Empty for models that do not reason.
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(abc.ABC):
    name: str = "llm"

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> LLMReply:
        """One round trip. *messages* is the OpenAI chat-completions format.

        When *on_delta* is given the adapter streams: it is awaited as
        ``on_delta(kind, text)`` for each piece of the reply as it arrives,
        *kind* being :data:`DELTA_TEXT` for answer text and
        :data:`DELTA_THINKING` for a reasoning trace. Two rules keep a live
        transcript honest across the retry ladder in
        :mod:`lumi.llm.openai_compat`:

        - the first delivered piece of answer text **commits** the call — no
          retry, key rotation or parameter step-down follows it, so text a
          viewer has seen is never delivered a second time;
        - a reasoning trace is delivered by at most one attempt per call, so
          the retry after an empty (thinking-only) completion cannot replay a
          trace that already went out.

        Without a callback nothing streams: the reply arrives whole, which is
        what every caller without a live channel gets.
        """

    def describe(self) -> str:
        return self.name


__all__ = [
    "LLMClient",
    "LLMReply",
    "ToolCall",
    "LLMError",
    "USAGE_PROMPT",
    "USAGE_COMPLETION",
    "USAGE_REASONING",
    "DELTA_TEXT",
    "DELTA_THINKING",
]
