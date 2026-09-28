"""LLM client contract.

Kept deliberately small: one method, plain dicts in and out. The OpenAI-compatible
adapter in :mod:`lumi.llm.openai_compat` is the only implementation that ships,
but anything speaking the OpenAI chat-completions API works by changing two
config values.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """The model call failed in a way worth showing the owner."""


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
    ) -> LLMReply:
        """One round trip. *messages* is the OpenAI chat-completions format."""

    def describe(self) -> str:
        return self.name


__all__ = ["LLMClient", "LLMReply", "ToolCall", "LLMError"]
