"""LLM clients.

The registry is a dict rather than a plugin scan: with one adapter that covers
every OpenAI-compatible endpoint, a scan would be ceremony. When a second
adapter earns its place, add it to :data:`PROVIDERS` and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Config, LLMConfig
from ..util.log import get_logger
from .base import LLMClient, LLMError, LLMReply, ToolCall
from .openai_compat import OpenAICompatClient

log = get_logger(__name__)

PROVIDERS: dict[str, Callable[[LLMConfig], LLMClient]] = {
    "openai": OpenAICompatClient,
    "openai-compatible": OpenAICompatClient,
    "ollama": OpenAICompatClient,  # same wire format, different default base_url
    "lmstudio": OpenAICompatClient,
    "vllm": OpenAICompatClient,
    "groq": OpenAICompatClient,
    "openrouter": OpenAICompatClient,
    "together": OpenAICompatClient,
}


def build_llm(config: Config) -> LLMClient:
    """Instantiate the client named by ``llm.provider``."""
    key = config.llm.provider.strip().lower()
    factory = PROVIDERS.get(key)
    if factory is None:
        raise LLMError(
            f"unknown llm.provider {config.llm.provider!r}. Known: {', '.join(sorted(PROVIDERS))}"
        )
    return factory(config.llm)


__all__ = [
    "LLMClient",
    "LLMReply",
    "LLMError",
    "ToolCall",
    "OpenAICompatClient",
    "build_llm",
    "PROVIDERS",
]
