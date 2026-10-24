"""LLM clients.

The registry is a dict rather than a plugin scan: with one adapter that covers
every OpenAI-compatible endpoint, a scan would be ceremony. When a second
adapter earns its place, add it to :data:`PROVIDERS` and nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ..config import Config, LLMConfig
from ..util.log import get_logger
from .base import (
    USAGE_COMPLETION,
    USAGE_PROMPT,
    USAGE_REASONING,
    LLMClient,
    LLMError,
    LLMReply,
    ToolCall,
)
from .keypool import STRATEGIES, KeyPool
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
    """Instantiate the client named by ``llm.provider``, plus its routes.

    ``[llm.vision]`` builds a second client — its own provider, endpoint, key
    pool and model — attached to answer every image-bearing request. That is
    what makes "text model on provider A with key A, vision model on provider B
    with key B" pure configuration: two lines in ``config.toml``, two env vars.
    An unknown provider in either block raises here, at boot, rather than on
    the first photo that arrives.
    """
    key = config.llm.provider.strip().lower()
    factory = PROVIDERS.get(key)
    if factory is None:
        raise LLMError(
            f"unknown llm.provider {config.llm.provider!r}. Known: {', '.join(sorted(PROVIDERS))}"
        )
    client = factory(config.llm)
    route = config.llm.vision_route_of()
    if route is not None:
        vision_key = route.provider.strip().lower()
        vision_factory = PROVIDERS.get(vision_key)
        if vision_factory is None:
            raise LLMError(
                f"unknown llm.vision.provider {route.provider!r}. "
                f"Known: {', '.join(sorted(PROVIDERS))}"
            )
        cast(OpenAICompatClient, client).vision = cast(OpenAICompatClient, vision_factory(route))
        log.info(
            "vision route ready: model=%s endpoint=%s key=%s",
            route.vision_model_of(),
            route.base_url_of(),
            route.api_key_env,
        )
    return client


__all__ = [
    "LLMClient",
    "LLMReply",
    "LLMError",
    "ToolCall",
    "USAGE_PROMPT",
    "USAGE_COMPLETION",
    "USAGE_REASONING",
    "KeyPool",
    "OpenAICompatClient",
    "build_llm",
    "PROVIDERS",
    "STRATEGIES",
]
