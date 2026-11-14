"""LLM clients.

One adapter covers every endpoint that speaks the OpenAI chat-completions
wire format — which is all of them this bot targets: OpenAI, OpenRouter,
Ollama, vLLM, LM Studio, Groq, a LAN server. ``llm.base_url`` picks the
endpoint; there is no provider preset to choose, so there is no registry to
scan either.
"""

from __future__ import annotations

from ..config import Config
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


def build_llm(config: Config) -> LLMClient:
    """Instantiate the client ``[llm]`` describes, plus its routes.

    ``[llm.vision]`` builds a second client — its own endpoint, key pool and
    model — attached to answer every image-bearing request. That is what
    makes "text model on endpoint A with key A, vision model on endpoint B
    with key B" pure configuration: two lines in ``config.toml``, two env
    vars. Both clients are the same adapter, so nothing here can fail on a
    name the registry does not know — there is no registry.
    """
    client = OpenAICompatClient(config.llm)
    if (route := config.llm.vision_route_of()) is not None:
        client.vision = OpenAICompatClient(route)
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
    "STRATEGIES",
]
