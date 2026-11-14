"""The image-generation client: an OpenAI-style ``POST /images/generations``.

Not a chat model: a prompt goes in, image bytes come back. That is a different
conversation than chat completions, so it gets its own client rather than a
prompt — the same reasoning as :mod:`lumi.llm.ocr_api`. The wire format is the
OpenAI Images API; any compatible service works by pointing
``tools.image_gen.base_url`` at it.

.. code-block:: bash

    curl https://api.openai.com/v1/images/generations \\
      -H "Authorization: Bearer $OPENAI_API_KEY" \\
      -d '{"model": "gpt-image-1", "prompt": "a red cube", "n": 1}'

A provider answers with ``data[0].b64_json`` (OpenAI's gpt-image-1, and most
compatible servers) or ``data[0].url`` (dall-e-3's default) — both are
accepted here, so switching model or endpoint is a config edit, not a tool
change.
"""

from __future__ import annotations

import base64
from typing import Any

from ..config import ImageGenConfig
from ..util.log import get_logger
from ..util.text import format_error, truncate
from .base import LLMError

log = get_logger(__name__)

#: Characters of a failed response body quoted back to the model. Enough to
#: name the problem ("content policy violation", "model not found"), short
#: enough to not eat the context.
ERROR_BODY_CHARS = 400

#: Ceiling on a downloaded image (URL responses). The artifact layer has its
#: own cap; this one stops the download itself from pulling something huge.
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


class ImageGenClient:
    """One HTTP call: prompt in, image bytes out.

    Deliberately not built on the OpenAI SDK — this is not chat completions,
    and pretending it is would mean bending someone else's retry policy around
    a different endpoint. One request, one clear failure; the tool above it
    decides what to say.
    """

    def __init__(self, config: ImageGenConfig) -> None:
        self.config = config

    @property
    def endpoint(self) -> str:
        """The full URL: ``<base_url>/images/generations``, whatever the slash state."""
        if problem := self.config.config_problem():
            raise LLMError(problem)
        return self.config.base_url_of() + "/images/generations"

    def describe(self) -> str:
        return f"{self.config.model_of()} via {self.endpoint}"

    async def generate(self, prompt: str, size: str = "") -> tuple[str, bytes]:
        """Generate one image. Returns ``(mime, bytes)``.

        Raises :class:`LLMError` with the endpoint, status and a snippet of
        the provider's own response body — an image API fails in specific
        ways (content policy, unknown size) and the owner needs the exact
        words, not a guess.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - openai always pulls httpx in
            raise LLMError("the httpx package is not installed") from exc

        key = self.config.api_key()
        # A key is mandatory only where the config says the endpoint cannot
        # work without one (the hosted API). A keyless custom endpoint is a
        # real configuration — a LAN server, a local queue — so no header
        # rather than a refusal; if that endpoint does want auth, its own 401
        # names the problem better than a guess here.
        if not key and self.config.needs_key():
            raise LLMError(
                f"{self.config.api_key_env} is not set — the image endpoint needs it in .env."
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}

        payload: dict[str, Any] = {
            "model": self.config.model_of(),
            "prompt": prompt,
            "n": 1,
        }
        if size := size.strip() or self.config.size.strip():
            payload["size"] = size

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=headers,
            ) as http:
                response = await http.post(self.endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach {self.endpoint}: {format_error(exc)}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"{self.endpoint} returned HTTP {response.status_code}: "
                f"{truncate(response.text, ERROR_BODY_CHARS)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.endpoint} returned a non-JSON body") from exc

        items = data.get("data") or []
        if not items:
            raise LLMError(f"{self.endpoint} returned no images")
        first = items[0]
        if b64 := first.get("b64_json"):
            return "image/png", base64.b64decode(b64)
        if url := first.get("url"):
            return await self._download(str(url))
        raise LLMError(
            f"{self.endpoint} returned an image with neither b64_json nor url: "
            f"{truncate(str(first), ERROR_BODY_CHARS)}"
        )

    async def _download(self, url: str) -> tuple[str, bytes]:
        """Fetch an image the provider hosted at a URL instead of inlining."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - openai always pulls httpx in
            raise LLMError("the httpx package is not installed") from exc

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds, follow_redirects=True
            ) as http:
                response = await http.get(url)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"could not download the generated image from {url}: {format_error(exc)}"
            ) from exc
        if response.status_code >= 400:
            raise LLMError(
                f"the image host returned HTTP {response.status_code} for the generated image"
            )
        if len(response.content) > MAX_DOWNLOAD_BYTES:
            raise LLMError(
                f"the generated image is {len(response.content) / (1024 * 1024):.1f} MB, "
                "over the download cap"
            )
        mime = response.headers.get("content-type", "image/png").split(";")[0].strip()
        if not mime.startswith("image/"):
            raise LLMError(f"the image host returned {mime!r}, not an image")
        return mime, response.content


__all__ = ["ImageGenClient", "MAX_DOWNLOAD_BYTES"]
