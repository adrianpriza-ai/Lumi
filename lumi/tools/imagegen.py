"""Generate an image from a text prompt.

The wire format lives in :mod:`lumi.llm.image_api` — the OpenAI Images API
shape, pointed at any compatible endpoint by ``tools.image_gen.base_url``.
What this tool adds on top is the project contract: the image lands under
``tools.image_gen.output_dir`` (a normal writable project path the model can
then read, edit or process like any other file) and travels to the chat as an
artifact, the same file-delivery path ``files`` uses for uploads out.

Generation is delegated, not local: no diffusion model runs in-process. The
tool is available only when an endpoint and key are configured, and
``available()`` says which piece is missing so ``/doctor`` and ``/tools`` can
show the owner why.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactStore
from ..config import Config
from ..llm.base import LLMError
from ..llm.image_api import ImageGenClient
from ..paths import is_within, relative_to_root
from ..util.log import get_logger
from .base import Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)

#: Returned MIME -> file suffix. Anything else a provider invents stays
#: unnamed: better a clear failure than a mystery extension on disk.
MIME_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

#: A filename stem built from the prompt: readable, filesystem-safe, and
#: bounded — the prompt itself can be hundreds of chars of arbitrary text.
STEM_CHARS = 40


def _stem(prompt: str) -> str:
    """A short safe stem from *prompt*, plus a timestamp so files sort."""
    words = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    stem = "-".join(w for w in words.split("-") if w)[:STEM_CHARS].rstrip("-") or "image"
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"


class ImageGenTool(Tool):
    name = "image_gen"
    description = (
        "Generate an image from a text prompt with a text-to-image model "
        "(config: tools.image_gen). Use it when the owner asks for a picture, "
        "illustration, logo draft or diagram-of-a-scene. The finished image is "
        "saved into the project and sent to the chat. NOT for reading or "
        "editing existing images — `read_file` looks at them, `files` edits "
        "them — and not for charts from data, which plot code renders better."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "What to draw. Be specific: subject, style, mood, "
                    "composition. The prompt goes to the image model verbatim."
                ),
            },
            "size": {
                "type": "string",
                "description": (
                    'Image size as "WIDTHxHEIGHT", e.g. "1024x1024" or '
                    '"1792x1024". Empty = the configured default '
                    "(tools.image_gen.size). Providers reject sizes they "
                    "do not know; their exact words come back if so."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self._config = config
        self._settings = config.tools.image_gen
        #: Lazily created: the store needs the bot's upload cap, which lives
        #: in the bot config section.
        self._store: ArtifactStore | None = None
        #: Built on first use — a fresh client owned by this tool, so image
        #: API retries never tangle with the chat loop's state. Injected in
        #: tests.
        self._api: ImageGenClient | None = None

    @property
    def store(self) -> ArtifactStore:
        if self._store is None:
            self._store = ArtifactStore(
                self._config.root,
                max_bytes=max(1, self._config.bot.max_upload_mb) * 1024 * 1024,
            )
        return self._store

    def _client(self) -> ImageGenClient:
        if self._api is None:
            self._api = ImageGenClient(self._settings)
        return self._api

    # -- availability ------------------------------------------------------ #

    def available(self) -> tuple[bool, str]:
        if not self._settings.enabled:
            return False, "disabled by tools.image_gen.enabled"
        if problem := self._settings.endpoint_problem():
            return False, problem
        # The hosted API bills per call, so it needs a key; a custom endpoint
        # may be self-hosted and keyless, and that is a usable configuration.
        if self._settings.needs_key() and not self._settings.api_key():
            return (
                False,
                f"{self._settings.api_key_env} is unset (needed by tools.image_gen.provider)",
            )
        return True, ""

    # -- dispatch ---------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return ToolResult.failure("`prompt` is required — describe what to draw.")
        try:
            mime, data = await self._client().generate(prompt, str(arguments.get("size", "")))
        except LLMError as exc:
            raise ToolError(str(exc)) from exc

        suffix = MIME_SUFFIXES.get(mime)
        if suffix is None:
            raise ToolError(
                f"the image endpoint returned {mime!r}, which I don't know how to save."
            )
        path = self._save(prompt, suffix, data)
        artifact = self.store.send(path, origin=self.name, project_root=self._config.root)
        return ToolResult(
            text=(
                f"Generated {relative_to_root(path)} "
                f"({len(data) / 1024:.0f} KB, {mime}). It has been sent to the chat; "
                "the file also stays in the project."
            ),
            summary=f"{relative_to_root(path)} generated",
            data={"path": relative_to_root(path), "bytes": len(data), "mime": mime},
            artifacts=[artifact],
        )

    # -- the project side ---------------------------------------------------- #

    def _save(self, prompt: str, suffix: str, data: bytes) -> Path:
        """Write the image under the configured output dir, inside the project."""
        root = self._config.root
        out_dir = (root / self._settings.output_dir).resolve()
        if not is_within(out_dir, root.resolve()):
            raise ToolError(
                f"tools.image_gen.output_dir {self._settings.output_dir!r} is outside "
                "the project; generated images must live under the project root."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{_stem(prompt)}-{uuid.uuid4().hex[:6]}{suffix}"
        try:
            path.write_bytes(data)
        except OSError as exc:
            raise ToolError(f"could not save the generated image: {exc}") from exc
        log.info("image saved: %s (%d bytes)", relative_to_root(path), len(data))
        return path


__all__ = ["ImageGenTool", "MIME_SUFFIXES"]
