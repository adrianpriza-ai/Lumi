"""Read text from PDFs and document images.

With ``tools.ocr.base_url`` set, documents go to a Mistral-compatible
``POST /ocr`` API. Without it, pypdfium2 extracts embedded PDF text locally;
scanned pages use ``llm.file_model`` or the default model.
Local image files always use the photo route (vision when set, else default).

The local machinery lives in :mod:`lumi.tools.docread`, shared with
``read_file`` and ``files: read`` — one implementation of "what a PDF or an
image turns into", so the tools cannot drift apart. What this tool adds on
top is the URL route (a dedicated OCR endpoint can fetch a public document
itself) and the transcription contract: *text out*, never an attached image.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.base import LLMError
from ..llm.ocr_api import CONTENT_TYPES, OCRDocumentClient
from ..paths import is_within, project_root, relative_to_root, resolve
from ..util.net import validate_fetch_url
from ..util.text import fence_untrusted, truncate
from .base import Tool, ToolContext, ToolError, ToolResult
from .docread import (
    IMAGE_MIMES,
    MAX_TEXT_CHARS,
    DocumentReader,
    max_pages_clamped,
)
from .docread import (
    read_b64 as _read_b64,
)
from .docread import (
    sniffs_pdf as _sniffs_pdf,
)


def _missing_local_library() -> tuple[str, str] | None:
    """``(package, pip name)`` of a missing local-PDF library, or ``None``.

    The local route opens every PDF with pypdfium2 and renders scanned pages
    with Pillow. Both are declared dependencies, so this only fires in a
    partial install — where the honest answer is "unavailable", not a promise
    of a transcription the call cannot deliver.
    """
    try:
        import pypdfium2  # noqa: F401 - the PDF reader
    except ImportError:
        return "pypdfium2", "pypdfium2"
    try:
        import PIL  # noqa: F401 - renders scanned pages
    except ImportError:
        return "Pillow", "Pillow"
    return None


def _result(pages: list[tuple[int, str]], *, source: str) -> ToolResult:
    """Assemble ``(page_number, markdown)`` pairs into one tool result."""
    if not pages:
        return ToolResult.failure(f"the OCR API returned no pages for {source}.")
    text = fence_untrusted(
        truncate(
            "\n\n".join(f"[page {number}]\n{content}" for number, content in pages),
            MAX_TEXT_CHARS,
        )
    )
    return ToolResult(
        text=text,
        summary=f"{source}: {len(pages)} page(s) via OCR API",
        data={"pages": len(pages), "scanned": len(pages)},
    )


class OCRTool(Tool):
    name = "ocr"
    description = (
        "Extract text from a PDF, scan or document image (receipt, screenshot, "
        "scanned page, paper). Works on project files and — with a dedicated OCR "
        "endpoint — on public document URLs directly. Use it when the owner sends "
        "or references a PDF and wants its contents. NOT for code or plain-text "
        "files: `read_file` handles those without a model call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the PDF/image file inside the project, or an http(s) "
                    "URL to a document (URLs need tools.ocr.base_url set)."
                ),
            },
            "max_pages": {
                "type": "integer",
                "description": (
                    "Stop after this many pages. Local path: default from "
                    "tools.ocr.max_pages, read the rest with a later call. "
                    "Dedicated OCR API: only the first N pages are sent."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self._config = config
        #: The shared PDF/image machinery — clients, guards, rendering, paging.
        self._reader = DocumentReader(config)
        #: Built on first use — a fresh client owned by this tool, so OCR API
        #: retries never tangle with the chat loop's state.
        self._api: Any = None

    # The vision clients live on the reader now; these two properties keep the
    # injection point tests (and callers) used: assigning ``tool._llm`` still
    # swaps the client that transcribes scanned pages.
    @property
    def _llm(self) -> Any:
        return self._reader._llm

    @_llm.setter
    def _llm(self, value: Any) -> None:
        self._reader._llm = value

    @property
    def _image_llm(self) -> Any:
        return self._reader._image_llm

    @_image_llm.setter
    def _image_llm(self, value: Any) -> None:
        self._reader._image_llm = value

    def _ocr_client(self) -> Any:
        return self._reader._ocr_client()

    def _image_client(self) -> Any:
        return self._reader._image_client()

    # -- availability ------------------------------------------------------ #

    def available(self) -> tuple[bool, str]:
        ocr = self._config.tools.ocr
        if not ocr.enabled:
            return False, "disabled by tools.ocr.enabled"
        if problem := ocr.config_problem():
            return False, problem
        # The hosted API bills per call, so it needs a key; a custom endpoint
        # may be self-hosted and keyless, and that is a usable configuration.
        if ocr.needs_key() and not ocr.api_key():
            return False, f"{ocr.api_key_env} is unset (needed by the OCR endpoint)"
        # Only the dedicated API is an httpx client. The local path uses
        # pypdfium2 and the vision model, so a missing httpx must not switch
        # off PDF and image reading that would still work.
        if ocr.uses_api():
            try:
                import httpx  # noqa: F401
            except ImportError:
                return False, "httpx is not installed; the OCR endpoint needs it"
            return True, ""
        if missing := _missing_local_library():
            package, install = missing
            return False, (
                f"{package} is not installed; the local PDF path needs it "
                f"(pip install {install}), or set [tools.ocr] base_url"
            )
        return True, ""

    # -- dispatch ---------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = str(arguments.get("path", "")).strip()
        if not raw:
            return ToolResult.failure("`path` is required — a project file path or an http(s) URL.")
        if raw.lower().startswith(("http://", "https://")):
            if reason := await asyncio.to_thread(validate_fetch_url, raw):
                return ToolResult.failure(
                    f"refusing to OCR {raw!r}: {reason} (only globally reachable http/https hosts)"
                )
            return await self._url(raw, arguments)
        return await self._file(Path(raw), arguments, ctx)

    @staticmethod
    def _requested_pages(arguments: dict[str, Any]) -> list[int] | None:
        """0-based pages for the dedicated API, or ``None`` for the whole document.

        Only the model's explicit ``max_pages`` ask limits the API path — the
        configured ``tools.ocr.max_pages`` default is a local-render knob, and
        silently capping a document the endpoint pages itself would hide most
        of a long PDF behind a number the model never asked for.
        """
        limit = max_pages_clamped(arguments.get("max_pages"), 0)
        return list(range(limit)) if limit > 0 else None

    async def _url(self, url: str, arguments: dict[str, Any]) -> ToolResult:
        """OCR a public document URL — only the dedicated API can do this."""
        ocr = self._config.tools.ocr
        if not ocr.uses_api():
            return ToolResult.failure(
                "Reading a document from a URL needs a dedicated OCR endpoint. "
                'Set `[tools.ocr] base_url = "https://api.mistral.ai/v1"` (and '
                "MISTRAL_API_KEY) in config.toml, or download the file into the "
                "project and call `ocr` on its path."
            )
        try:
            pages = await self._api_client().ocr(
                document_url=url, pages=self._requested_pages(arguments)
            )
        except LLMError as exc:
            raise ToolError(str(exc)) from exc
        return _result(pages, source=url)

    async def _file(self, path: Path, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not is_within(path := resolve(path), project_root()):
            return ToolResult.failure(
                f"{relative_to_root(path)} is outside the project; I only read files inside it."
            )
        # Same read boundary as the files tool: another chat's uploads are not
        # this chat's, and a credential file is not OCR material — its
        # transcription would leave for a third-party API with no tap.
        if (problem := self._reader.read_problem(path, ctx.chat_id)) is not None:
            return ToolResult.failure(
                f"{relative_to_root(path)} cannot be read as media: {problem}."
            )

        suffix = path.suffix.lower()
        ocr = self._config.tools.ocr
        is_pdf = suffix == ".pdf" or _sniffs_pdf(path)
        # PDF magic takes precedence over a misleading image extension.
        mime = IMAGE_MIMES.get(suffix)
        if mime and not is_pdf:
            return await self._image_locally(path, mime)

        if ocr.uses_api():
            if not is_pdf and suffix != ".docx":
                return ToolResult.failure(
                    f"{relative_to_root(path)} is not a supported OCR document. "
                    "For text and code files use `read_file` instead."
                )
            try:
                pages = await self._api_client().ocr(
                    document_b64=_read_b64(path),
                    content_type="application/pdf" if is_pdf else CONTENT_TYPES[suffix],
                    pages=self._requested_pages(arguments),
                )
            except LLMError as exc:
                raise ToolError(str(exc)) from exc
            return _result(pages, source=relative_to_root(path))

        max_pages = max_pages_clamped(arguments.get("max_pages"), ocr.max_pages)
        if is_pdf:
            return await self._reader.read_pdf(path, start_page=1, max_pages=max_pages)
        return ToolResult.failure(
            f"{relative_to_root(path)} is not a PDF or image. "
            "For text and code files use `read_file` instead."
        )

    # -- the local image path ----------------------------------------------- #

    def _api_client(self) -> OCRDocumentClient:
        if self._api is None:
            self._api = OCRDocumentClient(self._config.tools.ocr)
        return self._api

    async def _image_locally(self, path: Path, mime: str) -> ToolResult:
        # The ocr tool's contract is text out: a transcription, never an
        # attached image — attaching is read_file's route. An image file is an
        # image: the vision route when one is set, else the default.
        text = (
            await self._reader.transcribe(_read_b64(path), client=self._image_client(), mime=mime)
        ).strip()
        return ToolResult(
            text=fence_untrusted(text or "(no text found in the image)"),
            summary=f"{relative_to_root(path)}: transcribed",
            data={"pages": 1, "scanned": 1},
        )


__all__ = ["OCRTool"]
