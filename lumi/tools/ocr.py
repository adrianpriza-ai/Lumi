"""Read text from PDFs and document images.

With ``tools.ocr.provider`` set to ``mistral`` or ``custom``, documents go to
a Mistral-compatible ``POST /ocr`` API. Without it, pypdfium2 extracts embedded
PDF text locally; scanned pages use ``llm.ocr_model`` or the default model.
Local image files always use the photo route (vision when set, else default).
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.base import LLMError
from ..llm.ocr_api import CONTENT_TYPES, OCRDocumentClient
from ..llm.openai_compat import OpenAICompatClient
from ..paths import is_within, project_root, relative_to_root, resolve
from ..util.log import get_logger
from ..util.net import validate_fetch_url
from ..util.text import fence_untrusted, truncate
from .base import Tool, ToolContext, ToolError, ToolResult
from .files import foreign_uploads
from .safety import read_guard

log = get_logger(__name__)

#: A page yielding fewer than this many embedded chars counts as a scan:
#: footers and page numbers are not text, and rendering such a page is
#: cheaper than apologising for a missing paragraph later.
MIN_EMBEDDED_CHARS = 20

#: Per-page instruction when transcribing a rendered page. Short on purpose — the page
#: is the payload, the prompt is only the job.
OCR_PROMPT = (
    "Transcribe all text in this image, in reading order. Keep the original "
    "language; do not summarise, translate or comment. If there is no text, "
    "reply with exactly: (no text)"
)

#: Image formats the vision path accepts, by extension.
IMAGE_MIMES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: Ceiling on one call's text output (~30k tokens). A huge scan is still
#: readable page by page with max_pages; one colossal tool result is how a
#: conversation evicts its own history.
MAX_TEXT_CHARS = 120_000

#: Refuse (without reading) files bigger than this. Both paths charge by the
#: byte — the API upload and the context window alike — so a runaway file
#: gets a named limit instead of a bill.
MAX_FILE_MB = 50


class OCRTool(Tool):
    name = "ocr"
    description = (
        "Extract text from a PDF, scan or document image (receipt, screenshot, "
        "scanned page, paper). Works on project files and — with a dedicated OCR "
        "provider — on public document URLs directly. Use it when the owner sends "
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
                    "URL to a document (URLs need a dedicated OCR provider)."
                ),
            },
            "max_pages": {
                "type": "integer",
                "description": (
                    "Stop after this many pages (local path only; default from "
                    "tools.ocr.max_pages)."
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self._config = config
        #: Built on first use — a fresh key pool and session, owned by this
        #: tool, so OCR retries never tangle with the chat loop's state.
        #: ``_llm`` serves scanned PDF pages (the ocr route); ``_image_llm``
        #: serves image files (the vision route, else the default).
        self._llm: Any = None
        self._image_llm: Any = None
        self._api: Any = None

    # -- availability ------------------------------------------------------ #

    def available(self) -> tuple[bool, str]:
        ocr = self._config.tools.ocr
        if not ocr.enabled:
            return False, "disabled by tools.ocr.enabled"
        if problem := ocr.endpoint_problem():
            return False, problem
        if ocr.provider_of() and not ocr.api_key():
            return False, f"{ocr.api_key_env} is unset (needed by tools.ocr.provider)"
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
            return await self._url(raw)
        return await self._file(Path(raw), arguments, ctx)

    async def _url(self, url: str) -> ToolResult:
        """OCR a public document URL — only the dedicated API can do this."""
        ocr = self._config.tools.ocr
        if not ocr.provider_of():
            return ToolResult.failure(
                "Reading a document from a URL needs a dedicated OCR provider. "
                'Set `[tools.ocr] provider = "mistral"` (and MISTRAL_API_KEY) in '
                "config.toml, or download the file into the project and call `ocr` on its path."
            )
        try:
            pages = await self._api_client().ocr(document_url=url)
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
        if (foreign := foreign_uploads(path, project_root(), ctx.chat_id)) is not None:
            return ToolResult.failure(
                f"workspace/uploads/{foreign} belongs to another chat; I only read this chat's uploads."
            )
        if (guard := read_guard(path)) is not None:
            return ToolResult.failure(
                f"{guard} holds credentials, so it is not sent for OCR."
            )
        if not path.is_file():
            return ToolResult.failure(f"no such file: {relative_to_root(path)}")
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            return ToolResult.failure(
                f"{relative_to_root(path)} is over {MAX_FILE_MB} MB — too big to OCR in one call."
            )

        suffix = path.suffix.lower()
        ocr = self._config.tools.ocr
        is_pdf = suffix == ".pdf" or _sniffs_pdf(path)
        # PDF magic takes precedence over a misleading image extension.
        mime = IMAGE_MIMES.get(suffix)
        if mime and not is_pdf:
            return await self._image_locally(path, mime)

        if ocr.provider_of():
            if not is_pdf and suffix != ".docx":
                return ToolResult.failure(
                    f"{relative_to_root(path)} is not a supported OCR document. "
                    "For text and code files use `read_file` instead."
                )
            try:
                pages = await self._api_client().ocr(
                    document_b64=_read_b64(path),
                    content_type="application/pdf" if is_pdf else CONTENT_TYPES[suffix],
                )
            except LLMError as exc:
                raise ToolError(str(exc)) from exc
            return _result(pages, source=relative_to_root(path))

        max_pages = _max_pages(arguments.get("max_pages"), ocr.max_pages)
        if is_pdf:
            return await self._pdf_locally(path, max_pages)
        return ToolResult.failure(
            f"{relative_to_root(path)} is not a PDF or image. "
            "For text and code files use `read_file` instead."
        )

    # -- the local path ----------------------------------------------------- #

    def _ocr_client(self) -> Any:
        if self._llm is None:
            # [llm.ocr] decides provider, endpoint, key and model — a route,
            # exactly like [llm.vision]. Falls back to the *default* model:
            # a scanned page is document work, and a configured vision model
            # for photos must not change what reads a PDF.
            self._llm = OpenAICompatClient(self._config.llm.ocr_route_of())
        return self._llm

    def _image_client(self) -> Any:
        if self._image_llm is None:
            # An image file is an image: the vision route when one is set,
            # else the default — the same rule a photo in chat follows.
            route = self._config.llm.vision_route_of()
            self._image_llm = OpenAICompatClient(route or self._config.llm)
        return self._image_llm

    def _api_client(self) -> OCRDocumentClient:
        if self._api is None:
            self._api = OCRDocumentClient(self._config.tools.ocr)
        return self._api

    async def _pdf_locally(self, path: Path, max_pages: int) -> ToolResult:
        pdf = _open_pdf(path)
        try:
            total = len(pdf)
            if total == 0:
                return ToolResult.failure(f"{relative_to_root(path)} has no pages.")
            limit = total if not max_pages else min(total, max_pages)

            # Two passes, and the order is the whole point: the free pass reads
            # what the PDF already contains; only what it could not read is
            # rendered and priced. Most PDFs never reach the second pass.
            embedded: dict[int, str] = {}
            scanned: list[int] = []
            for index in range(limit):
                text = _embedded_text(pdf[index]).strip()
                if len(text) >= MIN_EMBEDDED_CHARS:
                    embedded[index] = text
                else:
                    scanned.append(index)

            rendered: dict[int, str] = {}
            for index in scanned:
                image_b64 = _render_page(pdf[index], self._config.tools.ocr.render_scale)
                rendered[index] = (
                    await self._transcribe(image_b64, client=self._ocr_client())
                ).strip()

            parts = [
                f"[page {index + 1}]\n{embedded.get(index) or rendered.get(index, '')}"
                for index in range(limit)
            ]
            note = (
                f" ({len(scanned)} of {limit} page(s) scanned with a vision model)"
                if scanned
                else ""
            )
            text = fence_untrusted(truncate("\n\n".join(parts), MAX_TEXT_CHARS)) + note
            if total > limit:
                text += (
                    f"\n\n[stopped at page {limit}; {total - limit} more page(s) "
                    "not read — call again with a larger max_pages]"
                )
            return ToolResult(
                text=text,
                summary=f"{relative_to_root(path)}: {limit} page(s), {len(scanned)} scanned",
                data={"pages": limit, "scanned": len(scanned), "total_pages": total},
            )
        finally:
            pdf.close()

    async def _image_locally(self, path: Path, mime: str) -> ToolResult:
        image_b64 = _read_b64(path)
        text = (await self._transcribe(image_b64, client=self._image_client(), mime=mime)).strip()
        return ToolResult(
            text=fence_untrusted(text or "(no text found in the image)"),
            summary=f"{relative_to_root(path)}: transcribed",
            data={"pages": 1, "scanned": 1},
        )

    async def _transcribe(self, image_b64: str, *, client: Any, mime: str = "image/png") -> str:
        try:
            reply = await client.complete(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                        ],
                    }
                ]
            )
        except LLMError as exc:
            raise ToolError(f"the model could not transcribe a page: {exc}") from exc
        return reply.text or "(no text)"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


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


def _max_pages(requested: Any, configured: int) -> int:
    """The page ceiling for this call: the model's ask, clamped by config. 0 = all."""
    try:
        value = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        value = configured
    return max(0, value)


def _read_b64(path: Path) -> str:
    """A file's bytes, base64 — sync on purpose, in a sync helper: ASYNC240
    exists so an async function does not block the loop on its own, and the
    files tool reads the same way for the same reason (a local read is what
    the tool *is*; a thread would only move the wait)."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _sniffs_pdf(path: Path) -> bool:
    """True when the file starts with a PDF magic number, whatever it is called."""
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _open_pdf(path: Path) -> Any:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ToolError(
            "pypdfium2 is not installed — the local PDF path needs it "
            "(pip install pypdfium2), or configure [tools.ocr] provider."
        ) from exc
    try:
        return pdfium.PdfDocument(str(path))
    except Exception as exc:  # noqa: BLE001 - pdfium raises whatever it likes
        raise ToolError(f"could not open {relative_to_root(path)} as a PDF: {exc}") from exc


def _embedded_text(page: Any) -> str:
    """The text the PDF already contains — the free pass."""
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range()
    finally:
        textpage.close()


def _render_page(page: Any, scale: float) -> str:
    """Render one page to a base64 PNG. This is the priced path."""
    try:
        from PIL import Image  # noqa: F401 - to_pil needs it; fail early, clearly
    except ImportError as exc:
        raise ToolError(
            "Pillow is not installed — rendering scanned pages needs it (pip install Pillow), "
            "or configure [tools.ocr] provider."
        ) from exc
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil()
    finally:
        bitmap.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = ["OCRTool"]
