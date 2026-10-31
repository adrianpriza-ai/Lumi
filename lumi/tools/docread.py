"""Reading PDFs and images: the machinery the ``ocr`` tool and the read tools share.

One body of code decides what a PDF or an image turns into, because every
caller wants the same two routes:

* **PDF** — pypdfium2 extracts the text the document already contains for
  free; pages that come back empty (scans) are rendered to PNG and
  transcribed by a vision model. The output is page-marked text.
* **Image** — the bytes go to the model as an image. The ``ocr`` tool asks a
  vision model to transcribe them; ``read_file`` and ``files: read`` attach
  the image to the tool result instead, and the LLM client routes that turn
  to the configured vision model — the same path a photo in chat follows.

Everything here inherits the read boundary of the callers: paths were already
resolved and secret-checked by the time they arrive, but :meth:`DocumentReader.check`
re-asks anyway, because a second opinion on a credential file costs nothing
and a missed one costs a transcription sent to a third-party API.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from ..config import Config
from ..llm.base import LLMError
from ..llm.openai_compat import OpenAICompatClient
from ..paths import is_within, project_root, relative_to_root
from ..util.text import fence_untrusted, truncate
from .base import ToolError, ToolResult
from .files import foreign_uploads
from .safety import read_guard

# Import note: this module sits *under* files/base and is imported by explore
# and ocr. files.py must keep importing it lazily (inside its read action) —
# docread.top-level imports files here, so a top-level import back would close
# a cycle.

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

#: What ``read_file`` asks a vision model to say about an image in a reply
#: chain or another place the model cannot look directly. Description, not
#: transcription: a photo's meaning is what the model is missing.
DESCRIBE_PROMPT = (
    "Describe this image: what it shows, any visible text (transcribe it "
    "verbatim), and anything notable. Do not speculate beyond what is visible."
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

#: An image attached to a tool result rides in the conversation's base64 —
#: every later request pays for it until the elider drops it. These caps keep
#: one attachment from being that request: bigger images are downscaled with
#: Pillow, never dropped (a photo the model cannot see is a photo it will
#: make another tool call about).
ATTACH_MAX_BYTES = 4 * 1024 * 1024
ATTACH_MAX_SIDE = 2000


def media_kind(path: Path) -> tuple[str, str] | None:
    """``("pdf", "application/pdf")`` or ``("image", mime)`` for *path*, else ``None``.

    Extension first, then PDF magic: a PDF named ``.jpg`` is a PDF. ``None``
    means "not media" — the caller reads it as text, exactly as before.
    """
    suffix = path.suffix.lower()
    try:
        is_pdf = suffix == ".pdf" or sniffs_pdf(path)
    except OSError:
        return None
    if is_pdf:
        return "pdf", "application/pdf"
    if mime := IMAGE_MIMES.get(suffix):
        return "image", mime
    return None


def stat_size(path: Path) -> int:
    """A file's size in bytes, as its own sync helper — same reasoning as
    :func:`read_b64` directly below: a local stat is what the read *is*."""
    return path.stat().st_size


def read_b64(path: Path) -> str:
    """A file's bytes, base64 — sync on purpose, in a sync helper: ASYNC240
    exists so an async function does not block the loop on its own, and the
    files tool reads the same way for the same reason (a local read is what
    the tool *is*; a thread would only move the wait)."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sniffs_pdf(path: Path) -> bool:
    """True when the file starts with a PDF magic number, whatever it is called."""
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def open_pdf(path: Path) -> Any:
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


def embedded_text(page: Any) -> str:
    """The text the PDF already contains — the free pass."""
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range()
    finally:
        textpage.close()


def render_page(page: Any, scale: float) -> str:
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


def max_pages_clamped(requested: Any, configured: int) -> int:
    """The page ceiling for this call: the model's ask, clamped by config. 0 = all."""
    try:
        value = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        value = configured
    return max(0, value)


def prepare_attachment(b64: str, mime: str) -> tuple[str, str]:
    """Shrink *b64* to a payload the conversation can carry, ``(mime, b64)``.

    A Telegram photo is a few hundred kilobytes and passes through untouched;
    a rendered PDF page or a 12-megapixel screenshot gets downscaled to
    :data:`ATTACH_MAX_SIDE` on the long side. Runs only when Pillow is
    importable — the render path already requires it, so an image reaching
    here without Pillow means the file itself was the attachment, and
    rescaling is skipped rather than turning a read into a refusal.
    """
    if len(b64) * 3 // 4 <= ATTACH_MAX_BYTES:
        return mime, b64
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with the render path
        return mime, b64
    try:
        image = Image.open(io.BytesIO(base64.b64decode(b64)))
        image.thumbnail((ATTACH_MAX_SIDE, ATTACH_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "image/png", base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001 - a bad image is still returned, just big
        return mime, b64


class DocumentReader:
    """PDF and image reading built on one :class:`~lumi.config.Config`.

    Holds the two vision clients (they are built lazily, on first use, so a
    process that never touches a scan never opens a second endpoint) and the
    read guards every caller would otherwise re-type.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        #: ``_llm`` serves scanned PDF pages (the ocr route); ``_image_llm``
        #: serves image descriptions (the vision route, else the default).
        self._llm: Any = None
        self._image_llm: Any = None

    # -- availability and guards ------------------------------------------- #

    def available(self) -> tuple[bool, str]:
        """Whether media reading is on at all: the ``ocr`` tool's own switch."""
        ocr = self._config.tools.ocr
        if not ocr.enabled:
            return False, "disabled by tools.ocr.enabled"
        if problem := ocr.endpoint_problem():
            return False, problem
        return True, ""

    def read_problem(self, path: Path, chat_id: int | str | None) -> str | None:
        """Why *path* must not be read as media, or ``None`` when it may be.

        The same boundary the ``files`` tool and ``read_file`` enforce on
        every other read — re-asked here because this class is also the one
        that ships bytes to a model, and a credential that slips past one
        caller becomes a transcription on someone else's API.
        """
        if not is_within(path, project_root()):
            return "it is outside the project"
        if (foreign := foreign_uploads(path, project_root(), chat_id)) is not None:
            return (
                f"workspace/uploads/{foreign} belongs to another chat; "
                "each chat only sees its own uploads"
            )
        if (guard := read_guard(path)) is not None:
            return f"{guard} holds credentials"
        if not path.is_file():
            return "there is no such file"
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            return f"it is over {MAX_FILE_MB} MB"
        return None

    # -- clients ------------------------------------------------------------ #

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

    # -- the two routes ------------------------------------------------------ #

    async def read_pdf(
        self, path: Path, *, start_page: int = 1, max_pages: Any = None
    ) -> ToolResult:
        """Pages ``[start_page, start_page + limit)`` of *path* as fenced text.

        ``offset`` and ``limit`` in the caller's arguments become the page
        window, so a long PDF is paged exactly like a long text file — the
        result reports the next offset instead of pretending it read
        everything.
        """
        # open_pdf raises ToolError for a corrupt document — the registry
        # turns that into a tool failure, so let it fly rather than flattening
        # it here.
        pdf = open_pdf(path)
        try:
            total = len(pdf)
            if total == 0:
                return ToolResult.failure(f"{relative_to_root(path)} has no pages.")
            start = max(1, int(start_page or 1))
            if start > total:
                return ToolResult.failure(
                    f"{relative_to_root(path)} has {total} page(s) — offset {start} is past "
                    f"the end. Call again with offset={total if total > 1 else 1}."
                )
            limit = max_pages_clamped(max_pages, self._config.tools.ocr.max_pages)
            remaining = total - start + 1
            if limit <= 0 or limit > remaining:
                limit = remaining

            # Two passes, and the order is the whole point: the free pass reads
            # what the PDF already contains; only what it could not read is
            # rendered and priced. Most PDFs never reach the second pass.
            embedded: dict[int, str] = {}
            scanned: list[int] = []
            for index in range(start - 1, start - 1 + limit):
                text = embedded_text(pdf[index]).strip()
                if len(text) >= MIN_EMBEDDED_CHARS:
                    embedded[index] = text
                else:
                    scanned.append(index)

            rendered: dict[int, str] = {}
            for index in scanned:
                image_b64 = render_page(pdf[index], self._config.tools.ocr.render_scale)
                rendered[index] = (await self.transcribe(image_b64)).strip()

            parts = [
                f"[page {index + 1}]\n{embedded.get(index) or rendered.get(index, '')}"
                for index in range(start - 1, start - 1 + limit)
            ]
            note = (
                f" ({len(scanned)} of {limit} page(s) scanned with a vision model)"
                if scanned
                else ""
            )
            text = fence_untrusted(truncate("\n\n".join(parts), MAX_TEXT_CHARS)) + note
            last = start - 1 + limit
            if last < total:
                text += (
                    f"\n\n[stopped at page {last}; {total - last} more page(s) "
                    f"not read — call again with offset={last + 1}]"
                )
            return ToolResult(
                text=text,
                summary=(
                    f"{relative_to_root(path)}: pages {start}-{last} of {total}, "
                    f"{len(scanned)} scanned"
                ),
                data={
                    "pages": limit,
                    "scanned": len(scanned),
                    "total_pages": total,
                    "next_offset": last + 1 if last < total else None,
                },
            )
        finally:
            pdf.close()

    async def read_image(self, path: Path, mime: str) -> ToolResult:
        """*path* as a tool result with the image attached for the model to see.

        The LLM client routes any request carrying an image to the configured
        vision model — the same rule that shows a chat photo to the model —
        so attaching is the vision path, not a substitute for one.
        """
        display = relative_to_root(path)
        size = stat_size(path)
        attach_mime, attach_b64 = prepare_attachment(read_b64(path), mime)
        return ToolResult(
            text=(
                f"--- {display} ({size:,} bytes, image) ---\n"
                "The image is attached to this result. Look at it; "
                f"{display} is its path if you need to process the file itself."
            ),
            images=[(attach_mime, attach_b64)],
            summary=f"read {display} (image attached)",
            data={"path": str(path), "size": size, "kind": "image", "mime": attach_mime},
        )

    async def transcribe(
        self, image_b64: str, *, client: Any = None, mime: str = "image/png"
    ) -> str:
        """One vision-model side call: the page/image's text, verbatim.

        Scanned PDF pages go through *client* when the caller passes one — the
        ``ocr`` tool passes its image client for image files, because an image
        file is an image (the vision route when one is set), while pages it
        renders use the ocr route. ``None`` means the ocr route.
        """
        try:
            reply = await (client or self._ocr_client()).complete(
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

    async def describe(self, image_b64: str, *, mime: str = "image/png") -> str:
        """One vision-model side call: what the image shows (see :data:`DESCRIBE_PROMPT`)."""
        try:
            reply = await self._image_client().complete(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DESCRIBE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                        ],
                    }
                ]
            )
        except LLMError as exc:
            raise ToolError(f"the model could not describe the image: {exc}") from exc
        return reply.text or "(no description)"

    async def describe_file(self, path: Path, mime: str) -> str:
        """The vision model's description of an image file on disk."""
        return await self.describe(read_b64(path), mime=mime)


__all__ = [
    "ATTACH_MAX_BYTES",
    "ATTACH_MAX_SIDE",
    "DESCRIBE_PROMPT",
    "IMAGE_MIMES",
    "MAX_FILE_MB",
    "MAX_TEXT_CHARS",
    "MIN_EMBEDDED_CHARS",
    "DocumentReader",
    "embedded_text",
    "max_pages_clamped",
    "media_kind",
    "open_pdf",
    "prepare_attachment",
    "read_b64",
    "render_page",
    "sniffs_pdf",
]
