"""The document-OCR client: a Mistral-style ``POST /ocr`` endpoint.

Not a chat model: the whole document goes in — a public URL or the bytes
base64-encoded — and per-page markdown comes back. That is a different
conversation than "look at this image and transcribe it", so it gets its own
client rather than a prompt. The wire format is Mistral's OCR API
(docs.mistral.ai/api/endpoint/ocr); any compatible service works by pointing
``tools.ocr.base_url`` at it.

.. code-block:: bash

    curl https://api.mistral.ai/v1/ocr \\
      -H "Authorization: Bearer $MISTRAL_API_KEY" \\
      -d '{"model": "mistral-ocr-latest",
           "document": {"type": "document_url",
                        "document_url": "https://arxiv.org/pdf/2201.04234"}}'
"""

from __future__ import annotations

import json
from typing import Any

from ..config import OCRConfig
from ..util.log import get_logger
from ..util.text import format_error, truncate
from .base import LLMError

log = get_logger(__name__)

#: Characters of a failed response body quoted back to the model. Enough to
#: name the problem ("model not found"), short enough to not eat the context.
ERROR_BODY_CHARS = 400

#: Fallback content types, by suffix, when the caller does not know the file's
#: MIME. Only the dedicated endpoint needs these — it types the document
#: itself; the local path never sends a document anywhere.
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def error_body(text: str) -> str:
    """The endpoint's own error words, or the raw body when it has none.

    Mistral answers errors as JSON — ``{"message": "..."}``, FastAPI-style
    ``{"detail": "..."}`` — and that message is what the owner needs; a raw
    wall of HTML or a minified body is the fallback.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return truncate(text, ERROR_BODY_CHARS)
    if isinstance(data, dict):
        for field in ("message", "detail"):
            value = data.get(field)
            if isinstance(value, str) and value.strip():
                return truncate(value, ERROR_BODY_CHARS)
    return truncate(text, ERROR_BODY_CHARS)


def page_numbers(pages_data: list[dict[str, Any]], requested: list[int] | None) -> list[int]:
    """1-based page numbers for the endpoint's returned pages.

    When specific pages were requested, the response follows that order, so
    the requested numbers (0-based) are the truth. Otherwise the endpoint's
    ``index`` field decides — the documented convention is 0-based, but a
    sequence that already starts at 1 is read as finished page numbers, since
    compatible servers disagree and a wrong page number is worse than no
    number. Missing or garbled indexes fall back to position.
    """
    count = len(pages_data)
    if requested:
        return [number + 1 for number in requested]
    indexes = [page.get("index") for page in pages_data]
    if all(isinstance(index, int) for index in indexes):
        if indexes == list(range(0, count)):
            return list(range(1, count + 1))
        if indexes == list(range(1, count + 1)):
            return list(indexes)
    return list(range(1, count + 1))


class OCRDocumentClient:
    """One HTTP call: document in, per-page text out.

    Deliberately not built on the OpenAI SDK — this is not chat completions,
    and pretending it is would mean bending someone else's retry policy around
    a different endpoint. One request, one clear failure; the tool above it
    decides what to say.
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config

    @property
    def endpoint(self) -> str:
        """The full OCR URL: ``<base_url>/ocr``, whatever the trailing slash state."""
        if problem := self.config.config_problem():
            raise LLMError(problem)
        return self.config.base_url_of() + "/ocr"

    def describe(self) -> str:
        return f"{self.config.model_of()} via {self.endpoint}"

    def _payload(
        self,
        document: dict[str, Any],
        pages: list[int] | None,
    ) -> dict[str, Any]:
        """The request body, per the documented Mistral shape.

        Only fields the owner configured are sent — an empty ``table_format``
        or an ``extract_header`` left off means "the endpoint decides", and
        ``pages`` goes out only when the caller actually wants a window.
        ``include_blocks`` is the one unconditional lie-by-omission: the
        documented default ``true`` doubles the payload with paragraph
        bounding boxes this client only ever throws away.
        """
        payload: dict[str, Any] = {
            "model": self.config.model_of(),
            "document": document,
            "include_blocks": False,
        }
        if pages:
            payload["pages"] = pages
        if table_format := self.config.table_format.strip().lower():
            payload["table_format"] = table_format
        if self.config.extract_header:
            payload["extract_header"] = True
        if self.config.extract_footer:
            payload["extract_footer"] = True
        return payload

    async def ocr(
        self,
        *,
        document_url: str | None = None,
        document_b64: str | None = None,
        content_type: str = "application/pdf",
        pages: list[int] | None = None,
    ) -> list[tuple[int, str]]:
        """OCR one document. Exactly one of *document_url* / *document_b64*.

        *pages* is optional and 0-based (Mistral's ``pages`` parameter:
        ``None`` reads the whole document). Returns ``(page_number, text)``
        pairs in page order. Raises :class:`LLMError` with the endpoint,
        status and the provider's own response body — an OCR API fails in
        specific ways ("unsupported file type", "model not found") and the
        owner needs the exact words, not a guess.
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - openai always pulls httpx in
            raise LLMError("the httpx package is not installed") from exc

        endpoint = self.endpoint
        if document_url and document_b64:
            raise LLMError("provide exactly one of document_url or document_b64")
        if document_url:
            document: dict[str, Any] = {"type": "document_url", "document_url": document_url}
        elif document_b64:
            document = {
                "type": "document_url",
                "document_url": f"data:{content_type};base64,{document_b64}",
            }
        else:  # pragma: no cover - the tool checks this first
            raise LLMError("no document to OCR")

        key = self.config.api_key()
        # A key is mandatory only where the config says the endpoint cannot
        # work without one (the hosted API). A keyless custom endpoint is a
        # real configuration — a LAN server, a local queue — so no header
        # rather than a refusal; if that endpoint does want auth, its own 401
        # names the problem better than a guess here.
        if not key and self.config.needs_key():
            raise LLMError(
                f"{self.config.api_key_env} is not set — the OCR endpoint needs it in .env."
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers=headers,
            ) as http:
                response = await http.post(endpoint, json=self._payload(document, pages))
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach {self.endpoint}: {format_error(exc)}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"{self.endpoint} returned HTTP {response.status_code}: "
                f"{error_body(response.text)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.endpoint} returned a non-JSON body") from exc

        pages_data = data.get("pages") or []
        if pages_data:
            numbers = page_numbers(pages_data, pages)
            return [
                (number, str(page.get("markdown") or ""))
                for number, page in zip(numbers, pages_data)
            ]
        # Some compatible servers answer with one flat body instead of pages;
        # take it rather than declaring an empty document.
        content = str(data.get("content") or "").strip()
        if content:
            return [(1, content)]
        raise LLMError(f"{self.endpoint} returned no pages and no content")


__all__ = ["OCRDocumentClient", "CONTENT_TYPES", "error_body", "page_numbers"]
