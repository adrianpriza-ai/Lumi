"""The document-OCR client: a Mistral-style ``POST /ocr`` endpoint.

Not a chat model: the whole document goes in — a public URL or the bytes
base64-encoded — and per-page markdown comes back. That is a different
conversation than "look at this image and transcribe it", so it gets its own
client rather than a prompt. The wire format is Mistral's OCR API; any
compatible service works by pointing ``tools.ocr.base_url`` at it.

.. code-block:: bash

    curl https://api.mistral.ai/v1/ocr \\
      -H "Authorization: Bearer $MISTRAL_API_KEY" \\
      -d '{"model": "mistral-ocr-latest",
           "document": {"type": "document_url",
                        "document_url": "https://arxiv.org/pdf/2201.04234"}}'
"""

from __future__ import annotations

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
        if problem := self.config.endpoint_problem():
            raise LLMError(problem)
        return self.config.base_url.strip().rstrip("/") + "/ocr"

    def describe(self) -> str:
        return f"{self.config.model} via {self.endpoint}"

    async def ocr(
        self,
        *,
        document_url: str | None = None,
        document_b64: str | None = None,
        content_type: str = "application/pdf",
    ) -> list[tuple[int, str]]:
        """OCR one document. Exactly one of *document_url* / *document_b64*.

        Returns ``(page_number, text)`` pairs in page order. Raises
        :class:`LLMError` with the endpoint, status and a snippet of the
        provider's own response body — an OCR API fails in specific ways
        ("unsupported file type", "model not found") and the owner needs the
        exact words, not a guess.
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
        if not key:
            raise LLMError(
                f"{self.config.api_key_env} is not set — the OCR endpoint needs it in .env."
            )

        payload = {"model": self.config.model, "document": document}
        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers={"Authorization": f"Bearer {key}"},
            ) as http:
                response = await http.post(endpoint, json=payload)
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

        pages = data.get("pages") or []
        if pages:
            return [
                (int(page.get("page_number") or index), str(page.get("markdown") or ""))
                for index, page in enumerate(pages, start=1)
            ]
        # Some compatible servers answer with one flat body instead of pages;
        # take it rather than declaring an empty document.
        content = str(data.get("content") or "").strip()
        if content:
            return [(1, content)]
        raise LLMError(f"{self.endpoint} returned no pages and no content")


__all__ = ["OCRDocumentClient", "CONTENT_TYPES"]
