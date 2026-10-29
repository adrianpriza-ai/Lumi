"""The ocr tool: dedicated OCR API dispatch, embedded-text extraction, vision fallback."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from lumi.config import OCRConfig
from lumi.llm.base import LLMError, LLMReply
from lumi.llm.ocr_api import OCRDocumentClient
from lumi.llm.openai_compat import OpenAICompatClient, request_model
from lumi.tools.base import ToolContext, ToolError
from lumi.tools.ocr import OCRTool, _sniffs_pdf
from lumi.util.text import UNTRUSTED_BEGIN, UNTRUSTED_END

#: The shape of a message whose context carries an image — what decides the
#: per-request model id in :func:`request_model`.
IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        ],
    }
]

#: A valid 1x1 transparent PNG.
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)


def make_pdf(*texts: str) -> bytes:
    """A minimal but real PDF: one page per text, Helvetica, computed xref."""
    objects: list[bytes] = []
    count = max(1, len(texts))
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode())
    font_id = 3 + 2 * count
    for i, text in enumerate(texts or [""]):
        if text:
            content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {4 + 2 * i} 0 R >>".encode()
            )
            objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        else:
            objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


class StubLLM:
    """Stands in for the vision client; raises if called with no replies left."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, tools=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("the vision model was called unexpectedly")
        return LLMReply(text=self.replies.pop(0))

    def describe(self) -> str:
        return "stub"


class StubOCRServer:
    """Stands in for OCRDocumentClient; records how the tool called it."""

    def __init__(self, pages: list[tuple[int, str]]) -> None:
        self.pages = pages
        self.calls: list[dict] = []

    async def ocr(self, *, document_url=None, document_b64=None, content_type="application/pdf"):
        self.calls.append(
            {
                "document_url": document_url,
                "document_b64": document_b64,
                "content_type": content_type,
            }
        )
        return self.pages


def ctx() -> ToolContext:
    return ToolContext(source="cli")


# --------------------------------------------------------------------------- #
# the local path: embedded text is free, scans cost
# --------------------------------------------------------------------------- #


async def test_a_text_pdf_never_touches_the_model(config, project: Path) -> None:
    path = project / "workspace" / "text.pdf"
    path.write_bytes(make_pdf("Hello Lumi OCR, the quick brown fox", "second page text with more words"))
    tool = OCRTool(config)
    tool._llm = StubLLM([])  # any call raises

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "Hello Lumi OCR" in result.text
    assert "second page text" in result.text
    assert result.text.startswith(UNTRUSTED_BEGIN)  # transcribed text is data
    assert result.data["scanned"] == 0


async def test_a_scanned_page_falls_back_to_the_vision_model(config, project: Path) -> None:
    path = project / "workspace" / "scan.pdf"
    path.write_bytes(make_pdf(""))  # blank page: nothing embedded
    tool = OCRTool(config)
    stub = StubLLM(["transcribed words"])
    tool._llm = stub

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "transcribed words" in result.text
    assert "1 of 1 page(s) scanned" in result.text
    # The page went out as an image, not as text.
    parts = stub.calls[0][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_max_pages_limits_the_work_and_names_the_rest(config, project: Path) -> None:
    path = project / "workspace" / "long.pdf"
    path.write_bytes(make_pdf("page one with plenty of words", "page two with plenty of words", "page three with plenty of words"))
    tool = OCRTool(config)
    tool._llm = StubLLM([])

    result = await tool.invoke({"path": str(path), "max_pages": 2}, ctx())

    assert "page one" in result.text
    assert "page two" in result.text
    assert "page three" not in result.text  # over max_pages
    assert "1 more page(s)" in result.text


@pytest.mark.parametrize("suffix", [".bin", ".png"])
async def test_the_magic_bytes_beat_the_extension(config, project: Path, suffix) -> None:
    path = project / "workspace" / f"upload{suffix}"
    path.write_bytes(make_pdf("sneaky pdf with enough text to count"))
    tool = OCRTool(config)
    tool._llm = StubLLM([])
    tool._image_llm = StubLLM([])

    result = await tool.invoke({"path": str(path)}, ctx())
    assert "sneaky pdf" in result.text


async def test_an_image_is_transcribed(config, project: Path) -> None:
    path = project / "workspace" / "receipt.png"
    path.write_bytes(base64.b64decode(PNG_1PX))
    tool = OCRTool(config)
    stub = StubLLM(["TOTAL 42.00"])
    tool._image_llm = stub

    result = await tool.invoke({"path": str(path)}, ctx())
    assert result.text == f"{UNTRUSTED_BEGIN}\nTOTAL 42.00\n{UNTRUSTED_END}"
    url = stub.calls[0][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


async def test_an_image_still_uses_the_vision_route_with_a_dedicated_ocr_provider(
    config, project: Path
) -> None:
    path = project / "workspace" / "receipt.png"
    path.write_bytes(base64.b64decode(PNG_1PX))
    config.tools.ocr.provider = "mistral"
    image = StubLLM(["TOTAL 42.00"])
    tool = OCRTool(config)
    tool._image_llm = image

    class FailingOCRServer:
        async def ocr(self, **kwargs):
            raise AssertionError("image files must not use the document OCR API")

    tool._api = FailingOCRServer()
    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "TOTAL 42.00" in result.text
    assert result.text.startswith(UNTRUSTED_BEGIN)


async def test_a_dedicated_ocr_provider_does_not_accept_plain_text_files(
    config, project: Path
) -> None:
    path = project / "workspace" / "notes.txt"
    path.write_text("just text", encoding="utf-8")
    config.tools.ocr.provider = "mistral"

    result = await OCRTool(config).invoke({"path": str(path)}, ctx())

    assert not result.ok
    assert "read_file" in result.text


# --------------------------------------------------------------------------- #
# model routing: photos -> vision, PDFs -> ocr, everything -> default
# --------------------------------------------------------------------------- #


def test_a_vision_model_never_changes_what_reads_a_pdf(config) -> None:
    config.llm.vision_model = "gpt-4o"
    tool = OCRTool(config)
    # The scanned page goes to the default, not the vision model…
    assert request_model(IMAGE_MESSAGES, tool._ocr_client().config) == config.llm.model_of()
    # …while an image file follows the photo rule.
    assert request_model(IMAGE_MESSAGES, tool._image_client().config) == "gpt-4o"


def test_an_ocr_model_takes_pdfs_and_leaves_images_alone(config) -> None:
    config.llm.vision_model = "gpt-4o"
    config.llm.ocr_model = "paddleocr-v4"
    tool = OCRTool(config)
    assert request_model(IMAGE_MESSAGES, tool._ocr_client().config) == "paddleocr-v4"
    assert request_model(IMAGE_MESSAGES, tool._image_client().config) == "gpt-4o"


def test_with_nothing_set_images_and_pdfs_share_the_default(config) -> None:
    tool = OCRTool(config)
    assert request_model(IMAGE_MESSAGES, tool._ocr_client().config) == config.llm.model_of()
    assert request_model(IMAGE_MESSAGES, tool._image_client().config) == config.llm.model_of()


@pytest.mark.parametrize("vision_model", ["", "photo-model"])
@pytest.mark.parametrize("ocr_model", ["", "document-model"])
async def test_routing_matrix_through_file_dispatch(
    config, project, monkeypatch, vision_model, ocr_model
) -> None:
    config.llm.vision_model = vision_model
    config.llm.ocr_model = ocr_model
    used_models = []

    async def capture(client, messages, *, tools=None):
        used_models.append(request_model(messages, client.config))
        return LLMReply(text="transcribed")

    monkeypatch.setattr(OpenAICompatClient, "complete", capture)
    pdf = project / "workspace" / "scan.pdf"
    pdf.write_bytes(make_pdf(""))
    image = project / "workspace" / "photo.png"
    image.write_bytes(base64.b64decode(PNG_1PX))
    tool = OCRTool(config)

    assert (await tool.invoke({"path": str(pdf)}, ctx())).ok
    assert (await tool.invoke({"path": str(image)}, ctx())).ok
    assert used_models == [ocr_model or "test-model", vision_model or "test-model"]
    assert request_model([{"role": "user", "content": "text"}], config.llm) == "test-model"


async def test_plain_text_files_are_refused(config, project: Path) -> None:
    path = project / "workspace" / "notes.txt"
    path.write_text("just text", encoding="utf-8")
    result = await OCRTool(config).invoke({"path": str(path)}, ctx())
    assert not result.ok
    assert "read_file" in result.text


async def test_paths_outside_the_project_are_refused(config) -> None:
    result = await OCRTool(config).invoke({"path": "/etc/passwd"}, ctx())
    assert not result.ok
    assert "outside the project" in result.text


async def test_a_missing_file_says_so(config, project: Path) -> None:
    result = await OCRTool(config).invoke({"path": "workspace/nope.pdf"}, ctx())
    assert not result.ok
    assert "no such file" in result.text


async def test_another_chats_uploads_are_out_of_reach(config, project: Path) -> None:
    path = project / "workspace" / "uploads" / "42" / "paper.pdf"
    path.parent.mkdir(parents=True)
    path.write_bytes(make_pdf("hello"))
    result = await OCRTool(config).invoke(
        {"path": str(path)}, ToolContext(source="cli", chat_id=43)
    )
    assert not result.ok
    assert "another chat" in result.text


async def test_a_credential_file_is_not_sent_for_ocr(config, project: Path) -> None:
    """A PDF is still a document, but one whose name marks it as credentials
    is not handed to a third-party OCR API with no tap."""
    path = project / "workspace" / ".env.pdf"
    path.write_bytes(make_pdf("hello"))
    result = await OCRTool(config).invoke({"path": str(path)}, ctx())
    assert not result.ok
    assert "credentials" in result.text


# --------------------------------------------------------------------------- #
# the dedicated OCR API ([tools.ocr] provider = "mistral")
# --------------------------------------------------------------------------- #


async def test_a_local_file_goes_to_the_api_base64(config, project: Path, monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-1")
    config.tools.ocr.provider = "mistral"
    path = project / "workspace" / "paper.pdf"
    data = make_pdf("hello api")
    path.write_bytes(data)
    tool = OCRTool(config)
    stub = StubOCRServer([(1, "# Page one"), (2, "# Page two")])
    tool._api = stub

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "[page 1]\n# Page one" in result.text
    assert "[page 2]\n# Page two" in result.text
    call = stub.calls[0]
    assert call["document_b64"] == base64.b64encode(data).decode("ascii")
    assert call["content_type"] == "application/pdf"
    assert call["document_url"] is None


async def test_a_url_goes_to_the_api_as_a_document_url(config, monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-1")
    config.tools.ocr.provider = "mistral"
    tool = OCRTool(config)
    stub = StubOCRServer([(1, "arxiv text")])
    tool._api = stub

    result = await tool.invoke({"path": "https://arxiv.org/pdf/2201.04234"}, ctx())

    assert result.ok
    assert stub.calls[0]["document_url"] == "https://arxiv.org/pdf/2201.04234"
    assert "arxiv text" in result.text


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.pdf",
        "http://localhost/x.pdf",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/x.pdf",
        "http://[::1]/x.pdf",
    ],
)
async def test_a_private_host_url_is_refused_before_the_api(config, monkeypatch, url: str) -> None:
    """SSRF-shaped OCR URLs never reach the OCR API."""
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-1")
    config.tools.ocr.provider = "mistral"
    tool = OCRTool(config)
    stub = StubOCRServer([(1, "should not be called")])
    tool._api = stub

    result = await tool.invoke({"path": url}, ctx())

    assert not result.ok
    assert "refusing to OCR" in result.text
    assert stub.calls == []


async def test_a_url_without_a_provider_explains_the_fix(config) -> None:
    result = await OCRTool(config).invoke({"path": "https://example.com/x.pdf"}, ctx())
    assert not result.ok
    assert "tools.ocr" in result.text


async def test_the_tool_is_unavailable_without_the_provider_key(config, monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    config.tools.ocr.provider = "mistral"
    ok, reason = OCRTool(config).available()
    assert not ok
    assert "MISTRAL_API_KEY" in reason


def test_the_ocr_enabled_flag_is_honoured(config) -> None:
    config.tools.ocr.enabled = False
    ok, reason = OCRTool(config).available()
    assert not ok
    assert "disabled" in reason


@pytest.mark.parametrize("base_url", ["", "https://api.mistral.ai/v1/", "not-a-url"])
async def test_invalid_custom_endpoint_is_blocked_before_http(config, monkeypatch, base_url) -> None:
    config.tools.ocr.provider = "custom"
    config.tools.ocr.base_url = base_url
    monkeypatch.setenv("MISTRAL_API_KEY", "test-ocr-key")

    def unexpected_http(**kwargs):
        pytest.fail("an invalid custom endpoint must not upload documents")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_http)
    ok, reason = OCRTool(config).available()
    assert not ok
    assert "base_url" in reason
    with pytest.raises(LLMError, match="base_url"):
        await OCRDocumentClient(config.tools.ocr).ocr(document_b64="JVBERg==")


# --------------------------------------------------------------------------- #
# OCRDocumentClient: the Mistral wire shape
# --------------------------------------------------------------------------- #


def test_the_endpoint_is_base_url_plus_ocr() -> None:
    client = OCRDocumentClient(OCRConfig(provider="mistral"))
    assert client.endpoint == "https://api.mistral.ai/v1/ocr"
    client = OCRDocumentClient(OCRConfig(provider="mistral", base_url="http://host/v1/"))
    assert client.endpoint == "http://host/v1/ocr"


async def test_the_request_matches_the_mistral_shape(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"pages": [{"page_number": 1, "markdown": "# Hi"}]}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-9")
    client = OCRDocumentClient(OCRConfig(provider="mistral", api_key_env="MISTRAL_API_KEY"))

    pages = await client.ocr(document_url="https://arxiv.org/pdf/2201.04234")

    assert pages == [(1, "# Hi")]
    assert captured["url"] == "https://api.mistral.ai/v1/ocr"
    assert captured["client"]["headers"]["Authorization"] == "Bearer mk-9"
    assert captured["json"]["model"] == "mistral-ocr-latest"
    assert captured["json"]["document"] == {
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234",
    }


async def test_an_error_response_quotes_the_body(monkeypatch) -> None:
    class FakeResponse:
        status_code = 400
        text = json.dumps({"detail": "unsupported file type"})

        def json(self):
            return {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setenv("MISTRAL_API_KEY", "mk-9")
    client = OCRDocumentClient(OCRConfig(provider="mistral"))

    with pytest.raises(LLMError, match="HTTP 400.*unsupported file type"):
        await client.ocr(document_url="https://example.com/x.pdf")


async def test_a_missing_key_is_named_before_any_request(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    client = OCRDocumentClient(OCRConfig(provider="mistral"))
    with pytest.raises(LLMError, match="MISTRAL_API_KEY is not set"):
        await client.ocr(document_url="https://example.com/x.pdf")


@pytest.mark.parametrize("provider", ["mistral", "custom"])
async def test_local_pdf_upload_uses_a_document_data_uri(monkeypatch, provider) -> None:
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"pages": [{"index": 0, "markdown": "text"}]})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setenv("MISTRAL_API_KEY", "test-ocr-key")
    client = OCRDocumentClient(OCRConfig(provider=provider, base_url="https://ocr.example/v1"))
    encoded = base64.b64encode(make_pdf("hello")).decode("ascii")

    assert await client.ocr(document_b64=encoded) == [(1, "text")]
    assert json.loads(requests[0].content)["document"] == {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{encoded}",
    }


def test_the_sniffer_recognises_a_pdf_header(config, project: Path) -> None:
    path = project / "workspace" / "x.dat"
    path.write_bytes(make_pdf("x"))
    assert _sniffs_pdf(path)
    path.write_bytes(b"not a pdf")
    assert not _sniffs_pdf(path)


async def test_tool_errors_surface_as_tool_errors(config, project: Path) -> None:
    """A corrupt PDF names itself instead of killing the turn."""
    path = project / "workspace" / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nnot really a pdf")
    with pytest.raises(ToolError, match="could not open"):
        await OCRTool(config).invoke({"path": str(path)}, ctx())
