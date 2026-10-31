"""Media reads: PDFs and images through ``read_file`` and ``files: read``.

The shared machinery (:mod:`lumi.tools.docread`) decides what a PDF or a
picture turns into; these tests pin the tool-level behavior — page windows,
attached images, the guards that survive a second implementation — plus the
agent plumbing that turns ``ToolResult.images`` into a multimodal tool
message, and the elider that lets old images leave the window again.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from test_ocr import PNG_1PX, StubLLM, make_pdf

from lumi.agent import _tool_message_content
from lumi.bot import _quoted_prefix
from lumi.chatlog import ChatLog, LogEntry, media_entry_from
from lumi.context import ELIDED_FLAG, content_chars
from lumi.tools.base import ToolContext, ToolError, ToolResult
from lumi.tools.docread import DocumentReader, media_kind, prepare_attachment
from lumi.tools.explore import ReadFileTool
from lumi.tools.files import FilesTool
from lumi.tools.more_context import MoreContextTool


def ctx(**kwargs) -> ToolContext:
    return ToolContext(source="cli", **kwargs)


# --------------------------------------------------------------------------- #
# read_file: PDFs
# --------------------------------------------------------------------------- #


async def test_read_file_reads_a_text_pdf_without_the_model(config, project: Path) -> None:
    path = project / "workspace" / "doc.pdf"
    path.write_bytes(make_pdf("Hello Lumi read_file, embedded text only"))
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "Hello Lumi read_file" in result.text
    assert "[page 1]" in result.text
    assert result.data["total_pages"] == 1
    assert result.data["next_offset"] is None


async def test_read_file_pages_a_pdf_in_pages(config, project: Path) -> None:
    path = project / "workspace" / "long.pdf"
    path.write_bytes(
        make_pdf("page one with plenty of words", "page two with plenty of words", "page three with plenty of words")
    )
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path), "offset": 2, "limit": 1}, ctx())

    assert result.ok
    assert "page two" in result.text
    assert "page one" not in result.text and "page three" not in result.text
    assert result.data["next_offset"] == 3


async def test_read_file_pdf_offset_past_the_end_names_the_page_count(
    config, project: Path
) -> None:
    path = project / "workspace" / "short.pdf"
    path.write_bytes(make_pdf("only one page with plenty of words"))
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path), "offset": 5}, ctx())

    assert not result.ok
    assert "has 1 page(s)" in result.text


async def test_a_jpg_that_is_really_a_pdf_reads_as_a_pdf(config, project: Path) -> None:
    """PDF magic beats a misleading image extension, as with the ocr tool."""
    path = project / "workspace" / "sneaky.jpg"
    path.write_bytes(make_pdf("sneaky pdf with enough text to count"))
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert "sneaky pdf" in result.text
    assert "[page 1]" in result.text


# --------------------------------------------------------------------------- #
# read_file: images
# --------------------------------------------------------------------------- #


async def test_read_file_attaches_an_image(config, project: Path) -> None:
    path = project / "workspace" / "pic.png"
    path.write_bytes(base64.b64decode(PNG_1PX))
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path)}, ctx())

    assert result.ok
    assert len(result.images) == 1
    mime, data = result.images[0]
    assert mime == "image/png"
    assert data == PNG_1PX
    assert str(path) in result.data["path"]


async def test_read_file_media_when_disabled_names_the_switch(config, project: Path) -> None:
    path = project / "workspace" / "pic.png"
    path.write_bytes(base64.b64decode(PNG_1PX))
    config.tools.ocr.enabled = False
    tool = ReadFileTool(config)

    result = await tool.invoke({"path": str(path)}, ctx())

    assert not result.ok
    assert "tools.ocr.enabled" in result.text


async def test_read_file_refuses_another_chats_upload_image(config, project: Path) -> None:
    path = project / "workspace" / "uploads" / "other" / "pic.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(base64.b64decode(PNG_1PX))
    tool = ReadFileTool(config)

    with pytest.raises(ToolError, match="another chat"):
        await tool.invoke({"path": str(path)}, ctx(chat_id="mine"))


# --------------------------------------------------------------------------- #
# files: read
# --------------------------------------------------------------------------- #


async def test_files_read_returns_pdf_text(config, project: Path) -> None:
    path = project / "workspace" / "doc.pdf"
    path.write_bytes(make_pdf("Hello from the files tool PDF"))
    tool = FilesTool(config)

    result = await tool.invoke({"action": "read", "path": str(path)}, ctx())

    assert result.ok
    assert "Hello from the files tool PDF" in result.text


async def test_files_read_attaches_an_image(config, project: Path) -> None:
    path = project / "workspace" / "pic.jpg"
    path.write_bytes(base64.b64decode(PNG_1PX))
    tool = FilesTool(config)

    result = await tool.invoke({"action": "read", "path": str(path)}, ctx())

    assert result.ok
    # The mime comes from the extension, not the bytes inside.
    assert result.images and result.images[0][0] == "image/jpeg"


async def test_files_read_refuses_secret_named_media(config, project: Path) -> None:
    """A credential that happens to carry an image extension stays refused."""
    path = project / "workspace" / "id_rsa.png"
    path.write_bytes(base64.b64decode(PNG_1PX))
    tool = FilesTool(config)

    with pytest.raises(ToolError, match="credentials"):
        await tool.invoke({"action": "read", "path": str(path)}, ctx())


# --------------------------------------------------------------------------- #
# the shared reader's own guards
# --------------------------------------------------------------------------- #


def test_media_kind_sniffs_pdf_over_extension(tmp_path: Path) -> None:
    real = tmp_path / "real.pdf"
    real.write_bytes(make_pdf("x"))
    assert media_kind(real) == ("pdf", "application/pdf")

    named_jpg = tmp_path / "real.jpg"
    named_jpg.write_bytes(make_pdf("x"))
    assert media_kind(named_jpg) == ("pdf", "application/pdf")

    png = tmp_path / "pic.png"
    png.write_bytes(base64.b64decode(PNG_1PX))
    assert media_kind(png) == ("image", "image/png")

    text = tmp_path / "notes.txt"
    text.write_text("just words")
    assert media_kind(text) is None


def test_reader_re_asks_the_guards(config, project: Path) -> None:
    """The reader checks foreign uploads itself: it is the component that
    ships bytes to a model, so it does not trust the caller's earlier check."""
    reader = DocumentReader(config)
    foreign = project / "workspace" / "uploads" / "other" / "pic.png"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(base64.b64decode(PNG_1PX))

    problem = reader.read_problem(foreign, "mine")

    assert problem is not None
    assert "another chat" in problem


async def test_reader_pdf_scan_pages_use_the_ocr_client(config, project: Path) -> None:
    path = project / "workspace" / "scan.pdf"
    path.write_bytes(make_pdf(""))  # blank page -> rendered and transcribed
    reader = DocumentReader(config)
    reader._llm = StubLLM(["transcribed words"])

    result = await reader.read_pdf(path)

    assert result.ok
    assert "transcribed words" in result.text
    assert result.data["scanned"] == 1


# --------------------------------------------------------------------------- #
# agent plumbing: ToolResult.images -> multimodal tool message
# --------------------------------------------------------------------------- #


def test_a_plain_tool_result_stays_text() -> None:
    result = ToolResult(text="just words")
    assert _tool_message_content(result) == "just words"


def test_an_attached_image_becomes_an_image_url_part() -> None:
    result = ToolResult(text="--- pic.png (image) ---", images=[("image/png", "AA")])

    content = _tool_message_content(result)

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "--- pic.png (image) ---"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AA"


def test_content_chars_counts_text_parts_only_for_images() -> None:
    message = {
        "role": "tool",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10_000}},
        ],
    }
    assert content_chars(message) == len("hello")


# --------------------------------------------------------------------------- #
# the elider lets attached images leave the window
# --------------------------------------------------------------------------- #


def _over_budget_agent(config):
    from lumi.context import ContextWindow

    return ContextWindow(config, llm=None, history=None)


class _Conv:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.chat_id = "test"
        self.messages = messages
        self.elided = 0


def test_elider_drops_image_parts_and_keeps_text(config) -> None:
    window = _over_budget_agent(config)
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "tool",
            "name": "read_file",
            "content": [
                {"type": "text", "text": "--- pic.png ---"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4000}},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    conv = _Conv(messages)
    window._elide_tool_output(conv, over=1)

    assert conv.elided == 1
    tool_message = messages[1]
    kinds = [part.get("type") for part in tool_message["content"]]
    assert "image_url" not in kinds
    assert "text" in kinds
    assert tool_message["content"][0]["text"] == "--- pic.png ---"
    assert tool_message.get(ELIDED_FLAG) is True


def test_elider_skips_list_content_without_images(config) -> None:
    window = _over_budget_agent(config)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "name": "read_file", "content": [{"type": "text", "text": "words"}]},
        {"role": "assistant", "content": "done"},
    ]

    conv = _Conv(messages)
    window._elide_tool_output(conv, over=1)

    assert conv.elided == 0
    assert messages[1]["content"] == [{"type": "text", "text": "words"}]


# --------------------------------------------------------------------------- #
# attachments: downscaled before they ride in the conversation
# --------------------------------------------------------------------------- #


def test_prepare_attachment_downscales_a_big_image(monkeypatch) -> None:
    """A payload over the cap is shrunk, never dropped; a tiny image of huge
    dimensions rides as-is, because the base64 is what the window pays for."""
    import io

    from PIL import Image

    monkeypatch.setattr("lumi.tools.docread.ATTACH_MAX_BYTES", 1000)
    image = Image.new("RGB", (4000, 3000), (200, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    big = base64.b64encode(buffer.getvalue()).decode("ascii")

    mime, out = prepare_attachment(big, "image/png")

    assert mime == "image/png"
    shrunk = Image.open(io.BytesIO(base64.b64decode(out)))
    assert max(shrunk.size) <= 2000
    assert len(out) < len(big)


# --------------------------------------------------------------------------- #
# the chatlog and more_context: dropped media is a pointer, not a hole
# --------------------------------------------------------------------------- #


class FakeMessage:
    def __init__(self, *, caption: str = "", sender: str = "Owner") -> None:
        self.caption = caption
        self.message_id = 5
        self.from_user = type("U", (), {"first_name": sender, "username": sender})()
        self.date = None


def test_media_entry_renders_with_its_path() -> None:
    entry = media_entry_from(FakeMessage(caption="see attached"), "document", "workspace/uploads/1/notes.txt")

    assert entry.kind == "document"
    assert "see attached" in entry.render(1)
    assert "workspace/uploads/1/notes.txt" in entry.render(1)
    assert "read_file" in entry.render(1)


def test_media_entry_without_a_path_says_so() -> None:
    entry = media_entry_from(FakeMessage(), "photo", "")

    assert entry.kind == "photo"
    assert entry.text.startswith("(a photo that could not be saved)")
    assert "not saved" in entry.render(1)


async def test_more_context_lists_attachment_paths(config) -> None:
    log = ChatLog()
    log.record(
        "chat",
        LogEntry(message_id=1, author="Owner", text="see attached", clock="10:00",
                 kind="document", path="workspace/uploads/chat/notes.txt"),
    )
    log.record("chat", LogEntry(message_id=2, author="Dana", text="thanks!", clock="10:01"))

    result = await MoreContextTool(config, log=log).invoke({}, ctx(chat_id="chat"))

    assert result.ok
    assert "workspace/uploads/chat/notes.txt" in result.text
    assert "read_file" in result.text
    assert "thanks!" in result.text


# --------------------------------------------------------------------------- #
# reply chain: media notes ride inside the quote block
# --------------------------------------------------------------------------- #


def _message(message_id: int, text: str = "") -> Any:
    return type(
        "M",
        (),
        {"message_id": message_id, "text": text, "caption": "", "from_user": None,
         "reply_to_message": None},
    )()


def test_quoted_prefix_single_link_carries_a_media_note() -> None:
    reply = _message(2, "@lumi what is this?")
    photo = _message(1)
    photo.photo = [{"file_id": "x"}]
    reply.reply_to_message = photo

    prefix = _quoted_prefix(reply, {1: "[this is a photo, saved as workspace/uploads/1/photo-1.jpg]"})

    assert "saved as workspace/uploads/1/photo-1.jpg" in prefix
    assert "(a photo with no text)" not in prefix


def test_quoted_prefix_chain_media_note_lands_in_its_own_block() -> None:
    newest = _message(3, "no it isn't")
    middle = _message(2, "look above")
    oldest = _message(1)
    oldest.document = type("D", (), {"file_name": "spec.pdf"})()
    newest.reply_to_message = middle
    middle.reply_to_message = oldest

    prefix = _quoted_prefix(newest, {1: "[this is a document, saved as workspace/uploads/1/spec.pdf]"})

    assert "saved as workspace/uploads/1/spec.pdf" in prefix
    # oldest first: the note is in the first block, not the last
    assert prefix.index("saved as workspace") < prefix.index("look above")


def test_quoted_prefix_without_media_is_unchanged() -> None:
    reply = _message(2, "try this")
    older = _message(1, "the plan")
    reply.reply_to_message = older

    assert _quoted_prefix(reply) == _quoted_prefix(reply, {})
