"""The artifact harness: outgoing file delivery and incoming documents.

Covers the store itself (validation, staging, dedup, ingestion), the ``files``
tool's upload surface, and the agent loop's collection of artifacts into a
turn. The Telegram delivery layer is covered in ``test_bot.py``.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM, make_reply

from lumi.artifacts import DEFAULT_MAX_BYTES, ArtifactError, ArtifactStore
from lumi.tools.base import ToolContext
from lumi.tools.files import FilesTool


def ctx(**kwargs) -> ToolContext:
    return ToolContext(**kwargs)


@pytest.fixture
def store(config) -> ArtifactStore:
    return ArtifactStore(config.root, max_bytes=1 * 1024 * 1024)


# --------------------------------------------------------------------------- #
# the store: outgoing
# --------------------------------------------------------------------------- #


async def test_send_stages_a_copy_in_the_outbox(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "report.md"
    target.write_text("# report", encoding="utf-8")

    artifact = store.send(target, origin="test", project_root=config.root)

    assert artifact.absolute.is_file(), "the outbox copy must exist"
    assert artifact.absolute != target, "the outbox copy is not the original"
    assert artifact.absolute.read_text(encoding="utf-8") == "# report"
    assert artifact.path == "workspace/report.md"
    assert artifact.size == len("# report")
    assert artifact.origin == "test"
    assert artifact.absolute.is_relative_to(store.outbox_dir()), "staged inside the outbox"


async def test_send_survives_deletion_of_the_original(store: ArtifactStore, config) -> None:
    """The whole point of the outbox: the chat still gets the file."""
    target = config.root / "workspace" / "temp.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact = store.send(target, project_root=config.root)
    target.unlink()
    assert artifact.absolute.read_text(encoding="utf-8") == "a,b\n1,2\n"


async def test_send_dedupes_unchanged_files(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "same.md"
    target.write_text("same", encoding="utf-8")
    first = store.send(target, project_root=config.root)
    second = store.send(target, project_root=config.root)
    assert first.absolute == second.absolute


async def test_send_a_rewritten_file_gets_a_new_copy(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "changing.md"
    target.write_text("v1", encoding="utf-8")
    first = store.send(target, project_root=config.root)
    target.write_text("v2 — longer now", encoding="utf-8")
    second = store.send(target, project_root=config.root)
    assert first.absolute != second.absolute
    assert second.absolute.read_text(encoding="utf-8") == "v2 — longer now"


async def test_send_refuses_paths_outside_the_project(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="outside the project"):
        store.send("/etc/os-release", project_root=config.root)


async def test_send_refuses_missing_files(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="not a file"):
        store.send("workspace/nope.md", project_root=config.root)


async def test_send_refuses_unknown_extensions(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "evil.exe"
    target.write_bytes(b"MZ")
    with pytest.raises(ArtifactError, match="refusing to deliver"):
        store.send(target, project_root=config.root)


async def test_send_refuses_extensionless_files(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "Makefile"
    target.write_text("all:\n\techo hi\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="no extension"):
        store.send(target, project_root=config.root)


async def test_send_refuses_empty_files(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "empty.txt"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ArtifactError, match="empty"):
        store.send(target, project_root=config.root)


async def test_send_refuses_oversized_files(store: ArtifactStore, config) -> None:
    target = config.root / "workspace" / "big.txt"
    target.write_text("x" * (2 * 1024 * 1024), encoding="utf-8")
    with pytest.raises(ArtifactError, match="over the"):
        store.send(target, project_root=config.root)


async def test_send_resolves_relative_paths_against_the_project(
    store: ArtifactStore, config
) -> None:
    (config.root / "workspace" / "rel.txt").write_text("hi", encoding="utf-8")
    artifact = store.send("workspace/rel.txt", project_root=config.root)
    assert artifact.path == "workspace/rel.txt"


def test_the_default_cap_is_twenty_megabytes() -> None:
    assert DEFAULT_MAX_BYTES == 20 * 1024 * 1024


def test_recent_returns_the_newest_last(store: ArtifactStore, config) -> None:
    for name in ("a.md", "b.md", "c.md"):
        (config.root / "workspace" / name).write_text(name, encoding="utf-8")
        store.send(config.root / "workspace" / name, project_root=config.root)
    names = [a.path for a in store.recent(3)]
    assert names[-1].endswith("c.md")
    assert len(store.recent(2)) == 2


def test_caption_is_human_readable(store: ArtifactStore, config) -> None:
    (config.root / "workspace" / "tiny.txt").write_text("hi", encoding="utf-8")
    artifact = store.send(config.root / "workspace" / "tiny.txt", origin="files",
                          project_root=config.root)
    caption = artifact.caption()
    assert "workspace/tiny.txt" in caption
    assert "2 B" in caption
    assert "files" in caption


# --------------------------------------------------------------------------- #
# the store: incoming documents
# --------------------------------------------------------------------------- #


async def test_ingest_stores_under_workspace_uploads(store: ArtifactStore, config) -> None:
    artifact = store.ingest(b"hello", "notes.txt", 42, config.root)
    assert artifact.path == "workspace/uploads/42/notes.txt"
    assert artifact.absolute.read_bytes() == b"hello"
    assert artifact.origin == "telegram upload"


async def test_ingest_never_overwrites(store: ArtifactStore, config) -> None:
    first = store.ingest(b"one", "doc.md", 42, config.root)
    second = store.ingest(b"two", "doc.md", 42, config.root)
    assert first.absolute != second.absolute
    assert first.absolute.read_bytes() == b"one"
    assert second.absolute.read_bytes() == b"two"


async def test_ingest_sanitises_hostile_filenames(store: ArtifactStore, config) -> None:
    artifact = store.ingest(b"x", "../../etc/passwd.txt", 42, config.root)
    assert ".." not in artifact.absolute.name
    assert artifact.absolute.parent.is_relative_to(config.root / "workspace" / "uploads")


async def test_ingest_strips_weird_characters(store: ArtifactStore, config) -> None:
    artifact = store.ingest(b"x", "my report (final!).md", 7, config.root)
    name = artifact.absolute.name
    assert name.endswith(".md")
    assert all(c.isalnum() or c in "._- " for c in name.rsplit(".", 1)[0])


async def test_ingest_refuses_disallowed_extensions(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="not on the artifact allowlist"):
        store.ingest(b"MZ", "evil.exe", 42, config.root)


async def test_ingest_refuses_extensionless_documents(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="no extension"):
        store.ingest(b"x", "README", 42, config.root)


async def test_ingest_refuses_oversized_documents(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="over the"):
        store.ingest(b"x" * (2 * 1024 * 1024), "big.txt", 42, config.root)


async def test_ingest_refuses_empty_documents(store: ArtifactStore, config) -> None:
    with pytest.raises(ArtifactError, match="empty"):
        store.ingest(b"", "empty.txt", 42, config.root)


# --------------------------------------------------------------------------- #
# the files tool: upload surface
# --------------------------------------------------------------------------- #


@pytest.fixture
def files(config) -> FilesTool:
    return FilesTool(config)


async def test_write_with_upload_attaches_an_artifact(files: FilesTool, config) -> None:
    result = await files.invoke(
        {"action": "write", "path": "workspace/out.md", "content": "# hi", "upload": True},
        ctx(source="cli"),
    )
    assert result.ok
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.path == "workspace/out.md"
    assert artifact.absolute.is_file(), "staged in the outbox"
    assert (config.shell_cwd / "out.md").read_text(encoding="utf-8") == "# hi"
    assert "will be sent to the chat" in result.text


async def test_write_without_upload_attaches_nothing(files: FilesTool) -> None:
    result = await files.invoke(
        {"action": "write", "path": "workspace/quiet.md", "content": "x"}, ctx(source="cli")
    )
    assert result.ok
    assert result.artifacts == []


async def test_upload_action_sends_an_existing_file(files: FilesTool, config) -> None:
    (config.shell_cwd / "made-by-shell.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    result = await files.invoke({"action": "upload", "path": "workspace/made-by-shell.csv"}, ctx())
    assert result.ok
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "workspace/made-by-shell.csv"
    # The upload action never modifies the file.
    assert (config.shell_cwd / "made-by-shell.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


async def test_upload_of_a_missing_file_fails_cleanly(files: FilesTool) -> None:
    result = await files.invoke({"action": "upload", "path": "workspace/ghost.md"}, ctx())
    assert not result.ok
    assert "no such file" in result.text
    assert result.artifacts == []


async def test_upload_of_a_disallowed_file_degrades_to_a_note(
    files: FilesTool, config
) -> None:
    (config.root / "workspace" / "binary.exe").write_bytes(b"MZ")
    result = await files.invoke({"action": "upload", "path": "workspace/binary.exe"}, ctx())
    assert not result.ok
    assert "refusing to deliver" in result.text


async def test_uploads_disabled_refuses_the_upload_action(files: FilesTool, config) -> None:
    config.tools.files.uploads = False
    (config.shell_cwd / "x.md").write_text("x", encoding="utf-8")
    result = await files.invoke({"action": "upload", "path": "workspace/x.md"}, ctx())
    assert not result.ok
    assert "uploads = false" in result.text


async def test_uploads_disabled_ignores_the_write_flag(files: FilesTool, config) -> None:
    config.tools.files.uploads = False
    result = await files.invoke(
        {"action": "write", "path": "workspace/y.md", "content": "y", "upload": True}, ctx()
    )
    # The write itself still succeeds; only delivery is off.
    assert result.ok
    assert result.artifacts == []
    assert "will be sent" not in result.text


async def test_upload_outside_the_project_is_refused(files: FilesTool) -> None:
    """A readable-but-not-deliverable path fails the delivery, not the tool."""
    result = await files.invoke({"action": "upload", "path": "/etc/os-release"}, ctx())
    assert not result.ok
    assert "outside the project" in result.text


def test_unknown_action_names_upload_as_an_option(files: FilesTool) -> None:
    with pytest.raises(Exception, match="upload"):
        import asyncio

        asyncio.run(files.invoke({"action": "teleport"}, ctx()))


def test_summary_line_mentions_delivery(files: FilesTool) -> None:
    assert "send files to the chat" in files.summary_line()
    files.settings.uploads = False
    assert "delivery disabled" in files.summary_line()


def test_the_schema_advertises_upload(files: FilesTool) -> None:
    spec = files.spec()["function"]
    actions = spec["parameters"]["properties"]["action"]["enum"]
    assert "upload" in actions
    assert "upload" in spec["parameters"]["properties"]


# --------------------------------------------------------------------------- #
# the agent loop: collection
# --------------------------------------------------------------------------- #


async def test_the_agent_collects_artifacts_into_the_turn(config) -> None:
    from lumi.agent import Agent
    from lumi.memory import History, MemoryFile
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config,
        Personality.load(config.personality_file),
        memory,
        History(config.history_dir),
        build_registry(config, memory),
        FakeLLM(
            [
                make_reply(
                    "",
                    [
                        (
                            "c1",
                            "files",
                            {
                                "action": "write",
                                "path": "workspace/agent-made.md",
                                "content": "made by the agent",
                                "upload": True,
                            },
                        )
                    ],
                ),
                make_reply("here is your file"),
            ]
        ),
    )
    result = await agent.handle(42, "make me a file")

    assert result.text == "here is your file"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "workspace/agent-made.md"
    assert result.artifacts[0].absolute.is_file()


async def test_a_turn_without_uploads_collects_no_artifacts(config) -> None:
    from lumi.agent import Agent
    from lumi.memory import History, MemoryFile
    from lumi.personality import Personality
    from lumi.tools import build_registry

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(
        config,
        Personality.load(config.personality_file),
        memory,
        History(config.history_dir),
        build_registry(config, memory),
        FakeLLM([make_reply("just words")]),
    )
    result = await agent.handle(42, "hi")
    assert result.artifacts == []
