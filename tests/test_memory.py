"""MEMORY.md handling and the JSONL transcript."""

from __future__ import annotations

from pathlib import Path

import pytest

from lumi.memory import MANAGED_END, MANAGED_START, History, MemoryFile


@pytest.fixture
def memory(tmp_path: Path) -> MemoryFile:
    path = tmp_path / "MEMORY.md"
    path.write_text(
        "# Memory\n\n"
        "## Facts\n\n"
        f"{MANAGED_START}\n"
        "<!-- scaffolding comment -->\n"
        f"{MANAGED_END}\n\n"
        "## Context\n\nhand-written notes that must survive\n",
        encoding="utf-8",
    )
    return MemoryFile(path=path, max_chars=6000)


# --------------------------------------------------------------------------- #
# remember
# --------------------------------------------------------------------------- #


def test_remember_appends_inside_the_markers(memory: MemoryFile) -> None:
    memory.remember("the owner prefers tabs")
    text = memory.text
    assert MANAGED_START in text and MANAGED_END in text
    assert text.index(MANAGED_START) < text.index("prefers tabs") < text.index(MANAGED_END)


def test_remember_preserves_hand_written_content(memory: MemoryFile) -> None:
    memory.remember("a new fact")
    assert "hand-written notes that must survive" in memory.text
    assert "<!-- scaffolding comment -->" in memory.text
    assert "## Context" in memory.text


def test_remember_is_idempotent_for_an_exact_duplicate(memory: MemoryFile) -> None:
    assert "remembered" in memory.remember("same fact")
    assert "already remembered" in memory.remember("same fact")
    assert memory.managed() == ["same fact"]


def test_remember_rejects_empty_text(memory: MemoryFile) -> None:
    assert "empty" in memory.remember("   ")
    assert memory.managed() == []


def test_remember_collapses_whitespace(memory: MemoryFile) -> None:
    memory.remember("spaced    out\n\nfact")
    assert memory.managed() == ["spaced out fact"]


def test_remember_truncates_an_enormous_fact(memory: MemoryFile) -> None:
    memory.remember("x" * 2000)
    assert len(memory.managed()[0]) <= 500


def test_remember_creates_the_file_and_markers(tmp_path: Path) -> None:
    fresh = MemoryFile(path=tmp_path / "sub" / "MEMORY.md")
    fresh.remember("works even from nothing")
    assert fresh.managed() == ["works even from nothing"]
    assert MANAGED_START in fresh.text


def test_remember_keeps_the_file_parseable(tmp_path: Path) -> None:
    fresh = MemoryFile(path=tmp_path / "MEMORY.md")
    for i in range(5):
        fresh.remember(f"fact {i}")
    assert fresh.managed() == [f"fact {i}" for i in range(5)]


# --------------------------------------------------------------------------- #
# forget
# --------------------------------------------------------------------------- #


def test_forget_removes_the_newest_first(memory: MemoryFile) -> None:
    for fact in ("one", "two", "three"):
        memory.remember(fact)
    assert memory.forget(1) == ["three"]
    assert memory.managed() == ["one", "two"]


def test_forget_can_remove_several(memory: MemoryFile) -> None:
    for fact in ("one", "two", "three"):
        memory.remember(fact)
    assert memory.forget(2) == ["two", "three"]
    assert memory.managed() == ["one"]


def test_forget_more_than_exists_clears_everything(memory: MemoryFile) -> None:
    for fact in ("one", "two"):
        memory.remember(fact)
    assert memory.forget(99) == ["one", "two"]
    assert memory.managed() == []


def test_forget_keeps_scaffolding_and_hand_written_text(memory: MemoryFile) -> None:
    memory.remember("a fact")
    memory.forget(1)
    text = memory.text
    assert "<!-- scaffolding comment -->" in text
    assert "hand-written notes that must survive" in text
    assert text.count(MANAGED_START) == 1  # not duplicated
    assert text.count("scaffolding comment") == 1


def test_forget_on_an_empty_memory_is_a_no_op(memory: MemoryFile) -> None:
    assert memory.forget(1) == []
    assert "scaffolding comment" in memory.text


def test_forget_zero_does_nothing(memory: MemoryFile) -> None:
    memory.remember("keep me")
    assert memory.forget(0) == []
    assert memory.managed() == ["keep me"]


# --------------------------------------------------------------------------- #
# read paths
# --------------------------------------------------------------------------- #


def test_search_matches_case_insensitively(memory: MemoryFile) -> None:
    memory.remember("Deploys with Fly.io")
    memory.remember("Prefers Neovim")
    assert memory.search("fly") == ["Deploys with Fly.io"]
    assert memory.search("NEOVIM") == ["Prefers Neovim"]
    assert memory.search("nothing here") == []


def test_search_ignores_single_characters(memory: MemoryFile) -> None:
    memory.remember("a fact about x")
    assert memory.search("x") == ["a fact about x"]  # falls back to everything


def test_for_prompt_truncates_and_says_so(tmp_path: Path) -> None:
    small = MemoryFile(path=tmp_path / "MEMORY.md", max_chars=200)
    small.remember("x" * 400)
    rendered = small.for_prompt()
    assert len(rendered) <= 200
    assert "elided" in rendered


def test_for_prompt_passes_short_memory_through(memory: MemoryFile) -> None:
    assert "hand-written notes" in memory.for_prompt()


def test_stats(memory: MemoryFile) -> None:
    memory.remember("one")
    memory.remember("two")
    stats = memory.stats()
    assert stats["managed_count"] == 2
    assert stats["bullet_count"] == 2
    assert stats["chars"] > 0


def test_reload_picks_up_external_edits(memory: MemoryFile) -> None:
    memory.remember("one")
    memory.path.write_text(
        memory.path.read_text(encoding="utf-8").replace("one", "edited by hand"),
        encoding="utf-8",
    )
    memory.ensure_loaded()
    assert memory.managed() == ["edited by hand"]


# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #


@pytest.fixture
def history(tmp_path: Path) -> History:
    return History(tmp_path / "history")


def test_append_and_read(history: History) -> None:
    history.append(7, "user", "hello", session="s1")
    turns = history.read(7)
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert turns[0].content == "hello"
    assert turns[0].session == "s1"


def test_unicode_survives_the_round_trip(history: History) -> None:
    history.append(7, "user", "emoji 🌙 and ünïcode")
    assert history.read(7)[0].content == "emoji 🌙 and ünïcode"


def test_malformed_lines_are_skipped_not_fatal(history: History) -> None:
    history.append(7, "user", "good")
    with history.path_for(7).open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    history.append(7, "user", "also good")
    assert [t.content for t in history.read(7)] == ["good", "also good"]


def test_chats_are_separate_files(history: History) -> None:
    history.append(1, "user", "chat one")
    history.append(2, "user", "chat two")
    assert history.read(1)[0].content == "chat one"
    assert history.read(2)[0].content == "chat two"


def test_recent_dialogue_excludes_tool_chatter(history: History) -> None:
    history.append(1, "user", "run ls")
    history.append(1, "tool", "output of ls", tool="run_shell")
    history.append(1, "assistant", "here it is")
    dialogue = history.recent_dialogue(1, 10)
    assert [d["role"] for d in dialogue] == ["user", "assistant"]


def test_recent_dialogue_drops_a_leading_assistant(history: History) -> None:
    history.append(1, "assistant", "orphan")
    history.append(1, "user", "real question")
    assert history.recent_dialogue(1, 10) == [{"role": "user", "content": "real question"}]


def test_recent_dialogue_respects_the_turn_window(history: History) -> None:
    for i in range(10):
        history.append(1, "user", f"q{i}")
        history.append(1, "assistant", f"a{i}")
    dialogue = history.recent_dialogue(1, 2)
    assert len(dialogue) == 4
    assert dialogue[0]["content"] == "q8"


def test_clear_removes_the_transcript(history: History) -> None:
    history.append(1, "user", "secret")
    assert history.clear(1) is True
    assert history.read(1) == []
    assert history.clear(1) is False


def test_read_of_an_unknown_chat_is_empty(history: History) -> None:
    assert history.read(999) == []


def test_sessions_are_distinct(history: History) -> None:
    assert history.new_session(1) != history.new_session(1)
