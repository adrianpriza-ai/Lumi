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


def make_window(config):
    """A context window at the test size, for the memory budget rules."""
    from lumi.context import ContextWindow

    config.llm.context_window = 4_000
    config.llm.context_headroom = 400
    return ContextWindow(config)


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


def test_for_prompt_drops_the_oldest_facts_not_the_newest(tmp_path: Path) -> None:
    """The bug this pins: the file is truncated from the top, which threw away
    the most recent facts — the ones still being acted on — and kept ancient
    ones nobody would ever ask about again."""
    memory = MemoryFile(path=tmp_path / "MEMORY.md", max_chars=0)
    memory.load()
    for i in range(60):
        memory.remember(f"fact {i} " + "x" * 100)

    rendered = memory.for_prompt(1_500)
    assert "fact 59" in rendered, "the newest fact must always survive"
    assert "fact 0 " not in rendered, "the oldest fact is the one to let go"
    assert "elided" in rendered
    assert len(rendered) <= 1_500


def test_for_prompt_keeps_the_structure_of_the_file(tmp_path: Path) -> None:
    memory = MemoryFile(path=tmp_path / "MEMORY.md", max_chars=0)
    memory.path.write_text(
        f"# Memory\n\n## Facts\n\n{MANAGED_START}\n<!-- managed -->\n{MANAGED_END}\n"
        "\n## Context\n\nnotes\n",
        encoding="utf-8",
    )
    memory.load()
    for i in range(60):
        memory.remember(f"fact {i} " + "x" * 100)

    rendered = memory.for_prompt(1_500)
    assert "## Facts" in rendered
    assert "<!-- managed -->" in rendered
    assert "## Context" in rendered
    assert rendered.count(MANAGED_START) == 1
    assert rendered.count(MANAGED_END) == 1


def test_one_oversized_fact_is_cut_but_still_sent(tmp_path: Path) -> None:
    """A file whose newest fact cannot fit has stopped being memory. It is cut
    rather than dropped, and the cap still holds."""
    memory = MemoryFile(path=tmp_path / "MEMORY.md", max_chars=400)
    memory.load()
    memory.remember("y" * 5_000)

    rendered = memory.for_prompt()
    assert len(rendered) <= 400
    assert "yyyy" in rendered


def test_a_zero_budget_means_the_window_decides(config) -> None:
    """max_memory_chars = 0 is not "unbounded": the file is held to an eighth
    of the window, because it goes in the part of the prompt nothing can
    condense away."""
    from lumi.context import ContextWindow

    window = make_window(config)
    limit = window.memory_limit()
    assert limit == window.chars_for(window.budget // 8)
    assert 0 < limit < config.llm.context_window // 2
    # A bigger window gives memory more room, but never all of it.
    config.llm.context_window = config.llm.context_window * 2
    assert 1.9 * limit < ContextWindow(config).memory_limit() < config.llm.context_window // 2


def test_an_explicit_memory_cap_wins_over_the_window_share(config) -> None:
    config.llm.max_memory_chars = 5_000
    assert make_window(config).memory_limit() == 5_000


def test_prompt_size_reports_what_did_not_fit(tmp_path: Path) -> None:
    memory = MemoryFile(path=tmp_path / "MEMORY.md", max_chars=0)
    memory.load()
    for i in range(60):
        memory.remember(f"fact {i} " + "x" * 100)

    size = memory.prompt_size(1_500)
    assert size["over"] is True
    assert size["dropped"] > 0
    assert size["sent_chars"] <= 1_500
    assert size["chars"] > size["sent_chars"]


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
    dialogue = history.recent_dialogue(1, max_messages=10)
    assert [d["role"] for d in dialogue.messages] == ["user", "assistant"]


def test_recent_dialogue_drops_a_leading_assistant(history: History) -> None:
    history.append(1, "assistant", "orphan")
    history.append(1, "user", "real question")
    assert history.recent_dialogue(1, max_messages=10).messages == [
        {"role": "user", "content": "real question"}
    ]


def test_recent_dialogue_takes_everything_that_fits(history: History) -> None:
    for i in range(10):
        history.append(1, "user", f"q{i}")
        history.append(1, "assistant", f"a{i}")
    dialogue = history.recent_dialogue(1)
    assert len(dialogue.messages) == 20
    assert dialogue.messages[0]["content"] == "q0"


def test_recent_dialogue_stops_at_the_token_budget(history: History) -> None:
    """The budget is a budget: a chatty transcript loses its oldest turns, and
    loses only those, rather than being cut to a fixed message count."""
    for _ in range(20):
        history.append(1, "user", "x" * 400)
        history.append(1, "assistant", "y" * 400)
    dialogue = history.recent_dialogue(1, budget_tokens=2000)
    assert 0 < len(dialogue.messages) < 40
    assert dialogue.messages[-1]["content"].startswith("y")


def test_recent_dialogue_respects_a_message_ceiling(history: History) -> None:
    for i in range(10):
        history.append(1, "user", f"q{i}")
        history.append(1, "assistant", f"a{i}")
    dialogue = history.recent_dialogue(1, max_messages=4)
    assert len(dialogue.messages) == 4
    assert dialogue.messages[0]["content"] == "q8"


def test_a_condensed_record_replaces_the_turns_it_covers(history: History) -> None:
    """A summary row is the only version of the turns behind it, so replay must
    hand back the record and the turns after it — not both."""
    history.append(1, "user", "old question")
    history.append(1, "assistant", "old answer")
    history.append(1, "summary", "they were talking about the router")
    history.append(1, "user", "new question")
    history.append(1, "assistant", "new answer")

    dialogue = history.recent_dialogue(1)
    assert dialogue.summary == "they were talking about the router"
    assert [m["content"] for m in dialogue.messages] == ["new question", "new answer"]


def test_tail_reads_only_the_end_of_the_file(history: History) -> None:
    """Replay must not cost the whole transcript: a year of JSONL is not
    something to pull into memory to answer the next message."""
    for i in range(500):
        history.append(1, "user", f"q{i}" + "z" * 200)
    rows = history.tail(1, max_bytes=4_000)
    assert 0 < len(rows) < 500
    assert rows[-1].content.startswith("q499")


def test_stats_do_not_read_the_transcript(history: History) -> None:
    history.append(1, "user", "hello")
    stats = history.stats(1)
    assert stats["bytes"] > 0
    assert stats["turns"] == 1
    assert stats["truncated"] is False


def test_clear_removes_the_transcript(history: History) -> None:
    history.append(1, "user", "secret")
    assert history.clear(1) is True
    assert history.read(1) == []
    assert history.clear(1) is False


def test_read_of_an_unknown_chat_is_empty(history: History) -> None:
    assert history.read(999) == []


def test_sessions_are_distinct(history: History) -> None:
    assert history.new_session(1) != history.new_session(1)
