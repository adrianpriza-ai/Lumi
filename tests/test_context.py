"""The context window: estimating, fitting, condensing, and replaying.

Unit-level, so each rule is pinned on its own: what counts as a token, what is
never dropped, what happens to a tool result when the window fills, and what a
fresh process inherits from a transcript.
"""

from __future__ import annotations

from pathlib import Path

from lumi.context import (
    ELIDED_FLAG,
    SUMMARY_FLAG,
    ContextWindow,
    blocks,
    content_chars,
    flatten,
)
from lumi.memory import History, MemoryFile
from lumi.personality import Personality


class ScriptedLLM:
    """Answers a condensation call with a record, and everything else with text."""

    def __init__(self, record: str = "the record", refuse: bool = False) -> None:
        self.record = record
        self.refuse = refuse
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        if self.refuse:
            return type("Reply", (), {"text": ""})()
        if "condensing the earlier part" in str(messages[0].get("content", "")):
            return type("Reply", (), {"text": self.record})()
        return type("Reply", (), {"text": "answered"})()


class FakeConv:
    """The parts of a Conversation the context manager touches."""

    def __init__(self, messages=None, chat_id: int = 1) -> None:
        self.chat_id = chat_id
        self.messages = list(messages or [])
        self.session = "s1"
        self.summary = ""
        self.summarised = 0
        self.elided = 0
        self.dropped = 0
        self.truncated = False


def make_window(config, llm=None, history=None) -> ContextWindow:
    config.llm.context_window = 4_000
    config.llm.context_headroom = 400
    config.llm.context_keep_recent = 4
    return ContextWindow(config, llm=llm, history=history)


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #


def test_a_multimodal_turn_counts_its_text_not_its_image(config) -> None:
    """A base64 photo is charged in vision tokens by the provider, not in
    characters; counting the megabytes of it would be nonsense."""
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 50_000}},
        ],
    }
    assert content_chars(message) == len("what is this")


def test_tool_call_arguments_count_towards_the_estimate(config) -> None:
    window = make_window(config)
    small = {"role": "assistant", "tool_calls": [
        {"function": {"name": "files", "arguments": "{}"}}
    ]}
    large = {"role": "assistant", "tool_calls": [
        {"function": {"name": "files", "arguments": "x" * 20_000}}
    ]}
    assert window.estimate([large]) - window.estimate([small]) > 5_000


def test_the_provider_prompt_count_recalibrates_the_estimate(config) -> None:
    """Every provider reports the real prompt size, so the guess is corrected
    from it rather than trusted for the whole conversation. Code is denser than
    prose, so a real reading usually makes the estimate more cautious."""
    window = make_window(config)
    messages = [{"role": "user", "content": "def f(x): return x ** 2"}]
    before = window.tokens_for(messages[0]["content"])
    window.observe({"prompt": window.estimate(messages) * 2}, messages)

    assert window.report(FakeConv(messages)).calibrated is True
    assert window.chars_per_token < 3.5
    assert window.tokens_for(messages[0]["content"]) >= before


def test_an_implausible_prompt_count_is_ignored(config) -> None:
    """One odd reading must not be able to make the window look twice its size."""
    window = make_window(config)
    messages = [{"role": "user", "content": "hello there"}]
    before = window.chars_per_token
    window.observe({"prompt": 1}, messages)
    assert window.chars_per_token == before
    window.observe({"prompt": 10_000_000}, messages)
    assert window.chars_per_token == before
    window.observe({"prompt": 0}, messages)
    assert window.chars_per_token == before


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #


def test_an_assistant_and_its_tool_results_are_one_block() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": "output"},
        {"role": "assistant", "content": "done"},
    ]
    groups = blocks(messages)
    assert [len(g) for g in groups] == [1, 1, 2, 1]
    assert flatten(groups) == messages


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


async def test_a_conversation_within_the_window_is_left_alone(config) -> None:
    window = make_window(config)
    conv = FakeConv([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    await window.prepare(conv)
    assert len(conv.messages) == 2
    assert conv.elided == conv.summarised == conv.dropped == 0


async def test_old_tool_output_is_elided_before_anything_else(config) -> None:
    window = make_window(config)
    conv = FakeConv(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "log it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {}}]},
            {"role": "tool", "tool_call_id": "1", "name": "run_shell", "content": "x" * 40_000},
            {"role": "assistant", "content": "all done"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ]
    )
    before = window.estimate(conv.messages)
    await window.prepare(conv)

    assert before > window.budget
    assert conv.elided == 1
    assert conv.summarised == 0, "eliding was enough; no call needed"
    tool = next(m for m in conv.messages if m.get("role") == "tool")
    assert tool[ELIDED_FLAG] is True
    assert "run_shell" in tool["content"]
    assert window.estimate(conv.messages) <= window.budget


async def test_the_record_replaces_the_turns_it_covers(config) -> None:
    llm = ScriptedLLM("they were working on the router")
    window = make_window(config, llm=llm)
    messages = [{"role": "system", "content": "s"}]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"question {i} " + "detail " * 200},
            {"role": "assistant", "content": f"answer {i} " + "detail " * 200},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert conv.summarised > 0
    assert conv.summary == "they were working on the router"
    assert len(llm.calls) == 1, "condensing is one extra call, not one per message"
    note = next(m for m in conv.messages if m.get(SUMMARY_FLAG))
    assert "they were working on the router" in note["content"]
    assert window.estimate(conv.messages) <= window.budget


async def test_the_newest_turns_are_never_condensed_away(config) -> None:
    window = make_window(config, llm=ScriptedLLM())
    messages = [{"role": "system", "content": "s"}]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"old question {i} " + "detail " * 200},
            {"role": "assistant", "content": f"old answer {i} " + "detail " * 200},
        ]
    messages += [{"role": "user", "content": "the newest question"}]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert conv.messages[-1]["content"] == "the newest question"
    assert any("the newest question" in str(m.get("content")) for m in conv.messages)


async def test_a_refused_record_costs_nothing_but_the_drops(config) -> None:
    """If the model will not write a record, the turns are still dropped rather
    than the conversation being stuck — the record is a nicety, not a hostage."""
    window = make_window(config, llm=ScriptedLLM(refuse=True))
    messages = [{"role": "system", "content": "s"}]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"question {i} " + "detail " * 200},
            {"role": "assistant", "content": f"answer {i} " + "detail " * 200},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert conv.summary == ""
    assert conv.dropped > 0
    assert window.estimate(conv.messages) <= window.budget


async def test_compaction_can_be_switched_off(config) -> None:
    config.llm.compaction = False
    llm = ScriptedLLM()
    window = make_window(config, llm=llm)
    messages = [{"role": "system", "content": "s"}]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"question {i} " + "detail " * 200},
            {"role": "assistant", "content": f"answer {i} " + "detail " * 200},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert llm.calls == []
    assert conv.summarised == 0
    assert conv.dropped > 0


async def test_a_window_smaller_than_the_floor_is_reported_not_mangled(config) -> None:
    """Cutting the system prompt to make a number fit would be the worst
    possible trade, so the misconfiguration is named instead."""
    config.llm.context_window = 300
    config.llm.context_headroom = 0
    config.llm.context_keep_recent = 2
    window = ContextWindow(config, llm=ScriptedLLM())
    system = "s" * 4_000
    conv = FakeConv(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    await window.prepare(conv)

    assert conv.truncated is True
    assert conv.messages[0]["content"] == system
    assert "context_window" in "\n".join(window.report(conv).lines())


async def test_the_record_is_capped_against_the_window(config) -> None:
    """A chatty summariser must not be able to replace the conversation with
    its own monologue."""
    window = make_window(config, llm=ScriptedLLM("r" * 200_000))
    messages = [{"role": "system", "content": "s"}]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"question {i} " + "detail " * 200},
            {"role": "assistant", "content": f"answer {i} " + "detail " * 200},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert len(conv.summary) <= window.chars_for(window.budget // 4)
    assert window.estimate(conv.messages) <= window.budget


async def test_only_one_record_is_ever_installed(config) -> None:
    """Repeated condensation updates the record in place; a second note would
    read to the model as a second, contradictory history."""
    llm = ScriptedLLM()
    window = make_window(config, llm=llm)
    messages = [{"role": "system", "content": "s"}]
    for i in range(10):
        messages += [
            {"role": "user", "content": f"question {i} " + "detail " * 600},
            {"role": "assistant", "content": f"answer {i} " + "detail " * 600},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)
    assert len([m for m in conv.messages if m.get(SUMMARY_FLAG)]) == 1

    for round_ in range(3):
        conv.messages += [
            {"role": "user", "content": f"q{round_} " + "detail " * 900},
            {"role": "assistant", "content": f"a{round_} " + "detail " * 900},
        ]
        await window.prepare(conv)
        notes = [m for m in conv.messages if m.get(SUMMARY_FLAG)]
        assert len(notes) == 1
        assert conv.messages[1].get(SUMMARY_FLAG) is True


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def make_history(config) -> History:
    return History(config.history_dir)


def test_replay_packs_as_much_of_the_transcript_as_fits(config) -> None:
    history = make_history(config)
    for i in range(40):
        history.append(1, "user", f"question {i} " + "detail " * 100)
        history.append(1, "assistant", f"answer {i} " + "detail " * 100)
    window = make_window(config, history=history)
    config.llm.context_window = 6_000
    conv = FakeConv([{"role": "system", "content": "s" * 500}])
    window.replay(conv)

    assert len(conv.messages) > 1
    assert window.estimate(conv.messages) <= window.budget
    assert conv.messages[-1]["content"].startswith("answer 39")


def test_replay_hands_back_to_the_soft_target_after_installing_a_record(config) -> None:
    """The summary note is installed after transcript packing and counts too."""
    history = make_history(config)
    history.append(1, "summary", "record " + "important detail " * 40)
    for i in range(20):
        history.append(1, "user", f"question {i} " + "detail " * 100)
        history.append(1, "assistant", f"answer {i} " + "detail " * 100)

    window = make_window(config, history=history)
    config.llm.context_target = 0.5
    conv = FakeConv([{"role": "system", "content": "s" * 500}])
    window.replay(conv)

    assert window.estimate(conv.messages) <= window.target
    assert conv.messages[1].get(SUMMARY_FLAG) is True
    assert conv.messages[-1]["content"].startswith("answer 19")


def test_replay_inherits_a_record_written_by_a_previous_process(config) -> None:
    history = make_history(config)
    history.append(1, "user", "old question")
    history.append(1, "assistant", "old answer")
    history.append(1, "summary", "they were wiring the router")
    history.append(1, "user", "and then?")
    history.append(1, "assistant", "then this")

    window = make_window(config, history=history)
    conv = FakeConv([{"role": "system", "content": "s"}])
    window.replay(conv)

    assert conv.summary == "they were wiring the router"
    assert conv.messages[1].get(SUMMARY_FLAG) is True
    assert [m["content"] for m in conv.messages[2:]] == ["and then?", "then this"]


def test_replay_of_an_unknown_chat_is_empty(config) -> None:
    window = make_window(config, history=make_history(config))
    conv = FakeConv([{"role": "system", "content": "s"}])
    window.replay(conv)
    assert len(conv.messages) == 1


def test_the_whole_memory_file_reaches_the_prompt(config) -> None:
    """A memory file that gets cut is a memory the model cannot use."""
    path = Path(config.memory_file)
    path.write_text("# Memory\n\n" + "\n".join(f"- fact {i}" for i in range(1_000)), encoding="utf-8")
    memory = MemoryFile(path, 0)
    memory.load()
    assert memory.for_prompt() == memory.text.strip()
    assert len(memory.for_prompt()) > 6_000


def test_personality_and_memory_still_build_a_prompt(config) -> None:
    """A smoke test for the pieces the agent assembles every turn."""
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    assert Personality.load(config.personality_file).text.strip()
    assert "Memory" in memory.for_prompt()


# --------------------------------------------------------------------------- #
# the fill target (llm.context_target)
# --------------------------------------------------------------------------- #


def test_the_target_defaults_to_the_hard_budget(config) -> None:
    """No target set: the fill line is the hard budget, as before."""
    window = make_window(config)
    assert window.target == window.budget  # context_target defaults to 1.0
    assert window.effective_budget == window.budget


def test_a_lower_target_pulls_the_fill_line_down(config) -> None:
    window = make_window(config)
    config.llm.context_target = 0.5
    assert window.target == window.budget // 2
    assert window.effective_budget == window.target


def test_the_target_never_raises_the_hard_budget(config) -> None:
    """A target of 1.0 on a window with headroom must not exceed the floor."""
    window = make_window(config)
    config.llm.context_target = 1.0
    assert window.target == window.budget  # 1.0 of the budget, not of the window…
    assert window.effective_budget == window.budget  # …and the floor is what holds


def test_an_absurd_target_is_clamped_to_the_budget(config) -> None:
    window = make_window(config)
    config.llm.context_target = 5.0
    assert window.target == window.budget
    assert window.effective_budget == window.budget


def test_a_tiny_target_still_has_a_floor(config) -> None:
    """Even a 1% target keeps a few tokens, so the estimator never divides by 0."""
    window = make_window(config)
    config.llm.context_target = 0.01
    assert window.target >= 256


def test_a_nonsense_target_falls_back_to_the_budget(config) -> None:
    """A target outside (0.0, 1.0] is a misconfiguration, and it falls back to
    1.0 — the pre-target behaviour. Falling back to the floor instead would
    crush the conversation to the system prompt and its tail."""
    window = make_window(config)
    for broken in (-1.0, 0.0, 5.0, "nonsense", None):
        config.llm.context_target = broken
        assert window.target == window.budget, broken
        assert window.effective_budget == window.budget, broken


async def test_a_negative_target_does_not_crush_the_conversation(config) -> None:
    """The same fallback, seen from the conversation: a bad setting must not
    cost the bot its history."""
    window = make_window(config)
    config.llm.context_target = -1.0
    window.llm = ScriptedLLM()
    messages = [{"role": "system", "content": "PERSONALITY " * 20}]
    for i in range(8):
        messages += [
            {"role": "user", "content": f"q{i} " + "detail " * 60},
            {"role": "assistant", "content": f"a{i} " + "detail " * 60},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert len(conv.messages) == len(messages)
    assert conv.dropped == 0
    assert conv.summarised == 0


def test_memory_share_follows_the_soft_target(config) -> None:
    window = make_window(config)
    config.llm.context_target = 0.5
    assert window.memory_limit() == window.chars_for(window.effective_budget // 8)


async def test_a_soft_target_trims_messages_back_to_the_fill_line(config) -> None:
    """Above the target but below the hard budget, older *messages* are dropped
    until the conversation is back under the fill line."""
    window = make_window(config)  # window 4000, budget 3600
    config.llm.context_target = 0.5  # fill line = half the budget = 1800
    window.llm = ScriptedLLM()  # compaction may summarise, then drop the rest

    system = {"role": "system", "content": "PERSONALITY INSTRUCTIONS" * 20}
    messages = [system]
    for i in range(12):
        messages += [
            {"role": "user", "content": f"old question {i} " + "detail " * 60},
            {"role": "assistant", "content": f"old answer {i} " + "detail " * 60},
        ]
    recent = [
        {"role": "user", "content": "the newest question"},
        {"role": "assistant", "content": "the newest answer"},
    ]
    conv = FakeConv(messages + recent)

    before = window.estimate(conv.messages)
    assert window.budget >= before > window.target  # in the target→budget band
    await window.prepare(conv)

    assert window.estimate(conv.messages) <= window.target
    # The system prompt survives untouched, and so do the newest turns.
    assert conv.messages[0] is system
    assert conv.messages[0]["content"] == "PERSONALITY INSTRUCTIONS" * 20
    assert {m["content"] for m in conv.messages[-2:]} == {
        "the newest question",
        "the newest answer",
    }
    # Something was given up to get under the target.
    assert conv.dropped + conv.summarised > 0


async def test_the_system_prompt_is_never_a_shed_candidate(config) -> None:
    """Even when the system prompt alone is large, trimming works around it —
    the identity and memory are not negotiable, only the conversation is."""
    window = make_window(config)
    config.llm.context_target = 0.5
    window.llm = ScriptedLLM()

    system = {"role": "system", "content": "IDENTITY " * 400}  # ~big, on purpose
    messages = [system]
    for i in range(16):
        messages += [
            {"role": "user", "content": f"q{i} " + "bulk " * 50},
            {"role": "assistant", "content": f"a{i} " + "bulk " * 50},
        ]
    conv = FakeConv(messages)
    await window.prepare(conv)

    assert conv.messages[0] is system
    assert conv.messages[0]["content"] == "IDENTITY " * 400


async def test_the_protected_tail_never_swallows_the_system_prompt(config) -> None:
    """``keep_recent`` larger than the whole conversation counts back past the
    newest turn to the system prompt, which put the head inside the tail — the
    same dict object twice — so the request carried the identity twice and the
    last-resort shrink then cut it in place. The tail may never contain the
    head."""
    window = make_window(config)  # window 4000, budget 3600, keep_recent 4
    config.llm.context_target = 0.5  # fill line 1800
    window.llm = ScriptedLLM()

    system = {"role": "system", "content": "IDENTITY " * 1_000}
    conv = FakeConv(
        [
            system,  # the whole conversation: keep_recent reaches past it
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    before = window.estimate(conv.messages)
    assert window.budget > before > window.target  # over the fill line, inside the budget
    await window.prepare(conv)

    assert sum(m is system for m in conv.messages) == 1
    assert [m["role"] for m in conv.messages] == ["system", "user", "assistant"]
    assert conv.messages[0]["content"] == "IDENTITY " * 1_000
    assert conv.elided == 0  # the shrink had nothing to cut


async def test_an_unreachable_target_is_clamped_to_the_protected_tail(config, caplog) -> None:
    """A soft target cannot be lower than the system prompt plus recent turns.

    The old path dropped every middle turn, then quietly finished above the
    configured target. The reachable floor should be used instead and named in
    the log.
    """
    window = make_window(config)
    config.llm.context_target = 0.5
    window.llm = ScriptedLLM()

    system = {"role": "system", "content": "IDENTITY " * 1_000}
    recent = [
        {"role": "user", "content": "recent question " + "detail " * 150},
        {"role": "assistant", "content": "recent answer " + "detail " * 150},
        {"role": "user", "content": "latest question " + "detail " * 150},
        {"role": "assistant", "content": "latest answer " + "detail " * 150},
    ]
    middle = [
        {"role": "user", "content": "older question " + "detail " * 80},
        {"role": "assistant", "content": "older answer " + "detail " * 80},
    ]
    conv = FakeConv([system, *middle, *recent])

    assert window.estimate(conv.messages) > window.target
    assert window.estimate([system, *recent]) > window.target

    with caplog.at_level("WARNING"):
        await window.prepare(conv)

    used = window.estimate(conv.messages)
    assert window.target < used <= window.budget
    assert any("context target" in record.message and "unreachable" in record.message for record in caplog.records)


def test_the_report_names_the_target_when_one_is_set(config) -> None:
    window = make_window(config)
    config.llm.context_target = 0.7
    conv = FakeConv([{"role": "system", "content": "s"}])
    body = "\n".join(window.report(conv).lines())
    assert window.report(conv).target == int(window.budget * 0.7)
    assert "fill target" in body
    # Both shares, because the setting is a share of the budget while the
    # number the owner cares about is the window.
    assert "70%" in body  # 2520 of a 3600-token budget
    assert "63%" in body  # 2520 of a 4000-token window


def test_the_report_omits_the_target_line_by_default(config) -> None:
    window = make_window(config)
    conv = FakeConv([{"role": "system", "content": "s"}])
    body = "\n".join(window.report(conv).lines())
    assert "fill target" not in body
