"""The agent loop: tool dispatch, approval handshake, and the iteration cap.

Driven with a scripted LLM so every branch is reachable without a network call:
plain answers, tool use, a tool that needs approval, denial, and a runaway loop.
"""

from __future__ import annotations

import json

from conftest import FakeLLM, make_reply

from lumi.agent import APPROVAL_TTL_SECONDS, Agent
from lumi.llm.base import LLMError
from lumi.memory import History, MemoryFile
from lumi.personality import Personality
from lumi.tools import build_registry

CHAT = 1234


class CondensingLLM(FakeLLM):
    """Answers normally, and plays along when asked to condense a conversation.

    Distinguishing the two by the request rather than by call order is what
    keeps the context tests readable: a compaction happens whenever the window
    fills, which is not something a test should have to count.
    """

    def __init__(self, answer: str = "answer") -> None:
        super().__init__([])
        self.answer = answer
        self.condensed = 0

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        first = str((messages[0] if messages else {}).get("content", ""))
        if "condensing the earlier part" in first:
            self.condensed += 1
            return make_reply(f"record {self.condensed}: questions up to {self.condensed} were settled")
        return make_reply(self.answer)


def make_agent(config, replies, llm=None) -> Agent:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    return Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=llm if llm is not None else FakeLLM(replies),
    )


# --------------------------------------------------------------------------- #
# plain conversation
# --------------------------------------------------------------------------- #


async def test_plain_answer(config) -> None:
    agent = make_agent(config, [make_reply("hello there")])
    result = await agent.handle(CHAT, "hi")

    assert result.text == "hello there"
    assert result.ok
    assert result.tools_used == []
    assert not result.needs_approval


async def test_system_prompt_carries_personality_and_memory(config) -> None:
    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "hi")

    system = agent.llm.calls[0][0]
    assert system["role"] == "system"
    assert "You are Lumi" in system["content"]  # PERSONALITY.md
    assert "lumi:managed" in system["content"]  # MEMORY.md
    assert "run_shell" in system["content"]  # tool list


async def test_system_prompt_teaches_telegram_html(config) -> None:
    """The formatting section must pin the model to HTML and away from Markdown."""
    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "hi")

    content = agent.llm.calls[0][0]["content"]
    assert "## Formatting" in content
    assert "<b>like this</b>" in content
    assert "Never use backticks" in content


async def test_system_prompt_hides_tools_without_a_key(config, monkeypatch) -> None:
    """An API-key-gated tool is invisible to the model when its key is unset.

    ``CONTEXT7_API_KEY`` is missing here; the tool is registered but
    ``available()`` says no, so the system prompt must not advertise it.
    """
    monkeypatch.delenv("CONTEXT7_API_KEY", raising=False)
    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "hi")

    system = agent.llm.calls[0][0]["content"]
    assert "context7" not in system.lower() or "_unavailable_" not in system
    # And it should not appear as a callable tool spec either.
    tools = [s for s in agent.registry.specs() if s["function"]["name"] == "context7"]
    assert tools == []


async def test_system_prompt_mentions_context7_when_the_key_is_set(config, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "hi")

    system = agent.llm.calls[0][0]["content"]
    assert "`context7`" in system
    assert any(s["function"]["name"] == "context7" for s in agent.registry.specs())


async def test_history_is_replayed(config) -> None:
    history = History(config.history_dir)
    history.append(CHAT, "user", "earlier question")
    history.append(CHAT, "assistant", "earlier answer")

    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "now what?")

    roles = [m["role"] for m in agent.llm.calls[0]]
    assert roles[:3] == ["system", "user", "assistant"]
    assert agent.llm.calls[0][1]["content"] == "earlier question"
    assert agent.llm.calls[0][-1]["content"] == "now what?"


async def test_turn_is_written_to_history(config) -> None:
    agent = make_agent(config, [make_reply("remembered this")])
    await agent.handle(CHAT, "hello")

    dialogue = History(config.history_dir).recent_dialogue(CHAT)
    assert [d["role"] for d in dialogue.messages] == ["user", "assistant"]
    assert {"role": "user", "content": "hello"} in dialogue.messages
    assert {"role": "assistant", "content": "remembered this"} in dialogue.messages


async def test_llm_error_is_reported_not_raised(config) -> None:
    class Broken(FakeLLM):
        async def complete(self, messages, *, tools=None):
            raise LLMError("upstream is on fire")

    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    agent = Agent(config, Personality.load(config.personality_file), memory,
                  History(config.history_dir), build_registry(config, memory), Broken([]))
    result = await agent.handle(CHAT, "hi")

    assert not result.ok
    assert "on fire" in result.error


# --------------------------------------------------------------------------- #
# tool use
# --------------------------------------------------------------------------- #


async def test_tool_call_then_answer(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("call_1", "run_shell", {"command": "echo hi"})]),
            make_reply("it said hi"),
        ],
    )
    result = await agent.handle(CHAT, "run echo hi")

    assert result.text == "it said hi"
    assert result.tools_used == ["run_shell"]
    assert result.iterations == 2


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


async def test_a_plain_answer_carries_no_reasoning(config) -> None:
    agent = make_agent(config, [make_reply("hello there")])
    result = await agent.handle(CHAT, "hi")
    assert not result.has_reasoning
    assert result.reasoning == ""


async def test_the_trace_reaches_the_turn_result(config) -> None:
    agent = make_agent(
        config, [make_reply("42", reasoning="counted the letters first", reasoning_tokens=310)]
    )
    result = await agent.handle(CHAT, "how many r's in strawberry?")

    assert result.has_reasoning
    assert result.reasoning == "counted the letters first"
    assert result.reasoning_tokens == 310


async def test_traces_accumulate_across_iterations(config) -> None:
    """A tool-using turn thinks more than once, and both passes are interesting."""
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})], reasoning="need to run it"),
            make_reply("it said hi", reasoning="it worked"),
        ],
    )
    result = await agent.handle(CHAT, "run echo hi")

    assert result.reasoning == "need to run it\n\nit worked"


async def test_a_turn_that_pauses_for_approval_keeps_its_thinking(config) -> None:
    """The most interesting trace of all is the one produced before the bot
    stopped to ask the owner. Discarding it would hide exactly the moment worth
    reading."""
    agent = make_agent(
        config,
        [
            make_reply(
                "",
                [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})],
                reasoning="this deletes files; I should ask first",
            )
        ],
    )
    result = await agent.handle(CHAT, "clean up")

    assert result.needs_approval
    assert "ask first" in result.reasoning


async def test_reasoning_tokens_sum_across_iterations(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})], reasoning_tokens=100),
            make_reply("done", reasoning_tokens=50),
        ],
    )
    result = await agent.handle(CHAT, "run it")
    assert result.reasoning_tokens == 150


async def test_the_trace_is_not_replayed_into_the_conversation(config) -> None:
    """Replaying a thinking trace is not something chat-completions asks for, and
    several providers reject the whole request if an assistant turn carries a
    field they do not recognise."""
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})], reasoning="first pass"),
            make_reply("done", reasoning="second pass"),
        ],
    )
    await agent.handle(CHAT, "run it")

    for call in agent.llm.calls:
        for message in call:
            if message["role"] == "assistant":
                assert "first pass" not in str(message)
                assert "second pass" not in str(message)
                assert "reasoning" not in message


async def test_tool_result_is_fed_back_to_the_model(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})]),
            make_reply("done"),
        ],
    )
    await agent.handle(CHAT, "run it")

    second_call = agent.llm.calls[1]
    tool_messages = [m for m in second_call if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "c1"
    assert tool_messages[0]["name"] == "run_shell"
    assert "hi" in tool_messages[0]["content"]


async def test_every_tool_call_gets_exactly_one_tool_message(config) -> None:
    """Two parallel calls must produce two tool messages, or the next request 400s."""
    agent = make_agent(
        config,
        [
            make_reply("", [
                ("c1", "run_shell", {"command": "echo one"}),
                ("c2", "run_shell", {"command": "echo two"}),
            ]),
            make_reply("both done"),
        ],
    )
    await agent.handle(CHAT, "run both")

    tool_messages = [m for m in agent.llm.calls[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]


async def test_assistant_message_preserves_tool_call_shape(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("thinking", [("c1", "run_shell", {"command": "ls"})]),
            make_reply("done"),
        ],
    )
    await agent.handle(CHAT, "ls please")

    assistant = next(m for m in agent.llm.calls[1] if m["role"] == "assistant")
    assert assistant["content"] == "thinking"
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "run_shell"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls"}


async def test_unknown_tool_is_reported_to_the_model(config) -> None:
    agent = make_agent(
        config,
        [make_reply("", [("c1", "teleport", {})]), make_reply("sorry")],
    )
    result = await agent.handle(CHAT, "teleport me")

    tool_message = next(m for m in agent.llm.calls[1] if m["role"] == "tool")
    assert "Unknown tool" in tool_message["content"]
    assert result.text == "sorry"


async def test_iteration_cap_stops_a_runaway_loop(config) -> None:
    config.llm.max_tool_iterations = 3
    agent = make_agent(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "echo loop"})])] * 5,
    )
    result = await agent.handle(CHAT, "loop forever")

    assert result.iterations == 3
    assert "tool iteration limit" in result.error
    assert "one thing at a time" in result.text

    # The stop is recorded *in* the conversation as well as in the result:
    # the next turn used to open on a dangling tool result with no sign the
    # model had been cut off.
    messages = agent.conversation(CHAT).messages
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == result.text

    # And the invariant the cap must not break: one tool message per
    # tool_call_id, even on the iteration that hit the cap.
    asked = {
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    answered = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    assert asked == answered


# --------------------------------------------------------------------------- #
# approval
# --------------------------------------------------------------------------- #


async def test_dangerous_command_needs_approval(config) -> None:
    agent = make_agent(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
    )
    result = await agent.handle(CHAT, "clean up")

    assert result.needs_approval
    assert len(result.pending) == 1
    action = result.pending[0]
    assert action.tool == "run_shell"
    assert action.arguments["command"] == "rm -rf workspace/tmp"
    assert action.preview == "rm -rf workspace/tmp"
    assert "deleting" in action.reason
    # Nothing may have run.
    assert result.tools_used == []


async def test_approval_message_tells_the_model_it_did_not_run(config) -> None:
    agent = make_agent(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
    )
    await agent.handle(CHAT, "clean up")

    # The loop stopped before a second model call, so inspect the conversation.
    messages = agent.conversation(CHAT).messages
    tool_message = next(m for m in messages if m["role"] == "tool")
    assert "Not executed" in tool_message["content"]
    assert "approval" in tool_message["content"]


def last_user_message(agent) -> str:
    """The most recent user message — the outcome injected by resolve()/run_tool()."""
    users = [m.get("content", "") for m in agent.llm.calls[1] if m["role"] == "user"]
    assert users, "the second model call had no user message"
    return users[-1]


async def test_approval_in_a_parallel_set_answers_every_tool_call(config) -> None:
    """An approval must not strand the *other* tool calls in the same reply.

    An assistant message carrying N ``tool_calls`` needs exactly N
    ``role: "tool"`` messages before the next request, or every later turn
    400s and the malformed pair stays in the conversation until ``/reset``.
    """
    agent = make_agent(
        config,
        [
            make_reply("", [
                ("c1", "run_shell", {"command": "rm x.txt"}),
                ("c2", "run_shell", {"command": "ls"}),
                ("c3", "run_shell", {"command": "echo hi"}),
            ]),
            make_reply("done"),
        ],
    )
    result = await agent.handle(CHAT, "tidy up")

    assert result.needs_approval
    assert result.pending[0].arguments["command"] == "rm x.txt"

    messages = agent.conversation(CHAT).messages
    assistant = next(m for m in reversed(messages) if m.get("tool_calls"))
    asked = [c["id"] for c in assistant["tool_calls"]]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert sorted(answered) == sorted(asked), (
        f"assistant asked for {asked}, only {answered} were answered — "
        "the next request would 400"
    )

    # The sibling calls that needed no approval still ran and reported back.
    by_id = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    assert "Not executed" in by_id["c1"]
    assert "Not executed" not in by_id["c2"]
    assert "Not executed" not in by_id["c3"]


async def test_approving_runs_the_command_and_continues(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "mkdir -p made"})]),
            make_reply("that directory exists now"),
        ],
    )
    first = await agent.handle(CHAT, "make it")
    action = first.pending[0]
    second = await agent.resolve(CHAT, action.id, approved=True)

    assert second.text == "that directory exists now"
    assert second.tools_used == ["run_shell"]
    assert (config.shell_cwd / "made").is_dir(), "the approved command must really run"
    assert "Approved by the owner" in last_user_message(agent)


async def test_denying_records_the_refusal_and_continues(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "mkdir -p denied"})]),
            make_reply("fair enough, i left it alone"),
        ],
    )
    first = await agent.handle(CHAT, "make it")
    second = await agent.resolve(CHAT, first.pending[0].id, approved=False)

    assert "left it alone" in second.text
    assert second.tools_used == []  # nothing ran
    assert not (config.shell_cwd / "denied").exists()
    outcome = last_user_message(agent)
    assert "declined" in outcome
    assert "do not retry" in outcome


async def test_denied_command_really_does_not_run(config) -> None:
    target = config.shell_cwd / "keepme.txt"
    target.write_text("precious", encoding="utf-8")

    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": f"rm {target}"})]),
            make_reply("ok"),
        ],
    )
    first = await agent.handle(CHAT, "delete it")
    await agent.resolve(CHAT, first.pending[0].id, approved=False)

    assert target.exists(), "a declined command must not touch the filesystem"
    assert target.read_text(encoding="utf-8") == "precious"


async def test_approving_a_real_dangerous_command_works(config) -> None:
    target = config.shell_cwd / "temp.txt"
    target.write_text("x", encoding="utf-8")

    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": f"rm {target}"})]),
            make_reply("deleted"),
        ],
    )
    first = await agent.handle(CHAT, "delete it")
    await agent.resolve(CHAT, first.pending[0].id, approved=True)

    assert not target.exists()


async def test_resolving_an_unknown_action_errors(config) -> None:
    agent = make_agent(config, [make_reply("ok")])
    await agent.handle(CHAT, "hi")
    result = await agent.resolve(CHAT, "does-not-exist", approved=True)
    assert "already been handled" in result.error


async def test_pending_queue_drains(config) -> None:
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm a"})]),
            make_reply("ok"),
        ],
    )
    first = await agent.handle(CHAT, "delete a")
    action = first.pending[0]
    await agent.resolve(CHAT, action.id, approved=False)
    assert agent.conversation(CHAT).pending == []


# --------------------------------------------------------------------------- #
# direct tool invocation (the /run, /search, /fetch path)
# --------------------------------------------------------------------------- #


async def test_run_tool_executes_without_the_model_choosing_it(config) -> None:
    agent = make_agent(config, [make_reply("the output was: hello")])
    result = await agent.run_tool(CHAT, "run_shell", {"command": "echo hello"})

    assert result.tools_used == ["run_shell"]
    assert "the output was: hello" in result.text


async def test_run_tool_can_also_need_approval(config) -> None:
    agent = make_agent(config, [])
    result = await agent.run_tool(CHAT, "run_shell", {"command": "rm -rf workspace/x"})

    assert result.needs_approval
    assert "confirm" in result.text.lower()
    assert agent.llm.calls == []  # never reached the model


# --------------------------------------------------------------------------- #
# maintenance
# --------------------------------------------------------------------------- #


async def test_reset_drops_in_memory_state(config) -> None:
    """Agent.reset() forgets the live conversation but keeps the durable transcript.

    The bot's /reset does both, which is the behaviour users actually see; see
    test_reset_command_clears_the_transcript_too.
    """
    agent = make_agent(config, [make_reply("one"), make_reply("two")])
    await agent.handle(CHAT, "first")
    agent.reset(CHAT)
    await agent.handle(CHAT, "second")

    # Re-seeded from the transcript, so "first" is still in the prompt.
    contents = [m.get("content") for m in agent.llm.calls[1]]
    assert "second" in contents
    assert "first" in contents


async def test_reset_command_clears_the_transcript_too(config) -> None:
    agent = make_agent(config, [make_reply("one"), make_reply("two")])
    await agent.handle(CHAT, "first")

    # This is what the /reset handler does.
    agent.reset(CHAT)
    History(config.history_dir).clear(CHAT)
    await agent.handle(CHAT, "second")

    assert len(agent.llm.calls[1]) == 2  # system + the new message only
    assert agent.llm.calls[1][1]["content"] == "second"


async def test_reload_rereads_personality(config) -> None:
    agent = make_agent(config, [make_reply("ok")])
    (config.personality_file).write_text("You are a pirate now.\n", encoding="utf-8")
    notes = agent.reload()
    assert any("personality reloaded" in note for note in notes)
    assert "pirate" in agent.personality.text


async def test_conversations_are_per_chat(config) -> None:
    agent = make_agent(config, [make_reply("a"), make_reply("b")])
    await agent.handle(1, "in chat one")
    await agent.handle(2, "in chat two")
    assert agent.conversation(1) is not agent.conversation(2)


async def test_a_long_conversation_is_condensed_not_forgotten(config) -> None:
    """A conversation that outgrows its window keeps its past as a record.

    The old failure mode was a message count: past 60 messages the oldest
    prefix was deleted, and the model was later asked about something it had
    never seen. Here the window is tiny, and what survives is the record plus
    the recent turns verbatim.
    """
    config.llm.context_window = 4_000
    config.llm.context_headroom = 400
    config.llm.context_keep_recent = 4
    config.llm.max_tool_iterations = 1

    agent = make_agent(config, [], llm=CondensingLLM("answer"))

    for i in range(8):
        await agent.handle(CHAT, f"question {i} " + "detail " * 300)

    conv = agent.conversation(CHAT)
    assert agent.llm.condensed > 0, "the window filled, so something had to be condensed"
    assert conv.summary, "the past should survive as a record, not vanish"
    assert conv.summarised > 0
    assert agent.context.estimate(conv.messages) <= agent.context.budget
    # The newest turn is still there verbatim — that is the one being answered.
    assert any("question 7 " in str(m.get("content", "")) for m in conv.messages)
    # And the record survived on disk, so a restart inherits it too.
    assert any(t.role == "summary" for t in History(config.history_dir).tail(CHAT))


async def test_the_window_is_not_managed_by_message_count(config) -> None:
    """Fifty short messages must not be trimmed: there was room for all of them."""
    config.llm.max_tool_iterations = 1
    agent = make_agent(config, [make_reply(f"reply {i}") for i in range(25)])
    for i in range(25):
        await agent.handle(CHAT, f"message {i}")

    conv = agent.conversation(CHAT)
    assert len(conv.messages) >= 25
    assert conv.dropped == 0
    assert conv.summary == ""


async def test_a_restart_replays_the_record_and_the_turns_after_it(config) -> None:
    """A condensed record is written to the transcript, so a fresh process
    resumes the thread instead of starting the conversation over."""
    config.llm.context_window = 4_000
    config.llm.context_headroom = 400
    config.llm.context_keep_recent = 4
    config.llm.max_tool_iterations = 1
    agent = make_agent(config, [], llm=CondensingLLM("answer"))
    for i in range(8):
        await agent.handle(CHAT, f"question {i} " + "detail " * 300)
    record = agent.conversation(CHAT).summary
    assert record

    # A new agent over the same transcript: what /reset does, without the delete.
    restarted = make_agent(config, [make_reply("welcome back")])
    await restarted.handle(CHAT, "what were we doing?")
    sent = restarted.llm.calls[0]
    assert any("condensed record" in str(m.get("content", "")) for m in sent)
    assert any("what were we doing?" in str(m.get("content", "")) for m in sent)


async def test_a_tool_pair_is_never_split(config) -> None:
    """Compaction moves an assistant's tool calls with their results, or the
    next request is a 400 rather than a shorter conversation."""
    from lumi.context import blocks

    config.llm.context_window = 2_000
    config.llm.context_headroom = 200
    config.llm.context_keep_recent = 2
    config.llm.max_tool_iterations = 1
    agent = make_agent(config, [], llm=CondensingLLM("answer"))
    for i in range(8):
        await agent.handle(CHAT, f"question {i} " + "detail " * 200)

    conv = agent.conversation(CHAT)
    for group in blocks(conv.messages):
        if group[0].get("role") == "assistant" and group[0].get("tool_calls"):
            assert [m["role"] for m in group[1:]] == ["tool"]


async def test_memory_of_chats_in_is_bounded(config) -> None:
    """Each conversation is capped by the window, so the number of chats is what
    decides how much the process holds — and an evicted one replays from disk."""
    config.llm.max_conversations = 3
    config.llm.max_tool_iterations = 1
    agent = make_agent(config, [make_reply(f"reply {i}") for i in range(20)])
    for chat_id in range(1, 6):
        await agent.handle(chat_id, f"hello from {chat_id}")
        await agent.handle(chat_id, f"and again from {chat_id}")

    assert agent.conversations_in_memory() <= 4  # the live one plus the cap
    # The most recent chat is never the one evicted.
    assert agent.conversation_state(5) is not None


def test_disabled_tools_are_not_registered(config) -> None:
    config.tools.enabled = ["memory"]
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    registry = build_registry(config, memory)
    assert registry.names() == ["memory"]


def test_specs_are_well_formed(config) -> None:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    for spec in build_registry(config, memory).specs():
        function = spec["function"]
        assert spec["type"] == "function"
        assert function["name"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"


# --------------------------------------------------------------------------- #
# stale approvals
# --------------------------------------------------------------------------- #


async def test_an_unanswered_approval_expires_and_a_late_tap_is_refused(config) -> None:
    """A day-old button must not run the day-old command: the action expires,
    is cancelled, and the model is told rather than left waiting."""
    agent = make_agent(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
    )
    result = await agent.handle(CHAT, "delete the tmp folder")
    assert result.needs_approval

    conv = agent.conversation(CHAT)
    conv.pending[0].created -= APPROVAL_TTL_SECONDS + 1

    late = await agent.resolve(CHAT, "c1", True)

    assert "expired" in late.error.lower()
    assert conv.pending == []
    # The model hears that the wait ended — a user note naming the expiry.
    assert conv.messages[-1]["role"] == "user"
    assert "expired" in conv.messages[-1]["content"]
    # And the command did not run: no approval outcome was ever recorded.
    assert not any("Approved by the owner" in str(m.get("content", "")) for m in conv.messages)


async def test_a_dangling_approval_stops_pinning_the_conversation(config) -> None:
    """A fresh approval protects its chat from eviction (the owner must be able
    to answer it); once expired the chat is evictable like any other, so
    ``max_conversations`` cannot be exceeded without limit."""
    config.llm.max_conversations = 1
    agent = make_agent(
        config,
        [make_reply("", [("c1", "run_shell", {"command": "rm -rf workspace/tmp"})])],
    )
    await agent.handle(CHAT, "delete the tmp folder")
    assert agent.conversation_state(CHAT).pending  # sanity: it is waiting

    agent.conversation(999)
    # The pending approval still keeps its chat in memory…
    assert agent.conversation_state(CHAT) is not None

    agent.conversation(CHAT).pending[0].created -= APPROVAL_TTL_SECONDS + 1
    agent.conversation(1000)

    # …but once it has expired, the stale chat is evicted and the cap holds.
    assert agent.conversation_state(CHAT) is None
    assert agent.conversations_in_memory() == 1
