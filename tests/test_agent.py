"""The agent loop: tool dispatch, approval handshake, and the iteration cap.

Driven with a scripted LLM so every branch is reachable without a network call:
plain answers, tool use, a tool that needs approval, denial, and a runaway loop.
"""

from __future__ import annotations

import json

from conftest import FakeLLM, make_reply

from lumi.agent import Agent
from lumi.llm.base import LLMError
from lumi.memory import History, MemoryFile
from lumi.personality import Personality
from lumi.tools import build_registry

CHAT = 1234


def make_agent(config, replies) -> Agent:
    memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
    memory.load()
    return Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(replies),
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

    dialogue = History(config.history_dir).recent_dialogue(CHAT, 10)
    assert {"role": "user", "content": "hello"} in dialogue
    assert {"role": "assistant", "content": "remembered this"} in dialogue


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


async def test_message_trimming_keeps_a_user_boundary(config, monkeypatch) -> None:
    import lumi.agent as agent_module

    monkeypatch.setattr(agent_module, "MAX_MESSAGES", 6)
    agent = make_agent(config, [make_reply(f"reply {i}") for i in range(8)])
    for i in range(8):
        await agent.handle(CHAT, f"message {i}")

    messages = agent.conversation(CHAT).messages
    assert len(messages) <= 8
    assert messages[0]["role"] in {"system", "user"}


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
