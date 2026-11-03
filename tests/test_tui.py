"""The stdio serve loop: protocol shape, event order, and survival.

Driven in-process against the scripted ``FakeLLM`` — no Rust and no network —
because the TUI's whole contract with the backend is these JSON lines. The
schema under test is the block at the top of :mod:`lumi.tui`.
"""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any

import pytest
from conftest import FakeLLM, make_reply, make_store

from lumi import tui
from lumi.agent import Agent
from lumi.memory import History
from lumi.personality import Personality
from lumi.tools import build_registry


def make_agent(config, replies: list) -> Agent:
    """An agent wired the way ``_build_agent`` wires one, with a scripted LLM."""
    memory = make_store(config)
    return Agent(
        config=config,
        personality=Personality.load(config.personality_file),
        memory=memory,
        history=History(config.history_dir),
        registry=build_registry(config, memory),
        llm=FakeLLM(replies),
    )


class Frontend:
    """An in-process frontend for the serve loop.

    Commands go in with :meth:`push`, events land in :attr:`events` the moment
    the loop writes them — so a reply that depends on an earlier event (an
    approval id) can be sent as soon as it is known, which is what a real
    frontend does and what a pre-scripted stdin cannot express.
    """

    def __init__(self, config: Any, agent: Agent) -> None:
        self.config = config
        self.agent = agent
        self.events: list[dict[str, Any]] = []
        self._inbox: queue.Queue[str] = queue.Queue()
        self._arrived = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        owner = self

        class _Stdin:
            def readline(self) -> str:
                return owner._inbox.get()

        class _Stdout:
            def write(self, data: str) -> int:
                for line in data.splitlines():
                    if line.strip():
                        owner.events.append(json.loads(line))
                owner._arrived.set()
                return len(data)

            def flush(self) -> None:
                pass

        self._stdin: Any = _Stdin()
        self._stdout: Any = _Stdout()

    async def __aenter__(self) -> Frontend:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Always unblock the reader and join the loop, even when the assertion
        # below failed: a thread parked on an empty inbox would hang the run.
        try:
            await self.finish()
        except Exception:  # noqa: BLE001 - never mask the test's own failure
            if exc_type is None:
                raise

    async def start(self) -> dict[str, Any]:
        """Start the loop and wait for the ``ready`` event."""
        self._task = asyncio.create_task(
            tui.serve(self.agent, self.config, stdin=self._stdin, stdout=self._stdout)
        )
        self._task.add_done_callback(lambda _: self._arrived.set())
        return (await self.wait_for("ready"))[0]

    def push(self, command: dict[str, Any]) -> None:
        self._inbox.put(json.dumps(command))

    def push_raw(self, line: str) -> None:
        """Send a line that is not a command object at all."""
        self._inbox.put(line)

    def close(self) -> None:
        """End stdin, the way a pipe closing or the frontend quitting does."""
        self._inbox.put("")

    async def wait_for(self, event: str, count: int = 1) -> list[dict[str, Any]]:
        """Block until *count* events named *event* have arrived."""
        while True:
            matched = [e for e in self.events if e["ev"] == event]
            if len(matched) >= count:
                return matched
            if self._task is not None and self._task.done():
                raise AssertionError(
                    f"serve ended waiting for {event!r}; got {[e['ev'] for e in self.events]}"
                )
            # No await between the check above and this: the writer only runs
            # on this loop, so it cannot slip an event in between.
            self._arrived.clear()
            await self._arrived.wait()

    async def finish(self) -> list[dict[str, Any]]:
        """Close stdin, let the loop drain, and hand back every event."""
        self.close()
        if self._task is None:
            return self.events
        await self._task
        return self.events


# --------------------------------------------------------------------------- #
# ready
# --------------------------------------------------------------------------- #


async def test_ready_announces_the_backend_before_any_command(config) -> None:
    """The header comes from Python, so the frontend never parses config."""
    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        ready = await fe.start()

        assert ready["model"] == config.llm.model_of()
        assert ready["host"] == config.llm.base_url_of()
        assert "run_shell" in ready["tools"]
        assert isinstance(ready["search"], str)
        assert ready["turns"] == 0
        assert [e["ev"] for e in fe.events] == ["ready"]


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #


async def test_submit_emits_the_turn_sequence(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("hello there")])) as fe:
        await fe.start()

        fe.push({"cmd": "submit", "text": "hi"})
        await fe.wait_for("turn_done")

        assert [e["ev"] for e in fe.events] == ["ready", "step", "result", "turn_done"]
        assert fe.events[1] == {"ev": "step", "kind": "thinking", "detail": ""}
        result = fe.events[2]
        assert result["text"] == "hello there"
        assert result["error"] == ""
        assert result["tools_used"] == []
        assert result["artifacts"] == []
        assert isinstance(result["elapsed"], float)


async def test_a_tui_turn_is_asked_for_markdown_not_telegram_html(config) -> None:
    """The protocol's source is ``tui``, so the prompt must not be Telegram's.

    Without this the TUI would be told to emit ``<b>`` tags its renderer shows
    as literal angle brackets — the bug §6.1 fixes, seen from the other side.
    """
    agent = make_agent(config, [make_reply("ok")])
    async with Frontend(config, agent) as fe:
        await fe.start()

        fe.push({"cmd": "submit", "text": "hi"})
        await fe.wait_for("turn_done")

        system = agent.llm.calls[0][0]["content"]
        assert "renders Markdown" in system
        assert "parses HTML" not in system


# --------------------------------------------------------------------------- #
# approve
# --------------------------------------------------------------------------- #


async def test_approve_resolves_the_pending_action_and_continues(config) -> None:
    target = config.shell_cwd / "made.txt"
    target.write_text("scratch", encoding="utf-8")

    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "rm made.txt"})]),
            make_reply("it is gone"),
        ],
    )
    async with Frontend(config, agent) as fe:
        await fe.start()

        fe.push({"cmd": "submit", "text": "make it"})
        approvals = await fe.wait_for("approval")
        await fe.wait_for("turn_done")

        assert len(approvals) == 1
        assert approvals[0]["tool"] == "run_shell"
        assert "rm made.txt" in approvals[0]["preview"]
        assert approvals[0]["reason"]

        # The id is only knowable once the backend has sent it — which is the
        # whole reason a pre-scripted stdin cannot express this exchange.
        fe.push({"cmd": "approve", "id": approvals[0]["id"], "ok": True})
        await fe.wait_for("turn_done", count=2)

        results = [e for e in fe.events if e["ev"] == "result"]
        assert len(results) == 2
        assert results[1]["text"] == "it is gone"
        assert results[1]["tools_used"] == ["run_shell"]
        assert not target.exists(), "the approved command must really run"
        # Approvals come before their result, and turn_done is always last.
        # Each exchange announces its steps too: the first stops on the tool
        # that needs asking, the second runs it and thinks again.
        assert [e["ev"] for e in fe.events] == [
            "ready",
            "step",
            "step",
            "approval",
            "result",
            "turn_done",
            "step",
            "step",
            "result",
            "turn_done",
        ]
        assert fe.events[6]["kind"] == "tool"
        assert fe.events[7]["kind"] == "thinking"


# --------------------------------------------------------------------------- #
# reset, quit
# --------------------------------------------------------------------------- #


async def test_reset_clears_the_conversation(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()

        fe.push({"cmd": "reset"})
        cleared = await fe.wait_for("cleared")

        assert len(cleared) == 1


async def test_quit_closes_the_registry(config) -> None:
    agent = make_agent(config, [make_reply("ok")])
    closed: list[str] = []
    original = agent.registry.aclose

    async def spy() -> None:
        closed.append("closed")
        await original()

    agent.registry.aclose = spy

    async with Frontend(config, agent) as fe:
        await fe.start()
        fe.push({"cmd": "quit"})
        events = await fe.finish()

    assert closed == ["closed"]
    assert [e["ev"] for e in events] == ["ready"]


# --------------------------------------------------------------------------- #
# malformed input and unknown commands
# --------------------------------------------------------------------------- #


async def test_a_malformed_line_is_reported_and_the_loop_survives(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("still here")])) as fe:
        await fe.start()

        fe.push_raw("this is not json\n")
        errors = await fe.wait_for("error")
        assert "malformed" in errors[0]["text"]

        fe.push_raw('[1, 2, 3]\n')
        await fe.wait_for("error", count=2)

        fe.push({"hello": "world"})
        await fe.wait_for("error", count=3)
        assert 'missing "cmd"' in fe.events[-1]["text"]

        # Still alive: the next well-formed command runs normally.
        fe.push({"cmd": "submit", "text": "hi"})
        await fe.wait_for("turn_done")
        assert [e for e in fe.events if e["ev"] == "result"][0]["text"] == "still here"


async def test_an_unknown_command_is_an_error_not_a_shutdown(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()

        fe.push({"cmd": "teleport"})
        errors = await fe.wait_for("error")

        assert errors[0]["text"] == "unknown command: teleport"
        fe.push({"cmd": "reset"})
        await fe.wait_for("cleared")


async def test_cancel_stops_the_turn_and_seals_dangling_tool_calls(config) -> None:
    """§6.2: a turn killed mid-tool leaves tool_calls without their replies,
    and the provider 400s on that pair until /reset. Cancel must repair it."""
    agent = make_agent(
        config,
        [
            make_reply("", [("c1", "run_shell", {"command": "echo hi"})]),
            make_reply("after cancel"),
        ],
    )

    # Park the tool call so the turn is genuinely in flight when cancel lands.
    gate = asyncio.Event()
    original = agent.registry.invoke

    async def blocking_invoke(name: str, arguments: dict, ctx: Any) -> Any:
        if name == "run_shell":
            await gate.wait()
        return await original(name, arguments, ctx)

    agent.registry.invoke = blocking_invoke

    async with Frontend(config, agent) as fe:
        await fe.start()

        fe.push({"cmd": "submit", "text": "run it"})
        steps = await fe.wait_for("step", count=2)
        assert steps[1] == {"ev": "step", "kind": "tool", "detail": "run_shell"}

        fe.push({"cmd": "cancel"})
        await fe.wait_for("turn_done")

        notices = [e for e in fe.events if e["ev"] == "notice"]
        assert notices[-1]["text"] == "turn cancelled"

        conv = agent.conversation_state(tui.CHAT_ID)
        assert conv is not None
        tool_replies = [m for m in conv.messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_replies] == ["c1"]
        assert tool_replies[0]["content"] == "cancelled by user"

        # The repaired conversation is replayable: the next turn goes through.
        fe.push({"cmd": "submit", "text": "again"})
        await fe.wait_for("result")
        sent = agent.llm.calls[-1]
        assert [m.get("tool_call_id") for m in sent if m.get("role") == "tool"] == ["c1"]
        results = [e for e in fe.events if e["ev"] == "result"]
        assert results[-1]["text"] == "after cancel"


async def test_quit_cancels_the_running_turn_instead_of_waiting(config) -> None:
    """Leaving mid-turn must not hang the TUI until the turn finishes."""
    agent = make_agent(
        config, [make_reply("", [("c1", "run_shell", {"command": "echo hi"})])]
    )
    gate = asyncio.Event()
    original = agent.registry.invoke

    async def blocking_invoke(name: str, arguments: dict, ctx: Any) -> Any:
        if name == "run_shell":
            await gate.wait()
        return await original(name, arguments, ctx)

    agent.registry.invoke = blocking_invoke

    async with Frontend(config, agent) as fe:
        await fe.start()
        fe.push({"cmd": "submit", "text": "run it"})
        await fe.wait_for("step", count=2)

        fe.push({"cmd": "quit"})
        await asyncio.wait_for(fe.finish(), timeout=5)

    conv = agent.conversation_state(tui.CHAT_ID)
    assert conv is not None
    assert [m["tool_call_id"] for m in conv.messages if m.get("role") == "tool"] == ["c1"]


async def test_cancel_when_idle_is_a_notice_not_an_error(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()

        fe.push({"cmd": "cancel"})
        notice = await fe.wait_for("notice")

        assert notice[0]["text"] == "nothing to cancel"
        assert not [e for e in fe.events if e["ev"] == "error"]


# --------------------------------------------------------------------------- #
# read-only commands
# --------------------------------------------------------------------------- #


async def test_read_only_commands_answer_with_a_notice(config) -> None:
    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()

        fe.push({"cmd": "tools"})
        tools = await fe.wait_for("notice")
        assert "run_shell" in tools[0]["text"]

        fe.push({"cmd": "memory"})
        memory = await fe.wait_for("notice", count=2)
        assert isinstance(memory[1]["text"], str)

        fe.push({"cmd": "context"})
        context = await fe.wait_for("notice", count=3)
        assert "window" in context[2]["text"]

        fe.push({"cmd": "reasoning", "on": False})
        reasoning = await fe.wait_for("notice", count=4)
        assert reasoning[3]["text"] == "live steps off"
        assert config.llm.show_reasoning is False

        fe.push({"cmd": "reasoning"})
        await fe.wait_for("error")
        assert "boolean" in fe.events[-1]["text"]


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


async def test_search_passes_the_query_and_limit_straight_through(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lumi.tools.base import ToolResult

    seen: list[dict[str, Any]] = []

    class StubWeb:
        providers: dict[str, Any] = {}

        def __init__(self, config: Any) -> None:
            self.config = config

        def available(self) -> tuple[bool, str]:
            return True, ""

        async def invoke(self, arguments: dict[str, Any], ctx: Any) -> ToolResult:
            seen.append(dict(arguments))
            return ToolResult(text=f"found {arguments['query']}")

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(tui, "WebTool", StubWeb)

    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()
        assert fe.events[0]["search"] == "on"

        fe.push({"cmd": "search", "query": "lumi tui", "limit": 3})
        notice = await fe.wait_for("notice")

        assert notice[0]["text"] == "found lumi tui"
        assert seen == [{"action": "search", "query": "lumi tui", "max_results": 3}]


async def test_search_reports_when_no_provider_is_usable(
    config, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubWeb:
        providers: dict[str, Any] = {}

        def __init__(self, config: Any) -> None:
            self.config = config

        def available(self) -> tuple[bool, str]:
            return False, "no provider is usable"

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(tui, "WebTool", StubWeb)

    async with Frontend(config, make_agent(config, [make_reply("ok")])) as fe:
        await fe.start()
        assert fe.events[0]["search"] == "off"

        fe.push({"cmd": "search", "query": "anything"})
        errors = await fe.wait_for("error")

        assert errors[0]["text"] == "web search unavailable: no provider is usable"

        fe.push({"cmd": "search"})
        await fe.wait_for("error", count=2)
        assert 'needs a "query"' in fe.events[-1]["text"]
