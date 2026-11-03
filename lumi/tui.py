"""Serve the agent to a frontend over stdin/stdout, one JSON object per line.

This is what makes the Rust UI a frontend instead of a second agent: Python
keeps the loop, tools, safety, memory, config and LLM client, and Rust keeps
the terminal. ``lumi chat`` and the Telegram bot are the other two consumers of
the same :class:`~lumi.agent.Agent`; this module adds no behaviour of its own,
it translates the REPL's commands into a byte stream.

**Precondition** (handoff-tui.md §3): logging's console handler writes to
stderr (:mod:`lumi.util.log`), so stdout carries the protocol and nothing else.
There is no ``print()`` on this path — one stray line corrupts the stream.

Protocol
--------

Two message kinds, one JSON object per line: ``{"cmd": …}`` in, ``{"ev": …}``
out. Both sides test against this block.

**Frontend → backend**

| cmd | maps to |
|---|---|
| ``submit {text}`` | ``agent.handle("cli", text, source="tui", hooks=…)`` |
| ``approve {id, ok}`` | ``agent.resolve("cli", id, ok, source="tui", hooks=…)`` |
| ``cancel`` | cancel the in-flight turn, seal any dangling tool calls → ``notice`` + ``turn_done`` |
| ``reset`` | ``agent.reset("cli")`` → ``cleared`` |
| ``context`` | ``agent.context_report("cli")`` → ``notice`` (one per line) |
| ``tools`` | ``agent.registry.describe()`` → ``notice`` |
| ``memory`` | ``agent.memory_for("cli").for_prompt(limit)`` → ``notice`` |
| ``reasoning {on}`` | ``config.llm.show_reasoning = on`` → ``notice`` (the REPL's ``/reasoning``) |
| ``search {query, limit}`` | ``WebTool`` invoked directly, as ``lumi search`` does → ``notice`` |
| ``quit`` | close the loop, ``await agent.registry.aclose()`` |

**Backend → frontend**

| ev | source |
|---|---|
| ``ready {model, host, tools[], search, turns}`` | once, before the first command |
| ``step {kind, detail}`` | ``TurnHooks.on_step`` — ``("thinking", "")`` and ``("tool", detail)`` |
| ``message {text}`` | ``TurnHooks.on_message`` — the ``send`` tool, mid-turn |
| ``delta {text}`` | ``TurnHooks.on_delta`` — one piece of the answer as the model produces it |
| ``thinking_delta {text}`` | the reasoning trace, streamed while ``reasoning`` is on |
| ``result {text, error, tools_used[], artifacts[], elapsed, streamed}`` | ``TurnResult``; ``streamed: true`` means ``text`` already went out as ``delta`` events and must not be rendered again |
| ``approval {id, tool, preview, reason}`` | ``TurnResult.pending`` — one per action, always before its ``result`` |
| ``notice {text}`` / ``error {text}`` | slash-command replies, exceptions, malformed input |
| ``cleared`` | after ``reset`` |
| ``turn_done`` | last event of every ``submit``/``approve``, including one that stopped on an approval |

Rules the frontend can rely on:

- The chat id is always ``"cli"`` — the same scope ``lumi chat`` uses, so the
  TUI, the REPL and ``lumi memory --scope cli`` share one memory file and one
  transcript.
- A turn runs as its own task while the reader keeps reading, so a command can
  arrive mid-turn. Commands that touch the conversation wait for the in-flight
  turn first; read-only ones do not. That separation is what lets ``cancel``
  land mid-turn.
- Several pending actions come back as several ``approval`` events with
  explicit ids. Answer by id; there is no ``pending[0]`` fallback here (§4).
- The answer streams: ``delta`` events carry it piece by piece while the turn
  runs, and ``result.streamed`` then says the same text is in ``result`` —
  render one or the other, never both. The backend commits the stream at the
  first piece, so its retry and key-rotation ladder can never replay text the
  frontend already showed. ``thinking_delta`` follows the same switch as the
  live steps: the ``reasoning`` command (§7b).
- ``error`` ends nothing. The loop survives a malformed line and an exception
  alike: a turn that raised emits ``error`` then ``turn_done``, a rejected
  command emits ``error`` alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from .config import Config
from .llm.base import DELTA_THINKING
from .tools.base import ToolContext
from .tools.web import WebTool
from .util.log import get_logger
from .util.text import format_error

log = get_logger(__name__)

#: The chat every non-Telegram frontend drives. Same scope as ``lumi chat``, so
#: a memory saved from the TUI is the one ``lumi memory --scope cli`` shows.
CHAT_ID = "cli"

#: The ``source`` every turn from here carries. It is what makes
#: :meth:`~lumi.agent.Agent.system_prompt` hand out the Markdown format block
#: instead of Telegram's HTML — the TUI's renderer expects Markdown.
SOURCE = "tui"


def _sender(stream: Any) -> Callable[[dict[str, Any]], None]:
    """One event per line, flushed as it is written.

    Synchronous on purpose: the turn task and the reader both emit, and a write
    that cannot be awaited is a write no other task can split in half.
    """

    def send(event: dict[str, Any]) -> None:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()

    return send


def _hooks(config: Config, send: Callable[[dict[str, Any]], None]) -> Any:
    """TurnHooks writing events instead of printing — the shape
    ``_hooks_for`` builds for the REPL, minus the terminal."""
    from .agent import TurnHooks

    async def on_message(text: str) -> None:
        send({"ev": "message", "text": text})

    async def on_step(kind: str, detail: str) -> None:
        send({"ev": "step", "kind": kind, "detail": detail})

    async def on_delta(kind: str, text: str) -> None:
        # The trace rides the same switch as the live steps — checked at
        # emit time, so a `/reasoning` that lands mid-turn takes effect on
        # the piece about to be sent. The answer itself always streams.
        if kind == DELTA_THINKING and not config.llm.show_reasoning:
            return
        send({"ev": kind, "text": text})

    return TurnHooks(
        on_message=on_message,
        on_step=on_step if config.llm.show_reasoning else None,
        on_delta=on_delta,
    )


def _emit_turn(send: Callable[[dict[str, Any]], None], result: Any) -> None:
    """One ``TurnResult`` as its events: approvals first, ``turn_done`` last.

    A turn that stops on an approval still ends with ``turn_done`` — the
    exchange is over, and the modal is driven by the ``approval`` events that
    preceded it, not by the absence of a terminator.
    """
    for action in result.pending:
        send(
            {
                "ev": "approval",
                "id": action.id,
                "tool": action.tool,
                "preview": action.preview,
                "reason": action.reason,
            }
        )
    send(
        {
            "ev": "result",
            "text": result.text,
            "error": result.error,
            "tools_used": list(result.tools_used),
            "artifacts": [
                {"path": artifact.path, "size": artifact.size, "mime": artifact.mime}
                for artifact in result.artifacts
            ],
            "elapsed": round(float(result.elapsed), 3),
            "streamed": bool(result.streamed),
        }
    )
    send({"ev": "turn_done"})


def _search_status(web: Any) -> str:
    """A short status-bar string: the usable providers, or ``off``."""
    ok, _reason = web.available()
    if not ok:
        return "off"
    usable = [name for name, provider in web.providers.items() if provider.available()[0]]
    return ", ".join(usable) or "on"


def _ready(agent: Any, config: Config, web: Any) -> dict[str, Any]:
    """The header the frontend renders instead of parsing config itself."""
    return {
        "ev": "ready",
        "model": config.llm.model_of(),
        "host": config.llm.base_url_of(),
        "tools": agent.registry.names(),
        "search": _search_status(web),
        "turns": 0,
    }


async def _search(
    message: dict[str, Any],
    config: Config,
    web: Any,
    send: Callable[[dict[str, Any]], None],
) -> bool:
    """Run one search the way ``lumi search`` does — directly, not through the
    model — and answer with a ``notice`` (or an ``error`` when it cannot run)."""
    query = message.get("query")
    if not isinstance(query, str) or not query.strip():
        send({"ev": "error", "text": 'search needs a "query"'})
        return True
    ok, reason = web.available()
    if not ok:
        send({"ev": "error", "text": f"web search unavailable: {reason}"})
        return True
    arguments: dict[str, Any] = {"action": "search", "query": query.strip()}
    limit = message.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        arguments["max_results"] = limit
    result = await web.invoke(arguments, ToolContext(cwd=config.shell_cwd, source=SOURCE))
    send({"ev": "notice" if result.ok else "error", "text": result.text})
    return True


async def serve(
    agent: Any,
    config: Config,
    *,
    stdin: Any | None = None,
    stdout: Any | None = None,
) -> None:
    """Run the protocol until stdin closes or a ``quit`` arrives.

    *stdin*/*stdout* default to the process streams; both exist so a test can
    drive the loop in-process. The search transport and the tool registry are
    released on the way out whatever happens — a frontend that hangs up
    mid-turn must not leave connections for the GC to complain about.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    send = _sender(stdout)
    web = WebTool(config)
    turn: asyncio.Task[None] | None = None

    async def settle() -> None:
        """Wait for the in-flight turn before anything touches the conversation."""
        nonlocal turn
        if turn is not None:
            await turn
            turn = None

    async def drive(run: Callable[[], Awaitable[Any]]) -> None:
        """Run one turn and emit its events. Never raises into the reader."""
        try:
            result = await run()
        except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the stream
            log.exception("turn failed")
            send({"ev": "error", "text": format_error(exc)})
            send({"ev": "turn_done"})
            return
        _emit_turn(send, result)

    async def dispatch(line: str) -> bool:
        """Handle one command line. ``False`` stops the loop."""
        nonlocal turn

        try:
            message = json.loads(line)
        except ValueError as exc:
            send({"ev": "error", "text": f"malformed line: {format_error(exc)}"})
            return True
        if not isinstance(message, dict):
            send({"ev": "error", "text": "malformed line: expected a JSON object"})
            return True
        cmd = message.get("cmd")
        if not isinstance(cmd, str) or not cmd:
            send({"ev": "error", "text": 'malformed line: missing "cmd"'})
            return True

        try:
            if cmd == "submit":
                text = message.get("text")
                if not isinstance(text, str) or not text.strip():
                    send({"ev": "error", "text": 'submit needs a non-empty "text"'})
                    return True
                await settle()
                turn = asyncio.create_task(
                    drive(
                        lambda: agent.handle(
                            CHAT_ID, text, source=SOURCE, hooks=_hooks(config, send)
                        )
                    )
                )
                return True

            if cmd == "approve":
                action_id = message.get("id")
                if not isinstance(action_id, str) or not action_id:
                    send({"ev": "error", "text": 'approve needs an "id"'})
                    return True
                approved = bool(message.get("ok"))
                await settle()
                turn = asyncio.create_task(
                    drive(
                        lambda: agent.resolve(
                            CHAT_ID,
                            action_id,
                            approved,
                            source=SOURCE,
                            hooks=_hooks(config, send),
                        )
                    )
                )
                return True

            if cmd == "cancel":
                if turn is None or turn.done():
                    send({"ev": "notice", "text": "nothing to cancel"})
                    return True
                turn.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await turn
                turn = None
                # The turn may have died between the assistant message that
                # carries `tool_calls` and their `role: "tool"` replies; the
                # provider 400s on that pair forever, so close it here (§6.2).
                agent.seal_dangling_tool_calls(CHAT_ID)
                send({"ev": "notice", "text": "turn cancelled"})
                send({"ev": "turn_done"})
                return True

            if cmd == "reset":
                await settle()
                agent.reset(CHAT_ID)
                send({"ev": "cleared"})
                return True

            if cmd == "context":
                send({"ev": "notice", "text": "\n".join(agent.context_report(CHAT_ID).lines())})
                return True

            if cmd == "tools":
                send({"ev": "notice", "text": agent.registry.describe()})
                return True

            if cmd == "memory":
                send(
                    {
                        "ev": "notice",
                        "text": agent.memory_for(CHAT_ID).for_prompt(agent.memory_limit()),
                    }
                )
                return True

            if cmd == "reasoning":
                on = message.get("on")
                if not isinstance(on, bool):
                    send({"ev": "error", "text": 'reasoning needs a boolean "on"'})
                    return True
                config.llm.show_reasoning = on
                send({"ev": "notice", "text": f"live steps {'on' if on else 'off'}"})
                return True

            if cmd == "search":
                return await _search(message, config, web, send)

            if cmd == "quit":
                # Leaving is not a reason to wait for a long turn: cancel it
                # the way `esc` does. EOF still settles instead (below), so a
                # piped `printf … | lumi tui` is not cut off mid-answer.
                if turn is not None and not turn.done():
                    turn.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await turn
                    turn = None
                    agent.seal_dangling_tool_calls(CHAT_ID)
                return False

            send({"ev": "error", "text": f"unknown command: {cmd}"})
            return True
        except Exception as exc:  # noqa: BLE001 - one bad command must not kill the stream
            log.exception("command %s failed", cmd)
            send({"ev": "error", "text": format_error(exc)})
            return True

    try:
        send(_ready(agent, config, web))
        while True:
            # A thread, so the loop stays free while a turn runs — that is what
            # lets a later `cancel` land mid-turn (§5.1).
            line = await asyncio.to_thread(stdin.readline)
            if not line:
                break  # EOF: the frontend hung up
            if not await dispatch(line):
                break
    finally:
        # A piped `printf '{"cmd":"submit",…}' | lumi tui` closes stdin while the
        # turn is still running: finish it, so the answer is not truncated.
        await settle()
        await web.aclose()
        await agent.registry.aclose()
