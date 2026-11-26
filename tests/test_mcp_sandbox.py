"""The Code Mode sandbox's boundary, exercised through the real child process.

These tests drive :func:`~lumi.tools.mcp.sandbox.run_code` directly: each one
spawns an interpreter, feeds it a script and reads the RPC frames back. That is
slower than stubbing the child, but the point here is exactly what the child
can and cannot reach, so a stub would prove nothing.
"""

from __future__ import annotations

import time

import lumi.tools.mcp.sandbox as sandbox
from lumi.tools.mcp.sandbox import run_code


async def never(*_args):
    raise AssertionError("no tool call should happen")


# --------------------------------------------------------------------------- #
# the namespace a script gets
# --------------------------------------------------------------------------- #


async def test_io_import_is_blocked(tmp_path) -> None:
    """``io.open`` is the unrestricted builtin ``open``; the module stays out."""
    target = tmp_path / "harmless.txt"
    target.write_text("not a secret", encoding="utf-8")
    code = "import io\nprint(io.open(" + repr(str(target)) + ").read())"

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert not run.ok
    assert "not available in the MCP sandbox" in run.error
    assert "not a secret" not in run.output


async def test_codemode_callables_have_no_host_globals() -> None:
    """Introspecting the API must not hand the script the bootstrap namespace."""
    code = (
        "print('rpc', sorted(codemode._rpc.__globals__))\n"
        "print('call', sorted(codemode.call.__globals__))\n"
        "print('helper', sorted(codemode.echo.__globals__))\n"
    )
    tools = [{"helper": "echo", "server": "demo", "tool": "echo", "description": ""}]

    async def echo(_server, _tool, arguments):
        return arguments.get("text")

    run = await run_code(code=code, tools=tools, on_call=echo, timeout=10, max_calls=1)

    assert run.ok, run.error
    for line in run.output.splitlines():
        assert line.endswith("['ToolError', '_emit']")


async def test_globals_escape_to_builtins_is_blocked() -> None:
    """The critical escape: ``_rpc.__globals__['builtins']`` must not exist."""
    code = (
        "try:\n"
        "    g = codemode._rpc.__globals__\n"
        "    g['builtins'].__import__('subprocess').check_output(['true'])\n"
        "    print('ESCAPED')\n"
        "except Exception as exc:\n"
        "    print('blocked', type(exc).__name__)\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip() == "blocked KeyError"


async def test_eval_exec_compile_and_open_are_absent() -> None:
    code = "print([n for n in ('eval', 'exec', 'compile', 'open') if n in __builtins__])"

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip() == "[]"


async def test_allowed_imports_work_and_others_are_refused() -> None:
    good = await run_code(
        code="import math\nprint(math.sqrt(16))",
        tools=[],
        on_call=never,
        timeout=10,
        max_calls=1,
    )
    assert good.ok, good.error
    assert good.output.strip() == "4.0"

    bad = await run_code(
        code="import socket\nsocket.socket()",
        tools=[],
        on_call=never,
        timeout=10,
        max_calls=1,
    )
    assert not bad.ok
    assert "not available in the MCP sandbox" in bad.error


async def test_helper_rpc_round_trip_still_works() -> None:
    """The rebuilt callables must still speak the parent's protocol."""

    async def echo(_server, _tool, arguments):
        return {"echo": arguments.get("text")}

    tools = [{"helper": "echo", "server": "demo", "tool": "echo", "description": ""}]
    run = await run_code(
        code="print(codemode.echo(text='hi')['echo'].upper())",
        tools=tools,
        on_call=echo,
        timeout=10,
        max_calls=1,
    )

    assert run.ok, run.error
    assert run.output.strip() == "HI"
    assert run.trail == ["demo.echo"]


# --------------------------------------------------------------------------- #
# resource limits and failure reporting
# --------------------------------------------------------------------------- #


async def test_timeout_stops_a_sleeping_script() -> None:
    started = time.monotonic()
    run = await run_code(
        code="import time\ntime.sleep(30)",
        tools=[],
        on_call=never,
        timeout=0.3,
        max_calls=1,
    )

    assert not run.ok
    assert "ran longer than 0.3s" in run.error
    assert time.monotonic() - started < 10


async def test_oversized_stdout_frame_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "MAX_FRAME_CHARS", 200)
    run = await run_code(
        code="print('x' * 500)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert not run.ok
    assert "too large" in run.error


async def test_output_truncation_marker(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "CHILD_OUTPUT_LIMIT", 100)
    run = await run_code(
        code="print('x' * 200)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert run.ok, run.error
    assert run.output.endswith("...[output truncated]")


async def test_bootstrap_stderr_is_reported_on_failure(monkeypatch) -> None:
    """A crash before the done frame is explained, even with a protocol error."""
    monkeypatch.setattr(
        sandbox,
        "BOOTSTRAP",
        "import sys\nsys.stderr.write('bootstrap broke')\nsys.exit(3)\n",
    )
    run = await run_code(
        code="print(1)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert not run.ok
    assert "ended without a result" in run.error
    assert "sandbox exited 3" in run.error
    assert "bootstrap broke" in run.error
