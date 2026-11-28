"""The Code Mode sandbox's boundary, exercised through the real child process.

These tests drive :func:`~lumi.tools.mcp.sandbox.run_code` directly: each one
spawns an interpreter, feeds it a script and reads the RPC frames back. That is
slower than stubbing the child, but the point here is exactly what the child
can and cannot reach, so a stub would prove nothing.
"""

from __future__ import annotations

import json
import time

from conftest import skip_without_sandbox

import lumi.tools.mcp.sandbox as sandbox
from lumi.tools.mcp.sandbox import run_code

pytestmark = skip_without_sandbox()


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
        "b = codemode._rpc.__globals__['__builtins__']\n"
        "print('absent', sorted(n for n in ('open', 'eval', 'exec', 'compile') if n not in b))\n"
    )
    tools = [{"helper": "echo", "server": "demo", "tool": "echo", "description": ""}]

    async def echo(_server, _tool, arguments):
        return arguments.get("text")

    run = await run_code(code=code, tools=tools, on_call=echo, timeout=10, max_calls=1)

    assert run.ok, run.error
    lines = run.output.splitlines()
    for line in lines[:3]:
        assert line.endswith("['ToolError', '__builtins__', '_emit']")
    assert lines[3] == "absent ['compile', 'eval', 'exec', 'open']"


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


async def test_import_closure_does_not_hand_over_an_importer() -> None:
    """Unpacking the restricted import must not yield something that imports."""
    code = (
        "imp = __builtins__['__import__']\n"
        "try:\n"
        "    real = [c.cell_contents for c in (imp.__closure__ or ()) "
        "if callable(c.cell_contents)][0]\n"
        "    os_mod = real('os', None, None, (), 0)\n"
        "    print('ESCAPED', os_mod)\n"
        "except Exception as exc:\n"
        "    print('blocked', type(exc).__name__)\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip().startswith("blocked")
    assert "ESCAPED" not in run.output


async def test_a_refused_import_is_a_catchable_import_error() -> None:
    """The hook raises ``ImportError``, so a script must be able to say so.

    ``ImportError`` was missing from the allowlist, so the one exception the
    sandbox's own import raises could not be named: `except ImportError` died
    with ``NameError`` and took the script with it.
    """
    code = "try:\n    import socket\nexcept ImportError as exc:\n    print('caught', str(exc))\n"

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip() == "caught module socket is not available in the MCP sandbox"


async def test_import_table_serves_submodules_and_nothing_else() -> None:
    """Allowlisted imports still bind; a dotted name outside the table does not."""
    code = (
        "import collections.abc as cabc\n"
        "from collections.abc import Mapping\n"
        "from json import dumps\n"
        "print(Mapping is cabc.Mapping, dumps({'a': 1}))\n"
        "try:\n"
        "    import socket\n"
        "except Exception as exc:\n"
        "    print('root blocked', 'not available' in str(exc))\n"
        "try:\n"
        "    import json.tool\n"
        "except Exception as exc:\n"
        "    print('submodule blocked', 'not available' in str(exc))\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.splitlines() == [
        'True {"a": 1}',
        "root blocked True",
        "submodule blocked True",
    ]


async def test_eval_exec_compile_and_open_are_absent() -> None:
    code = (
        "print([n for n in ('eval', 'exec', 'compile', 'open', 'globals', 'locals') "
        "if n in __builtins__])"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip() == "[]"


async def test_builtin_functions_do_not_expose_the_real_builtins() -> None:
    """``print.__self__`` is the real builtins module; it must be disarmed.

    Builtin functions cannot be hidden — each carries ``__self__`` — so the
    interpreter's own builtins module is disarmed in place instead.
    """
    code = (
        "b = print.__self__\n"
        "try:\n"
        "    b.open('/etc/passwd')\n"
        "    print('ESCAPED')\n"
        "except RuntimeError:\n"
        "    print('open blocked')\n"
        "try:\n"
        "    b.__import__('os')\n"
        "    print('ESCAPED')\n"
        "except ImportError as exc:\n"
        "    print('import blocked', 'not available' in str(exc))\n"
        "print(not hasattr(b, '__loader__'), not hasattr(b, '__spec__'))\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert "ESCAPED" not in run.output
    assert run.output.splitlines() == [
        "open blocked",
        "import blocked True",
        "True True",
    ]


async def test_exception_frames_carry_no_real_builtins() -> None:
    """A caught ToolError's traceback must not expose host builtins or modules.

    Frames take their builtins from the ``__builtins__`` entry of the globals
    they run with; every script-reachable callable's globals pin the restricted
    mapping, so ``f_builtins`` on a traceback frame is the sandbox's.
    """
    code = (
        "try:\n"
        "    codemode.call('x', 'y')\n"
        "except Exception as exc:\n"
        "    tb = exc.__traceback__\n"
        "    while tb:\n"
        "        f = tb.tb_frame\n"
        "        if 'open' in f.f_builtins or 'sys' in f.f_globals or 'os' in f.f_globals:\n"
        "            print('LEAK')\n"
        "        tb = tb.tb_next\n"
        "print('frames clean')\n"
    )

    async def refuse(*_args):
        raise RuntimeError("refused on purpose")

    run = await run_code(code=code, tools=[], on_call=refuse, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert "LEAK" not in run.output
    assert run.output.strip() == "frames clean"


async def test_module_internals_do_not_reach_the_host() -> None:
    """Allowlisted modules carry no live capability references.

    ``uuid.os`` used to return the real cwd and ``enum.bltns`` was the real
    builtins module; the scrub replaces both with stubs, disarms every
    reachable module's ``__builtins__``, and removes ``__spec__``/``__loader__``.
    """
    code = (
        "import enum, typing, uuid\n"
        "try:\n"
        "    uuid.os.getcwd()\n"
        "    print('ESCAPED')\n"
        "except AttributeError:\n"
        "    print('uuid.os stubbed')\n"
        "try:\n"
        "    enum.bltns.eval('1+1')\n"
        "    print('ESCAPED')\n"
        "except RuntimeError:\n"
        "    print('bltns disarmed')\n"
        "try:\n"
        "    uuid.__builtins__.open('/etc/passwd')\n"
        "    print('ESCAPED')\n"
        "except RuntimeError:\n"
        "    print('module builtins disarmed')\n"
        "print(typing.sys.modules.get('os') is None)\n"
        "print(not hasattr(uuid, '__spec__'), not hasattr(uuid, '__loader__'))\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=15, max_calls=1)

    assert run.ok, run.error
    assert "ESCAPED" not in run.output
    assert run.output.splitlines() == [
        "uuid.os stubbed",
        "bltns disarmed",
        "module builtins disarmed",
        "True",
        "True True",
    ]


async def test_import_machinery_globals_cannot_reach_host_files(tmp_path) -> None:
    """The real sys/os globals may be introspected, but the OS namespace is isolated."""
    host_file = tmp_path / "host-secret.txt"
    host_file.write_text("must stay outside the child", encoding="utf-8")
    code = (
        "cls = [c for c in object.__subclasses__() if c.__name__ == '_wrap_close'][0]\n"
        "real_sys = cls.__init__.__globals__['sys']\n"
        "real_os = real_sys.modules['os']\n"
        "print('cwd', real_os.getcwd())\n"
        "try:\n"
        f"    real_os.open({str(host_file)!r}, real_os.O_RDONLY)\n"
        "    print('HOST FILE READ')\n"
        "except OSError:\n"
        "    print('host file hidden')\n"
        "try:\n"
        f"    real_os.remove({str(host_file)!r})\n"
        "    print('HOST FILE REMOVED')\n"
        "except OSError:\n"
        "    print('host remove blocked')\n"
        "try:\n"
        "    real_os.open('/tmp/escape', real_os.O_CREAT | real_os.O_WRONLY, 0o600)\n"
        "    print('HOST FILE WRITE')\n"
        "except OSError:\n"
        "    print('scratch read-only')\n"
        "try:\n"
        "    real_os.open('/escape', real_os.O_CREAT | real_os.O_WRONLY, 0o600)\n"
        "    print('ROOT WRITE')\n"
        "except OSError:\n"
        "    print('root read-only')\n"
        "r = real_sys.modules['resource']\n"
        "print('limits', r.getrlimit(r.RLIMIT_FSIZE)[1] == 0, "
        "r.getrlimit(r.RLIMIT_NPROC)[1] <= 8, "
        "r.getrlimit(r.RLIMIT_AS)[1] <= 512 * 1024 * 1024)\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.splitlines() == [
        "cwd /tmp",
        "host file hidden",
        "host remove blocked",
        "scratch read-only",
        "root read-only",
        "limits True True True",
    ]
    assert host_file.read_text(encoding="utf-8") == "must stay outside the child"


async def test_allowlisted_modules_still_work_after_the_scrub() -> None:
    """The scrub must not cost the data-processing modules their function."""
    code = (
        "import collections, copy, dataclasses, decimal, enum, fractions, json\n"
        "import pprint, random, re, typing, uuid\n"
        "class Color(enum.Enum):\n"
        "    RED = 1\n"
        "print(Color.RED.name, len(uuid.uuid4().hex))\n"
        "print(json.dumps({'a': [1, 2]}))\n"
        "print(len(uuid.uuid4().hex), len(uuid.uuid3(uuid.NAMESPACE_DNS, 'x').hex))\n"
        "print(random.Random(1).randint(1, 10) > 0)\n"
        "print(str(decimal.Decimal('0.1') + decimal.Decimal('0.2')), "
        "str(fractions.Fraction(1, 3) + fractions.Fraction(1, 6)))\n"
        "print(collections.Counter('aab').most_common(1) == [('a', 2)])\n"
        "print(bool(re.match(r'\\d+', '123')))\n"
        "print('a' in pprint.pformat({'a': [1] * 40}))\n"
        "print(copy.deepcopy({'a': [1, {'b': 2}]})['a'][1]['b'])\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=15, max_calls=1)

    assert run.ok, run.error
    assert run.output.splitlines() == [
        "RED 32",
        '{"a": [1, 2]}',
        "32 32",
        "True",
        "0.3 1/2",
        "True",
        "True",
        "True",
        "2",
    ]


async def test_a_forged_done_frame_cannot_exceed_the_output_limit() -> None:
    """A script that reaches _emit and forges the final frame gets no extra room.

    print() is capped by the child's capture; the parent enforces the same
    ceiling on the done frame it accepts, so the forged channel returns no
    more than print would have.
    """
    code = "codemode._rpc.__globals__['_emit']({'done': True, 'output': 'x' * (400000)})\n"

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert len(run.output) <= sandbox.CHILD_OUTPUT_LIMIT + len("\n...[output truncated]")
    assert run.output.count("x") < 400000


async def test_the_emit_writer_budget_stops_a_flood() -> None:
    """A single frame past the emit budget fails in the child, not the parent."""
    code = "codemode._rpc.__globals__['_emit']({'done': True, 'output': 'x' * 5000000})\n"

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert not run.ok
    assert "output budget" in run.error


async def test_class_statements_and_attribute_tools_work() -> None:
    """Model-written classes, setattr/delattr, dir and vars are usable."""
    code = (
        "class Thing:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n"
        "\n"
        "t = Thing()\n"
        "setattr(t, 'value', 2)\n"
        "print(t.value)\n"
        "delattr(t, 'value')\n"
        "print(hasattr(t, 'value'))\n"
        "print('value' in dir(t))\n"
        "print(sorted(vars(t)))\n"
    )

    run = await run_code(code=code, tools=[], on_call=never, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.split() == ["2", "False", "False", "[]"]


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


async def test_stderr_flood_does_not_block_the_script(monkeypatch) -> None:
    """A child filling the stderr pipe must still reach its result."""
    monkeypatch.setattr(
        sandbox,
        "BOOTSTRAP",
        sandbox.BOOTSTRAP.replace(
            "sys.stdout = _capture",
            'sys.stdout = _capture\nsys.stderr.write("x" * 500000)\nsys.stderr.flush()',
            1,
        ),
    )

    run = await run_code(
        code="print('after the flood')", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert run.ok, run.error
    assert run.output.strip() == "after the flood"


async def test_budget_exhaustion_fails_later_calls_without_a_round_trip() -> None:
    """Only the first over-budget call reaches the parent; the rest fail locally."""
    calls: list[str] = []

    async def echo(_server, _tool, arguments):
        calls.append(str(arguments))
        return arguments

    tools = [{"helper": "echo", "server": "demo", "tool": "echo", "description": ""}]
    code = (
        "ran = 0\n"
        "caught = 0\n"
        "for _ in range(4):\n"
        "    try:\n"
        "        codemode.echo(n=1)\n"
        "        ran += 1\n"
        "    except codemode.ToolError:\n"
        "        caught += 1\n"
        "print('ran', ran, 'caught', caught)\n"
    )

    run = await run_code(code=code, tools=tools, on_call=echo, timeout=10, max_calls=1)

    assert run.ok, run.error
    assert run.output.strip() == "ran 1 caught 3"
    assert len(calls) == 1
    assert run.calls == 2  # one allowed call, one refused by the parent
    assert run.trail == ["demo.echo"]


async def test_oversized_stdout_frame_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "MAX_FRAME_CHARS", 200)
    run = await run_code(
        code="print('x' * 500)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert not run.ok
    assert "too large" in run.error


async def test_a_frame_exactly_at_the_size_cap_is_accepted(monkeypatch) -> None:
    """The cap is on the frame's payload, not on the newline that ends it.

    A payload of exactly ``MAX_FRAME_CHARS`` used to satisfy _FrameReader and
    then be rejected by the caller, which measured the terminator too.
    """
    printed = "x" * 50 + "\n"  # print's newline is part of the output
    frame = json.dumps({"done": True, "output": printed, "error": ""})

    monkeypatch.setattr(sandbox, "MAX_FRAME_CHARS", len(frame))
    at_cap = await run_code(
        code="print('x' * 50)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert at_cap.ok, at_cap.error
    assert at_cap.output == printed

    monkeypatch.setattr(sandbox, "MAX_FRAME_CHARS", len(frame) - 1)
    over_cap = await run_code(
        code="print('x' * 50)", tools=[], on_call=never, timeout=10, max_calls=1
    )

    assert not over_cap.ok
    assert "too large" in over_cap.error


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
