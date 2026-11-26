"""The Code Mode sandbox: run one model-written script in a child interpreter.

Cloudflare's Code Mode runs generated TypeScript in a V8 isolate with no
network access, and every MCP call is an RPC back to the supervisor. This is
the same shape in the language this project can run without a new dependency:
the script runs in a fresh Python interpreter (``-I -S``: isolated, no site
packages) with

* no inherited secrets (the child gets :func:`~lumi.util.env.scrubbed_env`),
* a restricted namespace: ``open``, ``eval``, ``exec`` and ``compile`` are
  absent, and ``import`` is limited to an allowlist of pure-data stdlib
  modules,
* one API object, ``codemode``, whose functions emit a JSON request on stdout
  and wait for the parent's reply on stdin,

so the child holds no MCP session, OAuth token or header. The parent validates
every request against the tools discovered for this run and enforces the call
budget. ``print()`` is how the script returns data: stdout is captured and sent
back to the model when the script ends.

This is a conformance boundary, not a jail. It is meant to keep ordinary
model-authored code away from the filesystem, the network and unapproved
servers by accident, and a runaway script costs a killed process rather than
the turn. It does not withstand deliberate Python introspection: the child
runs as the same OS user, and a script that hunts through object internals can
find a way out. Treat ``execute`` as an approved script runner, not as a
security boundary.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from ...util.env import scrubbed_env
from ...util.log import get_logger

log = get_logger(__name__)

#: Largest stdout the child buffers. The parent caps its result too; this
#: bound exists so a print loop cannot grow the child's memory without limit.
CHILD_OUTPUT_LIMIT = 200_000

#: Per-call JSON line ceiling on the parent side. An MCP server can return a
#: whole page; the tool layer truncates the final answer, but the line still
#: has to arrive first.
MAX_FRAME_CHARS = 4_000_000


@dataclass(slots=True)
class CodeRun:
    """What one sandboxed script produced."""

    ok: bool
    output: str = ""
    error: str = ""
    calls: int = 0
    #: One line per call, for the step log and the tool's summary.
    trail: list[str] = field(default_factory=list)


#: The child program. It is passed with ``-c``; keep it self-contained: the
#: child has no access to Lumi's package, only the stdlib modules it imports.
#:
#: Every callable the script can reach is rebuilt with ``types.FunctionType``
#: over ``_API_GLOBALS``, a dict holding only ``ToolError`` and ``_emit``. The
#: host imports, streams and real ``__import__`` live in closure cells or are
#: wiped from the module namespace before user code runs, so
#: ``codemode._rpc.__globals__['builtins']`` and friends do not exist.
BOOTSTRAP = r'''
import builtins
import io
import json
import sys
import traceback
import types

_proto_out = sys.stdout
_proto_in = sys.stdin

config = json.loads(_proto_in.readline())


class ToolError(Exception):
    pass


# The entire global namespace of every script-reachable callable. Keep it
# this small: introspecting __globals__ must not reveal a host capability.
_API_GLOBALS = {"ToolError": ToolError}


def _make_emit(_dumps, _write, _flush):
    def _emit(payload):
        _write(_dumps(payload, ensure_ascii=False, default=repr) + "\n")
        # stdout is a pipe here: without a flush the parent would wait for a
        # reply to a request still sitting in the child's buffer.
        _flush()

    return _emit


_emit_tmpl = _make_emit(json.dumps, _proto_out.write, _proto_out.flush)
_emit = types.FunctionType(
    _emit_tmpl.__code__, _API_GLOBALS, "_emit", None, _emit_tmpl.__closure__
)
_API_GLOBALS["_emit"] = _emit


def _make_rpc(_readline, _loads):
    def _rpc(server, tool, arguments):
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a dict")
        _emit({"call": {"server": server, "tool": tool, "arguments": arguments}})
        line = _readline()
        if not line:
            raise ToolError("the host closed the connection")
        reply = _loads(line)
        if not reply.get("ok"):
            raise ToolError(reply.get("error") or "the call failed")
        return reply.get("value")

    return _rpc


_rpc_tmpl = _make_rpc(_proto_in.readline, json.loads)
_rpc = types.FunctionType(
    _rpc_tmpl.__code__, _API_GLOBALS, "_rpc", None, _rpc_tmpl.__closure__
)


class _CodeMode:
    """The one object handed to the script; no host globals hang off it."""


codemode = _CodeMode()
codemode.ToolError = ToolError
codemode._rpc = _rpc


def _make_call(_rpc):
    def call(server, tool, arguments=None):
        return _rpc(server, tool, {} if arguments is None else arguments)

    return call


_call_tmpl = _make_call(_rpc)
codemode.call = types.FunctionType(
    _call_tmpl.__code__, _API_GLOBALS, "call", None, _call_tmpl.__closure__
)


def _make_helper(_rpc, _server, _tool, _name, _doc):
    def helper(**arguments):
        return _rpc(_server, _tool, arguments)

    helper.__name__ = _name
    helper.__doc__ = _doc
    return helper


for _item in config["tools"]:
    _helper_tmpl = _make_helper(
        _rpc,
        _item["server"],
        _item["tool"],
        _item["helper"],
        _item.get("description") or (_item["server"] + "." + _item["tool"]),
    )
    _helper = types.FunctionType(
        _helper_tmpl.__code__,
        _API_GLOBALS,
        _helper_tmpl.__name__,
        None,
        _helper_tmpl.__closure__,
    )
    _helper.__doc__ = _helper_tmpl.__doc__
    setattr(codemode, _item["helper"], _helper)


_ALLOWED_IMPORTS = {
    "base64",
    "binascii",
    "collections",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "html",
    "itertools",
    "json",
    "math",
    "operator",
    "pprint",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "time",
    "typing",
    "unicodedata",
    "uuid",
}


def _make_import(_real_import, _allowed):
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level != 0 or root not in _allowed:
            raise ImportError("module " + name + " is not available in the MCP sandbox")
        return _real_import(name, globals, locals, fromlist, level)

    return _import


_import_tmpl = _make_import(builtins.__import__, frozenset(_ALLOWED_IMPORTS))
_import = types.FunctionType(
    _import_tmpl.__code__, {}, "_import", None, _import_tmpl.__closure__
)

_ALLOWED_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod",
    "str", "sum", "super", "tuple", "type", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception",
    "IndexError", "KeyError", "LookupError", "NameError",
    "NotImplementedError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
]
_safe = {}
for _name in _ALLOWED_NAMES:
    _safe[_name] = getattr(builtins, _name)
_safe["__import__"] = _import
_safe["ToolError"] = ToolError


class _Capture(io.StringIO):
    def __init__(self, limit):
        super().__init__()
        self._limit = limit
        self._size = 0
        self.truncated = False

    def write(self, text):
        text = str(text)
        room = self._limit - self._size
        if room <= 0:
            self.truncated = True
            return len(text)
        chunk = text[:room]
        self._size += len(chunk)
        if len(chunk) < len(text):
            self.truncated = True
        return super().write(chunk)


_CAPTURE_GLOBALS = {"__builtins__": _safe}
for _method in ("__init__", "write"):
    _fn = getattr(_Capture, _method)
    setattr(
        _Capture,
        _method,
        types.FunctionType(
            _fn.__code__, _CAPTURE_GLOBALS, _fn.__name__, _fn.__defaults__, _fn.__closure__
        ),
    )


def _make_formatter(_format_exc):
    def _format_error():
        return _format_exc()

    return _format_error


_formatter_tmpl = _make_formatter(traceback.format_exc)
_format_error = types.FunctionType(
    _formatter_tmpl.__code__, {}, "_format_error", None, _formatter_tmpl.__closure__
)
_compile = compile
_capture = _Capture(int(config.get("output_limit", 200000)))
sys.stdout = _capture

# Drop every other name from the bootstrap namespace before the script runs.
# From here on the only objects a script can reach either carry _API_GLOBALS /
# _CAPTURE_GLOBALS, or are plain data (`config`, `_safe`, `codemode`).
_KEEP = {
    "config", "codemode", "_API_GLOBALS", "_capture", "_emit", "_safe",
    "_format_error", "_compile",
}
for _key in list(globals()):
    if _key not in _KEEP:
        del globals()[_key]

_error = ""
try:
    exec(
        _compile(config["code"], "<codemode>", "exec"),
        {"__name__": "__main__", "__builtins__": _safe, "codemode": codemode},
    )
except BaseException:
    _error = _format_error()

_output = _capture.getvalue()
if _capture.truncated:
    _output += "\n...[output truncated]"
_emit({"done": True, "output": _output, "error": _error})
'''


CallHandler = Callable[[str, str, dict[str, Any]], Awaitable[Any]]


class _FrameTooLarge(Exception):
    """The child sent a frame past the parent's size ceiling."""


class _FrameReader:
    """Read newline-terminated frames with a hard byte cap.

    ``StreamReader.readuntil`` has no per-call limit on supported Pythons, and
    a plain ``readline`` would buffer the child's entire line first. Reading in
    bounded chunks into a small bytearray keeps a single huge frame from
    exhausting the parent's memory.
    """

    def __init__(self, stream: asyncio.StreamReader) -> None:
        self._stream = stream
        self._buf = bytearray()

    async def readline(self) -> bytes:
        cap = MAX_FRAME_CHARS + 1
        while True:
            end = self._buf.find(b"\n")
            if end != -1:
                line = bytes(self._buf[: end + 1])
                del self._buf[: end + 1]
                return line
            if len(self._buf) > MAX_FRAME_CHARS:
                self._buf.clear()
                raise _FrameTooLarge
            chunk = await self._stream.read(max(1, cap - len(self._buf)))
            if not chunk:
                line = bytes(self._buf)
                self._buf.clear()
                return line
            self._buf.extend(chunk)


async def run_code(
    *,
    code: str,
    tools: Sequence[dict[str, Any]],
    on_call: CallHandler,
    timeout: float,  # noqa: ASYNC109 - a configured budget, not a per-call API knob
    max_calls: int,
) -> CodeRun:
    """Run *code* in the sandbox, dispatching its tool calls through *on_call*.

    *tools* is the metadata the child turns into ``codemode.<helper>()``
    functions (helper, server, tool, description). *on_call* is awaited in the
    parent for every request; it must return a JSON-serialisable value or
    raise, and the exception's text becomes a catchable ``ToolError`` in the
    script. Runs over budget are killed; nothing of the child survives.
    """
    env = scrubbed_env(
        overrides={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    )
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-X",
            "utf8",
            "-c",
            BOOTSTRAP,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        return CodeRun(ok=False, error=f"could not start the code sandbox: {exc}")

    run = CodeRun(ok=False)
    try:
        # asyncio.wait_for spawns a task on 3.11-3.13; a cancel scope opened
        # there by on_call could never be exited from this task. A timeout
        # context keeps the whole exchange in one task.
        async with asyncio.timeout(timeout):
            run = await _converse(
                process,
                code=code,
                tools=tools,
                on_call=on_call,
                run=run,
                max_calls=max_calls,
            )
    except TimeoutError:
        run.ok = False
        run.error = f"the script ran longer than {timeout:g}s and was stopped"
    except Exception as exc:  # noqa: BLE001 - a broken protocol is reported, not raised
        run.ok = False
        run.error = f"the code sandbox failed: {type(exc).__name__}: {exc}"
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        with suppress(Exception):
            await process.wait()
        with suppress(Exception):
            if process.stdin is not None:
                process.stdin.close()
    if not run.ok and process.returncode not in (None, 0):
        # Read stderr even when the protocol already gave us an error: a
        # bootstrap crash is often the only explanation of a failed run.
        stderr = b""
        with suppress(Exception):
            if process.stderr is not None:
                stderr = await process.stderr.read()
        detail = stderr.decode("utf-8", "replace").strip()
        if run.error:
            if detail:
                run.error += f" (sandbox exited {process.returncode}: {detail[:1000]})"
        else:
            run.error = f"the code sandbox exited with status {process.returncode}"
            if detail:
                run.error += f": {detail[:1000]}"
    return run


async def _converse(
    process: asyncio.subprocess.Process,
    *,
    code: str,
    tools: Sequence[dict[str, Any]],
    on_call: CallHandler,
    run: CodeRun,
    max_calls: int,
) -> CodeRun:
    """Feed the child its config, then serve its calls until it finishes."""
    assert process.stdin is not None and process.stdout is not None
    request = {
        "code": code,
        "tools": list(tools),
        "output_limit": CHILD_OUTPUT_LIMIT,
    }
    process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    await process.stdin.drain()

    reader = _FrameReader(process.stdout)
    while True:
        try:
            line = await reader.readline()
        except _FrameTooLarge:
            run.ok = False
            run.error = "the code sandbox sent a frame that was too large"
            return run
        if not line:
            run.ok = False
            run.error = "the code sandbox ended without a result"
            return run
        if len(line) > MAX_FRAME_CHARS:
            run.ok = False
            run.error = "the code sandbox sent a frame that was too large"
            return run
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            run.ok = False
            run.error = f"the code sandbox sent malformed output: {exc}"
            return run

        if payload.get("done"):
            run.ok = not payload.get("error")
            run.output = str(payload.get("output") or "")
            run.error = str(payload.get("error") or "")
            return run

        call = payload.get("call")
        if not isinstance(call, dict):
            run.ok = False
            run.error = "the code sandbox sent an unknown request"
            return run
        server = str(call.get("server") or "")
        tool = str(call.get("tool") or "")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        run.calls += 1
        if run.calls > max_calls:
            reply = {
                "ok": False,
                "error": f"call budget exhausted ({max_calls} MCP calls per script)",
            }
        else:
            try:
                value = await on_call(server, tool, arguments)
                reply = {"ok": True, "value": value}
            except Exception as exc:  # noqa: BLE001 - the script may catch this
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        line_out = json.dumps(reply, ensure_ascii=False, default=repr) + "\n"
        if len(line_out) > MAX_FRAME_CHARS:
            line_out = json.dumps(
                {"ok": False, "error": "the tool result was too large to pass to the script"},
                ensure_ascii=False,
            ) + "\n"
        process.stdin.write(line_out.encode("utf-8"))
        await process.stdin.drain()
        if run.calls <= max_calls:
            run.trail.append(f"{server}.{tool}")


__all__ = ["CodeRun", "run_code", "BOOTSTRAP"]
