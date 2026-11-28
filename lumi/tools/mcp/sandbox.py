"""The Code Mode sandbox: run one model-written script in an isolated child.

Cloudflare's Code Mode runs generated TypeScript in a V8 isolate with no
network access, and every MCP call is an RPC back to the supervisor. This is
the same shape in Python. On Linux, bubblewrap is a required runtime
dependency; the script runs in a fresh interpreter (``-I -S``: isolated, no
site packages) with

* no inherited secrets (the child gets :func:`~lumi.util.env.scrubbed_env`),
* a restricted namespace: ``open``, ``eval``, ``exec`` and ``compile`` are
  absent, and ``import`` is limited to an allowlist of pure-data stdlib
  modules,
* one API object, ``codemode``, whose functions emit a JSON request on stdout
  and wait for the parent's reply on stdin,

so the child holds no MCP session, OAuth token or header. On Linux, bubblewrap
also gives the child private user, PID, network and mount namespaces, with only
the read-only Python runtime and an empty scratch directory visible. Execution
fails closed when that isolation is unavailable. The parent validates every
request against the tools discovered for this run and enforces the call budget.
``print()`` is how the script returns data: stdout is captured and sent back to
the model when the script ends.

The Python namespace restrictions are defense in depth, not the boundary:
Python introspection can reach interpreter internals. The OS boundary is
bubblewrap's user, PID, network and mount namespaces; the namespace exposes no
host home, project files, or writable host paths. A runaway script costs a
killed process rather than the turn. The import statement is served from a
table built before the script runs, so
``__builtins__['__import__']`` cannot import anything new. The allowlisted
modules are disarmed in place too: references to capability modules that
stdlib keeps in its own namespaces (``uuid.os``, ``enum.bltns``,
``typing.sys``) are replaced with stubs, every reachable module's
``__builtins__`` points at the same restricted mapping the script gets, and
``__spec__``/``__loader__`` are removed — a module's dict is also the
``__globals__`` of every function defined in it, so attribute access and
closure unpacking reach the same scrubbed view. The interpreter's own builtins
module cannot be hidden (builtin functions carry ``__self__``), so it is
disarmed instead: ``open``, ``eval``, ``exec`` and ``compile`` are replaced
with raising stubs and ``__import__`` becomes the sandboxed one.

What remains: deliberate C-level introspection can still reach interpreter
internals, including the real ``os`` module and ``_emit``'s budget-throttled
writer. Bubblewrap limits those capabilities to a private PID and network
namespace, a read-only Python runtime, and an empty read-only scratch mount.
This is an OS-enforced boundary on Linux, but not a defense against kernel
vulnerabilities. The approval prompt remains necessary because approved MCP
calls can still have consequential effects.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
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

#: Largest stderr kept for the failure report. The stream is read past this
#: point regardless: an undrained pipe fills and blocks the child mid-run.
STDERR_LIMIT = 16_000


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
#: over ``_API_GLOBALS``, a dict holding only ``ToolError``, ``_emit`` and the
#: restricted ``__builtins__``. The protocol streams and the import table live
#: in closure cells; the bootstrap namespace itself is wiped before user code
#: runs, so ``codemode._rpc.__globals__['builtins']`` and friends do not exist.
BOOTSTRAP = r'''
import builtins
import io
import json
import sys
import resource
import types

_proto_out = sys.stdout
_proto_in = sys.stdin

config = json.loads(_proto_in.readline())


def _cap_resource(_kind, _requested):
    _soft, _hard = resource.getrlimit(_kind)
    _cap = _requested if _hard == resource.RLIM_INFINITY else min(_requested, _hard)
    resource.setrlimit(_kind, (_cap, _cap))


_cap_resource(resource.RLIMIT_CPU, int(config["cpu_seconds"]))
_cap_resource(resource.RLIMIT_AS, 512 * 1024 * 1024)
_cap_resource(resource.RLIMIT_FSIZE, 0)
_cap_resource(resource.RLIMIT_NOFILE, 32)
_cap_resource(resource.RLIMIT_NPROC, 8)


class ToolError(Exception):
    pass


# The entire global namespace of every script-reachable callable. Keep it
# this small: introspecting __globals__ must not reveal a host capability.
_API_GLOBALS = {"ToolError": ToolError}


# --- the import table -------------------------------------------------------

_ALLOWED_IMPORTS = {
    "annotationlib",
    "ast",
    "base64",
    "binascii",
    "collections",
    "copy",
    "csv",
    "dataclasses",
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


#: Dotted names a script may import in their own right. `import a.b` and
#: `from a.b import c` need the leaf module in the table; the roots alone are
#: not enough once imports stop going through a real importer.
_ALLOWED_SUBMODULES = ("collections.abc", "html.entities", "html.parser", "json.decoder")


#: Built before user code runs, and the only source of imports afterwards.
#: A lookup has no importer in its closure or globals, so
#: `__builtins__['__import__']` cannot reach a module outside this table even
#: if the script unpacks the closure cells.
_MODULES = {}
for _name in sorted(_ALLOWED_IMPORTS | set(_ALLOWED_SUBMODULES)):
    try:
        _MODULES[_name] = builtins.__import__(_name, None, None, ("_",), 0)
    except Exception:
        # A module this interpreter cannot load simply stays unimportable.
        pass


def _make_import(_modules):
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or not isinstance(name, str):
            raise ImportError(
                "only absolute imports of allowlisted modules are available in the MCP sandbox"
            )
        module = _modules.get(name)
        if "." in name:
            if module is None:
                raise ImportError("module " + name + " is not available in the MCP sandbox")
            # `import a.b` binds the top package; `from a.b import c` needs a.b.
            return module if fromlist else _modules[name.split(".")[0]]
        if module is None:
            raise ImportError("module " + name + " is not available in the MCP sandbox")
        return module

    return _import


_import_tmpl = _make_import(_MODULES)
_import = types.FunctionType(
    _import_tmpl.__code__, {}, "_import", _import_tmpl.__defaults__, _import_tmpl.__closure__
)


# --- the script's builtins --------------------------------------------------

# __build_class__ makes `class` statements work; setattr/delattr/dir/vars round
# out the attribute tools started by getattr/hasattr. Code-execution and
# scope-inspection names (eval, exec, compile, open, globals, locals) stay out.
# ImportError is here because `_import` raises it: a script must be able to name
# the one exception its own import statement hands it.
_ALLOWED_NAMES = [
    "__build_class__", "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "delattr", "dict", "dir", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
    "str", "sum", "super", "tuple", "type", "vars", "zip",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "ImportError",
    "IndexError", "KeyError", "LookupError", "NameError", "OSError",
    "NotImplementedError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
]
_safe = {}
for _name in _ALLOWED_NAMES:
    _safe[_name] = getattr(builtins, _name)
_safe["__import__"] = _import
_safe["ToolError"] = ToolError

# Frames take their builtins from the globals mapping's `__builtins__` entry,
# so this line is what keeps a traceback frame reached from a caught ToolError
# from carrying the interpreter's builtins.
_API_GLOBALS["__builtins__"] = _safe


def _make_blocked(_message):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError(_message)

    return _blocked


_blocked_tmpl = _make_blocked("that builtin is not available in the MCP sandbox")
_blocked = types.FunctionType(
    _blocked_tmpl.__code__, {}, "_blocked", None, _blocked_tmpl.__closure__
)

# Builtin functions carry `__self__` (the real builtins module) and module
# dicts carry `__builtins__`, so the interpreter's builtins cannot be hidden
# from a script — only disarmed. Every module the script can reach shares this
# one mapping, which stops `print.__self__.open` and
# `uuid.__builtins__['open']` alike. `__import__` becomes the sandboxed one;
# the rest raise. The bootstrap itself still needs the real exec and compile,
# so capture those first.
_exec = exec
_compile = compile
for _name in (
    "open", "eval", "exec", "compile", "input", "breakpoint",
    "exit", "quit", "help", "license", "credits", "copyright",
):
    setattr(builtins, _name, _blocked)
builtins.__import__ = _import
for _attr in ("__spec__", "__loader__"):
    try:
        delattr(builtins, _attr)
    except AttributeError:
        pass


# --- stdout capture ---------------------------------------------------------

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

_capture = _Capture(int(config.get("output_limit", 200000)))
sys.stdout = _capture


# --- capability stubs and the module scrub ----------------------------------

def _make_stub(_name, _members):
    _stub = types.ModuleType(_name)
    for _key, _value in _members.items():
        setattr(_stub, _key, _value)
    return _stub


# Only data and scrubbed objects: every member here is either plain state or
# something the script could already get another way.
_SYS_STUB = _make_stub("sys", {
    "stdout": _capture,
    "modules": _MODULES,
    "maxsize": sys.maxsize,
    "maxunicode": sys.maxunicode,
    "byteorder": sys.byteorder,
    "version": sys.version,
    "platform": sys.platform,
    "hash_info": sys.hash_info,
    "float_info": sys.float_info,
    "implementation": sys.implementation,
})

_OS_STUB = _make_stub("os", {})


def _make_intern():
    # namedtuple and friends call sys.intern at call time; the real one is a
    # builtin whose __self__ is the real sys module, so serve a no-op instead
    # — interning only canonicalises strings, nothing depends on it here.
    def _intern(_s):
        return _s

    return _intern


_intern_tmpl = _make_intern()
_SYS_STUB.intern = types.FunctionType(
    _intern_tmpl.__code__, {}, "intern", None, _intern_tmpl.__closure__
)

#: Roots of modules that are host capabilities. When an allowlisted module's
#: namespace holds a reference to one, it is replaced or removed below; stdlib
#: keeps such references under many names (`uuid.os`, `enum.bltns`,
#: `random._os`, `typing.sys`).
_BLOCKED_MODULE_ROOTS = {
    "_posixsubprocess", "_thread", "_winapi", "antigravity", "asyncio",
    "atexit", "code", "codeop", "compileall", "concurrent", "ctypes",
    "curses", "dbm", "fcntl", "ftplib", "gc", "getpass", "glob", "http",
    "imaplib", "importlib", "io", "linecache", "marshal", "mmap",
    "multiprocessing", "nt", "ntpath", "pathlib", "pickle", "pkgutil",
    "platform", "poplib", "posix", "posixpath", "pty", "py_compile",
    "resource", "runpy", "selectors", "select", "shelve", "shutil",
    "signal", "site", "smtplib", "socket", "ssl", "subprocess",
    "syslog", "tarfile", "telnetlib", "tempfile", "termios", "threading",
    "tkinter", "tty", "turtle", "urllib", "warnings", "weakref",
    "webbrowser", "zipfile", "zipimport",
}


def _scrub_module(_module, _seen):
    """Disarm one module in place, then recurse into the modules it exposes.

    A module's ``__dict__`` is also the ``__globals__`` of every function
    defined in it, so replacing a reference here hides it from attribute
    access and ``__globals__`` unpacking at once.
    """
    if _module in _seen:
        return
    _seen.add(_module)
    _d = vars(_module)
    for _key, _value in list(_d.items()):
        if not isinstance(_value, types.ModuleType):
            continue
        _root = _value.__name__.split(".")[0]
        if _root == "sys":
            _d[_key] = _SYS_STUB
        elif _root == "os":
            _d[_key] = _OS_STUB
        elif _root == "builtins":
            _d[_key] = builtins
        elif _root in _BLOCKED_MODULE_ROOTS:
            del _d[_key]
        else:
            _scrub_module(_value, _seen)
    _d["__builtins__"] = builtins
    for _attr in ("__spec__", "__loader__"):
        if _attr in _d:
            del _d[_attr]


_seen = set()
for _mod in list(_MODULES.values()):
    _scrub_module(_mod, _seen)
# `types` is reachable through module references the scrub keeps
# (`typing.types`, `csv.types`), so it gets the same treatment.
_scrub_module(types, _seen)


def _make_uuid4(_uuid_module, _randbits):
    def _uuid4():
        # The stock uuid4 reads os.urandom through this module's globals;
        # with `os` scrubbed it cannot run, so draw the bits from the random
        # module instead (seeded by the interpreter's own entropy).
        _i = _randbits(128)
        _i &= _uuid_module._RFC_4122_CLEARFLAGS_MASK
        _i |= _uuid_module._RFC_4122_VERSION_4_FLAGS
        return _uuid_module.UUID._from_int(_i)

    return _uuid4


if (
    "uuid" in _MODULES
    and "random" in _MODULES
    and hasattr(_MODULES["uuid"], "_RFC_4122_CLEARFLAGS_MASK")
):
    _uuid4_tmpl = _make_uuid4(_MODULES["uuid"], _MODULES["random"].Random().getrandbits)
    _MODULES["uuid"].uuid4 = types.FunctionType(
        _uuid4_tmpl.__code__, {}, "_uuid4", None, _uuid4_tmpl.__closure__
    )


# --- the protocol -----------------------------------------------------------

def _make_writer(_write, _flush, _budget):
    def _out(_text):
        _budget[0] -= len(_text)
        if _budget[0] < 0:
            _budget[0] = 0
            raise ToolError("the sandbox's output budget is exhausted")
        _write(_text)
        # stdout is a pipe here: without a flush the parent would wait for a
        # reply to a request still sitting in the child's buffer.
        _flush()

    return _out


_writer_tmpl = _make_writer(
    _proto_out.write, _proto_out.flush, [int(config.get("emit_budget", 4_200_000))]
)
_out = types.FunctionType(
    _writer_tmpl.__code__, _API_GLOBALS, "_out", None, _writer_tmpl.__closure__
)


def _make_emit(_dumps, _out):
    def _emit(payload):
        _out(_dumps(payload, ensure_ascii=False, default=repr) + "\n")

    return _emit


_emit_tmpl = _make_emit(json.dumps, _out)
_emit = types.FunctionType(
    _emit_tmpl.__code__, _API_GLOBALS, "_emit", None, _emit_tmpl.__closure__
)
_API_GLOBALS["_emit"] = _emit


def _make_rpc(_readline, _loads):
    # Set once the parent says the budget is spent; every later call then
    # raises here instead of sending a request the parent can only refuse.
    spent = [False]

    def _rpc(server, tool, arguments):
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a dict")
        if spent[0]:
            raise ToolError("the call budget is already exhausted")
        _emit({"call": {"server": server, "tool": tool, "arguments": arguments}})
        line = _readline()
        if not line:
            raise ToolError("the host closed the connection")
        reply = _loads(line)
        if reply.get("fatal"):
            spent[0] = True
            raise ToolError(reply.get("error") or "the call budget is exhausted")
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
    _call_tmpl.__code__, _API_GLOBALS, "call", _call_tmpl.__defaults__, _call_tmpl.__closure__
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


def _make_formatter(_sys_module):
    def _format_error():
        # A hand-rolled traceback. The traceback module drags in linecache
        # (which reads files), io and codeop (which compile), and none of
        # those can be disarmed without breaking it; the model only needs
        # frames and messages to fix its script.
        _exc = _sys_module.exc_info()[1]
        if _exc is None:
            return "unknown sandbox error"
        _lines = ["Traceback (most recent call last):"]

        def _render(_e):
            _tb = _e.__traceback__
            while _tb is not None:
                _code = _tb.tb_frame.f_code
                _lines.append(
                    '  File "%s", line %d, in %s'
                    % (_code.co_filename, _tb.tb_lineno, _code.co_name)
                )
                _tb = _tb.tb_next
            _lines.append("%s: %s" % (type(_e).__name__, _e))

        def _chain(_e, _seen_ids):
            _render(_e)
            _other = _e.__cause__
            _label = "\nThe above exception was the direct cause of the following exception:"
            if _other is None:
                _other = _e.__context__
                _label = "\nDuring handling of the above exception, another exception occurred:"
            if _other is None or id(_other) in _seen_ids:
                return
            _seen_ids.add(id(_other))
            _lines.append(_label)
            _lines.append("Traceback (most recent call last):")
            _chain(_other, _seen_ids)

        _chain(_exc, {id(_exc)})
        return "\n".join(_lines)

    return _format_error


_formatter_tmpl = _make_formatter(sys)
_format_error = types.FunctionType(
    _formatter_tmpl.__code__, {}, "_format_error", None, _formatter_tmpl.__closure__
)

# Drop every other name from the bootstrap namespace before the script runs.
# From here on the only objects a script can reach either carry _API_GLOBALS /
# _CAPTURE_GLOBALS, or are plain data (`config`, `_safe`, `codemode`).
_KEEP = {
    "config", "codemode", "_API_GLOBALS", "_capture", "_emit", "_safe",
    "_format_error", "_compile", "_exec",
}
for _key in list(globals()):
    if _key not in _KEEP:
        del globals()[_key]

_error = ""
try:
    _exec(
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


async def _drain_stderr(stream: asyncio.StreamReader) -> bytes:
    """Read the child's stderr to EOF, keeping a bounded head.

    Nothing normally writes there, but warnings from an allowed import (or an
    import-time traceback) can fill the pipe buffer; if the parent only reads
    after the run, the child blocks until the wall-clock timeout kills it.
    """
    kept = bytearray()
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        room = STDERR_LIMIT - len(kept)
        if room > 0:
            kept.extend(chunk[:room])
    return bytes(kept)


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
    script. Calls past *max_calls* fail in the child without another round
    trip; a script that outlives *timeout* is killed.
    """
    env = scrubbed_env(
        overrides={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    )
    try:
        scratch = tempfile.TemporaryDirectory(prefix="lumi-codemode-")
    except OSError as exc:
        return CodeRun(ok=False, error=f"could not create the code sandbox scratch: {exc}")
    scratch_dir = Path(scratch.name)
    try:
        await asyncio.to_thread(scratch_dir.chmod, 0o500)
    except OSError as exc:
        scratch.cleanup()
        return CodeRun(ok=False, error=f"could not prepare the code sandbox scratch: {exc}")
    command = _bubblewrap_command(scratch_dir)
    if command is None:
        scratch.cleanup()
        return CodeRun(
            ok=False,
            error=(
                "secure code execution requires Linux, bubblewrap (bwrap), "
                "and a Python runtime under /usr"
            ),
        )
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        scratch.cleanup()
        return CodeRun(ok=False, error=f"could not start the code sandbox: {exc}")
    except BaseException:
        scratch.cleanup()
        raise

    run = CodeRun(ok=False)
    stderr = b""
    stderr_task: asyncio.Task[bytes] | None = None
    if process.stderr is not None:
        stderr_task = asyncio.create_task(_drain_stderr(process.stderr))
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
                cpu_seconds=max(1, math.ceil(timeout)),
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
        if stderr_task is not None:
            try:
                # The pipe reaches EOF when the child dies; do not let a
                # process that inherited stderr hold the cleanup open.
                async with asyncio.timeout(1.0):
                    stderr = await stderr_task
            except Exception:
                stderr_task.cancel()
        scratch.cleanup()
    if not run.ok and process.returncode not in (None, 0):
        # Report stderr even when the protocol already gave us an error: a
        # bootstrap crash is often the only explanation of a failed run.
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
    cpu_seconds: int,
) -> CodeRun:
    """Feed the child its config, then serve its calls until it finishes."""
    assert process.stdin is not None and process.stdout is not None
    request = {
        "code": code,
        "tools": list(tools),
        "output_limit": CHILD_OUTPUT_LIMIT,
        "emit_budget": MAX_FRAME_CHARS + CHILD_OUTPUT_LIMIT,
        "cpu_seconds": cpu_seconds,
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
        # The ceiling is on the payload; the newline that terminates the frame
        # is not part of it. _FrameReader allowed a frame of exactly
        # MAX_FRAME_CHARS payload bytes, so counting the terminator here would
        # reject it one byte later than the reader does.
        if line.endswith(b"\n"):
            line = line[:-1]
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
            # The child caps its own capture, but a script can reach _emit and
            # forge the final frame; the same cap is enforced here so the
            # forged channel cannot return more than print would have.
            output = str(payload.get("output") or "")
            if len(output) > CHILD_OUTPUT_LIMIT:
                output = output[:CHILD_OUTPUT_LIMIT] + "\n...[output truncated]"
            run.output = output
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
            # `fatal` tells the child to raise locally for every later call,
            # so a loop pays one round trip for the budget, not one per call.
            reply = {
                "ok": False,
                "fatal": True,
                "error": f"call budget exhausted ({max_calls} MCP calls per script)",
            }
        else:
            try:
                value = await on_call(server, tool, arguments)
                reply = {"ok": True, "value": value}
            except Exception as exc:  # noqa: BLE001 - the script may catch this
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # Same ceiling, same rule: measure the payload, then add its newline.
        line_out = json.dumps(reply, ensure_ascii=False, default=repr)
        if len(line_out) > MAX_FRAME_CHARS:
            line_out = json.dumps(
                {"ok": False, "error": "the tool result was too large to pass to the script"},
                ensure_ascii=False,
            )
        process.stdin.write((line_out + "\n").encode("utf-8"))
        await process.stdin.drain()
        if run.calls <= max_calls:
            run.trail.append(f"{server}.{tool}")


__all__ = ["CodeRun", "run_code", "BOOTSTRAP"]


#: Directories that must never be mounted into the jail even read-only: they
#: are either everything, or places a user or the running process keeps files
#: that are not the interpreter's. An interpreter whose prefix is one of these
#: (or lives under one of them) cannot be confined by prefix, so code mode is
#: refused on that host rather than run with the directory exposed. The root
#: itself is checked separately: every path is "under" it, so it must never be
#: an ancestor test.
_NEVER_MOUNT = (
    Path("/tmp"),
    Path("/var"),
    Path("/home"),
    Path("/root"),
    Path("/etc"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/run"),
    Path("/mnt"),
    Path("/media"),
    Path("/srv"),
    Path("/data"),
)


def _prefix_of(executable: Path) -> Path:
    """The install prefix a Python executable belongs to.

    ``<prefix>/bin/python`` for a normal install; ``<prefix>/python`` for a
    Windows-style layout that still shows up on Linux via Wine or a flatpak
    wrapper. Resolves symlinks first, so a venv's ``bin/python`` — a link to
    the base interpreter — reports the *base* prefix, which is the tree that
    holds the stdlib.
    """
    resolved = executable.resolve()
    if resolved.parent.name == "bin":
        return resolved.parent.parent
    return resolved.parent


def _mountable_prefix(prefix: Path, runtime_root: Path) -> bool:
    """Whether *prefix* is a directory the sandbox may bind-mount read-only.

    True for an install prefix under ``/usr`` (already mounted as a whole) and
    for a self-contained runtime tree elsewhere — a pyenv shim directory, a
    distro package under ``/opt``, a CI tool cache. False for anything too
    broad to expose: the root, a home, a temp or system directory, or a path
    that is not a directory at all. Fail closed: an unverifiable prefix is not
    mountable.
    """
    try:
        resolved = prefix.resolve()
    except OSError:
        return False
    if resolved == runtime_root or resolved.is_relative_to(runtime_root):
        return True
    if resolved == Path("/").resolve():
        return False
    for never in _NEVER_MOUNT:
        if resolved == never.resolve() or resolved.is_relative_to(never.resolve()):
            # /usr is handled above; anything else under these is a home, a
            # temp dir or a system directory, none of which is a runtime tree.
            return False
    return resolved.is_dir()


def _bubblewrap_command(scratch_dir: Path) -> list[str] | None:
    """Build a fail-closed Linux sandbox command for the child interpreter.

    The child runs ``sys._base_executable`` with ``-I -S`` — isolated, no
    site-packages — so the only tree it needs is the interpreter's own prefix
    plus the system libraries. ``/usr`` is always mounted read-only. An
    interpreter installed outside ``/usr`` (a pyenv, a distro package under
    ``/opt``, GitHub's tool cache) has its prefix mounted read-only as well, at
    the same path, which keeps the jail to exactly what the runtime is: no
    host filesystem, no network, nothing writable but the scratch directory.

    Returns None — and the caller refuses to run code — when the runtime cannot
    be confined that way: no Linux, no bubblewrap, or a prefix that is too
    broad to mount (the root, a home, a temp dir, or anything containing this
    project). Fail closed, never fall back to an unjailed interpreter.
    """
    if not sys.platform.startswith("linux"):
        return None
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        return None

    executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    runtime_root = Path("/usr").resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    # Every prefix the interpreter's own files live under. A venv's base_prefix
    # is the real runtime; the executable sits inside it, so the two normally
    # coincide — but a symlinked or relocatable install may split them, and
    # then both trees are needed for the stdlib import to succeed. /usr is
    # already mounted whole, so it is not a prefix to add — but a prefix that
    # *is* the root is refused below, never quietly dropped.
    prefixes = {base_prefix, _prefix_of(executable)}
    if any(not _mountable_prefix(p, runtime_root) for p in prefixes):
        return None
    prefixes.discard(runtime_root)

    command = [
        bubblewrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    # Runtime prefixes outside /usr: read-only, at their own path, so the
    # interpreter finds its stdlib exactly where it expects it.
    for prefix in sorted(prefixes):
        if prefix.is_dir():
            command.extend(("--ro-bind", str(prefix), str(prefix)))
    for path in ("/lib", "/lib64"):
        if Path(path).exists():
            command.extend(("--ro-bind", path, path))
    command.extend(
        (
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--remount-ro",
            "/dev",
            "--remount-ro",
            "/proc",
            "--ro-bind",
            str(scratch_dir),
            "/tmp",
            "--chdir",
            "/tmp",
            "--remount-ro",
            "/",
            "--clearenv",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "PYTHONIOENCODING",
            "utf-8",
            "--setenv",
            "PYTHONUTF8",
            "1",
            "--",
            str(executable),
            "-I",
            "-S",
            "-X",
            "utf8",
            "-c",
            BOOTSTRAP,
        )
    )
    return command
