# Code Mode implementation review

Review of `lumi/tools/mcp/codemode.py` + `lumi/tools/mcp/sandbox.py` against
Cloudflare's Code Mode pattern ([blog post](https://blog.cloudflare.com/code-mode-mcp/),
[agents docs](https://developers.cloudflare.com/agents/tools/codemode/)).

Re-checked #1 against the current tree after another round of sandbox hardening.
The `object.__subclasses__()` route still reaches the real import machinery, so
Python-level scrubbing alone is not a security boundary. Code Mode execution now
fails closed unless it can run inside Linux bubblewrap namespaces. **#1 now routes through the OS boundary; #2–#7 remain closed.** The regression
test for the escape was added but has not yet been run. The existing suite
result below predates this change.

## What's right

- The shape is correct: three actions (`search` / `describe` / `execute`),
  generated helper API, one approved script per run, output fenced as untrusted,
  call budget, wall-clock budget, env scrubbing, trust gate checked *before* the
  approval prompt.
- The parent/child JSON-lines protocol with `MAX_FRAME_CHARS` and
  `CHILD_OUTPUT_LIMIT` is sound, and `_FrameReader` avoids unbounded
  `readline` buffering.
- Rebuilt `types.FunctionType` globals (`_API_GLOBALS`, `_CAPTURE_GLOBALS`)
  show real care about not leaking host capabilities through `__globals__`.
- `import` is served from a fixed preloaded table (`sandbox.py:160-195`), so the
  hook's closure hands over no importer.
- Allowlisted modules are scrubbed in place (`sandbox.py:301-422`) and the
  interpreter's `builtins` is disarmed (`230-261`): `uuid.os`, `typing.sys`,
  `enum.bltns` and `print.__self__.open` no longer reach the host.
- stderr is drained concurrently (`657-672`), frame caps agree
  (`625-654`, `799-805`), and budget exhaustion fails fast in the child
  (`463-483`, `840-845`).

## Bugs 1–7: current status

### 1. Sandbox escape — contained by an OS namespace

**First exploit (fixed and still fixed).** The restricted `__import__` no longer
ships the real one. `_MODULES` is built from a fixed allowlist before user code
runs (`sandbox.py:160-195`); unpacking `__builtins__['__import__'].__closure__`
yields a dict, not a callable importer, and the four-line PoC from the original
review raises `IndexError`. Pinned by
`tests/test_mcp_sandbox.py:80-98`.

**Second round (fixed).** Allowlisted modules are now disarmed in place:
capability references are replaced with stubs, every reachable module's
`__builtins__` is the restricted mapping, `__spec__`/`__loader__` are removed
(`sandbox.py:301-422`), and the interpreter's own `builtins` module is patched
so `open`, `eval`, `exec`, `compile` and friends raise (`230-261`). Verified on
the current tree: `uuid.os.getcwd()` raises `AttributeError`,
`print.__self__.open` raises `RuntimeError`, `__builtins__['open']` is absent,
and `import socket` is refused. Tests at
`tests/test_mcp_sandbox.py:154-254`.

**The Python-level path remains reachable.** `object` is in `_ALLOWED_NAMES`
(`sandbox.py:205-217`), and the import machinery loaded at interpreter start
keeps the *real* `sys` module in its function globals. The scrub does not reach
it: `importlib` is in `_BLOCKED_MODULE_ROOTS` (`347-359`) and is never recursed
into. The original demonstration therefore still reaches real `os` objects.
The effective fix is outside Python:

`run_code` launches the child through bubblewrap with private user, PID,
network and mount namespaces. Only `/usr`, the required runtime library paths,
`/dev`, `/proc` for the private PID namespace, and a per-run empty read-only
scratch directory are mounted. The host home, repository, `/etc`, and writable
host paths are not visible. The `object.__subclasses__()` route can still
inspect `sys.modules['os']`, but filesystem operations are confined to the
read-only runtime and scratch mount, process visibility is limited to the
private PID namespace, and networking is unshared. Child CPU, address space,
file size, descriptor count and process count are capped. Unsupported
platforms, runtimes outside `/usr`, missing bubblewrap, or failed namespace
creation do not fall back to unsandboxed execution.

**Verdict:** the interpreter escape is still demonstrable, but the Linux
bubblewrap configuration confines it away from the host filesystem and
network. This is an OS namespace boundary, not a defense against kernel
vulnerabilities. A regression test now exercises the exact `_wrap_close` path
and checks that host files are inaccessible and scratch is read-only; it still
needs to be run in an environment where bubblewrap namespaces are available.

### 2. Helper-name collisions silently retarget tools — FIXED

`_api` (`codemode.py:538-567`) tracks every assigned helper name in a `set[str]`
and bumps a suffix until the candidate is free (`543-550`). Bases `x_2`, `x`, `x`
now yield `srv_x_2`, `srv_x`, `srv_x_3`, so the bootstrap cannot overwrite a
helper with a different tool's. Regression test:
`tests/test_mcp_codemode.py:977`.

### 3. `class` statements can't run in the sandbox — FIXED

`_ALLOWED_NAMES` (`sandbox.py:205-217`) includes `__build_class__`, `setattr`,
`delattr`, `dir` and `vars`; `ImportError` was also added because the import
hook raises it (`203-204`). `eval`, `exec`, `compile`, `open`, `globals` and
`locals` remain absent. Tests: `tests/test_mcp_sandbox.py:100-113` and `318`.

### 4. Child stderr is never drained while the script runs — FIXED

`_drain_stderr` (`sandbox.py:657-672`) starts as a task in `run_code`
(`714-716`) before `_converse`, reads to EOF while keeping a bounded head
(`STDERR_LIMIT = 16_000`, `72`), and is joined with a 1s cap in cleanup
(`745-752`). A 500 KB stderr flood no longer blocks the script; test at
`tests/test_mcp_sandbox.py:402`.

### 5. Frame-size checks disagree by one byte — FIXED

`_converse` strips the newline before measuring (`sandbox.py:799-805`), so a
payload of exactly `MAX_FRAME_CHARS` bytes is accepted by both reader and
caller. Tests at `tests/test_mcp_sandbox.py:462`.

### 6. SSRF guard is keyed on the tool's *name* — FIXED

`ServerSpec.fetches_urls` (`servers.py:283-312`) decides from the server, not
the tool name: an explicit `fetches: true|false` wins; otherwise a local server
or a remote endpoint that is itself loopback/private counts as a fetcher, and a
public remote server does not. `on_call` uses `pair[0].fetches_urls`
(`codemode.py:493-500`), and the approval UI surfaces an opt-out as "URL
arguments unchecked" (`tests/test_mcp_codemode.py:725-743`).

Residual design note: MCP declares no fetch capability, so the default is
inferred from transport. A public remote server that follows caller-supplied
URLs into internal networks needs `fetches: true` in its config entry; that is
an explicit owner choice rather than a name regex.

### 7. `_url_targets` misses non-literal URLs — FIXED

`_URL_PATTERN` (`codemode.py:89-99`) matches a URL anywhere inside a string —
any `scheme://…`, plus host-shaped protocol-relative `//host/…` — and
`_url_targets` (`593-620`) walks nested dicts/lists/tuples and dedupes.
`validate_fetch_url` refuses non-http(s) schemes, so an embedded
`file:///etc/passwd` is caught too. Test:
`tests/test_mcp_codemode.py:747`.

Residual: a URL obfuscated without `://` or a host-shaped `//` prefix is still
not extracted — outside the original finding, but worth remembering when adding
schemes.

## Gaps vs. Cloudflare's Code Mode — current state

### 8. `search` gives signatures, not schemas — ADDRESSED

`search` still renders one-line signatures (`codemode.py:657-673`), but the new
`describe` action returns the tool's advertised `inputSchema` verbatim via
`_input_schema` (`640-655`), so nested objects, `enum` and per-parameter
descriptions no longer have to be guessed.

### 9. No `describe` step; descriptions truncated — ADDRESSED

`describe` (`codemode.py:369-420`) returns the full description and schema
(capped at `DESCRIBE_MAX_CHARS = 12000`), `search` appends a pointer to it
(`352-354`), and `invoke` routes the new action (`197-203`).

### 10. Approval preview truncated at 2000 chars — ADDRESSED

`_preview` (`codemode.py:674+`) now builds an AST-based structural summary of
every `codemode` call site in source order, including aliases and `getattr`
indirection, under `PREVIEW_MAX_CHARS = 900`, sized to fit the bot's 1200-char
card (`bot.py:1239-1254`).

### 11. Sandbox cost not stated in the prompt — ADDRESSED

The tool description (`codemode.py:118`) and the approval reason
(`466-470`) now describe containment rather than promising an absence of
filesystem/network/subprocess access.

### 12. Minor — PARTLY ADDRESSED

- Budget exhaustion fail-fast: **fixed** (`sandbox.py:463-483`, `840-845`; test
  `tests/test_mcp_sandbox.py:422`).
- `data["servers"]`/`ToolResult.servers` still list every discovered server, not
  the ones called (`codemode.py:521, 531`). `run.trail` exists
  (`sandbox.py:84`) but is not used here — cosmetic, still open.
- Parent→child error strings still include `type(exc).__name__` and the
  exception text (`sandbox.py:852`) — still worth a glance.
- `search` still requires all query terms to match the concatenated
  `server + name + description` string (`codemode.py:301-309`) — acceptable,
  still AND-only.

## Suggested priority now

1. #1 is the only substantive item left. The namespace hardening closed the
   two known Python-level paths, but the `object.__subclasses__()` →
   `_wrap_close.__init__.__globals__['sys']` → `sys.modules['os']` path is a
   live, four-line escape. Either accept it explicitly (current position:
   approval is the gate, prompt says so) or move the script to a real boundary
   with a separate OS user / seccomp / a JS isolate.
2. Cheap cleanup from #12: report called servers from `run.trail`, and consider
   trimming exception text sent into the script.
3. Capability work is largely closed by `describe`; `search` still shows
   signatures rather than schemas by design, to keep the catalog small.
