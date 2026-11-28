# Code Mode implementation review

Review of `lumi/tools/mcp/codemode.py` + `lumi/tools/mcp/sandbox.py` against
Cloudflare's Code Mode pattern ([blog post](https://blog.cloudflare.com/code-mode-mcp/),
[agents docs](https://developers.cloudflare.com/agents/tools/codemode/)).

Re-checked on the current tree (commit `3999fc2`, 2026-11-28, plus the #12
cleanup). Code Mode execution now fails closed unless it can run inside Linux
bubblewrap namespaces, so the live `object.__subclasses__()` escape is contained
by the OS boundary rather than by Python scrubbing. **#1 routes through that
boundary; #2–#7 are fixed.** The suite was run on this tree with bubblewrap
present (`/usr/bin/bwrap`, Python under `/usr`): `tests/test_mcp_sandbox.py` +
`tests/test_mcp_sandbox_isolation.py` → **25 passed**, and
`tests/test_mcp_codemode.py` → **52 passed**. The `_wrap_close` regression test
is among the passing tests. Outside a Linux/bwrap environment the integration
tests skip, and execution itself refuses to run rather than falling back.

## What's right

- The shape is correct: three actions (`search` / `describe` / `execute`),
  generated helper API, one approved script per run, output fenced as untrusted,
  call budget, wall-clock budget, env scrubbing, trust gate checked *before* the
  approval prompt.
- The parent/child JSON-lines protocol with `MAX_FRAME_CHARS` and
  `CHILD_OUTPUT_LIMIT` is sound, and `_FrameReader` (`sandbox.py:648-677`)
  avoids unbounded `readline` buffering.
- Rebuilt `types.FunctionType` globals (`_API_GLOBALS`, `_CAPTURE_GLOBALS`)
  show real care about not leaking host capabilities through `__globals__`.
- `import` is served from a fixed preloaded table (`sandbox.py:142-195`), so the
  hook's closure hands over no importer.
- Allowlisted modules are scrubbed in place (`sandbox.py:385-421`) and the
  interpreter's `builtins` is disarmed (`274-287`): `uuid.os`, `typing.sys`,
  `enum.bltns` and `print.__self__.open` no longer reach the host.
- stderr is drained concurrently (`680-695`), frame caps agree (`648-677`,
  `846-853`), and budget exhaustion fails fast in the child (`489-507`,
  `884-892`).

## Bugs 1–7: current status

### 1. Sandbox escape — contained by an OS namespace

**First exploit (fixed and still fixed).** The restricted `__import__` no longer
ships the real one. `_MODULES` is built from a fixed allowlist before user code
runs (`sandbox.py:142-195`); unpacking
`__builtins__['__import__'].__closure__` yields module/helpers, not a callable
importer, and the four-line PoC from the original review raises. Pinned by
`tests/test_mcp_sandbox.py:88-105`.

**Second round (fixed).** Allowlisted modules are disarmed in place:
capability references are replaced with stubs, every reachable module's
`__builtins__` is the restricted mapping, `__spec__`/`__loader__` are removed
(`sandbox.py:385-421`), and the interpreter's own `builtins` module is patched
so `open`, `eval`, `exec`, `compile` and friends raise (`274-287`). Verified on
the current tree: `uuid.os.getcwd()` raises `AttributeError`,
`print.__self__.open` raises `RuntimeError`, `__builtins__['open']` is absent,
and `import socket` is refused. Tests at
`tests/test_mcp_sandbox.py:162-262`.

**The Python-level path remains reachable.** `object` is in `_ALLOWED_NAMES`
(`sandbox.py:228-242`), and the import machinery loaded at interpreter start
keeps the *real* `sys` module in its function globals. The scrub does not reach
it: `importlib` is in `_BLOCKED_MODULE_ROOTS` (`sandbox.py:370-383`) and is
never recursed into. The original demonstration therefore still reaches real
`os` objects, exactly as the review found. The effective fix is outside Python:

`run_code` launches the child through bubblewrap with private user, PID,
network and mount namespaces (`sandbox.py:728-750`, `914-982`). Only `/usr`,
the required runtime library paths, `/dev`, `/proc` for the private PID
namespace, and a per-run empty read-only scratch directory are mounted. The
host home, repository, `/etc`, and writable host paths are not visible. The
`object.__subclasses__()` route can still inspect `sys.modules['os']`, but
filesystem operations are confined to the read-only runtime and scratch mount,
process visibility is limited to the private PID namespace, and networking is
unshared. Child CPU, address space, file size, descriptor count and process
count are capped (`sandbox.py:124-128`). Unsupported platforms, runtimes
outside `/usr`, missing bubblewrap, or failed namespace creation do not fall
back to unsandboxed execution (`728-750`; tested in
`tests/test_mcp_sandbox_isolation.py`).

**Verdict:** the interpreter escape is still demonstrable, but the Linux
bubblewrap configuration confines it away from the host filesystem and
network. This is an OS namespace boundary, not a defense against kernel
vulnerabilities. The regression test `_wrap_close` → real `sys` → real `os`
(`tests/test_mcp_sandbox.py:265-312`) passes here: cwd is `/tmp`, the host file
cannot be read or removed, scratch and `/` are read-only, and the resource
limits are in force.

### 2. Helper-name collisions silently retarget tools — FIXED

`_api` (`codemode.py:623-658`) tracks every assigned helper name in a `set[str]`
and bumps a suffix until the candidate is free (`639-642`). Bases `x_2`, `x`, `x`
now yield `srv_x_2`, `srv_x`, `srv_x_3`, so the bootstrap cannot overwrite a
helper with a different tool's. Regression test:
`tests/test_mcp_codemode.py:1110`.

### 3. `class` statements can't run in the sandbox — FIXED

`_ALLOWED_NAMES` (`sandbox.py:228-242`) includes `__build_class__`, `setattr`,
`delattr`, `dir` and `vars`; `ImportError` was also added because the import
hook raises it (`218-242`). `eval`, `exec`, `compile`, `open`, `globals` and
`locals` remain absent. Tests: `tests/test_mcp_sandbox.py:375-395` (class and
attribute tools) and `108-121` (catchable `ImportError`).

### 4. Child stderr is never drained while the script runs — FIXED

`_drain_stderr` (`sandbox.py:680-695`) starts as a task in `run_code`
(`757-758`) before `_converse`, reads to EOF while keeping a bounded head
(`STDERR_LIMIT = 16_000`, `81`), and is joined with a 1s cap in cleanup
(`788-794`). A 500 KB stderr flood no longer blocks the script; test at
`tests/test_mcp_sandbox.py:459-477`.

### 5. Frame-size checks disagree by one byte — FIXED

`_converse` strips the newline before measuring (`sandbox.py:846-853`), so a
payload of exactly `MAX_FRAME_CHARS` bytes is accepted by both reader and
caller. Tests at `tests/test_mcp_sandbox.py:519-543`.

### 6. SSRF guard is keyed on the tool's *name* — FIXED

`ServerSpec.fetches_urls` (`servers.py:293-312`) decides from the server, not
the tool name: an explicit `fetches: true|false` wins; otherwise a local server
or a remote endpoint that is itself loopback/private counts as a fetcher, and a
public remote server does not. `on_call` uses `pair[0].fetches_urls`
(`codemode.py:560-577`), and the approval UI surfaces an opt-out as "URL
arguments unchecked" (`tests/test_mcp_codemode.py:858-877`).

Residual design note: MCP declares no fetch capability, so the default is
inferred from transport. A public remote server that follows caller-supplied
URLs into internal networks needs `fetches: true` in its config entry; that is
an explicit owner choice rather than a name regex.

### 7. `_url_targets` misses non-literal URLs — FIXED

`_URL_PATTERN` (`codemode.py:94-101`) matches a URL anywhere inside a string —
any `scheme://…`, plus host-shaped protocol-relative `//host/…` — and
`_url_targets` (`678-710`) walks nested dicts/lists/tuples and dedupes.
`validate_fetch_url` refuses non-http(s) schemes, so an embedded
`file:///etc/passwd` is caught too. Test:
`tests/test_mcp_codemode.py:880-892`.

Residual: a URL obfuscated without `://` or a host-shaped `//` prefix is still
not extracted — outside the original finding, but worth remembering when adding
schemes.

## Gaps vs. Cloudflare's Code Mode — current state

### 8. `search` gives signatures, not schemas — ADDRESSED

`search` still renders one-line signatures (`codemode.py:416`, `742-756`), but
the new `describe` action returns the tool's advertised `inputSchema` verbatim
via `_input_schema` (`725-739`), so nested objects, `enum` and per-parameter
descriptions no longer have to be guessed.

### 9. No `describe` step; descriptions truncated — ADDRESSED

`describe` (`codemode.py:442-501`) returns the full description and schema
(capped at `DESCRIBE_MAX_CHARS = 12000`, `67`), `search` appends a pointer to it
(`424-427`), and `invoke` routes the new action (`206-214`).

### 10. Approval preview truncated at 2000 chars — ADDRESSED

`_preview` (`codemode.py:759-865`) now builds an AST-based structural summary of
every `codemode` call site in source order, including aliases, `getattr`
indirection and bare `vars()` calls, under `PREVIEW_MAX_CHARS = 900` (`78`),
sized to fit the bot's 1200-char card (`bot.py:1290-1320`).

### 11. Sandbox cost not stated in the prompt — ADDRESSED

The tool description (`codemode.py:118`) and the approval reason
(`539-543`) now describe containment rather than promising an absence of
filesystem/network/subprocess access.

### 12. Minor — FIXED

- Budget exhaustion fail-fast: **fixed** (`sandbox.py:489-507`, `884-892`; test
  `tests/test_mcp_sandbox.py:479-507`).
- `data["servers"]` now lists the servers actually dispatched, tracked in
  `_execute`'s `on_call` (`codemode.py:558-567`) and reported on both paths
  (`606`, `616`). A server that was discovered but never called no longer
  appears; `run.trail` is not consulted because a server name may itself
  contain the `server.tool` separator. Tests cover the success and failed-run
  cases (`tests/test_mcp_codemode.py:442`).
- Host exception text no longer reaches the script. `on_call` re-raises
  `ToolError`/`MCPCallError` unchanged, so a server's own failure message stays
  catchable (`codemode.py:578-581`, test at `tests/test_mcp_codemode.py:894`),
  and rewrites any other exception as `"<Class>: the tool call failed"`
  (`582-586`), so an internal message cannot carry local paths or credentials
  into model context (test at `467`).
- `search` matches all query terms first and falls back to any-term matching
  only when that returns nothing (`codemode.py:319-363`), with the body saying
  so (`410`). A narrow query stays narrow; a query whose terms never share a
  tool still returns candidates instead of an empty catalog.
- Matches are ranked, not just filtered: exact tool name, then a word in the
  name, a name substring, the server, then the description (`326-346`), with a
  stable server/tool tie-break. `limit` (default 50, max 200) caps the listing;
  when more matched than are shown, the header reads `shown of matched` and the
  body says to narrow (`387-408`). Helper names are still assigned over the
  whole catalog before filtering, so a ranked or server-filtered listing never
  advertises a name `execute` would have numbered differently. Tests: ranking
  `tests/test_mcp_codemode.py:144`, limit `162`, helper stability under ranking
  `187`, AND fallback `124`, and the no-match case at `204`, which still fails
  when no term matches anything.

## Suggested priority now

1. #1 no longer needs new machinery: the bubblewrap boundary is implemented,
   fail-closed, and covered by a passing regression test. The remaining risk is
   inherent to the OS boundary (kernel vulnerabilities) and should be stated as
   such; the approval prompt stays necessary because approved MCP calls can
   still have consequential effects.
2. #12 is closed. `search`'s AND-then-OR fallback keeps narrow queries narrow
   and only loosens when AND would return nothing, so there is nothing left to
   schedule there.
3. Capability work is largely closed by `describe`; `search` still shows
   signatures rather than schemas by design, to keep the catalog small.
