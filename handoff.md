# Code review — handoff

Written 2026-10-06 after a full read-through of `lumi/` (~11.6k lines) plus the
tool, LLM, and Telegram layers. Every item marked **VERIFIED** was reproduced
with a throwaway test against the real test harness before it was written down;
the repro is included so it can be turned into a regression test as-is.

**Baseline (unchanged by this review):**

```
./.venv/bin/ruff check lumi tests   → All checks passed!
./.venv/bin/pytest -q               → 837 passed in ~40s
```

No source files were modified. The only artefact of this review is this document.

---

## Contents

1. [Verified bugs](#1-verified-bugs) — 8, each reproducible
2. [Worth a look](#2-worth-a-look) — lower-confidence or judgement calls
3. [Minor improvements](#3-minor-improvements)
4. [Test gaps](#4-test-gaps)
5. [Repo notes](#5-repo-notes)
6. [What looked solid](#6-what-looked-solid)
7. [Suggested order of work](#7-suggested-order-of-work)

---

## 1. Verified bugs

### 1.1 The `context7` tool crashes on every real call — **HIGH**

`lumi/tools/context7.py:101`

```python
client = httpx.AsyncClient(
    timeout=httpx.Timeout(self.settings.timeout_seconds),   # ← self.settings
    ...
```

`Context7Client.__init__` (line 81) only assigns `self.config`. `self.settings`
exists on `Context7Tool` (line 366), not on `Context7Client`, so the first HTTP
call raises:

```
AttributeError: 'Context7Client' object has no attribute 'settings'
```

The registry turns it into `ToolResult.failure("context7 crashed: …")`, so the
model sees a broken tool rather than a stack trace — and with
`CONTEXT7_API_KEY` set the tool *is* advertised to the model (`available()`
passes), which is exactly when this fires.

Why it has never been noticed: `tests/test_context7.py:95` monkeypatches
`client._client_for` with its own factory, so the real body is never executed.

An AST sweep of the whole package for `self.<attr>` reads with no matching
assignment reports exactly one hit: this one.

**Fix:** `self.config.timeout_seconds` (one word).

**Regression test:** construct `Context7Client(Context7Config(api_key_env=...))`
with a key set, call `_client_for("k")`, assert an `httpx.AsyncClient` comes
back — no monkeypatching.

---

### 1.2 Parallel tool calls + one approval = a permanently 400-ing conversation — **HIGH**

`lumi/agent.py:425` (`_turn`) and `lumi/agent.py:444` (`_handle_tool_call`)

```python
for call in reply.tool_calls:
    if await self._handle_tool_call(conv, call, ctx, tools, result):
        # Something needs the owner; stop rather than continue with a
        # half-finished set of side effects.
        result.tools_used = tools
        ...
        return result
```

The comment is right about the side effects but the loop stops *mid-set*: if the
model returns three `tool_calls` and the second one raises `NeedsApproval`, only
the second gets a `role: "tool"` message. The third has none.

Every provider validates this strictly — an assistant message carrying
`tool_calls` must be followed by one `tool` message per `tool_call_id`. So the
*next* `llm.complete()` 400s, the turn dies with `LLMError`, and because the
malformed pair stays in `conv.messages`, **every subsequent turn fails too** until
the owner types `/reset`.

**VERIFIED** (`run_shell` `rm x.txt` = CONFIRM tier, `ls` = ALLOW):

```
TOOL MESSAGES: [('c1', "Not executed. It deleting files, so it needs the owner's app...")]
MISSING TOOL RESPONSES: ['c2']
```

**Fix:** when a call needs approval, still dispatch the *remaining* calls in the
same reply (they have no approval requirement of their own — or record each that
does), appending a `role: "tool"` message for every one before returning. The
invariant to enforce is "one tool message per tool_call_id, always", which the
comment on `lumi/agent.py:465` already states ("Exactly one tool message per
tool_call, or the next request 400s"); the early `return` is the only thing that
violates it.

**Regression test:** one scripted reply with two tool calls, the first needing
approval; after the turn, assert every `tool_call_id` in the last assistant
message has a matching `role: "tool"` message.

---

### 1.3 Photo captions are silently dropped — **MEDIUM** (README says otherwise)

`README.md:73` promises:

> Send the bot a photo and it looks at it. **The caption becomes your question**;
> a photo with no caption gets a default "what's in this image?".

It doesn't. PTB's `filters.TEXT` is `bool(message.text)` (checked against
`telegram/ext/filters.py:2777` in v22.8) and a photo message has `text = None`,
`caption = …` — so a captioned photo matches `~filters.TEXT` and lands in
`on_non_text` (`lumi/bot.py:1329`), which builds the prompt itself:

```python
await run_agent(update, context, "what's in this image?", image_b64=image_b64)
```

The caption handling that *does* exist in `on_text` (`lumi/bot.py:1275`,
`if message.photo:`) is unreachable dead code — `on_text` only ever fires for
real text messages.

**VERIFIED:**

```
USER MESSAGES SENT TO MODEL: [[{'type': 'text', 'text': "what's in this image?"}, …]]
AssertionError: caption never reached the model
```

**Fix:** in `on_non_text`, prefer `message.caption` when present:

```python
prompt = (message.caption or "").strip() or "what's in this image?"
await run_agent(update, context, prompt, image_b64=image_b64)
```

and delete the dead branch in `on_text`. (Documents already do this correctly in
`on_document`.)

**Regression test:** update carrying `photo` + `caption`; assert the caption text
reaches the model.

---

### 1.4 `/whitelist_add_user` has no effect until the process restarts — **MEDIUM**

`lumi/bot.py:1504-1512`

```python
allowed_user_ids: list[int] = []
if owner is not None:
    allowed_user_ids.append(owner)
for uid in config.bot.whitelisted_users:
    ...
owner_filter = filters.User(user_id=allowed_user_ids) if allowed_user_ids else filters.User(user_id=0)
```

`owner_filter` is a snapshot taken once when handlers are built. The handler
appends to `config.bot.whitelisted_users` and persists it, but PTB's filter —
which runs *before* every handler — still has the old id list, so the new user
falls through to `MessageHandler(on_stranger, filters=~owner_filter)` and gets
"this bot is private".

Removal works (the handlers re-check `_is_allowed` against live config); only
*addition* is broken. Asymmetric, which is what makes it confusing.

**VERIFIED** on one long-lived application:

```
ADD RESULT: added <code>777</code> to the user whitelist.
777 SAYS:   this bot is private. your id is not whitelisted.
```

**Fix options (pick one):**
- Replace the static `filters.User(...)` with a custom `UpdateFilter` that calls
  `_is_allowed(update.effective_user, config)` at match time (preferred — one
  source of truth), or
- keep a mutable id set in `bot_data` that `build_handlers` reads through, or
- document loudly that a restart is required and have the command say so.

**Regression test:** one application, `/whitelist_add_user 777`, then a message
from `777` on the *same* application.

---

### 1.5 `tools.files.max_read_chars = 0` returns an empty file — **MEDIUM**

`lumi/tools/files.py:185`

```python
cap = int(arguments.get("max_bytes") or self.settings.max_read_chars)
...
body = handle.read(cap)          # read(0) == ""
```

`0` is documented as "no cap" in three places — `lumi/config.py:352`,
`config.toml:212` and `CONFIGURATION.md:98` — and `/config` even renders it as
`read no cap` (`lumi/bot.py:279`). In practice the read returns nothing and the
header claims the file was truncated:

**VERIFIED:**

```
READ WITH cap=0 -> '--- workspace/notes.txt (500 bytes, truncated) ---\n'
AssertionError: documented 'no cap' returned an empty file
```

**Fix:**

```python
cap = int(arguments.get("max_bytes") or 0) or self.settings.max_read_chars
if cap <= 0:
    body = handle.read()          # documented "no cap"
else:
    body = handle.read(cap)
```

(and keep `size > cap` comparisons guarded by `cap > 0`).

---

### 1.6 `lumi chat` never prints the answer after you approve something — **MEDIUM**

`lumi/__main__.py:181-184`

```python
if result.pending:
    print("\n".join(...))
    answer = (await _prompt("  run it? [y/N] ")).strip().lower()
    for action in list(result.pending):
        ...
        result = await agent.resolve("cli", action.id, approved, source="cli")
elif result.text:
    print(result.text)
```

`agent.resolve()` returns the *resumed* turn — i.e. the model's actual reply to
the command it just ran — and that value is assigned to `result` inside the
branch that owns the `elif`, so `result.text` is never printed. The whole point
of approving a command in a REPL is to see what happened next.

**VERIFIED** (scripted turn → `rm x.txt` needs approval → `y` → resumed reply):

```
CLI OUTPUT >>> … lumi >   needs approval — deleting files: rm x.txt
                  working…
                  bye
AssertionError: approved turn's answer was never printed
```

**Fix:** after the approval loop, print the final result the same way the normal
path does — restructure so both branches fall through to a shared
`if result.text: print(result.text)`.

---

### 1.7 `/whitelist_*` can corrupt `config.local.toml` — **MEDIUM**

`lumi/bot.py:1483` (`_save_whitelist`) → `lumi/bot.py:732/746`
(`_toml_value` / `_write_toml`)

The whole file is parsed, mutated in memory and rewritten with a hand-rolled
TOML writer. Two ways this loses data:

1. **`except Exception: existing = {}`** (line 1491) — if the file is already
   unparseable, it is *silently replaced* by the whitelist alone. Whatever was in
   there is gone.
2. **`_toml_value` cannot round-trip a string containing a newline** (it only
   escapes `\` and `"`), multi-line `"""` strings, inline tables, arrays of
   tables, datetimes or integers-with-expressions.

**VERIFIED** — a `config.local.toml` containing one multi-line string, passed
through `_write_toml`:

```
[bot]
motd = "line one
line two"
…
REPARSE FAILED: Illegal character '\n' (at line 4, column 17)
```

The consequence is worse than a lost value: `_read_toml` raises `ConfigError` on
the next boot, so `lumi run` refuses to start with
`configuration error: … is not valid TOML`.

**Fix (cheap and safe):** store the whitelist in its own small file
(`data/whitelist.json`) instead of rewriting the user's config; or, if it must
stay in `config.local.toml`, only patch the two known keys textually and abort
loudly (rather than `existing = {}`) when the file cannot be re-serialised.

---

### 1.8 Tapping a stale inline button raises an unhandled `BadRequest` — **MEDIUM**

`lumi/bot.py:1230, 1241, 1430, 1435, 1443, 1446` — every `query.answer(...)`
call is unguarded, unlike `query.edit_message_text`, which `_edit_thinking`
wraps carefully.

Telegram expires callback queries quickly (and PTB processes updates
sequentially, so a tap that queues behind a long agent turn can easily be
answered late). `query.answer()` then raises:

```
telegram.error.BadRequest: Query is too old and response timeout expired
                           or query id is invalid
```

which propagates to `_error_handler` and is logged as an unhandled error — and
because it raises *before* the intended follow-up, the tap does nothing at all:
`_expand_thinking` never expands, `button` never resolves the approval.

This is not hypothetical — it is in the repo. `something.txt` (committed in
`4dc53c5 somehing`) is a captured log of exactly this traceback twice, ending in
`^C`.

**Fix:** wrap every `await query.answer(...)` in
`contextlib.suppress(BadRequest, TelegramError)` (or a tiny local helper
`async def _toast(query, text, **kw)`), and keep going: `edit_message_text` and
the approval resolution do not depend on the toast succeeding.

---

## 2. Worth a look

These I did **not** reproduce; they need a decision or a second opinion.

| # | Where | Concern |
|-|-|-|
| 2.1 | `lumi/config.py:347` (`FilesConfig`) vs `lumi/tools/files.py:152` | **`readable_from_project` does nothing.** Both branches of the check raise; the flag only changes the wording of the error. In-project reads are always allowed regardless of its value. Either implement it (block in-project reads when false) or drop the knob from `config.toml`, `config.py` and the docs. |
| 2.2 | `lumi/config.py:283` (`LLMConfig.model_of`) | Precedence quirk: an explicit `model = "gpt-4.1-mini"` in `config.toml` is treated as "unset" because it equals `OPENAI_DEFAULT_MODEL`, so `OPENAI_MODEL` from the environment wins. Harmless for every other value; surprising if someone pins the default deliberately. |
| 2.3 | `lumi/context.py` (`ContextWindow.observe`) | Reads `usage.get("prompt")`. `openai_compat._parse` maps `prompt_tokens → prompt`, so it is consistent today, but the key name is a private contract between two modules with no shared constant. A rename on either side silently disables calibration. |
| 2.4 | `lumi/agent.py:370` (`_turn`, iteration cap) | On the cap, `result.text` is a canned message that is **not** appended to `conv.messages`, so the conversation ends on a `role: "tool"` message and the model is never told it was cut off. Next turn starts from a half-finished tool exchange. |
| 2.5 | `lumi/agent.py` (`_evict_quietest`) | A chat with a pending approval blocks eviction (`if not candidates: return`), so `max_conversations` can be exceeded without limit if approvals are left dangling — no TTL/cleanup for stale `PendingAction`s. |
| 2.6 | `lumi/tools/context7.py:108` (`aclose`) | Docstring says "Called when the tool is garbage-collected", but there is no `__del__` and no shutdown hook, so the cached `httpx.AsyncClient`s are never closed. Harmless for a long-running bot, noisy under `pytest -W error::ResourceWarning`. |
| 2.7 | `lumi/tools/web/providers/mcp_provider.py` | Every `search`/`fetch` spawns the MCP server process, lists tools, then spawns it again for the call. Correct, but two process launches per query; a cached session per server would roughly halve latency. |
| 2.8 | `lumi/bot.py:746` (`_write_toml`) | Also: sections are emitted after top-level keys, and a key containing `.` would be written as a bare table path. Fine for the current whitelist use, fragile if anything else ever writes config through it. |

---

## 3. Minor improvements

- **`/config` memory-cap arithmetic** — `lumi/bot.py:237`:
  `(window - headroom) // 8 * 3` is labelled *"an eighth of the window"* but is
  3/8 of it, and it ignores `llm.chars_per_token`. The real rule is
  `ContextWindow.memory_limit()` = `budget // 8 * chars_per_token`. Print
  `agent.memory_limit()` instead of re-deriving it, so display and behaviour can
  never drift.
- **Shell timeout message** — `lumi/tools/shell.py:288` prints
  `self._timeout(None)` (the *configured* timeout) even when the model passed
  `timeout_seconds`. Print the timeout actually used (`_run` already has it).
- **`lumi doctor` omits Exa** — `lumi/doctor.py:195-196` checks Tavily and
  Firecrawl keys but never reports `EXA_API_KEY`, although Exa sits in the
  default `provider_order`. One more tuple entry.
- **`lumi doctor` leaks file handles** — `lumi/doctor.py:311` does
  `sum(1 for f in files for _ in f.open(...))` and never closes the handles.
  Use a `with` / `read_text` per file.
- **CLI `/memory` ignores the window share** — `lumi/__main__.py:161` calls
  `agent.memory.for_prompt()` with no limit, so it prints the whole file while
  the bot's `/memory` and the system prompt both apply
  `agent.memory_limit()`. Use the same limit for consistency.
- **`MemoryFile.load()` sets `_mtime` before creating the file** —
  `lumi/memory.py:83` stats a file that doesn't exist yet, writes it, and leaves
  `_mtime = 0.0`, so the very first `ensure_loaded()` always reloads. One extra
  read; move the `stat` after the write.
- **`History.new_session(chat_id)` ignores its argument** — `lumi/memory.py:540`.
  Harmless (the uuid makes ids unique), but the parameter implies per-chat
  scoping that isn't there.
- **`ArtifactStore._recent_order` allows duplicates** — `lumi/artifacts.py:281`
  re-registering the same file appends the key twice, so `recent()` can list it
  twice and the eviction pops early. Append only when the key is new.
- **`something.txt`** — a pasted crash log committed at the repo root (commit
  `4dc53c5 "somehing"`). It contains local paths (`/home/momoi/...`). Delete it
  and keep logs under `data/logs/` (already gitignored).

---

## 4. Test gaps

837 tests, all green, and they genuinely cover a lot — but every bug above
slipped through because the tests are shaped around the *happy path of each
unit* rather than the seams between units:

| Gap | Why the suite misses it |
|-|-|
| **Filters/routing** | No test sends a photo *with* a caption; `make_update` in `tests/test_bot.py` cannot even represent one. Routing bugs (1.3) are invisible without it. |
| **Multi-call replies** | Every scripted reply in the suite has `tool_calls` of length ≤ 1, or all-allowed calls, so the approval/multi-call seam (1.2) is never crossed. |
| **Long-lived application** | `send()`/`build()` construct a fresh `Application` per test, so anything that must survive *within* a process — the `owner_filter` snapshot (1.4) — looks fine. |
| **Real `context7._client_for`** | Patched out in `tests/test_context7.py:95`, which is precisely why 1.1 survived. |
| **Config round-trip** | Nothing re-parses `config.local.toml` after `_save_whitelist` writes it (1.7). |
| **`0` means no cap** | `max_read_chars = 0` (and the other documented zero-states) are never exercised through the tools themselves (1.5). |
| **Callback staleness** | No test makes `query.answer` raise `BadRequest`, though the codebase already special-cases `BadRequest` everywhere else. |

Suggested additions (one small file each): `tests/test_bot_routing.py` for the
caption/whitelist cases, and a handful of cases in `tests/test_agent.py` for the
multi-tool-call invariant. The invariant worth asserting everywhere is:

> *for every assistant message with `tool_calls` in `conv.messages`, there is
> exactly one `role: "tool"` message per `tool_call_id`.*

That single assertion would have caught 1.2.

---

## 5. Repo notes

- **Uncommitted work in the tree** (not mine, left untouched): 10 files,
  −84/+1 lines. It is a dead-code sweep — unused helpers removed
  (`PendingAction.to_tool_call`, `Artifact.to_dict`, `Agent.close`,
  `BotConfig.log_prefix`, `DEFAULT_CONNECT_TIMEOUT`, `artifact_caption`, …).
  `ruff` and the full suite pass on top of it. Worth committing on its own.
- `.env` is gitignored and untracked; the tracked `.env.example` contains only
  placeholders. `data/` and `workspace/` are ignored. No credentials in history
  by grep (`sk-`/`BOT_TOKEN=` hits are all docs/placeholders).
- CI (`.github/workflows/ci.yml`) runs `ruff check lumi tests` then
  `pytest -q` on 3.11/3.12/3.13, so a new test file is exercised immediately —
  and a *failing* repro file must not be left in `tests/`.

---

## 6. What looked solid

Worth knowing so a future reader does not "fix" these:

- **`lumi/context.py`** — the token budgeting, block grouping
  (`assistant+tool_calls` always move together), summary install/refresh and the
  four-stage fallback (elide → summarise → drop → shrink) are carefully
  reasoned and well tested. This is the strongest module in the repo.
- **`lumi/tools/safety.py`** — deny beats confirm, `ask_before_risky = false`
  upgrades confirm to deny, `effective_cwd` is judged after `cd`. The policy
  engine itself looks correct; I found no bypass in the classification order.
- **Path confinement** — `files._resolve` and `paths.is_within` use
  `resolve()` + parent checks, not string prefixes; `shell._resolve_cwd`
  handles absolute input correctly (`root / "/etc"` → `/etc` → rejected).
- **Secret scrubbing** — `shell._build_env` is an allowlist, so the
  `SECRET_PATTERN` is only used for the log line; nothing sneaks through by
  having an unexpected name.
- **Error containment** — `registry.invoke` catches everything except
  `NeedsApproval`; `bot.run_agent`/`run_tool`/`_resolve` all catch and render.
  A tool bug cannot take the bot down.
- **`lumi/llm/keypool.py` + `reasoning.py`** — duck-typed retry classification,
  cooldown parking, parameter step-down variants: well-designed and covered by
  `tests/test_llm.py`.

---

## 7. Suggested order of work

1. **1.1** context7 `self.settings` → one-word fix, currently a broken feature.
2. **1.2** tool-call/`role: "tool"` invariant → correctness + prevents an
   unrecoverable conversation.
3. **1.7** `config.local.toml` corruption → can stop the bot from booting.
4. **1.8** `query.answer` guard → small, and it is already in a shipped log.
5. **1.4** whitelist filter → either fix or document; users will hit this.
6. **1.3** photo caption + README, **1.5** `max_read_chars = 0`,
   **1.6** CLI approval output.
7. Turn each repro into a permanent regression test (see §4), then pick off §3.

For each fix: change the source, add the regression test, run
`./.venv/bin/ruff check lumi tests && ./.venv/bin/pytest -q`, and confirm the
baseline is still green before moving on.
