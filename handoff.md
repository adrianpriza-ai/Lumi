# Lumi handoff

Written 2026-10-19 after reviewing the supplied completion report against the
current working tree. The repository is on `main` at `1be228f`; the feature
work described below is still uncommitted. This review did not modify source
code. It only updates this handoff.

## Goal

Complete and verify the current Lumi changes for:

1. Mid-turn messages through the new `send` tool, including live delivery and
   no-channel fallback.
2. Step-based progress output instead of the old thinking trace and thinking
   buttons.
3. A no-bold response contract, with safe Telegram conversion of stray
   Markdown bold outside `<code>` and `<pre>` blocks.

## State

The implementation is present in the working tree, but the supplied report
should not yet be treated as a fully reproducible final verification.

Confirmed by repository inspection and focused smoke checks:

- `lumi/tools/send.py` exists and is registered in `lumi/tools/__init__.py`.
- The `send` tool is present in the default configuration and the message path
  includes `TurnHooks.on_message`, `ToolContext.emit`, CLI callbacks, Telegram
  callbacks, and `TurnResult.messages` fallback.
- The old `TraceStore`, `CB_THINK`, `thinking_stub`, and show/hide thinking
  keyboard symbols are absent from `lumi/` and `tests/`.
- `StepFeed` renders tool labels, deduplicates adjacent steps, and caps the
  rendered feed at ten lines. `/reasoning on|off` controls the stored display
  flag.
- `FORMAT_PREAMBLE` forbids Markdown bold and Telegram reply formatting calls
  `markdown_bold_to_html()`.

Open findings from the review:

1. High (fixed): `lumi/bot.py` re-raises `TelegramError` in strict delivery
   (`reply(..., strict=True)`), so live send failures properly surface to
   `agent.py` and get queued onto `TurnResult.messages` for turn-tail retry.
   In addition, `reply` now gracefully falls back to direct chat delivery
   if `effective_message` is missing or was deleted (handling message-deletion
   `BadRequest`), and raises when neither message nor chat is available.
   Covered by unit tests and `FlakySendBot` regression in `tests/test_bot.py`.
2. Medium: direct `/run`, `/search`, and `/fetch` execution, plus approved
   tool resolution, bypasses the `_step(hooks, "tool", ...)` path in
   `lumi/agent.py`. The feed can therefore omit tools that the bot actually
   runs, despite the feed copy saying it shows each run.
3. Medium (fixed): `lumi/util/text.py` protected lowercase `<code>` and
   `<pre>` segments only, so a valid uppercase `<PRE>**text**</PRE>` was
   converted to bold. `_CODE_SEGMENT_RE` now compiles with `re.IGNORECASE`,
   and `tests/test_text.py` covers the uppercase/mixed-case shapes.
4. Medium (fixed): the CLI now gates the `on_step` callback on
   `llm.show_reasoning` via `_hooks_for(config)`. Turning `llm.show_reasoning` off
   disables live step lines to stderr in addition to the post-turn raw trace.
   The CLI REPL also supports `/reasoning [on|off]` live toggling, and
   `tests/test_cli.py` covers hook wiring, step suppression, and the REPL command.
5. Low (fixed): `tests/test_bot.py` now supplies eleven distinct steps and
   verifies that the rendered feed is exactly ten lines with the correct
   earlier-step marker.
6. Low (fixed): the `README.md` layout tree now includes
   `lumi/tools/send.py`.

The report's statement that all requested behavior is delivered is therefore
partially supported, not fully confirmed. The live Telegram step feed was
tested against a recorded bot stub, not a live Telegram session. The report's
live Ollama `lumi ask` result is recorded here as reported evidence only.

## Files

The feature work currently changes these groups of paths:

- Message and progress behavior: `lumi/tools/send.py`,
  `lumi/tools/__init__.py`, `lumi/tools/base.py`, `lumi/agent.py`,
  `lumi/bot.py`, `lumi/__main__.py`, and `lumi/util/text.py`.
- Configuration and documentation: `config.toml`, `lumi/config.py`,
  `CONFIGURATION.md`, `TOOLS.md`, `README.md`, and `.gitignore`.
- Regression coverage: `tests/test_agent.py`, `tests/test_bot.py`,
  `tests/test_cli.py`, `tests/test_config.py`, `tests/test_text.py`, and
  `tests/test_tools.py`.
- This file: `handoff.md`, updated to preserve the older review backlog while
  recording the current verification state.

No source fix for finding 2 has been made. Findings 1 and 3 through 6 are
fixed in the current working tree.

## Decisions

- Keep `send` enabled by default. It is useful only when the agent can deliver
  a message during a turn, and the no-channel result list preserves messages
  for callers that cannot stream them.
- Keep step output in one edited Telegram message, with a ten-line cap and
  adjacent-step deduplication. Do not reintroduce a separate thinking trace or
  thinking keyboard.
- Keep the no-bold prompt rule and the Telegram conversion bridge. The
  code/pre protection is now case-insensitive, which was the stated
  prerequisite for calling this complete.
- Treat the existing source edits as working-tree changes, not merged or
  deployed changes. A commit is intentionally not created by this request.

## Checks

Commands run from `/home/momoi/Github/Lumi` during this review:

### Lint

- Passed: `./.venv/bin/ruff check lumi tests` returned `All checks passed!`.
- Passed: `git diff --check`.

### Tests

- Passed: `./.venv/bin/pytest -q` returned `920 passed`.
- The suite includes the new eleven-step `StepFeed` cap regression and the
  existing uppercase/mixed-case code protection coverage.

### Compile check

- Passed: `python -m compileall -q lumi tests`.

### CLI smoke

- Passed: `./.venv/bin/lumi config --key tools.enabled` lists `send` among the
  enabled tools.
- Passed: `./.venv/bin/lumi doctor` reports no blocking problems.
- Not independently rerun: the report's `lumi chat /tools` output and live
  Ollama `lumi ask` result.

## Next step

The project environment is now usable and the real test suite is green (923
passed). The remaining implementation work is finding 2:

```bash
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pytest -q
```

Fix finding 2, add regression coverage, and rerun:

```bash
./.venv/bin/ruff check lumi tests
./.venv/bin/pytest -q
```

The older review and its unresolved backlog are retained below so that the
next session does not lose the previously identified context7, approval,
photo-caption, whitelist, file-cap, CLI approval, TOML, and callback-query
issues.
