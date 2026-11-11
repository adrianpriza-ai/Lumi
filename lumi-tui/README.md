# lumi-tui

A Codex-CLI-style terminal UI for Lumi, written in Rust. It is a **frontend**,
not a second agent: the whole brain — model, tools, safety engine, memory,
context window — is the Python `lumi` package, served over JSON lines by a
`lumi tui-serve` child process that this binary spawns and owns. Rust keeps only
the terminal: raw mode, the composer, scrollback, markdown rendering, the
approval modal and the spinner.

Users do not run this binary. `lumi` with no arguments (or `lumi tui`) execs it,
the way Vibe's `vibe` command runs its bundled `vibe-rs`.

```
 ◆ lumi  chat                                    gpt-4.1-mini · api.openai.com/v1
● you
what changed in rust this week
✓ web (what changed in rust this week) · done
● lumi
This week in Rust: ...

 enter send · shift+enter newline · ↑ history · /help commands        search: tavily · 1 turns
╭ you ────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

## Build and run

Rust 1.75+ (any recent stable) and a working Python install of this repo:

From the repo root:

```bash
cargo build --release --manifest-path lumi-tui/Cargo.toml   # once
lumi                                                        # opens the UI
```

The TUI starts `lumi tui-serve` itself, so there is no server to run and no port
to configure. Quitting the TUI kills the child. Running
`./lumi-tui/target/release/lumi-tui` directly still works while you are working
on the Rust side, and `$LUMI_TUI` points the launcher at a build elsewhere.

`pip install .` compiles the crate during the wheel build and bundles the
binary at `lumi/_bin/lumi-tui` inside the platform-tagged wheel, so an installed
`lumi` needs no second build step — the same shape as Vibe bundling `vibe-rs`.
Editable installs skip that copy and use the checkout's `target/release/` build.

## Configuration

The TUI parses **no** configuration of its own. Everything — `.env`,
`config.toml`, `LUMI__*` overrides, `PERSONALITY.md`, `data/memory/cli/MEMORY.md`
— is read by the Python backend, exactly as `lumi chat` reads it. The TUI walks
up from your current directory looking for `config.toml` only to decide the
child's working directory, so the backend resolves the same project you
launched from. The header (`model`, `host`, `search`) is filled from the
backend's `ready` event, not from local parsing.

### Which backend is started

In order:

1. `$LUMI_BACKEND` if set — treated as a command line, with `tui-serve`
   appended (the dev escape hatch: `LUMI_BACKEND="uv run lumi"`,
   `LUMI_BACKEND="/path/to/.venv/bin/lumi"`).
2. `<root>/.venv/bin/lumi` if it exists.
3. `lumi` on `PATH`.
4. `python3 -m lumi`.

The child's **stderr** goes to `data/logs/lumi-tui-backend.log`; its **stdout**
is the protocol and nothing else.

## What it does

- **The same agent as `lumi chat` and the Telegram bot** — all 13 tools
  (`shell`, `files`, `read_file`, `list_files`, `grep`, `memory`,
  `more_context`, `web`, `mcp`, `context7`, `ocr`, `send`, `skills`), the
  context window with compaction, the key pool and retry ladder, and the
  scoped `cli` memory are the Python implementations, not copies.
- **The safety engine applies.** A `run_shell` call is judged by
  `lumi/tools/safety.py` before the modal even appears; a hard deny is a hard
  deny, not a prompt. The TUI cannot be used to bypass the policy.
- **Approvals** — a modal shows the exact command/preview and why it needs
  approval. `y` runs it, `n`/`esc` declines. Several pending actions queue and
  are answered one at a time by id.
- **Cancel** — `esc` (or `ctrl+c`) aborts the running turn. The backend kills
  the in-flight tool process group and seals any dangling tool calls, so the
  next turn replays cleanly instead of 400ing until `/reset`.
- **Multi-line input** — `shift+enter` opens a new line (via the Kitty
  keyboard protocol where supported); `ctrl+j` and `alt+enter` work
  everywhere. `enter` sends.
- **Scrollback** — `pgup`/`pgdn`, mouse wheel, `alt+↑/↓`; the view pins to the
  bottom until you scroll up.
- **Editing** — input history on `↑/↓` at the first line, bracketed paste,
  emacs-ish `home`/`end`, multi-line cursor movement.
- **Step traces** — thinking and tool calls between two messages fold into
  one line (`▸ 3 steps · thinking×2, web_search`), with the newest thought
  dimmed alongside it while collapsed. `ctrl+t` (or `/trace`) opens every
  trace in place and closes it again; a lone tool call keeps its own exec
  line, since there is nothing to fold.
- **Slash commands** — `/help`, `/search <query>`, `/reasoning on|off`,
  `/trace`, `/tools`, `/context`, `/memory`, `/reset`, `/quit`. `/model` is
  not a frontend switch: it points you at `config.toml`, because the backend
  would contradict a local override on the next command.

No token streaming yet: a turn emits live *steps* (thinking, each tool start)
and then the whole answer. That is the same behaviour as `lumi chat`; streaming
is a planned follow-up on the Python LLM client.

## Keys

| key | action |
| --- | --- |
| `enter` | send |
| `shift+enter` / `ctrl+j` / `alt+enter` | new line |
| `↑` / `↓` | cursor between lines; history at the edge |
| `pgup` / `pgdn` / wheel | scrollback |
| `ctrl+t` | expand / collapse the thinking + tool traces |
| `y` / `n` | answer the approval modal |
| `esc` | decline approval / cancel turn / clear composer |
| `ctrl+c` | cancel turn, then quit when idle and empty |

## Layout

- `src/main.rs` — terminal setup, key routing, the UI loop
- `src/backend.rs` — spawns `lumi tui`, speaks the JSON-lines protocol, maps
  backend events onto the UI's `AgentEvent` enum, translates slash commands
- `src/app.rs` — transcript blocks, scroll state, the multi-line composer
- `src/ui.rs` — frame drawing (header, transcript, status, composer, approval)
- `src/markdown.rs` — markdown → styled lines (fences, headings, bullets,
  inline code/bold) with unicode-aware wrapping

The protocol schema is the doc block at the top of `lumi/tui.py`.

## Tests

```bash
cargo test
```

Covers command translation, protocol event mapping (turn sequence, tool
open/close, approval queue and de-duplication, `turn_done` busy semantics),
malformed/unknown events, and the composer's multi-line behaviour
(shift+enter, ctrl+j, paste, backspace join, history recall).

The backend itself is tested on the Python side, with no Rust involved:

```bash
printf '{"cmd":"submit","text":"hi"}\n' | lumi tui-serve
```
