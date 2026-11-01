# lumi-tui

A Codex-CLI-style terminal UI for Lumi, written in Rust: streaming chat with
any OpenAI-compatible endpoint, agentic tool calls, web search through Tavily
and Firecrawl, and an approval prompt before anything runs on your machine.

```
 ◆ lumi  chat                                    gpt-4.1-mini · api.openai.com/v1
● you
what changed in rust this week
✓ web_search(what changed in rust this week) · 5 hits · 0.8s
● lumi
This week in Rust: ...

 enter send · shift+enter newline · ↑ history · /help commands        search: tavily · 1 turns
╭ you ────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```

## Build

Rust 1.75+ (any recent stable):

```bash
cd lumi-tui
cargo build --release
./target/release/lumi-tui
```

## Configuration

It reads the same things `lumi chat` does, from the project root (walks up
from your current directory looking for `config.toml`):

- `.env` — `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
  `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`
- `config.toml` — `[llm] base_url` / `model` / `api_key_env`
- `LUMI__LLM__BASE_URL` / `LUMI__LLM__MODEL` / `LUMI__LLM__MAX_TOOL_ITERATIONS`
  override the file, same rule as the Python side
- `PERSONALITY.md` and `data/memory/cli/MEMORY.md` are folded into the system
  prompt when present, so the TUI speaks with the same voice and remembers the
  same cli-scope facts as `lumi chat`

Search endpoints can be redirected with `TAVILY_API_URL` and
`FIRECRAWL_API_URL` (proxies, gateways, tests).

## What it does

- **Streaming** — tokens render as they arrive, from any OpenAI-compatible
  server (OpenAI, Ollama, LM Studio, vLLM, Groq, OpenRouter).
- **Tools** — the model can call:
  - `web_search` — Tavily REST, falling back to Firecrawl search
  - `fetch_page` — Firecrawl scrape to markdown
  - `shell` — runs `sh -c` in the project root, **after you approve** a modal
    showing the exact command (`y` to run, `n`/`esc` to decline)
  - `read_file` — project-root-confined text reads
- **Multi-line input** — `shift+enter` opens a new line (via the Kitty
  keyboard protocol where supported); `ctrl+j` and `alt+enter` work
  everywhere. `enter` sends.
- **Scrollback** — `pgup`/`pgdn`, mouse wheel, `alt+↑/↓`; the view pins to the
  bottom until you scroll up.
- **Editing** — input history on `↑/↓` at the first line, bracketed paste,
  emacs-ish `home`/`end`, multi-line cursor movement.
- **Slash commands** — `/help`, `/search <query>`, `/model [name]`,
  `/reasoning on|off`, `/tools`, `/reset`, `/quit`.
- **Cancel** — `esc` aborts the running turn; `ctrl+c` cancels, clears, and
  only quits on an empty composer.

## Keys

| key | action |
| --- | --- |
| `enter` | send |
| `shift+enter` / `ctrl+j` / `alt+enter` | new line |
| `↑` / `↓` | cursor between lines; history at the edge |
| `pgup` / `pgdn` / wheel | scrollback |
| `esc` | cancel turn / clear composer |
| `ctrl+c` | cancel turn, then quit when idle and empty |

## Layout

- `src/main.rs` — terminal setup, key routing, the UI loop
- `src/app.rs` — transcript blocks, scroll state, the multi-line composer
- `src/ui.rs` — frame drawing (header, transcript, status, composer, approval)
- `src/markdown.rs` — markdown → styled lines (fences, headings, bullets,
  inline code/bold) with unicode-aware wrapping
- `src/agent.rs` — the turn loop: stream → tool calls → approve → repeat
- `src/llm.rs` — OpenAI-compatible SSE client with tool-call assembly
- `src/search.rs` — Tavily + Firecrawl REST clients
- `src/config.rs` — `.env` / `config.toml` / `LUMI__*` resolution

## Tests

```bash
cargo test
```

Covers key-pool parsing, path escape refusal, and the composer's multi-line
behaviour (shift+enter, ctrl+j, paste, backspace join, history recall).
