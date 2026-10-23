# Lumi

A Telegram AI agent that is a folder, not an install.

Clone it, drop a token in `.env`, run it. `PERSONALITY.md` is the voice, every
chat gets its own markdown memory under `data/memory/`, `config.toml` is the
behaviour, and every tool it can reach is a file you can read before you trust
it.

```bash
git clone <this repo> lumi && cd lumi
cp .env.example .env          # fill in TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID

# plain venv — no extra tooling needed
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -e ".[dev]"

./.venv/bin/lumi doctor       # tells you exactly what is missing
./.venv/bin/lumi chat         # talk to it in your terminal, no token needed
./.venv/bin/lumi run          # start the Telegram bot
```

If you prefer `uv`, it does the same in one step:

```bash
uv sync
uv run lumi doctor
```

Either way `.venv/` is gitignored, so a fresh clone starts empty — that is
expected, and it is why the two steps above are always both needed.

## Why it is built this way

**Everything is relative.** Paths resolve from the directory containing `pyproject.toml`. State goes in `./data`, the shell runs in `./workspace`, and the bot pins `HOME` to the project root so a stray `~/.something` cannot appear. Move the folder to another machine and it still works. The bot does not read or write your home directory unless you deliberately configure it.

**Markdown is the configuration.** There is no prompt DSL and no schema to learn. You paste PERSONALITY.md into the system prompt verbatim; edit it and send `/reload`. It is the *voice* — how the bot talks. What it is allowed to do (the boundaries, the approval rules) is hardcoded in `lumi/agent.py`, so editing the voice file cannot quietly turn the safety rules off. Memory is also markdown: each chat gets its own `data/memory/<chat_id>/MEMORY.md`, a bullet list you can hand-edit, with a managed region the bot appends to and `/forget` pops from. A private chat's memory belongs to that user and a group's to the group — nothing learned in one chat is visible in another.

**The safety layer is a file you can audit.** `lumi/tools/safety.py` is one screen of `(regex, tier, reason)` triples. `lumi shell "rm -rf /"` prints the verdict and the reason, so you can check the policy without provoking it.

**Web access is pluggable five ways.** The Firecrawl SDK, the Exa SDK, the Tavily SDK, You.com's keyless MCP search, and a generic MCP client that reads `.mcp.json` (plus `~/.agents/mcp.json`, for servers shared across every project). They normalise to the same interface, so adding another is one file. The MCP path means a Tavily, Firecrawl, or Exa MCP server works with no code change at all — just add it to `.mcp.json`.

## Commands

| Command | What it does |
|-|-|
| `lumi run` | Start the Telegram bot |
| `lumi chat` | REPL against the same agent, no Telegram required |
| `lumi ask "..."` | One question, one answer, exits |
| `lumi shell "git status"` | Run one command through the safety engine |
| `lumi search "..."` | Web search from the terminal |
| `lumi memory` / `--add` / `--forget N` / `--scope <chat_id>` | Inspect and edit a chat's memory (default: yours) |
| `lumi doctor` | Full environment check |
| `lumi config [--key llm.model]` | Print the resolved configuration |

In Telegram: `/start` (the help), `/memory`, `/remember`, `/forget`, `/personality`, `/reasoning`, `/context`, `/status`, `/doctor`, `/reload`, `/reset`, plus `/approve` and `/deny`. Owner only: `/own_help` (lists `/config`, `/env`, and the whitelist commands). Anything else is just a message to the agent.

Send the bot a photo and it looks at it. The caption becomes your question; a
photo with no caption gets a default "what's in this image?". Images go to `llm.vision_model` when one is set, so `llm.model` can stay text-only. Unset, everything goes to `llm.model`, which then has to be vision-capable. See [CONFIGURATION.md](CONFIGURATION.md#the-llm) for the multimodal note and [Configuration](CONFIGURATION.md) for how the model is resolved.

Files work in both directions. Send the bot a document and it is stored under `workspace/uploads/` and handed to the model with its path — read it, summarise it, run code against it. Ask the model to produce a file (a report, a CSV, a generated image) and it comes back as a downloadable document. See [File delivery](TOOLS.md#file-delivery-documents) for the harness and its limits.

In group chats the bot only answers when **mentioned** (`@Lumi_a_bot ...`), **replied to**, or sent a slash command targeting it (`/start@Lumi_a_bot`). See [Group chats](TELEGRAM.md#group-chats) for the full rule and the `bot.group_reply_mode` config knob. Each chat also has its own long-term memory, so what the bot learns in a group stays in that group.

## Documentation

| Document | Covers |
|-|-|
| [CONFIGURATION.md](CONFIGURATION.md) | `config.toml` and precedence, picking an endpoint, reasoning models and thinking traces, key pools and rotation |
| [TOOLS.md](TOOLS.md) | The tools, shell safety and its threat model, web providers (Firecrawl / Exa / Tavily / MCP), Context7, mid-turn messages, and how to add a tool |
| [TELEGRAM.md](TELEGRAM.md) | Group chats and when the bot answers, and getting past `api.telegram.org` timeouts |

Start with `lumi doctor`: it names whichever of these is misconfigured.
## Layout

```
lumi/
├── config.py          config loading, dataclasses, env overrides
├── paths.py           every path, resolved from the project root
├── personality.py     PERSONALITY.md -> system prompt
├── memory.py          per-chat MEMORY.md files + JSONL transcripts (read tail-first)
├── context.py         the context window: what fits, what is condensed
├── artifacts.py       the file harness — outbox for documents, intake for uploads
├── agent.py           the tool-calling loop, including approvals
├── bot.py             Telegram handlers (all the edge cases live here)
├── doctor.py          environment diagnostics
├── llm/
│   ├── base.py        LLMClient contract, LLMReply, ToolCall
│   ├── keypool.py     key rotation, failure classification, masking
│   ├── reasoning.py   thinking traces, token ceilings, param negotiation
│   └── openai_compat.py  every OpenAI-compatible endpoint
├── tools/
│   ├── base.py        Tool ABC, ToolResult, NeedsApproval
│   ├── registry.py    dispatch, error containment
│   ├── safety.py      the command policy — read this one
│   ├── shell.py       execution: scrubbed env, timeouts, output caps
│   ├── files.py       path-confined read/write/edit
│   ├── explore.py     read_file / list_files / grep — read-only exploration
│   ├── memory_tool.py
│   ├── context7.py    version-specific library docs
│   ├── send.py        mid-turn owner messages
│   └── web/           tool + providers/{firecrawl,exa,tavily,you,mcp}
├── util/
│   ├── keys.py        comma-separated key parsing
│   ├── log.py         logging setup
│   └── text.py        truncation, chunking, error formatting
└── __main__.py        the CLI
```

Runtime state, all gitignored:

```
data/
├── history/<chat_id>.jsonl   append-only transcripts
├── memory/<chat_id>/MEMORY.md  one long-term memory file per chat
├── logs/lumi.log
├── outbox/                   staged copies of files sent to the chat
└── cache/
workspace/
├── uploads/<chat_id>/        documents received from the chat
└── ...                       everything else the shell and model produce
```

## Development

```bash
./.venv/bin/pytest
./.venv/bin/ruff check lumi tests
```

## Want `lumi` on your PATH

`./.venv/bin/lumi` always works. To get a bare `lumi` command in any directory,
either activate the venv per shell:

```bash
source .venv/bin/activate
```

or install it once with `pipx`, which gives you its own environment and does not
touch the project's `.venv`:

```bash
pipx install .
lumi doctor
```

Note that a `pipx` install is a *copy*, not an editable link: changes to the
source are not picked up until you reinstall with `pipx install --force .`. For
development, prefer the editable venv above.

## Licence

MIT. See [LICENSE](LICENSE).
