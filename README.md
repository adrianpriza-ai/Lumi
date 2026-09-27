# Lumi

A Telegram AI agent that is a folder, not an install.

Clone it, drop a token in `.env`, run it. `PERSONALITY.md` is the voice,
`MEMORY.md` is the memory, `config.toml` is the behaviour, and every tool it can
reach is a file you can read before you trust it.

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

**Everything is relative.** Paths resolve from the directory containing
`pyproject.toml`. State goes in `./data`, the shell runs in `./workspace`, and
the bot pins `HOME` to the project root so a stray `~/.something` cannot appear.
Move the folder to another machine and it still works. Nothing is read from or
written to your home directory unless you deliberately configure it.

**Markdown is the configuration.** There is no prompt DSL and no schema to
learn. `PERSONALITY.md` is pasted into the system prompt verbatim; edit it and
send `/reload`. `MEMORY.md` is a bullet list you can hand-edit, with a managed
region the bot appends to and `/forget` pops from.

**The safety layer is a file you can audit.** `lumi/tools/safety.py` is one
screen of `(regex, tier, reason)` triples. `lumi shell "rm -rf /"` prints the
verdict and the reason, so you can check the policy without provoking it.

**Web access is pluggable three ways.** The Tavily SDK, the Firecrawl SDK, and a
generic MCP client that reads `.mcp.json`. They normalise to the same interface,
so adding a fourth is one file. The MCP path means a Tavily or Firecrawl MCP
server works with no code change at all — just add it to `.mcp.json`.

## Commands

| Command | What it does |
| --- | --- |
| `lumi run` | Start the Telegram bot |
| `lumi chat` | REPL against the same agent, no Telegram required |
| `lumi ask "..."` | One question, one answer, exits |
| `lumi shell "git status"` | Run one command through the safety engine |
| `lumi search "..."` | Web search from the terminal |
| `lumi memory` / `--add` / `--forget N` | Inspect and edit `MEMORY.md` |
| `lumi doctor` | Full environment check |
| `lumi config [--key llm.model]` | Print the resolved configuration |

In Telegram: `/help` `/run` `/search` `/fetch` `/memory` `/remember` `/forget`
`/personality` `/tools` `/status` `/doctor` `/reload` `/reset`, plus `/approve`
and `/deny`. Anything that is not a command is just a message to the agent.

## Configuration

`config.toml` is committed and holds no secrets. `.env` holds the secrets and is
gitignored. Every config value can also be set as an environment variable:

```bash
LUMI__LLM__MODEL=qwen3:8b
LUMI__TOOLS__SHELL__TIMEOUT_SECONDS=10
LUMI__TOOLS__WEB__PROVIDER_ORDER=mcp,tavily
```

Precedence: dataclass defaults < `config.toml` < `config.local.toml` (gitignored)
< `LUMI__*` environment variables.

### The LLM

Any OpenAI-compatible endpoint: OpenAI, Ollama, LM Studio, vLLM, llama.cpp,
Groq, OpenRouter, Together, Nous. Put the key in `.env`, and the endpoint
wherever suits you:

```bash
# .env — per machine. This is the usual place.
OPENAI_API_KEY=sk-or-whatever
OPENAI_BASE_URL=http://localhost:20128/v1
OPENAI_MODEL=nemotron-3-nano-reasoning
```

```toml
# config.toml — per project, overrides the environment
[llm]
base_url = "https://openrouter.ai/api/v1"
model = "anthropic/claude-sonnet-4.5"
api_key_env = "OPENROUTER_API_KEY"   # this provider's own key variable
```

Resolution order, most specific wins:

1. `base_url` / `model` / `api_key_env` in `config.toml`
2. `LUMI__LLM__BASE_URL`, `LUMI__LLM__MODEL`, `LUMI__LLM__API_KEY_ENV`
3. `OPENAI_BASE_URL` (or `OPENAI_API_BASE`), `OPENAI_MODEL`, and the key named by `api_key_env`
4. OpenAI and `gpt-4.1-mini`

Always check what actually took effect:

```bash
lumi config --key llm.base_url   # -> http://localhost:20128/v1
lumi config --key llm.model      # -> nemotron-3-nano-reasoning
lumi doctor | grep llm           # -> model via endpoint (from OPENAI_BASE_URL)
```

If a call fails, the error names the endpoint, model and key variable it used, so
a key rejected by the wrong host is obvious immediately. `temperature = null`
omits the parameter, which some reasoning models require.

## Tools

| Tool | Capability |
| --- | --- |
| `run_shell` | Shell commands, with the allow/confirm/deny engine |
| `files` | Read anywhere in the project, write only in `workspace/` |
| `memory` | Remember and recall; forgetting is human-only by design |
| `web` | Search and fetch, across three interchangeable providers |

### Shell safety

Three tiers, resolved in order, strictest wins:

- **deny** — never runs, not even with approval. `rm -rf /`, `sudo`, piping a
  download into a shell, writing to `/dev/sda`, touching `.bashrc` or `.ssh`,
  power state changes, fork bombs.
- **confirm** — the owner gets a Confirm/Cancel button in the chat. Anything that
  deletes, moves, overwrites, changes permissions, installs, commits, or
  redirects into a file.
- **allow** — everything else runs immediately.

On top of that: writes outside the project are blocked, `HOME` is pinned to the
project, the child environment is scrubbed of anything matching
`*KEY*|*TOKEN*|*SECRET*|*PASSWORD*` (so a command cannot exfiltrate your API
keys), every command runs in its own process group and is killed on timeout, and
output is capped.

Set `tools.shell.ask_before_risky = false` to refuse the confirm tier outright,
or add your own patterns:

```toml
[tools.shell]
extra_deny    = ["\\bnpm\\s+publish\\b"]
extra_confirm = ["\\bterraform\\b"]
```

### Threat model

This is a guardrail, not a sandbox. Shell is Turing-complete; a determined
adversary who reaches the model's prompt can find a phrasing the regexes miss.
What the engine buys you is that the *common* destructive commands — the ones a
hallucination or a careless prompt produces — cannot run by accident.

If you need a hard boundary, run Lumi in a container or under a dedicated
unprivileged user. Owner gating is by Telegram user id and is enforced twice: as
a handler filter, and as an explicit check in every privileged handler.

## Web providers

`tools.web.provider_order` is tried left to right; the first available provider
answers, and one that fails at call time falls through to the next.

```toml
[tools.web]
provider_order = ["tavily", "firecrawl", "mcp"]
```

- **tavily** — best for agentic search. LLM-ready snippets, optional bundled
  answer, keyless tier with a low rate limit.
- **firecrawl** — best for reading a specific page. Renders JavaScript, returns
  clean markdown.
- **mcp** — reads `.mcp.json` from the project root, expands `${VAR}` from the
  environment, and speaks MCP over streamable HTTP or stdio. It discovers each
  server's tools and reads their input schemas, so it works with a server whose
  parameter is called `q` instead of `query`.

```json
{
  "mcpServers": {
    "tavily-remote": { "type": "remote", "url": "${TAVILY_MCP_URL}" },
    "firecrawl-remote": { "type": "remote", "url": "${FIRECRAWL_MCP_URL}" }
  }
}
```

```bash
# .env
TAVILY_MCP_URL=https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}
FIRECRAWL_MCP_URL=https://mcp.firecrawl.dev/${FIRECRAWL_API_KEY}/v2/mcp
```

Local servers work too:

```json
{ "mcpServers": { "my-search": { "type": "local", "command": ["npx", "-y", "my-mcp-server"] } } }
```

## Layout

```
lumi/
├── config.py          config loading, dataclasses, env overrides
├── paths.py           every path, resolved from the project root
├── personality.py     PERSONALITY.md -> system prompt
├── memory.py          MEMORY.md + JSONL transcripts
├── agent.py           the tool-calling loop, including approvals
├── bot.py             Telegram handlers (all the edge cases live here)
├── doctor.py          environment diagnostics
├── llm/               client contract + OpenAI-compatible adapter
├── tools/
│   ├── base.py        Tool ABC, ToolResult, NeedsApproval
│   ├── registry.py    dispatch, error containment
│   ├── safety.py      the command policy — read this one
│   ├── shell.py       execution: scrubbed env, timeouts, output caps
│   ├── files.py       path-confined read/write
│   ├── memory_tool.py
│   └── web/           tool + providers/{tavily,firecrawl,mcp}
└── __main__.py        the CLI
```

Runtime state, all gitignored:

```
data/
├── history/<chat_id>.jsonl   append-only transcripts
├── logs/lumi.log
└── cache/
workspace/                    the shell's working directory
```

## Adding a tool

```python
# lumi/tools/weather.py
from .base import Tool, ToolContext, ToolResult

class WeatherTool(Tool):
    name = "weather"
    description = "Look up the current weather for a city. Use when asked about weather."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
        "additionalProperties": False,
    }

    async def invoke(self, arguments, ctx: ToolContext) -> ToolResult:
        city = arguments["city"]
        return ToolResult(text=f"It is 18°C and raining in {city}.", summary=f"weather for {city}")
```

Register it in `lumi/tools/__init__.py:build_registry`. It is now in the model's
tool list, and `/tools` will show it. To make it ask permission first, raise
`NeedsApproval` from `invoke` when `ctx.confirmed` is false — the agent loop
already turns that into a Confirm/Cancel pair.

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

MIT. See `LICENSE`.
