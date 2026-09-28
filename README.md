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
Move the folder to another machine and it still works. The bot doesn't read from or write to your home directory unless you deliberately configure it.

**Markdown is the configuration.** There is no prompt DSL and no schema to
learn. You paste PERSONALITY.md into the system prompt verbatim; edit it and
send `/reload`. `MEMORY.md` is a bullet list you can hand-edit, with a managed
region the bot appends to and `/forget` pops from.

**The safety layer is a file you can audit.** `lumi/tools/safety.py` contains one
screen of `(regex, tier, reason)` triples. `lumi shell "rm -rf /"` prints the
verdict and the reason, so you can check the policy without provoking it.

**Web access is pluggable three ways.** The Tavily SDK, the Firecrawl SDK, and a
generic MCP client that reads `.mcp.json`. They normalise to the same interface,
so adding a fourth is one file. The MCP path means a Tavily or Firecrawl MCP
server works with no code change at all — just add it to `.mcp.json`.

## Commands

| Command | What it does |
|-|-|
| `lumi run` | Start the Telegram bot |
| `lumi chat` | REPL against the same agent, no Telegram required |
| `lumi ask "..."` | One question, one answer, exits |
| `lumi shell "git status"` | Run one command through the safety engine |
| `lumi search "..."` | Web search from the terminal |
| `lumi memory` / `--add` / `--forget N` | Inspect and edit `MEMORY.md` |
| `lumi doctor` | Full environment check |
| `lumi config [--key llm.model]` | Print the resolved configuration |

In Telegram: `/help` `/run` `/search` `/fetch` `/memory` `/remember` `/forget`
`/personality` `/tools` `/reasoning` `/status` `/doctor` `/reload` `/reset`, plus
`/approve`, `/deny`, and the whitelist commands. Anything that is not a command
is just a message to the agent.

Send the bot a photo and it looks at it. The caption becomes your question; a
photo with no caption gets a default "what's in this image?". This needs a
vision-capable model — a text-only model will simply fail to use the image, so
point `llm.model` at a multimodal one.

In group chats the bot only answers when **mentioned** (`@Lumi_a_bot ...`),
**replied to**, or sent a slash command targeting it (`/help@Lumi_a_bot`). See
[Group chats](#group-chats) for the full rule and the `bot.group_reply_mode`
config knob.

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

Images sent in Telegram are passed through the standard OpenAI multimodal
content array, so any endpoint that accepts `image_url` parts works. Pick a
vision-capable model (`gpt-4o`, `claude-3.5+`, `gemini-2.*`, `llama-3.2-vision`,
`qwen2-vl`, …) or photo support will not work.

Resolution order, most specific wins:

1. `base_url` / `model` / `api_key_env` / `key_strategy` in `config.toml`
2. `LUMI__LLM__BASE_URL`, `LUMI__LLM__MODEL`, `LUMI__LLM__API_KEY_ENV`, `LUMI__LLM__KEY_STRATEGY`
3. `OPENAI_BASE_URL` (or `OPENAI_API_BASE`), `OPENAI_MODEL`, and the key named by `api_key_env`
4. OpenAI and `gpt-4.1-mini`

The reasoning settings (`reasoning`, `reasoning_effort`, `reasoning_tokens`,
`show_reasoning`) are Lumi's own, so they resolve from `config.toml` and
`LUMI__LLM__*` only — no bare environment variable. See
[Reasoning models](#reasoning-models).

Always check what actually took effect:

```bash
lumi config --key llm.base_url      # -> http://localhost:20128/v1
lumi config --key llm.model         # -> nemotron-3-nano-reasoning
lumi config --key llm.key_strategy  # -> fallback
lumi doctor | grep llm              # -> model via endpoint (from OPENAI_BASE_URL), keys: fallback
```

If a call fails, the error names the endpoint, model and key variable it used, so
a key rejected by the wrong host is obvious immediately. `temperature = null`
omits the parameter, which some reasoning models require.

### Reasoning models

A reasoning model thinks before it answers, and that breaks three conventions
the rest of the program relies on. All three are handled by one flag:

```toml
[llm]
reasoning = true
reasoning_effort = "medium"   # empty = "medium"; the value switches thinking on
reasoning_tokens = 2000       # headroom for the trace, on top of max_tokens
show_reasoning = true
```

`reasoning = true` does three things:

| | Why |
|-|-|
| drops `temperature` | these models reject any value but their own default, and a 400 on every turn is the usual symptom |
| moves the limit to `max_completion_tokens` | o-series and gpt-5 reject the old `max_tokens` name outright |
| adds `reasoning_tokens` to the ceiling | thinking is billed as output, so without headroom a long trace eats the answer |

If the endpoint rejects one of those parameters anyway, Lumi drops it and retries
with a plainer request rather than making you work out which combination your
provider wants. The same applies to `reasoning_effort`. Leaving `reasoning` on
for a model that does not reason is harmless — the extra parameters go unused.

`reasoning_effort` is `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or
`max`. Empty is not the same as `"none"`: empty sends `medium` while `reasoning`
is on, and `"none"` actively tells the model to skip thinking. Sending a value
matters more than which value: on some OpenAI-compatible endpoints (Ollama's,
for instance) `reasoning_effort` doubles as the thinking on/off switch, so a
model that ships thinking-off by default only ever thinks when a value is sent.
Ollama's own docs put it as: *"reasoning_effort and reasoning.effort control
model thinking."*

If a reasoning model 400s and you have not set the flag, the error says so and
names the line to add.

#### Seeing the thinking

When a model returns a trace, the chat gets one collapsed line above the answer,
in the order it actually happened — the model thought, then it spoke:

> 🧠 thought for 12s · 1,200 reasoning tokens · 2,140 chars — `[show thinking]`

Tapping the button expands the trace in place, and collapses it again. Nothing
is pre-typed into the message, so a chat scrolled back through stays readable
unless you ask for the detail. Traces expire after 30 minutes or 64 turns,
whichever comes first — an unopened one is not worth keeping.

`/reasoning` reports the current setup; `/reasoning off` (or `on`) hides the
trace for the running process. `lumi ask` prints the trace to **stderr**, so
`lumi ask "..." > answer.md` still captures nothing but the answer.

The trace is read from `reasoning_content`, `reasoning` or `thinking` depending
on the provider, and a model that inlines `<think>` tags instead of using a
field has them stripped out — otherwise raw XML ends up in the chat. It is never
written back into the conversation history.

### Several keys

`OPENAI_API_KEY` takes a comma-separated list, so one dead key or one exhausted
quota does not take the bot offline:

```bash
OPENAI_API_KEY=sk-...90, sk-...91, sk-...92
```

Commas, semicolons and newlines all separate, and quotes and stray whitespace
are stripped, so a long list can be written one key per line and a trailing
separator is not a broken key.

`llm.key_strategy` decides how they are spent:

| Strategy | Behaviour | Use it when |
|-|-|-|
| `fallback` (default) | First key, moving on only when a call fails in a way another key would survive — 401, 403, 429, 5xx, dropped connection. A 400 fails immediately. | Keys are backups for one account |
| `round_robin` | One call each, in turn, so quotas are shared evenly. | Keys live on separate accounts |
| `random` | A random key per call. Same balance as round-robin, no shared cursor. | Same, and you do not care about order |

A failed call retries the same request on the next key within the same turn, so
a 429 mid-conversation is invisible to you. A key that fails is parked for a
minute before it is retried, so a rate limit that clears itself recovers
without a restart. If every key is parked, the one that has been cooling the
longest is probed anyway, and if all of them are dead the error says how many
were tried. With a single key in the variable, every one of these is a no-op and
the behaviour is unchanged.

`CONTEXT7_API_KEY` accepts the same shape and the same three strategies via
`tools.context7.key_strategy`, so the `context7` tool enjoys the same
self-healing when one of a pool of keys is rate-limited.

The web providers do too: `TAVILY_API_KEY` and `FIRECRAWL_API_KEY` are both
pools under the hood (`tools.web.tavily.key_strategy`,
`tools.web.firecrawl.key_strategy`). Tavily additionally accepts a **keyless**
mode — leave the variable unset and the SDK falls back to its free tier
(lower rate limit; `search` and `extract` only). Firecrawl has no keyless
tier, so an unset env var means the provider is unavailable, not a fallback
option. Both share one parser
and one rotation policy — `lumi/util/keys.py` and `lumi/llm/keypool.py` — so
the rules above hold for every one of them.

`TAVILY_API_KEY` and `FIRECRAWL_API_KEY` take a single key each. A comma in
either is not a pool; it is a broken key.

Keys are read at startup, so a pool is picked up by a restart. `lumi doctor`
shows how many keys it found and which strategy is active.

## Tools

| Tool | Capability |
|-|-|
| `run_shell` | Shell commands, with the allow/confirm/deny engine |
| `files` | Read anywhere in the project, write only in `workspace/` |
| `memory` | Remember and recall; forgetting is human-only by design |
| `web` | Search and fetch, across three interchangeable providers |
| `context7` | Up-to-date library docs from https://context7.com; auto-enables on `CONTEXT7_API_KEY` |

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

## Group chats

In a private chat the bot replies to every owner message. In a group or
supergroup it follows `bot.group_reply_mode` (default `mention`):

- **`mention`** — the bot only responds when it is **mentioned**
  (`@Lumi_a_bot`), **replied to**, or sent a **slash command that targets it**
  (`/help@Lumi_a_bot`). Plain `/help` in a group is left alone because Telegram
  routes commands without `@botname` to whichever bot claims them first.
- **`always`** — replies to every owner message regardless of chat type.
- **`off`** — never replies in groups; private chats still work as normal.

The detection covers three signals:

1. **Slash command with `@botname`** — `cmd.split("@", 1)[1]` must equal the
   bot's username (case-insensitive).
2. **Reply to a message from the bot** — `reply.from_user.is_bot` and the
   usernames match.
3. **`@botname` mention in entities** — `MessageEntity.MENTION` with the
   bot's username, or `MessageEntity.TEXT_MENTION` pointing at the bot.

Add chat ids to `bot.always_reply_chats` to override the mode for one
specific group (e.g. a private "Lumi lab"):

```toml
[bot]
group_reply_mode = "mention"
always_reply_chats = ["-1001234567890"]
```

## Reaching api.telegram.org

If the bot dies at startup with `telegram.error.TimedOut` / `ConnectTimeout`,
the TCP connection to `api.telegram.org` failed. Three knobs under `[bot]`
address it:

- `proxy_url` — route the Bot API through a SOCKS5 or HTTP proxy (e.g.
  `socks5://127.0.0.1:9050`). The usual fix where Telegram is throttled or
  blocked.
- `connect_timeout` — TCP connect timeout in seconds (default 15). The stock
  5s can be too tight on a slow network.
- `bootstrap_retries` — extra startup attempts after a network failure
  (default 3, so a single hiccup delays boot instead of aborting it).

`lumi doctor` and the Telegram `/doctor` command report which settings are in
effect.

## Web providers

`tools.web.provider_order` is tried left to right; the first available provider
answers, and one that fails at call time falls through to the next.

```toml
[tools.web]
provider_order = ["tavily", "firecrawl", "mcp"]
```

- **tavily** — best for agentic search. LLM-ready snippets, optional bundled
  answer. **Works without an API key** — the SDK runs in its free tier
  (low rate limit; `search` and `extract` only) when `TAVILY_API_KEY` is
  unset. With one or more keys, a failed key is parked for a minute and the
  next one answers; `tools.web.tavily.key_strategy` picks
  `fallback` / `round_robin` / `random`.
- **firecrawl** — best for reading a specific page. Renders JavaScript,
  returns clean markdown. No keyless tier — a `FIRECRAWL_API_KEY` is
  required, but a comma-separated pool with the same three strategies
  (`tools.web.firecrawl.key_strategy`) keeps the bot up when one key is
  rate-limited.
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

## Context7 (library docs)

A separate tool from `web`. Where `web` finds pages on the open internet,
`context7` returns version-specific documentation and code examples for a
named library. The training data the model runs on is older than most
libraries; this is how it catches up.

The tool is **auto-validated**: the moment `CONTEXT7_API_KEY` is set in `.env`,
the tool appears in the model's tool list; if the variable is unset, the tool
is hidden from the system prompt and `available()` returns a clear reason. The
same `OPENAI_API_KEY` rotation pattern applies — a comma-separated pool with
one of `fallback` (default), `round_robin`, or `random`:

```bash
# .env
CONTEXT7_API_KEY=ctx7sk-...91, ctx7sk-...92
```

Get a key at https://context7.com/dashboard (keys start with `ctx7sk`). The
tool reaches the public Context7 HTTP API (`https://context7.com/api/v2`):

- `resolve_library_id` — turn a name like `react` or `tavily` into a
  Context7 library ID such as `/facebook/react`. Pass `query` to rank the
  candidates by relevance to the task.
- `query_docs` — given a Context7 ID and a question, return the best
  snippets. Always pass the ID from `resolve_library_id`, prefixed with `/`.

A key that returns 401/403/429/5xx is parked for a minute, the same way the
LLM client handles dead credentials. A 400 or 404 fails immediately — it is
not the key's fault.

Disable the tool explicitly by removing it from `tools.enabled` or by
setting `tools.context7.enabled = false`.

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
│   ├── files.py       path-confined read/write
│   ├── memory_tool.py
│   ├── context7.py    version-specific library docs
│   └── web/           tool + providers/{tavily,firecrawl,mcp}
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
