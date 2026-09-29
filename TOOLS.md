# Tools

[← Back to README](README.md) · [CONFIGURATION.md](CONFIGURATION.md) · [TELEGRAM.md](TELEGRAM.md)

| Tool | Capability |
|-|-|
| `run_shell` | Shell commands, with the allow/confirm/deny engine |
| `files` | Read anywhere in the project, write only in `workspace/`, send files to the chat |
| `memory` | Remember and recall; forgetting is human-only by design |
| `web` | Search and fetch, across three interchangeable providers |
| `context7` | Up-to-date library docs from https://context7.com; auto-enables on `CONTEXT7_API_KEY` |

## File delivery (documents)

The bot exchanges files with the chat through one small harness
(`lumi/artifacts.py`). Tools never talk to Telegram: a tool that produces a file
registers it with the harness, the agent loop collects the registrations onto
the turn, and the presentation layer delivers them — Telegram as documents, the
CLI as printed paths.

**Outgoing.** The model has two ways to hand you a file:

- `files` with `upload: true` on a write — for anything it generates directly.
- the `files` action `upload` — for a file that already exists, e.g. one a shell
  command produced. The file is not modified.

Each delivered file is copied into `data/outbox/` first, so it survives even if
the original is later deleted or overwritten. A staged copy is deduplicated on
(name, size, mtime), so re-sending an unchanged file does not pile up copies.

**Incoming.** Send the bot a document and it is stored under
`workspace/uploads/<chat_id>/` (filename sanitised, never overwritten), and the
model gets a prompt naming the path and the caption. From there it is an
ordinary project file — read it, process it with the shell, rename it.

**Limits, all enforced with a model-readable reason:** files must have an
extension from an allowlist (text, code, data, documents, images, common
archives — no `.exe`), be non-empty, live under the project, and fit under
`bot.max_upload_mb` (default 20 MB; Telegram's own document ceiling is 50 MB).
A blocked delivery degrades to a note in the chat, never a failed turn. Turn
the whole feature off with `tools.files.uploads = false` — the `files` tool
keeps working for reads and writes.

## Shell safety

Three tiers, resolved in order, strictest wins:

- **deny** — never runs, not even with approval. `rm -rf /`, `sudo`, piping a
  download into a shell, writing to `/dev/sda`, touching `.bashrc` or `.ssh`,
  writing to a `.git/hooks` file, persistent `git config --global`, reverse
  shells, `find / -delete`, power state changes, fork bombs.
- **confirm** — the owner gets a Confirm/Cancel button in the chat. Anything that
  deletes, moves, overwrites, changes permissions, installs, commits, or
  redirects into a file.
- **allow** — everything else runs immediately.

On top of that: writes outside the project are blocked, `HOME` is pinned to the
project, the child environment is scrubbed of anything matching
`*KEY*|*TOKEN*|*SECRET*|*PASSWORD*` (so a command cannot exfiltrate your API
keys), every command runs in its own process group and is killed on timeout, and
output is capped.

The checks read the token stream, not just the raw text, so a few things are
caught that a regex over the command line would miss: `cd .. && rm -rf data` is
judged from the directory the shell ends up in rather than the one it started
in, `env cp a /etc/x` is judged as the `cp` it really is, `find` is read-only
until it is handed `-delete` or `-exec`, and a path inside an interpreter's
quoted string — `python3 -c "open('/etc/passwd','w')"` — is treated as the write
target it is. Paths are resolved before they are judged, so a symlink is
classified by what it points at.

Set `tools.shell.writable = ["workspace"]` to confine writes to the scratch
folder; anything outside it then asks first, and a recursive delete outside it
is denied. The default is empty, meaning the whole project, because a coding bot
has to be able to edit the code it works on.

Set `tools.shell.ask_before_risky = false` to refuse the confirm tier outright,
or add your own patterns:

```toml
[tools.shell]
writable      = ["workspace"]
extra_deny    = ["\\bnpm\\s+publish\\b"]
extra_confirm = ["\\bterraform\\b"]
```

### Threat model

This is a guardrail, not a sandbox. Shell is Turing-complete; a determined
adversary who reaches the model's prompt can find a phrasing the checks miss.
What the engine buys you is that the *common* destructive commands — the ones a
hallucination or a careless prompt produces — cannot run by accident. It is
strongest against the shapes a model actually writes (`rm -rf`, `sudo`, a
download piped into a shell) and weakest against a determined effort to
rephrase: a command the classifier cannot follow, such as one that builds its
path from a variable, is escalated to a prompt rather than resolved, so it
cannot run unattended, but it can run if the owner says yes.

If you need a hard boundary, run Lumi in a container or under a dedicated
unprivileged user. Owner gating is by Telegram user id and is enforced twice: as
a handler filter, and as an explicit check in every privileged handler.

## Web providers

`tools.web.provider_order` is tried left to right; the first available provider
answers, and one that fails at call time falls through to the next.

```toml
[tools.web]
provider_order = ["firecrawl", "exa", "tavily", "mcp"]
```

- **firecrawl** — best for reading a specific page. Renders JavaScript,
  returns clean markdown. No keyless tier — a `FIRECRAWL_API_KEY` is
  required, but a comma-separated pool with the same three strategies
  (`tools.web.firecrawl.key_strategy`) keeps the bot up when one key is
  rate-limited.
- **exa** — neural search built for agents: natural-language queries in,
  semantically ranked pages out, with highlights or full text per hit. Needs
  an `EXA_API_KEY` (no keyless tier); same rotation rules as the others
  (`tools.web.exa.key_strategy`). Optional `search_type` (`auto` / `neural` /
  `keyword`) and `category` (`news`, `github`, `paper`, `pdf`, ...) knobs in
  `[tools.web.providers.exa]`.
- **tavily** — best for agentic search. LLM-ready snippets, optional bundled
  answer. **Works without an API key** — the SDK runs in its free tier
  (low rate limit; `search` and `extract` only) when `TAVILY_API_KEY` is
  unset. With one or more keys, a failed key is parked for a minute and the
  next one answers; `tools.web.tavily.key_strategy` picks
  `fallback` / `round_robin` / `random`.
- **mcp** — reads `.mcp.json` from the project root, expands `${VAR}` from the
  environment, and speaks MCP over streamable HTTP or stdio. It discovers each
  server's tools and reads their input schemas, so it works with a server whose
  parameter is called `q` instead of `query`.

```json
{
  "mcpServers": {
    "tavily-remote": { "type": "remote", "url": "${TAVILY_MCP_URL}" },
    "firecrawl-remote": { "type": "remote", "url": "${FIRECRAWL_MCP_URL}" },
    "exa-remote": { "type": "remote", "url": "https://mcp.exa.ai/mcp" }
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
