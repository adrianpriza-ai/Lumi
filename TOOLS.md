# Tools

[← Back to README](README.md) · [CONFIGURATION.md](CONFIGURATION.md) · [TELEGRAM.md](TELEGRAM.md)

| Tool | Capability |
|-|-|
| `run_shell` | Shell commands, with the allow/confirm/deny engine |
| `files` | Read anywhere in the project, write and edit only in `workspace/`, send files to the chat |
| `memory` | Remember and recall; forgetting is human-only by design |
| `web` | Search and fetch, across three interchangeable providers |
| `context7` | Up-to-date library docs from https://context7.com; auto-enables on `CONTEXT7_API_KEY` |

## Editing files

The `files` tool is where edits happen, in three shapes:

- `write` creates a file or replaces it wholesale. Replacing a non-empty file
  asks the owner first and shows a diff of what would change (`overwrite: true`
  skips the prompt once the owner has seen it).
- `append` adds to the end and never asks — nothing is lost.
- `edit` swaps an exact `old_string` for a `new_string`. It needs no approval,
  because it can only touch the text it names: the rest of the file is
  untouched by construction, and a `old_string` that matches more than once is
  refused rather than guessed at (add context to make it unique, or set
  `replace_all`). It shows the same diff `write` does, and refuses binary files.

`edit` is the right default for a change to an existing file; `write` is for
creating one or replacing all of it. Both are confined to `tools.files.writable`
(default: `workspace/`) and both refuse the shell policy's protected paths —
a `.git/hooks` file or an ssh key is off-limits through either tool.

## File delivery (documents)

The bot exchanges files with the chat through one small harness (`lumi/artifacts.py`). Tools never talk to Telegram: a tool that produces a file registers it with the harness, the agent loop collects the registrations onto the turn, and the presentation layer delivers them — Telegram as documents, the CLI as printed paths.

**Outgoing.** The model has two ways to hand you a file:

- `files` with `upload: true` on a write — for anything it generates directly.
- the `files` action `upload` — for a file that already exists, e.g. one a shell command produced. The file is not modified.

Each delivered file is copied into `data/outbox/` first, so it survives even if the original is later deleted or overwritten. A staged copy is deduplicated on (name, size, mtime), so re-sending an unchanged file does not pile up copies.

**Incoming.** Send the bot a document and it is stored under `workspace/uploads/<chat_id>/` (filename sanitised, never overwritten), and the model gets a prompt naming the path and the caption. From there it is an ordinary project file — read it, process it with the shell, rename it.

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
- **confirm** — the owner gets a Confirm/Cancel button in the chat. Anything
  that destroys or overwrites state: deleting, clobbering a file that already
  exists (whether the clobber arrives as `mv`, `cp`, `tee`, or a `>` onto a
  live name), changing permissions, installing, committing, or writing
  outside the workspace.
- **allow** — everything else runs immediately, and creating is not
  clobbering: `mkdir`, `touch`, a move or copy onto a name that does not exist
  yet, a `>` that only creates a file, a `>>` append inside the workspace, and
  `2>/dev/null` all run without a tap. There is nothing to lose, so there is
  nothing to confirm — and asking on every harmless command would only teach
  the owner to stop reading the prompt.

On top of that: writes outside the project are blocked, `HOME` is pinned to the project, the child environment is scrubbed of anything matching `*KEY*|*TOKEN*|*SECRET*|*PASSWORD*` (so a command cannot exfiltrate your API keys), and every command runs in its own process group and is killed on timeout.

Output is not truncated. `tools.shell.max_output_chars` is `0` by default, so the model sees the whole thing; what gets dropped instead is *old* output, when the conversation outgrows the context window — at which point the model is told a command ran and that its output is no longer in front of it. Set the cap to a number if you would rather always see a short version.

The checks read the token stream, not just the raw text, so a few things are
caught that a regex over the command line would miss: `cd .. && rm -rf data` is
judged from the directory the shell ends up in rather than the one it started
in, `env cp a /etc/x` is judged as the `cp` it really is, `find` is read-only
until it is handed `-delete` or `-exec`, and a path inside an interpreter's
quoted string — `python3 -c "open('/etc/passwd','w')"` — is treated as the write
target it is. Paths are resolved before they are judged, so a symlink is
classified by what it points at.

A target the classifier cannot follow does not get guessed at: anything built
from a variable or a substitution — `mv a $DEST`, `echo x > $OUT`,
`python3 -c "open('$OUT','w')"` — is escalated to a prompt, because the path
written on the command line and the path used at run time are not the same
path. Guessing "it is probably in the project" is how a guardrail silently
becomes decoration.

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

`tools.web.provider_order` is tried left to right; the first available provider answers, and one that fails at call time falls through to the next.

```toml
[tools.web]
provider_order = ["firecrawl", "exa", "tavily", "mcp"]
```

- **firecrawl** — best for reading a specific page. Renders JavaScript, returns clean markdown. No keyless tier — a `FIRECRAWL_API_KEY` is required, but a comma-separated pool with the same three strategies (`tools.web.firecrawl.key_strategy`) keeps the bot up when one key is rate-limited.
- **exa** — neural search built for agents: natural-language queries in, semantically ranked pages out, with highlights or full text per hit. Needs an `EXA_API_KEY` (no keyless tier); same rotation rules as the others (`tools.web.exa.key_strategy`). Optional `search_type` (`auto` / `neural` / `keyword`) and `category` (`news`, `github`, `paper`, `pdf`, ...) knobs in `[tools.web.providers.exa]`.
- **tavily** — best for agentic search. LLM-ready snippets, optional bundled answer. **Works without an API key** — the SDK runs in its free tier (low rate limit; `search` and `extract` only) when `TAVILY_API_KEY` is unset. With one or more keys, a failed key is parked for a minute and the next one answers; `tools.web.tavily.key_strategy` picks `fallback` / `round_robin` / `random`.
- **mcp** — reads `.mcp.json` from the project root, expands `${VAR}` from the environment, and speaks MCP over streamable HTTP or stdio. It discovers each server's tools and reads their input schemas, so it works with a server whose parameter is called `q` instead of `query`.

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

### How much comes back

A search returns `tools.web.max_results` hits (10 by default) and page text is not truncated: `tools.web.max_content_chars` is `0`, so a hit the model can actually read is a hit it will not have to search for again. The model is told it may ask for more than the default, up to 50.

A thin result set is topped up rather than answered — if the first provider returns fewer than `min_results` hits, the next provider in the order fills the gap (deduplicated by URL) before the model sees it. Every result carries the date it was searched and, where the provider knows it, the publish date, so a stale page cannot pass for a current one.

## Context7 (library docs)

A separate tool from `web`. Where `web` finds pages on the open internet, `context7` returns version-specific documentation and code examples for a named library. The training data the model runs on is older than most libraries; this is how it catches up.

The tool is **auto-validated**: the moment `CONTEXT7_API_KEY` is set in `.env`, the tool appears in the model's tool list; if the variable is unset, the tool is hidden from the system prompt and `available()` returns a clear reason. The same `OPENAI_API_KEY` rotation pattern applies — a comma-separated pool with one of `fallback` (default), `round_robin`, or `random`:

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
