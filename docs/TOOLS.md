# Tools

[← Back to README](../README.md) · [CONFIGURATION.md](CONFIGURATION.md) · [TELEGRAM.md](TELEGRAM.md) · [DEPLOYMENT.md](DEPLOYMENT.md)

| Tool | Capability |
|-|-|
| `run_shell` | Shell commands, with the allow/confirm/deny engine |
| `files` | Read anywhere in the project, write and edit only in `workspace/`, send files to the chat |
| `read_file` | Read a file by line window (`offset`/`limit`), capped at ~30k tokens per call; PDFs through OCR, images attached for the model to look at |
| `list_files` | List one directory, like `ls` |
| `grep` | Multi-word content search across the project |
| `memory` | Remember and recall, scoped to the chat it is used in; forgetting is human-only by design |
| `more_context` | Up to 10 messages of room chatter the bot never answered (`offset` ≤ 9, ~30k-token cap), dropped media included with its saved path |
| `ocr` | Extract text from PDFs, scans, and document images |
| `web` | Search and fetch, across three interchangeable providers |
| `context7` | Up-to-date library docs from https://context7.com; auto-enables on `CONTEXT7_API_KEY` |
| `send` | Send the owner a message *mid-turn*, so one reply can arrive in parts |
| `skills` | List and read the owner's skills (`data/skills/`, `~/.agents/skills/`), no approval to read |

## Reading and searching (`read_file`, `list_files`, `grep`)

Three small read-only tools, separate from `files` because their job is only to
*look* — they never ask for approval and never write anything. `files` stays the
write/edit/upload tool.

- `read_file` reads a window of lines (1-based `offset`, `limit`) and numbers
  them. One call is capped at ~30k tokens; an oversized read is **refused with
  the remedy** — "use offset and limit" — rather than silently truncated,
  because a cut-off read looks like the whole truth to the model that asked.
  The result reports the total line count and the next `offset` to continue from.
  A *single* line longer than the cap is a dead end for `offset`/`limit` — no
  window can include it — so the refusal names the line and points at
  `run_shell` for that one line.
- **Media reads**: a PDF is read through OCR — embedded text is extracted for
  free, scanned pages are rendered and transcribed by a vision model — and
  comes back as page-marked text, with `offset`/`limit` paging *pages* instead
  of lines. An image file (png, jpg, webp, gif) is attached to the result as a
  picture the model looks at directly; the LLM client routes that request to
  the configured vision model, exactly the path an owner-sent photo takes.
  `files: read` takes the same two routes, sharing the machinery
  (`lumi/tools/docread.py`) with the `ocr` tool so the tools cannot drift
  apart. All the read guards still apply: secrets are refused, another chat's
  uploads are invisible, and `tools.ocr.enabled = false` turns media reads off
  with the `ocr` tool.
- `list_files` is `ls`: one directory, directories first with entry counts,
  files with sizes. Venvs, `node_modules` and caches stay hidden. A listing
  longer than 500 entries is cut, and the body says how many were hidden —
  the header's totals still describe the whole directory.
- `grep` searches file contents. A multi-word query is an AND: every word must
  appear on the same line (case-insensitive by default), so `config token
  limit` returns lines containing all three. Results are
  `path:line: content`, scoppable with `glob` and `path`; binary files and
  files over 1 MB are skipped. A `glob` behaves the way a path glob does:
  `**` spans directories (`**/*.py` finds the root files too) while `*` and `?`
  stop at a slash, and a bare `*.md` matches at any depth. A scan that hits
  20,000 files stops and says so, the same way an over-long match list does —
  a partial answer never pretends to be complete.

Remove any of them from `tools.enabled` to take it away from the model.

## Mid-turn messages (`send`)

One turn produces one answer, which is wrong for long work: you would rather
hear "phase 1 done, starting phase 2" when it happens than read it three
minutes later as part of a wall of text. The `send` tool is how the model says
that out loud. It reaches the chat the moment the model calls it — the agent
loop delivers it, the tool itself never touches Telegram — and then the turn
keeps going; the final answer still arrives when the turn ends.

Like every other tool it is a file: `lumi/tools/send.py`. Remove `"send"` from
`tools.enabled` to take it away from the model.

## The room's own conversation (`more_context`)

In a group the bot only sees messages addressed to it, so the model's history
is a series of interruptions with the conversation between them missing. When
the owner says "as they said above", there is nothing above — this tool is
how the model asks for it.

It reads a per-chat log of exactly the messages the bot *dropped*
(`lumi/chatlog.py`): never the ones addressed to it, which are already in the
conversation, and never a private chat, where every message already became a
turn. Media is in the log too: when a dropped photo or document can be saved,
the bot stores it under `workspace/uploads/<chat>/` (the same place a document
sent *to* the bot lands) and the log line names the path, so the model can
`read_file` it — PDFs and images included. An attachment that cannot be saved
still shows, marked "not saved".

- `offset` (0-9): skip this many of the newest messages before reading — use
  it to walk further back.
- `limit` (1-10, default 10): how many messages to return, oldest first.

One call is capped at 10 messages and ~30k tokens; a result over the cap is
**refused with the remedy** — "call again with a smaller `limit`" — never
truncated, the same rule `read_file` follows: a cut-off answer reads like the
whole truth to the model that asked.

The log is in memory (10 messages per chat) and starts fresh on a restart;
long-term memory is still `MEMORY.md`, and the transcript is still the bot's
own conversation. Reading the room at all needs Telegram's group privacy
setting disabled — see [TELEGRAM.md](TELEGRAM.md). Remove `"more_context"`
from `tools.enabled` to take it away from the model.

## OCR and document images (`ocr`)

The `ocr` tool reads PDFs, scans, and document images from project paths. It
keeps the input-type routing explicit:

- without a dedicated provider, embedded PDF text is extracted locally without
  a model call;
- locally rendered PDF pages use `llm.ocr.model`, then `llm.ocr_model`, then
  the default `llm.model` route, never the photo-only vision route;
- image files use `llm.vision_model` or the `[llm.vision]` route, even when a
  dedicated OCR API is configured;
- a dedicated `tools.ocr` provider can process whole PDFs and public document
  URLs through a Mistral-compatible `POST /ocr` endpoint.

The local machinery — rendering, guards, page windows — lives in
`lumi/tools/docread.py` and is shared with `read_file` and `files: read`, so
those tools read media the same way. What `ocr` keeps to itself is the
transcription contract: *text out*, never an attached image (attaching is
`read_file`'s route), plus the URL route above.

The committed configuration enables the tool. Remove `"ocr"` from
`tools.enabled` or set `tools.ocr.enabled = false` to disable it. Set
`tools.ocr.provider = "mistral"` for the
dedicated API path. With no provider, local PDF extraction needs `pypdfium2`
and only scanned pages use the configured model.

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

One gate applies to all three: a file that was created or changed **during the
current session** gets one owner approval on its first `write`, `append` or
`edit`, and is then treated normally for the rest of the session. Files that
predate the process are ordinary project material and follow the bullets above.
This is the write half of the write-then-run chain — fetched content can talk
the model into authoring a file, so the owner sees what is being written once
before it changes; the shell asks separately before *running* such a file
(`exec:recent-script`).

`edit` is the right default for a change to an existing file; `write` is for
creating one or replacing all of it. Both are confined to `tools.files.writable`
(default: `workspace/` and `data/skills/`) and both refuse the shell policy's protected paths —
a `.git/hooks` file, an ssh key, or the bot's own `config.toml`,
`data/whitelist.json` or `.mcp.json` is off-limits through either tool.

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

## Skills (`skills`)

A **skill** is a directory with a `SKILL.md` in it — a name, a one-line
description and the instructions for a job the owner does more than once. The
tool reads them from two roots, in this order:

- `data/skills/<name>/SKILL.md` — this project's skills, committed next to the
  code they describe and maintained with the `files` tool;
- `~/.agents/skills/<name>/SKILL.md` — the standard user-level directory, so a
  skill installed once is available in every project.

A project skill shadows a user skill of the same name, and `list` says so.

The **catalog — every skill's name and one-line description — is part of the
system prompt**, so the model sees what exists on every turn and loads the one
that matches the task in front of it before starting. The bodies stay on disk:
a 30-skill install costs a line per skill in the prompt, not a skill's worth of
context, and the block stops growing at 60 entries (the rest is one `list`
away). With no skills, or with the tool off, there is no section at all.

Two actions, both **read-only, so neither ever asks for approval** — the same
contract `read_file` has:

- `list` — every skill found, with its description and which root it came
  from. An empty listing says where to put the first one.
- `read` — load a skill's `SKILL.md` into the conversation, by `name`. Pass
  `file` to read any other file inside that skill's directory (a script or a
  reference the skill points at); the file must stay inside the skill, and the
  secret-file guard still applies, so a `.env` parked in a skill directory is
  refused like any other.

Creating or changing a skill is the `files` tool's job: `data/skills` is one of
its writable roots, so `files` with `action: "write"` and
`path: "data/skills/<name>/SKILL.md"` is how a skill is born. The shell's
policy covers it too — `data/skills` is not on the protected list (`data/memory`,
`data/history` and `data/whitelist.json` are), so nothing about it prompts.

Skill contents are instructions the *owner* kept, and they are read like any
other project file: they add to the rules the model was given, never replace
them. Remove `"skills"` from `tools.enabled` to take the tool away.

## Shell safety

Three tiers, resolved in order, strictest wins:

- **deny** — never runs, not even with approval. `rm -rf /`, `sudo`, piping a
  download into a shell, writing to `/dev/sda`, touching `.bashrc` or `.ssh`,
  writing to a `.git/hooks` file, rewriting the bot's own state
  (`data/whitelist.json`, `.mcp.json`, `config.toml`), persistent
  `git config --global`, reverse shells, `find / -delete`, power state
  changes, fork bombs.
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

Reads are bounded the same way. A read-only command may read inside the project
and may read a system path — `cat /etc/passwd` is fine — but a read of
anything else on the machine asks the owner first, the same way a write
outside the project does. Files whose *contents* are credentials are refused
outright, even to `cat`: `.env` and its variants, private keys, and
anything inside `.ssh`, `.aws`, `.gnupg`, `.kube` or `.docker`. The
environment scrub hides secrets from a command's *environment*; without a
read-side guard, `cat .env` would hand the same values straight back.

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
a handler filter, and as an explicit check in every privileged handler. What
the guardrails assume about the machine they run on is written down in
[DEPLOYMENT.md](DEPLOYMENT.md).

## Web providers

`tools.web.provider_order` is tried left to right; the first available provider answers, and one that fails at call time falls through to the next.

```toml
[tools.web]
provider_order = ["firecrawl", "exa", "tavily", "you", "mcp"]
```

- **firecrawl** — best for reading a specific page. Renders JavaScript, returns clean markdown. No keyless tier — a `FIRECRAWL_API_KEY` is required, but a comma-separated pool with the same three strategies (`tools.web.firecrawl.key_strategy`) keeps the bot up when one key is rate-limited.
- **exa** — neural search built for agents: natural-language queries in, semantically ranked pages out, with highlights or full text per hit. Needs an `EXA_API_KEY` (no keyless tier); same rotation rules as the others (`tools.web.exa.key_strategy`). Optional `search_type` (`auto` / `neural` / `keyword`) and `category` (`news`, `github`, `paper`, `pdf`, ...) knobs in `[tools.web.providers.exa]`.
- **tavily** — best for agentic search. LLM-ready snippets, optional bundled answer. **Works without an API key** — the SDK runs in its free tier (low rate limit; `search` and `extract` only) when `TAVILY_API_KEY` is unset. With one or more keys, a failed key is parked for a minute and the next one answers; `tools.web.tavily.key_strategy` picks `fallback` / `round_robin` / `random`.
- **you** — You.com's hosted MCP endpoint, `https://api.you.com/mcp?profile=free`. **Keyless by design**: no account and no API key, so a fresh clone still has a working search (low rate limit). A `YOU_API_KEY`, when set, is sent as an `X-API-Key` header. Disable it with `enabled = false` in `[tools.web.providers.you]`.
- **mcp** — reads `.mcp.json` from the project root and `~/.agents/mcp.json` (for servers configured once for every project; the project file wins on a duplicate server name, and `tools.web.providers.mcp.user_config_file = ""` turns the user file off), expands `${VAR}` from the environment, and speaks MCP over streamable HTTP or stdio. It discovers each server's tools and reads their input schemas, so it works with a server whose parameter is called `q` instead of `query`.

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
tool list, and `lumi chat`'s `/tools` will show it. To make it ask permission first, raise
`NeedsApproval` from `invoke` when `ctx.confirmed` is false — the agent loop
already turns that into a Confirm/Cancel pair.
