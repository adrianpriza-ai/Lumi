# Configuration

[← Back to README](README.md) · [TOOLS.md](TOOLS.md) · [TELEGRAM.md](TELEGRAM.md)

`config.toml` is committed and holds no secrets. `.env` holds the secrets and is
gitignored. Every config value can also be set as an environment variable:

```bash
LUMI__LLM__MODEL=qwen3:8b
LUMI__TOOLS__SHELL__TIMEOUT_SECONDS=10
LUMI__TOOLS__WEB__PROVIDER_ORDER=mcp,tavily
```

Precedence: dataclass defaults < `config.toml` < `config.local.toml` (gitignored)
< `LUMI__*` environment variables.

## The LLM

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
content array, so any endpoint that accepts `image_url` parts works. Set
`vision_model` to a vision-capable id (`gpt-4o`, `claude-3.5+`, `gemini-2.*`,
`llama-3.2-vision`, `qwen2-vl`, …) and photo turns go to it while everything
else stays on `model` — same endpoint, same keys. Leave it empty and photos go
to `model` too, which then has to be vision-capable or they will fail.

Resolution order, most specific wins:

1. `base_url` / `model` / `vision_model` / `api_key_env` / `key_strategy` in `config.toml`
2. `LUMI__LLM__BASE_URL`, `LUMI__LLM__MODEL`, `LUMI__LLM__VISION_MODEL`, `LUMI__LLM__API_KEY_ENV`, `LUMI__LLM__KEY_STRATEGY`
3. `OPENAI_BASE_URL` (or `OPENAI_API_BASE`), `OPENAI_MODEL`, `OPENAI_VISION_MODEL`, and the key named by `api_key_env`
4. OpenAI and `gpt-4.1-mini`

The reasoning settings (`reasoning`, `reasoning_effort`, `reasoning_tokens`,
`show_reasoning`) are Lumi's own, so they resolve from `config.toml` and
`LUMI__LLM__*` only — no bare environment variable. See
[Reasoning models](#reasoning-models).

Always check what actually took effect:

```bash
lumi config --key llm.base_url      # -> http://localhost:20128/v1
lumi config --key llm.model         # -> nemotron-3-nano-reasoning
lumi config --key llm.vision_model  # -> gpt-4o (empty means: use llm.model for photos)
lumi config --key llm.key_strategy  # -> fallback
lumi doctor | grep llm              # -> model via endpoint (from OPENAI_BASE_URL), keys: fallback
```

If a call fails, the error names the endpoint, model and key variable it used, so
a key rejected by the wrong host is obvious immediately. `temperature = null`
omits the parameter, which some reasoning models require.

## No output ceiling

`max_tokens` is `0` by default, which sends **no** `max_tokens` /
`max_completion_tokens` parameter at all. The provider's own output limit
applies, and the model writes as long as it has something to say.

A number here is not a "generous default", it is a wall the model stops at — a
2000-token answer is not a long answer, it is an answer cut off mid-sentence.
If your provider needs a specific limit, set it:

```toml
[llm]
max_tokens = 32000
```

`reasoning_tokens` adds headroom on top for the thinking trace, and is also `0`
by default: with no ceiling being sent there is nothing to add it to, and the
provider's own limit already covers the trace.

The same rule applies to the tools — nothing is cut at the source any more:

| Setting | Default | 0 means |
|-|-|-|
| `llm.max_tokens` | `0` | no ceiling sent; the provider decides |
| `llm.reasoning_tokens` | `0` | no separate thinking headroom |
| `tools.shell.max_output_chars` | `0` | the command's whole output is sent |
| `tools.web.max_content_chars` | `0` | no page or hit is cut short |
| `tools.files.max_read_chars` | `200000` | no cap |

`llm.max_memory_chars` is the exception, and deliberately so — see
[the memory file](#the-memory-file) below.

What does get cut is decided by the context window instead, and only when the
conversation actually outgrows the model — see
[The context window](#the-context-window).

## The context window

History is budgeted in **tokens against the model's window**, not counted in
messages. A message count is what makes a bot forget: sixty messages is a busy
afternoon, and the moment the count is exceeded the oldest prefix is deleted, so
the model is later asked about something it has never seen and answers from
imagination. With a 200k window you get the last 200k tokens instead, which on a
normal day is *everything*.

```toml
[llm]
context_window = 200000        # your model's real limit
context_headroom = 16000       # kept free for the reply
context_keep_recent = 12       # tail messages never condensed
compaction = true              # summarise before dropping
history_turns = 0              # replay as much as fits
max_conversations = 32         # chats held in memory at once
```

When the window fills, the oldest part is condensed rather than deleted, in
this order:

1. **Old tool output is elided** to one line saying what ran. It is the
   bulkiest and least conversational thing in any context, and the model
   already has what it needed from it.
2. **The oldest turns are summarised** by the model itself, with instructions to
   keep decisions, facts about you, files and versions, and anything unfinished
   — and to leave out the sequence of tool calls. The record replaces the turns
   it covers.
3. **Only if that is not enough** are turns dropped verbatim, and only if the
   model would not write a record.
4. **Last resort**, a single message too big for the window on its own gets cut.
   The system prompt and the turn being answered are never cut.

The record is written to `data/history/<chat_id>.jsonl` as a `summary` row, so a
restart inherits the thread instead of starting it over. `/context` shows the
window, what is in use, and the record itself.

Two invariants worth knowing: an assistant's tool calls always move with their
tool results (half of either is a 400, not a shorter conversation), and the
newest `context_keep_recent` messages are never condensed — losing the newest
turns is how a model starts contradicting the message it was replying to.

### Memory

`MEMORY.md` is the one part of the prompt with no second chance: the system
prompt is never cut, so an oversized memory file would grow into the window with
nothing able to reclaim it, and the conversation would quietly get smaller
instead. It is therefore always bounded — at an eighth of the window by default,
which is room for hundreds of facts:

```toml
[llm]
max_memory_chars = 0    # 0 = an eighth of the window (~64k chars at 200k tokens)
```

Set a number to cap it harder. When the file does not fit, the **oldest** of the
bot's own facts are elided — the recent ones are what it is still acting on, and
dropping them in favour of ancient ones is how a memory becomes useless while
still looking full. The hand-written parts of the file (headings, prose, links)
are kept as long as they fit.

Whatever is elided is stated in the prompt itself, because a model told it has
"all" of its memory and quietly missing the first forty facts will confidently
tell you it was never told:

```
<!-- 186 older remembered fact(s) elided to fit the memory budget; the ones below are the most recent -->
```

`/memory` and `lumi doctor` both report when this is happening and how many
facts are affected, so a memory that is on disk but not being sent is visible
rather than inferred.

### Memory usage

Each conversation is bounded by the window it was assembled for, transcripts are
read tail-first rather than whole (a megabyte at a time, however large the file
has grown), and at most `max_conversations` chats are held at once. An evicted
chat replays itself from its transcript on the next message, so the default of 32
is a memory ceiling with no functional cost. Measured at 32 chats × 25 turns:
**2.6 MiB** of live text, with the transcripts on disk.

## Reasoning models

A reasoning model thinks before it answers, and that breaks three conventions
the rest of the program relies on. All three are handled by one flag:

```toml
[llm]
reasoning = true
reasoning_effort = "medium"   # empty = "medium"; the value switches thinking on
reasoning_tokens = 0          # 0 = no separate headroom (the default)
show_reasoning = true
```

`reasoning = true` does three things:

| | Why |
|-|-|
| drops `temperature` | these models reject any value but their own default, and a 400 on every turn is the usual symptom |
| moves a configured limit to `max_completion_tokens` | o-series and gpt-5 reject the old `max_tokens` name outright (with `max_tokens = 0` there is no limit to move) |
| adds `reasoning_tokens` to a configured ceiling | thinking is billed as output, so headroom keeps a long trace from eating the answer |

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

### Seeing the thinking

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

`/context` reports what the model is holding for this chat right now: the window
and budget, the fill, what has been condensed, and the record itself when there
is one. It exists because "the bot forgot" is otherwise impossible to argue
with — a bot that forgets is usually a bot that deleted your history, and you
should be able to see that rather than infer it.

## Owner-only introspection: `/config` and `/env`

Two Telegram commands show what the running process is built on, and both are
locked to `TELEGRAM_OWNER_ID` — whitelisted users get the bot but not the
backend view.

`/config` prints the resolved configuration: the model and endpoint in use,
web provider order, shell and file limits, group mode. Nothing here is secret;
it is owner-only because a shared group has no business reading the layout.

`/env` lists every credential variable with its value **masked** (`sk…efgh`,
plus set/unset, pool size, and strategy), any `LUMI__` overrides, and a
reminder that keys are read at startup — edit `.env` and restart. Full values
are never sent to the chat on purpose: a Telegram chat lives on Telegram's
servers and on every device logged into the account, so even an owner-only
command keeps its secrets. Use `lumi doctor` in a terminal when you need the
unchanged view of what is set.

The trace is read from `reasoning_content`, `reasoning` or `thinking` depending
on the provider, and a model that inlines `<think>` tags instead of using a
field has them stripped out — otherwise raw XML ends up in the chat. It is never
written back into the conversation history.

## Several keys

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

The web providers do too: `TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, and
`EXA_API_KEY` are all pools under the hood (`tools.web.tavily.key_strategy`,
`tools.web.firecrawl.key_strategy`, `tools.web.exa.key_strategy`). Tavily
additionally accepts a **keyless** mode — leave the variable unset and the SDK
falls back to its free tier (lower rate limit; `search` and `extract` only).
Firecrawl and Exa have no keyless tier, so an unset env var means the provider
is unavailable, not a fallback option. All of them share one parser
and one rotation policy — `lumi/util/keys.py` and `lumi/llm/keypool.py` — so
the rules above hold for every one of them.

`TAVILY_API_KEY`, `FIRECRAWL_API_KEY`, and `EXA_API_KEY` take a single key
each. A comma in any of them is not a pool; it is a broken key.

Keys are read at startup, so a pool is picked up by a restart. `lumi doctor`
shows how many keys it found and which strategy is active.
