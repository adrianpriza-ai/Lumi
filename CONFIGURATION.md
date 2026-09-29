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

## Reasoning models

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
