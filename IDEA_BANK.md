# Idea bank

Notes the assistant keeps while working on Lumi, so decisions survive across
sessions and ideas do not get lost. Read this before making large changes;
append decisions as they are made. (Maintained by the assistant — the owner
can edit freely.)

## Decisions made

### 2026-11-22 — model catalog, /model command, ~/.config/lumi

- **Config layering**: `~/.config/lumi/config.toml` is a *base* layer; the
  repo's `config.toml` and `config.local.toml` override it; env vars
  (`LUMI__…`) stay on top. The repo stays self-contained (tests, committed
  defaults). Direction: repo wins, user dir fills gaps.
- **Model catalog**: JSONC files, merged per provider (user layer first,
  project overrides):
  1. `~/.config/lumi/models.json` (user, primary)
  2. `.models.json` at the repo root (project, hidden file)
  Only files with these exact names are loaded; anything else is inert.
- **Catalog files hold API keys** → both are on the safety secret-file
  denylist, so the agent's shell/file tools cannot read them.
- **`/model <id>` semantics**: per chat (Telegram chat id; TUI/CLI share
  `"cli"`), persisted in `data/chat-models.json`, survives restarts.
  `/model default` returns to the configured default. Owner-only on Telegram
  (keys get spent).
- **Format**: OpenCode-inspired, client-adapted. Supported: provider
  `id`/`aliases`, `options.baseURL`, `apiKey`/`apiKeys` (literal,
  `${VAR}`, `file:/path`, or empty → `[llm].api_key_env`), `headers`,
  `timeoutSecs`, `keyStrategy` (mapped: `first`→fallback, `rotate`→
  round_robin, `random`→random), model `name`, `upstreamName`, `as` (alias),
  `limit.context/output`, `description`, `modalities`, `thinking`, `kind`.
  Proxy-only keys (`maxInFlight`, `keyless` passthrough) are accepted but
  ignored.
- **`~/.config/lumi/` layout**: `config.toml`, `models.json`, `skills/`
  (skill root, between project and `~/.agents/skills` in shadowing), and
  `agents/` reserved for a future agents feature (not implemented).

## Open ideas (not built yet)

- `~/.config/lumi/agents/` — user-defined agent personas/configs the
  `/model`-style commands could switch between (the reason the config dir
  exists as a hub).
- `~/.config/lumi/mcp.json` as a user MCP config layer next to
  `~/.agents/mcp.json` (MCP currently reads only `~/.agents/mcp.json`,
  `lumi/config.py:1123`).
- `keyStrategy: "sticky"` — pin one conversation to one key for prompt-cache
  hits. Needs a KeyPool mode keyed by conversation, not just a strategy.
- Non-chat catalog kinds (`image`, `video`, `tts`, `stt`, `ocr`,
  `moderation`) — route them into the existing `image_gen` / `ocr` /
  audio tooling instead of only listing them in `/model list`.
- Per-provider system prompts (`"prompt": "prompts/x.txt"` in the catalog)
  layered under PERSONALITY.md.
- Per-chat `chars_per_token` calibration — context estimation is shared
  across chats; per-model calibration would tighten per-chat windows.
- Client lifecycle: per-chat catalog clients hold one `AsyncOpenAI`/httpx
  pool each, and an evicted conversation drops its client to the GC. An
  `aclose()` on `LLMClient` (or a shared pool cache keyed by provider)
  would bound the churn.
- `/model` in `lumi config --key …` / `lumi doctor --json` surfaces.
- TUI: a model picker overlay (arrow keys) instead of typing ids.

## Review outcomes (2026-11-22, post-implementation review)

Fixed after review: malformed `headers` shape no longer crashes the boot
(P1); a non-chat `defaultModel` is refused at load (P1); TUI `model`
keywords fold case like the CLI/Telegram (P2); catalog files are write-
protected too, not only read-denied (P2); a mid-turn `/model` now applies
from the next turn instead of mid-tool-loop (P3); `${VAR}` key pools split
on whitespace (P3); duplicate-provider-id folds warn on endpoint/key
disagreement (P3); `config_path` now names the highest config layer
loaded. Regression tests for both P1s live in `tests/test_catalog.py`.
