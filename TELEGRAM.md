# Telegram

[← Back to README](README.md) · [CONFIGURATION.md](CONFIGURATION.md) · [TOOLS.md](TOOLS.md)

Two things about the Telegram side trip people up: when the bot is allowed to
answer in a group, and getting a TCP connection to `api.telegram.org` at all.
Both are configuration; nothing here needs a code change.

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
  (default -1, retry forever). The counter only accumulates while the network
  is continuously down — one success resets the loop by ending it — so any
  finite value is a total budget, not a per-outage one. -1 keeps retrying
  startup until Telegram is reachable, matching the polling loop, which
  already retries disconnections indefinitely once the bot is running.
  A bad token still aborts immediately in either mode.

`lumi doctor` and the Telegram `/doctor` command report which settings are in
effect.
