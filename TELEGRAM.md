# Telegram

[← Back to README](README.md) · [CONFIGURATION.md](CONFIGURATION.md) · [TOOLS.md](TOOLS.md)

Two things about the Telegram side trip people up: when the bot is allowed to answer in a group, and getting a TCP connection to `api.telegram.org` at all. Both are configuration; nothing here needs a code change.

## Group chats

In a private chat the bot replies to every owner message. In a group or
supergroup it follows `bot.group_reply_mode` (default `mention`):

- **`mention`** — the bot only responds when it is **mentioned**
  (`@Lumi_a_bot`), **replied to**, or sent a **slash command that targets it**
  (`/start@Lumi_a_bot`). Plain `/start` in a group is left alone because Telegram
  routes commands without `@botname` to whichever bot claims them first.
- **`always`** — replies to every owner message regardless of chat type.
- **`off`** — never replies in groups; private chats still work as normal.

The detection covers three signals:

1. **Reply to a message from the bot** — the replied-to message has to be
   *this* bot's (matched by username). A reply to another person, or to
   another bot in the same room, leaves the bot alone.
2. **Slash command with `@botname`** — `cmd.split("@", 1)[1]` must equal the
   bot's username (case-insensitive).
3. **`@botname` mention in entities** — `MessageEntity.MENTION` with the
   bot's username, or `MessageEntity.TEXT_MENTION` pointing at the bot.

A command sent as a *reply* to the bot's own message counts too — quoting
`/start` under the bot's answer addresses the bot, so the `@botname` suffix
is not needed there. Telegram hands a replied-to command to the bot whose
message it quotes.

Mention offsets in a group are read the way Telegram counts them (UTF-16 code
units), so an emoji in front of the mention does not throw the detection off.

### Replies carry their quote

When a message that reaches the bot is a reply, the message it quotes is
prepended to the prompt before the model sees it:

```
[the message this replies to, from Other:
"""
an earlier human message
"""
]

@Lumi_a_bot try to fix this
```

Without that the model only ever saw "try to fix this" — the quoted message
was addressed to a person, so it is not part of the bot's transcript, and
"this" had nothing to point at. A quote with no text (a photo, a sticker) is
named as what it was instead. Messages that are not replies reach the model
unchanged.

Replies themselves chain — a reply to a reply to a reply — and one link is
rarely the whole story ("no it isn't" answers the message *above* the quote).
So the chain is walked too, up to **5 links**, rendered oldest first:

```
[reply chain: the 2 message(s) this replies to, oldest first:
(from Other):
"""
the deploy token is expired
"""

(from Owner):
"""
which token though
"""
]

@Lumi_a_bot fix it
```

A chain deeper than 5 is cut at the cap, and the prompt says so rather than
pretending the quote started there.

### The room when the bot is silent

Mention mode means the bot only joins a conversation it is addressed in — so
the model's history is a series of interruptions with the talk between them
missing. Every group message the bot *drops* is therefore kept in a small
in-memory log (`lumi/chatlog.py`, 10 messages per chat) that the
`more_context` tool reads back: up to 10 messages per call, `offset` up to 9,
and a call that would exceed ~30k tokens is refused with the remedy ("call
again with a smaller limit") instead of being truncated.

What goes in: text, from anyone, when the bot chose not to answer — the
mention rule and the stranger gate both log before they stay silent. What
does not: anything addressed to the bot (already in the conversation),
private chats (every message there already became a turn), photos, files and
any other attachment (caption or not — the log is conversation, not an
inbox), other bots' lines, anything older than the ring buffer, and anything
at all past a restart — the log is working context, not a record.

**Telegram's group privacy setting decides whether this sees anything.**
With privacy mode on, Telegram never delivers unaddressed messages to a bot
in the first place, so the log stays empty and the tool reports exactly that.
To let the bot read the room: @BotFather → `/mybots` → your bot → Bot
Settings → Group Privacy → Disable.

### Who the bot answers

Two different questions, two different settings:

- **Who may talk to the bot** — the whitelist. The owner, every id in
  `bot.whitelisted_users`, and every member of a group in
  `bot.whitelisted_groups`. A whitelisted group means the bot talks to *the
  room*: anyone in it can use the bot without being added to the user list.
- **When the bot answers** — `bot.group_reply_mode`. A whitelisted group is
  still a group, so the mention rule applies there too: members have to
  address the bot, and unaddressed chatter is ignored.

Anyone else who addresses the bot in a group gets one line — "this bot is
private, your id is not whitelisted" — and nothing more. "Addresses the bot"
covers a mention, a reply to one of the bot's messages, and a command aimed at
it; anything else a stranger says in the room is ignored silently, so the bot
does not announce itself in every conversation the room happens to have.

`bot.always_reply_chats` is the one setting that changes *when* the bot
answers in a specific chat — it lifts the mention rule for that chat id
(e.g. a private "Lumi lab"):

```toml
[bot]
group_reply_mode = "mention"
always_reply_chats = ["-1001234567890"]
```

Watch the tools when you whitelist a group: every member of that group can
then reach the bot's shell and files through it.

## Arriving in a group

Adding the bot to a group produces two events, and the bot answers neither:

- the `my_chat_member` update (its own membership changed) — logged, silent;
- the group's service message ("X added Lumi_a_bot") — a service notice, not
  something to reply to.

Service messages in general (someone joined, someone left, the chat was
renamed, a message was pinned) are ignored: the bot only speaks when a person
actually says something to it.

## Memory per chat

Long-term memory is scoped to the chat it was learned in:
`data/memory/<chat_id>/MEMORY.md`. Your private chat gets its own file (your
telegram id), each group gets one shared file for the group, and none of them
see each other. `/memory`, `/remember` and `/forget` in a chat act on that
chat's file — run them in a group to manage the group's memory. From the
terminal, `lumi memory` edits your own scope and `lumi memory --scope
<chat_id>` reaches any chat's, including a group's negative id.

## Reaching api.telegram.org

If the bot dies at startup with `telegram.error.TimedOut` / `ConnectTimeout`, the TCP connection to `api.telegram.org` failed. Three knobs under `[bot]` address it:

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
