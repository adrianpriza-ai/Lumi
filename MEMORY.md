# Memory

Long-term memory is **one file per chat**, not one file for the bot:

```
data/memory/<chat_id>/MEMORY.md
```

A private chat's id is the user's id, a group's id is the group's — so every
conversation has its own memory, and nothing the bot learns in one chat is
visible in another. The bot reads the current chat's file before every message
and can append to it via the `memory` tool; `/memory`, `/remember` and
`/forget` in a chat act on that chat's file. From the terminal, `lumi memory`
edits your own scope (shared with your private chat when `TELEGRAM_OWNER_ID`
is set), and `lumi memory --scope <chat_id>` opens any other chat's file.

You can edit any of these files by hand — the bot never rewrites lines you
wrote, it only appends inside the managed region.

Format: one fact per line, as a markdown bullet. Keep them short and
declarative; a line that starts with `-` and a date is treated as a managed
memory (see `/forget`). Anything else is free-form context.

## This file

The `MEMORY.md` at the project root is the **legacy seed**. The first time the
owner's scope is opened, its content is copied into
`data/memory/<owner_id>/MEMORY.md` (or `data/memory/cli/` when no
`TELEGRAM_OWNER_ID` is configured); after that the root file is not read. It is
kept as the template below and nothing more.

<!-- lumi:managed:start -->
<!-- Managed memories are appended below this line and removed from the top. -->
<!-- lumi:managed:end -->

## Context

_Empty. Free-form notes, project background, links, whatever is useful._
