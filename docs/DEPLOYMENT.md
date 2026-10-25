# Deployment

[← Back to README](../README.md) · [TOOLS.md](TOOLS.md) · [TELEGRAM.md](TELEGRAM.md) · [CONFIGURATION.md](CONFIGURATION.md)

Lumi is a Telegram bot that runs commands, reads files and fetches pages on
behalf of the person who set it up. The application-layer controls — the shell
safety engine, the owner gate on approvals, the environment scrub, the
secret-file read guard — live in the code and are described in
[TOOLS.md](../TOOLS.md#shell-safety).

They are guardrails, not a sandbox. Everything below the application layer is
the operator's job, and nothing in the repository does it for you. This page
lists what the controls assume about the deployment, so the gap between "the
code is careful" and "the machine is careful" is written down.

## What the controls assume

| Assumption | What breaks without it |
|-|-|
| The bot runs as an unprivileged user, not root | A command that escapes the workspace writes as root. `rm -rf` inside the home directory is a deny, but a bug or a bypass is not a second line of defence. |
| The bot's home directory is not the owner's home directory | `tools.shell.home = "inherit"` widens what commands can reach. Lumi warns at startup when this is set. |
| Only the bot's user can read `.env` and `data/whitelist.json` | The env scrub hides secrets from commands the bot runs; it cannot hide them from another process running as the same user. |
| Outbound network access is restricted to what the bot needs | A command that exfiltrates a secret needs somewhere to send it. The allowlist is the Telegram API, the model endpoint, the web providers, and `api.telegram.org`. |
| The project directory is the bot's own | The shell's write root defaults to the whole project. A checkout that also holds unrelated files gives the model more to write to. |

## The state files worth protecting

Everything the bot persists lives under the project directory:

- `.env` — the credentials for Telegram, the model provider and the web
  providers. The shell's environment scrub keeps these out of anything a
  command runs, and the read guard refuses to read the file at all. Neither
  helps against another process running as the same user.
- `data/whitelist.json` — who may talk to the bot. It is written by
  `/whitelist_add_user` and read at startup. It sits inside the shell's
  default write root, but both the shell and the files tool refuse to write
  it, so a group member cannot add themselves and have it survive a restart.
- `data/` — per-chat transcripts and long-term memory (`MEMORY.md` per
  chat). These contain whatever was said in the chat.
- `workspace/` — the shell's working directory and uploaded files.

None of these are encrypted at rest, and `data/` is not gitignored by
default in a fresh checkout — check `.gitignore` before committing anything
from a live instance.

## A container or a dedicated user

The simplest way to satisfy the table above is to run the bot as a user that
owns nothing else:

```sh
# a dedicated account, no login shell, no home of its own worth keeping
sudo useradd --system --create-home --home-dir /var/lib/lumi --shell /usr/sbin/nologin lumi

# the checkout, owned by that account alone
sudo chown -R lumi:lumi /opt/lumi
sudo -u lumi -H /opt/lumi/.venv/bin/python -m lumi run
```

A container works the same way, with the added benefit that the filesystem
outside the mounted project is simply not there:

```sh
docker run --rm -it \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --mount type=bind,src=/opt/lumi,dst=/opt/lumi \
  --env-file /opt/lumi/.env \
  --network bridge \
  /opt/lumi/.venv/bin/python -m lumi run
```

Note what the container does *not* do: it does not stop the bot from reaching
the network, and it does not stop a command from writing to the mounted
project. Those are the network policy and the write zone, which are separate
knobs.

## Egress

The bot needs to reach the Telegram Bot API and whatever model and web
provider you configured. A deny-by-default egress policy is the strongest
control available for the exfiltration path — a fetched page that talks the
model into `web.fetch` has nowhere to send what it stole if the destination
is not reachable.

The hosts to allow depend on your configuration: `api.telegram.org` always,
plus your model endpoint and any web provider you enable. If you run a local
model, nothing outbound is needed for it at all.

## A systemd unit

A unit is the easy way to keep the process running with the right
environment. It is a starting point, not a hardened profile — read the
`NoNewPrivileges` and `ProtectHome` lines against your own setup before
using it.

```ini
[Unit]
Description=Lumi Telegram agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lumi
Group=lumi
WorkingDirectory=/opt/lumi
ExecStart=/opt/lumi/.venv/bin/python -m lumi run
Restart=on-failure
RestartSec=10

# The bot reads its own checkout and writes to data/ and workspace/ only.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/opt/lumi/data /opt/lumi/workspace
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
```

## Checking a deployment

`lumi doctor` checks the configuration, the model endpoint and the web
providers. Two things it cannot check are worth checking by hand:

- `ls -l .env data/whitelist.json` — both should be readable only by the
  account the bot runs as.
- `curl` from a shell command — if the bot's user can reach arbitrary hosts,
  an exfiltration path exists regardless of what the safety engine refuses.
