"""The environment child processes are allowed to inherit.

Two places start a program whose output eventually reaches the model: the
``run_shell`` tool and a locally spawned MCP server. Letting either inherit the
bot's environment hands over ``OPENAI_API_KEY`` and ``TELEGRAM_BOT_TOKEN``
without a single command naming them, so both build their environment here from
the same allowlist rather than each keeping its own copy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Environment variables that pass through to a child process.
ENV_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TERM",
        "SHELL",
        "USER",
        "LOGNAME",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "TMPDIR",
        # Build tooling that legitimately needs to know where it is.
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "NODE_PATH",
        "PYTHONPATH",
        "GOPATH",
        "GOROOT",
    }
)

#: Anything matching this is stripped, whatever its name.
#:
#: The scrub is allowlist-first (see :data:`ENV_PASSTHROUGH`): a variable that
#: is not on the list is dropped whether or not its name looks secret. This
#: pattern is the second gate, for the names that *do* slip onto the list by
#: accident, and for the names that hold a secret without saying so.
#:
#: Word-ish boundaries, not a bare substring: "PASSWORD" and "MY_PASS" are
#: secrets, "PASSPORT" and "PASSED" are not, and "MONKEY" is not a key.
SECRET_PATTERN = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:KEYS?|TOKENS?|SECRETS?|PASSWORDS?|PASSWD|PASS|PASSPHRASE|CREDENTIALS?|"
    r"AUTH(?:ORIZATION)?|SESSION(?:S|_ID)?|COOKIE|DATABASE_URL|DB_URL|DSN|"
    r"WEBHOOK(?:_URL)?|PRIVATE(?:_KEY)?|SIGNATURE|CERT(?:IFICATE)?|"
    # The run-together spellings, which have no separator to break on.
    r"APIKEY|SECRETKEY|ACCESSTOKEN|AUTHTOKEN|PRIVATEKEY)"
    r"(?:[^A-Za-z0-9]|$)",
    re.IGNORECASE,
)


def scrubbed_env(
    *,
    home: Path | str | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """A minimal, secret-free environment for a child process.

    Only :data:`ENV_PASSTHROUGH` survives, so a variable is dropped by default
    rather than by whether its name looks secret. *home* pins ``HOME`` when the
    caller wants it (the shell keeps it inside the project); *overrides* are
    values the owner configured explicitly, such as an MCP server's ``env``
    block, and are applied last.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ENV_PASSTHROUGH and not SECRET_PATTERN.search(key)
    }
    if home is not None:
        env["HOME"] = str(home)
    if overrides:
        env.update(overrides)
    return env


__all__ = ["ENV_PASSTHROUGH", "SECRET_PATTERN", "scrubbed_env"]
