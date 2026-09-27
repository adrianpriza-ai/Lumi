"""Parse comma-separated API keys out of environment variables.

One key is the common case; a comma-separated list is a rotation pool. Several
parts of Lumi need this — the LLM client, the new Context7 tool, and anything
else that talks to a third-party API behind a Bearer token. Centralising the
parser keeps the rules in one place:

- Separators: ``,`` ``;`` and a literal newline, so a long list can be written
  one key per line in ``.env``.
- Whitespace around each entry is stripped.
- Surrounding quotes (``"sk-…"`` or ``'sk-…'``) are stripped, which is what you
  end up with when a value is wrapped to keep trailing punctuation safe.
- Blank entries and duplicates collapse. A trailing comma in a hand-edited
  ``.env`` is not a broken key.

This module has no I/O of its own — it takes a string and returns a list — so it
is trivially testable and safe to import from anywhere.
"""

from __future__ import annotations

import re

#: Characters accepted between keys. ``,`` is the documented one; ``;`` and a
#: literal newline are accepted so a long list can be split across lines.
_SEPARATORS = (",", ";", "\n")

#: A compiled ``|`` alternation of escaped separators, ready for ``re.split``.
_SEPARATOR_PATTERN = re.compile("|".join(re.escape(sep) for sep in _SEPARATORS))


def parse_key_list(raw: str) -> list[str]:
    """Turn *raw* into a clean list of keys, in the order they were written.

    Empty input returns ``[]``; whitespace, quotes, blanks and duplicates are
    filtered out. The original order is preserved so that the *first* key in
    ``OPENAI_API_KEY`` (or similar) stays the primary one — every existing
    caller already depends on that.
    """
    if not raw:
        return []
    keys: list[str] = []
    for part in _SEPARATOR_PATTERN.split(raw):
        key = part.strip().strip("\"'")
        if key and key not in keys:
            keys.append(key)
    return keys


def parse_env_var(name: str) -> list[str]:
    """Parse the value of *name* out of ``os.environ``.

    Returns ``[]`` when the variable is unset, empty, or all-whitespace.
    Equivalent to ``parse_key_list(os.environ.get(name) or "")``.
    """
    import os

    return parse_key_list(os.environ.get(name) or "")


__all__ = ["parse_key_list", "parse_env_var"]