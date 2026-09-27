"""Several API keys, one client.

``OPENAI_API_KEY`` accepts a comma-separated list, so a setup that has hit a
rate limit or exhausted one account's quota can keep going without restarting
anything:

.. code-block:: bash

    OPENAI_API_KEY=sk-...90, sk-...91, sk-...92

Three ways to spend them, chosen with ``llm.key_strategy``:

``fallback``     (default) Use the first key. Only move to the next one when
                 this call fails in a way another key might survive — 401, 403,
                 429, a 5xx, a dropped connection. A 400 is not the key's
                 fault, so it fails immediately. This is what you want when the
                 keys are backups for one account.
``round_robin``  Spread calls evenly. A good default when the keys belong to
                 different accounts and you are load-balancing your own quota.
``random``       Pick at random. Same effect as round-robin in aggregate, but
                 no shared cursor to serialise on.

A key that fails with a retryable error is parked for :data:`DEFAULT_COOLDOWN`
seconds so the next message does not hammer a dead credential. It is retried
once the cooldown expires, which is how a rate limit that clears itself
recovers without operator involvement. If every key is parked, the one that has
been cooling the longest is probed anyway, so a single-key setup behaves exactly
as it did before this module existed.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Iterable, Sequence

from ..util.log import get_logger

log = get_logger(__name__)

#: Accepted values for ``llm.key_strategy``.
STRATEGIES = ("fallback", "round_robin", "random")

#: How long a failed key is skipped before it is tried again.
DEFAULT_COOLDOWN = 60.0

#: HTTP statuses that are this key's problem, not the request's.
KEY_STATUSES = frozenset({401, 403, 408, 409, 429})

#: Statuses a different key would fail on identically.
FATAL_STATUSES = frozenset({400, 404, 405, 413, 422})

#: Substrings that mean "try another key" when the exception carries no status
#: code, which is the case for some OpenAI-compatible proxies.
RETRYABLE_HINTS = (
    "invalid_api_key",
    "incorrect api key",
    "invalid key",
    "unauthorized",
    "forbidden",
    "rate limit",
    "rate_limit",
    "too many requests",
    "insufficient_quota",
    "quota",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "server error",
    " 500",
    " 502",
    " 503",
    " 504",
)

_STATUS_RE = re.compile(r"\b([1-5]\d{2})\b")


def is_retryable(exc: BaseException) -> bool:
    """Whether *exc* looks like a problem with this key rather than the request.

    Duck-typed on purpose: the OpenAI SDK's exception classes are not imported
    here, so this works for any HTTP client an adapter might use and stays
    testable without a network.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status in FATAL_STATUSES:
            return False
        return status in KEY_STATUSES or status >= 500

    name = type(exc).__name__
    if any(marker in name for marker in ("BadRequest", "NotFound", "Unprocessable", "Conflict")):
        return False
    if any(marker in name for marker in ("Connection", "Timeout", "RateLimit", "Authentication", "Permission", "Server")):
        return True

    message = str(exc).lower()
    if any(hint in message for hint in RETRYABLE_HINTS):
        return True
    found = _STATUS_RE.search(message)
    if found:
        code = int(found.group(1))
        return code not in FATAL_STATUSES and (code in KEY_STATUSES or code >= 500)
    return False


def mask(key: str) -> str:
    """A key safe to put in a log line.

    The first three and last four characters, which is enough to tell
    ``sk-proj-aaa…b91c`` from ``sk-proj-bbb…b92d`` in a warning. Anything short
    enough that those would overlap is described by length instead, so a
    four-character key never reveals itself.
    """
    if not key:
        return "(empty)"
    if len(key) < 8:
        return f"(len {len(key)})"
    return f"{key[:3]}…{key[-4:]}"


class KeyPool:
    """Chooses which key to send, and remembers which ones are misbehaving.

    Not thread-safe, and does not need to be: one bot drives one event loop.
    """

    def __init__(
        self,
        keys: Iterable[str],
        strategy: str = "fallback",
        *,
        cooldown: float = DEFAULT_COOLDOWN,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.keys = [key for key in keys if key]
        wanted = (strategy or "fallback").strip().lower()
        self.strategy = wanted if wanted in STRATEGIES else "fallback"
        if self.strategy != wanted:
            log.warning(
                "unknown llm.key_strategy %r; using %r. Known: %s",
                strategy, self.strategy, ", ".join(STRATEGIES),
            )
        self.cooldown = cooldown
        self._clock = clock
        self._cursor = 0
        #: key -> the moment it was last failed, for the cooldown.
        self._parked_at: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def pick(self, exclude: Sequence[str] = ()) -> str | None:
        """The key to use next, or None when every key has been tried.

        *exclude* is the set already attempted in the current call, which is what
        stops a failing request from retrying the same dead key forever.
        """
        if not self.keys:
            return None
        banned = set(exclude)
        ready = [key for key in self._available() if key not in banned]
        if not ready:
            # All cooling down, or all tried already. Probe whichever has been
            # out of rotation longest rather than giving up while a key exists.
            untried = [key for key in self.keys if key not in banned]
            if not untried:
                return None
            return min(untried, key=lambda key: self._parked_at.get(key, float("-inf")))

        if self.strategy == "round_robin":
            # A cursor, not a pointer: when a key is parked the list shrinks and
            # the rotation stays even across the surviving keys.
            chosen = ready[self._cursor % len(ready)]
            self._cursor += 1
        elif self.strategy == "random":
            chosen = random.choice(ready)
        else:  # fallback
            chosen = ready[0]
        return chosen

    def report(self, key: str, *, ok: bool, retryable: bool = True) -> None:
        """Record the outcome of a call made with *key*."""
        if not ok and retryable:
            self._parked_at[key] = self._clock()
            log.warning(
                "key %s failed; parked for %.0fs", mask(key), self.cooldown
            )
        elif ok:
            # A success clears the cooldown, so a key that was rate-limited
            # gets back into rotation the moment it works again.
            self._parked_at.pop(key, None)

    def describe(self) -> str:
        """One line for logs and /doctor."""
        if not self.keys:
            return "no keys"
        if len(self.keys) == 1:
            return "1 key"
        return f"{len(self.keys)} keys ({self.strategy}, {len(self._available())} ready)"

    def _available(self) -> list[str]:
        now = self._clock()
        cutoff = now - self.cooldown
        return [key for key in self.keys if self._parked_at.get(key, float("-inf")) <= cutoff]


__all__ = [
    "DEFAULT_COOLDOWN",
    "KEY_STATUSES",
    "STRATEGIES",
    "KeyPool",
    "is_retryable",
    "mask",
]
