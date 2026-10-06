"""PERSONALITY.md — the bot's voice, as a plain-text system prompt.

The file is the prompt. There is no template language, no placeholders to
remember, and nothing to escape. Write it like you'd brief a person.

Reloaded on mtime change, so editing the file mid-conversation takes effect on
the next ``/reload`` (or automatically, see :meth:`Personality.load`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .util.log import get_logger

log = get_logger(__name__)

FALLBACK_PERSONALITY = """\
You are Lumi, a concise and direct assistant. Answer first, explain only when
the explanation earns its place. Say when you are uncertain instead of guessing.
"""


@dataclass(slots=True)
class Personality:
    """Cached contents of PERSONALITY.md."""

    path: Path
    text: str
    mtime: float = 0.0

    @classmethod
    def load(cls, path: Path) -> Personality:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            log.warning("PERSONALITY.md not found at %s; using the fallback voice", path)
            return cls(path=path, text=FALLBACK_PERSONALITY.strip())

        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            log.error("could not read %s: %s", path, exc)
            text = FALLBACK_PERSONALITY.strip()
            mtime = 0.0

        if not text:
            text = FALLBACK_PERSONALITY.strip()

        return cls(path=path, text=text, mtime=mtime)

    def reload(self) -> str:
        """Re-read from disk unconditionally. Returns a short status line."""
        self.text = Personality.load(self.path).text
        self.mtime = self.path.stat().st_mtime if self.path.is_file() else 0.0
        return f"personality reloaded ({len(self.text)} chars)"

    def __str__(self) -> str:
        return self.text
