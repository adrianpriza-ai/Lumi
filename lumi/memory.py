"""Two kinds of memory.

``MEMORY.md``
    Curated, human-readable long-term facts. The model sees all of it (up to a
    character cap) before every message and can append to it. You can edit it by
    hand; lines you wrote are never rewritten.

``data/history/<chat_id>.jsonl``
    Append-only transcripts. Prior turns are replayed into the prompt so the bot
    has conversational continuity, and the same file is a durable audit log.

Managed memories live between HTML marker comments so ``/forget`` can pop the
most recent ones without touching anything you typed yourself.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .util.log import get_logger
from .util.text import truncate

log = get_logger(__name__)

MANAGED_START = "<!-- lumi:managed:start -->"
MANAGED_END = "<!-- lumi:managed:end -->"
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_DATE_PREFIX_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}(?:[ T][\d:]{5})?\]\s*")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# MEMORY.md
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MemoryFile:
    """A markdown file with a managed, append-only bullet region."""

    path: Path
    max_chars: int = 6000

    _text: str = field(default="", init=False, repr=False)
    _mtime: float = field(default=0.0, init=False, repr=False)

    def load(self) -> str:
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = 0.0
        if not self.path.is_file():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._text = f"# Memory\n\n## Facts\n\n{MANAGED_START}\n{MANAGED_END}\n"
            self._write(self._text)
            return self._text
        try:
            self._text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("could not read %s: %s", self.path, exc)
            self._text = ""
        return self._text

    def _write(self, text: str) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)  # atomic; a crash mid-write cannot corrupt memory

    @property
    def text(self) -> str:
        if not self._text:
            self.load()
        return self._text

    def ensure_loaded(self) -> None:
        try:
            current = self.path.stat().st_mtime
        except OSError:
            current = 0.0
        if current != self._mtime:
            log.debug("MEMORY.md changed on disk; reloading")
            self.load()

    def changed_on_disk(self) -> bool:
        try:
            return self.path.stat().st_mtime != self._mtime
        except OSError:
            return False

    # -- managed region ---------------------------------------------------- #

    def _managed_bounds(self, text: str) -> tuple[int, int] | None:
        start = text.find(MANAGED_START)
        end = text.find(MANAGED_END)
        if start == -1 or end == -1 or end < start:
            return None
        return start + len(MANAGED_START), end

    def _managed_lines(self, text: str | None = None) -> list[str]:
        text = self.text if text is None else text
        bounds = self._managed_bounds(text)
        if bounds is None:
            return []
        start, end = bounds
        return [ln for ln in text[start:end].splitlines() if _BULLET_RE.match(ln)]

    def managed(self) -> list[str]:
        """Managed memories, newest last, with the date prefix stripped."""
        self.ensure_loaded()
        out = []
        for line in self._managed_lines():
            match = _BULLET_RE.match(line)
            assert match is not None
            out.append(_DATE_PREFIX_RE.sub("", match.group(1)).strip())
        return [item for item in out if item]

    def remember(self, fact: str, *, source: str = "bot") -> str:
        """Append one fact. Returns a short confirmation."""
        fact = " ".join(fact.split())
        if not fact:
            return "nothing to remember (the text was empty)"
        if len(fact) > 500:
            fact = truncate(fact, 500, marker="…")
        if fact.casefold() in {existing.casefold() for existing in self.managed()}:
            return f"already remembered: {fact}"

        self.ensure_loaded()
        text = self.text
        line = f"- [{_today()}] {fact}"

        bounds = self._managed_bounds(text)
        if bounds is None:
            # No markers in the file: create the managed region at the end.
            self._write(text.rstrip() + f"\n\n{MANAGED_START}\n{line}\n{MANAGED_END}\n")
        else:
            start, end = bounds
            self._write(text[:start] + text[start:end].rstrip("\n") + f"\n{line}\n" + text[end:])

        self.load()
        log.info("remembered (%s): %s", source, truncate(fact, 80, marker="…"))
        return f"remembered: {fact}"

    def forget(self, count: int = 1) -> list[str]:
        """Drop the *count* most recent managed memories. Returns what was removed."""
        self.ensure_loaded()
        text = self.text
        bounds = self._managed_bounds(text)
        if bounds is None:
            return []
        start, end = bounds
        # split("\n") rather than splitlines(): keeping the empty leading and
        # trailing elements preserves the block's exact newline shape, so the
        # markers never gain or lose a blank line.
        lines = text[start:end].split("\n")
        positions = [i for i, line in enumerate(lines) if _BULLET_RE.match(line)]
        if not positions or count <= 0:
            return []
        drop = set(positions[-count:])
        # Only bullet lines are ours to remove; blank lines and the explanatory
        # comment inside the markers are the human's scaffolding and stay put.
        kept = [line for i, line in enumerate(lines) if i not in drop]
        self._write(text[:start] + "\n".join(kept) + text[end:])
        self.load()
        removed = [
            _DATE_PREFIX_RE.sub("", _BULLET_RE.match(lines[i]).group(1)).strip()  # type: ignore[union-attr]
            for i in sorted(drop)
        ]
        log.info("forgot %d memories", len(removed))
        return removed

    def search(self, query: str, limit: int = 10) -> list[str]:
        """Case-insensitive keyword search across the whole file."""
        self.ensure_loaded()
        needles = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]
        if not needles:
            return self.managed()[-limit:]
        hits = []
        for line in self.text.splitlines():
            if not _BULLET_RE.match(line):
                continue
            haystack = line.lower()
            if any(n in haystack for n in needles):
                hits.append(_DATE_PREFIX_RE.sub("", _BULLET_RE.match(line).group(1)).strip())  # type: ignore[union-attr]
        return hits[-limit:]

    def for_prompt(self, limit: int | None = None) -> str:
        """The block handed to the model, capped and clearly marked when cut.

        *limit* defaults to ``max_chars``; pass a smaller one when rendering for
        a chat, where the file is being shown rather than injected.
        """
        self.ensure_loaded()
        budget = self.max_chars if limit is None else limit
        body = self.text.strip()
        if not body:
            return "_No long-term memory yet._"
        if len(body) <= budget:
            return body
        # Keep the managed (recent) region, drop the oldest bulk to stay in budget.
        keep = max(0, budget - 80)
        return truncate(body, keep, marker="\n… [older memory elided]")

    def stats(self) -> dict[str, Any]:
        self.ensure_loaded()
        managed = self.managed()
        return {            "path": str(self.path),
            "chars": len(self.text),
            "managed_count": len(managed),
            "bullet_count": sum(1 for ln in self.text.splitlines() if _BULLET_RE.match(ln)),
        }


# --------------------------------------------------------------------------- #
# JSONL transcripts
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Turn:
    ts: str
    role: str
    content: str
    session: str = ""
    name: str = ""
    tool: str = ""

    def to_json(self) -> str:
        payload: dict[str, Any] = {"ts": self.ts, "role": self.role, "content": self.content}
        for key in ("session", "name", "tool"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


class History:
    """Append-only JSONL transcript, one file per chat."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, chat_id: int | str) -> Path:
        return self.directory / f"{chat_id}.jsonl"

    def append(
        self,
        chat_id: int | str,
        role: str,
        content: str,
        *,
        session: str = "",
        name: str = "",
        tool: str = "",
    ) -> Turn:
        self.directory.mkdir(parents=True, exist_ok=True)
        turn = Turn(
            ts=_now(), role=role, content=content, session=session, name=name, tool=tool
        )
        try:
            with self.path_for(chat_id).open("a", encoding="utf-8") as handle:
                handle.write(turn.to_json() + "\n")
        except OSError as exc:
            log.error("could not write history for chat %s: %s", chat_id, exc)
        return turn

    def read(self, chat_id: int | str, limit: int | None = None) -> list[Turn]:
        path = self.path_for(chat_id)
        if not path.is_file():
            return []
        turns: list[Turn] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("skipping malformed history line in %s", path.name)
                        continue
                    turns.append(
                        Turn(
                            ts=str(raw.get("ts", "")),
                            role=str(raw.get("role", "user")),
                            content=str(raw.get("content", "")),
                            session=str(raw.get("session", "")),
                            name=str(raw.get("name", "")),
                            tool=str(raw.get("tool", "")),
                        )
                    )
        except OSError as exc:
            log.error("could not read history for chat %s: %s", chat_id, exc)
            return []
        return turns[-limit:] if limit else turns

    def recent_dialogue(self, chat_id: int | str, turns: int) -> list[dict[str, str]]:
        """The last *turns* user/assistant pairs, ready to drop into a prompt.

        Tool chatter is deliberately excluded: replaying it wastes context and
        the model re-derives tool state from the live loop anyway.
        """
        raw = self.read(chat_id)
        dialogue = [
            {"role": t.role, "content": t.content}
            for t in raw
            if t.role in {"user", "assistant"} and t.content.strip()
        ]
        # keep the window balanced: drop the orphan leading message
        while dialogue and dialogue[0]["role"] == "assistant":
            dialogue.pop(0)
        return dialogue[-turns * 2 :]

    def new_session(self, chat_id: int | str) -> str:
        """Start a new session id; used to segment a transcript after /reset.

        Two resets in the same second must still get distinct ids, otherwise the
        audit log cannot tell the sessions apart.
        """
        return f"s{int(time.time())}-{uuid4().hex[:6]}"

    def clear(self, chat_id: int | str) -> bool:
        path = self.path_for(chat_id)
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            log.error("could not clear history for %s: %s", chat_id, exc)
            return False

    def stats(self, chat_id: int | str) -> dict[str, Any]:
        turns = self.read(chat_id)
        return {
            "path": str(self.path_for(chat_id)),
            "turns": len(turns),
            "bytes": self.path_for(chat_id).stat().st_size
            if self.path_for(chat_id).is_file()
            else 0,
        }
