"""Two kinds of memory.

``MEMORY.md``
    Curated, human-readable long-term facts. The model sees all of it (up to a
    character cap) before every message and can append to it. You can edit it by
    hand; lines you wrote are never rewritten.

``data/history/<chat_id>.jsonl``
    Append-only transcripts. Prior turns are replayed into the prompt so the bot
    has conversational continuity, and the same file is a durable audit log.
    Read tail-first and never whole (see :meth:`History.tail`), because a
    transcript that has been running for a month is a transcript nobody wants
    in memory. A ``summary`` row records turns that were condensed out of the
    context window, so a restart resumes the thread instead of starting over.

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

#: Announced inside the prompt when the memory file did not fit. Wording
#: matters: the model has to know its memory is partial, or it will answer
#: "I don't have that" about a fact that is on disk and simply was not sent.
_ELIDED_BULLETS = (
    "<!-- ",
    " older remembered fact(s) elided to fit the memory budget; the ones below are the most recent -->",
)
_ELIDED_TAIL = "\n… [older memory elided]"

#: Room held back for the "these were elided" line, so the announcement itself
#: is never the thing that pushes the block over its budget.
_ELIDED_NOTE_ROOM = 120


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
    #: Cap on the block injected into the system prompt. 0 (the default) sends
    #: the whole file: this is the model's long-term memory, and cutting it is
    #: the same failure as cutting the conversation.
    max_chars: int = 0

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

        The cap exists because this file goes into the *system prompt*, which
        nothing in :mod:`lumi.context` is allowed to cut: an oversized memory
        would eat the window silently, with no log line and nothing to reclaim
        it later. So it is bounded here, on purpose, and sized so that a memory
        file full of facts cannot crowd out the conversation.

        What gets elided is the **oldest** of the model's own facts, not the
        newest: the recent ones are what it is still acting on, and dropping
        them in favour of ancient ones is how a memory becomes useless while
        still looking full. The hand-written parts of the file are the
        structure — headings, prose, links — and are kept as long as they fit.

        Whatever goes is announced in the text, because a model told it has
        "all" of its memory and quietly missing the first forty facts will
        confidently tell you it was never told. *limit* defaults to
        ``max_chars``; pass a smaller one when rendering for a chat.
        """
        self.ensure_loaded()
        budget = self.max_chars if limit is None else limit
        body = self.text.strip()
        if not body:
            return "_No long-term memory yet._"
        if budget <= 0 or len(body) <= budget:
            return body
        return self._elide(body, budget)

    def _elide(self, body: str, budget: int) -> str:
        """*body* cut to *budget*, keeping the structure and the newest facts.

        ``_managed_bounds`` hands back the span *inside* the markers, so
        ``before`` already ends with the start marker and ``after`` already
        begins with the end one; they are rejoined as they were.
        """
        bounds = self._managed_bounds(body)
        if bounds is None:
            # No markers to single anything out, so the head is what goes.
            return truncate(body, max(0, budget - 40), marker=_ELIDED_TAIL)

        start, end = bounds
        before, region, after = body[:start], body[start:end], body[end:]
        lines = region.split("\n")
        scaffolding = "\n".join(line for line in lines if line and not _BULLET_RE.match(line))
        bullets = [line for line in lines if _BULLET_RE.match(line)]

        # The hand-written parts are the structure — headings, prose, links —
        # and are kept as long as they fit; the facts get what is left.
        overhead = len(before) + len(after) + len(scaffolding) + _ELIDED_NOTE_ROOM
        if overhead >= budget:
            before = truncate(before, max(0, int(budget * 0.4)), marker=_ELIDED_TAIL)
            overhead = len(before) + len(after) + len(scaffolding) + _ELIDED_NOTE_ROOM
        allowance = max(80, budget - overhead)

        kept: list[str] = []
        used = 0
        for line in reversed(bullets):
            cost = len(line) + 1
            if used + cost <= allowance:
                kept.append(line)
                used += cost
                continue
            if not kept:
                # One fact longer than the whole allowance. It is still sent —
                # cut, not dropped — because a memory file whose newest fact
                # cannot fit is a file that has stopped being memory.
                kept.append(truncate(line, allowance, marker=_ELIDED_TAIL))
            break
        kept.reverse()
        dropped = len(bullets) - len(kept)

        note = f"{_ELIDED_BULLETS[0]}{dropped}{_ELIDED_BULLETS[1]}" if dropped else ""
        # Rejoined with explicit newlines: the region's own leading/trailing
        # blank lines are gone by now, and a model reading a wall of
        # `<!-- --><!-- -->` gains nothing from the tidiness being lost.
        out = "\n".join(
            part
            for part in (
                before.strip("\n"),
                scaffolding.strip("\n"),
                note.strip("\n"),
                "\n".join(kept),
                after.strip("\n"),
            )
            if part
        )
        # Last line of defence: the budget is the whole reason this method
        # exists, so a pathological file still has to obey it.
        return truncate(out, budget, marker=_ELIDED_TAIL) if len(out) > budget else out

    def stats(self) -> dict[str, Any]:
        self.ensure_loaded()
        managed = self.managed()
        return {            "path": str(self.path),
            "chars": len(self.text),
            "managed_count": len(managed),
            "bullet_count": sum(1 for ln in self.text.splitlines() if _BULLET_RE.match(ln)),
        }

    def prompt_size(self, limit: int | None = None) -> dict[str, Any]:
        """What the prompt block would look like, and what it cost to fit.

        Reported rather than inferred, because the one failure this design has
        to be honest about is a memory that is on disk but was not sent — which
        looks exactly like a memory that was never learned.
        """
        self.ensure_loaded()
        budget = self.max_chars if limit is None else limit
        rendered = self.for_prompt(limit)
        rendered_lines = set(rendered.splitlines())
        dropped = sum(
            1
            for line in self.text.splitlines()
            if _BULLET_RE.match(line) and line not in rendered_lines
        )
        return {
            "chars": len(self.text),
            "sent_chars": len(rendered),
            "budget": budget,
            "dropped": dropped,
            "over": bool(budget > 0 and len(self.text) > budget),
        }


# --------------------------------------------------------------------------- #
# JSONL transcripts
# --------------------------------------------------------------------------- #

#: Most bytes of a transcript read into memory at once. A 200k-token window is
#: ~800k characters of text, so a megabyte is a comfortable envelope for
#: replaying one — and it is the ceiling on what a read can cost, however large
#: the transcript has grown.
TAIL_READ_BYTES = 1_000_000

#: Per-message overhead, matching :data:`lumi.context.MESSAGE_OVERHEAD_TOKENS`.
#: Written out rather than imported so this module stands on its own.
_REPLAY_OVERHEAD_TOKENS = 4


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


@dataclass(slots=True)
class Dialogue:
    """What a replay found: the turns to send, plus the record of what came before them."""

    messages: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""

    def __bool__(self) -> bool:
        return bool(self.messages or self.summary)


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

    @staticmethod
    def _parse(blob: str, path: Path) -> list[Turn]:
        turns: list[Turn] = []
        for line in blob.splitlines():
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
        return turns

    def read(self, chat_id: int | str, limit: int | None = None) -> list[Turn]:
        """Every row. For reading a transcript deliberately, not for replay."""
        path = self.path_for(chat_id)
        if not path.is_file():
            return []
        try:
            blob = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.error("could not read history for chat %s: %s", chat_id, exc)
            return []
        turns = self._parse(blob, path)
        return turns[-limit:] if limit else turns

    def tail(self, chat_id: int | str, max_bytes: int = TAIL_READ_BYTES) -> list[Turn]:
        """The newest rows, reading only the end of the file.

        Replay only ever needs the tail, and a transcript is append-only, so
        seeking to the end beats loading a file that has been growing for
        months — and it bounds what one read can cost, which is the whole
        point: memory here is a budget, not a place to accumulate. The line the
        seek lands inside is discarded; it is both incomplete and the oldest
        thing wanted.

        A *max_bytes* of 0 or less means "the default", not "all of it" — the
        default is the ceiling that keeps this bounded, and a caller that wants
        a deliberate whole-file read has :meth:`read` for that.
        """
        path = self.path_for(chat_id)
        try:
            size = path.stat().st_size
        except OSError:
            return []
        if size == 0:
            return []
        limit = max_bytes if max_bytes > 0 else TAIL_READ_BYTES
        try:
            with path.open("rb") as handle:
                if limit < size:
                    handle.seek(size - limit)
                    handle.readline()  # the partial line at the cut
                blob = handle.read()
        except OSError as exc:
            log.error("could not read history for chat %s: %s", chat_id, exc)
            return []
        return self._parse(blob.decode("utf-8", errors="replace"), path)

    def recent_dialogue(
        self,
        chat_id: int | str,
        *,
        max_messages: int | None = None,
        budget_tokens: int | None = None,
        chars_per_token: float = 3.5,
    ) -> Dialogue:
        """The newest user/assistant turns, packed to fit.

        Packed backwards from the end of the transcript and cut where the budget
        runs out — the opposite of taking a fixed window of turns and hoping. A
        fixed window is forty messages on a quiet day and four on a busy one,
        and the busy day is exactly when the past matters.

        Tool chatter is deliberately excluded: the model re-derives tool state
        from the live loop, and replaying it is pure cost. So is anything the
        newest ``summary`` row already stands in for, which is how a condensed
        record survives a restart without the turns behind it going twice.
        """
        rows = self.tail(chat_id)

        summary = ""
        start = 0
        for index in range(len(rows) - 1, -1, -1):
            if rows[index].role == "summary" and rows[index].content.strip():
                summary = rows[index].content.strip()
                start = index + 1
                break

        dialogue = [
            {"role": t.role, "content": t.content}
            for t in rows[start:]
            if t.role in {"user", "assistant"} and t.content.strip()
        ]
        # Drop the orphan leading message: an answer with no question in front
        # of it reads as the model talking to itself.
        while dialogue and dialogue[0]["role"] == "assistant":
            dialogue.pop(0)
        if not dialogue:
            return Dialogue(summary=summary)

        divisor = max(1.0, chars_per_token)
        kept: list[dict[str, str]] = []
        used = 0
        for message in reversed(dialogue):
            cost = int(len(message["content"]) / divisor) + _REPLAY_OVERHEAD_TOKENS
            if kept and (
                (budget_tokens is not None and used + cost > budget_tokens)
                or (max_messages is not None and len(kept) >= max_messages)
            ):
                break
            kept.append(message)
            used += cost
        kept.reverse()
        return Dialogue(messages=kept, summary=summary)

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
        """Size and shape of a transcript, without reading the whole thing."""
        path = self.path_for(chat_id)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return {
            "path": str(path),
            "turns": len(self.tail(chat_id)),
            "bytes": size,
            "truncated": size > TAIL_READ_BYTES,
        }
