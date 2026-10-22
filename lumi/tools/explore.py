"""The read-only exploration tools: ``read_file``, ``list_files`` and ``grep``.

Deliberately narrower than the ``files`` tool, which stays the write / edit /
upload tool. These three are what a model reaches for when it just needs to
*look* at the project: they never ask for approval, never write anything, and
are read only inside the project directory (plus the same handful of system
paths ``files`` whitelists).

The one hard rule lives in :class:`ReadFileTool`: a single read may hand back
at most ~30k tokens (:data:`MAX_READ_TOKENS`). A tool result is part of the
conversation forever, so one accidental full read of a 2 MB source file would
evict the rest of the chat's history from the context window. An oversized
read is refused with the exact remedy — ``offset`` and ``limit`` — instead of
being silently truncated, because a cut-off read looks like the whole truth to
the model that just asked for it.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from ..config import Config
from ..paths import is_within, relative_to_root
from .base import Tool, ToolContext, ToolError, ToolResult
from .files import MAX_LIST_ENTRIES, READABLE_SYSTEM_PATHS

#: Rough chars-per-token for the read cap. Same order of magnitude as the
#: divisor in :mod:`lumi.context`; the cap is a guardrail, not a budget, so a
#: stable estimate beats a dependency on the context window's calibration.
CHARS_PER_TOKEN = 4

#: One ``read_file`` call may return at most this many tokens (~120k chars).
#: Bigger reads are refused with instructions to use ``offset`` and ``limit``.
MAX_READ_TOKENS = 30_000

#: Refuse (without reading) when a whole-file read would exceed this many
#: bytes. Cheap first gate: the stat check catches the 5 MB offenders before
#: a single byte is parsed.
MAX_READ_BYTES = MAX_READ_TOKENS * CHARS_PER_TOKEN

#: Files bigger than this are skipped by ``grep`` rather than scanned.
MAX_GREP_FILE_BYTES = 1_000_000

#: Directories never listed or grepped. Same noise as the ``files`` tool plus
#: the usual build/venv suspects — a search that dives into ``node_modules``
#: burns a minute to report what nobody wanted.
NOISE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
    }
)

#: Default and ceiling for one grep call's match list.
GREP_DEFAULT_RESULTS = 50
GREP_MAX_RESULTS = 200

#: A matching line is truncated to this many characters — the match is the
#: point, and a minified bundle line is not something anyone wanted whole.
GREP_LINE_CHARS = 300


def _resolve_read(root: Path, raw: Any) -> Path:
    """Resolve a read-only path: anywhere under *root*, plus whitelisted system files.

    Same boundary as the ``files`` tool's read branch — resolved path plus an
    ``is_within`` check, never a string prefix match.
    """
    if not raw or not str(raw).strip():
        raise ToolError("path is required")
    candidate = Path(str(raw).strip()).expanduser()
    requested = str(candidate)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if is_within(resolved, root):
        return resolved
    if requested in READABLE_SYSTEM_PATHS or str(resolved) in READABLE_SYSTEM_PATHS:
        return resolved
    raise ToolError(f"reading {resolved} is outside the project directory")


def _display(root: Path, path: Path) -> str:
    return relative_to_root(path)


def _is_binary_head(path: Path) -> bool:
    """True when the first 8 KB contain a NUL byte — the cheap binary sniff."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(8192)
    except OSError:
        return True


class ReadFileTool(Tool):
    name = "read_file"
    description = """\
Read a text file in this project, with line numbers.

Parameters:
- `path` (required): file relative to the project root.
- `offset`: 1-based line number to start from. Default 1.
- `limit`: how many lines to return from `offset`.

One call returns at most ~30k tokens. If a whole-file read would exceed that
it is refused, not truncated — re-issue with `offset` and `limit` (start at
`offset=1 limit=500`, then raise `offset` to the line after the last one you
saw; the result tells you the next offset). The result also reports the file's
total line count, so you can plan the windows.

Prefer this over `run_shell cat/head/tail`: it is confined to the project,
never hangs, and the line numbers make `files.edit` calls unambiguous. Use the
`files` tool when you intend to write, edit or upload the file instead."""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to read, relative to the project root.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start from. Defaults to 1.",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Required for large files.",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.root = config.root

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve_read(self.root, arguments.get("path"))
        display = _display(self.root, path)
        if not path.exists():
            return ToolResult.failure(f"no such file: {display}")
        if path.is_dir():
            return ToolResult.failure(
                f"{display} is a directory — use list_files to look inside it"
            )

        offset = _positive_int(arguments.get("offset"), default=1, name="offset")
        limit = arguments.get("limit")
        limit = _positive_int(limit, default=None, name="limit") if limit is not None else None

        size = path.stat().st_size
        whole_file = offset == 1 and limit is None
        if whole_file and size > MAX_READ_BYTES:
            return ToolResult.failure(
                f"{display} is {size:,} bytes — too big for one read "
                f"(cap {MAX_READ_TOKENS:,} tokens, about {MAX_READ_BYTES:,} bytes). "
                "Re-issue with offset and limit, e.g. offset=1 limit=500, "
                "then keep raising offset until you have what you need."
            )

        if _is_binary_head(path):
            return ToolResult.failure(
                f"{display} looks like a binary file. For its contents, describe it or "
                "process it with run_shell instead."
            )

        cap_chars = MAX_READ_TOKENS * CHARS_PER_TOKEN
        collected: list[tuple[int, str]] = []
        chars = 0
        last_line = 0
        stopped_at_cap = False
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if lineno < offset:
                        continue
                    if limit is not None and len(collected) >= limit:
                        break
                    text = line.rstrip("\n")
                    if chars + len(text) > cap_chars:
                        stopped_at_cap = True
                        break
                    collected.append((lineno, text))
                    chars += len(text)
                    last_line = lineno
        except OSError as exc:
            return ToolResult.failure(f"could not read {display}: {exc}")

        if not collected:
            total = _count_lines(path)
            if total is not None and offset > total:
                return ToolResult.failure(
                    f"offset {offset} is past the end of {display} — the file has "
                    f"{total:,} line(s)."
                )
            return ToolResult.failure(f"{display} has no content at offset {offset}.")

        if stopped_at_cap:
            return ToolResult.failure(
                f"{display}: the window starting at offset {offset} is over the "
                f"{MAX_READ_TOKENS:,}-token read cap. Use a smaller limit "
                "(e.g. limit=200) and walk forward with offset."
            )

        next_offset = last_line + 1
        more = limit is not None and _has_more_lines(path, next_offset)
        return ToolResult(
            text=self._render(path, display, collected, size, more, next_offset),
            data={
                "path": str(path),
                "offset": offset,
                "lines_returned": len(collected),
                "first_line": collected[0][0],
                "last_line": last_line,
                "next_offset": next_offset if more else None,
                "more": more,
            },
            summary=f"read {display} lines {collected[0][0]}-{last_line}",
        )

    def _render(
        self,
        path: Path,
        display: str,
        collected: list[tuple[int, str]],
        size: int,
        more: bool,
        next_offset: int,
    ) -> str:
        total = _count_lines(path) if size <= MAX_READ_BYTES else None
        first, last = collected[0][0], collected[-1][0]
        span = f"lines {first:,}-{last:,}"
        if total is not None:
            span += f" of {total:,}"
        header = f"--- {display} ({span}, {size:,} bytes) ---"
        body = "\n".join(f"{lineno:6d}  {text}" for lineno, text in collected)
        tail = ""
        if more:
            tail = (
                f"\n\n(more lines remain — continue with offset={next_offset}; "
                f"the file has {total:,} lines in total)"
                if total is not None
                else f"\n\n(more lines remain — continue with offset={next_offset})"
            )
        return f"{header}\n{body}{tail}"

    def summary_line(self) -> str:
        return f"`{self.name}` — read a project file by line window ({MAX_READ_TOKENS:,}-token cap)"


class ListFilesTool(Tool):
    name = "list_files"
    description = """\
List the entries of a directory in this project, like `ls`. Directories come
first (marked with `/` and an entry count), then files with their sizes.
Version-control, venv, cache and node_modules directories are hidden.

`path` is optional and defaults to the project root. One level only — use
`grep` to find files by their contents, or pass a subdirectory to drill in."""
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list, relative to the project root. Defaults to '.'.",
            }
        },
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.root = config.root

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = arguments.get("path") or "."
        path = _resolve_read(self.root, raw)
        display = _display(self.root, path)
        if not path.exists():
            return ToolResult.failure(f"no such directory: {display}")
        if not path.is_dir():
            return ToolResult.failure(
                f"{display} is a file, not a directory — use read_file to read it"
            )

        entries: list[str] = []
        files = 0
        dirs = 0
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            return ToolResult.failure(f"could not list {display}: {exc}")
        for child in children:
            if child.name in NOISE_DIRS:
                continue
            if child.is_dir():
                dirs += 1
                try:
                    count = sum(1 for _ in child.iterdir())
                except OSError:
                    count = "?"
                entries.append(f"{child.name}/  ({count} entries)")
            else:
                files += 1
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{child.name}  ({size:,} bytes)")
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"… truncated at {MAX_LIST_ENTRIES} entries")
                break

        header = f"--- {display}/ ({dirs} dir(s), {files} file(s)) ---"
        return ToolResult(
            text=f"{header}\n" + "\n".join(entries or ["(empty)"]),
            data={"path": str(path), "dirs": dirs, "files": files},
            summary=f"listed {display} ({dirs + files} entries)",
        )

    def summary_line(self) -> str:
        return f"`{self.name}` — list one directory of the project, like ls"


class GrepTool(Tool):
    name = "grep"
    description = """\
Search file contents across this project, like grep — but a query of several
words works as an AND: every word must appear somewhere on the matching line
(order does not matter, matching is case-insensitive unless `case_sensitive`
is set). "config token limit" therefore returns only lines containing all
three words.

Results are `path:line-number: matching-line`, most useful with `glob` (e.g.
`**/*.py`) to scope the search, `path` to narrow it to a subdirectory, and
`max_results` to bound the output. Binary files, venvs, node_modules, caches
and files over 1 MB are skipped.

Use this instead of `run_shell grep/rg` — no approval step, no quoting
surprises, and it never touches anything outside the project. For an exact
phrase containing spaces, put it in the query anyway: the whole line still has
to contain every word, so a distinctive multi-word phrase works as well."""
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Words to search for, separated by spaces. Every word must appear "
                    "on a line for it to match."
                ),
            },
            "path": {
                "type": "string",
                "description": "File or directory to search, relative to the project root.",
            },
            "glob": {
                "type": "string",
                "description": (
                    "Only search files whose relative path or name matches this "
                    "fnmatch glob, e.g. `**/*.py` or `*.md`."
                ),
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Match case-sensitively. Default false.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Stop after this many matches. Default {GREP_DEFAULT_RESULTS}.",
                "minimum": 1,
                "maximum": GREP_MAX_RESULTS,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.root = config.root

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        words = query.split()
        if not words:
            raise ToolError("query is required for a search")
        case_sensitive = bool(arguments.get("case_sensitive"))
        needles = list(words) if case_sensitive else [w.casefold() for w in words]
        max_results = int(arguments.get("max_results") or GREP_DEFAULT_RESULTS)
        max_results = max(1, min(max_results, GREP_MAX_RESULTS))

        start = _resolve_read(self.root, arguments.get("path") or ".")
        if not start.exists():
            return ToolResult.failure(f"no such path: {_display(self.root, start)}")
        glob = str(arguments.get("glob") or "").strip()

        matches: list[str] = []
        files_hit = 0
        files_scanned = 0
        skipped_large = 0
        truncated = False

        for file in self._candidates(start, glob):
            if skipped_large + files_scanned > 20_000:  # runaway tree guard
                break
            try:
                too_big = file.stat().st_size > MAX_GREP_FILE_BYTES
            except OSError:
                continue
            if too_big:
                skipped_large += 1
                continue
            if _is_binary_head(file):
                continue
            files_scanned += 1
            hit_here = False
            try:
                with file.open("r", encoding="utf-8", errors="replace") as handle:
                    for lineno, line in enumerate(handle, start=1):
                        text = line.rstrip("\n")
                        if not _line_matches(text, needles, case_sensitive):
                            continue
                        hit_here = True
                        shown = text.strip()
                        if len(shown) > GREP_LINE_CHARS:
                            shown = shown[:GREP_LINE_CHARS] + " …"
                        rel = (
                            file.relative_to(self.root)
                            if is_within(file, self.root)
                            else file
                        )
                        matches.append(f"{rel}:{lineno}: {shown}")
                        if len(matches) >= max_results:
                            truncated = True
                            break
            except OSError:
                continue
            if hit_here:
                files_hit += 1
            if truncated:
                break

        if not matches:
            hint = (
                " Remember: every word must appear on the same line — try fewer words, "
                "or a single distinctive word."
            )
            return ToolResult(
                text=f"no matches for {query!r} in {files_scanned} file(s).{hint}",
                data={"query": query, "files_scanned": files_scanned, "matches": []},
                summary=f"no matches for {query!r}",
            )

        scope = _display(self.root, start)
        header = f"--- {len(matches)} match(es) for {query!r} in {scope} ---"
        note = ""
        if truncated:
            note = (
                f"\n\n(stopped at {max_results} matches — narrow with `glob`, `path`, "
                "more words, or a higher max_results)"
            )
        if skipped_large:
            note += f"\n({skipped_large} file(s) over 1 MB were skipped)"
        return ToolResult(
            text=f"{header}\n" + "\n".join(matches) + note,
            data={
                "query": query,
                "files_scanned": files_scanned,
                "files_matched": files_hit,
                "matches": matches,
                "truncated": truncated,
            },
            summary=f"{len(matches)} match(es) for {query!r} in {files_hit} file(s)",
        )

    def _candidates(self, start: Path, glob: str = ""):
        """Every file to scan, deterministically ordered, noise pruned."""
        if start.is_file():
            yield start
            return
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in NOISE_DIRS)
            for name in sorted(filenames):
                file = Path(dirpath) / name
                if glob and not _glob_matches(file, self.root, glob):
                    continue
                yield file

    def summary_line(self) -> str:
        return f"`{self.name}` — multi-word content search across the project"


def _line_matches(text: str, needles: list[str], case_sensitive: bool) -> bool:
    hay = text if case_sensitive else text.casefold()
    return all(needle in hay for needle in needles)


def _glob_matches(file: Path, root: Path, glob: str) -> bool:
    """fnmatch *glob* against the file's root-relative path and its bare name.

    Both forms are accepted so `*.md` works at any depth while `docs/**` scopes
    to a subtree, the way a person typing a glob expects it to behave.
    """
    try:
        rel = str(file.relative_to(root))
    except ValueError:
        rel = str(file)
    return fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(file.name, glob)


def _count_lines(path: Path) -> int | None:
    """Total lines in *path*, or None when it is too big to count cheaply."""
    try:
        if path.stat().st_size > MAX_READ_BYTES:
            return None
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _has_more_lines(path: Path, next_offset: int) -> bool:
    """Whether *path* has any line at *next_offset* or beyond."""
    total = _count_lines(path)
    if total is not None:
        return next_offset <= total
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, _ in enumerate(handle, start=1):
                if lineno >= next_offset:
                    return True
        return False
    except OSError:
        return False


def _positive_int(value: Any, *, default: int | None, name: str) -> int | None:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be an integer, got {value!r}") from None
    if parsed < 1:
        raise ToolError(f"{name} must be >= 1, got {parsed}")
    return parsed


__all__ = [
    "ReadFileTool",
    "ListFilesTool",
    "GrepTool",
    "MAX_READ_TOKENS",
    "MAX_GREP_FILE_BYTES",
]
