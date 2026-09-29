"""The ``files`` tool: read and write inside the project.

Reads are allowed anywhere under the project root (so the bot can inspect its
own source), plus a small set of read-only system paths. Writes are confined to
the directories listed in ``tools.files.writable``, and overwriting an existing
non-empty file needs approval.

Path handling is the security boundary here, so it is done with
``Path.resolve()`` plus an ``is_within`` check — no string prefix matching,
which is trivially bypassed by ``../`` or a symlink.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactError, ArtifactStore
from ..config import Config
from ..paths import is_within, relative_to_root
from ..util.log import get_logger
from ..util.text import truncate
from .base import NeedsApproval, Tool, ToolContext, ToolError, ToolResult

log = get_logger(__name__)

#: Read-only escapes, so the bot can check a system fact without a shell.
READABLE_SYSTEM_PATHS: tuple[str, ...] = (
    "/etc/os-release",
    "/etc/hostname",
    "/proc/uptime",
    "/proc/meminfo",
    "/proc/cpuinfo",
    "/sys/kernel/osrelease",
)

MAX_LIST_ENTRIES = 500


class FilesTool(Tool):
    name = "files"
    description = """
Read, write, list, search and upload files inside this project.

Paths are relative to the project root. Reads work anywhere in the project;
writes only work inside the project's workspace directory, and overwriting an
existing file needs the owner's approval.

Set `upload: true` on a write to also send the finished file to the chat as a
document — use it whenever the owner asked for "a file" they can download. The
`upload` action sends a file that already exists (e.g. one a shell command
produced) without modifying it. Delivery is handled for you: just say which
file, and it arrives in the chat.

Prefer this over shell redirection for anything that involves file contents:
it validates the path, caps the size, and shows a diff before it overwrites.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "list", "search", "stat", "upload"],
                "description": "What to do.",
            },
            "path": {
                "type": "string",
                "description": "File or directory, relative to the project root.",
            },
            "content": {
                "type": "string",
                "description": "New file contents. Required for write and append.",
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "Set true to overwrite a non-empty file without asking the owner. "
                    "Leave it out to get an approval prompt instead."
                ),
            },
            "pattern": {
                "type": "string",
                "description": "Glob or substring. Required for search, e.g. `**/*.py`.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Optional read cap in bytes (default from config).",
                "minimum": 1,
            },
            "upload": {
                "type": "boolean",
                "description": (
                    "write only: also send the file to the chat as a document when "
                    "the write succeeds."
                ),
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.settings = config.tools.files
        self.root = config.root
        self.writable_roots = self.settings.writable_roots()
        # Lazily created: the store needs the bot's upload cap, which lives in
        # the bot config section.
        self._store: ArtifactStore | None = None

    @property
    def store(self) -> ArtifactStore:
        if self._store is None:
            self._store = ArtifactStore(
                self.config.root,
                max_bytes=max(1, self.config.bot.max_upload_mb) * 1024 * 1024,
            )
        return self._store

    def available(self) -> tuple[bool, str]:
        if not self.settings.enabled:
            return False, "disabled in config (tools.files.enabled = false)"
        return True, ""

    # -- path resolution --------------------------------------------------- #

    def _resolve(self, raw: str | None, *, for_write: bool) -> Path:
        if not raw or not str(raw).strip():
            raise ToolError("path is required")
        candidate = Path(str(raw).strip()).expanduser()
        # Keep the literal path around: /etc/os-release resolves to
        # /usr/lib/os-release, and the read-only whitelist is written in terms of
        # the paths a human would actually type.
        requested = str(candidate)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

        if for_write:
            if not any(is_within(resolved, root) for root in self.writable_roots):
                allowed = ", ".join(relative_to_root(r) for r in self.writable_roots)
                raise ToolError(
                    f"writing to {resolved} is not allowed. Writable locations: {allowed}. "
                    "Ask the owner to widen tools.files.writable in config.toml if this is intended."
                )
            return resolved

        if is_within(resolved, self.root):
            return resolved
        if requested in READABLE_SYSTEM_PATHS or str(resolved) in READABLE_SYSTEM_PATHS:
            return resolved
        if not self.settings.readable_from_project:
            raise ToolError(f"reading {resolved} is outside the project and is not permitted")
        raise ToolError(f"reading {resolved} is outside the project directory")

    def _display(self, path: Path) -> str:
        return relative_to_root(path)

    # -- actions ----------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        handler = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "list": self._list,
            "search": self._search,
            "stat": self._stat,
            "upload": self._upload,
        }.get(action)
        if handler is None:
            raise ToolError(
                f"unknown action {action!r}; use read, write, append, list, search, stat or upload"
            )
        return await handler(arguments, ctx)

    async def _read(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=False)
        if not path.exists():
            return ToolResult.failure(f"no such file: {self._display(path)}")
        if path.is_dir():
            return await self._list(arguments, ctx)

        cap = int(arguments.get("max_bytes") or self.settings.max_read_chars)
        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                body = handle.read(cap)
        except OSError as exc:
            return ToolResult.failure(f"could not read {self._display(path)}: {exc}")

        header = f"--- {self._display(path)} ({size} bytes"
        header += ", truncated" if size > cap else ""
        header += ") ---"
        return ToolResult(
            text=f"{header}\n{body}",
            data={"path": str(path), "size": size, "truncated": size > cap},
            summary=f"read {self._display(path)} ({size} bytes)",
        )

    async def _write(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=True)
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > self.settings.max_write_chars:
            raise ToolError(
                f"content is {len(content)} chars, over the {self.settings.max_write_chars} limit. "
                "Write it in pieces with append instead."
            )

        if path.is_dir():
            raise ToolError(f"{self._display(path)} is a directory")

        existed = path.exists()
        previous = path.read_text(encoding="utf-8", errors="replace") if existed else ""
        overwrite = bool(arguments.get("overwrite"))

        if existed and previous.strip() and not overwrite and not ctx.confirmed:
            raise NeedsApproval(
                tool=self.name,
                arguments=dict(arguments),
                reason=f"{self._display(path)} already exists and is not empty",
                preview=_short_diff(previous, content),
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".lumi-tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)  # atomic
        except OSError as exc:
            return ToolResult.failure(f"could not write {self._display(path)}: {exc}")

        verb = "overwrote" if existed else "created"
        log.info("%s %s (%d bytes)", verb, self._display(path), len(content))
        diff = _short_diff(previous, content) if existed else ""

        result = ToolResult(
            text=f"{verb} {self._display(path)} ({len(content)} bytes)\n{diff}",
            ok=True,
            data={"path": str(path), "bytes": len(content), "created": not existed},
            summary=f"{verb} {self._display(path)}",
        )
        if arguments.get("upload") and self.settings.uploads:
            await self._attach(result, path)
        return result

    # -- delivery to the chat ----------------------------------------------- #

    async def _attach(self, result: ToolResult, path: Path) -> None:
        """Stage *path* onto *result* as an artifact, best effort.

        A failure to deliver must not fail the write that already succeeded, so
        every error path here degrades to a note in the result text.
        """
        try:
            artifact = self.store.send(path, origin=self.name, project_root=self.root)
        except ArtifactError as exc:
            result.text += f"\n(note: not sent to the chat — {exc})"
            return
        except OSError as exc:
            result.text += f"\n(note: not sent to the chat — {exc})"
            return
        result.artifacts.append(artifact)
        result.text += f"\n(will be sent to the chat as a document: {artifact.path})"

    async def _upload(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Send an existing project file to the chat as a document.

        The file itself is not modified; this only stages it for delivery. Use
        it after generating something with the shell, or when the owner asks
        for a file that already exists.
        """
        if not self.settings.uploads:
            return ToolResult.failure(
                "file delivery is disabled in config (tools.files.uploads = false)"
            )
        raw = arguments.get("path")
        if not raw or not str(raw).strip():
            raise ToolError("path is required for an upload")
        path = self._resolve(raw, for_write=False)
        if not path.is_file():
            return ToolResult.failure(f"no such file: {self._display(path)}")

        result = ToolResult(text="", ok=True, summary=f"upload {self._display(path)}")
        await self._attach(result, path)
        if not result.artifacts:
            result.ok = False
            result.text = result.text.strip() or "could not stage that file for delivery"
        return result

    async def _append(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=True)
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if not content:
            return ToolResult.failure("nothing to append")

        created = not path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            return ToolResult.failure(f"could not append to {self._display(path)}: {exc}")

        verb = "created" if created else "appended to"
        return ToolResult(
            text=f"{verb} {self._display(path)} (+{len(content)} bytes)",
            data={"path": str(path), "appended": len(content), "created": created},
            summary=f"{verb} {self._display(path)}",
        )

    async def _list(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path") or ".", for_write=False)
        if not path.exists():
            return ToolResult.failure(f"no such directory: {self._display(path)}")
        if not path.is_dir():
            return ToolResult.failure(f"{self._display(path)} is a file, not a directory")

        entries: list[str] = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}:
                continue
            if child.is_dir():
                count = sum(1 for _ in child.iterdir())
                entries.append(f"{child.name}/  ({count} entries)")
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{child.name}  ({size} bytes)")
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"… truncated at {MAX_LIST_ENTRIES} entries")
                break

        return ToolResult(
            text=f"--- {self._display(path)}/ ---\n" + "\n".join(entries or ["(empty)"]),
            data={"path": str(path), "count": len(entries)},
            summary=f"listed {self._display(path)} ({len(entries)} entries)",
        )

    async def _search(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(arguments.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("pattern is required for a search")
        start = self._resolve(arguments.get("path") or ".", for_write=False)
        if not start.is_dir():
            start = start.parent

        try:
            matches = sorted(
                str(p.relative_to(start))
                for p in start.glob(pattern)
                if p.is_file() and not _ignored(p)
            )
        except (ValueError, OSError) as exc:
            return ToolResult.failure(f"bad pattern {pattern!r}: {exc}")

        return ToolResult(
            text=f"{len(matches)} match(es) for {pattern!r} under {self._display(start)}:\n"
            + ("\n".join(matches[:MAX_LIST_ENTRIES]) or "(none)"),
            data={"pattern": pattern, "matches": matches[:MAX_LIST_ENTRIES]},
            summary=f"found {len(matches)} match(es) for {pattern!r}",
        )

    async def _stat(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=False)
        if not path.exists():
            return ToolResult.failure(f"no such path: {self._display(path)}")
        try:
            info = path.stat()
        except OSError as exc:
            return ToolResult.failure(f"could not stat {self._display(path)}: {exc}")

        kind = "directory" if path.is_dir() else "file"
        lines = [
            f"{self._display(path)}",
            f"  type: {kind}",
            f"  size: {info.st_size} bytes",
            f"  modified: {info.st_mtime:.0f}",
        ]
        if path.is_dir():
            lines.append(f"  entries: {sum(1 for _ in path.iterdir())}")
        return ToolResult(text="\n".join(lines), data={"path": str(path), "is_dir": path.is_dir()})

    def summary_line(self) -> str:
        allowed = ", ".join(relative_to_root(r) for r in self.writable_roots)
        delivery = "can send files to the chat" if self.settings.uploads else "delivery disabled"
        return (
            f"`{self.name}` — read/write/list/search the project (writable: {allowed}), "
            f"{delivery}"
        )


def _ignored(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"})


def _short_diff(previous: str, current: str, context: int = 2, limit: int = 40) -> str:
    """Compact unified diff, so an approved overwrite shows what changed."""
    if previous == current:
        return "(no change)"
    diff = list(
        difflib.unified_diff(
            previous.splitlines(),
            current.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=context,
        )
    )
    if not diff:
        return "(no textual change)"
    body = truncate("\n".join(diff[:limit]), 1500)
    if len(diff) > limit:
        body += f"\n… {len(diff) - limit} more diff lines"
    return body


__all__ = ["FilesTool", "READABLE_SYSTEM_PATHS", "MAX_LIST_ENTRIES"]
