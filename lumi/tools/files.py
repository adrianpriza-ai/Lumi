"""The ``files`` tool: read, write and edit inside the project.

Reads are allowed anywhere under the project root (so the bot can inspect its
own source), plus a small set of read-only system paths — except for secret
files (``.env``, keys, credentials), which are refused by name: the env scrub
only protects the child process, so the read path has to refuse too. Writes
are confined to the directories listed in ``tools.files.writable``, and
overwriting an existing non-empty file needs approval.

``edit`` is the surgical path: an exact ``old_string``/``new_string`` swap that
can only touch the text it names, so it runs without the approval prompt that a
whole-file overwrite asks for — with one exception. A file that appeared or
changed *during this session* may have been authored from fetched or injected
content, so the first write, append or edit to such a file asks the owner once
(:data:`lumi.tools.safety.SESSION_START` and the ``_approved`` set are what make
"once" mean once). Files that predate the process are ordinary project
material and keep the rules above. This is the write half of the
write-then-run chain; the shell asks the run half.

Path handling is the security boundary here, so it is done with
``Path.resolve()`` plus an ``is_within`` check — no string prefix matching,
which is trivially bypassed by ``../`` or a symlink. The shell's
protected-path deny list applies here too (see :func:`protected_write`), so
widening ``tools.files.writable`` cannot reopen what ``run_shell`` refuses,
and the secret-file read guard (see :func:`read_guard`) applies to every
read, list, stat and upload alike.
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
from .safety import SESSION_START, protected_write, read_guard

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


def foreign_uploads(path: Path, root: Path, chat_id: int | str | None) -> str | None:
    """The *other* chat whose upload directory *path* sits in, if any.

    Incoming documents are stored per chat (``workspace/uploads/<chat_id>/``),
    but every read inside the project used to be allowed from any chat, so
    chat B could list and pull chat A's files back to itself with no approval
    anywhere. Read-side calls only: the component is compared as a string with
    the requesting ``chat_id``, and ``None`` (the local CLI, which has no chat)
    is not restricted.
    """
    if chat_id is None:
        return None
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if (
        len(parts) >= 3
        and parts[0] == "workspace"
        and parts[1] == "uploads"
        and parts[2] != str(chat_id)
    ):
        return parts[2]
    return None


class FilesTool(Tool):
    name = "files"
    description = """
Read, write, edit, list, search and upload files inside this project.

Paths are relative to the project root. Reads work anywhere in the project
except secret files (.env, keys, credentials), which are refused; writes only
work inside the project's workspace directory, and overwriting an existing
file needs the owner's approval. PDFs are read through OCR (page-marked text)
and image files are attached as pictures the model can look at directly —
prefer `read_file` for reading either.

`edit` is how you change an existing file: give the exact `old_string` you
found and the `new_string` that replaces it. It needs no approval (it can
only touch the text it names), shows a diff, and leaves the rest of the file
alone — prefer it over `write` whenever you are changing less than the whole
file. The `old_string` must match exactly once unless `replace_all` is true.

One exception: a file that was created or changed *during this session* gets
one owner approval on its first write, append or edit, then goes on normally.
That is the write half of the write-then-run chain — a file this session
authored may come from fetched content, so its authorship is shown to the
owner once before it changes. Skill files under `data/skills/` always get
that one approval, even when they predate the session: their text is put in
front of every chat, so the owner always sees the first change.

Set `upload: true` on a write or an edit to also send the finished file to the
chat as a document — use it whenever the owner asked for "a file" they can
download. The `upload` action sends a file that already exists (e.g. one a
shell command produced) without modifying it. Delivery is handled for you:
just say which file, and it arrives in the chat.

Prefer this over shell redirection for anything that involves file contents:
it validates the path, caps the size, and shows a diff before it overwrites.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "edit", "list", "search", "stat", "upload"],
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
            "old_string": {
                "type": "string",
                "description": (
                    "edit: the exact text to replace, copied verbatim from the file. "
                    "Required for edit."
                ),
            },
            "new_string": {
                "type": "string",
                "description": "edit: the replacement text. Required for edit.",
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "edit: replace every occurrence of old_string instead of requiring "
                    "it to match exactly once."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "Set true to overwrite a non-empty file without asking the owner. "
                    "Leave it out to get an approval prompt instead. Files authored "
                    "during this session ask the owner once regardless."
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
                    "write and edit: also send the file to the chat as a document "
                    "when the action succeeds."
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
        #: Lazily created: the shared PDF/image reader, used by the read action
        #: when the file turns out to be media. Imported lazily with it (see
        #: _read_media — docread.top-level imports this module).
        self._reader: Any = None
        #: Files the owner has already let this session touch (or that predate
        #: the session). One tap per file, not one tap per call — a gate the
        #: model has to clear on every keystroke would be tapped blind.
        self._approved: set[str] = set()

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

    def _resolve(
        self,
        raw: str | None,
        *,
        for_write: bool,
        chat_id: int | str | None = None,
    ) -> Path:
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
            # Writes are per chat exactly like reads: an upload that predates
            # this session carries no mtime freshness, so editing another
            # chat's file used to land with no tap at all (R14) — the
            # first-write prompt only ever fired for session-authored files.
            if (foreign := foreign_uploads(resolved, self.root, chat_id)) is not None:
                raise ToolError(
                    f"writing to {resolved} is not allowed: workspace/uploads/{foreign} "
                    "belongs to another chat. Each chat only sees its own uploads."
                )
            guard = protected_write(resolved, self.root)
            if guard is not None:
                raise ToolError(
                    f"writing to {resolved} is not allowed: {guard} holds credentials or "
                    "runs code on someone else's command, and run_shell refuses it too. "
                    "Ask the owner to do it by hand if it is really needed."
                )
            return resolved

        if is_within(resolved, self.root):
            if (foreign := foreign_uploads(resolved, self.root, chat_id)) is not None:
                raise ToolError(
                    f"reading {resolved} is not allowed: workspace/uploads/{foreign} "
                    "belongs to another chat. Each chat only sees its own uploads."
                )
            guard = read_guard(candidate) or read_guard(resolved)
            if guard is not None:
                raise ToolError(
                    f"reading {resolved} is not allowed: {guard} holds credentials. "
                    "run_shell refuses to print it too — ask the owner if a value "
                    "from it is really needed."
                )
            return resolved
        if requested in READABLE_SYSTEM_PATHS or str(resolved) in READABLE_SYSTEM_PATHS:
            return resolved
        raise ToolError(f"reading {resolved} is outside the project directory")

    def _display(self, path: Path) -> str:
        return relative_to_root(path)

    def _is_project_skill(self, path: Path) -> bool:
        """Whether *path* is a skill file the system prompt will read back."""
        return is_within(path, self.config.data_dir / "skills")

    def _session_authored(self, path: Path) -> bool:
        """Whether *path* still needs its one owner approval this session.

        True for a file that does not exist yet and for one whose mtime is at
        or after :data:`SESSION_START` — in both cases the content the next
        write would replace or extend was produced during this run, which is
        the provenance finding 3 cares about. False once the owner has approved
        this path, so a file costs one confirmation per session.

        A file under ``data/skills`` is always True until approved, whatever
        its mtime. Skill text is catalogued into the system prompt of every
        chat, so a skill that predates the session is not thereby trustworthy:
        the model can edit it, and an edit made after reading a hostile page
        would otherwise reach every future turn with no tap (R8).
        """
        if str(path) in self._approved:
            return False
        if self._is_project_skill(path):
            return True
        try:
            return path.stat().st_mtime >= SESSION_START
        except OSError:
            return True

    def _require_owner(self, path: Path, ctx: ToolContext, arguments: dict[str, Any], preview: str) -> None:
        """Raise :class:`NeedsApproval` unless *path* is already cleared."""
        if ctx.confirmed or not self._session_authored(path):
            return
        if self._is_project_skill(path):
            reason = (
                f"{self._display(path)} is a skill file, and its text is read into "
                "every chat's system prompt, so the owner approves the first write to "
                "it this session"
            )
        else:
            reason = (
                f"{self._display(path)} was created or changed during this session, "
                "so the owner approves the first write to it"
            )
        raise NeedsApproval(
            tool=self.name,
            arguments=dict(arguments),
            reason=reason,
            preview=preview,
        )

    # -- actions ----------------------------------------------------------- #

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(arguments.get("action", "")).strip().lower()
        handler = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "edit": self._edit,
            "list": self._list,
            "search": self._search,
            "stat": self._stat,
            "upload": self._upload,
        }.get(action)
        if handler is None:
            raise ToolError(
                f"unknown action {action!r}; use read, write, append, edit, "
                "list, search, stat or upload"
            )
        return await handler(arguments, ctx)

    async def _read(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=False, chat_id=ctx.chat_id)
        if not path.exists():
            return ToolResult.failure(f"no such file: {self._display(path)}")
        if path.is_dir():
            return await self._list(arguments, ctx)

        # Media before the text read: a PDF is full of NUL bytes and a photo
        # is over the char cap, and both are readable — just not as text.
        # Lazy import: docread.top-level imports this module (foreign_uploads),
        # so a top-level import here would close a cycle.
        from .docread import media_kind

        if media := media_kind(path):
            return await self._read_media(path, *media, ctx=ctx)

        # 0 (from the argument or the setting) means "no cap".
        cap = int(arguments.get("max_bytes") or 0) or self.settings.max_read_chars
        try:
            size = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                body = handle.read() if cap <= 0 else handle.read(cap)
        except OSError as exc:
            return ToolResult.failure(f"could not read {self._display(path)}: {exc}")

        truncated = cap > 0 and size > cap
        header = f"--- {self._display(path)} ({size} bytes"
        header += ", truncated" if truncated else ""
        header += ") ---"
        return ToolResult(
            text=f"{header}\n{body}",
            data={"path": str(path), "size": size, "truncated": truncated},
            summary=f"read {self._display(path)} ({size} bytes)",
        )

    async def _read_media(self, path: Path, kind: str, mime: str, ctx: ToolContext) -> ToolResult:
        """A PDF or an image: the shared reader's routes, behind the same guards.

        PDFs come back as page-marked text (scanned pages transcribed by a
        vision model), images as an attached picture the model looks at
        directly. The path was resolved and secret-checked by ``_resolve``;
        the reader re-asks anyway — it is the component that ships bytes to a
        model, and a credential that slips past one check becomes a
        transcription on someone else's API.
        """
        from .docread import DocumentReader

        if self._reader is None:
            self._reader = DocumentReader(self.config)
        display = self._display(path)
        ok, reason = self._reader.available()
        if not ok:
            return ToolResult.failure(
                f"{display} is a {'PDF' if kind == 'pdf' else 'picture'}, but reading it "
                f"is {reason}. Ask the owner to enable tools.ocr, or process the file "
                "with run_shell."
            )
        if problem := self._reader.read_problem(path, ctx.chat_id):
            return ToolResult.failure(f"cannot read {display}: {problem}.")
        if kind == "pdf":
            return await self._reader.read_pdf(path)
        return await self._reader.read_image(path, mime)

    async def _write(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=True, chat_id=ctx.chat_id)
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

        self._require_owner(path, ctx, arguments, _short_diff(previous, content))

        if existed and previous.strip() and not overwrite and not ctx.confirmed:
            raise NeedsApproval(
                tool=self.name,
                arguments=dict(arguments),
                reason=f"{self._display(path)} already exists and is not empty",
                preview=_short_diff(previous, content),
            )

        try:
            self._atomic_write(path, content)
        except OSError as exc:
            return ToolResult.failure(f"could not write {self._display(path)}: {exc}")

        verb = "overwrote" if existed else "created"
        self._approved.add(str(path))
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
        path = self._resolve(arguments.get("path"), for_write=False, chat_id=ctx.chat_id)
        if not path.is_file():
            return ToolResult.failure(f"no such file: {self._display(path)}")

        result = ToolResult(text="", ok=True, summary=f"upload {self._display(path)}")
        await self._attach(result, path)
        if not result.artifacts:
            result.ok = False
            result.text = result.text.strip() or "could not stage that file for delivery"
        return result

    async def _append(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path"), for_write=True, chat_id=ctx.chat_id)
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if not content:
            return ToolResult.failure("nothing to append")

        created = not path.exists()
        self._require_owner(path, ctx, arguments, truncate(content, 400))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            return ToolResult.failure(f"could not append to {self._display(path)}: {exc}")

        verb = "created" if created else "appended to"
        self._approved.add(str(path))
        return ToolResult(
            text=f"{verb} {self._display(path)} (+{len(content)} bytes)",
            data={"path": str(path), "appended": len(content), "created": created},
            summary=f"{verb} {self._display(path)}",
        )

    # -- editing an existing file ------------------------------------------ #

    async def _edit(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Replace exact text: a small change stays a small change.

        Deliberately skips the approval prompt that a whole-file ``write``
        asks for. The result is fully determined by ``old_string`` and
        ``new_string`` — the tool cannot clobber what it does not name — and
        a tap on every small fix would teach the owner to tap without
        reading. Everything else still applies: writable roots, the
        protected-path guard, a diff of what changed.

        The one exception is a file this session authored: its first edit
        goes through :meth:`_require_owner`, because ``old_string`` matching
        is no reassurance when the whole file arrived from fetched content.
        """
        path = self._resolve(arguments.get("path"), for_write=True, chat_id=ctx.chat_id)
        old = arguments.get("old_string")
        new = arguments.get("new_string")
        if not isinstance(old, str) or not old:
            raise ToolError("old_string is required for an edit")
        if not isinstance(new, str):
            raise ToolError("new_string must be a string")
        if old == new:
            raise ToolError("old_string and new_string are identical; there is nothing to change")
        if not path.exists():
            return ToolResult.failure(
                f"no such file: {self._display(path)} — edit only changes existing files; "
                "use write to create one"
            )
        if path.is_dir():
            raise ToolError(f"{self._display(path)} is a directory")

        # The whole file has to fit in memory to swap a piece of it, so the
        # same cap that bounds a write bounds an edit — with a reason the
        # model can act on instead of a silent failure.
        limit = self.settings.max_write_chars
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return ToolResult.failure(f"could not read {self._display(path)}: {exc}")
        if len(raw) > limit:
            raise ToolError(
                f"{self._display(path)} is {len(raw)} bytes, over the {limit}-byte limit for "
                "the files tool (tools.files.max_write_chars). Edit it with a shell command "
                "instead, or ask the owner to raise the limit."
            )
        if b"\x00" in raw:
            return ToolResult.failure(
                f"{self._display(path)} looks like a binary file; edit only works on text"
            )
        try:
            previous = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(f"{self._display(path)} is not valid UTF-8 text")

        replace_all = bool(arguments.get("replace_all"))
        count = previous.count(old)
        if count == 0:
            return ToolResult.failure(
                f"old_string not found in {self._display(path)}. Read the file first and "
                "copy the exact text — character for character, whitespace included."
            )
        if count > 1 and not replace_all:
            return ToolResult.failure(
                f"old_string matches {count} times in {self._display(path)}. Include a few "
                "lines of surrounding context to make it unique, or set replace_all: true."
            )

        updated = previous.replace(old, new) if replace_all else previous.replace(old, new, 1)
        made = count if replace_all else 1
        self._require_owner(path, ctx, arguments, _short_diff(previous, updated))
        try:
            self._atomic_write(path, updated)
        except OSError as exc:
            return ToolResult.failure(f"could not edit {self._display(path)}: {exc}")

        self._approved.add(str(path))
        log.info("edited %s (%d replacement(s))", self._display(path), made)
        result = ToolResult(
            text=f"edited {self._display(path)} ({made} replacement(s))\n"
            + _short_diff(previous, updated),
            ok=True,
            data={"path": str(path), "replacements": made, "bytes": len(updated)},
            summary=f"edited {self._display(path)} ({made} replacement(s))",
        )
        if arguments.get("upload") and self.settings.uploads:
            await self._attach(result, path)
        return result

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write through a temp file and rename, so a crash cannot halve a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".lumi-tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    async def _list(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self._resolve(arguments.get("path") or ".", for_write=False, chat_id=ctx.chat_id)
        if not path.exists():
            return ToolResult.failure(f"no such directory: {self._display(path)}")
        if not path.is_dir():
            return ToolResult.failure(f"{self._display(path)} is a file, not a directory")

        children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        shown = 0
        hidden = 0
        entries: list[str] = []
        for child in children:
            if child.name in {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}:
                continue
            if shown >= MAX_LIST_ENTRIES:
                hidden += 1
                continue
            shown += 1
            if child.is_dir():
                count = sum(1 for _ in child.iterdir())
                entries.append(f"{child.name}/  ({count} entries)")
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{child.name}  ({size} bytes)")

        # The count describes the directory, not the rendered list: the
        # truncation marker must not be counted as an entry, and a directory
        # of 600 files must not read as "500 files".
        text = f"--- {self._display(path)}/ ---\n" + "\n".join(entries or ["(empty)"])
        if hidden:
            text += f"\n… {hidden} more entries not shown"
        return ToolResult(
            text=text,
            data={"path": str(path), "count": shown, "truncated": hidden > 0},
            summary=f"listed {self._display(path)} ({shown} entries)",
        )

    async def _search(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(arguments.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("pattern is required for a search")
        start = self._resolve(arguments.get("path") or ".", for_write=False, chat_id=ctx.chat_id)
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
        path = self._resolve(arguments.get("path"), for_write=False, chat_id=ctx.chat_id)
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
            f"`{self.name}` — read/write/edit/list/search the project "
            f"(writable: {allowed}), {delivery}"
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
