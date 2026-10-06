"""The artifact harness: how files travel between the chat and the project.

Two directions, one registry.

**Out** — a tool writes a file under the project and wants the chat to have it.
It does not touch Telegram; it calls :func:`register`, which validates the path,
copies it into a stable outbox, and returns an :class:`Artifact`. The agent loop
collects every artifact it meets into :attr:`lumi.agent.TurnResult.artifacts`,
and the presentation layer (Telegram, the CLI) decides what to do with them.
This keeps tools free of any transport dependency: the same registry serves the
Telegram bot, ``lumi chat``, and a future channel with no changes to the tools.

**In** — the owner sends a document to Telegram. :func:`ingest` drops it into
``workspace/uploads/<chat_id>/`` and the bot hands the model both the path and
the file's facts. From there the model treats it like any other project file:
it can read it with the ``files`` tool, rename it, or process it with the shell.

The outbox exists so nothing the model produced is ever lost when a chat fails
or the owner ignores it: every artifact has a copy on disk under ``data`` that
is independent of any single message, plus the original inside the project if
it came from the writable roots.
"""

from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .paths import is_within, relative_to_root
from .util.log import get_logger

log = get_logger(__name__)

#: Default ceiling on a single stored artifact. Telegram's own Bot API document
#: limit is 50 MB; a 20 MB default keeps the outbox useful without letting one
#: runaway export fill the disk. Overridable per store.
DEFAULT_MAX_BYTES = 20 * 1024 * 1024

#: File kinds we are willing to store. Anything else is refused — the point of
#: the harness is files the model produces on purpose, not whatever a tool
#: happened to touch.
ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst",
        ".pdf", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml",
        ".html", ".htm", ".css", ".js", ".mjs", ".ts", ".tsx", ".jsx",
        ".py", ".pyi", ".ipynb", ".sh", ".bash", ".zsh", ".ps1",
        ".rs", ".go", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs",
        ".rb", ".php", ".swift", ".sql", ".lua", ".r", ".jl",
        ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp",
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z",
        ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".rtf",
        ".epub", ".log", ".ini", ".cfg", ".conf", ".env", ".diff", ".patch", ".xml",
    }
)


def _guess_mime(suffix: str) -> str:
    """A conservative MIME type for Telegram's send_document."""
    table = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
        ".html": "text/html",
        ".zip": "application/zip",
        ".gz": "application/gzip",
        ".tgz": "application/gzip",
        ".tar": "application/x-tar",
    }
    return table.get(suffix.lower(), "application/octet-stream")


@dataclass(slots=True)
class Artifact:
    """A file a tool produced, ready for the presentation layer to deliver.

    ``path`` is the project-relative view the chat should show a human.
    ``absolute`` is the concrete file on disk — the presentation layer reads
    from it and nothing else does. Both point at the same bytes; the split
    exists because the project-relative path is what survives a folder move.
    """

    path: str
    absolute: Path
    size: int
    mime: str
    #: Where it came from: the tool that registered it. Shown in the CLI and
    #: useful for tracing a file back to the call that made it.
    origin: str = ""
    #: When it was registered. Purely informational.
    created: float = 0.0

    def caption(self) -> str:
        """A short human line for a document message."""
        size = self.size
        if size >= 1024 * 1024:
            human = f"{size / (1024 * 1024):.1f} MB"
        elif size >= 1024:
            human = f"{size / 1024:.1f} KB"
        else:
            human = f"{size} B"
        origin = f" — from {self.origin}" if self.origin else ""
        return f"{self.path} ({human}){origin}"


class ArtifactError(RuntimeError):
    """Raised when a file cannot become an artifact. The model reads the reason."""


class ArtifactStore:
    """Validates, copies and remembers files a tool wants to send to the chat.

    One store per process. ``send`` is the entry point tools call; everything
    else is plumbing. The store deliberately knows nothing about Telegram —
    delivering an artifact is the caller's job, and the CLI proves that by
    delivering nothing at all.
    """

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self._recent: dict[str, Artifact] = {}
        # Bounded: a long-lived process should not grow this without limit.
        self._recent_order: list[str] = []
        self._recent_limit = 200

    # -- the outbox -------------------------------------------------------- #

    def send(
        self, path: Path | str, *, origin: str = "", project_root: Path | None = None
    ) -> Artifact:
        """Register *path* for delivery and return its :class:`Artifact`.

        The file is copied into the outbox so it survives even if the tool later
        deletes or rewrites the original. Relative paths resolve against
        *project_root* (default: the store's own idea of it).

        Raises :class:`ArtifactError` with a model-readable reason when the
        path is missing, unreadable, disallowed or too large — callers turn
        that into a tool error, not a crash.
        """
        root = project_root if project_root is not None else self.root
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ArtifactError(f"could not resolve {candidate}: {exc}") from exc

        if not resolved.is_file():
            raise ArtifactError(f"{relative_to_root(resolved)} is not a file (or does not exist)")
        if not is_within(resolved, root):
            raise ArtifactError(
                f"{resolved} is outside the project; artifacts must live under it"
            )

        suffix = resolved.suffix.lower()
        if suffix and suffix not in ALLOWED_SUFFIXES:
            raise ArtifactError(
                f"refusing to deliver a {suffix!r} file. Allowed: documents, text, "
                "code, data, images, and common archives."
            )
        if not suffix:
            raise ArtifactError("the file has no extension; artifacts need one")

        size = resolved.stat().st_size
        if size == 0:
            raise ArtifactError(f"{relative_to_root(resolved)} is empty — nothing to send")
        if size > self.max_bytes:
            raise ArtifactError(
                f"{relative_to_root(resolved)} is {size / (1024 * 1024):.1f} MB, over the "
                f"{self.max_bytes / (1024 * 1024):.0f} MB artifact limit"
            )

        out = self._outbox_path(resolved)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, out)
        except OSError as exc:
            raise ArtifactError(f"could not stage {relative_to_root(resolved)}: {exc}") from exc

        artifact = Artifact(
            path=relative_to_root(resolved),
            absolute=out,
            size=size,
            mime=_guess_mime(suffix),
            origin=origin,
            created=time.time(),
        )
        self._remember(artifact)
        log.info("artifact staged: %s (%d bytes, origin=%s)", artifact.path, size, origin)
        return artifact

    def _outbox_path(self, source: Path) -> Path:
        """Where the outbox copy lives: <chat-token>/dedup-key.ext.

        The dedup key is name + size + mtime, so re-sending an unchanged file
        does not pile up copies, while a genuinely rewritten file gets a new one.
        """
        stat = source.stat()
        key = uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{stat.st_size}|{int(stat.st_mtime)}")
        return self.outbox_dir() / key.hex[:2] / f"{key.hex[:8]}-{source.name}"

    def outbox_dir(self) -> Path:
        return self.root / "data" / "outbox"

    # -- ingestion (documents arriving from a chat) ------------------------- #

    def ingest(self, data: bytes, name: str, chat_id: int | str, project_root: Path) -> Artifact:
        """Store an incoming document under ``workspace/uploads/<chat>/``.

        Returns an artifact whose ``path`` is the *project* location — that is
        where the model will read it from. The path is sanitised to a safe
        basename: Telegram filenames can be anything, and only the extension is
        trusted after sanitisation.
        """
        suffix = Path(name).suffix.lower().lstrip(".")
        if suffix and f".{suffix}" not in ALLOWED_SUFFIXES:
            raise ArtifactError(
                f"refusing to store a .{suffix} file; it is not on the artifact allowlist"
            )
        if not suffix:
            raise ArtifactError("the document has no extension; give it one and send it again")
        if len(data) == 0:
            raise ArtifactError("the document is empty")
        if len(data) > self.max_bytes:
            raise ArtifactError(
                f"the document is {len(data) / (1024 * 1024):.1f} MB, over the "
                f"{self.max_bytes / (1024 * 1024):.0f} MB limit"
            )

        safe = _safe_basename(name)
        if not safe:
            raise ArtifactError(f"the filename {name!r} is not usable")

        folder = project_root / "workspace" / "uploads" / str(chat_id)
        target = self._unique_path(folder / safe)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ArtifactError(f"could not store {safe}: {exc}") from exc

        return Artifact(
            path=relative_to_root(target),
            absolute=target,
            size=len(data),
            mime=_guess_mime(target.suffix),
            origin="telegram upload",
            created=time.time(),
        )

    @staticmethod
    def _unique_path(target: Path) -> Path:
        """Never overwrite: user-uploaded files are irreplaceable."""
        if not target.exists():
            return target
        stem, suffix = target.stem, target.suffix
        for i in range(1, 1000):
            candidate = target.with_name(f"{stem}-{i}{suffix}")
            if not candidate.exists():
                return candidate
        # 999 collisions is not a realistic session; last resort, timestamp it.
        return target.with_name(f"{stem}-{int(time.time())}{suffix}")

    # -- dedup and recall --------------------------------------------------- #

    def _remember(self, artifact: Artifact) -> None:
        self._recent[artifact.absolute.as_posix()] = artifact
        self._recent_order.append(artifact.absolute.as_posix())
        while len(self._recent_order) > self._recent_limit:
            stale = self._recent_order.pop(0)
            self._recent.pop(stale, None)

    def recent(self, limit: int = 10) -> list[Artifact]:
        """The most recent artifacts, newest last. For ``/status`` and tests."""
        return [self._recent[key] for key in self._recent_order[-limit:]]


def _safe_basename(name: str) -> str:
    """A filesystem-safe stem + extension from an arbitrary filename."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    cleaned = "".join(c if c.isalnum() or c in "._- " else "_" for c in stem).strip()
    cleaned = cleaned.strip(".") or "upload"
    return f"{cleaned[:80]}{suffix.lower()}"


__all__ = [
    "Artifact",
    "ArtifactStore",
    "ArtifactError",
    "ALLOWED_SUFFIXES",
    "DEFAULT_MAX_BYTES",
]
