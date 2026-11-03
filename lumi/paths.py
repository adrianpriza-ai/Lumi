"""Path resolution.

Every runtime artefact lives under the project directory. The only escape
hatch is the ``LUMI_HOME`` environment variable, which is meant for tests and
for people who want state elsewhere — it is never required.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: ``<repo>/lumi/paths.py`` -> parents[1] is ``<repo>``.
PACKAGE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT: Path = PACKAGE_DIR.parent

#: The Rust frontend's binary name on this platform.
UI_BINARY_NAME = "lumi-tui.exe" if os.name == "nt" else "lumi-tui"


def project_root() -> Path:
    """Directory that acts as the sandbox root and the base for all relative paths."""
    override = os.environ.get("LUMI_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_PROJECT_ROOT


def ui_binary() -> Path | None:
    """The Rust UI binary, or None when it has not been built.

    A wheel bundles it at ``lumi/_bin`` (see ``hatch_build.py``); a checkout
    builds it under ``lumi-tui/target``. ``$LUMI_TUI`` overrides both, while
    ``LUMI_HOME`` deliberately does not move the build — it only relocates
    runtime state.
    """
    override = os.environ.get("LUMI_TUI")
    if override:
        return Path(override)
    bundled = PACKAGE_DIR / "_bin" / UI_BINARY_NAME
    if bundled.is_file():
        return bundled
    for profile in ("release", "debug"):
        candidate = DEFAULT_PROJECT_ROOT / "lumi-tui" / "target" / profile / UI_BINARY_NAME
        if candidate.is_file():
            return candidate
    found = shutil.which("lumi-tui")
    return Path(found) if found else None


def resolve(relative: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Resolve *relative* against the project root (or *base*) without touching the FS."""
    root = base if base is not None else project_root()
    p = Path(relative).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def is_within(candidate: Path, root: Path) -> bool:
    """True when *candidate* is *root* or lives underneath it.

    Uses resolved paths on both sides so ``../../etc`` cannot sneak through.
    """
    try:
        c = candidate.resolve()
        r = root.resolve()
    except OSError:  # pragma: no cover - unresolvable path (broken symlink loop)
        return False
    return c == r or r in c.parents


def relative_to_root(path: Path) -> str:
    """Human-facing path, relative to the project root when possible."""
    root = project_root()
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)
