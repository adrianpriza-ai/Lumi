from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import pytest

import lumi.tools.mcp.sandbox as sandbox
from lumi.tools.mcp.sandbox import run_code


async def never(*_args):
    raise AssertionError("no tool call should happen")


async def test_refuses_execution_when_isolation_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_bubblewrap_command", lambda _scratch: None)

    run = await run_code(
        code="print('should not run')", tools=[], on_call=never, timeout=1, max_calls=1
    )

    assert not run.ok
    assert "requires Linux, bubblewrap" in run.error


# --------------------------------------------------------------------------- #
# which interpreter prefixes the jail is willing to mount
# --------------------------------------------------------------------------- #


def _command_for(executable: str, base_prefix: str, monkeypatch) -> list[str] | None:
    """``_bubblewrap_command`` as if run by the interpreter described.

    ``sys`` is replaced rather than the interpreter, and bubblewrap is assumed
    present, so the decision depends only on the paths — testable on any host.
    """
    import sys as real_sys
    from types import SimpleNamespace

    fake = SimpleNamespace(
        platform=real_sys.platform,
        _base_executable=executable,
        executable=executable,
        base_prefix=base_prefix,
    )
    monkeypatch.setattr(sandbox, "sys", fake)
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: "/usr/bin/bwrap")
    return sandbox._bubblewrap_command(Path("/tmp"))


def _ro_binds(command: list[str]) -> list[tuple[str, str]]:
    return [(command[i + 1], command[i + 2]) for i, arg in enumerate(command) if arg == "--ro-bind"]


def test_the_root_is_never_a_mount_ancestor(monkeypatch) -> None:
    """Every path is \"under\" /, so the root must be an equality check. The
    first version of the relocated-prefix support got this wrong and refused
    every interpreter that was not under /usr."""
    assert sandbox._mountable_prefix(Path("/"), Path("/usr")) is False
    assert sandbox._mountable_prefix(Path("/usr"), Path("/usr")) is True
    assert sandbox._mountable_prefix(Path("/usr/local"), Path("/usr")) is True


def _outside_runtime_trees(path: Path) -> bool:
    """Whether *path* is a mountable directory not already under /usr.

    The relocated-runtime case needs a prefix outside /usr, not just one the
    sandbox would accept — /usr is mounted whole, so a prefix under it would
    take the other branch. The root is an equality check: every path is
    "under" /, so an ancestor test here would rule out every location.
    """
    resolved = path.resolve()
    if resolved == Path("/").resolve() or resolved.is_relative_to(Path("/usr").resolve()):
        return False
    return sandbox._mountable_prefix(path, Path("/usr").resolve())


def _relocated_prefix() -> tuple[Path, Path]:
    """A real directory outside every refused tree, where a runtime might live.

    pytest's ``tmp_path`` is under /tmp and HOME is refused too — both on
    purpose, since code mode must not expose them to the jailed interpreter.
    So look for a writable spot outside the refused trees instead: the runner's
    tool cache under /opt in CI, a user-owned tree under /opt on a workstation.
    Returns the temporary base (to remove afterwards) and the fake prefix.
    """
    import tempfile

    for top in sorted(Path("/").iterdir()):
        if not top.is_dir() or not _outside_runtime_trees(top):
            continue
        bases = [top]
        # A directory we cannot list cannot host our runtime either.
        with suppress(OSError):
            bases.extend(
                child
                for child in top.iterdir()
                if child.is_dir() and _outside_runtime_trees(child)
            )
        for base in bases:
            try:
                root = Path(tempfile.mkdtemp(prefix="lumi-runtime-", dir=str(base)))
            except OSError:
                continue
            prefix = root / "Python" / "3.11.9" / "x64"
            (prefix / "bin").mkdir(parents=True)
            (prefix / "lib").mkdir()
            return root, prefix
    pytest.skip("no writable directory outside the sandbox refuse-list to host a runtime")


def test_a_relocated_runtime_is_mounted_read_only_at_its_own_path(monkeypatch) -> None:
    """A Python installed outside /usr — a pyenv, a distro package, the CI tool
    cache — has its prefix mounted read-only at the same path, so the jailed
    interpreter finds its stdlib where it expects it."""
    import shutil

    root, prefix = _relocated_prefix()
    try:
        command = _command_for(str(prefix / "bin" / "python3"), str(prefix), monkeypatch)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert command is not None, "a real directory outside the refused ones must be mountable"
    binds = _ro_binds(command)
    assert ("/usr", "/usr") in binds
    assert (str(prefix), str(prefix)) in binds
    # Read-only throughout: no bind in the command is writable.
    assert all(arg != "--bind" and arg != "--rw-bind" for arg in command)
    assert "--unshare-all" in command and "--clearenv" in command
    assert "-I" in command and "-S" in command


def test_a_prefix_that_is_too_broad_is_refused(monkeypatch) -> None:
    """A prefix that would expose a home, a temp dir or a system directory is
    not mountable, and the sandbox refuses to run rather than jail loosely."""
    for too_broad in ("/home/runner/.pyenv/versions/3.12", "/tmp/py", "/var/lib/py", "/etc/py"):
        assert sandbox._mountable_prefix(Path(too_broad), Path("/usr")) is False, too_broad
        assert _command_for(f"{too_broad}/bin/python3", too_broad, monkeypatch) is None, too_broad


def test_the_root_as_prefix_is_refused_end_to_end(monkeypatch) -> None:
    """A runtime whose prefix is / cannot be confined, so no command is built —
    even when the executable resolves under /usr, where its other prefix would
    be dropped as already mounted. That is the case that used to slip through:
    subtracting / from the prefix set hid the refusal and built a command."""
    for executable in ("/usr/bin/python3", "/bin/python3"):  # /bin → /usr/bin on many distros
        assert _command_for(executable, "/", monkeypatch) is None, executable
    assert sandbox._mountable_prefix(Path("/"), Path("/usr")) is False


def test_a_missing_runtime_directory_is_refused(monkeypatch) -> None:
    """A prefix that does not exist is not a runtime: fail closed."""
    assert _command_for("/opt/nope/bin/python3", "/opt/nope", monkeypatch) is None
