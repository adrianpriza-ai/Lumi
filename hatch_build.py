"""Compile the Rust UI into the wheel — Lumi's equivalent of `vibe-rs`.

The `lumi` command and the Rust binary it execs live in one artifact: the wheel.
That is the packaging half of "one command, not two binaries to install": a
`pip install .` produces a `lumi` that opens the UI with no separate cargo step,
the way Vibe bundles its native `vibe-rs` into the Python package.

Editable installs skip the hook: a checkout already has the crate's
`target/release/` build, and the launcher finds it without copying anything.

If cargo is missing and no prebuilt binary exists, the wheel is built without
the UI rather than failing the install; `lumi` then prints the build command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parent
CRATE = ROOT / "lumi-tui"
BINARY = "lumi-tui.exe" if os.name == "nt" else "lumi-tui"


def platform_tag() -> str:
    """The platform part of this machine's wheel tag, e.g. ``linux_x86_64``."""
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


class UiBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel" or build_data.get("editable"):
            return
        binary = self._build()
        if binary is None:
            self.app.display_warning(
                "lumi: cargo is not on PATH and no prebuilt lumi-tui exists, so "
                "this wheel ships no Rust UI (`lumi` will explain how to build it)."
            )
            return
        build_data["force_include"][str(binary)] = f"lumi/_bin/{BINARY}"
        # A wheel carrying a native executable is not pure Python and cannot be
        # installed on another platform. It is still interpreter-independent, so
        # one artifact covers every Python version the project supports.
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag()}"

    def _build(self) -> Path | None:
        cargo = shutil.which("cargo")
        if cargo:
            subprocess.run(
                [
                    cargo,
                    "build",
                    "--release",
                    "--locked",
                    "--manifest-path",
                    str(CRATE / "Cargo.toml"),
                ],
                cwd=self.root,
                check=True,
            )
        built = CRATE / "target" / "release" / BINARY
        return built if built.is_file() else None
