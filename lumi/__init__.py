"""Lumi — a portable Telegram AI agent.

Everything is resolved relative to the project root (the directory containing
``pyproject.toml``), so a ``git clone`` anywhere is a working install with
nothing to copy. An optional user layer — ``~/.config/lumi``, moved by
``$LUMI_USER_CONFIG`` or ``$XDG_CONFIG_HOME`` — can carry a model catalog,
skills and config defaults that follow the owner across projects; without it
nothing else is needed.
"""

from __future__ import annotations

__version__ = "0.5.0"
__all__ = ["__version__"]
