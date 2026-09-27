"""Logging setup: console plus an optional rotating file under ``data/logs``."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;199m",
}
_RESET = "\033[0m"


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.color:
            return text
        tint = _LEVEL_COLORS.get(record.levelname)
        return f"{tint}{text}{_RESET}" if tint else text


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    root.handlers.clear()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(_ConsoleFormatter(color=sys.stderr.isatty()))
    root.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            rotating = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            rotating.setLevel(logging.DEBUG)
            rotating.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
            )
            root.addHandler(rotating)
        except OSError as exc:  # read-only fs, bad path, ... — console still works
            root.warning("file logging disabled: %s", exc)

    # These libraries are extremely chatty at INFO.
    for noisy in ("httpx", "httpcore", "telegram.ext.Updater", "telegram.ext.Application"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
