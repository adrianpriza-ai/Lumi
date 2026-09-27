"""Environment diagnostics.

Answers "why isn't this working?" without guesswork. Runs from the Telegram
``/doctor`` command and from ``lumi doctor`` in a terminal, and never raises:
every check records its own failure.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass

from .config import Config
from .memory import MemoryFile
from .paths import relative_to_root
from .tools import safety
from .tools.registry import ToolRegistry
from .util.log import get_logger
from .util.text import format_error

log = get_logger(__name__)

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str = ""

    @property
    def icon(self) -> str:
        return {OK: "✓", WARN: "!", FAIL: "✗"}.get(self.status, "?")

    def render(self) -> str:
        line = f"{self.icon} {self.name}"
        return f"{line} — {self.detail}" if self.detail else line


def _check_import(module: str, label: str, extra: str = "") -> Check:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        return Check(label, FAIL, f"not installed ({exc})")
    version = getattr(mod, "__version__", None)
    detail = f"{version}" if version else "installed"
    if extra:
        detail += f", {extra}"
    return Check(label, OK, detail)


def run_checks(config: Config, registry: ToolRegistry | None = None) -> list[Check]:
    checks: list[Check] = []

    # -- python and packages ---------------------------------------------- #
    version = ".".join(str(part) for part in sys.version_info[:3])
    too_old = sys.version_info < (3, 11)
    checks.append(Check("python", FAIL if too_old else OK, f"{version} at {sys.executable}"))
    if too_old:
        checks.append(Check("python version", FAIL, "3.11+ is required (tomllib, StrEnum)"))

    for module, label, extra in (
        ("telegram", "python-telegram-bot", ""),
        ("openai", "openai", ""),
        ("tavily", "tavily-python", ""),
        ("firecrawl", "firecrawl", ""),
        ("mcp", "mcp", ""),
        ("dotenv", "python-dotenv", ""),
    ):
        checks.append(_check_import(module, label, extra))

    # -- layout ------------------------------------------------------------ #
    checks.append(Check("project root", OK, str(config.root)))
    checks.append(
        Check(
            "config",
            OK if config.config_path else WARN,
            str(config.config_path) if config.config_path else "no config.toml; using built-in defaults",
        )
    )
    for label, path in (
        ("PERSONALITY.md", config.personality_file),
        ("MEMORY.md", config.memory_file),
    ):
        checks.append(Check(label, OK if path.is_file() else FAIL, relative_to_root(path)))

    for label, path in (
        ("data dir", config.data_dir),
        ("history dir", config.history_dir),
        ("shell cwd", config.shell_cwd),
    ):
        exists = path.is_dir()
        status = OK if exists else WARN
        detail = relative_to_root(path) if exists else f"{relative_to_root(path)} (will be created on demand)"
        checks.append(Check(label, status, detail))

    # -- credentials ------------------------------------------------------- #
    token = config.telegram_token
    checks.append(
        Check(
            "TELEGRAM_BOT_TOKEN",
            OK if token else FAIL,
            f"set (…{token[-6:]})" if token else "missing — add it to .env",
        )
    )
    owner = config.owner_id
    checks.append(
        Check(
            "TELEGRAM_OWNER_ID",
            OK if owner else FAIL,
            f"{owner} (only this id gets the shell and file tools)" if owner else "missing — see .env.example",
        )
    )
    key = config.llm.api_key()
    checks.append(
        Check(
            f"{config.llm.api_key_env}",
            OK if key else FAIL,
            "set" if key else "missing — the bot cannot answer anything",
        )
    )
    checks.append(
        Check(
            "llm",
            OK,
            f"{config.llm.model_of()} via {config.llm.base_url_of()} (from {config.llm.where_from()})",
        )
    )

    # -- shell ------------------------------------------------------------- #
    shell_ok = shutil.which("bash") or shutil.which("sh")
    checks.append(
        Check("shell", OK if shell_ok else WARN, shell_ok or "no bash/sh on PATH; commands may fail")
    )
    checks.append(Check("shell policy", OK, safety.describe_policy()))
    checks.append(
        Check(
            "shell cwd",
            OK,
            f"{relative_to_root(config.shell_cwd)} (HOME pinned to the project: "
            f"{config.tools.shell.home == 'project'})",
        )
    )

    # -- web providers ----------------------------------------------------- #
    for env_name, enabled in (
        (config.tools.web.tavily.api_key_env, config.tools.web.tavily.enabled),
        (config.tools.web.firecrawl.api_key_env, config.tools.web.firecrawl.enabled),
    ):
        if not enabled:
            checks.append(Check(env_name, WARN, "provider disabled in config"))
        else:
            present = bool(config.env(env_name))
            checks.append(
                Check(env_name, OK if present else WARN, "set" if present else "unset — that provider will be skipped")
            )

    mcp_path = config.root / config.tools.web.mcp.config_file
    if mcp_path.is_file():
        try:
            from .tools.web.providers.mcp_provider import MCPProvider

            provider = MCPProvider(config.tools.web.mcp, config.root)
            servers = provider.servers()
            unresolved = [f"{s.name} (needs {s.unresolved()})" for s in servers if s.unresolved()]
            detail = f"{len(servers)} server(s): {', '.join(s.name for s in servers) or 'none'}"
            if unresolved:
                checks.append(Check("mcp servers", WARN, f"{detail}; unset: {', '.join(unresolved)}"))
            else:
                checks.append(Check("mcp servers", OK, detail))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("mcp servers", FAIL, format_error(exc)))
    else:
        checks.append(Check("mcp servers", WARN, f"no {mcp_path.name} in the project"))

    # -- tools ------------------------------------------------------------- #
    if registry is not None:
        for tool in registry.all():
            ok, reason = tool.available()
            checks.append(
                Check(f"tool {tool.name}", OK if ok else WARN, reason or tool.summary_line())
            )
        checks.append(Check("tools registered", OK, ", ".join(registry.names()) or "none"))
    else:
        try:
            from .tools import build_registry

            memory = MemoryFile(config.memory_file, config.llm.max_memory_chars)
            memory.load()
            built = build_registry(config, memory)
            for tool in built.all():
                ok, reason = tool.available()
                checks.append(Check(f"tool {tool.name}", OK if ok else WARN, reason or tool.summary_line()))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("tools", FAIL, format_error(exc)))

    # -- history ----------------------------------------------------------- #
    try:
        history_dir = config.history_dir
        files = list(history_dir.glob("*.jsonl")) if history_dir.is_dir() else []
        total = sum(1 for f in files for _ in f.open("r", encoding="utf-8"))
        checks.append(Check("history", OK, f"{len(files)} chat(s), {total} turns"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("history", WARN, format_error(exc)))

    return checks


def platform_line() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


__all__ = ["Check", "run_checks", "platform_line", "OK", "WARN", "FAIL"]
