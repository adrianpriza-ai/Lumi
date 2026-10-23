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

from . import __version__
from .config import Config
from .paths import relative_to_root
from .tools import safety
from .tools.registry import ToolRegistry
from .util.log import get_logger
from .util.text import format_error

log = get_logger(__name__)

OK = "ok"
WARN = "warn"
FAIL = "fail"


def _reasoning_note(config: Config) -> str:
    """How the model is being driven, appended to the llm check.

    Worth stating even when it is off: the most common report of a reasoning
    model that "just does not work" is a turn that 400s on temperature, and
    this line is where the answer is.
    """
    if not config.llm.reasoning:
        return ", reasoning off"
    effort = config.llm.effort_of() or "provider default"
    return (
        f", reasoning on at {effort} "
        f"(temperature omitted, {config.llm.reasoning_tokens:,} thinking tokens)"
    )


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
    checks.append(Check("lumi", OK, __version__))
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
    checks.append(
        Check("PERSONALITY.md", OK if config.personality_file.is_file() else FAIL,
              relative_to_root(config.personality_file))
    )

    for label, path in (
        ("data dir", config.data_dir),
        ("history dir", config.history_dir),
        ("memory dir", config.memory_dir),
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
    if config.bot.proxy_url:
        checks.append(
            Check(
                "telegram proxy",
                OK,
                f"{config.bot.proxy_url} (Bot API via bot.proxy_url)",
            )
        )
    retries = config.bot.bootstrap_retries
    checks.append(
        Check(
            "telegram network",
            OK,
            f"connect timeout {config.bot.connect_timeout:g}s, "
            f"{'unlimited' if retries < 0 else f'{retries + 1}'} bootstrap attempt(s)",
        )
    )
    keys = config.llm.api_keys()
    if keys:
        detail = (
            f"{len(keys)} key(s), {config.llm.strategy_of()} strategy"
            if len(keys) > 1
            else "set"
        )
    else:
        detail = "missing — the bot cannot answer anything"
    checks.append(Check(f"{config.llm.api_key_env}", OK if keys else FAIL, detail))
    vision = config.llm.vision_model_of()
    checks.append(
        Check(
            "llm",
            OK,
            f"{config.llm.model_of()} via {config.llm.base_url_of()} "
            f"(from {config.llm.where_from()}), keys: {config.llm.strategy_of()}"
            + (f", vision: {vision}" if vision else "")
            + _reasoning_note(config),
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
    # Tavily has a keyless tier (provider stays available without a key);
    # Firecrawl and Exa do not — an unset key means the provider is skipped.
    # Each provider also takes a key pool, so the doctor reports the strategy
    # when more than one key is configured.
    for provider_name, cfg in (
        ("tavily", config.tools.web.tavily),
        ("firecrawl", config.tools.web.firecrawl),
        ("exa", config.tools.web.exa),
    ):
        if not cfg.enabled:
            checks.append(Check(cfg.api_key_env, WARN, "provider disabled in config"))
            continue

        keys = cfg.api_keys()
        if not keys:
            if provider_name == "tavily":
                checks.append(
                    Check(
                        cfg.api_key_env,
                        OK,
                        "unset — using tavily's keyless tier (free, low rate limit)",
                    )
                )
            else:
                checks.append(
                    Check(
                        cfg.api_key_env,
                        WARN,
                        "unset — that provider will be skipped",
                    )
                )
        elif len(keys) == 1:
            checks.append(Check(cfg.api_key_env, OK, "set"))
        else:
            checks.append(
                Check(
                    cfg.api_key_env,
                    OK,
                    f"{len(keys)} key(s), {cfg.key_strategy_of()} strategy",
                )
            )

    # -- context7 (a tool, not a web provider, but the env var is the same shape) #
    ctx7 = config.tools.context7
    if not ctx7.enabled:
        checks.append(Check(ctx7.api_key_env, WARN, "tool disabled in config"))
    elif ctx7.api_keys():
        detail = (
            f"{len(ctx7.api_keys())} key(s), {ctx7.key_strategy_of()} strategy"
            if len(ctx7.api_keys()) > 1
            else "set"
        )
        checks.append(Check(ctx7.api_key_env, OK, detail))
    else:
        checks.append(
            Check(
                ctx7.api_key_env,
                WARN,
                "unset — the context7 tool is hidden from the model until it is set",
            )
        )

    # Both files are read (the project's .mcp.json and ~/.agents/mcp.json), so
    # the check runs when either exists.
    try:
        from .tools.web.providers.mcp_provider import MCPProvider

        mcp = MCPProvider(config.tools.web.mcp, config.root)
        if any(path.is_file() for path in mcp.config_paths):
            servers = mcp.servers()
            unresolved = [f"{s.name} (needs {s.unresolved()})" for s in servers if s.unresolved()]
            detail = f"{len(servers)} server(s): {', '.join(s.name for s in servers) or 'none'}"
            if unresolved:
                checks.append(Check("mcp servers", WARN, f"{detail}; unset: {', '.join(unresolved)}"))
            else:
                checks.append(Check("mcp servers", OK, detail))
        else:
            checks.append(
                Check("mcp servers", WARN, f"no {' or '.join(mcp.sources)} found")
            )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("mcp servers", FAIL, format_error(exc)))

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

            built = build_registry(config, config.build_memory_store())
            for tool in built.all():
                ok, reason = tool.available()
                checks.append(Check(f"tool {tool.name}", OK if ok else WARN, reason or tool.summary_line()))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("tools", FAIL, format_error(exc)))

    # -- memory ------------------------------------------------------------ #
    try:
        from .context import ContextWindow

        store = config.build_memory_store()
        memory = store.get(config.default_memory_scope)
        size = memory.prompt_size(ContextWindow(config).memory_limit())
        scopes = store.scopes()
        detail = (
            f"{size['chars']:,} chars, {memory.stats()['managed_count']} remembered fact(s) "
            f"in scope {config.default_memory_scope} · {len(scopes)} chat scope(s) on disk"
        )
        if size["over"]:
            detail += (
                f" — over the {size['budget']:,}-char budget, so the oldest "
                f"{size['dropped']} are not sent (raise llm.max_memory_chars)"
            )
        checks.append(Check("memory", WARN if size["over"] else OK, detail))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("memory", WARN, format_error(exc)))

    # -- history ----------------------------------------------------------- #
    try:
        history_dir = config.history_dir
        files = list(history_dir.glob("*.jsonl")) if history_dir.is_dir() else []
        # read_text per file, not an unclosed handle per file.
        total = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)
        checks.append(Check("history", OK, f"{len(files)} chat(s), {total} turns"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("history", WARN, format_error(exc)))

    return checks


def platform_line() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


__all__ = ["Check", "run_checks", "platform_line", "OK", "WARN", "FAIL"]
