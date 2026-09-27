"""Configuration loading.

Order of precedence, lowest to highest:

1. the dataclass defaults in this module
2. ``config.toml`` in the project root (committed; behaviour only)
3. ``config.local.toml`` in the project root (gitignored; personal overrides)
4. environment variables named ``LUMI__<SECTION>__<KEY>``
5. ``.env`` in the project root, loaded for interpolation of the above

``config.toml`` is parsed with :mod:`tomllib` from the standard library, so
there is no YAML/TOML dependency to install.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .paths import project_root, resolve
from .util.log import get_logger

log = get_logger(__name__)

ENV_PREFIX = "LUMI__"
CONFIG_FILENAME = "config.toml"
LOCAL_CONFIG_FILENAME = "config.local.toml"
ENV_FILENAME = ".env"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing something the bot cannot start without."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class BotConfig:
    startup_chat_id: str = ""
    require_owner: bool = True
    log_prefix: str = "lumi"


@dataclass(slots=True)
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0.7
    max_tokens: int = 2000
    max_tool_iterations: int = 8
    history_turns: int = 20
    max_memory_chars: int = 6000

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


@dataclass(slots=True)
class ShellConfig:
    enabled: bool = True
    cwd: str = "workspace"
    timeout_seconds: int = 60
    max_output_chars: int = 6000
    ask_before_risky: bool = True
    home: str = "project"  # "project" | "inherit"
    extra_deny: list[str] = field(default_factory=list)
    extra_confirm: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FilesConfig:
    enabled: bool = True
    writable: list[str] = field(default_factory=lambda: ["workspace"])
    readable_from_project: bool = True
    max_read_chars: int = 20_000
    max_write_chars: int = 200_000

    def writable_roots(self) -> list[Path]:
        return [resolve(p) for p in self.writable]


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    auto_remember: bool = True


@dataclass(slots=True)
class TavilyConfig:
    enabled: bool = True
    api_key_env: str = "TAVILY_API_KEY"
    search_depth: str = "basic"
    topic: str = "general"
    include_answer: bool = True
    include_raw_content: bool = True
    base_url: str = "https://api.tavily.com"

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


@dataclass(slots=True)
class FirecrawlConfig:
    enabled: bool = True
    api_key_env: str = "FIRECRAWL_API_KEY"
    base_url: str = "https://api.firecrawl.dev"
    only_main_content: bool = True
    auto_scrape_top_n: int = 3

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


@dataclass(slots=True)
class MCPConfig:
    enabled: bool = True
    config_file: str = ".mcp.json"
    timeout_seconds: int = 60
    search_tools: list[str] = field(
        default_factory=lambda: ["tavily_search", "firecrawl_search", "search"]
    )
    fetch_tools: list[str] = field(
        default_factory=lambda: ["tavily_extract", "firecrawl_scrape", "scrape", "extract"]
    )


@dataclass(slots=True)
class WebConfig:
    enabled: bool = True
    provider_order: list[str] = field(default_factory=lambda: ["tavily", "firecrawl", "mcp"])
    max_results: int = 5
    max_content_chars: int = 6000
    timeout_seconds: int = 90
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    firecrawl: FirecrawlConfig = field(default_factory=FirecrawlConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)


@dataclass(slots=True)
class ToolsConfig:
    enabled: list[str] = field(default_factory=lambda: ["shell", "files", "memory", "web"])
    shell: ShellConfig = field(default_factory=ShellConfig)
    files: FilesConfig = field(default_factory=FilesConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def is_enabled(self, name: str) -> bool:
        return name in self.enabled


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "data/logs/lumi.log"


@dataclass(slots=True)
class Config:
    """The whole tree. ``lumi.config.load_config()`` builds one of these."""

    bot: BotConfig = field(default_factory=BotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Runtime-only, filled in by load_config().
    root: Path = field(default_factory=project_root)
    personality_file: Path = field(default_factory=lambda: project_root() / "PERSONALITY.md")
    memory_file: Path = field(default_factory=lambda: project_root() / "MEMORY.md")
    data_dir: Path = field(default_factory=lambda: project_root() / "data")
    config_path: Path | None = None

    # -- derived paths ----------------------------------------------------- #

    @property
    def history_dir(self) -> Path:
        return self.data_dir / "history"

    @property
    def log_file(self) -> Path:
        return resolve(self.logging.file, base=self.root)

    @property
    def shell_cwd(self) -> Path:
        return resolve(self.tools.shell.cwd, base=self.root)

    def env(self, name: str) -> str | None:
        return os.environ.get(name) or None

    @property
    def telegram_token(self) -> str | None:
        return self.env("TELEGRAM_BOT_TOKEN")

    @property
    def owner_id(self) -> int | None:
        raw = self.env("TELEGRAM_OWNER_ID")
        if not raw:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            log.error("TELEGRAM_OWNER_ID=%r is not an integer; owner gating disabled", raw)
            return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_dotenv(root: Path) -> None:
    """Load ``.env`` from the project root without clobbering real env vars."""
    path = root / ENV_FILENAME
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv as _load
    except ImportError:  # pragma: no cover - dependency always present in practice
        log.warning("python-dotenv is not installed; skipping %s", path)
        return
    _load(path, override=False)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}


def _env_overrides() -> dict[str, Any]:
    """Collect ``LUMI__SECTION__KEY=value`` pairs into a nested dict."""
    out: dict[str, Any] = {}
    for raw_key, value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in raw_key[len(ENV_PREFIX) :].split("__") if p]
        if len(parts) < 2:
            continue
        cursor = out
        for part in parts[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):  # pragma: no cover - conflicting override
                nxt = cursor[part] = {}
            cursor = nxt
        cursor[parts[-1]] = value
    return out


_HINT_CACHE: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints for *cls*.

    ``from __future__ import annotations`` turns every annotation into a string,
    so dataclass ``field.type`` is unusable for coercion. This resolves them once
    per class and caches the result.
    """
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = get_type_hints(cls)
    return _HINT_CACHE[cls]


def _is_optional(hint: Any) -> bool:
    return get_origin(hint) in (Union, UnionType) and type(None) in get_args(hint)


def _coerce(value: Any, hint: Any) -> Any:
    """Coerce a raw config value (often a string from the env) to *hint*."""
    if hint is Any or hint is None:
        return value

    origin = get_origin(hint)

    # Optional[X] / X | None
    if origin in (Union, UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        if len(args) == 1:
            return _coerce(value, args[0])
        return value

    if is_dataclass(hint):
        if isinstance(value, dict):
            return _build(hint, value)
        log.warning("expected a table for %s, got %r; keeping the default", hint.__name__, value)
        return None

    if origin in (list, list):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        args = get_args(hint)
        item_hint = args[0] if args else Any
        return [_coerce(item, item_hint) for item in value]

    if hint is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if hint is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            # Not fatal: a typo in an env var should not stop the bot from
            # booting. _build logs it and keeps the default.
            log.warning("expected an integer, got %r", value)
            return None

    if hint is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning("expected a number, got %r", value)
            return None

    if hint is str:
        return str(value)

    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a nested dict, ignoring unknown keys.

    Keys that are not fields of *cls* are dropped with a debug log, so a config
    written for a future version still boots.
    """
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    for key, raw in data.items():
        if key not in known:
            log.debug("ignoring unknown config key %s.%s", cls.__name__, key)
            continue
        hint = hints.get(key)
        value = _coerce(raw, hint)
        if value is None and raw is not None and not _is_optional(hint):
            # Coercion failed for a field with no null state: keep the default
            # rather than silently blanking it out.
            log.warning("%s.%s: cannot interpret %r; keeping the default", cls.__name__, key, raw)
            continue
        kwargs[key] = value
    return cls(**kwargs)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; *overlay* wins."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(root: Path | None = None) -> Config:
    """Build a :class:`Config` from disk plus environment overrides."""
    root = (root or project_root()).resolve()
    load_dotenv(root)

    merged: dict[str, Any] = {}
    config_path: Path | None = None
    for name in (CONFIG_FILENAME, LOCAL_CONFIG_FILENAME):
        path = root / name
        if path.is_file():
            merged = _merge(merged, _read_toml(path))
            config_path = config_path or path

    if config_path is None:
        log.info("no %s found; using built-in defaults", CONFIG_FILENAME)
    else:
        log.debug("loaded config from %s", config_path)

    merged = _merge(merged, _env_overrides())

    # Unknown top-level sections are ignored rather than fatal, so a config
    # written for a newer version still boots.
    known = {f.name for f in fields(Config)}
    data = {k: v for k, v in merged.items() if k in known}

    config = _build(Config, data)
    config.root = root
    config.config_path = config_path

    data_dir = resolve(config.data_dir, base=root)
    config.data_dir = data_dir
    config.personality_file = resolve("PERSONALITY.md", base=root)
    config.memory_file = resolve("MEMORY.md", base=root)
    return config


def validate(config: Config) -> list[str]:
    """Return a list of human-readable problems. Empty means good to go."""
    problems: list[str] = []

    if not config.llm.api_key():
        problems.append(
            f"{config.llm.api_key_env} is unset — the bot cannot talk to a model. "
            "Add it to .env, or point llm.base_url at a local server."
        )
    if config.telegram_token is None:
        problems.append("TELEGRAM_BOT_TOKEN is unset — the Telegram bot cannot start.")
    if config.bot.require_owner and config.owner_id is None:
        problems.append(
            "TELEGRAM_OWNER_ID is unset. The shell and file tools are locked to the owner, "
            "so the bot refuses to start without it. Find your id via "
            "https://api.telegram.org/bot<TOKEN>/getUpdates"
        )
    if not config.personality_file.is_file():
        problems.append(f"PERSONALITY.md not found at {config.personality_file}")
    if not config.memory_file.is_file():
        problems.append(f"MEMORY.md not found at {config.memory_file}")

    for name in config.tools.enabled:
        if name not in {"shell", "files", "memory", "web"}:
            problems.append(f"unknown tool {name!r} in tools.enabled")

    providers = [p for p in config.tools.web.provider_order if p not in {"tavily", "firecrawl", "mcp"}]
    if providers:
        problems.append(f"unknown web providers in tools.web.provider_order: {', '.join(providers)}")

    return problems
