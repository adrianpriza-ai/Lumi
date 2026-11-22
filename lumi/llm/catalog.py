"""The model catalog: several providers, each with models, in one JSONC file.

``[llm]`` describes *one* endpoint. The catalog describes many, so
``/model zen/minimax-m2.7`` can point a chat at another provider, key pool and
context size without a restart or a config edit. The format follows the
OpenCode-style ``models.json`` the owner keeps (JSONC: comments and trailing
commas allowed), adapted for a direct client instead of a proxy:

.. code-block:: jsonc

    {
      "defaultModel": "zen/nemotron-3.5-lightning-free",
      "provider": {
        "Zen": {
          "id": "zen",                    // ids are "<provider-id>/<model>"
          "aliases": ["opencode"],
          "options": {
            "baseURL": "https://opencode.ai/zen/v1",
            "apiKey": "",                 // literal, ${VAR}, or "file:/path"
            "apiKeys": [],                // pool: array, space-separated, file:
            "headers": {},
            "timeoutSecs": 300,
            "keyStrategy": "rotate"       // first|rotate|random|sticky
          },
          "models": {
            "nemotron-3.5-lightning-free": {
              "name": "Nemotron 3.5 Lightning",
              "upstreamName": "…",        // id sent upstream; default: the key
              "as": "nemotron",           // extra alias for selection
              "limit": { "context": 1000000, "output": 128000 },
              "description": "…",
              "modalities": { "input": ["text", "image"], "output": ["text"] },
              "thinking": ["low", "high"]
            }
          }
        }
      }
    }

Two files are loaded, the user home first and the project over it, both
optional, merged per provider with :func:`lumi.config._merge`:

1. ``~/.config/lumi/models.json`` — the owner's personal catalog
2. ``<project>/.models.json`` — the project's catalog (a hidden file: it may
   hold literal keys, and secret files must not read like documentation)

Proxy-only keys the format may still carry (``maxInFlight``,
``"keyless"`` passthrough) are accepted and ignored — a client talks to the
provider itself. ``kind`` marks non-chat models (``image``, ``video``, ``tts``,
``stt``, ``ocr``, ``moderation``); only chat models are selectable with
``/model``, and a kind is inferred from the modalities when it is not written.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import LLMConfig, _merge
from ..paths import user_config_dir
from ..util.log import get_logger
from .openai_compat import OpenAICompatClient

log = get_logger(__name__)

#: The user-level catalog file, inside :func:`lumi.paths.user_config_dir`.
CATALOG_FILENAME = "models.json"
#: The project-level catalog. Hidden on purpose: literal keys live here, and
#: a file the shell tools list should not read like a sample.
PROJECT_CATALOG_FILENAME = ".models.json"

#: Kinds a model entry can carry. Only ``"chat"`` models are selectable.
KINDS = ("chat", "image", "image_edit", "video", "tts", "stt", "ocr", "moderation")

#: ``keyStrategy`` in the catalog mapped onto :data:`lumi.config.KEY_STRATEGIES`.
KEY_STRATEGY_MAP = {
    "first": "fallback",
    "fallback": "fallback",
    "rotate": "round_robin",
    "round_robin": "round_robin",
    "random": "random",
    # "sticky" pins one conversation to one key for prompt-cache hits; the
    # KeyPool has no per-conversation mode yet, so the safest pool behaviour
    # stands. Recorded in IDEA_BANK.md as future work.
    "sticky": "fallback",
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_KEY_FILE_BYTES = 1_048_576  # a key file is lines of secrets, never a dataset


class CatalogError(ValueError):
    """A model id could not be resolved, or a catalog could not be read."""


# --------------------------------------------------------------------------- #
# JSONC
# --------------------------------------------------------------------------- #


def strip_jsonc(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments and trailing commas from *text*.

    One pass with a string-aware state machine, because comments and trailing
    commas must both be ignored *inside* strings and honoured outside them:
    a ``"url": "https://x//y"`` must survive, and ``{ "a": 1, }`` must parse.
    The newline a line comment ends on is kept, so error positions stay close
    to where the writer wrote them.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False

    def skip_line_comment(j: int) -> int:
        while j < n and text[j] != "\n":
            j += 1
        return j

    def skip_block_comment(j: int) -> int:
        end = text.find("*/", j)
        return n if end == -1 else end + 2

    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and i + 1 < n and text[i + 1] in "/*":
            i = skip_line_comment(i + 2) if text[i + 1] == "/" else skip_block_comment(i + 2)
            continue
        if char == ",":
            # A comma is trailing — and dropped — when only whitespace and
            # comments separate it from the closing brace or bracket.
            j = i + 1
            while j < n:
                nxt = text[j]
                if nxt in " \t\r\n":
                    j += 1
                elif nxt == "/" and j + 1 < n and text[j + 1] == "/":
                    j = skip_line_comment(j + 2)
                elif nxt == "/" and j + 1 < n and text[j + 1] == "*":
                    j = skip_block_comment(j + 2)
                else:
                    break
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(char)
        i += 1
    return "".join(out)


def parse_jsonc(text: str) -> Any:
    """Parse JSONC text to Python objects. Raises ``ValueError`` on bad JSON."""
    return json.loads(strip_jsonc(text))


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in *value*.

    The same spelling the MCP config uses. A local copy of the provider's
    expander: catalog values *are* secrets, so the variant that tracks which
    variables were sourced (for child-env scrubbing) has no job here.
    """

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or (default or "")

    return _ENV_PATTERN.sub(substitute, value)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ModelInfo:
    """One selectable model: its catalog id, its wire id, and its metadata."""

    #: Canonical id, ``"<provider-id>/<key>"`` — what ``/model`` accepts.
    id: str
    provider_id: str
    provider_name: str
    #: The key it is filed under in the provider's ``models`` table.
    key: str
    #: The id sent upstream: ``upstreamName`` when set, otherwise *key*.
    model: str
    name: str = ""
    description: str = ""
    #: Context window in tokens; 0 = the provider decides (unknown).
    context: int = 0
    #: Output ceiling in tokens; 0 = unknown. Budgets the reply headroom.
    output: int = 0
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    #: Reasoning efforts, lowest first — metadata until a use needs it.
    thinking: tuple[str, ...] = ()
    kind: str = "chat"
    aliases: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return self.name or self.key

    @property
    def vision_capable(self) -> bool:
        return "image" in self.input_modalities


@dataclass(slots=True)
class ProviderInfo:
    """One upstream API: endpoint, resolved key pool, and its models."""

    id: str
    name: str
    aliases: tuple[str, ...] = ()
    base_url: str = ""
    #: Resolved literal keys, or None to inherit the ``[llm]`` env pool.
    keys: tuple[str, ...] | None = None
    #: Where the keys came from, for listings: ``literal``, ``file:<path>``
    #: or ``env`` (the inherited pool).
    keys_from: str = "env"
    key_strategy: str = "fallback"
    headers: dict[str, str] = field(default_factory=dict)
    #: Request timeout in seconds; 0 = the adapter default.
    timeout: float = 0.0
    models: dict[str, ModelInfo] = field(default_factory=dict)

    def matches(self, name: str) -> bool:
        lowered = name.strip().lower()
        return lowered in (self.id.lower(), *(alias.lower() for alias in self.aliases))


# --------------------------------------------------------------------------- #
# Building from raw JSONC
# --------------------------------------------------------------------------- #


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return cleaned or "provider"


def _string_list(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(str(item) for item in raw if isinstance(item, str))
    return ()


def _read_key_file(path: str) -> list[str]:
    """Keys from a ``file:`` reference: one per line, ``#`` comments allowed."""
    key_path = Path(path).expanduser()
    try:
        if key_path.stat().st_size > _KEY_FILE_BYTES:
            raise CatalogError(f"key file {path!r} is implausibly large")
        text = key_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CatalogError(f"key file {path!r} is unreadable: {exc}") from exc
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _resolve_keys(options: dict[str, Any]) -> tuple[tuple[str, ...] | None, str]:
    """The provider's literal keys, or ``None`` to use the ``[llm]`` env pool.

    ``apiKey`` is one key, ``apiKeys`` a pool (array, space-separated string,
    or ``file:`` path). Both may name files; every value gets ``${VAR}``
    expansion. ``"keyless"`` — the proxy's pass-the-client's-key-through —
    means nothing to a direct client, so it counts as unset.
    """
    keys: list[str] = []
    source = ""

    def add(spec: str, label: str) -> None:
        nonlocal source
        text = spec.strip()
        if not text or text == "keyless":
            return
        if text.startswith("file:"):
            path_part = text[5:].strip()
            keys.extend(_read_key_file(path_part))
            if not source:
                source = f"file:{path_part}"
            return
        expanded = _expand_env(text)
        # A pool written as one string splits on whitespace — including one
        # that arrives through ${VAR} ("KEY_POOL=sk-a sk-b").
        for part in expanded.split() if (" " in text or "${" in text) else (expanded,):
            if part:
                keys.append(part)
        if not source:
            source = label

    api_keys = options.get("apiKeys")
    if isinstance(api_keys, str) and api_keys.strip() and not api_keys.strip().startswith("file:"):
        # A space-separated pool written as one string.
        add(api_keys, "literal")
    elif api_keys:
        for item in _string_list(api_keys):
            add(item, "literal")
    if isinstance(options.get("apiKey"), str):
        add(options["apiKey"], "literal")

    if keys:
        seen: set[str] = set()
        return tuple(key for key in keys if not (key in seen or seen.add(key))), source or "literal"
    return None, "env"


def _infer_kind(raw: dict[str, Any], inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    """The model's kind: written, or read off the modalities."""
    kind = str(raw.get("kind") or "").strip().lower()
    if kind in KINDS:
        return kind
    if "audio" in outputs:
        return "tts"
    if "video" in outputs:
        return "video"
    if "image" in outputs:
        return "image"
    if "audio" in inputs:
        return "stt"
    return "chat"


def _build_provider(name: str, raw: dict[str, Any]) -> ProviderInfo:
    """One provider table entry, validated leniently: junk fields are dropped,
    junk *values* fall back to defaults, nothing here refuses to boot."""
    options = raw.get("options") if isinstance(raw.get("options"), dict) else {}
    provider_id = str(raw.get("id") or "").strip() or _slug(name)
    keys, keys_from = _resolve_keys(options)
    strategy = str(options.get("keyStrategy") or "").strip().lower()
    if strategy and strategy not in KEY_STRATEGY_MAP:
        log.warning(
            "provider %r: unknown keyStrategy %r; using 'fallback'. Known: %s",
            name,
            strategy,
            ", ".join(KEY_STRATEGY_MAP),
        )
    raw_headers = options.get("headers")
    headers = {
        str(key): _expand_env(str(value))
        for key, value in (raw_headers.items() if isinstance(raw_headers, dict) else [])
        if isinstance(value, (str, int, float))
    }
    try:
        timeout = float(options.get("timeoutSecs") or 0)
    except (TypeError, ValueError):
        timeout = 0.0

    provider = ProviderInfo(
        id=provider_id,
        name=str(raw.get("name") or name),
        aliases=tuple(alias.strip() for alias in _string_list(raw.get("aliases")) if alias.strip()),
        base_url=_expand_env(str(options.get("baseURL") or "")).strip(),
        keys=keys,
        keys_from=keys_from,
        key_strategy=KEY_STRATEGY_MAP.get(strategy, "fallback"),
        headers=headers,
        timeout=timeout,
    )

    models_raw = raw.get("models") if isinstance(raw.get("models"), dict) else {}
    for model_key, model_raw in models_raw.items():
        if not isinstance(model_raw, dict):
            continue
        provider.models[str(model_key)] = _build_model(provider, str(model_key), model_raw)
    return provider


def _build_model(provider: ProviderInfo, key: str, raw: dict[str, Any]) -> ModelInfo:
    modalities = raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
    inputs = tuple(
        str(item).strip().lower() for item in _string_list(modalities.get("input")) if str(item).strip()
    )
    outputs = tuple(
        str(item).strip().lower() for item in _string_list(modalities.get("output")) if str(item).strip()
    )
    limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    aliases = [str(raw["as"]).strip()] if raw.get("as") else []
    aliases += [alias.strip() for alias in _string_list(raw.get("aliases")) if alias.strip()]
    return ModelInfo(
        id=f"{provider.id}/{key}",
        provider_id=provider.id,
        provider_name=provider.name,
        key=key,
        model=str(raw.get("upstreamName") or "").strip() or key,
        name=str(raw.get("name") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        context=_positive_int(limit.get("context")),
        output=_positive_int(limit.get("output")),
        input_modalities=inputs,
        output_modalities=outputs,
        thinking=tuple(
            str(item).strip().lower()
            for item in _string_list(raw.get("thinking"))
            if str(item).strip()
        ),
        kind=_infer_kind(raw, inputs, outputs),
        aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
    )


def _positive_int(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


class ModelCatalog:
    """The loaded catalog: providers, resolution, and client building.

    Empty by design when no files exist — everything falls back to ``[llm]``
    and ``/model`` has nothing to switch between, exactly the pre-catalog
    behaviour. Parse problems are collected in :attr:`problems` (and logged)
    rather than raised: one broken file must not take the bot down.
    """

    def __init__(
        self,
        providers: dict[str, ProviderInfo],
        default: str = "",
        problems: list[str] | None = None,
        sources: list[Path] | None = None,
    ) -> None:
        self.providers = providers
        self.default = default
        self.problems = problems or []
        self.sources = sources or []
        # Indexes for resolution: by id, by every name the provider answers to.
        self._by_id: dict[str, ProviderInfo] = {p.id: p for p in providers.values()}
        self._by_name: dict[str, ProviderInfo] = {}
        for provider in providers.values():
            for name in (provider.id, provider.name, *provider.aliases):
                if name:
                    self._by_name.setdefault(name.lower(), provider)

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, config: Any) -> ModelCatalog:
        """Load and merge the catalog layers. Never raises: problems are
        collected and logged so ``/model`` and ``lumi doctor`` can show them."""
        layers = [
            user_config_dir() / CATALOG_FILENAME,
            config.root / PROJECT_CATALOG_FILENAME,
        ]
        merged: dict[str, Any] = {}
        sources: list[Path] = []
        problems: list[str] = []
        for path in layers:
            if not path.is_file():
                continue
            try:
                raw = parse_jsonc(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError) as exc:
                problem = f"{path}: {exc}"
                problems.append(problem)
                log.warning("model catalog ignored: %s", problem)
                continue
            if isinstance(raw, dict):
                sources.append(path)
                merged = _merge(merged, raw)
            else:
                problem = f"{path}: expected a JSON object, got {type(raw).__name__}"
                problems.append(problem)
                log.warning("model catalog ignored: %s", problem)

        providers_raw = merged.get("provider") if isinstance(merged.get("provider"), dict) else {}
        if isinstance(merged.get("providers"), dict):
            providers_raw = {**providers_raw, **merged["providers"]}
        providers: dict[str, ProviderInfo] = {}
        for name, raw in providers_raw.items():
            if not isinstance(raw, dict):
                continue
            try:
                provider = _build_provider(str(name), raw)
            except CatalogError as exc:
                # A bad key file (or any provider-level problem) drops that
                # provider only — one broken entry must not take the catalog,
                # and the bot, down.
                problem = f"provider {name!r}: {exc}"
                problems.append(problem)
                log.warning("model catalog: %s", problem)
                continue
            existing = providers.get(provider.id)
            if existing is None:
                providers[provider.id] = provider
                continue
            # Two sections claiming one id: fold the models together, the
            # first section's endpoint and keys standing, later models winning
            # per key — the same merge the layers themselves get.
            if (
                existing.base_url != provider.base_url
                or existing.keys != provider.keys
                or existing.key_strategy != provider.key_strategy
            ):
                log.warning(
                    "provider %r: defined twice with different endpoint or key options; "
                    "the first definition's options stand",
                    provider.id,
                )
            overlap = sorted(set(existing.models) & set(provider.models))
            if overlap:
                log.warning(
                    "provider %r: models %s defined twice; the later definition wins",
                    provider.id,
                    ", ".join(overlap),
                )
            existing.models.update(provider.models)
            existing.aliases = tuple(dict.fromkeys((*existing.aliases, *provider.aliases)))

        default = str(merged.get("defaultModel") or "").strip()
        catalog = cls(providers, default=default, problems=problems, sources=sources)
        if default:
            try:
                resolved_default = catalog.resolve(default)
            except CatalogError as exc:
                catalog.problems.append(f"defaultModel {default!r}: {exc}")
                catalog.default = ""
            else:
                # The default is what every un-switched chat runs on, so a
                # media kind there would send ordinary turns to an image or
                # TTS endpoint — refused here, like an unresolvable default.
                if resolved_default.kind != "chat":
                    problem = f"defaultModel {default!r} is a {resolved_default.kind} model, not a chat model"
                    catalog.problems.append(problem)
                    log.warning("model catalog: %s", problem)
                    catalog.default = ""
        return catalog

    # -- lookup ------------------------------------------------------------ #

    def _model_in(self, provider: ProviderInfo, name: str) -> ModelInfo | None:
        """*name* inside one provider: its key, an alias, or the ``as`` id."""
        if name in provider.models:
            return provider.models[name]
        lowered = name.strip().lower()
        for model in provider.models.values():
            if lowered in (model.key.lower(), *(alias.lower() for alias in model.aliases)):
                return model
        return None

    def _provider_for(self, name: str) -> ProviderInfo | None:
        return self._by_name.get(name.strip().lower())

    def default_model(self) -> ModelInfo | None:
        """The catalog's ``defaultModel``, or None when unset or unresolvable."""
        return self.resolve(self.default) if self.default else None

    def resolve(self, text: str) -> ModelInfo:
        """The model *text* names: ``provider/model``, an alias spelling, a
        bare model key unique across providers, or ``default``.

        Raises :class:`CatalogError` with the closest thing to a suggestion a
        flat listing can give — the ids the catalog actually has.
        """
        query = (text or "").strip()
        if not query:
            raise CatalogError("no model id given; try /model list")
        if query.lower() == "default":
            if not self.default:
                raise CatalogError("no defaultModel is set in the catalog")
            return self.resolve(self.default)

        if "/" in query:
            provider_name, model_name = query.split("/", 1)
            provider = self._provider_for(provider_name)
            if provider is None:
                known = ", ".join(sorted(self._by_id)) or "none"
                raise CatalogError(f"unknown provider {provider_name!r}; known: {known}")
            model = self._model_in(provider, model_name)
            if model is None:
                known = ", ".join(sorted(provider.models)) or "none"
                raise CatalogError(f"{provider.id!r} has no model {model_name!r}; models: {known}")
            return model

        matches = {
            model.id: model
            for provider in self.providers.values()
            if (model := self._model_in(provider, query)) is not None
        }
        if len(matches) == 1:
            return next(iter(matches.values()))
        if not matches:
            known = ", ".join(sorted(model.id for provider in self.providers.values() for model in provider.models.values()))
            raise CatalogError(f"no model named {query!r}; models: {known or 'none — the catalog is empty'}")
        raise CatalogError(f"{query!r} is ambiguous: {', '.join(sorted(matches))}")

    def provider_of(self, model: ModelInfo) -> ProviderInfo:
        return self._by_id[model.provider_id]

    def list_models(self, kind: str | None = "chat") -> list[ModelInfo]:
        """Every model, optionally one kind, in a stable display order."""
        models = [model for provider in self.providers.values() for model in provider.models.values()]
        if kind is not None:
            models = [model for model in models if model.kind == kind]
        return sorted(models, key=lambda model: (model.provider_id.lower(), model.key.lower()))

    def is_empty(self) -> bool:
        return not self.providers

    # -- clients ------------------------------------------------------------ #

    def build_client(
        self,
        model: ModelInfo,
        base: LLMConfig,
        vision: OpenAICompatClient | None = None,
    ) -> OpenAICompatClient:
        """The client *model* runs on: an ``[llm]``-shaped config with the
        provider's endpoint, keys and knobs on top.

        Everything the catalog does not say inherits from *base*, so a
        switched chat keeps the configured temperature, reasoning mode and
        retry ladder. Literal keys are handed to the client directly; only
        when the provider declares none does the ``[llm]`` env pool stand.
        """
        provider = self.provider_of(model)
        has_own_keys = provider.keys is not None
        config = replace(
            base,
            model=model.model,
            base_url=provider.base_url or base.base_url,
            context_window=model.context or base.context_window,
            # The model handles its own images, or a vision client is attached
            # below — the route table must not fire underneath a switch.
            vision_model="",
            api_key_env="" if has_own_keys else base.api_key_env,
            key_strategy=provider.key_strategy or base.key_strategy,
            extra_headers=dict(provider.headers),
            timeout=provider.timeout or base.timeout,
        )
        return OpenAICompatClient(config, vision=vision, keys=provider.keys)


# --------------------------------------------------------------------------- #
# Per-chat selections
# --------------------------------------------------------------------------- #


class ModelSelections:
    """The per-chat model choice, persisted in ``data/chat-models.json``.

    ``/model <id>`` writes one line here; a restart reads it back and the
    chat keeps its model. Keys are chat ids as strings — Telegram ids and the
    frontends' shared ``"cli"`` alike. The file holds model *ids*, never
    keys, so unlike the catalogs it is not a secret file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("chat model selections ignored (%s): %s", self.path, exc)
            return
        if isinstance(raw, dict):
            self._data = {
                str(chat): str(model)
                for chat, model in raw.items()
                if isinstance(model, str) and model
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self, chat_id: int | str) -> str:
        """The model id saved for *chat_id*, or ``""``."""
        self._load()
        return self._data.get(str(chat_id), "")

    def set(self, chat_id: int | str, model_id: str) -> None:
        self._load()
        self._data[str(chat_id)] = model_id
        self._save()

    def clear(self, chat_id: int | str) -> bool:
        """Forget *chat_id*'s choice. True when there was one."""
        self._load()
        if str(chat_id) not in self._data:
            return False
        del self._data[str(chat_id)]
        self._save()
        return True

    def all(self) -> dict[str, str]:
        self._load()
        return dict(self._data)


__all__ = [
    "CATALOG_FILENAME",
    "PROJECT_CATALOG_FILENAME",
    "KINDS",
    "KEY_STRATEGY_MAP",
    "CatalogError",
    "ModelCatalog",
    "ModelInfo",
    "ModelSelections",
    "ProviderInfo",
    "parse_jsonc",
    "strip_jsonc",
]
