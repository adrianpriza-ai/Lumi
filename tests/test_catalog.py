"""The model catalog: JSONC parsing, layering, resolution and client building."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from lumi.config import LLMConfig, load_config
from lumi.llm.catalog import (
    CatalogError,
    ModelCatalog,
    ModelSelections,
    parse_jsonc,
    strip_jsonc,
)

ONE_PROVIDER = """\
{
  "defaultModel": "zen/mini",   // the chat default
  "provider": {
    "Zen": {
      "id": "zen",
      "aliases": ["opencode"],
      "options": {
        "baseURL": "https://zen.example/v1",
        "apiKey": "sk-zen-1",
        "timeoutSecs": 90,
        "headers": { "X-Tenant": "lab" }
      },
      "models": {
        "mini": {
          "name": "Mini",
          "limit": { "context": 100000, "output": 8000 },
          "description": "small and quick",
          "modalities": { "input": ["text"], "output": ["text"] }
        },
        "vision": {
          "name": "Vision",
          "limit": { "context": 200000, "output": 8000 },
          "modalities": { "input": ["text", "image"], "output": ["text"] }
        }
      }
    }
  }
}
"""


# --------------------------------------------------------------------------- #
# JSONC
# --------------------------------------------------------------------------- #


def test_strip_jsonc_removes_line_and_block_comments() -> None:
    text = '{\n  // line\n  /* block\n   spanning */\n  "a": 1 // trailing\n}\n'
    assert json.loads(strip_jsonc(text)) == {"a": 1}


def test_strip_jsonc_tolerates_trailing_commas() -> None:
    text = '{"a": [1, 2,], "b": {"c": 1,},}'
    assert json.loads(strip_jsonc(text)) == {"a": [1, 2], "b": {"c": 1}}


def test_strip_jsonc_leaves_strings_alone() -> None:
    # // inside a URL, a comma before a brace inside a string, an escaped quote.
    text = '{"url": "https://x.example/v1//y", "snip": "a,} b", "q": "say \\"hi\\"" }'
    assert json.loads(strip_jsonc(text)) == json.loads(text)


def test_parse_jsonc_reads_the_owner_catalog_shape() -> None:
    data = parse_jsonc(ONE_PROVIDER)
    assert data["provider"]["Zen"]["id"] == "zen"
    assert data["defaultModel"] == "zen/mini"


def test_parse_jsonc_reports_bad_json() -> None:
    with pytest.raises(ValueError):
        parse_jsonc('{"a": }')


# --------------------------------------------------------------------------- #
# Loading and merging
# --------------------------------------------------------------------------- #


@pytest.fixture
def catalog_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root with the user catalog pointed at a scratch directory."""
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("LUMI_USER_CONFIG", str(tmp_path / "usercfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return project


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_catalog(project: Path, extra_user: str = "") -> ModelCatalog:
    config = load_config(project)
    if extra_user:
        write(Path(os.environ["LUMI_USER_CONFIG"]) / "models.json", extra_user)
    return ModelCatalog.load(config)


def test_load_without_files_is_empty_and_quiet(catalog_env: Path) -> None:
    catalog = make_catalog(catalog_env)
    assert catalog.is_empty()
    assert catalog.problems == []
    assert catalog.sources == []
    assert catalog.list_models() == []


def test_load_reads_the_project_catalog(catalog_env: Path) -> None:
    write(catalog_env / ".models.json", ONE_PROVIDER)
    catalog = make_catalog(catalog_env)
    assert not catalog.is_empty()
    assert catalog.problems == []
    assert len(catalog.list_models(kind="chat")) == 2
    assert catalog.default == "zen/mini"
    assert catalog.default_model() is not None


def test_project_layer_overrides_the_user_layer_per_provider(catalog_env: Path) -> None:
    write(catalog_env / ".models.json", ONE_PROVIDER)
    user = make_catalog(
        catalog_env,
        '{ "provider": { "Zen": { "options": { "baseURL": "https://user.example/v1" }, '
        '"models": { "extra": { "name": "Extra" } } } } }',
    )
    zen = user.provider_of(user.resolve("zen/mini"))
    # The project's endpoint wins; the user layer's extra model survives.
    assert zen.base_url == "https://zen.example/v1"
    assert user.resolve("zen/extra").id == "zen/extra"
    # "extra" carries no modalities, so it classifies as chat and joins the list.
    assert [m.id for m in user.list_models(kind="chat")] == [
        "zen/extra",
        "zen/mini",
        "zen/vision",
    ]


def test_a_broken_catalog_file_becomes_a_problem_not_a_crash(catalog_env: Path) -> None:
    write(catalog_env / ".models.json", "{ not json ]")
    catalog = make_catalog(catalog_env)
    assert catalog.is_empty()
    assert len(catalog.problems) == 1
    assert ".models.json" in catalog.problems[0]


def test_an_unresolvable_defaultModel_is_reported_and_cleared(catalog_env: Path) -> None:
    write(catalog_env / ".models.json", ONE_PROVIDER.replace('"zen/mini"', '"zen/nope"'))
    catalog = make_catalog(catalog_env)
    assert catalog.default == ""
    assert any("defaultModel" in problem for problem in catalog.problems)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def catalog(catalog_env: Path) -> ModelCatalog:
    write(catalog_env / ".models.json", ONE_PROVIDER)
    return make_catalog(catalog_env)


def test_resolve_by_full_id_and_display(catalog: ModelCatalog) -> None:
    model = catalog.resolve("zen/mini")
    assert model.id == "zen/mini"
    assert model.model == "mini"  # the wire id: the key, no upstreamName
    assert model.name == "Mini"
    assert model.context == 100_000
    assert model.output == 8_000
    assert model.description == "small and quick"
    assert model.kind == "chat"
    assert not model.vision_capable


def test_resolve_by_provider_alias_and_model_alias(catalog: ModelCatalog) -> None:
    assert catalog.resolve("opencode/mini").id == "zen/mini"
    catalog.providers["zen"].models["mini"].aliases = ("small",)
    assert catalog.resolve("zen/small").id == "zen/mini"


def test_resolve_bare_model_key_when_unique(catalog: ModelCatalog) -> None:
    assert catalog.resolve("vision").id == "zen/vision"
    assert catalog.resolve("vision").vision_capable


def test_resolve_default_keyword(catalog: ModelCatalog) -> None:
    assert catalog.resolve("default").id == "zen/mini"


def test_resolve_errors_name_what_exists(catalog: ModelCatalog) -> None:
    with pytest.raises(CatalogError, match="unknown provider"):
        catalog.resolve("nope/mini")
    with pytest.raises(CatalogError, match="has no model"):
        catalog.resolve("zen/nope")
    with pytest.raises(CatalogError, match="no model named"):
        catalog.resolve("zzz")
    with pytest.raises(CatalogError, match="no model id"):
        catalog.resolve("  ")


def test_a_bare_key_across_providers_must_be_unique(catalog_env: Path) -> None:
    write(catalog_env / ".models.json", ONE_PROVIDER)
    other = """\
    {
      "provider": { "Other": {
        "id": "other", "options": { "baseURL": "https://other.example/v1" },
        "models": { "mini": { "name": "Mini elsewhere" } }
      } }
    }
    """
    write(Path(os.environ["LUMI_USER_CONFIG"]) / "models.json", other)
    with pytest.raises(CatalogError, match="ambiguous"):
        make_catalog(catalog_env).resolve("mini")


def test_upstream_name_and_as_alias(catalog_env: Path) -> None:
    write(
        catalog_env / ".models.json",
        ONE_PROVIDER.replace(
            '"name": "Mini"', '"name": "Mini", "upstreamName": "mini-preview", "as": "m"'
        ),
    )
    catalog = make_catalog(catalog_env)
    model = catalog.resolve("zen/mini")
    assert model.model == "mini-preview"
    assert catalog.resolve("zen/m").id == "zen/mini"


# --------------------------------------------------------------------------- #
# Kinds
# --------------------------------------------------------------------------- #


def test_non_chat_kinds_are_listed_but_marked(catalog_env: Path) -> None:
    write(
        catalog_env / ".models.json",
        """\
        {
          "provider": { "Media": {
            "id": "media", "options": { "baseURL": "https://media.example/v1" },
            "models": {
              "flux": { "name": "Flux", "kind": "image", "modalities": { "input": ["text"], "output": ["image"] } },
              "veo": { "name": "Veo", "modalities": { "input": ["text"], "output": ["video"] } },
              "voice": { "name": "Voice", "modalities": { "input": ["text"], "output": ["audio"] } },
              "ear": { "name": "Ear", "modalities": { "input": ["audio"], "output": ["text"] } }
            }
          } }
        }
        """,
    )
    catalog = make_catalog(catalog_env)
    # Only the explicit kinds landed; the modalities classify the rest.
    assert catalog.resolve("media/flux").kind == "image"
    assert catalog.resolve("media/veo").kind == "video"
    assert catalog.resolve("media/voice").kind == "tts"
    assert catalog.resolve("media/ear").kind == "stt"
    assert catalog.list_models(kind="chat") == []


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def keys_of(catalog_env: Path, options: dict[str, Any]) -> tuple[str, ...] | None:
    write(
        catalog_env / ".models.json",
        json.dumps(
            {
                "provider": {
                    "P": {
                        "id": "p",
                        "options": options,
                        "models": {"m": {"name": "M"}},
                    }
                }
            }
        ),
    )
    provider = make_catalog(catalog_env).providers["p"]
    return provider.keys


def test_keys_literal_array_and_space_separated(catalog_env: Path) -> None:
    assert keys_of(catalog_env, {"apiKey": "sk-a"}) == ("sk-a",)
    assert keys_of(catalog_env, {"apiKeys": ["sk-a", "sk-b"]}) == ("sk-a", "sk-b")
    assert keys_of(catalog_env, {"apiKeys": "sk-a sk-b"}) == ("sk-a", "sk-b")


def test_keys_from_a_file_one_per_line(catalog_env: Path, tmp_path: Path) -> None:
    key_file = write(tmp_path / "keys" / "zen.txt", "sk-line1\n\n# comment\nsk-line2\n")
    assert keys_of(catalog_env, {"apiKey": f"file:{key_file}"}) == ("sk-line1", "sk-line2")


def test_keys_expand_env_vars(catalog_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEN_KEY", "sk-from-env")
    assert keys_of(catalog_env, {"apiKey": "${ZEN_KEY}"}) == ("sk-from-env",)
    monkeypatch.delenv("ZEN_KEY")
    assert keys_of(catalog_env, {"apiKey": "${ZEN_KEY:-sk-fallback}"}) == ("sk-fallback",)


def test_keyless_and_empty_fall_back_to_the_env_pool(catalog_env: Path) -> None:
    # "keyless" is the proxy's pass-through; a direct client counts it unset.
    assert keys_of(catalog_env, {"apiKey": "keyless"}) is None
    assert keys_of(catalog_env, {}) is None


def test_an_unreadable_key_file_is_a_problem(catalog_env: Path) -> None:
    catalog = make_catalog(
        catalog_env,
        '{ "provider": { "P": { "id": "p", "options": { "apiKey": "file:/nope/keys.txt" }, '
        '"models": { "m": { "name": "M" } } } } }',
    )
    assert any("unreadable" in problem for problem in catalog.problems)
    assert "p" not in catalog.providers


def test_key_strategy_maps_onto_the_pool(catalog_env: Path) -> None:
    write(
        catalog_env / ".models.json",
        json.dumps(
            {
                "provider": {
                    "P": {
                        "id": "p",
                        "options": {"keyStrategy": "rotate", "apiKeys": ["a", "b"]},
                        "models": {"m": {"name": "M"}},
                    }
                }
            }
        ),
    )
    assert make_catalog(catalog_env).providers["p"].key_strategy == "round_robin"


# --------------------------------------------------------------------------- #
# Client building
# --------------------------------------------------------------------------- #


def test_build_client_synthesises_the_config(catalog: ModelCatalog) -> None:
    base = LLMConfig(model="test-model", context_window=100_000, api_key_env="OPENAI_API_KEY")
    client = catalog.build_client(catalog.resolve("zen/mini"), base)
    assert client.config.model_of() == "mini"
    assert client.config.base_url_of() == "https://zen.example/v1"
    assert client.config.context_window == 100_000
    assert client.config.timeout == 90.0
    assert client.config.extra_headers == {"X-Tenant": "lab"}
    # Literal keys flow to the pool, and the env lookup stands down.
    assert [key for key in client._pool.keys] == ["sk-zen-1"]
    assert client.config.api_key_env == ""


def test_build_client_inherits_what_the_catalog_does_not_say(catalog: ModelCatalog) -> None:
    base = LLMConfig(
        model="test-model",
        temperature=0.2,
        max_tokens=512,
        reasoning=True,
        context_window=100_000,
    )
    client = catalog.build_client(catalog.resolve("zen/mini"), base)
    assert client.config.temperature == 0.2
    assert client.config.max_tokens == 512
    assert client.config.reasoning is True


def test_build_client_adopts_the_model_context_limit(catalog: ModelCatalog) -> None:
    base = LLMConfig(model="test-model", context_window=200_000)
    model = catalog.resolve("zen/mini")  # context 100k < configured 200k
    client = catalog.build_client(model, base)
    assert client.config.context_window == 100_000


def test_build_client_without_keys_keeps_the_env_pool(catalog_env: Path) -> None:
    write(
        catalog_env / ".models.json",
        '{ "provider": { "P": { "id": "p", "options": { "baseURL": "https://p.example/v1" }, '
        '"models": { "m": { "name": "M" } } } } }',
    )
    catalog = make_catalog(catalog_env)
    base = LLMConfig(model="test-model", api_key_env="OPENAI_API_KEY")
    client = catalog.build_client(catalog.resolve("p/m"), base)
    assert client.config.api_key_env == "OPENAI_API_KEY"


# --------------------------------------------------------------------------- #
# Selections
# --------------------------------------------------------------------------- #


def test_selections_round_trip_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "data" / "chat-models.json"
    store = ModelSelections(path)
    assert store.get("cli") == ""
    store.set("cli", "zen/mini")
    store.set(12345, "zen/vision")
    assert path.is_file()

    reopened = ModelSelections(path)
    assert reopened.get("cli") == "zen/mini"
    assert reopened.get(12345) == "zen/vision"
    assert reopened.all() == {"cli": "zen/mini", "12345": "zen/vision"}

    assert reopened.clear("cli") is True
    assert reopened.clear("cli") is False
    assert reopened.get("cli") == ""
    assert json.loads(path.read_text()) == {"12345": "zen/vision"}


def test_selections_survive_a_broken_file(tmp_path: Path) -> None:
    path = tmp_path / "chat-models.json"
    path.write_text("{ broken", encoding="utf-8")
    store = ModelSelections(path)
    assert store.get("cli") == ""
    store.set("cli", "zen/mini")
    assert json.loads(path.read_text()) == {"cli": "zen/mini"}


# --------------------------------------------------------------------------- #
# review regressions: a broken catalog must never take the bot down
# --------------------------------------------------------------------------- #


def test_a_malformed_headers_value_does_not_crash_the_load(catalog_env: Path) -> None:
    """``headers`` of the wrong shape is dropped, not an AttributeError at boot."""
    write(
        catalog_env / ".models.json",
        json.dumps(
            {
                "provider": {
                    "P": {
                        "id": "p",
                        "options": {"baseURL": "https://p.example/v1", "headers": ["wrong"]},
                        "models": {"m": {"name": "M"}},
                    }
                }
            }
        ),
    )
    catalog = make_catalog(catalog_env)
    assert catalog.problems == []
    assert catalog.providers["p"].headers == {}


def test_a_non_chat_defaultModel_is_refused(catalog_env: Path) -> None:
    """The default is what every un-switched chat runs on; a media kind there
    would send ordinary turns at an image endpoint."""
    write(
        catalog_env / ".models.json",
        json.dumps(
            {
                "defaultModel": "media/flux",
                "provider": {
                    "Media": {
                        "id": "media",
                        "options": {"baseURL": "https://media.example/v1"},
                        "models": {
                            "flux": {"name": "Flux", "kind": "image"},
                        },
                    }
                },
            }
        ),
    )
    catalog = make_catalog(catalog_env)
    assert catalog.default == ""
    assert any("not a chat model" in problem for problem in catalog.problems)


# --------------------------------------------------------------------------- #
# Declared keys that resolve to nothing
# --------------------------------------------------------------------------- #


def _provider_with(options: dict[str, Any]) -> str:
    return json.dumps(
        {
            "provider": {
                "P": {
                    "id": "p",
                    "options": {"baseURL": "https://p.example/v1", **options},
                    "models": {"m": {"name": "M"}},
                }
            }
        }
    )


def test_a_declared_but_unset_key_var_refuses_the_env_pool(catalog_env: Path) -> None:
    """``apiKey: "${UNSET_VAR}"`` resolving to nothing must drop the provider,
    not silently inherit the ``[llm]`` pool — that fallback sends the
    configured provider's credentials to this provider's endpoint."""
    write(catalog_env / ".models.json", _provider_with({"apiKey": "${P_KEY}"}))

    catalog = make_catalog(catalog_env)

    assert "p" not in catalog.providers
    assert any("refusing to fall back" in problem for problem in catalog.problems)


def test_an_empty_key_file_is_declared_but_resolves_to_nothing(
    catalog_env: Path, tmp_path: Path
) -> None:
    """A ``file:`` reference that yields no keys — comments only, or an empty
    file — is the same declared-but-empty trap as an unset ${VAR}."""
    key_file = write(tmp_path / "keys" / "empty.txt", "# nothing but comments\n")
    write(catalog_env / ".models.json", _provider_with({"apiKey": f"file:{key_file}"}))

    catalog = make_catalog(catalog_env)

    assert "p" not in catalog.providers
    assert any("refusing to fall back" in problem for problem in catalog.problems)


def test_keys_split_on_the_env_pool_spelling(catalog_env: Path) -> None:
    # The comma/semicolon spelling every other key pool in the codebase takes
    # (OPENAI_API_KEY and friends), not just whitespace.
    assert keys_of(catalog_env, {"apiKeys": "sk-a,sk-b"}) == ("sk-a", "sk-b")
    assert keys_of(catalog_env, {"apiKeys": "sk-a; sk-b"}) == ("sk-a", "sk-b")


def test_a_provider_without_a_key_strategy_inherits_the_pool(
    catalog_env: Path,
) -> None:
    """``keyStrategy`` unset means inherit, not a silent ``fallback``: a
    ``[llm]`` pool configured to rotate keeps rotating after a /model switch
    instead of pinning to the first key."""
    write(
        catalog_env / ".models.json",
        _provider_with({"apiKey": "sk-a", "apiKeys": ["sk-a", "sk-b"]}),
    )
    catalog = make_catalog(catalog_env)

    base = LLMConfig(
        model="test-model",
        api_key_env="OPENAI_API_KEY",
        key_strategy="round_robin",
    )
    client = catalog.build_client(catalog.resolve("p/m"), base)
    assert client.config.key_strategy == "round_robin"

    # And with nothing configured, the pool's default stands as before.
    client = catalog.build_client(catalog.resolve("p/m"), LLMConfig(model="test-model"))
    assert client.config.key_strategy == "fallback"


def test_selections_save_leaves_no_tmp_behind(catalog_env: Path) -> None:
    """The persisted-selections tmp file is pid-suffixed and removed: the bot
    and the TUI are two processes writing one file, and a fixed tmp name lets
    interleaved writes publish garbled JSON."""
    selections = ModelSelections(catalog_env / "data" / "chat-models.json")
    selections.set("1", "p/m")
    selections.clear("1")

    assert selections.get("1") == ""
    assert list((catalog_env / "data").glob("*tmp*")) == []
