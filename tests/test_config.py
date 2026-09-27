"""Config loading: defaults, TOML, env overrides, and coercion."""

from __future__ import annotations

import pytest

from lumi.config import (
    BotConfig,
    Config,
    ConfigError,
    LLMConfig,
    ToolsConfig,
    load_config,
    validate,
)


def test_loads_the_committed_config(project) -> None:
    config = load_config(project)
    assert config.root == project
    assert config.llm.model == "test-model"
    assert config.tools.shell.cwd == "workspace"
    assert config.owner_id == 42
    assert config.telegram_token == "test-token"


def test_paths_are_derived_from_the_project_root(project) -> None:
    config = load_config(project)
    assert config.personality_file == project / "PERSONALITY.md"
    assert config.memory_file == project / "MEMORY.md"
    assert config.data_dir == project / "data"
    assert config.history_dir == project / "data" / "history"
    assert config.shell_cwd == project / "workspace"
    assert config.log_file == project / "data" / "logs" / "lumi.log"


def test_env_overrides_toml(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__MODEL", "override-model")
    monkeypatch.setenv("LUMI__TOOLS__SHELL__TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("LUMI__BOT__REQUIRE_OWNER", "false")
    config = load_config(project)

    assert config.llm.model == "override-model"
    assert config.tools.shell.timeout_seconds == 7
    assert config.bot.require_owner is False


def test_env_override_of_a_list(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__TOOLS__WEB__PROVIDER_ORDER", "mcp, tavily ,firecrawl")
    assert load_config(project).tools.web.provider_order == ["mcp", "tavily", "firecrawl"]


def test_optional_float_can_be_nulled(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__TEMPERATURE", "none")
    assert load_config(project).llm.temperature is None


def test_optional_float_still_parses_numbers(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__TEMPERATURE", "0.15")
    assert load_config(project).llm.temperature == pytest.approx(0.15)


def test_unparseable_int_keeps_the_default(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__TOOLS__SHELL__TIMEOUT_SECONDS", "not-a-number")
    assert load_config(project).tools.shell.timeout_seconds == 60


def test_local_config_overrides_the_committed_one(project) -> None:
    (project / "config.local.toml").write_text("[llm]\nmodel = 'local-model'\n", encoding="utf-8")
    assert load_config(project).llm.model == "local-model"


def test_unknown_keys_are_ignored(project) -> None:
    (project / "config.toml").write_text(
        "[llm]\nmodel = 'x'\nfuture_option = 1\n\n[unknown_section]\nfoo = 'bar'\n",
        encoding="utf-8",
    )
    config = load_config(project)
    assert config.llm.model == "x"


def test_malformed_toml_is_reported_clearly(project) -> None:
    (project / "config.toml").write_text("this is not = = toml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(project)


def test_defaults_apply_with_no_config_file(project) -> None:
    (project / "config.toml").unlink()
    config = load_config(project)
    assert isinstance(config, Config)
    assert config.llm.provider == "openai"
    assert config.tools.enabled == ["shell", "files", "memory", "web"]


def test_dotenv_is_read_from_the_project_root(project) -> None:
    config = load_config(project)
    assert config.llm.api_key() == "test-key"
    assert config.llm.api_key_env == "OPENAI_API_KEY"


def test_real_env_beats_dotenv(project, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    assert load_config(project).llm.api_key() == "from-the-environment"


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def test_validate_passes_on_a_complete_setup(config) -> None:
    assert validate(config) == []


def test_validate_flags_a_missing_token(config, monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert any("TELEGRAM_BOT_TOKEN" in p for p in validate(config))


def test_validate_flags_a_missing_owner(project, monkeypatch) -> None:
    # The fixture's .env supplies the owner id, so remove it to simulate a
    # fresh clone where the owner has not been filled in yet.
    (project / ".env").unlink()
    monkeypatch.delenv("TELEGRAM_OWNER_ID", raising=False)
    config = load_config(project)
    config.bot.require_owner = True  # the fixture turns this off
    assert config.owner_id is None
    assert any("OWNER" in p for p in validate(config))


def test_validate_flags_a_missing_api_key(config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    assert any("OPENAI_API_KEY" in p for p in validate(config))


def test_validate_flags_a_missing_personality_file(config) -> None:
    config.personality_file.unlink()
    assert any("PERSONALITY.md" in p for p in validate(config))


def test_validate_flags_a_bad_owner_id(config, monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "not-a-number")
    assert config.owner_id is None


def test_validate_flags_unknown_tools(config) -> None:
    config.tools.enabled = ["shell", "telepathy"]
    assert any("telepathy" in p for p in validate(config))


def test_validate_flags_unknown_providers(config) -> None:
    config.tools.web.provider_order = ["tavily", "askjeeves"]
    assert any("askjeeves" in p for p in validate(config))


# --------------------------------------------------------------------------- #
# coercion internals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [("true", True), ("False", False), ("1", True), ("0", False), ("yes", True), ("no", False)],
)
def test_bool_coercion(raw: str, expected: bool) -> None:
    from lumi.config import _build

    assert _build(BotConfig, {"require_owner": raw}).require_owner is expected


def test_nested_dataclasses_are_rebuilt() -> None:
    from lumi.config import _build

    tools = _build(ToolsConfig, {"shell": {"timeout_seconds": 3}})
    assert tools.shell.timeout_seconds == 3
    assert tools.files.max_read_chars == 20000  # untouched default


def test_type_hints_resolve_under_postponed_annotations() -> None:
    """The future-annotations import must not leak strings into the coercer."""
    from lumi.config import _build

    assert _build(LLMConfig, {"temperature": None}).temperature is None
