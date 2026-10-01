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
    assert config.tools.enabled == ["shell", "files", "memory", "web", "context7"]


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


def test_validate_accepts_context7_in_tools_enabled(config) -> None:
    """The context7 tool joins the known list without a validator complaint."""
    config.tools.enabled = ["shell", "files", "memory", "web", "context7"]
    assert not [p for p in validate(config) if "context7" in p and "unknown" in p]


def test_validate_flags_an_unknown_context7_strategy(config, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_API_KEY", "ctx7sk-91")
    config.tools.context7.key_strategy = "telepathy"
    assert any("key_strategy" in p and "context7" in p for p in validate(config))


def test_validate_flags_an_unknown_tavily_strategy(config) -> None:
    config.tools.web.tavily.key_strategy = "telepathy"
    assert any("tavily" in p and "key_strategy" in p for p in validate(config))


def test_validate_flags_an_unknown_firecrawl_strategy(config) -> None:
    config.tools.web.firecrawl.key_strategy = "telepathy"
    assert any("firecrawl" in p and "key_strategy" in p for p in validate(config))


@pytest.mark.parametrize("strategy", ["fallback", "round_robin", "random"])
def test_validate_accepts_every_documented_strategy(config, strategy: str) -> None:
    config.tools.web.tavily.key_strategy = strategy
    config.tools.web.firecrawl.key_strategy = strategy
    assert not [p for p in validate(config) if "key_strategy" in p]


@pytest.mark.parametrize("mode", ["mention", "always", "off"])
def test_validate_accepts_every_documented_group_reply_mode(config, mode: str) -> None:
    config.bot.group_reply_mode = mode
    assert not [p for p in validate(config) if "group_reply_mode" in p]


def test_validate_flags_an_unknown_group_reply_mode(config) -> None:
    config.bot.group_reply_mode = "loud"  # not a real value
    assert any("group_reply_mode" in p for p in validate(config))


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
    assert tools.files.max_read_chars == 200_000  # untouched default


def test_type_hints_resolve_under_postponed_annotations() -> None:
    """The future-annotations import must not leak strings into the coercer."""
    from lumi.config import _build

    assert _build(LLMConfig, {"temperature": None}).temperature is None


# --------------------------------------------------------------------------- #
# endpoint resolution
# --------------------------------------------------------------------------- #


def test_base_url_falls_back_to_the_openai_default(config, monkeypatch) -> None:
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert config.llm.base_url_of() == "https://api.openai.com/v1"
    assert config.llm.where_from() == "default"


def test_openai_base_url_env_is_honoured(config, monkeypatch) -> None:
    """The bug this guards: OPENAI_BASE_URL documented but never read."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert config.llm.base_url_of() == "http://localhost:11434/v1"
    assert config.llm.where_from() == "OPENAI_BASE_URL"


def test_openai_api_base_alias_is_honoured(config, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
    assert config.llm.base_url_of() == "https://api.groq.com/openai/v1"


def test_empty_env_var_is_treated_as_unset(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    assert config.llm.base_url_of() == "https://api.openai.com/v1"


def test_explicit_config_beats_the_env(config, monkeypatch) -> None:
    """A project-level setting must win over a machine-wide one."""
    config.llm.base_url = "http://localhost:1234/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert config.llm.base_url_of() == "http://localhost:1234/v1"
    assert config.llm.where_from() == "config"


def test_lum_env_override_beats_openai_base_url(project, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LUMI__LLM__BASE_URL", "http://localhost:9999/v1")
    assert load_config(project).llm.base_url_of() == "http://localhost:9999/v1"


def test_config_toml_base_url_wins_over_env(project, monkeypatch) -> None:
    (project / "config.toml").write_text(
        "[llm]\nmodel = 'm'\nbase_url = 'http://127.0.0.1:8000/v1'\n", encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert load_config(project).llm.base_url_of() == "http://127.0.0.1:8000/v1"


def test_model_falls_back_to_openai_model(config, monkeypatch) -> None:
    """config.toml ships the default model, so the env must still win over it."""
    config.llm.model = "gpt-4.1-mini"  # the shipped default
    monkeypatch.setenv("OPENAI_MODEL", "nemotron-3-nano-reasoning")
    assert config.llm.model_of() == "nemotron-3-nano-reasoning"


def test_explicit_model_beats_openai_model(config, monkeypatch) -> None:
    config.llm.model = "qwen3:8b"
    monkeypatch.setenv("OPENAI_MODEL", "nemotron-3-nano-reasoning")
    assert config.llm.model_of() == "qwen3:8b"


# --------------------------------------------------------------------------- #
# vision model
# --------------------------------------------------------------------------- #


def test_vision_model_is_unset_by_default(config) -> None:
    assert config.llm.vision_model_of() == ""


def test_vision_model_resolves_from_config(config) -> None:
    config.llm.vision_model = "gpt-4o"
    assert config.llm.vision_model_of() == "gpt-4o"


def test_vision_model_falls_back_to_its_env_var(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_VISION_MODEL", "qwen2-vl")
    assert config.llm.vision_model_of() == "qwen2-vl"


def test_explicit_vision_model_beats_the_env(config, monkeypatch) -> None:
    config.llm.vision_model = "gpt-4o"
    monkeypatch.setenv("OPENAI_VISION_MODEL", "qwen2-vl")
    assert config.llm.vision_model_of() == "gpt-4o"


def test_blank_vision_model_counts_as_unset(config, monkeypatch) -> None:
    config.llm.vision_model = "   "
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)
    assert config.llm.vision_model_of() == ""


def test_vision_model_survives_a_toml_load(project) -> None:
    (project / "config.toml").write_text(
        "[llm]\nmodel = 'm'\nvision_model = 'gpt-4o'\n", encoding="utf-8"
    )
    assert load_config(project).llm.vision_model == "gpt-4o"


def test_api_key_env_is_configurable(config, monkeypatch) -> None:
    """A provider with its own key name should need no code change."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-123")
    config.llm.api_key_env = "GROQ_API_KEY"
    assert config.llm.api_key() == "gsk-123"

    config.llm.api_key_env = "DEFINITELY_NOT_SET"
    assert config.llm.api_key() is None
    assert any("DEFINITELY_NOT_SET" in p for p in validate(config))


# --------------------------------------------------------------------------- #
# key pools
# --------------------------------------------------------------------------- #


def test_a_single_key_is_still_a_single_key(config) -> None:
    assert config.llm.api_keys() == ["test-key"]


def test_a_comma_separated_list_is_a_pool(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-...90, sk-...91 ,sk-...92")
    assert config.llm.api_keys() == ["sk-...90", "sk-...91", "sk-...92"]
    # api_key() stays the primary key, so every existing caller keeps working.
    assert config.llm.api_key() == "sk-...90"


@pytest.mark.parametrize("separator", [",", ";", "\n"])
def test_several_separators_are_accepted(config, monkeypatch, separator: str) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", f"sk-90{separator}sk-91")
    assert config.llm.api_keys() == ["sk-90", "sk-91"]


def test_a_trailing_separator_is_not_a_broken_key(config, monkeypatch) -> None:
    """Stray punctuation in a hand-edited .env must not become a key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-90, sk-91,")
    assert config.llm.api_keys() == ["sk-90", "sk-91"]


def test_duplicate_keys_collapse(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-90,sk-90,sk-91")
    assert config.llm.api_keys() == ["sk-90", "sk-91"]


def test_quotes_around_a_key_are_stripped(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", '"sk-90", \'sk-91\'')
    assert config.llm.api_keys() == ["sk-90", "sk-91"]


def test_an_empty_variable_has_no_keys(config, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert config.llm.api_keys() == []
    assert config.llm.api_key() is None


def test_the_key_strategy_defaults_to_fallback(config) -> None:
    assert config.llm.strategy_of() == "fallback"


@pytest.mark.parametrize("strategy", ["fallback", "round_robin", "random"])
def test_every_documented_strategy_is_accepted(config, strategy: str) -> None:
    config.llm.key_strategy = strategy
    assert config.llm.strategy_of() == strategy
    assert not [p for p in validate(config) if "key_strategy" in p]


def test_an_unknown_strategy_is_reported(config) -> None:
    config.llm.key_strategy = "roundrobin"
    assert config.llm.strategy_of() == "fallback"
    assert any("key_strategy" in p for p in validate(config))


def test_the_strategy_can_come_from_the_environment(config, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__KEY_STRATEGY", "round_robin")
    assert load_config(config.root).llm.strategy_of() == "round_robin"


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


def test_reasoning_is_off_by_default(config) -> None:
    """Off by default, so a model that does not reason behaves exactly as it did
    before this existed."""
    assert config.llm.reasoning is False
    assert config.llm.reasoning_effort == ""
    assert not [p for p in validate(config) if "reasoning" in p]


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_every_documented_effort_is_accepted(config, effort: str) -> None:
    config.llm.reasoning_effort = effort
    assert config.llm.effort_of() == effort
    assert not [p for p in validate(config) if "reasoning_effort" in p]


def test_effort_is_normalised(config) -> None:
    config.llm.reasoning_effort = "  HIGH "
    assert config.llm.effort_of() == "high"


def test_an_empty_effort_sends_nothing_in_plain_mode(config) -> None:
    """reasoning off, effort unset: the parameter stays out of the request."""
    assert config.llm.reasoning is False
    assert config.llm.effort_of() == ""


def test_an_empty_effort_sends_the_default_in_reasoning_mode(config) -> None:
    """On endpoints like Ollama's, reasoning_effort's presence is the thinking
    on/off switch — omitting it would leave a default-off model silent."""
    config.llm.reasoning = True
    assert config.llm.effort_of() == "medium"


def test_effort_none_still_means_none(config) -> None:
    """``"none"`` actively asks a model that thinks to skip thinking; it must
    not be collapsed into the default."""
    config.llm.reasoning = True
    config.llm.reasoning_effort = "none"
    assert config.llm.effort_of() == "none"


def test_an_unknown_effort_is_reported_and_omitted(config) -> None:
    """Reported so a typo is visible, but omitted rather than fatal — the
    provider's default is a perfectly good answer."""
    config.llm.reasoning_effort = "turbo"
    assert config.llm.effort_of() == ""
    assert any("reasoning_effort" in p for p in validate(config))


def test_a_negative_thinking_budget_is_reported(config) -> None:
    config.llm.reasoning = True
    config.llm.reasoning_tokens = -1
    assert any("reasoning_tokens" in p for p in validate(config))


def test_the_reasoning_settings_can_come_from_the_environment(config, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__LLM__REASONING", "true")
    monkeypatch.setenv("LUMI__LLM__REASONING_EFFORT", "high")
    monkeypatch.setenv("LUMI__LLM__REASONING_TOKENS", "8000")
    llm = load_config(config.root).llm
    assert llm.reasoning is True
    assert llm.effort_of() == "high"
    assert llm.reasoning_tokens == 8000


def test_the_trace_is_shown_by_default(config) -> None:
    assert config.llm.show_reasoning is True


# --------------------------------------------------------------------------- #
# bot network settings
# --------------------------------------------------------------------------- #


def test_network_defaults(config) -> None:
    """Direct connection, generous connect timeout, unlimited startup retries."""
    assert config.bot.proxy_url == ""
    assert config.bot.connect_timeout == 15.0
    assert config.bot.bootstrap_retries == -1


def test_network_settings_can_come_from_toml(project) -> None:
    (project / "config.toml").write_text(
        "[bot]\n"
        "require_owner = false\n"
        "proxy_url = 'socks5://127.0.0.1:9050'\n"
        "connect_timeout = 30.5\n"
        "bootstrap_retries = 7\n",
        encoding="utf-8",
    )
    bot = load_config(project).bot
    assert bot.proxy_url == "socks5://127.0.0.1:9050"
    assert bot.connect_timeout == 30.5
    assert bot.bootstrap_retries == 7


def test_network_settings_can_come_from_the_environment(project, monkeypatch) -> None:
    monkeypatch.setenv("LUMI__BOT__PROXY_URL", "http://127.0.0.1:7890")
    monkeypatch.setenv("LUMI__BOT__CONNECT_TIMEOUT", "20")
    monkeypatch.setenv("LUMI__BOT__BOOTSTRAP_RETRIES", "5")
    bot = load_config(project).bot
    assert bot.proxy_url == "http://127.0.0.1:7890"
    assert bot.connect_timeout == 20.0
    assert bot.bootstrap_retries == 5


def test_a_non_positive_connect_timeout_is_reported(config) -> None:
    config.bot.connect_timeout = 0
    assert any("connect_timeout" in p for p in validate(config))


def test_below_minus_one_bootstrap_retries_are_reported(config) -> None:
    """-1 is the documented 'retry forever' value; anything lower is a typo."""
    config.bot.bootstrap_retries = -2
    assert any("bootstrap_retries" in p for p in validate(config))


def test_minus_one_bootstrap_retries_validate(config) -> None:
    """-1 means retry startup forever and is accepted."""
    config.bot.bootstrap_retries = -1
    assert not any("bootstrap_retries" in p for p in validate(config))
