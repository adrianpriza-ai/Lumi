"""The shared key-list parser used by every tool that accepts a pool of API keys."""

from __future__ import annotations

import pytest

from lumi.util.keys import parse_env_var, parse_key_list


def test_empty_input_returns_empty_list() -> None:
    assert parse_key_list("") == []


def test_a_single_key_stays_a_single_key() -> None:
    assert parse_key_list("ctx7sk-abc") == ["ctx7sk-abc"]


def test_a_comma_separated_list_is_a_pool() -> None:
    raw = "ctx7sk-90, ctx7sk-91 ,ctx7sk-92"
    assert parse_key_list(raw) == ["ctx7sk-90", "ctx7sk-91", "ctx7sk-92"]


@pytest.mark.parametrize("separator", [",", ";", "\n"])
def test_several_separators_are_accepted(separator: str) -> None:
    assert parse_key_list(f"ctx7sk-90{separator}ctx7sk-91") == ["ctx7sk-90", "ctx7sk-91"]


def test_a_trailing_separator_is_not_a_broken_key() -> None:
    assert parse_key_list("ctx7sk-90, ctx7sk-91,") == ["ctx7sk-90", "ctx7sk-91"]


def test_duplicate_keys_collapse() -> None:
    assert parse_key_list("ctx7sk-90,ctx7sk-90,ctx7sk-91") == ["ctx7sk-90", "ctx7sk-91"]


def test_quotes_around_a_key_are_stripped() -> None:
    assert parse_key_list('"ctx7sk-90", \'ctx7sk-91\'') == ["ctx7sk-90", "ctx7sk-91"]


def test_an_empty_variable_has_no_keys(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT7_TEST_VAR", raising=False)
    assert parse_env_var("CONTEXT7_TEST_VAR") == []


def test_env_var_honours_the_set_value(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT7_TEST_VAR", "ctx7sk-a, ctx7sk-b")
    assert parse_env_var("CONTEXT7_TEST_VAR") == ["ctx7sk-a", "ctx7sk-b"]


def test_order_is_preserved() -> None:
    """The first key in the list stays the primary one."""
    assert parse_key_list("ctx7sk-z, ctx7sk-a, ctx7sk-m") == ["ctx7sk-z", "ctx7sk-a", "ctx7sk-m"]


def test_whitespace_around_keys_is_stripped() -> None:
    assert parse_key_list("   ctx7sk-90   ,    ctx7sk-91  ") == ["ctx7sk-90", "ctx7sk-91"]