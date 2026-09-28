"""The provider-agnostic half of reasoning support.

These are the functions that decide what counts as a thinking trace, so they are
tested directly rather than through a model call: the field names and the tag
shapes are the part most likely to drift, and the part with no useful default
if it is wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lumi.llm.reasoning import (
    completion_ceiling,
    rejects_param,
    split_think,
    trace_of,
)

# --------------------------------------------------------------------------- #
# trace_of
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
def test_every_known_field_name_is_probed(field: str) -> None:
    assert trace_of(SimpleNamespace(**{field: "step one, step two"})) == "step one, step two"


def test_a_dict_message_is_accepted_too() -> None:
    assert trace_of({"reasoning_content": "from a plain dict"}) == "from a plain dict"


def test_no_trace_field_means_no_trace() -> None:
    assert trace_of(SimpleNamespace(content="just an answer")) == ""


def test_an_empty_trace_is_not_a_trace() -> None:
    assert trace_of(SimpleNamespace(reasoning_content="   ", content="hi")) == ""


def test_a_structured_trace_is_ignored_rather_than_stringified() -> None:
    """OpenRouter also offers a `reasoning_details` list; a repr of it is worse
    than showing nothing, and it lives under a name we do not probe anyway."""
    assert trace_of(SimpleNamespace(reasoning=[{"type": "reasoning.text"}])) == ""


# --------------------------------------------------------------------------- #
# split_think
# --------------------------------------------------------------------------- #


def test_plain_text_is_untouched() -> None:
    assert split_think("just an answer") == ("just an answer", "")
    assert split_think("") == ("", "")


def test_a_closed_think_block_is_moved_out_of_the_answer() -> None:
    answer, trace = split_think("<think>weighing the options</think>Here is the answer.")
    assert answer == "Here is the answer."
    assert trace == "weighing the options"


def test_several_blocks_are_joined() -> None:
    answer, trace = split_think("<think>first</think>A<think>second</think>B")
    assert answer == "AB"
    assert trace == "first\n\nsecond"


def test_an_unclosed_think_block_is_still_captured() -> None:
    """What a model emits when it runs out of tokens mid-thought: the whole tail
    is thinking, and the answer never arrived."""
    answer, trace = split_think("the setup. <think>still weighing this out")
    assert answer == "the setup."
    assert trace == "still weighing this out"


def test_the_tag_name_and_attributes_do_not_matter() -> None:
    for text in ("<thinking>x</thinking>", "<reasoning>x</reasoning>", '<think foo="1">x</think>'):
        answer, trace = split_think(f"{text}answer")
        assert (answer, trace) == ("answer", "x"), text


def test_tag_matching_is_case_insensitive() -> None:
    assert split_think("<THINK>x</THINK>answer") == ("answer", "x")


def test_a_lone_angle_bracket_is_not_a_think_tag() -> None:
    assert split_think("if a < b then b > a") == ("if a < b then b > a", "")


def test_a_literal_tag_mention_in_prose_is_treated_as_a_think_block() -> None:
    """Documented trade-off: an unclosed ``<think>`` anywhere is read as a trace.

    Models open their thinking at the very start of ``content``, so requiring
    the tag to be first would be stricter — but it would also mean one model
    emitting a short preamble before thinking leaks its entire trace into the
    chat as visible prose. Leaking is worse than swallowing a stray mention.
    """
    answer, trace = split_think("wrap the trace in <think> tags, like this: <think>")
    assert trace == "tags, like this: <think>"
    assert "<think>" not in answer


# --------------------------------------------------------------------------- #
# completion_ceiling
# --------------------------------------------------------------------------- #


def test_the_ceiling_adds_room_for_thinking() -> None:
    assert completion_ceiling(2000, 2000) == 4000


def test_zero_headroom_is_the_same_as_no_headroom() -> None:
    assert completion_ceiling(2000, 0) == 2000


def test_negative_values_are_floored_not_propagated() -> None:
    """A negative ceiling is rejected by every provider, so a typo in config
    should not turn into an unexplainable 400."""
    assert completion_ceiling(-5, 100) == 100
    assert completion_ceiling(100, -5) == 100


# --------------------------------------------------------------------------- #
# rejects_param
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message",
    [
        "Unsupported parameter: 'max_completion_tokens' is not supported with this model.",
        "Unrecognized request argument supplied: reasoning_effort",
        "unknown parameter: reasoning_effort",
        "reasoning_effort is not supported with this model",
    ],
)
def test_unknown_parameter_phrasings_are_recognised(message: str) -> None:
    assert rejects_param(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        # A bad *value* is a real problem. Retrying without the parameter would
        # bury the message the owner needs to read.
        "temperature is only supported when set to 1",
        "max_tokens must be an integer",
        "messages is required",
    ],
)
def test_value_problems_are_not_mistaken_for_unknown_parameters(message: str) -> None:
    assert not rejects_param(RuntimeError(message))
