"""Message splitting and truncation."""

from __future__ import annotations

from lumi.util.text import (
    TELEGRAM_LIMIT,
    escape_markdown,
    format_error,
    split_message,
    truncate,
    truncate_middle,
)


def test_short_text_is_one_chunk() -> None:
    assert split_message("hello") == ["hello"]


def test_empty_text_is_no_chunks() -> None:
    assert split_message("") == []


def test_splits_on_paragraphs() -> None:
    text = "\n\n".join(["word " * 30] * 20)
    chunks = split_message(text, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks).replace("\n", "") .startswith("word")


def test_every_chunk_respects_the_limit() -> None:
    text = "a" * 20000
    for limit in (50, 100, 1000, 4000):
        for chunk in split_message(text, limit):
            assert len(chunk) <= limit, f"limit {limit} violated"


def test_no_content_is_lost() -> None:
    text = "\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(40))
    rebuilt = "".join(split_message(text, 300))
    for i in range(40):
        assert f"paragraph {i}" in rebuilt


def test_hard_split_for_unbroken_input() -> None:
    chunks = split_message("z" * 500, 100)
    assert len(chunks) >= 5
    assert all(len(c) <= 100 for c in chunks)


def test_never_splits_inside_a_fenced_code_block() -> None:
    block = "```python\n" + "print('x')\n" * 100 + "```"
    chunks = split_message("intro\n\n" + block, 300)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0, "an unbalanced fence would break Telegram"
        assert len(chunk) <= 300


def test_unterminated_fence_is_balanced_across_chunks() -> None:
    """A code block larger than a message still has to render correctly."""
    body = "```\n" + "x" * 2000  # no closing fence at all
    chunks = split_message(body, 300)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0
        assert len(chunk) <= 300


def test_fence_that_fits_closes_on_the_same_chunk() -> None:
    body = "a" * 100 + "\n```\nshort block\n```\n" + "b" * 300
    assert len(body) > 400  # otherwise there is nothing to split
    chunks = split_message(body, 400)
    assert len(chunks) > 1
    assert chunks[0].rstrip().endswith("```")


def test_word_boundaries_are_preferred() -> None:
    chunks = split_message("alpha beta gamma delta epsilon zeta", 20)
    assert all(not c.endswith("alph") for c in chunks)
    assert " ".join(chunks).split() == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def test_default_limit_leaves_room_for_the_api() -> None:
    assert TELEGRAM_LIMIT == 4096


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("hello", 100) == "hello"


def test_truncate_marks_the_cut() -> None:
    result = truncate("x" * 500, 100)
    assert len(result) <= 100
    assert "truncated" in result


def test_truncate_middle_keeps_both_ends() -> None:
    result = truncate_middle("START" + "y" * 500 + "END", 100)
    assert result.startswith("START")
    assert result.endswith("END")
    assert len(result) <= 100


# --------------------------------------------------------------------------- #
# escaping
# --------------------------------------------------------------------------- #


def test_escape_markdown_escapes_the_dangerous_set() -> None:
    escaped = escape_markdown("a_b*c`d[e")
    for char in "_*`[":
        assert f"\\{char}" in escaped


def test_escape_leaves_ordinary_text_alone() -> None:
    assert escape_markdown("plain text 123") == "plain text 123"


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


def test_format_error_includes_the_type() -> None:
    assert format_error(ValueError("bad value")) == "ValueError: bad value"


def test_format_error_without_a_message() -> None:
    assert "KeyError" in format_error(KeyError())


def test_format_error_truncates_a_huge_message() -> None:
    assert len(format_error(ValueError("x" * 5000))) < 400
