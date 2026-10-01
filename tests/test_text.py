"""Message splitting and truncation."""

from __future__ import annotations

from lumi.util.text import (
    TELEGRAM_LIMIT,
    escape_html,
    escape_markdown,
    format_error,
    markdown_code_to_html,
    sanitize_html,
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


def test_html_tags_are_reopened_after_a_split() -> None:
    """A chunk cut inside <pre> must still parse on its own."""
    text = "<pre>" + "x" * 400 + "</pre>"
    chunks = split_message(text, 120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("<pre>") == chunk.count("</pre>")
        assert chunk.count("<b>") == chunk.count("</b>")


def test_inline_tags_reopen_across_a_split() -> None:
    """Realistic nesting depth (a few tags) survives cuts at a small limit."""
    text = ("<b>bold " * 4) + ("body " * 80) + "end</b>"
    chunks = split_message(text, 120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("<b>") == chunk.count("</b>")


def test_pathological_nesting_terminates_and_respects_the_limit() -> None:
    """Depth that cannot fit its own closers must not loop forever: chunks stay
    within the limit and the sender's plain-text fallback absorbs the mess."""
    text = "<b>bold " * 30 + "end</b>"
    chunks = split_message(text, 100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).count("<b>") >= 30


def test_a_tag_cut_in_half_is_not_balanced_away() -> None:
    """A literal partial tag (escaped by the sanitizer later) splits cleanly."""
    text = "word " * 60 + "<brken " + "word " * 60
    chunks = split_message(text, 100)
    assert "".join(chunks).replace("\n", "")


def test_stray_tag_with_attributes_is_not_counted() -> None:
    """Only the shapes the sanitizer keeps may be balanced."""
    text = '<b onclick="x">' + "y" * 300
    chunks = split_message(text, 100)
    assert all("</b>" not in chunk for chunk in chunks)


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
# Telegram HTML
# --------------------------------------------------------------------------- #


def test_escape_html_handles_the_three_reserved_characters() -> None:
    assert escape_html('a < b & c > "d"') == "a &lt; b &amp; c &gt; \"d\""
    assert escape_html("underscores_and_asterisks*stay*") == "underscores_and_asterisks*stay*"


def test_sanitize_html_passes_telegram_tags_through() -> None:
    text = "<b>bold</b> and <code>x_1</code> plus <pre>line1\nline2</pre>"
    assert sanitize_html(text) == text


def test_sanitize_html_escapes_stray_angle_brackets() -> None:
    assert "&lt;script&gt;" in sanitize_html("<script>alert(1)</script>")
    assert "2 &lt; 3" in sanitize_html("if 2 < 3 then ok")


def test_sanitize_html_escapes_bare_ampersands_but_not_entities() -> None:
    assert "R&amp;D" in sanitize_html("R&D at 5%")
    assert sanitize_html("a && b") == "a &amp;&amp; b"
    assert "&amp;amp;" not in sanitize_html("&amp;gt; already escaped")  # not double-escaped
    assert "&lt;" in sanitize_html("&lt; pre-escaped entity")


def test_sanitize_html_keeps_underscores_and_asterisks_visible() -> None:
    """The whole point: content the legacy Markdown mode choked on survives."""
    text = "saved to workspace/uploads/42/notes_test.txt, max_tokens = 4096"
    assert sanitize_html(text) == text


def test_sanitize_html_strips_tag_attributes() -> None:
    """A whitelisted tag carrying extra attributes is escaped into visible text."""
    text = '<b onclick="alert(1)">hi</b>'
    out = sanitize_html(text)
    assert out.startswith("&lt;b"), "the dangerous opener must become literal text"
    assert out.endswith("</b>")
    out = sanitize_html('see <a href="https://example.com/x">the page</a>')
    assert '<a href="https://example.com/x">' in out


def test_markdown_code_bridges_fences_and_spans() -> None:
    out = markdown_code_to_html("look:\n```python\nprint('hi')\n```\nand `x_1` inline")
    assert "<pre>print(&#39;hi&#39;)</pre>" in out or "<pre>print('hi')</pre>" in out
    assert "<code>x_1</code>" in out
    assert "```" not in out


def test_markdown_code_bridge_leaves_a_lone_backtick_alone() -> None:
    assert markdown_code_to_html("it's a ` mystery") == "it's a ` mystery"


def test_sanitize_plus_bridge_is_idempotent() -> None:
    """Escaped text sent through the pipeline again must not double-escape."""
    once = sanitize_html(markdown_code_to_html("a & b with `x < y`"))
    twice = sanitize_html(once)
    assert once == twice
    assert "&amp;amp;" not in twice


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


def test_format_error_includes_the_type() -> None:
    assert format_error(ValueError("bad value")) == "ValueError: bad value"


def test_format_error_without_a_message() -> None:
    assert "KeyError" in format_error(KeyError())


def test_format_error_truncates_a_huge_message() -> None:
    assert len(format_error(ValueError("x" * 5000))) < 400
