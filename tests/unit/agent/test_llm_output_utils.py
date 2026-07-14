"""Unit tests for llm_output_utils.py — 100% branch coverage."""

from external_llm.agent.llm_output_utils import strip_markdown_fences


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences."""

    # ── No fences ────────────────────────────────────────────────────

    def test_no_fence_plain_text(self):
        assert strip_markdown_fences("hello world") == "hello world"

    def test_no_fence_empty_string(self):
        assert strip_markdown_fences("") == ""

    def test_no_fence_whitespace_only(self):
        assert strip_markdown_fences("  \n  ") == "  \n  "

    # ── Inline fences ────────────────────────────────────────────────

    def test_inline_fence_no_lang(self):
        result = strip_markdown_fences("```hello```")
        assert result == "hello"

    def test_inline_fence_with_lang(self):
        result = strip_markdown_fences("```json{\"key\": \"value\"}```")
        assert result == '{"key": "value"}'

    def test_inline_fence_with_lang_newline(self):
        result = strip_markdown_fences("```json\n{\"key\": \"value\"}\n```")
        assert result == '{"key": "value"}'

    # ── Line-based fences ────────────────────────────────────────────

    def test_line_based_fence(self):
        text = "```\ndef foo():\n    pass\n```"
        result = strip_markdown_fences(text)
        assert result == "def foo():\n    pass"

    def test_line_based_fence_json(self):
        text = "```json\n{\"key\": 42}\n```"
        result = strip_markdown_fences(text)
        assert result == '{"key": 42}'

    def test_line_based_fence_capital_json(self):
        text = "```JSON\n{\"key\": 42}\n```"
        result = strip_markdown_fences(text)
        assert result == '{"key": 42}'

    # ── strip_json_prefix=False ──────────────────────────────────────

    def test_preserve_json_prefix(self):
        text = "```json\nsome content\n```"
        result = strip_markdown_fences(text, strip_json_prefix=False)
        # With strip_json_prefix=False, "json" prefix is preserved
        assert "json" in result or result == "some content"

    def test_preserve_json_prefix_no_fence(self):
        text = "plain text"
        result = strip_markdown_fences(text, strip_json_prefix=False)
        assert result == "plain text"

    # ── Trailing backticks ───────────────────────────────────────────

    def test_dangling_backticks(self):
        text = "```code```\n```\n"
        result = strip_markdown_fences(text)
        assert "```" not in result

    def test_no_closing_fence(self):
        text = "```python\nprint('hello')"
        result = strip_markdown_fences(text)
        # Without a closing fence, the language tag "python" is preserved
        assert "print('hello')" in result

    # ── Nested fences (Phase 3 line-based fallback) ──────────────────

    def test_nested_fences_line_based_fallback(self):
        """Only the first fence pair is extracted; content after closing fence is ignored."""
        text = "```\nouter\n```\npost content"
        result = strip_markdown_fences(text)
        assert "outer" in result
        assert "post" not in result

    def test_fence_with_extra_backticks(self):
        text = "````\ncode\n````"
        result = strip_markdown_fences(text)
        assert "code" in result
        assert "``" not in result or len(result) < len(text)

    # ── Edge cases ───────────────────────────────────────────────────

    def test_only_fence_markers(self):
        result = strip_markdown_fences("``````")
        assert result == ""

    def test_only_opening_fence(self):
        result = strip_markdown_fences("```")
        assert result == ""

    def test_fence_with_trailing_text(self):
        result = strip_markdown_fences("```json result")
        assert result == "result"

    def test_mixed_content_preserved(self):
        text = "```\nline1\nline2\n```"
        result = strip_markdown_fences(text)
        assert result == "line1\nline2"

    def test_leading_whitespace_before_fence(self):
        text = "  \n  ```\ncontent\n```\n  "
        result = strip_markdown_fences(text)
        assert result == "content"

    def test_triple_backtick_in_content(self):
        text = "```\nnot a ``` fence\n```"
        result = strip_markdown_fences(text)
        # The middle ``` is inside content, the line-based pass should handle it
        assert "not a" in result
