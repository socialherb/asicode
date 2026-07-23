"""Tests for indent-visibility feedback: read_file gutter (Option A) + write-tool
indent metadata (Option E).

Option A: ``read_file`` prefixes each line with an indent gutter ``│N│`` so the
leading-whitespace column count is a readable number (no space-counting).

Option E: ``edit_text``/``modify_symbol`` success metadata surfaces the *actual*
indent at the edit site (``matched_indent`` / ``symbol_def_indent``), mirroring
the gutter — so the LLM can self-verify it matched the file's depth without a
read_file round-trip.
"""
import pytest

from external_llm.agent.symbol_search import SymbolDef
from external_llm.agent.tool_handlers.read_tools import (
    _INDENT_GUTTER_BAR,
    ReadToolsMixin,
    _format_numbered_line,
    _split_source_lines,
)
from external_llm.agent.tool_handlers.write_tools import (
    WriteToolsMixin,
    _leading_indent_width,
)
from external_llm.agent.tool_registry import ToolResult
from pathlib import Path


class _Harness(WriteToolsMixin):
    """Minimal concrete host for WriteToolsMixin handlers."""

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._repo_root_override = None
        self._applied_patches = []

    @property
    def _effective_repo_root(self):
        return self.repo_root

    def _make_result(self, **kwargs):
        kwargs.setdefault("content", "")
        return ToolResult(**kwargs)

    def _run_syntax_check_for_file(self, path):
        return {"ok": True, "skipped": True, "reason": "test"}

    def _secure_path(self, path, *, confine=False):
        p = Path(path)
        if p.is_absolute():
            return None
        resolved = Path(self.repo_root) / path
        return resolved if resolved.exists() else None

    def _invalidate_cache_after_write(self, files):
        pass


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


# ── Option A: read_file indent gutter ────────────────────────────────────────────


class TestReadFileIndentGutter:
    def test_gutter_reports_leading_whitespace_count(self):
        assert _format_numbered_line(1, "def foo():") == "     1 │ 0│ def foo():"
        assert _format_numbered_line(2, "    x = 1") == "     2 │ 4│     x = 1"
        assert _format_numbered_line(3, "        return x") == "     3 │ 8│         return x"

    def test_blank_line_reports_zero(self):
        assert _format_numbered_line(4, "") == "     4 │ 0│ "

    def test_tab_counts_as_width_one(self):
        # Consistent with min_indent in common/indent_utils.
        assert _format_numbered_line(5, "\tpass") == "     5 │ 1│ \tpass"

    def test_gutter_uses_box_drawing_bar_not_ascii_pipe(self):
        # U+2502 (│) is never a valid column-0 code prefix, so a naive LLM copy
        # of the line (which starts at the code, past the gutter) cannot
        # accidentally include the gutter. ASCII | appears in code (type unions,
        # bitwise-or) and must NOT be used.
        assert _INDENT_GUTTER_BAR == "\u2502"
        line = _format_numbered_line(1, "x | y")
        assert line.count("\u2502") == 2  # exactly the two gutter bars
        assert " | " in line  # the real code pipe is preserved verbatim

    def test_line_number_right_aligned(self):
        # 6-wide field, right-aligned — aligns with the pre-existing read_file
        # numbering so multi-hundred-line files stay readable.
        assert _format_numbered_line(1, "x").startswith("     1 ")
        assert _format_numbered_line(999, "x").startswith("   999 ")


# ── Option E: edit_text matched_indent metadata ──────────────────────────────────


class TestEditTextMatchedIndentMetadata:
    def test_matched_indent_reflects_edit_site(self, harness, tmp_path):
        target = tmp_path / "t.py"
        target.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        x = 1\n"
            "        return x\n",
            encoding="utf-8",
        )
        # Edit the line at indent 8 (inside the method body).
        result = harness._tool_edit_text({
            "file_path": "t.py",
            "old_string": "        x = 1",
            "new_string": "        x = 2",
        })
        assert result.ok
        assert result.metadata["matched_line"] == 3
        assert result.metadata["matched_indent"] == 8

    def test_matched_indent_top_level_is_zero(self, harness, tmp_path):
        target = tmp_path / "t.py"
        target.write_text("ALPHA = 1\nBETA = 2\n", encoding="utf-8")
        result = harness._tool_edit_text({
            "file_path": "t.py",
            "old_string": "ALPHA = 1",
            "new_string": "ALPHA = 10",
        })
        assert result.ok
        assert result.metadata["matched_indent"] == 0
        assert result.metadata["matched_line"] == 1

    def test_reindent_applied_flag_set_when_new_string_indent_corrected(
        self, harness, tmp_path
    ):
        # Multi-line block where the LLM strips ALL indentation from old_string.
        # The fallback matcher reconstructs old_string at the file's actual
        # indent (12), and _reindent_to_match shifts new_string to match — so
        # reindent_applied=True. (Single-line substring matches never trigger
        # reindent because they match exactly regardless of indent.)
        target = tmp_path / "t.py"
        target.write_text(
            "class C:\n"
            "    def m(self):\n"
            "        if flag:\n"
            "            a = 1\n"
            "            b = 2\n"
            "        after()\n",
            encoding="utf-8",
        )
        # Both old and new sent flush-LEFT (wrong) — the file's block is at 12.
        result = harness._tool_edit_text({
            "file_path": "t.py",
            "old_string": "a = 1\nb = 2",
            "new_string": "a = 10\nb = 20",
        })
        assert result.ok
        # Reindent fired: new_string shifted from 0 to 12.
        assert result.metadata.get("reindent_applied") is True
        assert result.metadata["matched_indent"] == 12
        # And the file content is correctly indented (not flush-left).
        written = target.read_text(encoding="utf-8")
        assert "            a = 10\n            b = 20" in written

    def test_reindent_applied_absent_when_indent_already_correct(
        self, harness, tmp_path
    ):
        target = tmp_path / "t.py"
        target.write_text("    x = 1\n", encoding="utf-8")
        result = harness._tool_edit_text({
            "file_path": "t.py",
            "old_string": "    x = 1",
            "new_string": "    x = 2",  # same indent → no reindent
        })
        assert result.ok
        assert "reindent_applied" not in result.metadata
        assert result.metadata["matched_indent"] == 4

    def test_batch_mode_omits_per_edit_indent_detail(self, harness, tmp_path):
        # Batch mode does not surface per-edit matched_indent (diff carries it).
        target = tmp_path / "t.py"
        target.write_text("a = 1\nb = 2\n", encoding="utf-8")
        result = harness._tool_edit_text({
            "file_path": "t.py",
            "edits": [
                {"old_string": "a = 1", "new_string": "a = 10"},
                {"old_string": "b = 2", "new_string": "b = 20"},
            ],
        })
        assert result.ok
        assert "matched_indent" not in result.metadata
        assert "matched_line" not in result.metadata


# ── Option E: modify_symbol symbol_def_indent metadata ───────────────────────────


class TestModifySymbolDefIndentMetadata:
    def test_method_def_indent_reported(self, harness, tmp_path):
        target = tmp_path / "t.py"
        target.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n",
            encoding="utf-8",
        )
        result = harness._tool_modify_symbol({
            "file_path": "t.py",
            "symbol": "bar",
            "code": "    def bar(self):\n        return 2\n",
        })
        assert result.ok
        # bar's def line is line 2 at indent 4.
        assert result.metadata["symbol_def_line"] == 2
        assert result.metadata["symbol_def_indent"] == 4

    def test_top_level_def_indent_is_zero(self, harness, tmp_path):
        target = tmp_path / "t.py"
        target.write_text(
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n"
            "\n"
            "def baz():\n"
            "    pass\n",
            encoding="utf-8",
        )
        result = harness._tool_modify_symbol({
            "file_path": "t.py",
            "symbol": "baz",
            "code": "def baz():\n    return 2\n",
        })
        assert result.ok
        assert result.metadata["symbol_def_indent"] == 0
        assert result.metadata["symbol_def_line"] == 5


# ── Option A: read_symbol indent gutter (parity with read_file) ──────────────────


class _FakeSymbolSearcher:
    """Returns a single pre-built SymbolDef regardless of query."""

    def __init__(self, sym):
        self._sym = sym

    def find_symbol(self, name, search_path=None):
        return [self._sym] if self._sym else []


class _ReadHarness(ReadToolsMixin):
    """Minimal concrete host for ReadToolsMixin._tool_read_symbol."""

    def __init__(self, repo_root, sym):
        self.repo_root = str(repo_root)
        self._symbol_searcher = _FakeSymbolSearcher(sym)

    def _make_result(self, **kwargs):
        kwargs.setdefault("content", "")
        return ToolResult(**kwargs)


class TestReadSymbolIndentGutter:
    def _build(self, tmp_path, src, sym_line, sym_end=None):
        (tmp_path / "t.py").write_text(src, encoding="utf-8")
        sym = SymbolDef(
            file="t.py", line=sym_line, kind="function",
            name="bar", end_line=sym_end,
        )
        return _ReadHarness(tmp_path, sym)

    def test_lines_carry_indent_gutter(self, tmp_path):
        src = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n"
        )
        harness = self._build(tmp_path, src, sym_line=2, sym_end=3)
        result = harness._tool_read_symbol({"name": "bar", "context_lines": 0})
        assert result.ok
        # def line is line 2 at indent 4; body line 3 at indent 8.
        assert "     2 │ 4│     def bar(self):" in result.content
        assert "     3 │ 8│         return 1" in result.content

    def test_line_numbers_start_at_window_start(self, tmp_path):
        # context_lines=1 → start = sym.line-1-1 = line 1 (1-based).
        src = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        return 1\n"
            "    def baz(self):\n"
            "        return 2\n"
        )
        harness = self._build(tmp_path, src, sym_line=2, sym_end=3)
        result = harness._tool_read_symbol({"name": "bar", "context_lines": 1})
        # First numbered line must be line 1, not line 2.
        assert "     1 │ 0│ class Foo:" in result.content

    def test_header_carries_gutter_legend(self, tmp_path):
        src = "def bar():\n    return 1\n"
        harness = self._build(tmp_path, src, sym_line=1, sym_end=2)
        result = harness._tool_read_symbol({"name": "bar", "context_lines": 0})
        assert "leading-indent column count" in result.content
        assert _INDENT_GUTTER_BAR in result.content

    def test_gutter_uses_box_drawing_bar_not_ascii_pipe(self, tmp_path):
        src = "def bar():\n    return a | b\n"
        harness = self._build(tmp_path, src, sym_line=1, sym_end=2)
        result = harness._tool_read_symbol({"name": "bar", "context_lines": 0})
        # The body line keeps its real ASCII pipe; gutter uses U+2502.
        assert "│ 4│     return a | b" in result.content

    def test_form_feed_does_not_misalign_symbol_slice(self, tmp_path):
        # Regression: str.splitlines() counts \f (form-feed) as a line break,
        # but ast.lineno counts \n only. A form-feed-only line above a def
        # made read_symbol index a splitlines() array with an ast lineno,
        # slicing/displaying the WRONG lines. Now aligned to \n via
        # _split_source_lines (sym.line comes from ast, so the array must
        # use the same \n-only model).
        src = "# header\n\x0c\n\ndef bar():\n    return 1\n"
        harness = self._build(tmp_path, src, sym_line=4, sym_end=5)
        result = harness._tool_read_symbol({"name": "bar", "context_lines": 0})
        assert result.ok
        # ast says def bar is at L4 (\n-only counting). The fix must show the
        # actual def there — NOT the empty line splitlines() would place at
        # index 3 (its L4) due to the \f being counted as a line break.
        assert "     4 │ 0│ def bar():" in result.content
        assert "     5 │ 4│     return 1" in result.content
        assert "def bar" in result.content


class TestSplitSourceLinesHelper:
    """Unit tests for the ast/git-aligned (\n-only) line splitter."""

    def test_aligns_with_ast_newline_model(self):
        # splitlines() yields 6 elements (def at index 4 = L5);
        # _split_source_lines yields 5 (def at index 3 = L4), matching ast.lineno.
        src = "# header\n\x0c\n\ndef bar():\n    return 1\n"
        assert _split_source_lines(src) == ["# header", "\x0c", "", "def bar():", "    return 1"]
        assert _split_source_lines(src).index("def bar():") == 3  # 0-indexed → L4

    def test_trailing_newline_dropped(self):
        assert _split_source_lines("a\nb\n") == ["a", "b"]
        assert _split_source_lines("a\nb") == ["a", "b"]  # no trailing newline

    def test_empty_and_single_newline(self):
        # Matches str.splitlines() edge behavior: "" → [], "\n" → [""].
        assert _split_source_lines("") == []
        assert _split_source_lines("\n") == [""]


# ── _leading_indent_width unit ──────────────────────────────────────────────


class TestLeadingIndentWidth:
    def test_spaces(self):
        assert _leading_indent_width("    x") == 4
        assert _leading_indent_width("        y") == 8

    def test_first_non_blank_line(self):
        # Only the first non-blank line matters (a body's def-line anchor).
        assert _leading_indent_width("\n\n    x") == 4

    def test_empty_returns_zero(self):
        assert _leading_indent_width("") == 0
        assert _leading_indent_width("   \n  \n") == 0

    def test_tab_counts_as_one(self):
        assert _leading_indent_width("\tpass") == 1
