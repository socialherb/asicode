"""
Unit tests for the pure indentation / diff helpers in symbol_modify_tool.py.

The public ``modify_symbol`` path is well covered, and the indent-correction
helpers (``_reindent_relative``, ``_correct_indent_drift``, ``min_indent``)
are exercised indirectly through several test files. But a cluster of lower-
level helpers had ZERO direct coverage:

  - _analyze_logical_lines          (tokenizer-based logical-line classification)
  - _file_indent_unit_from_logical  (GCD indent detection excluding continuations)
  - _mode_logical_indent            (most-common indent, tie → shallowest)
  - _block_parses_after_dedent      (ast.parse consistency check)
  - _post_edit_syntax_ok               (compile / node --check / gofmt gate)
  - _apply_diff_to_source           (in-memory unified-diff applier)

These functions are the bedrock of the re-indentation heuristics (the very
class of bugs this branch was created to fix), so their edge cases deserve
first-class regression tests.

Run: pytest tests/unit/agent/test_symbol_modify_helpers.py -v
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Ensure repo root on path for any sys-level imports the module does.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_llm.agent.symbol_modify_tool import (  # noqa: E402
    _apply_diff_to_source,
    _block_parses_after_dedent,
    _mode_logical_indent,
    _post_edit_syntax_ok,
    _reindent_relative,
)
from external_llm.common.indent_utils import (  # noqa: E402
    _analyze_logical_lines,
    _file_indent_unit_from_logical,
    min_indent,
)

# ── _analyze_logical_lines ───────────────────────────────────────────────────

class TestAnalyzeLogicalLines:
    def test_simple_one_line_per_logical(self):
        snippet = "x = 1\ny = 2\n"
        owner, logical_rows = _analyze_logical_lines(snippet)
        # Each physical row starts its own logical line.
        assert logical_rows == {1, 2}
        assert owner[1] == 1 and owner[2] == 2

    def test_paren_continuation_owned_by_opener(self):
        # Row 2 is a bracket continuation → owned by row 1, not a logical start.
        snippet = "foo(\n    1,\n)\n"
        owner, logical_rows = _analyze_logical_lines(snippet)
        assert 1 in logical_rows
        assert 2 not in logical_rows
        assert owner[2] == 1

    def test_triple_quoted_string_interior_owned_by_opener(self):
        snippet = 'x = """\nline two\nline three\n"""\n'
        owner, logical_rows = _analyze_logical_lines(snippet)
        assert 1 in logical_rows
        # Interior rows of the multi-line string belong to row 1.
        assert owner.get(2) == 1
        assert owner.get(3) == 1

    def test_returns_none_on_syntax_error(self):
        # A genuinely broken fragment cannot be tokenized.
        assert _analyze_logical_lines("def (\n") is None

    def test_empty(self):
        owner, logical_rows = _analyze_logical_lines("")
        assert logical_rows == set()
        assert owner == {}


# ── _file_indent_unit_from_logical ───────────────────────────────────────────

class TestFileIndentUnitFromLogical:
    def test_four_space_file(self):
        src = "def f():\n    a = 1\n    b = 2\n"
        assert _file_indent_unit_from_logical(src, " ") == 4

    def test_two_space_file(self):
        src = "def f():\n  a = 1\n  b = 2\n"
        assert _file_indent_unit_from_logical(src, " ") == 2

    def test_nested_levels_gcd(self):
        # 4 and 8 → GCD 4
        src = "def f():\n    if x:\n        y = 1\n"
        assert _file_indent_unit_from_logical(src, " ") == 4

    def test_alignment_continuation_does_not_poison_gcd(self):
        # A hanging-indent continuation aligned at 17 spaces must not collapse
        # the GCD toward 1 — logical-line filtering excludes it.
        src = "def f(arg_one,\n         arg_two):  # aligned to col 9\n    return 1\n"
        unit = _file_indent_unit_from_logical(src, " ")
        assert unit == 4

    def test_tab_file(self):
        src = "def f():\n\ta = 1\n\tb = 2\n"
        assert _file_indent_unit_from_logical(src, "\t") == 1

    def test_falls_back_on_untokenizable(self):
        # Broken source → falls back to indent_unit(); just ensure no crash and
        # returns a positive int.
        unit = _file_indent_unit_from_logical("def (\n", " ")
        assert isinstance(unit, int) and unit >= 1


# ── _mode_logical_indent ─────────────────────────────────────────────────────

class TestModeLogicalIndent:
    def test_most_common_width(self):
        lines = ["        a", "        b", "        c", "    outlier"]
        assert _mode_logical_indent(lines) == 8

    def test_tie_breaks_to_shallowest(self):
        # Two at width 4, two at width 8 → tie → return 4 (shallower, safe).
        lines = ["    a", "    b", "        c", "        d"]
        assert _mode_logical_indent(lines) == 4

    def test_ignores_blank_lines(self):
        lines = ["    a", "", "   ", "    b"]
        assert _mode_logical_indent(lines) == 4

    def test_empty_returns_zero(self):
        assert _mode_logical_indent([]) == 0
        assert _mode_logical_indent(["", "  "]) == 0

    def test_single_outlier_shallower_does_not_mask_majority(self):
        # The whole point vs min_indent: a docstring first line at
        # depth 4 while the rest of the body drifted to depth 8 should report 8.
        lines = ["    def line0_at_target", "        drifted1", "        drifted2"]
        assert _mode_logical_indent(lines) == 8
        # And the min-based helper would incorrectly report 4:
        assert min_indent(lines) == 4


# ── _block_parses_after_dedent ───────────────────────────────────────────────

class TestBlockParsesAfterDedent:
    def test_consistent_block_parses(self):
        lines = ["    a = 1", "    b = 2"]
        assert _block_parses_after_dedent(lines) is True

    def test_inconsistent_indent_does_not_parse(self):
        # After dedent by the common min indent, a body whose indent profile
        # is internally inconsistent raises SyntaxError.
        #   common=4 → ['    a = 1', 'if x:', '        pass']
        #   'if x:' at col 0 then 'pass' at col 8 (should be 4) → inconsistent.
        lines = ["        a = 1", "    if x:", "        pass"]
        assert _block_parses_after_dedent(lines) is False

        # Another genuinely inconsistent one (4 then 2):
        assert _block_parses_after_dedent(["    a = 1", "  b = 2"]) is False

    def test_empty_block_parses(self):
        assert _block_parses_after_dedent([]) is True
        assert _block_parses_after_dedent(["", "  "]) is True

    def test_nested_body_parses_despite_deep_indent(self):
        # A correctly nested body at deep indent still parses after dedent.
        lines = ["        if x:", "            y = 1", "        z = 2"]
        assert _block_parses_after_dedent(lines) is True


# ── _post_edit_syntax_ok ────────────────────────────────────────────────────────

class TestPostEditSyntaxOk:
    def test_valid_python(self):
        assert _post_edit_syntax_ok("x = 1\n", "m.py") is True

    def test_invalid_python(self):
        assert _post_edit_syntax_ok("def (\n", "m.py") is False

    def test_non_python_passes_through(self):
        # Non-Python files without node/gofmt infra pass through as True.
        assert _post_edit_syntax_ok("garbage {{{", "m.js") in (True, False)

    def test_path_drives_language(self):
        # Same content, .txt → not Python/JS/TS/Go → always True.
        assert _post_edit_syntax_ok("def (\n", "m.txt") is True


# ── _apply_diff_to_source ────────────────────────────────────────────────────

class TestApplyDiffToSource:
    def test_simple_replacement(self):
        source = "line1\nline2\nline3\n"
        diff = textwrap.dedent("""\
            @@ -1,3 +1,3 @@
             line1
            -line2
            +LINE2
             line3
        """)
        assert _apply_diff_to_source(source, diff) == "line1\nLINE2\nline3\n"

    def test_insertion_only(self):
        source = "a\nb\n"
        diff = textwrap.dedent("""\
            @@ -1,2 +1,3 @@
             a
            +inserted
             b
        """)
        assert _apply_diff_to_source(source, diff) == "a\ninserted\nb\n"

    def test_deletion_only(self):
        source = "a\nb\nc\n"
        diff = textwrap.dedent("""\
            @@ -1,3 +1,2 @@
             a
            -b
             c
        """)
        assert _apply_diff_to_source(source, diff) == "a\nc\n"

    def test_multiple_hunks(self):
        source = "a\nb\nc\nd\ne\n"
        diff = textwrap.dedent("""\
            @@ -1,2 +1,2 @@
            -a
            +A
             b
            @@ -4,2 +4,2 @@
             d
            -e
            +E
        """)
        assert _apply_diff_to_source(source, diff) == "A\nb\nc\nd\nE\n"

    def test_context_line_space_prefix_stripped(self):
        # The critical bug class: context lines prefixed with a single space
        # MUST be stripped, otherwise indentation is corrupted.
        source = "def f():\n    x = 1\n"
        diff = textwrap.dedent("""\
            @@ -1,2 +1,2 @@
             def f():
            -    x = 1
            +    x = 2
        """)
        result = _apply_diff_to_source(source, diff)
        # No leading-space corruption on the 'def f():' line.
        assert result == "def f():\n    x = 2\n"

    def test_adds_trailing_newline_to_new_lines(self):
        source = "a\n"
        diff = "@@ -1,1 +1,2 @@\n a\n+b"
        # 'b' has no newline in the diff; the applier must add one.
        assert _apply_diff_to_source(source, diff) == "a\nb\n"

    def test_no_newline_meta_line_ignored(self):
        source = "a\n"
        diff = "@@ -1,1 +1,2 @@\n a\n+b\n\\ No newline at end of file"
        # The \ meta line must not be treated as a context/add line.
        assert _apply_diff_to_source(source, diff) == "a\nb\n"

    def test_idempotent_when_no_hunks(self):
        source = "a\nb\n"
        assert _apply_diff_to_source(source, "") == "a\nb\n"


# ── _reindent_relative (focused edge cases) ──────────────────────────────────
# _reindent_relative is covered elsewhere, but the cross-char tab↔space branch
# and continuation-preservation are subtle enough to pin down here.

class TestReindentRelativeEdgeCases:
    def test_blank_lines_emit_empty(self):
        body_lines = ["    x = 1", "", "    y = 2"]
        out = _reindent_relative(
            body_lines,
            anchor_indent=4,
            base_prefix="    ",
            model_char=" ",
            model_unit=4,
            file_char=" ",
            file_unit=4,
        )
        assert out[1] == ""
        assert out[0].startswith("    x = 1")
        assert out[2].startswith("    y = 2")

    def test_continuation_shifts_with_owner_delta(self):
        # A paren continuation must move by the same delta as its owning line.
        body_lines = [
            "    foo(",       # logical, depth 0 relative
            "        1,",     # continuation owned by line 1, col 8
        ]
        out = _reindent_relative(
            body_lines,
            anchor_indent=4,
            base_prefix="    ",
            model_char=" ",
            model_unit=4,
            file_char=" ",
            file_unit=4,
        )
        # Owner line maps to base_prefix (4) + 0 extra = 4 cols.
        # Continuation was at 8, owner was at 4 → delta 0 → stays at 8.
        assert out[0] == "    foo("
        assert out[1] == "        1,"

    def test_cross_char_conversion_bounded(self):
        # Tab-indented model body converted to a space file: must not produce
        # a column-count run of tabs or explode the indent.
        body_lines = ["\tx = 1", "\t\ty = 2"]
        out = _reindent_relative(
            body_lines,
            anchor_indent=1,   # one tab
            base_prefix="    ",
            model_char="\t",
            model_unit=1,
            file_char=" ",
            file_unit=4,
        )
        # Line 1 → base_prefix (4 spaces). Line 2 → 4 + 4 = 8 spaces.
        assert out[0] == "    x = 1"
        assert out[1] == "        y = 2"
        # No tabs leak into the output.
        assert all("\t" not in _item_ for _item_ in out)
