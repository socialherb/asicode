"""Unit tests for WriteToolsEditMixin — edit_text / edit_file / create_file /
modify_symbol family.

Covers the pure and near-pure helpers that the existing write-tools tests do
not reach: anchor resolution fallbacks, near-match hinting, missing-path
suggestions, fallback matching (_resolve_with_fallback), scoped replacement,
edited-line-region diagnosis, indentation/structural hints, raw-args recovery,
and the end-to-end handlers on tmp files with stubbed side-effect hooks.
"""

from __future__ import annotations

import pytest

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_handlers.write_tools_edit_mixin import WriteToolsEditMixin
from external_llm.agent.tool_registry import ToolResult


class _Harness(WriteToolsMixin):
    """Minimal concrete host for the edit mixin."""

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._repo_root_override = None
        self._applied_patches = []
        self._text_edited_files = set()

    @property
    def _effective_repo_root(self):
        return self.repo_root

    def _make_result(self, **kwargs):
        kwargs.setdefault("content", "")
        return ToolResult(**kwargs)

    def _run_syntax_check_for_file(self, path, **kwargs):
        return {"ok": True, "skipped": True, "reason": "test"}

    def _secure_path(self, path, *, confine=False):
        from pathlib import Path as _Path

        repo = _Path(self.repo_root).resolve()
        p = _Path(path)
        resolved = p.resolve() if p.is_absolute() else (repo / path).resolve()
        try:
            resolved.relative_to(repo)
        except ValueError:
            return None
        return resolved

    def _should_soft_fail_verify(self, verify_detail, snapshots):
        return False


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


# ── _resolve_edit_anchor ────────────────────────────────────────────────────


class TestResolveEditAnchor:
    CONTENT = "alpha\nbeta\ngamma\ndelta\n"

    def test_exact_unique(self, harness):
        pos, actual, ratio = harness._resolve_edit_anchor(self.CONTENT, "beta")
        assert self.CONTENT[pos : pos + 4] == "beta"
        assert actual == "beta"
        assert ratio == 0.0

    def test_exact_multiple_with_line_hint(self, harness):
        content = "x\nbeta\ny\nbeta\nz\n"
        pos, actual, _ratio = harness._resolve_edit_anchor(content, "beta", line=4)
        assert actual == "beta"
        # line 4 is the second "beta" (0-indexed line 3)
        assert content[pos : pos + 4] == "beta"
        assert pos > content.find("beta")  # NOT the first occurrence

    def test_exact_multiple_without_line_hint_raises(self, harness):
        content = "beta\nbeta\n"
        with pytest.raises(ValueError, match="not unique"):
            harness._resolve_edit_anchor(content, "beta")

    def test_line_hint_fallback_to_content(self, harness):
        pos, actual, _ratio = harness._resolve_edit_anchor(self.CONTENT, "ZZZ", line=3)
        assert actual == "gamma"
        assert self.CONTENT[pos : pos + 5] == "gamma"

    def test_line_hint_empty_line_falls_through(self, harness):
        content = "a\n\nb\n"
        with pytest.raises(ValueError, match="anchor text not found"):
            harness._resolve_edit_anchor(content, "ZZZ", line=2)

    def test_first_line_strip_match(self, harness):
        content = "    beta\n    gamma\n"
        _pos, actual, _ratio = harness._resolve_edit_anchor(content, "beta\ngamma")
        assert actual == "    beta\n    gamma"

    def test_first_line_multiple_strip_matches_raises(self, harness):
        content = "  beta\n  beta\n"
        with pytest.raises(ValueError, match="not unique"):
            harness._resolve_edit_anchor(content, "beta")

    def test_multi_line_reconstruction(self, harness):
        content = "a\n  beta\n  gamma\nz\n"
        _pos, actual, _ratio = harness._resolve_edit_anchor(content, "beta\ngamma")
        assert actual == "  beta\n  gamma"

    def test_multi_line_progressive_fallback(self, harness):
        content = "a\n  beta\n  gamma\n  delta\nz\n"
        _pos, actual, _ = harness._resolve_edit_anchor(content, "beta\ngamma\ndelta")
        assert actual == "  beta\n  gamma\n  delta"

    def test_not_found_raises_with_suggestions(self, harness):
        with pytest.raises(ValueError, match="anchor text not found"):
            harness._resolve_edit_anchor(self.CONTENT, "alpa")


# ── _current_file_head_snippet / _patch_failure_snippet / _raw_repr ─────────


class TestCurrentFileHeadSnippet:
    def test_basic(self, harness):
        out = harness._current_file_head_snippet("a\nb\nc\n")
        assert "│1│ a" in out
        assert "│3│ c" in out

    def test_empty_returns_empty(self, harness):
        assert harness._current_file_head_snippet("") == ""

    def test_max_lines_tail(self, harness):
        out = harness._current_file_head_snippet("\n".join(f"l{i}" for i in range(40)))
        assert "more lines" in out

    def test_max_chars_truncation_returns_empty(self, harness):
        long_line = "x" * 5000
        out = harness._current_file_head_snippet(long_line + "\n")
        assert out == ""  # single line already exceeds char budget


class TestPatchFailureSnippet:
    def test_path_hint_existing_file(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        out = harness._patch_failure_snippet("unused patch", "t.txt")
        assert "│1│ a" in out

    def test_path_hint_missing_file_empty(self, harness):
        assert harness._patch_failure_snippet("patch", "missing.txt") == ""

    def test_extract_from_patch(self, harness, tmp_path):
        (tmp_path / "f.txt").write_text("hello\n", encoding="utf-8")
        patch = "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-hello\n+bye\n"
        out = harness._patch_failure_snippet(patch, None)
        assert "│1│ hello" in out

    def test_no_target_empty(self, harness):
        assert harness._patch_failure_snippet("not a patch", None) == ""


class TestRawRepr:
    def test_basic(self, harness):
        out = harness._raw_repr("a\nb")
        assert "Raw old_string (repr)" in out
        assert "a\\nb" in out  # repr-escaped newline

    def test_empty(self, harness):
        assert harness._raw_repr("") == ""

    def test_truncated_long(self, harness):
        out = harness._raw_repr("a\nb\nc\nd\n")
        assert "total lines" in out


# ── _near_match_hint ────────────────────────────────────────────────────────


class TestNearMatchHint:
    def test_close_match_returns_numbered_block(self, harness):
        content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        out = harness._near_match_hint(content, "    return 3")
        assert "Closest match" in out
        assert "your old_string vs file" in out

    def test_whitespace_drift_note(self, harness):
        content = "x = 1\n y = 2\nz = 3\n"
        out = harness._near_match_hint(content, "y = 2")
        assert "Whitespace differs" in out

    def test_markdown_decoration_note(self, harness):
        content = "a = `code`\n"
        out = harness._near_match_hint(content, "a = code")
        assert "Markdown decoration differs" in out

    def test_multiline_content_change_ratio_survives_autojunk(self, harness):
        """P24-1: char-level autojunk collapse on multi-line old_strings.

        The window scorer joins old_lines with '\\n' and runs
        SequenceMatcher with autojunk=True (default). For any old_string of
        >=200 chars the common chars (e/t/s/space/\\n/...) exceed the 1%
        popularity threshold and are purged from b2j. When the drift is at a
        *popular* char (whitespace) the match survives, but when a
        *non-popular* char changed (a single digit/letter — the normal case
        for a content edit) the aligned anchor vanishes and every remaining
        anchor is misaligned, so the window scan reports ~46% for a block
        that is 99.9% identical. The 0.88 note threshold is then
        unreachable and the hint misleads the model. autojunk=False restores
        ~100%.
        """
        lines = [f"    self.attr_{i} = compute_value_{i}(input_{i}, options_{i}, flags[{i}])" for i in range(60)]
        content = "\n".join(lines)
        old = content.replace("compute_value_10", "compute_value_1X")  # 1-char content edit
        out = harness._near_match_hint(content, old)
        assert "~9" in out or "~100%" in out, f"ratio collapsed (autojunk): {out[:200]}"

    def test_window_scan_bounded_not_full_blob_per_candidate(self, harness):
        """Performance contract for the hint window scan.

        The scorer used to iterate EVERY start in [ci-window+1, ci] for each
        candidate line — a 60-line old_string with 5 candidates scored up to
        300 full-blob SequenceMatcher comparisons (~19s for one hint). The
        scan is now bounded to at most 3 starts per candidate, so the same
        pathological input must stay well under a second. This pins the
        bound: if the window scan regresses to unbounded, this test's
        2s ceiling fails.
        """
        import time

        lines = [f"    self.attr_{i} = compute_value_{i}(input_{i}, options_{i}, flags[{i}])" for i in range(60)]
        content = "\n".join(lines)
        old = content.replace("compute_value_10", "compute_value_1X")
        t0 = time.monotonic()
        out = harness._near_match_hint(content, old)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"window scan unbounded again: {elapsed:.2f}s"
        assert "~9" in out or "~100%" in out, f"ratio collapsed (autojunk): {out[:200]}"

    def test_no_match_returns_empty(self, harness):
        content = "aaa\nbbb\nccc\n"
        assert harness._near_match_hint(content, "zzzzzz") == ""

    def test_empty_inputs(self, harness):
        assert harness._near_match_hint("", "x") == ""
        assert harness._near_match_hint("x\n", "") == ""


# ── _suggest_missing_paths ──────────────────────────────────────────────────


class TestSuggestMissingPaths:
    def test_exact_basename_suggestion(self, harness, tmp_path, monkeypatch):
        from external_llm.agent.tool_handlers import write_tools_edit_mixin as mod

        (tmp_path / "my_file.py").write_text("x", encoding="utf-8")
        monkeypatch.setattr(mod, "_repo_file_index", lambda root: ["my_file.py"])
        out = harness._suggest_missing_paths("my_fil.py")
        assert out == ". Did you mean: my_file.py"

    def test_no_match_empty(self, harness, monkeypatch):
        from external_llm.agent.tool_handlers import write_tools_edit_mixin as mod

        monkeypatch.setattr(mod, "_repo_file_index", lambda root: ["a.py", "b.py"])
        assert harness._suggest_missing_paths("zzz.py") == ""

    def test_empty_input(self, harness, monkeypatch):
        from external_llm.agent.tool_handlers import write_tools_edit_mixin as mod

        monkeypatch.setattr(mod, "_repo_file_index", lambda root: ["a.py"])
        assert harness._suggest_missing_paths("") == ""


# ── _ast_fail_hint ──────────────────────────────────────────────────────────


class TestAstFailHint:
    def test_symbol_close_match(self, harness):
        source = "def compute_result():\n    pass\n"
        out = harness._ast_fail_hint(source, [], "compute_resultt")
        assert "Did you mean" in out and "compute_result" in out

    def test_symbol_not_found_lists_defined(self, harness):
        source = "def alpha():\n    pass\n"
        out = harness._ast_fail_hint(source, [], "omega")
        assert "not found" in out

    def test_replace_expr_near_match(self, harness):
        source = "x = 1\n"
        out = harness._ast_fail_hint(source, [{"type": "replace_expr", "old": "x = 2"}], "")
        assert "replace_expr" in out and "Closest match" in out

    def test_delete_stmt_near_match(self, harness):
        source = "y = 1\n"
        out = harness._ast_fail_hint(source, [{"type": "delete_stmt", "pattern": "y = 2"}], "")
        assert "delete_stmt" in out

    def test_no_candidates_empty(self, harness):
        source = "z = 1\n"
        out = harness._ast_fail_hint(source, [{"type": "replace_expr", "old": "qqqq"}], "")
        assert out == ""

    def test_broken_source_empty(self, harness):
        out = harness._ast_fail_hint("def broken(:\n", [], "sym")
        assert out == ""


# ── _resolve_with_fallback ──────────────────────────────────────────────────


class TestResolveWithFallback:
    def test_exact_match(self, harness):
        content = "a\nb\n"
        resolved, count, fb, _split = harness._resolve_with_fallback(content, "b")
        assert resolved == "b" and count == 1 and fb is None

    def test_trailing_whitespace_tolerant(self, harness):
        content = "a\nb  \nc\n"
        resolved, count, fb, _split = harness._resolve_with_fallback(content, "b\nc")
        assert resolved == "b  \nc" and count == 1
        assert fb is not None

    def test_indent_tolerant(self, harness):
        content = "a\n    b\nc\n"
        resolved, count, _fb, _split = harness._resolve_with_fallback(content, "a\nb")
        assert resolved == "a\n    b" and count == 1

    def test_unicode_decorative_tolerant(self, harness):
        content = "a\n\u2500\u2500 b\nc\n"  # box-drawing horizontal lines
        _resolved, count, _fb, _split = harness._resolve_with_fallback(content, "-- b")
        assert count == 1

    def test_no_match(self, harness):
        _resolved, count, fb, _split = harness._resolve_with_fallback("a\nb\n", "zzz")
        assert count == 0 and fb is None

    def test_multiple_ws_matches_count(self, harness):
        content = "b\nb\n"
        resolved, count, _fb, _split = harness._resolve_with_fallback(content, "b")
        assert count == 2
        assert resolved == "b"  # ambiguous → caller's original


# ── _edited_line_regions (static) ───────────────────────────────────────────


class TestEditedLineRegions:
    def test_line_inside_edited_region(self, harness):
        orig = "a\nb\nc\nd\n"
        mod = "a\nB\nc\nd\n"
        in_region, regions = harness._edited_line_regions(orig, mod, 2)
        assert in_region is True
        assert regions == [(2, 2)]

    def test_line_outside_edited_region(self, harness):
        orig = "a\nb\nc\nd\n"
        mod = "a\nB\nc\nd\n"
        in_region, _regions = harness._edited_line_regions(orig, mod, 4)
        assert in_region is False

    def test_context_window(self, harness):
        orig = "a\nb\nc\nd\ne\n"
        mod = "a\nb\nC\nd\ne\n"
        # line 1 is 1 away from edited line 3 → within context=1
        in_region, _ = harness._edited_line_regions(orig, mod, 1, context=1)
        assert in_region is False  # distance 2 > context 1
        in_region, _ = harness._edited_line_regions(orig, mod, 2, context=1)
        assert in_region is True

    def test_pure_deletion_anchors_region(self, harness):
        orig = "a\nb\nc\n"
        mod = "a\nc\n"
        in_region, _regions = harness._edited_line_regions(orig, mod, 2)
        assert in_region is True

    def test_identical_content_safe_default(self, harness):
        in_region, regions = harness._edited_line_regions("a\n", "a\n", 1)
        assert in_region is True and regions == []


# ── _indentation_hint / _structural_imbalance_hint (static) ─────────────────


class TestIndentationHint:
    def test_unexpected_indent(self, harness):
        content = "def f():\n    pass\n        x = 1\n"
        out = harness._indentation_hint(content, 3, "unexpected indent")
        assert "Reduce this line" in out and "8" in out

    def test_unindent_does_not_match(self, harness):
        content = "def f():\n    if x:\n        pass\n  y = 1\n"
        out = harness._indentation_hint(content, 4, "unindent does not match")
        assert "dedents to" in out

    def test_expected_indented_block(self, harness):
        content = "def f():\nx = 1\n"
        out = harness._indentation_hint(content, 2, "expected an indented block")
        assert "must be indented deeper" in out

    def test_no_hint_for_unknown_message(self, harness):
        content = "a\nb\n"
        assert harness._indentation_hint(content, 1, "something else") == ""

    def test_out_of_range_empty(self, harness):
        assert harness._indentation_hint("a\n", 99, "unexpected indent") == ""


class TestStructuralImbalanceHint:
    def test_missing_except(self, harness):
        out = harness._structural_imbalance_hint("expected 'except' or 'finally'")
        assert "try:" in out and "except" in out

    def test_unclosed_paren(self, harness):
        assert "(" in harness._structural_imbalance_hint("'(' was never closed")

    def test_unclosed_bracket(self, harness):
        assert "[" in harness._structural_imbalance_hint("'[' was never closed")

    def test_unclosed_brace(self, harness):
        assert "{" in harness._structural_imbalance_hint("'{' was never closed")

    def test_unexpected_eof(self, harness):
        assert "EOF" in harness._structural_imbalance_hint(
            "unexpected EOF while parsing"
        ) or "bracket" in harness._structural_imbalance_hint("unexpected EOF while parsing")

    def test_unknown_message_empty(self, harness):
        assert harness._structural_imbalance_hint("just an error") == ""


# ── _apply_scoped_replacement / _apply_one_edit_text ────────────────────────


class TestApplyScopedReplacement:
    def test_in_scope_unique_replaces(self, harness):
        content = "x = 1\ny = 2\nx = 1\n"
        res = harness._apply_scoped_replacement(content, "f.py", "x = 1", "x = 9", (1, 2))
        assert res["ok"]
        assert res["new_content"] == "x = 9\ny = 2\nx = 1\n"
        assert res["occurrences"] == 1

    def test_out_of_scope_fails_with_count(self, harness):
        content = "x = 1\ny = 2\n"
        res = harness._apply_scoped_replacement(content, "f.py", "x = 1", "x = 9", (2, 2))
        assert not res["ok"]
        assert "OUTSIDE the scope" in res["error"]
        assert res["metadata"]["out_of_scope_count"] == 1

    def test_multiple_in_scope_fails(self, harness):
        content = "x = 1\nx = 1\ny = 2\n"
        res = harness._apply_scoped_replacement(content, "f.py", "x = 1", "x = 9", (1, 3))
        assert not res["ok"]
        assert "occurrences" in res["error"]

    def test_empty_old_string_rejected(self, harness):
        res = harness._apply_scoped_replacement("a\n", "f.py", "   ", "x", (1, 1))
        assert not res["ok"]

    def test_whitespace_fallback_scope_splice(self, harness):
        content = "  x = 1\ny = 2\n"
        res = harness._apply_scoped_replacement(content, "f.py", "x = 1", "x = 9", (1, 1))
        assert res["ok"]
        assert res["new_content"] == "  x = 9\ny = 2\n"


class TestApplyOneEditText:
    def test_single_replacement(self, harness):
        res = harness._apply_one_edit_text("a\nb\nc\n", "f.py", "b", "B", False)
        assert res["ok"]
        assert res["new_content"] == "a\nB\nc\n"
        assert res["occurrences"] == 1

    def test_not_found_with_near_match(self, harness):
        res = harness._apply_one_edit_text("foo\nbar\n", "f.py", "baaz", "x", False)
        assert not res["ok"]
        assert res["metadata"]["failure_class"] == "search_string_mismatch"
        assert res["metadata"]["near_match"] is True

    def test_not_found_no_near_match(self, harness):
        res = harness._apply_one_edit_text("aaa\nbbb\n", "f.py", "zzzz", "x", False)
        assert not res["ok"]
        assert res["metadata"]["near_match"] is False

    def test_multiple_occurrences_fails_with_contexts(self, harness):
        res = harness._apply_one_edit_text("b\nb\n", "f.py", "b", "B", False)
        assert not res["ok"]
        assert "2 occurrences" in res["error"]
        assert "match 1" in res["error"]

    def test_replace_all(self, harness):
        res = harness._apply_one_edit_text("b\nb\n", "f.py", "b", "B", True)
        assert res["ok"]
        assert res["new_content"] == "B\nB\n"
        assert res["occurrences"] == 2

    def test_replace_all_high_count_warning(self, harness):
        res = harness._apply_one_edit_text("b\n" * 25, "f.py", "b", "B", True)
        assert res["ok"]
        assert "max recommended" in res["high_count_warning"]

    def test_replace_all_fallback_position_splice(self, harness):
        content = "  x = 1\n  x = 1\n"
        res = harness._apply_one_edit_text(content, "f.py", "x = 1", "x = 9", True)
        assert res["ok"]
        assert res["new_content"] == "  x = 9\n  x = 9\n"

    def test_empty_old_string_rejected(self, harness):
        res = harness._apply_one_edit_text("a\n", "f.py", "  ", "x", False)
        assert not res["ok"]

    def test_scoped_delegation(self, harness):
        res = harness._apply_one_edit_text("x = 1\ny = 2\nx = 1\n", "f.py", "x = 1", "x = 9", False, scope=(1, 2))
        assert res["ok"]
        assert res["new_content"] == "x = 9\ny = 2\nx = 1\n"

    def test_fallback_reindent_applied(self, harness):
        # old_string is absent verbatim (indent drift) → indent-tolerant fallback
        # resolves it; new_string's first-line indent is rebased to the match's.
        content = "def f():\n    body\n"
        res = harness._apply_one_edit_text(content, "f.py", "def f():\nbody", "    def g():\n    pass", False)
        assert res["ok"]
        assert res["new_content"] == "def g():\npass\n"
        assert res["reindent_applied"] is True


# ── _tool_edit_text end-to-end ──────────────────────────────────────────────


class TestToolEditText:
    def test_single_success(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        r = harness._tool_edit_text({"file_path": "t.py", "old_string": "x = 1", "new_string": "x = 10"})
        assert r.ok, r.error
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "x = 10\ny = 2\n"
        assert r.metadata.get("matched_line") == 1

    def test_batch_success(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {
                "file_path": "t.py",
                "edits": [
                    {"old_string": "a = 1", "new_string": "a = 10"},
                    {"old_string": "b = 2", "new_string": "b = 20"},
                ],
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "a = 10\nb = 20\n"

    def test_batch_atomic_failure(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {
                "file_path": "t.py",
                "edits": [
                    {"old_string": "a = 1", "new_string": "a = 10"},
                    {"old_string": "zzz", "new_string": "b = 20"},
                ],
            }
        )
        assert not r.ok
        assert r.metadata.get("failed_edit_index") == 1
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "a = 1\nb = 2\n"  # untouched

    def test_scope_restricts_matching(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {
                "file_path": "t.py",
                "old_string": "x = 1",
                "new_string": "x = 9",
                "scope_start_line": 1,
                "scope_end_line": 2,
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "x = 9\ny = 2\nx = 1\n"

    def test_scope_validation_errors(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
        r = harness._tool_edit_text({"file_path": "t.py", "old_string": "x", "new_string": "y", "scope_start_line": 1})
        assert not r.ok and "together" in r.error
        r = harness._tool_edit_text(
            {"file_path": "t.py", "old_string": "x", "new_string": "y", "scope_start_line": 3, "scope_end_line": 1}
        )
        assert not r.ok and "<=" in r.error
        r = harness._tool_edit_text(
            {
                "file_path": "t.py",
                "old_string": "x",
                "new_string": "y",
                "scope_start_line": 1,
                "scope_end_line": 2,
                "replace_all": True,
            }
        )
        assert not r.ok and "replace_all" in r.error

    def test_mixed_mode_rejected(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {
                "file_path": "t.py",
                "edits": [{"old_string": "x", "new_string": "y"}],
                "old_string": "x",
                "new_string": "y",
            }
        )
        assert not r.ok and "Cannot mix" in r.error

    def test_syntax_gate_refuses_broken_python(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("def f():\n    pass\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {"file_path": "t.py", "old_string": "def f():\n    pass", "new_string": "def f():\npass"}
        )
        assert not r.ok
        assert r.metadata.get("failure_class") == "syntax_invalid_after_edit"
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "def f():\n    pass\n"  # untouched

    def test_syntax_gate_allows_fixing_broken_file(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("def f():\npass\n", encoding="utf-8")
        r = harness._tool_edit_text(
            {"file_path": "t.py", "old_string": "def f():\npass", "new_string": "def f():\n    pass\n"}
        )
        assert r.ok, r.error  # pre-existing broken file → gate opens

    def test_missing_file_error(self, harness, tmp_path):
        r = harness._tool_edit_text({"file_path": "nope.py", "old_string": "x", "new_string": "y"})
        assert not r.ok and "File not found" in r.error

    def test_old_string_required(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x\n", encoding="utf-8")
        r = harness._tool_edit_text({"file_path": "t.py", "new_string": "y"})
        assert not r.ok and "old_string is required" in r.error

    def test_path_blocked_outside_repo(self, harness, tmp_path):
        outside = tmp_path.parent / "outside.py"
        outside.write_text("x = 1\n", encoding="utf-8")
        r = harness._tool_edit_text({"file_path": str(outside), "old_string": "x", "new_string": "y"})
        assert not r.ok and "outside repo" in r.error

    def test_reread_retry_when_file_changed(self, harness, tmp_path):
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        orig_apply = harness._apply_one_edit_text
        calls = {"n": 0}

        def flaky_apply(content, file_path, old, new, replace_all, scope=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # simulate a stale-entry mismatch, then change the file on disk
                # (fresh content still contains the old_string so retry succeeds)
                target.write_text("alpha\nbeta\nextra\n", encoding="utf-8")
                return {
                    "ok": False,
                    "error": "old_string not found",
                    "metadata": {"failure_class": "search_string_mismatch"},
                }
            return orig_apply(content, file_path, old, new, replace_all, scope)

        harness._apply_one_edit_text = flaky_apply
        r = harness._tool_edit_text({"file_path": "t.txt", "old_string": "beta", "new_string": "gamma"})
        assert calls["n"] == 2
        assert (r.metadata or {}).get("reread_retried") is True
        assert (r.metadata or {}).get("reread_retry_success") is True


# ── _tool_edit_file end-to-end ──────────────────────────────────────────────


class TestToolEditFile:
    def test_replace_op(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "replace", "anchor": "b", "content": "B"}]}
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nB\nc\n"

    def test_insert_after(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "insert_after", "anchor": "a", "content": "A1"}]}
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nA1\nb\n"

    def test_insert_after_block_header_moves_to_block_end(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("def f():\n    body\nx = 1\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.py", "operations": [{"type": "insert_after", "anchor": "def f():", "content": "y = 2"}]}
        )
        assert r.ok, r.error
        content = (tmp_path / "t.py").read_text(encoding="utf-8")
        # inserted as a SIBLING after the block, not nested inside the body
        assert "y = 2" in content
        assert content.index("y = 2") > content.index("body")

    def test_insert_after_idempotent_skip(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nX\nb\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "insert_after", "anchor": "a", "content": "X"}]}
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nX\nb\n"  # unchanged

    def test_insert_before(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "insert_before", "anchor": "b", "content": "B0"}]}
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nB0\nb\n"

    def test_unknown_op_type_fails(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "teleport", "anchor": "a", "content": "x"}]}
        )
        assert not r.ok and "unknown type" in r.error

    def test_missing_anchor_fails(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_edit_file({"path": "t.txt", "operations": [{"type": "replace", "content": "x"}]})
        assert not r.ok and "anchor" in r.error

    def test_anchor_not_found(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "replace", "anchor": "zzz", "content": "x"}]}
        )
        assert not r.ok and "anchor text not found" in r.error

    def test_ops_recovered_from_raw_arguments(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        raw = '{"path": "t.txt", "operations": [{"type": "replace", "anchor": "a", "content": "A"}]}'
        r = harness._tool_edit_file({"__raw_arguments": raw})
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "A\nb\n"

    def test_syntax_rollback_on_bad_edit(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("def f():\n    pass\n", encoding="utf-8")
        harness._run_syntax_check_for_file = lambda p: {
            "ok": False,
            "skipped": False,
            "errors": [{"line": 1, "col": 0, "message": "boom"}],
        }
        r = harness._tool_edit_file(
            {"path": "t.py", "operations": [{"type": "replace", "anchor": "def f():", "content": "def g():"}]}
        )
        assert not r.ok
        assert r.metadata.get("rollback_reason") == "syntax_error"
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "def f():\n    pass\n"  # rolled back

    def test_file_not_found(self, harness, tmp_path):
        r = harness._tool_edit_file(
            {"path": "missing.txt", "operations": [{"type": "replace", "anchor": "x", "content": "y"}]}
        )
        assert not r.ok and "File not found" in r.error

    def test_empty_operations_fails(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_edit_file({"path": "t.txt", "operations": []})
        assert not r.ok and "operations list cannot be empty" in r.error

    def test_content_contains_anchor_warning(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("foo\n", encoding="utf-8")
        r = harness._tool_edit_file(
            {"path": "t.txt", "operations": [{"type": "replace", "anchor": "foo", "content": "foo bar"}]}
        )
        assert r.ok
        assert "edit_warnings" in (r.metadata or {})


# ── _tool_create_file end-to-end ────────────────────────────────────────────


class TestToolCreateFile:
    def test_creates_file_with_parents(self, harness, tmp_path):
        r = harness._tool_create_file({"path": "sub/dir/t.txt", "content": "hello"})
        assert r.ok, r.error
        assert (tmp_path / "sub" / "dir" / "t.txt").read_text(encoding="utf-8") == "hello"

    def test_exists_without_overwrite_fails(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("old", encoding="utf-8")
        r = harness._tool_create_file({"path": "t.txt", "content": "new"})
        assert not r.ok and "already exists" in r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "old"

    def test_overwrite_true(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("old", encoding="utf-8")
        r = harness._tool_create_file({"path": "t.txt", "content": "new", "overwrite": True})
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "new"

    def test_python_syntax_gate(self, harness, tmp_path):
        r = harness._tool_create_file({"path": "bad.py", "content": "def broken(:"})
        assert not r.ok and "syntax error" in r.error
        assert not (tmp_path / "bad.py").exists()

    def test_path_required(self, harness):
        r = harness._tool_create_file({"content": "x"})
        assert not r.ok and "path is required" in r.error

    def test_path_blocked_outside_repo(self, harness, tmp_path):
        outside = tmp_path.parent / "evil.txt"
        r = harness._tool_create_file({"path": str(outside), "content": "x"})
        assert not r.ok and "outside repo" in r.error
        assert not outside.exists()

    def test_description_in_content(self, harness, tmp_path):
        r = harness._tool_create_file({"path": "t.txt", "content": "x", "description": "note"})
        assert r.ok and "note" in r.content


# ── _extract_ops_from_raw / _run_syntax_check_for_file / misc ───────────────


class TestExtractOpsFromRaw:
    def test_truncated_json_recovers_ops(self, harness):
        raw = '{"path": "t.txt", "operations": [{"type": "replace", "anchor": "a", "content": "A"}], "extra": '
        ops = harness._extract_ops_from_raw(raw)
        assert ops == [{"type": "replace", "anchor": "a", "content": "A"}]

    def test_no_operations_key_empty(self, harness):
        assert harness._extract_ops_from_raw('{"path": "t.txt"}') == []

    def test_unclosed_array_empty(self, harness):
        assert harness._extract_ops_from_raw('{"operations": [{"type": "x"}') == []

    def test_invalid_json_empty(self, harness):
        assert harness._extract_ops_from_raw('{"operations": [oops]}') == []


class TestRunSyntaxCheckForFile:
    def test_unknown_provider_skipped(self, harness, tmp_path):
        (tmp_path / "t.xyzunknown").write_text("x", encoding="utf-8")
        out = harness._run_syntax_check_for_file("t.xyzunknown")
        assert out.get("skipped") is True

    def test_missing_file_skipped(self, harness):
        out = harness._run_syntax_check_for_file("missing.py")
        assert out.get("skipped") in (True, None) or out.get("ok") is not None


class TestNormRepoRelAndRecord:
    def test_norm_repo_rel_absolute(self, harness, tmp_path):
        assert harness._norm_repo_rel(str(tmp_path / "sub" / "f.py")) == "sub/f.py"

    def test_norm_repo_rel_relative(self, harness):
        assert harness._norm_repo_rel("sub/f.py") == "sub/f.py"
        assert harness._norm_repo_rel("") == ""

    def test_record_text_edit(self, harness):
        harness._record_text_edit("a/b.py")
        assert "a/b.py" in harness._text_edited_files


# ── _tool_modify_symbol end-to-end ──────────────────────────────────────────


class TestToolModifySymbol:
    def test_modify_symbol_success(self, harness, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        r = harness._tool_modify_symbol({"file_path": "m.py", "symbol": "foo", "code": "def foo():\n    return 42\n"})
        assert r.ok, r.error
        assert "return 42" in (tmp_path / "m.py").read_text(encoding="utf-8")
        assert r.metadata.get("symbol_def_line") == 1

    def test_missing_fields(self, harness):
        assert not harness._tool_modify_symbol({}).ok
        assert not harness._tool_modify_symbol({"file_path": "m.py"}).ok
        assert not harness._tool_modify_symbol({"file_path": "m.py", "symbol": "foo"}).ok

    def test_path_traversal_blocked(self, harness, tmp_path):
        outside = tmp_path.parent / "m.py"
        outside.write_text("def foo():\n    pass\n", encoding="utf-8")
        r = harness._tool_modify_symbol(
            {"file_path": str(outside), "symbol": "foo", "code": "def foo():\n    return 1\n"}
        )
        assert not r.ok and "blocked" in r.error

    def test_file_not_found(self, harness):
        r = harness._tool_modify_symbol({"file_path": "missing.py", "symbol": "foo", "code": "def foo():\n    pass\n"})
        assert not r.ok and "File not found" in r.error

    def test_dry_run_restores_file(self, harness, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        r = harness._tool_modify_symbol(
            {"file_path": "m.py", "symbol": "foo", "code": "def foo():\n    return 42\n", "dry_run": True}
        )
        assert r.ok, r.error
        assert "preview" in r.content
        # file restored to pre-edit content
        assert (tmp_path / "m.py").read_text(encoding="utf-8") == "def foo():\n    return 1\n"


# ── _tool_edit_text additional paths ────────────────────────────────────────


class TestToolEditTextExtra:
    def test_reread_retry_not_triggered_when_file_unchanged(self, harness, tmp_path):
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        calls = {"n": 0}

        def fail_once(content, file_path, old, new, replace_all, scope=None):
            calls["n"] += 1
            return {
                "ok": False,
                "error": "old_string not found",
                "metadata": {"failure_class": "search_string_mismatch"},
            }

        harness._apply_one_edit_text = fail_once
        r = harness._tool_edit_text({"file_path": "t.txt", "old_string": "beta", "new_string": "gamma"})
        # file unchanged on disk → no retry, plain failure with fresh-content snippet
        assert calls["n"] == 1
        assert not r.ok
        assert (r.metadata or {}).get("reread_retried") is None
        assert (r.metadata or {}).get("reread_snippet") is True  # head snippet attached

    def test_non_utf8_latin1_roundtrip(self, harness, tmp_path):
        # latin-1 file with non-UTF8 byte: edit_text must read/write with latin-1
        # and preserve untouched bytes exactly (no U+FFFD corruption)
        data = "caf\xe9\n".encode("latin-1")  # café in latin-1
        (tmp_path / "t.txt").write_bytes(data)
        r = harness._tool_edit_text({"file_path": "t.txt", "old_string": "caf\xe9", "new_string": "x"})
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_bytes() == b"x\n"

    def test_unicode_encode_error_reported(self, harness, tmp_path):
        # latin-1 file (é byte forces the latin-1 read path); Chinese chars in
        # new_string are not representable in latin-1 → encode failure before write
        (tmp_path / "t.txt").write_bytes("caf\xe9\n".encode("latin-1"))
        r = harness._tool_edit_text({"file_path": "t.txt", "old_string": "caf\xe9", "new_string": "\u4e2d\u6587"})
        assert not r.ok and "not representable" in r.error

    def test_high_count_warning_metadata(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("b\n" * 25, encoding="utf-8")
        r = harness._tool_edit_text({"file_path": "t.txt", "old_string": "b", "new_string": "B", "replace_all": True})
        assert r.ok, r.error
        assert "high_count_warnings" in (r.metadata or {})


# ── _tool_create_file extra paths ───────────────────────────────────────────


class TestToolCreateFileExtra:
    def test_non_python_language_syntax_gate(self, harness, tmp_path):
        # .ts files have a registered provider; a broken one must be refused
        r = harness._tool_create_file({"path": "bad.ts", "content": "const x: = 1"})
        if "syntax error" in (r.error or ""):
            assert not r.ok
            assert not (tmp_path / "bad.ts").exists()
        else:
            # provider may be unavailable in this env — either way file state is sane
            assert r.ok or not (tmp_path / "bad.ts").exists()

    def test_meta_soft_fail_marker(self, harness, tmp_path):
        # non-python soft-fail path: provider reports a soft-fail error →
        # _should_soft_fail_verify returns False in harness → refused
        from external_llm.languages import LanguageRegistry as LR  # noqa: N817 — test-local shorthand

        prov = LR.instance().get("x.go")
        if prov is None:
            import pytest as _pt

            _pt.skip("go provider not registered")
        r = harness._tool_create_file({"path": "x.go", "content": "package p\nfunc f() {"})
        assert not r.ok


# ── _tool_modify_symbol extra paths ─────────────────────────────────────────


class TestToolModifySymbolExtra:
    def test_failure_result(self, harness, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        r = harness._tool_modify_symbol(
            {"file_path": "m.py", "symbol": "nonexistent_symbol", "code": "def foo():\n    return 42\n"}
        )
        assert not r.ok
        assert "failed" in r.error

    def test_dry_run_preview_only_when_apply_fails(self, harness, tmp_path):
        (tmp_path / "m.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        r = harness._tool_modify_symbol(
            {"file_path": "m.py", "symbol": "nope", "code": "def nope():\n    pass\n", "dry_run": True}
        )
        assert r.ok  # dry run failure → preview_only success
        assert r.metadata.get("preview_only") is True


# ── B2: atomic funnel entry + symlink preservation (edit_text / edit_file) ──


class TestB2AtomicFunnelAndSymlinks:
    def test_edit_text_write_goes_through_atomic_funnel(self, harness, tmp_path, monkeypatch):
        """edit_text's write must land in atomic_write_bytes — not raw
        write_bytes — so the repo file index is invalidated without any
        dispatch-level help (handlers are invoked directly here)."""
        import external_llm.common.repo_files as common_rf

        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
        key = common_rf.canonical_repo_key(str(tmp_path))
        common_rf._FILE_INDEX_CACHE.pop(key, None)

        def fake_listing(root):
            return ["app.py"]

        monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)
        assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
        assert key in common_rf._FILE_INDEX_CACHE

        r = harness._tool_edit_text({"file_path": "app.py", "old_string": "x = 1", "new_string": "x = 2"})
        assert r.ok, r.error
        assert (tmp_path / "app.py").read_text(encoding="utf-8") == "x = 2\n"
        assert key not in common_rf._FILE_INDEX_CACHE, "edit_text must invalidate through the atomic funnel"

    def test_edit_text_encoding_write_is_atomic(self, harness, tmp_path, monkeypatch):
        """The non-UTF-8 (latin-1) write path — the one that used raw
        write_bytes — must also go through the atomic funnel."""
        import external_llm.common.repo_files as common_rf

        # latin-1 content: é = 0xE9
        (tmp_path / "app.py").write_bytes("x = 'café'\n".encode("latin-1"))
        key = common_rf.canonical_repo_key(str(tmp_path))
        common_rf._FILE_INDEX_CACHE.pop(key, None)

        def fake_listing(root):
            return ["app.py"]

        monkeypatch.setattr(common_rf, "git_list_repo_files", fake_listing)
        assert common_rf.cached_repo_file_list(str(tmp_path)) == ["app.py"]
        assert key in common_rf._FILE_INDEX_CACHE

        r = harness._tool_edit_text(
            {
                "file_path": "app.py",
                "old_string": "café",
                "new_string": "café!",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "app.py").read_bytes() == "x = 'café!'\n".encode("latin-1"), (
            "latin-1 bytes must round-trip exactly (no UTF-8 re-encode)"
        )
        assert key not in common_rf._FILE_INDEX_CACHE

    def test_edit_text_preserves_symlink(self, harness, tmp_path):
        """A repo-internal symlink must stay a symlink: the atomic write lands
        on the target through the RESOLVED path, like create_file."""
        (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "link.py").symlink_to("real.py")

        r = harness._tool_edit_text({"file_path": "link.py", "old_string": "x = 1", "new_string": "x = 2"})
        assert r.ok, r.error
        assert (tmp_path / "link.py").is_symlink(), "symlink must not be replaced by a regular file"
        assert (tmp_path / "real.py").read_text(encoding="utf-8") == "x = 2\n"

    def test_edit_file_preserves_symlink(self, harness, tmp_path):
        """Same guarantee for edit_file — both success and rollback writes use
        the resolved path."""
        (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "link.py").symlink_to("real.py")

        r = harness._tool_edit_file(
            {"path": "link.py", "operations": [{"type": "replace", "anchor": "x = 1", "content": "x = 2"}]}
        )
        assert r.ok, r.error
        assert (tmp_path / "link.py").is_symlink(), "symlink must not be replaced by a regular file"
        assert (tmp_path / "real.py").read_text(encoding="utf-8") == "x = 2\n"


# ── real _run_syntax_check_for_file: non-UTF-8 read path ─────────────────────


class _RealSyntaxHarness(WriteToolsEditMixin):
    """Minimal real-mixin host for _run_syntax_check_for_file (no stub).

    The shared ``harness`` fixture stubs ``_run_syntax_check_for_file`` away,
    so the encoding-fallback behaviour of the REAL method is tested here.
    """

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self._repo_root_override = None
        self.defer_semantic_check = lambda *a, **k: False


class TestRealSyntaxCheckForFile:
    def test_cp949_file_gate_runs_not_skipped(self, tmp_path):
        """Non-UTF-8 (cp949) source: the gate must RUN (latin-1 fallback),
        not silently skip with reason="exception" — the historical fail-open
        where UnicodeDecodeError escaped the OSError guard."""
        h = _RealSyntaxHarness(tmp_path)
        (tmp_path / "t.py").write_bytes("# 한국어 주석\nx = 1\n".encode("cp949"))
        out = h._run_syntax_check_for_file("t.py")
        assert out.get("skipped") is not True, out
        assert out.get("language") == "python", out
        assert out.get("ok") is True, out

    def test_missing_file_returns_file_read_error(self, tmp_path):
        h = _RealSyntaxHarness(tmp_path)
        out = h._run_syntax_check_for_file("missing.py")
        assert out == {"ok": True, "skipped": True, "reason": "file_read_error"}, out

    def test_broken_utf8_still_refused(self, tmp_path):
        """The gate must still catch syntax errors after the reader change."""
        h = _RealSyntaxHarness(tmp_path)
        (tmp_path / "t.py").write_text("def f():\npass\n", encoding="utf-8")
        out = h._run_syntax_check_for_file("t.py")
        assert out.get("ok") is False, out
        assert out.get("errors"), out
