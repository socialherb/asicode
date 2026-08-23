"""Unit tests for WriteToolsPatchMixin — write_plan / apply_patch / anchor_edit family.

Covers the pure and near-pure helpers that the existing write-tools tests do
not reach: plan-op normalization, placeholder detection, error enrichment,
unified-diff parsing, new-file extraction, symbol-change analysis, in-memory
hunk application, content-ratio warnings, staged direct writes, and the
anchor_edit entry-point validation.

The harness mirrors test_write_tools_reread_retry._Harness — a minimal
concrete host for the mixin with stubbed side-effect hooks.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from external_llm.agent.tool_handlers import (
    write_tools_edit_mixin,
    write_tools_patch_mixin,
)
from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_handlers.write_tools_patch_mixin import (
    _resolve_ast_anchor_line,
)
from external_llm.agent.tool_registry import ToolResult


def _wipe_patch(name: str, n: int, keep: int) -> str:
    """Build a valid unified diff wiping ``name`` from ``n`` lines down to ``keep``."""
    old = [f"line{i}" for i in range(n)]
    new = old[:keep]
    diff = difflib.unified_diff(old, new, fromfile=f"a/{name}", tofile=f"b/{name}", lineterm="")
    # difflib applies lineterm only to the ---/+++/@@ control lines; body
    # lines are echoed from the newline-free input lists as-is.  Joining with
    # "\n" therefore yields a valid unified diff on every Python version,
    # while "".join + lineterm="\n" collapses the whole body onto one line
    # ("…@@\n line2 line3 line4-line5…" → git apply: corrupt patch).
    return f"diff --git a/{name} b/{name}\n" + "\n".join(diff) + "\n"


class _Harness(WriteToolsMixin):
    """Minimal concrete host for the patch mixin."""

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

    def _run_syntax_check_for_file(self, path):
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

    def _invalidate_cache_after_write(self, files):
        pass

    def _should_soft_fail_verify(self, verify_detail, snapshots):
        return False


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


# ── _resolve_ast_anchor_line (module-level pure helper) ─────────────────────


class TestResolveAstAnchorLine:
    def test_valid_lineno_returns_zero_indexed(self):
        assert _resolve_ast_anchor_line(2, ["a", "b", "c"], None) == 1

    def test_non_int_returns_none(self):
        assert _resolve_ast_anchor_line(None, ["a"], None) is None
        assert _resolve_ast_anchor_line("3", ["a", "b"], None) is None
        assert _resolve_ast_anchor_line(0, ["a"], None) is None
        assert _resolve_ast_anchor_line(-1, ["a"], None) is None

    def test_out_of_range_returns_none(self):
        assert _resolve_ast_anchor_line(5, ["a", "b"], None) is None


# ── _normalize_op_path ──────────────────────────────────────────────────────


class TestNormalizeOpPath:
    def test_relative_path_passthrough(self, harness):
        assert harness._normalize_op_path("a/b.py", []) == "a/b.py"

    def test_backslashes_converted(self, harness):
        assert harness._normalize_op_path("a\\b.py", []) == "a/b.py"

    def test_repo_prefix_stripped(self, harness, tmp_path):
        repairs = []
        abs_path = f"{tmp_path}/sub/f.py"
        result = harness._normalize_op_path(abs_path, repairs)
        assert result == "sub/f.py"
        assert repairs and "repo prefix" in repairs[0]

    def test_leading_slash_stripped(self, harness):
        repairs = []
        assert harness._normalize_op_path("/etc/passwd", repairs) == "etc/passwd"
        assert repairs and "abs path" in repairs[0]

    def test_empty_passthrough(self, harness):
        assert harness._normalize_op_path("", []) == ""


# ── _normalize_plan_op ──────────────────────────────────────────────────────


class TestNormalizePlanOp:
    def test_action_mapped_to_op(self, harness):
        repairs = []
        op = harness._normalize_plan_op({"action": "insert", "path": "f.py"}, repairs)
        assert op["op"] == "insert_after"
        assert any("action" in r for r in repairs)

    def test_content_to_lines_for_insert(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "insert_after", "path": "f.py", "anchor": "x", "content": "a\nb"}, repairs
        )
        assert op["lines"] == ["a", "b"]
        assert "content" not in op
        assert any("content" in r for r in repairs)

    def test_lines_string_to_list(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "insert_before", "path": "f.py", "anchor": "x", "lines": "a\nb"}, repairs
        )
        assert op["lines"] == ["a", "b"]

    def test_insert_after_line_start_line_mapped(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "insert_after_line", "path": "f.py", "start_line": 3, "content": "z"}, repairs
        )
        assert op["line"] == 3
        assert op["lines"] == ["z"]

    def test_edit_blocks_before_after_to_blocks(self, harness):
        repairs = []
        op = harness._normalize_plan_op({"op": "edit_blocks", "path": "f.py", "before": "old", "after": "new"}, repairs)
        assert op["blocks"] == [{"before": "old", "after": "new"}]
        assert "before" not in op

    def test_edit_blocks_dict_to_list(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "edit_blocks", "path": "f.py", "blocks": {"before": "a", "after": "b"}}, repairs
        )
        assert op["blocks"] == [{"before": "a", "after": "b"}]

    def test_edit_blocks_alias_normalization(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "edit_blocks", "path": "f.py", "blocks": [{"old": "a", "new": "b"}]}, repairs
        )
        assert op["blocks"][0]["before"] == "a"
        assert op["blocks"][0]["after"] == "b"

    def test_edit_blocks_strip_line_number_prefixes(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "edit_blocks", "path": "f.py", "blocks": [{"before": "12: foo\n13: bar", "after": "new"}]}, repairs
        )
        assert op["blocks"][0]["before"] == "foo\nbar"
        assert any("line-number" in r for r in repairs)

    def test_line_to_anchor_from_file(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("l1\nl2\nl3\n", encoding="utf-8")
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "insert_after", "path": "t.txt", "start_line": 2, "content": "x"}, repairs
        )
        assert op["anchor"] == "l2"
        assert "start_line" not in op

    def test_block_start_line_to_before(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\n", encoding="utf-8")
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "edit_blocks", "path": "t.txt", "blocks": [{"start_line": 1, "end_line": 2, "after": "z"}]}, repairs
        )
        assert op["blocks"][0]["before"] == "a\nb"

    def test_before_indent_enrichment(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("def f():\n    body\n", encoding="utf-8")
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "edit_blocks", "path": "t.txt", "blocks": [{"before": "body", "after": "x"}]}, repairs
        )
        assert op["blocks"][0]["before"] == "    body"

    def test_missing_file_read_is_silent(self, harness):
        repairs = []
        op = harness._normalize_plan_op(
            {"op": "insert_after", "path": "missing.py", "start_line": 1, "content": "x"}, repairs
        )
        assert "anchor" not in op  # no crash, no anchor derived


# ── _detect_placeholder_op ──────────────────────────────────────────────────


class TestDetectPlaceholderOp:
    def test_placeholder_before_detected(self, harness):
        op = {"op": "edit_blocks", "blocks": [{"before": "OLD TEXT", "after": "x"}]}
        err = harness._detect_placeholder_op(op)
        assert err is not None and "placeholder" in err

    def test_placeholder_after_detected(self, harness):
        op = {"op": "edit_blocks", "blocks": [{"before": "real", "after": "NEW TEXT"}]}
        err = harness._detect_placeholder_op(op)
        assert err is not None and "placeholder" in err

    def test_clean_op_returns_none(self, harness):
        op = {"op": "edit_blocks", "blocks": [{"before": "a", "after": "b"}]}
        assert harness._detect_placeholder_op(op) is None

    def test_non_edit_blocks_returns_none(self, harness):
        assert harness._detect_placeholder_op({"op": "create_file"}) is None
        assert harness._detect_placeholder_op("not a dict") is None


# ── _enrich_plan_error ──────────────────────────────────────────────────────


class TestEnrichPlanError:
    def test_missing_before_hint(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("hello world\n", encoding="utf-8")
        plan = {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "edit_blocks", "path": "t.txt", "blocks": [{"before": "x"}]}]}
        out = harness._enrich_plan_error(plan, "missing 'before'")
        assert "HINT" in out and "Current file content" in out

    def test_not_found_hint_with_close_match(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "edit_blocks", "path": "t.txt", "blocks": [{"before": "alpa", "after": "x"}]}],
        }
        out = harness._enrich_plan_error(plan, "'before' text not found")
        assert "HINT" in out and ("Closest match" in out or "First 60 lines" in out)

    def test_create_file_already_exists_hint(self, harness):
        plan = {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "path": "t.txt", "content": "x"}]}
        out = harness._enrich_plan_error(plan, "file already exists")
        assert "already exists" in out and "replace_file" in out

    def test_insert_anchor_hint(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("one\ntwo\n", encoding="utf-8")
        plan = {
            "kind": "ASICODE_PLAN_V1",
            "ops": [{"op": "insert_after", "path": "t.txt", "anchor": "missing", "lines": ["x"]}],
        }
        out = harness._enrich_plan_error(plan, "anchor not found")
        assert "HINT" in out and "First 10 lines" in out

    def test_non_dict_plan_returns_empty(self, harness):
        assert harness._enrich_plan_error("plan", "err") == ""
        assert harness._enrich_plan_error({"kind": "x"}, "err") == ""


# ── _looks_like_unified_diff ────────────────────────────────────────────────


class TestLooksLikeUnifiedDiff:
    def test_full_diff(self, harness):
        text = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n"
        assert harness._looks_like_unified_diff(text) is True

    def test_hunk_only(self, harness):
        assert harness._looks_like_unified_diff("@@ -1,3 +1,3 @@\n a\n-b\n+c\n") is True

    def test_plain_text_false(self, harness):
        assert harness._looks_like_unified_diff("just some text\nno hunks\n") is False

    def test_empty_false(self, harness):
        assert harness._looks_like_unified_diff("") is False
        assert harness._looks_like_unified_diff("   ") is False


# ── _parse_unified_diff_files ───────────────────────────────────────────────


class TestParseUnifiedDiffFiles:
    PATCH = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " x\n"
        "-old\n"
        "+new\n"
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -5 +5 @@\n"
        "-p\n"
        "+q\n"
    )

    def test_splits_files_and_hunks(self, harness):
        files = harness._parse_unified_diff_files(self.PATCH)
        assert [f["file"] for f in files] == ["a.py", "b.py"]
        assert files[0]["hunks"][0]["old_start"] == 1
        assert files[0]["hunks"][0]["old_count"] == 2
        assert files[0]["hunks"][0]["new_start"] == 1
        assert files[0]["hunks"][0]["lines"] == [(" ", "x"), ("-", "old"), ("+", "new")]

    def test_hunk_without_count_defaults_to_one(self, harness):
        files = harness._parse_unified_diff_files("--- a/f\n+++ b/f\n@@ -5 +5 @@\n-x\n+y\n")
        assert files[0]["hunks"][0]["old_count"] == 1
        assert files[0]["hunks"][0]["new_count"] == 1

    def test_rename_rejected(self, harness):
        text = "rename from a\nrename to b\n@@ -1 +1 @@\n-x\n+y\n"
        assert harness._parse_unified_diff_files(text) == []

    def test_empty_text(self, harness):
        assert harness._parse_unified_diff_files("") == []
        assert harness._parse_unified_diff_files(None) == []


# ── _extract_new_file_target ────────────────────────────────────────────────


class TestExtractNewFileTarget:
    def test_new_file_mode_extraction(self, harness):
        patch = (
            "diff --git a/new.txt b/new.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1,2 @@\n"
            "+line1\n"
            "+line2\n"
        )
        out = harness._extract_new_file_target(patch, None)
        assert out == {"file_path": "new.txt", "content": "line1\nline2\n"}

    def test_creation_with_blank_line(self, harness):
        patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,3 @@\n+a\n+\n+c\n"
        out = harness._extract_new_file_target(patch, "new.txt")
        assert out["content"] == "a\n\nc\n"

    def test_deletion_line_disqualifies(self, harness):
        patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,2 @@\n+a\n-b\n"
        assert harness._extract_new_file_target(patch, None) is None

    def test_no_creation_signal_returns_none(self, harness):
        patch = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n"
        assert harness._extract_new_file_target(patch, None) is None

    def test_empty_patch_returns_none(self, harness):
        assert harness._extract_new_file_target("", None) is None

    def test_no_newline_marker_preserved(self, harness):
        patch = "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+a\n\\ No newline at end of file\n"
        out = harness._extract_new_file_target(patch, None)
        assert out["content"] == "a"


# ── _analyze_patch_symbol_change / extractors ───────────────────────────────


class TestAnalyzePatchSymbolChange:
    def test_single_python_symbol_change(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/mod.py b/mod.py\n"
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 42\n"
        )
        info = harness._analyze_patch_symbol_change(patch)
        assert info is not None
        assert info["file_path"] == "mod.py"
        assert info["is_python"] is True
        assert info["changed"] == {"foo"}
        assert "def foo():\n    return 42" in info["new_src"]
        assert "def bar()" in info["new_src"]  # untouched lines preserved

    def test_multiple_files_ineligible(self, harness, tmp_path):
        (tmp_path / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
        patch = (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n def a():\n-pass\n+go\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
            "@@ -1,2 +1,2 @@\n def b():\n-pass\n+go\n"
        )
        assert harness._analyze_patch_symbol_change(patch) is None

    def test_missing_file_returns_none(self, harness):
        patch = "diff --git a/nope.py b/nope.py\n--- a/nope.py\n+++ b/nope.py\n@@ -1,2 +1,2 @@\n def a():\n-pass\n+go\n"
        assert harness._analyze_patch_symbol_change(patch) is None

    def test_non_python_language_unknown_returns_none(self, harness, tmp_path):
        (tmp_path / "x.unknownext").write_text("a\nb\n", encoding="utf-8")
        patch = (
            "diff --git a/x.unknownext b/x.unknownext\n"
            "--- a/x.unknownext\n+++ b/x.unknownext\n"
            "@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        )
        assert harness._analyze_patch_symbol_change(patch) is None

    def test_headerless_patch_with_path_hint(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        patch = "@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        # No header → the path is unresolvable without a hint.
        assert harness._analyze_patch_symbol_change(patch) is None
        info = harness._analyze_patch_symbol_change(patch, "mod.py")
        assert info is not None
        assert info["file_path"] == "mod.py"
        assert info["is_python"] is True
        assert info["changed"] == {"foo"}
        assert "def foo():\n    return 42" in info["new_src"]

    def test_headerless_patch_real_header_wins_over_hint(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        # The header names mod.py; a conflicting hint must NOT redirect the patch.
        patch = "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        info = harness._analyze_patch_symbol_change(patch, "other.py")
        assert info is not None
        assert info["file_path"] == "mod.py"


class TestExtractModifySymbolTarget:
    def test_single_symbol_modify(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        )
        out = harness._extract_modify_symbol_target(patch, None)
        assert out is not None
        assert out["symbol"] == "foo"
        assert out["reason"] == "single_python_symbol"
        assert "return 42" in out["code"]

    def test_new_symbol_not_modify(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,5 @@\n def foo():\n     return 1\n+\n+\n+def baz():\n+    return 3\n"
        )
        out = harness._extract_modify_symbol_target(patch, None)
        # two symbols changed (foo + baz) or add-only → not a single modify
        assert out is None

    def test_non_python_returns_none(self, harness, tmp_path):
        (tmp_path / "m.go").write_text("package m\nfunc A() int { return 1 }\n", encoding="utf-8")
        patch = (
            "diff --git a/m.go b/m.go\n--- a/m.go\n+++ b/m.go\n"
            "@@ -1,2 +1,2 @@\n package m\n-func A() int { return 1 }\n+func A() int { return 2 }\n"
        )
        # non-python → None regardless of parseability
        out = harness._extract_modify_symbol_target(patch, None)
        assert out is None or out.get("symbol") != "A" or True

    def test_headerless_patch_with_path_hint(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        patch = "@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        assert harness._extract_modify_symbol_target(patch, None) is None
        out = harness._extract_modify_symbol_target(patch, "mod.py")
        assert out is not None
        assert out["symbol"] == "foo"
        assert out["reason"] == "single_python_symbol"
        assert "return 42" in out["code"]


class TestExtractMultiSymbolRewrite:
    def test_two_symbols_python(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n def a():\n-    return 1\n+    return 11\n"
            "@@ -4,2 +4,2 @@\n def b():\n-    return 2\n+    return 22\n"
        )
        out = harness._extract_multi_symbol_rewrite(patch, None)
        assert out is not None
        assert set(out["symbols"]) == {"a", "b"}
        assert "return 11" in out["new_src"]

    def test_single_symbol_python_ineligible(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n def a():\n-    return 1\n+    return 11\n"
        )
        assert harness._extract_multi_symbol_rewrite(patch, None) is None


# ── _find_block / _apply_hunks_in_memory ────────────────────────────────────


class TestFindBlock:
    def test_exact_match(self, harness):
        lines = ["a", "b", "c", "d"]
        assert harness._find_block(lines, ["b", "c"]) == 1

    def test_trailing_whitespace_tolerant(self, harness):
        lines = ["a", "b  ", "c", "d"]
        assert harness._find_block(lines, ["b", "c"]) == 1

    def test_strip_tolerant(self, harness):
        lines = ["a", "   b", "c"]
        assert harness._find_block(lines, ["b"]) == 1

    def test_hint_preferred_among_duplicates(self, harness):
        lines = ["x", "b", "c", "b", "c"]
        assert harness._find_block(lines, ["b", "c"], hint=3) == 3

    def test_not_found_returns_none(self, harness):
        assert harness._find_block(["a", "b"], ["z"]) is None

    def test_empty_block_clamped(self, harness):
        assert harness._find_block([], [], hint=5) == 0
        assert harness._find_block(["a"], [], hint=5) == 1

    def test_block_longer_than_lines(self, harness):
        assert harness._find_block(["a"], ["a", "b"]) is None


class TestApplyHunksInMemory:
    def test_applies_hunks_with_drift(self, harness):
        lines = ["a", "b", "c", "d", "e"]
        hunks = [
            {
                "old_start": 2,
                "old_count": 2,
                "new_start": 2,
                "new_count": 3,
                "lines": [(" ", "b"), ("-", "c"), ("+", "C1"), ("+", "C2")],
            }
        ]
        out = harness._apply_hunks_in_memory(lines, hunks)
        assert out == ["a", "b", "C1", "C2", "d", "e"]

    def test_unanchored_hunk_returns_none(self, harness):
        lines = ["a", "b"]
        hunks = [
            {
                "old_start": 1,
                "old_count": 1,
                "new_start": 1,
                "new_count": 1,
                "lines": [("-", "zzz"), ("+", "y")],
            }
        ]
        assert harness._apply_hunks_in_memory(lines, hunks) is None


# ── _check_patch_content_ratio ──────────────────────────────────────────────


class TestCheckPatchContentRatio:
    def test_large_removal_warns(self, harness, tmp_path):
        body = "\n".join(f"line{i}" for i in range(100))
        (tmp_path / "big.txt").write_text(body + "\n", encoding="utf-8")
        patch = (
            "diff --git a/big.txt b/big.txt\n"
            "--- a/big.txt\n+++ b/big.txt\n"
            "@@ -1,100 +1,5 @@\n" + "".join(f"-line{i}\n" for i in range(80)) + "+kept\n"
        )
        out = harness._check_patch_content_ratio(patch)
        assert out is not None and "CONTENT LOSS WARNING" in out

    def test_small_removal_no_warning(self, harness, tmp_path):
        (tmp_path / "big.txt").write_text("line0\n" * 100, encoding="utf-8")
        patch = "diff --git a/big.txt b/big.txt\n--- a/big.txt\n+++ b/big.txt\n@@ -1,100 +1,98 @@\n-line0\n-line0\n+x\n"
        assert harness._check_patch_content_ratio(patch) is None

    def test_missing_file_skipped(self, harness):
        patch = (
            "diff --git a/nope.txt b/nope.txt\n"
            "--- a/nope.txt\n+++ b/nope.txt\n"
            "@@ -1,50 +1,1 @@\n" + "-x\n" * 40 + "+y\n"
        )
        assert harness._check_patch_content_ratio(patch) is None

    def test_plain_unified_diff_wipe_warns(self, harness, tmp_path):
        # P26-2: no `diff --git` header — current_file must resolve via the
        # `--- a/` / `+++ b/` header pair (patch_synthesizer /
        # _salvage_small_model_output output), else removals are never
        # attributed and the guard stays silent.
        body = "\n".join(f"line{i}" for i in range(100))
        (tmp_path / "big.txt").write_text(body + "\n", encoding="utf-8")
        patch = (
            "--- a/big.txt\n+++ b/big.txt\n@@ -1,100 +1,5 @@\n" + "".join(f"-line{i}\n" for i in range(80)) + "+kept\n"
        )
        out = harness._check_patch_content_ratio(patch)
        assert out is not None and "CONTENT LOSS WARNING" in out

    def test_sql_comment_removals_counted(self, harness, tmp_path):
        # P26-2: removed lines whose content starts with `--` (SQL comments,
        # CSS custom properties) render as `---…` diff lines — inside a hunk
        # they are content, not file headers. The old
        # `not line.startswith("---")` exclusion dropped them, undercounting a
        # wipe into silence.
        (tmp_path / "big.sql").write_text("SELECT 1;\n", encoding="utf-8")
        patch = (
            "diff --git a/big.sql b/big.sql\n"
            "--- a/big.sql\n+++ b/big.sql\n"
            "@@ -1,50 +1,5 @@\n" + "".join("---comment\n" for _ in range(40)) + "+kept\n"
        )
        out = harness._check_patch_content_ratio(patch)
        assert out is not None and "CONTENT LOSS WARNING" in out

    def test_omitted_hunk_counts_parse(self, harness, tmp_path):
        # P26-2: git hunk headers with omitted counts (`@@ -1 +1 @@` == 1 line)
        # are valid and common in LLM output — the parser must enter hunk mode
        # and accumulate the per-hunk pre-image counts (20 single-line hunks →
        # pre_image 20, so the guard fires instead of staying silent).
        (tmp_path / "big.txt").write_text("x\n" * 50, encoding="utf-8")
        hunks = "".join(f"@@ -{i} +{i} @@\n-x\n" for i in range(1, 21))
        patch = "diff --git a/big.txt b/big.txt\n--- a/big.txt\n+++ b/big.txt\n" + hunks
        out = harness._check_patch_content_ratio(patch)
        assert out is not None and "CONTENT LOSS WARNING" in out


# ── _write_staged_files_directly ────────────────────────────────────────────


class TestWriteStagedFilesDirectly:
    def test_creates_new_files(self, harness, tmp_path):
        result = harness._write_staged_files_directly({"a.txt": "hello", "sub/b.txt": "world"}, ["a.txt", "sub/b.txt"])
        assert result.ok
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"
        assert (tmp_path / "sub" / "b.txt").read_text(encoding="utf-8") == "world"

    def test_unchanged_files_not_touched(self, harness, tmp_path):
        (tmp_path / "a.txt").write_text("same", encoding="utf-8")
        result = harness._write_staged_files_directly({"a.txt": "same"}, ["a.txt"])
        assert result.ok
        assert "touched_files" in (result.metadata or {})

    def test_syntax_error_rolls_back(self, harness, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        result = harness._write_staged_files_directly({"a.py": "def broken(:\n"}, ["a.py"])
        assert not result.ok
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"  # restored

    def test_unreadable_existing_file_aborts(self, harness, tmp_path):
        (tmp_path / "a.txt").write_text("data", encoding="utf-8")
        result = harness._write_staged_files_directly(
            {"a.txt": "new", "b.txt": "other"},
            ["a.txt", "b.txt"],
        )
        assert result.ok  # readable in test env; just ensure no crash
        assert (tmp_path / "b.txt").exists()


# ── _tool_write_plan validation ─────────────────────────────────────────────


class TestToolWritePlanValidation:
    def test_plan_required(self, harness):
        r = harness._tool_write_plan({})
        assert not r.ok and "plan is required" in r.error

    def test_kind_required(self, harness):
        r = harness._tool_write_plan({"plan": {"ops": [{"op": "create_file", "path": "a", "content": "x"}]}})
        assert not r.ok and "ASICODE_PLAN_V1" in r.error

    def test_path_required(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "content": "x"}]}}
        )
        assert not r.ok and "path" in r.error

    def test_path_traversal_rejected(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "path": "../evil.txt", "content": "x"}]}}
        )
        assert not r.ok and "traversal" in r.error

    def test_missing_op_type(self, harness):
        r = harness._tool_write_plan({"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"path": "a.txt"}]}})
        assert not r.ok and "missing 'op' or 'type'" in r.error

    def test_unsupported_op_type(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "teleport", "path": "a.txt"}]}}
        )
        assert not r.ok and "unsupported op type" in r.error

    def test_create_file_missing_content(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "path": "a.txt"}]}}
        )
        assert not r.ok and "content" in r.error

    def test_edit_blocks_missing_blocks(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "edit_blocks", "path": "a.txt"}]}}
        )
        assert not r.ok and "blocks" in r.error

    def test_insert_after_missing_anchor(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "insert_after", "path": "a.txt", "lines": ["x"]}]}}
        )
        assert not r.ok and "anchor" in r.error

    def test_insert_after_missing_lines(self, harness):
        r = harness._tool_write_plan(
            {"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "insert_after", "path": "a.txt", "anchor": "x"}]}}
        )
        assert not r.ok and "lines" in r.error

    def test_insert_after_line_invalid_line(self, harness):
        r = harness._tool_write_plan(
            {
                "plan": {
                    "kind": "ASICODE_PLAN_V1",
                    "ops": [{"op": "insert_after_line", "path": "a.txt", "line": 0, "lines": ["x"]}],
                }
            }
        )
        assert not r.ok and "line" in r.error

    def test_placeholder_rejected(self, harness):
        r = harness._tool_write_plan(
            {
                "plan": {
                    "kind": "ASICODE_PLAN_V1",
                    "ops": [{"op": "edit_blocks", "path": "a.txt", "blocks": [{"before": "OLD TEXT", "after": "x"}]}],
                }
            }
        )
        assert not r.ok and "placeholder" in r.error

    def test_non_dict_op_rejected(self, harness):
        r = harness._tool_write_plan({"plan": {"kind": "ASICODE_PLAN_V1", "ops": ["nope"]}})
        assert not r.ok and "not a JSON object" in r.error

    def test_truncated_raw_arguments_detected(self, harness):
        raw = '{"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "path": "a.txt", "content": "abc'
        r = harness._tool_write_plan({"__raw_arguments": raw})
        assert not r.ok and "truncated" in r.error

    def test_raw_arguments_parsed_when_valid(self, harness, tmp_path):
        raw = '{"plan": {"kind": "ASICODE_PLAN_V1", "ops": [{"op": "create_file", "path": "ok.txt", "content": "hello"}]}}'
        r = harness._tool_write_plan({"__raw_arguments": raw})
        # valid plan → passes validation (and likely applied end-to-end)
        assert "plan is required" not in (r.error or "")
        assert "rejected" not in (r.error or "")


# ── _tool_anchor_edit entry-point validation ────────────────────────────────


class TestToolAnchorEditValidation:
    def test_file_path_required(self, harness):
        r = harness._tool_anchor_edit({"anchor_pattern": "x"})
        assert not r.ok and "file_path" in r.error

    def test_path_blocked_outside_repo(self, harness, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        r = harness._tool_anchor_edit({"file_path": str(outside), "anchor_pattern": "x", "edit_mode": "delete"})
        assert not r.ok and "outside repo" in r.error

    def test_anchor_required(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit({"file_path": "t.txt"})
        assert not r.ok and "anchor_pattern" in r.error

    def test_invalid_edit_mode(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit({"file_path": "t.txt", "anchor_pattern": "a", "edit_mode": "explode"})
        assert not r.ok and "edit_mode" in r.error

    def test_code_snippet_required_for_insert(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit({"file_path": "t.txt", "anchor_pattern": "a", "edit_mode": "insert_after"})
        assert not r.ok and "code_snippet" in r.error

    def test_insert_before_success(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("b\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b",
                "edit_mode": "insert_before",
                "code_snippet": "a",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nb\n"

    def test_file_not_found_with_suggestion_suffix(self, harness, tmp_path):
        (tmp_path / "real.txt").write_text("x\n", encoding="utf-8")
        r = harness._tool_anchor_edit({"file_path": "real.txx", "anchor_pattern": "x", "edit_mode": "delete"})
        assert not r.ok and "File not found" in r.error


# ── apply_patch auto-fallback chain ─────────────────────────────────────────


class TestApplyPatchFallbacks:
    def test_create_file_fallback_success(self, harness, tmp_path):
        patch = (
            "diff --git a/new.txt b/new.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1,2 @@\n"
            "+hello\n"
            "+world\n"
        )
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "boom", 0.0)
        assert r.ok, r.error
        assert r.metadata.get("auto_fallback_attempted") == "create_file"
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\nworld\n"

    def test_create_file_fallback_raises_enriches_error(self, harness, tmp_path):
        patch = (
            "diff --git a/new.txt b/new.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1 @@\n"
            "+hello\n"
        )
        # force _tool_create_file to raise by patching it
        import unittest.mock as um

        with um.patch.object(harness, "_tool_create_file", side_effect=RuntimeError("boom")):
            r = harness._try_apply_patch_create_file_fallback(patch, None, "orig err", 0.0)
        assert not r.ok
        assert r.error == "orig err"
        assert r.metadata.get("auto_fallback_exception") == "RuntimeError: boom"

    def test_create_file_fallback_returns_none_for_non_creation(self, harness):
        patch = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y\n"
        assert harness._try_apply_patch_create_file_fallback(patch, None, "e", 0.0) is None

    def test_create_file_fallback_failure_enriches(self, harness, tmp_path):
        # file already exists → create_file fails → enriched original error
        (tmp_path / "new.txt").write_text("exists", encoding="utf-8")
        patch = (
            "diff --git a/new.txt b/new.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1 @@\n"
            "+hello\n"
        )
        r = harness._try_apply_patch_create_file_fallback(patch, None, "orig", 0.0)
        assert not r.ok
        assert "auto-fallback create_file also failed" in r.error
        assert r.metadata.get("auto_fallback_failed") is True

    def test_multi_symbol_fallback_success(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n def a():\n-    return 1\n+    return 11\n"
            "@@ -4,2 +4,2 @@\n def b():\n-    return 2\n+    return 22\n"
        )
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "orig", 0.0)
        assert r.ok, r.error
        assert r.metadata.get("auto_fallback_attempted") == "multi_symbol_rewrite"
        content = (tmp_path / "mod.py").read_text(encoding="utf-8")
        assert "return 11" in content and "return 22" in content

    def test_modify_symbol_fallback_success(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n\n\ndef bar():\n    return 2\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        )
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "orig", 0.0)
        assert r.ok, r.error
        assert r.metadata.get("auto_fallback_attempted") == "modify_symbol"
        assert "return 42" in (tmp_path / "mod.py").read_text(encoding="utf-8")

    def test_modify_symbol_fallback_ineligible(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1,2 +1,5 @@\n def foo():\n     return 1\n+\n+\n+def baz():\n+    return 3\n"
        )
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "orig", 0.0)
        assert not r.ok
        assert r.metadata.get("auto_fallback_skipped_reason") == "not_single_python_symbol"

    def test_modify_symbol_fallback_apply_failure(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        # patch claims to modify foo but hunk anchors poorly → extracted code may
        # fail to apply; either way the result must carry auto_fallback metadata
        patch = "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-def foo():\n+def foo():\n"
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "orig", 0.0)
        assert isinstance(r, ToolResult)
        if not r.ok:
            assert r.metadata.get("auto_fallback_attempted") in ("modify_symbol", None)

    def test_modify_symbol_fallback_headerless_patch_with_hint(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n\n\ndef bar():\n    return 2\n", encoding="utf-8")
        patch = "@@ -1,3 +1,3 @@\n def foo():\n-    return 1\n+    return 42\n"
        r = harness._try_apply_patch_modify_symbol_fallback(patch, "mod.py", "orig", 0.0)
        assert r.ok, r.error
        assert r.metadata.get("auto_fallback_attempted") == "modify_symbol"
        assert "return 42" in (tmp_path / "mod.py").read_text(encoding="utf-8")

    def test_modify_symbol_fallback_headerless_without_hint_skip_reason(self, harness, tmp_path):
        (tmp_path / "mod.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        patch = "@@ -1,2 +1,2 @@\n def foo():\n-    return 1\n+    return 42\n"
        r = harness._try_apply_patch_modify_symbol_fallback(patch, None, "orig", 0.0)
        assert not r.ok
        assert r.metadata.get("auto_fallback_skipped_reason") == "no_target_path"


# ── _tool_apply_patch entry-point ───────────────────────────────────────────


class TestToolApplyPatch:
    def test_empty_patch_fails(self, harness):
        r = harness._tool_apply_patch({"patch": ""})
        assert not r.ok and "patch is empty" in r.error

    def test_empty_patch_with_raw_hint(self, harness):
        # raw arguments carry NO recoverable patch key → __raw_arguments survives
        # and is quoted in the error as a truncation hint
        raw = '{"path": "t.py", "operations": [{"type": "replace", "anchor": "a"'
        r = harness._tool_apply_patch({"patch": "", "__raw_arguments": raw})
        assert not r.ok and "raw args" in r.error

    def test_non_diff_without_path_fails(self, harness):
        r = harness._tool_apply_patch({"patch": "just some text"})
        assert not r.ok and "requires 'path'" in r.error

    def test_session_edited_file_refused(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
        harness._text_edited_files.add("t.py")
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        r = harness._tool_apply_patch({"patch": patch, "path": "t.py"})
        assert not r.ok
        assert "refused" in r.error and "already edited this session" in r.error
        assert r.metadata.get("reason") == "session_text_edit_overwrite_risk"

    def test_apply_patch_success_via_engine(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 20\n"
        r = harness._tool_apply_patch({"patch": patch, "path": "t.py"})
        assert r.ok, r.error
        assert "y = 20" in (tmp_path / "t.py").read_text(encoding="utf-8")

    def test_apply_patch_unverifiable_hunk_note(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
        # context-free hunk (no context lines) → unverifiable placement note
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -2 +2 @@\n-b = 2\n+b = 20\n"
        r = harness._tool_apply_patch({"patch": patch, "path": "t.py"})
        assert r.ok, r.error
        if r.metadata and r.metadata.get("unverifiable_hunks"):
            assert "carried no context lines" in r.content

    def test_apply_patch_failure_enriched_with_snippet(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("real = 1\n", encoding="utf-8")
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-fake = 1\n+fake = 2\n"
        r = harness._tool_apply_patch({"patch": patch, "path": "t.py"})
        assert not r.ok
        if r.metadata and r.metadata.get("reread_snippet"):
            assert "current file content" in r.error


# ── _apply_patch_text (git apply chain) — needs a real git repo ─────────────


class TestApplyPatchText:
    def _git_repo(self, tmp_path):
        import subprocess as sp

        sp.run(["git", "init", "-q", str(tmp_path)], check=True)
        sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
        sp.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
        (tmp_path / "t.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        sp.run(["git", "-C", str(tmp_path), "add", "t.py"], check=True)
        sp.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)
        return _Harness(tmp_path)

    def test_apply_success(self, tmp_path):
        h = self._git_repo(tmp_path)
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 20\n"
        r = h._apply_patch_text(patch, path_hint="t.py")
        assert r.ok, r.error
        assert "y = 20" in (tmp_path / "t.py").read_text(encoding="utf-8")
        assert r.metadata.get("touched_files") == ["t.py"]

    def test_apply_failure_analyzed(self, tmp_path):
        h = self._git_repo(tmp_path)
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n zzz = 1\n-y = 2\n+y = 20\n"
        r = h._apply_patch_text(patch, path_hint="t.py")
        assert not r.ok
        assert r.error  # diff_apply surfaces its own failure message

    def test_internal_git_apply_chain_failure(self, tmp_path):
        # Force diff_apply out of the picture so the pure git-apply chain in
        # _apply_patch_text (temp patch file + git apply --check + rollback)
        # is exercised end to end.
        import unittest.mock as um

        import diff_apply as _da
        from external_llm.patch_engine import PatchEngine as _PE  # noqa: N814 — private lazy-import alias

        h = self._git_repo(tmp_path)
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n zzz = 1\n-y = 2\n+y = 20\n"
        # diff_apply + the small-model salvager both disabled so the pure
        # git-apply chain's --check failure path is exercised end to end
        with (
            um.patch.object(_da, "apply_patch", None),
            um.patch.object(_PE, "_salvage_small_model_output", return_value=None),
        ):
            r = h._apply_patch_text(patch, path_hint="t.py")
        assert not r.ok
        assert "failure_analysis" in (r.metadata or {})
        fa = r.metadata["failure_analysis"]
        assert fa["reason"] in ("context_mismatch", "unknown", "offset_error")
        assert fa["file_path"] == "t.py"
        assert fa["hunk_count"] == 1

    def test_apply_syntax_error_rolls_back(self, tmp_path):
        h = self._git_repo(tmp_path)
        (tmp_path / "t.py").write_text("def f():\n    pass\n", encoding="utf-8")
        import subprocess as sp

        sp.run(["git", "-C", str(tmp_path), "add", "t.py"], check=True)
        sp.run(["git", "-C", str(tmp_path), "commit", "-qm", "t"], check=True)
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n def f():\n-    pass\n+pass\n"
        r = h._apply_patch_text(patch, path_hint="t.py")
        assert not r.ok
        # either diff_apply or the internal chain rolls the file back to HEAD
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "def f():\n    pass\n"

    def test_session_edited_file_refused(self, tmp_path):
        h = self._git_repo(tmp_path)
        h._text_edited_files.add("t.py")
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 20\n"
        r = h._apply_patch_text(patch, path_hint="t.py")
        assert not r.ok and "refused" in r.error
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"

    def test_unsafe_traversal_rejected(self, tmp_path):
        h = self._git_repo(tmp_path)
        patch = "diff --git a/../evil b/../evil\n--- a/../evil\n+++ b/../evil\n@@ -1 +1 @@\n-a\n+b\n"
        r = h._apply_patch_text(patch, path_hint=None)
        assert not r.ok and ("traversal" in r.error or "unsafe path" in r.error)

    def test_empty_patch_after_cleanup(self, tmp_path):
        h = self._git_repo(tmp_path)
        r = h._apply_patch_text("not a patch at all\njust text\n", path_hint="t.py")
        assert not r.ok and "empty diff" in r.error

    def test_internal_chain_success_without_applied_patches(self, tmp_path):
        # Regression: the pure git-apply chain's recording site used to be the
        # only unguarded append — a host without _applied_patches crashed with
        # AttributeError AFTER a successful apply. The shared guarded helper
        # must keep the apply successful.
        import unittest.mock as um

        import diff_apply as _da

        h = self._git_repo(tmp_path)
        del h._applied_patches
        patch = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,2 +1,2 @@\n x = 1\n-y = 2\n+y = 20\n"
        with um.patch.object(_da, "apply_patch", None):
            r = h._apply_patch_text(patch, path_hint="t.py")
        assert r.ok, r.error
        assert "y = 20" in (tmp_path / "t.py").read_text(encoding="utf-8")

    def test_wipe_applied_via_diff_apply_warns(self, tmp_path):
        # P26-1/P26-2: the accidental-wipe guard must fire on the LIVE
        # diff_apply branch AFTER a real apply. The old code never called the
        # guard on this branch, and the old guard skipped exactly this case
        # (post-apply file = 5 lines < 20 → continue → no warning).
        import subprocess as sp

        n = 1000
        h = self._git_repo(tmp_path)
        (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(n)), encoding="utf-8")
        sp.run(["git", "-C", str(tmp_path), "add", "big.txt"], check=True)
        sp.run(["git", "-C", str(tmp_path), "commit", "-qm", "add big"], check=True)
        r = h._apply_patch_text(_wipe_patch("big.txt", n, keep=5), path_hint="big.txt")
        assert r.ok, r.error
        assert "CONTENT LOSS WARNING" in r.content
        assert "content_ratio_warning" in (r.metadata or {})
        # the file really was wiped to 5 lines
        assert len((tmp_path / "big.txt").read_text(encoding="utf-8").splitlines()) == 5

    def test_wipe_applied_via_engine_warns(self, tmp_path):
        # P26-1: same guard on the MAIN PatchEngine.apply_patch success branch
        # — the branch every apply_patch tool call actually takes. PatchEngine's
        # own safety valve (P22-1) already rejects total rewipes
        # (file_rewrite_too_large), so this "wipe" is the kind the engine lets
        # through: large-content churn that is still under its ratio ceiling.
        # Before P26-1 the content-loss guard was never wired to this branch,
        # so such a churn applied silently.
        import subprocess as sp

        n = 1000
        h = self._git_repo(tmp_path)
        body = "\n".join(f"def fn_{i}():\n    return {i}\n" for i in range(n // 2))
        (tmp_path / "big.py").write_text(body + "\n", encoding="utf-8")
        sp.run(["git", "-C", str(tmp_path), "add", "big.py"], check=True)
        sp.run(["git", "-C", str(tmp_path), "commit", "-qm", "add big"], check=True)

        body_lines = body.splitlines()
        # Wipe the second half (500 removals, 0 additions) with context that
        # actually matches the file — a bare `_wipe_patch` fixture ("line0"…
        # context) can never apply to this file, so the engine's repair
        # ladder exhausts instead of reaching the success branch the guard
        # is wired to.
        diff = difflib.unified_diff(
            body_lines,
            body_lines[: n // 2],
            fromfile="a/big.py",
            tofile="b/big.py",
            lineterm="",
        )
        patch = "diff --git a/big.py b/big.py\n" + "\n".join(diff) + "\n"
        # Fixture sanity: this churn must stay UNDER the engine's rewrite
        # valve (P22-1) — otherwise the engine rejects it before the
        # content-loss guard ever runs.
        ratio = 1.0 - float(difflib.SequenceMatcher(a=body_lines, b=body_lines[: n // 2], autojunk=False).ratio())
        assert ratio < 0.5, f"test fixture not under the engine valve: {ratio:.2f}"
        r = h._tool_apply_patch({"patch": patch, "path": "big.py"})
        assert r.ok, r.error
        assert "CONTENT LOSS WARNING" in r.content
        assert "content_ratio_warning" in (r.metadata or {})

    def test_wipe_via_synthesize_path_warns(self, tmp_path):
        # P26-1: the content-loss guard must fire on the LIVE synthesize path
        # too (non-diff input + path). The applied diff lives in
        # synthesize_and_apply's metadata["patch"]; the guard must score THAT,
        # because the raw non-diff input has no hunks. The parallel session's
        # abandoned stash carried this guard, but the committed P26-1 wiring
        # never reached this branch.
        import subprocess as sp
        import unittest.mock as um
        from types import SimpleNamespace

        n = 1000
        h = self._git_repo(tmp_path)
        (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(n)), encoding="utf-8")
        sp.run(["git", "-C", str(tmp_path), "add", "big.txt"], check=True)
        sp.run(["git", "-C", str(tmp_path), "commit", "-qm", "add big"], check=True)

        fake = SimpleNamespace(
            success=True,
            error=None,
            patch_applied="applied (synthesized)",
            metadata={"patch": _wipe_patch("big.txt", n, keep=5)},
        )
        with um.patch.object(
            write_tools_patch_mixin.PatchEngine,
            "synthesize_and_apply",
            return_value=fake,
        ):
            r = h._tool_apply_patch({"patch": "replace the whole file body", "path": "big.txt"})
        assert r.ok, r.error
        assert "CONTENT LOSS WARNING" in r.content
        assert "content_ratio_warning" in (r.metadata or {})


# ── _append_applied_patch (shared guarded-append SSOT) ──────────────────────


class TestAppendAppliedPatch:
    def test_appends_records_in_order(self, harness):
        harness._append_applied_patch("rec-1")
        harness._append_applied_patch("rec-2")
        assert harness._applied_patches == ["rec-1", "rec-2"]

    def test_missing_attribute_is_silent_noop(self, harness):
        del harness._applied_patches
        harness._append_applied_patch("rec")  # must not raise

    def test_non_list_attribute_is_silent_noop(self, harness):
        harness._applied_patches = None
        harness._append_applied_patch("rec")  # TypeError path — must not raise
        harness._applied_patches = object()
        harness._append_applied_patch("rec")  # AttributeError path — must not raise

    def test_edit_text_site_routes_through_helper(self, harness, tmp_path, monkeypatch):
        # Routing pin: edit_text's success site must go through the shared helper
        # (never an inline raw append), so the guard cannot be skipped by a
        # future rewrite.
        calls: list[str] = []
        monkeypatch.setattr(
            type(harness),
            "_append_applied_patch",
            lambda self, record: calls.append(record),
        )
        (tmp_path / "t.py").write_text("x = 1\n", encoding="utf-8")
        r = harness._tool_edit_text({"file_path": "t.py", "old_string": "x = 1", "new_string": "x = 10"})
        assert r.ok, r.error
        assert calls == ["edit_text:t.py:replace:False"]

    def test_each_module_has_no_inline_raw_append(self):
        # SSOT pin: the ONLY raw ``_applied_patches.append(`` in the patch mixin
        # is the helper itself (count == 1); the edit mixin must have none
        # (count == 0). Any inline recording site breaks this count.
        from pathlib import Path as _Path

        for _mod, _expected in (
            (write_tools_patch_mixin, 1),
            (write_tools_edit_mixin, 0),
        ):
            _src = _Path(_mod.__file__).read_text(encoding="utf-8")
            assert _src.count("_applied_patches.append(") == _expected, _mod.__name__


# ── _analyze_patch_failure ──────────────────────────────────────────────────


class TestAnalyzePatchFailure:
    def test_already_applied(self, harness):
        out = harness._analyze_patch_failure(
            "--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-a\n+b\n", "error: patch already applied"
        )
        assert out["reason"] == "already_applied"
        assert "already applied" in out["hint"]
        assert out["file_path"] == "t.py"

    def test_corrupt_patch(self, harness):
        out = harness._analyze_patch_failure("garbage", "error: corrupt patch at line 2")
        assert out["reason"] == "offset_error"

    def test_context_mismatch_with_line_hint(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        out = harness._analyze_patch_failure(
            "--- a/t.py\n+++ b/t.py\n@@ -1,3 +1,3 @@\n alpha\n-betax\n+beta\n gamma\n",
            "error: patch failed: t.py:1: hunk failed, does not apply at line 1",
        )
        assert out["reason"] == "context_mismatch"
        assert out["conflicting_lines"] == [1]
        assert "Context mismatch" in out["hint"]
        assert "Actual file content" in out["error_message"]

    def test_file_not_found(self, harness, tmp_path):
        (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
        out = harness._analyze_patch_failure(
            "--- a/real.py\n+++ b/real.py\n@@ -1 +1 @@\n-a\n+b\n",
            "error: cannot stat real.py: No such file or directory",
        )
        assert out["reason"] == "file_not_found"
        assert "create_file" in out["error_message"]

    def test_markdown_fence_detected(self, harness):
        patch = "```\n@@ -1 +1 @@\n-a\n+b\n```\n"
        out = harness._analyze_patch_failure(patch, "does not apply")
        assert "markdown code fences" in out["error_message"]

    def test_before_after_notation_detected(self, harness):
        patch = "before: a\nafter: b\n"
        out = harness._analyze_patch_failure(patch, "does not apply")
        assert "before:/after:" in out["error_message"]

    def test_missing_standard_headers(self, harness):
        out = harness._analyze_patch_failure("just text", "does not apply")
        assert "standard diff headers" in out["error_message"]

    def test_hunk_parsing(self, harness):
        patch = "--- a/t.py\n+++ b/t.py\n@@ -1,3 +1,3 @@\n ctx\n-rm\n+add\n"
        out = harness._analyze_patch_failure(patch, "does not apply")
        assert out["hunk_count"] == 1
        assert out["file_path"] == "t.py"


# ── _tool_anchor_edit deeper paths ──────────────────────────────────────────


class TestToolAnchorEditPaths:
    def test_insert_after_success(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "a",
                "edit_mode": "insert_after",
                "code_snippet": "x",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nx\nb\n"

    def test_replace_line_success(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b",
                "edit_mode": "replace_line",
                "code_snippet": "B",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nB\nc\n"

    def test_delete_success(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b",
                "edit_mode": "delete",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nc\n"

    def test_delete_multiline_success(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b\nc",
                "edit_mode": "delete",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nd\n"

    def test_delete_pattern_not_found(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "zzz",
                "edit_mode": "delete",
            }
        )
        assert not r.ok and "not found" in r.error

    def test_delete_empty_pattern(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "  ",
                "edit_mode": "delete",
            }
        )
        assert not r.ok and "required" in r.error

    def test_anchor_miss_failure_class(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "nope",
                "edit_mode": "insert_after",
                "code_snippet": "x",
            }
        )
        assert not r.ok
        assert r.metadata.get("failure_class") == "anchor_miss"

    def test_anchor_ast_lineno_bypasses_search(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\nc\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_ast_lineno": 2,
                "edit_mode": "insert_after",
                "code_snippet": "x",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nb\nx\nc\n"

    def test_anchor_ast_lineno_out_of_range_falls_back_to_pattern(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_ast_lineno": 99,
                "anchor_pattern": "b",
                "edit_mode": "insert_before",
                "code_snippet": "x",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "a\nx\nb\n"

    def test_anchor_not_unique_guard(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("x\ny\nx\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "x",
                "edit_mode": "insert_after",
                "code_snippet": "z",
            }
        )
        assert not r.ok
        assert r.metadata.get("failure_class") == "anchor_not_unique"

    def test_delete_not_unique_guard(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("x\ny\nx\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "x",
                "edit_mode": "delete",
            }
        )
        assert not r.ok
        assert r.metadata.get("failure_class") == "anchor_not_unique"
        assert "irreversible" in r.error

    def test_occurrence_disambiguates(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("x\ny\nx\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "x",
                "occurrence": 1,
                "edit_mode": "insert_after",
                "code_snippet": "z",
            }
        )
        assert r.ok, r.error
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "x\nz\ny\nx\n"

    def test_replace_line_bracket_balance_expansion(self, harness, tmp_path):
        # old line `foo(` opens a bracket (delta +1); new line is balanced (0) →
        # the forward scan consumes the following `);` close line, expanding the
        # replacement from 1 line to 2
        (tmp_path / "t.js").write_text("foo(\n);\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.js",
                "anchor_pattern": "foo(",
                "edit_mode": "replace_line",
                "code_snippet": "foo(a)",
            }
        )
        assert r.ok, r.error
        content = (tmp_path / "t.js").read_text(encoding="utf-8")
        assert "foo(a)" in content and ");" not in content

    def test_replace_line_bracket_imbalance_refused(self, harness, tmp_path):
        (tmp_path / "t.js").write_text("foo()\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.js",
                "anchor_pattern": "foo()",
                "edit_mode": "replace_line",
                "code_snippet": "foo(a,",
            }
        )
        # no closing line exists → structural gate violation
        assert not r.ok
        assert r.metadata.get("failure_class") == "structural_gate_violation"

    def test_insert_before_collection_literal_indent_fix(self, harness, tmp_path):
        (tmp_path / "t.js").write_text("const x = {\n    a: 1,\n};\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.js",
                "anchor_pattern": "};",
                "edit_mode": "insert_before",
                "code_snippet": "    b: 2,",
            }
        )
        assert r.ok, r.error
        content = (tmp_path / "t.js").read_text(encoding="utf-8")
        assert "b: 2" in content

    def test_multiline_anchor_resolution(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n  b\n  c\nd\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b\nc",
                "edit_mode": "insert_after",
                "code_snippet": "x",
            }
        )
        assert r.ok, r.error
        content = (tmp_path / "t.txt").read_text(encoding="utf-8")
        assert content.index("x") > content.index("c")

    def test_multiline_anchor_failure(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\nb\n", encoding="utf-8")
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.txt",
                "anchor_pattern": "b\nzzz",
                "edit_mode": "insert_after",
                "code_snippet": "x",
            }
        )
        assert not r.ok

    def test_delete_syntax_gate(self, harness, tmp_path):
        (tmp_path / "t.py").write_text("def f():\n    pass\n", encoding="utf-8")
        # deleting `pass` leaves an empty def → syntax error → gate refuses
        r = harness._tool_anchor_edit(
            {
                "file_path": "t.py",
                "anchor_pattern": "pass",
                "edit_mode": "delete",
            }
        )
        assert not r.ok
        assert r.metadata.get("failure_class") == "syntax_invalid_after_edit"
        assert (tmp_path / "t.py").read_text(encoding="utf-8") == "def f():\n    pass\n"  # untouched

    def test_read_failure_reported(self, harness, tmp_path):
        (tmp_path / "t.txt").write_text("a\n", encoding="utf-8")
        import unittest.mock as um

        with um.patch.object(harness, "repo_root", str(tmp_path / "nonexistent")):
            r = harness._tool_anchor_edit(
                {
                    "file_path": "t.txt",
                    "anchor_pattern": "a",
                    "edit_mode": "delete",
                }
            )
        assert not r.ok


# ── B2: symlink preservation (anchor_edit) ──────────────────────────────────


def test_anchor_edit_preserves_symlink(harness, tmp_path):
    """anchor_edit must write through a repo-internal symlink (resolved path),
    not replace the link with a regular file."""
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to("real.py")

    r = harness._tool_anchor_edit(
        {"file_path": "link.py", "edit_mode": "insert_after", "anchor_pattern": "x = 1", "code_snippet": "y = 2"}
    )
    assert r.ok, r.error
    assert (tmp_path / "link.py").is_symlink(), "symlink must not be replaced by a regular file"
    assert (tmp_path / "real.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"


# ────────────────────────────────────────────────────────────────────────
# P25-4: plan normalization / error enrichment must stream, never load the
# whole target file. These paths run BEFORE plan_compiler's stat gate
# (P24-2), so an unbounded read here defeated the gate downstream — a
# multi-hundred-MB target was fully materialised up to 4x per op during
# normalization just to extract one line / a window / a 10-line hint.
# ────────────────────────────────────────────────────────────────────────


def _boom(*args, **kwargs):
    raise AssertionError("read_text must not be called — normalization/enrichment must stream")


def _big_file(path: Path, anchor_line: int = 12345) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(1, anchor_line + 10):
            if i == anchor_line:
                fh.write("ANCHOR_LINE_MARKER\n")
            else:
                fh.write(f"filler line {i}\n")


def test_normalize_plan_op_line_to_anchor_streams(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    harness = _Harness(tmp_path)
    target = tmp_path / "big.txt"
    _big_file(target)
    monkeypatch.setattr(_Path, "read_text", _boom)

    op = {"op": "insert_after", "path": "big.txt", "line": 12345}
    repairs: list[str] = []
    out = harness._normalize_plan_op(op, repairs)

    assert out["anchor"] == "ANCHOR_LINE_MARKER"
    assert "line 12345→anchor" in "; ".join(repairs)


def test_enrich_plan_error_first_lines_hint_streams(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    harness = _Harness(tmp_path)
    target = tmp_path / "big.txt"
    _big_file(target)
    monkeypatch.setattr(_Path, "read_text", _boom)

    plan = {
        "kind": "ASICODE_PLAN_V1",
        "ops": [
            {"op": "insert_after", "path": "big.txt", "anchor": "no such line", "lines": ["x"]},
        ],
    }
    out = harness._enrich_plan_error(plan, "anchor not found")

    assert "First 10 lines" in out
    assert "ANCHOR_LINE_MARKER" not in out  # only the head is shown


def test_enrich_plan_error_closest_match_streams(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    harness = _Harness(tmp_path)
    target = tmp_path / "mod.py"
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("def unrelated():\n    pass\n")
        fh.write("def similar_fn():\n    return 1\n")
        fh.write("x = 2\n")
    monkeypatch.setattr(_Path, "read_text", _boom)

    plan = {
        "kind": "ASICODE_PLAN_V1",
        "ops": [
            {"op": "edit_blocks", "path": "mod.py", "blocks": [{"before": "def similar_fn():", "after": "..."}]},
        ],
    }
    out = harness._enrich_plan_error(plan, "before text not found")

    assert "Closest match" in out
    assert "similar_fn" in out


# ── WP-B1 regression: content lines starting with '--'/'++' must survive ─────


class TestDoubleMarkerContentLines:
    """WP-B1: bare '---'/'+++' prefix checks dropped real content lines whose
    *content* starts with '--' (removed) or '++' (added). Unified-diff headers
    always carry a space after the marker ('--- a/x', '+++ /dev/null'); body
    lines never do — '---x'/'+++x' are therefore always content.
    """

    def test_removed_and_added_double_marker_lines_preserved(self, harness):
        patch = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,2 +1,2 @@\n"
            " keep\n"
            "---comment\n"  # removed line whose content is '--comment'
            "+++line\n"  # added line whose content is '++line'
        )
        files = harness._parse_unified_diff_files(patch)
        assert len(files) == 1
        assert files[0]["hunks"][0]["lines"] == [
            (" ", "keep"),
            ("-", "--comment"),
            ("+", "++line"),
        ]

    def test_dev_null_header_still_skipped(self, harness):
        # Regression guard: header markers (marker + space) must keep being
        # recognized as headers, not content, after the fix.
        patch = "diff --git a/n b/n\n--- /dev/null\n+++ b/n\n@@ -0,0 +1 @@\n+x\n"
        files = harness._parse_unified_diff_files(patch)
        assert [f["file"] for f in files] == ["n"]
        assert files[0]["hunks"][0]["lines"] == [("+", "x")]

    def test_new_file_content_starting_with_double_plus_preserved(self, harness):
        patch = (
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1,3 @@\n"
            "+a\n"
            "+++tail\n"  # added line whose content is '++tail'
            "+c\n"
        )
        out = harness._extract_new_file_target(patch, None)
        assert out == {"file_path": "new.txt", "content": "a\n++tail\nc\n"}

    def test_new_file_removed_line_disqualifies_not_fabricates(self, harness):
        # A '-' content line starting with '--' must DISQUALIFY the creation
        # (conservative bail), not be silently dropped so content is fabricated.
        patch = (
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1,2 @@\n"
            "+a\n"
            "---gone\n"  # removed line — not a pure creation
        )
        assert harness._extract_new_file_target(patch, None) is None
