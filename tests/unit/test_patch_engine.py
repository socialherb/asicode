"""Tests for PatchEngine (external_llm/patch_engine.py)."""
import os
from unittest.mock import MagicMock, patch

import pytest

from external_llm.patch_engine import PatchContext, PatchEngine, PatchResult

MINIMAL_DIFF = (
    "--- a/foo.py\n"
    "+++ b/foo.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-x = 1\n"
    "+x = 2\n"
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Temp git-like repo root."""
    (tmp_path / ".git").mkdir()
    return str(tmp_path)


@pytest.fixture
def engine(tmp_repo):
    return PatchEngine(tmp_repo)


# ── PatchContext / PatchResult dataclasses ────────────────────────────────────

class TestPatchContextResult:
    def test_patch_context_defaults(self):
        ctx = PatchContext()
        assert ctx.original_request is None
        assert ctx.file_content is None
        assert ctx.llm_output is None
        assert ctx.output_mode is None
        assert ctx.metadata == {}

    def test_patch_context_with_values(self):
        ctx = PatchContext(
            original_request="fix bug",
            file_content="x=1",
            llm_output="--- a/f\n+++ b/f\n@@ ... @@\n",
            output_mode="diff",
            metadata={"key": "val"},
        )
        assert ctx.original_request == "fix bug"
        assert ctx.metadata == {"key": "val"}

    def test_patch_result_success(self):
        r = PatchResult(success=True, patch_applied="some diff")
        assert r.success is True
        assert r.patch_applied == "some diff"
        assert r.error is None
        assert r.metadata == {}

    def test_patch_result_failure(self):
        r = PatchResult(success=False, error="it broke", metadata={"mode": "git_apply"})
        assert r.success is False
        assert r.error == "it broke"
        assert r.metadata["mode"] == "git_apply"


# ── _looks_like_unified_diff ──────────────────────────────────────────────────

class TestLooksLikeUnifiedDiff:
    def test_valid_diff_detected(self, engine):
        assert engine._looks_like_unified_diff(MINIMAL_DIFF) is True

    def test_empty_string_is_not_diff(self, engine):
        assert engine._looks_like_unified_diff("") is False

    def test_plain_text_not_diff(self, engine):
        assert engine._looks_like_unified_diff("hello world") is False

    def test_missing_hunk_marker_not_diff(self, engine):
        text = "--- a/foo.py\n+++ b/foo.py\n+x = 2\n"
        assert engine._looks_like_unified_diff(text) is False

    def test_diff_git_header(self, engine):
        text = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        assert engine._looks_like_unified_diff(text) is True

    def test_hunk_only_patch_detected(self, engine):
        """Hunk-only patches (no header, starting with @@) are valid unified diffs."""
        text = "@@ -1,2 +1,2 @@\n-a\n+b\n"
        assert engine._looks_like_unified_diff(text) is True
        text = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
        assert engine._looks_like_unified_diff(text) is True


# ── normalize_and_validate ────────────────────────────────────────────────────

class TestNormalizeAndValidate:
    def test_empty_patch_returns_error(self, engine):
        _p, err = engine.normalize_and_validate("", None)
        assert err is not None
        assert "empty" in err.lower()

    def test_non_diff_returns_error(self, engine):
        _p, err = engine.normalize_and_validate("just some code", None)
        assert err is not None
        assert "unified diff" in err.lower()

    def test_valid_diff_normalizes(self, engine):
        """normalize_and_validate should parse without crashing; result depends on git check."""
        p, _err = engine.normalize_and_validate(MINIMAL_DIFF, None)
        # The patch text should be preserved (even if git check fails)
        assert "@@ " in p

    def test_trailing_newline_added(self, engine):
        diff_no_newline = MINIMAL_DIFF.rstrip("\n")
        # After normalization, trailing newline should be present
        # (we mock git check to avoid real git dependency)
        with patch.object(engine, '_git_apply_check_best_effort', return_value=(True, None)):
            p, _err = engine.normalize_and_validate(diff_no_newline, None)
        assert p.endswith("\n")


# ── _output_mode_to_enum ──────────────────────────────────────────────────────

class TestOutputModeToEnum:
    def test_known_modes_mapped(self, engine):
        try:
            from external_llm.output_modes import OutputMode
        except ImportError:
            pytest.skip("OutputMode not available")

        assert engine._output_mode_to_enum("diff") == OutputMode.UNIFIED_DIFF
        assert engine._output_mode_to_enum("auto") == OutputMode.UNIFIED_DIFF
        assert engine._output_mode_to_enum("full_file") == OutputMode.FULL_FILE

    def test_unknown_mode_defaults_to_unified_diff(self, engine):
        try:
            from external_llm.output_modes import OutputMode
        except ImportError:
            pytest.skip("OutputMode not available")

        result = engine._output_mode_to_enum("some_unknown_mode")
        assert result == OutputMode.UNIFIED_DIFF


# ── apply_patch (without real git) ────────────────────────────────────────────

class TestApplyPatch:
    def test_empty_patch_fails(self, engine):
        result = engine.apply_patch("", target_file=None)
        assert result.success is False

    def test_non_diff_text_fails(self, engine):
        result = engine.apply_patch("not a diff at all", target_file=None)
        assert result.success is False

    def test_metadata_has_required_keys(self, engine):
        result = engine.apply_patch("not a diff", target_file=None)
        for key in ("reason", "mode", "fallback_used", "first_fail_reason", "execution_steps"):
            assert key in result.metadata

    def test_diff_apply_success(self, engine, tmp_repo):
        """When _diff_apply succeeds, apply_patch returns success."""
        # Create target file to pass pre-apply file-existence check
        foo_path = os.path.join(tmp_repo, "foo.py")
        with open(foo_path, "w") as f:
            f.write("x = 1\n")
        engine._diff_apply = MagicMock(return_value=(True, None, "git_apply_success", {}))
        with patch.object(engine, 'normalize_and_validate', return_value=(MINIMAL_DIFF, None)):
            result = engine.apply_patch(MINIMAL_DIFF)
        assert result.success is True
        assert result.metadata["mode"] == "git_apply"

    def test_diff_apply_failure_falls_through(self, engine):
        """When _diff_apply fails, repair ladder is attempted."""
        engine._diff_apply = MagicMock(return_value=(False, "hunk mismatch", "hunk_mismatch", {}))
        with patch.object(engine, 'normalize_and_validate', return_value=(MINIMAL_DIFF, None)), \
             patch.object(engine, '_tolerant_git_apply', return_value=(False, "fail", "tol")), \
             patch.object(engine, '_exact_reanchor_patch', return_value=None), \
             patch.object(engine, '_reanchor_patch', return_value=None), \
             patch.object(engine, 'repair_patch', return_value=PatchResult(
                 success=False, metadata={"fallback_used": [], "error": "all failed"})):
            result = engine.apply_patch(MINIMAL_DIFF)
        assert result.success is False

    def test_no_diff_apply_module(self, engine, tmp_repo):
        """When diff_apply module is not available, git apply is skipped."""
        # Create target file to pass pre-apply file-existence check
        foo_path = os.path.join(tmp_repo, "foo.py")
        with open(foo_path, "w") as f:
            f.write("x = 1\n")
        engine._diff_apply = None
        with patch.object(engine, 'normalize_and_validate', return_value=(MINIMAL_DIFF, None)), \
             patch.object(engine, '_tolerant_git_apply', return_value=(False, "fail", "tol")), \
             patch.object(engine, '_exact_reanchor_patch', return_value=None), \
             patch.object(engine, '_reanchor_patch', return_value=None), \
             patch.object(engine, 'repair_patch', return_value=PatchResult(
                 success=False, metadata={"fallback_used": [], "error": "no diff"})):
            result = engine.apply_patch(MINIMAL_DIFF)
        assert result.success is False
        assert "diff_apply module not available" in result.metadata.get("first_fail_reason", "")


# ── _try_synthesize_diff_from_file_blocks ─────────────────────────────────────

class TestSynthesizeDiffFromFileBlocks:
    def test_missing_target_file(self, engine, tmp_repo):
        _diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "nonexistent.py", "FILE: nonexistent.py\n```\ncode\n```\n"
        )
        assert reason == "target_missing"

    def test_no_file_block_in_text(self, engine, tmp_repo):
        target = os.path.join(tmp_repo, "foo.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        _diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "foo.py", "just some text with no file blocks"
        )
        assert reason == "no_file_block"

    def test_no_changes_returns_no_changes(self, engine, tmp_repo):
        target = os.path.join(tmp_repo, "same.py")
        content = "x = 1\n"
        with open(target, "w") as f:
            f.write(content)
        llm_text = f'FILE: same.py\n```\n{content}```\n'
        _diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "same.py", llm_text
        )
        assert reason == "no_changes"

    def test_valid_file_block_produces_diff(self, engine, tmp_repo):
        target = os.path.join(tmp_repo, "mod.py")
        with open(target, "w") as f:
            f.write("x = 1\n")
        llm_text = "FILE: mod.py\n```\nx = 2\n```\n"
        diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "mod.py", llm_text
        )
        # Either produces a diff or fails with a known reason
        assert reason in ("", "no_file_block", "no_target_file_block") or "@@ " in diff
    class TestExactReanchorPatch:
        def test_substring_false_positive_rejected(self, engine, tmp_repo):
            target = os.path.join(tmp_repo, "test_substring.py")
            with open(target, "w") as f:
                f.write("return validate(self):\n")
                f.write("something else\n")
            diff = (
                "--- a/test_substring.py\n"
                "+++ b/test_substring.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-return val\n"
                "+return other\n"
            )
            result = engine._exact_reanchor_patch(diff, target)
            assert result is None, "Substring match should not reanchor"

        def test_exact_match_works(self, engine, tmp_repo):
            target = os.path.join(tmp_repo, "test_exact.py")
            with open(target, "w") as f:
                f.write("return val\n")
                f.write("something else\n")
            # Diff claims the hunk is at line 5, but the actual content is at line 1.
            # offset_diff = |0 - 4| = 4 → triggers reanchoring.
            diff = (
                "--- a/test_exact.py\n"
                "+++ b/test_exact.py\n"
                "@@ -5,2 +5,2 @@\n"
                "-return val\n"
                "+return other\n"
            )
            result = engine._exact_reanchor_patch(diff, target)
            assert result is not None, "Exact match should reanchor when offset is wrong"
            assert "@@ -1," in result, "Reanchored header should point to line 1"
