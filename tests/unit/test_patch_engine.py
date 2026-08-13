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
        # P22-1: the rewrite valve now compares LINE sequences, so a fixture that
        # swaps the file's only line (100% change) is rejected by policy — real
        # FILE-block output is a partial edit of the full file, mirror that.
        target = os.path.join(tmp_repo, "mod.py")
        with open(target, "w") as f:
            f.write("x = 1\ny = 2\nz = 3\n")
        llm_text = "FILE: mod.py\n```\nx = 1\ny = 9\nz = 3\n```\n"
        diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "mod.py", llm_text
        )
        assert reason in ("file_block_synth", "") or "@@ " in diff
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


# ── repair_patch METHOD header (nested-class chain support) ────────────────


class TestRepairPatchMethodHeader:
    """``repair_patch`` parses ``METHOD:<class>.<method>`` headers and forwards
    them to ``ASTRewriter.replace_method``, which explicitly accepts a dotted
    ``class_name`` chain for nested classes (resolved via
    ``class_chain = class_name.split('.')`` — see ast_rewrite.py).

    Regression: the header was unpacked with ``path.split('.')`` which raises
    ``ValueError`` for nested-class chains (e.g. ``METHOD:A.B.method`` → 3
    parts), silently aborting the AST-rewrite fallback. Fixed by
    ``path.rsplit('.', 1)``."""

    @staticmethod
    def _isolate_ast_rewrite(engine):
        """Enable only the AST-rewrite rung; disable all later fallbacks so the
        test is deterministic regardless of optional imports."""
        engine.ast_rewriter = MagicMock()
        engine.ast_rewriter.replace_method.return_value = object()
        engine.ast_rewriter.generate_patch.return_value = "diff --git a/t b/t\n"
        engine.symbol_searcher = None
        engine.semantic_patcher = None
        engine.patch_synthesizer = None
        engine.hybrid_parser = None
        return engine

    @pytest.mark.parametrize("header, expect_class, expect_method", [
        ("METHOD:A.method", "A", "method"),
        ("METHOD:A.B.method", "A.B", "method"),
        ("METHOD:A.B.C.method", "A.B.C", "method"),
    ])
    def test_method_header_class_chain_rsplit(
        self, engine, tmp_repo, header, expect_class, expect_method,
    ):
        target = os.path.join(tmp_repo, "target.py")
        llm_output = header + "\nFILE: target.py\n```\ndef method(self):\n    return 42\n```\n"
        self._isolate_ast_rewrite(engine)

        fake_blocks = [{
            "path": "target.py",
            "text": "def method(self):\n    return 42\n",
            "content": "def method(self):\n    return 42\n",
        }]
        with patch("external_llm.patch_engine.parse_file_blocks", return_value=fake_blocks):
            result = engine.repair_patch(
                patch_text="garbage",
                target_file=target,
                failure_reason="hunk mismatch",
                llm_output=llm_output,
            )

        assert result.success is True
        assert result.metadata["mode"] == "ast_method"
        engine.ast_rewriter.replace_method.assert_called_once()
        # call args positional: (target_file, class_name, method_name, new_code)
        _tf, class_name, method_name, _code = engine.ast_rewriter.replace_method.call_args.args
        assert class_name == expect_class
        assert method_name == expect_method


# ── _sanitize_patch_lines: hunk-body region tracking ──────────────────────────

class TestSanitizePatchLinesBodyRegion:
    """``_sanitize_patch_lines`` must distinguish the *header* region (between
    sections) from the *body* region (inside a hunk, after ``@@``).

    Regression: the function de-indented every line whose *content* looked like a
    diff marker. A context line such as ``' +++ b/other.py'`` (single-space prefix
    = context, content happens to be a header-looking string) was de-indented into
    a real ``+++`` header, corrupting the patch (``git apply``: "corrupt patch" or
    wrong file). Body lines now preserve their leading char verbatim; only the
    header region is de-indented, and a new ``diff --git`` returns to header mode."""

    @staticmethod
    def _sanitize(patch: str) -> str:
        return PatchEngine._sanitize_patch_lines(patch)

    def test_context_line_with_marker_content_preserved(self):
        """A context line whose CONTENT looks like a marker is file content, not a
        header — its single-space prefix must be preserved verbatim."""
        patch = (
            "diff --git a/notes.md b/notes.md\n"
            "--- a/notes.md\n"
            "+++ b/notes.md\n"
            "@@ -1,3 +1,3 @@\n"
            " +++ b/other.py\n"   # context: 1 space + marker-like content
            "-old line\n"
            "+new line\n"
        )
        out = self._sanitize(patch)
        out_lines = out.splitlines()
        # The marker-like context line is NOT turned into a header line.
        assert "+++ b/other.py" not in out_lines
        # It survives with its leading space (verbatim body content).
        assert " +++ b/other.py" in out_lines
        # Exactly one genuine +++ header remains.
        assert sum(1 for ln in out_lines if ln.startswith("+++ ")) == 1

    def test_multi_section_diff_git_returns_to_header(self):
        """An indented ``diff --git`` after a hunk body re-enters header mode
        (de-indented), proving the body→header transition works."""
        patch = (
            "  diff --git a/a.py b/a.py\n"
            "  --- a/a.py\n"
            "  +++ a/a.py\n"
            "  @@ -1,1 +1,1 @@\n"
            " -x\n"
            " +y\n"
            "  diff --git a/b.py b/b.py\n"   # indented 2nd section header
            "  --- a/b.py\n"
            "  +++ b/b.py\n"
            "  @@ -1,1 +1,1 @@\n"
            " -p\n"
            " +q\n"
        )
        out = self._sanitize(patch)
        out_lines = out.splitlines()
        # Both section headers de-indented (header region in both cases).
        assert "diff --git a/a.py b/a.py" in out_lines
        assert "diff --git a/b.py b/b.py" in out_lines

    def test_context_fence_inside_body_preserved(self):
        """A bare-fence context line (whitespace + ```) is file content when inside
        a hunk — must NOT be dropped like header-region wrapper fences. The old code
        dropped it (``stripped.startswith('```')`` → continue), causing a hunk line
        count mismatch and a corrupt patch."""
        patch = (
            "diff --git a/r.md b/r.md\n"
            "--- a/r.md\n"
            "+++ b/r.md\n"
            "@@ -1,4 +1,4 @@\n"
            " ```\n"          # context: bare fence (old code dropped this)
            " old code\n"
            " ```\n"          # context: bare fence (old code dropped this)
            "+new code\n"
        )
        out = self._sanitize(patch)
        out_lines = out.splitlines()
        # Bare-fence context lines survive in body (not dropped).
        assert out_lines.count(" ```") == 2

    def test_header_markers_still_deindented(self):
        """Regression guard: header-region markers are STILL de-indented."""
        patch = (
            "  diff --git a/x b/x\n"
            "  --- a/x\n"
            "  +++ b/x\n"
            "  @@ -1,1 +1,1 @@\n"
            " -a\n"
            " +b\n"
        )
        out = self._sanitize(patch)
        out_lines = out.splitlines()
        assert out_lines[0] == "diff --git a/x b/x"
        assert "--- a/x" in out_lines
        assert "+++ b/x" in out_lines
        assert "@@ -1,1 +1,1 @@" in out_lines


# ── _salvage_small_model_output: large-file guard ─────────────────────────────

class TestSalvageLargeFileGuard:
    """``_salvage_small_model_output`` runs an O(NxMxL) sliding-window
    ``SequenceMatcher.ratio()`` (Strategy 1/2) over the WHOLE file with no window
    limit, unlike ``_reanchor_patch`` which caps at >2000 lines. Salvage now bails
    early for >2000-line files (consistent with reanchor's documented contract)."""

    def test_large_file_bails_fast(self, engine, tmp_repo):
        """A file with >2000 lines returns None immediately (no O(NxM) scan).
        Note: target_file is repo-relative (matches internal call sites)."""
        big_rel = "big.py"
        with open(os.path.join(tmp_repo, big_rel), "w") as f:
            f.write("\n".join(f"line_{i} = {i}" for i in range(2500)) + "\n")
        result = engine._salvage_small_model_output("-old\n+new", big_rel)
        assert result is None

    def test_small_file_proceeds_past_guard(self, engine, tmp_repo):
        """A <=2000-line file is NOT short-circuited; salvage strategies run and
        can synthesize a real diff (Strategy 1: 'old'→'new' exact match)."""
        small_rel = "small.py"
        with open(os.path.join(tmp_repo, small_rel), "w") as f:
            f.write("old\n")
        result = engine._salvage_small_model_output("-old\n+new", small_rel)
        assert isinstance(result, str)
        assert "new" in result


# ── _apply_diff_once: header synthesis (no duplicates) ───────────────────────

class TestApplyDiffOnceHeaderSynthesis:
    """``_apply_diff_once`` synthesizes missing ``---``/``+++`` headers for
    fragment-only diffs. When EXACTLY ONE header was present, the old code
    prepended both synthesized headers while the original fragment still lived
    in the body — producing two ``+++`` (or two ``---``) lines and a "corrupt
    patch". The pre-hunk fragment is now stripped before reassembly."""

    @staticmethod
    def _capture(engine):
        """Replace ``engine._diff_apply`` with a recorder; return the captured
        ``normalized`` dict (so the test never depends on the real diff_apply)."""
        captured = {}

        def fake(repo_root, normalized, file_path_hint=None):
            captured["normalized"] = normalized
            return (True, "ok", "", {})

        engine._diff_apply = fake
        return captured

    def test_both_headers_missing_synthesizes_pair(self, engine):
        """A ``diff --git`` section with hunks but NO ``---``/``+++`` headers →
        synthesize exactly one pair from the ``b/`` path (no body fragment to
        duplicate)."""
        cap = self._capture(engine)
        patch = (
            "diff --git a/foo.py b/foo.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        engine._apply_diff_once(patch, "foo.py")
        lines = cap["normalized"].splitlines()
        assert sum(1 for ln in lines if ln.startswith("--- ")) == 1
        assert sum(1 for ln in lines if ln.startswith("+++ ")) == 1
        assert lines[0] == "--- a/foo.py"
        assert lines[1] == "+++ b/foo.py"

    def test_one_header_present_no_duplicate(self, engine):
        """Only ``--- a/foo.py`` present (no ``+++``) → strip the orphan old
        header from the body, then synthesize a clean pair. No duplicate
        ``---`` line; exactly one ``+++`` line."""
        cap = self._capture(engine)
        patch = "--- a/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        engine._apply_diff_once(patch, "foo.py")
        lines = cap["normalized"].splitlines()
        assert sum(1 for ln in lines if ln.startswith("--- ")) == 1
        assert sum(1 for ln in lines if ln.startswith("+++ ")) == 1
        assert lines[0] == "--- a/foo.py"
        assert lines[1] == "+++ b/foo.py"
        # The hunk header survives the body-strip.
        assert "@@ -1,1 +1,1 @@" in lines


# ── _keep_only_target_file_section: exact-vs-basename matching ────────────────

class TestKeepOnlyTargetFileSectionMatching:
    """``_keep_only_target_file_section`` picks which ``diff --git`` section to
    keep when the model emitted several. The old single-pass loop tried an
    exact-path match and a basename match in the SAME iteration and broke on the
    first hit — so an EARLIER basename collision (``src/utils.py``) won over a
    LATER exact match (``tests/utils.py``); ``_force_target_file_paths`` then
    rewrote the wrong section's headers onto the target. Exact matches across
    ALL sections are now tried before any basename fallback."""

    @staticmethod
    def _keep(patch, target):
        return PatchEngine._keep_only_target_file_section(patch, target)

    def test_exact_match_preferred_over_earlier_basename(self):
        """Target ``tests/utils.py``: section[0]=src/utils.py (basename only),
        section[1]=tests/utils.py (exact). The exact section must win — NOT the
        earlier basename collision."""
        patch = (
            "diff --git a/src/utils.py b/src/utils.py\n"
            "--- a/src/utils.py\n"
            "+++ b/src/utils.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-SRC\n"
            "+src_new\n"
            "diff --git a/tests/utils.py b/tests/utils.py\n"
            "--- a/tests/utils.py\n"
            "+++ b/tests/utils.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-TST\n"
            "+tst_new\n"
        )
        out = self._keep(patch, "tests/utils.py")
        assert "diff --git a/tests/utils.py b/tests/utils.py" in out
        assert "+tst_new" in out
        # The basename-colliding earlier section is NOT chosen.
        assert "+src_new" not in out

    def test_basename_fallback_when_no_exact_match(self):
        """No exact path match anywhere → basename fallback still selects the
        section whose basename equals the target's basename."""
        patch = (
            "diff --git a/lib/x.py b/lib/x.py\n"
            "--- a/lib/x.py\n"
            "+++ b/lib/x.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-a\n"
            "+b\n"
        )
        out = self._keep(patch, "x.py")
        assert "diff --git a/lib/x.py b/lib/x.py" in out


# ── P22 round regressions ─────────────────────────────────────────────────────

class TestP22FileBlockRegressions:
    def test_large_file_single_line_edit_passes_rewrite_valve(self, engine, tmp_repo):
        """P22-1 regression: a character-level SequenceMatcher (autojunk ON)
        collapses ratio() toward 0 once b exceeds ~200 chars (every popular
        char is junk), so a single-line edit in a >40-50 KB file was rejected
        as file_rewrite_too_large — the FILE-block path was dead for every
        realistic source file. The matcher must run on LINES with autojunk
        disabled (changed_lines_est already assumes a line ratio)."""
        lines = []
        for i in range(2000):
            lines.append(f"def fn{i}():\n    return {i} + 1  # pad\n")
        content = "".join(lines)  # 4000 lines, ~70 KB
        assert len(content) > 60_000
        target = os.path.join(tmp_repo, "big_mod.py")
        with open(target, "w") as f:
            f.write(content)
        new_content = content.replace("return 1999 + 1  # pad", "return 9999  # pad", 1)
        llm_text = f"FILE: big_mod.py\n```\n{new_content}```\n"
        diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "big_mod.py", llm_text
        )
        assert reason != "file_rewrite_too_large", reason
        assert reason in ("file_block_synth", "") or "@@ " in diff

    def test_escape_path_rejected_before_read(self, engine, tmp_repo):
        """P22-2: prompt-derived paths must not escape the repo (../SECRET.txt)."""
        secret = os.path.join(os.path.dirname(tmp_repo), "SECRET.txt")
        with open(secret, "w") as f:
            f.write("TOP_SECRET_TOKEN=abc123\n")
        _diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "../SECRET.txt", "FILE: ../SECRET.txt\n```\ncode\n```\n"
        )
        assert reason == "target_missing"

    def test_salvage_rejects_escape_path(self, engine, tmp_repo):
        """P22-2: _salvage_small_model_output must refuse paths outside the repo."""
        secret = os.path.join(os.path.dirname(tmp_repo), "SECRET.py")
        with open(secret, "w") as f:
            f.write("SECRET = 1\n")
        assert engine._salvage_small_model_output("-old\n+new\n", "../SECRET.py") is None

    def test_file_block_synth_rejects_oversize_target(self, engine, tmp_repo):
        """P22-4: old_text was read unbounded before the new_text char cap
        applied; now an over-size target is rejected on stat() alone."""
        big = os.path.join(tmp_repo, "huge.py")
        with open(big, "w") as f:
            f.write("x = 0\n" * 50_000)  # 300 KB > _MAX_FILE_CHARS (250_000)
        _diff, reason = engine._try_synthesize_diff_from_file_blocks(
            tmp_repo, "huge.py", "FILE: huge.py\n```\nx = 1\n```\n"
        )
        assert reason == "file_too_large"

    def test_salvage_skips_huge_file_without_loading(self, engine, tmp_repo):
        """P22-4: salvage previously read the whole file before its >2000-line
        skip; now a stat() pre-check returns None without loading it."""
        big = os.path.join(tmp_repo, "huge_salvage.py")
        with open(big, "w") as f:
            f.write("x = 0\n" * 60_000)  # 360 KB > _SALVAGE_SKIP_MAX_BYTES (256 KiB)
        assert engine._salvage_small_model_output("+new\n", "huge_salvage.py") is None

    def test_auto_repair_proceeds_on_non_utf8_file(self, engine, tmp_repo, monkeypatch):
        """P22-5: a dead path.read_text(encoding='utf-8') raised
        UnicodeDecodeError on non-UTF-8 files and silently disabled AST repair
        (swallowed by the surrounding except). The repair path must now reach
        ASTRewriter."""
        target = os.path.join(tmp_repo, "latin.py")
        with open(target, "wb") as f:
            f.write(b"def foo():\n    return 1\n# caf\xe9\n")

        class _FakeRewriter:
            def __init__(self, repo_root):
                pass

            def replace_function(self, file_path, symbol_name, new_code):
                return "replaced"

            def replace_class(self, file_path, symbol_name, new_code):
                return None

            def generate_patch(self, file_path, result):
                return "PATCHED"

        monkeypatch.setattr("external_llm.ast_rewrite.ASTRewriter", _FakeRewriter)
        patch = (
            "@@ -1,3 +1,3 @@\n"
            "-def foo():\n"
            "-    return 1\n"
            "+def foo():\n"
            "+    return 2\n"
        )
        out = engine._auto_repair_patch(patch, "latin.py")
        assert out == "PATCHED"
