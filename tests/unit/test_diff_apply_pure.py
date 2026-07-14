"""
Unit tests for the pure helpers in diff_apply.py.

These functions (_clean_diff_lines, _parse_hunk_header, _recount_hunks,
_rewrite_patch_paths, _upgrade_hunk_fragment, _clean_diff,
_classify_git_apply_output, _extract_files_from_git_apply_output,
_is_probably_binary_file, _has_conflict_markers, _resolve_inside_repo_path)
were previously exercised only indirectly through the integration-level
PatchEngine / git-apply path, which made failure branches and edge cases
hard to pin down. These tests target each helper directly.

Run: pytest tests/unit/test_diff_apply_pure.py -v
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# diff_apply.py lives at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diff_apply

# ── _clean_diff_lines ────────────────────────────────────────────────────────

class TestCleanDiffLines:
    def test_empty_input_returns_empty(self):
        assert diff_apply._clean_diff_lines("", strict=False) == []
        assert diff_apply._clean_diff_lines(None, strict=False) == []  # type: ignore[arg-type]

    def test_strips_code_fence(self):
        text = "```diff\ndiff --git a/x b/x\n```"
        out = diff_apply._clean_diff_lines(text, strict=False)
        assert "```diff" not in out
        assert "```" not in out
        assert "diff --git a/x b/x" in out

    def test_strips_agent_chain_hints(self):
        text = "[CHAIN-HINT] something\ndiff --git a/x b/x\n[TOOL CHAIN HINT] y\nTypical next steps: foo"
        out = diff_apply._clean_diff_lines(text, strict=False)
        assert all(not _item_.startswith("[CHAIN-HINT]") for _item_ in out)
        assert all(not _item_.startswith("[TOOL CHAIN HINT]") for _item_ in out)
        assert all(not _item_.startswith("Typical next steps") for _item_ in out)

    def test_drops_index_lines(self):
        text = "index 1234567..89abcde 100644\ndiff --git a/x b/x"
        out = diff_apply._clean_diff_lines(text, strict=False)
        assert all(not _item_.startswith("index ") for _item_ in out)

    def test_trims_leading_and_trailing_blank_lines(self):
        text = "\n\n   \ndiff --git a/x b/x\n\n  \n"
        out = diff_apply._clean_diff_lines(text, strict=False)
        assert out[0] == "diff --git a/x b/x"
        assert out[-1].strip() or out[-1].startswith("diff --git")

    def test_strict_requires_header_or_hunk(self):
        # No diff markers → empty under strict
        text = "some random prose\nwithout any diff markers"
        assert diff_apply._clean_diff_lines(text, strict=True) == []

    def test_strict_skips_leading_junk_before_diff(self):
        text = "Here is the patch:\ndiff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b"
        out = diff_apply._clean_diff_lines(text, strict=True)
        assert out[0].startswith("diff --git")

    def test_strict_truncates_after_noise_budget_exceeded(self):
        # Diff then a wall of prose (>3 noise lines) → truncated.
        text = textwrap.dedent("""\
            diff --git a/x b/x
            @@ -1 +1 @@
            -a
            +b
            Explanation line one.
            Explanation line two.
            Explanation line three.
            Explanation line four.
            This should be dropped.
        """)
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "This should be dropped." not in joined

    def test_strict_tolerates_small_noise_then_resumes(self):
        # A couple of noise lines mid-diff are tolerated, hunk continues.
        text = textwrap.dedent("""\
            diff --git a/x b/x
            @@ -1,3 +1,3 @@
             ctx1
            -old
            +new
             ctx2
        """)
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "+new" in joined and "-old" in joined and " ctx2" in joined

    def test_strict_preserves_blank_context_inside_hunk(self):
        text = textwrap.dedent("""\
            diff --git a/x b/x
            @@ -1,3 +1,3 @@
             a

            -b
            +c
        """)
        out = diff_apply._clean_diff_lines(text, strict=True)
        # The blank line is a valid context line within the hunk.
        assert "" in out


# ── _parse_hunk_header ───────────────────────────────────────────────────────

class TestParseHunkHeader:
    def test_full_form(self):
        assert diff_apply._parse_hunk_header("@@ -10,5 +12,7 @@") == (10, 5, 12, 7)

    def test_counts_default_to_one_when_omitted(self):
        assert diff_apply._parse_hunk_header("@@ -10 +12 @@") == (10, 1, 12, 1)
        assert diff_apply._parse_hunk_header("@@ -10,3 +12 @@") == (10, 3, 12, 1)

    def test_leading_whitespace_tolerated(self):
        assert diff_apply._parse_hunk_header("   @@ -1 +1 @@") == (1, 1, 1, 1)

    def test_non_hunk_returns_none(self):
        assert diff_apply._parse_hunk_header("not a header") is None
        assert diff_apply._parse_hunk_header("@@ malformed") is None


# ── _recount_hunks ───────────────────────────────────────────────────────────

class TestRecountHunks:
    def test_recomputes_counts_from_body(self):
        lines = [
            "@@ -1,99 +1,99 @@",   # bogus counts
            " ctx",
            "-old",
            "+new",
            "+extra",
        ]
        out = diff_apply._recount_hunks(lines)
        # old side: ctx + old = 2 ; new side: ctx + new + extra = 3
        assert out[0] == "@@ -1,2 +1,3 @@"

    def test_skips_no_newline_meta_line_in_count(self):
        lines = [
            "@@ -1,99 +1,99 @@",
            " ctx",
            "+last",
            "\\ No newline at end of file",
        ]
        out = diff_apply._recount_hunks(lines)
        # The \ meta line is NOT counted; new side = ctx + last = 2
        assert out[0] == "@@ -1,1 +1,2 @@"
        # And the meta line is preserved in output
        assert out[-1] == "\\ No newline at end of file"

    def test_stops_at_next_hunk_or_file_header(self):
        lines = [
            "@@ -1,99 +1,99 @@",
            "+x",
            "@@ -5,99 +5,99 @@",
            "+y",
        ]
        out = diff_apply._recount_hunks(lines)
        assert out[0] == "@@ -1,0 +1,1 @@"
        assert out[2] == "@@ -5,0 +5,1 @@"

    def test_does_not_miscount_code_lines_starting_with_plusplus(self):
        # A real code line '+++thing' (no space) must count as an addition,
        # not be mistaken for the '+++ ' file header.
        lines = [
            "@@ -1,99 +1,99 @@",
            "+++b/c",   # no space — code addition
        ]
        out = diff_apply._recount_hunks(lines)
        assert out[0] == "@@ -1,0 +1,1 @@"

    def test_passes_through_non_hunk_lines(self):
        lines = ["diff --git a/x b/x", "--- a/x", "+++ b/x", "random"]
        out = diff_apply._recount_hunks(lines)
        assert out == lines


# ── _rewrite_patch_paths ─────────────────────────────────────────────────────

class TestRewritePatchPaths:
    def test_no_target_returns_unchanged(self):
        lines = ["diff --git a/x b/x", "--- a/x", "+++ b/x"]
        assert diff_apply._rewrite_patch_paths(lines, "") == lines

    def test_rewrites_diff_and_minus_plus_headers(self):
        lines = ["diff --git a/old b/old", "--- a/old", "+++ b/old"]
        out = diff_apply._rewrite_patch_paths(lines, "new/path.py")
        assert out[0] == "diff --git a/new/path.py b/new/path.py"
        assert out[1] == "--- a/new/path.py"
        assert out[2] == "+++ b/new/path.py"

    def test_preserves_dev_null_for_new_file(self):
        lines = ["--- /dev/null", "+++ b/old"]
        out = diff_apply._rewrite_patch_paths(lines, "created.py")
        assert out[0] == "--- /dev/null"
        assert out[1] == "+++ b/created.py"


# ── _upgrade_hunk_fragment ───────────────────────────────────────────────────

class TestUpgradeHunkFragment:
    def test_wraps_naked_hunk_into_full_diff(self):
        lines = ["@@ -1 +1 @@", "-a", "+b"]
        out = diff_apply._upgrade_hunk_fragment(lines, "mod.py")
        assert out[0] == "diff --git a/mod.py b/mod.py"
        assert out[1] == "--- a/mod.py"
        assert out[2] == "+++ b/mod.py"
        assert out[3:] == lines

    def test_noop_when_already_has_diff_header(self):
        lines = ["diff --git a/x b/x", "--- a/x", "+++ b/x", "@@ -1 +1 @@"]
        assert diff_apply._upgrade_hunk_fragment(lines, "y.py") == lines

    def test_noop_when_no_hunk(self):
        lines = ["some context"]
        assert diff_apply._upgrade_hunk_fragment(lines, "y.py") == lines

    def test_empty_target_returns_unchanged(self):
        lines = ["@@ -1 +1 @@"]
        assert diff_apply._upgrade_hunk_fragment(lines, "") == lines


# ── _clean_diff (integration of the above) ───────────────────────────────────

class TestCleanDiff:
    def test_empty_returns_empty(self):
        assert diff_apply._clean_diff("", "/repo", "x.py") == ""
        assert diff_apply._clean_diff("no diff here", "/repo", "x.py") == ""

    def test_fenced_diff_cleaned_and_paths_rewritten(self):
        text = "```diff\n@@ -1 +1 @@\n-a\n+b\n```"
        out = diff_apply._clean_diff(text, "/repo", "target.py")
        # Fenced naked hunk → upgraded + recounted
        assert out.startswith("diff --git a/target.py b/target.py")
        assert "@@ -1,1 +1,1 @@" in out


# ── _classify_git_apply_output ───────────────────────────────────────────────

class TestClassifyGitApplyOutput:
    @pytest.mark.parametrize("msg,expected", [
        ("error: corrupt patch at line 3", diff_apply.REASON_PATCH_MALFORMED),
        ("fatal: patch fragment without header", diff_apply.REASON_PATCH_MALFORMED),
        ("error: malformed patch", diff_apply.REASON_PATCH_MALFORMED),
        ("repository lacks the necessary blob", diff_apply.REASON_PATCH_MALFORMED),
        ("can't find file to patch", diff_apply.REASON_PATH_INVALID),
        ("No such file or directory", diff_apply.REASON_PATH_INVALID),
        ("error: patch failed: x.py:42", diff_apply.REASON_CONFLICT),
        ("hunk failed at 10", diff_apply.REASON_CONFLICT),
        ("Patch does not apply", diff_apply.REASON_CONFLICT),
        ("some unrecognized message", diff_apply.REASON_UNKNOWN),
        ("", diff_apply.REASON_UNKNOWN),
    ])
    def test_classification(self, msg, expected):
        assert diff_apply._classify_git_apply_output(msg) == expected

    def test_case_insensitive(self):
        assert diff_apply._classify_git_apply_output("CORRUPT PATCH") == diff_apply.REASON_PATCH_MALFORMED


# ── _extract_files_from_git_apply_output ─────────────────────────────────────

class TestExtractFilesFromGitApplyOutput:
    def test_empty_returns_empty(self):
        assert diff_apply._extract_files_from_git_apply_output("") == []
        assert diff_apply._extract_files_from_git_apply_output(None) == []  # type: ignore[arg-type]

    def test_extracts_from_patch_failed(self):
        out = "error: patch failed: src/app.py:42"
        assert diff_apply._extract_files_from_git_apply_output(out) == ["src/app.py"]

    def test_extracts_from_checking_applying(self):
        out = "Checking patch src/a.py...\nApplying patch src/a.py..."
        assert diff_apply._extract_files_from_git_apply_output(out) == ["src/a.py"]

    def test_strips_a_b_prefixes_and_quotes(self):
        out = 'error: "b/src/x.py": No such file'
        assert diff_apply._extract_files_from_git_apply_output(out) == ["src/x.py"]

    def test_drops_dev_null(self):
        out = "Applying patch /dev/null..."
        assert diff_apply._extract_files_from_git_apply_output(out) == []

    def test_dedupes_and_sorts(self):
        out = "Checking patch b/c.py...\npatch failed: a/c.py:1"
        # both normalize to c.py
        assert diff_apply._extract_files_from_git_apply_output(out) == ["c.py"]


# ── _is_probably_binary_file ─────────────────────────────────────────────────

class TestIsProbablyBinaryFile:
    def test_text_file_not_binary(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("hello world")
        assert diff_apply._is_probably_binary_file(p, 1024) is False

    def test_null_byte_is_binary(self, tmp_path):
        p = tmp_path / "b.dat"
        p.write_bytes(b"hello\x00world")
        assert diff_apply._is_probably_binary_file(p, 1024) is True

    def test_missing_file_returns_false(self, tmp_path):
        # Exception path → False (not crash)
        assert diff_apply._is_probably_binary_file(tmp_path / "nope", 1024) is False


# ── _has_conflict_markers ────────────────────────────────────────────────────

class TestHasConflictMarkers:
    def _write(self, tmp_path, text):
        p = tmp_path / "f.py"
        p.write_text(text)
        return p

    def test_full_conflict_block_detected(self, tmp_path):
        p = self._write(
            tmp_path,
            "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> branch\n",
        )
        assert diff_apply._has_conflict_markers(p, 0) is True

    def test_single_marker_alone_not_conflict(self, tmp_path):
        # A real git conflict always has all three markers in order.
        # A lone marker (e.g. Markdown setext '=======' underline) must NOT fire.
        for marker in ("<" * 7, "=" * 7, ">" * 7):
            p = self._write(tmp_path, f"{marker} branch\n")
            assert diff_apply._has_conflict_markers(p, 0) is False, marker

    def test_markdown_setext_heading_not_conflict(self, tmp_path):
        p = self._write(tmp_path, "Title\n=======\n\nbody text\n")
        assert diff_apply._has_conflict_markers(p, 0) is False

    def test_markers_out_of_order_not_conflict(self, tmp_path):
        p = self._write(tmp_path, ">>>>>>> x\n=======\n<<<<<<< y\n")
        assert diff_apply._has_conflict_markers(p, 0) is False

    def test_plain_text_no_markers(self, tmp_path):
        p = self._write(tmp_path, "x = '======='\nprint('<<<<<<< not a marker')\n")
        # The patterns are line-anchored, so these inline occurrences don't fire.
        assert diff_apply._has_conflict_markers(p, 0) is False

    def test_marker_with_leading_whitespace_not_detected(self, tmp_path):
        # Leading spaces before the markers → not real git conflict markers,
        # even when the full triplet appears in order.
        p = self._write(
            tmp_path,
            "  <<<<<<< HEAD\n  =======\n  >>>>>>> branch\n",
        )
        assert diff_apply._has_conflict_markers(p, 0) is False

    def test_setext_heading_before_conflict_still_detected(self, tmp_path):
        # Regression: a Markdown setext '=======' heading appearing BEFORE a
        # real conflict block must NOT mask the genuine block. The first-match
        # approach picked the heading as the separator, breaking open<sep<close
        # ordering → false negative (conflict markers silently left in file).
        p = self._write(
            tmp_path,
            "Title\n=======\n\n<<<<<<< HEAD\nfoo\n=======\nbar\n>>>>>>> feature\n",
        )
        assert diff_apply._has_conflict_markers(p, 0) is True

    def test_multiple_conflict_blocks_detected(self, tmp_path):
        p = self._write(
            tmp_path,
            "<<<<<<< HEAD\na\n=======\nb\n>>>>>>> branch\n"
            "<<<<<<< HEAD\nc\n=======\nd\n>>>>>>> branch\n",
        )
        assert diff_apply._has_conflict_markers(p, 0) is True

    def test_missing_file_returns_false(self, tmp_path):
        assert diff_apply._has_conflict_markers(tmp_path / "nope", 0) is False


# ── _resolve_inside_repo_path ────────────────────────────────────────────────

class TestResolveInsideRepoPath:
    def test_resolves_inside(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "sub").mkdir(parents=True)
        p = diff_apply._resolve_inside_repo_path(repo, "sub/a.py")
        assert p == (repo / "sub" / "a.py").resolve()

    def test_rejects_path_traversal(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(ValueError, match="path_outside_repo"):
            diff_apply._resolve_inside_repo_path(repo, "../../etc/passwd")


# ── _rollback (report + mixed tracked/new pathspec regression) ──────────────

class TestRollbackReport:
    """Regression tests for the _rollback report dict (Design 10).

    The critical scenario: touched_files mixing a modified tracked file with a
    patch-created NEW file. Passing both to a single `git restore` makes git
    abort on the unknown pathspec, silently restoring NOTHING. _rollback must
    restore pre-existing files separately and verify the result.
    """

    def _init_repo(self, tmp_path):
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args):
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                cwd=str(repo), check=True, capture_output=True,
            )

        git("init", "-q")
        (repo / "a.txt").write_text("v1\n")
        git("add", "a.txt")
        git("commit", "-qm", "init")
        return repo

    def test_mixed_tracked_and_new_file_rollback(self, tmp_path):
        repo = self._init_repo(tmp_path)
        # Simulate a failed apply: tracked file modified + new file created
        (repo / "a.txt").write_text("v2-from-patch\n")
        (repo / "b.txt").write_text("new-from-patch\n")

        snapshot = {"pre_untracked": set(), "pre_exists": {"a.txt": True, "b.txt": False}}
        report = diff_apply._rollback(repo, ["a.txt", "b.txt"], snapshot=snapshot)

        assert report["attempted"] is True
        # Tracked file restored despite the new-file pathspec (the old bug
        # aborted the whole `git restore`, leaving a.txt at v2).
        assert (repo / "a.txt").read_text() == "v1\n"
        # New file removed
        assert not (repo / "b.txt").exists()
        assert report["verified"] is True
        assert report["remaining_dirty"] == []
        assert report["restore_failed"] == []

    def test_report_flags_remaining_dirty_on_partial_failure(self, tmp_path):
        repo = self._init_repo(tmp_path)
        (repo / "a.txt").write_text("v2\n")

        # Wrong snapshot claims a.txt is a NEW file -> restore is skipped and
        # deletion is attempted; verification must surface the inconsistency
        # instead of reporting a clean rollback.
        snapshot = {"pre_untracked": set(), "pre_exists": {"a.txt": False}}
        report = diff_apply._rollback(repo, ["a.txt"], snapshot=snapshot)

        assert report["verified"] is False
        assert "a.txt" in " ".join(report["remaining_dirty"])


# ── _count_hunk_body: empty line counting (Bug 1 regression test) ────────────

class TestCountHunkBodyEmptyLines:
    """Bug 1: _count_hunk_body must count empty lines as context (like git apply)."""

    def test_empty_line_inside_hunk_is_counted_as_context(self):
        # A hunk with a blank context line between a and the deletion.
        # git apply counts blank lines as context=1 (old+1, new+1).
        lines = [
            "@@ -1,2 +1,3 @@",
            " a",
            "",       # blank context line — MUST be counted
            "+new",
            " b",
        ]
        result = diff_apply._count_hunk_body(lines, 0)
        assert result is not None
        end, actual_old, actual_new, claimed_old, claimed_new = result
        # old: ' a' + '' + ' b' = 3 ; new: ' a' + '' + '+new' + ' b' = 4
        assert actual_old == 3
        assert actual_new == 4
        assert end == 5  # consumed all lines

    def test_whitespace_only_line_counted_as_context(self):
        lines = [
            "@@ -1,1 +1,2 @@",
            "   ",     # whitespace-only — valid context
            "+new",
        ]
        result = diff_apply._count_hunk_body(lines, 0)
        assert result is not None
        _, actual_old, actual_new, _, _ = result
        assert actual_old == 1  # the whitespace line
        assert actual_new == 2  # whitespace + new

    def test_recount_with_empty_lines_produces_git_apply_compatible_header(self):
        # Exact reproduction of the reported bug: body [' a', '', '+new', ' b']
        # with a bogus claimed header. The recount must produce @@ -1,3 +1,4 @@
        # (matching git apply's own counting) not @@ -1,2 +1,3 @@.
        lines = [
            "@@ -1,99 +1,99 @@",
            " a",
            "",
            "+new",
            " b",
        ]
        out = diff_apply._recount_hunks(lines)
        assert out[0] == "@@ -1,3 +1,4 @@"


# ── _count_hunk_body: early-stop fix (Bug 2 regression test) ─────────────────

class TestCountHunkBodyEarlyStop:
    """Bug 2: _count_hunk_body must NOT truncate valid trailing context when
    claimed counts are exhausted. LLM claimed counts are unreliable — the
    recount exists precisely because of this."""

    def test_trailing_context_preserved_after_claimed_counts_met(self):
        # Claimed: @@ -1,1 +1,1 @@  but body has 3 actual lines.
        # The old early-stop would truncate after the first line.
        lines = [
            "@@ -1,1 +1,1 @@",
            " ctx1",
            " ctx2",   # trailing context — must not be truncated
            " ctx3",   # more trailing context
        ]
        result = diff_apply._count_hunk_body(lines, 0)
        assert result is not None
        end, actual_old, actual_new, claimed_old, claimed_new = result
        assert claimed_old == 1
        assert claimed_new == 1
        # All trailing context must be consumed (old early-stop would give end=2)
        assert actual_old == 3
        assert actual_new == 3
        assert end == 4

    def test_stops_at_non_context_line_after_claimed_counts_met(self):
        # After claimed counts are met, only context lines (" "-prefix, "\ ",
        # blank) are absorbed.  "+"/"-" lines are a hunk boundary stop.
        lines = [
            "@@ -1,1 +1,1 @@",    # 0
            " ctx1",               # 1 — consumed, meets claimed (1,1)
            "+extra",              # 2 — "+" after claimed → stop (not context)
            "more context",        # 3 — would be context but already stopped
        ]
        result = diff_apply._count_hunk_body(lines, 0)
        assert result is not None
        end, actual_old, actual_new, _, _ = result
        assert end == 2  # body consumed lines[1:2] = [" ctx1"]
        assert actual_old == 1  # ctx1 only
        assert actual_new == 1  # ctx1 only
    def test_multifile_bare_header_not_absorbed_by_prior_hunk(self):
            # Bug 3+2 interaction regression: bare "--- f2.py" / "+++ f2.py"
            # headers after a hunk must NOT be absorbed as deletion/addition lines
            # by the prior hunk's _count_hunk_body.
            lines = [
                "@@ -1,2 +1,2 @@",    # 0 — hunk 1 header
                " a",                   # 1
                " b",                   # 2
                "--- f2.py",            # 3 — bare header, must NOT be consumed as -
                "+++ f2.py",            # 4 — bare header, must NOT be consumed as +
                "@@ -1,1 +1,1 @@",     # 5 — hunk 2 header
                "-old",                 # 6
                "+new",                 # 7
            ]
            result = diff_apply._count_hunk_body(lines, 0)
            assert result is not None
            end, actual_old, actual_new, claimed_old, claimed_new = result
            assert claimed_old == 2
            assert claimed_new == 2
            assert end == 3  # consumed lines[1:3] = [" a", " b"]; "--- f2.py" is boundary
            assert actual_old == 2
            assert actual_new == 2

            # Hunk 2 must also parse correctly (boundary check uses bare pair detection)
            result2 = diff_apply._count_hunk_body(lines, 5)
            assert result2 is not None
            end2, _, _, _, _ = result2
            assert end2 == 8  # consumed lines[6:8] = ["-old", "+new"]
    def test_multifile_bare_header_before_claimed_counts_met(self):
            # Bare header pair appears BEFORE claimed counts are met (inflated
            # claimed count scenario). Must still be recognized as boundary via
            # the ---/+++ pair check, not consumed as deletion/addition lines.
            lines = [
                "@@ -1,50 +1,50 @@",    # 0 — inflated claimed (50 old, 50 new)
                " a",                   # 1
                " b",                   # 2
                "--- f2.py",            # 3 — bare header, must be boundary
                "+++ f2.py",            # 4
                "@@ -1,1 +1,1 @@",     # 5 — hunk 2 header
                "-old",
                "+new",
            ]
            result = diff_apply._count_hunk_body(lines, 0)
            assert result is not None
            end, actual_old, actual_new, _, _ = result
            assert end == 3  # consumed lines[1:3] only; "--- f2.py" is boundary
            assert actual_old == 2
            assert actual_new == 2


# ── _clean_diff_lines: ---/+++ pair without a/b/ prefix (Bug 3 regression) ──

class TestCleanDiffLinesBareHeaders:
    """Bug 3: LLM often emits ``--- f.py`` / ``+++ f.py`` without the a/ or b/
    prefix. The strict cleaner must detect these as a consecutive pair and
    preserve them, rather than dropping them as preamble noise."""

    def test_bare_header_pair_preserved(self):
        text = "--- f.py\n+++ f.py\n@@ -1,3 +1,3 @@\n ctx\n-old\n+new\n ctx2"
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "--- f.py" in joined
        assert "+++ f.py" in joined
        assert "+new" in joined

    def test_bare_header_pair_with_hunk_only_fragment(self):
        # Naked hunk preceded by bare ---/+++ pair (no diff --git header)
        text = "--- src/app.py\n+++ src/app.py\n@@ -1 +1 @@\n-a\n+b"
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "--- src/app.py" in joined
        assert "+++ src/app.py" in joined
        assert "-a" in joined
        assert "+b" in joined

    def test_lone_bare_minus_without_plus_is_dropped(self):
        # A bare "--- comment" without a following "+++ " is not a file header
        # (could be SQL/Lua comment preamble). Must be dropped.
        text = "--- some preamble\n@@ -1 +1 @@\n-a\n+b"
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "--- some preamble" not in joined
        assert "-a" in joined  # hunk still preserved

    def test_canonical_a_b_prefix_still_works(self):
        # The standard "--- a/f.py" / "+++ b/f.py" must still work.
        text = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "--- a/x.py" in joined
        assert "+++ b/x.py" in joined

    def test_mixed_prefix_bare_plus_kept_when_paired(self):
        # "--- a/f.py" (canonical) followed by "+++ f.py" (bare) — the bare
        # plus is kept because the preceding kept line is a --- header.
        text = "--- a/x.py\n+++ x.py\n@@ -1 +1 @@\n-a\n+b"
        out = diff_apply._clean_diff_lines(text, strict=True)
        joined = "\n".join(out)
        assert "--- a/x.py" in joined
        assert "+++ x.py" in joined
