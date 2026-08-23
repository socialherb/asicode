"""Regression tests: context-mismatch auto re-read + bounded retry (write_tools).

When edit_text fails with ``search_string_mismatch`` (old_string did not match
what we read at entry), the handler re-reads the file from disk ONCE:

* fresh content != entry content  -> the file moved under us (parallel editor /
  another session on the same checkout) -> the whole edit list is retried ONCE
  against the fresh content (``_reread_retry`` recursion guard, depth-1 bound);
* fresh content == entry content -> unchanged; a retry would be deterministic
  and pointless -> the normal failure is returned, enriched with a
  fresh-content head snippet when near-match hinting found nothing, so the LLM
  can craft a correct old_string without an extra read round-trip.

apply_patch's failure path attaches the same snippet (``_patch_failure_snippet``)
but deliberately does NOT auto-retry (its repair ladder already ran; a retry
could double-apply a partially-landed patch).

The retry/snippet events are surfaced via metadata (``reread_retried`` /
``reread_retry_success`` / ``reread_snippet``) so tool_failure_log records them
and failure-pattern analysis can measure the mechanism's effect.

Run: pytest tests/unit/agent/test_write_tools_reread_retry.py -v
"""

from __future__ import annotations

import pytest

from external_llm.agent.tool_handlers.write_tools import WriteToolsMixin
from external_llm.agent.tool_registry import ToolResult


class _Harness(WriteToolsMixin):
    """Minimal concrete host (mirrors test_write_tools_bugfixes._Harness)."""

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


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


# ── Auto re-read + bounded retry ────────────────────────────────────────────


class TestRereadRetry:
    def test_retry_once_when_file_changed_under_us(self, harness, tmp_path):
        """File changed between entry read and failure -> retry succeeds once."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        orig_apply = harness._apply_one_edit_text
        calls = {"n": 0}

        def _fake_apply(content, file_path, old_string, new_string, replace_all, scope=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate a parallel editor landing a change between our entry
                # read (alpha/beta/gamma) and the failure.
                target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
                return {
                    "ok": False,
                    "error": f"old_string not found in {file_path}\n",
                    "metadata": {
                        "matched": False,
                        "near_match": False,
                        "failure_class": "search_string_mismatch",
                    },
                }
            return orig_apply(content, file_path, old_string, new_string, replace_all, scope)

        harness._apply_one_edit_text = _fake_apply
        result = harness._tool_edit_text(
            {
                "file_path": "t.txt",
                "old_string": "beta",
                "new_string": "BETA",
            }
        )

        assert result.ok, result.error
        assert calls["n"] == 2, "exactly one retry after the forced failure"
        assert result.metadata.get("reread_retried") is True
        assert result.metadata.get("reread_retry_success") is True
        # The retry ran against the FRESH content and landed.
        out = target.read_text(encoding="utf-8")
        assert "BETA" in out
        assert "beta\n" not in out

    def test_no_retry_when_file_unchanged_snippet_attached(self, harness, tmp_path):
        """Unchanged file -> no retry; failure carries a fresh-content snippet."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        orig_apply = harness._apply_one_edit_text
        calls = {"n": 0}

        def _counting_apply(content, file_path, old_string, new_string, replace_all, scope=None):
            calls["n"] += 1
            return orig_apply(content, file_path, old_string, new_string, replace_all, scope)

        harness._apply_one_edit_text = _counting_apply
        result = harness._tool_edit_text(
            {
                "file_path": "t.txt",
                "old_string": "zzz",  # absent AND no near match
                "new_string": "YYY",
            }
        )

        assert not result.ok
        assert calls["n"] == 1, "no retry when the file did not change"
        assert result.metadata.get("reread_retried") is None
        assert result.metadata.get("reread_snippet") is True
        assert "old_string not found" in result.error
        assert "current file content" in result.error
        assert "│1│ alpha" in result.error

    def test_retry_bounded_to_once(self, harness, tmp_path):
        """Retry that still fails must NOT trigger a second retry."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        orig_apply = harness._apply_one_edit_text
        calls = {"n": 0}

        def _fake_apply(content, file_path, old_string, new_string, replace_all, scope=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # The parallel editor REPLACES the whole file: old_string ("beta")
                # is gone, so even the retry against fresh content must fail.
                target.write_text("x\ny\n", encoding="utf-8")
                return {
                    "ok": False,
                    "error": f"old_string not found in {file_path}\n",
                    "metadata": {
                        "matched": False,
                        "near_match": False,
                        "failure_class": "search_string_mismatch",
                    },
                }
            return orig_apply(content, file_path, old_string, new_string, replace_all, scope)

        harness._apply_one_edit_text = _fake_apply
        result = harness._tool_edit_text(
            {
                "file_path": "t.txt",
                "old_string": "beta",
                "new_string": "BETA",
            }
        )

        assert not result.ok
        assert calls["n"] == 2, "retry happened once; no second retry (depth-1 bound)"
        assert result.metadata.get("reread_retried") is True
        assert result.metadata.get("reread_retry_success") is None

    def test_no_snippet_when_near_match_present(self, harness, tmp_path):
        """A near-match hint already shows file content -> no snippet duplicate."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        def _near_fail_apply(content, file_path, old_string, new_string, replace_all, scope=None):
            return {
                "ok": False,
                "error": f"old_string not found in {file_path}\n  [did you mean] ...\n",
                "metadata": {
                    "matched": False,
                    "near_match": True,
                    "failure_class": "search_string_mismatch",
                },
            }

        harness._apply_one_edit_text = _near_fail_apply
        result = harness._tool_edit_text(
            {
                "file_path": "t.txt",
                "old_string": "zzz",
                "new_string": "YYY",
            }
        )

        assert not result.ok
        assert result.metadata.get("reread_snippet") is None
        assert "current file content" not in result.error

    def test_batch_mode_retry_and_metadata(self, harness, tmp_path):
        """Batch mode shares the same re-read retry path."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        orig_apply = harness._apply_one_edit_text
        calls = {"n": 0}

        def _fake_apply(content, file_path, old_string, new_string, replace_all, scope=None):
            calls["n"] += 1
            if calls["n"] == 1:
                target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
                return {
                    "ok": False,
                    "error": f"old_string not found in {file_path}\n",
                    "metadata": {
                        "matched": False,
                        "near_match": False,
                        "failure_class": "search_string_mismatch",
                    },
                }
            return orig_apply(content, file_path, old_string, new_string, replace_all, scope)

        harness._apply_one_edit_text = _fake_apply
        result = harness._tool_edit_text(
            {
                "file_path": "t.txt",
                "edits": [
                    {"old_string": "beta", "new_string": "BETA"},
                    {"old_string": "gamma", "new_string": "GAMMA"},
                ],
            }
        )

        assert result.ok, result.error
        # First attempt fails (1 call), retry applies both edits (2 calls).
        assert calls["n"] == 3
        assert result.metadata.get("reread_retried") is True
        assert result.metadata.get("reread_retry_success") is True
        out = target.read_text(encoding="utf-8")
        assert "BETA" in out and "GAMMA" in out


# ── apply_patch stale-target snippet (no auto-retry) ───────────────────────


class TestPatchFailureSnippet:
    def test_snippet_attached_to_patch_failure(self, harness, tmp_path):
        """Failed apply_patch on a stale target gets the current file head."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        patch_text = (
            "diff --git a/t.txt b/t.txt\n--- a/t.txt\n+++ b/t.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BBB\n gamma\n"
        )
        result = harness._tool_apply_patch({"patch": patch_text, "path": "t.txt"})
        # The patch SHOULD apply cleanly here (context matches); the snippet
        # path is exercised directly via the helper below instead.
        assert result.ok, result.error

    def test_patch_failure_snippet_helper(self, harness, tmp_path):
        """Helper attaches the head of the primary target on failure."""
        target = tmp_path / "t.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        snip = harness._patch_failure_snippet("diff --git a/t.txt b/t.txt\n", "t.txt")
        assert "current file content" in snip
        assert "│1│ alpha" in snip
        assert "gamma" in snip

    def test_patch_failure_snippet_derives_target_from_patch(self, harness, tmp_path):
        """No path hint -> target derived from the patch's first file."""
        target = tmp_path / "other.txt"
        target.write_text("one\ntwo\n", encoding="utf-8")
        snip = harness._patch_failure_snippet(
            "diff --git a/other.txt b/other.txt\n--- a/other.txt\n+++ b/other.txt\n",
            None,
        )
        assert "current file content" in snip
        assert "│1│ one" in snip

    def test_patch_failure_snippet_missing_target_returns_empty(self, harness, tmp_path):
        """No readable target -> empty snippet (graceful degradation)."""
        assert harness._patch_failure_snippet("diff --git a/nope.txt b/nope.txt\n", None) == ""
        assert harness._patch_failure_snippet("not a diff at all", "ghost.txt") == ""
