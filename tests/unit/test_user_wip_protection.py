"""
Fix 2 (user-WIP protection): `git apply --check` is a NON-MUTATING dry-run.

Both `--check`-failure branches in diff_apply.apply_patch must NOT call
_rollback(), because _rollback() runs `git restore --staged --worktree` and
would silently revert user work-in-progress (pre-existing uncommitted edits)
back to HEAD — on a code path that never mutated the working tree.

  Path A (~line 795): `--check` fails with CONFLICT and skip_3way is True
                      (the freshly-edited / untracked / fake-SHA case).
  Path B (~line 881): `--check` fails with a NON-conflict reason
                      (PATCH_MALFORMED / PATH_INVALID / UNKNOWN).

The fix replaces `_rollback(...)` with a bare `_cleanup_reject_files(...)` in
both branches and reports `rollback_performed: False` plus
`rollback_skipped_reason: "check_is_dryrun_worktree_unchanged"`.

These tests prove the invariant: after a `--check`-only failure, user-WIP on
disk is byte-identical to before the call.

Run: pytest tests/unit/test_user_wip_protection.py -v
"""
from __future__ import annotations

import subprocess
import textwrap

import pytest


@pytest.fixture
def git_repo(tmp_path):
    """Minimal git repo with a committed app.py."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)

    (repo / "app.py").write_text(textwrap.dedent("""\
        def greet(name):
            msg = "Hello, " + name
            return msg


        def add(a, b):
            return a + b


        def multiply(a, b):
            return a * b
    """))
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    return repo


class TestUserWipProtectionOnCheckFailure:
    """`--check` never mutates the working tree, so neither failure branch may
    call _rollback() (which would `git restore --worktree` user-WIP to HEAD)."""

    # A git-conventional hunk whose context matches the committed fixture.
    _MATCHING_HUNK = (
        "diff --git a/app.py b/app.py\n"
        "index abcdef0..1234567 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -4,7 +4,7 @@ def greet(name):\n"
        " \n"
        " \n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b + 1\n"
        " \n"
        " \n"
        " def multiply(a, b):\n"
    )

    @staticmethod
    def _apply(git_repo, patch, skip_3way):
        from diff_apply import apply_patch

        return apply_patch(
            git_repo, patch, file_path_hint="app.py", skip_3way=skip_3way,
        )

    @staticmethod
    def _make_wip(git_repo):
        """Stage user work-in-progress on app.py WITHOUT git add (worktree != HEAD)."""
        app = git_repo / "app.py"
        original = app.read_text()
        wip = original.replace("def greet(name):", "def greet_v2(name):  # USER WIP")
        app.write_text(wip)
        return original, wip

    def test_skip3way_conflict_path_preserves_user_wip(self, git_repo):
        """Path A (~line 795): skip_3way + CONFLICT. The primary fix.

        The freshly-edited / untracked / fake-SHA case where the caller tells
        diff_apply to skip 3-way. Before Fix 2 this silently reverted user-WIP
        to HEAD via `git restore --worktree` inside _rollback().
        """
        _original, wip = self._make_wip(git_repo)
        app = git_repo / "app.py"
        assert "USER WIP" in app.read_text()

        # Drift the context so `--check` fails with CONFLICT (rc=1).
        drift = self._MATCHING_HUNK.replace(
            "@@ -4,7 +4,7 @@ def greet(name):",
            "@@ -4,7 +4,7 @@ WRONG_CONTEXT_FUNC",
        ).replace("def add(a, b):", "def add_drifted(a, b):", 1)

        ok, _m, reason, d = self._apply(git_repo, drift, skip_3way=True)
        assert not ok
        assert reason == "CONFLICT", d
        assert d["used_strategy"] == "git-apply-check-3way-skipped", d
        # Fix 2 signal: rollback was deliberately skipped (check is a dry-run).
        assert d["rollback_performed"] is False, d
        assert d["rollback_skipped_reason"] == "check_is_dryrun_worktree_unchanged", d

        # THE FIX: worktree was never mutated by `--check`, so WIP must survive.
        assert app.read_text() == wip, (
            "user-WIP destroyed by skip_3way --check path!"
        )
        assert "USER WIP" in app.read_text()

    def test_nonconflict_checkfailure_path_preserves_user_wip(self, git_repo, monkeypatch):
        """Path B (~line 881): `--check` fails with a NON-CONFLICT reason.

        In production this branch is rare: `_clean_diff` normalizes malformed
        patches (it strips invalid hunk-body lines, turning "corrupt patch" into
        a clean hunk) and missing files are caught earlier (MISSING_PATH_HINT).
        The realistic non-CONFLICT triggers are exotic --check failures such as
        a git timeout (REASON_UNKNOWN). To deterministically exercise this
        branch, we force the classifier to return a non-CONFLICT reason on a
        drift patch that genuinely fails --check (rc=1). This bypasses the
        3-way/autostash ladder and lands in the generic --check-failure return.
        """
        import diff_apply

        _original, wip = self._make_wip(git_repo)
        app = git_repo / "app.py"
        assert "USER WIP" in app.read_text()

        # Force --check failure to classify as non-CONFLICT -> skip 3-way/autostash
        # -> reach the generic --check-failure return (~line 881).
        monkeypatch.setattr(
            diff_apply, "_classify_git_apply_output",
            lambda out: diff_apply.REASON_PATCH_MALFORMED,
        )

        # Drift the context so `--check` genuinely fails (rc=1) with a CONFLICT-
        # style message; the forced classifier rebrands it PATCH_MALFORMED.
        drift = self._MATCHING_HUNK.replace(
            "@@ -4,7 +4,7 @@ def greet(name):",
            "@@ -4,7 +4,7 @@ WRONG_CONTEXT_FUNC",
        ).replace("def add(a, b):", "def add_drifted(a, b):", 1)

        ok, _m, reason, d = self._apply(git_repo, drift, skip_3way=False)
        assert not ok
        # Confirm we exercised the NON-conflict branch (path B), not 3-way.
        assert reason == diff_apply.REASON_PATCH_MALFORMED, d
        assert d["used_strategy"] == "git-apply-check", d
        assert d["rollback_performed"] is False, d
        assert d["rollback_skipped_reason"] == "check_is_dryrun_worktree_unchanged", d

        # THE FIX: no _rollback -> no `git restore --worktree` -> WIP survives.
        assert app.read_text() == wip, (
            "user-WIP destroyed by non-conflict --check path!"
        )
        assert "USER WIP" in app.read_text()
