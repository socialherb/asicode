"""Regression: subprocess calls in the HTTP patch-apply path must be bounded.

A hung git/python process (lock contention, NFS, pathological repo) must never
stall an HTTP /edit/run or /edit/apply request indefinitely. Every git/
py_compile subprocess in the apply path now passes ``timeout=``, and every
call site degrades gracefully on ``subprocess.TimeoutExpired`` (a
SubprocessError->Exception subclass caught by the surrounding handler) instead
of propagating an unhandled exception or hanging.

These are *behavioral* tests: they force a timeout and assert the public
helpers return their documented graceful value (no hang, no stray raise),
rather than asserting the ``timeout=`` kwarg is present (which would be a
source-contract grep blind to refactors). Monkeypatching uses plain callables,
not MagicMock, so a missing attribute on the fake surfaces immediately.

Run: pytest tests/unit/test_patch_subprocess_timeout.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# diff_apply.py and services/ live at / under the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import diff_apply
from services import patch_helpers


def _timeout_raiser():
    """Return a plain callable that always raises TimeoutExpired (no MagicMock)."""

    def _boom(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", "git")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=0.001)

    return _boom


class TestGitApplyCheckOnlyTimeout:
    def test_timeout_maps_to_check_exception(self, monkeypatch):
        """git_apply_check_only must return CHECK_EXCEPTION, not raise/hang."""
        monkeypatch.setattr(patch_helpers.subprocess, "run", _timeout_raiser())
        ok, out, taxonomy = patch_helpers.git_apply_check_only(
            "/nonexistent/repo", "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
        )
        assert ok is False
        assert taxonomy == "CHECK_EXCEPTION"
        assert isinstance(out, str)


class TestDiffApplyHelperTimeouts:
    """The apply-path git helpers must degrade gracefully on subprocess timeout."""

    def test_run_git_apply_returns_timeout_sentinel(self, monkeypatch):
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        rc, msg = diff_apply._run_git_apply(Path("/tmp"), ["--check"], "x")
        assert rc == -1
        assert "timeout" in msg.lower()

    def test_git_status_porcelain_returns_none_on_timeout(self, monkeypatch):
        """Timeout must yield None (unknown), NOT "" (verified clean): a hung
        git must never masquerade as a pristine tree for the 3-way gate."""
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_porcelain(Path("/tmp")) is None

    def test_git_status_porcelain_untracked_false_on_timeout(self, monkeypatch):
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_porcelain(Path("/tmp"), include_untracked=False) is None

    def test_is_worktree_clean_returns_none_on_timeout(self, monkeypatch):
        """The tri-state core: unverifiable cleanliness is None, not True."""
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._is_worktree_clean(Path("/tmp")) is None

    def test_git_status_untracked_returns_none_on_timeout(self, monkeypatch):
        """Timeout must yield None, NOT set(): an empty set would let rollback
        git clean user files that pre-existed the apply attempt."""
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_untracked(Path("/tmp")) is None

    def test_rollback_survives_timeout(self, monkeypatch, tmp_path):
        """_rollback must not raise when every git subprocess times out."""
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        report = diff_apply._rollback(tmp_path, ["x.py"], snapshot={})
        # Timeout during restore -> attempted True, verified False (best-effort)
        assert isinstance(report, dict)
        assert report.get("verified") is False

    def test_rollback_unknown_pre_untracked_skips_cleanup(self, monkeypatch, tmp_path):
        """pre_untracked=None (status timed out) must NOT git clean user files.

        Old contract collapsed timeout to set(), so a pre-apply snapshot taken
        during a hung git would make _rollback treat every untracked file as
        'newly created' and delete it. Fail-closed: skip the cleanup instead.
        """
        user_file = tmp_path / "user_untracked.txt"
        user_file.write_text("user data\n")
        monkeypatch.setattr(diff_apply, "_git_status_untracked", lambda repo: None)
        report = diff_apply._rollback(
            tmp_path,
            ["x.py"],
            snapshot={"pre_untracked": None, "pre_exists": {}},
        )
        assert user_file.exists(), "fail-closed rollback deleted a pre-existing untracked file"
        assert report.get("cleanup_skipped_reason") == "pre_untracked_unknown", report

    def test_apply_patch_3way_skipped_when_clean_state_unverifiable(self, monkeypatch, tmp_path):
        """A git-status timeout at the 3-way gate must refuse the merge.

        Old behavior: timeout collapsed to "" -> _is_worktree_clean() True ->
        no autostash -> `git apply --3way` runs against an unverifiable tree,
        potentially writing conflict markers into user-WIP files. New behavior:
        None -> fail closed with REASON_3WAY_SKIPPED_UNVERIFIABLE; the tree is
        untouched (--check is a dry run and nothing below it has run).
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
        (repo / "app.py").write_text("def greet(name):\n    return name\n\ndef add(a, b):\n    return a + b\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

        # Drift the context so `--check` fails with CONFLICT (rc=1) and the
        # 3-way gate is reached.
        patch = (
            "diff --git a/app.py b/app.py\n"
            "index abcdef0..1234567 100644\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -3,3 +3,3 @@\n"
            " \n"
            " def add_drifted(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 1\n"
        )

        # git status can't be verified (hung git): the gate must refuse 3-way.
        monkeypatch.setattr(diff_apply, "_is_worktree_clean", lambda repo: None)
        monkeypatch.setattr(diff_apply, "_git_status_porcelain", lambda repo, **kw: " M app.py")

        before = (repo / "app.py").read_text()
        ok, _msg, reason, d = diff_apply.apply_patch(str(repo), patch, file_path_hint="app.py", skip_3way=False)
        assert not ok
        assert reason == diff_apply.REASON_3WAY_SKIPPED_UNVERIFIABLE, (reason, d)
        assert d["used_strategy"] == "git-apply-3way-skipped-unverifiable", d
        assert d["rollback_performed"] is False, d
        assert d["rollback_skipped_reason"] == "worktree_clean_unverifiable", d
        # Diagnostics surface the timeout instead of collapsing to "".
        assert d["git_status_porcelain_before"] == " M app.py", d
        # --check is a dry run and 3-way never ran: the tree must be untouched.
        assert (repo / "app.py").read_text() == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
