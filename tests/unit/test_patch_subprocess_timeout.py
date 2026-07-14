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

import diff_apply  # noqa: E402
import services.patch_helpers as patch_helpers  # noqa: E402


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

    def test_git_status_porcelain_returns_empty_on_timeout(self, monkeypatch):
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_porcelain(Path("/tmp")) == ""

    def test_git_status_porcelain_untracked_false_on_timeout(self, monkeypatch):
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_porcelain(Path("/tmp"), include_untracked=False) == ""

    def test_git_status_untracked_returns_empty_set_on_timeout(self, monkeypatch):
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        assert diff_apply._git_status_untracked(Path("/tmp")) == set()

    def test_rollback_survives_timeout(self, monkeypatch, tmp_path):
        """_rollback must not raise when every git subprocess times out."""
        monkeypatch.setattr(diff_apply.subprocess, "run", _timeout_raiser())
        report = diff_apply._rollback(tmp_path, ["x.py"], snapshot={})
        # Timeout during restore -> attempted True, verified False (best-effort)
        assert isinstance(report, dict)
        assert report.get("verified") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
