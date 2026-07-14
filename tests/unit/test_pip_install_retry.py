"""Behavioral guards for the shared in-REPL pip installer (``asi._pip_install``).

Covers two classes of bug the ``/claude`` one-shot SDK install surfaced:

1. PEP 668 externally-managed retry losing the caller's explicit ``label`` —
   on Homebrew/system Python the retry *is* the normal path, so the regression
   hid the nice label for the entire real install duration.
2. A successful in-process install not invalidating the import finder cache,
   so a re-check via ``find_spec`` could read a just-installed package as
   "still missing" (stale FileFinder mtime cache).
"""

import io

import asi


class _FakeCompleted:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _force_non_tty(monkeypatch):
    """Replace sys.stderr with a non-tty stream so the label surfaces via
    ``_print`` (the tty branch writes a spinner to stderr directly)."""
    monkeypatch.setattr(asi.sys, "stderr", io.StringIO())


def _silence_print(monkeypatch):
    monkeypatch.setattr(asi, "_print", lambda *a, **k: None)


class TestPep668RetryPreservesLabel:
    def test_retry_threads_explicit_label_through(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            # First attempt: PEP 668 externally-managed failure (the common path
            # on Homebrew Python). Second attempt: success.
            if len(calls) == 1:
                return _FakeCompleted(1, stderr="externally-managed-environment")
            return _FakeCompleted(0)

        monkeypatch.setattr(asi.subprocess, "run", fake_run)
        _force_non_tty(monkeypatch)

        recorded = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: recorded.append(a[0]))

        ok = asi._pip_install(["-e", "/x[collaborate]"], label="claude_agent_sdk")

        assert ok is True
        assert len(calls) == 2  # retry actually happened
        # Both invocations surface the explicit label — before the fix the retry
        # reverted to the pkgs[0] default ("-e (+1)").
        install_lines = [p for p in recorded if "Installing" in p]
        assert install_lines == [
            "  Installing claude_agent_sdk …",
            "  Installing claude_agent_sdk …",
        ], install_lines

    def test_no_label_default_is_still_pkgs0(self, monkeypatch):
        """Callers that don't pass label keep the documented pkgs[0] default."""
        monkeypatch.setattr(asi.subprocess, "run", lambda cmd, **kw: _FakeCompleted(0))
        _force_non_tty(monkeypatch)
        recorded = []
        monkeypatch.setattr(asi, "_print", lambda *a, **k: recorded.append(a[0]))

        assert asi._pip_install(["sentence-transformers", "faiss-cpu"]) is True
        assert any("Installing sentence-transformers (+1)" in p for p in recorded)


class TestInvalidateCachesOnSuccess:
    def test_success_invalidates_import_cache(self, monkeypatch):
        monkeypatch.setattr(asi.subprocess, "run", lambda cmd, **kw: _FakeCompleted(0))
        _force_non_tty(monkeypatch)
        _silence_print(monkeypatch)

        invalidated = []
        import importlib

        monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))

        assert asi._pip_install(["some-pkg"]) is True
        assert invalidated == [True]

    def test_failure_does_not_invalidate(self, monkeypatch):
        monkeypatch.setattr(
            asi.subprocess, "run", lambda cmd, **kw: _FakeCompleted(1, stderr="nope")
        )
        _force_non_tty(monkeypatch)
        _silence_print(monkeypatch)

        invalidated = []
        import importlib

        monkeypatch.setattr(importlib, "invalidate_caches", lambda: invalidated.append(True))

        assert asi._pip_install(["some-pkg"]) is False
        assert invalidated == []
