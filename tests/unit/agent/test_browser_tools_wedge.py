"""Tests for browser_action hard-timeout wedge recovery and screenshot placement.

Regression coverage:

  - A hard timeout on the single-thread browser executor used to wedge every
    subsequent ``browser_action`` (the stuck worker blocks the only thread).
    ``_reset_browser_on_wedge`` abandons the wedged worker and recreates the
    executor so the session recovers.
  - Screenshots are written under ``.asicode/`` (not ``<repo_root>/screenshots``)
    so the user's working tree is not polluted.
"""
from __future__ import annotations

import os
import threading

import external_llm.agent.tool_handlers.browser_tools as browser_tools
from external_llm.agent.tool_handlers.browser_tools import BrowserActionToolsMixin


class _Host(BrowserActionToolsMixin):
    """Minimal concrete host with a stubbed navigate handler."""

    repo_root = "."

    def __init__(self) -> None:
        self.handler_threads: list[int] = []

    def _make_result(self, ok, content, error=None, metadata=None):
        return {"ok": ok, "content": content, "error": error}

    def _browser_navigate(self, args):  # override real handler (no Playwright needed)
        self.handler_threads.append(threading.get_ident())
        return self._make_result(ok=True, content="navigated")


class _HungFuture:
    def result(self, timeout=None):
        raise browser_tools._FutureTimeout()


class _WedgedExecutor:
    """Stand-in for an executor whose worker is stuck inside a Playwright call."""

    def __init__(self) -> None:
        self.shutdown_called = False
        self.cancel_futures_seen = None

    def submit(self, fn):
        return _HungFuture()

    def shutdown(self, wait=False, cancel_futures=False):
        self.shutdown_called = True
        self.cancel_futures_seen = cancel_futures


# ── screenshot placement ────────────────────────────────────────────────

def test_screenshot_dir_under_asicode(tmp_path):
    host = _Host()
    host.repo_root = str(tmp_path)
    d = host._screenshot_dir()
    assert d.startswith(str(tmp_path))
    assert os.path.join(".asicode", "screenshots") in d
    assert os.path.isdir(d)
    # Must NOT be the old <repo_root>/screenshots location.
    assert d != os.path.join(str(tmp_path), "screenshots")


# ── wedge recovery ──────────────────────────────────────────────────────

def test_reset_browser_on_wedge_clears_state_and_recreates_executor(monkeypatch):
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", object())
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", object())
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", object())

    fake_before = _WedgedExecutor()
    monkeypatch.setattr(browser_tools, "_BROWSER_EXECUTOR", fake_before)

    browser_tools._reset_browser_on_wedge()

    # Class-level browser refs dropped (so the next call re-initializes).
    assert BrowserActionToolsMixin._browser is None
    assert BrowserActionToolsMixin._page is None
    assert BrowserActionToolsMixin._playwright is None
    # The wedged executor was shut down (cancel_futures best-effort).
    assert fake_before.shutdown_called
    assert fake_before.cancel_futures_seen is True
    # A new executor took its place.
    after = browser_tools._BROWSER_EXECUTOR
    assert after is not fake_before
    after.shutdown(wait=False)  # teardown the executor created during the test


def test_browser_action_recovers_after_hard_timeout(monkeypatch):
    """After a hard-timeout wedge, the very next browser_action must succeed.

    Previously the wedged single worker queued every later submit behind the
    hung call, so all subsequent actions timed out too (permanent wedging).
    """
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", object())
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", object())

    wedged = _WedgedExecutor()
    monkeypatch.setattr(browser_tools, "_BROWSER_EXECUTOR", wedged)

    host = _Host()
    res = host._tool_browser_action({"action": "navigate", "url": "http://x"})

    # The wedged call reports a recoverable error, not a silent hang.
    assert not res["ok"]
    assert "reset" in res["error"].lower()
    assert wedged.shutdown_called
    assert BrowserActionToolsMixin._browser is None
    assert BrowserActionToolsMixin._page is None
    assert browser_tools._BROWSER_EXECUTOR is not wedged

    new_executor = browser_tools._BROWSER_EXECUTOR
    try:
        # The next action runs on the fresh executor and succeeds.
        host.handler_threads = []
        res2 = host._tool_browser_action({"action": "navigate", "url": "http://x"})
        assert res2["ok"], res2
        assert len(host.handler_threads) == 1
    finally:
        new_executor.shutdown(wait=False)


# ── atexit handler hygiene ──────────────────────────────────────────────

def test_wedge_recovery_does_not_accumulate_atexit_handlers(monkeypatch):
    """Each _reset_browser_on_wedge must NOT re-register an atexit handler.

    The old _new_browser_executor() called atexit.register on every recreation,
    so N wedges piled up N dead handlers (each retaining a reference to its
    stuck-worker executor until process exit). Teardown now lives in ONE
    module-level handler (_shutdown_browser_executor_at_exit) registered at
    import; recovery must not touch atexit at all.
    """
    registrations = []
    monkeypatch.setattr(
        browser_tools.atexit, "register", lambda *a, **k: registrations.append(a)
    )

    # Drive several wedge recoveries; each recreates the executor.
    created = []
    for _ in range(5):
        monkeypatch.setattr(browser_tools, "_BROWSER_EXECUTOR", _WedgedExecutor())
        browser_tools._reset_browser_on_wedge()
        created.append(browser_tools._BROWSER_EXECUTOR)

    # No atexit registration happened during recovery.
    assert registrations == [], registrations

    # Teardown the executors created during the test.
    for ex in created:
        ex.shutdown(wait=False)
