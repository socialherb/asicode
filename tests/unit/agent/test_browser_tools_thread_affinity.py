"""Regression tests for browser_action thread affinity.

Playwright's sync API binds its driver to the thread that called
``sync_playwright().start()``; the browser/page objects may only be used from
that same thread. Tool calls, however, are dispatched on a shared
``ThreadPoolExecutor`` where ``browser_action`` (a read tool) runs unserialized
on any worker. Reusing the shared browser singleton from a different (or
already-exited) worker raised:

    error: cannot switch to a different thread (which happens to have exited)

The fix pins every Playwright call to one dedicated single-thread executor.
These tests lock in that affinity without requiring Playwright/Chromium.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import external_llm.agent.tool_handlers.browser_tools as browser_tools
from external_llm.agent.tool_handlers.browser_tools import (
    HAS_PLAYWRIGHT,
    BrowserActionToolsMixin,
)


class _Host(BrowserActionToolsMixin):
    """Minimal concrete host: stubs _make_result and records handler threads."""

    repo_root = "."

    def __init__(self) -> None:
        self.handler_threads: list[int] = []

    def _make_result(self, ok, content, error=None, metadata=None):
        return {"ok": ok, "content": content, "error": error}

    def _browser_navigate(self, args):  # override real handler
        self.handler_threads.append(threading.get_ident())
        return self._make_result(ok=True, content="navigated")


def test_browser_action_pins_handlers_to_single_thread(monkeypatch):
    """All handler executions run on one thread, regardless of caller thread.

    The affinity logic (dispatch onto the dedicated executor) is independent of
    Playwright, but ``_tool_browser_action`` short-circuits when Playwright is
    absent. Force the guard open so the stubbed handler actually runs.
    """
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    host = _Host()
    caller_threads: set[int] = set()

    def call(_):
        caller_threads.add(threading.get_ident())
        return host._tool_browser_action({"action": "navigate", "url": "http://x"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(40)))

    assert all(r["ok"] for r in results)
    # Callers genuinely spread across multiple pool workers...
    assert len(caller_threads) > 1
    # ...yet every Playwright handler invocation ran on exactly one thread.
    handler_threads = set(host.handler_threads)
    assert len(handler_threads) == 1, f"affinity broken: {handler_threads}"
    # And that thread is the dedicated browser worker, never a caller thread.
    assert handler_threads.isdisjoint(caller_threads)


def test_browser_action_serializes_concurrent_calls(monkeypatch):
    """Concurrent browser_action calls never overlap on the shared page."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    host = _Host()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _slow_navigate(args):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        # busy a moment so an unserialized impl would overlap here
        threading.Event().wait(0.01)
        with lock:
            active -= 1
        return host._make_result(ok=True, content="ok")

    host._browser_navigate = _slow_navigate  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(
            lambda _: host._tool_browser_action({"action": "navigate", "url": "http://x"}),
            range(20),
        ))

    assert max_active == 1, f"calls overlapped: max_active={max_active}"


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="Playwright not installed")
def test_real_browser_navigates_across_caller_threads():
    """End-to-end: a real browser drives cleanly when called from many threads.

    Without the fix this raises 'cannot switch to a different thread'.
    """
    class _RealHost(BrowserActionToolsMixin):
        repo_root = "."

        def _make_result(self, ok, content, error=None, metadata=None):
            return {"ok": ok, "content": content, "error": error}

    host = _RealHost()  # uses the real _browser_navigate handler
    data_url = "data:text/html,<title>Hi</title><body>hello</body>"

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(
                lambda _: host._tool_browser_action(
                    {"action": "navigate", "url": data_url, "timeout": 8000}
                ),
                range(6),
            ))
        assert all(r["ok"] for r in results), [r["error"] for r in results if not r["ok"]]
    finally:
        host._tool_browser_action({"action": "close"})
