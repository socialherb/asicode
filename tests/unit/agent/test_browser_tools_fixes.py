"""Regression tests for browser_action correctness fixes.

Covers:

  - ``_clamp_per_call_timeout_ms`` keeps every per-call Playwright timeout below
    the dedicated-executor hard ceiling, so a generous caller request (e.g.
    180000ms) resolves as a clean Playwright timeout instead of tripping the
    session-resetting wedge (``_reset_browser_on_wedge``).
  - navigate/extract metadata reports real content length (``length``) and the
    un-truncated total (``total_length``), not the marker-inflated ``len(text)``.
  - ``_get_browser`` stops the Playwright driver if ``chromium.launch()`` fails,
    so a launch failure does not leak a node driver process.

No real browser: actions run on the dedicated executor against a fake page, or
exercise module-level helpers / monkeypatched ``sync_playwright``.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.tool_handlers import browser_tools
from external_llm.agent.tool_handlers.browser_tools import BrowserActionToolsMixin


class _NavHost(BrowserActionToolsMixin):
    """Concrete host that uses the REAL navigate/extract handlers.

    Unlike the _Host in test_browser_tools_wedge.py (which overrides
    _browser_navigate), this host keeps the production handlers so the metadata
    + truncation path is exercised end-to-end via a fake page.
    """

    repo_root = "."

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


class _FakePage:
    """Minimal stand-in for a Playwright Page: inner_text/title/url/goto."""

    def __init__(self, body_text: str, title: str = "T", url: str = "https://x/") -> None:
        self._body = body_text
        self._title = title
        self._url = url

    def inner_text(self, selector: str) -> str:
        return self._body

    def title(self) -> str:
        return self._title

    @property
    def url(self) -> str:
        return self._url

    def goto(self, url, timeout=None, wait_until=None) -> None:
        self._url = url


# ── _clamp_per_call_timeout_ms ─────────────────────────────────────────


def test_clamp_per_call_timeout_ms_passes_normal_values():
    assert browser_tools._clamp_per_call_timeout_ms(30000) == 30000
    assert browser_tools._clamp_per_call_timeout_ms(1000) == 1000


def test_clamp_per_call_timeout_ms_clamps_above_ceiling():
    ceil = browser_tools._PER_CALL_TIMEOUT_CEIL_MS
    hard_ms = browser_tools._BROWSER_HARD_TIMEOUT_SEC * 1000
    # The ceiling must sit strictly below the hard timeout so a per-call timeout
    # always resolves before the wedge path fires.
    assert ceil < hard_ms
    clamped = browser_tools._clamp_per_call_timeout_ms(180000)
    assert clamped == ceil
    assert clamped < hard_ms


def test_clamp_per_call_timeout_ms_floors_at_1000():
    assert browser_tools._clamp_per_call_timeout_ms(0) == 1000
    assert browser_tools._clamp_per_call_timeout_ms(-5) == 1000


def test_clamp_per_call_timeout_ms_recovers_from_bad_input():
    assert browser_tools._clamp_per_call_timeout_ms("not-a-number") == 30000
    assert browser_tools._clamp_per_call_timeout_ms(None) == 30000


# ── navigate / extract length metadata ─────────────────────────────────


@pytest.fixture
def _real_browser_state(monkeypatch):
    """Put the module into the 'Playwright ready, no live browser' state."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    # Ensure the class-level browser refs are clean for the duration of a test.
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", None)
    # Cached per browser, so it must not leak a previous test's probe result.
    monkeypatch.setattr(BrowserActionToolsMixin, "_user_agent", None)


def test_navigate_length_metadata_excludes_marker(_real_browser_state, monkeypatch):
    host = _NavHost()
    big = "x" * 25000
    page = _FakePage(big)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: page)

    res = host._tool_browser_action({"action": "navigate", "url": "https://x", "max_chars": 5000})
    assert res["ok"], res
    assert res["metadata"]["length"] == 5000  # real content, not marker-inflated
    assert res["metadata"]["total_length"] == 25000  # full body size
    assert "TRUNCATED" in res["content"]


def test_navigate_length_metadata_when_not_truncated(_real_browser_state, monkeypatch):
    host = _NavHost()
    page = _FakePage("short body")
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: page)

    res = host._tool_browser_action({"action": "navigate", "url": "https://x"})
    assert res["ok"], res
    assert res["metadata"]["length"] == len("short body")
    assert res["metadata"]["total_length"] == len("short body")
    assert "TRUNCATED" not in res["content"]


def test_extract_length_metadata_excludes_marker(_real_browser_state, monkeypatch):
    host = _NavHost()
    big = "y" * 25000
    page = _FakePage(big)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: page)

    res = host._tool_browser_action({"action": "extract", "max_chars": 5000})
    assert res["ok"], res
    assert res["metadata"]["length"] == 5000
    assert res["metadata"]["total_length"] == 25000
    assert "TRUNCATED" in res["content"]


# ── _get_browser launch-failure driver cleanup ─────────────────────────


def test_get_browser_stops_driver_when_launch_fails(monkeypatch):
    """If chromium.launch() raises, the just-started Playwright driver must be
    stopped (no orphan node process) and the failure re-raised."""
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", None)
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)

    stopped = {"called": False}

    class _FakeLauncher:
        def launch(self, headless=True):
            raise RuntimeError("no chromium binary")

    class _FakeDriver:
        chromium = _FakeLauncher()

        def start(self):
            return self

        def stop(self):
            stopped["called"] = True

    monkeypatch.setattr(browser_tools, "sync_playwright", lambda: _FakeDriver())

    host = _NavHost()
    with pytest.raises(RuntimeError, match="no chromium binary"):
        host._get_browser()

    assert stopped["called"] is True  # driver stopped → no leak
    assert BrowserActionToolsMixin._playwright is None  # not assigned on failure
    assert BrowserActionToolsMixin._browser is None


# ── _render_and_eval: isolated page (never clobbers the shared session) ──


def test_render_and_eval_uses_isolated_page_and_closes_it(_real_browser_state, monkeypatch):
    """_render_and_eval renders on a FRESH throwaway page — not the shared _page —
    and closes it, so an automated search never destroys the user's interactive
    browser_action session (open tab / login state)."""

    class _EvalPage:
        def __init__(self):
            self.closed = False
            self.evaluated = None
            self.goto_url = None

        def goto(self, url, timeout=None, wait_until=None):
            self.goto_url = url

        def evaluate(self, js):
            self.evaluated = js
            return [{"title": "hello world", "url": "https://x/", "snippet": "s"}]

        def close(self):
            self.closed = True

        def is_closed(self):
            return self.closed

    class _ProbePage(_EvalPage):
        """Answers the one-off navigator.userAgent probe with a HEADLESS UA."""

        def evaluate(self, js):
            self.evaluated = js
            return "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/149.0.0.0 Safari/537.36"

    shared = _EvalPage()
    probe = _ProbePage()
    fresh = _EvalPage()
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", shared)

    handed_out = []

    class _FakeBrowser:
        def new_page(self, user_agent=None):
            handed_out.append(user_agent)
            # First call is the UA probe (raw, no user_agent), then the real page.
            return probe if len(handed_out) == 1 else fresh

    host = _NavHost()
    monkeypatch.setattr(host, "_get_browser", lambda: _FakeBrowser())

    out = host._render_and_eval("https://search.naver.com/x", "() => []", timeout_ms=5000)

    assert out == [{"title": "hello world", "url": "https://x/", "snippet": "s"}]
    assert fresh.evaluated == "() => []"  # eval ran on the fresh page
    assert fresh.goto_url == "https://search.naver.com/x"
    assert fresh.closed is True  # throwaway page closed
    assert probe.closed is True  # UA probe page closed too
    # The render page is created with the de-headlessed UA: leaving
    # "HeadlessChrome" in the string is by itself enough for some anti-bot
    # systems to refuse the request (measured on Startpage 2026-08-05).
    assert handed_out[0] is None  # probe itself is raw
    assert "HeadlessChrome" not in handed_out[1]
    assert "Chrome/149.0.0.0" in handed_out[1]
    assert shared.closed is False  # shared session untouched
    assert BrowserActionToolsMixin._page is shared  # shared _page not replaced


def test_render_and_eval_raises_when_playwright_unavailable(monkeypatch):
    """When Playwright cannot be made available, _render_and_eval raises RuntimeError
    so the search fallback chain records a real reason rather than a silent empty."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    host = _NavHost()
    monkeypatch.setattr(host, "_ensure_playwright_installed", lambda: False)
    with pytest.raises(RuntimeError, match="Playwright"):
        host._render_and_eval("https://x/", "() => []")


# ── wait-action cancellation (P0-1) ─────────────────────────────────────────


class _WaitHost(_NavHost):
    """Host with a registry-style ``config.cancel_event`` (ESC wiring)."""

    def __init__(self, cancel_event=None) -> None:
        self.config = types.SimpleNamespace(cancel_event=cancel_event)


def test_wait_no_selector_aborts_immediately_when_cancel_set(_real_browser_state, monkeypatch):
    """ESC already pressed → a no-selector wait raises AgentCancelled instantly
    instead of sleeping the full clamped timeout on the browser worker."""
    ev = threading.Event()
    ev.set()
    host = _WaitHost(cancel_event=ev)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())

    with pytest.raises(AgentCancelled):
        host._tool_browser_action({"action": "wait", "timeout": 30000})


def test_wait_no_selector_wires_live_cancel_event(_real_browser_state, monkeypatch):
    """The wait's sleep is interruptible_sleep (client.py SSOT) wired to the
    registry's LIVE cancel_event — a mid-wait ESC aborts the turn."""
    ev = threading.Event()
    host = _WaitHost(cancel_event=ev)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())
    calls: list[tuple] = []

    def _fake_sleep(seconds, cancel_event):
        calls.append((seconds, cancel_event))
        return False  # not cancelled → wait completes normally

    monkeypatch.setattr(browser_tools, "interruptible_sleep", _fake_sleep)

    res = host._tool_browser_action({"action": "wait", "timeout": 30000})
    assert res["ok"], res
    assert calls == [(30.0, ev)]


def test_wait_no_selector_absent_config_sleeps_without_cancel(_real_browser_state, monkeypatch):
    """Duck-typed hosts without ``config`` fall back to a non-cancelable wait —
    no AttributeError from the defensive getattr."""
    host = _NavHost()  # no .config attribute
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())
    calls: list[tuple] = []

    def _fake_sleep(seconds, cancel_event):
        calls.append((seconds, cancel_event))
        return False

    monkeypatch.setattr(browser_tools, "interruptible_sleep", _fake_sleep)

    res = host._tool_browser_action({"action": "wait", "timeout": 30000})
    assert res["ok"], res
    assert calls == [(30.0, None)]


def test_wait_no_selector_per_call_scope_aborts_immediately(_real_browser_state, monkeypatch):
    """An abandoned dispatch (its per-call scope set, agent ESC unset) must
    abort the no-selector wait through the ``_live_cancel_event`` wiring — the
    MCP-timeout case: the browser worker's pool slot is released instead of
    sleeping out the clamped timeout."""
    from external_llm.agent.cancel_scope import call_cancel_scope

    host = _WaitHost(cancel_event=threading.Event())  # agent ESC unset
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())

    scope_ev = threading.Event()
    scope_ev.set()  # caller abandoned the call
    with call_cancel_scope(scope_ev), pytest.raises(AgentCancelled):
        host._tool_browser_action({"action": "wait", "timeout": 30000})


def test_wait_no_selector_scope_plus_config_forces_composite(_real_browser_state, monkeypatch):
    """With both a per-call scope AND config.cancel_event live, ``_live_cancel_event``
    yields a composite — a scope set alone must still trip the wait (identity
    preserved only in the single-source case, pinned by the ESC tests above)."""
    from external_llm.agent.cancel_scope import call_cancel_scope

    host = _WaitHost(cancel_event=threading.Event())  # unset, live
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())
    calls: list[tuple] = []

    def _fake_sleep(seconds, cancel_event):
        calls.append((seconds, cancel_event))
        return False  # not cancelled → completes normally

    monkeypatch.setattr(browser_tools, "interruptible_sleep", _fake_sleep)

    scope_ev = threading.Event()
    with call_cancel_scope(scope_ev):
        res = host._tool_browser_action({"action": "wait", "timeout": 30000})
    assert res["ok"], res
    assert len(calls) == 1 and calls[0][0] == 30.0
    # Composite: neither identity — but is_set() ORs both sources.
    ce = calls[0][1]
    assert ce is not None and ce is not scope_ev
    assert ce.is_set() is False


def test_wait_no_selector_mid_wait_cancel_aborts_quickly(_real_browser_state, monkeypatch):
    """ESC pressed DURING the wait → the real interruptible_sleep is interrupted
    and AgentCancelled propagates through the browser executor (the worker is
    not left sleeping for the full timeout)."""
    ev = threading.Event()
    host = _WaitHost(cancel_event=ev)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: object())
    threading.Timer(0.05, ev.set).start()

    t0 = time.monotonic()
    with pytest.raises(AgentCancelled):
        host._tool_browser_action({"action": "wait", "timeout": 5000})
    # Aborted in ~50ms; the uncancelled wait would have run the full 5s.
    assert time.monotonic() - t0 < 2.0


def test_wait_with_selector_ignores_cancel_event(_real_browser_state, monkeypatch):
    """The selector path is unchanged: Playwright's wait_for_selector owns the
    wait even when the cancel event is already set."""
    ev = threading.Event()
    ev.set()
    host = _WaitHost(cancel_event=ev)
    waited: list[str] = []

    class _Page:
        def wait_for_selector(self, selector, timeout=None):
            waited.append(selector)

    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: _Page())

    res = host._tool_browser_action({"action": "wait", "selector": ".ok", "timeout": 30000})
    assert res["ok"], res
    assert waited == [".ok"]
