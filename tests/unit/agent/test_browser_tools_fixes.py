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

import pytest

import external_llm.agent.tool_handlers.browser_tools as browser_tools
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


def test_navigate_length_metadata_excludes_marker(_real_browser_state, monkeypatch):
    host = _NavHost()
    big = "x" * 25000
    page = _FakePage(big)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: page)

    res = host._tool_browser_action({"action": "navigate", "url": "https://x", "max_chars": 5000})
    assert res["ok"], res
    assert res["metadata"]["length"] == 5000          # real content, not marker-inflated
    assert res["metadata"]["total_length"] == 25000    # full body size
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

    assert stopped["called"] is True                  # driver stopped → no leak
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

    shared = _EvalPage()
    fresh = _EvalPage()
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", shared)

    class _FakeBrowser:
        def new_page(self):
            return fresh

    host = _NavHost()
    monkeypatch.setattr(host, "_get_browser", lambda: _FakeBrowser())

    out = host._render_and_eval("https://search.naver.com/x", "() => []", timeout_ms=5000)

    assert out == [{"title": "hello world", "url": "https://x/", "snippet": "s"}]
    assert fresh.evaluated == "() => []"                # eval ran on the fresh page
    assert fresh.goto_url == "https://search.naver.com/x"
    assert fresh.closed is True                          # throwaway page closed
    assert shared.closed is False                        # shared session untouched
    assert BrowserActionToolsMixin._page is shared       # shared _page not replaced


def test_render_and_eval_raises_when_playwright_unavailable(monkeypatch):
    """When Playwright cannot be made available, _render_and_eval raises RuntimeError
    so the search fallback chain records a real reason rather than a silent empty."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    host = _NavHost()
    monkeypatch.setattr(host, "_ensure_playwright_installed", lambda: False)
    with pytest.raises(RuntimeError, match="Playwright"):
        host._render_and_eval("https://x/", "() => []")
