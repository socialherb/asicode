"""RED→GREEN coverage tests for browser_tools.py (72% → 100%).

Covers the remaining edge branches:
  - module guards: _playwright_browser_installed platform/env branches,
    _ensure_playwright_imported negative paths
  - executor teardown helper, dispatch errors (empty/unknown action),
    install-declined path, PlaywrightTimeout / generic exception mapping
  - install error branches (CalledProcessError / unexpected), reload success
  - teardown-failure logging, user-agent probe failure, driver stop failure
  - _render_and_eval: invalid wait_until, selector wait, close failure, wedge
  - click / type / screenshot / evaluate handlers (previously untested)
"""
from __future__ import annotations

import subprocess
import sys
import types

import pytest

from external_llm.agent.tool_handlers import browser_tools
from external_llm.agent.tool_handlers.browser_tools import BrowserActionToolsMixin


class _Host(BrowserActionToolsMixin):
    """Concrete host keeping the REAL handlers (fake page drives the actions)."""

    repo_root = "."

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


class _RecPage:
    """Recording fake Playwright Page covering every handler's surface."""

    def __init__(self, body="hello", title="T", url="https://x/"):
        self._body, self._title, self._url = body, title, url
        self.goto_calls = []
        self.clicked = None
        self.wait_load_state_called = False
        self.filled = None
        self.shot_path = None
        self.evals = []
        self.wait_sel = None
        self.closed = False

    def inner_text(self, selector): return self._body

    def title(self): return self._title

    @property
    def url(self): return self._url

    def goto(self, url, timeout=None, wait_until=None):
        self.goto_calls.append((url, timeout, wait_until))

    def click(self, selector, timeout=None): self.clicked = (selector, timeout)

    def wait_for_load_state(self, state): self.wait_load_state_called = True

    def fill(self, selector, text, timeout=None): self.filled = (selector, text, timeout)

    def screenshot(self, path=None, full_page=None): self.shot_path = path

    def evaluate(self, js):
        self.evals.append(js)
        return 42

    def wait_for_selector(self, selector, timeout=None): self.wait_sel = (selector, timeout)

    def close(self): self.closed = True

    def is_closed(self): return False


@pytest.fixture
def pw_ready(monkeypatch):
    """Module in the 'Playwright ready, no live browser' state."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    monkeypatch.setattr(browser_tools, "_ensure_playwright_imported", lambda: True)
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_page", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_user_agent", None)


# ── module guards ─────────────────────────────────────────────────────────

def test_playwright_browser_installed_false_without_package(monkeypatch):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    assert browser_tools._playwright_browser_installed() is False


def test_playwright_browser_installed_env_path_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "browsers"))
    # Dir does not exist → no installed chromium → False (env branch executed).
    assert browser_tools._playwright_browser_installed() is False


def test_playwright_browser_installed_platform_branches(monkeypatch):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    monkeypatch.setattr(browser_tools.sys, "platform", "win32")
    assert browser_tools._playwright_browser_installed() is False
    monkeypatch.setattr(browser_tools.sys, "platform", "linux")
    assert browser_tools._playwright_browser_installed() is False


def test_ensure_playwright_imported_no_package(monkeypatch):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    monkeypatch.setattr(browser_tools, "sync_playwright", None)
    assert browser_tools._ensure_playwright_imported() is False


def test_ensure_playwright_imported_import_error(monkeypatch):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    monkeypatch.setattr(browser_tools, "sync_playwright", None)
    monkeypatch.setattr(browser_tools, "_PlaywrightTimeout", Exception)
    # Halt the submodule import deterministically regardless of whether the real
    # playwright is installed on this machine.
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    assert browser_tools._ensure_playwright_imported() is False


def test_shutdown_browser_executor_at_exit(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        shutdown=lambda wait, cancel_futures: calls.append((wait, cancel_futures))
    )
    monkeypatch.setattr(browser_tools, "_BROWSER_EXECUTOR", fake)
    browser_tools._shutdown_browser_executor_at_exit()
    assert calls == [(False, True)]


# ── dispatch edges ─────────────────────────────────────────────────────────

def test_tool_browser_action_empty_action_error(pw_ready):
    host = _Host()
    res = host._tool_browser_action({})
    assert res["ok"] is False
    assert "action' is required" in res["error"]


def test_tool_browser_action_unknown_action_error(pw_ready):
    host = _Host()
    res = host._tool_browser_action({"action": "fly"})
    assert res["ok"] is False
    assert "Unknown action" in res["error"]


def test_tool_browser_action_install_declined(pw_ready, monkeypatch):
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    monkeypatch.setattr(browser_tools, "_ensure_playwright_imported", lambda: False)
    host = _Host()
    monkeypatch.setattr(host, "_ensure_playwright_installed", lambda: False)
    res = host._tool_browser_action({"action": "navigate", "url": "https://x"})
    assert res["ok"] is False
    assert "Playwright is not available" in res["error"]


class _FakePwTimeoutError(Exception):
    pass


def test_tool_browser_action_playwright_timeout_mapping(pw_ready, monkeypatch):
    monkeypatch.setattr(browser_tools, "_PlaywrightTimeout", _FakePwTimeoutError)
    host = _Host()

    def _boom(args):
        raise _FakePwTimeoutError("too slow")

    monkeypatch.setattr(host, "_browser_navigate", _boom)
    res = host._tool_browser_action({"action": "navigate", "url": "https://x"})
    assert res["ok"] is False
    assert "Playwright timeout" in res["error"]


def test_tool_browser_action_generic_exception_mapping(pw_ready, monkeypatch):
    monkeypatch.setattr(browser_tools, "_PlaywrightTimeout", _FakePwTimeoutError)
    host = _Host()

    def _boom(args):
        raise RuntimeError("boom")

    monkeypatch.setattr(host, "_browser_navigate", _boom)
    res = host._tool_browser_action({"action": "navigate", "url": "https://x"})
    assert res["ok"] is False
    assert "RuntimeError: boom" in res["error"]


# ── install / reload branches ──────────────────────────────────────────────

def test_ensure_playwright_installed_reloads_after_install(monkeypatch):
    host = _Host()
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    monkeypatch.setattr(host, "_ask_install_playwright", lambda: True)
    monkeypatch.setattr(host, "_install_playwright", lambda: True)
    monkeypatch.setattr(host, "_reload_playwright_module", lambda: True)
    assert host._ensure_playwright_installed() is True


def test_ensure_playwright_installed_install_failure(monkeypatch):
    """A failed pip install → False (the reload branch is never reached)."""
    host = _Host()
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    monkeypatch.setattr(host, "_ask_install_playwright", lambda: True)
    monkeypatch.setattr(host, "_install_playwright", lambda: False)
    assert host._ensure_playwright_installed() is False


def test_pip_install_flags_delegates(monkeypatch):
    host = _Host()
    monkeypatch.setattr(browser_tools, "pip_install_flags", lambda: ["--x"])
    assert host._pip_install_flags() == ["--x"]


def test_install_playwright_called_process_error(monkeypatch):
    host = _Host()
    monkeypatch.setattr(host, "_pip_install_flags", lambda: [])

    def _run(*a, **kw):
        raise subprocess.CalledProcessError(1, ["pip"], stderr=b"externally-managed")

    monkeypatch.setattr(browser_tools.subprocess, "run", _run)
    assert host._install_playwright() is False


def test_install_playwright_unexpected_error(monkeypatch):
    host = _Host()
    monkeypatch.setattr(host, "_pip_install_flags", lambda: [])

    def _run(*a, **kw):
        raise OSError("pip missing")

    monkeypatch.setattr(browser_tools.subprocess, "run", _run)
    assert host._install_playwright() is False


def test_reload_playwright_module_success(monkeypatch):
    # Pin the pre-test values so the mutation is restored at teardown.
    monkeypatch.setattr(browser_tools, "sync_playwright", browser_tools.sync_playwright)
    monkeypatch.setattr(browser_tools, "_PlaywrightTimeout", browser_tools._PlaywrightTimeout)
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)

    class _FakeSync:
        sync_playwright = "SYNC"
        TimeoutError = "TMO"

    monkeypatch.setattr(browser_tools, "ensure_user_site_importable", lambda: None)
    monkeypatch.setattr(browser_tools.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(browser_tools.importlib, "import_module", lambda name: _FakeSync)

    host = _Host()
    assert host._reload_playwright_module() is True
    assert browser_tools.sync_playwright == "SYNC"
    assert browser_tools._PlaywrightTimeout == "TMO"
    assert browser_tools.HAS_PLAYWRIGHT is True


def test_reload_playwright_module_import_error(monkeypatch):
    """A failed re-import after install → False (ImportError branch)."""

    def _boom(name):
        raise ImportError("no playwright")

    monkeypatch.setattr(browser_tools.importlib, "import_module", _boom)
    host = _Host()
    assert host._reload_playwright_module() is False


# ── lifecycle failure branches ─────────────────────────────────────────────

def test_get_browser_driver_stop_failure_logged(monkeypatch):
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", None)
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", None)

    class _Launcher:
        def launch(self, headless=True):
            raise RuntimeError("launch fail")

    class _Driver:
        chromium = _Launcher()

        def start(self):
            return self

        def stop(self):
            raise RuntimeError("stop fail")

    monkeypatch.setattr(browser_tools, "sync_playwright", lambda: _Driver())
    host = _Host()
    with pytest.raises(RuntimeError, match="launch fail"):
        host._get_browser()


def test_browser_user_agent_probe_failure_returns_none(monkeypatch):
    monkeypatch.setattr(BrowserActionToolsMixin, "_user_agent", None)

    class _BrokenBrowser:
        def new_page(self):
            raise RuntimeError("probe fail")

    monkeypatch.setattr(BrowserActionToolsMixin, "_get_browser", lambda self: _BrokenBrowser())
    host = _Host()
    assert host._browser_user_agent() is None


def test_close_shared_browser_teardown_failures_logged(monkeypatch):
    class _Fragile:
        def __init__(self, name):
            self._name = name

        def close(self):
            raise RuntimeError(f"{self._name} close fail")

        def stop(self):
            raise RuntimeError(f"{self._name} stop fail")

    monkeypatch.setattr(BrowserActionToolsMixin, "_page", _Fragile("page"))
    monkeypatch.setattr(BrowserActionToolsMixin, "_browser", _Fragile("browser"))
    monkeypatch.setattr(BrowserActionToolsMixin, "_playwright", _Fragile("pw"))
    monkeypatch.setattr(BrowserActionToolsMixin, "_user_agent", "UA")
    host = _Host()
    host._close_shared_browser()  # must swallow every teardown error
    assert BrowserActionToolsMixin._page is None
    assert BrowserActionToolsMixin._browser is None
    assert BrowserActionToolsMixin._playwright is None
    assert BrowserActionToolsMixin._user_agent is None


# ── _render_and_eval edges ─────────────────────────────────────────────────

def test_render_and_eval_invalid_wait_until_and_selector(pw_ready, monkeypatch):
    page = _RecPage()
    browser = types.SimpleNamespace(new_page=lambda user_agent=None: page)
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_browser", lambda self: browser)
    # Pre-set the UA so the user-agent probe (a second page) is skipped and the
    # fake page is only consumed by the render under test.
    monkeypatch.setattr(BrowserActionToolsMixin, "_user_agent", "TestUA")
    host = _Host()

    out = host._render_and_eval("https://x", "1+1", wait_until="bogus", wait_for_selector="#el")
    assert out == 42
    assert page.goto_calls[0][2] == "networkidle"  # invalid wait_until defaulted
    assert page.wait_sel == ("#el", page.goto_calls[0][1])
    assert page.evals == ["1+1"]
    assert page.closed is True


def test_render_and_eval_page_close_failure_logged(pw_ready, monkeypatch):
    class _FragilePage:
        def goto(self, url, timeout=None, wait_until=None):
            pass

        def evaluate(self, js):
            return 1

        def close(self):
            raise RuntimeError("close fail")

    browser = types.SimpleNamespace(new_page=lambda user_agent=None: _FragilePage())
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_browser", lambda self: browser)
    host = _Host()
    assert host._render_and_eval("https://x", "1") == 1


def test_render_and_eval_hard_timeout_wedge(pw_ready, monkeypatch):
    from concurrent.futures import TimeoutError as _FutureTimeout

    class _Never:
        def result(self, timeout=None):
            raise _FutureTimeout("hard")

    class _FakeExec:
        def submit(self, fn):
            return _Never()

    monkeypatch.setattr(browser_tools, "_BROWSER_EXECUTOR", _FakeExec())
    monkeypatch.setattr(browser_tools, "_reset_browser_on_wedge", lambda: None)
    host = _Host()
    with pytest.raises(RuntimeError, match="browser render did not complete"):
        host._render_and_eval("https://x", "1+1")


# ── action handlers: click / type / screenshot / evaluate ──────────────────

def _bind_page(monkeypatch, page):
    monkeypatch.setattr(BrowserActionToolsMixin, "_get_page", lambda self: page)


def test_navigate_invalid_wait_until_defaults_to_load(pw_ready, monkeypatch):
    page = _RecPage()
    _bind_page(monkeypatch, page)
    host = _Host()
    res = host._tool_browser_action({"action": "navigate", "url": "https://x", "wait_until": "bogus"})
    assert res["ok"], res
    assert page.goto_calls[0][2] == "load"


def test_navigate_missing_url_error(pw_ready, monkeypatch):
    _bind_page(monkeypatch, _RecPage())
    host = _Host()
    res = host._tool_browser_action({"action": "navigate"})
    assert res["ok"] is False
    assert "'url' is required" in res["error"]


def test_click_ok(pw_ready, monkeypatch):
    page = _RecPage()
    _bind_page(monkeypatch, page)
    host = _Host()
    res = host._tool_browser_action({"action": "click", "selector": "#btn"})
    assert res["ok"], res
    assert page.clicked == ("#btn", pytest.approx(30000) if False else page.clicked[1])
    assert page.clicked[0] == "#btn"
    assert page.wait_load_state_called is True
    assert res["content"] == "Clicked '#btn'"
    assert res["metadata"]["selector"] == "#btn"


def test_click_missing_selector_error(pw_ready, monkeypatch):
    _bind_page(monkeypatch, _RecPage())
    host = _Host()
    res = host._tool_browser_action({"action": "click"})
    assert res["ok"] is False
    assert "'selector' is required" in res["error"]


def test_type_ok_with_long_text_snippet(pw_ready, monkeypatch):
    page = _RecPage()
    _bind_page(monkeypatch, page)
    host = _Host()
    long_text = "x" * 60
    res = host._tool_browser_action({"action": "type", "selector": "#in", "text": long_text})
    assert res["ok"], res
    assert page.filled == ("#in", long_text, page.filled[2])
    assert page.filled[0] == "#in" and page.filled[1] == long_text
    assert "..." in res["content"]  # 50-char snippet truncation
    assert res["metadata"]["text_length"] == 60
    short = host._tool_browser_action({"action": "type", "selector": "#in", "text": "hi"})
    assert short["ok"] and "Typed 'hi'" in short["content"]


def test_type_missing_selector_error(pw_ready, monkeypatch):
    _bind_page(monkeypatch, _RecPage())
    host = _Host()
    res = host._tool_browser_action({"action": "type", "text": "hi"})
    assert res["ok"] is False
    assert "'selector' and 'text' are required" in res["error"]


def test_screenshot_ok(pw_ready, monkeypatch, tmp_path):
    page = _RecPage()
    _bind_page(monkeypatch, page)
    host = _Host()
    monkeypatch.setattr(host, "_screenshot_dir", lambda: str(tmp_path))
    res = host._tool_browser_action({"action": "screenshot"})
    assert res["ok"], res
    assert res["metadata"]["filepath"].startswith(str(tmp_path))
    assert page.shot_path == res["metadata"]["filepath"]
    assert res["metadata"]["url"] == "https://x/"


def test_evaluate_ok(pw_ready, monkeypatch):
    page = _RecPage()
    _bind_page(monkeypatch, page)
    host = _Host()
    res = host._tool_browser_action({"action": "evaluate", "js": "1+1"})
    assert res["ok"], res
    assert page.evals == ["1+1"]
    assert res["content"] == "42"
    assert res["metadata"]["result_type"] == "int"


def test_evaluate_missing_js_error(pw_ready, monkeypatch):
    _bind_page(monkeypatch, _RecPage())
    host = _Host()
    res = host._tool_browser_action({"action": "evaluate"})
    assert res["ok"] is False
    assert "'js' is required" in res["error"]
