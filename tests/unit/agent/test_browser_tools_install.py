"""Regression tests for on-the-fly Playwright install with user consent.

Tests the install flow in ``_ensure_playwright_installed``:

  - Fast path: already installed (HAS_PLAYWRIGHT=True)
  - User consent flow: decline, consent (happy path), ask_user exception
  - Frozen environment guard (sys.frozen)

Flag wiring only: that _install_playwright threads the shared PEP 668 flags
(external_llm.pip_env.pip_install_flags) into the pip invocation. The flag
decision itself is covered by tests/unit/test_pip_env.py.

These tests do NOT require Playwright or Chromium to be installed; every
external dependency (subprocess, importlib) is mocked away.
"""
import sys

import external_llm.agent.tool_handlers.browser_tools as browser_tools
from external_llm.agent.tool_handlers.browser_tools import (
    BrowserActionToolsMixin,
)
from external_llm.agent.tool_registry import ToolResult


class _InstallHost(BrowserActionToolsMixin):
    """Minimal concrete host for testing install flow.

    ``_make_result`` is required by the mixin ABC but is not called by the
    install code path (which goes through ``_tool_ask_user`` instead).
    """

    repo_root = "."

    def _make_result(self, ok, content, error=None, metadata=None):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


def test_already_installed(monkeypatch):
    """HAS_PLAYWRIGHT=True → return True immediately, no ask_user call."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", True)
    host = _InstallHost()

    def _should_not_call(_args):
        raise AssertionError("_tool_ask_user must not be called when already installed")

    monkeypatch.setattr(host, "_tool_ask_user", _should_not_call, raising=False)

    assert host._ensure_playwright_installed() is True


def test_user_declines(monkeypatch):
    """User says 'no' → return False, _install_playwright not called."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    host = _InstallHost()
    monkeypatch.setattr(
        host, "_tool_ask_user",
        lambda args: ToolResult(ok=True, metadata={"answer": "no"}),
        raising=False,
    )

    install_called = False

    def _fail_install():
        nonlocal install_called
        install_called = True
        return True

    monkeypatch.setattr(host, "_install_playwright", _fail_install, raising=False)

    assert host._ensure_playwright_installed() is False
    assert not install_called, "_install_playwright must NOT be called when user declines"


def test_user_consents(monkeypatch):
    """User says 'yes' + install succeeds + reload succeeds → return True."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    host = _InstallHost()
    monkeypatch.setattr(
        host, "_tool_ask_user",
        lambda args: ToolResult(ok=True, metadata={"answer": "yes"}),
        raising=False,
    )
    monkeypatch.setattr(host, "_install_playwright", lambda: True, raising=False)
    monkeypatch.setattr(host, "_reload_playwright_module", lambda: True, raising=False)

    assert host._ensure_playwright_installed() is True


def test_ask_user_raises(monkeypatch):
    """ask_user raises → return False, _install_playwright not called."""
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    host = _InstallHost()
    monkeypatch.setattr(host, "_tool_ask_user", lambda args: (_ for _ in ()).throw(RuntimeError("prompt down")), raising=False)

    install_called = False

    def _fail_install():
        nonlocal install_called
        install_called = True

    monkeypatch.setattr(host, "_install_playwright", _fail_install, raising=False)

    assert host._ensure_playwright_installed() is False
    assert not install_called, "_install_playwright must NOT be called when ask_user fails"


def test_frozen_env(monkeypatch):
    """sys.frozen=True → return False, no ask_user/install attempted.

    Frozen environments (PyInstaller, py2exe, …) cannot run
    ``sys.executable -m pip``; the guard short-circuits before any
    interaction and the existing error path shows manual install instructions.
    """
    monkeypatch.setattr(browser_tools, "HAS_PLAYWRIGHT", False)
    # sys.frozen does not exist on CPython; setattr with raising=False creates it.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    host = _InstallHost()

    ask_called = False

    def _fail_ask(_args):
        nonlocal ask_called
        ask_called = True

    monkeypatch.setattr(host, "_tool_ask_user", _fail_ask, raising=False)

    try:
        assert host._ensure_playwright_installed() is False
        assert not ask_called, "_tool_ask_user must NOT be called in frozen environment"
    finally:
        # monkeypatch.restore restores sys.frozen automatically at test teardown
        pass


# ── PEP 668 externally-managed environment handling ──────────────────── #
# The flag-decision logic itself lives in external_llm.pip_env and is covered
# by tests/unit/test_pip_env.py. Here we only assert browser wiring: that
# _install_playwright threads whatever flags the (shared) helper returns.

def test_install_uses_flags(monkeypatch):
    """_install_playwright threads the env flags into the pip invocation."""
    monkeypatch.setattr(
        BrowserActionToolsMixin, "_pip_install_flags",
        staticmethod(lambda: ["--user", "--break-system-packages"]),
    )
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        class _R:  # minimal CompletedProcess stand-in
            returncode = 0
        return _R()

    monkeypatch.setattr(browser_tools.subprocess, "run", _fake_run)
    assert _InstallHost()._install_playwright() is True
    pip_cmd = calls[0]
    assert pip_cmd[:5] == [sys.executable, "-m", "pip", "install", "playwright"]
    assert pip_cmd[-2:] == ["--user", "--break-system-packages"]
    # Chromium step carries no pip flags (browser binaries, not a pip package).
    assert calls[1] == [sys.executable, "-m", "playwright", "install", "chromium"]
