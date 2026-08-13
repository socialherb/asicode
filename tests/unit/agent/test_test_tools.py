"""run_tests tool handler: config timeout, mid-run cancellation, cancelled contract.

Covers _tool_run_tests (TestToolsMixin) with a stub host + a fake TestRunner:
the handler is what changed (timeout source, cancel_check wiring, cancelled
result contract), and a real ToolRegistry would drag in a full git-repo fixture
for no extra signal.
"""
from __future__ import annotations

import sys
import threading
import types
from typing import ClassVar

import pytest

from external_llm.agent.tool_handlers.test_tools import TestToolsMixin
from external_llm.agent.tool_registry import AgentConfig


class _Host(TestToolsMixin):
    """Minimal host exposing only what _tool_run_tests touches."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.repo_root = "."

    def _make_result(self, **kwargs):
        # Mirror ToolResult's field defaults + override semantics so paths that
        # don't set a field still read the same as production.
        fields = {
            "ok": False,
            "content": "",
            "error": None,
            "metadata": {},
            "execution_time": 0.0,
            "partial_failure": False,
            "retryable": True,
            "retry_count": 0,
        }
        fields.update(kwargs)
        return types.SimpleNamespace(**fields)


def _fake_result(**overrides):
    base = {
        "ok": True,
        "exit_code": 0,
        "duration_ms": 100,
        "timed_out": False,
        "cancelled": False,
        "failing_tests": [],
        "passed_count": 1,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "xpassed_count": 0,
        "xfailed_count": 0,
        "failed_test_details": [],
        "error_test_details": [],
        "summary_line": "1 passed",
        "first_traceback": None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _FakeRunner:
    """Captures every run/run_pytest call; returns the canned class-level result."""

    calls: ClassVar[list[dict]] = []
    result = None
    python_executable = sys.executable

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_provider(cls, *args, **kwargs):
        return cls()

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def run_pytest(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def host(monkeypatch):
    # _tool_run_tests lazy-imports TestRunner from ..test_runner, so patch the
    # class on that module (not on tool_handlers.test_tools).
    monkeypatch.setattr(
        "external_llm.agent.test_runner.TestRunner", _FakeRunner
    )
    _FakeRunner.calls = []
    return _Host(AgentConfig())


def test_timeout_comes_from_config(host):
    host.config.test_timeout_sec = 77
    _FakeRunner.result = _fake_result()
    res = host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert res.ok is True
    assert _FakeRunner.calls[0]["timeout_sec"] == 77


def test_default_timeout_is_300(host):
    _FakeRunner.result = _fake_result()
    host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert _FakeRunner.calls[0]["timeout_sec"] == 300


def test_cancel_check_reads_live_config(host):
    _FakeRunner.result = _fake_result()
    host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    cancel_check = _FakeRunner.calls[0]["cancel_check"]
    # No cancel_event configured → never cancels.
    assert cancel_check() is False
    # Set while the run is "in flight" → the poll sees it (live read, not a
    # dispatch-entry snapshot — the REPL swaps cancel_event per turn).
    host.config.cancel_event = threading.Event()
    host.config.cancel_event.set()
    assert cancel_check() is True
    host.config.cancel_event.clear()
    assert cancel_check() is False


def test_cancelled_run_reports_cancelled(host):
    _FakeRunner.result = _fake_result(ok=False, cancelled=True)
    res = host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert res.ok is False
    assert res.error == "Operation cancelled during test execution"
    assert res.retryable is False
    assert res.metadata["cancelled"] is True
    assert res.metadata["timed_out"] is False
    assert "CANCELLED" in res.content
    assert "NOT a test failure" in res.content


def test_timed_out_run_not_marked_cancelled(host):
    _FakeRunner.result = _fake_result(ok=False, timed_out=True)
    res = host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert res.metadata["cancelled"] is False
    assert res.metadata["timed_out"] is True
    assert "TIMED OUT" in res.content
    assert res.retryable is True  # default result path keeps retryable as-is
