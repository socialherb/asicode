"""Regression tests for job(output, wait_timeout) hardening.

Three defects fixed together (same bug class as the browser/web retry sleeps):

1. clamp — a model-supplied wait_timeout=99999 pinned a serial worker thread
   (and with ``job`` in _SERIAL_TOOLS, the whole turn) for hours.
2. cancel — the wait ignored ESC for its whole budget; mid-wait cancellation
   must abort promptly via AgentCancelled.
3. MCP ceiling — the outer asyncio.wait_for (120s default) could fire before
   the inner wait budget, discarding the result (see asi_mcp_adapter
   _INNER_TIMEOUT_TOOLS); the derived ceiling must clear the inner budget.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.background_job_manager import BackgroundJobManager
from external_llm.agent.tool_handlers.git_tools import ShellToolsMixin
from external_llm.agent.tool_handlers.shell_policy import SHELL_TIMEOUT_MAX


class _FakeJobInfo:
    def __init__(self, status="running"):
        self.job_id = "j1"
        self.status = status
        self.elapsed = 1.0
        self.stdout = "out"
        self.stderr = "err"


class _StubBgManager:
    """Records the args each call got; returns a canned info."""

    def __init__(self, info):
        self.info = info
        self.wait_calls = []
        self.get_calls = 0

    def wait_for_completion(self, job_id, timeout=120.0, poll_interval=1.0,
                            cancel_event=None):
        self.wait_calls.append((timeout, cancel_event))
        return self.info

    def get_info(self, job_id):
        self.get_calls += 1
        return self.info


class _Handler(ShellToolsMixin):
    """Minimal ShellToolsMixin instance: only the parts _job_output touches."""

    def __init__(self, bg, config):
        self._bg = bg
        self.config = config

    def _make_result(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def _get_bg_manager(self):
        return self._bg


def _handler(bg=None, cancel_event=None):
    return _Handler(
        bg or _StubBgManager(_FakeJobInfo()),
        SimpleNamespace(cancel_event=cancel_event),
    )


class TestWaitTimeoutClamp:
    def test_absurd_wait_timeout_clamped_to_schema_max(self):
        bg = _StubBgManager(_FakeJobInfo(status="completed"))
        h = _handler(bg)
        res = h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": 99999})
        assert bg.wait_calls == [(float(SHELL_TIMEOUT_MAX), None)]
        assert res.ok

    def test_negative_wait_timeout_treated_as_immediate(self):
        bg = _StubBgManager(_FakeJobInfo())
        h = _handler(bg)
        res = h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": -5})
        assert bg.wait_calls == []  # get_info path, no blocking wait
        assert bg.get_calls == 1
        assert res.ok

    def test_garbage_wait_timeout_treated_as_immediate(self):
        bg = _StubBgManager(_FakeJobInfo())
        h = _handler(bg)
        res = h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": "abc"})
        assert bg.wait_calls == []
        assert res.ok

    def test_missing_wait_timeout_treated_as_immediate(self):
        bg = _StubBgManager(_FakeJobInfo())
        h = _handler(bg)
        res = h._tool_job({"action": "output", "job_id": "j1"})
        assert bg.wait_calls == []
        assert res.ok


class TestWaitCancellation:
    def test_pre_set_cancel_raises_agent_cancelled(self):
        ev = threading.Event()
        ev.set()
        bg = _StubBgManager(_FakeJobInfo(status="running"))
        h = _handler(bg, cancel_event=ev)
        with pytest.raises(AgentCancelled):
            h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": 30})
        # The live cancel event must reach the manager (REPL swaps it per turn).
        assert bg.wait_calls == [(30.0, ev)]

    def test_no_cancel_returns_result_normally(self):
        bg = _StubBgManager(_FakeJobInfo(status="completed"))
        h = _handler(bg, cancel_event=threading.Event())
        res = h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": 30})
        assert res.ok
        assert "Job ID: j1" in res.content

    def test_cancel_event_passed_even_when_config_has_none(self):
        bg = _StubBgManager(_FakeJobInfo(status="completed"))
        h = _handler(bg, cancel_event=None)
        res = h._tool_job({"action": "output", "job_id": "j1", "wait_timeout": 5})
        assert bg.wait_calls == [(5.0, None)]
        assert res.ok


class _FakeProc:
    """Minimal subprocess.Popen stand-in (same shape as
    test_background_job_manager.py): poll() returning None keeps it "running"."""

    _next_pid = 9_000_000

    def __init__(self):
        _FakeProc._next_pid += 1
        self.pid = _FakeProc._next_pid
        self.stdout = None
        self.stderr = None

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def _force_kill_through_fake_proc(monkeypatch):
    """Never touch a real OS process group for fake pids (kill path)."""
    from external_llm.agent import background_job_manager as bjm

    monkeypatch.setattr(bjm.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))


class TestWaitForCompletionIntegration:
    def test_mid_wait_cancel_returns_promptly(self):
        mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
        try:
            jid = mgr.start("sleep 30", _FakeProc())
            ev = threading.Event()
            t0 = time.monotonic()
            timer = threading.Timer(0.1, ev.set)
            timer.start()
            try:
                info = mgr.wait_for_completion(
                    jid, timeout=30.0, poll_interval=0.05, cancel_event=ev
                )
            finally:
                timer.cancel()
            elapsed = time.monotonic() - t0
            assert info is not None and info.status == "running"
            assert elapsed < 5.0, f"cancel took {elapsed:.2f}s — sleep not interruptible"
        finally:
            mgr.shutdown()

    def test_completion_still_detected(self):
        """A finishing job must still be reported before the deadline."""
        mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
        try:
            jid = mgr.start("sleep 30", _FakeProc())
            # _FakeProc.poll returns None forever; simulate completion by
            # swapping in a done proc (poll -> 0).
            done_proc = _FakeProc()
            done_proc.poll = lambda: 0
            mgr._jobs[jid].proc = done_proc
            t0 = time.monotonic()
            info = mgr.wait_for_completion(
                jid, timeout=10.0, poll_interval=0.02, cancel_event=threading.Event()
            )
            assert info is not None and info.status == "completed"
            assert time.monotonic() - t0 < 5.0
        finally:
            mgr.shutdown()
