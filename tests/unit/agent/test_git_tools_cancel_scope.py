"""git_tools' three cancel channels must observe the per-call scope, not just ESC.

``_capture_bounded``'s poll loop (a running bash), ``_tool_shell_exec``'s
dispatch-entry check, and ``_job_output``'s blocking wait all read only
``config.cancel_event`` — the whole-turn ESC. A call abandoned by ITS caller
(MCP ``wait_for`` timeout, aborted ``dispatch_parallel`` batch) sets a
per-call scope event none of them saw: the bash ran out its whole budget
(120-300 s), a dead dispatch was waved through the entry check, and the job
wait pinned the serial-turn thread for up to ``wait_timeout``.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
import types

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.background_job_manager import BackgroundJobManager
from external_llm.agent.cancel_scope import call_cancel_scope
from external_llm.agent.tool_handlers.git_tools import ShellToolsMixin


class _Host(ShellToolsMixin):
    """Duck-typed host: just what the three observers touch."""

    def __init__(self, root: str, config):
        self.repo_root = root
        self.config = config
        self._bg: BackgroundJobManager | None = None

    # Registry-level helpers the mixin borrows from its host (ToolRegistry).
    def _correct_bias_path(self, command: str) -> str:
        return command

    def _secure_path(self, p: str) -> str:
        return p

    def _get_bg_manager(self) -> BackgroundJobManager:
        assert self._bg is not None, "test bug: no manager installed"
        return self._bg

    def _make_result(self, ok=False, content="", error=None, metadata=None, **kw):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


@pytest.fixture
def host(tmp_path):
    return _Host(str(tmp_path), types.SimpleNamespace(cancel_event=threading.Event()))


def _kill(proc) -> None:
    """Best-effort teardown: nothing a test starts may outlive it."""
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):  # pragma: no cover
        proc.wait(timeout=5)


def test_shell_exec_entry_observes_per_call_scope(host):
    """The dispatch-entry check must OR the live per-call scope in — a call
    abandoned before it started is rejected, not executed."""
    scope = threading.Event()
    scope.set()  # caller abandoned the dispatch; config.cancel_event stays unset
    with call_cancel_scope(scope):
        res = host._tool_shell_exec({"command": "echo should-not-run", "timeout": 5})

    assert not res["ok"]
    assert res["error"] == "Operation cancelled before shell execution"


def test_running_bash_aborts_when_caller_abandons(host):
    """An already-running bash whose dispatch was abandoned must stop at the
    next poll tick, not run out the whole budget (then misreport timeout)."""
    proc = subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    scope = threading.Event()
    try:
        with call_cancel_scope(scope):
            scope.set()  # abandoned mid-run
            t0 = time.monotonic()
            _out, _err, status = host._capture_bounded(proc, timeout=3, cap=10_000)
            elapsed = time.monotonic() - t0
    finally:
        _kill(proc)

    assert status == "cancelled", f"abandoned bash ran to status={status!r}"
    assert elapsed < 2, f"cancel observed only after {elapsed:.1f}s of a 3s budget"


def test_job_wait_aborts_when_caller_abandons(host):
    """``job(action=output, wait_timeout=...)`` blocking the turn must return
    at the next poll tick when ITS dispatch is abandoned — raising
    AgentCancelled, not pinning the serial thread for the full wait."""
    host._bg = BackgroundJobManager()
    # start_new_session is NOT cosmetic here: BackgroundJob.kill() signals the
    # process GROUP, so a session-less Popen would put pytest's own pgid in the
    # blast radius and the cleanup would terminate the test run itself.
    proc = subprocess.Popen(
        ["sleep", "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    job_id = host._bg.start("sleep 30", proc)
    scope = threading.Event()
    try:
        with call_cancel_scope(scope):
            scope.set()  # abandoned while the wait blocks
            t0 = time.monotonic()
            with pytest.raises(AgentCancelled):
                host._job_output({"job_id": job_id, "wait_timeout": 3})
            elapsed = time.monotonic() - t0
    finally:
        host._bg.kill(job_id)
        _kill(proc)

    assert elapsed < 2, f"job wait observed the scope only after {elapsed:.1f}s"
