"""ToolRegistry.dispatch_parallel must honor cancel_event during collection.

A bare future.result() blocks until the slowest tool finishes — ESC during a
long read (huge grep, web fetch) would freeze the whole turn. The cancel-aware
poll raises AgentCancelled at the CANCEL_POLL_INTERVAL cadence; an
AgentCancelled raised INSIDE a tool must propagate — not be wrapped into a
ToolResult error that the caller would feed back to the LLM as a tool failure.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.tool_registry import ToolRegistry, ToolResult


def _registry(ev, dispatch):
    reg = ToolRegistry.__new__(ToolRegistry)
    reg.config = SimpleNamespace(
        cancel_event=ev,
        parallel_tool_execution_enabled=True,
    )
    reg.dispatch = dispatch
    return reg


def _two_read_calls():
    return [
        {"tool": "bash", "args": {"command": "ls -la"}},
        {"tool": "bash", "args": {"command": "git status"}},
    ]


def test_dispatch_parallel_cancel_event_preempts_collection():
    """ESC mid-collection must abort at the poll cadence, not after the
    slowest tool finishes."""
    ev = threading.Event()
    finished = {"v": 0}

    def _slow_dispatch(tool, args):
        time.sleep(1.5)
        finished["v"] += 1
        return ToolResult(ok=True, content="ok", error="")

    reg = _registry(ev, _slow_dispatch)
    out = {}

    def _run():
        try:
            reg.dispatch_parallel(_two_read_calls())
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # let the tools start in the pool
    ev.set()  # user presses ESC
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"cancel did not preempt dispatch_parallel (exc={out.get('exc')!r})"
    )
    assert finished["v"] == 0, "cancel must abort while tools are still running"


def test_dispatch_parallel_agent_cancelled_from_tool_propagates():
    """AgentCancelled raised inside a tool must abort the batch — NOT be
    wrapped into a ToolResult error the caller would feed back to the LLM."""
    ev = threading.Event()

    def _cancelling_dispatch(tool, args):
        raise AgentCancelled("cancelled by user in tool")

    reg = _registry(ev, _cancelling_dispatch)
    with pytest.raises(AgentCancelled):
        reg.dispatch_parallel(_two_read_calls())


# ── Per-call scope propagation (cooperative cancellation) ──────────────────
# dispatch_parallel installs a cancel scope per submitted call; when the batch
# aborts (ESC), the still-running workers must OBSERVE the abandonment at
# their next checkpoint and free their pool slots — not run to completion.


def test_dispatch_parallel_abort_cancels_in_flight_workers():
    """ESC mid-batch: AgentCancelled raised AND both workers exit early."""
    from external_llm.agent.cancel_scope import current_cancel_event

    ev = threading.Event()
    FULL = 3.0
    exits: list[float] = []
    lock = threading.Lock()

    def _scope_aware_dispatch(tool, args):
        t0 = time.monotonic()
        while time.monotonic() - t0 < FULL:
            ce = current_cancel_event()
            if ce is not None and ce.is_set():
                break  # checkpoint observed — abandon early
            time.sleep(0.01)
        with lock:
            exits.append(time.monotonic())
        return ToolResult(ok=True, content="ok", error="")

    reg = _registry(ev, _scope_aware_dispatch)
    out = {}

    def _run():
        try:
            reg.dispatch_parallel(_two_read_calls())
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.3)  # both workers running in the pool
    ev.set()  # user presses ESC
    t.join(timeout=10)
    assert isinstance(out.get("exc"), AgentCancelled)
    # Both workers must have exited well before their full duration: the
    # abort path set their per-call scope events (finally in dispatch_parallel).
    deadline = t0 + FULL  # when an UNPROPAGATED batch would still be running
    for _ in range(200):  # wait up to ~2s for cooperative exits
        with lock:
            if len(exits) >= 2:
                break
        time.sleep(0.01)
    assert len(exits) == 2, (
        f"workers did not exit cooperatively ({len(exits)}/2 by "
        f"{time.monotonic() - t0:.1f}s; full run would end at {FULL:.0f}s)"
    )
    assert max(exits) < deadline, "a worker ran to completion — scope never set"
