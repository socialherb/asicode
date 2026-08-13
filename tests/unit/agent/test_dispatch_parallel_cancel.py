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
    ev.set()         # user presses ESC
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"cancel did not preempt dispatch_parallel (exc={out.get('exc')!r})")
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
