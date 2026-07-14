"""Regression guard: design_chat_loop.respond() must emit a `design_thinking_stop`
event on EVERY exit path (normal final answer, LLM error, unexpected exception,
AgentCancelled) so the CLI thinking-ticker is torn down before the final message
is rendered.

Root cause this pins: the normal final-answer return path (no tool_calls → early
return at `if not _effective_tool_calls`) skips the `design_thinking` emit that
lives *after* it, leaving the ticker spinning while the caller renders the final
message. The fix emits an explicit `design_thinking_stop` in respond()'s finally
block, covering every exit path once.
"""
from __future__ import annotations

import types

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.client import LLMRateLimitError


def _make_loop():
    """Lightweight DesignChatLoop without the heavy __init__."""
    loop = DesignChatLoop.__new__(DesignChatLoop)
    loop.registry = types.SimpleNamespace(config=types.SimpleNamespace())
    return loop


def _capture_events(loop, impl_fn):
    """Run respond() with a capturing stream_callback; return (result, events)."""
    loop._respond_impl = impl_fn
    events: list[tuple[str, dict]] = []

    def _cb(name, data=None):
        events.append((name, data or {}))

    result = loop.respond([], stream_callback=_cb)
    return result, events


def _stop_emitted(events):
    return any(name == "design_thinking_stop" for name, _ in events)


def test_normal_final_answer_emits_stop():
    """The early-return final-answer path (no tool_calls) must still emit stop."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        # Simulate the normal final-answer path: _respond_impl fires
        # design_thinking_start, then returns without emitting design_thinking
        # (the bug path before the fix).
        if sc:
            sc("design_thinking_start", {})
        result.content = "final answer"
        return result

    loop = _make_loop()
    result, events = _capture_events(loop, _impl)
    assert result.content == "final answer"
    assert _stop_emitted(events), (
        "normal final-answer path must emit design_thinking_stop so the CLI "
        "ticker is torn down before the final message renders"
    )


def test_llmclient_error_emits_stop():
    """The LLMClientError handler path must emit stop via the finally block."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        if sc:
            sc("design_thinking_start", {})
        raise LLMRateLimitError("429 busy")

    loop = _make_loop()
    result, events = _capture_events(loop, _impl)
    assert result.is_error is True
    assert _stop_emitted(events), (
        "LLMClientError path must emit design_thinking_stop so the ticker "
        "does not keep spinning over the error message"
    )


def test_unexpected_exception_emits_stop():
    """The generic Exception handler path must emit stop via the finally block."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        if sc:
            sc("design_thinking_start", {})
        raise ValueError("genuine bug")

    loop = _make_loop()
    result, events = _capture_events(loop, _impl)
    assert result.is_error is True
    assert _stop_emitted(events), (
        "unexpected-exception path must emit design_thinking_stop"
    )


def test_agent_cancelled_emits_stop():
    """AgentCancelled is re-raised, but the finally must still emit stop."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        if sc:
            sc("design_thinking_start", {})
        raise AgentCancelled("user pressed ESC")

    loop = _make_loop()
    events: list[tuple[str, dict]] = []

    def _cb(name, data=None):
        events.append((name, data or {}))

    loop._respond_impl = _impl
    with pytest.raises(AgentCancelled):
        loop.respond([], stream_callback=_cb)
    assert _stop_emitted(events), (
        "AgentCancelled re-raise path must still emit design_thinking_stop in "
        "the finally block — the ticker must not survive a user cancel"
    )


def test_no_stream_callback_does_not_crash():
    """When stream_callback is None (non-interactive caller), finally is a no-op."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        result.content = "ok"
        return result

    loop = _make_loop()
    loop._respond_impl = _impl
    # stream_callback defaults to None — must not raise.
    result = loop.respond([])
    assert result.content == "ok"


def test_stop_emitted_exactly_once_per_respond():
    """The finally emits stop exactly once, even if _respond_impl also emitted it."""
    def _impl(msgs, sc, rc, mi, tc, result, **kw):
        if sc:
            sc("design_thinking_start", {})
            sc("design_thinking", {"content": "intermediate", "elapsed": 1.0})
        result.content = "final"
        return result

    loop = _make_loop()
    result, events = _capture_events(loop, _impl)
    stop_count = sum(1 for name, _ in events if name == "design_thinking_stop")
    assert stop_count == 1, (
        f"design_thinking_stop must be emitted exactly once by the finally, "
        f"got {stop_count}"
    )
