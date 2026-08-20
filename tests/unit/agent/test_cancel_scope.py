"""Per-call cooperative cancellation scope (agent.cancel_scope).

Contracts sealed here:
  * thread-local STACK semantics — innermost scope observed, symmetric pop on
    exceptions, nothing leaks to the next dispatch on a pooled thread;
  * ``effective_cancel`` merging — single source returned as-is (zero wrapper
    on the common serial path), composite OR when scope + fallback coexist;
  * ``ToolRegistry._dispatch_impl`` entry checkpoint — a call abandoned while
    its worker was still queued returns "Operation cancelled" without running
    the handler (the pool slot frees immediately instead of after the tool's
    full run).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from external_llm.agent.cancel_scope import (
    CallCancelledError,
    call_cancel_scope,
    current_cancel_event,
    effective_cancel,
    raise_if_call_cancelled,
)
from external_llm.agent.tool_registry import ToolRegistry


class TestScopeStack:
    def test_no_scope_current_is_none(self):
        assert current_cancel_event() is None
        assert effective_cancel() is None
        raise_if_call_cancelled()  # no-op without a scope — must not raise

    def test_push_pop_and_innermost_observation(self):
        outer, inner = threading.Event(), threading.Event()
        with call_cancel_scope(outer):
            assert current_cancel_event() is outer
            with call_cancel_scope(inner):
                assert current_cancel_event() is inner
            assert current_cancel_event() is outer
        assert current_cancel_event() is None

    def test_exception_inside_nested_scope_pops_cleanly(self):
        outer, inner = threading.Event(), threading.Event()
        with (
            pytest.raises(RuntimeError),
            call_cancel_scope(outer),
            call_cancel_scope(inner),
        ):
            raise RuntimeError("boom")
        assert current_cancel_event() is None, (
            "exception must not leave a stale scope for the next dispatch"
        )

    def test_raise_if_call_cancelled_only_when_set(self):
        ev = threading.Event()
        with call_cancel_scope(ev):
            raise_if_call_cancelled()  # unset → silent
            ev.set()
            with pytest.raises(CallCancelledError):
                raise_if_call_cancelled()


class TestEffectiveCancelMerge:
    def test_single_fallback_returned_as_is(self):
        ev = threading.Event()
        assert effective_cancel(ev) is ev

    def test_scope_only_returned_as_is(self):
        ev = threading.Event()
        with call_cancel_scope(ev):
            assert effective_cancel() is ev

    def test_scope_and_fallback_composite_or(self):
        scope, cfg = threading.Event(), threading.Event()
        with call_cancel_scope(scope):
            merged = effective_cancel(cfg)
            assert merged is not None and merged is not scope and merged is not cfg
            assert merged.is_set() is False
            cfg.set()  # fallback (agent ESC) alone must trip the composite
            assert merged.is_set() is True
            cfg.clear()
            scope.set()  # scope alone too
            assert merged.is_set() is True


class _BareRegistry:
    """ToolRegistry without __init__: the entry cancel check runs BEFORE any
    instance attribute beyond ``config`` is touched, so this is the minimal
    harness for it. An AttributeError below the checkpoint proves the call got
    PAST the gate (``_arg_repairer`` is the first attr it reaches)."""

    @staticmethod
    def make(cancel_event=None):
        reg = ToolRegistry.__new__(ToolRegistry)
        reg.config = SimpleNamespace(cancel_event=cancel_event)
        return reg


class TestDispatchEntryCheckpoint:
    def test_scope_set_before_start_returns_cancelled_without_running(self):
        reg = _BareRegistry.make(cancel_event=None)  # agent ESC NOT set
        ev = threading.Event()
        ev.set()  # …but the per-call scope is (caller abandoned a queued call)
        with call_cancel_scope(ev):
            result = reg._dispatch_impl("read_file", {"path": "x"})
        assert result.ok is False
        assert result.error == "Operation cancelled"

    def test_unset_scope_lets_the_call_proceed(self):
        reg = _BareRegistry.make(cancel_event=None)
        ev = threading.Event()
        # Past the checkpoint → crashes on the first real instance attr,
        # proving the gate did not fire on an UNSET scope.
        with call_cancel_scope(ev), pytest.raises(AttributeError):
            reg._dispatch_impl("read_file", {"path": "x"})
