"""Per-call cooperative cancellation scope.

CPython cannot forcibly terminate a running thread. An abandoned tool call —
an MCP ``wait_for`` timeout on an executor worker, or an ESC that aborts a
``dispatch_parallel`` batch mid-flight — therefore keeps occupying its pool
slot unless the call *cooperates*: whoever abandons the call sets a
``threading.Event``, and the call observes it at natural boundaries (dispatch
entry, scanner-to-scanner, opt-in scanner internals) and stops early.

The scope is a thread-local STACK installed by executor submit sites around
``ToolRegistry.dispatch``. Threads without a scope (the serial agent loop,
plain tests) see ``None`` everywhere and keep their exact current behavior.
Pooled threads get a fresh Event per submission, so a set event can never leak
into a later dispatch.

This is deliberately separate from ``config.cancel_event`` (agent-loop ESC:
user intent for the WHOLE turn) — a scope event is one caller abandoning ONE
call. Checkpoints that must observe both use :func:`effective_cancel`, which
returns a composite exposing ``is_set()`` — the only method the cooperative
channel consumes (vulture's ``_is_cancelled``, the dispatch entry check, the
scanner-to-scanner checkpoint).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class CallCancelledError(Exception):
    """Raised inside a dispatch whose caller set the call's cancel event."""


class _CompositeCancel:
    """``is_set()``/``wait()`` OR over several cancel sources; duck-types an Event.

    Cooperative consumers call ``is_set()`` (dispatch checkpoints) or
    ``wait(timeout)`` (``interruptible_sleep`` in the browser/web-search
    backoffs), so the composite must serve both — a read-only union where each
    source keeps its own setter, and any set source trips the composite.
    """

    __slots__ = ("_events",)

    _POLL_GRANULARITY = 0.02  # 20ms sleep grain when waiting on >1 Event

    def __init__(self, *events) -> None:
        self._events = tuple(e for e in events if e is not None)

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until any source is set (True) or *timeout* elapses (False).

        Mirrors ``threading.Event.wait`` semantics: ``timeout=None`` waits
        forever. Uses ``is_set()`` polling because cross-event blocking would
        need a condition/propagator — there is at most one blocking wait at a
        time in any pool worker, and 20ms grain is far below the 1.5s+
        backoffs it guards, so responsiveness is preserved.
        """
        if self.is_set():
            return True
        if timeout is None:
            while not self.is_set():
                time.sleep(self._POLL_GRANULARITY)
            return True
        deadline = time.monotonic() + timeout
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self._POLL_GRANULARITY, remaining))
        return True


_local = threading.local()


@contextmanager
def call_cancel_scope(event: threading.Event) -> Iterator[threading.Event]:
    """Install *event* as this thread's innermost per-call cancel source.

    Symmetric push/pop: the ``finally`` pops exactly what the ``try`` pushed
    (LIFO by construction — nested scopes unwind inside-out), so an exception
    from any depth never leaves a stale entry for the NEXT dispatch on this
    pooled thread.
    """
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    stack.append(event)
    try:
        yield event
    finally:
        stack.pop()


def current_cancel_event() -> threading.Event | None:
    """Innermost per-call cancel event for THIS thread, ``None`` if no scope."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


def effective_cancel(
    *fallbacks: threading.Event | None,
) -> threading.Event | _CompositeCancel | None:
    """Merge the active scope with *fallbacks* (e.g. ``config.cancel_event``).

    Returns ``None`` when no source exists (serial dispatch: zero behavior
    change), the single live source itself when only one exists (no wrapper
    allocation on the common paths), else a composite whose ``is_set()`` ORs
    every source.
    """
    live = [s for s in (current_cancel_event(), *fallbacks) if s is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]
    return _CompositeCancel(*live)


def raise_if_call_cancelled() -> None:
    """Raise :class:`CallCancelledError` when this call's scope event is set.

    No-op without an active scope — safe to call from any layer.
    """
    ev = current_cancel_event()
    if ev is not None and ev.is_set():
        raise CallCancelledError("call cancelled by its caller")
