"""Regression tests for the per-session compress-lock registry (leak #7).

Guards the 7th instance of the per-repo/per-session module-level leak class
fixed across this repo. ``_MODULE_COMPRESS_LOCKS`` (in ``context_manager``)
is a process-global registry keyed by ``session_id`` — every distinct design
chat session that triggers background compression creates a ``threading.Lock``
entry. The only explicit removal path is
``DesignSessionManager.delete_session`` (the ``.pop`` in ``design_session.py``),
which fires solely on an explicit DELETE request; normal sessions are created,
used, and abandoned, so a plain ``dict`` would leak one Lock per session for the
process lifetime of a long-lived webapp/CLI.

Fix (mirrors leak #6 ``_INSIGHTS_THREAD_LOCKS`` and the proven
``orchestrator._file_locks`` pattern): a ``weakref.WeakValueDictionary``. Locks
are weakref-able, so idle entries are GC'd once no caller holds a strong
reference; active compressions keep entries alive because the caller binds the
returned Lock to a local that outlives the operation
(``schedule_background_compress`` / ``compact_now``), preserving cross-instance
Lock identity exactly when dedup must hold. ``_get_compress_lock`` uses a
strong local binding to survive the assign-to-return window — a naive
assign-then-re-read races with GC under ``WeakValueDictionary``.

Mutation guards: if ``_MODULE_COMPRESS_LOCKS`` is reverted to a plain ``dict``,
``test_gc_evicts_idle_session_lock`` fails (the entry survives the GC). If the
strong-local-binding is dropped in favor of assign-then-re-read, the identity /
dedup tests become racy (intermittently creating a fresh Lock and breaking
dedup).
"""
from __future__ import annotations

import gc

import pytest

from external_llm.agent.context_manager import (
    SessionCompressionContext,
    _MODULE_COMPRESS_LOCKS,
)


@pytest.fixture(autouse=True)
def _clear_lock_registry():
    """Isolation: clear the module-level WeakValueDictionary around each test."""
    _MODULE_COMPRESS_LOCKS.clear()
    gc.collect()
    yield
    _MODULE_COMPRESS_LOCKS.clear()
    gc.collect()


@pytest.fixture
def ctx(tmp_path):
    return SessionCompressionContext(str(tmp_path))


def test_lock_registry_is_weakvaluedictionary():
    """The registry must be a WeakValueDictionary (plain dict leaks per session)."""
    import weakref

    assert isinstance(_MODULE_COMPRESS_LOCKS, weakref.WeakValueDictionary)


def test_gc_evicts_idle_session_lock(ctx):
    """Idle session locks must be reclaimed by GC (the core leak fix).

    Mutation guard: revert ``_MODULE_COMPRESS_LOCKS`` to a plain ``dict`` and
    this assertion fails — the entry survives the release+GC.
    """
    sid = "idle-session-xyz"
    lock = ctx._get_compress_lock(sid)
    assert sid in _MODULE_COMPRESS_LOCKS  # alive while strong ref held

    del lock
    gc.collect()

    assert sid not in _MODULE_COMPRESS_LOCKS, (
        "idle session lock leaked — registry is not weakly-referenced"
    )


def test_strong_ref_keeps_entry_alive_across_return(ctx):
    """The returned Lock is a strong reference, so the weak entry survives
    until the caller drops it — no assign-to-return GC race."""
    sid = "held-session"
    lock = ctx._get_compress_lock(sid)
    try:
        # While the caller holds `lock`, the entry MUST remain present even
        # under explicit GC. A naive inline assign-then-re-read that lost the
        # strong binding between assign and return would intermittently drop
        # the entry here.
        gc.collect()
        assert sid in _MODULE_COMPRESS_LOCKS
    finally:
        del lock


def test_same_session_same_lock_identity_while_held(ctx):
    """Two lookups for the same session, while a strong ref is held, return the
    IDENTICAL Lock — the invariant that makes cross-instance dedup work."""
    sid = "shared-session"
    first = ctx._get_compress_lock(sid)
    second = ctx._get_compress_lock(sid)  # `first` still holds a strong ref
    try:
        assert first is second
    finally:
        del first, second


def test_overlapping_same_session_dedup_blocks_second(ctx):
    """Simulate two concurrent schedule_background_compress callers for the
    same session. Dedup relies on both seeing the SAME Lock while either holds
    a strong reference: caller A wins the non-blocking acquire, caller B's
    acquire on the same Lock then fails (skip).

    Under WeakValueDictionary this holds because caller A's bound local keeps
    the entry alive across caller B's lookup. A broken implementation that let
    the entry be GC'd between A's acquire and B's lookup would hand B a fresh
    Lock and both would run — redundant LLM summary cost.
    """
    sid = "overlapping-session"
    lock_a = ctx._get_compress_lock(sid)
    assert lock_a.acquire(blocking=False) is True  # caller A wins
    try:
        lock_b = ctx._get_compress_lock(sid)  # entry alive (lock_a strong ref)
        assert lock_b is lock_a  # SAME identity -> dedup works
        assert lock_b.acquire(blocking=False) is False  # caller B blocked
    finally:
        lock_a.release()
        del lock_a


def test_distinct_sessions_distinct_locks(ctx):
    """Different sessions must get independent Locks (no accidental sharing)."""
    a = ctx._get_compress_lock("sess-a")
    b = ctx._get_compress_lock("sess-b")
    try:
        assert a is not b
    finally:
        del a, b


def test_delete_session_pop_is_harmless_on_weak_registry(ctx):
    """DesignSessionManager.delete_session does
    ``self._ctx._compress_locks.pop(session_id, None)``. On a
    WeakValueDictionary this must be a no-op when the entry was already GC'd
    (the common idle case) — ``.pop`` with a default never raises."""
    sid = "deleted-session"
    lock = ctx._get_compress_lock(sid)
    del lock
    gc.collect()
    assert sid not in _MODULE_COMPRESS_LOCKS  # already reclaimed

    # The delete_session cleanup path must not raise on a missing entry.
    assert ctx._compress_locks.pop(sid, None) is None


def test_per_request_instances_share_one_registry(tmp_path):
    """Each per-request SessionCompressionContext aliases the SAME
    module-global registry, so cross-instance dedup of overlapping compressions
    holds (the original rationale for making it module-level)."""
    ctx1 = SessionCompressionContext(str(tmp_path))
    ctx2 = SessionCompressionContext(str(tmp_path))
    sid = "cross-instance-session"
    lock1 = ctx1._get_compress_lock(sid)
    lock2 = ctx2._get_compress_lock(sid)
    try:
        assert lock1 is lock2  # shared identity across instances
    finally:
        del lock1, lock2
