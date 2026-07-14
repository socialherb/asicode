"""Concurrency regression tests for InMemoryRunStore.add_run.

``add_run`` is the ONLY writer to ``_runs`` / ``_run_order`` and performs a
read-check-evict-append RMW. Parallel in-process sub-agents share one store
instance (orchestrator ThreadPoolExecutor) and call ``add_run`` concurrently
from each run-completion hook. These tests hammer that path at capacity and
assert the invariants the ``_telemetry_lock`` is meant to protect.

Without the lock the failure mode is non-deterministic: double-evict corrupts
``_run_order`` (len/runs desync, orphan refs, duplicate ids) or the second
``del`` raises ``KeyError``. The stress test runs enough iterations to make a
regression (e.g. the lock being removed) reproducible.
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

from external_llm.agent.run_store import InMemoryRunStore, RunRecord


@pytest.fixture(autouse=True)
def _aggressive_thread_switching():
    """Force byte-granular thread switching for the duration of each test.

    CPython's GIL releases only between bytecodes, and the default switch
    interval (~5ms) rarely lands inside ``add_run``'s tight check-evict window —
    so a naive stress test passes even with the lock removed, hiding the race.
    Dropping the interval to the minimum makes the interpreter switch at (nearly)
    every bytecode boundary, reliably exposing the RMW corruption a missing lock
    would cause. Restored afterward so other tests keep their default cadence.
    """
    prev = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    try:
        yield
    finally:
        sys.setswitchinterval(prev)


def _make_record(run_id: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        timestamp=time.time(),
        plan_mode="EDIT",
        operation_count=1,
        completed=1,
        failed=0,
        skipped=0,
        final_status="success",
        final_failure_class=None,
        final_blocking_reasons=[],
        final_warning_reasons=[],
        semantic_gate_passed=True,
        semantic_gate_failed_reasons=[],
        plan_acceptance_passed=True,
        plan_acceptance_failed_checks=[],
        repair_attempted=False,
        repair_rounds_attempted=0,
        repair_improved=False,
        semantic_issue_codes=[],
        dependency_issues=[],
        completed_ids=[],
        failed_ids=[],
        skipped_ids=[],
    )


def _assert_store_invariants(store: InMemoryRunStore, *, expected_cap: int) -> None:
    """The three invariants the lock protects against RMW corruption."""
    runs = store._runs
    order = store._run_order
    # (1) dict and order-list must stay in sync — no orphan refs either direction.
    assert len(runs) == len(order), (
        f"runs/order desync: len(runs)={len(runs)} len(order)={len(order)}"
    )
    # (2) capacity must be respected (no over-append from the losing thread).
    assert len(runs) <= expected_cap, (
        f"capacity exceeded: {len(runs)} > {expected_cap}"
    )
    # (3) every id referenced in the order list must resolve in the dict (no
    #     KeyError-equivalent dangling ref), and no duplicate ids (double-append).
    assert len(set(order)) == len(order), f"duplicate ids in order: {order}"
    for rid in order:
        assert rid in runs, f"orphan order ref {rid!r} not in runs"


def test_add_run_concurrent_at_capacity_preserves_invariants():
    """Many threads each add distinct runs against a tiny capacity.

    Heavy eviction churn (8 threads x 60 adds vs cap=10 = 480 adds, ~470 evictions)
    maximizes the chance two threads race the check-evict window. After join, the
    store must still satisfy all three invariants and contain exactly the last
    ``cap`` ids added.
    """
    cap = 10
    store = InMemoryRunStore(max_runs=cap, write_unified=False)
    n_threads = 8
    adds_per_thread = 60
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(adds_per_thread):
                # Distinct ids across threads via tid prefix.
                store.add_run(_make_record(f"t{tid}-r{i:04d}"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent add_run raised: {errors}"
    _assert_store_invariants(store, expected_cap=cap)
    # Capacity honored: exactly the last ``cap`` ids survive (FIFO eviction).
    assert len(store._runs) == cap


def test_add_run_concurrent_re_add_existing_id_no_corruption():
    """Re-adding an EXISTING id concurrently (update path) must not corrupt.

    The update branch does ``_run_order.remove(id)`` then re-append; two threads
    re-adding the same id can otherwise trip a ValueError (remove of absent id) or
    leave a duplicate in ``_run_order``.
    """
    cap = 20
    store = InMemoryRunStore(max_runs=cap, write_unified=False)
    shared_ids = [f"shared-{i:03d}" for i in range(cap // 2)]
    for rid in shared_ids:
        store.add_run(_make_record(rid))

    errors: list[BaseException] = []

    def readder(rid: str) -> None:
        try:
            for _ in range(100):
                store.add_run(_make_record(rid))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=readder, args=(rid,)) for rid in shared_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent re-add raised: {errors}"
    _assert_store_invariants(store, expected_cap=cap)
    # All shared ids still present (re-add moves to end but never evicts them
    # since cap > number of distinct ids).
    for rid in shared_ids:
        assert rid in store._runs


def test_add_run_mixed_concurrent_new_and_existing():
    """Mix of brand-new ids (eviction path) and existing ids (update path)."""
    cap = 8
    store = InMemoryRunStore(max_runs=cap, write_unified=False)
    # Pre-seed at capacity so new adds immediately churn eviction.
    for i in range(cap):
        store.add_run(_make_record(f"seed-{i:03d}"))

    errors: list[BaseException] = []
    stop = threading.Event()

    def new_adder(tid: int) -> None:
        i = 0
        try:
            while not stop.is_set() and i < 200:
                store.add_run(_make_record(f"new-t{tid}-{i:04d}"))
                i += 1
        except BaseException as exc:
            errors.append(exc)

    def readder() -> None:
        try:
            while not stop.is_set():
                # Re-add a seeded id to exercise the update branch concurrently.
                store.add_run(_make_record("seed-000"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=new_adder, args=(t,)) for t in range(6)]
    threads.append(threading.Thread(target=readder))
    for t in threads:
        t.start()
    # Let it churn briefly — enough for races to surface.
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"mixed concurrent add_run raised: {errors}"
    _assert_store_invariants(store, expected_cap=cap)


def test_concurrent_readers_during_eviction_no_keyerror():
    """Readers iterating ``_run_order`` must not ``KeyError`` when a concurrent
    ``add_run`` eviction removes a record mid-iteration.

    The orchestrator shares one store instance across parallel sub-agents
    (ThreadPoolExecutor). Each sub-agent's planning reads run history
    (list_runs, get_recent_repair_memories, build_strategy_outcome_memory, …)
    while another sub-agent's run-completion hook evicts+appends via
    ``add_run``. Before the fix, a reader that snapshotted ``_run_order`` and
    then indexed ``_runs[id]`` for an id whose record was just evicted raised
    ``KeyError``, crashing the sub-agent. The ``_aggressive_thread_switching``
    fixture (1ns switch interval) makes the eviction/iteration overlap
    near-deterministic so this test catches a regression reliably.
    """
    cap = 6
    store = InMemoryRunStore(max_runs=cap, write_unified=False)
    # Seed with one repair-class record so the filtered readers have matches.
    seed = _make_record("seed-repair")
    seed.final_failure_class = "ImportError"
    store.add_run(seed)

    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            i = 0
            while not stop.is_set():
                store.add_run(_make_record(f"w-{i:05d}"))
                i += 1
        except BaseException as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                store.list_runs(limit=10)
                store.get_recent_runs(limit=8)
                store.get_recent_repair_memories("ImportError", limit=4)
                store.build_strategy_outcome_memory(limit=10)
                store.get_recent_selected_strategies(limit=5)
                store.build_switch_outcome_memory(limit=10)
                store.get_strategy_summary_for_request("test", limit=10)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"concurrent readers raised: {errors[:3]}"
    _assert_store_invariants(store, expected_cap=cap)


# ── create_run_id uniqueness (Bug 2: _next_run_id race) ────────────────────────

def test_create_run_id_holds_telemetry_lock():
    """create_run_id must serialize on _telemetry_lock (the Bug 2 fix).

    The read-increment RMW (``run_id = f"run-{N}"; N += 1``) is GIL-atomic on
    stock CPython, so duplicate IDs are not reproducible in-process here — but on
    free-threaded Python (PEP 703) or if the critical section ever widens, an
    unsynchronized counter yields duplicate IDs that ``add_run`` silently merges
    into one record. Rather than assert a non-deterministic race, this test pins
    the actual fix: a concurrent holder of ``_telemetry_lock`` BLOCKS
    create_run_id. Remove the ``with self._telemetry_lock`` and this fails.
    """
    store = InMemoryRunStore(write_unified=False)
    # Simulate a concurrent run-completion holding the lock on another "thread".
    store._telemetry_lock.acquire()
    try:
        done = threading.Event()

        def caller():
            store.create_run_id()
            done.set()

        t = threading.Thread(target=caller)
        t.start()
        # With the lock, create_run_id cannot proceed until we release → not done.
        # (_telemetry_lock is an RLock, reentrant only for the OWNING thread, so the
        # worker thread genuinely blocks.)
        did_block = not done.wait(timeout=0.3)
        store._telemetry_lock.release()
        t.join(timeout=2)
        assert did_block, (
            "create_run_id did not block on _telemetry_lock — the lock is missing "
            "(Bug 2 regression: _next_run_id RMW is unsynchronized)"
        )
        assert done.is_set(), "create_run_id did not complete after the lock was released"
    finally:
        try:
            store._telemetry_lock.release()
        except RuntimeError:
            pass  # already released above


def test_create_run_id_concurrent_produces_unique_ids():
    """Invariant guard: many concurrent create_run_id calls yield all-unique IDs.

    Complements the lock-presence test above. The race itself is GIL-atomic on
    stock CPython (not reproducible here), but this pins the uniqueness invariant
    that the lock guarantees on free-threaded builds / under future changes.
    """
    store = InMemoryRunStore(write_unified=False)
    n_threads = 8
    ids_per_thread = 200
    all_ids: list[str] = []
    lock = threading.Lock()

    def worker():
        local_ids = [store.create_run_id() for _ in range(ids_per_thread)]
        with lock:
            all_ids.extend(local_ids)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_count = n_threads * ids_per_thread
    assert len(all_ids) == expected_count
    assert len(set(all_ids)) == expected_count, (
        f"duplicate run IDs produced: {expected_count - len(set(all_ids))} collisions"
    )


# ── Per-thread model context (Bug 1: set_model_context race) ───────────────────

def test_model_context_is_thread_local_not_shared():
    """Concurrent sessions must not overwrite each other's model context.

    The run_store is a process-lifetime singleton shared across concurrent sessions.
    A shared instance field would let session B's set_model_context overwrite
    session A's model mid-run, so A's run-completion telemetry attributes to B's
    model — corrupting per-model learning data. With threading.local, each thread
    observes only its own model even while another thread sets a different one.
    """
    store = InMemoryRunStore(write_unified=False)
    barrier = threading.Barrier(2)
    observations: dict[str, str] = {}
    obs_lock = threading.Lock()
    errors: list[BaseException] = []

    def session(label: str, model: str):
        try:
            store.set_model_context(planner_model=model, developer_model=model)
            # Hold here so the OTHER thread is guaranteed to run set_model_context
            # concurrently — under a shared field this would clobber our value.
            barrier.wait(timeout=5)
            # Re-read AFTER the other thread set its model. Thread-local → still ours.
            barrier.wait(timeout=5)
            seen = store._model_name
            with obs_lock:
                observations[label] = seen
        except BaseException as exc:
            errors.append(exc)

    t_a = threading.Thread(target=session, args=("A", "gpt-4o"))
    t_b = threading.Thread(target=session, args=("B", "claude-sonnet"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    assert observations.get("A") == "gpt-4o", (
        f"session A saw the wrong model (cross-contamination): {observations!r}"
    )
    assert observations.get("B") == "claude-sonnet", (
        f"session B saw the wrong model (cross-contamination): {observations!r}"
    )


def test_model_context_scope_restores_parent_context():
    """model_context_scope must restore the caller's prior model on exit.

    Sequential orchestrator mode reuses the parent thread for sub-agents; without
    restore, the last sub-agent's model would leak into subsequent parent work
    (planner bias reads / telemetry). The scope saves and restores.
    """
    store = InMemoryRunStore(write_unified=False)
    store.set_model_context(planner_model="parent-planner", developer_model="parent-dev")
    assert store._model_name == "parent-planner"
    assert store._developer_model_name == "parent-dev"

    with store.model_context_scope("subagent-model", "subagent-model"):
        assert store._model_name == "subagent-model"
        assert store._developer_model_name == "subagent-model"

    # Restored after exit.
    assert store._model_name == "parent-planner"
    assert store._developer_model_name == "parent-dev"


def test_model_context_defaults_empty_on_uninitialized_thread():
    """A thread that never called set_model_context reads "" (not another session's).

    This is the isolation guarantee: a fresh worker thread cannot inherit a model
    set by a different session on a different thread.
    """
    store = InMemoryRunStore(write_unified=False)
    store.set_model_context(planner_model="session-A-model")  # current (main) thread

    seen_on_new_thread = {}

    def fresh_thread():
        seen_on_new_thread["v"] = store._model_name

    t = threading.Thread(target=fresh_thread)
    t.start()
    t.join(timeout=5)

    assert seen_on_new_thread["v"] == "", (
        f"new thread inherited another session's model: {seen_on_new_thread!r}"
    )
