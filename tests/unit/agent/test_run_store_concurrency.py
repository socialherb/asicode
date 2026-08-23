"""Concurrency regression tests for the live InMemoryRunStore surface.

Two areas are covered:

* Per-thread model context (``set_model_context`` / ``model_context_scope``):
  the store is a process-lifetime singleton shared across concurrent sessions;
  a shared instance field would let session B's model overwrite session A's
  mid-run. ``threading.local`` keeps each worker thread isolated, and a fresh
  thread must never inherit another session's model.
* Adaptive hub cache: the hub is keyed by the per-model persistence namespace
  (not a single shared instance field), so a sub-agent's flush cannot carry the
  parent's accumulated blob into its namespace, and racing the lazy init must
  build exactly one hub.
"""

from __future__ import annotations

import sys
import threading

import pytest

from external_llm.agent.run_store import InMemoryRunStore


@pytest.fixture(autouse=True)
def _aggressive_thread_switching():
    """Force byte-granular thread switching for the duration of each test.

    CPython's GIL releases only between bytecodes, and the default switch
    interval (~5ms) rarely lands inside the hub lazy-init / context RMW
    windows — so a naive stress test passes even with the lock removed, hiding
    the race. Dropping the interval to the minimum makes the interpreter switch
    at (nearly) every bytecode boundary, reliably exposing the corruption a
    missing lock would cause. Restored afterward so other tests keep their
    default cadence.
    """
    prev = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    try:
        yield
    finally:
        sys.setswitchinterval(prev)


# ── Per-thread model context (Bug 1: set_model_context race) ───────────────────


def test_model_context_is_thread_local_not_shared():
    """Concurrent sessions must not overwrite each other's model context.

    The run_store is a process-lifetime singleton shared across concurrent sessions.
    A shared instance field would let session B's set_model_context overwrite
    session A's model mid-run, so A's run-completion telemetry attributes to B's
    model — corrupting per-model learning data. With threading.local, each thread
    observes only its own model even while another thread sets a different one.
    """
    store = InMemoryRunStore()
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
    assert observations.get("A") == "gpt-4o", f"session A saw the wrong model (cross-contamination): {observations!r}"
    assert observations.get("B") == "claude-sonnet", (
        f"session B saw the wrong model (cross-contamination): {observations!r}"
    )


def test_model_context_scope_restores_parent_context():
    """model_context_scope must restore the caller's prior model on exit.

    Sequential orchestrator mode reuses the parent thread for sub-agents; without
    restore, the last sub-agent's model would leak into subsequent parent work
    (planner bias reads / telemetry). The scope saves and restores.
    """
    store = InMemoryRunStore()
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
    store = InMemoryRunStore()
    store.set_model_context(planner_model="session-A-model")  # current (main) thread

    seen_on_new_thread = {}

    def fresh_thread():
        seen_on_new_thread["v"] = store._model_name

    t = threading.Thread(target=fresh_thread)
    t.start()
    t.join(timeout=5)

    assert seen_on_new_thread["v"] == "", f"new thread inherited another session's model: {seen_on_new_thread!r}"


# ── Adaptive hub: cache key must track the persistence namespace ─────────────
# The namespace is thread-local (per model context) but the hub used to be a
# single shared instance field, so the two disagreed. The hub was loaded once
# under whichever namespace the first caller's thread had — in practice the
# parent's generic "adaptive_hub", since agent_loop constructs the store with no
# model_name — and a sub-agent thread inside model_context_scope then SAVED that
# same shared object under "adaptive_hub/<model>". The per-model namespace was
# therefore write-only, and every sub-agent flush copied the parent's whole blob
# into it, after which the two copies drifted.


def _trace_namespaces(monkeypatch):
    """Record (op, namespace) for every hub load/save; returns the list."""
    from external_llm.editor.learning import strategy_state

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(strategy_state, "read_namespace", lambda ns, *a, **k: seen.append(("read", ns)) or {})
    monkeypatch.setattr(strategy_state, "write_namespace", lambda ns, v, *a, **k: seen.append(("write", ns)) or True)
    return seen


def test_hub_cache_is_keyed_by_namespace_not_shared(monkeypatch):
    """Each model context gets its OWN hub, so saves cannot cross namespaces."""
    seen = _trace_namespaces(monkeypatch)
    store = InMemoryRunStore()

    parent_hub = store._get_adaptive_hub()
    sub_hub_holder: dict[str, object] = {}

    def sub_agent():
        with store.model_context_scope(planner_model="m-1", developer_model="m-1"):
            sub_hub_holder["hub"] = store._get_adaptive_hub()

    t = threading.Thread(target=sub_agent)
    t.start()
    t.join(timeout=5)

    assert parent_hub is not None and sub_hub_holder.get("hub") is not None
    assert parent_hub is not sub_hub_holder["hub"], (
        "sub-agent reused the parent's hub object; its save would write the "
        "parent's accumulated blob into adaptive_hub/m-1"
    )
    assert set(store._adaptive_hubs) == {"adaptive_hub", "adaptive_hub/m-1"}
    hub_reads = [ns for op, ns in seen if op == "read" and ns.startswith("adaptive_hub")]
    assert "adaptive_hub/m-1" in hub_reads, "per-model namespace was never READ — it was write-only before this fix"


def test_sub_agent_save_does_not_carry_the_parent_blob(monkeypatch):
    """The per-model namespace must receive the SUB-AGENT's state, not the parent's.

    Asserting only on the namespace written is not enough — that was already
    "adaptive_hub/<model>" before the fix. The defect was in the payload: the
    shared hub carried the parent's accumulated signals, so every sub-agent
    flush duplicated them into the per-model namespace. So this checks the
    saved value, and is written to fail against the pre-fix store.
    """
    from external_llm.editor.learning import strategy_state

    writes: list[tuple[str, object]] = []
    monkeypatch.setattr(strategy_state, "read_namespace", lambda ns, *a, **k: {})
    monkeypatch.setattr(
        strategy_state,
        "write_namespace",
        lambda ns, v, *a, **k: writes.append((ns, v)) or True,
    )
    store = InMemoryRunStore()
    store.record_tool_usage("phase", "parent_only_tool", True)
    assert [ns for ns, _ in writes] == ["adaptive_hub"], writes
    writes.clear()

    def sub_agent():
        with store.model_context_scope(planner_model="m-2", developer_model="m-2"):
            store.record_tool_usage("phase", "sub_only_tool", True)

    t = threading.Thread(target=sub_agent)
    t.start()
    t.join(timeout=5)

    assert [ns for ns, _ in writes] == ["adaptive_hub/m-2"], writes
    payload = repr(writes[0][1])
    assert "sub_only_tool" in payload, "sub-agent's own signal missing from its namespace"
    assert "parent_only_tool" not in payload, "parent's accumulated signals were duplicated into adaptive_hub/m-2"


def test_concurrent_hub_init_builds_exactly_one(monkeypatch):
    """Racing the lazy init must not orphan a hub.

    Without the lock two threads each build one, the loser is overwritten in the
    map, and any signals already recorded into it are dropped — saving reads the
    map, so the orphan is never persisted.
    """
    _trace_namespaces(monkeypatch)
    for _ in range(20):
        store = InMemoryRunStore()
        start = threading.Barrier(12)
        ids: set[int] = set()
        ids_lock = threading.Lock()

        def grab(_store=store, _start=start, _ids=ids, _ids_lock=ids_lock):
            _start.wait(timeout=5)
            hub = _store._get_adaptive_hub()
            with _ids_lock:
                _ids.add(id(hub))

        threads = [threading.Thread(target=grab) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(ids) == 1, f"{len(ids)} distinct hubs built for one namespace"
