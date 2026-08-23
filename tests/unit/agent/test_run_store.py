"""
Unit tests for the live InMemoryRunStore surface — adaptive-hub batching
(record_tool_usage + batch_adaptive_signals).
"""

from unittest.mock import patch

import external_llm.agent.run_store as _rs
from external_llm.agent.run_store import InMemoryRunStore


def test_batch_adaptive_signals_collapses_n_saves_into_one():
    """batch_adaptive_signals() defers persistence: N record_* calls → 1 save."""
    store = InMemoryRunStore()

    # Without batch: each record_* call triggers a separate save.
    with patch.object(store, "_save_adaptive_hub_state", wraps=store._save_adaptive_hub_state) as spy:
        store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
        store.record_tool_usage("MAIN_AGENT", "symbol_modify", True)
        store.record_tool_usage("MAIN_AGENT", "semantic_edit", True)
        store.record_tool_usage("MAIN_AGENT", "context_gather", True)
        calls_without = spy.call_count

    # With batch: only one save at block exit.
    with patch.object(store, "_save_adaptive_hub_state", wraps=store._save_adaptive_hub_state) as spy:
        with store.batch_adaptive_signals():
            store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
            store.record_tool_usage("MAIN_AGENT", "symbol_modify", True)
            store.record_tool_usage("MAIN_AGENT", "semantic_edit", True)
            store.record_tool_usage("MAIN_AGENT", "context_gather", True)
        calls_with = spy.call_count

    assert calls_without == 4, f"expected 4 saves without batch, got {calls_without}"
    assert calls_with == 1, f"expected 1 save with batch, got {calls_with}"


def test_batch_adaptive_signals_flushes_on_exception():
    """Even if an exception fires mid-block, the deferred save still runs (finally)."""
    store = InMemoryRunStore()

    with patch.object(store, "_save_adaptive_hub_state", wraps=store._save_adaptive_hub_state) as spy:
        try:
            with store.batch_adaptive_signals():
                store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert spy.call_count == 1, f"expected 1 flush on exception, got {spy.call_count}"


def test_batch_adaptive_signals_is_reentrant():
    """A nested block must not end the outer batch early."""
    store = InMemoryRunStore()
    with patch.object(store, "_save_adaptive_hub_state") as spy:
        with store.batch_adaptive_signals():
            store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
            with store.batch_adaptive_signals():
                store.record_tool_usage("MAIN_AGENT", "symbol_modify", True)
            inner_exit_calls = spy.call_count
            store.record_tool_usage("MAIN_AGENT", "semantic_edit", True)
        total = spy.call_count

    assert inner_exit_calls == 0, f"inner block flushed and ended the outer batch ({inner_exit_calls} saves)"
    assert total == 1, f"expected a single flush at outer exit, got {total}"


def test_batch_adaptive_signals_is_thread_local():
    """One thread batching must not suspend another thread's writes.

    The store is a process-lifetime singleton that the orchestrator hands to
    concurrent subagent loops. Batching state kept in an instance field would
    let one thread's block silence every other thread.
    """
    import threading

    store = InMemoryRunStore()
    entered, release = threading.Event(), threading.Event()

    def _batcher():
        with store.batch_adaptive_signals():
            entered.set()
            release.wait(5)

    with patch.object(store, "_save_adaptive_hub_state") as spy:
        t = threading.Thread(target=_batcher)
        t.start()
        assert entered.wait(5), "batching thread never entered the block"
        spy.reset_mock()
        for _ in range(4):
            store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
        main_writes = spy.call_count
        release.set()
        t.join(5)

    assert main_writes == 4, f"main thread wrote {main_writes}/4 signals — another thread's batch leaked across threads"


def test_batch_adaptive_signals_debounces_instead_of_suspending():
    """A long batch still flushes periodically, bounding crash data loss."""
    store = InMemoryRunStore()
    n = _rs._HUB_FLUSH_MAX_PENDING * 3
    with patch.object(store, "_save_adaptive_hub_state") as spy:
        with store.batch_adaptive_signals():
            for _ in range(n):
                store.record_tool_usage("MAIN_AGENT", "apply_patch", True)
        calls = spy.call_count

    assert calls >= 3, f"{n} signals produced only {calls} flush(es); a crash would discard the whole batch"
    assert calls < n, f"debounce did nothing: {calls} flushes for {n} signals"


def test_batch_adaptive_signals_no_write_when_nothing_recorded():
    """An empty block must not cost a write."""
    store = InMemoryRunStore()
    with patch.object(store, "_save_adaptive_hub_state") as spy:
        with store.batch_adaptive_signals():
            pass
        assert spy.call_count == 0, f"empty batch wrote {spy.call_count} time(s)"
