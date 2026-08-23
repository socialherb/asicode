"""Unit tests for _thread_pool.py — 100% coverage."""

import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from external_llm.agent._thread_pool import shared_pool


class TestSharedPool:
    """Tests for the global ThreadPoolExecutor."""

    def test_pool_is_executor(self):
        """shared_pool is a ThreadPoolExecutor with a dynamically sized pool."""
        expected = max(4, min(32, (os.cpu_count() or 1) + 4))
        assert hasattr(shared_pool, "submit")
        assert hasattr(shared_pool, "map")
        assert shared_pool._max_workers == expected

    def test_pool_can_execute(self):
        """Pool can submit and complete a task."""
        future = shared_pool.submit(lambda: 42)
        assert future.result() == 42

    def test_pool_is_singleton(self):
        """Repeated import returns the same instance."""
        from external_llm.agent._thread_pool import shared_pool as sp2

        assert sp2 is shared_pool


def test_idle_workers_exit_when_executor_is_dropped():
    """CPython's weakref callback is what replaced the deleted registry ``__del__``.

    ToolRegistry used to carry a ``__del__`` that shut down its thread pool
    from a GC finalizer (redundant, and it crashed the interpreter on 3.12).
    It was deleted because ThreadPoolExecutor registers
    ``weakref.ref(self, weakref_cb)`` whose callback puts None on the work
    queue, so idle workers wake and exit once the executor becomes
    unreachable. If this ever stops holding, dropping a ThreadPoolExecutor
    would start leaking idle worker threads in a long-lived process — pinned
    here rather than assumed (moved from the deleted
    test_registry_executor_lifecycle.py).
    """
    before = set(threading.enumerate())
    pool = ThreadPoolExecutor(max_workers=2)
    pool.submit(lambda: None).result(timeout=10)
    workers = [t for t in threading.enumerate() if t not in before]
    assert workers, "expected at least one worker thread to have been spawned"

    del pool
    gc.collect()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not any(t.is_alive() for t in workers):
            break
        time.sleep(0.05)

    assert not any(t.is_alive() for t in workers), (
        "idle workers did not exit after the executor became unreachable — "
        "CPython's weakref cleanup no longer covers what __del__ used to"
    )
