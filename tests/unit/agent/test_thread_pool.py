"""Unit tests for _thread_pool.py — 100% coverage."""

import os

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
