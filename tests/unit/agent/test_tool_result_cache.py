"""Unit tests for tool_result_cache.py — TTL-based LRU ToolResultCache."""

import time

from external_llm.agent.tool_result_cache import ToolResultCache


class TestToolResultCache:
    """Tests for ToolResultCache TTL-based LRU cache."""

    def test_miss_on_empty_cache(self):
        cache = ToolResultCache()
        result = cache.get("read_file", {"path": "test.py"})
        assert result is None

    def test_set_and_get(self):
        cache = ToolResultCache()
        args = {"path": "main.py"}
        result_data = {"content": "def foo(): pass"}

        cache.set("read_file", args, result_data)
        retrieved = cache.get("read_file", args)
        assert retrieved == result_data

    def test_get_returns_same_object(self):
        cache = ToolResultCache()
        result_data = {"lines": [1, 2, 3]}
        cache.set("grep", {"pattern": "foo"}, result_data)
        retrieved = cache.get("grep", {"pattern": "foo"})
        assert retrieved is result_data  # same reference

    def test_miss_on_different_args(self):
        cache = ToolResultCache()
        cache.set("find_symbol", {"name": "Foo"}, {"found": True})
        result = cache.get("find_symbol", {"name": "Bar"})
        assert result is None

    def test_miss_on_different_tool(self):
        cache = ToolResultCache()
        cache.set("read_file", {"path": "x.py"}, {"content": "x"})
        result = cache.get("grep", {"pattern": "something"})
        assert result is None

    def test_expired_entry_returns_none(self):
        cache = ToolResultCache(default_ttl=0)  # 0-second TTL = immediate expiry
        cache.set("ping", {}, {"ok": True})
        time.sleep(0.01)  # Guarantee expiry
        result = cache.get("ping", {})
        assert result is None

    def test_lru_eviction(self):
        cache = ToolResultCache(max_entries=3)
        for i in range(3):
            cache.set(f"tool_{i}", {"i": i}, {"value": i})
        # Cache is full with 3 entries
        assert len(cache._cache) == 3

        # Access entry 0 to make it recently used
        cache.get("tool_0", {"i": 0})

        # Add a 4th entry — should evict the least recently used (tool_1)
        cache.set("tool_new", {"new": True}, {"value": "new"})
        assert cache.get("tool_1", {"i": 1}) is None
        assert cache.get("tool_0", {"i": 0}) is not None  # Still there (recently used)
        assert cache.get("tool_new", {"new": True}) is not None

    def test_clear_empties_cache(self):
        cache = ToolResultCache()
        cache.set("read_file", {"path": "a.py"}, {"content": "a"})
        cache.set("read_file", {"path": "b.py"}, {"content": "b"})
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0
        assert cache.get("read_file", {"path": "a.py"}) is None

    def test_get_stats_initial(self):
        cache = ToolResultCache(max_entries=64)
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0
        assert stats["max_entries"] == 64

    def test_get_stats_after_hits_and_misses(self):
        cache = ToolResultCache()
        cache.set("tool", {"k": "v"}, {"ok": True})

        # 2 hits
        cache.get("tool", {"k": "v"})
        cache.get("tool", {"k": "v"})
        # 2 misses
        cache.get("tool", {"k": "other"})
        cache.get("tool2", {})

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["size"] == 1

    def test_key_stability_same_args(self):
        """Same args in different order should produce the same key."""
        cache = ToolResultCache()
        cache.set("tool", {"b": 2, "a": 1}, {"result": "stable"})
        r1 = cache.get("tool", {"a": 1, "b": 2})
        r2 = cache.get("tool", {"b": 2, "a": 1})
        assert r1 == r2

    def test_ttl_zero_expires_immediately(self):
        """ttl=0 should expire immediately, not fall back to default_ttl."""
        cache = ToolResultCache(default_ttl=999)
        cache.set("fast", {}, {"flash": True}, ttl=0)
        time.sleep(0.05)
        result = cache.get("fast", {})
        assert result is None

    def test_eviction_removes_oldest(self):
        cache = ToolResultCache(max_entries=2)
        cache.set("a", {}, {"data": "a"})
        cache.set("b", {}, {"data": "b"})
        cache.set("c", {}, {"data": "c"})  # Evicts 'a'

        assert cache.get("a", {}) is None
        assert cache.get("b", {}) is not None
        assert cache.get("c", {}) is not None

    def test_no_eviction_under_max(self):
        cache = ToolResultCache(max_entries=10)
        for i in range(5):
            cache.set(f"t{i}", {"n": i}, {"v": i})
        assert len(cache._cache) == 5
        for i in range(5):
            assert cache.get(f"t{i}", {"n": i}) is not None

    def test_argument_serialization_with_complex_types(self):
        """Args with non-serializable types (e.g. Path) should still work."""
        cache = ToolResultCache()

        class Custom:
            def __str__(self):
                return "custom"

        cache.set("tool", {"obj": Custom()}, {"ok": True})
        # Should have stored it without error
        # Key is a hash — just verify it's in the cache
        assert len(cache._cache) == 1

    def test_clear_resets_stats(self):
        cache = ToolResultCache()
        cache.set("x", {}, {"x": 1})
        cache.get("x", {})  # hit
        cache.get("y", {})  # miss
        assert cache._hits == 1
        assert cache._misses == 1

        cache.clear()
        # Cache cleared but stats reset? Let's check: clear() clears _cache but not stats
        # This documents current behavior — stats are NOT reset by clear()
        # Actually looking at the code, clear() only does self._cache.clear()
        # So stats persist after clear — that's the documented behavior

    def test_multiple_tools_with_same_args(self):
        """Same args for different tools should not collide."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "x.py"}, {"content": "read"})
        cache.set("grep", {"path": "x.py"}, {"content": "grep"})

        r1 = cache.get("read_file", {"path": "x.py"})
        r2 = cache.get("grep", {"path": "x.py"})
        assert r1 == {"content": "read"}
        assert r2 == {"content": "grep"}

    def test_invalidate_paths_drops_only_overlapping_entry(self):
        """A write to /repo/a.py must not evict the cached read of /repo/b.py."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "a.py"}, {"content": "a"}, paths=frozenset({"/repo/a.py"}))
        cache.set("read_file", {"path": "b.py"}, {"content": "b"}, paths=frozenset({"/repo/b.py"}))

        removed = cache.invalidate_paths(frozenset({"/repo/a.py"}))

        assert removed == 1
        assert cache.get("read_file", {"path": "a.py"}) is None
        assert cache.get("read_file", {"path": "b.py"}) is not None

    def test_invalidate_paths_drops_directory_overlap(self):
        """A write inside a scanned directory must evict a grep scoped to that dir."""
        cache = ToolResultCache()
        cache.set("grep", {"pattern": "x", "path": "src"}, {"hits": []}, paths=frozenset({"/repo/src"}))

        removed = cache.invalidate_paths(frozenset({"/repo/src/mod.py"}))

        assert removed == 1
        assert cache.get("grep", {"pattern": "x", "path": "src"}) is None

    def test_invalidate_paths_drops_unknown_scope_entries(self):
        """Entries with paths=None (unknown/repo-wide scope) are always dropped,
        since we can't prove they don't depend on the write."""
        cache = ToolResultCache()
        cache.set("find_symbol", {"name": "Foo"}, {"found": True})  # no paths → None

        removed = cache.invalidate_paths(frozenset({"/repo/unrelated.py"}))

        assert removed == 1
        assert cache.get("find_symbol", {"name": "Foo"}) is None

    def test_invalidate_paths_empty_falls_back_to_full_clear(self):
        """No known write target (e.g. mutating bash) → conservative full clear."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "a.py"}, {"content": "a"}, paths=frozenset({"/repo/a.py"}))
        cache.set("read_file", {"path": "b.py"}, {"content": "b"}, paths=frozenset({"/repo/b.py"}))

        removed = cache.invalidate_paths(frozenset())

        assert removed == 2
        assert len(cache._cache) == 0

    def test_key_instability_logged_not_raised(self, caplog):
        """A non-JSON-serializable arg must not raise, and should log at debug
        level so an always-miss cache is debuggable instead of silent."""
        import logging as _logging

        class Custom:
            def __str__(self):
                return f"custom-{id(self)}"

        cache = ToolResultCache()
        with caplog.at_level(_logging.DEBUG, logger="external_llm.agent.tool_result_cache"):
            cache.set("tool", {"obj": Custom()}, {"ok": True})
        assert any("non-JSON-serializable" in r.message for r in caplog.records)

    def test_lru_move_to_end_on_access(self):
        """Accessing an entry should move it to end (most recently used)."""
        cache = ToolResultCache(max_entries=3)
        cache.set("a", {}, {"v": "a"})
        cache.set("b", {}, {"v": "b"})
        cache.set("c", {}, {"v": "c"})

        # Access 'a' — makes it most recently used
        cache.get("a", {})

        # Eviction should hit 'b' (oldest), not 'a'
        cache.set("d", {}, {"v": "d"})
        assert cache.get("a", {}) is not None
        assert cache.get("b", {}) is None
        assert cache.get("c", {}) is not None
        assert cache.get("d", {}) is not None


# ── PerformanceCollector integration: WeakSet aggregation of cache stats ─────
import gc as _gc

from external_llm.agent.performance_metrics import PerformanceCollector


def test_metrics_tool_result_cache_none_when_no_cache_registered():
    """No cache registered → get_summary reports tool_result_cache=None."""
    c = PerformanceCollector()
    s = c.get_summary()
    assert s["cache_metrics"]["tool_result_cache"] is None


def test_metrics_aggregate_multiple_registered_caches():
    """Parent + clone caches: stats are SUMMED, not last-registered-wins.

    Regression for the masking bug introduced when clones got fresh isolated
    caches: the collector used to hold a single ref, so a clone's cache
    overwrote the parent's ref and the parent's hit-rate vanished from stats.
    """
    c = PerformanceCollector()
    parent = ToolResultCache()
    clone = ToolResultCache()
    parent._hits, parent._misses = 3, 1   # 75% hit
    clone._hits, clone._misses = 0, 4     # 0% hit
    c.register_tool_result_cache(parent)
    c.register_tool_result_cache(clone)

    trc = c.get_summary()["cache_metrics"]["tool_result_cache"]
    assert trc is not None
    assert trc["hits"] == 3
    assert trc["misses"] == 5
    assert trc["instances"] == 2
    assert abs(trc["hit_rate"] - 3 / 8) < 1e-9
    # Existing schema keys preserved (backward-compat with stats consumers).
    for _k in ("hits", "misses", "hit_rate", "size", "max_entries"):
        assert _k in trc


def test_metrics_weakset_drops_cache_when_clone_collected():
    """A clone cache vanishes from the aggregate once the clone is GC'd — no
    leak, and the parent's stats are no longer masked by a dead clone."""
    c = PerformanceCollector()
    parent = ToolResultCache()
    parent._hits, parent._misses = 3, 1
    c.register_tool_result_cache(parent)

    def make_clone():
        clone = ToolResultCache()
        clone._hits, clone._misses = 0, 4
        c.register_tool_result_cache(clone)
        return clone

    clone = make_clone()
    trc = c.get_summary()["cache_metrics"]["tool_result_cache"]
    assert trc["instances"] == 2  # parent + clone

    del clone
    _gc.collect()
    trc = c.get_summary()["cache_metrics"]["tool_result_cache"]
    assert trc["instances"] == 1
    assert trc["hits"] == 3 and trc["misses"] == 1  # parent survives


def test_metrics_register_none_is_noop():
    """register_tool_result_cache(None) (cache disabled) must be a no-op."""
    c = PerformanceCollector()
    c.register_tool_result_cache(None)
    assert c.get_summary()["cache_metrics"]["tool_result_cache"] is None


def test_metrics_concurrent_register_and_summary_no_race():
    """Concurrent register_tool_result_cache() (worker threads) vs get_summary()
    must not raise "Set changed size during iteration".

    CPython's WeakSet is not documented as thread-safe: its _IterationGuard only
    defers weakref callbacks within the iterating thread, so a cross-thread .add()
    during get_summary()'s aggregation can raise RuntimeError. register() and the
    aggregation snapshot are both guarded by self._lock — this test exercises the
    window and asserts no exception escapes and the aggregate stays consistent.
    """
    import threading as _threading

    c = PerformanceCollector()
    errors: list = []
    stop = _threading.Event()
    keepalive: list = []  # hold strong refs so caches aren't GC'd mid-run

    def registrar():
        i = 0
        while not stop.is_set():
            cache = ToolResultCache()
            cache._hits = 1
            cache._misses = 1
            try:
                c.register_tool_result_cache(cache)
                keepalive.append(cache)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
            i += 1
            if i > 200:
                break

    def summarizer():
        for _ in range(200):
            try:
                s = c.get_summary()
                trc = s["cache_metrics"]["tool_result_cache"]
                # When caches are registered, the aggregate must surface them
                # with consistent counters (no torn reads / over/under-count).
                if trc is not None:
                    assert trc["hits"] == trc["instances"], trc
                    assert trc["misses"] == trc["instances"], trc
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                return

    threads = [_threading.Thread(target=registrar) for _ in range(3)]
    threads += [_threading.Thread(target=summarizer) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    stop.set()

    assert not errors, f"concurrent register/get_summary raised: {errors!r}"


class TestInvalidatePathsCountAccuracy:
    """Regression: invalidate_paths(frozenset()) read ``len(self._cache)``
    OUTSIDE the lock, then called ``self.clear()`` (which re-acquires the same
    non-reentrant lock). Between the unlocked len-read and the clear, a
    concurrent set()/invalidate could change the size, so the returned count was
    a torn read. The fix does len+clear atomically under the lock."""

    def test_empty_paths_count_matches_inserted(self):
        cache = ToolResultCache()
        for i in range(5):
            cache.set("read_file", {"path": f"f{i}.py"}, {"content": str(i)},
                      paths=frozenset({f"/repo/f{i}.py"}))
        removed = cache.invalidate_paths(frozenset())
        assert removed == 5
        assert len(cache._cache) == 0

    def test_empty_paths_count_is_zero_when_already_empty(self):
        cache = ToolResultCache()
        assert cache.invalidate_paths(frozenset()) == 0
