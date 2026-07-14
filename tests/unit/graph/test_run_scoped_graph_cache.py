"""Tests for RunScopedGraphCache."""
from external_llm.graph.run_scoped_graph_cache import (
    RunScopedGraphCache,
    get_global_graph_cache,
    reset_global_graph_cache,
)


class TestRunScopedGraphCache:
    def test_put_and_get(self):
        cache = RunScopedGraphCache()
        cache.put("key1", {"data": "test"}, category="enrichment")
        assert cache.get("key1") == {"data": "test"}

    def test_get_miss(self):
        cache = RunScopedGraphCache()
        assert cache.get("nonexistent") is None

    def test_has(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value")
        assert cache.has("key1") is True
        assert cache.has("key2") is False

    def test_lru_eviction(self):
        cache = RunScopedGraphCache(max_entries=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # should evict "a"
        assert cache.get("a") is None
        assert cache.get("d") == 4

    def test_lru_access_refreshes(self):
        cache = RunScopedGraphCache(max_entries=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # refresh "a"
        cache.put("d", 4)  # should evict "b" (oldest untouched)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_clear(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None
        assert len(cache._cache) == 0

    def test_make_key_deterministic(self):
        key1 = RunScopedGraphCache.make_key("enrichment", symbols=["a", "b"], files=["f1.py"])
        key2 = RunScopedGraphCache.make_key("enrichment", symbols=["a", "b"], files=["f1.py"])
        assert key1 == key2

    def test_make_key_different_inputs(self):
        key1 = RunScopedGraphCache.make_key("enrichment", symbols=["a"])
        key2 = RunScopedGraphCache.make_key("enrichment", symbols=["b"])
        assert key1 != key2

    def test_make_key_order_independent(self):
        key1 = RunScopedGraphCache.make_key("enrichment", symbols=["b", "a"])
        key2 = RunScopedGraphCache.make_key("enrichment", symbols=["a", "b"])
        assert key1 == key2  # sorted internally

    def test_make_key_different_categories(self):
        key1 = RunScopedGraphCache.make_key("enrichment", symbols=["a"])
        key2 = RunScopedGraphCache.make_key("safety_issues", symbols=["a"])
        assert key1 != key2

    def test_put_overwrites_existing(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"

    def test_eviction_count_tracked(self):
        cache = RunScopedGraphCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        stats = cache.get_stats()
        assert stats["evictions"] >= 1


class TestDirtyFileInvalidation:
    def test_invalidate_increments_generation(self):
        cache = RunScopedGraphCache()
        gen_before = cache.generation
        cache.invalidate_for_files(["a.py"])
        assert cache.generation == gen_before + 1

    def test_invalidate_removes_matching_entries(self):
        cache = RunScopedGraphCache()
        cache.put("key1", {"primary_files": ["a.py", "b.py"]})
        cache.put("key2", {"primary_files": ["c.py"]})
        evicted = cache.invalidate_for_files(["a.py"])
        assert evicted == 1
        assert cache.get("key1") is None  # invalidated
        assert cache.get("key2") is not None  # kept

    def test_invalidate_empty_list(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value")
        evicted = cache.invalidate_for_files([])
        assert evicted == 0
        assert cache.get("key1") is not None

    def test_dirty_files_tracked(self):
        cache = RunScopedGraphCache()
        cache.invalidate_for_files(["a.py", "b.py"])
        assert "a.py" in cache.dirty_files
        assert "b.py" in cache.dirty_files

    def test_clear_resets_dirty_files(self):
        cache = RunScopedGraphCache()
        cache.invalidate_for_files(["a.py"])
        cache.clear()
        assert len(cache.dirty_files) == 0

    def test_invalidate_impact_files_field(self):
        cache = RunScopedGraphCache()
        cache.put("key1", {"impact_files": ["x.py"]})
        evicted = cache.invalidate_for_files(["x.py"])
        assert evicted == 1
        assert cache.get("key1") is None

    def test_invalidate_multiple_files(self):
        cache = RunScopedGraphCache()
        cache.put("key1", {"primary_files": ["a.py"]})
        cache.put("key2", {"primary_files": ["b.py"]})
        cache.put("key3", {"primary_files": ["c.py"]})
        evicted = cache.invalidate_for_files(["a.py", "b.py"])
        assert evicted == 2
        assert cache.get("key3") is not None

    def test_generation_increments_on_clear(self):
        cache = RunScopedGraphCache()
        gen_before = cache.generation
        cache.clear()
        assert cache.generation == gen_before + 1


class TestCacheStats:
    def test_stats_initial(self):
        cache = RunScopedGraphCache()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_stats_after_operations(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_debug_summary(self):
        cache = RunScopedGraphCache()
        cache.put("key1", "value1")
        summary = cache.get_debug_summary()
        assert "cache_size" in summary
        assert "hit_rate" in summary
        assert "generation" in summary

    def test_stats_has_all_fields(self):
        cache = RunScopedGraphCache()
        stats = cache.get_stats()
        for field in ("size", "max_entries", "hits", "misses", "hit_rate",
                      "invalidations", "evictions", "generation", "dirty_file_count"):
            assert field in stats

    def test_hit_rate_zero_when_no_accesses(self):
        cache = RunScopedGraphCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0


class TestGlobalSingleton:
    def test_get_global_returns_instance(self):
        reset_global_graph_cache()
        cache = get_global_graph_cache()
        assert isinstance(cache, RunScopedGraphCache)

    def test_get_global_same_instance(self):
        reset_global_graph_cache()
        c1 = get_global_graph_cache()
        c2 = get_global_graph_cache()
        assert c1 is c2

    def test_reset_creates_new_instance(self):
        reset_global_graph_cache()
        c1 = get_global_graph_cache()
        c1.put("key1", "value1")
        reset_global_graph_cache()
        c2 = get_global_graph_cache()
        assert c2.get("key1") is None  # new instance

    def test_reset_clears_old_cache(self):
        reset_global_graph_cache()
        c1 = get_global_graph_cache()
        c1.put("k", "v")
        reset_global_graph_cache()
        c2 = get_global_graph_cache()
        assert c2.get_stats()["size"] == 0
