"""Unit tests for file_cache.py — 100% branch coverage."""

import os
import time
from pathlib import Path

import pytest

from external_llm.agent.file_cache import (
    FileContentCache,
    get_global_file_cache,
    reset_global_file_cache,
)

# ======================================================================
# FileContentCache
# ======================================================================

class TestFileContentCache:
    """Tests for FileContentCache with temporary files."""

    @pytest.fixture
    def cache(self):
        return FileContentCache(max_size=10)

    @pytest.fixture
    def tmpfile(self, tmp_path: Path) -> Path:
        p = tmp_path / "test.txt"
        p.write_text("hello world\nline2\nline3\n")
        return p

    # ── _make_key ────────────────────────────────────────────────────

    def test_make_key_file_only(self, cache):
        key = cache._make_key("/foo/bar.py", None, None)
        assert key == "/foo/bar.py"

    def test_make_key_with_start(self, cache):
        key = cache._make_key("/foo/bar.py", 5, None)
        assert key == "/foo/bar.py:start:5"

    def test_make_key_with_both(self, cache):
        key = cache._make_key("/foo/bar.py", 5, 10)
        assert key == "/foo/bar.py:start:5:end:10"

    def test_make_key_start_zero(self, cache):
        key = cache._make_key("/foo/bar.py", 0, None)
        assert key == "/foo/bar.py:start:0"

    # ── get / set ────────────────────────────────────────────────────

    def test_set_and_get(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "content", total_lines=3, showing="full")
        result = cache.get(fp)
        assert result == "content"

    def test_get_with_metadata(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "content", total_lines=3, showing="full")
        meta = cache.get_with_metadata(fp)
        assert meta is not None
        content, total_lines, showing = meta
        assert content == "content"
        assert total_lines == 3
        assert showing == "full"

    def test_get_with_start_end(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "partial content", start_line=1, end_line=2, total_lines=3, showing="partial")
        result = cache.get(fp, start_line=1, end_line=2)
        assert result == "partial content"

    def test_get_partial_and_full_are_separate(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "full content", total_lines=3, showing="full")
        cache.set(fp, "partial content", start_line=1, end_line=2, total_lines=3, showing="partial")
        # Full get without range
        assert cache.get(fp) == "full content"
        # Partial get with range
        assert cache.get(fp, start_line=1, end_line=2) == "partial content"

    def test_get_miss(self, cache):
        assert cache.get("/nonexistent.py") is None

    def test_get_with_metadata_miss(self, cache):
        assert cache.get_with_metadata("/nonexistent.py") is None

    def test_get_mtime_changed_invalidates(self, cache, tmpfile: Path):
        """When file mtime changes, cached entry should be invalidated."""
        fp = str(tmpfile)
        cache.set(fp, "old content")
        assert cache.get(fp) == "old content"

        # Modify the file to change mtime
        time.sleep(0.05)
        tmpfile.write_text("modified content")
        # mtime might not change fast enough on some filesystems; force it
        os.utime(fp, None)

        result = cache.get(fp)
        assert result is None, "Cache should miss after mtime change"

    def test_mtime_stored_as_int_nanoseconds(self, cache, tmpfile: Path):
        """mtime cached as integer nanoseconds (st_mtime_ns), not float seconds.

        Sub-millisecond edit→re-read cycles can collapse to the same float
        mtime (float64 resolution is ~238ns at epoch magnitude), serving stale
        content from cache. st_mtime_ns is integer-nanosecond precise. Mirrors
        the dev_tool tail-meta fix (see test_dev_tool_tail_meta.py).
        """
        fp = str(tmpfile)
        cache.set(fp, "content")
        key = cache._make_key(fp, None, None)
        _content, cached_mtime, _tl, _sh = cache.cache[key]
        assert isinstance(cached_mtime, int)
        assert cached_mtime == os.stat(fp).st_mtime_ns

    def test_get_file_deleted_invalidates(self, cache, tmpfile: Path):
        """When file is deleted, cached entry should be invalidated."""
        fp = str(tmpfile)
        cache.set(fp, "content")
        assert cache.get(fp) == "content"

        os.remove(fp)
        assert cache.get(fp) is None

    def test_set_nonexistent_file_does_not_cache(self, cache):
        """set() should silently skip when file doesn't exist."""
        cache.set("/nonexistent/file.py", "content")
        assert cache.get("/nonexistent/file.py") is None

    def test_set_file_deleted_before_get_returns_none(self, cache, tmpfile: Path):
        """When file is deleted, get() should return None via mtime check."""
        fp = str(tmpfile)
        cache.set(fp, "content")
        # Overwrite mtime reference: delete file, get() try to access mtime → OSError → cache miss
        os.remove(fp)
        assert cache.get(fp) is None

    # ── LRU eviction ─────────────────────────────────────────────────

    def test_lru_eviction(self, cache, tmpfile: Path):
        """Once max_size is reached, oldest entry should be evicted."""
        cache.max_size = 3
        files = []
        for i in range(4):
            p = tmpfile.parent / f"file{i}.txt"
            p.write_text(f"content{i}")
            files.append(p)
            cache.set(str(p), f"content{i}")

        # file0 (oldest) should be evicted
        assert cache.get(str(files[0])) is None
        # file1, file2, file3 should still be there
        assert cache.get(str(files[1])) == "content1"
        assert cache.get(str(files[2])) == "content2"
        assert cache.get(str(files[3])) == "content3"

    def test_get_refreshes_lru_order(self, cache, tmpfile: Path):
        """Accessing an entry with get() should move it to end (most recent)."""
        cache.max_size = 2
        f1 = tmpfile.parent / "f1.txt"
        f2 = tmpfile.parent / "f2.txt"
        f3 = tmpfile.parent / "f3.txt"
        f1.write_text("a"); f2.write_text("b"); f3.write_text("c")
        cache.set(str(f1), "a")
        cache.set(str(f2), "b")
        # Access f1 — should move to end
        cache.get(str(f1))
        # Now set f3 — should evict f2 (least recently used), not f1
        cache.set(str(f3), "c")
        assert cache.get(str(f1)) == "a", "f1 should still be cached (was recently accessed)"
        assert cache.get(str(f2)) is None, "f2 should be evicted (LRU)"

    # ── invalidate ────────────────────────────────────────────────────

    def test_invalidate_single_key(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "content")
        assert cache.get(fp) is not None
        cache.invalidate(fp)
        assert cache.get(fp) is None

    def test_invalidate_all_keys_for_file(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "full", total_lines=3, showing="full")
        cache.set(fp, "partial", start_line=1, end_line=2, total_lines=3, showing="partial")
        assert cache.get(fp) is not None
        assert cache.get(fp, start_line=1, end_line=2) is not None

        cache.invalidate(fp)
        assert cache.get(fp) is None
        assert cache.get(fp, start_line=1, end_line=2) is None

    def test_invalidate_only_matching_file(self, cache, tmpfile: Path):
        fp1 = str(tmpfile)
        fp2 = str(tmpfile.parent / "other.txt")
        tmpfile.parent.joinpath("other.txt").write_text("other")
        cache.set(fp1, "content1")
        cache.set(fp2, "content2")
        cache.invalidate(fp1)
        assert cache.get(fp1) is None
        assert cache.get(fp2) == "content2"

    def test_invalidate_nonexistent_file(self, cache):
        """invalidate() on a file not in cache should be a no-op."""
        cache.invalidate("/nonexistent.py")  # should not raise

    # ── clear ─────────────────────────────────────────────────────────

    def test_clear(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "content")
        cache.set(fp, "p2", start_line=2, total_lines=3)
        cache.get(fp)  # hit
        assert cache.hits == 1
        cache.clear()
        assert cache.get(fp) is None  # miss after clear
        assert cache.hits == 0  # reset by clear
        # misses were also reset by clear; the get() above is a new miss
        assert cache.misses == 1

    # ── get_stats ─────────────────────────────────────────────────────

    def test_get_stats_empty(self, cache):
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_get_stats_after_ops(self, cache, tmpfile: Path):
        fp = str(tmpfile)
        cache.set(fp, "content")
        cache.get(fp)  # hit
        cache.get("/nonexistent.py")  # miss
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_get_stats_hit_rate_zero_no_ops(self, cache):
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

    # ── max_size boundary ─────────────────────────────────────────────

    def test_set_at_capacity(self, cache, tmpfile: Path):
        """Filling the cache exactly to max_size should not evict."""
        cache.max_size = 3
        files = []
        for i in range(3):
            p = tmpfile.parent / f"f{i}.txt"
            p.write_text(f"c{i}")
            files.append(p)
            cache.set(str(p), f"c{i}")
        for f in files:
            assert cache.get(str(f)) is not None, f"{f} should be cached"

    def test_set_overflow_retains_recent(self, cache, tmpfile: Path):
        """Overflow by 1 should evict oldest only."""
        cache.max_size = 2
        f1 = tmpfile.parent / "a.txt"; f1.write_text("a")
        f2 = tmpfile.parent / "b.txt"; f2.write_text("b")
        f3 = tmpfile.parent / "c.txt"; f3.write_text("c")
        cache.set(str(f1), "a")
        cache.set(str(f2), "b")
        cache.set(str(f3), "c")  # evicts f1
        assert cache.get(str(f1)) is None
        assert cache.get(str(f2)) == "b"


# ======================================================================
# Global cache functions
# ======================================================================

class TestGlobalCache:
    def test_get_global_file_cache_returns_singleton(self):
        c1 = get_global_file_cache()
        c2 = get_global_file_cache()
        assert c1 is c2

    def test_reset_global_file_cache_creates_new(self):
        c1 = get_global_file_cache()
        reset_global_file_cache(max_size=500)
        c2 = get_global_file_cache()
        assert c2 is not c1
        assert c2.max_size == 500

    def test_reset_defaults_to_1000(self):
        reset_global_file_cache()
        c = get_global_file_cache()
        assert c.max_size == 1000
