"""Tests for ``_archived_bm25_entries`` caching layer.

Covers:
  * Cache hit returns the identical entries list (object identity).
  * Cache miss builds and stores fresh entries.
  * LRU eviction, including that evicting a full cache does not break subsequent
    lookups of the surviving session.
  * Thread-safety: the module-level ``_ARCHIVED_BM25_CACHE_LOCK`` serialises
    concurrent access.
  * Small archive skip: archives ≤ 2000 turns are not cached.
  * Environent-variable cap via ``ASICODE_ARCHIVED_BM25_CACHE_MAX``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from external_llm.agent.design_chat_loop import (
    _ARCHIVED_BM25_CACHE,
    _ARCHIVED_BM25_CACHE_LOCK,
    _archived_bm25_entries,
)

import external_llm.agent.design_chat_loop as _dcl


class _FakeSessionMgr:
    """Minimal stand-in for DesignSessionManager that provides ``archive_path``.

    Creates a real file so ``_archive_sig`` can stat it.
    """

    def __init__(self, archive_path: Path):
        self._archive_path_val = archive_path

    def archive_path(self, sid: str) -> Path:
        return self._archive_path_val


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the global BM25 cache before each test to avoid cross-test pollution.
    
    Also disables the small-archive skip so all tests exercise the cache properly
    (``test_small_archive_skips_cache`` re-enables it explicitly).
    """
    with _ARCHIVED_BM25_CACHE_LOCK:
        _ARCHIVED_BM25_CACHE.clear()
    _dcl._ARCHIVE_SMALL_SKIP = 0
    yield


def _make_archive(lines: list[str]) -> Path:
    """Write a temporary archive JSONL and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    )
    for line in lines:
        f.write(line + "\n")
    f.close()
    return Path(f.name)


class TestArchivedBm25Cache:
    """Pin BM25 caching semantics for ``_archived_bm25_entries``."""

    # -- helpers to unpack the new tuple return ---------------------------------

    def _entries(self, *a, **kw) -> list:
        entries, _sig, _from_cache = _archived_bm25_entries(*a, **kw)
        return entries

    def _result(self, *a, **kw) -> tuple:
        return _archived_bm25_entries(*a, **kw)

    # -- existing tests (adapted to tuple return) -------------------------------

    def test_cache_miss_builds_entries(self):
        """First call tokenises and caches entries."""
        p = _make_archive([
            '{"content": "hello world test"}',
            '{"content": "another message here"}',
        ])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world test"}, {"content": "another message here"}]
            entries, sig, from_cache = _archived_bm25_entries(mgr, "s1", archived)
            assert len(entries) == 2
            assert not from_cache
            assert sig is not None
            # Each entry is (idx, tc_dict, token_count, text)
            assert entries[0][0] == 0
            assert entries[0][2] == 3  # "hello", "world", "test"
            assert entries[1][0] == 1
        finally:
            os.unlink(p)

    def test_cache_hit_returns_same_object(self):
        """Repeat call with same sig returns the identical list (object identity)."""
        p = _make_archive(['{"content": "hello world test"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world test"}]
            r1, sig1, fc1 = _archived_bm25_entries(mgr, "s1", archived)
            r2, sig2, fc2 = _archived_bm25_entries(mgr, "s1", archived)
            assert r1 is r2         # object identity — not just equality
            assert sig1 == sig2
            assert fc1 is False     # first call: miss
            assert fc2 is True      # second call: hit
        finally:
            os.unlink(p)

    def test_cache_miss_different_sid_does_not_hit(self):
        """Different session ID is a separate cache entry."""
        p = _make_archive(['{"content": "hello world test"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world test"}]
            r1, _s1, fc1 = _archived_bm25_entries(mgr, "s1", archived)
            r2, _s2, fc2 = _archived_bm25_entries(mgr, "s2", archived)
            assert r1 is not r2     # Different sig → different objects
            assert fc1 is False
            assert fc2 is False     # s2 is a miss
        finally:
            os.unlink(p)

    def test_cache_returns_identical_ranking(self):
        """Cache hit produces the same BM25-relevant data (tc, doc_len) as fresh build."""
        p = _make_archive([
            '{"content": "hello world"}',
            '{"content": "goodbye world"}',
        ])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world"}, {"content": "goodbye world"}]

            # Force miss by clearing cache
            with _ARCHIVED_BM25_CACHE_LOCK:
                _ARCHIVED_BM25_CACHE.clear()

            fresh, _, fc1 = _archived_bm25_entries(mgr, "s1", archived)
            cached, _, fc2 = _archived_bm25_entries(mgr, "s1", archived)

            assert fc1 is False
            assert fc2 is True

            # Term-frequency dicts and token counts must match
            for f_entry, c_entry in zip(fresh, cached, strict=True):
                assert f_entry[1] == c_entry[1]  # tc dict
                assert f_entry[2] == c_entry[2]  # token count
        finally:
            os.unlink(p)

    def test_lru_eviction(self):
        """When cache exceeds MAX (2), oldest entry is evicted."""
        p1 = _make_archive(['{"content": "first session data"}'])
        p2 = _make_archive(['{"content": "second session data"}'])
        p3 = _make_archive(['{"content": "third session data"}'])
        try:
            m1, m2, m3 = (_FakeSessionMgr(p) for p in (p1, p2, p3))
            archived = [{"content": "dummy"}]

            r1, _, _ = _archived_bm25_entries(m1, "s1", archived)
            r2, _, _ = _archived_bm25_entries(m2, "s2", archived)
            # Cache now has max 2 entries (s1, s2)

            # Access s1 to make it MRU
            _archived_bm25_entries(m1, "s1", archived)

            # Insert s3 → evicts s2 (LRU)
            _archived_bm25_entries(m3, "s3", archived)

            # s1 should still be in cache (was MRU)
            assert _archived_bm25_entries(m1, "s1", archived)[0] is r1
            # s2 should have been evicted
            assert _archived_bm25_entries(m2, "s2", archived)[0] is not r2
        finally:
            for p in (p1, p2, p3):
                os.unlink(p)

    def test_lock_held_during_cache_access(self):
        """Cache hits under lock return correct data (thread-safety smoke test)."""
        p = _make_archive(['{"content": "thread safety check"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "thread safety check"}]

            # Populate cache
            r1, _, _ = _archived_bm25_entries(mgr, "s1", archived)

            # Under explicit lock, cache must still be readable
            with _ARCHIVED_BM25_CACHE_LOCK:
                assert _ARCHIVED_BM25_CACHE is not None
                # Cache should have the entry
                assert len(_ARCHIVED_BM25_CACHE) > 0

            # Hit from normal path must still work
            r2, _, fc2 = _archived_bm25_entries(mgr, "s1", archived)
            assert r1 is r2
            assert fc2 is True
        finally:
            os.unlink(p)

    # -- new tests for small-archive skip and from_cache flag -------------------

    def test_small_archive_skips_cache(self):
        """Archives ≤2000 turns bypass BM25 caching entirely."""
        _dcl._ARCHIVE_SMALL_SKIP = 2000  # re-enable production threshold
        p = _make_archive(['{"content": "small archive test"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "small archive test"}]
            # First call — miss (small archive skips cache, so no store)
            entries1, sig1, fc1 = _archived_bm25_entries(mgr, "s_small", archived)
            assert fc1 is False
            assert len(entries1) == 1
            # Second call — still a miss (was never stored)
            entries2, sig2, fc2 = _archived_bm25_entries(mgr, "s_small", archived)
            assert fc2 is False  # not cached because small
            assert entries2 is not entries1  # fresh build each time
        finally:
            os.unlink(p)

    def test_from_cache_flag(self):
        """from_cache=True only when BM25 cache actually hits."""
        p = _make_archive([
            '{"content": "turn one"}',
            '{"content": "turn two"}',
            '{"content": "turn three"}',
        ])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "turn one"}, {"content": "turn two"}, {"content": "turn three"}]
            # First call: miss
            _, sig, from_cache = _archived_bm25_entries(mgr, "s_flag", archived)
            assert from_cache is False
            assert sig is not None
            # Second call: hit
            _, sig2, from_cache2 = _archived_bm25_entries(mgr, "s_flag", archived)
            assert from_cache2 is True
            assert sig == sig2
        finally:
            os.unlink(p)


class TestParseCacheMax:
    """Pin ``_parse_cache_max`` parsing edge cases."""

    # -- valid inputs ----------------------------------------------------------

    def test_default_when_unset(self):
        """``None`` (env unset) → 2."""
        val = _dcl._parse_cache_max(None)
        assert val == 2

    def test_positive_int_string(self):
        """``"5"`` → 5."""
        assert _dcl._parse_cache_max("5") == 5

    def test_one_is_minimum(self):
        """``"0"`` → 1 (``max(1, …)`` floor)."""
        assert _dcl._parse_cache_max("0") == 1

    def test_negative_becomes_inf(self):
        """``"-1"`` → ``inf`` (unlimited)."""
        val = _dcl._parse_cache_max("-1")
        assert val == float("inf")

    def test_inf_token(self):
        """``"inf"`` → ``inf``."""
        assert _dcl._parse_cache_max("inf") == float("inf")

    def test_unlimited_token(self):
        """``"unlimited"`` → ``inf``."""
        assert _dcl._parse_cache_max("unlimited") == float("inf")

    # -- invalid / edge-case inputs (must degrade, not crash) ----------------

    def test_empty_string(self):
        """``""`` → 2 (safe default)."""
        assert _dcl._parse_cache_max("") == 2

    def test_whitespace_only(self):
        """``"   "`` → 2."""
        assert _dcl._parse_cache_max("   ") == 2

    def test_non_numeric(self):
        """``"abc"`` → 2."""
        assert _dcl._parse_cache_max("abc") == 2

    def test_float_string(self):
        """``"2.5"`` → 2 (``int("2.5")`` raises ``ValueError``)."""
        assert _dcl._parse_cache_max("2.5") == 2

    def test_strip_leading_trailing(self):
        """``" 3 "`` → 3."""
        assert _dcl._parse_cache_max(" 3 ") == 3
