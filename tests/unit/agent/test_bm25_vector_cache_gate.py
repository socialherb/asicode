"""Tests for ``_VECTOR_CACHE_INDEXED_ARCHIVES`` gate.

The archive-sig gate ensures that the second ``search_design_history`` call
on the *same* (unchanged) archive skips 14k SHA-256 hashes + 14k
``_ensure_model_loaded`` calls by avoiding ``vector_cache.add_document``
for the archived portion.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from external_llm.agent.design_chat_loop import (
    _ARCHIVED_BM25_CACHE,
    _ARCHIVED_BM25_CACHE_LOCK,
    _VECTOR_CACHE_INDEXED_ARCHIVES,
    _VECTOR_CACHE_LOCK,
    _archived_bm25_entries,
)

import external_llm.agent.design_chat_loop as _dcl


class _FakeSessionMgr:
    """Minimal stand-in providing ``archive_path`` (creates a real file for ``stat``)."""

    def __init__(self, archive_path: Path):
        self._archive_path_val = archive_path

    def archive_path(self, sid: str) -> Path:
        return self._archive_path_val


@pytest.fixture(autouse=True)
def _clean_state():
    """Clear caches before each test to avoid cross-test pollution."""
    with _ARCHIVED_BM25_CACHE_LOCK:
        _ARCHIVED_BM25_CACHE.clear()
    with _VECTOR_CACHE_LOCK:
        _VECTOR_CACHE_INDEXED_ARCHIVES.clear()
    _dcl._ARCHIVE_SMALL_SKIP = 0  # always cache (no small-archive skip)
    yield


def _make_archive(lines: list[str]) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    )
    for line in lines:
        f.write(line + "\n")
    f.close()
    return Path(f.name)


class TestVectorCacheGate:
    """Pin ``_VECTOR_CACHE_INDEXED_ARCHIVES`` gating behaviour."""

    def test_sig_is_deterministic(self):
        """Archive signature is a deterministic 3-tuple ``(sid, size, mtime_ns)``."""
        p = _make_archive(['{"content": "hello world"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world"}]
            entries, sig, from_cache = _archived_bm25_entries(mgr, "s1", archived)
            assert sig is not None
            assert not from_cache
            assert isinstance(sig, tuple)
            assert len(sig) == 3
            assert sig[0] == "s1"           # sid
            assert isinstance(sig[1], int)  # size
            assert isinstance(sig[2], int)  # mtime_ns
            # Same archive → same sig (deterministic)
            entries2, sig2, fc2 = _archived_bm25_entries(mgr, "s1", archived)
            assert sig == sig2
        finally:
            os.unlink(p)

    def test_second_call_caches_bm25(self):
        """Second call on same sig triggers BM25 cache hit (from_cache=True),
        which means the call site can pass ``archive_sig`` to ``index_docs``
        for vector-cache skip logic."""
        p = _make_archive(['{"content": "hello world test"}'])
        try:
            mgr = _FakeSessionMgr(p)
            archived = [{"content": "hello world test"}]
            r1, sig1, fc1 = _archived_bm25_entries(mgr, "s1", archived)
            r2, sig2, fc2 = _archived_bm25_entries(mgr, "s1", archived)

            assert fc1 is False       # miss (build)
            assert fc2 is True        # hit (reuse)
            assert sig1 == sig2
            assert r1 is r2           # object identity — BM25 cache worked
        finally:
            os.unlink(p)

    def test_different_archive_is_miss(self):
        """Different (size, mtime) → different sig → BM25 cache miss."""
        p1 = _make_archive(['{"content": "first"}'])
        p2 = _make_archive(['{"content": "second longer content"}'])
        try:
            m1, m2 = _FakeSessionMgr(p1), _FakeSessionMgr(p2)
            archived = [{"content": "dummy"}]

            r1, s1, fc1 = _archived_bm25_entries(m1, "s1", archived)
            r2, s2, fc2 = _archived_bm25_entries(m2, "s2", archived)

            assert s1 != s2          # different file sizes → different sigs
            assert fc1 is False
            assert fc2 is False      # different sig → cache miss
        finally:
            for p in (p1, p2):
                os.unlink(p)

    def test_bm25_eviction_clears_vector_set(self):
        """When BM25 cache evicts the oldest entry, the corresponding sig is
        removed from ``_VECTOR_CACHE_INDEXED_ARCHIVES``."""
        # Force small max so we can trigger eviction with 3 sessions
        _dcl._ARCHIVED_BM25_CACHE_MAX = 2

        p1 = _make_archive(['{"content": "session one"}'])
        p2 = _make_archive(['{"content": "session two"}'])
        p3 = _make_archive(['{"content": "session three"}'])
        try:
            m1, m2, m3 = (_FakeSessionMgr(p) for p in (p1, p2, p3))
            archived = [{"content": "dummy"}]

            # Insert s1, s2 → cache has 2 entries
            r1, sig1, _ = _archived_bm25_entries(m1, "s1", archived)
            r2, sig2, _ = _archived_bm25_entries(m2, "s2", archived)

            # Manually record both sigs in VECTOR_CACHE_INDEXED_ARCHIVES (as
            # index_docs would do in production)
            with _VECTOR_CACHE_LOCK:
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig1] = None
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig2] = None

            with _VECTOR_CACHE_LOCK:
                assert sig1 in _VECTOR_CACHE_INDEXED_ARCHIVES
                assert sig2 in _VECTOR_CACHE_INDEXED_ARCHIVES

            # Touch s1 to make it MRU, then insert s3 → evicts s2 (LRU)
            _archived_bm25_entries(m1, "s1", archived)
            r3, sig3, _ = _archived_bm25_entries(m3, "s3", archived)
            # Simulate index_docs recording sig3 (as production does)
            with _VECTOR_CACHE_LOCK:
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig3] = None

            # s2's sig should have been discarded from VECTOR_CACHE_INDEXED_ARCHIVES
            with _VECTOR_CACHE_LOCK:
                assert sig1 in _VECTOR_CACHE_INDEXED_ARCHIVES  # MRU, still alive
                assert sig2 not in _VECTOR_CACHE_INDEXED_ARCHIVES  # evicted
                assert sig3 in _VECTOR_CACHE_INDEXED_ARCHIVES  # just inserted
        finally:
            for p in (p1, p2, p3):
                os.unlink(p)
        # Restore
        _dcl._ARCHIVED_BM25_CACHE_MAX = _dcl._parse_cache_max()

    def test_vector_set_cap_is_fifo_deterministic(self):
        """The safety cap on ``_VECTOR_CACHE_INDEXED_ARCHIVES`` evicts the
        OLDEST-inserted sig (FIFO), not an arbitrary one (the previous
        ``set.pop()`` dropped a random element, possibly a still-active sig,
        forcing a redundant re-index on the next search). Determinism keeps
        still-active sigs alive so repeat searches keep benefiting from the
        skip. This path (cap reached) was NOT covered by
        ``test_bm25_eviction_clears_vector_set``, which exercises the BM25
        LRU-eviction path instead."""
        _orig_max = _dcl._VECTOR_CACHE_INDEXED_ARCHIVES_MAX
        _dcl._VECTOR_CACHE_INDEXED_ARCHIVES_MAX = 2
        try:
            with _VECTOR_CACHE_LOCK:
                _VECTOR_CACHE_INDEXED_ARCHIVES.clear()
                sig_a = ("a", 1, 10)
                sig_b = ("b", 2, 20)
                sig_c = ("c", 3, 30)
                # Insert a, b → at cap (2). Inserting c pushes the OLDEST (a) out.
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig_a] = None
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig_b] = None
                _VECTOR_CACHE_INDEXED_ARCHIVES[sig_c] = None
                while len(_VECTOR_CACHE_INDEXED_ARCHIVES) > _dcl._VECTOR_CACHE_INDEXED_ARCHIVES_MAX:
                    _VECTOR_CACHE_INDEXED_ARCHIVES.popitem(last=False)

                # FIFO: sig_a (oldest-inserted) evicted; sig_b and sig_c survive.
                assert sig_a not in _VECTOR_CACHE_INDEXED_ARCHIVES
                assert sig_b in _VECTOR_CACHE_INDEXED_ARCHIVES
                assert sig_c in _VECTOR_CACHE_INDEXED_ARCHIVES
        finally:
            _dcl._VECTOR_CACHE_INDEXED_ARCHIVES_MAX = _orig_max
