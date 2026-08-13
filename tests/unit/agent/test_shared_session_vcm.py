"""Regression tests for the shared session ``VectorCacheManager`` memoisation.

Background: each ``search_design_history()`` call previously constructed a
fresh ``VectorCacheManager`` (~77ms on-disk reload of ``faiss_index.bin`` +
~23MB ``metadata.json``) and relied on ``__del__`` to flush the dirty
(<100-doc) tail back to disk *between* calls.  That flush is unreliable (an
exception traceback can pin the frame local; a GC cycle can delay ``__del__``),
so the next call's reload could miss the tail — and the archive-sig skip gate
(``_VECTOR_CACHE_INDEXED_ARCHIVES``) would then never re-insert it, silently
dropping the most-recent archived turn from vector re-ranking.

The fix memoises one VCM at module level (``_get_session_vcm``), and
``VectorCacheManager`` now serialises its own FAISS index with an instance
lock because ``search_design_history`` dispatches concurrently in the shared
pool (multi-tool batches run read-tools in parallel threads).  These tests
pin both properties WITHOUT requiring the faiss / sentence-transformers
stack (they monkeypatch a fake VCM or a probe index).
"""
from __future__ import annotations

import threading
import time

import pytest

import external_llm.agent.design_chat_loop as _dcl
from external_llm.agent.design_chat_loop import (
    _get_session_vcm,
    _reset_session_vcm_for_test,
    _SessionSearcher,
)


@pytest.fixture(autouse=True)
def _reset_vcm():
    """Drop the memoised VCM before AND after each test for isolation."""
    _reset_session_vcm_for_test()
    yield
    _reset_session_vcm_for_test()


class _FakeVCM:
    """Fake VCM whose ``add_document`` / ``search`` detect concurrent access."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.add_calls = 0
        self.search_calls = 0
        self._guard = threading.Lock()

    def _enter(self) -> None:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)  # widen the race window so an unserialised overlap is visible

    def _leave(self) -> None:
        with self._guard:
            self.active -= 1

    def add_document(self, doc_key: str, text: str) -> None:
        self.add_calls += 1
        self._enter()
        try:
            pass
        finally:
            self._leave()

    def search(self, query: str, top_k: int = 5) -> list:
        self.search_calls += 1
        self._enter()
        try:
            return []
        finally:
            self._leave()


class TestSharedSessionVCM:
    def test_memoised_single_construction(self, monkeypatch):
        """``_get_session_vcm()`` constructs the VCM exactly once; subsequent
        calls return the SAME instance (the old per-call path rebuilt every
        time)."""
        instances: list = []

        def _fake_ctor(path):
            inst = _FakeVCM()
            instances.append(inst)
            return inst

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        monkeypatch.setattr(_dcl, "VectorCacheManager", _fake_ctor)

        a = _get_session_vcm()
        b = _get_session_vcm()
        c = _get_session_vcm()
        assert a is b is c
        assert len(instances) == 1, "VCM must be constructed once, not per call"

    def test_returns_none_when_vector_stack_unavailable(self, monkeypatch):
        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", False)
        assert _get_session_vcm() is None

    def test_construction_failure_returns_none_and_retries(self, monkeypatch):
        """If the first construction raises, the call returns None but the
        singleton slot stays empty so a LATER (successful) construction can
        populate it — graceful self-heal instead of a permanent failure cache."""
        calls: list = []

        def _flaky(path):
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return _FakeVCM()

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        monkeypatch.setattr(_dcl, "VectorCacheManager", _flaky)

        assert _get_session_vcm() is None      # first call: construction raises → None
        vcm = _get_session_vcm()                # second call: succeeds
        assert vcm is not None
        assert _get_session_vcm() is vcm        # memoised thereafter

    def test_io_lock_serialises_concurrent_vcm_access(self, monkeypatch, tmp_path):
        """Concurrent ``index_docs()`` (FAISS mutate via ``add_document``) and
        ``search()`` (FAISS read) on the SHARED VCM are serialised by the
        manager's INTERNAL ``_io_lock``: no two threads are ever inside
        ``index.add`` / ``index.search`` simultaneously (``max_active`` stays
        1). Without the lock, FAISS add/search interleave on one
        ``IndexFlatIP`` and corrupt it.

        Two INDEPENDENT searchers share the VCM — this mirrors production,
        where each ``search_design_history()`` call builds its own searcher
        (independent BM25 arrays) but they all share the memoised VCM. The
        searchers must NOT share their BM25 arrays (those are unlocked), so
        only the VCM is the contended resource."""
        np = pytest.importorskip("numpy")
        import external_llm.agent.vector_cache as _vc

        class _ProbeIndex:
            """Records whether add/search ever overlap on one index."""

            def __init__(self):
                self.ntotal = 0
                self.searches = 0
                self.active = 0
                self.max_active = 0
                self._guard = threading.Lock()

            def _enter(self):
                with self._guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.005)  # widen the race window so overlap is visible

            def _leave(self):
                with self._guard:
                    self.active -= 1

            def add(self, embeddings):
                self._enter()
                try:
                    self.ntotal += int(embeddings.shape[0])
                finally:
                    self._leave()

            def search(self, query, k):
                self._enter()
                try:
                    self.searches += 1
                    return (
                        np.zeros((1, k), dtype="float32"),
                        np.zeros((1, k), dtype=np.int64),
                    )
                finally:
                    self._leave()

        class _FakeModel:
            def encode(self, texts, **kwargs):
                n = len(texts) if isinstance(texts, list) else 1
                return np.ones((n, 4), dtype="float32")

        monkeypatch.setattr(_dcl, "_HAS_VECTOR_CACHE", True)
        monkeypatch.setattr(_vc, "HAS_NUMPY", True)
        monkeypatch.setattr(_vc, "HAS_FAISS", True)

        class _StubFaiss:
            @staticmethod
            def normalize_L2(_a):
                return None

        monkeypatch.setattr(_vc, "_faiss", _StubFaiss())

        mgr = _vc.VectorCacheManager(str(tmp_path / "vcm"))
        probe = _ProbeIndex()
        mgr.index = probe
        mgr.embedding_model = _FakeModel()
        monkeypatch.setattr(_dcl, "VectorCacheManager", lambda path: mgr)

        vcm = _get_session_vcm()
        assert vcm is mgr

        searcher_a = _SessionSearcher(session_prefix="a", vector_cache=vcm)
        searcher_b = _SessionSearcher(session_prefix="b", vector_cache=vcm)
        # Seed both so search() proceeds past the empty-doc guard.
        searcher_a.index_docs([("id0", "seed document text here")])
        searcher_b.index_docs([("id0", "another seed document text")])
        assert probe.ntotal == 2

        errors: list = []

        def _add_loop():
            try:
                for i in range(8):
                    searcher_a.index_docs([("a" + str(i), "added text content " + str(i))])
            except Exception as e:
                errors.append(e)

        def _search_loop():
            try:
                for _ in range(8):
                    searcher_b.search("query", top_k=1)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_add_loop)
        t2 = threading.Thread(target=_search_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"concurrent VCM access raised: {errors}"
        # The manager's INTERNAL lock guarantees add/search never overlap on
        # the shared FAISS index.
        assert probe.max_active == 1, (
            f"FAISS add/search overlapped (max_active={probe.max_active} > 1) — "
            "the shared VCM is missing its serialisation lock"
        )
        assert probe.ntotal > 2
        assert probe.searches > 0
