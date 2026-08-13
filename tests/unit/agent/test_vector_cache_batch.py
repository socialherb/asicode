"""Unit tests for VectorCacheManager.add_documents batched-encode path.

Mocks the embedding model so these run without sentence-transformers (only
numpy + FAISS required). Locks the contract that batching is equivalent to
per-file add_document, dedups by doc_id, and handles empty/mixed input.
"""
import shutil
import tempfile

import pytest

try:
    import numpy as np

    from external_llm.agent.vector_cache import HAS_FAISS, HAS_NUMPY, VectorCacheManager
    _OK = HAS_NUMPY and HAS_FAISS
except ImportError:
    _OK = False


def _make_det_model(dim):
    """Fake embedding model with a content-keyed deterministic encode()."""
    import hashlib

    class _Det:
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            single = isinstance(texts, str)
            if single:
                texts = [texts]
            out = np.zeros((len(texts), dim), dtype=np.float32)
            for i, t in enumerate(texts):
                seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
                out[i] = np.random.RandomState(seed).randn(dim).astype(np.float32)
            return out[0] if single else out

    return _Det()


@pytest.mark.skipif(not _OK, reason="numpy or faiss not available")
class TestAddDocumentsBatch:
    @pytest.fixture
    def manager(self, tmp_path):
        m = VectorCacheManager(str(tmp_path))
        m.embedding_model = _make_det_model(m.dimension)
        yield m
        shutil.rmtree(str(tmp_path), ignore_errors=True)

    def test_batch_adds_all(self, manager):
        items = [(f"file{i}.py", f"content {i}") for i in range(5)]
        manager.add_documents(items)
        assert manager.index.ntotal == 5
        assert len(manager.id_to_doc) == 5
        assert len(manager._doc_id_to_idx) == 5
        # metadata recorded in input order
        assert manager.id_to_doc[0]["file_path"] == "file0.py"
        assert manager.id_to_doc[4]["file_path"] == "file4.py"

    def test_empty_is_noop(self, manager):
        # Empty input must short-circuit before loading the index (lazy init).
        assert manager.index is None
        manager.add_documents([])  # must not raise
        assert manager.index is None

    def test_dedup_skips_existing(self, manager):
        manager.add_documents([("a.py", "aaa"), ("b.py", "bbb")])
        assert manager.index.ntotal == 2
        # Re-adding the same (path, content) pairs is a no-op.
        manager.add_documents([("a.py", "aaa"), ("b.py", "bbb")])
        assert manager.index.ntotal == 2
        assert len(manager.id_to_doc) == 2

    def test_mixed_new_and_dup(self, manager):
        manager.add_documents([("a.py", "aaa")])
        # a.py already present; b.py and c.py are new.
        manager.add_documents([("a.py", "aaa"), ("b.py", "bbb"), ("c.py", "ccc")])
        assert manager.index.ntotal == 3
        paths = {manager.id_to_doc[i]["file_path"] for i in range(3)}
        assert paths == {"a.py", "b.py", "c.py"}

    def test_intra_batch_dup_collapses_to_one(self, manager):
        """The same (file_path, content) pair appearing twice in ONE add_documents
        call must collapse to a single indexed row.

        Regression: _doc_id_to_idx is updated only AFTER the dedup loop, so the
        old loop accepted both copies — two identical FAISS rows, the reverse
        lookup overwriting to keep only the last idx (orphan first row), and
        duplicate documents surfacing in search results. The in-batch ``seen``
        set must reject the second copy up front.
        """
        items = [("a.py", "aaa"), ("a.py", "aaa"), ("b.py", "bbb")]
        manager.add_documents(items)
        # Exactly two distinct documents, NOT three.
        assert manager.index.ntotal == 2
        assert len(manager.id_to_doc) == 2
        assert len(manager._doc_id_to_idx) == 2
        # No orphan row: every indexed FAISS row must be reachable via the
        # reverse lookup. Before the fix the first a.py row had idx=0 in
        # id_to_doc but _doc_id_to_idx["a.py"] was overwritten to 1, leaving
        # idx=0 unreachable (orphan) — the two sets would differ.
        reachable_idxs = set(manager._doc_id_to_idx.values())
        assert set(manager.id_to_doc) == reachable_idxs, "orphan row detected"
        paths = {manager.id_to_doc[i]["file_path"] for i in range(2)}
        assert paths == {"a.py", "b.py"}

    def test_batch_equivalent_to_per_file(self, manager, tmp_path):
        """Batch and per-file add produce identical indexed vectors + metadata."""
        items = [(f"f{i}.py", f"content {i}") for i in range(6)]
        manager.add_documents(items)

        single_dir = tempfile.mkdtemp()
        try:
            m_single = VectorCacheManager(single_dir)
            m_single.embedding_model = _make_det_model(m_single.dimension)
            for p, c in items:
                m_single.add_document(p, c)

            assert manager.index.ntotal == m_single.index.ntotal
            # Reconstructed (post-normalize) vectors must match row-for-row.
            vb = manager.index.reconstruct_n(0, manager.index.ntotal)
            vs = m_single.index.reconstruct_n(0, m_single.index.ntotal)
            assert vb.shape == vs.shape
            assert np.allclose(vb, vs, atol=1e-6)
            # metadata file_paths in same order
            pb = [manager.id_to_doc[i]["file_path"] for i in range(vb.shape[0])]
            ps = [m_single.id_to_doc[i]["file_path"] for i in range(vs.shape[0])]
            assert pb == ps
        finally:
            shutil.rmtree(single_dir, ignore_errors=True)

    def test_checkpoint_fires_on_boundary_cross(self, manager, monkeypatch):
        """A batch crossing a 100-doc boundary must checkpoint even when the
        final ntotal is NOT an exact multiple of 100.

        Regression for the old ``ntotal % 100 == 0`` gate: a cold-start build
        flushes the entire file walk as one batch, so ntotal is just the file
        count — almost never divisible by 100. The gate then skipped the save,
        leaving the whole encode reliant on __del__ (which never runs under
        SIGKILL/OOM/os._exit) and discarding the entire batch on crash.
        """
        real_save = manager._save_index
        saves = []

        def spy():
            saves.append(manager.index.ntotal)
            real_save()

        monkeypatch.setattr(manager, "_save_index", spy)
        # 150 docs from empty → crosses 100; 150 % 100 == 50 != 0 (old gate).
        items = [(f"f{i}.py", f"content {i}") for i in range(150)]
        manager.add_documents(items)
        assert manager.index.ntotal == 150
        assert len(saves) == 1, "expected exactly one checkpoint on boundary cross"
        assert manager._dirty is False, "save must clear the dirty flag"

    def test_no_checkpoint_below_boundary(self, manager, monkeypatch):
        """A sub-boundary batch must NOT save — mirrors per-file's <100 tail,
        which relies on __del__. Both paths share the same loss-on-crash bound
        (<100 docs since the last boundary save).
        """
        saves = []
        monkeypatch.setattr(manager, "_save_index", lambda: saves.append(1))
        items = [(f"f{i}.py", f"content {i}") for i in range(50)]
        manager.add_documents(items)
        assert manager.index.ntotal == 50
        assert saves == [], "no boundary crossed → no save"
        assert manager._dirty is True, "dirty flag carries the unsaved tail"
