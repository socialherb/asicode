"""
Unit tests for embedding-model cache invalidation in VectorCacheManager.

When the embedding model changes, persisted vectors live in a different
semantic space and must not be reused. These tests verify the on-disk model
marker drives reuse-vs-rebuild correctly. The embedding model itself is stubbed
out (None) so the tests exercise only the index/marker logic without loading a
real SentenceTransformer.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import faiss
import numpy as np

from external_llm.agent.vector_cache import VectorCacheManager


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Never load a real SentenceTransformer in these marker-logic tests.

    The model resolves lazily inside ``_ensure_index_loaded`` →
    ``_ensure_model_loaded`` — i.e. in the test body, AFTER construction — so a
    ``with patch(...)`` scoped to ``VectorCacheManager(...)`` creation never
    intercepted the load (real model: ~7s + network download on cache-miss
    hosts). Function-scoped monkeypatch keeps the whole test under the stub,
    same contract as test_vector_cache_lazy_and_migration._no_real_model.
    """
    import external_llm.agent.vector_cache as vc

    monkeypatch.setattr(vc, "get_global_embedding_model", lambda: None)


def _seed_cache(cache_dir, model_name, dim=384, n_docs=3):
    """Write a faiss index + metadata + model marker as if built by *model_name*."""
    index = faiss.IndexFlatIP(dim)
    vecs = np.ones((n_docs, dim), dtype="float32")
    index.add(vecs)
    faiss.write_index(index, str(cache_dir / "faiss_index.bin"))
    import json
    # Metadata is persisted as JSON; keys are stringified on disk and restored
    # to int by VectorCacheManager on load.
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({str(i): {"file_path": f"f{i}", "content": "c"} for i in range(n_docs)}, f)
    (cache_dir / "embedding_model.txt").write_text(model_name, encoding="utf-8")


def _manager(cache_dir):
    # Model resolution is lazy (_ensure_index_loaded) and stubbed for the whole
    # test by the _no_real_model autouse fixture — plain construction is enough.
    return VectorCacheManager(str(cache_dir))


def test_reuses_cache_when_model_matches(tmp_path):
    mgr0 = _manager(tmp_path)
    _seed_cache(tmp_path, mgr0.model_name)
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()
    assert mgr.index.ntotal == 3  # vectors reused


def test_rebuilds_when_model_differs(tmp_path):
    _seed_cache(tmp_path, "some-old-model")
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()
    assert mgr.index.ntotal == 0  # stale vectors discarded
    assert mgr.id_to_doc == {}


def test_rebuilds_when_marker_missing(tmp_path):
    _seed_cache(tmp_path, "x")
    (tmp_path / "embedding_model.txt").unlink()  # legacy pre-marker cache
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()
    assert mgr.index.ntotal == 0


def test_save_writes_marker(tmp_path):
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()  # load empty index first so _save_index has something to write
    mgr._save_index()
    marker = tmp_path / "embedding_model.txt"
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == mgr.model_name
