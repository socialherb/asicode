"""F1 regressions: the embedding model must resolve BEFORE the index loads.

``_load_or_create_index`` compares the persisted model marker against
``self.model_name`` and creates ``IndexFlatIP(self.dimension)``; both values
are only correct AFTER ``_ensure_model_loaded`` corrected them (an offline
fallback changes the name, a non-384 model changes the dimension). Loading the
index first used to cause three distinct bugs:

* F1-a: offline fallback -> a valid fallback-built cache was discarded every
  session (marker compared against the configured name) -> full re-embed loop.
* F1-b: a marker-matched multilingual cache reused under a fallback session ->
  two semantic spaces mixed into one index, silently.
* F1-c: a non-384 model -> index built at 384 -> every add/search broken.

These tests drive the REAL loader path (configured->fallback candidate loop)
with a stub SentenceTransformer, so they run offline.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import numpy as np

import external_llm.agent.vector_cache as vc
from external_llm.agent.vector_cache import VectorCacheManager

_FAILING = "model-that-fails-to-load"


class _StubST:
    """SentenceTransformer stand-in: the configured/multilingual names fail
    (simulated offline); ``all-MiniLM-L6-v2`` (the real fallback) loads."""

    dim = 384

    def __init__(self, name: str):
        if name in (_FAILING, vc.DEFAULT_EMBEDDING_MODEL):
            raise OSError("simulated offline")

    def get_embedding_dimension(self) -> int:
        return type(self).dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        return np.ones((n, type(self).dim), dtype="float32")


@pytest.fixture(autouse=True)
def _isolate_global_model(monkeypatch):
    vc.reset_global_embedding_model()
    monkeypatch.setattr(vc, "SentenceTransformer", _StubST)
    monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setenv("ASICODE_EMBEDDING_MODEL", _FAILING)
    yield
    vc.reset_global_embedding_model()


def _seed_marker(cache_dir, model_name, n_docs=3):
    """Write a cache as if built by *model_name* (384-dim vectors)."""
    import faiss

    index = faiss.IndexFlatIP(384)
    index.add(np.ones((n_docs, 384), dtype="float32"))
    faiss.write_index(index, str(cache_dir / "faiss_index.bin"))
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {str(i): {"file_path": f"f{i}", "doc_id": f"d{i}"} for i in range(n_docs)}
        ),
        encoding="utf-8",
    )
    (cache_dir / "embedding_model.txt").write_text(model_name, encoding="utf-8")


def test_offline_fallback_cache_reused_next_session(tmp_path):
    """F1-a: the marker must be compared against the LOADED model name. The
    previous order compared it against the configured name first (mismatch ->
    rebuild), then loaded the fallback — repeating the full re-embed every
    session."""
    # Session 1: configured model unavailable -> fallback builds the cache.
    first = VectorCacheManager(str(tmp_path))
    first.add_document("src/a.py", "hello world")
    first._flush_if_dirty()
    assert (tmp_path / "embedding_model.txt").read_text(encoding="utf-8") == (
        "all-MiniLM-L6-v2"
    )

    # Session 2: fresh process — no model loaded yet, configured name is the
    # failing one, marker on disk is the fallback's.
    vc.reset_global_embedding_model()
    second = VectorCacheManager(str(tmp_path))
    assert second.model_name == _FAILING, "precondition: unloaded model -> configured name"
    second._ensure_index_loaded()
    assert second.index.ntotal == 1, "fallback-built cache must be reused, not re-embedded"
    second._dirty = False


def test_fallback_session_rebuilds_multilingual_marked_cache(tmp_path, monkeypatch):
    """F1-b: the marker matches the CONFIGURED name while the session actually
    loads the fallback — reusing the index would mix two semantic spaces."""
    monkeypatch.delenv("ASICODE_EMBEDDING_MODEL", raising=False)  # configured = default
    _seed_marker(tmp_path, vc.DEFAULT_EMBEDDING_MODEL, n_docs=3)
    vc.reset_global_embedding_model()
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()  # model resolves first -> fallback name
    assert mgr.model_name == "all-MiniLM-L6-v2"
    assert mgr.index.ntotal == 0, "fallback session must NOT reuse multilingual vectors"
    mgr.add_document("src/a.py", "hello")
    assert mgr.index.ntotal == 1, "fallback vector joined the multilingual index (contamination)"
    mgr._dirty = False


def test_non_384_model_gets_matching_index_dimension(tmp_path, monkeypatch):
    """F1-c: a 768-dim model must produce a 768-dim index, not the default."""
    monkeypatch.setattr(_StubST, "dim", 768)
    monkeypatch.setenv("ASICODE_EMBEDDING_MODEL", "mpnet-768")
    vc.reset_global_embedding_model()
    mgr = VectorCacheManager(str(tmp_path))
    assert mgr.dimension == 384, "precondition: ctor default until the model resolves"
    mgr._ensure_index_loaded()
    assert mgr.index.d == 768, "index built at the default 384 -> every add fails"
    mgr.add_document("src/a.py", "hello")
    assert mgr.index.ntotal == 1, "768-dim add must succeed"
    mgr._dirty = False
