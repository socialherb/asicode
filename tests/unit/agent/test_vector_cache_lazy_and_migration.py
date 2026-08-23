"""Vector-cache startup cost: lazy index load + legacy `content` migration.

Two defects lived here, and they compound:

1. ``VectorCacheManager.__init__`` loaded the FAISS index and JSON-parsed the
   metadata synchronously, so every ``ToolRegistry`` construction paid it —
   including the several the CLI makes per session, and every test using the
   ``tool_registry`` fixture. Measured on this repo: 0.33 s and +561 MB RSS per
   registry, 917 MB for two, with no sharing between them.
2. Each metadata entry stored a full copy of the file's text. That was ~117 MB
   of a 124 MB metadata.json — 96% — while the FAISS index it accompanies is
   5.1 MB. Nothing reads it: snippets come from the BM25 ``_doc_texts``.

Dropping ``content`` for NEW entries alone fixes nothing for anyone who already
has a cache: the load path reads the old entries back and ``_save_index`` dumps
``id_to_doc`` verbatim, re-persisting the bulk on every save. The strip must
happen on LOAD, with the dirty flag set, so the file shrinks once and stays
small.
"""

from __future__ import annotations

import json

import pytest

from external_llm.agent.vector_cache import (
    HAS_FAISS,
    HAS_NUMPY,
    VectorCacheManager,
    get_configured_embedding_model_name,
)

pytestmark = pytest.mark.skipif(not (HAS_FAISS and HAS_NUMPY), reason="faiss/numpy not installed")

_DIM = 384


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Never load a real SentenceTransformer in these index-mechanics tests.

    ``_ensure_index_loaded`` now resolves the embedding model FIRST (F1), so
    without this every test would pay a 2-4s model load and the marker-based
    reuse logic would depend on network/offline fallback state.
    """
    import external_llm.agent.vector_cache as vc

    monkeypatch.setattr(vc, "get_global_embedding_model", lambda: None)


def _legacy_cache(tmp_path, n=50, content_size=20_000):
    """Write a cache in the pre-migration format (entries carry 'content')."""
    import faiss
    import numpy as np

    index = faiss.IndexFlatIP(_DIM)
    index.add(np.random.rand(n, _DIM).astype("float32"))
    faiss.write_index(index, str(tmp_path / "faiss_index.bin"))
    meta = {
        str(i): {
            "file_path": f"src/f{i}.py",
            "content": "X" * content_size,
            "doc_id": f"doc{i}",
            "embedding_norm": 1.0,
        }
        for i in range(n)
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "embedding_model.txt").write_text(get_configured_embedding_model_name(), encoding="utf-8")
    return (tmp_path / "metadata.json").stat().st_size


def test_construction_does_not_touch_the_index(tmp_path):
    """The whole point: ToolRegistry construction must not pay the load."""
    _legacy_cache(tmp_path)
    mgr = VectorCacheManager(str(tmp_path))
    assert mgr.index is None, "index loaded eagerly at construction"
    assert mgr.id_to_doc == {}
    assert mgr._doc_id_to_idx == {}


def test_first_use_loads_the_index(tmp_path):
    _legacy_cache(tmp_path, n=50)
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    assert mgr.index is not None
    assert mgr.index.ntotal == 50
    assert len(mgr.id_to_doc) == 50
    assert len(mgr._doc_id_to_idx) == 50, "reverse lookup not rebuilt after lazy load"


def test_ensure_index_loaded_is_idempotent(tmp_path):
    _legacy_cache(tmp_path, n=10)
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    first = mgr.index
    mgr._ensure_index_loaded()
    assert mgr.index is first, "second call re-loaded the index"


def test_legacy_content_is_stripped_on_load_and_marks_dirty(tmp_path):
    _legacy_cache(tmp_path, n=50)
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    assert all("content" not in d for d in mgr.id_to_doc.values()), (
        "legacy 'content' survived the load — every session keeps paying for it"
    )
    assert mgr._dirty is True, (
        "strip without dirty means _save_index never rewrites the file, so the bulk is re-read on every future session"
    )
    # The fields search actually uses must survive.
    sample = next(iter(mgr.id_to_doc.values()))
    assert sample["file_path"].startswith("src/")
    assert "doc_id" in sample and "embedding_norm" in sample


def test_migration_shrinks_the_persisted_metadata(tmp_path):
    before = _legacy_cache(tmp_path, n=50, content_size=20_000)
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    mgr._save_index()
    after = (tmp_path / "metadata.json").stat().st_size
    assert after < before / 10, f"expected >10x shrink, got {before} -> {after}"
    assert mgr._dirty is False, "_save_index did not clear the dirty flag"


def test_migration_is_idempotent_and_survives_a_reload(tmp_path):
    _legacy_cache(tmp_path, n=50)
    first = VectorCacheManager(str(tmp_path))
    first._ensure_index_loaded()
    first._save_index()
    size_after_first = (tmp_path / "metadata.json").stat().st_size

    second = VectorCacheManager(str(tmp_path))
    second._ensure_index_loaded()
    assert len(second.id_to_doc) == 50, "migration lost entries"
    assert second.index.ntotal == 50, "migration desynced index and metadata"
    assert second._dirty is False, "nothing left to migrate, yet marked dirty"
    second._save_index()
    assert (tmp_path / "metadata.json").stat().st_size == size_after_first


def test_search_result_shape_survives_migration(tmp_path):
    """`content` is intentionally empty in results; RAGSearcher takes snippets
    from the BM25 _doc_texts and only falls back to this field on desync."""
    _legacy_cache(tmp_path, n=10)
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    for doc in mgr.id_to_doc.values():
        assert set(doc) == {"file_path", "doc_id", "embedding_norm"}


def test_empty_cache_directory_is_created_lazily_but_usable(tmp_path):
    target = tmp_path / "nested" / "vector_cache"
    mgr = VectorCacheManager(str(target))
    assert target.is_dir(), "cache dir should still be created at construction"
    assert mgr.index is None
    mgr._ensure_index_loaded()
    assert mgr.index is not None and mgr.index.ntotal == 0
    assert mgr._dirty is False, "a fresh empty index has nothing to persist"


def test_cache_dir_is_derived_from_repo_root_not_cwd(tmp_path, monkeypatch):
    """A relative ``.asicode/vector_cache`` resolved against the process CWD, so
    running the agent from anywhere but the repo root served repo A's embeddings
    for repo B's searches — and littered `.asicode/` into arbitrary directories.
    """
    from external_llm.agent.rag_searcher import RAGSearcher

    repo = tmp_path / "repo"
    elsewhere = tmp_path / "elsewhere"
    repo.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    searcher = RAGSearcher(str(repo), vector_cache_enabled=True)
    if searcher.vector_cache_manager is None:
        pytest.skip("vector cache disabled (missing sentence-transformers)")
    assert searcher.vector_cache_manager.cache_dir.resolve() == (repo / ".asicode" / "vector_cache").resolve()
    assert not (elsewhere / ".asicode").exists(), "cache dir littered into CWD"
