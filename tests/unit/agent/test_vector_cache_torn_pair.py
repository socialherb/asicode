"""Vector cache torn-pair handling: staged save (F1) + tail recovery (F2).

The index (``faiss_index.bin``) and metadata (``metadata.json``) are persisted
as two separate files, so a crash mid-save can leave them inconsistent:

* ``_save_index`` now stages the index in a sibling ``.atomic_*`` temp file and
  commits it with one ``os.replace`` BEFORE the metadata write — the
  multi-MB write window is replaced by a single rename, so a crash cannot
  leave a truncated index; at worst it leaves index > metadata.
* ``_load_or_create_index`` treats index > metadata as a tail overflow
  (metadata keys are the gapless row sequence 0..n-1) and drops only the
  orphan tail instead of discarding the whole corpus for a full re-embed.

These tests use the REAL faiss/numpy stack (skipped when unavailable).
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import faiss
import numpy as np
import pytest

import external_llm.agent.vector_cache as vc
from external_llm.agent.vector_cache import VectorCacheManager


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """Never load a real SentenceTransformer in these index-mechanics tests.

    ``_ensure_index_loaded`` now resolves the embedding model FIRST (F1), so
    without this every test would pay a 2-4s model load and the marker-based
    reuse logic would depend on network/offline fallback state.
    """
    monkeypatch.setattr(vc, "get_global_embedding_model", lambda: None)


def _seed_pair(cache_dir, n_index_rows: int, metadata_keys) -> None:
    """Write a consistent-or-torn (index, metadata) pair as faiss would."""
    index = faiss.IndexFlatIP(384)
    index.add(np.ones((n_index_rows, 384), dtype="float32"))
    faiss.write_index(index, str(cache_dir / "faiss_index.bin"))
    (cache_dir / "metadata.json").write_text(
        json.dumps({str(k): {"file_path": f"f{k}", "doc_id": f"d{k}"} for k in metadata_keys}),
        encoding="utf-8",
    )


def _manager(cache_dir) -> VectorCacheManager:
    mgr = VectorCacheManager(str(cache_dir))
    (cache_dir / "embedding_model.txt").write_text(mgr.model_name, encoding="utf-8")
    return mgr


# ── F2: load-time tail recovery ───────────────────────────────────────────────


def test_load_recovers_orphan_tail_rows(tmp_path):
    """index longer than metadata = a save that died between the index commit
    and the metadata write.  The orphan TAIL is dropped, everything else is
    kept — no re-embed."""
    mgr = _manager(tmp_path)
    _seed_pair(tmp_path, n_index_rows=3, metadata_keys=[0, 1])  # 1 orphan row

    loaded, id_to_doc = mgr._load_or_create_index()

    assert loaded.ntotal == 2, "orphan tail row must be dropped, prefix kept"
    assert set(id_to_doc) == {0, 1}
    assert mgr._dirty is True, "trimmed state must be persisted on next save"
    # cleanup: keep the exit-flush registry quiet (manager stays registered)
    mgr._dirty = False


def test_load_rebuilds_on_gapped_metadata(tmp_path):
    """A metadata KEY GAP (not a tail overflow) cannot be safely truncated —
    the pair is discarded and rebuilt from scratch."""
    mgr = _manager(tmp_path)
    _seed_pair(tmp_path, n_index_rows=3, metadata_keys=[0, 2])  # key 1 missing

    loaded, id_to_doc = mgr._load_or_create_index()

    assert loaded.ntotal == 0, "gapped pair must be rebuilt, not half-loaded"
    assert id_to_doc == {}


def test_load_rebuilds_when_metadata_empty_but_index_has_rows(tmp_path):
    """index > metadata with NO metadata entries is not a tail overflow (the
    whole index is orphaned) — rebuild rather than pretend recovery."""
    mgr = _manager(tmp_path)
    _seed_pair(tmp_path, n_index_rows=3, metadata_keys=[])

    loaded, id_to_doc = mgr._load_or_create_index()

    assert loaded.ntotal == 0
    assert id_to_doc == {}


# ── F1: staged save ───────────────────────────────────────────────────────────


def test_save_index_stages_then_replaces(tmp_path):
    """The index is written to a sibling temp file and committed with a single
    rename; the final on-disk state is a consistent pair, no temp leftovers."""
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()
    mgr.index.add(np.ones((1, 384), dtype="float32"))
    mgr.id_to_doc[0] = {"file_path": "f0", "doc_id": "d0"}

    mgr._save_index()

    assert faiss.read_index(str(tmp_path / "faiss_index.bin")).ntotal == 1
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == {
        "0": {"file_path": "f0", "doc_id": "d0"}
    }
    assert (tmp_path / "embedding_model.txt").read_text(encoding="utf-8") == mgr.model_name
    assert mgr._dirty is False
    assert not list(tmp_path.glob(".atomic_*")), "temp file must be consumed by the rename"


def test_save_index_failure_leaves_previous_pair_intact(tmp_path, monkeypatch):
    """If the index serialization fails mid-save, the PREVIOUS pair on disk
    must be untouched (the live file is never truncated) and no temp file may
    leak."""
    mgr = _manager(tmp_path)
    mgr._ensure_index_loaded()
    (tmp_path / "faiss_index.bin").write_bytes(b"OLD-INDEX")
    (tmp_path / "metadata.json").write_text('{"old": true}', encoding="utf-8")
    mgr.index.add(np.ones((1, 384), dtype="float32"))
    mgr.id_to_doc[0] = {"file_path": "f0", "doc_id": "d0"}
    mgr._dirty = True

    def _boom(_index):
        raise OSError("disk full")

    class _StubFaiss:
        serialize_index = staticmethod(_boom)

    monkeypatch.setattr(vc, "_faiss", _StubFaiss())

    mgr._save_index()

    assert (tmp_path / "faiss_index.bin").read_bytes() == b"OLD-INDEX"
    assert (tmp_path / "metadata.json").read_text(encoding="utf-8") == '{"old": true}'
    assert mgr._dirty is True, "failed save must not clear the dirty flag"
    assert not list(tmp_path.glob(".atomic_*")), "temp file must be cleaned up on failure"
    mgr._dirty = False  # keep the exit-flush registry quiet


# ── exit-flush gate contract: one manager's failure must not abort the rest ──


class _StubFaiss:
    def __init__(self) -> None:
        self.writes = 0

    def serialize_index(self, index) -> bytes:
        self.writes += 1
        return b"stub-index"


def test_exit_flush_continues_after_one_manager_failure(monkeypatch, tmp_path, caplog):
    stub = _StubFaiss()
    monkeypatch.setattr(vc, "HAS_NUMPY", True)
    monkeypatch.setattr(vc, "HAS_FAISS", True)
    monkeypatch.setattr(vc, "_faiss", stub)

    ok = VectorCacheManager(str(tmp_path / "ok"))
    ok.index = object()
    ok.id_to_doc = {0: {"file_path": "src/a.py", "doc_id": "doc0"}}
    ok._dirty = True

    bad = VectorCacheManager(str(tmp_path / "bad"))
    bad.index = object()
    bad.id_to_doc = {0: {"file_path": "src/b.py", "doc_id": "doc1"}}
    bad._dirty = True

    def _boom():
        raise RuntimeError("boom")

    bad._save_index = _boom

    with caplog.at_level("WARNING"):
        vc._flush_live_caches()

    assert (tmp_path / "ok" / "metadata.json").exists(), "healthy manager must still flush"
    assert not (tmp_path / "bad" / "metadata.json").exists()
    assert any("exit flush failed" in r.message for r in caplog.records)
    bad._dirty = False  # keep the registry quiet for later flushes
