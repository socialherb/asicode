"""F2/F3 regressions around ``_save_index`` failure handling.

F2 — a FAILED checkpoint save must leave the dirty flag set. ``_save_index``
swallows its own failures and clears ``_dirty`` only on success; the checkpoint
branches used to set the flag only in the NON-checkpoint path, so a failed
checkpoint save left ``_dirty`` False and the exit flush skipped the whole
un-persisted tail (a cold-start batch of hundreds of documents, lost), and
``clear()`` set no flag at all (old disk rows resurrected next session).

F3 — durability + robustness of the staged save: the staged index must be
fsync'd BEFORE the rename (rename atomicity != data durability; the metadata
leg already fsyncs), and a vanished cache dir must degrade to a logged skip
instead of escaping ``clear()``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

import numpy as np

import external_llm.agent.vector_cache as vc
from external_llm.agent.vector_cache import VectorCacheManager


class _StubST:
    """384-dim deterministic model; the configured name loads fine."""

    def __init__(self, name: str):
        pass

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        return np.ones((n, 384), dtype="float32")


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    vc.reset_global_embedding_model()
    monkeypatch.setattr(vc, "SentenceTransformer", _StubST)
    monkeypatch.setattr(vc, "HAS_SENTENCE_TRANSFORMERS", True)
    yield
    vc.reset_global_embedding_model()


def _fail(*_a, **_k):
    raise OSError("simulated disk failure")


def test_failed_checkpoint_save_keeps_tail_dirty_for_exit_flush(tmp_path, monkeypatch):
    """F2: the checkpoint save at doc 100 fails -> the 100th doc must NOT be
    lost: _dirty stays set, the exit flush retries, the retry persists it."""
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    # Seed 99 rows directly (bypasses encode cost) so the next add crosses the
    # 100-doc checkpoint boundary.
    mgr.index.add(np.ones((99, 384), dtype="float32"))
    mgr.id_to_doc = {i: {"file_path": f"f{i}", "doc_id": f"d{i}"} for i in range(99)}
    mgr._doc_id_to_idx = {f"d{i}": i for i in range(99)}
    mgr._dirty = False

    real_awj = vc.atomic_write_json
    monkeypatch.setattr(vc, "atomic_write_json", _fail)

    mgr.add_document("f99.py", "the 100th document")  # idx 99 -> checkpoint -> fails

    assert mgr.index.ntotal == 100, "in-memory add must not roll back"
    assert mgr._dirty is True, "failed checkpoint must leave the tail dirty (was: silently lost)"

    monkeypatch.setattr(vc, "atomic_write_json", real_awj)
    mgr._flush_if_dirty()  # the exit flush retries the failed tail
    assert mgr._dirty is False

    fresh = VectorCacheManager(str(tmp_path))
    fresh._ensure_index_loaded()
    assert fresh.index.ntotal == 100, "exit flush did not persist the retried tail"
    fresh._dirty = False


def test_failed_clear_save_keeps_dirty_for_exit_flush(tmp_path, monkeypatch):
    """F2: clear() with a failing save must stay dirty so the old disk rows
    cannot resurrect on the next session."""
    mgr = VectorCacheManager(str(tmp_path))
    mgr.add_document("a.py", "hello")
    mgr._flush_if_dirty()

    real_wi = vc._faiss.write_index
    monkeypatch.setattr(vc._faiss, "write_index", _fail)

    mgr.clear()
    assert mgr.index.ntotal == 0, "in-memory state must be cleared"
    assert mgr._dirty is True, "failed clear must keep the flag (was: disk rows resurrect)"

    monkeypatch.setattr(vc._faiss, "write_index", real_wi)
    mgr._flush_if_dirty()
    assert mgr._dirty is False

    fresh = VectorCacheManager(str(tmp_path))
    fresh._ensure_index_loaded()
    assert fresh.index.ntotal == 0, "cleared cache resurrected old disk rows"
    fresh._dirty = False


def test_save_fsyncs_staged_index_before_rename(tmp_path, monkeypatch):
    """F3: rename atomicity does not imply data durability — the staged index
    must hit disk (fsync) BEFORE os.replace commits it under the final name."""
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    mgr.index.add(np.ones((1, 384), dtype="float32"))
    mgr.id_to_doc[0] = {"file_path": "f0", "doc_id": "d0"}
    mgr._dirty = True

    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def _replace(src, dst):
        events.append(f"replace:{Path(dst).name}")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(os, "replace", _replace)

    mgr._save_index()

    assert "fsync" in events, "staged index never reached disk"
    assert events.index("fsync") < events.index("replace:faiss_index.bin"), (
        "index renamed before its data was durable — a power loss commits empty data"
    )
    mgr._dirty = False


def test_clear_survives_deleted_cache_dir(tmp_path, monkeypatch):
    """F3: a vanished cache dir degrades to a logged skip. Previously the
    mkstemp FileNotFoundError escaped _save_index — clear() was the only
    caller with no safety net."""
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    for p in tmp_path.iterdir():
        p.unlink()
    tmp_path.rmdir()

    mgr.clear()  # must not raise

    assert mgr._dirty is True, "failed save leaves the flag set; exit flush retries quietly"


def test_save_does_not_block_search_and_preserves_dirty_on_concurrent_add(tmp_path, monkeypatch):
    """F5: disk I/O runs OUTSIDE the lock (a concurrent search must not
    block), and a mutation landing during the I/O keeps the dirty flag set so
    the next flush persists it (generation counter)."""
    mgr = VectorCacheManager(str(tmp_path))
    mgr._ensure_index_loaded()
    mgr.add_document("a.py", "hello")
    mgr._flush_if_dirty()
    mgr._dirty = False

    entered = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def _slow_replace(src, dst):
        if Path(dst).name == "faiss_index.bin":
            entered.set()
            assert release.wait(5), "test timed out waiting to release the save"
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _slow_replace)

    t = threading.Thread(target=mgr._save_index)
    t.start()
    try:
        assert entered.wait(5), "save never reached the rename step"
        # While the save's disk I/O is blocked, a search must NOT block (the
        # lock is free) and a concurrent add must keep the dirty flag set.
        t0 = time.monotonic()
        results = mgr.search("hello", top_k=1)
        assert time.monotonic() - t0 < 1.0, "search blocked behind the save's disk I/O"
        assert results, "search should return the existing doc"
        mgr.add_document("b.py", "world")
    finally:
        release.set()
    t.join(5)
    assert not t.is_alive(), "save did not finish after release"

    assert mgr._dirty is True, "add during save I/O must keep the flag set (generation changed)"

    mgr._flush_if_dirty()
    assert mgr._dirty is False

    fresh = VectorCacheManager(str(tmp_path))
    fresh._ensure_index_loaded()
    assert fresh.index.ntotal == 2, "the concurrent add must be persisted by the follow-up flush"
    fresh._dirty = False
