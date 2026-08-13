"""Vector cache must be flushed from ``atexit``, not from ``__del__``.

Observed at the end of a real CLI session::

    Exception ignored while calling deallocator <function VectorCacheManager.__del__ ...>:
      File "external_llm/agent/vector_cache.py", line 774, in __del__
      File "external_llm/common/atomic_io.py", line 192, in atomic_write_json
      File "external_llm/common/atomic_io.py", line 146, in _atomic_replace
      File "<frozen os>", line 1073, in fdopen
    ImportError: sys.meta_path is None, Python is likely shutting down

When the manager survives until the final GC pass, ``__del__`` runs *inside*
interpreter finalization: ``sys.modules`` has been purged and
``sys.meta_path`` is ``None``, so ``os.fdopen``'s lazy ``import io`` cannot
resolve (measured directly on 3.14: ``'io' in sys.modules`` is already False
there).  Two consequences, both bad: the dirty tail is silently lost, and the
flush dies *between* ``faiss.write_index`` and the metadata write — leaving a
torn pair that ``_load_or_create_index`` rejects on the next start
(index/metadata row-count mismatch) for a full re-embed.

``atexit`` hooks run before any of that teardown (measured: ``is_finalizing()``
is False and ``sys.meta_path`` is populated inside a hook), so that is where
the flush belongs.  ``__del__`` keeps only the live-interpreter case.

These tests stub faiss so they run everywhere.
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path

import external_llm.agent.vector_cache as vc
from external_llm.agent.vector_cache import VectorCacheManager

_REPO_ROOT = Path(__file__).resolve().parents[3]


class _StubFaiss:
    """Stands in for the faiss module: records + serializes the index."""

    def __init__(self) -> None:
        self.writes = 0

    def serialize_index(self, index) -> bytes:
        self.writes += 1
        return b"stub-index"


def _dirty_manager(monkeypatch, tmp_path) -> tuple[VectorCacheManager, _StubFaiss]:
    """A manager with un-persisted state, without needing the faiss stack."""
    stub = _StubFaiss()
    monkeypatch.setattr(vc, "HAS_NUMPY", True)
    monkeypatch.setattr(vc, "HAS_FAISS", True)
    monkeypatch.setattr(vc, "_faiss", stub)
    mgr = VectorCacheManager(str(tmp_path))
    mgr.index = object()  # non-None: stands in for a loaded IndexFlatIP
    mgr.id_to_doc = {0: {"file_path": "src/a.py", "doc_id": "doc0"}}
    mgr._dirty = True
    return mgr, stub


def test_del_does_no_io_during_finalization(monkeypatch, tmp_path):
    """The deallocator must bail out at shutdown instead of raising.

    Nothing it could do there works: the write path needs the import system.
    Attempting it produces an "Exception ignored" traceback AND a half-written
    cache pair, so the only correct behaviour is to leave it to atexit.
    """
    mgr, stub = _dirty_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "is_finalizing", lambda: True)

    mgr.__del__()

    assert stub.writes == 0, "__del__ ran the write path during finalization"
    assert not (tmp_path / "metadata.json").exists()


def test_del_still_flushes_while_the_interpreter_is_alive(monkeypatch, tmp_path):
    """A manager dropped mid-session (not at exit) still persists."""
    mgr, stub = _dirty_manager(monkeypatch, tmp_path)
    assert sys.is_finalizing() is False

    mgr.__del__()

    assert stub.writes == 1
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == {
        "0": {"file_path": "src/a.py", "doc_id": "doc0"}
    }


def test_del_flushes_when_is_finalizing_is_absent(monkeypatch, tmp_path):
    """The mid-session flush must survive a ``sys`` that lacks the probe.

    ``sys.is_finalizing`` has existed since 3.5 — every interpreter this
    module supports (>=3.10) ships it — so a bare ``sys.is_finalizing()``
    never actually raised in production. But a deallocator must be
    bulletproof against exotic ``sys`` stubs (embedded/restricted
    interpreters, packaging tools), and the guard's contract is: "absent
    probe → treat as not finalizing → flush still runs". This regression test
    simulates that environment by deleting the attribute outright rather than
    merely overriding it.
    """
    mgr, stub = _dirty_manager(monkeypatch, tmp_path)
    monkeypatch.delattr(sys, "is_finalizing", raising=False)
    assert not hasattr(sys, "is_finalizing"), "fixture failed to hide the probe"

    mgr.__del__()

    assert stub.writes == 1, "absent is_finalizing() silently skipped the flush"
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == {
        "0": {"file_path": "src/a.py", "doc_id": "doc0"}
    }


def test_exit_flush_persists_dirty_managers(monkeypatch, tmp_path):
    mgr, stub = _dirty_manager(monkeypatch, tmp_path)
    assert mgr in vc._live_managers, "manager not registered for the exit flush"

    vc._flush_live_caches()

    assert stub.writes == 1
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "embedding_model.txt").exists(), (
        "model marker not refreshed — next start discards the index as stale"
    )
    assert mgr._dirty is False


def test_exit_flush_skips_clean_managers(monkeypatch, tmp_path):
    """The dirty-flag optimisation must survive the move to atexit.

    A clean manager means the last periodic save already wrote the same bytes;
    re-dumping is O(n) I/O (5 MB index + metadata) for nothing.
    """
    mgr, stub = _dirty_manager(monkeypatch, tmp_path)
    mgr._dirty = False

    vc._flush_live_caches()

    assert stub.writes == 0
    assert not (tmp_path / "metadata.json").exists()


def test_exit_flush_does_not_resurrect_a_deleted_cache_dir(monkeypatch, tmp_path, caplog):
    """A cache dir removed mid-session stays removed, quietly.

    Without the guard, every dead manager logs a faiss "could not open ... for
    writing" warning (visible at the end of any pytest run, whose fixtures
    rmtree the cache dir while the manager is still alive), and the metadata
    write would ``os.makedirs`` the directory back.
    """
    cache_dir = tmp_path / "gone"
    mgr, stub = _dirty_manager(monkeypatch, cache_dir)
    for child in cache_dir.iterdir():
        child.unlink()
    cache_dir.rmdir()

    with caplog.at_level("WARNING"):
        vc._flush_live_caches()
        mgr.__del__()  # the deallocator path must be just as quiet

    assert stub.writes == 0
    assert not cache_dir.exists(), "exit flush recreated a deleted cache dir"
    assert not caplog.records, f"noisy exit flush: {[r.message for r in caplog.records]}"


def test_registry_does_not_keep_managers_alive(monkeypatch, tmp_path):
    """The registry must be weak: metadata dicts reach ~100 MB in real caches."""
    mgr, _stub = _dirty_manager(monkeypatch, tmp_path)
    ref = weakref.ref(mgr)

    del mgr
    gc.collect()

    assert ref() is None, "exit-flush registry pinned the manager for the process lifetime"
    assert len(list(vc._live_managers)) == 0


def test_process_exit_flushes_without_deallocator_noise(tmp_path):
    """End-to-end: a real interpreter exit persists the cache and stays quiet.

    ``sys.is_finalizing`` is forced True in the child so the deallocator can
    never be the writer — exactly its situation at shutdown.  If the file
    appears anyway, the atexit hook wrote it.
    """
    cache_dir = tmp_path / "vector_cache"
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from pathlib import Path
        import external_llm.agent.vector_cache as vc

        class _StubFaiss:
            def serialize_index(self, index):
                return b"stub-index"

        vc.HAS_NUMPY = True
        vc.HAS_FAISS = True
        vc._faiss = _StubFaiss()

        mgr = vc.VectorCacheManager({str(cache_dir)!r})
        mgr.index = object()
        mgr.id_to_doc = {{0: {{"file_path": "src/a.py", "doc_id": "doc0"}}}}
        mgr._dirty = True

        # Simulate the shutdown stage the deallocator actually runs in.
        sys.is_finalizing = lambda: True
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr
    assert "meta_path is None" not in proc.stderr, proc.stderr
    meta = cache_dir / "metadata.json"
    assert meta.exists(), "dirty cache was lost at process exit"
    assert json.loads(meta.read_text(encoding="utf-8")) == {
        "0": {"file_path": "src/a.py", "doc_id": "doc0"}
    }
