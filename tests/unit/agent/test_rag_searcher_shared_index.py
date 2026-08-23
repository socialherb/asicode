"""PERF-4 regression tests: process-wide shared RAG index + fingerprint
reconciliation.

A fresh ``RAGSearcher`` on a repo already indexed by an earlier instance of the
same process must NOT re-read + re-tokenize the whole corpus: it reuses the
shared build after a cheap walk+stat fingerprint diff (``_ensure_index``),
re-reading only files whose (mtime_ns, size) changed since the index was built /
invalidated.  Externally-modified files (edits outside the process write
funnel) are picked up by that reconciliation, preserving the freshness a
per-instance fresh build used to provide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import external_llm.agent.rag_searcher as rs_mod
from external_llm.agent.rag_searcher import (
    _SHARED_INDEXES,
    _SHARED_INDEXES_LOCK,
    RAGSearcher,
)


@pytest.fixture(autouse=True)
def _reset_shared_indexes():
    """Each test starts from an empty shared-index registry (unique tmp_path
    would mostly isolate anyway; this makes registry-size assertions exact)."""
    with _SHARED_INDEXES_LOCK:
        _SHARED_INDEXES.clear()
    yield
    with _SHARED_INDEXES_LOCK:
        _SHARED_INDEXES.clear()


def _seed(root: Path, n: int) -> None:
    for k in range(n):
        (root / f"doc{k}.py").write_text(f"# unique_token_doc{k}\ndef function_{k}(x, y):\n    return x + y\n")


def _build(root: Path) -> RAGSearcher:
    s = RAGSearcher(str(root), vector_cache_enabled=False)
    assert s.find_relevant_files("unique_token_doc0", top_k=5), "seed repo must be searchable"
    return s


def _counting_read_text(monkeypatch) -> list[str]:
    """Monkeypatch Path.read_text to record every file read; returns the log."""
    reads: list[str] = []
    orig = Path.read_text

    def _counting(self, *a, **k):
        reads.append(str(self))
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _counting)
    return reads


# ── reuse without rebuild ─────────────────────────────────────────────────────


def test_new_instance_reuses_shared_build_without_reread(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 20)
    _build(tmp_path)

    reads = _counting_read_text(monkeypatch)
    fresh = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    # First search reconciles via walk+stat only — no file may be re-read while
    # the fingerprint map still matches disk.
    assert fresh.find_relevant_files("unique_token_doc7", top_k=5)
    assert reads == [], f"shared build must not re-read files; read {len(reads)}: {reads}"

    # Second search takes the lock-free fast path.
    assert fresh.find_relevant_files("unique_token_doc3", top_k=5)
    assert reads == []


def test_reconcile_is_idempotent_across_instances(tmp_path: Path, monkeypatch) -> None:
    """A third instance created after two reconciles still performs zero reads
    (the diff must stay empty, not re-flag files each time)."""
    _seed(tmp_path, 10)
    _build(tmp_path)

    reads = _counting_read_text(monkeypatch)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert b.find_relevant_files("unique_token_doc1", top_k=5)
    assert reads == []

    c = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert c.find_relevant_files("unique_token_doc2", top_k=5)
    assert reads == [], f"repeated reconciles must stay empty; read {len(reads)}: {reads}"


# ── external changes (outside the write funnel) picked up with O(changed) ────


def test_external_edit_reconciled_without_full_reread(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 30)
    _build(tmp_path)

    # External edit — no invalidate_files (e.g. edited in another tool).
    (tmp_path / "doc0.py").write_text("fresh_external_marker_zzz\ndef f():\n    pass\n")

    reads = _counting_read_text(monkeypatch)
    fresh = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    res = fresh.find_relevant_files("fresh_external_marker_zzz", top_k=5)
    assert any(r.file == "doc0.py" for r in res), "external edit must be searchable"
    assert len(reads) == 1, f"reconciliation must re-read ONLY the changed file; read {len(reads)}: {reads}"


def test_external_delete_removed_by_new_instance(tmp_path: Path) -> None:
    _seed(tmp_path, 10)
    _build(tmp_path)

    (tmp_path / "doc3.py").unlink()

    fresh = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    res = fresh.find_relevant_files("unique_token_doc3", top_k=5)
    assert "doc3.py" not in [r.file for r in res], "deleted file must not be searchable"
    assert "doc3.py" not in fresh._rel_paths
    assert "doc3.py" not in fresh._s.fingerprints


def test_external_add_indexed_by_new_instance(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 5)
    _build(tmp_path)

    (tmp_path / "brand_new.py").write_text("brand_new_marker_qqq\ndef g():\n    pass\n")

    reads = _counting_read_text(monkeypatch)
    fresh = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    res = fresh.find_relevant_files("brand_new_marker_qqq", top_k=5)
    assert any(r.file == "brand_new.py" for r in res), "new file must be searchable"
    assert len(reads) == 1, f"only the new file must be read; read {len(reads)}: {reads}"


def test_tokenless_file_not_reread_on_every_reconcile(tmp_path: Path, monkeypatch) -> None:
    """A tokenless file stays OUT of the index but its fingerprint is stored, so
    a later instance does not re-read it every reconciliation."""
    _seed(tmp_path, 5)
    (tmp_path / "empty.py").write_text("   \n\n")
    _build(tmp_path)

    reads = _counting_read_text(monkeypatch)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert b.find_relevant_files("unique_token_doc0", top_k=5)
    assert reads == [], f"tokenless file must not be re-read; read {len(reads)}: {reads}"


# ── invalidate_files keeps fingerprints in sync ───────────────────────────────


def test_invalidate_refreshes_fingerprints_for_next_instance(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 10)
    a = _build(tmp_path)

    (tmp_path / "doc0.py").write_text("edited_via_invalidate\nchanged content\n")
    a.invalidate_files(["doc0.py"])

    reads = _counting_read_text(monkeypatch)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert b.find_relevant_files("edited_via_invalidate", top_k=5)
    assert reads == [], "invalidate_files must update fingerprints so the next instance skips the file"


def test_invalidate_delete_drops_fingerprint(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 5)
    a = _build(tmp_path)

    (tmp_path / "doc2.py").unlink()
    a.invalidate_files(["doc2.py"])
    assert "doc2.py" not in a._s.fingerprints

    reads = _counting_read_text(monkeypatch)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert b.find_relevant_files("unique_token_doc2", top_k=5)
    assert reads == []
    assert "doc2.py" not in b._rel_paths


# ── cross-instance coherence ──────────────────────────────────────────────────


def test_mutation_by_one_instance_visible_to_another(tmp_path: Path) -> None:
    _seed(tmp_path, 5)
    a = _build(tmp_path)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    assert b.find_relevant_files("unique_token_doc1", top_k=5)  # reconcile (no-op)

    (tmp_path / "doc4.py").write_text("propagated_token_abc\ndef h():\n    pass\n")
    a.invalidate_files(["doc4.py"])

    res = b.find_relevant_files("propagated_token_abc", top_k=5)
    assert any(r.file == "doc4.py" for r in res), "A's invalidation must be visible to B through the shared index"


def test_cross_instance_invalidation_not_served_from_sibling_cache(tmp_path: Path) -> None:
    """A's invalidation clears only A's search cache.  B's own pre-invalidation
    entry must be discarded by the generation-on-read check, not served for the
    5-min TTL."""
    _seed(tmp_path, 5)
    a = _build(tmp_path)
    b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    # Warm BOTH instances' caches with the same query.
    assert a.find_relevant_files("unique_token_doc0", top_k=5)
    assert b.find_relevant_files("unique_token_doc0", top_k=5)

    (tmp_path / "doc0.py").unlink()
    a.invalidate_files(["doc0.py"])

    res = b.find_relevant_files("unique_token_doc0", top_k=5)
    assert not any(r.file == "doc0.py" for r in res), "sibling cache served a stale pre-invalidation result"


# ── cap-mode fallback + registry bound ────────────────────────────────────────


def test_truncated_repo_falls_back_to_full_rebuild(tmp_path: Path, monkeypatch) -> None:
    """Under a file cap the walk itself is incomplete, so an incremental diff
    could drift from a fresh build — the reconcile must fall back to a full
    rebuild (and the fingerprint map must be refreshed by it)."""
    _seed(tmp_path, 8)
    orig_max = rs_mod._MAX_FILES
    rs_mod._MAX_FILES = 4
    try:
        a = _build(tmp_path)
        assert a.index_truncated is True

        builds: list[bool] = []
        orig_build = RAGSearcher._build_index

        def _spy_build(self):
            builds.append(True)
            return orig_build(self)

        monkeypatch.setattr(RAGSearcher, "_build_index", _spy_build)
        b = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
        assert b.find_relevant_files("unique_token_doc0", top_k=5)
        assert len(builds) == 1, "cap-mode repo must fall back to a full rebuild"
        assert b._s.fingerprints, "full rebuild must refresh the fingerprint map"
    finally:
        rs_mod._MAX_FILES = orig_max


def test_shared_registry_is_bounded(tmp_path: Path) -> None:
    """The shared-index registry is LRU-capped; cold repos are evicted and their
    next instance falls back to a full build."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i in range(rs_mod._SHARED_INDEXES_MAX + 3):
            root = Path(td) / f"r{i}"
            root.mkdir()
            (root / "a.py").write_text("x = 1\n")
            RAGSearcher(str(root), vector_cache_enabled=False)
        assert len(_SHARED_INDEXES) <= rs_mod._SHARED_INDEXES_MAX
        # The most recent root is still cached (LRU).  Registry keys are
        # Path.resolve()d (symlinks resolved), so resolve the expected key too.
        last = str((Path(td) / f"r{rs_mod._SHARED_INDEXES_MAX + 2}").resolve())
        assert last in _SHARED_INDEXES
