"""Batched incremental invalidation (P3, 2026-08-11).

RepositoryGraph.remove_file was historically called once per changed path, and
each call rebuilt call_edges / import_edges / _symbol_locations (tens of
thousands of entries on a real repo) from scratch — O(N paths x M edges).

P3 adds _remove_files(set) (one pass per list, set membership) and reparse_files
(batched reparse_file), and the facade now splits changed paths into reparse vs
deleted sets and hands each to the batched path. These tests pin the contract
that batched invalidation produces the SAME graph state as the per-path loop.
"""

import shutil
import tempfile
import textwrap
from pathlib import Path

from external_llm.graph.repository_graph import RepositoryGraph


def _make_repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="test_rg_batch_")
    for rel, src in files.items():
        full = Path(d) / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(src))
    return d


def _symbol_names(g):
    return sorted(s.name for s in g.symbols.values())


def test_remove_files_equivalent_to_remove_file_per_path():
    """_remove_files(set) must leave the same state as remove_file x N."""
    repo = _make_repo(
        {
            "a.py": "def a():\n    b()\n",
            "b.py": "def b():\n    pass\n",
            "c.py": "def c():\n    a()\n",
            "d.py": "def d():\n    c()\n",
        }
    )
    try:
        # Reference: remove a and c via the single-file API
        g_ref = RepositoryGraph(repo)
        g_ref.build()
        g_ref.remove_file("a.py")
        g_ref.remove_file("c.py")

        # Batch: remove a and c in one call
        g = RepositoryGraph(repo)
        g.build()
        g._remove_files({"a.py", "c.py"})

        assert _symbol_names(g) == _symbol_names(g_ref)
        assert sorted(g.file_symbols) == sorted(g_ref.file_symbols)
        assert len(g.call_edges) == len(g_ref.call_edges)
        assert len(g.import_edges) == len(g_ref.import_edges)
        # survivors untouched
        assert "b" in _symbol_names(g) and "d" in _symbol_names(g)
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_reparse_files_equivalent_to_reparse_file_per_path(monkeypatch):
    """reparse_files(list) must leave the same state as reparse_file x N."""
    repo = _make_repo(
        {
            "a.py": "def a():\n    pass\n",
            "b.py": "def b():\n    pass\n",
            "c.py": "def c():\n    pass\n",
        }
    )
    try:
        # Reference: reparse a and c with edits
        g_ref = RepositoryGraph(repo)
        g_ref.build()
        (Path(repo) / "a.py").write_text("def a_new():\n    pass\n")
        (Path(repo) / "c.py").write_text("def c_new():\n    pass\n")
        g_ref.reparse_file(str(Path(repo) / "a.py"))
        g_ref.reparse_file(str(Path(repo) / "c.py"))
        ref_names = _symbol_names(g_ref)

        # Reset files, then batch-reparse the same edits
        (Path(repo) / "a.py").write_text("def a():\n    pass\n")
        (Path(repo) / "c.py").write_text("def c():\n    pass\n")
        g = RepositoryGraph(repo)
        g.build()
        (Path(repo) / "a.py").write_text("def a_new():\n    pass\n")
        (Path(repo) / "c.py").write_text("def c_new():\n    pass\n")
        g.reparse_files([str(Path(repo) / "a.py"), str(Path(repo) / "c.py")])

        assert _symbol_names(g) == ref_names, f"batch reparse diverged from per-path: {_symbol_names(g)} != {ref_names}"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_remove_files_empty_is_noop():
    repo = _make_repo({"a.py": "def a():\n    pass\n"})
    try:
        g = RepositoryGraph(repo)
        g.build()
        before = len(g.symbols)
        g._remove_files(set())
        assert len(g.symbols) == before
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_reparse_files_dedups_when_same_path_passed_twice():
    """A rel set guards against double-processing the same file."""
    repo = _make_repo({"a.py": "def a():\n    pass\n"})
    try:
        g = RepositoryGraph(repo)
        g.build()
        (Path(repo) / "a.py").write_text("def a():\n    b()\n\ndef b():\n    pass\n")
        g.reparse_files(
            [
                str(Path(repo) / "a.py"),
                str(Path(repo) / "a.py"),  # duplicate
            ]
        )
        # a appears exactly once (no double-injection of edges)
        a_edges = [e for e in g.call_edges if e.caller == "a"]
        assert len(a_edges) == 1, f"duplicate reparse injected {len(a_edges)} caller=a edges"
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_reparse_files_does_not_load_disk_snapshot(monkeypatch):
    """P0-1: incremental reparse skips the on-disk snapshot tier entirely.

    A just-reparsed file can never match the disk manifest stamp, so loading
    the whole snapshot JSON for it is pure waste (measured ~0.93s stall +
    ~209MB resident on asicode, 2026-08-12).  reparse_files passes
    use_disk_tier=False; the regression guard asserts the snapshot loader is
    never consulted and ``_disk_cache`` stays None after the reparse.
    """
    repo = _make_repo({"a.py": "def a():\n    pass\n"})
    try:
        g = RepositoryGraph(repo)
        g.build()
        assert g._disk_cache is None  # build() releases it at the end

        calls: list[int] = []
        orig = RepositoryGraph._load_disk_cache_snapshot

        def spy(self):
            calls.append(1)
            return orig(self)

        monkeypatch.setattr(RepositoryGraph, "_load_disk_cache_snapshot", spy)
        (Path(repo) / "a.py").write_text("def a2():\n    pass\n")
        g.reparse_files([str(Path(repo) / "a.py")])

        assert calls == [], "reparse_files must not load the disk snapshot"
        assert g._disk_cache is None
        assert "a2" in _symbol_names(g)  # the file was parsed, not skipped
    finally:
        shutil.rmtree(repo, ignore_errors=True)
