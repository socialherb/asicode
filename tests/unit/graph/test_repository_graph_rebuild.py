"""Unit tests for RepositoryGraph.build()'s rebuild (idempotence) contract.

Re-calling ``build()`` on the same instance must produce a graph identical to
a fresh instance's single build: call/import edges and file_symbols are reset
(not appended to), and symbols of files deleted between builds must not linger
(P1, 2026-08-11).
"""
import os
import tempfile
import textwrap
from pathlib import Path

from external_llm.graph.repository_graph import RepositoryGraph


def _make_repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="test_rg_rebuild_")
    for rel_path, source in files.items():
        full = Path(d) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(source))
    return d


def _edge_tuples(graph):
    """Value snapshot — edge objects are recreated each build, compare tuples."""
    calls = [(e.caller, e.callee, e.file_path, e.line) for e in graph.call_edges]
    imports = [(e.importer, e.imported, e.import_type) for e in graph.import_edges]
    return calls, imports


def test_rebuild_does_not_duplicate_call_and_import_edges():
    """N builds == 1 build: edge lists must be identical, not doubled."""
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "b.py": "def fb():\n    return fa()\n",
            "c.py": "import a\nimport b\n",
        }
    )
    graph = RepositoryGraph(repo)
    graph.build()
    want_calls, want_imports = _edge_tuples(graph)
    for _ in range(3):
        graph.build()
        assert _edge_tuples(graph) == (want_calls, want_imports)
    assert len(graph.call_edges) == 1  # fb -> fa, exactly once
    assert len(graph.import_edges) == 2  # c -> a and c -> b, exactly once


def test_rebuild_resets_file_symbols():
    """file_symbols must not accumulate duplicate unique_ids per file."""
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "sub/b.py": "class B:\n    def m(self):\n        return 1\n",
        }
    )
    graph = RepositoryGraph(repo)
    for _ in range(3):
        graph.build()
        for rel, ids in graph.file_symbols.items():
            assert len(ids) == len(set(ids)), (rel, ids)
        assert len(graph.get_symbols_in_file("a.py")) == 1
        assert len(graph.get_symbols_in_file("sub/b.py")) == 2  # class + method


def test_rebuild_drops_state_of_deleted_files():
    """Symbols/edges of a file deleted between builds must not linger."""
    repo = _make_repo(
        {
            "a.py": "def fa():\n    return 1\n",
            "main.py": "import a\ndef run():\n    return fa()\n",
        }
    )
    graph = RepositoryGraph(repo)
    graph.build()
    assert graph.get_importers("a.py") == ["main.py"]
    assert graph.get_symbols_in_file("main.py")

    os.remove(Path(repo) / "main.py")
    graph.build()

    assert graph.get_importers("a.py") == []
    assert graph.get_symbols_in_file("main.py") == []
    assert "main.py" not in graph.file_symbols
    assert not any(e.file_path == "main.py" for e in graph.call_edges)
    assert not any(e.importer == "main.py" for e in graph.import_edges)
