"""Unit tests for RAGSearcher._walk_files directory pruning."""
from pathlib import Path

from external_llm.agent.rag_searcher import RAGSearcher


def test_walk_files_prunes_vendor_dirs(tmp_path: Path):
    """_walk_files prunes node_modules/.git via os.walk dirs[:] assignment,
    not rglob which descends into every directory before filtering."""
    (tmp_path / "src" / "main.py").parent.mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "node_modules" / "pkg" / "index.js").parent.mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")

    searcher = RAGSearcher(str(tmp_path))
    files = searcher._walk_files()
    rel_files = [str(f.relative_to(tmp_path)) for f in files]

    assert "src/main.py" in rel_files
    assert not any("node_modules" in f for f in rel_files), \
        "node_modules should be pruned"
    assert not any(".git" in f for f in rel_files), \
        ".git should be pruned"
