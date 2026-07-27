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


def test_walk_files_is_source_prioritized_and_deterministic(tmp_path: Path):
    """_walk_files must (1) visit source before tests/fixtures so a tight cap
    admits source first, and (2) yield a deterministic (sorted) order. Direct
    regression for the blindness bug: bare os.walk enumeration order starved
    entire subtrees (on this repo external_llm/ got 0 coverage under the cap)."""
    # src/ (source) + tests/ (deprioritized) with several indexed files each.
    src = tmp_path / "src"
    tests = tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    for i in range(4):
        (src / f"s{i}.py").write_text(f"x = {i}\n")
        (tests / f"t{i}.py").write_text(f"y = {i}\n")

    searcher = RAGSearcher(str(tmp_path))
    # Patch the cap down to 4 so the prioritization is observable; restore after.
    import external_llm.agent.rag_searcher as rs
    orig = rs._MAX_FILES
    rs._MAX_FILES = 4
    try:
        files = searcher._walk_files()
    finally:
        rs._MAX_FILES = orig
    names = {str(f.relative_to(tmp_path)) for f in files}
    assert names == {f"src/s{i}.py" for i in range(4)}, (
        f"source-prioritized walk must admit src/ before tests/; got {names}"
    )
    assert searcher.index_truncated is True, "cap hit must set index_truncated"


def test_walk_files_index_truncated_false_when_complete(tmp_path: Path):
    """index_truncated must be False when the whole tree fits under the cap."""
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    searcher = RAGSearcher(str(tmp_path))
    searcher._walk_files()
    assert searcher.index_truncated is False
