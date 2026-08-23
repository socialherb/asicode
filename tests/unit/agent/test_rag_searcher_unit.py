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
    assert not any("node_modules" in f for f in rel_files), "node_modules should be pruned"
    assert not any(".git" in f for f in rel_files), ".git should be pruned"


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


def test_dotfile_survives_incremental_edit(tmp_path: Path):
    """F-RAG-1 regression: the build walk indexes basename dotfiles
    (.eslintrc.js) but ``_prepare_files``' all-parts check dropped them on the
    first incremental edit — a one-edit-then-gone-from-search bug.  Only
    DIRECTORY parts may be pruned; the basename is never a directory."""
    (tmp_path / "src" / "app.py").parent.mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("print(1)\n")
    (tmp_path / ".eslintrc.js").write_text("module.exports = {}\n")

    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    searcher._ensure_index()
    assert ".eslintrc.js" in searcher._rel_paths, "build must index basename dotfiles"

    (tmp_path / ".eslintrc.js").write_text("module.exports = {a: 1}\n")
    searcher.invalidate_files([".eslintrc.js"])
    assert ".eslintrc.js" in searcher._rel_paths, "incremental edit must not drop the dotfile"
    assert "src/app.py" in searcher._rel_paths


def test_walk_files_applies_shared_skip_policy(tmp_path: Path):
    """4th-walker parity (B2' contract, F-RAG-2): the RAG walker must prune
    exactly the dirs the shared policy prunes (vendor/, .egg-info/, venv*/
    site-packages) and skip minified-bundle suffixes — or the corpus drifts
    from the CGI/RG graph universes (vendored bundles / venv site-packages
    can even starve real source under the file cap)."""
    files = {
        "src/app.py": "print(1)\n",
        "vendor/dep.py": "x = 1\n",
        "pkg.egg-info/meta.py": "y = 2\n",
        "venv310/lib/python3.14/site-packages/mod.py": "z = 3\n",
        "assets/lib.min.js": "function f() {}\n",
        "app.min.css": "a { color: red }\n",
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    rels = {str(f.relative_to(tmp_path)) for f in searcher._walk_files()}
    assert rels == {"src/app.py"}, rels


def test_prepare_files_matches_walk_admission(tmp_path: Path):
    """Incremental admission must equal walk admission (F-RAG-2): files under
    pruned dirs / with minified suffixes are refused in ``_prepare_files``
    too, so an invalidate_files call can never index what a rebuild drops."""
    for rel in ("vendor/dep.py", "lib.min.js", "app.min.css"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n", encoding="utf-8")

    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    prepared, fp_map = searcher._prepare_files(["vendor/dep.py", "lib.min.js", "app.min.css"])
    assert prepared == {}
    assert fp_map == {}


def test_invalidate_files_batches_vector_cache_adds(tmp_path: Path):
    """F3: the deferred vector-cache updates of an incremental update are
    flushed in ONE ``add_documents`` batch (single encode pass) instead of N
    per-file ``add_document`` calls — the per-file path must never be used."""
    from unittest.mock import MagicMock

    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")

    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    searcher._ensure_index()

    mock_mgr = MagicMock()
    searcher.vector_cache_manager = mock_mgr
    searcher.vector_cache_enabled = True

    (tmp_path / "a.py").write_text("def a():\n    return 10\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 20\n", encoding="utf-8")
    searcher.invalidate_files(["a.py", "b.py"])

    mock_mgr.add_documents.assert_called_once()
    added = mock_mgr.add_documents.call_args.args[0]
    assert {p for p, _t in added} == {"a.py", "b.py"}
    mock_mgr.add_document.assert_not_called()


def test_invalidate_files_mixed_batch_keeps_arrays_aligned(tmp_path: Path):
    """P2: a single incremental batch mixing updates, removals and appends must
    keep every parallel array in lockstep.  Removals are deferred and applied
    in descending index order, so a snapshot-based path→idx lookup stays valid
    for the whole loop and the mirror matches the arrays afterwards."""
    for i in range(4):
        (tmp_path / f"f{i}.py").write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")
    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    searcher._ensure_index()
    assert searcher._rel_paths == ["f0.py", "f1.py", "f2.py", "f3.py"]

    # One batch: update f1, delete f0 + f2 (unlinked), add f4.
    (tmp_path / "f1.py").write_text("def f1():\n    return 10\n", encoding="utf-8")
    (tmp_path / "f0.py").unlink()
    (tmp_path / "f2.py").unlink()
    (tmp_path / "f4.py").write_text("def f4():\n    return 4\n", encoding="utf-8")
    searcher.invalidate_files(["f0.py", "f1.py", "f2.py", "f4.py"])

    assert searcher._rel_paths == ["f1.py", "f3.py", "f4.py"]
    assert len(searcher._doc_token_counts) == len(searcher._rel_paths)
    assert len(searcher._doc_lengths) == len(searcher._rel_paths)
    assert len(searcher._doc_texts) == len(searcher._rel_paths)
    assert searcher._n_docs == len(searcher._rel_paths)
    assert "10" in searcher._doc_texts[0], "updated f1 must survive at its shifted index"
    assert "4" in searcher._doc_texts[2], "appended f4 must be present"
    # Mirror rebuilt to match the mutated arrays.
    assert searcher._rel_path_to_idx == {p: i for i, p in enumerate(searcher._rel_paths)}


def test_invalidate_files_removal_frees_cap_slot_in_same_batch(tmp_path: Path):
    """P2: with the index at the cap, deleting one file and adding a new one in
    the SAME batch must admit the newcomer — deferred removals must not shrink
    the effective capacity mid-batch (the old in-loop pop-then-append
    semantics)."""
    import external_llm.agent.rag_searcher as rs

    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("def c():\n    return 3\n", encoding="utf-8")
    searcher = RAGSearcher(str(tmp_path), vector_cache_enabled=False)
    orig = rs._MAX_FILES
    rs._MAX_FILES = 3
    try:
        searcher._ensure_index()
        assert searcher._n_docs == 3
        # Delete b.py (unlinked) and add d.py in the same batch.
        (tmp_path / "b.py").unlink()
        (tmp_path / "d.py").write_text("def d():\n    return 4\n", encoding="utf-8")
        searcher.invalidate_files(["b.py", "d.py"])
    finally:
        rs._MAX_FILES = orig

    assert searcher._n_docs == 3, "the removal must free the slot for the append"
    assert searcher._rel_paths == ["a.py", "c.py", "d.py"]
    assert searcher._rel_path_to_idx == {p: i for i, p in enumerate(searcher._rel_paths)}
