"""
Unit tests for ToolRegistry cache consistency improvements.
"""
import os
import shutil
import tempfile
import time
from unittest.mock import Mock

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def test_cache_invalidation_across_agents():
    """Test that cache invalidation affects all agents."""
    repo_root = tempfile.mkdtemp()
    test_file = os.path.join(repo_root, "test.py")

    # Initial file creation
    with open(test_file, "w") as f:
        f.write("original content\n")

    # Create two agents
    config = AgentConfig()
    agent1 = ToolRegistry(repo_root, config)
    agent2 = ToolRegistry(repo_root, config)

    # Both agents read the file (fill cache)
    agent1._tool_read_file({"path": "test.py"})
    agent2._tool_read_file({"path": "test.py"})

    # Modify file
    with open(test_file, "w") as f:
        f.write("modified content\n")

    # Wait for mtime change
    time.sleep(0.1)

    # agent1 invalidates cache
    agent1._invalidate_cache_after_write(["test.py"])

    # agent2 reads new content (cache invalidated, fresh read)
    result3 = agent2._tool_read_file({"path": "test.py"})
    assert "modified content" in result3.content


def test_rag_index_invalidation_on_write():
    """Test that RAG index is invalidated after write operations."""
    repo_root = tempfile.mkdtemp()
    config = AgentConfig()
    registry = ToolRegistry(repo_root, config)

    # Mock RAG searcher
    mock_rag = Mock()
    registry._rag_searcher = mock_rag

    # Call cache invalidation
    registry._invalidate_cache_after_write(["test.py"])

    # Verify invalidate_files was called
    mock_rag.invalidate_files.assert_called_once()


def test_invalidate_after_write_normalizes_absolute_snapshot_paths():
    """Regression (44bc2eb9): _snapshot_target_files builds ABSOLUTE target
    paths (os.path.join(repo_root, target)), but the three incremental
    invalidators — CallGraphIndexer, RAGSearcher, GraphFacade — all assume
    repo-relative and re-join against the root. An un-normalized absolute path
    survives their strip().lstrip("/") as "Users/.../a.py" and the re-join
    points at a nonexistent path, silently no-op'ing the invalidation: the
    index keeps answering with pre-write state ("can't find the symbol it just
    wrote"). The entry of _invalidate_cache_after_write must normalize both
    forms to repo-relative once, up front."""
    repo_root = tempfile.mkdtemp()
    try:
        with open(os.path.join(repo_root, "a.py"), "w") as f:
            f.write("def foo(): pass\n")
        registry = ToolRegistry(repo_root, AgentConfig())
        cgi = registry._call_graph.call_graph_indexer
        cgi.get_callers("foo")  # lazy build — index the original file
        assert "foo" in cgi._nodes

        # Semantic write (edit_text / edit_ast / anchor_edit / modify_symbol):
        # the snapshot that drives invalidation is keyed by absolute path.
        with open(os.path.join(repo_root, "a.py"), "w") as f:
            f.write("def foo(): pass\ndef gamma(): pass\n")
        registry._invalidate_cache_after_write(
            [os.path.join(repo_root, "a.py")]
        )
        # Without entry normalization gamma would be invisible (stale index).
        assert "gamma" in cgi._nodes

        # Relative form must keep working through the same path.
        with open(os.path.join(repo_root, "a.py"), "w") as f:
            f.write("def foo(): pass\ndef delta(): pass\n")
        registry._invalidate_cache_after_write(["a.py"])
        assert "delta" in cgi._nodes

        # RAGSearcher receives the RELATIVE form, never the absolute one.
        mock_rag = Mock()
        registry._rag_searcher = mock_rag
        registry._invalidate_cache_after_write(
            [os.path.join(repo_root, "a.py")]
        )
        mock_rag.invalidate_files.assert_called_once_with(["a.py"])
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_detect_repo_language_cached_per_repo_root(tmp_path, monkeypatch):
    """Turn 13122 fix 2: _detect_repo_language memoizes per repo_root — the
    os.walk (~250-450ms) must run at most once per repo, not on every
    ToolRegistry construction (IPC worker builds one per task)."""
    import os as _os
    import subprocess as _sp

    from external_llm.agent.tool_registry import ToolRegistry

    _sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")

    ToolRegistry._LANGUAGE_DETECTION_CACHE.pop(_os.path.normpath(str(tmp_path)), None)
    first = ToolRegistry._detect_repo_language(str(tmp_path))
    # Second call must be served from the cache — os.walk would blow up.
    def _no_walk(*a, **k):
        raise AssertionError("os.walk called on a cache hit")
    monkeypatch.setattr(_os, "walk", _no_walk)
    assert ToolRegistry._detect_repo_language(str(tmp_path)) == first
    # A Python repo caches None (all tools visible) — None must ALSO hit.
    assert first is None


def test_detect_repo_language_ignores_vendored_trees(tmp_path):
    """F7: vendored trees (site-packages/ venv310/ *.egg-info/) are excluded
    from the language-detection count via the shared walk_policy predicate.

    Pure-non-Python repos pick their dominant family by raw file count, so a
    vendored copy of language B inside site-packages/ would OUT-VOTE the real
    language-A source and mis-mask the Python-only tool set.  Here 1 real .ts
    must win over 20 vendored .go — without F7 the exact-match _COUNT_SKIP_DIRS
    descended into site-packages/ and returned GO instead of TYPESCRIPT."""
    import subprocess as _sp

    from external_llm.languages import LanguageId

    _sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "real.ts").write_text("export const x = 1;\n")
    vdir = tmp_path / "site-packages"
    vdir.mkdir()
    for i in range(20):
        (vdir / f"v{i}.go").write_text(f"package v\nfunc V{i}() {{}}\n")

    ToolRegistry._LANGUAGE_DETECTION_CACHE.pop(str(tmp_path), None)
    assert ToolRegistry._detect_repo_language(str(tmp_path)) is LanguageId.TYPESCRIPT


def test_cache_hit_metadata_not_aliased_to_cache_entry():
    """Regression: on a cache HIT, dispatch reconstructed a ToolResult whose
    ``metadata`` was a REFERENCE to the dict stored in the cache entry, then set
    ``result.metadata["cache_hit"] = True`` — mutating the cache entry's own
    dict. Any caller-side metadata addition then leaked back into the cache and
    propagated to every later hit (and cache_hit got permanently baked in).

    The fix copies the metadata dict when reconstructing the ToolResult, so the
    returned result's metadata is independent of the cached entry."""
    import tempfile

    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.agent.tool_result_cache import ToolResultCache

    repo_root = tempfile.mkdtemp()
    registry = ToolRegistry(repo_root, AgentConfig())
    # Enable the result cache directly (isolates the dispatch hit path).
    registry._tool_result_cache = ToolResultCache()

    # Seed the cache with a read-only entry carrying its own metadata.
    registry._tool_result_cache.set(
        "read_file", {"path": "x.py"},
        {"ok": True, "content": "DATA", "error": None, "metadata": {"original": True}},
    )

    # First dispatch → cache HIT.
    r1 = registry.dispatch("read_file", {"path": "x.py"})
    assert r1.ok is True
    assert r1.content == "DATA"
    assert r1.metadata.get("cache_hit") is True
    assert r1.metadata.get("original") is True

    # Caller mutates the returned result's metadata (e.g. adds provenance).
    r1.metadata["caller_added"] = "should-not-leak"

    # Second dispatch → cache HIT again. Before the fix, the cache entry's
    # metadata had been aliased+mutated, so this would carry caller_added AND a
    # permanently-baked cache_hit. After the fix it is a fresh copy.
    r2 = registry.dispatch("read_file", {"path": "x.py"})
    assert r2.ok is True
    assert r2.metadata.get("cache_hit") is True
    assert r2.metadata.get("original") is True
    # The caller-side mutation must NOT have polluted the cache entry.
    assert "caller_added" not in r2.metadata, (
        "cache entry metadata was aliased into the hit result and polluted "
        "by a caller-side mutation"
    )


def test_mid_read_external_rewrite_not_served_as_fresh(tmp_path):
    """TOCTOU regression at dispatch level: the result-cache signature must be
    captured BEFORE the read handler runs. When an external writer (background
    job, parallel session) rewrites the file mid-dispatch, the cached entry
    must be dropped by the next call — not served as fresh for the whole TTL.

    Without the pre-capture, set() snapshots the POST-read signature, so the
    stale content races a fresh signature and the guard never fires (the exact
    scenario the signature guard exists to catch)."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

    p = tmp_path / "f.py"
    p.write_text("VERSION = 1\n")
    registry = ToolRegistry(str(tmp_path), AgentConfig())
    assert registry._tool_result_cache is not None

    real_handler = registry._tool_read_file

    def racing_handler(args):
        result = real_handler(args)  # handler reads VERSION = 1
        p.write_text("VERSION = 2  # external writer, mid-dispatch\n")
        return result

    registry._tool_read_file = racing_handler

    r1 = registry.dispatch("read_file", {"path": "f.py"})
    assert "VERSION = 1" in r1.content

    # The file on disk is v2 — the cached v1 entry must be dropped by the
    # signature guard (pre-read sig v1 != current sig v2), so this call
    # re-reads and returns v2. Pre-fix it served the stale v1.
    r2 = registry.dispatch("read_file", {"path": "f.py"})
    assert "VERSION = 2" in r2.content


def test_network_reads_empty_scope_survive_file_writes(tmp_path):
    """search_web/web_fetch depend on NO repo file — a file write must not
    invalidate them. They must report an EMPTY scope (frozenset(), distinct
    from None=unknown scope), which invalidate_paths() keeps while still
    dropping unknown-scope entries."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.agent.tool_result_cache import ToolResultCache

    registry = ToolRegistry(str(tmp_path), AgentConfig())
    for tool in ("search_web", "web_fetch"):
        scope = registry._extract_read_scope_paths(tool, {"query": "asdf"})
        assert scope == frozenset(), f"{tool} scope must be empty, got {scope!r}"

    cache = ToolResultCache()
    cache.set("search_web", {"query": "asdf"}, {"ok": True}, paths=frozenset())
    cache.set(
        "read_file", {"path": "a.py"}, {"ok": True},
        paths=frozenset({"/r/a.py"}),
    )
    removed = cache.invalidate_paths(frozenset({"/r/a.py"}))
    assert removed == 1
    assert cache.get("search_web", {"query": "asdf"}) is not None  # survived
    assert cache.get("read_file", {"path": "a.py"}) is None        # dropped


def test_unknown_scope_reads_still_dropped_by_any_write(tmp_path):
    """Empty scope (network) and unknown scope (None) must stay distinct:
    unknown-scope entries keep the conservative always-drop behavior."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.agent.tool_result_cache import ToolResultCache

    registry = ToolRegistry(str(tmp_path), AgentConfig())
    # find_symbol without a path → repo-wide → unknown scope (None)
    scope = registry._extract_read_scope_paths("find_symbol", {"name": "Foo"})
    assert scope is None

    cache = ToolResultCache()
    cache.set("search_web", {"query": "asdf"}, {"ok": True}, paths=frozenset())
    cache.set("find_symbol", {"name": "Foo"}, {"ok": True})
    removed = cache.invalidate_paths(frozenset({"/r/anything.py"}))
    assert removed == 1  # only the unknown-scope entry
    assert cache.get("search_web", {"query": "asdf"}) is not None
    assert cache.get("find_symbol", {"name": "Foo"}) is None
