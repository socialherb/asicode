"""
Unit tests for ToolRegistry cache consistency improvements.
"""
import os
import tempfile
import time
from unittest.mock import Mock

from external_llm.agent.file_cache import get_global_file_cache, reset_global_file_cache
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def test_global_file_cache_singleton():
    """Test that get_global_file_cache() returns the same instance."""
    # Singleton verification
    cache1 = get_global_file_cache()
    cache2 = get_global_file_cache()
    assert cache1 is cache2, "Global file cache should be a singleton"

    # ToolRegistry uses global cache verification
    repo_root = tempfile.mkdtemp()
    registry = ToolRegistry(repo_root, AgentConfig())
    assert registry._file_cache is cache1, "ToolRegistry should use global cache"


def test_cache_invalidation_across_agents():
    """Test that cache invalidation affects all agents."""
    # Reset global cache to ensure clean state
    reset_global_file_cache()

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


def test_cache_hit_metadata_not_aliased_to_cache_entry():
    """Regression: on a cache HIT, dispatch reconstructed a ToolResult whose
    ``metadata`` was a REFERENCE to the dict stored in the cache entry, then set
    ``result.metadata["cache_hit"] = True`` — mutating the cache entry's own
    dict. Any caller-side metadata addition then leaked back into the cache and
    propagated to every later hit (and cache_hit got permanently baked in).

    The fix copies the metadata dict when reconstructing the ToolResult, so the
    returned result's metadata is independent of the cached entry."""
    import tempfile
    from external_llm.agent.tool_result_cache import ToolResultCache
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

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
