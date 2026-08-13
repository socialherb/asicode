"""Cache-scope mapping for the symbol/search read tools.

``_extract_read_scope_paths`` tags each cached read-only result with the
file/dir subtree it depends on, so a later write only evicts *overlapping*
entries instead of nuking the whole cache (see ``ToolResultCache.invalidate_paths``).

``read_file``/``grep``/``glob`` already report a precise scope. The symbol
tools (``read_symbol``/``find_symbol``/``find_references``) narrow their walk
the same way when given a path arg, but used to report ``None`` (unknown scope)
— so a single-file edit evicted *every* cached symbol lookup, the costliest
pattern in the edit→verify loop. These tests pin the fix and the three
exclusions that keep it from going stale.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.agent.tool_result_cache import ToolResultCache


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for rel in ("a.py", "b.py", "src/c.py", "src/deep/d.py"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def _registry(repo: Path) -> ToolRegistry:
    return ToolRegistry(str(repo), AgentConfig(rag_enabled=False))


class TestSymbolReadScope:
    """Direct mapping checks — the turn-980 measurement matrix, now pinned."""

    @pytest.mark.parametrize("tool, args, scoped", [
        # A path arg narrows the walk → result depends only on that subtree.
        ("read_symbol", {"name": "foo", "file_path": "src/c.py"}, True),
        ("find_symbol", {"name": "foo", "search_path": "src"}, True),
        ("find_references", {"name": "foo", "search_path": "src"}, True),
        # No path arg → repo-wide search → unknown scope.
        ("read_symbol", {"name": "foo"}, None),
        ("find_symbol", {"name": "foo"}, None),
        ("find_references", {"name": "foo"}, None),
        # find_symbol + include_inheritance enriches with subclasses/refs found
        # ANYWHERE in the repo (get_symbol_info) → repo-wide.
        ("find_symbol", {"name": "foo", "search_path": "src", "include_inheritance": True}, None),
        # A path escaping the repo: find_references falls back to a repo-wide rg
        # (_resolve_search_root(...) or self.repo_root), so it can't be scoped.
        ("find_references", {"name": "foo", "search_path": "../outside"}, None),
        ("find_symbol", {"name": "foo", "search_path": "/etc"}, None),
        ("read_symbol", {"name": "foo", "file_path": "../outside"}, None),
        # Graph traversal tools are repo-wide by nature — a file_path arg only
        # disambiguates the symbol, it does not bound the traversal.
        ("analyze_change_impact", {"symbol": "foo", "file_path": "src/c.py"}, None),
        ("query_dependency_graph", {"source": "src/", "mode": "subgraph"}, None),
    ])
    def test_scope_mapping(self, repo: Path, tool: str, args: dict, scoped):
        result = _registry(repo)._extract_read_scope_paths(tool, args)
        if scoped is None:
            assert result is None, f"{tool} {args} should be repo-wide (None)"
        else:
            assert result is not None and len(result) == 1, f"{tool} {args} should be scoped"
            assert all(os.path.isabs(p) for p in result)

    def test_scope_is_the_given_subtree(self, repo: Path):
        """The reported scope must be the normalized absolute path of the arg,
        so _paths_overlap can match a later write to a file under it."""
        reg = _registry(repo)
        scope = reg._extract_read_scope_paths("find_symbol", {"name": "foo", "search_path": "src"})
        assert scope == frozenset({os.path.normpath(str(repo / "src"))})

    def test_existing_tools_unchanged(self, repo: Path):
        """read_file/grep/glob scope behavior must not regress."""
        reg = _registry(repo)
        assert reg._extract_read_scope_paths("read_file", {"path": "a.py"}) == frozenset(
            {os.path.normpath(str(repo / "a.py"))}
        )
        assert reg._extract_read_scope_paths("grep", {"pattern": "x"}) is None
        assert reg._extract_read_scope_paths("glob", {"pattern": "*.py", "path": "src"}) is not None


class TestSymbolCacheSurvivesNonOverlappingWrite:
    """The fix's whole point: editing file B must NOT evict a cached symbol
    lookup scoped to file/subtree A. Before the fix these reported None and a
    write *anywhere* dropped them."""

    def test_find_symbol_scoped_to_src_survives_edit_of_a_py(self, repo: Path):
        reg = _registry(repo)
        cache = ToolResultCache()
        scope = reg._extract_read_scope_paths("find_symbol", {"name": "foo", "search_path": "src"})
        assert scope is not None  # the fix; was None before
        args = {"name": "foo", "search_path": "src"}
        cache.set("find_symbol", args, {"ok": True, "content": "hit", "error": "", "metadata": {}}, paths=scope)

        # Edit a.py — outside src/ — must leave the src-scoped entry intact.
        removed = cache.invalidate_paths(frozenset({os.path.normpath(str(repo / "a.py"))}))
        assert removed == 0
        assert cache.get("find_symbol", args) is not None

        # Edit src/c.py — under src/ — must drop it.
        removed = cache.invalidate_paths(frozenset({os.path.normpath(str(repo / "src" / "c.py"))}))
        assert removed == 1
        assert cache.get("find_symbol", args) is None

    def test_read_symbol_scoped_to_one_file_survives_edit_of_another(self, repo: Path):
        reg = _registry(repo)
        cache = ToolResultCache()
        scope = reg._extract_read_scope_paths("read_symbol", {"name": "foo", "file_path": "src/c.py"})
        assert scope is not None
        args = {"name": "foo", "file_path": "src/c.py"}
        cache.set("read_symbol", args, {"ok": True, "content": "hit", "error": "", "metadata": {}}, paths=scope)

        # Edit src/deep/d.py — different file, same parent dir — must survive.
        removed = cache.invalidate_paths(frozenset({os.path.normpath(str(repo / "src" / "deep" / "d.py"))}))
        assert removed == 0
        assert cache.get("read_symbol", args) is not None

        # Edit the scoped file itself → dropped.
        removed = cache.invalidate_paths(frozenset({os.path.normpath(str(repo / "src" / "c.py"))}))
        assert removed == 1
        assert cache.get("read_symbol", args) is None

    def test_unscoped_find_symbol_is_still_dropped_by_any_write(self, repo: Path):
        """Conservative guarantee preserved: a repo-wide (unscoped) lookup is
        still evicted by any write, since we can't prove it doesn't depend on
        the written file."""
        reg = _registry(repo)
        cache = ToolResultCache()
        scope = reg._extract_read_scope_paths("find_symbol", {"name": "foo"})
        assert scope is None
        args = {"name": "foo"}
        cache.set("find_symbol", args, {"ok": True, "content": "hit", "error": "", "metadata": {}}, paths=scope)

        removed = cache.invalidate_paths(frozenset({os.path.normpath(str(repo / "a.py"))}))
        assert removed == 1
        assert cache.get("find_symbol", args) is None

