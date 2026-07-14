"""Unit tests for async_tool_executor.py — full coverage.

Tests:
  - AsyncToolExecutor construction
  - _group_by_dependencies (with and without dependency graph)
  - _has_dependency
  - _execute_single (success and failure paths via mocked registry)
  - execute_parallel (orchestration with layered execution)
  - shutdown
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from external_llm.agent.async_tool_executor import AsyncToolExecutor, ToolResult

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_registry():
    """A MagicMock that looks like a ToolRegistry WITHOUT dependency_graph."""
    registry = MagicMock()
    registry.dispatch.return_value = ToolResult(ok=True, content="done", execution_time=0.0)
    # MagicMock auto-creates attributes on hasattr — explicitly clear it
    registry.dependency_graph = None
    return registry


@pytest.fixture
def mock_registry_with_dep_graph():
    """Registry with a dependency graph that has_path returns True for specific pairs."""
    registry = MagicMock()
    registry.dispatch.return_value = ToolResult(ok=True, content="done", execution_time=0.0)
    dep_graph = MagicMock()
    # add_dependency(from, to) adds edge from→to, meaning from must execute before to.
    # has_path("read_file", "grep")=True means read_file→grep edge: read_file runs first,
    # grep depends on read_file.
    dep_graph.has_path.side_effect = lambda src, tgt: (src, tgt) in [("read_file", "grep")]
    registry.dependency_graph = dep_graph
    return registry


# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """AsyncToolExecutor.__init__"""

    def test_default_construction(self, mock_registry):
        executor = AsyncToolExecutor(mock_registry, max_workers=3)
        assert executor.registry is mock_registry
        assert executor.max_workers == 3
        assert executor.executor is not None
        assert executor.dependency_graph is None

    def test_construction_with_dep_graph(self, mock_registry_with_dep_graph):
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        assert executor.dependency_graph is not None


# ═══════════════════════════════════════════════════════════════════════════
# _group_by_dependencies
# ═══════════════════════════════════════════════════════════════════════════


class TestGroupByDependencies:
    """_group_by_dependencies topological layering."""

    def test_no_dep_graph_all_in_one_group(self, mock_registry):
        executor = AsyncToolExecutor(mock_registry)
        tool_calls = [
            {"tool": "read_file", "args": {"path": "a.py"}},
            {"tool": "grep", "args": {"pattern": "foo"}},
            {"tool": "bash", "args": {"command": "ls"}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_empty_calls_returns_empty_group(self, mock_registry):
        executor = AsyncToolExecutor(mock_registry)
        groups = executor._group_by_dependencies([])
        # Returns [list(range(0))] = [[]] — a single empty group
        assert groups == [[]]


class TestGroupByDependenciesWithGraph:
    """_group_by_dependencies with dependency graph."""

    def test_single_tool(self, mock_registry_with_dep_graph):
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        groups = executor._group_by_dependencies([{"tool": "bash", "args": {}}])
        assert len(groups) == 1
        assert groups[0] == [0]

    def test_independent_tools_same_layer(self, mock_registry_with_dep_graph):
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        # read_file depends on grep via has_path
        tool_calls = [
            {"tool": "grep", "args": {}},
            {"tool": "bash", "args": {}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        # Both independent → one group
        assert len(groups) == 1

    def test_dependent_tools_separate_layers(self, mock_registry_with_dep_graph):
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        # read_file → grep edge: read_file (idx 0) has no deps, grep (idx 1) depends on read_file.
        # Topological sort: prerequisites first → read_file in layer 1, grep in layer 2.
        tool_calls = [
            {"tool": "read_file", "args": {}},
            {"tool": "grep", "args": {}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        assert len(groups) == 2
        # read_file (no deps) in first layer
        assert groups[0] == [0]
        # grep in second layer (depends on read_file)
        assert groups[1] == [1]

    def test_single_tool_layer_with_cyclic_dependency(self, mock_registry_with_dep_graph):
        """All remaining tools in one layer when cyclic dependency is detected."""
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        # Clear the side_effect and return True for all calls
        executor.dependency_graph.has_path.side_effect = None
        executor.dependency_graph.has_path.return_value = True
        tool_calls = [
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}},
            {"tool": "c", "args": {}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        # All nodes have incoming deps → all go into one fallback layer
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# _has_dependency
# ═══════════════════════════════════════════════════════════════════════════


class TestHasDependency:
    """_has_dependency delegates to dependency_graph.has_path."""

    def test_no_dep_graph_returns_false(self, mock_registry):
        executor = AsyncToolExecutor(mock_registry)
        assert executor._has_dependency("a", "b") is False

    def test_with_dep_graph_has_path(self, mock_registry_with_dep_graph):
        executor = AsyncToolExecutor(mock_registry_with_dep_graph)
        # _has_dependency(tool1, tool2) checks has_path(tool2, tool1).
        # has_path("read_file", "grep") = True ⇒ _has_dependency("grep", "read_file") = True
        assert executor._has_dependency("grep", "read_file") is True
        assert executor._has_dependency("read_file", "grep") is False


# ═══════════════════════════════════════════════════════════════════════════
# _execute_single
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteSingle:
    """_execute_single wraps dispatch in thread pool."""

    def test_success(self, mock_registry):
        async def _run():
            executor = AsyncToolExecutor(mock_registry)
            return await executor._execute_single({"tool": "read_file", "args": {"path": "a.py"}})
        result = asyncio.run(_run())
        assert result.ok is True
        assert result.content == "done"
        assert result.execution_time >= 0.0

    def test_dispatch_exception_returns_error_result(self, mock_registry):
        mock_registry.dispatch.side_effect = RuntimeError("unexpected error")
        async def _run():
            executor = AsyncToolExecutor(mock_registry)
            return await executor._execute_single({"tool": "bash", "args": {"cmd": "ls"}})
        result = asyncio.run(_run())
        assert result.ok is False
        assert "unexpected error" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# execute_parallel
# ═══════════════════════════════════════════════════════════════════════════


class TestExecuteParallel:
    """execute_parallel orchestrates parallel execution."""
    def test_malformed_tool_call_caught_as_exception(self, mock_registry):
            """A tool_call missing required keys raises an exception caught by gather."""
            async def _run():
                executor = AsyncToolExecutor(mock_registry)
                return await executor.execute_parallel([{"invalid": "no_tool_key"}])
            results = asyncio.run(_run())
            assert len(results) == 1
            assert results[0].ok is False

    def test_empty_calls_returns_empty_list(self, mock_registry):
        async def _run():
            executor = AsyncToolExecutor(mock_registry)
            return await executor.execute_parallel([])
        results = asyncio.run(_run())
        assert results == []

    def test_single_tool(self, mock_registry):
        async def _run():
            executor = AsyncToolExecutor(mock_registry)
            return await executor.execute_parallel([{"tool": "bash", "args": {"cmd": "ls"}}])
        results = asyncio.run(_run())
        assert len(results) == 1
        assert results[0].ok is True

    def test_multiple_independent_tools(self, mock_registry_with_dep_graph):
        async def _run():
            executor = AsyncToolExecutor(mock_registry_with_dep_graph)
            calls = [
                {"tool": "bash", "args": {"cmd": "date"}},
                {"tool": "grep", "args": {"pattern": "foo"}},
            ]
            return await executor.execute_parallel(calls)
        results = asyncio.run(_run())
        assert len(results) == 2
        assert all(r.ok for r in results)

    def test_one_tool_fails_others_succeed(self, mock_registry_with_dep_graph):
        """A failing tool produces an error result but doesn't block others."""
        dispatch_side_effects = {
            "bash": ToolResult(ok=True, content="ok", execution_time=0.0),
            "grep": RuntimeError("grep failed"),
        }

        def dispatch_side_effect(tool, args):
            result = dispatch_side_effects.get(tool)
            if isinstance(result, Exception):
                raise result
            return result

        mock_registry_with_dep_graph.dispatch.side_effect = dispatch_side_effect
        async def _run():
            executor = AsyncToolExecutor(mock_registry_with_dep_graph)
            calls = [
                {"tool": "bash", "args": {}},
                {"tool": "grep", "args": {}},
            ]
            return await executor.execute_parallel(calls)
        results = asyncio.run(_run())
        assert len(results) == 2
        # bash succeeded
        bash_result = next(r for r in results if r.ok)
        assert bash_result.content == "ok"
        # grep failed
        grep_result = next(r for r in results if not r.ok)
        assert "grep failed" in grep_result.error


# ═══════════════════════════════════════════════════════════════════════════
# Shutdown
# ═══════════════════════════════════════════════════════════════════════════


class TestShutdown:
    """Coverage for shutdown()."""

    def test_shutdown_does_not_raise(self, mock_registry):
        executor = AsyncToolExecutor(mock_registry)
        executor.shutdown()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# ToolResult fallback dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestToolResultFallback:
    """Coverage for the fallback ToolResult definition (ImportError path)."""

    def test_default_construction(self):
        tr = ToolResult(ok=True, content="hello")
        assert tr.ok is True
        assert tr.content == "hello"
        assert tr.error is None
        assert tr.execution_time == 0.0
        assert tr.retryable is True

    def test_full_construction(self):
        tr = ToolResult(ok=False, content="err", error="something broke",
                        execution_time=1.5, retryable=False, retry_count=2)
        assert tr.ok is False
        assert tr.error == "something broke"
        assert tr.execution_time == 1.5
        assert tr.retryable is False
        assert tr.retry_count == 2
