"""Regression tests for ``ToolDependencyGraph.has_path``.

These tests exercise the REAL ``ToolDependencyGraph`` (not a MagicMock).
Background: ``AsyncToolExecutor._has_dependency`` calls
``self.dependency_graph.has_path(tool2, tool1)``, but ``ToolDependencyGraph``
previously had NO public ``has_path`` method (only a private ``_has_path``
that handled the dict backend). With ``parallel_tool_execution_enabled``
defaulting to True, every multi-read tool batch hit
``AttributeError: 'ToolDependencyGraph' object has no attribute 'has_path'``
inside ``_group_by_dependencies``. The dispatcher's broad ``except Exception``
silently swallowed it and fell back to the thread pool, so the async parallel
executor NEVER worked and logged a full traceback on every parallel batch.

The pre-existing tests in ``test_async_tool_executor.py`` masked the bug
because their ``mock_registry_with_dep_graph`` fixture uses a ``MagicMock``
dependency graph, for which ``has_path`` is auto-created. These tests use the
real object so the missing method would raise ``AttributeError`` if reintroduced.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from external_llm.agent.async_tool_executor import AsyncToolExecutor, ToolResult
from external_llm.agent.tool_dependency_graph import ToolDependencyGraph


# ═══════════════════════════════════════════════════════════════════════════
# ToolDependencyGraph.has_path
# ═══════════════════════════════════════════════════════════════════════════


class TestHasPath:
    """Public ``has_path`` works for both networkx and dict backends."""

    @pytest.fixture
    def graph(self):
        # Initial edges from _build_initial_graph:
        #   find_symbol -> find_references -> apply_patch -> run_lint
        #   write_plan -> run_lint
        return ToolDependencyGraph()

    def test_direct_edge_returns_true(self, graph):
        assert graph.has_path("find_symbol", "find_references") is True

    def test_transitive_path_returns_true(self, graph):
        # find_symbol -> find_references -> apply_patch
        assert graph.has_path("find_symbol", "apply_patch") is True

    def test_reverse_direction_returns_false(self, graph):
        assert graph.has_path("apply_patch", "find_symbol") is False

    def test_absent_source_returns_false_no_crash(self, graph):
        # The critical regression: missing node must NOT raise.
        assert graph.has_path("grep", "find_symbol") is False

    def test_absent_target_returns_false_no_crash(self, graph):
        assert graph.has_path("find_symbol", "grep") is False

    def test_both_absent_returns_false_no_crash(self, graph):
        assert graph.has_path("read_file", "bash") is False

    def test_self_loop_unrelated_tool_returns_false(self, graph):
        assert graph.has_path("grep", "grep") is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration: AsyncToolExecutor with a REAL ToolDependencyGraph
# ═══════════════════════════════════════════════════════════════════════════


def _registry_with_real_graph():
    """Registry whose dependency_graph is a REAL ToolDependencyGraph.

    This is what production uses (tool_registry.py instantiates the real class
    when parallel_tool_execution_enabled is True, which is the default).
    """
    registry = MagicMock()
    registry.dispatch.return_value = ToolResult(ok=True, content="done", execution_time=0.0)
    registry.dependency_graph = ToolDependencyGraph()
    return registry


class TestHasDependencyWithRealGraph:
    """``_has_dependency`` must not raise AttributeError on the real graph."""

    def test_real_graph_dependency_check_no_crash(self):
        executor = AsyncToolExecutor(_registry_with_real_graph())
        # These tools are NOT in the dependency graph — must return False, not raise.
        assert executor._has_dependency("grep", "read_file") is False
        assert executor._has_dependency("read_file", "grep") is False

    def test_real_graph_known_dependency(self):
        executor = AsyncToolExecutor(_registry_with_real_graph())
        # find_references depends on find_symbol (edge find_symbol->find_references).
        # _has_dependency(tool1, tool2) checks has_path(tool2, tool1):
        #   has_path("find_symbol", "find_references") = True
        #   => _has_dependency("find_references", "find_symbol") = True
        assert executor._has_dependency("find_references", "find_symbol") is True


class TestGroupByDependenciesWithRealGraph:
    """``_group_by_dependencies`` must complete without AttributeError."""

    def test_read_only_tools_one_layer(self):
        """Independent read-only tools (not in graph) form a single parallel layer."""
        executor = AsyncToolExecutor(_registry_with_real_graph())
        tool_calls = [
            {"tool": "grep", "args": {}},
            {"tool": "read_file", "args": {}},
            {"tool": "bash", "args": {"command": "ls"}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_mixed_known_and_unknown_tools_no_crash(self):
        """Mixing graph tools (find_symbol) with unknown tools (grep) must not raise."""
        executor = AsyncToolExecutor(_registry_with_real_graph())
        tool_calls = [
            {"tool": "find_symbol", "args": {}},
            {"tool": "grep", "args": {}},
            {"tool": "find_references", "args": {}},
        ]
        groups = executor._group_by_dependencies(tool_calls)
        # All tool indices must be covered exactly once.
        flat = [i for layer in groups for i in layer]
        assert sorted(flat) == [0, 1, 2]
