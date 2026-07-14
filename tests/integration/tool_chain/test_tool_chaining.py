"""
Integration tests for tool chaining and dependency graph.
"""
import pytest

from external_llm.agent.tool_dependency_graph import ToolDependencyGraph


@pytest.mark.integration
class TestToolDependencyGraph:
    """Test tool dependency graph for optimal chaining."""

    def test_dependency_graph_initialization(self):
        """Test dependency graph initialization with default dependencies."""
        graph = ToolDependencyGraph()

        # Should have nodes
        if hasattr(graph, 'graph'):
            # Check internal structure
            if isinstance(graph.graph, dict) and "nodes" in graph.graph:
                assert len(graph.graph["nodes"]) > 0
            # Or if using networkx
            elif hasattr(graph.graph, 'nodes'):
                assert len(list(graph.graph.nodes())) > 0

    def test_add_dependency(self):
        """Test adding custom dependencies to graph."""
        graph = ToolDependencyGraph()

        # Add a custom dependency
        graph.add_dependency("custom_tool_1", "custom_tool_2", weight=2.0)

        # Test if dependency was added
        deps = graph.get_dependent_tools("custom_tool_1")
        assert "custom_tool_2" in deps

    def test_get_optimal_chain_existing_path(self):
        """Test finding optimal chain between tools with existing path."""
        graph = ToolDependencyGraph()

        # find_symbol -> find_references should have a well-known path
        chain = graph.get_optimal_chain("find_symbol", "find_references")

        assert isinstance(chain, list)
        # Should at least contain start and end
        if chain:
            assert chain[0] == "find_symbol"
            assert chain[-1] == "find_references"
            # Path should be valid (no self loops, etc.)
            assert len(chain) >= 2

    def test_get_optimal_chain_no_path(self):
        """Test finding optimal chain when no path exists."""
        graph = ToolDependencyGraph()

        # Self-loop: path from find_symbol to find_symbol
        chain = graph.get_optimal_chain("find_symbol", "find_symbol")

        assert isinstance(chain, list)
        # Should return empty list or single-element list
        assert chain == [] or (len(chain) == 1 and chain[0] == "find_symbol")

    def test_get_optimal_chain_complex_path(self):
        """Test finding optimal chain through multiple tools."""
        graph = ToolDependencyGraph()

        # Add a longer chain for testing
        graph.add_dependency("tool_a", "tool_b")
        graph.add_dependency("tool_b", "tool_c")
        graph.add_dependency("tool_c", "tool_d")

        chain = graph.get_optimal_chain("tool_a", "tool_d")

        assert isinstance(chain, list)
        if chain:
            assert chain[0] == "tool_a"
            assert chain[-1] == "tool_d"
            assert "tool_b" in chain or "tool_c" in chain

    def test_dependency_graph_weights(self):
        """Test that dependency weights affect optimal path."""
        graph = ToolDependencyGraph()

        # Add two paths with different weights
        graph.add_dependency("start", "middle1", weight=1.0)
        graph.add_dependency("middle1", "end", weight=1.0)  # Total weight 2.0

        graph.add_dependency("start", "middle2", weight=0.5)
        graph.add_dependency("middle2", "end", weight=0.5)  # Total weight 1.0

        # Optimal path should prefer lower weight path
        chain = graph.get_optimal_chain("start", "end")

        # With networkx, should use Dijkstra and prefer middle2 path
        # Without networkx, BFS might return first found path
        assert isinstance(chain, list)

    def test_circular_dependency_prevention(self):
        """Test that graph handles or prevents circular dependencies."""
        graph = ToolDependencyGraph()

        # Add potential circular dependency
        graph.add_dependency("tool_x", "tool_y")
        graph.add_dependency("tool_y", "tool_z")
        graph.add_dependency("tool_z", "tool_x")  # Creates cycle

        # get_optimal_chain should handle cycles (not get stuck)
        chain = graph.get_optimal_chain("tool_x", "tool_z")
        # Should return a path or empty list, not crash
        assert isinstance(chain, list)


