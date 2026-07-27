"""
Tool Dependency Graph for asicode Agent

Manages dependencies between tools for optimal chaining and parallel execution analysis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ToolDependencyGraph:
    """Manages tool dependency graph for optimal chaining and parallel execution."""

    def __init__(self):
        self.graph = self._create_empty_graph()
        self._build_initial_graph()

    def _create_empty_graph(self):
        """Create an empty adjacency-list graph."""
        return {"nodes": set(), "edges": {}}

    def _build_initial_graph(self):
        """Build initial tool dependency graph based on common workflows."""
        # Common tool sequences observed in agent sessions
        dependencies = [
            ("find_symbol", "find_references"),
            ("find_references", "apply_patch"),
            ("apply_patch", "run_lint"),
            ("write_plan", "run_lint"),
        ]

        for from_tool, to_tool in dependencies:
            self.add_dependency(from_tool, to_tool)

    def add_dependency(self, from_tool: str, to_tool: str, weight: float = 1.0):
        """Add a dependency edge from from_tool to to_tool."""
        if from_tool not in self.graph["edges"]:
            self.graph["edges"][from_tool] = {}
        self.graph["edges"][from_tool][to_tool] = weight
        self.graph["nodes"].add(from_tool)
        self.graph["nodes"].add(to_tool)

    def get_optimal_chain(self, start_tool: str, target_outcome: str) -> list[str]:
        """Calculate optimal tool chain from start_tool to target_outcome.

        Uses BFS to find the shortest path (minimum edge count).
        """
        return self._bfs_path(start_tool, target_outcome)

    def _bfs_path(self, start: str, target: str) -> list[str]:
        """Breadth-first search for path in simple adjacency list."""
        if start not in self.graph["edges"]:
            return []

        from collections import deque
        queue = deque([(start, [start])])
        visited = set([start])

        while queue:
            node, path = queue.popleft()
            if node == target:
                return path

            for neighbor in self.graph["edges"].get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))

        return []

    def _has_path(self, start: str, target: str) -> bool:
        """Check if there is a directed path from start to target."""
        if start not in self.graph["edges"]:
            return False

        from collections import deque
        queue = deque([start])
        visited = set([start])

        while queue:
            node = queue.popleft()
            if node == target:
                return True

            for neighbor in self.graph["edges"].get(node, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def has_path(self, source: str, target: str) -> bool:
        """Return True if a directed path exists from ``source`` to ``target``.

        Tools absent from the graph return False (no path). This is the public
        entry point used by the parallel executor's dependency check
        (``AsyncToolExecutor._has_dependency``) to decide whether two tools have
        an ordering relationship.
        """
        return self._has_path(source, target)

    def get_dependent_tools(self, tool: str) -> list[str]:
        """Get all tools that directly depend on the given tool."""
        return list(self.graph["edges"].get(tool, {}).keys())

    def record_transition(self, from_tool: str, to_tool: str, increment: float = 1.0):
        """Record a transition from from_tool to to_tool, increasing edge weight."""
        if from_tool not in self.graph["edges"]:
            self.graph["edges"][from_tool] = {}
        current_weight = self.graph["edges"][from_tool].get(to_tool, 0.0)
        self.graph["edges"][from_tool][to_tool] = current_weight + increment
        self.graph["nodes"].add(from_tool)
        self.graph["nodes"].add(to_tool)
