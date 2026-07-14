"""
Tool Dependency Graph for asicode Agent

Manages dependencies between tools for optimal chaining and parallel execution analysis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy networkx binding: importing networkx eagerly costs ~130ms and pulls in
# its entire submodule tree (algorithms/drawing/linalg/...). This module is
# imported unconditionally at top of tool_registry.py, but ToolDependencyGraph
# is only instantiated when parallel_tool_execution_enabled is on — so the
# eager import taxed every cold start (incl. subagent process spawns) for a
# feature most runs never use. _nx is bound on first access via _load_nx().
_nx = None  # type: ignore[assignment]
_HAS_NETWORKX_PROBED = False


def _load_nx():
    """Idempotently import networkx into module-global _nx. Returns the module
    or None (with a one-shot warning) when networkx is unavailable."""
    global _nx, _HAS_NETWORKX_PROBED
    if not _HAS_NETWORKX_PROBED:
        _HAS_NETWORKX_PROBED = True
        try:
            import networkx as nx  # type: ignore[import-untyped]
            _nx = nx
        except ImportError:
            logger.warning("networkx not installed, using simplified dependency graph")
            _nx = None
    return _nx


class ToolDependencyGraph:
    """Manages tool dependency graph for optimal chaining and parallel execution."""

    def __init__(self):
        self.graph = self._create_empty_graph()
        self._build_initial_graph()

    def _create_empty_graph(self):
        """Create an empty graph, using networkx if available."""
        if _load_nx() is not None:
            return _nx.DiGraph()
        else:
            # Simple adjacency list representation
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
        if _load_nx() is not None:
            self.graph.add_edge(from_tool, to_tool, weight=weight)
        else:
            if from_tool not in self.graph["edges"]:
                self.graph["edges"][from_tool] = {}
            self.graph["edges"][from_tool][to_tool] = weight
            self.graph["nodes"].add(from_tool)
            self.graph["nodes"].add(to_tool)

    def get_optimal_chain(self, start_tool: str, target_outcome: str) -> list[str]:
        """Calculate optimal tool chain from start_tool to target_outcome."""
        if _load_nx() is not None:
            try:
                # Use Dijkstra's algorithm to find shortest weighted path
                path = _nx.shortest_path(self.graph, start_tool, target_outcome, weight='weight')
                return path
            except (_nx.NetworkXNoPath, _nx.NodeNotFound):
                return []
        else:
            # Simplified BFS search without weights
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
        """Check if there's a path from start to target in simple graph."""
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
            if _load_nx() is not None:
                try:
                    return _nx.has_path(self.graph, source, target)
                except _nx.NodeNotFound:
                    return False
            return self._has_path(source, target)
    def get_dependent_tools(self, tool: str) -> list[str]:
        """Get all tools that directly depend on the given tool."""
        if _load_nx() is not None:
            return list(self.graph.successors(tool))
        else:
            return list(self.graph["edges"].get(tool, {}).keys())

    def record_transition(self, from_tool: str, to_tool: str, increment: float = 1.0):
        """Record a transition from from_tool to to_tool, increasing edge weight."""
        if _load_nx() is not None:
            if self.graph.has_edge(from_tool, to_tool):
                # Increase existing edge weight
                current_weight = self.graph[from_tool][to_tool].get('weight', 1.0)
                self.graph[from_tool][to_tool]['weight'] = current_weight + increment
            else:
                # Add new edge with initial weight
                self.graph.add_edge(from_tool, to_tool, weight=increment)
        else:
            if from_tool not in self.graph["edges"]:
                self.graph["edges"][from_tool] = {}
            current_weight = self.graph["edges"][from_tool].get(to_tool, 0.0)
            self.graph["edges"][from_tool][to_tool] = current_weight + increment
            self.graph["nodes"].add(from_tool)
            self.graph["nodes"].add(to_tool)
