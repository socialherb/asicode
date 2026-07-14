"""
Graph builder for constructing repository symbol graphs.
"""

from __future__ import annotations

from typing import Optional

from .repository_graph import RepositoryGraph


class GraphBuilder:
    """Builds repository symbol graphs."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def build_repo_graph(self, repo_root: Optional[str] = None) -> RepositoryGraph:
        """
        Build a RepositoryGraph for the given repository root.

        Args:
            repo_root: Optional override; uses self.repo_root if not provided.

        Returns:
            RepositoryGraph instance with populated symbols, calls, and imports.
        """
        root = repo_root or self.repo_root
        graph = RepositoryGraph(root)
        graph.build()
        return graph
