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
        # P3 Stage 3 (2026-08-12): agent-side RG builds write the on-disk
        # snapshot (collect_imported_names=True) — the release-gate snapshot
        # then self-heals on repos where no gate ever runs (third-party repos
        # using the agent), which is the precondition for CallGraphIndexer
        # having no disk tier of its own.  Warm/served builds pay ≈0 for the
        # name pass (_imported_names_for serves from cache/disk); the snapshot
        # rewrite itself is hint-gated (re-parse or count drift only).
        graph.build(collect_imported_names=True)
        return graph
