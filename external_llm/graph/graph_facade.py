"""
RepositoryGraphFacade — canonical entry point for all graph queries.

All callers should import and use this facade instead of accessing
RepositoryGraph or CallGraphIndexer directly.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .graph_builder import GraphBuilder
from .models import CallEdge, ImportEdge, SymbolNode
from .repository_graph import RepositoryGraph


class RepositoryGraphFacade:
    def __init__(self, call_graph_indexer=None, test_finder=None, repo_root: Optional[str] = None):
        self.call_graph_indexer = call_graph_indexer
        self.test_finder = test_finder
        self.repo_root = repo_root or os.getcwd()
        self._graph: Optional[RepositoryGraph] = None
        self._graph_builder = GraphBuilder(self.repo_root)
        # RLock (re-entrant) so the same thread can call _ensure_graph() recursively
        # without deadlock, while still blocking other threads during rebuild.
        # Invariant: _graph is either None (needs rebuild) or a fully-built
        # RepositoryGraph — never a partially-built object visible to callers.
        self._rebuild_lock = threading.RLock()

    def get_related_symbols(
        self,
        symbol: str,
        file_path: Optional[str] = None,
        depth: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if self.call_graph_indexer is None:
            return []
        return self.call_graph_indexer.get_related_symbols(
            symbol=symbol,
            file_path=file_path,
            depth=depth,
            limit=limit,
        )

    def get_test_targets(self, symbol: str) -> list[str]:
        if self.test_finder is None:
            return []
        try:
            return self.test_finder.find_tests_for_symbol(symbol)
        except Exception:
            return []

    # ── P1: Global Symbol Graph extensions ──────────────────────────────────

    def invalidate(self) -> None:
        """Mark the graph as stale so it rebuilds on the next access.

        Called whenever files are written (apply_patch, write_file, etc.) so that
        subsequent find_symbol / get_callers calls reflect the updated source.

        Thread-safe: holds _rebuild_lock so a concurrent _ensure_graph() build
        doesn't silently repopulate _graph with stale data after invalidation.
        """
        with self._rebuild_lock:
            self._graph = None

    def invalidate_files(self, changed_paths: list[str]) -> None:
        """Incrementally re-parse only the changed files instead of full rebuild.

        Args:
            changed_paths: List of relative paths that were modified.
        """
        with self._rebuild_lock:
            graph = self._graph
            if graph is None:
                return

            from ..languages import LanguageId
            lang_paths = [p for p in changed_paths if LanguageId.from_path(p.strip()) != LanguageId.UNKNOWN]
            if not lang_paths:
                return

            for rel_path in lang_paths:
                norm = rel_path.strip().lstrip("/")
                abs_path = os.path.join(self.repo_root, norm)

                if os.path.isfile(abs_path):
                    graph.reparse_file(abs_path)
                else:
                    graph.remove_file(norm)

        logger.debug("GSG incremental update: %d files reparsed", len(lang_paths))

    def _ensure_graph(self) -> RepositoryGraph:
        """Lazy-load (or rebuild if invalidated) the repository graph.

        Thread-safe: holds _rebuild_lock during build so concurrent readers
        never observe a partially-constructed graph.  The check-then-build
        pattern inside the lock prevents redundant rebuilds when multiple
        threads race on a None graph simultaneously.
        """
        # Fast path: graph already built — no lock needed for read.
        # Python's GIL makes this single attribute read atomic.
        if self._graph is not None:
            return self._graph

        with self._rebuild_lock:
            # Re-check inside lock: another thread may have built it while
            # we waited to acquire.
            if self._graph is None:
                self._graph = self._graph_builder.build_repo_graph(self.repo_root)
        return self._graph

    def get_symbol(self, name: str, file_path: Optional[str] = None) -> Optional[SymbolNode]:
        """Retrieve a symbol by name, optionally scoped to a file."""
        graph = self._ensure_graph()
        return graph.get_symbol(name, file_path)

    def get_callers(self, symbol_name: str, file_path: Optional[str] = None) -> list[CallEdge]:
        """Return all call edges where the given symbol is the callee (canonical CallEdge)."""
        if self.call_graph_indexer is not None:
            return self.call_graph_indexer.get_callers(symbol_name, file_path)
        graph = self._ensure_graph()
        return self._convert_repo_edges(graph.get_callers(symbol_name), graph)

    def get_callees(self, symbol_name: str, file_path: Optional[str] = None) -> list[CallEdge]:
        """Return all call edges where the given symbol is the caller (canonical CallEdge)."""
        if self.call_graph_indexer is not None:
            return self.call_graph_indexer.get_callees(symbol_name, file_path)
        graph = self._ensure_graph()
        return self._convert_repo_edges(graph.get_callees(symbol_name), graph)

    @staticmethod
    def _convert_repo_edges(edges: list, graph: RepositoryGraph = None) -> list[CallEdge]:
        """Convert RepositoryGraph internal CallEdge list to canonical CallEdge list.

        Direction-agnostic: both get_callers() and get_callees() return RepositoryGraph
        edges whose ``file_path`` is always the *caller's* file, so the callee_file is
        resolved from the symbol table identically for either direction.
        """
        result = []
        for e in edges:
            # Resolve callee_file from symbol table (e.file_path is the CALLER's file)
            _callee_node = graph.get_symbol(e.callee) if graph else None
            _callee_file = _callee_node.file_path if _callee_node else e.file_path
            result.append(CallEdge(
                caller_symbol=e.caller,
                caller_file=e.file_path,
                caller_line=e.line,
                callee_symbol=e.callee,
                callee_display=e.callee,
                callee_file=_callee_file,
                call_args=getattr(e, "call_args", []),
                is_mutating=getattr(e, "is_mutating", False),
            ))
        return result

    def get_file_dependencies(self, file_path: str) -> list[ImportEdge]:
        """Return all import edges where the given file is the importer."""
        graph = self._ensure_graph()
        return graph.get_file_dependencies(file_path)

    def get_importers(self, file_path: str) -> list[str]:
        """Return file paths that import the given file (reverse dependency lookup).

        Delegates to ``RepositoryGraph.get_importers()``.  Returns an empty list
        when the graph is unavailable or the file has no known importers.
        """
        try:
            graph = self._ensure_graph()
            return graph.get_importers(file_path)
        except Exception:
            return []

    def get_symbols_in_file(self, file_path: str) -> list[SymbolNode]:
        """Return all symbols defined in the given file."""
        graph = self._ensure_graph()
        return graph.get_symbols_in_file(file_path)

    def get_symbol_file(self, symbol_name: str) -> Optional[str]:
        """
        Get the file path where a symbol is defined.

        Args:
            symbol_name: Symbol name (qualified or simple).

        Returns:
            File path if symbol found, None otherwise.
        """
        graph = self._ensure_graph()
        node = graph.get_symbol(symbol_name)
        return node.file_path if node else None
