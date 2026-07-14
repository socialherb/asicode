"""
Graph-based dependency traversal for impact simulation.

Provides safe traversal functions to collect callers, callees, and dependencies
using the RepositoryGraphFacade.
"""

from __future__ import annotations

from collections import deque

from external_llm.graph.graph_facade import RepositoryGraphFacade
from external_llm.graph.repository_graph import CallEdge, ImportEdge


def collect_direct_callers(
    graph_facade: RepositoryGraphFacade, symbol: str
) -> list[str]:
    """
    Collect direct callers of a symbol.

    Args:
        graph_facade: RepositoryGraphFacade instance.
        symbol: Target symbol name (qualified or simple).

    Returns:
        List of caller symbol names. Empty list if graph not available or error.
    """
    try:
        callers: list[CallEdge] = graph_facade.get_callers(symbol)
        result: set[str] = set()
        for edge in callers:
            if edge.caller:
                result.add(edge.caller)
        return list(result)
    except Exception:
        return []


def collect_indirect_callers(
    graph_facade: RepositoryGraphFacade,
    symbol: str,
    depth: int = 2,
    max_nodes: int = 100,
) -> list[str]:
    """
    Collect indirect callers up to given depth via BFS.

    Args:
        graph_facade: RepositoryGraphFacade instance.
        symbol: Starting symbol.
        depth: Maximum call chain depth.
        max_nodes: Maximum number of nodes to collect.

    Returns:
        List of indirect caller symbol names (excluding the starting symbol).
    """
    if depth <= 0:
        return []

    try:
        visited: set[str] = set()
        queue = deque([(symbol, 0)])  # (symbol, current_depth)
        indirect_callers: set[str] = set()

        while queue and len(visited) < max_nodes:
            current_sym, current_depth = queue.popleft()
            if current_sym in visited:
                continue
            visited.add(current_sym)

            if current_depth >= depth:
                continue

            callers = collect_direct_callers(graph_facade, current_sym)
            for caller in callers:
                if caller == symbol:
                    continue

                next_depth = current_depth + 1

                if caller not in visited:
                    queue.append((caller, next_depth))

                    # Only depth >= 2 should be considered indirect.
                    if next_depth >= 2:
                        indirect_callers.add(caller)

        return list(indirect_callers)
    except Exception:
        return []


def collect_file_dependencies(
    graph_facade: RepositoryGraphFacade, file_path: str
) -> list[str]:
    """
    Collect import dependencies of a file.

    Args:
        graph_facade: RepositoryGraphFacade instance.
        file_path: Relative file path (as stored in graph).

    Returns:
        List of imported module or symbol names as recorded in ImportEdge.imported.
        These are not guaranteed to be resolved repo-relative file paths.
    """
    try:
        imports: list[ImportEdge] = graph_facade.get_file_dependencies(file_path)
        result: set[str] = set()
        for imp in imports:
            if imp.imported:
                result.add(imp.imported)
        return list(result)
    except Exception:
        return []


def collect_impacted_files(
    graph_facade: RepositoryGraphFacade,
    symbols: list[str],
    file_paths: list[str],
) -> list[str]:
    """
    Collect all files impacted by changes to given symbols and files.

    Includes:
    - Files containing target symbols
    - Files containing direct callers
    - The target files themselves

    Args:
        graph_facade: RepositoryGraphFacade instance.
        symbols: List of target symbol names.
        file_paths: List of target file paths.

    Returns:
        List of impacted file paths (deduplicated).
    """
    impacted: set[str] = set()

    # Add target files
    impacted.update(file_paths)

    # For each symbol, find its file and add it
    for sym in symbols:
        try:
            node = graph_facade.get_symbol(sym)
            if node and node.file_path:
                impacted.add(node.file_path)
        except Exception:
            pass

    # Add files of direct callers
    for sym in symbols:
        callers = collect_direct_callers(graph_facade, sym)
        for caller in callers:
            try:
                node = graph_facade.get_symbol(caller)
                if node and node.file_path:
                    impacted.add(node.file_path)
            except Exception:
                pass

    # Note:
    # collect_file_dependencies() currently returns imported module/symbol names,
    # not resolved repo-relative file paths. Those values must not be mixed into
    # impacted_files, which should contain only concrete file paths.
    #
    # If import-to-file resolution is added in the future, resolved internal
    # repository file paths can be appended here safely.
    return list(impacted)
