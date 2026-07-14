"""
Impact propagation engine: computes the set of symbols, files, and modules
affected by a code change, with depth/relation tracking.

Wraps existing dependency_traversal functions and adds typed metadata.
Does NOT reimplement BFS — delegates to collect_direct_callers,
collect_indirect_callers, collect_impacted_files.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.languages import LanguageId

logger = logging.getLogger(__name__)

# Default propagation limits
DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_NODES = 50
STRUCTURAL_OP_MAX_DEPTH = 3
STRUCTURAL_OP_KINDS = {
    "RENAME_SYMBOL", "MOVE_SYMBOL", "MODIFY_SYMBOL",
    "replace_symbol_body", "INSERT_AFTER_SYMBOL",
    "UPDATE_CALLERS", "refactor",
}


@dataclass
class PropagatedNode:
    """A single node in the impact propagation graph."""
    symbol: Optional[str] = None
    file_path: Optional[str] = None
    module: Optional[str] = None
    depth: int = 0
    relation: str = "origin"  # "origin" | "caller" | "same_file" | "same_module" | "import_dep"
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "file_path": self.file_path,
            "module": self.module,
            "depth": self.depth,
            "relation": self.relation,
            "reason_codes": self.reason_codes[:3],
        }


@dataclass
class ImpactSet:
    """Complete impact propagation result."""
    origin_symbols: list[str] = field(default_factory=list)
    origin_files: list[str] = field(default_factory=list)
    impacted_symbols: list[str] = field(default_factory=list)
    impacted_files: list[str] = field(default_factory=list)
    impacted_modules: list[str] = field(default_factory=list)
    propagated_nodes: list[PropagatedNode] = field(default_factory=list)
    max_depth_reached: int = 0
    truncated: bool = False

    def to_summary(self) -> dict[str, Any]:
        """Concise summary for metadata."""
        return {
            "origin_symbols": self.origin_symbols[:5],
            "origin_files": self.origin_files[:5],
            "impacted_symbol_count": len(self.impacted_symbols),
            "impacted_file_count": len(self.impacted_files),
            "impacted_module_count": len(self.impacted_modules),
            "max_depth_reached": self.max_depth_reached,
            "truncated": self.truncated,
            "propagated_node_count": len(self.propagated_nodes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_symbols": self.origin_symbols,
            "origin_files": self.origin_files,
            "impacted_symbols": self.impacted_symbols,
            "impacted_files": self.impacted_files,
            "impacted_modules": self.impacted_modules,
            "max_depth_reached": self.max_depth_reached,
            "truncated": self.truncated,
            "propagated_nodes": [n.to_dict() for n in self.propagated_nodes[:20]],
        }


class ImpactPropagationEngine:
    """
    Computes impact set from changed symbols/files using graph traversal.

    Wraps existing dependency_traversal functions, adding typed metadata
    with depth, relation, and reason tracking.

    Never raises — returns minimal ImpactSet on any error.
    """

    def __init__(self, graph_facade=None):
        self._facade = graph_facade

    def propagate(
        self,
        changed_symbols: Optional[list[str]] = None,
        changed_files: Optional[list[str]] = None,
        operation_kind: Optional[str] = None,
        max_depth: Optional[int] = None,
        max_nodes: int = DEFAULT_MAX_NODES,
        include_same_file: bool = True,
        include_same_module: bool = True,
    ) -> ImpactSet:
        """
        Compute impact set from changed symbols and files.

        Propagation order:
        1. Origin nodes (changed symbols/files)
        2. Caller propagation (BFS up call graph)
        3. Same-file sibling symbols
        4. Same-module file adjacency

        Args:
            changed_symbols: Symbols that were modified
            changed_files: Files that were modified
            operation_kind: Type of operation (affects default depth)
            max_depth: Max caller chain depth (default: 2, structural: 3)
            max_nodes: Max total propagated nodes
            include_same_file: Include sibling symbols in same file
            include_same_module: Include files in same module/package
        """
        impact = ImpactSet()
        symbols = list(changed_symbols or [])
        files = list(changed_files or [])

        impact.origin_symbols = symbols[:]
        impact.origin_files = files[:]

        # Determine max_depth from operation kind if not specified
        if max_depth is None:
            if operation_kind and operation_kind in STRUCTURAL_OP_KINDS:
                max_depth = STRUCTURAL_OP_MAX_DEPTH
            else:
                max_depth = DEFAULT_MAX_DEPTH

        seen_symbols: set[str] = set()
        seen_files: set[str] = set(files)
        seen_modules: set[str] = set()
        nodes: list[PropagatedNode] = []
        actual_max_depth = 0

        try:
            # 1. Origin nodes
            for sym in symbols:
                if sym in seen_symbols:
                    continue
                nodes.append(PropagatedNode(
                    symbol=sym, depth=0, relation="origin",
                    reason_codes=["CHANGED_SYMBOL"],
                ))
                seen_symbols.add(sym)

                # Resolve symbol file if not already in files
                if self._facade:
                    try:
                        node = self._facade.get_symbol(sym)
                        if node and hasattr(node, 'file_path') and node.file_path:
                            seen_files.add(node.file_path)
                            if hasattr(node, 'module') and node.module:
                                seen_modules.add(node.module)
                    except Exception:
                        pass

            for f in files:
                nodes.append(PropagatedNode(
                    file_path=f, depth=0, relation="origin",
                    reason_codes=["CHANGED_FILE"],
                ))
                mod = self._file_to_module(f)
                if mod:
                    seen_modules.add(mod)

            # 2. Caller propagation (using existing dependency_traversal)
            if self._facade and symbols:
                caller_nodes, caller_depth = self._propagate_callers(
                    symbols, max_depth, max_nodes - len(nodes),
                    seen_symbols, seen_files, seen_modules,
                )
                # Sync seen_* sets from returned nodes (supports mocking in tests)
                for cn in caller_nodes:
                    if cn.symbol:
                        seen_symbols.add(cn.symbol)
                    if cn.file_path:
                        seen_files.add(cn.file_path)
                    if cn.module:
                        seen_modules.add(cn.module)
                nodes.extend(caller_nodes)
                actual_max_depth = max(actual_max_depth, caller_depth)

                if len(nodes) >= max_nodes:
                    impact.truncated = True

            # 3. Same-file sibling symbols
            if include_same_file and self._facade and len(nodes) < max_nodes:
                sibling_nodes = self._propagate_same_file(
                    seen_files, seen_symbols, max_nodes - len(nodes),
                )
                nodes.extend(sibling_nodes)

            # 4. Same-module file adjacency
            if include_same_module and len(nodes) < max_nodes:
                module_nodes = self._propagate_same_module(
                    seen_files, seen_modules, max_nodes - len(nodes),
                )
                nodes.extend(module_nodes)

        except Exception as e:
            logger.debug("Impact propagation failed: %s", e)
            # Return best-effort result with what we have

        # Build final impact set
        impact.propagated_nodes = nodes[:max_nodes]
        impact.max_depth_reached = actual_max_depth
        if len(nodes) > max_nodes:
            impact.truncated = True

        # Deduplicate into lists
        impact.impacted_symbols = list(seen_symbols)
        impact.impacted_files = list(seen_files)
        impact.impacted_modules = list(seen_modules)

        return impact

    def _propagate_callers(
        self,
        symbols: list[str],
        max_depth: int,
        budget: int,
        seen_symbols: set[str],
        seen_files: set[str],
        seen_modules: set[str],
    ) -> tuple:
        """
        BFS caller propagation using existing dependency_traversal functions.
        Returns (nodes, max_depth_reached).
        """
        nodes: list[PropagatedNode] = []
        max_depth_reached = 0

        try:
            from external_llm.editor.simulator.dependency_traversal import (
                collect_direct_callers,
                collect_indirect_callers,
            )

            for sym in symbols:
                if len(nodes) >= budget:
                    break

                # Direct callers (depth=1)
                direct = collect_direct_callers(self._facade, sym)
                for caller in direct:
                    if caller in seen_symbols or len(nodes) >= budget:
                        continue
                    seen_symbols.add(caller)
                    max_depth_reached = max(max_depth_reached, 1)

                    # Resolve caller file
                    caller_file = None
                    caller_module = None
                    try:
                        caller_node = self._facade.get_symbol(caller)
                        if caller_node:
                            caller_file = getattr(caller_node, 'file_path', None)
                            caller_module = getattr(caller_node, 'module', None)
                            if caller_file:
                                seen_files.add(caller_file)
                            if caller_module:
                                seen_modules.add(caller_module)
                    except Exception:
                        pass

                    nodes.append(PropagatedNode(
                        symbol=caller, file_path=caller_file, module=caller_module,
                        depth=1, relation="caller",
                        reason_codes=[f"DIRECT_CALLER_OF_{sym}"],
                    ))

                # Indirect callers (depth 2+)
                if max_depth >= 2:
                    indirect = collect_indirect_callers(
                        self._facade, sym,
                        depth=max_depth,
                        max_nodes=budget - len(nodes),
                    )
                    for caller in indirect:
                        if caller in seen_symbols or len(nodes) >= budget:
                            continue
                        seen_symbols.add(caller)
                        # indirect callers are depth >= 2
                        depth = 2  # approximate; exact depth not tracked by collect_indirect_callers
                        max_depth_reached = max(max_depth_reached, depth)

                        caller_file = None
                        caller_module = None
                        try:
                            caller_node = self._facade.get_symbol(caller)
                            if caller_node:
                                caller_file = getattr(caller_node, 'file_path', None)
                                caller_module = getattr(caller_node, 'module', None)
                                if caller_file:
                                    seen_files.add(caller_file)
                                if caller_module:
                                    seen_modules.add(caller_module)
                        except Exception:
                            pass

                        nodes.append(PropagatedNode(
                            symbol=caller, file_path=caller_file, module=caller_module,
                            depth=depth, relation="caller",
                            reason_codes=[f"INDIRECT_CALLER_OF_{sym}"],
                        ))

        except Exception as e:
            logger.debug("Caller propagation failed: %s", e)

        return nodes, max_depth_reached

    def _propagate_same_file(
        self,
        seen_files: set[str],
        seen_symbols: set[str],
        budget: int,
    ) -> list[PropagatedNode]:
        """Add sibling symbols from files already in the impact set."""
        nodes: list[PropagatedNode] = []

        try:
            for f in list(seen_files):  # copy to avoid mutation during iteration
                if len(nodes) >= budget:
                    break
                try:
                    file_symbols = self._facade.get_symbols_in_file(f)
                    for sym_node in file_symbols:
                        name = getattr(sym_node, 'name', None) or getattr(sym_node, 'qualname', None)
                        if not name or name in seen_symbols:
                            continue
                        if len(nodes) >= budget:
                            break
                        seen_symbols.add(name)
                        nodes.append(PropagatedNode(
                            symbol=name, file_path=f,
                            module=getattr(sym_node, 'module', None),
                            depth=1, relation="same_file",
                            reason_codes=["SIBLING_IN_CHANGED_FILE"],
                        ))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Same-file propagation failed: %s", e)

        return nodes

    def _propagate_same_module(
        self,
        seen_files: set[str],
        seen_modules: set[str],
        budget: int,
    ) -> list[PropagatedNode]:
        """Add files in the same module/package directory."""
        nodes: list[PropagatedNode] = []

        try:
            # Get unique directories from seen_files
            dirs: set[str] = set()
            for f in seen_files:
                d = os.path.dirname(f)
                if d:
                    dirs.add(d)

            # For each directory, find .py files not yet in seen_files
            for d in dirs:
                if len(nodes) >= budget:
                    break
                # Walk the directory looking for Python files
                # But we don't have repo_root here, so use module names
                # Add the directory as a module indicator
                mod = d.replace('/', '.').replace('\\', '.')
                if mod and mod not in seen_modules:
                    seen_modules.add(mod)
                    nodes.append(PropagatedNode(
                        module=mod, depth=1, relation="same_module",
                        reason_codes=["SAME_PACKAGE_DIRECTORY"],
                    ))
        except Exception as e:
            logger.debug("Same-module propagation failed: %s", e)

        return nodes

    def _file_to_module(self, file_path: str) -> Optional[str]:
        """Convert file path to approximate module name."""
        if not file_path or LanguageId.from_path(file_path) is not LanguageId.PYTHON:
            return None
        return file_path[:-3].replace('/', '.').replace('\\', '.')
