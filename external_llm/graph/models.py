"""
Canonical model types for the Global Symbol Graph (P1).

This module defines the unified data models shared across all graph
subsystems (RepositoryGraph, CallGraphIndexer, RepositoryGraphFacade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    CONSTANT = "constant"  # module-level constant / variable assignment


class EdgeKind(str, Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    DEFINES = "defines"
    INHERITS = "inherits"
    CONTAINS = "contains"


@dataclass(frozen=True)
class SymbolId:
    """Canonical identity for a symbol across the repository."""
    module: str
    qualname: str
    file_path: str
    kind: SymbolKind

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolId):
            return NotImplemented
        return (
            self.module == other.module
            and self.qualname == other.qualname
            and self.file_path == other.file_path
            and self.kind == other.kind
        )

    def __hash__(self) -> int:
        return hash((self.module, self.qualname, self.file_path, self.kind))


@dataclass
class SymbolNode:
    """Node representing a symbol (function, class, module) in the repository."""
    name: str
    qualname: str
    module: str
    file_path: str
    kind: str  # kept as str for backward compat; use SymbolKind values
    start_line: int
    end_line: int
    language: Optional[str] = None  # e.g. "python", "typescript", "javascript"
    signature_hash: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None  # full function signature text with type annotations
    bases: Optional[list[str]] = None  # parent class names (for class symbols only)

    @property
    def symbol_id(self) -> SymbolId:
        try:
            sk = SymbolKind(self.kind)
        except ValueError:
            sk = SymbolKind.CONSTANT if self.kind == "constant" else SymbolKind.FUNCTION
        return SymbolId(
            module=self.module,
            qualname=self.qualname,
            file_path=self.file_path,
            kind=sk,
        )


@dataclass
class CallEdge:
    """Unified edge representing a function/method call."""
    caller_symbol: str
    caller_file: str
    caller_line: int
    callee_symbol: str
    callee_display: str
    callee_file: Optional[str] = None
    callee_line: Optional[int] = None
    confidence: float = 1.0
    edge_kind: EdgeKind = EdgeKind.CALLS
    call_args: list[str] = field(default_factory=list)
    """Literal positional arg values at call site — enables object identity.
    e.g. get_user(1, 'admin') → ["1", "'admin'"].
    Empty list means arguments were expressions, not literals.
    """
    is_mutating: bool = False
    """Heuristic: True when this call has write/side-effect semantics.
    e.g. db.save(user), session.commit(), cache.set(k, v).
    Used by graph_propagator to boost UPDATE_CALLERS weight for schema changes.
    """


@dataclass
class ImportEdge:
    """Edge representing an import relationship."""
    importer: str
    imported: str
    import_type: str  # "import", "from", "import_from"
    alias: Optional[str] = None
