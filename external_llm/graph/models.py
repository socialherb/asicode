"""
Canonical model types for the Global Symbol Graph (P1).

This module defines the unified data models shared across all graph
subsystems (RepositoryGraph, CallGraphIndexer, RepositoryGraphFacade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    language: str | None = None  # e.g. "python", "typescript", "javascript"
    signature_hash: str | None = None
    docstring: str | None = None
    signature: str | None = None  # full function signature text with type annotations
    bases: list[str] | None = None  # parent class names (for class symbols only)
    # P3 Stage 2: CGI-convention reconstruction fields — RG's snapshot is the
    # SSOT for CallGraphIndexer, which needs (a) the async-ness CGI encodes in
    # its def kind ("async_function" vs "function") and (b) the CGI defs symbol
    # (direct-class-qualified "Class.method", bare name otherwise) that differs
    # from RG's full-scope qualname (e.g. "Outer.inner" vs "inner").  Additive
    # defaults keep pre-P3 snapshots loadable: SymbolNode(**d) without these
    # keys still works, and the CGI conversion falls back to name/function.
    is_async: bool = False  # def is async def (CGI kind discriminator)
    cgi_symbol: str | None = None  # CGI-convention defs symbol
    # P3 Stage 2: AST nesting depth of the symbol's definition (module=0,
    # class/function body=1, ...).  CGI collects defs via ``ast.walk`` (BFS:
    # ALL depth-k symbols before any depth-(k+1) symbol), while RG traverses
    # DFS — qualname's dotted depth alone cannot reproduce BFS order because
    # if/for blocks (invisible in qualnames) add nesting levels.  Storing the
    # real AST depth lets the SSOT conversion sort defs (depth, start_line),
    # which IS ast.walk order.
    ast_depth: int = 0

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
    callee_file: str | None = None
    callee_line: int | None = None
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
    alias: str | None = None
