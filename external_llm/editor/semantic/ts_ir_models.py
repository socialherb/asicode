"""ts_ir_models.py — TS/JS Core Intermediate Representation.

Language-agnostic semantic IR for TypeScript/JavaScript code.
NO React/Node specifics — those belong in the Profile layer (P3).

Two layers:
1. **Structural IR** (P1+P2): imports, exports, functions, classes, call graph
2. **Execution IR** (P2.5): node identity, symbol table, usage graph, data flow

This is the foundation for:
- Call graph construction
- Symbol-level edit / refactor
- Data-flow-based repair
- Primitive system
- Cross-language VM alignment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ── enums ────────────────────────────────────────────────────────────────────


class SymbolKind(Enum):
    """Kind of a symbol."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    PARAM = "param"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
    IMPORT = "import"


class ExportKind(Enum):
    """Export type."""

    NAMED = "named"
    DEFAULT = "default"
    RE_EXPORT = "re_export"


# ── execution metadata (P2.5) ───────────────────────────────────────────────


@dataclass
class IRNodeMeta:
    """Stable identity + byte-precise location for any IR node.

    Every IR node can carry this metadata, enabling:
    - Precise byte-range edits (patch engine)
    - Cross-reference by node_id (symbol table ↔ usage)
    - Parent-child traversal
    """

    node_id: str  # deterministic hash: file:start_byte:end_byte
    start_byte: int
    end_byte: int
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    parent_id: Optional[str] = None


# ── core IR nodes ────────────────────────────────────────────────────────────


@dataclass
class TSTypeRef:
    """A lightweight type reference (not full type system)."""

    name: str
    is_array: bool = False
    is_optional: bool = False
    is_union: bool = False
    type_args: list[str] = field(default_factory=list)


@dataclass
class TSParam:
    """A function/method parameter."""

    name: str
    type_ref: Optional[TSTypeRef] = None
    has_default: bool = False
    is_rest: bool = False


@dataclass
class IRImport:
    """An import statement — source module + what's imported."""

    source: str
    specifiers: list[str] = field(default_factory=list)
    default_name: Optional[str] = None
    namespace_name: Optional[str] = None
    is_type_only: bool = False
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRExport:
    """An export declaration."""

    name: str
    kind: ExportKind = ExportKind.NAMED
    original_name: Optional[str] = None
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRVariable:
    """A top-level or scoped variable declaration."""

    name: str
    decl_kind: str = "const"
    type_ref: Optional[TSTypeRef] = None
    initializer_type: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRFunction:
    """A function or arrow function (NOT method — see IRClass)."""

    name: str
    params: list[TSParam] = field(default_factory=list)
    return_type: Optional[TSTypeRef] = None
    is_async: bool = False
    is_generator: bool = False
    is_exported: bool = False
    export_kind: Optional[ExportKind] = None
    calls: list[str] = field(default_factory=list)
    local_vars: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRMethod:
    """A class method."""

    name: str
    params: list[TSParam] = field(default_factory=list)
    return_type: Optional[TSTypeRef] = None
    is_async: bool = False
    is_static: bool = False
    is_getter: bool = False
    is_setter: bool = False
    calls: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRClassProperty:
    """A class/interface property with byte-precise location."""

    name: str
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRClass:
    """A class declaration."""

    name: str
    methods: list[IRMethod] = field(default_factory=list)
    properties: list[IRClassProperty] = field(default_factory=list)
    extends: Optional[str] = None
    implements: list[str] = field(default_factory=list)
    is_exported: bool = False
    export_kind: Optional[ExportKind] = None
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRInterface:
    """An interface declaration."""

    name: str
    properties: list[IRClassProperty] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    extends: list[str] = field(default_factory=list)
    is_exported: bool = False
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRTypeAlias:
    """A type alias (type X = ...)."""

    name: str
    is_exported: bool = False
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IREnum:
    """An enum declaration."""

    name: str
    members: list[str] = field(default_factory=list)
    is_exported: bool = False
    start_line: int = 0
    end_line: int = 0
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRCallSite:
    """A call from one function/method to another."""

    caller: str
    callee: str
    is_method_call: bool = False
    receiver: Optional[str] = None
    line: int = 0
    meta: Optional[IRNodeMeta] = None


# ── P2.5: symbol table, usage graph, data flow ──────────────────────────────


@dataclass
class IRSymbol:
    """Symbol table entry — every declared name gets one.

    Enables symbol-level queries:
    - "where is X defined?" → meta.start_byte..end_byte
    - "what kind is X?" → kind
    - "what scope?" → scope (function name or '<module>')
    """

    name: str
    kind: SymbolKind
    scope: str = "<module>"  # function/class name, or '<module>'
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRUsage:
    """A reference (read) of a symbol at a specific location.

    Enables:
    - "who uses X?" → filter by symbol name
    - "what does function Y reference?" → filter by scope
    """

    symbol: str  # name of the referenced symbol
    scope: str = "<module>"  # function/class where the usage occurs
    meta: Optional[IRNodeMeta] = None


@dataclass
class IRAssignment:
    """An assignment / binding — data flows from source to target.

    Enables lightweight data-flow analysis:
    - "where does X get its value?" → source + source_type
    - "what depends on function F's return?" → find assignments with source=F
    """

    target: str  # variable being assigned
    source: Optional[str] = None  # source symbol or function name
    source_type: Optional[str] = None  # "call", "variable", "literal", "new"
    scope: str = "<module>"  # function where assignment occurs
    meta: Optional[IRNodeMeta] = None


# ── module-level IR ──────────────────────────────────────────────────────────


@dataclass
class TSModule:
    """Complete language-agnostic IR for a single TS/JS module.

    Layers:
    - Structural: imports, exports, functions, classes, interfaces, etc.
    - Call graph: call_sites
    - Execution (P2.5): symbols, usages, assignments
    - Metadata: profile layer can attach extra info
    """

    file_path: str

    # ── structural declarations ──────────────────────────────────────
    imports: list[IRImport] = field(default_factory=list)
    exports: list[IRExport] = field(default_factory=list)
    functions: list[IRFunction] = field(default_factory=list)
    classes: list[IRClass] = field(default_factory=list)
    interfaces: list[IRInterface] = field(default_factory=list)
    type_aliases: list[IRTypeAlias] = field(default_factory=list)
    enums: list[IREnum] = field(default_factory=list)
    variables: list[IRVariable] = field(default_factory=list)

    # ── call graph ───────────────────────────────────────────────────
    call_sites: list[IRCallSite] = field(default_factory=list)

    # ── execution IR (P2.5) ──────────────────────────────────────────
    symbols: list[IRSymbol] = field(default_factory=list)
    usages: list[IRUsage] = field(default_factory=list)
    assignments: list[IRAssignment] = field(default_factory=list)

    # ── metadata ─────────────────────────────────────────────────────
    metadata: dict = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────

    @property
    def all_symbols(self) -> list[str]:
        """All declared symbol names (from structural IR)."""
        names: list[str] = []
        names.extend(f.name for f in self.functions)
        names.extend(c.name for c in self.classes)
        names.extend(i.name for i in self.interfaces)
        names.extend(t.name for t in self.type_aliases)
        names.extend(e.name for e in self.enums)
        names.extend(v.name for v in self.variables)
        return names

    @property
    def exported_symbols(self) -> set[str]:
        return {e.name for e in self.exports}

    @property
    def import_sources(self) -> set[str]:
        return {i.source for i in self.imports}

    def get_function(self, name: str) -> Optional[IRFunction]:
        """Find a top-level function by name, or a class method with matching name.

        Supports:
        - Bare names: ``greet``, ``helper``, ``fetchData``
        - Dotted names: ``Game.lockPiece``, ``Greeter.greet``

        Dotted names are resolved as ``ClassName.methodName`` — the class
        is looked up first, then the method within it is returned.
        """
        # ── Dotted name: ClassName.methodName ──────────────────────────
        if "." in name:
            cls_name, _, member_name = name.partition(".")
            for c in self.classes:
                if c.name == cls_name:
                    for m in c.methods:
                        if m.name == member_name:
                            return m  # type: ignore[return-value]
            return None

        # ── Bare name ──────────────────────────────────────────────────
        # 1. Top-level functions (exact match)
        for f in self.functions:
            if f.name == name:
                return f
        # 2. Class methods (IRMethod has `meta` like IRFunction)
        for c in self.classes:
            for m in c.methods:
                if m.name == name:
                    return m  # type: ignore[return-value] -- IRMethod has meta
        return None

    def get_class(self, name: str) -> Optional[IRClass]:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def callers_of(self, callee: str) -> list[str]:
        return list({cs.caller for cs in self.call_sites if cs.callee == callee})

    def callees_of(self, caller: str) -> list[str]:
        return list({cs.callee for cs in self.call_sites if cs.caller == caller})

    # ── P2.5 convenience ─────────────────────────────────────────────

    def get_symbol(self, name: str) -> Optional[IRSymbol]:
        """Find a symbol table entry by name."""
        for s in self.symbols:
            if s.name == name:
                return s
        return None

    def symbols_in_scope(self, scope: str) -> list[IRSymbol]:
        """All symbols declared in a given scope."""
        return [s for s in self.symbols if s.scope == scope]

    def usages_of(self, symbol: str) -> list[IRUsage]:
        """All usage sites of a symbol."""
        return [u for u in self.usages if u.symbol == symbol]

    def assignments_to(self, target: str) -> list[IRAssignment]:
        """All assignments to a variable."""
        return [a for a in self.assignments if a.target == target]

    def data_sources_of(self, target: str) -> list[str]:
        """What provides data to *target*? Returns source names."""
        return [
            a.source for a in self.assignments
            if a.target == target and a.source is not None
        ]
