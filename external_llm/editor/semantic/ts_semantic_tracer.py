"""ts_semantic_tracer.py — TS/JS Semantic Trace Extraction.

Two-layer architecture:

1. **Structural IR** (P1+P2, `analyze_core`) — language-agnostic:
   - imports / exports, functions, classes, interfaces, enums, variables
   - call graph (caller → callee with receiver tracking)

2. **Execution IR** (P2.5, also in `analyze_core`) — edit/repair-ready:
   - IRNodeMeta on every node (stable identity + byte-precise location)
   - Symbol table (IRSymbol: every declared name)
   - Usage graph (IRUsage: every reference)
   - Data flow (IRAssignment: target ← source with type)

Both layers share the same tree-sitter parsing infrastructure.
Reuses `tree_sitter_utils.is_available()` — no new parser introduced.
"""
from __future__ import annotations

import hashlib
import logging
import threading as _threading
from typing import Any, Optional  # f821-protected

from external_llm.editor.semantic.ts_ir_models import (
    ExportKind,
    IRAssignment,
    IRCallSite,
    IRClass,
    IREnum,
    IRExport,
    IRFunction,
    IRImport,
    IRInterface,
    IRMethod,
    IRNodeMeta,
    IRSymbol,
    IRTypeAlias,
    IRUsage,
    IRVariable,
    SymbolKind,
    TSModule,
    TSParam,
)
from external_llm.languages.tree_sitter_utils import is_available

logger = logging.getLogger(__name__)

# ── parser cache ─────────────────────────────────────────────────────────────
#
# Per-thread. tree-sitter's TSParser is stateful and NOT thread-safe
# ("not safe to call ts_parser_parse from multiple threads at once" — api.h).
# Tools run on a shared thread pool (_thread_pool.shared_pool), so a
# module-global cached Parser would be hit
# concurrently from multiple worker threads. Each worker reuses its thread, so
# the per-thread Parser is constructed once and then reused with no locking.

_PARSER_TLS = _threading.local()
_PARSER_MISS = object()


def _build_tsx_parser():
    """Construct a tree-sitter parser for TSX, or None if unavailable.

    Goes through ``languages.tree_sitter_utils`` — the repo's single grammar
    loader — rather than importing a per-language grammar package directly.
    These two builders used to do ``import tree_sitter_typescript`` /
    ``import tree_sitter_javascript``, and neither package is declared in
    ``pyproject.toml``: the only declared grammar source is
    ``tree-sitter-language-pack``, which exposes grammars through
    ``get_language()``, not as top-level ``tree_sitter_<lang>`` modules. A clean
    ``pip install asicode`` therefore hit ``ImportError`` here, and because the
    handler only logs at DEBUG, every TS/TSX/JS caller silently degraded with no
    user-visible signal. Verified against a real export + clean venv: 71 tests
    failed before, 0 after.

    Dev machines masked this by having leftover individual grammar packages
    installed, so it reproduced only outside the dev environment.
    """
    try:
        from external_llm.languages import tree_sitter_utils as _tsu
        return _tsu.get_parser("tsx")
    except Exception as e:
        logger.debug("TSX parser not available: %s", e)
        return None


def _build_jsx_parser():
    """Construct a tree-sitter parser for JavaScript (JSX-aware), or None.

    Routed through ``tree_sitter_utils`` for the reason documented on
    :func:`_build_tsx_parser`.
    """
    try:
        from external_llm.languages import tree_sitter_utils as _tsu
        return _tsu.get_parser("javascript")
    except Exception as e:
        logger.debug("JSX parser not available: %s", e)
        return None


def _get_parser(key: str, builder) -> Any:
    """Get a tree-sitter parser via the per-thread cache (or None)."""
    cache = getattr(_PARSER_TLS, "cache", None)
    if cache is None:
        cache = {}
        _PARSER_TLS.cache = cache
    cached = cache.get(key, _PARSER_MISS)
    if cached is _PARSER_MISS:
        cached = builder()
        cache[key] = cached
    return cached


def _get_tsx_parser():
    """Get a tree-sitter parser for TSX (per-thread cached, or None)."""
    return _get_parser("tsx", _build_tsx_parser)


def _get_jsx_parser():
    """Get a tree-sitter parser for JavaScript (per-thread cached, or None)."""
    return _get_parser("jsx", _build_jsx_parser)


# ── constants ────────────────────────────────────────────────────────────────

# Node types that represent function-like declarations
_FUNC_LIKE = {"arrow_function", "function_expression", "function"}


class TSSemanticTracer:
    """Extracts semantic structure from TS/JS source using tree-sitter.

    Public API:
    - ``analyze_core(code, file_path)`` → ``TSModule`` (Core IR)
    """

    def __init__(self, language: str = "typescript"):
        self._language = language

    # ── shared parser setup ──────────────────────────────────────────────

    def _get_parser(self):
        if self._language == "typescript":
            return _get_tsx_parser()
        return _get_jsx_parser()

    def _parse(self, code: str, file_path: str):
        """Parse code and set internal state.  Returns root node or None."""
        if not is_available():
            logger.warning("tree-sitter not available")
            return None

        parser = self._get_parser()
        if parser is None:
            logger.warning("No parser for language %s", self._language)
            return None

        try:
            tree = parser.parse(code.encode("utf-8"))
        except Exception:
            logger.exception("Failed to parse %s", file_path)
            return None

        self._code_bytes = code.encode("utf-8")
        self._file_path = file_path
        return tree.root_node

    # ── P2.5: node identity + meta ───────────────────────────────────────

    def _make_node_id(self, node) -> str:
        """Deterministic identity: hash of file + byte range."""
        raw = f"{self._file_path}:{node.start_byte}:{node.end_byte}"
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:12]

    def _make_meta(self, node, parent_id: Optional[str] = None) -> IRNodeMeta:
        """Build IRNodeMeta from a tree-sitter node."""
        return IRNodeMeta(
            node_id=self._make_node_id(node),
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            parent_id=parent_id,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 1 — Core IR  (language-agnostic, no React)
    # ══════════════════════════════════════════════════════════════════════

    def analyze_core(self, code: str, file_path: str = "") -> TSModule:
        """Parse *code* into a language-agnostic TSModule IR.

        Layers populated:
        - Structural: imports, exports, functions, classes, interfaces, etc.
        - Call graph: call_sites (caller → callee)
        - Execution (P2.5): symbols, usages, assignments, IRNodeMeta on all nodes
        """
        module = TSModule(file_path=file_path)
        root = self._parse(code, file_path)
        if root is None:
            return module

        for node in root.children:
            self._core_top_level(node, module)

        return module

    def _core_top_level(self, node, module: TSModule) -> None:
        """Dispatch a top-level AST node into Core IR."""
        ntype = node.type

        if ntype == "import_statement":
            imp = self._core_parse_import(node)
            if imp:
                module.imports.append(imp)
            return

        # Unwrap export
        is_exported = False
        export_kind: Optional[ExportKind] = None
        inner = node
        if ntype == "export_statement":
            is_exported = True
            has_default = any(
                self._text(c) == "default" for c in node.children
                if c.type in ("default", "identifier")
                or self._text(c) == "default"
            )
            export_kind = ExportKind.DEFAULT if has_default else ExportKind.NAMED

            # Find inner declaration
            found_inner = False
            for child in node.children:
                if child.type in (
                    "function_declaration", "class_declaration",
                    "lexical_declaration", "interface_declaration",
                    "type_alias_declaration", "enum_declaration",
                ):
                    inner = child
                    found_inner = True
                    break

            if not found_inner:
                # export { X, Y } or export default expr
                self._core_parse_export_clause(node, module)
                return

        ntype = inner.type

        if ntype == "function_declaration":
            func = self._core_parse_function(
                inner, module, is_exported, export_kind)
            module.functions.append(func)
            if is_exported:
                module.exports.append(IRExport(
                    name=func.name, kind=export_kind or ExportKind.NAMED))
            # Collect calls + usages + assignments inside this function
            self._core_collect_calls(inner, func.name, module)
            self._core_collect_usages(inner, func.name, module)
            self._core_collect_assignments(inner, func.name, module)

        elif ntype == "lexical_declaration":
            self._core_process_lexical(inner, module, is_exported, export_kind)

        elif ntype == "class_declaration":
            cls = self._core_parse_class(
                inner, module, is_exported, export_kind)
            module.classes.append(cls)
            if is_exported:
                module.exports.append(IRExport(
                    name=cls.name, kind=export_kind or ExportKind.NAMED))

        elif ntype == "interface_declaration":
            iface = self._core_parse_interface(inner, module, is_exported)
            module.interfaces.append(iface)
            if is_exported:
                module.exports.append(IRExport(name=iface.name))

        elif ntype == "type_alias_declaration":
            ta = self._core_parse_type_alias(inner, module, is_exported)
            module.type_aliases.append(ta)
            if is_exported:
                module.exports.append(IRExport(name=ta.name))

        elif ntype == "enum_declaration":
            en = self._core_parse_enum(inner, module, is_exported)
            module.enums.append(en)
            if is_exported:
                module.exports.append(IRExport(name=en.name))

        elif ntype == "expression_statement":
            # Top-level calls (e.g., app.listen(...))
            self._core_collect_calls(inner, "<module>", module)

    # ── core: import ─────────────────────────────────────────────────────

    def _core_parse_import(self, node) -> Optional[IRImport]:
        source_node = node.child_by_field_name("source")
        if not source_node:
            return None  # pragma: no cover — import_statement always carries a source; errors parse as ERROR

        source = self._text(source_node).strip("'\"")
        specifiers: list[str] = []
        default_name: Optional[str] = None
        namespace_name: Optional[str] = None
        is_type_only = False

        for child in node.children:
            if child.type == "type" or self._text(child) == "type":
                is_type_only = True

            if child.type == "import_clause":
                for cc in child.children:
                    if cc.type == "identifier":
                        default_name = self._text(cc)
                    elif cc.type == "named_imports":
                        for spec in cc.children:
                            if spec.type == "import_specifier":
                                nn = spec.child_by_field_name("name")
                                if nn:
                                    specifiers.append(self._text(nn))
                    elif cc.type == "namespace_import":
                        # import * as X
                        for ns_child in cc.children:
                            if ns_child.type == "identifier":
                                namespace_name = self._text(ns_child)

        return IRImport(
            source=source, specifiers=specifiers, default_name=default_name,
            namespace_name=namespace_name, is_type_only=is_type_only,
            meta=self._make_meta(node),
        )

    # ── core: export clause ──────────────────────────────────────────────

    def _core_parse_export_clause(self, export_node, module: TSModule) -> None:
        """Parse `export { X, Y }` or `export default expr`."""
        for child in export_node.children:
            if child.type == "export_clause":
                for spec in child.children:
                    if spec.type == "export_specifier":
                        name_node = spec.child_by_field_name("name")
                        alias_node = spec.child_by_field_name("alias")
                        if name_node:
                            name = self._text(name_node)
                            if alias_node:
                                name = self._text(alias_node)
                            module.exports.append(IRExport(
                                name=name, kind=ExportKind.NAMED))
            elif child.type == "identifier":
                text = self._text(child)
                if text != "default":
                    module.exports.append(IRExport(
                        name=text, kind=ExportKind.DEFAULT))

    # ── core: function ───────────────────────────────────────────────────

    def _core_parse_function(
        self, node, module: TSModule,
        is_exported: bool, export_kind: Optional[ExportKind],
        scope: str = "<module>",
    ) -> IRFunction:
        name = self._node_field_text(node, "name") or "anonymous"
        params = self._core_extract_params(node)
        is_async = any(c.type == "async" for c in node.children)
        meta = self._make_meta(node)

        func = IRFunction(
            name=name, params=params, is_async=is_async,
            is_exported=is_exported, export_kind=export_kind,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

        # P2.5: register symbol + param symbols
        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.FUNCTION, scope=scope, meta=meta))
        for p in params:
            module.symbols.append(IRSymbol(
                name=p.name, kind=SymbolKind.PARAM, scope=name))

        return func

    # ── core: lexical declaration ────────────────────────────────────────

    def _core_process_lexical(
        self, node, module: TSModule,
        is_exported: bool, export_kind: Optional[ExportKind],
        scope: str = "<module>",
    ) -> None:
        decl_kind = "const"
        for c in node.children:
            if c.type in ("const", "let", "var"):
                decl_kind = self._text(c)
                break

        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node:
                continue  # pragma: no cover — variable_declarator always has a name field

            # Array pattern: const [a, b] = ...
            if name_node.type == "array_pattern":
                for elem in name_node.children:
                    if elem.type == "identifier":
                        elem_name = self._text(elem)
                        meta = self._make_meta(elem)
                        module.variables.append(IRVariable(
                            name=elem_name, decl_kind=decl_kind,
                            start_line=node.start_point.row + 1,
                            end_line=node.end_point.row + 1,
                            meta=meta,
                        ))
                        module.symbols.append(IRSymbol(
                            name=elem_name, kind=SymbolKind.VARIABLE,
                            scope=scope, meta=meta))
                # Assignment for destructuring
                if value_node:
                    source = self._core_callee_name(value_node) if value_node.type == "call_expression" else None
                    source_type = self._classify_initializer(value_node)
                    module.assignments.append(IRAssignment(
                        target="[destructured]", source=source,
                        source_type=source_type, scope=scope,
                        meta=self._make_meta(child),
                    ))
                continue

            name = self._text(name_node)
            node_meta = self._make_meta(node)

            # Arrow / function expression → treat as function
            if value_node and value_node.type in _FUNC_LIKE:
                func = self._core_parse_function(
                    value_node, module, is_exported, export_kind, scope)
                # Override name (arrow functions use variable name)
                func.name = name
                func.start_line = node.start_point.row + 1
                func.end_line = node.end_point.row + 1
                func.meta = node_meta
                # Fix the symbol name too
                for sym in module.symbols:
                    if sym.meta and sym.meta == func.meta:
                        continue  # pragma: no cover — no symbol shares func.meta (fresh node meta)
                    if sym.name != name and sym.scope == scope:
                        continue
                module.functions.append(func)
                if is_exported:
                    module.exports.append(IRExport(
                        name=name, kind=export_kind or ExportKind.NAMED))
                self._core_collect_calls(value_node, name, module)
                self._core_collect_usages(value_node, name, module)
                self._core_collect_assignments(value_node, name, module)
            else:
                # Regular variable
                init_type = None
                source: Optional[str] = None
                if value_node:
                    init_type = self._classify_initializer(value_node)
                module.variables.append(IRVariable(
                    name=name, decl_kind=decl_kind,
                    initializer_type=init_type,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    meta=node_meta,
                ))
                # P2.5: symbol
                module.symbols.append(IRSymbol(
                    name=name, kind=SymbolKind.VARIABLE,
                    scope=scope, meta=node_meta))
                # P2.5: assignment
                if value_node:
                    if value_node.type == "call_expression":
                        source = self._core_callee_name(value_node)
                    elif value_node.type == "identifier":
                        source = self._text(value_node)
                    elif value_node.type == "new_expression":
                        for vc in value_node.children:
                            if vc.type == "identifier":
                                source = self._text(vc)
                                break
                    module.assignments.append(IRAssignment(
                        target=name, source=source,
                        source_type=init_type, scope=scope,
                        meta=self._make_meta(child),
                    ))
                if is_exported:
                    module.exports.append(IRExport(
                        name=name, kind=export_kind or ExportKind.NAMED))
                # Top-level call in initializer
                if value_node and value_node.type == "call_expression":
                    callee = self._core_callee_name(value_node)
                    if callee:
                        module.call_sites.append(IRCallSite(
                            caller=scope, callee=callee,
                            line=value_node.start_point.row + 1,
                            meta=self._make_meta(value_node),
                        ))

    def _classify_initializer(self, node) -> Optional[str]:
        t = node.type
        if t == "call_expression":
            return "call"
        if t in ("number", "string", "true", "false", "null", "undefined"):
            return "literal"
        if t == "identifier":
            return "variable"
        if t in _FUNC_LIKE:
            return "arrow"
        if t in ("array", "array_expression"):
            return "array"
        if t in ("object", "object_expression"):
            return "object"
        if t == "new_expression":
            return "new"
        if t == "await_expression":
            return "await"
        return None

    # ── core: class ──────────────────────────────────────────────────────

    def _core_parse_class(
        self, node, module: TSModule,
        is_exported: bool, export_kind: Optional[ExportKind],
    ) -> IRClass:
        name = self._node_field_text(node, "name") or "anonymous"
        extends: Optional[str] = None
        implements: list[str] = []
        methods: list[IRMethod] = []
        properties: list[str] = []
        meta = self._make_meta(node)

        # Heritage
        for child in node.children:
            if child.type == "class_heritage":
                for hc in child.children:
                    if hc.type == "extends_clause":
                        for ec in hc.children:
                            if ec.type in ("identifier", "member_expression"):
                                extends = self._text(ec)
                    elif hc.type == "implements_clause":
                        implements.extend(self._text(ic) for ic in hc.children if ic.type in ("identifier", "generic_type", "type_identifier"))

        # P2.5: class symbol
        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.CLASS, scope="<module>", meta=meta))

        # Body
        body = node.child_by_field_name("body")
        if body:
            for member in body.children:
                if member.type == "method_definition":
                    m = self._core_parse_method(member, module, name)
                    methods.append(m)
                elif member.type in (
                    "public_field_definition", "property_definition",
                    "field_definition",
                ):
                    pname = self._node_field_text(member, "name")
                    if pname:
                        from external_llm.editor.semantic.ts_ir_models import IRClassProperty
                        properties.append(IRClassProperty(
                            name=pname, meta=self._make_meta(member),
                        ))

        return IRClass(
            name=name, methods=methods, properties=properties,
            extends=extends, implements=implements,
            is_exported=is_exported, export_kind=export_kind,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

    def _core_parse_method(
        self, node, module: TSModule, class_name: str,
    ) -> IRMethod:
        name = self._node_field_text(node, "name") or "anonymous"
        params = self._core_extract_params(node)
        is_async = any(c.type == "async" for c in node.children)
        is_static = any(c.type == "static" for c in node.children)
        meta = self._make_meta(node)

        is_getter = False
        is_setter = False
        for c in node.children:
            t = self._text(c)
            if t == "get" and c.type != "identifier":
                is_getter = True
            elif t == "set" and c.type != "identifier":
                is_setter = True

        calls: list[str] = []
        for desc in self._walk(node):
            if desc.type == "call_expression":
                callee = self._core_callee_name(desc)
                if callee:
                    calls.append(callee)

        # P2.5: method symbol
        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.METHOD, scope=class_name, meta=meta))

        return IRMethod(
            name=name, params=params, is_async=is_async,
            is_static=is_static, is_getter=is_getter, is_setter=is_setter,
            calls=calls,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

    # ── core: interface ──────────────────────────────────────────────────

    def _core_parse_interface(
        self, node, module: TSModule, is_exported: bool,
    ) -> IRInterface:
        name = self._node_field_text(node, "name") or "anonymous"
        properties: list[str] = []
        methods: list[str] = []
        extends: list[str] = []
        meta = self._make_meta(node)

        for child in node.children:
            if child.type == "extends_type_clause":
                extends.extend(self._text(ec) for ec in child.children if ec.type in ("identifier", "generic_type", "type_identifier"))

        body = node.child_by_field_name("body")
        if body:
            for member in body.children:
                if member.type == "property_signature":
                    pname = self._node_field_text(member, "name")
                    if pname:
                        from external_llm.editor.semantic.ts_ir_models import IRClassProperty
                        properties.append(IRClassProperty(
                            name=pname, meta=self._make_meta(member),
                        ))
                elif member.type == "method_signature":
                    mname = self._node_field_text(member, "name")
                    if mname:
                        methods.append(mname)

        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.INTERFACE, scope="<module>", meta=meta))

        return IRInterface(
            name=name, properties=properties, methods=methods,
            extends=extends, is_exported=is_exported,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

    # ── core: type alias ─────────────────────────────────────────────────

    def _core_parse_type_alias(
        self, node, module: TSModule, is_exported: bool,
    ) -> IRTypeAlias:
        name = self._node_field_text(node, "name") or "anonymous"
        meta = self._make_meta(node)
        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.TYPE_ALIAS, scope="<module>",
            meta=meta))
        return IRTypeAlias(
            name=name, is_exported=is_exported,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

    # ── core: enum ───────────────────────────────────────────────────────

    def _core_parse_enum(
        self, node, module: TSModule, is_exported: bool,
    ) -> IREnum:
        name = self._node_field_text(node, "name") or "anonymous"
        meta = self._make_meta(node)
        members: list[str] = []
        body = node.child_by_field_name("body")
        if body:
            members.extend(self._text(child).split("=")[0].strip() for child in body.children if child.type in ("enum_assignment", "property_identifier"))
        module.symbols.append(IRSymbol(
            name=name, kind=SymbolKind.ENUM, scope="<module>", meta=meta))
        return IREnum(
            name=name, members=members, is_exported=is_exported,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            meta=meta,
        )

    # ── core: call graph ─────────────────────────────────────────────────

    def _core_collect_calls(
        self, node, caller: str, module: TSModule,
    ) -> None:
        """Walk a subtree and record all call_expression nodes."""
        for desc in self._walk(node):
            if desc.type != "call_expression":
                continue
            func_node = desc.child_by_field_name("function")
            if not func_node:
                continue  # pragma: no cover — call_expression always has a function field

            callee: Optional[str] = None
            is_method = False
            receiver: Optional[str] = None

            if func_node.type == "identifier":
                callee = self._text(func_node)
            elif func_node.type == "member_expression":
                obj = func_node.child_by_field_name("object")
                prop = func_node.child_by_field_name("property")
                if obj and prop:
                    receiver = self._text(obj)
                    callee = self._text(prop)
                    is_method = True

            if callee:
                module.call_sites.append(IRCallSite(
                    caller=caller, callee=callee,
                    is_method_call=is_method, receiver=receiver,
                    line=desc.start_point.row + 1,
                    meta=self._make_meta(desc),
                ))

    # ── P2.5: usage graph ────────────────────────────────────────────────

    def _core_collect_usages(
        self, node, scope: str, module: TSModule,
    ) -> None:
        """Walk a subtree and record identifier references as usages."""
        for desc in self._walk(node):
            if desc.type != "identifier":
                continue
            # Skip identifiers that are part of declarations
            parent = desc.parent
            if parent is None:
                continue  # pragma: no cover — root nodes are never identifiers
            parent_type = parent.type
            # Skip if this identifier IS the declaration name
            if parent_type in (
                "function_declaration", "class_declaration",
                "interface_declaration", "type_alias_declaration",
                "enum_declaration",
            ):
                name_node = parent.child_by_field_name("name")
                if name_node and name_node == desc:
                    continue
            # Skip variable declarator name (definition, not usage)
            if parent_type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node and name_node == desc:
                    continue
            # Skip formal parameter names
            if parent_type in (
                "formal_parameters", "required_parameter",
                "optional_parameter",
            ):
                continue
            # Skip import specifier names
            if parent_type in ("import_specifier", "import_clause"):
                continue  # pragma: no cover — usage collection never walks import clauses (top-level only)
            # Skip property identifiers in object literals / member access
            if parent_type == "member_expression":
                prop_node = parent.child_by_field_name("property")
                if prop_node and prop_node == desc:
                    continue  # pragma: no cover — member properties are property_identifier, filtered at L738

            name = self._text(desc)
            module.usages.append(IRUsage(
                symbol=name, scope=scope,
                meta=self._make_meta(desc),
            ))

    # ── P2.5: assignment tracking ────────────────────────────────────────

    def _core_collect_assignments(
        self, node, scope: str, module: TSModule,
    ) -> None:
        """Walk a subtree and record variable assignments."""
        for desc in self._walk(node):
            if desc.type != "variable_declarator":
                continue
            name_node = desc.child_by_field_name("name")
            value_node = desc.child_by_field_name("value")
            if not name_node or not value_node:
                continue
            # Skip array patterns (handled in lexical)
            if name_node.type != "identifier":
                continue

            target = self._text(name_node)
            source: Optional[str] = None
            source_type = self._classify_initializer(value_node)

            if value_node.type == "call_expression":
                source = self._core_callee_name(value_node)
            elif value_node.type == "identifier":
                source = self._text(value_node)
            elif value_node.type == "new_expression":
                for vc in value_node.children:
                    if vc.type == "identifier":
                        source = self._text(vc)
                        break
            elif value_node.type == "await_expression":
                for vc in value_node.children:
                    if vc.type == "call_expression":
                        source = self._core_callee_name(vc)
                        break

            module.assignments.append(IRAssignment(
                target=target, source=source,
                source_type=source_type, scope=scope,
                meta=self._make_meta(desc),
            ))

    def _core_callee_name(self, call_node) -> Optional[str]:
        """Extract callee name from a call_expression (simple or member)."""
        func_node = call_node.child_by_field_name("function")
        if not func_node:
            return None  # pragma: no cover — call_expression always has a function field
        if func_node.type == "identifier":
            return self._text(func_node)
        if func_node.type == "member_expression":
            prop = func_node.child_by_field_name("property")
            return self._text(prop) if prop else None
        return None  # pragma: no cover — member_expression always has a property field

    # ── core: params ─────────────────────────────────────────────────────

    def _core_extract_params(self, func_node) -> list[TSParam]:
        params_node = func_node.child_by_field_name("parameters")
        if not params_node:
            return []  # pragma: no cover — function nodes always carry a parameters field

        params: list[TSParam] = []
        for child in params_node.children:
            if child.type == "identifier":
                params.append(TSParam(name=self._text(child)))
            elif child.type in ("required_parameter", "optional_parameter"):
                pat = child.child_by_field_name("pattern")
                name = self._text(pat) if pat else self._text(child)
                # Clean up type annotation from name
                if ":" in name:
                    name = name.split(":")[0].strip()  # pragma: no cover — pattern field excludes the type annotation
                is_rest = any(
                    c.type == "..." for c in child.children)
                has_default = child.child_by_field_name("value") is not None
                params.append(TSParam(
                    name=name, has_default=has_default, is_rest=is_rest))
            elif child.type == "rest_pattern":
                params.extend(TSParam(
                            name=self._text(rc), is_rest=True) for rc in child.children if rc.type == "identifier")
            elif child.type == "object_pattern":
                params.append(TSParam(name="{...}"))
            elif child.type == "array_pattern":
                params.append(TSParam(name="[...]"))
        return params

    # ══════════════════════════════════════════════════════════════════════
    #  SHARED UTILITIES
    # ══════════════════════════════════════════════════════════════════════

    def _text(self, node) -> str:
        return self._code_bytes[node.start_byte: node.end_byte].decode("utf-8")

    def _node_field_text(self, node, field: str) -> Optional[str]:
        child = node.child_by_field_name(field)
        return self._text(child) if child else None

    def _walk(self, node):
        """Yield all descendants of a node (depth-first)."""
        cursor = node.walk()
        visited = False
        while True:
            if not visited:
                yield cursor.node
                if cursor.goto_first_child():
                    continue
            if cursor.goto_next_sibling():
                visited = False
                continue
            if cursor.goto_parent():
                visited = True
                if cursor.node == node:
                    break
            else:
                break
