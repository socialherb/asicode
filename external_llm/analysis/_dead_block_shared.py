"""Shared utilities for dead-block scanners.

Consolidates data models and AST helpers used by both
``dead_block_scanner`` and ``public_dead_code_scanner``
to eliminate ~200 lines of duplicated code.
Also re-used by ``duplicate_definition_scanner`` for
``_is_overload_decorator`` / ``_has_overload``.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tree-sitter availability ─────────────────────────────────────────────
try:
    from ..languages.tree_sitter_utils import (
        get_node_text as _ts_get_text,
    )
    from ..languages.tree_sitter_utils import (  # type: ignore
        parse_to_tree as _ts_parse_to_tree,
    )
    from ..languages.tree_sitter_utils import (
        query_matches as _ts_query_matches,
    )
    _HAS_TS = True
except ImportError:
    _HAS_TS = False

# Per-language definition node types: node_type → (kind, is_container, skip_if_enclosed)
#   kind: "function" | "class" | "assignment" | "interface" | "type_alias" | "enum"
#   is_container: True = recurse into children (class body, module body)
#   skip_if_enclosed: True = skip when inside a container (class methods)
_LANG_DEF_NODES: dict[str, dict[str, tuple]] = {
    "python": {
        "function_definition": ("function", False, True),
        "async_function_definition": ("function", False, True),
        "class_definition": ("class", True, True),
        "expression_statement": ("assignment", False, False),
        "annotated_assignment": ("assignment", False, False),
    },
    "typescript": {
        "function_declaration": ("function", False, True),
        "class_declaration": ("class", True, True),
        "interface_declaration": ("interface", True, True),
        "type_alias_declaration": ("assignment", False, False),
        "enum_declaration": ("class", True, True),
        "lexical_declaration": ("assignment", False, False),
        "variable_declaration": ("assignment", False, False),
        "module_declaration": ("class", True, True),
    },
    "javascript": {
        "function_declaration": ("function", False, True),
        "class_declaration": ("class", True, True),
        "lexical_declaration": ("assignment", False, False),
        "variable_declaration": ("assignment", False, False),
    },
    "go": {
        "function_declaration": ("function", False, True),
        "method_declaration": ("function", False, True),
        "type_declaration": ("class", True, True),
        "type_spec": ("assignment", False, False),
        "const_declaration": ("assignment", False, False),
        "var_declaration": ("assignment", False, False),
    },
    "java": {
        "class_declaration": ("class", True, True),
        "interface_declaration": ("class", True, True),
        "enum_declaration": ("class", True, True),
        "method_declaration": ("function", False, True),
        "field_declaration": ("assignment", False, False),
    },
    "kotlin": {
        "class_declaration": ("class", True, True),
        "object_declaration": ("class", True, True),
        "companion_object": ("class", True, True),
        "interface_declaration": ("class", True, True),
        "enum_declaration": ("class", True, True),
        "fun_declaration": ("function", False, True),
        "property_declaration": ("assignment", False, False),
    },
}

# Per-language assignment wrapper nodes — the name identifier is nested one level
# deeper inside an inner node (e.g. expression_statement → assignment → identifier,
# lexical_declaration → variable_declarator → identifier).
_LANG_ASSIGN_WRAPPERS: dict[str, set] = {
    "python": {"expression_statement", "annotated_assignment"},
    "typescript": {"lexical_declaration", "variable_declaration", "expression_statement"},
    "javascript": {"lexical_declaration", "variable_declaration"},
    "go": {"type_declaration", "var_declaration", "const_declaration", "type_spec"},
    "java": {"field_declaration"},
    "kotlin": {"property_declaration"},
}
_LANG_DEF_PARENT_TYPES: dict[str, set] = {
    "python": {"function_definition", "async_function_definition", "class_definition",
               "assignment", "annotated_assignment", "parameters", "lambda_parameters",
               "for_statement", "with_item", "import_statement", "import_from_statement",
               "alias", "typed_parameter", "default_parameter"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                   "required_parameter", "optional_parameter", "variable_declarator",
                   "lexical_declaration", "for_in_statement", "catch_clause",
                   "arrow_function", "assignment"},
    "javascript": {"function_declaration", "class_declaration", "variable_declarator",
                   "lexical_declaration", "for_in_statement", "catch_clause",
                   "arrow_function", "assignment"},
    "go": {"function_declaration", "method_declaration", "type_declaration",
           "type_spec", "parameter_declaration", "var_declaration", "short_var_declaration",
           "field_declaration", "receiver", "const_declaration", "const_spec", "var_spec"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration",
             "method_declaration", "formal_parameter", "variable_declarator",
             "field_declaration", "constructor_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "companion_object",
               "fun_declaration", "property_declaration", "parameter",
               "value_parameter", "variable_declaration", "destructured_parameter"},
}

CLUSTER_GAP_TOLERANCE = 5  # max blank-line gap between adjacent dead defs


# ── Data models ──────────────────────────────────────────────────────────


@dataclass
class DeadBlockMember:
    name: str
    symbol_kind: str  # "function" | "class" | "assignment" | "class_assignment"
    lineno: int
    end_lineno: int
    enclosing_class: Optional[str] = None


@dataclass
class DeadBlockCandidate:
    """A contiguous group of unused module-level definitions in one file."""
    file: str
    members: list[DeadBlockMember] = field(default_factory=list)
    cluster_start: int = 0
    cluster_end: int = 0
    confidence: float = 1.0
    is_singleton: bool = False
    includes_public: bool = False

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "members": [
                {
                    "name": m.name,
                    "symbol_kind": m.symbol_kind,
                    "lineno": m.lineno,
                    "end_lineno": m.end_lineno,
                    "enclosing_class": m.enclosing_class,
                }
                for m in self.members
            ],
            "cluster_start": self.cluster_start,
            "cluster_end": self.cluster_end,
            "includes_public": self.includes_public,
        }


# ── AST helpers ──────────────────────────────────────────────────────────


def _ts_extract_all_list(source: str, language: str = "python") -> set:
    """Tree-sitter version: return set of names registered in ``__all__`` literal.

    Only Python has the ``__all__`` convention; other languages always return empty.
    """
    if not _HAS_TS:
        return set()
    if language != "python":
        return set()
    tree = _ts_parse_to_tree(source, language)
    if tree is None:
        return set()
    # Query: __all__ = [ ... ]
    query_str = """
(module (expression_statement
  (assignment
    left: (identifier) @name
    right: (list (string (string_content) @item))) @def
  (#eq? @name "__all__")))
"""
    matches = _ts_query_matches(source, language, query_str)
    names: set = set()
    for match in matches:
        items = match.get("item", [])
        for cap in items:
            names.add(cap.text)

    # Detect dynamic __all__ (assignment to variable/expression, not literal list)
    if not names:
        any_all_query = """
    (module (expression_statement
      (assignment
        left: (identifier) @name)
      (#eq? @name "__all__")))
    """
        if _ts_query_matches(source, language, any_all_query):
            names.add("*__dynamic__*")
    return names


def _ts_collect_all_defs(
    source: str,
    language: str = "python",
) -> list[tuple[str, str, int, int, Optional[str]]]:
    """Tree-sitter version of ``_collect_all_defs``.

    Uses per-language ``_LANG_DEF_NODES`` map (supports Python, TypeScript,
    JavaScript, Go, Java, Kotlin).  Falls back to empty when *language* is
    not in the map.

    Collects module-level AND class-level definitions.
    Returns (name, kind, lineno, end_lineno, enclosing_class_or_None).
    """
    if not _HAS_TS:
        return []
    lang_defs = _LANG_DEF_NODES.get(language)
    if lang_defs is None:
        return []
    tree = _ts_parse_to_tree(source, language)
    if tree is None:
        return []
    out: list[tuple[str, str, int, int, Optional[str]]] = []
    root = tree.root_node

    def _walk(node, enclosing_class: Optional[str] = None):
        if not node:
            return
        node_type = node.type
        def_info = lang_defs.get(node_type)
        if def_info is not None:
            kind, is_container, skip_if_enclosed = def_info
            if skip_if_enclosed and enclosing_class is not None:
                return  # skip methods / inner definitions
            # Find the name identifier — may be nested inside assignment wrapper
            name_node = _ts_child_by_type(node, ("identifier",))
            if name_node is None:
                # Check if this is an assignment wrapper (e.g. expression_statement
                # wrapping an assignment node that contains the identifier)
                in_wrappers = language in _LANG_ASSIGN_WRAPPERS and node_type in _LANG_ASSIGN_WRAPPERS[language]
                if in_wrappers:
                    for child in (node.children or []):
                        name_node = _ts_child_by_type(child, ("identifier",))
                        if name_node is not None:
                            break
            if name_node is None:
                return
            name = _ts_get_text(source.encode("utf-8"), name_node)
            # Check overload / framework decorators via manual walk
            if language == "python" and _ts_has_overload_or_framework(node, source):
                return
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            decorator_node = _ts_child_by_type(node, ("decorator",))
            if decorator_node is not None:
                start = decorator_node.start_point[0] + 1
            out.append((name, kind, start, end, enclosing_class))
            if is_container:
                for child in (node.children or []):
                    _walk(child, enclosing_class=name)

        elif language == "python" and node_type == "decorated_definition":
            if _ts_has_overload_or_framework(node, source):
                return
        else:
            for child in (node.children or []):
                _walk(child, enclosing_class)

    _walk(root)
    return out


def _ts_child_by_type(node, type_names: tuple[str, ...]) -> Optional[object]:
    """Find first child with one of *type_names* (tree-sitter node helper)."""
    for child in (node.children or []):
        if child.type in type_names:
            return child
    return None


def _ts_has_overload_or_framework(node, source: str) -> bool:
    """Check if node has @overload or @fixture/@hookimpl/@hookspec decorator."""
    _FRAMEWORK_NAMES = {"fixture", "hookimpl", "hookspec"}
    for child in (node.children or []):
        if child.type != "decorator":
            continue
        # Walk decorator child for identifier
        dec_text = _ts_get_text(source.encode("utf-8"), child)
        # Strip leading @
        dec_name = dec_text.lstrip("@").strip()
        if dec_name == "overload":
            return True
        if "." in dec_name:
            _, _, attr = dec_name.rpartition(".")
            if attr in _FRAMEWORK_NAMES:
                return True
        if dec_name in _FRAMEWORK_NAMES:
            return True
    return False


def _ts_collect_name_references(source: str, language: str = "python") -> dict:
    """Tree-sitter version: map name -> list of linenos where it appears.

    Uses per-language ``_LANG_DEF_PARENT_TYPES`` to skip definition positions.
    Includes ``self.attr`` / ``cls.attr`` tracking via ``clsattr:{name}`` keys
    (Python only) and function parameter names for pytest fixture injection.
    """
    if not _HAS_TS:
        return {}
    lang_def_parents = _LANG_DEF_PARENT_TYPES.get(language)
    if lang_def_parents is None:
        return {}
    tree = _ts_parse_to_tree(source, language)
    if tree is None:
        return {}
    refs: dict = {}
    root = tree.root_node
    source_bytes = source.encode("utf-8")

    def _is_def_position(n) -> bool:
        """True if *n* is a definition position (name on left side)."""
        parent = n.parent
        if parent is None:
            return False
        pt = parent.type
        if pt in lang_def_parents:
            # For function/class definitions, check field name
            if hasattr(parent, "child_by_field_name") and parent.child_by_field_name("name") is n:
                return True
            if pt in ("function_definition", "async_function_definition", "class_definition",
                      "function_declaration", "class_declaration", "method_definition",
                      "method_declaration", "fun_declaration"):
                return parent.child_by_field_name("name") is n
            # Field-aware parents: only the binding (left) side is a
            # definition — the value/iterable side is a USE and must count
            # as a reference.  A blanket parent-type check here silently
            # dropped references like `for _x in _TABLE:` or `alias = _ORIG`,
            # making symbols used only in those positions look dead.
            if pt in ("assignment", "annotated_assignment",
                      "for_statement", "for_in_statement"):
                return parent.child_by_field_name("left") is n
            if pt == "variable_declarator":
                return parent.child_by_field_name("name") is n
            if pt == "with_item":
                # `with lock:` — an identifier directly under with_item is the
                # context-manager expression (a use); aliases live in as_pattern.
                return False
            if pt in ("expression_statement",):
                for c in (parent.children or []):
                    if c.type == "assignment":
                        return True
                return True
            if pt in ("parameters", "lambda_parameters",
                      "import_statement",
                      "import_from_statement", "alias", "typed_parameter",
                      "default_parameter", "required_parameter", "optional_parameter",
                      "lexical_declaration",
                      "catch_clause", "arrow_function", "parameter_declaration",
                      "var_declaration", "short_var_declaration", "field_declaration",
                      "receiver", "const_declaration", "formal_parameter",
                      "constructor_declaration", "value_parameter",
                      "variable_declaration", "destructured_parameter"):
                return True
        return False

    def _walk(n):
        if n is None:
            return
        if n.type == "identifier":
            if _is_def_position(n):
                # Def-position identifier — check if it's a parameter name
                # (pytest fixture injection: parameter name counts as reference)
                if n.parent and n.parent.type == "parameters":
                    name = _ts_get_text(source_bytes, n)
                    refs.setdefault(name, []).append(n.start_point[0] + 1)
            else:
                name = _ts_get_text(source_bytes, n)
                refs.setdefault(name, []).append(n.start_point[0] + 1)
        elif n.type == "type_identifier" and language == "go":
            # Go tree-sitter occasionally classifies a variable/const reference
            # inside a type-conversion-like call expression (``new(uint(X))``,
            # ``uint(X)``) as a ``type_identifier`` rather than ``identifier`` —
            # especially under partial/error trees.  Collecting these adds only
            # liveness evidence (false-positive reduction direction): a stray
            # ``uint``/``int`` hit is harmless, while a missed ``defaultMargin``
            # would be a false dead report.
            name = _ts_get_text(source_bytes, n)
            refs.setdefault(name, []).append(n.start_point[0] + 1)
        elif n.type == "attribute":
            # self.attr / cls.attr
            obj = n.child_by_field_name("object")
            attr = n.child_by_field_name("attribute")
            if obj and attr and obj.type == "identifier":
                obj_name = _ts_get_text(source_bytes, obj)
                if obj_name in ("self", "cls"):
                    attr_name = _ts_get_text(source_bytes, attr)
                    refs.setdefault(f"clsattr:{attr_name}", []).append(n.start_point[0] + 1)
        for child in (n.children or []):
            _walk(child)

    _walk(root)
    return refs


def _is_overload_decorator(dec: ast.expr) -> bool:
    """True if the decorator expression is ``@overload`` or ``@typing.overload``."""
    if isinstance(dec, ast.Name):
        return dec.id == "overload"
    if isinstance(dec, ast.Attribute):
        return dec.attr == "overload"
    return False


def _has_overload(func: ast.AST) -> bool:
    decs = getattr(func, "decorator_list", None) or []
    return any(_is_overload_decorator(d) for d in decs)


# Decorator attribute names that indicate framework-managed discovery/injection.
_FRAMEWORK_INJECTION_DECORATOR_NAMES: frozenset = frozenset({
    "fixture",
    "hookimpl",
    "hookspec",
})


def _has_framework_injection_decorator(node: ast.AST) -> bool:
    """Return True if node carries a framework-injection decorator."""
    for dec in getattr(node, "decorator_list", None) or []:
        _attr = None
        if isinstance(dec, ast.Name):
            _attr = dec.id
        elif isinstance(dec, ast.Attribute):
            _attr = dec.attr
        elif isinstance(dec, ast.Call):
            fn = dec.func
            if isinstance(fn, ast.Name):
                _attr = fn.id
            elif isinstance(fn, ast.Attribute):
                _attr = fn.attr
        if _attr in _FRAMEWORK_INJECTION_DECORATOR_NAMES:
            return True
    return False


def _extract_all_list(tree: ast.Module) -> set:
    """Return set of names registered in ``__all__`` literal, if any."""
    names: set = set()
    sentinel_dynamic = False
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "__all__":
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                value = node.value
        if value is None:
            continue
        for n in ast.walk(value):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                names.add(n.value)
            elif isinstance(n, ast.Name):
                sentinel_dynamic = True
    if sentinel_dynamic:
        names.add("*__dynamic__*")
    return names


def _is_dead_candidate(
    name: str,
    all_names: set,
    cross_file_referenced_names: Optional[set] = None,
    include_public: Optional[bool] = None,
) -> bool:
    """Decide whether a symbol can be considered for dead-code removal.

    include_public: explicit eligibility for non-``_`` names.  None keeps the
    legacy inference (public eligible iff ``cross_file_referenced_names`` was
    provided) — used by ``public_dead_code_scanner``.  ``dead_block_scanner``
    passes False so the cross-file set acts as suppression evidence only,
    matching its documented private-only contract.
    """
    if include_public is None:
        include_public = cross_file_referenced_names is not None
    if "*__dynamic__*" in all_names:
        return False
    # Blank identifier (Python ``_`` discard) is never a named definition
    # eligible for removal — it is the language's explicit "discard this
    # binding" marker.
    if name == "_":
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    if name in all_names:
        return False
    if not include_public and not name.startswith("_"):
        return False
    return True


def _is_dynamic_invocation_file(rel_path: str) -> bool:
    """True for files whose public symbols are invoked dynamically (test frameworks).

    Test functions/methods are collected by name convention, never imported
    or call-edged — public-symbol dead-code judgement there is unsound
    (measured 2026-06-12: 1784/2082 public "dead" members were test symbols).
    Private helpers in test files are still judged normally.

    Language-agnostic: matches test-file conventions for Python, Go, Java,
    Kotlin, TS/JS, and Rust.  Previously Python-only, which caused Go
    ``*_test.go`` / Java ``*Test.java`` test symbols to be flagged as dead.
    """
    parts = str(rel_path).replace("\\", "/").split("/")
    fname = parts[-1]
    # Directory-based convention is language-agnostic (tests/, test/, __tests__/).
    if any(p in ("tests", "test", "__tests__") for p in parts[:-1]):
        return True
    # Filename-based convention, dispatched by extension.
    root, ext = os.path.splitext(fname)
    ext = ext.lower()
    if ext == ".py":
        # pytest/unittest: test_*.py / *_test.py / conftest.py
        return fname.startswith("test_") or fname == "conftest.py" or fname.endswith("_test.py")
    if ext == ".go":
        # Go: *_test.go (the sole convention; symbols invoked via reflection)
        return fname.endswith("_test.go")
    if ext in (".java", ".kt"):
        # JUnit: Test*.java/kt, *Test.java/kt, *Tests.java/kt
        return root.startswith("Test") or root.endswith("Test") or root.endswith("Tests")
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        # Jest/Mocha/Vitest: *.test.ts/js, *.spec.ts/js
        return root.endswith(".test") or root.endswith(".spec")
    if ext == ".rs":
        # Rust: *_test.rs (best-effort; #[test] attribute is the real signal)
        return root.endswith("_test")
    return False


def _collect_all_defs(tree: ast.Module) -> list[tuple[str, str, int, int, Optional[str]]]:
    """Collect module-level AND class-level definitions.

    Returns (name, kind, lineno, end_lineno, enclosing_class_or_None).
    """
    out: list[tuple[str, str, int, int, Optional[str]]] = []

    def _collect_from_body(body: list, enclosing_class: Optional[str] = None):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if enclosing_class is not None:
                    continue  # skip methods
                if _has_overload(node):
                    continue
                if _has_framework_injection_decorator(node):
                    continue
                end = getattr(node, "end_lineno", node.lineno)
                deco_list = getattr(node, "decorator_list", None) or []
                start = deco_list[0].lineno if deco_list else node.lineno
                out.append((node.name, "function", start, end, None))
            elif isinstance(node, ast.ClassDef):
                end = getattr(node, "end_lineno", node.lineno)
                deco_list = getattr(node, "decorator_list", None) or []
                start = deco_list[0].lineno if deco_list else node.lineno
                out.append((node.name, "class", start, end, None))
                # Recurse into class body for assignments
                _collect_from_body(list(node.body), enclosing_class=node.name)
            elif isinstance(node, ast.Assign):
                if enclosing_class is not None:
                    continue  # class-level assignments are API-contract definitions
                for tgt in node.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    end = getattr(node, "end_lineno", node.lineno)
                    out.append((tgt.id, "assignment", node.lineno, end, None))
            elif isinstance(node, ast.AnnAssign):
                if enclosing_class is not None:
                    continue  # same rationale as above
                if not isinstance(node.target, ast.Name):
                    continue
                end = getattr(node, "end_lineno", node.lineno)
                out.append((node.target.id, "assignment", node.lineno, end, None))

    _collect_from_body(list(tree.body))
    return out


def _collect_name_references(tree: ast.Module) -> dict:
    """Map name -> list of linenos where it appears as a Load reference.

    Includes ``self.attr`` / ``cls.attr`` tracking via ``clsattr:{name}`` keys
    and function parameter names (``ast.arg``) for pytest fixture injection.
    """
    refs: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs.setdefault(node.id, []).append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.arg):
            # Parameter names: def test_foo(agent_loop): ... references agent_loop
            refs.setdefault(node.arg, []).append(getattr(node, "lineno", 0))
        elif isinstance(node, ast.Attribute):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                refs.setdefault(base.id, []).append(getattr(node, "lineno", 0))
                # self.attr / cls.attr -> also register as clsattr:attr_name
                if base.id in ("self", "cls"):
                    refs.setdefault(
                        f"clsattr:{node.attr}", []
                    ).append(getattr(node, "lineno", 0))
    return refs


def _is_externally_referenced(
    name: str,
    def_start: int,
    def_end: int,
    references: dict,
    cross_file_referenced_names: Optional[set] = None,
    is_class_attr: bool = False,
) -> bool:
    """True iff ``name`` is referenced outside [def_start, def_end]."""
    locs = (references.get(name) or [])[:]
    if is_class_attr:
        locs += references.get(f"clsattr:{name}") or []
    for ln in locs:
        if ln < def_start or ln > def_end:
            return True
    if cross_file_referenced_names and name in cross_file_referenced_names:
        return True
    return False


def _cluster_dead_members(
    dead_members: list[DeadBlockMember],
    gap_tol: int,
) -> tuple[list[list[DeadBlockMember]], set]:
    """Group dead members into clusters of adjacent definitions.

    Returns (clusters, clustered_member_keys) where each cluster has >=2
    members and clustered_member_keys is a set of (lineno, name) tuples
    for members that were placed into a cluster.
    """
    if not dead_members:
        return [], set()
    dead_members.sort(key=lambda m: m.lineno)
    clusters: list[list[DeadBlockMember]] = []
    clustered_members: set = set()
    current: list[DeadBlockMember] = [dead_members[0]]
    for m in dead_members[1:]:
        gap = m.lineno - current[-1].end_lineno
        if gap <= gap_tol:
            current.append(m)
        else:
            if len(current) >= 2:
                clusters.append(current)
                for cm in current:
                    clustered_members.add((cm.lineno, cm.name))
            current = [m]
    if len(current) >= 2:
        clusters.append(current)
        for cm in current:
            clustered_members.add((cm.lineno, cm.name))
    return clusters, clustered_members


# ── Shared scan core ─────────────────────────────────────────────────────────

def scan_dead_block_core(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int,
    cluster_gap_tolerance: Optional[int],
    cross_file_referenced_names: Optional[set],
    singleton_confidence: float,
    mark_public: bool,
    log_tag: str,
    include_public: Optional[bool] = None,
) -> tuple[list[DeadBlockCandidate], int]:
    """Shared scan loop behind ``scan_dead_blocks`` and ``scan_public_dead_blocks``.

    The two scanners were ~90% identical (structural similarity 0.968); the
    only real differences are parameterised here:
      - ``singleton_confidence``: 0.65 (private-only) vs 0.55 (public-capable)
      - ``mark_public``: whether to compute ``includes_public`` on candidates
      - ``log_tag``: log prefix

    Candidate semantics (``_is_dead_candidate``): private symbols are always
    candidates; public symbols only when ``cross_file_referenced_names`` is
    provided AND the name is absent from it.

    Returns ``(candidates, truncated_cluster_count)``.
    """
    from ..languages import LanguageId as _LanguageId
    from . import parse_cache as _pc

    gap_tol = cluster_gap_tolerance if cluster_gap_tolerance is not None else CLUSTER_GAP_TOLERANCE
    candidates: list[DeadBlockCandidate] = []
    truncated_total = 0

    def _pub(members: list[DeadBlockMember]) -> bool:
        if not mark_public:
            return False
        return any(not m.name.startswith("_") for m in members)


    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)
        src = _pc.read_source(abs_path)
        if src is None:
            continue

        _lang_id = _LanguageId.from_path(rel_path)
        _lang = _lang_id.value if _lang_id is not None else "python"

        # ── Primary: tree-sitter (language-agnostic) ──
        if _HAS_TS:
            # tree-sitter is error-tolerant; a partial tree from broken source
            # would under-count references and produce false positives.
            _pre_tree = _ts_parse_to_tree(src, _lang)
            if _pre_tree is None or _pre_tree.root_node.has_error:
                continue
            all_names = _ts_extract_all_list(src, language=_lang)
            if "*__dynamic__*" in all_names:
                continue
            defs = _ts_collect_all_defs(src, language=_lang)
            references = _ts_collect_name_references(src, language=_lang)
        else:
            # ── Fallback: AST (Python only) ──
            if _lang != "python":
                continue
            tree = _pc.parse_ast(abs_path)
            if tree is None:
                continue
            all_names = _extract_all_list(tree)
            if "*__dynamic__*" in all_names:
                continue
            defs = _collect_all_defs(tree)
            references = _collect_name_references(tree)

        _dynamic_invocation = _is_dynamic_invocation_file(rel_path)

        _effective_cross = cross_file_referenced_names

        dead_members: list[DeadBlockMember] = []
        for name, kind, lineno, end_lineno, enclosing_class in defs:
            # Class-level assignments are API-contract definitions — cross-file
            # instance/mixin attribute access (self._FOO from a sibling mixin
            # file) is invisible to single-file analysis (see module docstring).
            # The AST fallback never collects these; the tree-sitter collector
            # does (other languages need class fields), so filter here.
            if _lang == "python" and kind == "assignment" and enclosing_class is not None:
                continue
            # Test files: public symbols are pytest-invoked by convention —
            # never judge them dead even in public mode.
            if _dynamic_invocation and not name.startswith("_"):
                continue
            if not _is_dead_candidate(
                name, all_names, _effective_cross,
                include_public=include_public,
            ):
                continue
            if _is_externally_referenced(
                name, lineno, end_lineno, references,
                cross_file_referenced_names=_effective_cross,
                is_class_attr=(kind == "class_assignment"),
            ):
                continue
            dead_members.append(DeadBlockMember(
                name=name, symbol_kind=kind,
                lineno=lineno, end_lineno=end_lineno,
                enclosing_class=enclosing_class,
            ))

        if len(dead_members) < 2:
            if len(dead_members) == 1:
                m = dead_members[0]
                candidates.append(DeadBlockCandidate(
                    file=rel_path,
                    members=[m],
                    cluster_start=m.lineno,
                    cluster_end=m.end_lineno,
                    confidence=singleton_confidence,
                    is_singleton=True,
                    includes_public=_pub([m]),
                ))
            continue

        clusters, clustered_members = _cluster_dead_members(dead_members, gap_tol)

        emitted = 0
        for cluster in clusters:
            candidates.append(DeadBlockCandidate(
                file=rel_path,
                members=list(cluster),
                cluster_start=cluster[0].lineno,
                cluster_end=cluster[-1].end_lineno,
                confidence=1.0,
                includes_public=_pub(cluster),
            ))
            emitted += 1
            if emitted >= max_per_file:
                _remaining = len(clusters) - emitted
                truncated_total += _remaining
                logger.warning(
                    "[%s] %s: hit max_per_file=%d, truncating %d remaining cluster(s)",
                    log_tag, rel_path, max_per_file, _remaining,
                )
                break

        # Emit remaining singletons (not clustered) with lower confidence.
        _singleton_emitted = 0
        for m in dead_members:
            if (m.lineno, m.name) in clustered_members:
                continue
            candidates.append(DeadBlockCandidate(
                file=rel_path,
                members=[m],
                cluster_start=m.lineno,
                cluster_end=m.end_lineno,
                confidence=singleton_confidence,
                is_singleton=True,
                includes_public=_pub([m]),
            ))
            _singleton_emitted += 1
            if emitted + _singleton_emitted >= max_per_file:
                break

    if candidates:
        logger.info(
            "[%s] %d cluster(s) across %d file(s); total dead symbols=%d (public=%s)",
            log_tag, len(candidates), len({c.file for c in candidates}),
            sum(len(c.members) for c in candidates),
            any(c.includes_public for c in candidates),
        )

    return candidates, truncated_total
