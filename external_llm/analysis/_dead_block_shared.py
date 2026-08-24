"""Shared utilities for dead-block scanners.

Consolidates data models and AST helpers used by both
``dead_block_scanner`` and ``public_dead_code_scanner``
to eliminate ~200 lines of duplicated code.
Also re-used by ``duplicate_definition_scanner`` for
``_is_overload_decorator`` / ``_has_overload``.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass, field

from ..common.atomic_io import atomic_write_json
from ..languages.tree_sitter_utils import (
    get_node_text as _ts_get_text,
)
from ..languages.tree_sitter_utils import (
    parse_to_tree as _ts_parse_to_tree,
)
from ..languages.tree_sitter_utils import (
    query_matches as _ts_query_matches,
)

logger = logging.getLogger(__name__)

# ── Tree-sitter availability ─────────────────────────────────────────────
# tree_sitter_utils guards its own optional tree-sitter import, so this
# module always imports — the old try/except ImportError was dead code.
_HAS_TS = True

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
        # Wrapper-less assignment: the language-pack Python grammar puts
        # `assignment` directly under `module`, where the standalone
        # tree-sitter-python package interposes `expression_statement`. Both must
        # be listed or module-level assignments are invisible on the grammar every
        # real install actually gets. No double-count: `is_container=False` stops
        # the walk at the outer node, so the nested `assignment` under an
        # `expression_statement` is never visited.
        # Deliberately NOT in _LANG_ASSIGN_WRAPPERS — a bare `assignment` holds its
        # name identifier as a DIRECT child (field `left`), so the plain
        # _ts_child_by_type() lookup already finds it.
        "assignment": ("assignment", False, False),
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
        "function_declaration": (
            "function",
            False,
            True,
        ),  # fwcd kotlin grammar (language-pack AND standalone) — "fun_declaration" was a stale fork variant
        "property_declaration": ("assignment", False, False),
    },
}

# ── Import-time map validation ──────────────────────────────────────────────
# _LANG_DEF_NODES is hand-maintained per-language node data (grammar facts),
# so its SHAPE is validated at import: a typo'd kind or a missing flag would
# silently change dead-block reachability semantics.  The kind vocabulary is
# the one documented in the map's comment above.
_DEF_NODE_KINDS = frozenset({"function", "class", "assignment", "interface", "type_alias", "enum"})


def _validate_lang_def_nodes() -> None:
    for lang, defs in _LANG_DEF_NODES.items():
        for node_type, spec in defs.items():
            if (
                not isinstance(spec, tuple)
                or len(spec) != 3
                or not isinstance(spec[0], str)
                or spec[0] not in _DEF_NODE_KINDS
                or not isinstance(spec[1], bool)
                or not isinstance(spec[2], bool)
            ):
                raise ValueError(
                    f"_LANG_DEF_NODES[{lang!r}][{node_type!r}] = {spec!r} must be "
                    f"(kind, is_container, skip_if_enclosed) with kind in "
                    f"{sorted(_DEF_NODE_KINDS)}"
                )


_validate_lang_def_nodes()

# Per-language assignment wrapper nodes — the name identifier is nested one level
# deeper inside an inner node (e.g. expression_statement → assignment → identifier,
# lexical_declaration → variable_declarator → identifier).  The walk's wrapper
# branch is reached ONLY for node types in _LANG_DEF_NODES, so a wrapper that is
# not a def node is a dead entry — _validate_lang_assign_wrappers enforces it.
_LANG_ASSIGN_WRAPPERS: dict[str, set] = {
    "python": {"expression_statement", "annotated_assignment"},
    # "expression_statement" deliberately absent for TS: no TS grammar variant
    # makes it a def node (module-level `x = 1` is an assignment_expression —
    # a USE, not a binding), so a wrapper entry could never fire.
    "typescript": {"lexical_declaration", "variable_declaration"},
    "javascript": {"lexical_declaration", "variable_declaration"},
    "go": {"type_declaration", "var_declaration", "const_declaration", "type_spec"},
    "java": {"field_declaration"},
    "kotlin": {"property_declaration"},
}
_LANG_DEF_PARENT_TYPES: dict[str, set] = {
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "assignment",
        "annotated_assignment",
        "parameters",
        "lambda_parameters",
        "for_statement",
        "with_item",
        "import_statement",
        "import_from_statement",
        "alias",
        "typed_parameter",
        "default_parameter",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "required_parameter",
        "optional_parameter",
        "variable_declarator",
        "lexical_declaration",
        "for_in_statement",
        "catch_clause",
        "arrow_function",
        "assignment",
        # interface/type-alias/enum names are blanket-name-field
        # handled (probe-verified on the ACTIVE tree-sitter-typescript
        # grammar: interface_declaration/type_alias_declaration name =
        # type_identifier, enum_declaration name = identifier) — they
        # are listed so the reverse validator sees the vocabulary as
        # live: the reference walk visits none of these name nodes for
        # TS (type_identifier is collected for go/kotlin only), so
        # listing them is behavior-neutral but keeps the branch
        # vocabulary honest.
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "variable_declarator",
        "lexical_declaration",
        "for_in_statement",
        "catch_clause",
        "arrow_function",
        "assignment",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_spec",
        "parameter_declaration",
        "var_declaration",
        "short_var_declaration",
        "field_declaration",
        "receiver",
        "const_declaration",
        "const_spec",
        "var_spec",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "formal_parameter",
        "variable_declarator",
        "field_declaration",
        "constructor_declaration",
    },
    # kotlin: the fwcd grammar (language-pack AND standalone) names bindings via
    # unnamed children (simple_identifier / type_identifier), never a `name`
    # field — object/companion/parameter/property names have no handling shape
    # and are deliberately absent (they fall through to "reference", the safe
    # direction; the property NAME itself lives in a nested variable_declaration
    # child).  "fun_declaration" was a stale fork variant; the active grammar
    # emits function_declaration.  "value_parameter"/"destructured_parameter"
    # were fork-variant node types too — the active grammar emits a plain
    # "parameter" (unnamed children, no handling shape) and never those.
    "kotlin": {"class_declaration", "function_declaration", "variable_declaration"},
}

# ── Definition-position parent-type vocabulary (SSOT for _is_def_position) ───
# _is_def_position() decides binding vs reference from an identifier's DIRECT
# parent node type.  The branch conditions below are module constants so the
# import-time validator can enforce that every _LANG_DEF_PARENT_TYPES entry is
# covered by a branch — an uncovered parent type silently counts its bindings
# as references and dead-block under-reports (never flags a symbol that looks
# referenced).
_DEF_PARENT_NAME_FIELD_TYPES = frozenset(
    {
        # parent.child_by_field_name("name") == n (also subsumed by the blanket
        # name-field check at the top of _is_def_position)
        "function_definition",
        "async_function_definition",
        "class_definition",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "method_declaration",
        "variable_declarator",
    }
)
_DEF_PARENT_LEFT_FIELD_TYPES = frozenset(
    {
        # parent.child_by_field_name("left") == n — only the binding side is a
        # definition; the value/iterable side is a USE and must count as a
        # reference.
        "assignment",
        "annotated_assignment",
        "for_statement",
        "for_in_statement",
    }
)
_DEF_PARENT_BINDING_TYPES = frozenset(
    {
        # BINDING-aware branch: name field, else first/last identifier child.
        # NB: reachable only for types listed in some language's
        # _LANG_DEF_PARENT_TYPES (the `pt in lang_def_parents` gate) —
        # value_parameter / destructured_parameter were removed with their last
        # (kotlin) map reference: the fwcd grammar emits "parameter", never them.
        "parameters",
        "lambda_parameters",
        "import_statement",
        "import_from_statement",
        "alias",
        "typed_parameter",
        "default_parameter",
        "required_parameter",
        "optional_parameter",
        "lexical_declaration",
        "catch_clause",
        "arrow_function",
        "parameter_declaration",
        "var_declaration",
        "short_var_declaration",
        "field_declaration",
        "receiver",
        "const_declaration",
        "formal_parameter",
        "constructor_declaration",
        "variable_declaration",
    }
)
_DEF_PARENT_SPECIAL_TYPES = frozenset(
    {
        # with_item → always a use (`with lock:`; aliases live in as_pattern).
        # "expression_statement" was removed with its _is_def_position branch: no
        # language map lists it as a def-PARENT (python's map has it as a def-NODE
        # wrapper only), so the branch was globally unreachable — the reverse
        # validator below fails at import if such dead vocabulary reappears.
        "with_item",
    }
)
_DEF_PARENT_GRAMMAR_NAME_FIELD = frozenset(
    {
        # No explicit branch — covered by the blanket child_by_field_name("name")
        # check (name field probe-verified on the ACTIVE grammars:
        # tree-sitter-typescript / tree-sitter-go).
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "type_spec",
        "const_spec",
        "var_spec",
    }
)
_DEF_PARENT_COVERED = (
    _DEF_PARENT_NAME_FIELD_TYPES
    | _DEF_PARENT_LEFT_FIELD_TYPES
    | _DEF_PARENT_BINDING_TYPES
    | _DEF_PARENT_SPECIAL_TYPES
    | _DEF_PARENT_GRAMMAR_NAME_FIELD
)


# ── Import-time cross-map validation (wrappers / def parents) ────────────────
def _validate_lang_assign_wrappers() -> None:
    """Fail at import if _LANG_ASSIGN_WRAPPERS drifts from _LANG_DEF_NODES.

    The wrapper branch in ``_ts_collect_all_defs`` sits inside ``def_info is
    not None`` — it is reached ONLY for node types present in
    ``_LANG_DEF_NODES[language]``.  A wrapper entry for any other node type is
    therefore dead (it can never fire) and would hide a node-type removal.
    """
    def_keys = set(_LANG_DEF_NODES)
    wrapper_keys = set(_LANG_ASSIGN_WRAPPERS)
    if wrapper_keys != def_keys:
        raise ValueError(
            f"_LANG_ASSIGN_WRAPPERS languages {sorted(wrapper_keys)} != _LANG_DEF_NODES languages {sorted(def_keys)}"
        )
    for lang, wrappers in _LANG_ASSIGN_WRAPPERS.items():
        if not isinstance(wrappers, set) or not wrappers or not all(isinstance(w, str) and w for w in wrappers):
            raise ValueError(
                f"_LANG_ASSIGN_WRAPPERS[{lang!r}] must be a non-empty set of non-empty strings, got {wrappers!r}"
            )
        unknown = wrappers - set(_LANG_DEF_NODES[lang])
        if unknown:
            raise ValueError(
                f"_LANG_ASSIGN_WRAPPERS[{lang!r}] entries {sorted(unknown)} are "
                f"not _LANG_DEF_NODES[{lang!r}] node types — the walk's wrapper "
                "branch is only reached for def-node types, so these are dead "
                "entries"
            )


def _validate_lang_def_parent_types() -> None:
    """Fail at import if _LANG_DEF_PARENT_TYPES drifts from the branch vocabulary.

    FORWARD direction: ``_is_def_position`` returns False for a parent type with
    no handling branch, which silently counts the binding as a reference and
    makes dead-block under-report (a symbol that looks referenced is never
    flagged).  Every map entry must be covered by the blanket name-field check
    (``_DEF_PARENT_GRAMMAR_NAME_FIELD`` vocabulary) or one of the explicit
    branch sets.

    REVERSE direction: the branch vocabulary is reachable ONLY through the
    ``pt in lang_def_parents`` gate, so a covered item that no language map
    lists can never fire — it is dead vocabulary that silently drifts (e.g. a
    stale fork-grammar node type like kotlin's ``value_parameter``, or a
    def-parent entry no map ever lists like ``expression_statement``).  A
    covered-but-unreferenced item must be either listed by the map(s) whose
    grammar emits it or removed together with its branch.
    """
    def_keys = set(_LANG_DEF_NODES)
    parent_keys = set(_LANG_DEF_PARENT_TYPES)
    if parent_keys != def_keys:
        raise ValueError(
            f"_LANG_DEF_PARENT_TYPES languages {sorted(parent_keys)} != _LANG_DEF_NODES languages {sorted(def_keys)}"
        )
    for lang, parents in _LANG_DEF_PARENT_TYPES.items():
        if not isinstance(parents, set) or not parents or not all(isinstance(p, str) and p for p in parents):
            raise ValueError(
                f"_LANG_DEF_PARENT_TYPES[{lang!r}] must be a non-empty set of non-empty strings, got {parents!r}"
            )
        unknown = parents - _DEF_PARENT_COVERED
        if unknown:
            raise ValueError(
                f"_LANG_DEF_PARENT_TYPES[{lang!r}] entries {sorted(unknown)} have "
                "no handling branch in _is_def_position (blanket name-field "
                "check or explicit branch) — bindings under them would silently "
                "count as references (dead-block under-reports)"
            )
    # REVERSE: every covered vocabulary item must be referenced by at least one
    # language map — otherwise its branch can never fire (dead vocabulary).
    referenced = set().union(*_LANG_DEF_PARENT_TYPES.values())
    dead_vocab = _DEF_PARENT_COVERED - referenced
    if dead_vocab:
        raise ValueError(
            f"_DEF_PARENT_COVERED entries {sorted(dead_vocab)} are referenced by "
            "no _LANG_DEF_PARENT_TYPES map — their _is_def_position branches can "
            "never fire (dead vocabulary). Remove the branch/entry or list the "
            "type in the language map(s) whose grammar emits it."
        )


_validate_lang_assign_wrappers()
_validate_lang_def_parent_types()


CLUSTER_GAP_TOLERANCE = 5  # max blank-line gap between adjacent dead defs


# ── Data models ──────────────────────────────────────────────────────────


@dataclass
class DeadBlockMember:
    name: str
    symbol_kind: str  # "function" | "class" | "assignment" | "class_assignment"
    lineno: int
    end_lineno: int
    enclosing_class: str | None = None


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
    #
    # Both module-child shapes are matched. Python tree-sitter grammars disagree
    # on whether a statement-position assignment is wrapped: the standalone
    # ``tree-sitter-python`` package yields
    # ``module → expression_statement → assignment``, while the grammar bundled in
    # ``tree-sitter-language-pack`` yields ``module → assignment`` with no wrapper.
    # Only the pack is a declared dependency, so a wrapper-only query silently
    # matched NOTHING for every real install — ``__all__`` was ignored outright
    # and public re-exports got reported as dead code. The alternation keeps the
    # ``(module ...)`` anchor, so this stays equivalent to the ``ast`` twin
    # ``_extract_all_list``, which only scans ``tree.body``.
    query_str = """
(module [
  (expression_statement (assignment
    left: (identifier) @name
    right: (list (string (string_content) @item))))
  (assignment
    left: (identifier) @name
    right: (list (string (string_content) @item)))
 ] @def
 (#eq? @name "__all__"))
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
(module [
  (expression_statement (assignment left: (identifier) @name))
  (assignment left: (identifier) @name)
 ] @def
 (#eq? @name "__all__"))
"""
        if _ts_query_matches(source, language, any_all_query):
            names.add("*__dynamic__*")
    return names


def _ts_collect_all_defs(
    source: str,
    language: str = "python",
) -> list[tuple[str, str, int, int, str | None]]:
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
    out: list[tuple[str, str, int, int, str | None]] = []
    root = tree.root_node

    def _walk(node, enclosing_class: str | None = None):
        if not node:
            return
        node_type = node.type
        def_info = lang_defs.get(node_type)
        if def_info is not None:
            kind, is_container, skip_if_enclosed = def_info
            if skip_if_enclosed and enclosing_class is not None:
                return  # skip methods / inner definitions
            # Find the name identifier — may be nested inside assignment wrapper
            name_node = _ts_def_name_node(node)
            if name_node is None:
                # Assignment wrapper (e.g. expression_statement wrapping an
                # assignment node that contains the identifier).  The wrapper
                # can also wrap a bare expression (module-level call
                # ``_init()``, ternary, ...) whose identifiers are USES, not
                # definitions — only the binding side (fields ``name``/``left``)
                # may define a symbol.  Scanning any identifier inside a
                # ``call`` child misregisters the callee as an assignment def
                # whose only reference is its own line → false dead reports
                # (5 real cases across the repo, 2026-08).
                in_wrappers = language in _LANG_ASSIGN_WRAPPERS and node_type in _LANG_ASSIGN_WRAPPERS[language]
                if in_wrappers:
                    for child in node.children or []:
                        name_node = _ts_def_name_node(child, fields_only=True)
                        if name_node is None and child.type in _NAME_HOLDER_CHILD_TYPES:
                            # kotlin fwcd: name nested in variable_declaration
                            name_node = _ts_name_inside(child)
                        if name_node is not None:
                            break
            if name_node is None:
                return
            name = _ts_get_text(source.encode("utf-8"), name_node)
            # Check overload / framework decorators via manual walk
            if language == "python" and _ts_has_overload_or_framework(node, source):
                return
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1  # type: ignore[attr-defined]  # tree-sitter node
            decorator_node = _ts_child_by_type(node, ("decorator",))
            if decorator_node is not None:
                start = decorator_node.start_point[0] + 1  # type: ignore[attr-defined]  # tree-sitter node
            out.append((name, kind, start, end, enclosing_class))
            if is_container:
                for child in node.children or []:
                    _walk(child, enclosing_class=name)

        elif language == "python" and node_type == "decorated_definition":
            if _ts_has_overload_or_framework(node, source):
                return
        else:
            for child in node.children or []:
                _walk(child, enclosing_class)

    _walk(root)
    return out


def _ts_child_by_type(node, type_names: tuple[str, ...]) -> object | None:
    """Find first child with one of *type_names* (tree-sitter node helper)."""
    for child in node.children or []:
        if child.type in type_names:
            return child
    return None


# Definition nodes whose binding is a *field* (``name`` for declarations,
# ``left`` for assignments) — never a bare identifier scan.  Scanning any
# identifier inside these misreads USE positions as definitions: a module-level
# call ``_init()`` registers the callee as an assignment def, and ``x.y = z``
# registers the RHS identifier ``z``.  The resulting fake def's only reference
# is its own line, so it is reported dead (5 real false positives, 2026-08).
_ASSIGNMENT_LIKE_DEF_NODES = frozenset(
    {
        "assignment",
        "annotated_assignment",  # python
        "variable_declarator",
        "lexical_declaration",
        "variable_declaration",  # typescript / javascript
        "var_spec",
        "const_spec",
        "var_declaration",
        "const_declaration",  # go
        "field_declaration",  # java
        "property_declaration",  # kotlin
    }
)


# Identifier-like node types that can carry a binding name across the
# supported grammars: plain ``identifier`` (python/go/java/TS/JS),
# ``type_identifier`` (TS interface/type-alias names, kotlin type names),
# ``field_identifier`` (go method names) and ``simple_identifier`` (kotlin
# fwcd grammar — names are UNNAMED children, no ``name`` field).  The old
# ``== "identifier"`` restriction silently dropped go method names, TS
# interface/type-alias names and every kotlin def.
_NAME_NODE_TYPES = ("identifier", "type_identifier", "field_identifier", "simple_identifier")

# Wrapper children that HOLD the binding name as an unnamed identifier child
# (no ``name``/``left`` field): kotlin fwcd nests it as property_declaration →
# binding_pattern_kind → variable_declaration → simple_identifier.  These
# declaration containers are safe to search recursively — the assignment-like
# guard exists precisely for expression-bearing nodes (callee/RHS hazard).
_NAME_HOLDER_CHILD_TYPES = frozenset({"variable_declaration", "binding_pattern_kind"})


def _ts_name_inside(node) -> object | None:
    """First identifier-like child of a kotlin name-holder node (fwcd grammar)."""
    for child in node.children or []:
        if child.type in _NAME_NODE_TYPES:
            return child
        if child.type in _NAME_HOLDER_CHILD_TYPES:
            found = _ts_name_inside(child)
            if found is not None:
                return found
    return None


def _ts_def_name_node(node, fields_only: bool = False) -> object | None:
    """Return the binding-side identifier node of a definition node.

    Prefers field-aware lookup (``name`` for declarations, ``left`` for
    assignments).  Assignment-like nodes whose binding is not a plain
    identifier (``x.y = z``, ``a, b = ...``) define NO collectible name, and a
    bare expression (module-level call ``_init()``) has no binding at all — for
    those, the first-identifier fallback must not substitute (it picks up
    callee/RHS identifiers as fake definitions).  ``fields_only`` forces the
    strict path (used for wrapper children).
    """
    if hasattr(node, "child_by_field_name"):
        for fld in ("name", "left"):
            fn = node.child_by_field_name(fld)
            if fn is not None and fn.type in _NAME_NODE_TYPES:
                return fn
    if fields_only or node.type in _ASSIGNMENT_LIKE_DEF_NODES:
        return None
    return _ts_child_by_type(node, _NAME_NODE_TYPES)


def _ts_has_overload_or_framework(node, source: str) -> bool:
    """Check if node has @overload or @fixture/@hookimpl/@hookspec decorator."""
    _framework_names = {"fixture", "hookimpl", "hookspec"}
    for child in node.children or []:
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
            if attr in _framework_names:
                return True
        if dec_name in _framework_names:
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
            if hasattr(parent, "child_by_field_name") and parent.child_by_field_name("name") == n:
                return True
            if pt in _DEF_PARENT_NAME_FIELD_TYPES:
                return parent.child_by_field_name("name") == n
            # Field-aware parents: only the binding (left) side is a
            # definition — the value/iterable side is a USE and must count
            # as a reference.  A blanket parent-type check here silently
            # dropped references like `for _x in _TABLE:` or `alias = _ORIG`,
            # making symbols used only in those positions look dead.
            if pt in _DEF_PARENT_LEFT_FIELD_TYPES:
                return parent.child_by_field_name("left") == n
            if pt == "with_item":
                # `with lock:` — an identifier directly under with_item is the
                # context-manager expression (a use); aliases live in as_pattern.
                return False
            if pt in _DEF_PARENT_BINDING_TYPES:
                # BINDING-aware: only the name (binding) side is a definition —
                # the default-value / type / import-name side is a USE and must
                # count as a reference.  A blanket True here dropped references
                # like `def _verify(code=_SAMPLE_CODE):`) and reported live symbols
                # dead.  NB: tree-sitter returns a fresh Node object per access,
                # so identity (`is`) never holds across calls — compare with
                # structural equality (`==`).
                if hasattr(parent, "child_by_field_name"):
                    nf = parent.child_by_field_name("name")
                    if nf is not None:
                        return nf == n
                ids = [c for c in (parent.children or []) if c.type == "identifier"]
                if not ids:
                    return False
                return (ids[-1] if pt == "formal_parameter" else ids[0]) == n
        return False

    def _walk(n):
        if n is None:
            return
        # kotlin fwcd grammar emits every value identifier as
        # ``simple_identifier`` (never plain ``identifier``) — without this
        # branch kotlin reference collection was completely empty.
        if n.type == "identifier" or (language == "kotlin" and n.type == "simple_identifier"):
            if _is_def_position(n):
                # Def-position identifier — check if it's a parameter name
                # (pytest fixture injection: parameter name counts as reference)
                if n.parent and n.parent.type == "parameters":
                    name = _ts_get_text(source_bytes, n)
                    refs.setdefault(name, []).append(n.start_point[0] + 1)
            else:
                name = _ts_get_text(source_bytes, n)
                refs.setdefault(name, []).append(n.start_point[0] + 1)
        elif n.type == "type_identifier" and language in ("go", "kotlin"):
            # Go tree-sitter occasionally classifies a variable/const reference
            # inside a type-conversion-like call expression (``new(uint(X))``,
            # ``uint(X)``) as a ``type_identifier`` rather than ``identifier`` —
            # especially under partial/error trees.  Collecting these adds only
            # liveness evidence (false-positive reduction direction): a stray
            # ``uint``/``int`` hit is harmless, while a missed ``defaultMargin``
            # would be a false dead report.
            # kotlin fwcd class/interface/object names AND their type-annotation
            # usages (``val x: Foo``) are ``type_identifier`` — the same
            # liveness-evidence direction.
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
        for child in n.children or []:
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
_FRAMEWORK_INJECTION_DECORATOR_NAMES: frozenset = frozenset(
    {
        "fixture",
        "hookimpl",
        "hookspec",
    }
)


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
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
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
    cross_file_referenced_names: set | None = None,
    include_public: bool | None = None,
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
    return include_public or name.startswith("_")


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
        return root.startswith("Test") or root.endswith(("Test", "Tests"))
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        # Jest/Mocha/Vitest: *.test.ts/js, *.spec.ts/js
        return root.endswith((".test", ".spec"))
    if ext == ".rs":
        # Rust: *_test.rs (best-effort; #[test] attribute is the real signal)
        return root.endswith("_test")
    return False


def _collect_all_defs(tree: ast.Module) -> list[tuple[str, str, int, int, str | None]]:
    """Collect module-level AND class-level definitions.

    Returns (name, kind, lineno, end_lineno, enclosing_class_or_None).
    """
    out: list[tuple[str, str, int, int, str | None]] = []

    def _collect_from_body(body: list, enclosing_class: str | None = None):
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
                    refs.setdefault(f"clsattr:{node.attr}", []).append(getattr(node, "lineno", 0))
    return refs


def _is_externally_referenced(
    name: str,
    def_start: int,
    def_end: int,
    references: dict,
    cross_file_referenced_names: set | None = None,
    is_class_attr: bool = False,
) -> bool:
    """True iff ``name`` is referenced outside [def_start, def_end]."""
    locs = (references.get(name) or [])[:]
    if is_class_attr:
        locs += references.get(f"clsattr:{name}") or []
    for ln in locs:
        if ln < def_start or ln > def_end:
            return True
    return bool(cross_file_referenced_names and name in cross_file_referenced_names)


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


# ── Per-file extraction cache (dead_block / public_dead_code share it) ──────
# The extraction block inside ``scan_dead_block_core`` (parse + ``__all__`` +
# defs + references) is a PURE function of file content and is IDENTICAL for
# dead_block_scanner and public_dead_code_scanner — they share the core and
# differ only in the confidence/visibility FILTERING applied afterwards.  Each
# scanner previously re-parsed every file with tree-sitter (~3.2s) and
# re-walked every tree for name references (~4.3s); the pair paid 2x per gate
# run.  The extraction is therefore cached per file under a
# ``(st_mtime_ns, st_size)`` fingerprint — the same invalidation contract as
# the vulture scan cache — and persisted to ``.cache`` so a cache-hot run
# skips BOTH scanners' extraction (~12s → ~2s on this repo, 2026-08-16).
#
# Cache-file version: bump ``_DBX_CACHE_VERSION`` when the collectors'
# semantics change (def/reference shapes), or a stale cache would silently
# serve pre-change extraction results.  Corruption / version mismatch /
# read-write errors all fail OPEN to a full extraction (never wrong results).
_DBX_CACHE_VERSION = 1


def _dbx_cache_path(repo_root: str) -> str:
    from . import parse_cache as _pc

    return _pc.cache_file_path(repo_root, f"dead_block_extract_v{_DBX_CACHE_VERSION}.json")


def _dbx_load(repo_root: str) -> tuple[dict, bool]:
    """Load the extraction cache for *repo_root*; returns ``(cache, dirty=False)``.

    Fail-open: any read/parse error or version mismatch returns an empty
    cache — the caller recomputes everything and rewrites the file.  Values
    are ``abs_path -> (fingerprint, payload)`` where ``payload`` is None for
    a persisted "skip this file" decision and otherwise the serialized
    ``(lang, all_names, defs, references)`` extraction (all_names restored
    to a set so membership checks stay O(1)).

    An empty *repo_root* (unit-test convention) bypasses the cache entirely —
    the cache file would otherwise land in the CWD and grow with throwaway
    temp files.
    """
    if not repo_root:
        return {}, False
    cache_path = _dbx_cache_path(repo_root)  # outside try: CachePathError must propagate
    try:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != _DBX_CACHE_VERSION:
            return {}, False
        cache: dict = {}
        for path, entry in (payload.get("files") or {}).items():
            if not isinstance(entry, dict) or not isinstance(entry.get("fp"), list):
                continue
            pl = entry.get("payload")
            if pl is not None:
                pl = (pl.get("lang"), set(pl.get("all") or ()), pl.get("defs") or (), pl.get("refs") or {})
            cache[path] = (tuple(entry["fp"]), pl)
    except (OSError, ValueError, TypeError):
        logger.debug("dead-block extraction cache unreadable — full extraction", exc_info=True)
        return {}, False
    return cache, False


def _dbx_save(repo_root: str, cache: dict) -> None:
    """Persist *cache* to disk atomically (best-effort; failure costs a re-extraction).

    Whole-file replace via the canonical :func:`atomic_write_json` (B2): a
    crash mid-save — or another process saving the same file concurrently —
    can never leave a truncated cache; the previous one stays loadable.  See
    the disk-cache concurrency policy in ``parse_cache`` (lock-free,
    last-writer-wins).

    Empty *repo_root* skips the write (see ``_dbx_load`` — unit tests run
    with ``repo_root=""`` and must not pollute the CWD's ``.cache``).
    """
    if not repo_root:
        return
    try:
        cache_path = _dbx_cache_path(repo_root)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        files = {}
        for abs_path, (fp, pl) in cache.items():
            if pl is None:
                files[abs_path] = {"fp": list(fp), "payload": None}
                continue
            _lang, all_names, defs, references = pl
            files[abs_path] = {
                "fp": list(fp),
                "payload": {
                    "lang": _lang,
                    "all": sorted(all_names),
                    "defs": defs,
                    "refs": references,
                },
            }
        atomic_write_json(
            cache_path,
            {"format": _DBX_CACHE_VERSION, "files": files},
            indent=None,
            ensure_ascii=True,
        )
    except (OSError, TypeError, ValueError):
        logger.debug("dead-block extraction cache write failed", exc_info=True)


def _dbx_stat(abs_path: str) -> tuple[int, int] | None:
    """(st_mtime_ns, st_size) — delegates to the canonical parse_cache helper
    (single stat code path; order contract documented there, B1)."""
    from . import parse_cache as _pc

    return _pc.stat_fingerprint(abs_path)


def _extract_dead_block_file(abs_path: str, rel_path: str):
    """Per-file extraction — parse + ``__all__`` + defs + references (pure).

    Returns ``(lang, all_names, defs, references)`` or None when the file must
    be SKIPPED (unreadable, unparseable tree, dynamic ``__all__``, or
    non-Python without tree-sitter).  Pure function of file content — the
    cache above makes the two dead-block scanners pay it once, not twice.
    """
    from ..languages import LanguageId as _LanguageId
    from . import parse_cache as _pc

    src = _pc.read_source(abs_path)
    if src is None:
        return None
    _lang_id = _LanguageId.from_path(rel_path)
    _lang = _lang_id.value if _lang_id is not None else "python"

    # ── Primary: tree-sitter (language-agnostic) ──
    if _HAS_TS:
        # tree-sitter is error-tolerant; a partial tree from broken source
        # would under-count references and produce false positives.
        _pre_tree = _ts_parse_to_tree(src, _lang)
        if _pre_tree is None or _pre_tree.root_node.has_error:
            return None
        all_names = _ts_extract_all_list(src, language=_lang)
        if "*__dynamic__*" in all_names:
            return None
        defs = _ts_collect_all_defs(src, language=_lang)
        references = _ts_collect_name_references(src, language=_lang)
    else:
        # ── Fallback: AST (Python only) ──
        if _lang != "python":
            return None
        # Parse from the SAME src read above — no second stat, so the tree
        # cannot be a different file version from the source being analysed.
        try:
            tree = ast.parse(src, filename=abs_path)
        except SyntaxError:
            logger.debug("[DEAD_BLOCK] SyntaxError in %s — skipping", rel_path)
            return None
        all_names = _extract_all_list(tree)
        if "*__dynamic__*" in all_names:
            return None
        defs = _collect_all_defs(tree)
        references = _collect_name_references(tree)
    return _lang, all_names, defs, references


# ── Shared scan core ─────────────────────────────────────────────────────────


def scan_dead_block_core(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int,
    cluster_gap_tolerance: int | None,
    cross_file_referenced_names: set | None,
    singleton_confidence: float,
    mark_public: bool,
    log_tag: str,
    include_public: bool | None = None,
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
    gap_tol = cluster_gap_tolerance if cluster_gap_tolerance is not None else CLUSTER_GAP_TOLERANCE
    candidates: list[DeadBlockCandidate] = []
    truncated_total = 0

    def _pub(members: list[DeadBlockMember]) -> bool:
        if not mark_public:
            return False
        return any(not m.name.startswith("_") for m in members)

    _dbx_cache, _dbx_dirty = _dbx_load(repo_root or "")
    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)
        _dbx_fp = _dbx_stat(abs_path)
        if _dbx_fp is None:
            continue
        _dbx_entry = _dbx_cache.get(abs_path)
        if _dbx_entry is None or _dbx_entry[0] != _dbx_fp:
            # Miss (or stale fingerprint): extract and store — the extraction
            # is shared verbatim by dead_block_scanner and public_dead_code_
            # scanner, so the second scanner over the same files hits here.
            _dbx_entry = (_dbx_fp, _extract_dead_block_file(abs_path, rel_path))
            _dbx_cache[abs_path] = _dbx_entry
            _dbx_dirty = True
        if _dbx_entry is None:
            continue  # unreachable (assigned above), but keeps the type checker honest
        _dbx_analysis = _dbx_entry[1]
        if _dbx_analysis is None:
            continue  # cached skip decision (unreadable / broken / dynamic __all__)

        _lang, all_names, defs, references = _dbx_analysis
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
                name,
                all_names,
                _effective_cross,
                include_public=include_public,
            ):
                continue
            if _is_externally_referenced(
                name,
                lineno,
                end_lineno,
                references,
                cross_file_referenced_names=_effective_cross,
                is_class_attr=(kind == "class_assignment"),
            ):
                continue
            dead_members.append(
                DeadBlockMember(
                    name=name,
                    symbol_kind=kind,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    enclosing_class=enclosing_class,
                )
            )

        if len(dead_members) < 2:
            if len(dead_members) == 1:
                m = dead_members[0]
                candidates.append(
                    DeadBlockCandidate(
                        file=rel_path,
                        members=[m],
                        cluster_start=m.lineno,
                        cluster_end=m.end_lineno,
                        confidence=singleton_confidence,
                        is_singleton=True,
                        includes_public=_pub([m]),
                    )
                )
            continue

        clusters, clustered_members = _cluster_dead_members(dead_members, gap_tol)

        emitted = 0
        for cluster in clusters:
            candidates.append(
                DeadBlockCandidate(
                    file=rel_path,
                    members=list(cluster),
                    cluster_start=cluster[0].lineno,
                    cluster_end=cluster[-1].end_lineno,
                    confidence=1.0,
                    includes_public=_pub(cluster),
                )
            )
            emitted += 1
            if emitted >= max_per_file:
                _remaining = len(clusters) - emitted
                truncated_total += _remaining
                logger.warning(
                    "[%s] %s: hit max_per_file=%d, truncating %d remaining cluster(s)",
                    log_tag,
                    rel_path,
                    max_per_file,
                    _remaining,
                )
                break

        # Emit remaining singletons (not clustered) with lower confidence.
        _singleton_emitted = 0
        for m in dead_members:
            if (m.lineno, m.name) in clustered_members:
                continue
            candidates.append(
                DeadBlockCandidate(
                    file=rel_path,
                    members=[m],
                    cluster_start=m.lineno,
                    cluster_end=m.end_lineno,
                    confidence=singleton_confidence,
                    is_singleton=True,
                    includes_public=_pub([m]),
                )
            )
            _singleton_emitted += 1
            if emitted + _singleton_emitted >= max_per_file:
                break

    if _dbx_dirty:
        _dbx_save(repo_root or "", _dbx_cache)

    if candidates:
        logger.info(
            "[%s] %d cluster(s) across %d file(s); total dead symbols=%d (public=%s)",
            log_tag,
            len(candidates),
            len({c.file for c in candidates}),
            sum(len(c.members) for c in candidates),
            any(c.includes_public for c in candidates),
        )

    return candidates, truncated_total
