"""Duplicate-definition scanner — finds same-name top-level definitions.

Phase 3 detector #1.  Targets a different problem from the AST similarity
scanner: where similarity finds *near-duplicates that could share a helper*,
this scanner finds *exact name collisions* — two ``def foo`` or two
``_X = ...`` at module scope, which usually means the second occurrence is
a stale leftover that shadows or redefines the first.

Conservative scope (Phase 3 launch):
  - module-body only (no class methods — those have legitimate same-name
    overrides via inheritance)
  - simple-target assignments only (``_X = ...`` and ``_X: T = ...``)
  - skips ``@overload`` / ``@typing.overload`` decorated functions
  - skips definitions nested inside conditional blocks (``if``/``try``/etc.)

Each name with ≥ 2 qualifying occurrences becomes a
``DuplicateDefinitionCandidate``.  Adapter
(``StructuralWorkset.from_duplicate_definition_candidate``) maps these to
``kind="duplicate_definition"`` worksets that DPB dispatches to a
deterministic ``DELETE_SYMBOL_RANGE`` op targeting the second occurrence.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.languages import LanguageId as _LanguageId

from . import parse_cache
from ._dead_block_shared import _has_overload, _ts_child_by_type

logger = logging.getLogger(__name__)


# ── Candidate model ────────────────────────────────────────────────────────────

@dataclass
class DuplicateDefinitionCandidate:
    """Two or more top-level definitions sharing a name in the same file."""
    file: str
    name: str
    symbol_kind: str  # "function" | "class" | "assignment"
    # Each occurrence: (lineno, end_lineno).  Sorted by lineno ascending so
    # occurrences[0] is the first definition (kept) and occurrences[1:] are
    # candidates for deletion.
    occurrences: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "name": self.name,
            "symbol_kind": self.symbol_kind,
            "occurrences": [list(occ) for occ in self.occurrences],
        }


# ── Tree-sitter availability ─────────────────────────────────────────────
try:
    from ..languages.tree_sitter_utils import (
        get_node_text as _ts_get_text,
    )
    from ..languages.tree_sitter_utils import (  # type: ignore
        parse_to_tree as _ts_parse_to_tree,
    )
    _HAS_TS = True
except ImportError:
    _HAS_TS = False


# ── Internal helpers ─────────────────────────────────────────────────────────

# Per-language top-level definition node types (direct children of program root
# only; Python ``decorated_definition`` wrappers are unwrapped before matching).
_LANG_TOP_LEVEL_NODES: dict[str, set] = {
    "python": {"function_definition", "async_function_definition", "class_definition",
               "expression_statement"},
    "typescript": {"function_declaration", "class_declaration", "interface_declaration",
                   "type_alias_declaration", "enum_declaration", "lexical_declaration",
                   "variable_declaration", "module_declaration"},
    "javascript": {"function_declaration", "class_declaration", "lexical_declaration",
                   "variable_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration", "type_spec"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration",
             "method_declaration", "field_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "companion_object",
               "interface_declaration", "enum_declaration", "fun_declaration",
               "property_declaration"},
}

# Per-language kind derivation from node type.
_LANG_KIND_MAP: dict[str, str] = {
    # Python
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
    "expression_statement": "assignment",
    # TS/JS
    "function_declaration": "function",
    "class_declaration": "class",
    "interface_declaration": "class",
    "type_alias_declaration": "assignment",
    "enum_declaration": "class",
    "lexical_declaration": "assignment",
    "variable_declaration": "assignment",
    "module_declaration": "class",
    # Go / Java (method_declaration shared by both)
    "method_declaration": "function",
    "type_declaration": "class",
    "type_spec": "assignment",
    # Java
    "field_declaration": "assignment",
    "constructor_declaration": "function",
    # Kotlin
    "object_declaration": "class",
    "companion_object": "class",
    "fun_declaration": "function",
    "property_declaration": "assignment",
}


def _ts_collect_top_level_definitions(source: str, language: str = "python") -> list[tuple[str, str, int, int, Optional[str]]]:
    """Tree-sitter version: collect module-level definitions only.

    Uses per-language ``_LANG_TOP_LEVEL_NODES`` map (supports Python,
    TypeScript, JavaScript, Go, Java, Kotlin).

    Skips:
      - functions with ``@overload`` (Python only)
      - definitions inside conditionals (``if`` / ``try`` / ``with`` blocks)
      - tuple/multi-target assignments and non-Name targets
    """
    if not _HAS_TS:
        return []
    top_types = _LANG_TOP_LEVEL_NODES.get(language)
    if top_types is None:
        return []
    tree = _ts_parse_to_tree(source, language)
    if tree is None:
        return []
    out: list[tuple[str, str, int, int, Optional[str]]] = []
    source_bytes = source.encode("utf-8")
    _FRAMEWORK_NAMES = {"fixture", "hookimpl", "hookspec"}

    for child in tree.root_node.children:
        node = child
        ct = node.type

        # ── Python: @decorator wraps the real def in decorated_definition ──
        decorators = []
        if ct == "decorated_definition":
            decorators = [c for c in node.children if c.type == "decorator"]
            node = node.child_by_field_name("definition")
            if node is None:
                continue
            ct = node.type

        if ct not in top_types:
            continue

        # ── Python-specific: skip @overload / framework-registered defs ──
        if language == "python" and decorators:
            skip = False
            for dec in decorators:
                dec_name = _ts_get_text(source_bytes, dec).lstrip("@").strip()
                dec_name = dec_name.split("(")[0].strip()
                if dec_name == "overload" or dec_name.rpartition(".")[2] in _FRAMEWORK_NAMES:
                    skip = True
                    break
            if skip:
                continue

        # ── Extract name node(s) ──────────────────────────────────────
        name_nodes = []
        if ct == "expression_statement":  # Python assignment wrapper
            assign_node = _ts_child_by_type(node, ("assignment",))
            if assign_node is None:
                continue
            left = assign_node.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                name_nodes.append(left)
        elif ct in ("lexical_declaration", "variable_declaration"):
            # TS/JS: const/let/var — names live inside variable_declarator children
            for c in node.children:
                if c.type == "variable_declarator":
                    nn = c.child_by_field_name("name")
                    if nn is not None and nn.type == "identifier":
                        name_nodes.append(nn)
        elif ct == "type_declaration":
            # Go: type Foo struct{...} — name lives inside the type_spec child
            for c in node.children:
                if c.type == "type_spec":
                    nn = c.child_by_field_name("name")
                    if nn is not None:
                        name_nodes.append(nn)
        else:
            nn = node.child_by_field_name("name")
            if nn is None:
                for c in node.children:
                    if c.type == "identifier":
                        nn = c
                        break
            if nn is not None:
                name_nodes.append(nn)

        kind = _LANG_KIND_MAP.get(ct, "assignment")
        # ── Receiver type for Go methods (dedup disambiguation) ──────────
        # ``func (a *A) Render()`` and ``func (b *B) Render()`` are distinct
        # symbols that would collide without the receiver in the dedup key.
        receiver: Optional[str] = None
        if ct == "method_declaration":
            recv_node = node.child_by_field_name("receiver")
            if recv_node is not None:
                pdecl = _ts_child_by_type(recv_node, ("parameter_declaration",))
                if pdecl is not None:
                    type_node = pdecl.child_by_field_name("type")
                    if type_node is not None:
                        receiver = _ts_get_text(source_bytes, type_node).lstrip("*").strip() or None
        # Span includes decorators (the outer wrapper node) so deletion ops
        # remove the decorator lines together with the definition.
        start = child.start_point[0] + 1
        end = child.end_point[0] + 1
        for nn in name_nodes:
            out.append((_ts_get_text(source_bytes, nn), kind, start, end, receiver))
    return out


def _collect_top_level_definitions(tree: ast.Module) -> list[tuple[str, str, int, int, Optional[str]]]:
    """Walk module.body once.  Returns list of (name, kind, lineno, end_lineno).

    Skips:
      - functions with ``@overload`` (intentional name reuse for typing)
      - definitions inside conditionals (``if`` / ``try`` / ``with`` blocks)
      - tuple/multi-target assignments and non-Name targets
    """
    # 5th element (receiver) is always None for Python — Python has no
    # Go-style receiver; ``self``/``cls`` are implicit and not part of the
    # qualified name.  Kept for return-type parity with the tree-sitter path.
    out: list[tuple[str, str, int, int, Optional[str]]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _has_overload(node):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            out.append((node.name, "function", node.lineno, end, None))
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", node.lineno)
            out.append((node.name, "class", node.lineno, end, None))
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            out.append((tgt.id, "assignment", node.lineno, end, None))
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                continue
            end = getattr(node, "end_lineno", node.lineno)
            out.append((node.target.id, "assignment", node.lineno, end, None))
        # Deliberately skip: If, Try, With, For, While, Match — conditional
        # contexts where same-name definitions are legitimate (e.g.
        # ``if sys.version_info >= ...: def foo(): ... else: def foo(): ...``).
    return out


# ── Public scan API ───────────────────────────────────────────────────────────

def scan_duplicate_definitions(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_DUP_DEF_MAX,
) -> list[DuplicateDefinitionCandidate]:
    """Scan ``file_paths`` for top-level name collisions.

    Returns one candidate per (file, name) pair that has ≥ 2 qualifying
    occurrences.  Files that fail to parse are skipped silently — duplicate
    detection is supplementary signal and must never block the main pipeline.
    """
    candidates: list[DuplicateDefinitionCandidate] = []
    _truncated_total = 0  # collision groups dropped by max_per_file

    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)
        src = parse_cache.read_source(abs_path)
        if src is None:
            continue

        # ── Determine language ──
        _lang_id = _LanguageId.from_path(rel_path)
        _lang = _lang_id.value if _lang_id is not None else "python"

        # ── Primary: tree-sitter ──
        if _HAS_TS:
            defs = _ts_collect_top_level_definitions(src, language=_lang)
        else:
            # ── Fallback: AST (Python only) ──
            if _lang != "python":
                continue
            tree = parse_cache.parse_ast(abs_path)
            if tree is None:
                continue
            defs = _collect_top_level_definitions(tree)

        # Group by (name, kind).  Same-name across kinds (e.g. class shadowed
        # by assignment) is rarer and reported as separate candidates.
        groups: dict = {}
        for name, kind, lineno, end_lineno, _receiver in defs:
            # Dedup key includes the receiver type for Go methods: distinct
            # receiver types (``func (a *A) Render`` vs ``func (b *B) Render``)
            # are legitimately separate symbols, not duplicate definitions.
            key = (name, kind, _receiver)
            groups.setdefault(key, []).append((lineno, end_lineno))

        _collision_groups = [(name, kind, occs) for (name, kind, _r), occs in groups.items() if len(occs) >= 2]
        emitted = 0
        for name, kind, occs in _collision_groups:
            if len(occs) < 2:
                continue
            occs_sorted = sorted(occs, key=lambda x: x[0])
            candidates.append(DuplicateDefinitionCandidate(
                file=rel_path,
                name=name,
                symbol_kind=kind,
                occurrences=occs_sorted,
            ))
            emitted += 1
            if emitted >= max_per_file:
                _truncated_total += len(_collision_groups) - emitted
                logger.warning(
                    "[DUPLICATE_DEF] %s: hit max_per_file=%d, truncating %d remaining group(s)",
                    rel_path, max_per_file, len(_collision_groups) - emitted,
                )
                break

    if candidates:
        logger.info(
            "[DUPLICATE_DEF] %d duplicate name(s) across %d file(s)",
            len(candidates), len(set(c.file for c in candidates)),
        )

    if _truncated_total:
        # Function attribute consumed by ScannerRegistry.run() (reset via
        # `del` before each invocation).
        scan_duplicate_definitions._truncated = _truncated_total
    return candidates

