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
``DuplicateDefinitionCandidate``, which DPB dispatches to a deterministic
``DELETE_SYMBOL_RANGE`` op targeting the second occurrence.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass, field

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.common.atomic_io import atomic_write_json
from external_llm.languages import LanguageId as _LanguageId

from ..languages.tree_sitter_utils import (
    get_node_text as _ts_get_text,
)
from ..languages.tree_sitter_utils import (
    parse_to_tree as _ts_parse_to_tree,
)
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
# tree_sitter_utils guards its own optional tree-sitter import, so this
# module always imports — the old try/except ImportError was dead code.
_HAS_TS = True


# ── Internal helpers ─────────────────────────────────────────────────────────

# Per-language top-level definition node types (direct children of program root
# only; Python ``decorated_definition`` wrappers are unwrapped before matching).
_LANG_TOP_LEVEL_NODES: dict[str, set] = {
    # "assignment" and "expression_statement" are the same statement under the two
    # Python grammars (standalone tree-sitter-python wraps, the
    # tree-sitter-language-pack bundle does not); only the pack is a declared
    # dependency, so both shapes must be listed.
    # "annotated_assignment" (``x: T = ...``) is the same dual-shape story and is
    # emitted by the AST fallback path (ast.AnnAssign) — the TS path skipping it
    # was a silent parity gap, now pinned by the node-level contract in
    # test_duplicate_definition_lang_keys_match_registry.
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "expression_statement",
        "assignment",
        "annotated_assignment",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "lexical_declaration",
        "variable_declaration",
        "module_declaration",
    },
    "javascript": {"function_declaration", "class_declaration", "lexical_declaration", "variable_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration", "type_spec"},
    # method/field_declaration never appear at program root under the standard
    # java grammar (they live inside class bodies) but are kept for
    # grammar-variant coverage — the same reasoning as the python dual listing.
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "field_declaration",
    },
    # fwcd kotlin grammar (language-pack AND standalone) emits
    # function_declaration — "fun_declaration" was a stale fork variant that
    # silently skipped every top-level kotlin function.
    "kotlin": {
        "class_declaration",
        "object_declaration",
        "companion_object",
        "interface_declaration",
        "enum_declaration",
        "function_declaration",
        "property_declaration",
    },
}

# Per-language kind derivation from node type.
_LANG_KIND_MAP: dict[str, str] = {
    # Python
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
    "expression_statement": "assignment",
    "assignment": "assignment",  # wrapper-less form (language-pack grammar)
    "annotated_assignment": "assignment",  # x: T = ... (both grammar shapes)
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
    # Kotlin
    "object_declaration": "class",
    "companion_object": "class",
    "property_declaration": "assignment",
}


# ── Import-time judge-map validation ────────────────────────────────────────
# The two maps above are hand-maintained per-language node types (grammar
# facts — not derivable), so their internal consistency is enforced at import:
#   - a node type the walk CAN emit without a kind would be silently misjudged
#     as "assignment" by the old get(ct, ...) fallback;
#   - a kind entry nothing emits is a stale leftover that hides a node-type
#     removal (constructor_declaration was exactly that — java constructors
#     live inside class bodies, never at program root).
_SYMBOL_KINDS = frozenset({"function", "class", "assignment"})


def _validate_judge_maps() -> None:
    emitted = set().union(*_LANG_TOP_LEVEL_NODES.values())
    unkinded = emitted - set(_LANG_KIND_MAP)
    if unkinded:
        raise ValueError(f"_LANG_KIND_MAP is missing kinds for emitted node types: {sorted(unkinded)}")
    stale = set(_LANG_KIND_MAP) - emitted
    if stale:
        raise ValueError(
            "_LANG_KIND_MAP entries are never emitted by any _LANG_TOP_LEVEL_NODES "
            f"set (stale after a node-type removal): {sorted(stale)}"
        )
    bad = {ct: kind for ct, kind in _LANG_KIND_MAP.items() if kind not in _SYMBOL_KINDS}
    if bad:
        raise ValueError(f"_LANG_KIND_MAP kinds must be in {sorted(_SYMBOL_KINDS)}, got: {bad}")


_validate_judge_maps()


def _ts_collect_top_level_definitions(
    source: str, language: str = "python"
) -> list[tuple[str, str, int, int, str | None]]:
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
    out: list[tuple[str, str, int, int, str | None]] = []
    source_bytes = source.encode("utf-8")
    _framework_names = {"fixture", "hookimpl", "hookspec"}

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
                if dec_name == "overload" or dec_name.rpartition(".")[2] in _framework_names:
                    skip = True
                    break
            if skip:
                continue

        # ── Extract name node(s) ──────────────────────────────────────
        name_nodes = []
        if ct in ("expression_statement", "assignment", "annotated_assignment"):  # Python assignment, ±wrapper
            assign_node = (
                node
                if ct in ("assignment", "annotated_assignment")
                else _ts_child_by_type(node, ("assignment", "annotated_assignment"))
            )
            if assign_node is None:
                continue
            left = assign_node.child_by_field_name("left")  # type: ignore[attr-defined]  # tree-sitter node
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
                # kotlin fwcd grammar names bindings via unnamed children
                # (simple_identifier / type_identifier), never a `name` field.
                for c in node.children:
                    if c.type in ("identifier", "simple_identifier", "type_identifier"):
                        nn = c
                        break
                if nn is None and language == "kotlin" and ct == "property_declaration":
                    # property names nest one level deeper:
                    # property_declaration → variable_declaration →
                    # simple_identifier (binding_pattern_kind holds only the
                    # val/var keyword).  Destructuring (`val (a, b) = ...`)
                    # emits multi_variable_declaration instead and is
                    # deliberately skipped — mirroring the python
                    # multi-target exclusion.
                    vd = _ts_child_by_type(node, ("variable_declaration",))
                    if vd is not None:
                        nn = _ts_child_by_type(vd, ("simple_identifier",))
            if nn is not None:
                name_nodes.append(nn)

        kind = _LANG_KIND_MAP[ct]  # presence guaranteed by _validate_judge_maps at import
        # ── Receiver type for Go methods (dedup disambiguation) ──────────
        # ``func (a *A) Render()`` and ``func (b *B) Render()`` are distinct
        # symbols that would collide without the receiver in the dedup key.
        receiver: str | None = None
        if ct == "method_declaration":
            recv_node = node.child_by_field_name("receiver")  # type: ignore[attr-defined]  # tree-sitter node
            if recv_node is not None:
                pdecl = _ts_child_by_type(recv_node, ("parameter_declaration",))
                if pdecl is not None:
                    type_node = pdecl.child_by_field_name("type")  # type: ignore[attr-defined]  # tree-sitter node
                    if type_node is not None:
                        receiver = _ts_get_text(source_bytes, type_node).lstrip("*").strip() or None
        # Span includes decorators (the outer wrapper node) so deletion ops
        # remove the decorator lines together with the definition.
        start = child.start_point[0] + 1
        end = child.end_point[0] + 1
        out.extend((_ts_get_text(source_bytes, nn), kind, start, end, receiver) for nn in name_nodes)
    return out


def _collect_top_level_definitions(tree: ast.Module) -> list[tuple[str, str, int, int, str | None]]:
    """Walk module.body once.  Returns list of (name, kind, lineno, end_lineno).

    Skips:
      - functions with ``@overload`` (intentional name reuse for typing)
      - definitions inside conditionals (``if`` / ``try`` / ``with`` blocks)
      - tuple/multi-target assignments and non-Name targets
    """
    # 5th element (receiver) is always None for Python — Python has no
    # Go-style receiver; ``self``/``cls`` are implicit and not part of the
    # qualified name.  Kept for return-type parity with the tree-sitter path.
    out: list[tuple[str, str, int, int, str | None]] = []
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


# ── Per-file result disk cache (A307 pattern, 2026-08-21) ────────────────────
# The structural gate runs this scanner over the WHOLE repo on every commit
# (fresh process).  Tree-sitter parsing of every scanned file is the
# scanner's dominant cost even on a warm gate (measured 1.43s / 1004 files,
# P14-5 2026-08-21) — and unlike the AST scanners it has no shared
# parse_cache tier, because tree_sitter_utils keeps its own parse tree cache.
# The collection is a pure function of file content (no graph, no cross-file
# state), so a per-file (st_mtime_ns, st_size) fingerprint cache lets a warm
# gate reuse the previous process's definition lists instead of re-parsing
# every file.  Fail-open by construction: any read/format/version mismatch
# or a missing entry recomputes that file; a stale cache never changes
# verdicts, only costs a re-scan.  Version is embedded so scanner-logic
# changes invalidate the whole cache instead of serving old verdicts.
_DUP_DEF_CACHE_VERSION = 1


def _dup_def_cache_path(repo_root: str) -> str:
    return parse_cache.cache_file_path(repo_root, f"duplicate_def_v{_DUP_DEF_CACHE_VERSION}.json")


def _load_dup_def_cache(repo_root: str) -> tuple[dict[str, tuple], bool]:
    """Per-file def-list cache keyed by abs path: ``{path: (fp, defs)}``.

    *defs* entries are JSON-shaped ``[name, kind, lineno, end_lineno,
    receiver]`` lists (see the JSON-round-trip contract — P14-3: the in-memory
    form must match the on-disk form so a warm load never re-dirties).
    Returns ``(cache, dirty=False)``; fail-open: any read/format/version
    error yields an empty cache, and the caller recomputes everything.
    """
    if not repo_root:
        return {}, False
    path = _dup_def_cache_path(repo_root)
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError):
        logger.debug("[DUPLICATE_DEF] cache unreadable — full collection", exc_info=True)
        return {}, False
    if payload.get("format") != _DUP_DEF_CACHE_VERSION or not isinstance(payload.get("files"), dict):
        return {}, False
    cache: dict[str, tuple] = {}
    for abs_path, entry in payload["files"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("fp"), list):
            continue
        defs = entry.get("defs")
        if defs is None:
            cache[abs_path] = (tuple(entry["fp"]), None)
        elif isinstance(defs, list):
            cache[abs_path] = (tuple(entry["fp"]), [tuple(d) for d in defs])
    return cache, False


def _save_dup_def_cache(repo_root: str, cache: dict[str, tuple]) -> None:
    """Persist *cache* to disk atomically (best-effort; failure costs a re-scan).

    Empty *repo_root* skips the write (unit-test convention — throwaway temp
    files must not grow the CWD's ``.cache``).
    """
    if not repo_root:
        return
    path = _dup_def_cache_path(repo_root)
    try:
        files = {}
        for abs_path, (fp, defs) in cache.items():
            if defs is None:
                files[abs_path] = {"fp": list(fp), "defs": None}
            else:
                files[abs_path] = {"fp": list(fp), "defs": [list(d) for d in defs]}
        atomic_write_json(
            path,
            {"format": _DUP_DEF_CACHE_VERSION, "files": files},
            indent=None,
            ensure_ascii=True,
        )
    except (OSError, TypeError, ValueError):
        logger.debug("[DUPLICATE_DEF] cache write failed", exc_info=True)


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

    _cache, _dirty = _load_dup_def_cache(repo_root or "")
    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)
        # Fused read+fingerprint under ONE stat (B1 order contract, 2026-08-16):
        # the fingerprint may only key content read at or after this stat, so a
        # torn entry is unreachable instead of silently stale.  The previous
        # read_source-then-stat_fingerprint pair stat-ed twice and could pair a
        # post-write stamp with pre-write analysis.
        _fused = parse_cache.read_with_fingerprint(abs_path)
        if _fused is None:
            continue
        src, _fp = _fused

        # ── Determine language ──
        _lang_id = _LanguageId.from_path(rel_path)
        _lang = _lang_id.value if _lang_id is not None else "python"

        # ── Fingerprint cache: skip the tree-sitter/AST collection entirely on
        # a (mtime_ns, size) hit (the scanner's dominant cost, P14-5).  ``_fp``
        # came from the fused read above (B1 order — stamp precedes content).
        _cached = _cache.get(abs_path)
        if _cached is not None and _cached[0] == _fp:
            defs = _cached[1]
        else:
            # ── Primary: tree-sitter ──
            if _HAS_TS:
                defs = _ts_collect_top_level_definitions(src, language=_lang)
            else:
                # ── Fallback: AST (Python only) ──
                if _lang != "python":
                    continue
                # Parse from the SAME src the fingerprint describes — no second
                # stat, so fp/src/defs are one consistent file version.
                try:
                    tree = ast.parse(src, filename=abs_path)
                except SyntaxError:
                    logger.debug("[DUPLICATE_DEF] SyntaxError in %s — skipping", rel_path)
                    continue
                defs = _collect_top_level_definitions(tree)
            if _fp is not None:
                _cache[abs_path] = (_fp, defs)
                _dirty = True
        if defs is None:
            continue  # cached skip decision (unreadable / broken / no imports)

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
            candidates.append(
                DuplicateDefinitionCandidate(
                    file=rel_path,
                    name=name,
                    symbol_kind=kind,
                    occurrences=occs_sorted,
                )
            )
            emitted += 1
            if emitted >= max_per_file:
                _truncated_total += len(_collision_groups) - emitted
                logger.warning(
                    "[DUPLICATE_DEF] %s: hit max_per_file=%d, truncating %d remaining group(s)",
                    rel_path,
                    max_per_file,
                    len(_collision_groups) - emitted,
                )
                break

    if _dirty:
        _save_dup_def_cache(repo_root or "", _cache)

    if candidates:
        logger.info(
            "[DUPLICATE_DEF] %d duplicate name(s) across %d file(s)",
            len(candidates),
            len({c.file for c in candidates}),
        )

    if _truncated_total:
        # Function attribute consumed by ScannerRegistry.run() (reset via
        # `del` before each invocation).
        scan_duplicate_definitions._truncated = _truncated_total  # type: ignore[attr-defined]  # dynamic attr consumed by ScannerRegistry.run()
    return candidates
