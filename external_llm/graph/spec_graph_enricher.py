"""
SpecGraphEnricher — enriches a ResolvedExecutionSpec with graph-derived signals.

Phase 7.2 (P2): connects graph-derived data to the spec → planner → candidate → ranker pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from external_llm.languages.models import _get_language_group  # single source of truth

from .models import SymbolKind, SymbolNode

if TYPE_CHECKING:
    from ..agent.execution_spec import ResolvedExecutionSpec
    from .graph_facade import RepositoryGraphFacade

logger = logging.getLogger(__name__)


class SpecGraphEnricher:
    """Enriches a ResolvedExecutionSpec with graph-derived metadata.

    The enriched data is stored in spec.metadata["graph_context"] and never
    modifies any canonical spec field.  All facade calls are wrapped in
    try/except so failures never crash the pipeline.
    """

    def __init__(self, graph_facade: "RepositoryGraphFacade") -> None:
        self._facade = graph_facade

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, spec: "ResolvedExecutionSpec", cache=None) -> "ResolvedExecutionSpec":
        """Enrich spec.metadata["graph_context"] with graph-derived signals.

        Steps:
        1. Resolve target_symbols via facade.get_symbol()
        2. For resolved symbols, get callers/callees
        3. Expand target_files from resolved symbol file_paths
        4. Compute impact frontier (depth=1 callers/callees files)
        5. Find related_symbols via facade.get_related_symbols()
        6. Compute graph_confidence based on resolution success rate
        7. Store everything in spec.metadata["graph_context"]

        Args:
            spec: ResolvedExecutionSpec to enrich
            cache: Optional RunScopedGraphCache for deduplication

        IMPORTANT: Never fail — always return spec (possibly with minimal graph_context).
        """
        if spec is None:
            return spec
        # Guard: skip if already enriched
        if "graph_context" in spec.metadata:
            logger.debug("Spec already enriched, skipping")
            return spec

        target_symbols = list(getattr(spec, "target_symbols", None) or [])
        target_files = list(getattr(spec, "target_files", None) or [])

        # Cache check
        cache_key = None
        if cache is not None:
            try:
                from .run_scoped_graph_cache import RunScopedGraphCache
                cache_key = RunScopedGraphCache.make_key(
                    "enrichment",
                    target_symbols=target_symbols,
                    target_files=target_files,
                    generation=cache.generation,
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    spec.metadata["graph_context"] = cached
                    spec.metadata["graph_cache_hit"] = True
                    spec.metadata["graph_cache_key"] = cache_key
                    logger.info("Graph enrichment cache hit for key=%s", cache_key[:12])
                    return spec
            except Exception as e:
                logger.debug("Graph cache lookup failed: %s", e)

        logger.info(
            "Starting graph enrichment for spec with %d target_symbols, %d target_files",
            len(target_symbols), len(target_files),
        )

        try:
            graph_context = self._build_graph_context(spec)
            spec.metadata["graph_context"] = graph_context
            logger.info(
                "Graph enrichment complete: resolved=%d, unresolved=%d, "
                "impact_files=%d, confidence=%.2f",
                len(graph_context.get("resolved_symbols", [])),
                len(graph_context.get("unresolved_symbols", [])),
                len(graph_context.get("impact_files", [])),
                graph_context.get("graph_confidence", 0.0),
            )
            # N1 guard: low confidence (0.00) + unresolved symbols → downstream warning flag
            # Two triggers:
            #   (a) resolved_symbols empty — nothing at all was found
            #   (b) resolved_symbols non-empty but ALL target_symbols are unresolved
            #       (e.g. file enumeration found 10 symbols but none are the target)
            _gc = graph_context
            _unresolved_targets = set(_gc.get("unresolved_symbols", []))
            _resolved_names = {r.get("name") for r in _gc.get("resolved_symbols", [])}
            _all_targets_unresolved = (
                bool(target_symbols)
                and all(s in _unresolved_targets for s in target_symbols)
            )
            if (
                _gc.get("graph_confidence", 1.0) == 0.0
                and target_symbols
                and _gc.get("unresolved_symbols", [])
                and (
                    not _gc.get("resolved_symbols", [])
                    or _all_targets_unresolved
                )
            ):
                graph_context["_low_confidence_warning"] = True
                logger.warning(
                    "[GRAPH_LOW_CONFIDENCE] graph_confidence=0.00, resolved=%d targets all unresolved, "
                    "target_symbols=%s — hallucination risk, setting warning flag",
                    len(_gc.get("resolved_symbols", [])),
                    target_symbols,
                )

            # N1b guard: empty target_symbols due to hallucination removal
            if (
                not target_symbols
                and spec.metadata.get("_design_chat_removed_symbols")
            ):
                graph_context["_low_confidence_warning"] = True
                graph_context["graph_confidence"] = 0.0
                # Populate unresolved_symbols with hallucinated symbols
                removed = spec.metadata["_design_chat_removed_symbols"]
                current_unresolved = graph_context.get("unresolved_symbols", [])
                for sym in removed:
                    if sym not in current_unresolved:
                        current_unresolved.append(sym)
                graph_context["unresolved_symbols"] = current_unresolved
                logger.warning(
                    "[GRAPH_LOW_CONFIDENCE] target_symbols empty after symbol validation removed %s — "
                    "hallucination risk, setting warning flag",
                    removed,
                )

            # Store in cache after successful enrichment
            if cache is not None and cache_key:
                try:
                    cache.put(cache_key, graph_context, category="enrichment")
                    spec.metadata["graph_cache_hit"] = False
                    spec.metadata["graph_cache_key"] = cache_key
                except Exception as e:
                    logger.debug("Graph enrichment cache store failed: %s", e)
        except Exception as exc:
            logger.warning("SpecGraphEnricher.enrich() failed (non-fatal): %s", exc)
            # Minimal fallback to signal attempted but failed enrichment
            spec.metadata.setdefault("graph_context", {
                "resolved_symbols": [],
                "unresolved_symbols": [],
                "primary_files": [],
                "impact_files": [],
                "callers": {},
                "callees": {},
                "related_symbols": [],
                "graph_confidence": 0.0,
            })

        return spec

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_unresolved_via_ast(
        self,
        unresolved_symbols: list[str],
        target_files: list[str],
        spec: "ResolvedExecutionSpec",
    ) -> list[dict[str, Any]]:
        """AST-based fallback resolution for symbols the graph could not find.

        When the graph is stale (files modified after last rebuild),
        get_symbol() returns None.  This method parses target files via
        ast.parse() to confirm unresolved symbols actually exist.

        Returns a list of resolved symbol entries (minimal: name + file_path).
        """
        import ast as _py_ast
        import os as _os

        if not unresolved_symbols or not target_files:
            return []

        _resolved: list[dict[str, Any]] = []
        _unresolved_set = set(unresolved_symbols)
        # Extract repo_root early so Phase 4 can use it even when
        # target_files have no absolute paths.
        _meta = getattr(spec, "metadata", None) or {}
        _root = _meta.get("repo_root", "") or _os.getcwd()

        for _tf in target_files:
            if not _unresolved_set:
                break
            _abs_path = _tf
            if not _os.path.isabs(_tf):
                _abs_path = _os.path.join(_root, _tf)
            if not _os.path.isfile(_abs_path):
                continue
            try:
                with open(_abs_path, encoding="utf-8", errors="replace") as _fh:
                    _source = _fh.read()
            except OSError:
                continue
            try:
                _tree = _py_ast.parse(_source, filename=_abs_path)
            except SyntaxError:
                continue

            # ── Phase 1: collect bare names + class tree for dotted resolution ──
            _defined_names: set = set()
            _class_tree: dict = {}  # class_name -> {"methods": set, "fields": set, "classes": {}}

            def _collect_class_members(_cls_node: _py_ast.ClassDef) -> dict:
                """Recursively collect methods, fields, and nested classes."""
                _methods: set = set()
                _fields: set = set()
                _classes: dict = {}
                for _child in _cls_node.body:
                    if isinstance(_child, (_py_ast.FunctionDef, _py_ast.AsyncFunctionDef)):
                        _methods.add(_child.name)
                    elif isinstance(_child, _py_ast.ClassDef):
                        _classes[_child.name] = _collect_class_members(_child)
                    elif isinstance(_child, _py_ast.Assign):
                        for _tgt in _child.targets:
                            if isinstance(_tgt, _py_ast.Name):
                                _fields.add(_tgt.id)
                    elif isinstance(_child, _py_ast.AnnAssign) and isinstance(_child.target, _py_ast.Name):
                        _fields.add(_child.target.id)
                return {"methods": _methods, "fields": _fields, "classes": _classes}

            # Module-level names only — iterate _tree.body, NOT ast.walk(),
            # to prevent nested class members leaking into _defined_names.
            # _collect_class_members() (called inside ClassDef branch) handles
            # recursive member collection for dotted resolution.
            for _node in _tree.body:
                if isinstance(_node, (_py_ast.FunctionDef, _py_ast.AsyncFunctionDef)):
                    _defined_names.add(_node.name)
                elif isinstance(_node, _py_ast.ClassDef):
                    _defined_names.add(_node.name)
                    _class_tree[_node.name] = _collect_class_members(_node)
                elif isinstance(_node, _py_ast.Assign):
                    for _tgt in _node.targets:
                        if isinstance(_tgt, _py_ast.Name):
                            _defined_names.add(_tgt.id)
                elif isinstance(_node, _py_ast.AnnAssign) and isinstance(_node.target, _py_ast.Name):
                    _defined_names.add(_node.target.id)

            def _walk_class_tree(_parts: list, _tree_node: dict) -> str | None:
                """Walk A.B.C through class tree, return kind or None."""
                if not _parts:
                    return None
                _cur = _parts[0]
                _rest = _parts[1:]
                if not _rest:
                    if _cur in _tree_node.get("methods", set()):
                        return "function"
                    if _cur in _tree_node.get("fields", set()):
                        return "class_field"
                    if _cur in _tree_node.get("classes", {}):
                        return "class"
                    return None
                # More parts remain — must be a nested class
                _sub = _tree_node.get("classes", {}).get(_cur)
                if _sub is None:
                    return None
                return _walk_class_tree(_rest, _sub)

            for _sym in list(_unresolved_set):
                # ── Case 1: exact bare-name match ────────────────────────────
                if _sym in _defined_names:
                    _resolved.append({
                        "name": _sym,
                        "qualname": _sym,
                        "file_path": _tf,
                        "kind": "function",
                        "source": "ast_fallback",
                        "start_line": 0,
                        "end_line": 0,
                        "docstring": "",
                    })
                    _unresolved_set.discard(_sym)
                elif "." in _sym:
                    # ── Case 2: dotted name (ClassName.method / Nested.Class.field) ──
                    _parts = _sym.split(".")
                    _cls = _parts[0]
                    _cls_info = _class_tree.get(_cls)
                    if _cls_info is not None:
                        _kind = _walk_class_tree(_parts[1:], _cls_info)
                        if _kind is not None:
                            _resolved.append({
                                "name": _sym,
                                "qualname": _sym,
                                "file_path": _tf,
                                "kind": _kind,
                                "source": "ast_fallback",
                                "start_line": 0,
                                "end_line": 0,
                                "docstring": "",
                            })
                            _unresolved_set.discard(_sym)
                else:
                    # ── Case 3: stripped variant (leading underscore removed) ──
                    _bare = _sym.lstrip("_")
                    if _bare and _bare != _sym and _bare in _defined_names:
                        _resolved.append({
                            "name": _sym,
                            "qualname": _sym,
                            "file_path": _tf,
                            "kind": "function",
                            "source": "ast_fallback",
                            "start_line": 0,
                            "end_line": 0,
                            "docstring": "",
                        })
                        _unresolved_set.discard(_sym)

        # ── Phase 4: module-prefix resolution for remaining dotted symbols ──
        # If _shared_utils._TS_JS_EXTENSIONS is unresolved, try to find and
        # scan _shared_utils.py even if it's not in target_files.
        if _unresolved_set and _root:
            _orig_unresolved = set(_unresolved_set)
            for _sym in list(_orig_unresolved):
                if "." not in _sym:
                    continue
                _module_part = _sym.split(".")[0]
                # Try common locations: same dir, common package dirs
                _candidates = [
                    _os.path.join(_root, f"{_module_part}.py"),
                    _os.path.join(_root, "external_llm", "agent", f"{_module_part}.py"),
                ]
                # Also check target_files' directories
                for _tf in target_files:
                    _tf_dir = _os.path.dirname(
                        _tf if _os.path.isabs(_tf) else _os.path.join(_root, _tf)
                    )
                    _candidates.append(_os.path.join(_tf_dir, f"{_module_part}.py"))
                for _mod_path in _candidates:
                    if not _os.path.isfile(_mod_path):
                        continue
                    try:
                        with open(_mod_path, encoding="utf-8", errors="replace") as _mh:
                            _mod_source = _mh.read()
                        _mod_tree = _py_ast.parse(_mod_source, filename=_mod_path)
                    except (OSError, SyntaxError):
                        continue
                    _mod_names: set = set()
                    for _mod_node in _mod_tree.body:
                        if isinstance(_mod_node, (_py_ast.FunctionDef, _py_ast.AsyncFunctionDef)):
                            _mod_names.add(_mod_node.name)
                        elif isinstance(_mod_node, _py_ast.ClassDef):
                            _mod_names.add(_mod_node.name)
                        elif isinstance(_mod_node, _py_ast.Assign):
                            for _tgt in _mod_node.targets:
                                if isinstance(_tgt, _py_ast.Name):
                                    _mod_names.add(_tgt.id)
                        elif isinstance(_mod_node, _py_ast.AnnAssign) and isinstance(_mod_node.target, _py_ast.Name):
                            _mod_names.add(_mod_node.target.id)
                    _rest_part = _sym[len(_module_part) + 1:]  # e.g. "_TS_JS_EXTENSIONS"
                    if _rest_part in _mod_names:
                        _resolved.append({
                            "name": _sym,
                            "qualname": _sym,
                            "file_path": _mod_path,
                            "kind": "variable",
                            "source": "ast_fallback",
                            "start_line": 0,
                            "end_line": 0,
                            "docstring": "",
                        })
                        _unresolved_set.discard(_sym)
                        _resolved_module_path = _mod_path
                        break

        return _resolved
    @staticmethod
    def _get_depth_config(scope: str) -> dict[str, int]:
        """Return adaptive limits based on estimated scope.

        Larger scope → wider context collection.
        """
        configs = {
            # single_symbol: for explicit, high-confidence single-symbol changes.
            # Minimal scope — only direct callers/callees, no 2-hop or related symbols.
            "single_symbol": {"caller_cap": 2,  "callee_cap": 2,  "hop2_parent_cap": 0, "hop2_edge_cap": 0, "impact_cap": 5,  "file_enum_cap": 0, "related_cap": 0},
            "tiny":   {"caller_cap": 5,  "callee_cap": 5,  "hop2_parent_cap": 2, "hop2_edge_cap": 3, "impact_cap": 10, "file_enum_cap": 5,  "related_cap": 5},
            "small":  {"caller_cap": 10, "callee_cap": 10, "hop2_parent_cap": 3, "hop2_edge_cap": 5, "impact_cap": 20, "file_enum_cap": 10, "related_cap": 20},
            "medium": {"caller_cap": 15, "callee_cap": 15, "hop2_parent_cap": 5, "hop2_edge_cap": 8, "impact_cap": 30, "file_enum_cap": 15, "related_cap": 30},
            "large":  {"caller_cap": 20, "callee_cap": 20, "hop2_parent_cap": 8, "hop2_edge_cap": 10, "impact_cap": 40, "file_enum_cap": 20, "related_cap": 40},
        }
        if scope not in configs:
            logger.warning("Unknown scope '%s', falling back to 'small'", scope)
        return configs.get(scope, configs["small"])


    def _get_class_members(self, class_name: str, file_path: str) -> list[str]:
        """Return method names defined directly inside class_name (max 20).

        Plan A: When resolving a class symbol, collect its member list so the
        planner can use actual existing method names as insert_after_symbol anchors.
        """
        try:
            file_syms = self._facade.get_symbols_in_file(file_path) or []
            prefix = f"{class_name}."
            return [
                node.name for node in file_syms
                if node.kind == "method"
                and (node.qualname or "").startswith(prefix)
            ][:20]
        except Exception as exc:
            logger.debug("_get_class_members(%r, %r) failed: %s", class_name, file_path, exc)
            return []

    def _read_symbol_source(self, node: SymbolNode, max_lines: int = 30) -> str:
        """Read symbol source from disk.

        Returns full body for small symbols (≤max_lines), or
        signature+docstring for large symbols.  Total budget: ~8000 chars
        across all symbols (managed by caller).
        """
        try:
            abs_path = os.path.join(self._facade.repo_root, node.file_path)
            with open(abs_path, encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()
            start = node.start_line - 1  # 0-indexed
            end = min(node.end_line, len(all_lines))
            sym_lines = end - start
            if sym_lines <= 0:
                return ""
            if sym_lines <= max_lines:
                return "".join(all_lines[start:end])
            # Large symbol: extract signature + docstring only
            snippet = []
            found_colon = False
            for i in range(start, min(end, start + 10)):
                snippet.append(all_lines[i])
                if all_lines[i].rstrip().endswith(':'):
                    found_colon = True
                    # Look for docstring after the colon
                    for j in range(i + 1, min(end, i + 15)):
                        line = all_lines[j].strip()
                        snippet.append(all_lines[j])
                        if line.startswith(('"""', "'''")):
                            # Single-line docstring
                            if (line.endswith('"""') or line.endswith("'''")) and len(line) > 3:
                                break
                        if line.endswith('"""') or line.endswith("'''"):
                            break
                    break
            if not found_colon:
                snippet = list(all_lines[start:min(end, start + 5)])
            return "".join(snippet) + f"    # ... ({sym_lines} lines total)\n"
        except Exception as exc:
            logger.debug("_read_symbol_source(%r) failed: %s", node.qualname, exc)
            return ""

    def _build_graph_context(self, spec: "ResolvedExecutionSpec") -> dict[str, Any]:
        """Build the full graph_context dict from a spec."""
        target_symbols: list[str] = list(getattr(spec, "target_symbols", None) or [])
        target_files: list[str] = list(getattr(spec, "target_files", None) or [])

        resolved_symbols: list[dict[str, Any]] = []
        unresolved_symbols: list[str] = []
        callers_map: dict[str, list[dict[str, Any]]] = {}
        callees_map: dict[str, list[dict[str, Any]]] = {}
        primary_files_set: set = set()
        impact_files_set: set = set()
        _source_budget_remaining = 8000  # chars budget for symbol sources

        # P2: adaptive limits based on estimated scope
        _scope = getattr(spec, "estimated_scope", "small")
        # Detect single-symbol scenario from explicit target signals.
        # When the user named the target directly (explicit provenance) and
        # grounding found it with high confidence for exactly 1 symbol, scope is
        # inherently narrow — no need for wide graph expansion.
        # Three signals gate this path:
        #   1. allow_exploration=False — grounder determined exploration is
        #      unnecessary for this request (identifier_first or behavioral_first).
        #   2. target_provenance="explicit" — user named the target directly.
        #   3. grounding_confidence >= 0.75 — high confidence in the resolve.
        #   4. target_symbols <= 1 — exactly one symbol to change.
        # All four must be true; otherwise falls back to the original scope tier.
        _allow_exp = spec.metadata.get("grounding_policy_allow_exploration", True)
        _target_provenance = getattr(spec, "target_provenance", "") or ""
        _grounding_conf = float(spec.metadata.get("grounding_confidence", 0.0) or 0.0)
        # Use intent_symbols (canonical user-intended symbols) rather than
        # target_symbols (which may include grounder-expanded noise).
        _intent_symbols = list(getattr(spec, "intent_symbols", None) or [])
        if (
            not _allow_exp
            and _target_provenance == "explicit"
            and _grounding_conf >= 0.75
            and len(_intent_symbols) <= 1
        ):
            _scope = "single_symbol"
            logger.info(
                "GraphEnricher: single_symbol scope (allow_exp=%s, prov=%s, conf=%.2f, "
                "intent_syms=%d, target_syms=%d)",
                _allow_exp, _target_provenance, _grounding_conf,
                len(_intent_symbols), len(target_symbols),
            )
        _depth_level = self._get_depth_config(_scope)
        _file_enum_cap = _depth_level.get("file_enum_cap", 10)
        _related_cap = _depth_level.get("related_cap", 20)

        # ── Step 1: resolve each target symbol ───────────────────────────────
        # In single_symbol scope, resolve only the user-intended symbols
        # (intent_symbols) instead of the grounder-expanded target_symbols.
        # The grounder may add noise symbols from the same file that are
        # semantically unrelated to the actual bugfix target.
        _resolve_syms = _intent_symbols if (_scope == "single_symbol" and _intent_symbols) else target_symbols
        for sym_name in _resolve_syms:
            try:
                # Try file-scoped lookup first to avoid ambiguity when the same
                # symbol name exists in multiple files.
                node = None
                for file_hint in target_files:
                    node = self._facade.get_symbol(sym_name, file_hint)
                    if node is not None:
                        break
                # Fallback to unscoped lookup if no file hint matched
                if node is None:
                    node = self._facade.get_symbol(sym_name)
                    # ★ Root cause fix: language boundary guard for unscoped lookup.
                    # When file-scoped lookup (above) fails, unscoped get_symbol()
                    # searches the entire repo and may find a symbol from a different
                    # language (e.g., Python GameState for a TypeScript project).
                    # This contaminates downstream graph expansion:
                    #   wrong file → callee_files → target_files → P6.11 import injection
                    # Uses _LANGUAGE_EXTENSION_GROUPS so .ts/.tsx are same family.
                    if node is not None and node.file_path:
                        _node_ext = os.path.splitext(node.file_path)[1].lower()
                        if _node_ext:
                            _node_group = _get_language_group(_node_ext)
                            if _node_group >= 0:
                                # Collect language groups from target_files.
                                # Only reject when target_files *have* a language signal
                                # and the resolved node doesn't match any of them.
                                # target_files=[] → _target_groups=∅ → skip rejection.
                                _target_groups = {
                                    _get_language_group(os.path.splitext(f)[1].lower())
                                    for f in target_files
                                    if os.path.splitext(f)[1]
                                }
                                _target_groups.discard(-1)  # unknown exts are not a signal
                                if _target_groups and _node_group not in _target_groups:
                                    logger.info(
                                        "GraphEnricher: rejecting cross-language resolve of '%s' "
                                        "(node=%s, node_grp=%d, target_files=%s)",
                                        sym_name, node.file_path, _node_group,
                                        [os.path.basename(f) for f in target_files],
                                    )
                                    node = None

                if node is not None:
                    entry: dict[str, Any] = {
                        "name": node.name,
                        "qualname": node.qualname,
                        "file_path": node.file_path,
                        "kind": node.kind,
                        "start_line": node.start_line,
                        "end_line": node.end_line,
                        "docstring": node.docstring or "",
                    }
                    # Function signature with type annotations
                    if getattr(node, "signature", None):
                        entry["signature"] = node.signature
                    # Class: base classes + member methods
                    if node.kind == SymbolKind.CLASS and node.file_path:
                        entry["members"] = self._get_class_members(
                            node.name, node.file_path
                        )
                        if getattr(node, "bases", None):
                            entry["bases"] = node.bases
                    # Read symbol source code (budget-capped)
                    if _source_budget_remaining > 0:
                        _src = self._read_symbol_source(node, max_lines=30)
                        if _src and len(_src) <= _source_budget_remaining:
                            entry["source"] = _src
                            _source_budget_remaining -= len(_src)
                    resolved_symbols.append(entry)
                    if node.file_path:
                        primary_files_set.add(node.file_path)
                else:
                    unresolved_symbols.append(sym_name)
            except Exception as exc:
                logger.debug("get_symbol(%r) failed: %s", sym_name, exc)
                unresolved_symbols.append(sym_name)

        # ── Step 1b: AST-based fallback for unresolved symbols ──────────────
        # When the graph is stale (file was modified after last graph rebuild),
        # graph.get_symbol() returns None for symbols that actually exist.
        # Try file-level AST parse as a fallback before declaring confidence=0.
        if unresolved_symbols:
            _ast_resolved = self._resolve_unresolved_via_ast(
                unresolved_symbols, target_files, spec,
            )
            for _entry in _ast_resolved:
                resolved_symbols.append(_entry)
                if _entry.get("file_path"):
                    primary_files_set.add(_entry["file_path"])
            # Remove AST-resolved names from unresolved list
            _resolved_names = {e["name"] for e in _ast_resolved}
            unresolved_symbols = [s for s in unresolved_symbols if s not in _resolved_names]
            if _ast_resolved:
                logger.info(
                    "GraphEnricher: AST fallback resolved %d/%d unresolved symbols (%s)",
                    len(_ast_resolved),
                    len(unresolved_symbols) + len(_ast_resolved),
                    [e["name"] for e in _ast_resolved],
                )

        # ── Step 2: callers / callees for resolved symbols ────────────────────
        _caller_cap = _depth_level["caller_cap"]
        _callee_cap = _depth_level["callee_cap"]
        _hop2_parent_cap = _depth_level["hop2_parent_cap"]
        _hop2_edge_cap = _depth_level["hop2_edge_cap"]
        _impact_cap = _depth_level["impact_cap"]

        for sym_info in resolved_symbols:
            sym_name = sym_info["name"]
            sym_file = sym_info.get("file_path")

            # callers
            try:
                caller_edges = self._facade.get_callers(sym_name, sym_file) or []
                caller_list: list[dict[str, Any]] = []
                for edge in caller_edges[:_caller_cap]:
                    caller_list.append({
                        "symbol": getattr(edge, "caller_symbol", ""),
                        "file": getattr(edge, "caller_file", ""),
                        "confidence": getattr(edge, "confidence", 1.0),
                    })
                    f = getattr(edge, "caller_file", None)
                    if f:
                        impact_files_set.add(f)
                callers_map[sym_name] = caller_list
            except Exception as exc:
                logger.debug("get_callers(%r) failed: %s", sym_name, exc)

            # callees
            try:
                callee_edges = self._facade.get_callees(sym_name, sym_file) or []
                callee_list: list[dict[str, Any]] = []
                for edge in callee_edges[:_callee_cap]:
                    callee_list.append({
                        "symbol": getattr(edge, "callee_symbol", ""),
                        "file": getattr(edge, "callee_file", ""),
                        "confidence": getattr(edge, "confidence", 1.0),
                        "is_mutating": getattr(edge, "is_mutating", False),
                    })
                    f = getattr(edge, "callee_file", None)
                    if f:
                        impact_files_set.add(f)
                callees_map[sym_name] = callee_list
            except Exception as exc:
                logger.debug("get_callees(%r) failed: %s", sym_name, exc)

        # ── Step 2.5: 2-hop expansion (callers of callers, callees of callees) ──
        # Extends impact_files with transitive dependencies for deeper analysis.
        # P2: hop2 limits are adaptive based on scope.
        for sym_info in resolved_symbols:
            sym_name = sym_info["name"]
            # Expand callers' callers (hop 2)
            for caller_entry in callers_map.get(sym_name, [])[:_hop2_parent_cap]:
                _c_sym = caller_entry.get("symbol", "")
                if not _c_sym:
                    continue
                try:
                    hop2_edges = self._facade.get_callers(_c_sym) or []
                    for edge in hop2_edges[:_hop2_edge_cap]:
                        f = getattr(edge, "caller_file", None)
                        if f:
                            impact_files_set.add(f)
                except Exception:
                    pass
            # Expand callees' callees (hop 2)
            for callee_entry in callees_map.get(sym_name, [])[:_hop2_parent_cap]:
                _e_sym = callee_entry.get("symbol", "")
                if not _e_sym:
                    continue
                try:
                    hop2_edges = self._facade.get_callees(_e_sym) or []
                    for edge in hop2_edges[:_hop2_edge_cap]:
                        f = getattr(edge, "callee_file", None)
                        if f:
                            impact_files_set.add(f)
                except Exception:
                    pass

        # ── Step 3: enumerate target_files to expose file-level symbol map ───
        # Plan C: Always run regardless of target_symbols (skip only for single_symbol scope).
        # Already resolved names are not added again,
        # methods are already included in their parent class's "members" field, so skip.
        # Give the planner up to 10 top-level symbols in the file.
        # _file_enum_cap=0 means single_symbol scope, so this step itself is skipped.
        if target_files and _file_enum_cap > 0:
            _resolved_names = {s["name"] for s in resolved_symbols}
            _extra_count = 0
            for file_path in target_files:
                if _extra_count >= _file_enum_cap:
                    break
                try:
                    file_syms = self._facade.get_symbols_in_file(file_path) or []
                    for node in file_syms:
                        if _extra_count >= _file_enum_cap:
                            break
                        # Methods are already in parent class's members → skip
                        if node.kind == "method":
                            continue
                        # Already resolved names are not added again
                        if node.name in _resolved_names:
                            continue
                        entry: dict[str, Any] = {
                            "name": node.name,
                            "qualname": node.qualname,
                            "file_path": node.file_path,
                            "kind": node.kind,
                        }
                        if getattr(node, "signature", None):
                            entry["signature"] = node.signature
                        if node.kind == SymbolKind.CLASS and node.file_path:
                            entry["members"] = self._get_class_members(
                                node.name, node.file_path
                            )
                            if getattr(node, "bases", None):
                                entry["bases"] = node.bases
                        resolved_symbols.append(entry)
                        _resolved_names.add(node.name)
                        if node.file_path:
                            primary_files_set.add(node.file_path)
                        _extra_count += 1
                except Exception as exc:
                    logger.debug("get_symbols_in_file(%r) failed: %s", file_path, exc)

        # ── Step 4: related_symbols (scope-aware cap) ────────────────────────
        # Skipped entirely for single_symbol scope (_related_cap == 0).
        related_symbols: list[dict[str, Any]] = []
        if _related_cap > 0:
            for sym_name in (s["name"] for s in resolved_symbols):
                if len(related_symbols) >= _related_cap:
                    break
                try:
                    rels_raw = self._facade.get_related_symbols(sym_name, limit=_related_cap)
                    if rels_raw is None:
                        continue
                    # CallGraphIndexer returns dict with "callees", "callers", "related_symbols" keys
                    # Extract the related_symbols list (list of symbol name strings)
                    if isinstance(rels_raw, dict):
                        # Structured dict from CallGraphIndexer
                        sym_names_list = rels_raw.get("related_symbols", [])
                        for s_name in sym_names_list:
                            if len(related_symbols) >= _related_cap:
                                break
                            if isinstance(s_name, str):
                                related_symbols.append({
                                    "symbol": s_name,
                                    "file": "",
                                    "kind": "",
                                })
                            elif isinstance(s_name, dict):
                                related_symbols.append({
                                    "symbol": s_name.get("symbol", ""),
                                    "file": s_name.get("file", ""),
                                    "kind": s_name.get("kind", ""),
                                })
                    elif isinstance(rels_raw, list):
                        for r in rels_raw:
                            if len(related_symbols) >= _related_cap:
                                break
                            if isinstance(r, dict):
                                related_symbols.append({
                                    "symbol": r.get("symbol", ""),
                                    "file": r.get("file", ""),
                                    "kind": r.get("kind", ""),
                                })
                            elif isinstance(r, str):
                                related_symbols.append({
                                    "symbol": r, "file": "", "kind": "",
                                })
                            else:
                                related_symbols.append({
                                    "symbol": getattr(r, "name", str(r)),
                                    "file": getattr(r, "file_path", ""),
                                    "kind": getattr(r, "kind", ""),
                                })
                except Exception as exc:
                    logger.debug("get_related_symbols(%r) failed: %s", sym_name, exc)

        # ── Step 5: deduplicate files and compute graph_confidence ────────────
        # Remove primary files from impact files to keep them separate
        impact_files_set -= primary_files_set

        primary_files = _dedup_list(primary_files_set)
        impact_files = _dedup_list(impact_files_set)[:_impact_cap]  # adaptive cap

        # graph_confidence: Step 1(graph resolve) + Step 1b(AST fallback) resolve
        # ratio.  AST-resolved symbols get 0.7 weight (partial info, no graph edges).
        # Step 3 file-enumeration additions in the numerator would cause >1.0 overflow, so
        # inversely compute the actual resolved target count from unresolved count.
        if target_symbols:
            _resolved_target_count = len(target_symbols) - len(unresolved_symbols)
            # Count AST-fallback-resolved symbols (source="ast_fallback")
            _ast_resolved_count = sum(
                1 for s in resolved_symbols
                if s.get("source") == "ast_fallback"
            )
            _graph_resolved_count = _resolved_target_count - _ast_resolved_count
            # Full weight for graph-resolved, partial weight for AST-resolved
            _weighted_resolved = _graph_resolved_count + (_ast_resolved_count * 0.7)
            graph_confidence = min(1.0, _weighted_resolved / len(target_symbols))
        elif target_files and primary_files:
            # No target_symbols but target_files are specified and file enumeration
            # (Step 3) found symbols in at least one target file. Common for
            # new-symbol creation requests where the symbol doesn't exist yet.
            _covered = len(set(target_files) & set(primary_files))
            _coverage = _covered / len(target_files)
            graph_confidence = round(min(1.0, _coverage * 0.7), 4)
        else:
            graph_confidence = 0.0

        return {
            "resolved_symbols": resolved_symbols,
            "unresolved_symbols": unresolved_symbols,
            "primary_files": primary_files,
            "impact_files": impact_files,
            "callers": callers_map,
            "callees": callees_map,
            "related_symbols": related_symbols[:20],
            "graph_confidence": round(graph_confidence, 4),
        }


# ── Utility helpers ───────────────────────────────────────────────────────────

def _dedup_list(items) -> list[str]:
    """Return deduplicated list preserving approximate order (set → sorted)."""
    seen = set()
    result = []
    for item in sorted(items):  # sort for determinism
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
