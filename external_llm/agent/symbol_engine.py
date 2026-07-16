"""Symbol resolution and analysis utilities extracted from OperationExecutor.

All functions in this module are pure — they accept values and return values
without depending on OperationExecutor state.  Some take an ``executor``
parameter to access shared facilities (e.g. ``symbol_searcher``); the caller
passes ``self`` at the call site.

This is P2 of the God Object decomposition (StageContext → AstEngine →
SymbolEngine → …).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from .ast_engine import (
    _extract_def_name,
)
from .ast_engine import (
    extract_ast_call_names as _extract_ast_call_names,
)
from .ast_engine import (
    get_symbol_start_line_ast as _get_symbol_start_line_ast,
)

logger = logging.getLogger(__name__)

# ── Call-graph neighbour discovery ───────────────────────────────────────

def collect_graph_neighbors(
    graph_facade: Any,
    target_symbol: str,
    file_path: str,
    max_hops: int = 2,
) -> set:
    """Return bare method names structurally related to *target_symbol* via the call graph.

    Explores both directions (callee: target calls X; caller: X calls target) up to
    *max_hops* BFS levels.  Only same-file symbols are included — cross-file callers/
    callees are noise for sibling selection inside a single class.

    Returns a set of bare names (no class prefix).  Empty set on any failure.
    """
    if not graph_facade or not file_path:
        return set()

    _bare = target_symbol.split(".")[-1] if "." in target_symbol else target_symbol
    related: set = set()
    queue: list = [(target_symbol, 0)]
    visited: set = {target_symbol}

    while queue:
        sym, depth = queue.pop(0)
        if depth >= max_hops:
            continue

        for direction in ("callees", "callers"):
            try:
                if direction == "callees":
                    edges = graph_facade.get_callees(sym, file_path) or []
                else:
                    edges = graph_facade.get_callers(sym, file_path) or []
            except Exception as _ge:
                logger.debug("graph edge query failed sym=%s dir=%s: %s", sym, direction, _ge)
                continue

            for edge in edges:
                # Only same-file neighbors to avoid cross-module noise
                neighbor_file = getattr(edge, "callee_file" if direction == "callees" else "caller_file", "") or ""
                if neighbor_file and neighbor_file != file_path:
                    continue
                neighbor_sym = (
                    getattr(edge, "callee_symbol", "") if direction == "callees"
                    else getattr(edge, "caller_symbol", "")
                )
                if not neighbor_sym or neighbor_sym in visited:
                    continue
                visited.add(neighbor_sym)
                nb_bare = neighbor_sym.split(".")[-1] if "." in neighbor_sym else neighbor_sym
                # Skip the target itself
                if nb_bare == _bare:
                    continue
                related.add(nb_bare)
                if depth + 1 < max_hops:
                    queue.append((neighbor_sym, depth + 1))

    return related

# ── Tiered sibling ranking ───────────────────────────────────────────────

_TIER_REASON: dict[int, str] = {
    5: "graph",
    4: "direct_call",
    3: "intent_ref",
    2: "name_family",
    1: "private",
    0: "",
}

def rank_class_siblings_by_relevance(
    sigs: list[str],
    intent: str,
    target_symbol: str,
    budget_chars: int = 1_600,
    symbol_source: str = "",
    graph_neighbors: Optional[set] = None,
) -> tuple[list[tuple[str, str]], int]:
    """Select class sibling method signatures by relevance, not position.

    Ranking tiers (higher = shown first, stable within tier):
      5 — method reachable within 2 hops in the call graph (graph facade, multi-hop)
      4 — method is directly called by the target symbol (AST, 1-hop fallback)
      3 — method name appears in the op intent (textual reference)
      2 — shares name prefix with the target (same naming family)
      1 — underscore-private (likely a helper)
      0 — everything else

    Tiers 4-5 are structural/semantic; tier 3 is textual; tiers 1-2 are name heuristics.
    Budget cap is applied AFTER ranking so the highest-tier sigs are never crowded out.

    Returns:
        (selected, omitted_count) where selected is [(sig, reason_label), ...],
        preserving original positional order within each tier.
    """
    _bare = target_symbol.split(".")[-1] if "." in target_symbol else target_symbol
    _prefix = _bare[:max(4, len(_bare) // 2)].lower()
    _intent_words: set = set()
    if intent:
        _cur = []
        for _ch in intent:
            if _ch.isalnum() or _ch == '_':
                _cur.append(_ch)
            else:
                _w = ''.join(_cur)
                if len(_w) >= 4 and (_w[0].isidentifier() or _w[0] == '_'):
                    _intent_words.add(_w.lower())
                _cur = []
        _w = ''.join(_cur)
        if len(_w) >= 4 and (_w[0].isidentifier() or _w[0] == '_'):
            _intent_words.add(_w.lower())
    _graph_nb: set = {n.lower() for n in (graph_neighbors or set())}
    # Tier 4 fallback: AST direct call names (only used when graph_neighbors is empty)
    _ast_calls: set = (
        {n.lower() for n in _extract_ast_call_names(symbol_source, _bare)}
        if symbol_source and not _graph_nb
        else set()
    )

    def _tier(sig: str) -> int:
        m = re.search(r'def (\w+)', sig)
        if not m:
            return 0
        nm = m.group(1)
        nm_lower = nm.lower()
        if nm_lower in _graph_nb:
            return 5
        if nm_lower in _ast_calls:
            return 4
        if nm_lower in _intent_words:
            return 3
        if nm_lower.startswith(_prefix):
            return 2
        if nm.startswith("_"):
            return 1
        return 0

    tiers = [_tier(s) for s in sigs]
    ranked = sorted(range(len(sigs)), key=lambda i: -tiers[i])

    selected_items: list[tuple[int, str, int]] = []
    chars = 0
    omitted = 0
    for i in ranked:
        s = sigs[i]
        if chars + len(s) + 1 > budget_chars:
            omitted += 1
        else:
            chars += len(s) + 1
            selected_items.append((i, s, tiers[i]))

    selected_items.sort(key=lambda x: x[0])
    result: list[tuple[str, str]] = [(s, _TIER_REASON[t]) for _, s, t in selected_items]
    return result, omitted

# ── Anchor / definition utilities ────────────────────────────────────────

def anchor_is_valid_definition(
    name: str, file_path: str, executor: Any,
) -> bool:
    """Check if *name* resolves to a function/class/method definition.

    The symbol searcher uses ``ast.walk`` which descends into function bodies
    and matches local variable assignments as ``kind="constant"``.  INSERT_
    AFTER_SYMBOL requires a proper definition anchor, so we reject variables
    and constants here.
    """
    searcher = getattr(executor, "symbol_searcher", None)
    if not searcher:
        return True  # can't check, assume OK
    try:
        for r in searcher.find_symbol(name, search_path=file_path):
            kind = (
                r.kind if hasattr(r, "kind")
                else (r.get("kind", "") if isinstance(r, dict) else "")
            )
            if kind in ("function", "async_function", "class"):
                return True
        return False
    except Exception:
        return True  # on error, assume OK (conservative)

def resolve_fixspec_insert_anchor(
    anchor_names: list,
    anchor_line: Optional[int],
    file_path: str,
    fallback_symbol: str,
    executor: Any,
    target_kind: str = "",
) -> str:
    """Resolve a FixSpec INSERT target's anchor to an existing symbol.

    The LLM may place the *new* symbol name in ``anchor_names`` (e.g.
    ``"_read_text_file"``), which does not exist yet and causes checkpoint
    anchor-lost RED.  This helper:

    1. Tries each *anchor_names* candidate — returns the first that exists.
    2. Falls back to *anchor_line* → nearest existing symbol via tree-sitter,
       respecting the intended nesting level (module vs. class scope).
    3. Falls back to the original parent-scope extraction from *fallback_symbol*.

    ``target_kind`` comes from FixSpec target["kind"].  When it is NOT "method",
    the new symbol is module-level, so the anchor must also be module-level
    (no qualified class.method names).  Using a class-method anchor would cause
    INSERT_AFTER_SYMBOL to place code inside the class body.
    """
    # Infer whether the insertion target is a class member or module-level.
    # "method" → class scope; everything else ("function", "class", "module",
    # "") → module-level (INSERT_AFTER_SYMBOL must use a module-level anchor).
    module_level_only = target_kind.lower() not in ("method",)

    # 1. Try each anchor_names candidate — must exist AND respect scope.
    if anchor_names:
        for name in anchor_names:
            if executor._check_symbol_exists(name, file_path):
                # If module-level is required, reject class-qualified names.
                if module_level_only and "." in name:
                    logger.debug(
                        "[FIXSPEC_ANCHOR] skipping class-qualified anchor %r "
                        "(target_kind=%r requires module-level)",
                        name, target_kind,
                    )
                    continue
                # Reject local variables (e.g. 'key', 'provider') that
                # ast.walk finds as assignment targets (kind='constant')
                # but are not valid INSERT anchors.  Gating via the kind
                # field prevents non-definition symbols from short-circuiting
                # the anchor_line-based resolution that follows.
                if not anchor_is_valid_definition(name, file_path, executor):
                    logger.debug(
                        "[FIXSPEC_ANCHOR] skipping non-definition anchor %r "
                        "(local variable, not function/class/method)",
                        name,
                    )
                    continue
                # When anchor_line is set and the symbol starts AFTER
                # anchor_line, the FixSpec wants insertion BEFORE this
                # symbol (e.g., before the class definition).  Fall
                # through to position-based resolution (Step 2) which
                # finds the nearest symbol BEFORE the anchor line,
                # enabling correct pre-class insertion.
                if anchor_line is not None:
                    _sym_start = _get_symbol_start_line_ast(name, file_path)
                    if _sym_start is not None and _sym_start >= anchor_line:
                        logger.debug(
                            "[FIXSPEC_ANCHOR] anchor %r starts at line %d "
                            "(at or after anchor_line=%d) — falling through to "
                            "position-based resolution",
                            name, _sym_start, anchor_line,
                        )
                        continue
                return name

    # 2. Resolve from anchor_line via tree-sitter, filtered by scope.
    if anchor_line and file_path:
        resolved = resolve_symbol_near_line(
            file_path, anchor_line, executor, module_level_only=module_level_only,
        )
        if resolved:
            return resolved

    # 2.5 Text-based fallback: scan backwards from anchor_line for def/class.
    #     tree-sitter may fail on very large files or unavailable grammars.
    if anchor_line and file_path:
        resolved = resolve_anchor_text_based(
            file_path, anchor_line, module_level_only=module_level_only,
        )
        if resolved:
            return resolved

    # 2.7. File-wide text scan — last module-level def/class, no anchor_line needed.
    #      Fires when anchor_line is absent and tree-sitter failed.  Produces a
    #      valid existing anchor so step 3's new-symbol fallback is rarely needed.
    if file_path and module_level_only:
        resolved = find_last_def(
            os.path.join(executor._repo_root or ".", file_path)
            if not os.path.isabs(file_path) else file_path,
            module_level_only=True,
        )
        if resolved:
            logger.debug(
                "[FIXSPEC_ANCHOR] step 2.7 file-wide scan → anchor=%r for %s",
                resolved, file_path,
            )
            return resolved

    # 3. Fallback: scan file for ANY existing definition.
    _abs_path = ""
    if file_path:
        _abs_path = (
            os.path.join(executor._repo_root or ".", file_path)
            if not os.path.isabs(file_path) else file_path
        )
    if _abs_path and os.path.isfile(_abs_path):
        # Try module-level def (functions + classes not inside a class).
        resolved = find_last_def(_abs_path, module_level_only=True)
        if resolved:
            logger.debug(
                "[FIXSPEC_ANCHOR] step 3 file scan → module-level anchor=%r "
                "for %s (instead of non-existent anchor_names[0]=%r)",
                resolved, file_path,
                anchor_names[0] if anchor_names else "(none)",
            )
            return resolved
        # For method-level targets (module_level_only=False), also scan for
        # class methods — the last definition of any kind in the file.
        if not module_level_only:
            resolved = find_last_def(_abs_path, module_level_only=False)
            if resolved:
                logger.debug(
                    "[FIXSPEC_ANCHOR] step 3 file scan → any-level anchor=%r "
                    "for %s (instead of non-existent anchor_names[0]=%r)",
                    resolved, file_path,
                    anchor_names[0] if anchor_names else "(none)",
                )
                return resolved
        # Last resort within existing file: use the fallback symbol's parent
        # scope.  This at least references a known file.
        parent = fallback_symbol.rsplit(".", 1)
        if len(parent) == 2 and parent[0]:
            logger.debug(
                "[FIXSPEC_ANCHOR] step 3 file exists but no def found — "
                "parent scope fallback=%r for %s",
                parent[0], file_path,
            )
            return parent[0]
        # Final: return fallback_symbol itself — it may be wrong but honours
        # the caller's intent better than a hallucinated new-symbol name.
        logger.debug(
            "[FIXSPEC_ANCHOR] step 3 exhausted — returning fallback_symbol=%r",
            fallback_symbol,
        )
        return fallback_symbol
    # No file on disk (create_file scenario or path broken) — cannot scan.
    if anchor_names:
        logger.debug(
            "[FIXSPEC_ANCHOR] file %r does not exist — returning "
            "anchor_names[0]=%r; auto-correction D.3 will skip",
            file_path, anchor_names[0],
        )
        return anchor_names[0]
    parent = fallback_symbol.rsplit(".", 1)
    if len(parent) == 2 and parent[0]:
        return parent[0]
    return fallback_symbol

def find_last_def(file_path: str, module_level_only: bool = False) -> Optional[str]:
    """Return the last ``def``/``class`` name in *file_path*.

    When *module_level_only* is True (default False), only matches
    column-0 definitions (module-level symbols).  When False, also
    matches indented class methods and nested definitions.

    Requires no anchor_line and no tree-sitter — pure text scan over the
    entire file.  Used in ``resolve_fixspec_insert_anchor`` when
    anchor_line is unavailable and earlier strategies have failed.
    """
    _DEF_RE = re.compile(
        r"^(?:async\s+)?(?:def|class)\s+(\w+)" if module_level_only
        else r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)"
    )
    last_name: Optional[str] = None
    try:
        with open(file_path, encoding="utf-8", errors="replace") as _fh:
            for _line in _fh:
                _m = _DEF_RE.match(_line)
                if _m:
                    last_name = _m.group(1)
    except OSError:
        pass
    return last_name

def resolve_anchor_text_based(
    file_path: str,
    anchor_line: int,
    module_level_only: bool = False,
) -> Optional[str]:
    """Find the nearest existing definition above *anchor_line* via text scan.

    Acts as a tree-sitter-free fallback for ``resolve_symbol_near_line``.
    Scans backwards from ``anchor_line`` for ``def``/``class`` lines and
    returns the bare definition name.
    """
    try:
        _abs = file_path if os.path.isabs(file_path) else file_path
        if not os.path.isfile(_abs):
            return None
        with open(_abs, encoding="utf-8", errors="replace") as _fh:
            _lines = _fh.readlines()
    except OSError:
        return None

    # Pattern: def/class name (with optional async and indentation)
    _DEF_RE = re.compile(
        r"^(?P<indent>\s*)(?:async\s+)?(?:def|class)\s+(?P<name>\w+)"
    )

    # Scan backwards from anchor_line (1-indexed, exclusive).
    # Decorated functions are still valid anchors — only skip indented defs
    # when module_level_only=True (to avoid anchoring inside a class body).
    for _ln in range(min(anchor_line - 1, len(_lines)), 0, -1):
        _text = _lines[_ln - 1]
        _m = _DEF_RE.search(_text)
        if not _m:
            continue
        _indent = _m.group("indent")
        if module_level_only and _indent:
            continue
        _name = _m.group("name")
        if _name:
            return _name

    return None

def resolve_symbol_near_line(
    file_path: str, target_line: int, executor: Any,
    module_level_only: bool = False,
) -> Optional[str]:
    """Return the nearest definable symbol at or before *target_line*.

    Walks the full tree-sitter AST (not just module-level ``find_all_symbols``)
    so that class methods, nested functions, and other non-top-level definitions
    are visible.  Returns the qualified name (e.g. ``"AgentLoop._build_session_context"``)
    when possible.

    When ``module_level_only=True``, only module-level (non-class-member) symbols
    are considered.  This prevents a class method from being used as anchor for a
    module-level insertion, which would otherwise cause INSERT_AFTER_SYMBOL to
    place code inside the class body instead of at module scope.
    """
    try:
        from external_llm.languages.tree_sitter_utils import (
            get_parser,
            grammar_key_for_path,
            is_available,
            parse_to_tree,
        )
        if not is_available():
            logger.warning(
                "[FIXSPEC_ANCHOR] tree-sitter core unavailable for %s — "
                "anchor resolution fallback path used (line-proximity only)",
                file_path,
            )
            return None

        _repo = getattr(executor, "_repo_root", "") or ""
        abs_path = (
            os.path.join(_repo, file_path)
            if _repo and not os.path.isabs(file_path)
            else file_path
        )
        if not os.path.isfile(abs_path):
            return None

        with open(abs_path, encoding="utf-8") as f:
            content = f.read()

        lang = grammar_key_for_path(file_path)
        if not lang:
            return None

        if get_parser(lang) is None:
            logger.warning(
                "[FIXSPEC_ANCHOR] tree-sitter grammar unavailable for %s (lang=%s) — "
                "anchor resolution fallback path used",
                file_path, lang,
            )
            return None

        tree = parse_to_tree(content, lang)
        if tree is None:
            logger.warning(
                "[FIXSPEC_ANCHOR] tree-sitter parse failed for %s (lang=%s) — "
                "anchor resolution fallback path used",
                file_path, lang,
            )
            return None

        # Walk the full tree; collect definable nodes (skip nested functions —
        # only recurse into classes, not functions, so inner/local defs are not
        # mistaken for insertion anchors).
        _DEF_NODE_TYPES = {
            "function_definition", "async_function_definition",
            "class_definition", "decorated_definition",
            "function_declaration", "method_declaration",
            "method_definition", "constructor_declaration",
            "class_declaration", "interface_declaration",
            "type_alias_declaration", "enum_declaration",
            "object_declaration", "lexical_declaration",
            "variable_declaration", "type_declaration",
        }
        _RECURSE_TYPES = {
            "class_definition", "class_declaration",
            "interface_declaration", "enum_declaration",
            "object_declaration",
        }
        entries: list = []  # (qualified_name, start_line)
        code_bytes = content.encode("utf-8")

        def _walk(node, parent_class: str = "") -> None:
            if node.type in _DEF_NODE_TYPES:
                name = _extract_def_name(node, code_bytes)
                if name:
                    start = node.start_point.row + 1  # 1-indexed
                    if parent_class:
                        qualified = f"{parent_class}.{name}"
                    else:
                        qualified = name
                    entries.append((qualified, start))
                    # Only recurse into classes (not functions) to avoid
                    # collecting nested/local definitions as candidates.
                    if node.type in _RECURSE_TYPES:
                        for child in node.children:
                            _walk(child, qualified)
                    return  # stop descent regardless (avoid nested funcs)
            # Default: recurse
            for child in node.children:
                _walk(child, parent_class)

        _walk(tree.root_node)

        if not entries:
            return None

        # Apply scope filter before selecting the best candidate.
        if module_level_only:
            module_entries = [(q, s) for q, s in entries if "." not in q]
            if module_entries:
                entries = module_entries
            else:
                logger.debug(
                    "[FIXSPEC_ANCHOR] module_level_only=True but no module-level "
                    "symbols found before line %d in %s — using all entries as fallback",
                    target_line, file_path,
                )

        # Find the entry with the largest start_line < target_line.
        best: Optional[tuple] = None
        for qualified, start in entries:
            if start < target_line:
                if best is None or start > best[1]:
                    best = (qualified, start)
        if best:
            logger.debug(
                "[FIXSPEC_ANCHOR] resolved anchor=%r (start_line=%d) for target_line=%d "
                "in %s (module_level_only=%s)",
                best[0], best[1], target_line, file_path, module_level_only,
            )
            return best[0]
    except Exception as _ane:
        logger.debug(
            "[FIXSPEC_ANCHOR] resolve_symbol_near_line exception for %s: %s",
            file_path, _ane,
        )
    return None
