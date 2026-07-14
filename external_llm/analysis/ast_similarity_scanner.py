"""
AST Similarity Scanner — duplicate/similar symbol candidate discovery.

Role: find structurally similar symbol pairs in a file and surface them as
SimilarityCandidate objects.  The scanner is a *candidate generator*, not a
decision maker — planner LLM judges whether a pair is a true duplicate and
what refactoring action is appropriate.

Design: after spec_resolver identifies target files for low-spec cleanup /
refactor requests, this scanner runs over those files, normalises AST
representations, computes pairwise similarity, and injects the top candidates
into spec.metadata["duplicate_candidates"] for the planner.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from . import parse_cache

logger = logging.getLogger(__name__)

# ── Candidate data model ───────────────────────────────────────────────────────

@dataclass
class SimilarityCandidate:
    """A pair of symbols that are structurally similar."""
    file: str
    symbol_a: str           # qualified name, e.g. "MyClass._foo"
    symbol_b: str
    similarity: float       # 0.0–1.0
    duplication_kind: str   # see DUPLICATION_KINDS below
    shared_inputs: list[str] = field(default_factory=list)
    extractable: bool = False
    anchor_fingerprints: list[str] = field(default_factory=list)
    anchor_texts: list[str] = field(default_factory=list)
    # suggested_action values:
    #   "extract_shared_helper"  — extractable=True pair; helper extraction is appropriate
    #   "similar_structure_only" — structure is similar but extraction is not recommended
    #                              (extractable=False: low score, few anchors, or exit divergence)
    #   "consolidate_b_into_a"   — shared_prefix_suffix; inline one into the other
    #   "analyze"                — general similarity; requires LLM judgement
    #   "too_dissimilar"         — below min_similarity threshold
    #   "forced_pair_unresolved" — forced pair but symbols missing or normalisation failed
    suggested_action: str = ""
    suggested_primary_symbol: str = ""
    forced: bool = False    # True when pair was explicitly requested via intent_symbols
    # forced_reason classifies *why* a forced pair ended up in the result:
    #   ""                      — normal top-N candidate (may also be forced=True if in top-N)
    #   "scanner_limit_excluded"— computed but below min_similarity
    #   "normalise_failed"      — symbol found but AST normalisation raised
    #   "symbol_missing"        — one/both symbols absent from scanned files
    forced_reason: str = ""
    # shadow_overlaps: semantic overlap signals logged for distribution observation.
    # Not used for action decisions yet — populated to guide future threshold calibration.
    #   call_overlap       — Jaccard of call_shapes sets
    #   result_key_overlap — Jaccard of result dict key access sets
    shadow_overlaps: dict[str, float] = field(default_factory=dict)
    # Canonical edit_kind for this pair, declared by the scanner from
    # (suggested_action, extractable).  Downstream (DPB) consumes this directly
    # instead of guessing a fallback; empty string means the pair is not
    # actionable as a paired plan.
    paired_edit_kind: str = ""

    def __post_init__(self) -> None:
        if not self.paired_edit_kind:
            self.paired_edit_kind = paired_edit_kind_for(self.suggested_action)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "similarity": round(self.similarity, 3),
            "duplication_kind": self.duplication_kind,
            "shared_inputs": self.shared_inputs,
            "extractable": self.extractable,
            "anchor_fingerprints": self.anchor_fingerprints,
            "anchor_texts": self.anchor_texts[:3],
            "suggested_action": self.suggested_action,
            "suggested_primary_symbol": self.suggested_primary_symbol,
            "forced": self.forced,
            "forced_reason": self.forced_reason,
            "shadow_overlaps": self.shadow_overlaps,
            "paired_edit_kind": self.paired_edit_kind,
            "reason": _similarity_reason(self),
        }


# suggested_action → canonical edit_kind for the structural pair.  Detector-
# side declaration so DPB doesn't have to invent a default when IntentResult
# fails to propagate edit_kind.  Keep the table closed: any action not listed
# returns "" and the pair is treated as non-actionable downstream.
_PAIRED_EDIT_KIND_BY_ACTION: dict[str, str] = {
    "extract_shared_helper":  "helper_extraction",
    "consolidate_b_into_a":   "delegate_common_logic",
    "similar_structure_only": "local_patch",
    "analyze":                "local_patch",
}


def paired_edit_kind_for(suggested_action: str) -> str:
    """Return the canonical edit_kind for a structural pair, or "" if not actionable.

    The scanner already collapses ``("extract_shared_helper", extractable=False)``
    to ``"similar_structure_only"`` before construction (see _is_extractable
    enforcement), so this lookup only needs the action.
    """
    return _PAIRED_EDIT_KIND_BY_ACTION.get((suggested_action or "").strip(), "")




# ── Normalised AST repr ────────────────────────────────────────────────────────

# suggested_action → human-readable prose for the SCAN tool's ``reason`` field.
# Without this, candidates showed the bare identifier ``extract_shared_helper``
# instead of a sentence; the reason key fills the description/reason fallback
# chain so pairs render as ``file:line a ↔ b — <reason>``.
_SIMILARITY_ACTION_PROSE: dict[str, str] = {
    "extract_shared_helper": "extractable into a shared helper",
    "similar_structure_only": "similar structure, extraction not advised",
    "consolidate_b_into_a": "consolidate one into the other",
    "analyze": "general similarity, needs judgement",
    "too_dissimilar": "below similarity threshold",
    "forced_pair_unresolved": "forced pair unresolved",
}


def _similarity_reason(cand: "SimilarityCandidate") -> str:
    """One-line human summary of a similarity candidate."""
    prose = (
        _SIMILARITY_ACTION_PROSE.get(cand.suggested_action)
        or cand.suggested_action
        or "similar"
    )
    extras: list[str] = []
    if cand.shared_inputs:
        extras.append(f"{len(cand.shared_inputs)} shared input(s)")
    if cand.anchor_fingerprints:
        extras.append(f"{len(cand.anchor_fingerprints)} anchor(s)")
    tail = f"; {', '.join(extras)}" if extras else ""
    return f"{cand.duplication_kind} (sim {cand.similarity:.2f}); {prose}{tail}"


@dataclass
class NormalisedSymbol:
    """Normalised structural representation of a function/method body."""
    qualname: str
    params: list[str]           # parameter names (positional, no self/cls)
    skeleton: list[str]         # top-level stmt type sequence
    stmt_seq: list[str]         # richer normalised stmt strings
    call_shapes: list[str]      # (receiver_kind, method_tail) pairs
    exit_shapes: list[str]      # "return", "raise", "yield"
    guard_count: int            # leading guard-like if-stmts
    assign_density: float       # assignments / total stmts
    try_present: bool
    line_count: int
    anchor_fps: list[str]       # AST-canonical fingerprints (ast.dump hash)
    anchor_texts: list[str]     # ast.unparse representations (verifier-ready)
    result_keys: list[str] = field(default_factory=list)  # dict param key accesses (shadow signal)
    ident_tokens: list[str] = field(default_factory=list)  # distinctive identifiers/constants (role signal)


def _param_names(node: ast.FunctionDef) -> list[str]:
    args = node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    skip = {"self", "cls"}
    return [a.arg for a in all_args if a.arg not in skip]


_BUILTINS = frozenset({
    "len", "range", "enumerate", "zip", "map", "filter", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "print", "isinstance",
    "getattr", "setattr", "hasattr", "type", "repr", "sorted", "reversed",
    "min", "max", "sum", "any", "all", "open", "super", "next", "iter",
    "append", "extend", "update", "pop", "get", "items", "keys", "values",
    "format", "join", "split", "strip", "startswith", "endswith",
    "logging", "logger", "log",
})


def _norm_expr_str(node: ast.expr) -> str:
    """Produce a normalised string for an expression (no alpha-norm needed here)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return "NUM"
        if isinstance(node.value, str):
            return "STR"
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id if node.id in _BUILTINS else "NAME"
    if isinstance(node, ast.Attribute):
        return f"ATTR.{node.attr}"
    if isinstance(node, ast.Call):
        return f"CALL({_norm_expr_str(node.func)})"
    if isinstance(node, ast.BinOp):
        return "BINOP"
    if isinstance(node, ast.Compare):
        return "CMP"
    if isinstance(node, ast.BoolOp):
        return "BOOLOP"
    if isinstance(node, ast.UnaryOp):
        return "UNARY"
    if isinstance(node, ast.IfExp):
        return "IFEXP"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return "COLLECTION"
    if isinstance(node, ast.Dict):
        return "DICT"
    if isinstance(node, ast.Subscript):
        return f"SUBSCR({_norm_expr_str(node.value)})"
    return "EXPR"


def _norm_stmt(node: ast.stmt) -> str:
    """Richer stmt normalization: encodes structural shape beyond just the node type."""
    if isinstance(node, ast.Return):
        val = _norm_expr_str(node.value) if node.value else "None"
        return f"return:{val}"
    if isinstance(node, ast.Assign):
        n_tgts = len(node.targets)
        val_class = _norm_expr_str(node.value)
        return f"assign:{n_tgts}:{val_class}"
    if isinstance(node, ast.AugAssign):
        return f"augassign:{type(node.op).__name__}"
    if isinstance(node, ast.AnnAssign):
        return "annassign"
    if isinstance(node, ast.Expr):
        return f"expr:{_norm_expr_str(node.value)}"
    if isinstance(node, ast.If):
        # encode elif/else presence and branch count
        branches = 1
        cur = node
        while isinstance(cur.orelse, list) and len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            branches += 1
            cur = cur.orelse[0]
        has_else = bool(cur.orelse)
        return f"if:b{branches}:{'else' if has_else else 'noelse'}"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        body_len = len(node.body)
        has_else = bool(node.orelse)
        return f"for:{'else' if has_else else 'noelse'}:body{min(body_len, 5)}"
    if isinstance(node, ast.While):
        return f"while:body{min(len(node.body), 5)}"
    if isinstance(node, ast.Try):
        n_handlers = len(node.handlers)
        has_else = bool(node.orelse)
        has_final = bool(node.finalbody) if hasattr(node, 'finalbody') else False
        return f"try:h{n_handlers}:{'else' if has_else else ''}:{'fin' if has_final else ''}"
    if isinstance(node, ast.With):
        return f"with:{len(node.items)}"
    if isinstance(node, ast.Raise):
        exc_class = _norm_expr_str(node.exc) if node.exc else "None"
        return f"raise:{exc_class}"
    if isinstance(node, ast.Delete):
        return "del"
    if isinstance(node, ast.Pass):
        return "pass"
    if isinstance(node, (ast.Break, ast.Continue)):
        return type(node).__name__.lower()
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "nested_def"
    if isinstance(node, ast.ClassDef):
        return "nested_class"
    return "stmt"


def _call_shape(call_node: ast.Call) -> Optional[str]:
    func = call_node.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            recv = func.value.id if func.value.id in _BUILTINS else "OBJ"
        elif isinstance(func.value, ast.Attribute):
            recv = "ATTR"
        else:
            recv = "EXPR"
        return f"{recv}.{func.attr}"
    if isinstance(func, ast.Name):
        return f"CALL.{func.id}" if func.id in _BUILTINS else "CALL.local"
    return None


def _collect_calls(body: list[ast.stmt]) -> list[str]:
    shapes = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            s = _call_shape(node)
            if s:
                shapes.append(s)
    return shapes


def _exit_shapes(body: list[ast.stmt]) -> list[str]:
    exits = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Return):
            exits.append("return")
        elif isinstance(node, ast.Raise):
            exits.append("raise")
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            exits.append("yield")
    return list(dict.fromkeys(exits))


def _extract_result_keys(body: list[ast.stmt], var_name: str = "result") -> list[str]:
    """Extract string keys accessed on a dict parameter named var_name.

    Captures result.get("key") and result["key"] patterns — both read and write.
    Used as a shadow signal for semantic domain overlap between two functions.
    """
    keys: set = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        # result.get("key") pattern
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var_name
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        # result["key"] subscript (read or assignment target)
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == var_name
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return sorted(keys)


def _ident_tokens(func_node: ast.FunctionDef) -> list[str]:
    """Distinctive identifier/constant tokens of a function — its semantic domain.

    The structural metrics (skeleton/stmt_seq/call_shapes) alpha-normalise
    aggressively, so two functions in unrelated domains can score high on
    coincidental shape alone.  This token set keeps the *names*: attribute
    accesses, referenced non-builtin names, keyword-arg names, and string
    constants.  Genuine copy-paste duplicates call the same functions and
    touch the same attributes, so their token Jaccard stays high; pairs that
    merely share control-flow shape diverge sharply.

    Parameter names are excluded — they are measured separately by
    _param_role_similarity, and duplicates whose params were renamed at
    copy time should not be penalised here.
    """
    args = func_node.args
    params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    toks: set = set()
    for n in ast.walk(func_node):
        if isinstance(n, ast.Attribute) and n.attr not in _BUILTINS:
            toks.add(f"attr:{n.attr}")
        elif isinstance(n, ast.Name):
            if n.id in params or n.id in _BUILTINS or n.id in ("self", "cls"):
                continue
            toks.add(f"name:{n.id}")
        elif (isinstance(n, ast.Constant)
                and isinstance(n.value, str) and len(n.value) >= 3):
            toks.add(f"str:{n.value[:30]}")
        elif isinstance(n, ast.keyword) and n.arg:
            toks.add(f"kw:{n.arg}")
    return sorted(toks)


def _guard_count(body: list[ast.stmt]) -> int:
    count = 0
    for stmt in body:
        if not isinstance(stmt, ast.If):
            break
        if len(stmt.body) == 1 and isinstance(stmt.body[0], (ast.Return, ast.Raise)):
            count += 1
        else:
            break
    return count


def _extract_ast_anchors(body: list[ast.stmt]) -> tuple[list[str], list[str]]:
    """Extract 2–5 representative statements as AST-canonical anchors.

    Returns (fingerprints, anchor_texts) where:
    - fingerprints: md5(ast.dump(stmt, include_attributes=False))[:12]
      AST-canonical — immune to whitespace/quote/formatting changes.
    - anchor_texts: ast.unparse(stmt)
      Normalised code text — suitable for verifier text matching and
      for planner display.

    When the top-level body has very few statements (≤ 2), the function body
    is likely wrapped in a single try/with/if block.  In that case we
    recurse one level into that wrapper to find inner statements — otherwise
    functions whose entire body is `try: ... except: pass` would yield only
    1 anchor and never be considered extractable.
    """
    _SKIP = (ast.Pass, ast.Import, ast.ImportFrom)

    def _is_docstring(node: ast.stmt) -> bool:
        return isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant)

    def _inner_stmts(wrapper: ast.stmt) -> list[ast.stmt]:
        """Return the inner statement list of a wrapping compound statement."""
        if isinstance(wrapper, ast.Try):
            return list(wrapper.body)
        if isinstance(wrapper, ast.With):
            return list(wrapper.body)
        if isinstance(wrapper, ast.If):
            return list(wrapper.body)
        if isinstance(wrapper, (ast.For, ast.AsyncFor, ast.While)):
            return list(wrapper.body)
        return []

    def _candidates(stmts: list[ast.stmt]) -> list[ast.stmt]:
        return [s for s in stmts if not _is_docstring(s) and not isinstance(s, _SKIP)]

    candidates = _candidates(body)

    # If body has ≤ 2 meaningful stmts, look one level inside the wrapper
    if len(candidates) <= 2 and len(candidates) == 1:
        inner = _inner_stmts(candidates[0])
        if inner:
            inner_cands = _candidates(inner)
            if len(inner_cands) > len(candidates):
                candidates = inner_cands

    fps: list[str] = []
    texts: list[str] = []

    for stmt in candidates:
        try:
            dump = ast.dump(stmt, include_attributes=False)
            fp = hashlib.md5(dump.encode(), usedforsecurity=False).hexdigest()[:12]
            text = ast.unparse(stmt)
        except Exception:
            continue

        if not text or len(text.strip()) < 4:
            continue

        fps.append(fp)
        texts.append(text.strip())

        if len(fps) >= 5:
            break

    return fps, texts


def normalise_function(
    func_node: ast.FunctionDef,
    qualname: str,
) -> NormalisedSymbol:
    body = func_node.body
    skeleton = [type(s).__name__ for s in body]
    stmt_seq = [_norm_stmt(s) for s in body]
    calls = _collect_calls(body)
    exits = _exit_shapes(body)
    assigns = sum(1 for s in body if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)))
    total = len(body) or 1
    guard_c = _guard_count(body)
    has_try = any(isinstance(s, ast.Try) for s in body)
    line_count = (getattr(func_node, "end_lineno", func_node.lineno) - func_node.lineno + 1)
    fps, texts = _extract_ast_anchors(body)
    result_keys = _extract_result_keys(body)

    return NormalisedSymbol(
        qualname=qualname,
        params=_param_names(func_node),
        skeleton=skeleton,
        stmt_seq=stmt_seq,
        call_shapes=calls,
        exit_shapes=exits,
        guard_count=guard_c,
        assign_density=assigns / total,
        try_present=has_try,
        line_count=line_count,
        anchor_fps=fps,
        anchor_texts=texts,
        result_keys=result_keys,
        ident_tokens=_ident_tokens(func_node),
    )


# ── Similarity metrics ─────────────────────────────────────────────────────────

def _sequence_similarity_upper_bound(la: int, lb: int) -> float:
    """Cheap upper bound of ``_sequence_similarity`` from lengths alone.

    LCS length is at most ``min(la, lb)``, so the normalised score is at most
    ``2*min/(la+lb)``.  Used to prune pairs before the O(la*lb) DP runs.
    """
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return 2 * min(la, lb) / (la + lb)


def _sequence_similarity(a: list[str], b: list[str]) -> float:
    """Normalised LCS similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return 2 * dp[la][lb] / (la + lb)


def _set_similarity(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _param_role_similarity(a: NormalisedSymbol, b: NormalisedSymbol) -> float:
    """Compare parameter roles: shared-name Jaccard + count-ratio blend."""
    pa, pb = set(a.params), set(b.params)
    la, lb = len(a.params), len(b.params)

    # Jaccard on shared param names
    if pa or pb:
        name_jaccard = len(pa & pb) / len(pa | pb)
    else:
        name_jaccard = 1.0

    # count ratio
    if la == 0 and lb == 0:
        count_ratio = 1.0
    elif la == 0 or lb == 0:
        count_ratio = 0.0
    else:
        count_ratio = min(la, lb) / max(la, lb)

    return 0.6 * name_jaccard + 0.4 * count_ratio


def compute_similarity(a: NormalisedSymbol, b: NormalisedSymbol) -> float:
    """Weighted similarity score in [0, 1]."""
    return (
        0.40 * _sequence_similarity(a.skeleton, b.skeleton)
        + 0.25 * _sequence_similarity(a.stmt_seq, b.stmt_seq)
        + 0.15 * _set_similarity(a.call_shapes, b.call_shapes)
        + 0.10 * _set_similarity(a.exit_shapes, b.exit_shapes)
        + 0.10 * _param_role_similarity(a, b)
    )


# ── Duplication kind classifier ────────────────────────────────────────────────

_NEAR_DUPLICATE_MIN_PARAM_SIM = 0.30  # below this → functions have different roles


def _classify_duplication_kind(a: NormalisedSymbol, b: NormalisedSymbol, score: float) -> str:
    if score >= 0.90:
        # Structurally near-duplicate but with incompatible signatures?
        # e.g. one function takes (op, state, ctx) and the other takes no args —
        # they serve different purposes regardless of structural similarity.
        # Downgrade to shared_control_flow so the pair is not mistaken for a
        # genuine refactoring target.
        param_sim = _param_role_similarity(a, b)
        if param_sim < _NEAR_DUPLICATE_MIN_PARAM_SIM:
            pass  # fall through to kind-specific checks below
        else:
            return "near_duplicate"
    if a.try_present and b.try_present and _sequence_similarity(a.skeleton, b.skeleton) >= 0.70:
        return "shared_try_except_flow"
    if a.guard_count >= 1 and b.guard_count >= 1 and _sequence_similarity(a.skeleton, b.skeleton) >= 0.60:
        return "shared_guarded_core"
    if a.assign_density >= 0.35 and b.assign_density >= 0.35 and _sequence_similarity(a.stmt_seq, b.stmt_seq) >= 0.65:
        return "shared_accumulator_pattern"
    if (a.skeleton and b.skeleton
            and (a.skeleton[:2] == b.skeleton[:2] or a.skeleton[-2:] == b.skeleton[-2:])):
        return "shared_prefix_suffix"
    return "shared_control_flow"


# ── Shared inputs ─────────────────────────────────────────────────────────────

def _infer_shared_inputs(a: NormalisedSymbol, b: NormalisedSymbol) -> list[str]:
    return sorted(set(a.params) & set(b.params))


# ── Suggested action ──────────────────────────────────────────────────────────

# call_overlap threshold for structural kinds (shared_try_except_flow / shared_control_flow /
# shared_guarded_core).  Below this value the two functions call different method ecosystems
# — extraction produces a helper that cannot be meaningfully shared.
#
# Calibrated from shadow data (3 cases, 2026-05-07):
#   SL59 _normalise_*        : call_overlap=1.000 → extract ✅
#   SL62 _cached_*           : call_overlap=0.647 → extract ✅
#   SL60 _run_lineage_* pair : call_overlap=0.136 → similar_structure_only ✅
# Gap between lowest "extract" (0.647) and highest "no-extract" (0.136) is wide.
# 0.25 sits conservatively in that gap.  Raise if false-negatives emerge.
_CALL_OVERLAP_EXTRACT_THRESHOLD = 0.25

# ident_tokens Jaccard below this → the pair shares control-flow shape but
# lives in different semantic domains (calls different functions, touches
# different attributes/constants).  Such pairs are coincidental structure,
# not duplication — drop them from candidates entirely (forced pairs bypass).
#
# Calibrated from asi.py false positives vs copy-paste fixtures (2026-06-12):
#   _build_interrupt_note / _build_agent_interrupt_note : 0.073 → drop ✅
#   _strip_ansi / _ProgressPrinter._plain               : 0.062 → drop ✅
#   _drain_stdin / _run_esc_watcher (same stdin machinery): 0.583 → keep
#   copy-paste duplicate, params renamed                 : 0.692 → keep ✅
#   typical copy-paste duplicate                         : 0.778 → keep ✅
# Gap between drops (≤0.073) and keeps (≥0.583) is wide; 0.25 sits in it.
_IDENT_OVERLAP_MIN = 0.25


def _suggest_action(a: NormalisedSymbol, b: NormalisedSymbol, kind: str, score: float) -> tuple[str, str]:
    # call_overlap applies at every score level: low overlap means the two functions
    # call different method ecosystems — extracting a shared helper produces an
    # incoherent abstraction regardless of structural similarity score.
    call_overlap = _set_similarity(a.call_shapes, b.call_shapes)
    if score >= 0.90:
        if call_overlap < _CALL_OVERLAP_EXTRACT_THRESHOLD:
            logger.debug(
                "[AST_SCAN] call_overlap guard fired: pair=(%s, %s) sim=%.3f "
                "call_overlap=%.3f < %.2f → similar_structure_only",
                a.qualname, b.qualname, score, call_overlap,
                _CALL_OVERLAP_EXTRACT_THRESHOLD,
            )
            return "similar_structure_only", a.qualname
        return "extract_shared_helper", a.qualname
    if kind in ("shared_guarded_core", "shared_try_except_flow", "shared_control_flow"):
        if score >= 0.82:
            if call_overlap < _CALL_OVERLAP_EXTRACT_THRESHOLD:
                logger.debug(
                    "[AST_SCAN] call_overlap guard fired: pair=(%s, %s) sim=%.3f "
                    "call_overlap=%.3f < %.2f → similar_structure_only",
                    a.qualname, b.qualname, score, call_overlap,
                    _CALL_OVERLAP_EXTRACT_THRESHOLD,
                )
                return "similar_structure_only", a.qualname
            return "extract_shared_helper", a.qualname
        return "similar_structure_only", a.qualname
    if kind == "shared_prefix_suffix":
        primary = a.qualname if a.line_count <= b.line_count else b.qualname
        return "consolidate_b_into_a", primary
    return "analyze", a.qualname


# ── Extractable judgment ──────────────────────────────────────────────────────

def _is_extractable(
    na: NormalisedSymbol,
    nb: NormalisedSymbol,
    action: str,
    score: float,
    cross_anchor_count: int,
) -> bool:
    """True when extraction is structurally plausible — not just action==extract.

    Conditions:
    - action must suggest extraction
    - similarity strong enough that the shared core is non-trivial
    - at least 2 cross-source anchor statements (same AST fingerprint in BOTH symbols)
    - both symbols are non-trivial size
    - exit shapes are not too divergent
    - call ecosystems overlap enough that a shared helper is coherent
    """
    if action != "extract_shared_helper":
        return False
    if score < 0.82:
        return False
    # cross_anchor_count = anchors whose AST fingerprint appears in BOTH sources.
    # Anchors from only one source mean the "shared" region is one-sided — extraction
    # would produce a helper that the other source cannot actually call unchanged.
    if cross_anchor_count < 2:
        return False
    if na.line_count < _MIN_LINE_COUNT or nb.line_count < _MIN_LINE_COUNT:
        return False
    # exit divergence: if one raises and the other returns, extraction changes semantics
    exit_sim = _set_similarity(na.exit_shapes, nb.exit_shapes)
    if exit_sim < 0.4:
        return False
    # call ecosystem overlap: low overlap means the two functions call different
    # method sets — extraction produces an incoherent shared helper.
    call_overlap = _set_similarity(na.call_shapes, nb.call_shapes)
    if call_overlap < _CALL_OVERLAP_EXTRACT_THRESHOLD:
        return False
    return True


def _is_trivial_function_body(body: list[ast.stmt]) -> bool:
    """True when function body is a stub — ≤1 meaningful statement.

    A "meaningful" statement excludes docstrings, imports, and pass.
    Two stub functions are structurally similar by coincidence (both are
    trivially short), not by semantic duplication — the scanner should
    not report them as candidates.

    This is intentionally broad: any function with ≤1 meaningful statement
    produces noise in the similarity matrix because its skeleton and stmt_seq
    will match nearly any other such function regardless of semantics.
    These pairs have no refactoring value.
    """
    meaningful = [
        s for s in body
        if not (isinstance(s, ast.Expr)
                and isinstance(getattr(s, 'value', None), ast.Constant))
        and not isinstance(s, (ast.Pass, ast.Import, ast.ImportFrom))
    ]
    return len(meaningful) <= 1


# ── Symbol collection ──────────────────────────────────────────────────────────

@dataclass
class _SymbolEntry:
    qualname: str
    node: ast.FunctionDef
    parent_class: Optional[str]


def _collect_symbols(tree: ast.Module) -> list[_SymbolEntry]:
    entries: list[_SymbolEntry] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entries.append(_SymbolEntry(qualname=node.name, node=node, parent_class=None))
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entries.append(_SymbolEntry(
                        qualname=f"{node.name}.{child.name}",
                        node=child,
                        parent_class=node.name,
                    ))
    return entries


# ── Public API ────────────────────────────────────────────────────────────────

_MIN_LINE_COUNT = 5
_MIN_LINE_COUNT_CROSS = 8
_FILTER_BONUS = 0.05      # similarity bonus for filter-preferred symbols


def scan_similarity_candidates(
    repo_root: str,
    file_paths: list[str],
    symbol_filter: Optional[list[str]] = None,
    max_candidates: int = 20,
    min_similarity: float = 0.72,
    forced_pairs: Optional[list[tuple[str, str]]] = None,
) -> list[SimilarityCandidate]:
    """
    Scan files for structurally similar symbol pairs.

    symbol_filter is a *rerank bonus*, not a hard filter.  All symbols in
    the file are compared; pairs that include a filter symbol receive a small
    similarity bonus so they rank higher.  This avoids reinforcing an
    incorrect initial symbol set from upstream spec grounding.

    forced_pairs: list of (bare_name_a, bare_name_b) tuples that must be
    evaluated regardless of min_similarity.  Each is included in the result
    with forced=True so DPB can make an informed decision about synthetic
    pairs.  suggested_action is set to "too_dissimilar" when similarity is
    below min_similarity, allowing DPB to distinguish "scanner didn't check"
    from "scanner checked and found low similarity".
    """
    _filter_set: frozenset = frozenset(symbol_filter or [])
    _filter_bare: frozenset = frozenset(s.split(".")[-1] for s in _filter_set)

    all_candidates: list[SimilarityCandidate] = []
    # Per-file normalised symbols (qualname AND bare-name keys) — built once
    # here and reused by the forced-pairs section below, which previously
    # re-read and re-parsed every file from scratch.
    _normed_cache: dict[str, dict[str, NormalisedSymbol]] = {}

    for fpath in file_paths:
        abs_path = fpath if os.path.isabs(fpath) else os.path.join(repo_root, fpath)
        tree = parse_cache.parse_ast(abs_path)
        if tree is None:
            logger.debug("[AST_SCAN] cannot read/parse %s", fpath)
            continue

        all_entries = _collect_symbols(tree)

        _by_name: dict[str, NormalisedSymbol] = {}
        entries: list[_SymbolEntry] = []
        normed: list[NormalisedSymbol] = []
        for e in all_entries:
            try:
                n = normalise_function(e.node, e.qualname)
            except Exception:
                logger.debug("[AST_SCAN] normalise failed for %s in %s", e.qualname, fpath, exc_info=True)
                continue
            _by_name.setdefault(e.qualname, n)
            _by_name.setdefault(e.qualname.split(".")[-1], n)
            # pairwise scan skips trivially small symbols; forced pairs don't
            if n.line_count >= _MIN_LINE_COUNT:
                entries.append(e)
                normed.append(n)
        _normed_cache[fpath] = _by_name

        if len(entries) < 2:
            continue

        file_candidates: list[SimilarityCandidate] = []
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                na, nb = normed[i], normed[j]

                a_class = entries[i].parent_class
                b_class = entries[j].parent_class
                if a_class != b_class:
                    if na.line_count < _MIN_LINE_COUNT_CROSS or nb.line_count < _MIN_LINE_COUNT_CROSS:
                        continue

                # Skip pairs where BOTH symbols are trivial stubs —
                # they score high on structure but aren't semantically related.
                if (_is_trivial_function_body(entries[i].node.body)
                        and _is_trivial_function_body(entries[j].node.body)):
                    continue

                # Role gate: structural metrics alpha-normalise names away, so
                # same-shape functions from unrelated domains can pass them.
                # Identifier-level overlap restores that signal — below the
                # threshold the pair is coincidental structure, not duplication.
                ident_overlap = _set_similarity(na.ident_tokens, nb.ident_tokens)
                if ident_overlap < _IDENT_OVERLAP_MIN:
                    logger.debug(
                        "[AST_SCAN] ident_overlap gate: pair=(%s, %s) "
                        "ident_overlap=%.3f < %.2f → skipped",
                        na.qualname, nb.qualname, ident_overlap, _IDENT_OVERLAP_MIN,
                    )
                    continue

                # Cheap prune before the O(L²) LCS: the two sequence terms
                # (weights 0.40 + 0.25) are bounded by length ratio; the
                # remaining set/param terms are bounded by 1.0 (0.35 total).
                _seq_ub = _sequence_similarity_upper_bound(len(na.skeleton), len(nb.skeleton))
                if 0.65 * _seq_ub + 0.35 + _FILTER_BONUS < min_similarity:
                    continue

                raw_score = compute_similarity(na, nb)

                # Filter bonus applies only to ranking/filtering — not to kind/action
                # classification.  Inflating score before _classify_duplication_kind
                # pushes pairs across the 0.90 near_duplicate threshold spuriously,
                # and before _suggest_action bypasses call_overlap guards.
                bare_a = na.qualname.split(".")[-1]
                bare_b = nb.qualname.split(".")[-1]
                _filter_matched = bool(_filter_set) and (
                    na.qualname in _filter_set or nb.qualname in _filter_set
                    or bare_a in _filter_bare or bare_b in _filter_bare
                )
                ranking_score = min(1.0, raw_score + _FILTER_BONUS) if _filter_matched else raw_score

                if ranking_score < min_similarity:
                    continue

                # kind and action use raw_score — immune to filter bonus inflation
                kind = _classify_duplication_kind(na, nb, raw_score)
                shared = _infer_shared_inputs(na, nb)
                action, primary = _suggest_action(na, nb, kind, raw_score)

                # Cross-source anchor validation: only fingerprints present in BOTH
                # symbols are true shared anchors.  Anchors from the larger symbol
                # alone are one-sided and cannot locate the region in the other source.
                larger_norm = na if na.line_count >= nb.line_count else nb
                fps_set_a = set(na.anchor_fps)
                fps_set_b = set(nb.anchor_fps)
                cross_fps_set = fps_set_a & fps_set_b
                cross_anchor_count = len(cross_fps_set)
                # Use only cross-source anchors as the shared region hint.
                # Fall back to larger_norm anchors as texts (best-effort display/hint)
                # but pass cross_anchor_count as the structural gate signal.
                fps = [fp for fp in larger_norm.anchor_fps if fp in cross_fps_set]
                texts = [t for fp, t in zip(larger_norm.anchor_fps, larger_norm.anchor_texts, strict=False)
                         if fp in cross_fps_set]
                # If no cross-source anchors but each source has its own, provide
                # the larger symbol's anchors as texts for DPB display (non-gating).
                if not fps:
                    fps = larger_norm.anchor_fps
                    texts = larger_norm.anchor_texts

                extractable = _is_extractable(na, nb, action, raw_score, cross_anchor_count)
                if not extractable and action == "extract_shared_helper":
                    action = "similar_structure_only"

                call_overlap = _set_similarity(na.call_shapes, nb.call_shapes)
                result_key_overlap = _set_similarity(na.result_keys, nb.result_keys)
                exit_sim = _set_similarity(na.exit_shapes, nb.exit_shapes)
                logger.debug(
                    "[AST_SCAN_SHADOW] pair=(%s, %s) raw_sim=%.3f ranking_sim=%.3f action=%s "
                    "call_overlap=%.3f result_key_overlap=%.3f cross_anchors=%d exit_sim=%.3f "
                    "ident_overlap=%.3f",
                    na.qualname, nb.qualname, raw_score, ranking_score, action,
                    call_overlap, result_key_overlap, cross_anchor_count, exit_sim,
                    ident_overlap,
                )

                candidate = SimilarityCandidate(
                    file=fpath,
                    symbol_a=na.qualname,
                    symbol_b=nb.qualname,
                    similarity=raw_score,          # raw score — not inflated by filter bonus
                    duplication_kind=kind,
                    shared_inputs=shared[:6],
                    extractable=extractable,
                    anchor_fingerprints=fps[:5],
                    anchor_texts=texts[:3],
                    suggested_action=action,
                    suggested_primary_symbol=primary,
                    shadow_overlaps={
                        "call_overlap": round(call_overlap, 3),
                        "result_key_overlap": round(result_key_overlap, 3),
                        "exit_sim": round(exit_sim, 3),
                        "cross_anchor_count": cross_anchor_count,
                        "filter_bonus": round(_FILTER_BONUS if _filter_matched else 0.0, 3),
                        "ranking_sim": round(ranking_score, 3),
                        "param_sim": round(_param_role_similarity(na, nb), 3),
                        "ident_overlap": round(ident_overlap, 3),
                    },
                )
                file_candidates.append(candidate)

        file_candidates.sort(key=lambda c: c.shadow_overlaps.get("ranking_sim", c.similarity), reverse=True)
        all_candidates.extend(file_candidates[:max(5, max_candidates // max(1, len(file_paths)))])

    all_candidates.sort(key=lambda c: c.similarity, reverse=True)
    result = all_candidates[:max_candidates]

    # ── Forced pairs: compute on-demand for user-explicit intent symbols ──────
    # Always included in result even if below min_similarity so DPB can
    # distinguish "not checked" from "checked and low".
    if forced_pairs:
        # Reuse the per-file normalised symbols built in the main loop.
        _fp_normed: dict[str, dict[str, NormalisedSymbol]] = _normed_cache

        _result_pairs = {
            frozenset([c.symbol_a.split(".")[-1], c.symbol_b.split(".")[-1]])
            for c in result
        }

        for bare_a, bare_b in forced_pairs:
            _pair_key = frozenset([bare_a, bare_b])
            if _pair_key in _result_pairs:
                # already in top-N result — mark forced, clear forced_reason (resolved)
                for c in result:
                    if {c.symbol_a.split(".")[-1], c.symbol_b.split(".")[-1]} == _pair_key:
                        c.forced = True
                        c.forced_reason = ""
                continue

            # Not in top-N — compute similarity on-demand.
            # Always emit a candidate so DPB can distinguish "not checked" from
            # "checked and classified".  forced_reason carries the classification.
            _first_file = file_paths[0] if file_paths else ""
            _fc: Optional[SimilarityCandidate] = None

            for fpath, nm in _fp_normed.items():
                na = nm.get(bare_a)
                nb = nm.get(bare_b)
                if na is None or nb is None:
                    continue
                try:
                    score = compute_similarity(na, nb)
                except Exception:
                    # normalise_failed: symbols found but similarity computation failed
                    _fc = SimilarityCandidate(
                        file=fpath,
                        symbol_a=bare_a,
                        symbol_b=bare_b,
                        similarity=0.0,
                        duplication_kind="forced_pair",
                        suggested_action="forced_pair_unresolved",
                        forced=True,
                        forced_reason="normalise_failed",
                    )
                    break
                if _fc is None:
                    _is_low = score < min_similarity
                    if _is_low:
                        _kind = "forced_pair"
                        action = "too_dissimilar"
                    else:
                        _kind = _classify_duplication_kind(na, nb, score)
                        action, _ = _suggest_action(na, nb, _kind, score)
                    kind = "forced_pair" if _is_low else _kind
                    _cross_count = 0 if _is_low else len(set(na.anchor_fps) & set(nb.anchor_fps))
                    extractable = False if _is_low else _is_extractable(na, nb, action, score, _cross_count)
                    if not extractable and action == "extract_shared_helper":
                        action = "similar_structure_only"

                    call_overlap = 0.0 if _is_low else _set_similarity(na.call_shapes, nb.call_shapes)
                    result_key_overlap = 0.0 if _is_low else _set_similarity(na.result_keys, nb.result_keys)
                    exit_sim = 1.0 if _is_low else _set_similarity(na.exit_shapes, nb.exit_shapes)
                    logger.debug(
                        "[AST_SCAN_SHADOW] forced pair=(%s, %s) sim=%.3f action=%s "
                        "call_overlap=%.3f result_key_overlap=%.3f cross_anchors=%d exit_sim=%.3f",
                        na.qualname, nb.qualname, score, action,
                        call_overlap, result_key_overlap, _cross_count, exit_sim,
                    )

                    _fc = SimilarityCandidate(
                        file=fpath,
                        symbol_a=na.qualname,
                        symbol_b=nb.qualname,
                        similarity=score,
                        duplication_kind=kind,
                        suggested_action=action,
                        extractable=extractable,
                        forced=True,
                        forced_reason="scanner_limit_excluded" if _is_low else "",
                        shadow_overlaps={
                            "call_overlap": round(call_overlap, 3),
                            "result_key_overlap": round(result_key_overlap, 3),
                            "exit_sim": round(exit_sim, 3),
                            "cross_anchor_count": _cross_count,
                            "ident_overlap": round(
                                _set_similarity(na.ident_tokens, nb.ident_tokens), 3),
                        },
                    )
                break

            if _fc is None:
                # Cross-file fallback: symbols may live in different files.
                # Find each independently across all scanned files.
                _na_cross: Optional[NormalisedSymbol] = None
                _nb_cross: Optional[NormalisedSymbol] = None
                _file_a: str = _first_file
                _file_b: str = _first_file
                for fpath, nm in _fp_normed.items():
                    if _na_cross is None and bare_a in nm:
                        _na_cross = nm[bare_a]
                        _file_a = fpath
                    if _nb_cross is None and bare_b in nm:
                        _nb_cross = nm[bare_b]
                        _file_b = fpath
                    if _na_cross is not None and _nb_cross is not None:
                        break

                if _na_cross is not None and _nb_cross is not None:
                    try:
                        score = compute_similarity(_na_cross, _nb_cross)
                        _is_low = score < min_similarity
                        if _is_low:
                            _kind = "forced_pair"
                            action = "too_dissimilar"
                        else:
                            _kind = _classify_duplication_kind(_na_cross, _nb_cross, score)
                            action, _ = _suggest_action(_na_cross, _nb_cross, _kind, score)
                        kind = "forced_pair" if _is_low else _kind
                        # cross-file pair: extractable only if same-file merge is feasible
                        extractable = False
                        if action == "extract_shared_helper":
                            action = "similar_structure_only"
                        _fc = SimilarityCandidate(
                            file=_file_a,
                            symbol_a=_na_cross.qualname,
                            symbol_b=_nb_cross.qualname,
                            similarity=score,
                            duplication_kind=kind,
                            suggested_action=action,
                            extractable=extractable,
                            forced=True,
                            forced_reason="cross_file",
                            shadow_overlaps={
                                "cross_file_b": _file_b,
                            },
                        )
                    except Exception:
                        _fc = SimilarityCandidate(
                            file=_file_a,
                            symbol_a=bare_a,
                            symbol_b=bare_b,
                            similarity=0.0,
                            duplication_kind="forced_pair",
                            suggested_action="forced_pair_unresolved",
                            forced=True,
                            forced_reason="normalise_failed",
                        )

            if _fc is None:
                # symbol_missing: one or both symbols not found in any scanned file
                _fc = SimilarityCandidate(
                    file=_first_file,
                    symbol_a=bare_a,
                    symbol_b=bare_b,
                    similarity=0.0,
                    duplication_kind="forced_pair",
                    suggested_action="forced_pair_unresolved",
                    forced=True,
                    forced_reason="symbol_missing",
                )

            result.append(_fc)
            _result_pairs.add(_pair_key)
            logger.info(
                "[AST_SCAN] forced pair (%s, %s) sim=%.3f action=%s reason=%s",
                bare_a, bare_b, _fc.similarity, _fc.suggested_action, _fc.forced_reason,
            )

    if result:
        logger.info(
            "[AST_SCAN] %d candidate pair(s) across %d file(s) "
            "(top=%.3f, extractable=%d)",
            len(result), len(file_paths), result[0].similarity,
            sum(1 for c in result if c.extractable),
        )
    else:
        logger.debug("[AST_SCAN] no candidates (files=%d, min_sim=%.2f)", len(file_paths), min_similarity)

    return result


# ── Activation gate ───────────────────────────────────────────────────────────
