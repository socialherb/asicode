"""Contradictory / unreachable / redundant logic scanner.

AST-based deterministic analysis that finds common logic anti-patterns:
  1. **Constant conditions**: ``if True:``, ``if False:``, ``while False:``, ``if 0:``, etc.
  2. **Contradictory boolean expressions**: ``if x and not x:``
  3. **Duplicate conditions**: same condition checked twice in the same scope
  4. **Always-false assertions**: ``assert False``

Each candidate carries a confidence score:
  - 1.0: AST-proven (constant literal condition, always-false assert)
  - 0.9: heuristic (contradictory boolean with same Name reference)
  - 0.8: heuristic (duplicate condition, same structural form)

Duplicate detection uses three-layer logic:
  1. **Same chain** (same if/elif group): always a duplicate regardless of distance.
  2. **Cross-chain, no mutation barrier**: flag if no relevant name mutation exists
     between the two occurrences AND they are within max_dup_distance lines.
  3. **Cross-chain, mutation barrier present**: skip — the variable may have changed.

A *mutation barrier* exists when, between two identical conditions, the
statements contain a direct assignment (Assign/AugAssign/AnnAssign) to any name
referenced by the condition, or — for attribute-referencing conditions — any
function call (which may mutate object state).
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg

from . import parse_cache

logger = logging.getLogger(__name__)

# ast.TryStar (except*) exists from Python 3.11
_TRY_TYPES = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)


@dataclass
class ContradictoryCandidate:
    """One instance of contradictory / unreachable / redundant logic."""
    file: str
    symbol: str               # containing function/method/class name
    contradiction_kind: str   # one of the "constant_*"/"contradictory_*"/etc. kind strings
    lineno: int
    end_lineno: int
    detail: str               # human-readable description
    confidence: float         # 0.0–1.0
    node_kind: str = ""       # AST node type at lineno: "If" | "While" | "Assert"
    condition_dump: str = ""  # ast.dump(condition, annotate_fields=False) for If/While

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "symbol": self.symbol,
            "contradiction_kind": self.contradiction_kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "detail": self.detail,
            "confidence": round(self.confidence, 3),
            "node_kind": self.node_kind,
            "condition_dump": self.condition_dump,
        }


# ── Expression-level helpers ─────────────────────────────────────────────────


def _get_enclosing_symbol_name(tree: ast.Module, lineno: int) -> str:
    """Find the name of the function/class/method enclosing *lineno*."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", 0)
            if start <= lineno <= end:
                return node.name
    return "<module>"


def _is_constant_false(node: ast.expr) -> Optional[bool]:
    """True if definitely falsy constant, False if definitely truthy, None if unknown."""
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return not val
        if isinstance(val, (int, float)):
            return val == 0
        if isinstance(val, (str, bytes)) and len(val) == 0:
            return True
    return None


def _falsy_constant_kind(cond: ast.expr) -> str:
    """Classify a definitely-falsy condition into a contradiction-kind string."""
    if isinstance(cond, ast.Constant):
        val = cond.value
        if isinstance(val, bool):
            return "constant_false_condition"
        if isinstance(val, (int, float)):
            return "constant_zero_condition"
        if isinstance(val, (str, bytes)):
            return "constant_empty_condition"
    return "constant_false_condition"


def _collect_pos_neg_names(node: ast.BoolOp) -> tuple[set[str], set[str]]:
    """Collect positive (x) and negative (not x) name sets from BoolOp values."""
    pos_names: set[str] = set()
    neg_names: set[str] = set()
    for val in node.values:
        if isinstance(val, ast.Name) and isinstance(val.ctx, ast.Load):
            pos_names.add(val.id)
        elif isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.Not):
            if isinstance(val.operand, ast.Name):
                neg_names.add(val.operand.id)
    return pos_names, neg_names


def _check_boolop_tautology(node: ast.BoolOp) -> list[tuple[str, str]]:
    """Detect contradictory (``x and not x``) or redundant (``x or not x``) patterns.

    Returns a list of (kind, message) tuples. Empty list if no pattern detected.
    
    For And: detects ``x and not x`` (contradictory, always False).
    For Or: detects ``x or not x`` (redundant, always True).
    
    Requires the same name in both positive (``x``) and negative (``not x``)
    form as *direct children* of the BoolOp.  ``not x and y > 0`` is NOT flagged
    because ``x`` only appears in negative form.
    """
    results = []
    pos_names, neg_names = _collect_pos_neg_names(node)
    overlap = pos_names & neg_names
    if overlap:
        name = next(iter(overlap))
        if isinstance(node.op, ast.And):
            results.append(("contradictory_boolean", f"contradictory: '{name} and not {name}'"))
        elif isinstance(node.op, ast.Or):
            results.append(("always_true_boolean", f"redundant: '{name} or not {name}' is always True"))
    return results


def _check_condition(cond: ast.expr) -> list:
    """Return list of (kind, detail, confidence) for single-expression anti-patterns."""
    results: list = []
    cf = _is_constant_false(cond)
    if cf is True:
        results.append((_falsy_constant_kind(cond), "condition is always False", 1.0))
        return results
    if cf is False:
        results.append(("constant_true_condition", "condition is always True", 1.0))
        return results
    if isinstance(cond, ast.BoolOp):
        for kind, msg in _check_boolop_tautology(cond):
            results.append((kind, msg, 0.9))
    return results


# ── Mutation barrier helpers ─────────────────────────────────────────────────


def _extract_condition_names(cond: ast.expr) -> tuple[set[str], bool]:
    """Names referenced in *cond* and whether any are inside Attribute access.

    Returns:
        (simple_names, has_attr_access) where:
        - simple_names: bare Name.id values found anywhere in the condition
        - has_attr_access: True if the condition contains any Attribute node
          (meaning object state could be mutated by a method call)
    """
    simple_names: set[str] = set()
    has_attr_access = False
    for node in ast.walk(cond):
        if isinstance(node, ast.Name):
            simple_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            has_attr_access = True
    return simple_names, has_attr_access


def _assignment_target_overlaps(target: ast.expr, names: set[str]) -> bool:
    """True if an assignment target directly assigns to any name in *names*."""
    if isinstance(target, ast.Name):
        return target.id in names
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_assignment_target_overlaps(e, names) for e in target.elts)
    if isinstance(target, ast.Starred):
        return _assignment_target_overlaps(target.value, names)
    # Attribute/subscript writes don't directly rebind the bare name
    return False


# Known-pure call names — these cannot mutate object state, even when the
# condition depends on attribute access.  Used to reduce false mutation
# barriers in duplicate-condition detection.
_KNOWN_PURE_CALL_NAMES: frozenset = frozenset({
    # Logging — pure side-effect (no state mutation)
    "debug", "info", "warning", "warn", "error", "critical", "exception", "log",
    # Container read-only — getters do not mutate
    "get", "keys", "values", "items", "copy",
    # Pure builtins — no mutation of any object
    "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set", "frozenset",
    "sorted", "reversed", "enumerate", "zip", "range", "iter", "next",
    "isinstance", "issubclass", "hasattr", "getattr", "callable", "type", "id",
    "repr", "format", "print", "ascii",
    # String methods — strings are immutable
    "strip", "split", "join", "replace", "lower", "upper", "lstrip", "rstrip",
    "startswith", "endswith", "find", "index", "count",
})


def _is_known_pure_call(node: ast.Call) -> bool:
    """Return True if *node* is a call to a known-pure (non-mutating) target.

    Supports bare names (``isinstance(...)``) and attribute access
    (``self.logger.info(...)``, ``obj.get(...)``).
    """
    if isinstance(node.func, ast.Name):
        return node.func.id in _KNOWN_PURE_CALL_NAMES
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in _KNOWN_PURE_CALL_NAMES
    return False


def _has_name_mutation(
    stmts: list[ast.stmt],
    names: set[str],
    has_attr_access: bool,
) -> bool:
    """Return True if *stmts* contain a mutation barrier for *names*.

    Barriers:
    - Direct assignment (Assign / AugAssign / AnnAssign) to any name in *names*
      (checked recursively inside nested blocks).
    - Any NON-pure function Call when the condition uses attribute access —
      method calls may mutate object state that the condition depends on.
      Known-pure calls (loggers, getters, builtins) are NOT barriers.
    - Method call directly on any condition name, e.g. ``_allowed.add(x)`` —
      this is an in-place mutation even when the name is not rebound.
    """
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                if any(_assignment_target_overlaps(t, names) for t in node.targets):
                    return True
            elif isinstance(node, ast.AugAssign):
                if _assignment_target_overlaps(node.target, names):
                    return True
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name) and node.target.id in names:
                    return True
            elif isinstance(node, ast.NamedExpr):
                # walrus: `if (x := f()): ...` rebinds x in-place
                if isinstance(node.target, ast.Name) and node.target.id in names:
                    return True
            elif isinstance(node, ast.Call):
                if has_attr_access:
                    # When the condition has attribute access (e.g. self.config.mode),
                    # most calls could potentially mutate object state.  Skip only
                    # known-pure calls (getters, loggers, builtins).
                    if _is_known_pure_call(node):
                        continue
                    return True
                # Method call on a condition name (e.g. _allowed.add()) is an
                # in-place mutation even when has_attr_access is False.
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in names:
                        return True
    return False


# ── Body overlap helpers ─────────────────────────────────────────────────────


def _body_has_early_exit(body: list) -> bool:
    """Return True if any statement in *body* is a Return or Raise."""
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.Return, ast.Raise)):
                return True
    return False


def _bodies_have_overlapping_writes(body1: list, body2: list) -> bool:
    """Return True if two if-bodies write to any overlapping assignment targets.

    Two ``if cond:`` blocks that write to *completely disjoint* sets of names /
    attributes are independent guards for separate purposes — not a removable
    duplicate.  Return True (conservative) when either body has no write
    statements (e.g. only ``return`` / ``raise`` / ``pass``).
    """
    def _write_targets(stmts: list) -> set:
        targets: set = set()
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        targets.add(ast.dump(t, annotate_fields=False)[:60])
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets.add(ast.dump(node.target, annotate_fields=False)[:60])
        return targets

    t1 = _write_targets(body1)
    t2 = _write_targets(body2)
    if not t1 or not t2:
        return True  # conservative: trivial body (return/raise/pass only)
    return bool(t1 & t2)


# Names that appear in virtually every try/except block and carry no semantic
# meaning for distinguishing independent guard blocks from redundant ones.
_TRIVIAL_CALL_NAMES: frozenset = frozenset({
    "debug", "info", "warning", "warn", "error", "critical", "exception", "log",
    "append", "add", "get", "set", "update", "pop", "remove", "clear",
    "isinstance", "len", "str", "int", "float", "bool", "list", "dict",
    "tuple", "sorted", "range", "print", "repr", "format",
    "getattr", "setattr", "hasattr", "type",
})


def _bodies_have_overlapping_calls(body1: list, body2: list) -> bool:
    """Return True if the two if-bodies share any non-trivial function calls.

    Used as a secondary check when write-target overlap is inconclusive (both
    bodies consist entirely of side-effect calls with no assignments).  Bodies
    that call disjoint non-trivial functions are independent sequential guards
    — not removable duplicates.
    """
    def _nontrivial_calls(stmts: list) -> set:
        names: set = set()
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        n = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        n = node.func.attr
                    else:
                        continue
                    if n not in _TRIVIAL_CALL_NAMES:
                        names.add(n)
        return names

    c1 = _nontrivial_calls(body1)
    c2 = _nontrivial_calls(body2)
    if not c1 or not c2:
        return True  # conservative: can't determine from call names alone
    return bool(c1 & c2)


# ── if/elif chain flattener ──────────────────────────────────────────────────


def _branch_body_end(lineno: int, branch_body: list) -> int:
    """Return the last line of *branch_body*, falling back to *lineno* if empty."""
    if not branch_body:
        return lineno
    return max(
        getattr(stmt, "end_lineno", lineno) for stmt in branch_body
    )


def _collect_if_elif_chain(if_node: ast.If) -> tuple[list, list]:
    """Flatten an if/elif/else into (chain, else_body).

    chain: list of (condition, lineno, branch_end, body_stmts) for each branch.
      - lineno    : line of the if/elif keyword
      - branch_end: last line of the LOCAL branch body (excluding orelse/else)
    else_body: stmts in the final else clause, or [] if no else.
    """
    chain: list = []
    node: ast.If = if_node
    while True:
        body = list(node.body or [])
        chain.append((node.test, node.lineno, _branch_body_end(node.lineno, body), body))
        orelse = list(node.orelse or [])
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            node = orelse[0]
        else:
            return chain, orelse


# ── Core walker ──────────────────────────────────────────────────────────────

# scope_conditions entry: (dump_str, lineno, branch_end, chain_id, flat_idx, cond_node, body, node_kind)
# lineno     = line of the if/elif/while keyword
# branch_end = last line of the LOCAL branch body (excl. orelse/else)
# chain_id   = lineno of the top-level If/While node (unique per file)
# flat_idx   = index of the statement in the current flat body list
# body       = branch body statements (for overlap check)
# node_kind  = "If" | "While" — the AST node this condition came from
_SC = tuple[str, int, int, int, int, ast.expr, list, str]


def _check_body_stmts(
    body: list,
    enclosing_symbol: str,
    rel_path: str,
    max_dup_distance: int = _cfg.counts.SCANNER_CONTRADICTORY_DUP_DISTANCE,
) -> list:
    """Walk *body* statements and collect anti-pattern candidates.

    Scope invariant: ``scope_conditions`` accumulates only the if/elif/while
    conditions at *this* flat scope level.  Each branch body is recursed in a
    separate call so its conditions never pollute the parent scope's duplicate
    detector.

    Duplicate detection (three layers):
    1. Same chain_id → always duplicate.
    2. Cross-chain, no mutation barrier → duplicate if within max_dup_distance.
    3. Cross-chain, mutation barrier present → not a duplicate.
    """
    candidates: list = []
    scope_conditions: list[_SC] = []
    flat_stmts: list = list(body)   # for mutation barrier slicing

    for flat_idx, node in enumerate(flat_stmts):

        # Nested function/class: new scope, new enclosing_symbol
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.extend(
                _check_body_stmts(
                    list(node.body or []),
                    node.name, rel_path, max_dup_distance,
                )
            )

        # if/elif/else — process entire chain as one unit
        if isinstance(node, ast.If):
            chain_id = node.lineno   # unique per if/elif group in this file
            chain, else_body = _collect_if_elif_chain(node)
            for cond, lineno, branch_end, branch_body in chain:
                _cdump = ast.dump(cond, annotate_fields=False)
                for kind, detail, conf in _check_condition(cond):
                    candidates.append(ContradictoryCandidate(
                        file=rel_path,
                        symbol=enclosing_symbol,
                        contradiction_kind=kind,
                        lineno=lineno,
                        end_lineno=branch_end,
                        detail=detail,
                        confidence=conf,
                        node_kind="If",
                        condition_dump=_cdump,
                    ))
                scope_conditions.append((
                    ast.dump(cond, annotate_fields=False),
                    lineno, branch_end, chain_id, flat_idx, cond, branch_body, "If",
                ))
                candidates.extend(
                    _check_body_stmts(branch_body, enclosing_symbol, rel_path, max_dup_distance)
                )
            if else_body:
                candidates.extend(
                    _check_body_stmts(else_body, enclosing_symbol, rel_path, max_dup_distance)
                )

        # while — same flat_idx/chain_id concept, no elif
        elif isinstance(node, ast.While):
            cond = node.test
            chain_id = node.lineno
            while_end = getattr(node, "end_lineno", node.lineno)
            # while True: / while (x or not x): are idiomatic infinite-loop patterns
            # controlled by break/return — never flag as deletable.
            _WHILE_SKIP_KINDS = frozenset({"constant_true_condition", "always_true_boolean"})
            for kind, detail, conf in _check_condition(cond):
                if kind in _WHILE_SKIP_KINDS:
                    continue
                candidates.append(ContradictoryCandidate(
                    file=rel_path,
                    symbol=enclosing_symbol,
                    contradiction_kind=kind,
                    lineno=node.lineno,
                    end_lineno=while_end,
                    detail=detail,
                    confidence=conf,
                    node_kind="While",
                    condition_dump=ast.dump(cond, annotate_fields=False),
                ))
            scope_conditions.append((
                ast.dump(cond, annotate_fields=False),
                node.lineno, while_end, chain_id, flat_idx, cond, list(node.body or []), "While",
            ))
            candidates.extend(
                _check_body_stmts(list(node.body or []), enclosing_symbol, rel_path, max_dup_distance)
            )

        # assert
        elif isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant) and node.test.value is False:
                candidates.append(ContradictoryCandidate(
                    file=rel_path,
                    symbol=enclosing_symbol,
                    contradiction_kind="always_false_assert",
                    lineno=node.lineno,
                    end_lineno=node.lineno,
                    detail="'assert False' always fails",
                    confidence=1.0,
                    node_kind="Assert",
                    condition_dump=ast.dump(node.test, annotate_fields=False),
                ))

        # try/except (incl. 3.11+ except*) — each clause is its own sub-scope
        elif isinstance(node, _TRY_TYPES):
            candidates.extend(
                _check_body_stmts(list(node.body or []), enclosing_symbol, rel_path, max_dup_distance)
            )
            for handler in (node.handlers or []):
                candidates.extend(
                    _check_body_stmts(
                        list(handler.body or []), enclosing_symbol, rel_path, max_dup_distance,
                    )
                )
            candidates.extend(
                _check_body_stmts(list(node.orelse or []), enclosing_symbol, rel_path, max_dup_distance)
            )
            candidates.extend(
                _check_body_stmts(
                    list(getattr(node, "finalbody", None) or []),
                    enclosing_symbol, rel_path, max_dup_distance,
                )
            )

        # for / with — loop and context bodies are their own sub-scopes
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            candidates.extend(
                _check_body_stmts(list(node.body or []), enclosing_symbol, rel_path, max_dup_distance)
            )
            candidates.extend(
                _check_body_stmts(list(node.orelse or []), enclosing_symbol, rel_path, max_dup_distance)
            )
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            candidates.extend(
                _check_body_stmts(list(node.body or []), enclosing_symbol, rel_path, max_dup_distance)
            )

        # match — each case body is its own sub-scope
        elif isinstance(node, ast.Match):
            for case in (node.cases or []):
                candidates.extend(
                    _check_body_stmts(list(case.body or []), enclosing_symbol, rel_path, max_dup_distance)
                )

    # ── Duplicate detection ──────────────────────────────────────────────────
    # seen[dump_str] = (prev_lineno, prev_chain_id, prev_flat_idx, prev_body)
    # Always advances to the nearest occurrence so distance is to the closest prior.
    seen: dict = {}
    for dump_str, lineno, branch_end, chain_id, flat_idx, cond_node, body, node_kind in scope_conditions:
        if dump_str in seen:
            prev_lineno, prev_chain_id, prev_flat_idx, prev_body = seen[dump_str]
            is_dup = False

            if prev_chain_id == chain_id:
                # Same if/elif chain: condition appears in two branches → always dup
                # (the elif branch is unreachable regardless of any body mutations)
                is_dup = True
            else:
                # Cross-chain: check mutation barrier first
                intervening = flat_stmts[prev_flat_idx + 1 : flat_idx]
                cond_names, has_attr = _extract_condition_names(cond_node)
                if not _has_name_mutation(intervening, cond_names, has_attr):
                    # Adjacency gate: only directly consecutive `if x: A` /
                    # `if x: B` statements are merge candidates.  A repeated
                    # guard further apart is almost always an intentional
                    # phase boundary in a long function — sample verification
                    # (2026-06-12, 8 candidates at 12-46 lines apart) found
                    # 0 true positives under the distance-based gate.
                    if flat_idx == prev_flat_idx + 1:
                        is_dup = True

                if is_dup:
                    cond_names_chk, has_attr_chk = _extract_condition_names(cond_node)
                    # Check 1: if the first block's own body mutates the condition
                    # variable, the second check is testing the post-mutation value
                    # (e.g. fallback assignment then re-check) — not a dup.
                    if _has_name_mutation(prev_body, cond_names_chk, has_attr_chk):
                        is_dup = False
                    # Check 2: if the two bodies write to completely disjoint targets
                    # they serve independent purposes — not a removable dup.
                    elif not _bodies_have_overlapping_writes(prev_body, body):
                        is_dup = False
                    # Check 3: when write-target overlap is inconclusive (both bodies
                    # are side-effect-only), fall back to non-trivial call overlap.
                    # Bodies calling disjoint non-trivial functions are independent
                    # sequential guards under the same flag — not a removable dup.
                    elif not _bodies_have_overlapping_calls(prev_body, body):
                        is_dup = False
                    # Check 4: asymmetric exit profile — exactly one body ends with
                    # return/raise while the other does not.  This signals "side-effect
                    # accumulation" (first block) vs "early-exit decision" (second block)
                    # — two structurally distinct roles that must not be merged.
                    elif _body_has_early_exit(prev_body) != _body_has_early_exit(body):
                        is_dup = False

            if is_dup:
                # Same-chain dup (elif): always unreachable — flag as duplicate.
                # Cross-chain dup near the first occurrence: merge candidate,
                # not a delete-simple duplicate (bodies may differ in purpose).
                _is_same_chain = (prev_chain_id == chain_id)
                _distance = lineno - prev_lineno
                if not _is_same_chain and _distance <= max_dup_distance:
                    _kind = "mergeable_condition"
                    _conf = 0.85
                    _detail = (
                        f"same condition as line {prev_lineno} "
                        f"({_distance} lines apart) — merge candidate"
                    )
                else:
                    _kind = "duplicate_condition"
                    _conf = 0.8
                    _detail = f"duplicate condition previously checked at line {prev_lineno}"
                candidates.append(ContradictoryCandidate(
                    file=rel_path,
                    symbol=enclosing_symbol,
                    contradiction_kind=_kind,
                    lineno=lineno,
                    end_lineno=branch_end,
                    detail=_detail,
                    confidence=_conf,
                    node_kind=node_kind,
                    condition_dump=dump_str,
                ))

        # Always track the nearest prior occurrence for future distance checks
        seen[dump_str] = (lineno, chain_id, flat_idx, body)

    return candidates


def scan_contradictory_logic(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_CONTRADICTORY_MAX,
    max_dup_distance: int = _cfg.counts.SCANNER_CONTRADICTORY_DUP_DISTANCE,
) -> list[ContradictoryCandidate]:
    """Scan files for contradictory, unreachable, or redundant logic.

    Args:
        repo_root: Repository root path.
        file_paths: File paths to scan.
        max_per_file: Max candidates to emit per file.
        max_dup_distance: Secondary gate for cross-chain duplicates with no
            detected mutation barrier.  Same-chain duplicates ignore this limit.

    Returns:
        List of ``ContradictoryCandidate``.
    """
    candidates: list[ContradictoryCandidate] = []
    _truncated_total = 0  # candidates dropped by max_per_file

    for rel_path in file_paths or []:
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(repo_root or "", rel_path)
        tree = parse_cache.parse_ast(abs_path)
        if tree is None:
            continue

        emitted = 0
        file_candidates = _check_body_stmts(
            list(tree.body), "<module>", rel_path, max_dup_distance,
        )
        for c in file_candidates:
            candidates.append(c)
            emitted += 1
            if emitted >= max_per_file:
                _truncated_total += len(file_candidates) - emitted
                logger.warning(
                    "[CONTRADICTORY] %s: hit max_per_file=%d, truncating %d remaining",
                    rel_path, max_per_file, len(file_candidates) - emitted,
                )
                break

    if candidates:
        kinds = {}
        for c in candidates:
            kinds[c.contradiction_kind] = kinds.get(c.contradiction_kind, 0) + 1
        kind_summary = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        logger.info(
            "[CONTRADICTORY] %d candidate(s) across %d file(s): %s",
            len(candidates), len({c.file for c in candidates}),
            kind_summary,
        )

    if _truncated_total:
        # Function attribute consumed by ScannerRegistry.run() (reset via
        # `del` before each invocation).
        scan_contradictory_logic._truncated = _truncated_total
    return candidates
