"""guard_ir — GuardIR: single source of truth for guard statement semantics.

Step 1: GuardIR dataclass + parse_guard() factory (read-only IR).
Step 2: GuardPlacement + GuardFeasibility dataclasses + analyze_guard().
  - All pure-AST helpers (_classify_enclosing_function_flavor,
    _extract_guard_local_anchors, _extract_for_loop_target_names,
    _normalize_guard_for_contract, _extract_guard_control_flow)
    are implemented here as authoritative copies.  planner_agent.py
    delegates to these via thin wrappers so existing call sites are
    unchanged.
  - _infer_guard_placement_contract stays in planner_agent.py for now
    (it imports from .placement_contract; moved in Step 3+).

Circular-import safety: imports only stdlib (ast, dataclasses, logging).
"""
from __future__ import annotations

import ast
import dataclasses
import logging
import re
from collections.abc import Collection
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared builtin-name sets
# ---------------------------------------------------------------------------

# Union of _BUILTINS (precheck) and _PCL_BUILTIN_NAMES (local-anchor filter).
# Using the union is conservative: more names are excluded from "local state"
# analysis → fewer spurious NameError / anchor-ordering concerns.
_GUARD_BUILTIN_NAMES: frozenset[str] = frozenset({
    "True", "False", "None",
    "len", "type", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "callable", "iter", "next", "print", "repr", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "range", "enumerate", "zip", "map", "filter",
    "any", "all", "min", "max", "sum", "abs", "round", "sorted", "reversed",
    "open", "vars", "dir", "id", "hash", "hex", "oct", "bin", "chr", "ord",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "AssertionError", "OSError", "IOError",
})

# Python keyword tokens excluded from IR operand lists.
_PY_KW: frozenset[str] = frozenset(
    {"not", "in", "is", "and", "or", "True", "False", "None"}
)

# Compound statement AST types used by _has_compound_first_stmt.
_COMPOUND_STMT_TYPES: tuple = (
    ast.Try, ast.With, ast.For, ast.While, ast.If,
    ast.AsyncFor, ast.AsyncWith,
)
if hasattr(ast, "TryStar"):
    _COMPOUND_STMT_TYPES = _COMPOUND_STMT_TYPES + (ast.TryStar,)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GuardCondition:
    """AST-derived semantic model for the if-condition in a guard statement."""

    op_class: str
    operands: list[str]
    attribute_pairs: list[tuple[str, str]]

    def to_legacy_dict(self) -> dict:
        d: dict = {"op": self.op_class, "operands": self.operands}
        if self.attribute_pairs:
            d["attribute_pairs"] = self.attribute_pairs
        return d


@dataclasses.dataclass
class GuardPlacement:
    """Structural placement analysis for a guard (filled by analyze_guard)."""

    anchors: list[str]
    """Local-var anchor names the guard depends on (after-assignment ordering)."""

    had_unresolved: bool
    """True if the guard referenced non-param, non-builtin names before
    hallucination filtering.  ([], True) means all references were hallucinated."""

    hallucinated_bases: frozenset[str]
    """Base names confirmed absent from function scope (e.g. "error" in SL28)."""

    host_function_flavor: str
    """"async" | "generator" | "plain" | "unknown"."""

    loop_candidates: list[str]
    """Sorted for-loop target names in the host function (candidates for
    inside_block placement when guard uses break/continue)."""


@dataclasses.dataclass
class GuardFeasibility:
    """AST-op eligibility decision (filled by analyze_guard)."""

    ast_op_safe: bool
    """True → executor can use the deterministic ASTOpExecutor path."""

    reason_code: str
    """Why this decision was reached (e.g. "parameter_validation_raise")."""

    insert_scope: str
    """"function_body" | "for_loop" | "while_loop" | "" (unknown)."""

    loop_variable: str
    """For-loop variable matched to the guard's anchor (for_loop scope)."""

    requires_llm: bool
    """True → executor must use the LLM code-generation path."""


@dataclasses.dataclass
class GuardIR:
    """Canonical IR for a single guard statement (``if <cond>: <control>``)."""

    raw: str
    canonical: str
    """ast.unparse-stable form.  Empty string when parsing fails."""

    compact: str
    """Single-line collapsed form suitable for LLM prompts / op.metadata."""

    condition: Optional[GuardCondition]
    control: str
    """"continue" | "break" | "return" | "raise" | ""."""

    placement: Optional[GuardPlacement] = None
    """Filled by analyze_guard(); None until then."""

    feasibility: Optional[GuardFeasibility] = None
    """Filled by analyze_guard(); None until then."""

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def to_legacy_tuple(self) -> tuple[Optional[dict], Optional[str]]:
        """(condition_dict, control) compatible with _extract_guard_ir output."""
        if self.condition is None:
            return None, None
        return self.condition.to_legacy_dict(), self.control or None

    @property
    def is_parsed(self) -> bool:
        return self.canonical != ""

    @property
    def is_template_placeholder(self) -> bool:
        """True when this guard is an IntentResolver template placeholder.

        A bare-Name condition (e.g. "if condition: continue") without
        code-validated placement is a placeholder, not a real hypothesis.
        """
        if self.condition is None:
            return False
        if self.condition.op_class != "Name":
            return False
        if self.placement is not None:
            return False
        return True


# ---------------------------------------------------------------------------
# Internal: condition extraction helpers (Step 1)
# ---------------------------------------------------------------------------

def _compute_op_class(expr: ast.expr) -> str:
    if isinstance(expr, ast.UnaryOp):
        return type(expr.op).__name__
    if isinstance(expr, ast.BoolOp):
        return type(expr.op).__name__
    if isinstance(expr, ast.Compare) and expr.ops:
        return type(expr.ops[0]).__name__
    return type(expr).__name__


def _extract_control(stmt: ast.If) -> str:
    for _n in ast.walk(stmt):
        if isinstance(_n, ast.Continue):
            return "continue"
        if isinstance(_n, ast.Break):
            return "break"
        if isinstance(_n, ast.Return):
            return "return"
        if isinstance(_n, ast.Raise):
            return "raise"
    return ""


def _extract_condition(stmt: ast.If) -> GuardCondition:
    op_class = _compute_op_class(stmt.test)
    operands: list = []
    seen: set = set()
    attribute_pairs: list = []
    seen_pairs: set = set()
    for _n in ast.walk(stmt.test):
        tok: Optional[str] = None
        if isinstance(_n, ast.Name) and _n.id not in _PY_KW:
            tok = _n.id
        elif isinstance(_n, ast.Attribute) and _n.attr not in _PY_KW:
            tok = _n.attr
            if isinstance(_n.value, ast.Name) and _n.value.id not in _PY_KW:
                _pair = (_n.value.id, _n.attr)
                if _pair not in seen_pairs:
                    attribute_pairs.append(_pair)
                    seen_pairs.add(_pair)
        if tok and tok not in seen:
            operands.append(tok)
            seen.add(tok)
    return GuardCondition(op_class=op_class, operands=operands,
                          attribute_pairs=attribute_pairs)


def _make_compact(canonical: str) -> str:
    parts = [p.strip() for p in canonical.splitlines() if p.strip()]
    if not parts:
        return canonical
    if len(parts) == 2 and parts[0].endswith(":"):
        return parts[0] + " " + parts[1]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal: placement / feasibility helpers (Step 2)
# ---------------------------------------------------------------------------

def _classify_enclosing_function_flavor(file_content: str, symbol: str) -> str:
    """Return "async" | "generator" | "plain" | "unknown".

    Authoritative implementation; planner_agent.py delegates to this.
    """
    if not file_content or not symbol:
        return "unknown"
    try:
        _tree = ast.parse(file_content)
    except SyntaxError:
        return "unknown"
    _bare = symbol.split(".")[-1]
    _func = next(
        (n for n in ast.walk(_tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == _bare),
        None,
    )
    if _func is None:
        return "unknown"
    if isinstance(_func, ast.AsyncFunctionDef):
        return "async"

    def _contains_yield(node: ast.AST) -> bool:
        for _n in ast.iter_child_nodes(node):
            if isinstance(_n, (ast.Yield, ast.YieldFrom)):
                return True
            if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if _contains_yield(_n):
                return True
        return False

    return "generator" if _contains_yield(_func) else "plain"


def _extract_for_loop_target_names(file_content: str, symbol: str) -> list[str]:
    """Return sorted for-loop target names in the enclosing function.

    Authoritative implementation; planner_agent.py delegates to this.
    """
    if not file_content or not symbol:
        return []
    try:
        _tree = ast.parse(file_content)
    except SyntaxError:
        return []
    _bare = symbol.split(".")[-1]
    _func = next(
        (n for n in ast.walk(_tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == _bare),
        None,
    )
    if _func is None:
        return []
    _names: set = set()
    for _n in ast.walk(_func):
        if isinstance(_n, ast.For):
            for _t in ast.walk(_n.target):
                if isinstance(_t, ast.Name):
                    _names.add(_t.id)
    return sorted(_names)


def _extract_guard_control_flow(guard_tree: ast.AST) -> set[str]:
    """Return the set of control-flow primitives in a guard AST.

    Authoritative implementation; planner_agent.py delegates to this.
    """
    cf: set = set()
    for _n in ast.walk(guard_tree):
        if isinstance(_n, ast.Return):
            cf.add("return")
        elif isinstance(_n, ast.Raise):
            cf.add("raise")
        elif isinstance(_n, ast.Break):
            cf.add("break")
        elif isinstance(_n, ast.Continue):
            cf.add("continue")
        elif isinstance(_n, (ast.Yield, ast.YieldFrom)):
            cf.add("yield")
        elif isinstance(_n, ast.Await):
            cf.add("await")
    return cf


def _extract_guard_local_anchors(
    guard_statement: str,
    file_content: str,
    symbol: str,
) -> tuple[list[str], bool, frozenset[str]]:
    """Return (anchors, had_unresolved, hallucinated_bases).

    Authoritative implementation; planner_agent.py delegates to this.
    Uses _GUARD_BUILTIN_NAMES (superset of former _PCL_BUILTIN_NAMES) —
    conservative: more names treated as "safe", fewer spurious anchors.
    """
    if not guard_statement or not symbol:
        return [], False, frozenset()
    try:
        _gs_tree = ast.parse(guard_statement.strip(), mode="exec")
    except SyntaxError:
        return [], False, frozenset()

    _gs_names: set = {
        n.id for n in ast.walk(_gs_tree) if isinstance(n, ast.Name)
    }
    _param_names: set = set()
    _func_referenced: set = set()
    _func_assigned: set = set()
    _file_tree = None
    if file_content:
        try:
            _file_tree = ast.parse(file_content)
            _sym_bare = symbol.split(".")[-1]
            _func = next(
                (n for n in ast.walk(_file_tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == _sym_bare),
                None,
            )
            if _func:
                _args = _func.args
                _param_names = (
                    {a.arg for a in getattr(_args, "args", [])}
                    | {a.arg for a in getattr(_args, "posonlyargs", [])}
                    | {a.arg for a in getattr(_args, "kwonlyargs", [])}
                )
                if _args.vararg:
                    _param_names.add(_args.vararg.arg)
                if _args.kwarg:
                    _param_names.add(_args.kwarg.arg)
                for _ndw in ast.walk(_func):
                    if isinstance(_ndw, ast.Name):
                        _func_referenced.add(_ndw.id)
                    elif (isinstance(_ndw, ast.Attribute)
                            and isinstance(_ndw.value, ast.Name)):
                        _func_referenced.add(_ndw.value.id)
                for _ndw in ast.walk(_func):
                    if isinstance(_ndw, ast.Assign):
                        for _t in _ndw.targets:
                            for _tn in ast.walk(_t):
                                if isinstance(_tn, ast.Name):
                                    _func_assigned.add(_tn.id)
                    elif isinstance(_ndw, (ast.AnnAssign, ast.AugAssign)):
                        _t = _ndw.target
                        for _tn in ast.walk(_t):
                            if isinstance(_tn, ast.Name):
                                _func_assigned.add(_tn.id)
                    elif isinstance(_ndw, ast.NamedExpr):
                        _func_assigned.add(_ndw.target.id)
                    elif isinstance(_ndw, ast.For):
                        for _tn in ast.walk(_ndw.target):
                            if isinstance(_tn, ast.Name):
                                _func_assigned.add(_tn.id)
                    elif isinstance(_ndw, (ast.With, ast.AsyncWith)):
                        for _wi in getattr(_ndw, "items", []):
                            if getattr(_wi, "optional_vars", None):
                                for _tn in ast.walk(_wi.optional_vars):
                                    if isinstance(_tn, ast.Name):
                                        _func_assigned.add(_tn.id)
                    elif isinstance(_ndw, ast.ExceptHandler):
                        if getattr(_ndw, "name", None):
                            _func_assigned.add(_ndw.name)
        except (AttributeError, TypeError):
            pass

    _unresolved_set: set = _gs_names - _param_names - _GUARD_BUILTIN_NAMES
    _had_unresolved: bool = bool(_unresolved_set)
    _hallucinated_bases: frozenset[str] = frozenset()

    if _func_referenced:
        _hallucinated = _unresolved_set - _func_referenced
        if _hallucinated:
            _hallucinated_bases = frozenset(_hallucinated)
            logger.info(
                "[PCL] dropping %d hallucinated anchor name(s) not referenced "
                "anywhere in %s: %s",
                len(_hallucinated), symbol, sorted(_hallucinated),
            )
        _unresolved_set = _unresolved_set & _func_referenced

    _module_level_names: set = set()
    if _file_tree is not None:
        try:
            for _top in _file_tree.body:
                if isinstance(_top, (ast.Import, ast.ImportFrom)):
                    for _alias in getattr(_top, "names", []):
                        _n = getattr(_alias, "asname", None) or getattr(_alias, "name", "") or ""
                        if _n:
                            _module_level_names.add(_n.split(".")[0])
                elif isinstance(_top, ast.Assign):
                    for _t in _top.targets:
                        for _tn in ast.walk(_t):
                            if isinstance(_tn, ast.Name):
                                _module_level_names.add(_tn.id)
                elif isinstance(_top, ast.AnnAssign):
                    _t2 = _top.target
                    if isinstance(_t2, ast.Name):
                        _module_level_names.add(_t2.id)
                elif isinstance(_top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    _module_level_names.add(_top.name)
        except (AttributeError, TypeError):
            pass

    _module_globals = _unresolved_set & _module_level_names
    _module_globals -= _func_assigned
    if _module_globals:
        logger.debug(
            "[PCL] dropping %d module-level global anchor(s) from %s: %s",
            len(_module_globals), symbol, sorted(_module_globals),
        )
    _unresolved_set = _unresolved_set - _module_globals
    _unresolved: list = sorted(_unresolved_set)

    _attr_map: dict = {}
    for _nd in ast.walk(_gs_tree):
        if (isinstance(_nd, ast.Attribute)
                and isinstance(_nd.value, ast.Name)
                and _nd.value.id in _unresolved_set
                and _nd.value.id != "self"):
            _attr_map.setdefault(_nd.value.id, set()).add(_nd.attr)

    _anchors: list = []
    for _nm in _unresolved:
        if _nm in _attr_map:
            for _at in sorted(_attr_map[_nm]):
                _anchors.append(f"{_nm}.{_at}")
        else:
            _anchors.append(_nm)
    return _anchors, _had_unresolved, _hallucinated_bases


def _normalize_guard_for_contract(
    guard_statement: str,
    hallucinated_bases: Collection[str],
    effective_anchor_names: Collection[str],
) -> str:
    """Replace hallucinated_obj.attr → attr in a guard statement.

    When hallucinated_bases are used as attribute bases (obj.attr) in the
    guard, normalization would destroy the semantic relationship (SL28:
    ``if not error.name: continue`` → ``if not name: continue`` changes
    the meaning from "check error object's name attribute" to "check
    loop variable name").  The normalization is only valid for PCL
    placement matching, not for semantic transformation — if any
    hallucinated base appears as an attribute base, preserve the
    original guard.

    Authoritative implementation; planner_agent.py delegates to this.
    """
    if not guard_statement or not hallucinated_bases or not effective_anchor_names:
        return guard_statement
    _valid_attrs: set = set(effective_anchor_names)
    _hall_set: set = set(hallucinated_bases)
    try:
        _tree = ast.parse(guard_statement.strip(), mode="exec")
    except SyntaxError:
        return guard_statement

    # Guard: if any hallucinated base is used as attribute base (obj.attr),
    # normalization destroys semantic structure (SL28).  Preserve original.
    for _node in ast.walk(_tree):
        if (
            isinstance(_node, ast.Attribute)
            and isinstance(_node.value, ast.Name)
            and _node.value.id in _hall_set
        ):
            return guard_statement

    class _AttrNormalizer(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # type: ignore[override]
            self.generic_visit(node)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id in _hall_set
                and node.attr in _valid_attrs
            ):
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            return node

    _new_tree = _AttrNormalizer().visit(_tree)
    ast.fix_missing_locations(_new_tree)
    try:
        return ast.unparse(_new_tree)
    except (SyntaxError, TypeError, AttributeError):
        return guard_statement


def _has_compound_first_stmt(source: str, symbol_name: str) -> bool:
    """Return True if the first executable statement of symbol_name is a compound.

    Authoritative implementation; planner_agent.py delegates to this.
    """
    try:
        _tree = ast.parse(source)
        _bare = symbol_name.split(".")[-1]
        for _n in ast.walk(_tree):
            if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _n.name == _bare:
                _stmts = _n.body
                for _stmt in _stmts:
                    if isinstance(_stmt, ast.Expr) and isinstance(
                        getattr(_stmt, "value", None), ast.Constant
                    ):
                        continue
                    if isinstance(_stmt, ast.Pass):
                        continue
                    if isinstance(_stmt, ast.Expr) and isinstance(
                        getattr(_stmt, "value", None), ast.Constant
                    ) and getattr(_stmt.value, "value", None) is ...:
                        continue
                    return isinstance(_stmt, _COMPOUND_STMT_TYPES)
                return False
    except (SyntaxError, AttributeError, TypeError):
        pass
    return False


def _safe_unparse(node: "ast.AST") -> str:
    """ast.unparse with silent fallback for non-expression nodes."""
    try:
        return ast.unparse(node)
    except (SyntaxError, TypeError, AttributeError):
        return ""


def _compute_feasibility(
    guard_statement: str,
    gs_tree: ast.Module,
    func_node: ast.FunctionDef,
    file_tree: ast.Module,
    *,
    explicit_insert_scope: str = "",
    explicit_loop_variable: str = "",
    loop_iterable_src: str = "",
) -> GuardFeasibility:
    """Run the 3-rule AST-op eligibility check.

    Extracted from _precheck_guard_add_ast_op; called by analyze_guard.
    Uses _GUARD_BUILTIN_NAMES (unified set).
    """
    _ir_condition, _ir_control = _parse_guard_ir_fast(guard_statement)

    stmt_names: set = {n.id for n in ast.walk(gs_tree) if isinstance(n, ast.Name)}

    control_flow: set = set()
    for _n in ast.walk(gs_tree):
        if isinstance(_n, ast.Return):     control_flow.add("return")
        elif isinstance(_n, ast.Raise):    control_flow.add("raise")
        elif isinstance(_n, ast.Break):    control_flow.add("break")
        elif isinstance(_n, ast.Continue): control_flow.add("continue")
        elif isinstance(_n, (ast.Yield, ast.YieldFrom)): control_flow.add("yield")
        elif isinstance(_n, ast.Await):    control_flow.add("await")

    _args = func_node.args
    param_names: set = (
        {a.arg for a in getattr(_args, "args", [])}
        | {a.arg for a in getattr(_args, "posonlyargs", [])}
        | {a.arg for a in getattr(_args, "kwonlyargs", [])}
    )
    if _args.vararg:
        param_names.add(_args.vararg.arg)
    if _args.kwarg:
        param_names.add(_args.kwarg.arg)
    _known_names = param_names | _GUARD_BUILTIN_NAMES

    def _make(allow: bool, reason: str, scope: str = "function_body",
               loop_var: str = "", req_llm: bool = False) -> GuardFeasibility:
        return GuardFeasibility(
            ast_op_safe=allow, reason_code=reason,
            insert_scope=scope, loop_variable=loop_var, requires_llm=req_llm,
        )

    # Rule 1: Explicit anchor
    if explicit_insert_scope in ("for_loop", "while_loop"):
        if explicit_insert_scope == "for_loop" and not explicit_loop_variable:
            return _make(False, "for_loop_missing_loop_variable", req_llm=True)
        return _make(True, "explicit_anchor", scope=explicit_insert_scope,
                     loop_var=explicit_loop_variable)

    # Rule 1: Auto-detect loop control
    if {"break", "continue"} & control_flow:
        _for_loops = [n for n in ast.walk(func_node) if isinstance(n, ast.For)]
        _loop_target_names: set = set()
        for _fl in _for_loops:
            for _tn in ast.walk(_fl.target):
                if isinstance(_tn, ast.Name):
                    _loop_target_names.add(_tn.id)
        _matches = stmt_names & _loop_target_names
        if len(_matches) == 1:
            _match_var = next(iter(_matches))
            # Additionally verify exactly ONE loop node uses this variable.
            # Multiple loops with the same variable name → ambiguous (AST op
            # requires a single deterministic insertion target).
            _matching_loops = [
                _fl for _fl in _for_loops
                if any(
                    isinstance(_tn, ast.Name) and _tn.id == _match_var
                    for _tn in ast.walk(_fl.target)
                )
            ]
            if len(_matching_loops) == 1:
                return _make(True, "loop_control_unique_anchor",
                             scope="for_loop", loop_var=_match_var)
            # Multiple loops share the same variable — try to disambiguate via
            # iterable_src (e.g. "undefined_names" uniquely identifies one loop).
            if loop_iterable_src and len(_matching_loops) > 1:
                _iterable_filtered = [
                    _fl for _fl in _matching_loops
                    if _safe_unparse(_fl.iter) == loop_iterable_src
                ]
                if len(_iterable_filtered) == 1:
                    return _make(True, "loop_control_iterable_anchor",
                                 scope="for_loop", loop_var=_match_var)
        return _make(False, "loop_control_ambiguous_anchor", scope="", req_llm=True)

    # Rule 2: Local-state dependency
    _unresolved = stmt_names - _known_names
    if _unresolved:
        return _make(False, "local_state_dependent_guard", req_llm=True)

    # Rule 2b: self.attr initialization
    _self_attr_names: set = {
        n.attr for n in ast.walk(gs_tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
    }
    if _self_attr_names:
        _sym_bare = func_node.name
        _enclosing_class: str = ""
        for _nd in ast.walk(file_tree):
            if isinstance(_nd, ast.ClassDef):
                for _it in _nd.body:
                    if (isinstance(_it, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and _it.name == _sym_bare):
                        _enclosing_class = _nd.name
                        break
        if _enclosing_class:
            _init_attrs: set = set()
            for _nd in ast.walk(file_tree):
                if isinstance(_nd, ast.ClassDef) and _nd.name == _enclosing_class:
                    for _it in _nd.body:
                        if not (isinstance(_it, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and _it.name == "__init__"):
                            continue
                        for _st in ast.walk(_it):
                            _tgts: list = []
                            if isinstance(_st, ast.Assign):
                                _tgts = list(_st.targets)
                            elif isinstance(_st, ast.AnnAssign) and _st.target:
                                _tgts = [_st.target]
                            elif isinstance(_st, ast.AugAssign):
                                _tgts = [_st.target]
                            for _t in _tgts:
                                if (isinstance(_t, ast.Attribute)
                                        and isinstance(_t.value, ast.Name)
                                        and _t.value.id == "self"):
                                    _init_attrs.add(_t.attr)
            _uninit = _self_attr_names - _init_attrs
            if _uninit:
                return _make(False, "self_attr_not_initialized", req_llm=True)

    # Rule 3: yield / await
    if {"yield", "await"} & control_flow:
        return _make(False, "contract_risky_control_flow", req_llm=True)

    # Rule 3: return
    if "return" in control_flow:
        if stmt_names.issubset(_known_names):
            return _make(True, "parameter_return_guard")
        return _make(False, "local_state_return_guard", req_llm=True)

    # Rule 3: raise
    if "raise" in control_flow:
        if not stmt_names.issubset(_known_names):
            return _make(False, "raise_not_parameter_guard", req_llm=True)
        return _make(True, "parameter_validation_raise")

    # Default safe: parameter-only condition
    if stmt_names and stmt_names.issubset(_known_names):
        return _make(True, "parameter_guard")

    return _make(False, "no_safe_ast_guard_rule_matched", req_llm=True)


def _parse_guard_ir_fast(guard_stmt: str) -> tuple[Optional[dict], Optional[str]]:
    """Minimal re-implementation of _extract_guard_ir used internally.

    Kept separate from parse_guard to avoid re-building the full GuardIR in
    nested calls.  External callers should use parse_guard() instead.
    """
    try:
        src = guard_stmt.strip()
        _ir_tree = None
        try:
            _ir_tree = ast.parse(src, mode="exec")
        except SyntaxError:
            try:
                _ir_tree = ast.parse(src + "\n    pass", mode="exec")
            except SyntaxError:
                return None, None
        if not _ir_tree.body or not isinstance(_ir_tree.body[0], ast.If):
            return None, None
        stmt = _ir_tree.body[0]
        control = ""
        for _n in ast.walk(stmt):
            if isinstance(_n, ast.Continue):   control = "continue"; break
            elif isinstance(_n, ast.Break):    control = "break";    break
            elif isinstance(_n, ast.Return):   control = "return";   break
            elif isinstance(_n, ast.Raise):    control = "raise";    break
        if not control:
            return None, None
        op = _compute_op_class(stmt.test)
        operands: list = []
        seen: set = set()
        attribute_pairs: list = []
        seen_pairs: set = set()
        for _n in ast.walk(stmt.test):
            tok = None
            if isinstance(_n, ast.Name) and _n.id not in _PY_KW:
                tok = _n.id
            elif isinstance(_n, ast.Attribute) and _n.attr not in _PY_KW:
                tok = _n.attr
                if (isinstance(_n.value, ast.Name) and _n.value.id not in _PY_KW):
                    _pair = (_n.value.id, _n.attr)
                    if _pair not in seen_pairs:
                        attribute_pairs.append(_pair)
                        seen_pairs.add(_pair)
            if tok and tok not in seen:
                operands.append(tok)
                seen.add(tok)
        _ir: dict = {"op": op, "operands": operands}
        if attribute_pairs:
            _ir["attribute_pairs"] = attribute_pairs
        return _ir, control
    except (AttributeError, TypeError):
        return None, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_condensed_guard_src(src: str) -> Optional[str]:
    """Expand a condensed single-line guard to valid multi-line Python.

    Handles "if cond: stmt1 stmt2" (two statements on one line without
    semicolons — not valid Python) by splitting on exit keywords to produce
    "if cond:\n    stmt1\n    stmt2".

    This form appears in DPB-generated guard_statement strings that are
    extracted verbatim from natural-language requests.  Returns None when
    expansion is not applicable or results in a syntax error.
    """
    m = re.match(r'^(if\s+.+?):\s*(.+)$', src, re.DOTALL)
    if not m:
        return None
    head = m.group(1)
    body = m.group(2).strip()
    # Split body on exit keywords used as statement boundaries (no semicolons).
    parts = re.split(
        r'\s+(?=\b(?:continue|break|return(?:\s+\S+)?|raise\s+\w+)\b)',
        body,
    )
    if len(parts) < 2:
        return None
    indented = "\n    ".join(p.strip() for p in parts if p.strip())
    candidate = f"{head}:\n    {indented}"
    try:
        ast.parse(candidate, mode="exec")
        return candidate
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def parse_guard(raw: str) -> Optional[GuardIR]:
    """Parse *raw* into a GuardIR (Step 1: condition + control only).

    Returns None only when *raw* is empty.  Returns a GuardIR with
    condition=None for syntactically invalid or non-guard strings.
    """
    if not raw or not raw.strip():
        return None

    src = raw.strip()
    _tree: Optional[ast.Module] = None
    try:
        _tree = ast.parse(src, mode="exec")
    except SyntaxError:
        try:
            _tree = ast.parse(src + "\n    pass", mode="exec")
        except SyntaxError:
            pass

    if _tree is None or not _tree.body or not isinstance(_tree.body[0], ast.If):
        # Try to expand condensed single-line form:
        # "if cond: stmt1 stmt2" (no semicolons) → "if cond:\n    stmt1\n    stmt2"
        # This format is invalid Python but appears in DPB-generated guard_statement
        # descriptions extracted from natural-language requests.
        _expanded = _expand_condensed_guard_src(src)
        if _expanded:
            try:
                _tree = ast.parse(_expanded, mode="exec")
            except SyntaxError:
                pass
    if _tree is None or not _tree.body or not isinstance(_tree.body[0], ast.If):
        return GuardIR(raw=raw, canonical="", compact="", condition=None, control="")

    stmt: ast.If = _tree.body[0]
    try:
        canonical = ast.unparse(stmt)
    except (SyntaxError, TypeError, AttributeError):
        canonical = src
    compact = _make_compact(canonical)

    control = _extract_control(stmt)
    if not control:
        return GuardIR(raw=raw, canonical=canonical, compact=compact,
                       condition=None, control="")

    condition = _extract_condition(stmt)
    return GuardIR(raw=raw, canonical=canonical, compact=compact,
                   condition=condition, control=control)


def analyze_guard(
    ir: GuardIR,
    file_content: str,
    symbol: str,
    *,
    explicit_insert_scope: str = "",
    explicit_loop_variable: str = "",
    loop_iterable_src: str = "",
) -> GuardIR:
    """Fill ir.placement and ir.feasibility from the host function's AST.

    Returns a new GuardIR with placement + feasibility set.  ir itself is
    not mutated.  When the guard cannot be parsed or the symbol is missing,
    feasibility.ast_op_safe=False with reason_code="missing_symbol_or_guard".
    """
    _gs = ir.raw.strip() if ir else ""
    if not ir or not _gs or not symbol:
        _feas = GuardFeasibility(
            ast_op_safe=False, reason_code="missing_symbol_or_guard",
            insert_scope="function_body", loop_variable="", requires_llm=True,
        )
        _placement = GuardPlacement(
            anchors=[], had_unresolved=False, hallucinated_bases=frozenset(),
            host_function_flavor="unknown", loop_candidates=[],
        )
        return dataclasses.replace(ir, placement=_placement, feasibility=_feas)

    if not ir.is_parsed:
        _feas = GuardFeasibility(
            ast_op_safe=False, reason_code="guard_syntax_error",
            insert_scope="function_body", loop_variable="", requires_llm=True,
        )
        _placement = GuardPlacement(
            anchors=[], had_unresolved=False, hallucinated_bases=frozenset(),
            host_function_flavor=_classify_enclosing_function_flavor(file_content, symbol),
            loop_candidates=[],
        )
        return dataclasses.replace(ir, placement=_placement, feasibility=_feas)

    # is_parsed=True guarantees canonical is valid Python (raw may be condensed/unexpandable).
    # Use canonical for all subsequent AST operations to avoid re-encountering the syntax
    # error that was already resolved during parse_guard's expansion step.
    _gs = ir.canonical.strip() or _gs

    # Parse guard AST
    try:
        _gs_tree = ast.parse(_gs, mode="exec")
    except SyntaxError:
        try:
            _gs_tree = ast.parse(_gs + "\n    pass", mode="exec")
        except SyntaxError:
            _feas = GuardFeasibility(
                ast_op_safe=False, reason_code="guard_syntax_error",
                insert_scope="function_body", loop_variable="", requires_llm=True,
            )
            _placement = GuardPlacement(
                anchors=[], had_unresolved=False, hallucinated_bases=frozenset(),
                host_function_flavor=_classify_enclosing_function_flavor(file_content, symbol),
                loop_candidates=[],
            )
            return dataclasses.replace(ir, placement=_placement, feasibility=_feas)

    # Parse file AST
    try:
        _file_tree = ast.parse(file_content)
    except SyntaxError:
        _feas = GuardFeasibility(
            ast_op_safe=False, reason_code="file_parse_error",
            insert_scope="function_body", loop_variable="", requires_llm=True,
        )
        _placement = GuardPlacement(
            anchors=[], had_unresolved=False, hallucinated_bases=frozenset(),
            host_function_flavor="unknown", loop_candidates=[],
        )
        return dataclasses.replace(ir, placement=_placement, feasibility=_feas)

    # Find function node
    _sym_bare = symbol.split(".")[-1]
    _func_node = next(
        (n for n in ast.walk(_file_tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == _sym_bare),
        None,
    )
    if _func_node is None:
        _feas = GuardFeasibility(
            ast_op_safe=False, reason_code="target_symbol_not_found",
            insert_scope="function_body", loop_variable="", requires_llm=True,
        )
        _placement = GuardPlacement(
            anchors=[], had_unresolved=False, hallucinated_bases=frozenset(),
            host_function_flavor="unknown", loop_candidates=[],
        )
        return dataclasses.replace(ir, placement=_placement, feasibility=_feas)

    # Compute placement fields
    _anchors, _had_unresolved, _hallucinated = _extract_guard_local_anchors(
        _gs, file_content, symbol
    )
    _flavor = _classify_enclosing_function_flavor(file_content, symbol)
    _loop_cands = _extract_for_loop_target_names(file_content, symbol)

    _placement = GuardPlacement(
        anchors=_anchors,
        had_unresolved=_had_unresolved,
        hallucinated_bases=_hallucinated,
        host_function_flavor=_flavor,
        loop_candidates=_loop_cands,
    )

    # Compute feasibility
    _feas = _compute_feasibility(
        _gs, _gs_tree, _func_node, _file_tree,
        explicit_insert_scope=explicit_insert_scope,
        explicit_loop_variable=explicit_loop_variable,
        loop_iterable_src=loop_iterable_src,
    )

    return dataclasses.replace(ir, placement=_placement, feasibility=_feas)
