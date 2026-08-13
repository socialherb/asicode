"""guard_ir — GuardIR: single source of truth for guard statement semantics.

Parses guard statements (``if <cond>: <control>``) into a read-only IR:
GuardCondition (op_class/operands/attribute_pairs) + control keyword.  This is
the canonical parse step used by IntentResolver; placement/feasibility
analysis (former Step 2) moved out of this module in later steps.

Circular-import safety: imports only stdlib (ast, dataclasses, re).
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Shared builtin-name sets
# ---------------------------------------------------------------------------

# Python keyword tokens excluded from IR operand lists.
_PY_KW: frozenset[str] = frozenset(
    {"not", "in", "is", "and", "or", "True", "False", "None"}
)



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
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_condensed_guard_src(src: str, *, _return_tree: bool = False):
    """Expand a condensed single-line guard to valid multi-line Python.

    Handles "if cond: stmt1 stmt2" (two statements on one line without
    semicolons — not valid Python) by splitting on exit keywords to produce
    "if cond:\n    stmt1\n    stmt2".

    This form appears in DPB-generated guard_statement strings that are
    extracted verbatim from natural-language requests.  Returns None when
    expansion is not applicable or results in a syntax error.  With
    ``_return_tree=True``, returns ``(candidate, tree)`` instead of the bare
    candidate so the caller can reuse the already-parsed AST instead of
    re-parsing the same string (see parse_guard).
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
    _tree: Optional[ast.AST] = None
    with contextlib.suppress(SyntaxError):
        _tree = ast.parse(candidate, mode="exec")
    if _tree is None:
        return None
    if _return_tree:
        return candidate, _tree
    return candidate


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
        with contextlib.suppress(SyntaxError):
            _tree = ast.parse(src + "\n    pass", mode="exec")

    if _tree is None or not _tree.body or not isinstance(_tree.body[0], ast.If):
        # Try to expand condensed single-line form:
        # "if cond: stmt1 stmt2" (no semicolons) → "if cond:\n    stmt1\n    stmt2"
        # This format is invalid Python but appears in DPB-generated guard_statement
        # descriptions extracted from natural-language requests.
        # The expander already parsed the candidate for its own syntax check —
        # reuse that tree instead of parsing the same string a second time.
        _expanded = _expand_condensed_guard_src(src, _return_tree=True)
        if _expanded:
            _tree = _expanded[1]
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

