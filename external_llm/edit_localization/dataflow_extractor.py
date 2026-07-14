"""Intra-function dataflow fact extraction via AST analysis.

Extracts structural facts about what a symbol *does* — not what tokens
it contains. These facts drive edit localization scoring.

Design:
- Pure AST, no LLM, no string matching
- Focuses on intra-function def-use: assignments, derivations, returns
- Identifies structural roles: delegation, value-determination, construction
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional
# Noise tokens to exclude from extracted facts
_NOISE_NAMES: frozenset = frozenset({
    "self", "cls", "None", "True", "False",
    "__name__", "__main__",
    "utf-8", "replace", "r", "w", "rb", "wb",
    "end", "start",
})

_NOISE_STRINGS: frozenset = frozenset({
    "", "utf-8", "replace", "r", "w", "rb", "wb",
})


@dataclass
class SymbolFlowFacts:
    """Dataflow facts extracted from a single symbol body.

    These facts describe *what the function does structurally*, enabling
    edit localization without relying on keyword matching.
    """

    # --- Direct values ---
    string_literals: set[str] = field(default_factory=set)
    """String constants in assignments, comparisons, ternaries."""

    assigned_names: set[str] = field(default_factory=set)
    """Variable names that receive values (LHS of assignments)."""

    # --- Dataflow ---
    derives_from: dict[str, set[str]] = field(default_factory=dict)
    """def-use chains: variable -> set of variables it depends on.
    e.g. kind derives from {is_async} when: kind = "x" if is_async else "y"
    """

    # --- Structural patterns ---
    return_names: set[str] = field(default_factory=set)
    """Variable names or attribute accesses that appear in return statements."""

    delegation_calls: set[str] = field(default_factory=set)
    """Function names whose result is directly returned (pure delegation)."""

    constructor_calls: dict[str, set[str]] = field(default_factory=dict)
    """ClassName -> set of keyword argument names used in constructor calls."""

    attribute_writes: set[str] = field(default_factory=set)
    """Attribute names written: self.kind, obj.field -> 'kind', 'field'."""

    call_sites: dict[str, list[str]] = field(default_factory=dict)
    """Per-callee list of literal arg sets observed at each call site.
    Enables object identity: get_user → [["1"], ["2"]] means two distinct
    objects are created/retrieved. Used to match requests like "user id 1".
    e.g. {"get_user": [["1"], ["2"]], "fetch": [['"admin"']]}
    """

    mutating_calls: set[str] = field(default_factory=set)
    """Callee names where is_mutating=True in the call graph.
    Populated from graph edge data (not re-derived from AST here).
    Signals "this function writes to external state via these callees."
    """

    alias_chains: dict[str, str] = field(default_factory=dict)
    """Single-hop alias map: alias_name -> original_name for x = y assignments.
    Enables object identity: u2 = u1 → alias_chains["u2"] = "u1"
    Combined with call_sites, traces "user 1" → u1 → u2 → u2.email.
    Only captures direct Name assignments (x = y), not computed aliases.
    Use _resolve_alias_root() to follow multi-hop chains.
    """

    param_names: set[str] = field(default_factory=set)
    """Function parameter names (for distinguishing pass-through from transform)."""

    # --- Semantic tags (derived from above) ---
    tags: set[str] = field(default_factory=set)
    """Structural role tags, e.g.:
    - pure_delegation: function only delegates to another
    - value_determiner: function assigns/computes values directly
    - field_constructor: function builds objects with keyword fields
    - conditional_logic: function has if/else that determines values
    - collection_builder: function appends/extends to a collection
    - pass_through: function returns input with minimal transformation
    """


def extract_flow_facts(body_source: str) -> SymbolFlowFacts:
    """Extract dataflow facts from a function body source.

    Args:
        body_source: Source code of the function body (including def line).

    Returns:
        SymbolFlowFacts with all extracted structural information.
    """
    facts = SymbolFlowFacts()

    try:
        tree = ast.parse(body_source)
    except SyntaxError:
        return facts

    # Extract parameter names from the function definition
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            facts.param_names = _extract_param_names(node)
            break

    # Walk all nodes for fact extraction
    _extract_assignments(tree, facts)
    _extract_returns(tree, facts)
    _extract_constructor_calls(tree, facts)
    _extract_attribute_writes(tree, facts)
    _extract_comparisons(tree, facts)
    _extract_call_sites(tree, facts)

    # Build derivation chains and single-hop alias map
    _build_derives_from(tree, facts)
    _extract_alias_chains(tree, facts)

    # Derive semantic tags from extracted facts
    _derive_tags(facts)

    return facts


def _extract_call_sites(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Populate call_sites: per-callee list of literal arg sets.

    For each function call in the body, record the callee name and any
    literal positional argument values. This enables object identity:

        u1 = get_user(1)      # call_sites["get_user"] = [["1"], ...]
        u2 = get_user(2)      # call_sites["get_user"] = [["1"], ["2"]]

    Only ast.Constant args are captured — expressions are skipped (they add
    no identity signal since their value is dynamic).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Resolve callee name
        func = node.func
        if isinstance(func, ast.Name):
            callee = func.id
        elif isinstance(func, ast.Attribute):
            callee = func.attr
        else:
            continue

        # Collect literal positional args
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                args.append(repr(arg.value))

        facts.call_sites.setdefault(callee, []).append(args)


def _extract_param_names(func_node: ast.AST) -> set[str]:
    """Extract parameter names from function definition."""
    names: set[str] = set()
    args = func_node.args
    for arg in args.args + args.posonlyargs + args.kwonlyargs:
        if arg.arg not in _NOISE_NAMES:
            names.add(arg.arg)
    if args.vararg and args.vararg.arg not in _NOISE_NAMES:
        names.add(args.vararg.arg)
    if args.kwarg and args.kwarg.arg not in _NOISE_NAMES:
        names.add(args.kwarg.arg)
    return names


def _extract_assignments(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Extract assignment targets and string literal values."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue

        # Get target names
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                targets.extend(_names_from_target(t))
        elif isinstance(node, ast.AnnAssign) and node.target:
            targets.extend(_names_from_target(node.target))
        elif isinstance(node, ast.AugAssign):
            targets.extend(_names_from_target(node.target))

        for name in targets:
            if name not in _NOISE_NAMES:
                facts.assigned_names.add(name)

        # Extract string literals from value side
        value_node = getattr(node, "value", None)
        if value_node:
            for child in ast.walk(value_node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    val = child.value
                    if val not in _NOISE_STRINGS and len(val) > 1:
                        facts.string_literals.add(val)


def _extract_returns(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Extract return expression analysis — names, delegation calls."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue

        val = node.value

        # Direct name return: return result
        if isinstance(val, ast.Name):
            facts.return_names.add(val.id)

        # Direct function call return: return _walk_definitions(tree)
        elif isinstance(val, ast.Call):
            callee = _call_name(val)
            if callee:
                facts.delegation_calls.add(callee)
                facts.return_names.add(callee)

        # Attribute return: return self.result
        elif isinstance(val, ast.Attribute):
            facts.return_names.add(val.attr)

        # Collect all names referenced in return expression
        for child in ast.walk(val):
            if isinstance(child, ast.Name) and child.id not in _NOISE_NAMES:
                facts.return_names.add(child.id)


def _extract_constructor_calls(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Extract constructor/function calls with keyword arguments.

    Captures patterns like: DefinitionInfo(kind=kind, name=name)
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        callee = _call_name(node)
        if not callee:
            continue

        # Only track calls with keyword arguments (constructor-like patterns)
        if not node.keywords:
            continue

        # Heuristic: constructor calls typically start with uppercase
        # but also track calls with 2+ kwargs regardless
        is_constructor = callee[0].isupper() if callee else False
        has_significant_kwargs = len(node.keywords) >= 2

        if is_constructor or has_significant_kwargs:
            kwargs: set[str] = set()
            for kw in node.keywords:
                if kw.arg and kw.arg not in _NOISE_NAMES:
                    kwargs.add(kw.arg)
            if kwargs:
                existing = facts.constructor_calls.get(callee, set())
                facts.constructor_calls[callee] = existing | kwargs


def _extract_attribute_writes(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Extract attribute writes: self.kind = ..., obj.field = ..."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.target:
            targets = [node.target]

        for t in targets:
            if isinstance(t, ast.Attribute):
                if t.attr not in _NOISE_NAMES:
                    facts.attribute_writes.add(t.attr)


def _extract_comparisons(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Extract string constants from comparisons and ternary expressions."""
    for node in ast.walk(tree):
        # Comparisons: if x == "value"
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    val = comparator.value
                    if val not in _NOISE_STRINGS and len(val) > 1:
                        facts.string_literals.add(val)

        # Ternary: "a" if cond else "b"
        if isinstance(node, ast.IfExp):
            for sub in (node.body, node.orelse):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    val = sub.value
                    if val not in _NOISE_STRINGS and len(val) > 1:
                        facts.string_literals.add(val)


def _build_derives_from(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Build lightweight def-use chains within the function.

    For each assignment `x = expr`, records which names in `expr` feed
    into `x`. This allows tracing "kind depends on is_async" without
    full SSA.

    Also tracks conditional derivations:
      x = "a" if flag else "b"  =>  x derives from {flag}
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        target_names = []
        for t in node.targets:
            target_names.extend(_names_from_target(t))

        if not target_names:
            continue

        # Collect all Name references in the value expression
        source_names: set[str] = set()
        for child in ast.walk(node.value):
            if isinstance(child, ast.Name) and child.id not in _NOISE_NAMES:
                source_names.add(child.id)

        # Don't record self-references or trivial cases
        for tgt in target_names:
            if tgt in _NOISE_NAMES:
                continue
            deps = source_names - {tgt}
            if deps:
                existing = facts.derives_from.get(tgt, set())
                facts.derives_from[tgt] = existing | deps


def _extract_alias_chains(tree: ast.AST, facts: SymbolFlowFacts) -> None:
    """Build single-hop alias map from direct variable-to-variable assignments.

    Only captures ``x = y`` patterns where the RHS is a plain Name reference.
    Computed aliases (x = y + z, x = f(y)) are NOT captured here — those are
    tracked as derivations in derives_from.

    This enables object identity tracing:
        u1 = get_user(1)   → call_sites["get_user"] = [["1"]]
        u2 = u1            → alias_chains["u2"] = "u1"
        u2.email = "x"     → attribute_writes = {"email"}
    Combined: "user 1's email" matches get_user("1") → u1 → u2 → email.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Only single-target simple assignments
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        # RHS must be a plain variable reference (not a call, subscript, etc.)
        if not isinstance(node.value, ast.Name):
            continue

        alias = target.id
        original = node.value.id
        if alias in _NOISE_NAMES or original in _NOISE_NAMES:
            continue
        if alias == original:
            continue

        facts.alias_chains[alias] = original


def resolve_alias_root(name: str, alias_chains: dict[str, str], max_hops: int = 8) -> str:
    """Follow alias_chains to the root variable (the original, non-alias source).

    Stops if a cycle is detected or max_hops is reached.

    Example:
        alias_chains = {"u2": "u1", "u3": "u2"}
        resolve_alias_root("u3", ...) → "u1"
    """
    visited: set[str] = set()
    current = name
    for _ in range(max_hops):
        if current in visited:
            break  # cycle guard
        visited.add(current)
        parent = alias_chains.get(current)
        if parent is None:
            break
        current = parent
    return current


def _derive_tags(facts: SymbolFlowFacts) -> None:
    """Derive semantic tags from the extracted structural facts."""

    has_direct_values = bool(facts.string_literals) or bool(facts.attribute_writes)
    has_assignments = bool(facts.assigned_names - facts.param_names)
    has_constructors = bool(facts.constructor_calls)
    has_derivations = bool(facts.derives_from)
    has_delegation = bool(facts.delegation_calls)

    # Pure delegation: function only delegates, no value assignment
    if has_delegation and not has_direct_values and not has_assignments:
        facts.tags.add("pure_delegation")

    # Value determiner: directly assigns values (string literals, computed values)
    if has_direct_values and has_assignments:
        facts.tags.add("value_determiner")

    # Field constructor: builds objects with keyword fields
    if has_constructors:
        facts.tags.add("field_constructor")

    # Conditional logic: has derivation chains (x depends on y)
    if has_derivations:
        facts.tags.add("conditional_logic")

    # Collection builder: appends/extends pattern
    # Detected via assigned_names containing typical collection patterns
    # combined with constructor calls (building items to add)
    if has_constructors and has_derivations:
        facts.tags.add("collection_builder")

    # Pass-through: returns parameters with no significant transformation
    returned_params = facts.return_names & facts.param_names
    if returned_params and not has_direct_values and not has_derivations:
        facts.tags.add("pass_through")


# ---------- AST helpers ----------


def _names_from_target(target: ast.AST) -> list[str]:
    """Extract variable names from assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple) or isinstance(target, ast.List):
        result = []
        for elt in target.elts:
            result.extend(_names_from_target(elt))
        return result
    if isinstance(target, ast.Starred):
        return _names_from_target(target.value)
    return []


def _call_name(node: ast.Call) -> Optional[str]:
    """Extract the function name from a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None
