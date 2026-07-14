"""ast_transform_utils.py — Phase C.3: AST-Based Code Transformations.

Pure AST transformations on function bodies:
- reorder_calls: reorder call statements within a function
- replace_call_args: replace arguments of a specific call
- rewrite_return: replace the last return expression
- move_statement: move a statement before/after another
- remove_if_blocks_by_condition_attributes: remove if-blocks whose condition
  references specific self.X attributes (structural removal without LLM).

All transforms:
- Operate on ast.Module in-place
- Return True/False for success
- Never break syntax (caller validates after)
"""
from __future__ import annotations

import ast
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def reorder_calls(
    tree: ast.Module,
    func_name: str,
    desired_order: list[str],
) -> bool:
    """Reorder call statements within a function to match desired_order.

    Moves statements containing calls to the specified functions
    so they appear in the desired order. Non-matching statements
    stay in their original relative positions.

    Returns True if any reordering was performed.
    """
    func_node = _find_function(tree, func_name)
    if not func_node or not func_node.body:
        return False

    body = func_node.body

    # Map: call_name → index of statement containing it
    call_indices: dict[str, int] = {}
    for i, stmt in enumerate(body):
        for call_name in desired_order:
            if _stmt_calls(stmt, call_name) and call_name not in call_indices:
                call_indices[call_name] = i

    # Find which calls are present and out of order
    present_calls = [c for c in desired_order if c in call_indices]
    if len(present_calls) < 2:
        return False  # Nothing to reorder

    # Check if already in order
    indices = [call_indices[c] for c in present_calls]
    if indices == sorted(indices):
        return False  # Already in correct order

    # Extract the statements to reorder
    stmts_to_move: list[tuple[str, ast.stmt]] = []
    indices_to_remove: set[int] = set()
    for call_name in present_calls:
        idx = call_indices[call_name]
        stmts_to_move.append((call_name, body[idx]))
        indices_to_remove.add(idx)

    # Build new body: keep non-moved stmts, insert moved ones at first moved position
    first_moved_idx = min(indices_to_remove)
    new_body = []
    inserted = False

    for i, stmt in enumerate(body):
        if i in indices_to_remove:
            if not inserted:
                # Insert all moved stmts in desired order
                for call_name, moved_stmt in stmts_to_move:
                    new_body.append(moved_stmt)
                inserted = True
            # Skip original position
        else:
            new_body.append(stmt)

    func_node.body = new_body

    # Fix line numbers for unparse
    ast.fix_missing_locations(tree)

    logger.debug(
        "reorder_calls: %s → %s in %s()",
        [call_indices[c] for c in present_calls],
        list(range(first_moved_idx, first_moved_idx + len(present_calls))),
        func_name,
    )
    return True


def replace_call_args(
    tree: ast.Module,
    func_name: str,
    call_name: str,
    new_args: list[str],
) -> bool:
    """Replace the arguments of a specific function call.

    Finds the first call to call_name within func_name
    and replaces its arguments with new_args (parsed as expressions).

    Returns True if replacement was performed.
    """
    func_node = _find_function(tree, func_name)
    if not func_node:
        return False

    # Find the call node
    call_node = None
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name == call_name:
                call_node = node
                break

    if call_node is None:
        return False

    # Parse new arguments as expressions
    parsed_args = []
    for arg_str in new_args:
        try:
            expr_tree = ast.parse(arg_str, mode='eval')
            parsed_args.append(expr_tree.body)
        except SyntaxError:
            logger.debug("replace_call_args: failed to parse arg %r", arg_str)
            return False

    # Replace
    call_node.args = parsed_args
    call_node.keywords = []  # Clear keywords to avoid conflicts

    ast.fix_missing_locations(tree)
    logger.debug("replace_call_args: %s(%s) in %s()", call_name, new_args, func_name)
    return True


def rewrite_return(
    tree: ast.Module,
    func_name: str,
    new_return_expr: str,
) -> bool:
    """Replace the last return statement's value in a function.

    Returns True if replacement was performed.
    """
    func_node = _find_function(tree, func_name)
    if not func_node or not func_node.body:
        return False

    # Find last return
    last_return = None
    for stmt in reversed(func_node.body):
        if isinstance(stmt, ast.Return):
            last_return = stmt
            break

    if last_return is None:
        return False

    # Parse new return expression
    try:
        expr_tree = ast.parse(new_return_expr, mode='eval')
        new_value = expr_tree.body
    except SyntaxError:
        logger.debug("rewrite_return: failed to parse %r", new_return_expr)
        return False

    # Check if return already matches (idempotency)
    try:
        old_code = ast.unparse(last_return.value) if last_return.value else ""
        new_code = ast.unparse(new_value)
        if old_code == new_code:
            return False  # Already matches
    except Exception:
        pass

    last_return.value = new_value
    ast.fix_missing_locations(tree)
    logger.debug("rewrite_return: → %s in %s()", new_return_expr[:50], func_name)
    return True


def move_statement(
    tree: ast.Module,
    func_name: str,
    call_name: str,
    before_call: str,
) -> bool:
    """Move a statement containing call_name to before the statement containing before_call.

    Returns True if the move was performed.
    """
    func_node = _find_function(tree, func_name)
    if not func_node or not func_node.body:
        return False

    body = func_node.body

    # Find source and target indices
    source_idx = None
    target_idx = None

    for i, stmt in enumerate(body):
        if source_idx is None and _stmt_calls(stmt, call_name):
            source_idx = i
        if target_idx is None and _stmt_calls(stmt, before_call):
            target_idx = i

    if source_idx is None or target_idx is None:
        return False

    # Already in position
    if source_idx < target_idx:
        return False

    # Move: remove from source, insert before target
    stmt_to_move = body.pop(source_idx)
    # After popping, target_idx may have shifted
    if target_idx > source_idx:
        target_idx -= 1
    body.insert(target_idx, stmt_to_move)

    ast.fix_missing_locations(tree)
    logger.debug("move_statement: %s before %s in %s()", call_name, before_call, func_name)
    return True


def remove_if_blocks_by_condition_attributes(
    tree: ast.Module,
    func_name: str,
    attr_names: list[str],
) -> bool:
    """Remove if/elif/else blocks whose condition references specific self.X attributes.

    Walks the function body looking for ``if`` statements where the condition
    contains ``ast.Attribute(value=ast.Name('self'), attr=X)`` for any X in
    *attr_names*.

    When the **outermost** ``if`` matches, the ``if`` itself is removed but its
    ``orelse`` branches (elif/else) are **promoted** to the function body level
    so that remaining elif conditions are preserved.

    For ``elif`` branches inside an outer if whose condition does NOT match,
    only the elif branch is removed (the remaining chain is re-wired).

    Examples::

        # Before:                # After (remove vortex_active):
        if self.vortex_active:   if self.laser_active:
            ...                      ...
        elif self.laser_active:  elif self.spiral_active:
            ...                      ...
        elif self.spiral_active: else:
            ...                      ...
        else:
            ...

    Returns True if any blocks were removed.
    """
    func_node = _find_function(tree, func_name)
    if not func_node or not func_node.body:
        return False

    _attr_set = set(attr_names)
    modified = False

    # Walk backwards through body to safely remove items.
    i = len(func_node.body) - 1
    while i >= 0:
        stmt = func_node.body[i]
        if isinstance(stmt, ast.If):
            if _if_condition_has_attributes(stmt, _attr_set):
                # Remove the matched if, but PROMOTE its orelse (elif/else
                # branches) to the function body level.
                _promoted = list(stmt.orelse)  # copy before pop
                func_node.body.pop(i)
                modified = True  # removal happened (pop succeeded)
                # If no orelse (no elif/else) to promote, the body just shrunk
                # by 1. Decrement i to keep it valid — otherwise the next
                # `func_node.body[i]` would IndexError if this was the last
                # statement (common case: trailing `if self.X:` with no else).
                if not _promoted:
                    i -= 1
                    continue
                # Insert promoted branches at position i.
                for _p in reversed(_promoted):
                    func_node.body.insert(i, _p)
                # Do NOT decrement i: the promoted branches at position i
                # still need to be checked for matching conditions.
                continue
            else:
                # Check elif branches (elif = nested if in orelse).
                _modified_orelse = _prune_elif_branches(stmt, _attr_set)
                if _modified_orelse:
                    modified = True
        i -= 1

    if modified:
        ast.fix_missing_locations(tree)

    return modified


# ── Internal helpers for structural removal ──────────────────────

def _if_condition_has_attributes(if_node: ast.If, attr_set: set) -> bool:
    """Check if an if-node's condition references self.X for any X in attr_set."""
    for node in ast.walk(if_node.test):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in attr_set):
            return True
    return False


def _prune_elif_branches(if_node: ast.If, attr_set: set) -> bool:
    """Remove elif/else branches whose conditions match attr_set.

    Python's AST represents elif chains as nested ``If`` nodes in ``orelse``::

        If(test=A, body=[...], orelse=[
            If(test=B, body=[...], orelse=[
                If(test=C, body=[...], orelse=[...else body...])
            ])
        ])

    To remove ``elif B``, we replace ``orelse=[B_IF]`` with ``orelse=B_IF.orelse``
    (effectively splicing B's tail into A's else position).

    Mutates if_node.orelse in-place. Returns True if any branch removed.
    """
    modified = False
    while if_node.orelse:
        elif_node = if_node.orelse[0]
        if isinstance(elif_node, ast.If):
            if _if_condition_has_attributes(elif_node, attr_set):
                # Splice: replace orelse with the elif's own orelse.
                # This removes the elif but preserves its tail (further elifs / else).
                if_node.orelse = list(elif_node.orelse)
                modified = True
                # Re-check new orelse[0] in case it's another matching elif.
                continue
            else:
                # Recurse into the elif's orelse chain.
                if _prune_elif_branches(elif_node, attr_set):
                    modified = True
                break  # chain continuation handled by recursion
        else:
            # Bare else node (not an If) — cannot match attributes, stop.
            break
    return modified


def _extract_self_attrs_from_source(source: str) -> set[str]:
    """Extract all ``self.X`` attribute names from Python source code.

    Uses AST to find Attribute nodes where value is ``ast.Name('self')``.
    Filters out method calls (self.X() → X is a method, not an attribute check)
    and dunder methods.

    Returns a set of attribute names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    attrs: set[str] = set()
    calls_in_source: set[str] = set()

    # First pass: collect all method calls (self.X()) to filter them out.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"):
                calls_in_source.add(node.func.attr)

    # Second pass: collect self.X attribute references (excluding method calls).
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr not in calls_in_source
                and not node.attr.startswith("__")
                and not node.attr.endswith("__")):
            attrs.add(node.attr)

    # Third pass: in if-conditions, also include method calls on self (self.xxx())
    # since they are used as condition checks in if-blocks.
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for sub in ast.walk(node.test):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "self"):
                    attrs.add(sub.func.attr)

    return attrs

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    """Find a function or method definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def _get_call_name(call_node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    if isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return ""


def _stmt_calls(stmt: ast.stmt, call_name: str) -> bool:
    """Check if a statement contains a call to the named function."""
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            if _get_call_name(node) == call_name:
                return True
    return False


def safe_unparse(tree: ast.Module) -> Optional[str]:
    """Safely unparse an AST tree back to source code.

    Returns None if unparse fails.
    """
    try:
        return ast.unparse(tree)
    except Exception as e:
        logger.debug("safe_unparse failed: %s", e)
        return None
