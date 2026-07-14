"""libcst_transform_utils.py — Format-preserving CST function-body transforms.

General-purpose LibCST CSTTransformer passes for common in-function rewrites.
No dependency on semantic planner models — takes primitive parameters only.

Public API:
    reorder_calls(module, func_name, desired_order) -> module
    replace_call_args(module, func_name, call_name, new_args_src) -> module | None
    rewrite_return(module, func_name, new_return_expr) -> module | None
    move_statement(module, func_name, call_name, before_call) -> module
    stmt_contains_call(stmt, call_name) -> bool
"""
from __future__ import annotations

import logging

try:
    import libcst as cst
    import libcst.matchers as m
    _LIBCST_AVAILABLE = True
except ImportError:
    cst = None  # type: ignore[assignment]
    m = None    # type: ignore[assignment]
    _LIBCST_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _LIBCST_AVAILABLE:
    logger.warning("libcst not installed — CST function-body transforms disabled")


# ── Public helpers ────────────────────────────────────────────────────────────

def stmt_contains_call(stmt, call_name: str) -> bool:
    """Return True if the CST statement contains a call to call_name."""
    if not _LIBCST_AVAILABLE:
        return False
    try:
        for call_node in m.findall(stmt, m.Call()):
            func = call_node.func
            if isinstance(func, cst.Name) and func.value == call_name:
                return True
            if isinstance(func, cst.Attribute) and func.attr.value == call_name:
                return True
    except Exception:
        pass
    return False


def _parse_expr(expr_src: str):
    """Parse expr_src as a LibCST expression node, or return None."""
    try:
        return cst.parse_expression(expr_src.strip())
    except Exception:
        return None


def _parse_args(new_args_src: list[str]):
    """Parse argument source strings as a tuple of LibCST Arg nodes."""
    try:
        fake = f"_f({', '.join(new_args_src)})"
        call = cst.parse_expression(fake)
        if isinstance(call, cst.Call):
            return call.args
    except Exception:
        pass
    return None


# ── CSTTransformer factories ──────────────────────────────────────────────────

def reorder_calls(module, func_name: str, desired_order: list[str]):
    """Reorder call-containing statements within func_name to match desired_order."""
    if not _LIBCST_AVAILABLE:
        return module
    class _Reorder(cst.CSTTransformer):
        def _transform_body(self, body_stmts):
            stmts = list(body_stmts)
            call_indices = {}
            for i, stmt in enumerate(stmts):
                for cname in desired_order:
                    if cname not in call_indices and stmt_contains_call(stmt, cname):
                        call_indices[cname] = i

            present = [c for c in desired_order if c in call_indices]
            if len(present) < 2:
                return stmts

            indices = [call_indices[c] for c in present]
            if indices == sorted(indices):
                return stmts  # already ordered

            indices_set = set(indices)
            new_stmts = []
            inserted = False
            for i, stmt in enumerate(stmts):
                if i in indices_set:
                    if not inserted:
                        for cname in present:
                            new_stmts.append(stmts[call_indices[cname]])
                        inserted = True
                else:
                    new_stmts.append(stmt)
            return new_stmts

        def leave_FunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(updated.body.body)
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

        def leave_AsyncFunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(updated.body.body)
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

    return module.visit(_Reorder())


def replace_call_args(module, func_name: str, call_name: str, new_args_src: list[str]):
    """Replace arguments of the first call to call_name inside func_name."""
    if not _LIBCST_AVAILABLE:
        return None
    new_arg_nodes = _parse_args(new_args_src)
    if new_arg_nodes is None:
        return None

    class _ReplaceArgs(cst.CSTTransformer):
        def __init__(self):
            self._depth = 0
            self._replaced = False

        def visit_FunctionDef(self, node):
            if node.name.value == func_name:
                self._depth += 1
            return True

        def leave_FunctionDef(self, original, updated):
            if updated.name.value == func_name:
                self._depth -= 1
            return updated

        def visit_AsyncFunctionDef(self, node):
            if node.name.value == func_name:
                self._depth += 1
            return True

        def leave_AsyncFunctionDef(self, original, updated):
            if updated.name.value == func_name:
                self._depth -= 1
            return updated

        def leave_Call(self, original, updated):
            if self._depth == 0 or self._replaced:
                return updated
            func = updated.func
            name = None
            if isinstance(func, cst.Name):
                name = func.value
            elif isinstance(func, cst.Attribute):
                name = func.attr.value
            if name != call_name:
                return updated
            self._replaced = True
            return updated.with_changes(args=new_arg_nodes)

    return module.visit(_ReplaceArgs())


def rewrite_return(module, func_name: str, new_return_expr: str):
    """Replace the last return statement's value in func_name."""
    if not _LIBCST_AVAILABLE:
        return None
    new_value = _parse_expr(new_return_expr)
    if new_value is None:
        return None

    class _RewriteReturn(cst.CSTTransformer):
        def _transform_body(self, body_stmts):
            stmts = list(body_stmts)
            for i in range(len(stmts) - 1, -1, -1):
                stmt = stmts[i]
                if isinstance(stmt, cst.SimpleStatementLine):
                    for j, small in enumerate(stmt.body):
                        if isinstance(small, cst.Return):
                            try:
                                old_src = small.value.deep_clone().code if small.value else ""
                                if old_src.strip() == new_return_expr.strip():
                                    return stmts  # idempotent
                            except Exception:
                                pass
                            new_small = small.with_changes(value=new_value)
                            new_stmt = stmt.with_changes(
                                body=[*list(stmt.body[:j]), new_small, *list(stmt.body[j + 1:])]
                            )
                            stmts[i] = new_stmt
                            return stmts
                elif isinstance(stmt, cst.Return):
                    stmts[i] = stmt.with_changes(value=new_value)
                    return stmts
            return stmts

        def leave_FunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

        def leave_AsyncFunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

    return module.visit(_RewriteReturn())


def move_statement(module, func_name: str, call_name: str, before_call: str):
    """Move the statement containing call_name to before before_call in func_name."""
    if not _LIBCST_AVAILABLE:
        return module
    class _Move(cst.CSTTransformer):
        def _transform_body(self, body_stmts):
            stmts = list(body_stmts)
            source_idx = next(
                (i for i, s in enumerate(stmts) if stmt_contains_call(s, call_name)), None
            )
            target_idx = next(
                (i for i, s in enumerate(stmts) if stmt_contains_call(s, before_call)), None
            )
            if source_idx is None or target_idx is None:
                return stmts
            if source_idx < target_idx:
                return stmts  # already in position

            stmt_to_move = stmts.pop(source_idx)
            adj_target = target_idx if target_idx < source_idx else target_idx - 1
            stmts.insert(adj_target, stmt_to_move)
            return stmts

        def leave_FunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

        def leave_AsyncFunctionDef(self, original, updated):
            if updated.name.value != func_name:
                return updated
            new_body = self._transform_body(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_body))

    return module.visit(_Move())
