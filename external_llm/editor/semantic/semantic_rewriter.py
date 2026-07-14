"""semantic_rewriter.py — Phase C.3: AST-Based Rewrite Executor.

Applies RewritePlan operations using AST transformations.
All rewrites are syntax-validated before committing.
Rollback on any failure.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Optional

from external_llm.editor.semantic import ast_transform_utils as transforms
from external_llm.editor.semantic.libcst_transform_utils import (
    move_statement as _cst_move,
)
from external_llm.editor.semantic.libcst_transform_utils import (
    reorder_calls as _cst_reorder,
)
from external_llm.editor.semantic.libcst_transform_utils import (
    replace_call_args as _cst_replace_args,
)
from external_llm.editor.semantic.libcst_transform_utils import (
    rewrite_return as _cst_rewrite_return,
)
from external_llm.editor.semantic.semantic_rewrite_models import (
    RewriteOperation,
    RewriteOpType,
    RewritePlan,
    RewriteResult,
)
from external_llm.languages.libcst_utils import (
    parse_module as _cst_parse,
)
from external_llm.languages.libcst_utils import (
    splice_modified_functions as _splice_modified,
)

logger = logging.getLogger(__name__)

# Maximum diff size (lines changed) before we reject a rewrite
MAX_DIFF_LINES = 30


def apply_rewrite_plan(plan: RewritePlan) -> RewriteResult:
    """Apply a rewrite plan to a single file.

    Source generation priority:
      1. LibCST direct (primary) — no ast.unparse, preserves all formatting/comments.
         Tried first regardless of AST outcome; handles cases AST cannot (keyword args).
      2. LibCST splice (fallback) — replaces only modified functions, keeps rest verbatim.
      3. AST full unparse (last resort) — correct but formatting lost.
    """
    result = RewriteResult()

    if plan.is_empty:
        result.success = True
        return result

    file_path = plan.file_path
    if not os.path.isfile(file_path):
        result.error = f"File not found: {file_path}"
        return result

    # 1. Read + backup
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            original_source = f.read()
    except Exception as e:
        result.error = f"Read failed: {e}"
        return result

    backup = original_source

    # 2. Try LibCST path first — format-preserving, handles keyword args, etc.
    new_source = _apply_ops_via_libcst(original_source, plan.operations)
    modified_functions: set = set()

    if new_source is not None:
        # LibCST succeeded: credit all ops as applied if source actually changed
        if new_source != original_source:
            for op in plan.operations:
                result.applied_ops.append(f"{op.op_type}:{op.target_function}")
                modified_functions.add(op.target_function)
                logger.debug("[C.3/libcst] applied %s on %s()", op.op_type, op.target_function)
        else:
            result.success = True  # all ops were no-ops
            return result
    else:
        # 3. AST fallback — apply ops in-place, then generate source
        try:
            tree = ast.parse(original_source)
        except SyntaxError as e:
            result.error = f"Parse failed: {e}"
            return result

        any_applied = False
        for op in plan.operations:
            try:
                applied = _apply_operation(tree, op)
                if applied:
                    result.applied_ops.append(f"{op.op_type}:{op.target_function}")
                    any_applied = True
                    modified_functions.add(op.target_function)
                    logger.debug("[C.3/ast] applied %s on %s()", op.op_type, op.target_function)
                else:
                    result.skipped_ops.append(f"{op.op_type}:{op.target_function}")
            except Exception as e:
                result.skipped_ops.append(f"{op.op_type}:{op.target_function}:error:{e}")
                logger.debug("[C.3] op failed: %s — %s", op.op_type, e)

        if not any_applied:
            result.success = True
            return result

        new_source = _splice_modified(original_source, tree, modified_functions)
        if new_source is None:
            new_source = transforms.safe_unparse(tree)
            if new_source is None:
                result.error = "AST unparse failed"
                return result
            new_source = _format_unparsed(new_source)

    # 4. Validate syntax
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        result.error = f"Post-rewrite syntax error: {e}"
        _rollback(file_path, backup)
        return result

    # 5. Check diff size
    orig_lines = original_source.splitlines()
    new_lines = new_source.splitlines()
    diff_count = abs(len(new_lines) - len(orig_lines))
    for o, n in zip(orig_lines, new_lines, strict=False):
        if o != n:
            diff_count += 1
    if diff_count > MAX_DIFF_LINES:
        result.error = f"Diff too large: {diff_count} lines (max {MAX_DIFF_LINES})"
        return result

    # 6. Write
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_source)
        result.success = True
        result.files_modified.append(file_path)
        logger.info(
            "[C.3 REWRITE] %d ops applied to %s (diff=%d lines)",
            len(result.applied_ops), os.path.basename(file_path), diff_count,
        )
    except Exception as e:
        result.error = f"Write failed: {e}"
        _rollback(file_path, backup)

    return result


def apply_rewrite_plans(plans: list[RewritePlan]) -> RewriteResult:
    """Apply multiple rewrite plans. Aggregate results."""
    aggregate = RewriteResult(success=True)

    for plan in plans:
        if plan.is_empty:
            continue
        r = apply_rewrite_plan(plan)
        aggregate.applied_ops.extend(r.applied_ops)
        aggregate.skipped_ops.extend(r.skipped_ops)
        aggregate.files_modified.extend(r.files_modified)
        if not r.success:
            aggregate.success = False
            if r.error:
                aggregate.error = r.error  # Last error wins

    return aggregate


def _apply_operation(tree: ast.Module, op: RewriteOperation) -> bool:
    """Apply a single rewrite operation to the AST."""
    if op.op_type == RewriteOpType.REORDER_CALLS:
        order = op.payload.get("order", [])
        if not order:
            return False
        return transforms.reorder_calls(tree, op.target_function, order)

    if op.op_type == RewriteOpType.REPLACE_CALL_ARGS:
        call_name = op.payload.get("call_name", "")
        new_args = op.payload.get("new_args", [])
        if not call_name or not new_args:
            return False
        return transforms.replace_call_args(tree, op.target_function, call_name, new_args)

    if op.op_type == RewriteOpType.REWRITE_RETURN:
        new_return = op.payload.get("new_return", "")
        if not new_return:
            return False
        return transforms.rewrite_return(tree, op.target_function, new_return)

    if op.op_type == RewriteOpType.MOVE_STATEMENT:
        call_name = op.payload.get("call_name", "")
        before = op.payload.get("before", "")
        if not call_name or not before:
            return False
        return transforms.move_statement(tree, op.target_function, call_name, before)

    logger.debug("[C.3] unknown op_type: %s", op.op_type)
    return False


def _apply_ops_via_libcst(original_source: str, operations) -> Optional[str]:
    """Apply rewrite operations directly via LibCST — no ast.unparse.

    Each RewriteOpType is dispatched to the corresponding CSTTransformer pass.
    All regions not touched by an operation keep their original formatting,
    comments, and whitespace intact.

    Returns None if any step fails (caller falls back to splice / full unparse).
    """
    module = _cst_parse(original_source)
    if module is None:
        return None

    try:
        for op in operations:
            if op.op_type == RewriteOpType.REORDER_CALLS:
                order = op.payload.get("order", [])
                if order:
                    module = _cst_reorder(module, op.target_function, order)

            elif op.op_type == RewriteOpType.REPLACE_CALL_ARGS:
                call_name = op.payload.get("call_name", "")
                new_args = op.payload.get("new_args", [])
                if call_name and new_args:
                    module = _cst_replace_args(module, op.target_function, call_name, new_args)

            elif op.op_type == RewriteOpType.REWRITE_RETURN:
                new_return = op.payload.get("new_return", "")
                if new_return:
                    module = _cst_rewrite_return(module, op.target_function, new_return)

            elif op.op_type == RewriteOpType.MOVE_STATEMENT:
                call_name = op.payload.get("call_name", "")
                before = op.payload.get("before", "")
                if call_name and before:
                    module = _cst_move(module, op.target_function, call_name, before)

            if module is None:
                return None
    except Exception as exc:
        logger.debug("[C.3/libcst] op failed: %s", exc)
        return None

    try:
        result = module.code
        ast.parse(result)
        return result
    except Exception:
        return None


def _rollback(file_path: str, backup_content: str) -> None:
    """Restore file from backup."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(backup_content)
        logger.info("[C.3] rolled back %s", os.path.basename(file_path))
    except Exception as e:
        logger.error("[C.3] rollback failed for %s: %s", file_path, e)


def _format_unparsed(source: str) -> str:
    """Add minimal formatting to ast.unparse output.

    ast.unparse produces compact code without blank lines between
    definitions. This adds blank lines before class/function defs
    and after imports for readability.
    """
    lines = source.splitlines()
    result = []
    prev_was_import = False

    for _i, line in enumerate(lines):
        stripped = line.strip()

        # Add blank line before top-level def/class (but not the first one)
        if stripped.startswith(("def ", "async def ", "class ", "@")):
            if result and result[-1].strip():
                result.append("")

        # Add blank line after import block
        if prev_was_import and not stripped.startswith(("import ", "from ")):
            if result and result[-1].strip():
                result.append("")

        result.append(line)
        prev_was_import = stripped.startswith(("import ", "from "))

    # Ensure trailing newline
    text = "\n".join(result)
    if not text.endswith("\n"):
        text += "\n"
    return text
