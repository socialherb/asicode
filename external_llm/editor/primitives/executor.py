"""executor.py — Language-agnostic Primitive Executor.

Executes a sequence of PrimitiveOps against source code using CodeContext.
After each op, rebuilds the CodeContext so subsequent ops use correct positions.
"""
from __future__ import annotations

import logging

from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.delete_node import delete_node
from external_llm.editor.primitives.insert_import import insert_import, remove_import
from external_llm.editor.primitives.insert_statement import insert_statement
from external_llm.editor.primitives.models import (
    PrimitiveKind,
    PrimitiveOp,
    PrimitivePlan,
    PrimitiveResult,
)
from external_llm.editor.primitives.registry import PrimitiveRegistry
from external_llm.editor.primitives.rename_symbol import rename_symbol
from external_llm.editor.primitives.replace_function import replace_function_body
from external_llm.editor.primitives.update_call import update_call

logger = logging.getLogger(__name__)


def _make_code_context(code: str, file_path: str) -> CodeContext:
    """Create a CodeContext from code and file path."""
    return CodeContext.from_file_path(code, file_path)


class GenericPrimitiveExecutor:
    """Executes primitive operations on code of any supported language.

    Usage::

        executor = GenericPrimitiveExecutor()
        result = executor.execute(code, "file.ts", [op1, op2])
        if result.success:
            new_code = result.code
    """

    def __init__(self):
        self._registry = PrimitiveRegistry()
        self._register_builtins()

    def _register_builtins(self) -> None:
        self._registry.register(
            PrimitiveKind.REPLACE_FUNCTION_BODY, replace_function_body)
        self._registry.register(
            PrimitiveKind.INSERT_IMPORT, insert_import)
        self._registry.register(
            PrimitiveKind.REMOVE_IMPORT, remove_import)
        self._registry.register(
            PrimitiveKind.RENAME_SYMBOL, rename_symbol)
        self._registry.register(
            PrimitiveKind.UPDATE_CALL, update_call)
        self._registry.register(
            PrimitiveKind.INSERT_STATEMENT, insert_statement)
        self._registry.register(
            PrimitiveKind.DELETE_NODE, delete_node)

    def execute(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
    ) -> PrimitiveResult:
        """Execute a sequence of primitives, rebuilding CodeContext between ops."""
        if not ops:
            return PrimitiveResult(success=True, code=code, message="No ops")

        messages: list[str] = []
        current_code = code

        for i, op in enumerate(ops):
            ctx = _make_code_context(current_code, file_path)
            result = self._registry.execute(op, ctx)

            if not result.success:
                messages.append(f"[{i}] FAIL {op.kind.value}: {result.message}")
                logger.warning(
                    "Primitive %d (%s) failed: %s",
                    i, op.kind.value, result.message,
                )
                return PrimitiveResult(
                    success=False,
                    code=current_code,
                    message="; ".join(messages),
                )

            messages.append(f"[{i}] OK {op.kind.value}: {result.message}")
            current_code = result.code

        return PrimitiveResult(
            success=True,
            code=current_code,
            message="; ".join(messages),
        )

    def execute_plan(
        self, code: str, file_path: str, plan: PrimitivePlan,
    ) -> PrimitiveResult:
        """Execute a PrimitivePlan."""
        return self.execute(code, file_path, plan.ops)
