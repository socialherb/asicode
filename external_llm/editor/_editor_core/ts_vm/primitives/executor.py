"""executor.py — TS/JS Primitive Executor (delegates to generic primitives).

This module now delegates to the language-agnostic GenericPrimitiveExecutor
from external_llm.editor.primitives. The TSPrimitiveExecutor remains for backward
compatibility, but the actual primitive logic is shared across all languages.
After each op, reparses the code to rebuild the IR so that
subsequent ops use correct byte positions.
"""
from __future__ import annotations
from external_llm.editor._editor_core.ts_vm.primitives.models import (
    PrimitiveOp,
    PrimitivePlan,
    PrimitiveResult,
)
class TSPrimitiveExecutor:
    """Executes primitive operations on TS/JS code.

    Delegates to GenericPrimitiveExecutor internally.
    Maintains backward-compatible API for ASTRewriter and TSExecutionVM.

    Usage::

        executor = TSPrimitiveExecutor()
        result = executor.execute(code, "file.ts", [op1, op2])
        if result.success:
            new_code = result.code

    The module parameter is accepted for backward compatibility but ignored
    (CodeContext is rebuilt from scratch each time, which is fast due to
    lazy initialization).
    """

    def __init__(self, language: str = "typescript"):
        self._language = language
        # Delegate to language-agnostic generic executor
        from external_llm.editor.primitives.executor import GenericPrimitiveExecutor
        self._generic_executor = GenericPrimitiveExecutor()

    def execute(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
        module: object = None,
    ) -> PrimitiveResult:
        """Execute a sequence of primitives, reparsing between each op.

        Delegates to GenericPrimitiveExecutor. The module parameter is
        accepted for backward compatibility but not used.

        Args:
            code: Source code to modify.
            file_path: File path (used for IR construction).
            ops: Ordered list of primitive operations.
            module: Ignored (backward compatibility).

        Returns:
            PrimitiveResult with the final code and aggregate status.
        """
        return self._generic_executor.execute(code, file_path, ops)

    def execute_plan(
        self, code: str, file_path: str, plan: PrimitivePlan,
    ) -> PrimitiveResult:
        """Execute a PrimitivePlan."""
        return self.execute(code, file_path, plan.ops)
