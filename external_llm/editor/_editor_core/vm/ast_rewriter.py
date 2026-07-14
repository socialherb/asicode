"""ast_rewriter.py — AST-safe rewrite layer (language-agnostic).

Wraps GenericPrimitiveExecutor with parse-verify guards:
1. Before apply: build CodeContext
2. After apply: re-parse to confirm valid AST
3. If parse fails: reject mutation (caller can rollback)
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor.primitives.code_context import CodeContext
from external_llm.editor.primitives.executor import GenericPrimitiveExecutor
from external_llm.editor.primitives.models import PrimitiveOp, PrimitiveResult

logger = logging.getLogger(__name__)


class ASTRewriter:
    """AST-safe wrapper around GenericPrimitiveExecutor.

    Ensures every mutation produces parseable code. If not,
    the mutation is rejected and the original code is returned.
    """

    def __init__(self):
        self._executor = GenericPrimitiveExecutor()

    def apply(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
        _module: object = None,
    ) -> PrimitiveResult:
        """Apply primitives with AST parse-check after each op.

        Uses tree-sitter (via CodeContext) to verify parseability.
        If tree-sitter is not available, skips parse check.

        Returns PrimitiveResult. On parse failure, returns original code.
        """
        current_code = code
        messages: list[str] = []

        for i, op in enumerate(ops):
            result = self._executor.execute(current_code, file_path, [op])

            if not result.success:
                messages.append(f"[{i}] FAIL {op.kind.value}: {result.message}")
                return PrimitiveResult(
                    success=False, code=current_code,
                    message="; ".join(messages),
                )

            # Parse check via tree-sitter (if available)
            parse_errors = _check_parse(result.code, file_path)
            if parse_errors is not None:
                if parse_errors:
                    messages.append(
                        f"[{i}] REJECT {op.kind.value}: parse errors")
                    logger.warning(
                        "AST rewrite rejected for op %d (%s): parse errors",
                        i, op.kind.value)
                    return PrimitiveResult(
                        success=False, code=current_code,
                        message="; ".join(messages),
                    )

            messages.append(f"[{i}] OK {op.kind.value}")
            current_code = result.code

        return PrimitiveResult(
            success=True, code=current_code,
            message="; ".join(messages),
        )


def _check_parse(code: str, file_path: str) -> Optional[list[VerifyError]]:
    """Check that tree-sitter can parse without ERROR nodes.

    Returns:
        None if tree-sitter unavailable.
        [] if parse is clean.
        List[VerifyError] if parse errors found.
    """
    try:
        ctx = CodeContext.from_file_path(code, file_path)
        errors = ctx.parse_errors or []
        if errors:
            return [
                VerifyError(message=e, line=0, column=0)
                for e in errors[:5]
            ]
        return []
    except Exception:
        return None  # tree-sitter not available
