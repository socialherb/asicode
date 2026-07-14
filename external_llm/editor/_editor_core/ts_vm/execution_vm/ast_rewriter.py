"""ast_rewriter.py — AST-safe rewrite layer.

Wraps primitive execution with parse-verify guards:
1. Before apply: parse the target to get precise IR
2. After apply: re-parse to confirm the result is valid AST
3. If parse fails: reject the mutation (caller can rollback)

This is the safety net between raw byte-range primitives and
the verification layer. It catches structural breakage early,
before tsc/eslint even runs.
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor._editor_core.ts_vm.primitives.executor import TSPrimitiveExecutor
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveOp, PrimitiveResult
from external_llm.editor.semantic.ts_ir_models import TSModule
from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

logger = logging.getLogger(__name__)


class ASTRewriter:
    """AST-safe wrapper around TSPrimitiveExecutor.

    Ensures every mutation produces parseable code. If not,
    the mutation is rejected and the original code is returned.
    """

    def __init__(self, language: str = "typescript"):
        self._language = language
        self._tracer = TSSemanticTracer(language=language)
        self._executor = TSPrimitiveExecutor(language=language)

    def apply(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
        module: Optional[TSModule] = None,
    ) -> PrimitiveResult:
        """Apply primitives with AST parse-check after each op.

        Returns PrimitiveResult. On parse failure, returns the code
        from the last successful op (or original if first op failed).
        """
        if module is None:
            module = self._tracer.analyze_core(code, file_path)

        current_code = code
        messages: list[str] = []

        for i, op in enumerate(ops):
            # Execute single primitive
            result = self._executor.execute(
                current_code, file_path, [op], module)

            if not result.success:
                messages.append(f"[{i}] FAIL {op.kind.value}: {result.message}")
                return PrimitiveResult(
                    success=False, code=current_code,
                    message="; ".join(messages),
                )

            # AST parse check: can tree-sitter parse the result?
            new_module = self._tracer.analyze_core(result.code, file_path)
            if self._has_parse_errors(result.code, new_module):
                messages.append(
                    f"[{i}] REJECT {op.kind.value}: "
                    f"result has parse errors, reverting")
                logger.warning(
                    "AST rewrite rejected for op %d (%s): parse errors",
                    i, op.kind.value)
                return PrimitiveResult(
                    success=False, code=current_code,
                    message="; ".join(messages),
                )

            messages.append(f"[{i}] OK {op.kind.value}")
            current_code = result.code
            module = new_module

        return PrimitiveResult(
            success=True, code=current_code,
            message="; ".join(messages),
        )

    def _has_parse_errors(self, code: str, module: TSModule) -> bool:
        """Check if the parsed module indicates errors.

        tree-sitter is lenient — it produces partial trees even on errors.
        We detect errors by re-parsing and checking for ERROR nodes.
        """
        from external_llm.languages.tree_sitter_utils import is_available

        if not is_available():
            return False  # can't check, assume OK

        parser = self._tracer._get_parser()
        if parser is None:
            return False

        try:
            tree = parser.parse(code.encode("utf-8"))
            return self._tree_has_errors(tree.root_node)
        except Exception:
            return True

    def _tree_has_errors(self, node) -> bool:
        """Recursively check for ERROR or MISSING nodes in the tree."""
        if node.type == "ERROR" or node.is_missing:
            return True
        for child in node.children:
            if self._tree_has_errors(child):
                return True
        return False
