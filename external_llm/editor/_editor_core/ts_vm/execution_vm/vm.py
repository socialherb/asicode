"""vm.py — TS/JS Execution VM.

The VM orchestrates the full execution cycle:

    Primitive Plan
        ↓
    AST Rewrite (apply + parse check)
        ↓
    Verify (parse → tsc → eslint)
        ↓
    ╔═══════════════════╗
    ║  verify passed?   ║──yes──→ return success
    ╚═══════════════════╝
            │ no
    FailureClassifier → FailureType
        ↓
    RepairPlanner → repair ops (deterministic)
        ↓
    Re-apply + re-verify (max 3 attempts)
        ↓
    Still failing → rollback to original
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor._editor_core.ts_vm.execution_vm.ast_rewriter import ASTRewriter
from external_llm.editor._editor_core.ts_vm.execution_vm.models import VMResult
from external_llm.editor._editor_core.ts_vm.execution_vm.rollback_manager import RollbackManager
from external_llm.editor._editor_core.ts_vm.execution_vm.verifier import TSVerifier
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveOp, PrimitivePlan
from external_llm.editor._editor_core.ts_vm.repair.repair_planner import TSRepairPlanner
from external_llm.editor.semantic.ts_ir_models import TSModule
from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 3


class TSExecutionVM:
    """TS/JS Execution VM — apply, verify, repair, rollback.

    Usage::

        vm = TSExecutionVM()
        result = vm.execute(code, "file.ts", [op1, op2])
        if result.success:
            final_code = result.code
        else:
            # code is rolled back to original
            assert result.rolled_back
    """

    def __init__(
        self,
        language: str = "typescript",
        use_tsc: bool = True,
        use_eslint: bool = False,
        **kwargs,
    ):
        self._language = language
        self._rewriter = ASTRewriter(language=language)
        self._verifier = TSVerifier(
            language=language, use_tsc=use_tsc, use_eslint=use_eslint)
        self._repair_planner = TSRepairPlanner(language=language)
        self._tracer = TSSemanticTracer(language=language)

    def execute(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
        module: Optional[TSModule] = None,
    ) -> VMResult:
        """Execute primitives with full safety cycle.

        Steps:
        1. Apply primitives via AST rewriter (parse-checked)
        2. Verify result (parse + tsc + eslint)
        3. If verify fails → classify error → plan repair → re-apply
        4. If still failing → rollback to original code
        """
        rollback = RollbackManager(code)

        # ── Step 1: Apply ────────────────────────────────────────────
        apply_result = self._rewriter.apply(code, file_path, ops, module)

        if not apply_result.success:
            return VMResult(
                success=False,
                code=rollback.rollback(),
                message=f"Apply failed: {apply_result.message}",
                rolled_back=True,
            )

        current_code = apply_result.code
        rollback.push(current_code)

        # ── Defense-in-depth: no-op guard ───────────────────────────────
        # If the rewriter returned the same code as input AND there were
        # actual ops to execute, the operation had no effect.
        # Report failure so the caller can detect this rather than silently
        # writing unchanged content back to disk.
        # Empty op lists are trivially successful (nothing to do).
        if ops and current_code == code:
            return VMResult(
                success=False, code=rollback.rollback(),
                message="No-op: apply produced identical code",
                rolled_back=True,
            )

        # ── Step 2: Verify ───────────────────────────────────────────
        ok, errors = self._verifier.verify(current_code, file_path)

        if ok:
            return VMResult(
                success=True,
                code=current_code,
                message="OK",
            )

        # ── Step 3: Repair loop ──────────────────────────────────────
        repair_attempts = 0
        current_module = self._tracer.analyze_core(current_code, file_path)

        for attempt in range(MAX_REPAIR_ATTEMPTS):
            repair_attempts += 1
            logger.info(
                "Repair attempt %d/%d for %s: %d errors",
                attempt + 1, MAX_REPAIR_ATTEMPTS, file_path, len(errors),
            )

            # Plan repair
            plan = self._repair_planner.plan(
                current_code, errors, current_module)
            if plan is None:
                logger.info("No repair strategy matched, giving up")
                break

            # Apply repair
            if plan.is_raw:
                # Syntax fix: direct code replacement
                repaired = plan.raw_code
            else:
                # Semantic fix: execute repair ops via rewriter
                repair_result = self._rewriter.apply(
                    current_code, file_path, plan.ops, current_module)
                if not repair_result.success:
                    logger.info("Repair ops failed: %s", repair_result.message)
                    break
                repaired = repair_result.code

            # Re-verify
            ok, errors = self._verifier.verify(repaired, file_path)
            if ok:
                return VMResult(
                    success=True,
                    code=repaired,
                    message=(
                        f"Repaired ({plan.failure_type.value}) "
                        f"after {repair_attempts} attempt(s)"
                    ),
                    repair_attempts=repair_attempts,
                )

            current_code = repaired
            current_module = self._tracer.analyze_core(
                current_code, file_path)

        # ── Step 4: Rollback ─────────────────────────────────────────
        error_msgs = "; ".join(e.message for e in errors[:3])
        return VMResult(
            success=False,
            code=rollback.rollback(),
            message=(
                f"Verification failed after {repair_attempts} repair(s): "
                f"{error_msgs}"
            ),
            repair_attempts=repair_attempts,
            verify_errors=errors,
            rolled_back=True,
        )

    def execute_plan(
        self,
        code: str,
        file_path: str,
        plan: PrimitivePlan,
    ) -> VMResult:
        """Execute a PrimitivePlan."""
        return self.execute(code, file_path, plan.ops)
