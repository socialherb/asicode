"""vm.py — Language-agnostic Execution VM.

The VM orchestrates the full execution cycle:

    Primitive Plan
        |
    AST Rewrite (apply + parse check)
        |
    Verify (compile / javac / kotlinc / go build)
        |
    verify passed? --yes--> [contract propagation?] --> return success
        | no                             |
    FailureClassifier -> FailureType     write updated code to disk
        |                              (propagation ops also applied)
    RepairPlanner -> repair ops (deterministic)
        |
    Re-apply + re-verify (max 3 attempts)
        |
    Still failing -> rollback to original
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor._editor_core.vm.ast_rewriter import ASTRewriter
from external_llm.editor._editor_core.vm.failure_classifier import BaseFailureClassifier, create_failure_classifier
from external_llm.editor._editor_core.vm.models import VMResult
from external_llm.editor._editor_core.vm.repair_planner import RepairPlanner
from external_llm.editor._editor_core.vm.rollback_manager import RollbackManager
from external_llm.editor._editor_core.vm.verifier import BaseVerifier, create_verifier
from external_llm.editor.primitives.models import PrimitiveOp, PrimitivePlan

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 3


class GenericExecutionVM:
    """Language-agnostic Execution VM — apply, verify, repair, rollback.

    Usage::

        vm = GenericExecutionVM(language="python")
        result = vm.execute(code, "file.py", [op1, op2])
        if result.success:
            final_code = result.code
        else:
            assert result.rolled_back
    """

    def __init__(
        self,
        language: str = "python",
        verifier: Optional[BaseVerifier] = None,
        classifier: Optional[BaseFailureClassifier] = None,
        enable_propagation: bool = False,
    ):
        self._language = language
        self._rewriter = ASTRewriter()
        self._verifier = verifier or create_verifier(language)
        self._classifier = classifier or create_failure_classifier(language)
        self._repair_planner = RepairPlanner(language, self._classifier)
        self._enable_propagation = enable_propagation
        self._project_files: Optional[dict[str, str]] = None  # {path: code} for propagation

    def execute(
        self,
        code: str,
        file_path: str,
        ops: list[PrimitiveOp],
        _module: object = None,
    ) -> VMResult:
        """Execute primitives with full safety cycle.

        Steps:
        1. Apply primitives via AST rewriter (parse-checked)
        2. Verify result (compile / javac / kotlinc / go build)
        3. If verify fails -> classify error -> plan repair -> re-apply
        4. If still failing -> rollback to original code
        """
        rollback = RollbackManager(code)

        # -- Step 1: Apply ------------------------------------------------
        apply_result = self._rewriter.apply(code, file_path, ops)

        if not apply_result.success:
            return VMResult(
                success=False,
                code=rollback.rollback(),
                message=f"Apply failed: {apply_result.message}",
                rolled_back=True,
            )

        current_code = apply_result.code
        rollback.push(current_code)

        # -- Defense-in-depth: no-op guard ---------------------------------
        if ops and current_code == code:
            return VMResult(
                success=False, code=rollback.rollback(),
                message="No-op: apply produced identical code",
                rolled_back=True,
            )

        # -- Step 2: Verify ------------------------------------------------
        ok, errors = self._verifier.verify(current_code, file_path)

        if ok:
            # -- Contract propagation (opt-in) --------------------------------
            prop_files = self._propagate_contract(
                code, current_code, file_path,
                self._extract_changed_symbol(ops),
            )
            if prop_files:
                for p_path, p_code in prop_files.items():
                    p_ok, _ = self._verifier.verify(p_code, p_path)
                    if p_ok:
                        continue
                    logger.warning(
                        "Propagation verify failed for %s (non-blocking)",
                        p_path,
                    )
                return VMResult(
                    success=True,
                    code=current_code,
                    message=f"OK + propagated to {len(prop_files)} file(s)",
                    propagated_files=prop_files,
                )

            return VMResult(
                success=True,
                code=current_code,
                message="OK",
            )

        # -- Step 3: Repair loop -------------------------------------------
        repair_attempts = 0

        for attempt in range(MAX_REPAIR_ATTEMPTS):
            repair_attempts += 1
            logger.info(
                "Repair attempt %d/%d for %s: %d errors",
                attempt + 1, MAX_REPAIR_ATTEMPTS, file_path, len(errors),
            )

            # Plan repair
            plan = self._repair_planner.plan(
                current_code, errors)
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
                    current_code, file_path, plan.ops)
                if not repair_result.success:
                    logger.info("Repair ops failed: %s", repair_result.message)
                    break
                repaired = repair_result.code

            # Re-verify
            ok, errors = self._verifier.verify(repaired, file_path)
            if ok:
                # -- Contract propagation after repair (opt-in) ----------------
                prop_files = self._propagate_contract(
                    code, repaired, file_path,
                    self._extract_changed_symbol(ops),
                )
                if prop_files:
                    for p_path, p_code in prop_files.items():
                        p_ok, _ = self._verifier.verify(p_code, p_path)
                        if p_ok:
                            continue
                        logger.warning(
                            "Propagation (post-repair) verify failed for %s "
                            "(non-blocking)", p_path,
                        )
                    return VMResult(
                        success=True,
                        code=repaired,
                        message=(
                            f"Repaired ({plan.failure_type.value}) "
                            f"after {repair_attempts} attempt(s) "
                            f"+ propagated to {len(prop_files)} file(s)"
                        ),
                        repair_attempts=repair_attempts,
                        propagated_files=prop_files,
                    )

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

        # -- Step 4: Rollback ----------------------------------------------
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

    @staticmethod
    def _extract_changed_symbol(ops: list[PrimitiveOp]) -> str:
        """Extract the name of the changed symbol from the ops list.

        Returns the symbol name, or empty string if unknown.
        """
        for op in ops:
            if op.kind.value == "REPLACE_FUNCTION_BODY":
                return op.payload.get("name", "")
            if op.kind.value == "RENAME_SYMBOL":
                return op.payload.get("old_name", "")
        return ""

    def set_project_files(self, files: dict[str, str]) -> None:
        """Set the project-wide file map for contract propagation.

        Args:
            files: Mapping of file path → source code for all project files.
        """
        self._project_files = dict(files)

    def _propagate_contract(
        self,
        old_code: str,
        new_code: str,
        file_path: str,
        changed_symbol: str,
    ) -> Optional[dict[str, str]]:
        """Detect contract changes and propagate to dependent files.

        Compares old and new code to detect signature changes,
        then propagates to callers in other files.

        Args:
            old_code: Original code before modification.
            new_code: Modified code.
            file_path: Path of the modified file.
            changed_symbol: Name of the changed symbol (if known).

        Returns:
            Dict of {file_path: updated_code} for affected files,
            or None if no propagation needed.
        """
        if not self._enable_propagation or self._project_files is None:
            return None

        from external_llm.editor._editor_core.vm.contracts.contract_diff import ContractDiffer
        from external_llm.editor._editor_core.vm.contracts.contract_extractor import ContractExtractor
        from external_llm.editor._editor_core.vm.contracts.project_graph import ProjectGraphBuilder
        from external_llm.editor._editor_core.vm.contracts.propagation_planner import PropagationPlanner

        # 1. Extract old and new contracts
        extractor = ContractExtractor()
        old_contracts = extractor.extract_all(old_code, file_path)
        new_contracts = extractor.extract_all(new_code, file_path)

        if not old_contracts or not new_contracts:
            return None

        # 2. Diff contracts for the changed symbol
        differ = ContractDiffer()
        diffs = differ.diff_multi(old_contracts, new_contracts)

        # Filter to only the changed symbol
        relevant_diffs = [d for d in diffs if d.symbol == changed_symbol]
        if not relevant_diffs:
            return None

        # 3. Build project graph and plan propagation
        graph_builder = ProjectGraphBuilder()
        all_files = dict(self._project_files)

        # Update with the new code for the modified file
        all_files[file_path] = new_code

        graph = graph_builder.build(all_files)
        planner = PropagationPlanner()

        updated_files: dict[str, str] = {}
        for diff in relevant_diffs:
            plan = planner.propagate(graph, diff)
            if not plan.has_work:
                continue

            # 4. Apply propagation ops via rewriter
            for prop_op in plan.operations:
                source = all_files.get(prop_op.file_path)
                if source is None:
                    continue

                rr = self._rewriter.apply(source, prop_op.file_path, prop_op.ops)
                if rr.success:
                    updated_files[prop_op.file_path] = rr.code
                    all_files[prop_op.file_path] = rr.code

        return updated_files if updated_files else None

    def execute_plan(
        self,
        code: str,
        file_path: str,
        plan: PrimitivePlan,
    ) -> VMResult:
        """Execute a PrimitivePlan."""
        return self.execute(code, file_path, plan.ops)
