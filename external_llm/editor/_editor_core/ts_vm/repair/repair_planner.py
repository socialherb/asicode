"""repair_planner.py — Plans repair ops from verification errors.

Orchestrates:
1. Classify error → FailureType
2. Look up strategy → repair ops
3. Return ops (or None if no strategy matches)

The VM calls this in its repair loop.
"""
from __future__ import annotations

import logging
from typing import Optional

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveOp
from external_llm.editor._editor_core.ts_vm.repair.failure_classifier import FailureType, TSFailureClassifier
from external_llm.editor._editor_core.ts_vm.repair.repair_registry import RepairRegistry
from external_llm.editor.semantic.ts_ir_models import TSModule
from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer

logger = logging.getLogger(__name__)


class TSRepairPlanner:
    """Plans deterministic repairs from verification errors."""

    def __init__(self, language: str = "typescript"):
        self._classifier = TSFailureClassifier()
        self._registry = RepairRegistry()
        self._tracer = TSSemanticTracer(language=language)

    def plan(
        self,
        code: str,
        errors: list[VerifyError],
        module: Optional[TSModule] = None,
    ) -> Optional[RepairPlan]:
        """Analyze errors and produce a repair plan.

        Returns RepairPlan with ops, or None if no strategy matches.
        """
        if not errors:
            return None

        if module is None:
            module = self._tracer.analyze_core(code, "repair.ts")

        failure_type = self._classifier.classify(errors)
        logger.info(
            "Classified failure: %s (%d errors)",
            failure_type.value, len(errors),
        )

        strategy = self._registry.get(failure_type)
        if strategy is None:
            logger.info("No repair strategy for %s", failure_type.value)
            return None

        # Run strategy on the primary error
        ops = strategy(code, errors[0], module)
        if ops is None:
            return None

        # Check for raw code replacement (syntax repair shortcut)
        if (
            len(ops) == 1
            and "__raw_code__" in ops[0].payload
        ):
            return RepairPlan(
                failure_type=failure_type,
                ops=[],
                raw_code=ops[0].payload["__raw_code__"],
            )

        return RepairPlan(failure_type=failure_type, ops=ops)


class RepairPlan:
    """A planned repair: either primitive ops or raw code."""

    def __init__(
        self,
        failure_type: FailureType,
        ops: list[PrimitiveOp],
        raw_code: Optional[str] = None,
    ):
        self.failure_type = failure_type
        self.ops = ops
        self.raw_code = raw_code  # shortcut for syntax fixes

    @property
    def is_raw(self) -> bool:
        return self.raw_code is not None
