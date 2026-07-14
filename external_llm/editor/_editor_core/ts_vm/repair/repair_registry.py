"""repair_registry.py — Maps FailureType → repair strategy."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveOp
from external_llm.editor._editor_core.ts_vm.repair.failure_classifier import FailureType
from external_llm.editor._editor_core.ts_vm.repair.repair_strategies import (
    repair_argument_mismatch,
    repair_missing_return,
    repair_syntax_error,
    repair_unknown_symbol,
)
from external_llm.editor.semantic.ts_ir_models import TSModule

# Strategy signature: (code, error, module) → Optional[List[PrimitiveOp]]
RepairStrategy = Callable[
    [str, VerifyError, TSModule], Optional[list[PrimitiveOp]]]


class RepairRegistry:
    """Registry mapping FailureType to deterministic repair strategies."""

    def __init__(self):
        self._strategies: dict[FailureType, RepairStrategy] = {
            FailureType.UNKNOWN_SYMBOL: repair_unknown_symbol,
            FailureType.MISSING_IMPORT: repair_unknown_symbol,
            FailureType.SYNTAX_ERROR: repair_syntax_error,
            FailureType.ARGUMENT_MISMATCH: repair_argument_mismatch,
            FailureType.MISSING_RETURN: repair_missing_return,
        }

    def register(
        self, failure_type: FailureType, strategy: RepairStrategy,
    ) -> None:
        self._strategies[failure_type] = strategy

    def get(self, failure_type: FailureType) -> Optional[RepairStrategy]:
        return self._strategies.get(failure_type)

