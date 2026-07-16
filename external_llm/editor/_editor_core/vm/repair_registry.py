"""repair_registry.py — Maps FailureType -> repair strategy per language."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from external_llm.editor._editor_core.vm.classification import Classification
from external_llm.editor._editor_core.vm.failure_classifier import FailureType
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.repair_strategies import get_strategies
from external_llm.editor.primitives.models import PrimitiveOp

# Strategy signature: (code, error, classification) -> Optional[List[PrimitiveOp]]
# Phase 3: Changed from (code, error, classifier) to (code, error, classification)
# to use structured symbol extraction from Classification.
RepairStrategyFn = Callable[
    [str, VerifyError, Classification], Optional[list[PrimitiveOp]]]


class RepairRegistry:
    """Registry mapping FailureType to deterministic repair strategies."""

    def __init__(self, language: str):
        strategies = get_strategies(language)
        self._strategies: dict[FailureType, RepairStrategyFn] = dict(strategies)

    def register(
        self, failure_type: FailureType, strategy: RepairStrategyFn,
    ) -> None:
        self._strategies[failure_type] = strategy

    def get(self, failure_type: FailureType) -> Optional[RepairStrategyFn]:
        return self._strategies.get(failure_type)

