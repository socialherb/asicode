"""output_comparator.py — Phase F: Raw vs Reconstructed Candidate Comparison.

Compares the raw LLM output state with the reconstructed candidate
and decides which to adopt. Uses primitive coverage as primary signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from external_llm.editor.semantic.primitive_models import PrimitiveIR, ReconstructionCandidate

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing raw vs reconstructed candidates."""
    chosen: str = "raw"           # "raw" | "reconstructed"
    reason: str = ""
    raw_coverage: float = 0.0
    reconstructed_coverage: float = 0.0
    raw_contract_score: float = 0.0
    reconstructed_contract_estimate: float = 0.0
    improvement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "reason": self.reason,
            "raw_coverage": round(self.raw_coverage, 4),
            "reconstructed_coverage": round(self.reconstructed_coverage, 4),
            "improvement": round(self.improvement, 4),
        }


def compare_candidates(
    raw_ir: PrimitiveIR,
    candidate: ReconstructionCandidate,
    contract_score: float = 0.0,
) -> ComparisonResult:
    """Compare raw LLM output with reconstruction candidate.

    Decision criteria:
    1. If candidate fills no new primitives → keep raw
    2. If candidate improves coverage by >= 0.1 → use reconstructed
    3. If candidate has no patches/fragments → keep raw
    4. Otherwise → keep raw (conservative)
    """
    result = ComparisonResult(
        raw_coverage=raw_ir.overall_coverage,
        reconstructed_coverage=candidate.primitive_coverage_estimate,
        raw_contract_score=contract_score,
    )

    # No reconstruction attempted or no patches produced
    if not candidate.applied_primitives:
        result.chosen = "raw"
        result.reason = "no reconstruction patches"
        return result

    improvement = candidate.primitive_coverage_estimate - raw_ir.overall_coverage
    result.improvement = improvement

    # Estimate contract score improvement
    # Heuristic: each filled primitive improves contract by ~0.05
    result.reconstructed_contract_estimate = min(
        1.0,
        contract_score + len(candidate.applied_primitives) * 0.05,
    )

    # Decision
    if improvement >= 0.1:
        result.chosen = "reconstructed"
        result.reason = f"coverage +{improvement:.2f} ({len(candidate.applied_primitives)} primitives filled)"
    elif improvement > 0 and len(candidate.applied_primitives) >= 2:
        result.chosen = "reconstructed"
        result.reason = f"multiple primitives filled ({candidate.applied_primitives})"
    else:
        result.chosen = "raw"
        result.reason = f"improvement too small ({improvement:.2f})"

    logger.info(
        "[COMPARE] chosen=%s reason=%s coverage=%.2f→%.2f",
        result.chosen, result.reason,
        result.raw_coverage, result.reconstructed_coverage,
    )

    return result
