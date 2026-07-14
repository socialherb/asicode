"""
Risk estimation for impact analysis.

Converts impact reports into quantitative risk scores.
"""

from __future__ import annotations

from typing import Optional

from external_llm.editor.simulator.impact_models import ImpactConfig, ImpactReport


class RiskEstimator:
    """Estimates risk score from an impact report."""

    def __init__(self, config: Optional[ImpactConfig] = None):
        self.config = config or ImpactConfig()

    def estimate(self, report: ImpactReport) -> float:
        """
        Compute a risk score based on impact report.

        Scoring formula (adjustable via config):
        - direct_callers * weight_direct_callers
        - indirect_callers * weight_indirect_callers
        - dependencies * weight_dependencies
        - impacted_files * weight_impacted_files
        - missing tests penalty (if no tests affected)

        Returns:
            Risk score (0.0 ~ max_risk_score).
        """
        if not report:
            return 0.0

        # Count metrics
        n_direct = len(report.direct_callers)
        n_indirect = len(report.indirect_callers)
        n_deps = len(report.dependencies)
        n_files = len(report.impacted_files)
        has_tests = bool(report.affected_tests)

        # Compute weighted sum
        score = (
            n_direct * self.config.weight_direct_callers
            + n_indirect * self.config.weight_indirect_callers
            + n_deps * self.config.weight_dependencies
            + n_files * self.config.weight_impacted_files
        )

        # Apply missing tests penalty
        if not has_tests and self.config.include_tests:
            score += self.config.penalty_missing_tests

        # Clamp to max_risk_score
        if self.config.max_risk_score > 0:
            score = min(score, self.config.max_risk_score)

        return max(0.0, score)

    def normalize(self, score: float) -> float:
        """
        Normalize risk score to 0~1 range.

        Uses max_risk_score as upper bound.
        """
        if self.config.max_risk_score <= 0:
            return 0.0
        normalized = score / self.config.max_risk_score
        return max(0.0, min(1.0, normalized))
