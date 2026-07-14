"""
Composite risk evaluation: aggregates graph, safety, and planning signals
into a single risk level for policy-driven execution.

Score-based deterministic evaluator. NOT ML-based.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.agent.execution_thresholds import THRESHOLDS

logger = logging.getLogger(__name__)


@dataclass
class CompositeRisk:
    """Aggregated risk assessment from multiple signal sources."""
    level: str = "low"  # "low" | "medium" | "high" | "critical"
    score: int = 0
    reason_codes: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    source_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "score": self.score,
            "reason_codes": self.reason_codes[:10],
            "rationale": self.rationale[:5],
        }

    def to_summary(self) -> dict[str, Any]:
        """Concise summary for metadata."""
        return {
            "level": self.level,
            "score": self.score,
            "reason_codes": self.reason_codes[:5],
        }


class CompositeRiskEvaluator:
    """
    Deterministic rule-based evaluator that computes composite risk
    from graph, safety, and planning signals.

    Never raises — returns low-risk default on any error.
    """

    def evaluate(
        self,
        graph_confidence: Optional[float] = None,
        unresolved_symbol_count: int = 0,
        resolved_symbol_count: int = 0,
        impact_file_count: int = 0,
        caller_count: int = 0,
        safety_issue_codes: Optional[list[str]] = None,
        operation_kind: Optional[str] = None,
        is_structural_op: bool = False,
        graph_enriched: bool = False,
    ) -> CompositeRisk:
        """
        Evaluate composite risk from individual signals.

        Scoring rules:
        - Low confidence + structural op: +3
        - Unresolved symbols >= 2: +2
        - Impact files >= 6: +2
        - Caller count >= 10: +2
        - Structural/rename/refactor op: +1
        - Multiple safety issues (>= 2): +2
        - Each individual safety issue: +1 (max 3)
        - Precise target + confidence >= 0.8 + low impact: -2
        - All symbols resolved + none unresolved: -1

        Levels:
        - score >= 7: critical
        - score 4-6: high
        - score 2-3: medium
        - score <= 1: low
        """
        risk = CompositeRisk()

        try:
            score = 0

            # ── Positive risk factors (increase score) ──

            # Low confidence on structural op
            if (graph_confidence is not None
                    and graph_confidence < THRESHOLDS.risk_low_confidence_structural
                    and is_structural_op):
                score += 3
                risk.reason_codes.append("LOW_CONFIDENCE_STRUCTURAL")
                risk.rationale.append(f"Low graph confidence ({graph_confidence:.2f}) on structural op")

            # Unresolved symbols
            if unresolved_symbol_count >= THRESHOLDS.graph_unresolved_symbol_count:
                score += 2
                risk.reason_codes.append("UNRESOLVED_SYMBOLS")
                risk.rationale.append(f"{unresolved_symbol_count} unresolved symbols")

            # Wide impact
            if impact_file_count >= THRESHOLDS.graph_wide_impact_count:
                score += 2
                risk.reason_codes.append("WIDE_IMPACT")
                risk.rationale.append(f"{impact_file_count} impact files")

            # High caller count
            if caller_count >= THRESHOLDS.risk_high_caller_count:
                score += 2
                risk.reason_codes.append("HIGH_CALLER_COUNT")
                risk.rationale.append(f"{caller_count} callers")

            # Structural operation
            structural_kinds = {
                "RENAME_SYMBOL", "MOVE_SYMBOL", "MODIFY_SYMBOL",
                "replace_symbol_body", "INSERT_AFTER_SYMBOL",
                "UPDATE_CALLERS", "refactor",
            }
            if is_structural_op or (operation_kind and operation_kind in structural_kinds):
                score += 1
                risk.reason_codes.append("STRUCTURAL_OP")
                risk.rationale.append(f"Structural operation: {operation_kind or 'detected'}")

            # Multiple safety issues
            issue_codes = safety_issue_codes or []
            if len(issue_codes) >= 2:
                score += 2
                risk.reason_codes.append("MULTIPLE_SAFETY_ISSUES")
                risk.rationale.append(f"{len(issue_codes)} safety issues")
            elif len(issue_codes) == 1:
                score += 1
                risk.reason_codes.append("SAFETY_ISSUE")
                risk.rationale.append(f"Safety issue: {issue_codes[0]}")

            # ── Negative risk factors (decrease score) ──

            # Precise target with high confidence and low impact
            if (graph_confidence is not None
                    and graph_confidence >= THRESHOLDS.risk_high_confidence_precise
                    and resolved_symbol_count > 0
                    and impact_file_count < THRESHOLDS.risk_low_impact_file_count):
                score -= 2
                risk.reason_codes.append("PRECISE_LOW_IMPACT")
                risk.rationale.append("High confidence, resolved targets, low impact")

            # All resolved, none unresolved
            if resolved_symbol_count > 0 and unresolved_symbol_count == 0:
                score -= 1
                risk.reason_codes.append("ALL_RESOLVED")
                risk.rationale.append("All symbols resolved")

            # Clamp score
            score = max(0, score)
            risk.score = score

            # Determine level
            if score >= THRESHOLDS.risk_critical_score:
                risk.level = "critical"
            elif score >= THRESHOLDS.risk_high_score:
                risk.level = "high"
            elif score >= THRESHOLDS.risk_medium_score:
                risk.level = "medium"
            else:
                risk.level = "low"

            # Store source signals for observability
            risk.source_signals = {
                "graph_confidence": graph_confidence,
                "unresolved_symbol_count": unresolved_symbol_count,
                "resolved_symbol_count": resolved_symbol_count,
                "impact_file_count": impact_file_count,
                "caller_count": caller_count,
                "safety_issue_count": len(issue_codes),
                "is_structural_op": is_structural_op,
                "graph_enriched": graph_enriched,
            }

        except Exception as e:
            logger.debug("Composite risk evaluation failed: %s", e)
            risk = CompositeRisk()  # safe default: low

        return risk

    def evaluate_from_metadata(
        self,
        graph_context: Optional[dict] = None,
        safety_issues: Optional[list] = None,
        operation_kind: Optional[str] = None,
        is_structural_op: bool = False,
    ) -> CompositeRisk:
        """
        Convenience method: evaluate from metadata dicts (as stored in spec/plan).
        """
        gc = graph_context or {}

        # Extract from graph_context
        graph_confidence = gc.get("graph_confidence")
        resolved = gc.get("resolved_symbols", [])
        unresolved = gc.get("unresolved_symbols", [])
        impact_files = gc.get("impact_files", [])
        callers = gc.get("callers", {})

        # Count callers
        caller_count = sum(len(v) for v in callers.values()) if isinstance(callers, dict) else 0

        # Extract safety issue codes
        issue_codes = []
        if safety_issues:
            for issue in safety_issues:
                code = issue.get("code", "") if isinstance(issue, dict) else getattr(issue, "code", "")
                if code:
                    issue_codes.append(code)

        return self.evaluate(
            graph_confidence=graph_confidence,
            unresolved_symbol_count=len(unresolved),
            resolved_symbol_count=len(resolved),
            impact_file_count=len(impact_files),
            caller_count=caller_count,
            safety_issue_codes=issue_codes,
            operation_kind=operation_kind,
            is_structural_op=is_structural_op,
            graph_enriched=bool(gc),
        )
