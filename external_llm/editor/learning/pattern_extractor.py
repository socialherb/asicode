"""
Pattern extractor: identifies failure→strategy patterns from historical experiences.

Deterministic rule-based extraction. No ML.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_PATTERN_SAMPLES = 3
MIN_SUCCESS_RATE = 0.5


@dataclass
class StrategyPattern:
    """A learned pattern: context → best strategy."""
    context_key: str  # e.g., "test_failure|external_llm.agent"
    best_strategy: str
    success_rate: float
    sample_count: int
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # (strategy, success_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "best_strategy": self.best_strategy,
            "success_rate": round(self.success_rate, 3),
            "sample_count": self.sample_count,
            "alternatives": [(s, round(r, 3)) for s, r in self.alternatives[:3]],
        }


class PatternExtractor:
    """
    Extracts failure→strategy patterns from experience history.

    Groups experiences by (failure_type, module) and identifies
    the most successful strategy for each group.

    Never raises — returns empty results on error.
    """

    def extract_patterns(
        self,
        experiences: list[dict[str, Any]],
        min_samples: int = MIN_PATTERN_SAMPLES,
        min_success_rate: float = MIN_SUCCESS_RATE,
    ) -> list[StrategyPattern]:
        """
        Extract strategy patterns from experience list.
        """
        patterns = []

        try:
            # Group by context key
            groups: dict[str, list[dict]] = defaultdict(list)

            for exp in experiences:
                sig = exp.get("problem_signature", {})
                failure_type = sig.get("failure_type", "")
                module = sig.get("module", "")
                strategy = exp.get("strategy_used", "")

                if not strategy:
                    continue

                # Create context keys at different granularity levels
                if failure_type and module:
                    groups[f"{failure_type}|{module}"].append(exp)
                if failure_type:
                    groups[f"{failure_type}|*"].append(exp)

            # Extract best strategy per group
            for context_key, exps in groups.items():
                if len(exps) < min_samples:
                    continue

                # Count strategy successes + alignment accumulation (P8)
                strategy_stats: dict[str, dict[str, float]] = defaultdict(
                    lambda: {"success": 0, "total": 0, "alignment_sum": 0.0, "alignment_count": 0}
                )
                for exp in exps:
                    strat = exp.get("strategy_used", "")
                    if strat:
                        strategy_stats[strat]["total"] += 1
                        if exp.get("success"):
                            strategy_stats[strat]["success"] += 1
                        # P8: accumulate alignment scores
                        _align = exp.get("alignment_score", -1.0)
                        if isinstance(_align, (int, float)) and _align >= 0:
                            strategy_stats[strat]["alignment_sum"] += _align
                            strategy_stats[strat]["alignment_count"] += 1

                # Find best strategy — P8: use alignment-weighted score when available
                best_strategy = ""
                best_score = 0.0
                alternatives = []

                for strat, stats in sorted(strategy_stats.items()):
                    rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
                    # P8: alignment-weighted score = 0.6 * success_rate + 0.4 * avg_alignment
                    _align_sum = stats.get("alignment_sum", 0.0)
                    _align_count = stats.get("alignment_count", 0)
                    avg_alignment = _align_sum / _align_count if _align_count > 0 else rate
                    score = 0.6 * rate + 0.4 * avg_alignment
                    alternatives.append((strat, rate))
                    if score > best_score or (score == best_score and stats["total"] > strategy_stats.get(best_strategy, {}).get("total", 0)):
                        best_score = score
                        best_rate = rate
                        best_strategy = strat

                if best_strategy and best_rate >= min_success_rate:
                    patterns.append(StrategyPattern(
                        context_key=context_key,
                        best_strategy=best_strategy,
                        success_rate=best_rate,
                        sample_count=len(exps),
                        alternatives=[(s, r) for s, r in alternatives if s != best_strategy],
                    ))

        except Exception as e:
            logger.debug("Pattern extraction failed: %s", e)

        return patterns

    def get_strategy_for_context(
        self,
        patterns: list[StrategyPattern],
        failure_type: str = "",
        module: str = "",
    ) -> Optional[str]:
        """
        Look up best strategy for a specific context.

        Tries specific match first (failure_type|module), then general (failure_type|*).
        """
        try:
            # Specific match first
            specific_key = f"{failure_type}|{module}"
            for p in patterns:
                if p.context_key == specific_key:
                    return p.best_strategy

            # General match
            general_key = f"{failure_type}|*"
            for p in patterns:
                if p.context_key == general_key:
                    return p.best_strategy

        except Exception as e:
            # Mirrors extract_patterns() above: log so that a failure to look
            # up a strategy never silently disables learning-based selection.
            logger.debug("get_strategy_for_context failed: %s", e)

        return None
