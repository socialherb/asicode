"""
Strategy prioritizer: adjusts strategy ordering using experience patterns.

Boosts strategies with historical success for similar problems,
penalizes strategies with historical failures.

P11: Extended with ``unified_prioritize()`` — synthesizes learned_policy
Q-values + execution bias + failure penalty + experience patterns into
a single priority ranking.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXPERIENCE_BOOST = 0.2
EXPERIENCE_PENALTY = -0.15
MIN_CONFIDENCE_THRESHOLD = 0.3  # min success rate to apply boost


@dataclass
class PrioritizationResult:
    """Result of experience-based strategy prioritization."""
    strategies: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)
    matched_pattern_count: int = 0
    top_recommended: str = ""
    confidence: float = 0.0
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": self.strategies[:5],
            "adjustments": {k: round(v, 3) for k, v in list(self.adjustments.items())[:5]},
            "matched_pattern_count": self.matched_pattern_count,
            "top_recommended": self.top_recommended,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale[:3],
        }


class StrategyPrioritizer:
    """
    Adjusts strategy priority scores using experience patterns.

    Works additively with existing memory-aware ordering:
    1. Takes current strategy list with preference scores
    2. Finds matching experience patterns
    3. Applies boost/penalty adjustments
    4. Returns reordered list

    Never raises — returns unchanged order on error.
    """

    def prioritize(
        self,
        strategies: list[str],
        preference_scores: Optional[dict[str, float]] = None,
        problem_signature=None,
        experience_store=None,
        patterns: Optional[list] = None,
    ) -> PrioritizationResult:
        """
        Adjust strategy ordering based on experience.

        Args:
            strategies: Current strategy order
            preference_scores: Current scores from memory-aware ordering
            problem_signature: Current problem signature
            experience_store: ExperienceStore for similarity search
            patterns: Pre-extracted patterns (if available)
        """
        result = PrioritizationResult(strategies=list(strategies))
        scores = dict(preference_scores or {})

        try:
            # 1. Find matching experiences
            similar_experiences = []
            if experience_store and problem_signature:
                similar_experiences = experience_store.find_similar(problem_signature, limit=20)
                result.matched_pattern_count = len(similar_experiences)

            # 2. Extract or use provided patterns
            if patterns is None and similar_experiences:
                from external_llm.editor.learning.pattern_extractor import PatternExtractor
                extractor = PatternExtractor()
                patterns = extractor.extract_patterns(similar_experiences, min_samples=2)

            if not patterns:
                result.rationale.append("No applicable patterns found")
                return result

            # 3. Find best strategy from patterns
            sig_failure = ""
            sig_module = ""
            if problem_signature:
                if hasattr(problem_signature, 'failure_type'):
                    sig_failure = problem_signature.failure_type
                elif isinstance(problem_signature, dict):
                    sig_failure = problem_signature.get("failure_type", "")
                if hasattr(problem_signature, 'module'):
                    sig_module = problem_signature.module
                elif isinstance(problem_signature, dict):
                    sig_module = problem_signature.get("module", "")

            from external_llm.editor.learning.pattern_extractor import PatternExtractor
            extractor = PatternExtractor()
            best = extractor.get_strategy_for_context(patterns, sig_failure, sig_module)

            # Fallback: if no context-specific match, pick pattern with highest success rate
            if not best and patterns:
                top_pattern = max(patterns, key=lambda p: p.success_rate)
                if top_pattern.success_rate >= MIN_CONFIDENCE_THRESHOLD:
                    best = top_pattern.best_strategy

            if best:
                result.top_recommended = best
                # Find the pattern to get confidence
                for p in patterns:
                    if p.best_strategy == best:
                        result.confidence = p.success_rate
                        break

                result.rationale.append(f"Experience pattern suggests '{best}' (confidence={result.confidence:.2f})")

            # 4. Apply adjustments
            for pattern in patterns:
                strat = pattern.best_strategy
                if strat in scores:
                    if pattern.success_rate >= MIN_CONFIDENCE_THRESHOLD:
                        scores[strat] = scores.get(strat, 0.0) + EXPERIENCE_BOOST
                        result.adjustments[strat] = EXPERIENCE_BOOST
                        result.rationale.append(f"Boosted '{strat}' by {EXPERIENCE_BOOST} (success_rate={pattern.success_rate:.2f})")
                elif strat in strategies:
                    if pattern.success_rate >= MIN_CONFIDENCE_THRESHOLD:
                        scores[strat] = scores.get(strat, 0.0) + EXPERIENCE_BOOST
                        result.adjustments[strat] = EXPERIENCE_BOOST

                # Penalize low-success alternatives
                for alt_strat, alt_rate in pattern.alternatives:
                    if alt_rate < MIN_CONFIDENCE_THRESHOLD and alt_strat in strategies:
                        scores[alt_strat] = scores.get(alt_strat, 0.0) + EXPERIENCE_PENALTY
                        result.adjustments[alt_strat] = result.adjustments.get(alt_strat, 0.0) + EXPERIENCE_PENALTY

            # 5. Reorder by adjusted scores
            result.strategies = sorted(
                strategies,
                key=lambda s: -scores.get(s, 0.0),
            )

        except Exception as e:
            logger.debug("Strategy prioritization failed: %s", e)
            result.rationale.append(f"Prioritization failed: {str(e)[:60]}")

        return result

    # ------------------------------------------------------------------
    # P11: Unified priority synthesis
    # ------------------------------------------------------------------

    def unified_prioritize(
        self,
        strategies: list[str],
        *,
        learned_policy_scores: Optional[dict[str, float]] = None,
        strategy_policy_scores: Optional[dict[str, float]] = None,
        reward_ema: Optional[dict[str, float]] = None,
        execution_bias_by_strategy: Optional[dict[str, float]] = None,
        failure_penalty_by_strategy: Optional[dict[str, float]] = None,
        experience_adjustments: Optional[dict[str, float]] = None,
    ) -> PrioritizationResult:
        """Synthesize all learning signals into unified strategy ranking.

        Formula per strategy:
            score = learned_policy * 0.30
                  + strategy_policy * 0.20
                  + reward_ema * 0.20
                  + execution_adj * 0.15
                  + experience_adj * 0.05

        Args:
            strategies: Available strategy names.
            learned_policy_scores: Q-value scores from LearnedPolicyEngine.
            strategy_policy_scores: Q-scores from StrategyPolicyLearner.
            reward_ema: EMA reward per strategy from run_store.
            execution_bias_by_strategy: Per-strategy execution bias (net_bias) from strategy_execution_bias.
            failure_penalty_by_strategy: (deprecated, always empty) Per-strategy failure penalty.
            experience_adjustments: Experience pattern boosts/penalties.

        Returns:
            PrioritizationResult with unified ordering.
        """
        lp = learned_policy_scores or {}
        sp = strategy_policy_scores or {}
        ema = reward_ema or {}
        ebs = execution_bias_by_strategy or {}
        fps = failure_penalty_by_strategy or {}
        exp = experience_adjustments or {}

        scores: dict[str, float] = {}
        for s in strategies:
            w = (
                lp.get(s, 0.0) * 0.30
                + sp.get(s, 0.0) * 0.20
                + ema.get(s, 0.0) * 0.20
                + ebs.get(s, 0.0) * 0.15
                - fps.get(s, 0.0) * 0.10
                + exp.get(s, 0.0) * 0.05
            )
            scores[s] = round(w, 4)

        ordered = sorted(strategies, key=lambda s: -scores.get(s, 0.0))

        # Build rationale
        rationale: list[str] = []
        sources_used: list[str] = []
        if lp:
            sources_used.append("learned_policy")
        if sp:
            sources_used.append("strategy_policy")
        if ema:
            sources_used.append("reward_ema")
        if ebs:
            _ebs_summary = ", ".join(f"{k}:{v:+.2f}" for k, v in ebs.items() if abs(v) > 0.01)
            if _ebs_summary:
                sources_used.append(f"exec_bias({_ebs_summary})")
        if fps:
            _fps_summary = ", ".join(f"{k}:{v:.2f}" for k, v in fps.items() if v > 0.01)
            sources_used.append(f"failure_penalty({_fps_summary})")
        if exp:
            sources_used.append("experience")
        rationale.append(f"P11 unified: {', '.join(sources_used) or 'no signals'}")

        top = ordered[0] if ordered else ""
        conf = scores.get(top, 0.0)

        return PrioritizationResult(
            strategies=ordered,
            adjustments=scores,
            matched_pattern_count=len(exp),
            top_recommended=top,
            confidence=min(1.0, max(0.0, conf + 0.5)),  # normalize to [0, 1]
            rationale=rationale,
        )
