"""P7 full: Termination policy — alignment-based execution control.

Decides whether the execution loop should terminate, continue repair,
or accept partial results. Uses alignment score + hard fail gates +
plateau detection to make deterministic decisions.

Decisions:
  SUCCESS          — score >= threshold, no hard fails -> stop, accept
  PARTIAL          — score >= partial threshold -> stop, accept with warnings
  REPAIR_REQUIRED  — hard fails or low score -> continue repair loop
  STOP             — plateau detected or max iter -> stop, accept best effort

Legacy API (should_terminate / get_trend_analysis) preserved for backward compat.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Thresholds ─────────────────────────────────────────────────────────

SCORE_SUCCESS = 0.85
SCORE_PARTIAL = 0.60
PLATEAU_DELTA = 0.02
MAX_REPAIR_ITERATIONS = 2          # tightened: was 3
IMPROVEMENT_MIN = 0.05             # stop retry if improvement < this
INTENT_HARD_FLOOR = 0.3            # intent < this → FAIL
SEMANTIC_SOFT_CEILING = 0.4        # semantic < this → max REPAIR_REQUIRED


@dataclass
class TerminationDecision:
    decision: str  # SUCCESS | PARTIAL | REPAIR_REQUIRED | STOP
    reason: str
    score: float
    iteration: int

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "score": round(self.score, 4),
            "iteration": self.iteration,
        }


class TerminationPolicy:
    """Alignment-based termination + legacy checkpoint trend analysis.

    New API: decide() — alignment score driven
    Legacy API: should_terminate() / get_trend_analysis() — checkpoint driven
    """

    def __init__(
        self,
        repo_root: str = "",
        window_size: int = 5,
        improvement_threshold: float = 0.1,
        degradation_threshold: float = -0.1,
        success_threshold: float = SCORE_SUCCESS,
        partial_threshold: float = SCORE_PARTIAL,
        plateau_delta: float = PLATEAU_DELTA,
        max_iterations: int = MAX_REPAIR_ITERATIONS,
    ):
        # Legacy params
        self.repo_root = repo_root
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.degradation_threshold = degradation_threshold
        # P7 full params
        self.success_threshold = success_threshold
        self.partial_threshold = partial_threshold
        self.plateau_delta = plateau_delta
        self.max_iterations = max_iterations

    # ── P7 full: Alignment-based decision ───────────────────────────

    def decide(
        self,
        score: float,
        verification_result,  # VerificationResult
        prev_scores: list[float],
        iteration: int = 0,
        breakdown: Optional[dict[str, float]] = None,
    ) -> TerminationDecision:
        """Make termination decision based on alignment score.

        Args:
            score: Current alignment score (0~1)
            verification_result: Current verification result
            prev_scores: Alignment scores from previous iterations
            iteration: Current iteration number (0-based)
            breakdown: Optional {structural, semantic, intent} scores
        """
        _bd = breakdown or {}

        # ── Gate 0: Intent hard floor ───────────────────────────────
        _intent = _bd.get("intent", 1.0)
        if _intent < INTENT_HARD_FLOOR:
            return self._decision(
                "STOP", f"intent floor: {_intent:.2f} < {INTENT_HARD_FLOOR}",
                score, iteration,
            )

        # ── Gate 0b: Semantic soft ceiling ──────────────────────────
        _semantic = _bd.get("semantic", 1.0)
        _semantic_capped = _semantic < SEMANTIC_SOFT_CEILING

        # ── Gate 1: Hard fail checks ────────────────────────────────
        hard_fail = self._check_hard_fails(verification_result)
        if hard_fail:
            if iteration >= self.max_iterations:
                return self._decision(
                    "STOP", f"max iterations ({self.max_iterations}) with hard fail: {hard_fail}",
                    score, iteration,
                )
            return self._decision(
                "REPAIR_REQUIRED", f"hard fail: {hard_fail}", score, iteration,
            )

        # ── Gate 2: Max iterations ──────────────────────────────────
        if iteration >= self.max_iterations:
            _d = "PARTIAL" if score >= self.partial_threshold else "STOP"
            return self._decision(
                _d, f"max iterations ({self.max_iterations}), score={score:.3f}",
                score, iteration,
            )

        # ── Gate 3: Plateau / insufficient improvement ──────────────
        # Require TWO consecutive below-threshold deltas before declaring a
        # plateau. A single noisy step (e.g. 0.63 -> 0.61 -> 0.65) no longer
        # triggers premature termination; the trend must actually stall.
        if len(prev_scores) >= 3:
            _delta1 = abs(prev_scores[-1] - prev_scores[-2])
            _delta2 = abs(prev_scores[-2] - prev_scores[-3])
            if _delta1 < IMPROVEMENT_MIN and _delta2 < IMPROVEMENT_MIN:
                _last = prev_scores[-1]
                _prev = prev_scores[-3]
                _d = "PARTIAL" if score >= self.partial_threshold else "STOP"
                return self._decision(
                    _d, f"plateau: {_prev:.3f} -> {_last:.3f} (two-step deltas {_delta2:.3f},{_delta1:.3f} < {IMPROVEMENT_MIN})",
                    score, iteration,
                )
        elif len(prev_scores) == 2:
            _last = prev_scores[-1]
            _prev = prev_scores[-2]
            _delta = abs(_last - _prev)
            if _delta < IMPROVEMENT_MIN:
                _d = "PARTIAL" if score >= self.partial_threshold else "STOP"
                return self._decision(
                    _d, f"insufficient improvement: {_prev:.3f} -> {_last:.3f} (delta={_delta:.3f}<{IMPROVEMENT_MIN})",
                    score, iteration,
                )

        # ── Gate 4: Score-based ─────────────────────────────────────
        if score >= self.success_threshold and not _semantic_capped:
            return self._decision("SUCCESS", f"score {score:.3f} >= {self.success_threshold}", score, iteration)

        # Semantic capped: max status is REPAIR_REQUIRED (not SUCCESS)
        if _semantic_capped:
            if iteration == 0:
                return self._decision(
                    "REPAIR_REQUIRED",
                    f"semantic soft gate: {_semantic:.2f} < {SEMANTIC_SOFT_CEILING}, score={score:.3f}",
                    score, iteration,
                )
            return self._decision(
                "PARTIAL",
                f"semantic soft gate: {_semantic:.2f} < {SEMANTIC_SOFT_CEILING}, accepting partial",
                score, iteration,
            )

        if score >= self.partial_threshold:
            if iteration == 0 or self._has_improvement(score, prev_scores):
                return self._decision(
                    "REPAIR_REQUIRED", f"score {score:.3f} in partial range, improvement possible",
                    score, iteration,
                )
            return self._decision(
                "PARTIAL", f"score {score:.3f} >= {self.partial_threshold}, no improvement",
                score, iteration,
            )

        # Below partial threshold
        return self._decision(
            "REPAIR_REQUIRED", f"score {score:.3f} < {self.partial_threshold}", score, iteration,
        )

    def _decision(self, decision: str, reason: str, score: float, iteration: int) -> TerminationDecision:
        logger.info("termination: %s (score=%.3f iter=%d reason=%s)", decision, score, iteration, reason)
        return TerminationDecision(decision=decision, reason=reason, score=score, iteration=iteration)

    @staticmethod
    def _check_hard_fails(vr) -> Optional[str]:
        if not getattr(vr, 'syntax_ok', True):
            return "syntax_error"
        for reason in (getattr(vr, 'blocking_reasons', []) or []):
            rl = reason.lower()
            if "forbidden_token" in rl:
                return "forbidden_token"
            if "missing required symbol" in rl:
                return "required_symbol_missing"
            if "syntax error" in rl:
                return "syntax_error"
        return None

    @staticmethod
    def _has_improvement(current: float, prev_scores: list[float]) -> bool:
        if not prev_scores:
            return True
        return current > prev_scores[-1] + IMPROVEMENT_MIN

    # ── Legacy API (backward compat) ────────────────────────────────

    def should_terminate(self, recent_checkpoints) -> bool:
        """Legacy: Determine termination from checkpoint validity_score trends."""
        for checkpoint in recent_checkpoints:
            if hasattr(checkpoint, 'validity_score') and abs(checkpoint.validity_score - 1.0) < 1e-9:
                return True

        scores = []
        for checkpoint in recent_checkpoints[-self.window_size:]:
            if hasattr(checkpoint, 'validity_score'):
                scores.append(checkpoint.validity_score)
            else:
                scores.append(0.0)

        if len(scores) < 2:
            return False

        slope = self._calc_slope(scores)
        return slope >= self.improvement_threshold or slope <= self.degradation_threshold

    def get_trend_analysis(self, recent_checkpoints) -> dict[str, Any]:
        """Legacy: Analyze validity_score trends."""
        scores = []
        for checkpoint in recent_checkpoints[-self.window_size:]:
            scores.append(getattr(checkpoint, 'validity_score', 0.0))

        trend = self._calc_slope(scores) if len(scores) >= 2 else 0.0

        should_terminate = False
        reason = ""

        for checkpoint in recent_checkpoints:
            if hasattr(checkpoint, 'validity_score') and abs(checkpoint.validity_score - 1.0) < 1e-9:
                should_terminate = True
                reason = "Perfect validity_score (1.0) found"
                break

        if not should_terminate and len(scores) >= 2:
            if trend >= self.improvement_threshold:
                should_terminate = True
                reason = f"Improvement trend ({trend:.3f}) above threshold ({self.improvement_threshold})"
            elif trend <= self.degradation_threshold:
                should_terminate = True
                reason = f"Degradation trend ({trend:.3f}) below threshold ({self.degradation_threshold})"
            else:
                reason = f"Stable trend ({trend:.3f}) within thresholds [{self.degradation_threshold}, {self.improvement_threshold}]"
        elif not should_terminate:
            reason = "Insufficient data points for trend analysis"

        return {
            'scores': scores,
            'trend': trend,
            'should_terminate': should_terminate,
            'reason': reason,
        }

    @staticmethod
    def _calc_slope(scores: list[float]) -> float:
        n = len(scores)
        if n < 2:
            return 0.0
        sum_x = sum(range(n))
        sum_y = sum(scores)
        sum_xy = sum(i * s for i, s in enumerate(scores))
        sum_x2 = sum(i * i for i in range(n))
        denom = n * sum_x2 - sum_x * sum_x
        return (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0
