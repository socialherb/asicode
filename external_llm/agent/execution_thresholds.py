"""execution_thresholds.py — Centralized execution thresholds.

Single source of truth for all numeric thresholds used in the execution pipeline.
Prevents magic numbers from scattering across files and provides a clear extension
point for learning-based threshold adaptation.

Naming convention: CATEGORY_DESCRIPTION (e.g., LINEAGE_HARD_REJECT_RATIO)

Learning integration:
  - ``update_thresholds(**overrides)`` mutates the global singleton in-place.
  - ``reset_thresholds()`` reverts to factory defaults in-place.
  - ``ThresholdAdapter`` collects per-decision observations and applies EMA-based
    adaptation when enough evidence accumulates.

Singleton design:
  THRESHOLDS is a *mutable* singleton. Modules that do
  ``from .execution_thresholds import THRESHOLDS`` hold a reference to the same
  object. In-place mutations from ``update_thresholds()`` are immediately visible
  to all importers within the same process without any re-import.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionThresholds:
    """All execution pipeline thresholds in one place.

    These are defaults. Learning systems can override via ``update_thresholds()``.
    """

    # ── Lineage Verification ─────────────────────────────────────────
    lineage_hard_reject_ratio: float = 0.70  # ratio < this → rollback
    lineage_soft_downgrade_ratio: float = 0.85  # ratio < this + warnings → partial

    # ── Semantic Verification ────────────────────────────────────────
    semantic_pass_score: float = 0.40  # semantic score >= this → pass

    # ── Symbol/File Size ─────────────────────────────────────────────
    large_symbol_lines: int = 300  # > this → anchor_edit strategy
    focused_edit_file_lines: int = 500  # > this → focused edit mode

    # ── Strategy Selection ───────────────────────────────────────────
    strategy_confidence_gate: float = 0.80  # deterministic early exit
    strategy_confidence_floor: float = 0.50  # minimum adaptive gate

    # ── Candidate Ranking ────────────────────────────────────────────
    cold_start_temperature: float = 0.25  # < 3 recent runs
    high_failure_temperature: float = 0.60  # failure_rate > 0.3
    moderate_failure_temperature: float = 0.30  # failure_rate > 0.1
    stable_temperature: float = 0.10  # failure_rate <= 0.1
    temperature_base_weight: float = 0.60  # base vs policy blend
    # ── Quality Gate (correctness-before-strategy) ───────────────────
    # A candidate must meet BOTH thresholds to receive strategy/parsimony
    quality_gate_spec_alignment: float = 0.25    # spec_alignment floor (lowered from 0.45: simple edits score 0.30-0.50, 0.45 gate caused all to fall through to bypass)
    quality_gate_contract_satisfaction: float = 0.30  # contract_satisfaction floor
    quality_gate_spec_alignment_hard_block: float = 0.05  # spec_alignment < this → hard block (no fallback; 0.0 ≈ hallucination)
    changespec_missing_penalty: float = 0.25  # typed items expected but absent; no longer needs to beat gate (spec_alignment proxy handles it)

    # ── Reward Shaping ───────────────────────────────────────────────
    reward_replan_penalty: float = 0.30  # per replan
    reward_repair_penalty: float = 0.15  # per repair round
    reward_success_bonus: float = 0.80  # SUCCESS termination
    reward_stop_penalty: float = 1.00  # STOP termination
    reward_no_diff_penalty: float = 0.80  # no changes produced

    # ── Surgical Edit ────────────────────────────────────────────────
    surgical_similarity_threshold: float = 0.85  # SequenceMatcher ratio
    caller_count_surgical_gate: int = 5  # > this → surgical

    # ── Token Analysis ───────────────────────────────────────────────
    token_ratio_gate: float = 0.75  # fix specification match ratio
    min_analysis_tokens: int = 5  # minimum tokens for analysis

    # ── DPB: Intent-Explicit Pair Similarity Gate ────────────────────
    # Used in the intent-explicit pair path (_try_paired_local_patch): when the
    synthetic_pair_min_similarity: float = 0.50

    # ── Graph Planning Policy ─────────────────────────────────────────
    graph_high_confidence: float = 0.80    # "trusted" mode — proceed with symbol-focused strategy
    graph_low_confidence: float = 0.40     # "conservative" mode — avoid precise targeting
    graph_wide_impact_count: int = 6       # impact file count >= this → prefer decomposition
    graph_unresolved_symbol_count: int = 2 # unresolved symbols >= this → reduce scope

    # ── Structural Seed Protection (SSOT) ────────────────────────────
    # Single source of truth for deterministic structural seed confidence threshold.
    # Used by: deterministic_plan_builder, planner_helpers, strategy_router.
    structural_seed_protect_confidence: float = 0.88

    # ── Exploration Engine ────────────────────────────────────────────
    #confidence value ConfidenceScorer.calibrate() output criteria (calibrated scale).
    exploration_convergence_confidence: float = 0.75  # calibrated conf >= this → converged
    exploration_clear_winner_gap: float = 0.25         # score gap top-1 vs top-2 >= this → clear winner
    exploration_pool_narrow_conf: float = 0.65         # calibrated conf above this → min pool size
    exploration_pool_wide_conf: float = 0.35           # calibrated conf below this → max pool size
    exploration_pool_min_size: int = 12                # pool size at high confidence (focused)
    exploration_pool_max_size: int = 28                # pool size at low confidence (wide net)

    # ── Exploration Engine: plateau early-stop ────────────────────────
    # "Low and not improving" = a normal early-stop signal.
    # A separate hard floor exits regardless of delta when confidence stays very low,
    # but min_depth protects cold starts.
    exploration_low_conf_floor: float = 0.35     # conf below this is considered "low"
    exploration_min_delta: float = 0.03          # min per-depth confidence gain to avoid plateau (depth 1)
    exploration_stuck_delta_cap: float = 0.08    # delta cap at depth >= 2 when conf < floor.
                                                 # Covers "slow-creeping" cases where delta is
                                                 # positive but confidence stays persistently low.
    exploration_abort_low_conf_hard: float = 0.25  # abort below this after min depth regardless of delta
    exploration_abort_low_conf_min_depth: int = 2  # depth≥2 — pre-expansion gate at depth 1 covers cold-start, plateau gate gives one trial
    exploration_low_conf_persistent_margin: float = 0.05  # abort if confidence never rose > this from its minimum across depths

    # ── Candidate Ranking: Failure-Rate Temperature Bands ─────────────
    temperature_high_failure_rate: float = 0.30   # failure_rate above this → high_failure_temperature
    temperature_moderate_failure_rate: float = 0.10  # failure_rate above this → moderate_failure_temperature

    # ── Composite Risk Scoring Bands ──────────────────────────────────
    risk_critical_score: int = 7   # score >= this → "critical"
    risk_high_score: int = 4       # score >= this → "high"
    risk_medium_score: int = 2     # score >= this → "medium"
    risk_low_confidence_structural: float = 0.40  # graph_confidence < this AND structural → +3 risk
    risk_high_confidence_precise: float = 0.80    # graph_confidence >= this + precise target → -2 risk
    risk_high_caller_count: int = 10  # caller_count >= this → +2 risk
    risk_low_impact_file_count: int = 3  # impact_file_count < this → precise target condition

    # ── Closure Check Graph Query Budget ─────────────────────────────
    # Max fresh graph queries (cache misses) per _normalize_fixspec_closure_check call.
    # Cache hits are free. Prevents cost explosion when uncovered target count is large.
    # Sources 3a/3b/3c all share this budget; cache accumulates at executor lifetime.
    closure_graph_query_budget: int = 30

    # ── Executor Gate ───────────────────────────────────────────────
    executor_gc_suppression_threshold: float = 0.40
    # grounding_confidence below this → suppress per-op gate skip

    # ── Seed Bonus ──────────────────────────────────────────────────
    seed_bonus_confidence_floor: float = 0.60
    # structural seed conf below this → lose bonus


    # ── Paired Extract Gate (det_plan_extract) ──────────────────────
    paired_extract_similarity: float = 0.82    # similarity < this → not extractable
    paired_extract_exit_sim: float = 0.40      # exit_sim < this → not extractable
    paired_extract_call_overlap: float = 0.25  # call_overlap < this → not extractable
    paired_param_sim_gate: float = 0.30        # param_sim < this → skip pair



    # ── ThresholdAdapter Learning Parameters ─────────────────────────
    adapter_min_observations: int = 10   # min obs before first adaptation (was 20)
    adapter_alpha: float = 0.10          # EMA adaptation rate (slow = stable)
    adapter_correct_rate_gate: float = 0.80  # adapt if accuracy below this (was 0.85)


# ── Global mutable singleton ─────────────────────────────────────────────────
# Learning system can replace via update_thresholds().

THRESHOLDS = ExecutionThresholds()


def update_thresholds(**overrides) -> None:
    """Mutate global THRESHOLDS singleton in-place with learned values.

    Because THRESHOLDS is mutated rather than replaced, all modules that have
    already imported it (``from X import THRESHOLDS``) see the change immediately
    without re-importing.

    Only valid ExecutionThresholds fields are applied; unknown keys are silently
    ignored.
    """
    valid = {k: v for k, v in overrides.items() if hasattr(ExecutionThresholds, k)}
    if valid:
        for k, v in valid.items():
            setattr(THRESHOLDS, k, v)
        logger.info("[THRESHOLDS] Updated %d fields: %s", len(valid), list(valid.keys()))


def reset_thresholds() -> None:
    """Reset to factory defaults in-place (singleton object is preserved)."""
    defaults = ExecutionThresholds()
    for field in dataclasses.fields(defaults):
        setattr(THRESHOLDS, field.name, getattr(defaults, field.name))
    logger.info("[THRESHOLDS] Reset to factory defaults")


# ── ThresholdAdapter — observation-based EMA adaptation ──────────────────────


class ThresholdAdapter:
    """Adapts thresholds from execution outcomes using EMA.

    Principle: if a threshold is causing too many incorrect decisions
    (rejections that get overridden downstream, or passes that later fail),
    gradually shift it toward the empirical optimal boundary.

    Usage::

        THRESHOLD_ADAPTER.observe("lineage_hard_reject_ratio", 0.65, was_correct=True)
        # ... after enough observations, adapter auto-adjusts the threshold

    The adapter only modifies thresholds that have:
      1. Enough observations (>= THRESHOLDS.adapter_min_observations)
      2. Poor accuracy (< THRESHOLDS.adapter_correct_rate_gate)

    Adaptation speed is controlled by THRESHOLDS.adapter_alpha (EMA rate).
    All three parameters are themselves tunable via update_thresholds().
    """

    def __init__(self) -> None:
        self._observations: dict[str, list[tuple[float, bool]]] = {}

    def observe(self, threshold_name: str, value_at_decision: float, was_correct: bool) -> None:
        """Record whether a threshold-based decision was correct.

        Args:
            threshold_name: Name of the ExecutionThresholds field (e.g., "lineage_hard_reject_ratio").
            value_at_decision: The actual metric value when the decision was made.
            was_correct: Whether the threshold-based decision turned out to be correct.
        """
        if not hasattr(ExecutionThresholds, threshold_name):
            return  # Unknown field — skip silently

        if threshold_name not in self._observations:
            self._observations[threshold_name] = []
        self._observations[threshold_name].append((value_at_decision, was_correct))

        # Adapt when a full batch has accumulated, then clear for next batch.
        # Clearing (rather than sliding-window prune) ensures each adaptation
        # uses a clean, independent batch and the trigger rate matches
        # adapter_min_observations exactly — not every single new observation.
        obs = self._observations[threshold_name]
        if len(obs) >= THRESHOLDS.adapter_min_observations:
            self._adapt(threshold_name, obs)
            self._observations[threshold_name] = []

    def _adapt(self, name: str, observations: list[tuple[float, bool]]) -> None:
        """EMA-based threshold adjustment from observation history."""
        correct_vals = [v for v, ok in observations if ok]
        incorrect_vals = [v for v, ok in observations if not ok]

        if not incorrect_vals:
            return  # All correct — no adjustment needed

        correct_rate = len(correct_vals) / len(observations)
        if correct_rate >= THRESHOLDS.adapter_correct_rate_gate:
            return  # Threshold working well enough

        # Compute suggested boundary between correct and incorrect value distributions
        avg_correct = sum(correct_vals) / len(correct_vals) if correct_vals else 0
        avg_incorrect = sum(incorrect_vals) / len(incorrect_vals) if incorrect_vals else 0
        suggested = (avg_correct + avg_incorrect) / 2

        current = getattr(THRESHOLDS, name, None)
        if current is None:
            return

        # EMA blend: mostly keep current, slowly move toward suggested
        alpha = THRESHOLDS.adapter_alpha
        new_val = round(current * (1 - alpha) + suggested * alpha, 4)
        update_thresholds(**{name: new_val})
        logger.info(
            "[THRESHOLD_ADAPTER] %s: %.4f -> %.4f (correct_rate=%.2f, n=%d)",
            name, current, new_val, correct_rate, len(observations),
        )

    def reset(self) -> None:
        """Clear all observation history."""
        self._observations.clear()

    @property
    def observation_counts(self) -> dict[str, int]:
        """Current observation counts per threshold (for diagnostics)."""
        return {k: len(v) for k, v in self._observations.items()}


# Global adapter instance — import and call observe() from decision points
THRESHOLD_ADAPTER = ThresholdAdapter()
