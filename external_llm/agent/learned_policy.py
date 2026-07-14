"""learned_policy.py — Phase 6: Learned Policy Engine.

Unified module that:
- Defines enriched state features for strategy selection
- Computes shaped rewards with strong contrast
- Manages Q-table with state-action pairs
- Implements epsilon-greedy exploration with decay
- Repetition suppression (generic_create dominance prevention)
- Distillation → scoring integration (not just override)
- Full logging for observability

Architecture
------------
1. ``compute_state``        — build structured state from spec + context
2. ``compute_shaped_reward`` — enhanced reward with strong contrast
3. ``update``               — Q-table + stats update after execution
4. ``score_strategies``     — compute learned scores for all candidates
5. ``select_with_exploration`` — epsilon-greedy selection

This module is the SINGLE source of learned policy decisions.
All P8/P9/P11/P12/P13/P14 scattered boosts are consolidated here.
"""
from __future__ import annotations

import logging
import math
import random

# (re module removed — intent_mode uses structured lookup instead of regex)
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from external_llm.agent.operation_models import FailureClass, normalize_failure_class

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Q-learning parameters
# (Q_GAMMA removed — update step uses a bandit-style update without a
#  next-state discount term; see update() Q_ALPHA branch.)
Q_ALPHA = 0.15           # Learning rate (higher than prev 0.1 for faster adaptation)
Q_CLIP = 3.0             # Q-value clip range [-3, +3]

# Exploration parameters
EPSILON_INITIAL = 0.30   # Initial exploration rate
EPSILON_DECAY = 0.97     # Per-episode decay
EPSILON_FLOOR = 0.05     # Minimum exploration rate

# Repetition suppression
REP_WINDOW = 10          # Recent selection window
REP_PENALTY_BASE = 0.15  # Base penalty per repeat
REP_PENALTY_MAX = 0.60   # Max repetition penalty
REP_FAILURE_MULT = 2.0   # Multiplier when repeated strategy also failed

# Score scaling — learned signals must be able to flip heuristic
LEARNED_SCORE_SCALE = 1.5  # Multiplier for learned component
DISTILL_BOOST = 0.40       # Boost for distillation-recommended strategy
DISTILL_PENALTY = -0.15    # Penalty for distillation-discouraged strategies

# Reward shaping constants
REWARD_SUCCESS_BASE = 2.0
REWARD_FAILURE_BASE = -2.0
REWARD_FAST_SUCCESS_BONUS = 0.8   # Success with 0 repairs
REWARD_SEMANTIC_BONUS = 0.5       # Semantic verification passed

# ── Structured intent-mode lookup (replaces regex matching) ──────────────
# request_type is a structured field set by the planner. These sets replace
# the old word-boundary regexes (\bcreate\b etc.) with exact match against
# known request_type values set across the codebase.
_INTENT_MODE_MAP: dict[str, str] = {
    "create": "create",
    "product_create": "create",
    "extend": "extend",
    "product_extend": "extend",
    "add": "extend",
    "feature": "feature",
    "product_feature": "feature",
    "improve": "improve",
    "product_improve": "improve",
    "refactor": "improve",
}

# Import-related intent keywords (mapped to "integration" issue_type)
_IMPORT_INTENT_TOKENS: frozenset = frozenset({"import", "wiring"})
REWARD_REPAIR_PENALTY_PER = -0.4  # Per repair round
REWARD_REPAIR_MAX_PENALTY = -2.0  # Cap on repair penalty
REWARD_OVERSIZED_DIFF = -0.3      # Diff > 200 lines
REWARD_REPEATED_FAILURE = -0.6    # Same strategy failed again
REWARD_NO_DIFF = -1.5             # No diff produced
REWARD_RANGE = (-3.0, 4.0)        # Total reward range


# ── State Definition ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyState:
    """Structured state for strategy selection.

    Designed to be small but discriminative.
    Hashable for use as Q-table key.
    """
    intent_mode: str       # "create" | "extend" | "feature" | "improve" | "modify" | "unknown"
    task_family: str       # "auth" | "video" | "chat" | "api" | "general" | ...
    issue_type: str        # "import_error" | "integration" | "symbol_edit" | "test" | "none"
    complexity: str        # "simple" | "medium" | "complex"
    has_references: bool   # Whether reference files are available
    has_symbols: bool      # Whether target symbols are specified

    def to_key(self) -> str:
        """Compact string key for Q-table lookup."""
        refs = "R" if self.has_references else "r"
        syms = "S" if self.has_symbols else "s"
        return f"{self.intent_mode}|{self.task_family}|{self.issue_type}|{self.complexity}|{refs}{syms}"

    @staticmethod
    def from_key(key: str) -> "PolicyState":
        """Parse a key back to PolicyState."""
        parts = key.split("|")
        if len(parts) != 5:
            return PolicyState("unknown", "general", "none", "medium", False, False)
        flags = parts[4]
        return PolicyState(
            intent_mode=parts[0],
            task_family=parts[1],
            issue_type=parts[2],
            complexity=parts[3],
            has_references="R" in flags,
            has_symbols="S" in flags,
        )


def compute_state(
    spec: Any = None,
    context_info: Optional[dict[str, Any]] = None,
    failure_class: str = "",
) -> PolicyState:
    """Build a PolicyState from spec and context.

    Extracts meaningful features without being overly granular.
    """
    intent_mode = "unknown"
    task_family = "general"
    issue_type = "none"
    complexity = "medium"
    has_references = False
    has_symbols = False

    if spec is not None:
        # Intent mode from structured spec fields (not free-text keywords)
        _req_type = str(getattr(spec, "request_type", "") or "").lower()
        _intent_str = str(getattr(spec, "intent", "") or "").lower()
        _meta = getattr(spec, "metadata", None) or {}

        # Structured lookup replaces regex matching — request_type is a
        # structured field, not free text, so exact/keyword match suffices.
        intent_mode = _INTENT_MODE_MAP.get(_req_type, "unknown")

        # Task family from metadata or structural scope
        _product_type = _meta.get("product_type", "")
        if _product_type:
            task_family = _product_type
        else:
            task_family = "general"

        # Issue type from failure class — prefer structural FailureClass enum
        # over substring matching to avoid false positives.
        if failure_class:
            _fc_norm = normalize_failure_class(failure_class)
            if _fc_norm in (
                FailureClass.INVALID_IMPORT_STMT,
                FailureClass.INVALID_IMPORT_MODULE_PATH,
                FailureClass.MODULE_IMPORT_GATE,
                FailureClass.F821_UNDEFINED_NAME,
                FailureClass.UNDEFINED_NAME,
                FailureClass.NAME_REFERENCE_ERROR,
            ):
                issue_type = "import_error"
            elif _fc_norm in (
                FailureClass.ANCHOR_MISS,
                FailureClass.ANCHOR_LOSS,
               FailureClass.ANCHOR_MULTILINE_PATTERN,
                FailureClass.SYMBOL_NOT_FOUND,
                FailureClass.TARGET_NOT_FOUND,
            ):
                issue_type = "symbol_edit"
            elif _fc_norm in (
                FailureClass.SEMANTIC_VERIFICATION_FAILED,
                FailureClass.SEMANTIC_VERIFY_FAILED,
                FailureClass.VERIFICATION_FAILED,
                FailureClass.INTENT_ASSERTION_FAILED,
                FailureClass.PRESUPPOSITION_VIOLATED,
                FailureClass.PLACEMENT_VIOLATION,
            ):
                issue_type = "semantic"
            elif _fc_norm in (
                FailureClass.LINT_ERROR,
                FailureClass.SYNTAX_ERROR,
                FailureClass.SYNTAX_ERROR_AFTER_PATCH,
                FailureClass.SYNTAX_INVALID_AFTER_EDIT,
            ):
                issue_type = "syntax"
            elif _fc_norm == FailureClass.UNKNOWN:
                # Fallback to substring check for unknown/passthrough values
                _fc_raw = failure_class.lower()
                if "test" in _fc_raw:
                    issue_type = "test"
                else:
                    issue_type = failure_class[:20]
            else:
                issue_type = failure_class[:20]
        elif any(tok in _intent_str for tok in _IMPORT_INTENT_TOKENS):
            issue_type = "integration"

        # Complexity
        _scope = str(getattr(spec, "estimated_scope", "") or "").lower()
        _new_files = getattr(spec, "new_files", None) or []
        _target_files = getattr(spec, "target_files", None) or []
        total_files = len(set(_new_files) | set(_target_files))
        if _scope == "large" or total_files >= 5:
            complexity = "complex"
        elif _scope == "small" or total_files <= 1:
            complexity = "simple"
        else:
            complexity = "medium"

        # References / symbols
        has_references = bool(getattr(spec, "reference_files", None))
        has_symbols = bool(getattr(spec, "target_symbols", None))

    return PolicyState(
        intent_mode=intent_mode,
        task_family=task_family,
        issue_type=issue_type,
        complexity=complexity,
        has_references=has_references,
        has_symbols=has_symbols,
    )


# ── Reward Shaping ────────────────────────────────────────────────────────────

def compute_shaped_reward(
    success: bool,
    repair_rounds: int = 0,
    replan_count: int = 0,
    semantic_pass: bool = True,
    has_diff: bool = True,
    diff_lines: int = 0,
    fast_success: bool = False,
    is_repeated_failure: bool = False,
    alignment_score: float = 0.0,
    termination: str = "",
    # P11: Policy trace features
    diversity_mean: float = 0.0,
    no_effective_progress: bool = False,
    runtime_gate_failed: bool = False,
    # Phase 3: Test & token signals
    test_pass_count: int = 0,
    test_fail_count: int = 0,
    test_was_run: bool = False,
    total_tokens: int = 0,
) -> float:
    """Compute a strongly-contrasted reward signal.

    Design: success vs failure must produce VISIBLY different rewards.
    A perfect success ≈ +3.3, a total failure ≈ -3.0.

    """
    reward = 0.0

    # Base outcome
    if success:
        reward += REWARD_SUCCESS_BASE
    else:
        reward += REWARD_FAILURE_BASE

    # Fast success bonus (0 repairs needed)
    if success and fast_success and repair_rounds == 0:
        reward += REWARD_FAST_SUCCESS_BONUS

    # Semantic verification (binary)
    if success and semantic_pass:
        reward += REWARD_SEMANTIC_BONUS
    elif not semantic_pass:
        reward -= 0.3

    # Repair rounds penalty (diminishing)
    if repair_rounds > 0:
        repair_pen = max(REWARD_REPAIR_MAX_PENALTY,
                         REWARD_REPAIR_PENALTY_PER * min(repair_rounds, 5))
        reward += repair_pen

    # Replan penalty
    if replan_count > 0:
        reward -= 0.3 * min(replan_count, 3)

    # No diff = wasted execution
    if not has_diff:
        reward += REWARD_NO_DIFF

    # Oversized diff penalty
    if diff_lines > 200:
        reward += REWARD_OVERSIZED_DIFF

    # Repeated failure on same strategy
    if is_repeated_failure:
        reward += REWARD_REPEATED_FAILURE

    # Alignment bonus (P8 compat)
    if alignment_score > 0:
        reward += 0.3 * min(alignment_score, 1.0)

    # Termination type refinement
    if termination == "SUCCESS":
        reward += 0.2
    elif termination == "STOP":
        reward -= 0.4

    # First-pass success bonus
    if success and repair_rounds == 0:
        reward += 0.5

    # P11: Policy trace reward terms
    # Diversity: good diversity among candidates → slight bonus
    if diversity_mean >= 0.35:
        reward += 0.15
    # No effective progress (operations produced no meaningful changes) → penalty
    if no_effective_progress:
        reward -= 0.5
    # Runtime gate failure (compile/lint gate before execution) → penalty
    if runtime_gate_failed:
        reward -= 0.4

    # Phase 3: Test-based reward signal
    if test_was_run:
        if test_fail_count == 0 and test_pass_count > 0:
            reward += 1.0   # All tests pass — strong bonus
        elif test_fail_count > 0:
            _total = max(test_pass_count + test_fail_count, 1)
            _fail_ratio = test_fail_count / _total
            reward -= 0.5 + (_fail_ratio * 1.0)  # -0.5 to -1.5

    # Phase 3: Token efficiency signal
    if total_tokens > 0:
        if total_tokens > 50000:
            reward -= 0.2   # Expensive execution penalty
        elif total_tokens < 10000 and success:
            reward += 0.1   # Efficient execution bonus

    return round(max(REWARD_RANGE[0], min(REWARD_RANGE[1], reward)), 4)


# ── Learned Policy Engine ─────────────────────────────────────────────────────

class LearnedPolicyEngine:
    """Unified learned policy engine for strategy selection.

    Consolidates Q-table, EMA, distillation, exploration, and repetition
    suppression into a single coherent policy.

    Thread-safety: NOT thread-safe. Expected to be used per-run_store instance.
    """

    def __init__(self) -> None:
        # Q-table: state_key → {strategy: q_value}
        self._q_table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Visit counts: state_key → {strategy: count}
        self._visit_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_visits: int = 0

        # EMA per (state, strategy)
        self._state_ema: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Global strategy stats
        self._strategy_successes: dict[str, int] = defaultdict(int)
        self._strategy_failures: dict[str, int] = defaultdict(int)
        self._strategy_total: dict[str, int] = defaultdict(int)

        # Recent selection history for repetition suppression
        self._recent_selections: list[tuple[str, str, bool]] = []  # (state_key, strategy, success)

        # Distilled rules: context_pattern → {strategy, confidence, count}
        self._distilled_rules: dict[str, dict[str, Any]] = {}

        # Exploration state
        self._epsilon: float = EPSILON_INITIAL
        self._episode_count: int = 0

        # Failure tracking per (state, strategy)
        self._recent_failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ── Q-table operations ────────────────────────────────────────────────

    def get_q_value(self, state: PolicyState, strategy: str) -> float:
        """Get Q-value for (state, strategy) pair."""
        return self._q_table[state.to_key()].get(strategy, 0.0)

    def update(
        self,
        state: PolicyState,
        strategy: str,
        reward: float,
        success: bool,
    ) -> dict[str, Any]:
        """Update Q-table and all stats after execution.

        Returns dict with before/after values for logging.
        """
        key = state.to_key()
        old_q = self._q_table[key].get(strategy, 0.0)

        # Q-learning update
        new_q = old_q + Q_ALPHA * (reward - old_q)
        new_q = max(-Q_CLIP, min(Q_CLIP, new_q))
        self._q_table[key][strategy] = round(new_q, 4)

        # EMA update
        old_ema = self._state_ema[key].get(strategy, 0.0)
        new_ema = 0.3 * reward + 0.7 * old_ema
        self._state_ema[key][strategy] = round(new_ema, 4)

        # Visit counts
        self._visit_counts[key][strategy] += 1
        self._total_visits += 1

        # Global stats
        self._strategy_total[strategy] += 1
        if success:
            self._strategy_successes[strategy] += 1
        else:
            self._strategy_failures[strategy] += 1
            self._recent_failures[key][strategy] += 1

        # Recent selections
        self._recent_selections.append((key, strategy, success))
        if len(self._recent_selections) > REP_WINDOW * 3:
            self._recent_selections = self._recent_selections[-REP_WINDOW * 3:]

        # Decay epsilon
        self._episode_count += 1
        self._epsilon = max(EPSILON_FLOOR, self._epsilon * EPSILON_DECAY)

        # Auto-distill check
        self._maybe_distill(key, strategy, success)

        result = {
            "state": key,
            "strategy": strategy,
            "reward": reward,
            "success": success,
            "q_before": round(old_q, 4),
            "q_after": round(new_q, 4),
            "ema_before": round(old_ema, 4),
            "ema_after": round(new_ema, 4),
            "epsilon": round(self._epsilon, 4),
            "visit_count": self._visit_counts[key][strategy],
            "total_visits": self._total_visits,
        }

        logger.info(
            "[LEARN] state=%s strategy=%s reward=%.2f Q: %.3f -> %.3f "
            "EMA: %.3f -> %.3f visits=%d epsilon=%.3f",
            key, strategy, reward, old_q, new_q,
            old_ema, new_ema, self._visit_counts[key][strategy], self._epsilon,
        )

        return result

    # ── Strategy Scoring ──────────────────────────────────────────────────

    def score_strategies(
        self,
        state: PolicyState,
        strategies: list[str],
        heuristic_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, dict[str, float]]:
        """Compute learned scores for all candidate strategies.

        Returns dict of strategy → {component scores + final_score}.
        Learned signals are scaled to be able to FLIP heuristic rankings.
        """
        key = state.to_key()
        results: dict[str, dict[str, float]] = {}

        for strat in strategies:
            q_val = self._q_table[key].get(strat, 0.0)
            ema_val = self._state_ema[key].get(strat, 0.0)
            visits = self._visit_counts[key].get(strat, 0)

            # 1. Q-value weighted contribution
            # Scale Q to be significant: Q range is [-3, +3],
            # multiply by scale factor to match heuristic range (~0.1-0.4)
            q_contrib = q_val * 0.20 * LEARNED_SCORE_SCALE

            # 2. EMA contribution (smoothed reward)
            ema_contrib = ema_val * 0.10 * LEARNED_SCORE_SCALE

            # 3. UCB exploration bonus
            if visits == 0:
                ucb = 0.30  # Strong bonus for untried strategies
            else:
                ucb = 0.25 * math.sqrt(
                    math.log(max(self._total_visits, 1) + 1) / visits
                )
                ucb = min(0.40, ucb)  # Cap but higher than before

            # 4. Distillation boost
            distill_contrib = self._get_distill_score(key, strat)

            # 5. Repetition penalty
            rep_penalty = self._compute_repetition_penalty(key, strat)

            # 6. Recent failure penalty
            fail_penalty = self._compute_failure_penalty(key, strat)

            # 7. Success rate contribution
            total = self._strategy_total.get(strat, 0)
            if total >= 3:
                sr = self._strategy_successes.get(strat, 0) / total
                sr_contrib = 0.10 * (sr - 0.5) * LEARNED_SCORE_SCALE  # Centered at 0.5
            else:
                sr_contrib = 0.0

            # Total learned score
            learned_total = (
                q_contrib
                + ema_contrib
                + ucb
                + distill_contrib
                + rep_penalty
                + fail_penalty
                + sr_contrib
            )

            results[strat] = {
                "q_value": round(q_val, 4),
                "q_contrib": round(q_contrib, 4),
                "ema_value": round(ema_val, 4),
                "ema_contrib": round(ema_contrib, 4),
                "ucb_bonus": round(ucb, 4),
                "distill_contrib": round(distill_contrib, 4),
                "repetition_penalty": round(rep_penalty, 4),
                "failure_penalty": round(fail_penalty, 4),
                "success_rate_contrib": round(sr_contrib, 4),
                "learned_total": round(learned_total, 4),
                "visits": visits,
            }

        return results

    def select_with_exploration(
        self,
        state: PolicyState,
        strategies: list[str],
        final_scores: dict[str, float],
    ) -> tuple[str, bool]:
        """Epsilon-greedy selection with generic_create cooldown.

        Returns (selected_strategy, was_exploration).
        """
        if not strategies:
            return "generic_create", False

        # Sort by final score descending
        ranked = sorted(strategies, key=lambda s: final_scores.get(s, 0.0), reverse=True)
        best = ranked[0]

        # Epsilon-greedy exploration
        if random.random() < self._epsilon and len(ranked) > 1:
            # During exploration, prefer non-dominant strategies
            # Weighted by inverse visit count (explore less-tried ones)
            key = state.to_key()
            weights = []
            for s in ranked[1:]:
                visits = self._visit_counts[key].get(s, 0)
                # Inverse visit weight: less visited = more likely explored
                w = 1.0 / (1.0 + visits)
                weights.append(w)

            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
                chosen = random.choices(ranked[1:], weights=weights, k=1)[0]
            else:
                chosen = random.choice(ranked[1:])

            logger.info(
                "[POLICY] EXPLORATION: state=%s epsilon=%.3f chose=%s (best=%s score=%.3f)",
                state.to_key(), self._epsilon, chosen, best,
                final_scores.get(best, 0.0),
            )
            return chosen, True

        return best, False

    # ── Repetition Suppression ────────────────────────────────────────────

    def _compute_repetition_penalty(self, state_key: str, strategy: str) -> float:
        """Penalty for repeatedly selecting the same strategy in same state."""
        recent = self._recent_selections[-REP_WINDOW:]
        if not recent:
            return 0.0

        # Count how many times this strategy was selected in this state recently
        same_count = sum(
            1 for sk, st, _ in recent
            if sk == state_key and st == strategy
        )

        # Count failures among those selections
        fail_count = sum(
            1 for sk, st, success in recent
            if sk == state_key and st == strategy and not success
        )

        if same_count <= 1:
            return 0.0

        # Base penalty scales with repetition count
        penalty = -REP_PENALTY_BASE * (same_count - 1)

        # Extra penalty for repeated failures
        if fail_count > 0:
            penalty -= REP_PENALTY_BASE * fail_count * (REP_FAILURE_MULT - 1.0)

        return max(-REP_PENALTY_MAX, penalty)

    def _compute_failure_penalty(self, state_key: str, strategy: str) -> float:
        """Extra penalty based on accumulated failure count in this state."""
        fails = self._recent_failures.get(state_key, {}).get(strategy, 0)
        if fails == 0:
            return 0.0
        # Diminishing penalty: -0.1, -0.18, -0.24, ...
        return round(-0.10 * math.sqrt(fails), 4)

    # ── Distillation Integration ──────────────────────────────────────────

    def _maybe_distill(self, state_key: str, strategy: str, success: bool) -> None:
        """Auto-distill rules when sufficient evidence accumulates."""
        visits = self._visit_counts[state_key]
        total_state_visits = sum(visits.values())

        if total_state_visits < 5:
            return  # Not enough data

        # Find best strategy for this state
        q_row = self._q_table.get(state_key, {})
        if not q_row:
            return

        best_strat = max(q_row, key=q_row.get)
        best_q = q_row[best_strat]
        best_visits = visits.get(best_strat, 0)

        if best_visits < 3 or best_q < 0.5:
            return  # Not enough confidence

        # Check if best strategy is significantly better than alternatives
        others = [v for k, v in q_row.items() if k != best_strat]
        if others:
            second_best = max(others)
            gap = best_q - second_best
            if gap < 0.3:
                return  # Not enough gap

        # Compute confidence
        total_trials = self._strategy_total.get(best_strat, 0)
        success_count = self._strategy_successes.get(best_strat, 0)
        confidence = success_count / max(total_trials, 1)

        if confidence < 0.60:
            return

        self._distilled_rules[state_key] = {
            "strategy": best_strat,
            "confidence": round(confidence, 3),
            "q_value": round(best_q, 3),
            "visits": best_visits,
            "timestamp": time.time(),
        }

        logger.info(
            "[DISTILL] New rule: state=%s → %s (conf=%.2f Q=%.2f visits=%d)",
            state_key, best_strat, confidence, best_q, best_visits,
        )

    def _get_distill_score(self, state_key: str, strategy: str) -> float:
        """Score contribution from distilled rules."""
        rule = self._distilled_rules.get(state_key)
        if not rule:
            return 0.0

        if strategy == rule["strategy"]:
            # Boost proportional to confidence
            return DISTILL_BOOST * rule["confidence"]
        else:
            # Penalty for non-recommended strategies
            return DISTILL_PENALTY * rule["confidence"]

    # ── Observability ─────────────────────────────────────────────────────

    def log_selection(
        self,
        state: PolicyState,
        strategies: list[str],
        heuristic_scores: dict[str, float],
        learned_scores: dict[str, dict[str, float]],
        final_scores: dict[str, float],
        selected: str,
        was_exploration: bool,
    ) -> dict[str, Any]:
        """Log full selection details and return summary dict."""
        key = state.to_key()

        log_lines = [f"\n[POLICY] state={key}"]
        for strat in sorted(strategies, key=lambda s: final_scores.get(s, 0.0), reverse=True):
            h = heuristic_scores.get(strat, 0.0)
            ls = learned_scores.get(strat, {})
            f = final_scores.get(strat, 0.0)
            marker = " ***" if strat == selected else ""
            log_lines.append(
                f"  {strat}: heuristic={h:.3f} "
                f"Q={ls.get('q_value', 0):.3f} "
                f"EMA={ls.get('ema_value', 0):.3f} "
                f"UCB={ls.get('ucb_bonus', 0):.3f} "
                f"distill={ls.get('distill_contrib', 0):.3f} "
                f"rep_pen={ls.get('repetition_penalty', 0):.3f} "
                f"fail_pen={ls.get('failure_penalty', 0):.3f} "
                f"learned={ls.get('learned_total', 0):.3f} "
                f"final={f:.3f}{marker}"
            )
        log_lines.append(
            f"  exploration={'YES' if was_exploration else 'no'} "
            f"epsilon={self._epsilon:.3f} chosen={selected}"
        )

        log_text = "\n".join(log_lines)
        logger.info(log_text)

        return {
            "state": key,
            "strategies": {
                s: {
                    "heuristic": round(heuristic_scores.get(s, 0.0), 4),
                    "learned": learned_scores.get(s, {}),
                    "final": round(final_scores.get(s, 0.0), 4),
                }
                for s in strategies
            },
            "selected": selected,
            "exploration": was_exploration,
            "epsilon": round(self._epsilon, 4),
            "episode": self._episode_count,
        }

    # ── Serialization ─────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return summary for diagnostics."""
        return {
            "episode_count": self._episode_count,
            "epsilon": round(self._epsilon, 4),
            "total_visits": self._total_visits,
            "q_table_states": len(self._q_table),
            "distilled_rules": len(self._distilled_rules),
            "strategy_stats": {
                s: {
                    "total": self._strategy_total.get(s, 0),
                    "successes": self._strategy_successes.get(s, 0),
                    "failures": self._strategy_failures.get(s, 0),
                    "success_rate": round(
                        self._strategy_successes.get(s, 0) / max(self._strategy_total.get(s, 0), 1),
                        3,
                    ),
                }
                for s in self._strategy_total
            },
        }

    def get_state_q_table(self) -> dict[str, dict[str, float]]:
        """Return full Q-table for inspection."""
        return {k: dict(v) for k, v in self._q_table.items()}

    def get_distilled_rules(self) -> dict[str, dict[str, Any]]:
        """Return distilled rules for inspection."""
        return dict(self._distilled_rules)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full state for persistence."""
        return {
            "q_table": {k: dict(v) for k, v in self._q_table.items()},
            "visit_counts": {k: dict(v) for k, v in self._visit_counts.items()},
            "total_visits": self._total_visits,
            "state_ema": {k: dict(v) for k, v in self._state_ema.items()},
            "strategy_successes": dict(self._strategy_successes),
            "strategy_failures": dict(self._strategy_failures),
            "strategy_total": dict(self._strategy_total),
            "distilled_rules": dict(self._distilled_rules),
            "epsilon": self._epsilon,
            "episode_count": self._episode_count,
            "recent_failures": {k: dict(v) for k, v in self._recent_failures.items()},
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore state from serialized dict."""
        if not data:
            return
        for k, v in data.get("q_table", {}).items():
            self._q_table[k].update(v)
        for k, v in data.get("visit_counts", {}).items():
            self._visit_counts[k].update(v)
        self._total_visits = data.get("total_visits", 0)
        for k, v in data.get("state_ema", {}).items():
            self._state_ema[k].update(v)
        self._strategy_successes.update(data.get("strategy_successes", {}))
        self._strategy_failures.update(data.get("strategy_failures", {}))
        self._strategy_total.update(data.get("strategy_total", {}))
        self._distilled_rules.update(data.get("distilled_rules", {}))
        self._epsilon = data.get("epsilon", EPSILON_INITIAL)
        self._episode_count = data.get("episode_count", 0)
        for k, v in data.get("recent_failures", {}).items():
            self._recent_failures[k].update(v)
