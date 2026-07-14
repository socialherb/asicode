"""weight_learning.py — Rule-based online weight learning for adaptive scoring.

Maintains per-context-bucket learned weight state updated incrementally after
each run via conservative, rule-based deltas.  No ML; fully explainable.

Architecture
------------
1. ``LearningSignal``      — structured signal from one run / operation outcome
2. ``WeightBucketState``   — per-bucket learned weights + signal count
3. ``WeightLearner``       — state manager: update rules + confidence gating
4. Arithmetic helpers      — delta computation, clamp, normalize, blend
5. Integration entry points:
   - ``build_learning_signal_from_execution_metadata`` — planner/executor → signal
   - ``update_weights_from_monitor_result``            — external monitor → update

Context buckets
---------------
strict_reference_create  — has_strict_reference or reference_bound_context
graph_heavy              — graph_impact_level == "high"
default                  — everything else

Confidence gating (blend policy)
---------------------------------
signal_count < 5          → 100% static (no learning applied)
5 <= signal_count < 10    → 30% learned + 70% static
signal_count >= 10        → 60% learned + 40% static

Static profiles always act as a regularisation anchor — never 100% learned.

Update deltas are intentionally small (0.01–0.03) and always followed by
clamp → normalise to keep weights in [_W_MIN, _W_MAX] and sum == 1.0.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AXES: tuple[str, ...] = ("success", "repair", "contract", "complexity", "cost")

# Conservative update deltas
_DELTA_SMALL:  float = 0.01
_DELTA_MEDIUM: float = 0.02

# Per-axis weight clamp bounds
_W_MIN: float = 0.05
_W_MAX: float = 0.50

# Confidence gating tiers: (min_signal_count, learned_fraction, static_fraction)
# Applied in descending order of min_count; first match wins.
_CONFIDENCE_TIERS: tuple[tuple[int, float, float], ...] = (
    (10, 0.60, 0.40),   # >= 10 signals → 60% learned, 40% static
    (5,  0.30, 0.70),   # >=  5 signals → 30% learned, 70% static
)
# Below lowest tier → pure static
_MIN_SIGNALS_FOR_LEARNING: int = 5

# Bucket identifiers
BUCKET_STRICT_REFERENCE: str = "strict_reference_create"
BUCKET_GRAPH_HEAVY: str      = "graph_heavy"
BUCKET_DEFAULT: str          = "default"

_ALL_BUCKETS: tuple[str, ...] = (
    BUCKET_STRICT_REFERENCE,
    BUCKET_GRAPH_HEAVY,
    BUCKET_DEFAULT,
)

# Maps bucket → base static profile name (mirrors adaptive_scoring.WEIGHT_PROFILES)
_BUCKET_PROFILE_MAP: dict[str, str] = {
    BUCKET_STRICT_REFERENCE: "CONTRACT_HEAVY",
    BUCKET_GRAPH_HEAVY:      "GRAPH_HEAVY",
    BUCKET_DEFAULT:          "DEFAULT",
}

# Static profile weights — duplicated here to avoid circular imports with
# adaptive_scoring.py; both modules must stay independently importable.
_STATIC_BASE: dict[str, dict[str, float]] = {
    "CONTRACT_HEAVY": {
        "success": 0.30, "repair": 0.25, "contract": 0.35,
        "complexity": 0.05, "cost": 0.05,
    },
    "GRAPH_HEAVY": {
        "success": 0.25, "repair": 0.20, "contract": 0.15,
        "complexity": 0.20, "cost": 0.20,
    },
    "DEFAULT": {
        "success": 0.35, "repair": 0.30, "contract": 0.20,
        "complexity": 0.10, "cost": 0.05,
    },
}

# Repair burden ordinal mapping
_BURDEN_RANK: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Axis weights that should relax toward DEFAULT baseline on clean success
_DEFAULT_CONTRACT_BASELINE: float = 0.20
_DEFAULT_REPAIR_BASELINE:   float = 0.30
_RELAX_THRESHOLD:           float = 0.05   # relax only if > baseline + threshold


# ---------------------------------------------------------------------------
# Data structures — Lane-common execution signals
# ---------------------------------------------------------------------------

@dataclass
class ExecutionLearningSignal:
    """Lane-common execution outcome signal.

    Recorded by ALL lanes (PLANNER, MAIN_AGENT, FAST_PATH).
    Does NOT require strategy selection metadata.

    ``verification_depth`` captures signal reliability:
    - "full"          — PLANNER lane (compile→lint→semantic gate→pytest)
    - "self_reported" — MAIN_AGENT (agent's own success/failure judgement)
    - "none"          — FAST_PATH (trivial edit, no verification)
    """
    lane: str                        # "planner" | "main_agent" | "fast_path"
    success: bool
    final_status: str                # "success" | "max_turns" | "error" | "failed" | ...
    failure_class: str               # FailureType value or "" if success
    verification_depth: str          # "full" | "self_reported" | "none"

    # Repair info
    repair_rounds: int = 0
    repair_burden: str = "none"      # "none" | "low" | "medium" | "high"

    # Failure flags
    had_compile_failure: bool = False
    had_lint_failure: bool = False
    had_test_failure: bool = False
    had_semantic_failure: bool = False
    had_contract_failure: bool = False
    had_budget_failure: bool = False
    had_no_progress: bool = False
    had_anchor_loss: bool = False
    rollback_used: bool = False

    # Scope info
    file_count: int = 0
    diff_size_bucket: str = "small"  # "small" | "medium" | "large"

    # Context bucket for context-aware bias (coarse task classification)
    # "single_file" | "multi_file" | "refactor" | "create" | "unknown"
    context_bucket: str = "unknown"

    # Context features (auxiliary — used for fine-grained bias, not bucketing)
    has_symbol_target: bool = False   # True if target symbols were identified
    has_tests: bool = False           # True if tests existed for the target
    scope: str = "small"             # "small" | "medium" | "large" — estimated task scope

    # Execution style (MAIN_AGENT only; "structured" for PLANNER, "" for FAST_PATH)
    # Classified from tool sequence: quick / deliberate / repair_heavy / structured / ""
    execution_style: str = ""

    # Fallback tracking
    is_fallback: bool = False
    parent_lane: str = ""            # lane that failed before fallback
    parent_status: str = ""          # status of the parent lane

    # P8: Alignment-based learning signal (from P7 alignment scorer)
    alignment_score: float = -1.0    # -1 = not computed; 0~1 = computed
    termination_decision: str = ""   # "" | "SUCCESS" | "PARTIAL" | "REPAIR_REQUIRED" | "STOP"
    replan_count: int = 0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "success": self.success,
            "final_status": self.final_status,
            "failure_class": self.failure_class,
            "verification_depth": self.verification_depth,
            "repair_rounds": self.repair_rounds,
            "repair_burden": self.repair_burden,
            "had_compile_failure": self.had_compile_failure,
            "had_lint_failure": self.had_lint_failure,
            "had_test_failure": self.had_test_failure,
            "had_semantic_failure": self.had_semantic_failure,
            "had_contract_failure": self.had_contract_failure,
            "had_budget_failure": self.had_budget_failure,
            "had_no_progress": self.had_no_progress,
            "had_anchor_loss": self.had_anchor_loss,
            "rollback_used": self.rollback_used,
            "file_count": self.file_count,
            "diff_size_bucket": self.diff_size_bucket,
            "context_bucket": self.context_bucket,
            "has_symbol_target": self.has_symbol_target,
            "has_tests": self.has_tests,
            "scope": self.scope,
            "execution_style": self.execution_style,
            "is_fallback": self.is_fallback,
            "parent_lane": self.parent_lane,
            "parent_status": self.parent_status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionLearningSignal":
        return cls(
            lane=d.get("lane", ""),
            success=bool(d.get("success", False)),
            final_status=d.get("final_status", ""),
            failure_class=d.get("failure_class", ""),
            verification_depth=d.get("verification_depth", "none"),
            repair_rounds=int(d.get("repair_rounds", 0)),
            repair_burden=d.get("repair_burden", "none"),
            had_compile_failure=bool(d.get("had_compile_failure", False)),
            had_lint_failure=bool(d.get("had_lint_failure", False)),
            had_test_failure=bool(d.get("had_test_failure", False)),
            had_semantic_failure=bool(d.get("had_semantic_failure", False)),
            had_contract_failure=bool(d.get("had_contract_failure", False)),
            had_budget_failure=bool(d.get("had_budget_failure", False)),
            had_no_progress=bool(d.get("had_no_progress", False)),
            had_anchor_loss=bool(d.get("had_anchor_loss", False)),
            rollback_used=bool(d.get("rollback_used", False)),
            file_count=int(d.get("file_count", 0)),
            diff_size_bucket=d.get("diff_size_bucket", "small"),
            context_bucket=d.get("context_bucket", "unknown"),
            has_symbol_target=bool(d.get("has_symbol_target", False)),
            has_tests=bool(d.get("has_tests", False)),
            scope=d.get("scope", "small"),
            execution_style=d.get("execution_style", ""),
            is_fallback=bool(d.get("is_fallback", False)),
            parent_lane=d.get("parent_lane", ""),
            parent_status=d.get("parent_status", ""),
            timestamp=float(d.get("timestamp", 0.0)),
        )


@dataclass
class StrategyLearningSignal:
    """Planner-only strategy selection outcome signal.

    Recorded only when ``pre_execution_strategy_selection`` metadata exists.
    Separated from ExecutionLearningSignal to prevent lane contamination.
    """
    lane: str = "planner"            # always "planner" (future-proofing)
    strategy_name: str = ""          # e.g. "generic_create", "reference_bound_create"
    strategy_rank: int = 0           # 0-based rank in candidate list
    candidate_count: int = 1
    graph_risk_bucket: str = "low"   # "low" | "medium" | "high"
    success: bool = False
    repair_cost_bucket: str = "none" # "none" | "low" | "medium" | "high"
    switched_strategy: bool = False
    replan_count: int = 0
    final_status: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "strategy_name": self.strategy_name,
            "strategy_rank": self.strategy_rank,
            "candidate_count": self.candidate_count,
            "graph_risk_bucket": self.graph_risk_bucket,
            "success": self.success,
            "repair_cost_bucket": self.repair_cost_bucket,
            "switched_strategy": self.switched_strategy,
            "replan_count": self.replan_count,
            "final_status": self.final_status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyLearningSignal":
        return cls(
            lane=d.get("lane", "planner"),
            strategy_name=d.get("strategy_name", ""),
            strategy_rank=int(d.get("strategy_rank", 0)),
            candidate_count=int(d.get("candidate_count", 1)),
            graph_risk_bucket=d.get("graph_risk_bucket", "low"),
            success=bool(d.get("success", False)),
            repair_cost_bucket=d.get("repair_cost_bucket", "none"),
            switched_strategy=bool(d.get("switched_strategy", False)),
            replan_count=int(d.get("replan_count", 0)),
            final_status=d.get("final_status", ""),
            timestamp=float(d.get("timestamp", 0.0)),
        )


# ---------------------------------------------------------------------------
# Data structures — Weight learning (existing)
# ---------------------------------------------------------------------------

@dataclass
class LearningSignal:
    """Structured learning signal from one run / operation outcome.

    All fields are serialisable (no rich objects).
    """
    bucket: str                      # one of BUCKET_* constants
    selected_weight_profile: str     # "CONTRACT_HEAVY" | "GRAPH_HEAVY" | "DEFAULT"
    selected_strategy: str           # e.g. "generic_create", "reference_bound_create"
    success: bool
    repair_attempts: int             # 0 = no repair
    repair_burden: str               # "none" | "low" | "medium" | "high"
    contract_violation: bool
    semantic_failures: list[str]     # free-form reason strings
    budget_failure: bool
    graph_impact_level: str          # "low" | "medium" | "high"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket":                  self.bucket,
            "selected_weight_profile": self.selected_weight_profile,
            "selected_strategy":       self.selected_strategy,
            "success":                 self.success,
            "repair_attempts":         self.repair_attempts,
            "repair_burden":           self.repair_burden,
            "contract_violation":      self.contract_violation,
            "semantic_failures":       list(self.semantic_failures),
            "budget_failure":          self.budget_failure,
            "graph_impact_level":      self.graph_impact_level,
            "timestamp":               self.timestamp,
        }


@dataclass
class WeightBucketState:
    """Learned weight state for one context bucket."""
    weights: dict[str, float]
    signal_count: int  = 0
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights":      dict(self.weights),
            "signal_count": self.signal_count,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------

def resolve_bucket(
    has_strict_reference: bool = False,
    reference_bound_context: bool = False,
    graph_impact_level: str = "low",
) -> str:
    """Map context features to a weight learning bucket.

    Priority mirrors adaptive_scoring.select_weight_profile():
    1. strict reference or reference-bound context → strict_reference_create
    2. graph_impact_level == "high"                → graph_heavy
    3. otherwise                                   → default
    """
    if has_strict_reference or reference_bound_context:
        return BUCKET_STRICT_REFERENCE
    if graph_impact_level == "high":
        return BUCKET_GRAPH_HEAVY
    return BUCKET_DEFAULT


# ---------------------------------------------------------------------------
# Weight arithmetic helpers
# ---------------------------------------------------------------------------

def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    """Return a copy of ``w`` scaled so values sum to exactly 1.0."""
    total = sum(w.values())
    if total <= 0:
        equal = round(1.0 / len(_AXES), 6)
        return {k: equal for k in _AXES}
    return {k: round(v / total, 6) for k, v in w.items()}


def _clamp_weights(w: dict[str, float]) -> dict[str, float]:
    """Return a copy of ``w`` with each value clamped to [_W_MIN, _W_MAX]."""
    return {k: max(_W_MIN, min(_W_MAX, v)) for k, v in w.items()}


def _blend_weights(
    learned: dict[str, float],
    static: dict[str, float],
    learned_frac: float,
    static_frac: float,
) -> dict[str, float]:
    """Linear blend of learned and static weights, normalised."""
    blended = {
        k: learned_frac * learned.get(k, 0.0) + static_frac * static.get(k, 0.0)
        for k in _AXES
    }
    return _normalize_weights(blended)


def _compute_weight_delta(
    signal: LearningSignal,
    current_weights: dict[str, float],
) -> dict[str, float]:
    """Compute axis-wise weight deltas for one learning signal.

    Rules (additive; multiple may fire per signal):
    A. contract_violation or semantic contract failure:
         contract += MEDIUM, repair += SMALL, success -= SMALL
    B. high repair burden:
         repair += MEDIUM; graph_heavy bucket: complexity += SMALL
       medium repair burden:
         repair += SMALL
    C. graph-heavy failure (not success AND graph_impact == "high"):
         complexity += MEDIUM, cost += SMALL, success -= SMALL
    D. clean success — no repair, no contract violation:
         success += SMALL
         if contract/repair weights are notably above DEFAULT baseline:
             contract -= 0.5×SMALL, repair -= 0.5×SMALL

    Returns a dict {axis: delta} where positive means "increase this weight".
    """
    delta: dict[str, float] = {k: 0.0 for k in _AXES}

    # ── A: Contract / semantic failure ────────────────────────────────────
    has_semantic_contract_fail = any(
        "contract" in f.lower() for f in (signal.semantic_failures or [])
    )
    if signal.contract_violation or has_semantic_contract_fail:
        delta["contract"] += _DELTA_MEDIUM
        delta["repair"]   += _DELTA_SMALL
        delta["success"]  -= _DELTA_SMALL

    # ── B: Repair burden ──────────────────────────────────────────────────
    burden_rank = _BURDEN_RANK.get(signal.repair_burden, 0)
    if burden_rank >= 3:   # high
        delta["repair"] += _DELTA_MEDIUM
    elif burden_rank >= 2: # medium
        delta["repair"] += _DELTA_SMALL
    # Graph-heavy bucket: also lift complexity on significant burden
    if signal.bucket == BUCKET_GRAPH_HEAVY and burden_rank >= 2:
        delta["complexity"] += _DELTA_SMALL

    # ── C: Graph-heavy failure ─────────────────────────────────────────────
    if not signal.success and signal.graph_impact_level == "high":
        delta["complexity"] += _DELTA_MEDIUM
        delta["cost"]       += _DELTA_SMALL
        delta["success"]    -= _DELTA_SMALL

    # ── D: Clean success ──────────────────────────────────────────────────
    if signal.success and signal.repair_attempts == 0 and not signal.contract_violation:
        delta["success"] += _DELTA_SMALL
        # Gently relax over-elevated contract/repair toward DEFAULT baseline
        if current_weights.get("contract", 0) > _DEFAULT_CONTRACT_BASELINE + _RELAX_THRESHOLD:
            delta["contract"] -= _DELTA_SMALL * 0.5
        if current_weights.get("repair", 0) > _DEFAULT_REPAIR_BASELINE + _RELAX_THRESHOLD:
            delta["repair"] -= _DELTA_SMALL * 0.5

    return delta


def _apply_weight_delta(
    weights: dict[str, float],
    delta: dict[str, float],
) -> dict[str, float]:
    """Apply delta, then project onto the bounded probability simplex.

    Guarantees:
    - sum(result) == 1.0
    - _W_MIN <= result[k] <= _W_MAX for all k

    Algorithm:
    1. Add delta.
    2. Normalise (scale to sum=1.0).
    3. Clamp each axis to [_W_MIN, _W_MAX].
    4. Re-balance: any sum residual from step 3 is absorbed by the axis
       farthest from both bounds, keeping all axes within bounds.
    """
    updated = {k: weights.get(k, 0.0) + delta.get(k, 0.0) for k in _AXES}
    normed  = _normalize_weights(updated)
    clamped = _clamp_weights(normed)

    # Re-distribute sum residual onto the most flexible axis
    deficit = round(1.0 - sum(clamped.values()), 9)
    if abs(deficit) > 1e-9:
        for k in sorted(_AXES, key=lambda x: -clamped[x]):
            candidate = clamped[k] + deficit
            if _W_MIN - 1e-9 <= candidate <= _W_MAX + 1e-9:
                clamped[k] = max(_W_MIN, min(_W_MAX, candidate))
                break

    return clamped


# ---------------------------------------------------------------------------
# WeightLearner
# ---------------------------------------------------------------------------

class WeightLearner:
    """Manages per-bucket learned weight state with conservative online updates.

    Thread-safe — uses a lock for concurrent signal updates.

    Usage::

        learner = WeightLearner()

        # After each run outcome:
        sig = LearningSignal(bucket="strict_reference_create", ...)
        learner.update(sig)

        # In adaptive_scoring / strategy_selector:
        eff_weights, source, n = learner.get_effective_weights(
            "strict_reference_create", static_weights
        )
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Initialise each bucket from the corresponding static profile
        self._states: dict[str, WeightBucketState] = {
            bucket: WeightBucketState(
                weights=_STATIC_BASE[_BUCKET_PROFILE_MAP[bucket]].copy(),
                signal_count=0,
                last_updated=0.0,
            )
            for bucket in _ALL_BUCKETS
        }

    # ── State access ──────────────────────────────────────────────────────

    def get_bucket_state(self, bucket: str) -> WeightBucketState:
        """Return state for ``bucket``; falls back to default bucket."""
        return self._states.get(bucket, self._states[BUCKET_DEFAULT])

    def get_learned_weights(self, bucket: str) -> dict[str, float]:
        """Return the current learned (possibly still equal to static) weights."""
        return dict(self.get_bucket_state(bucket).weights)

    def get_effective_weights(
        self,
        bucket: str,
        static_weights: dict[str, float],
    ) -> tuple[dict[str, float], str, int]:
        """Return confidence-gated blended weights.

        Parameters
        ----------
        bucket:
            Context bucket name.
        static_weights:
            Base static profile weights to blend against.

        Returns
        -------
        ``(effective_weights, weight_source, signal_count)``
        ``weight_source``: "static" | "blended"
        """
        with self._lock:
            state = self.get_bucket_state(bucket)
            n = state.signal_count

            if n < _MIN_SIGNALS_FOR_LEARNING:
                return dict(static_weights), "static", n

            for min_count, lf, sf in _CONFIDENCE_TIERS:
                if n >= min_count:
                    blended = _blend_weights(state.weights, static_weights, lf, sf)
                    return blended, "blended", n

            # Safety fallback — should never reach here
            return dict(static_weights), "static", n

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, signal: LearningSignal) -> None:
        """Process one learning signal; update the relevant bucket's weights."""
        with self._lock:
            bucket = signal.bucket
            if bucket not in self._states:
                logger.debug(
                    "weight_learning: unknown bucket %r — skipping update", bucket
                )
                return

            state = self._states[bucket]
            delta = _compute_weight_delta(signal, state.weights)

            # Only persist the update if at least one axis changed meaningfully
            if all(abs(v) < 1e-9 for v in delta.values()):
                logger.debug(
                    "weight_learning: bucket=%s zero-delta signal — no update", bucket
                )
                return

            new_weights = _apply_weight_delta(state.weights, delta)
            state.weights     = new_weights
            state.signal_count += 1
            state.last_updated = signal.timestamp

            logger.debug(
                "weight_learning: bucket=%s n=%d delta=%s → weights=%s",
                bucket,
                state.signal_count,
                {k: f"{v:+.3f}" for k, v in delta.items() if abs(v) > 1e-9},
                {k: f"{v:.3f}" for k, v in new_weights.items()},
            )

    # ── Persistence ───────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all bucket states.

        This is also the canonical export format for persistence.
        Restore with :meth:`load_state`.
        """
        return {
            bucket: {
                "weights":      dict(state.weights),
                "signal_count": state.signal_count,
                "last_updated": state.last_updated,
                "base_profile": _BUCKET_PROFILE_MAP.get(bucket, "DEFAULT"),
            }
            for bucket, state in self._states.items()
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Restore bucket states from a previously persisted dict.

        Accepts the format produced by :meth:`get_summary`.  Unknown bucket
        keys are silently skipped; invalid or missing weight fields fall back
        to the current (static-initialised) defaults.  Clamp + normalise are
        re-applied on every bucket to guarantee weight invariants.
        """
        for bucket, data in state_dict.items():
            if bucket not in self._states or not isinstance(data, dict):
                continue
            raw_weights = data.get("weights")
            if not isinstance(raw_weights, dict):
                continue
            try:
                restored = _normalize_weights(_clamp_weights({
                    k: float(raw_weights.get(k, self._states[bucket].weights.get(k, 0.0)))
                    for k in _AXES
                }))
                self._states[bucket].weights      = restored
                self._states[bucket].signal_count = int(data.get("signal_count", 0))
                self._states[bucket].last_updated = float(data.get("last_updated", 0.0))
            except Exception:
                pass  # Keep existing state on per-bucket error; silently continue


# ---------------------------------------------------------------------------
# ExecutionLearner — lane-common execution outcome learner
# ---------------------------------------------------------------------------

# Execution learning axes — different from strategy weight axes

_EXEC_MIN_SIGNALS: dict[str, int] = {
    "planner": 4,
    "main_agent": 5,
    "fast_path": 10,
}

# Verification depth → signal weight (for weighted aggregation)
_VERIFICATION_WEIGHT: dict[str, float] = {
    "full": 1.0,
    "self_reported": 0.5,
    "none": 0.3,
}


@dataclass
class ExecutionBucketState:
    """Learned execution outcome state for one lane.

    Counts are floats because signals are weighted by verification_depth:
    full=1.0, self_reported=0.5, none=0.3.
    """
    lane: str
    signal_count: float = 0.0
    success_count: float = 0.0
    repair_count: float = 0.0
    budget_exhaust_count: float = 0.0
    verify_fail_count: float = 0.0
    rollback_count: float = 0.0
    fallback_count: float = 0.0
    total_repair_rounds: int = 0       # raw int (not weighted — actual rounds)
    # Fallback recovery tracking
    recovery_success_count: float = 0.0  # is_fallback=True AND success=True
    recovery_total_count: float = 0.0    # is_fallback=True (total)
    # Per-style tracking: {style_name: {signals: float, successes: float}}
    style_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-context tracking: {context_bucket: {signals, successes, repairs, budget_fails}}
    context_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # Dynamic bucket metadata: {bucket_name: {origin, created_at, reason}}
    bucket_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "signal_count": self.signal_count,
            "success_count": self.success_count,
            "repair_count": self.repair_count,
            "budget_exhaust_count": self.budget_exhaust_count,
            "verify_fail_count": self.verify_fail_count,
            "rollback_count": self.rollback_count,
            "fallback_count": self.fallback_count,
            "total_repair_rounds": self.total_repair_rounds,
            "recovery_success_count": self.recovery_success_count,
            "recovery_total_count": self.recovery_total_count,
            "style_stats": {k: dict(v) for k, v in self.style_stats.items()},
            "context_stats": {k: dict(v) for k, v in self.context_stats.items()},
            "bucket_metadata": {k: dict(v) for k, v in self.bucket_metadata.items()},
            "last_updated": self.last_updated,
        }

    @property
    def success_rate(self) -> float:
        return self.success_count / self.signal_count if self.signal_count > 0 else 0.0

    @property
    def repair_rate(self) -> float:
        return self.repair_count / self.signal_count if self.signal_count > 0 else 0.0

    @property
    def avg_repair_rounds(self) -> float:
        return self.total_repair_rounds / self.signal_count if self.signal_count > 0 else 0.0

    @property
    def budget_exhaust_rate(self) -> float:
        return self.budget_exhaust_count / self.signal_count if self.signal_count > 0 else 0.0

    @property
    def verify_fail_rate(self) -> float:
        return self.verify_fail_count / self.signal_count if self.signal_count > 0 else 0.0

    @property
    def recovery_success_rate(self) -> float:
        """Rate at which fallback executions succeed (rescue rate)."""
        return self.recovery_success_count / self.recovery_total_count if self.recovery_total_count > 0 else 0.0

    def get_style_success_rate(self, style: str) -> float:
        """Success rate for a specific execution_style."""
        s = self.style_stats.get(style)
        if not s or s.get("signals", 0) <= 0:
            return 0.0
        return s.get("successes", 0) / s["signals"]

    def get_context_stats(self, context_bucket: str) -> Optional[dict[str, float]]:
        """Return stats for a specific context bucket, or None if no data."""
        return self.context_stats.get(context_bucket)


def _compute_reward(success: bool, plan_quality: float, base_success: float = 1.0, base_fail: float = -0.5) -> float:
    """Compute reward based on plan quality and success status.

    Returns plan_quality if nonzero, else base_success if success else base_fail.
    """
    if plan_quality != 0:
        return plan_quality
    return base_success if success else base_fail


class ExecutionLearner:
    """Lane-common execution outcome learner.

    Tracks per-lane aggregated execution statistics with thread-safe updates.
    Unlike WeightLearner (which adjusts strategy scoring weights), this learner
    tracks raw outcome rates to answer questions like:
    - "Does MAIN_AGENT frequently exhaust max_turns?"
    - "Does PLANNER's repair rate increase over time?"
    - "Are fallback executions generally successful?"

    Thread-safe via threading.Lock.
    """

    _LANES: tuple[str, ...] = ("planner", "main_agent", "fast_path")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ExecutionBucketState] = {
            lane: ExecutionBucketState(lane=lane)
            for lane in self._LANES
        }
        # Recent signals for pattern analysis (FIFO, max 200 per lane)
        self._recent_signals: dict[str, list[dict[str, Any]]] = {
            lane: [] for lane in self._LANES
        }
        self._max_recent_per_lane: int = 200

    def update(self, signal: ExecutionLearningSignal) -> None:
        """Process one execution signal; update the relevant lane's state.

        Uses verification_depth-based weighting so that signals from lanes
        with weaker verification (self_reported, none) contribute less to
        aggregated rates than fully-verified signals (full).

        Weight mapping: full=1.0, self_reported=0.5, none=0.3
        """
        with self._lock:
            lane = signal.lane
            if lane not in self._states:
                logger.debug(
                    "execution_learner: unknown lane %r — skipping", lane
                )
                return

            w = _VERIFICATION_WEIGHT.get(signal.verification_depth, 1.0)

            state = self._states[lane]
            state.signal_count += w
            if signal.success:
                state.success_count += w
            if signal.repair_rounds > 0:
                state.repair_count += w
            state.total_repair_rounds += signal.repair_rounds
            if signal.had_budget_failure:
                state.budget_exhaust_count += w
            if (signal.had_compile_failure or signal.had_lint_failure
                    or signal.had_test_failure or signal.had_semantic_failure):
                state.verify_fail_count += w
            if signal.rollback_used:
                state.rollback_count += w
            if signal.is_fallback:
                state.fallback_count += w
                state.recovery_total_count += w
                if signal.success:
                    state.recovery_success_count += w

            # Per-style tracking
            if signal.execution_style:
                ss = state.style_stats.setdefault(
                    signal.execution_style, {"signals": 0.0, "successes": 0.0}
                )
                ss["signals"] += w
                if signal.success:
                    ss["successes"] += w

            # Per-context tracking + dynamic bucket management
            if signal.context_bucket and signal.context_bucket != "unknown":
                # Record to original bucket
                _record_ctx = self._record_to_context_bucket(
                    state, signal.context_bucket, w, signal
                )
                # Check for dynamic bucket assignment
                _dynamic = self._resolve_dynamic_bucket(state, signal.context_bucket, signal)
                if _dynamic and _dynamic != signal.context_bucket:
                    self._record_to_context_bucket(state, _dynamic, w, signal)
                # Periodic: check for divergence → auto-create new bucket
                self._check_divergence_and_create(state, signal.context_bucket)
                # Periodic: merge similar buckets + enforce cap
                if state.signal_count % 10 < w:  # roughly every 10 signals
                    self._merge_similar_buckets(state)
                    self._enforce_bucket_cap(state)

            state.last_updated = signal.timestamp

            # Append to recent signals (FIFO eviction)
            recent = self._recent_signals[lane]
            recent.append(signal.to_dict())
            if len(recent) > self._max_recent_per_lane:
                self._recent_signals[lane] = recent[-self._max_recent_per_lane:]

            logger.debug(
                "execution_learner: lane=%s n=%d success_rate=%.2f "
                "repair_rate=%.2f budget_exhaust_rate=%.2f",
                lane, state.signal_count, state.success_rate,
                state.repair_rate, state.budget_exhaust_rate,
            )

    # ── Dynamic bucket management (all called under self._lock) ──────────

    # Static buckets cannot be merged or deleted
    _STATIC_BUCKETS = frozenset({"single_file", "multi_file", "create", "refactor"})
    _MAX_CONTEXT_BUCKETS = 12   # 4 static × 3 tiers max = 12
    _DIVERGENCE_MIN_SIGNALS = 5.0
    _DIVERGENCE_THRESHOLD = 0.25   # |context_rate - lane_rate| > Δ → create
    _MERGE_THRESHOLD = 0.10        # distance(A, B) < ε → merge  (hysteresis: create > merge)
    _INHERITANCE_FACTOR = 0.5      # new bucket inherits 50% of parent stats

    @staticmethod
    def _record_to_context_bucket(
        state: ExecutionBucketState,
        bucket: str,
        w: float,
        signal: "ExecutionLearningSignal",
    ) -> None:
        """Record a signal to a specific context bucket (no lock needed — called under lock)."""
        cs = state.context_stats.setdefault(
            bucket, {
                "signals": 0.0, "successes": 0.0, "repairs": 0.0, "budget_fails": 0.0,
                "verify_fails": 0.0,
                "with_symbol": 0.0, "with_symbol_success": 0.0,
                "with_tests": 0.0, "with_tests_success": 0.0,
            }
        )
        cs["signals"] += w
        if signal.success:
            cs["successes"] += w
        if signal.repair_rounds > 0:
            cs["repairs"] += w
        if signal.had_budget_failure:
            cs["budget_fails"] += w
        if not signal.success and (signal.had_test_failure or signal.had_compile_failure
                                   or signal.had_semantic_failure):
            cs["verify_fails"] = cs.get("verify_fails", 0) + w
        # Context feature tracking
        if signal.has_symbol_target:
            cs["with_symbol"] = cs.get("with_symbol", 0) + w
            if signal.success:
                cs["with_symbol_success"] = cs.get("with_symbol_success", 0) + w
        if signal.has_tests:
            cs["with_tests"] = cs.get("with_tests", 0) + w
            if signal.success:
                cs["with_tests_success"] = cs.get("with_tests_success", 0) + w

    # Absolute floor/ceiling for tier thresholds (safety bounds)
    _TIER_HEAVY_MIN = 2
    _TIER_HEAVY_MAX = 5
    _TIER_MEDIUM_MIN = 1

    @staticmethod
    def _classify_signal_tier(
        signal: "ExecutionLearningSignal",
        lane_state: Optional["ExecutionBucketState"] = None,
    ) -> str:
        """Classify a signal into a difficulty tier.

        Uses adaptive thresholds when lane_state is available:
        - heavy threshold = avg_repair_rounds + 1σ (clamped to [2, 5])
        - medium threshold = max(1, heavy_threshold / 2)

        Additionally considers failure_class for quality-aware classification:
        - VERIFY_FAIL (test failure) without repair → at least medium
        - NO_PROGRESS → at least medium
        - PARSE_FAIL → at least medium (broken output)

        Falls back to fixed thresholds (heavy≥3, medium≥1) when no lane data.

        Returns "heavy" | "medium" | "light".
        """
        # Adaptive thresholds from lane stats
        heavy_thresh = 3  # default
        medium_thresh = 1  # default
        if lane_state and lane_state.signal_count >= 4:
            avg = lane_state.avg_repair_rounds
            _sigma = max(avg * lane_state.repair_rate, 0.5) if lane_state.repair_rate > 0 else 1.0
            heavy_thresh = max(
                ExecutionLearner._TIER_HEAVY_MIN,
                min(int(avg + _sigma + 0.5), ExecutionLearner._TIER_HEAVY_MAX),
            )
            medium_thresh = max(ExecutionLearner._TIER_MEDIUM_MIN, heavy_thresh // 2)

        # Repair-based classification
        if not signal.success and signal.repair_rounds >= heavy_thresh:
            return "heavy"
        if signal.repair_rounds >= medium_thresh or signal.had_budget_failure:
            return "medium"

        # Failure-class boost: certain failure types are inherently non-trivial
        # even without repair attempts (e.g. test failure = code is wrong)
        if not signal.success:
            fc = signal.failure_class or ""
            if fc in ("VERIFY_FAIL", "NO_PROGRESS", "PARSE_FAIL"):
                return "medium"
            if signal.had_compile_failure or signal.had_test_failure:
                return "medium"

        return "light"

    def _resolve_dynamic_bucket(
        self, state: ExecutionBucketState, original_bucket: str,
        signal: Optional["ExecutionLearningSignal"] = None,
    ) -> Optional[str]:
        """Check if this signal should also be assigned to a dynamic tier bucket.

        Returns the dynamic bucket name, or None if no reassignment.
        Multi-tier: {bucket}_light, {bucket}_medium, {bucket}_heavy.
        Only assigns to tiers that already exist (created by divergence detection).
        """
        if signal is None:
            return None
        tier = self._classify_signal_tier(signal, lane_state=state)
        derived_name = f"{original_bucket}_{tier}"
        if derived_name in state.context_stats:
            return derived_name
        # Fall back: medium → heavy if exact tier doesn't exist
        if tier == "medium":
            heavy_name = f"{original_bucket}_heavy"
            if heavy_name in state.context_stats:
                return heavy_name
        # Light: only mirror if _light bucket already exists (created by divergence)
        return None

    def _check_divergence_and_create(
        self, state: ExecutionBucketState, bucket: str
    ) -> None:
        """Detect if a bucket's rates diverge significantly from lane average.

        Creates tier-appropriate derived bucket:
        - Strong divergence (Δ > 2×threshold) → {bucket}_heavy
        - Moderate divergence (Δ > threshold)  → {bucket}_medium

        Only creates from static buckets. At most one tier is created per check.
        """
        cs = state.context_stats.get(bucket)
        if not cs or cs.get("signals", 0) < self._DIVERGENCE_MIN_SIGNALS:
            return
        if bucket not in self._STATIC_BUCKETS:
            return

        _s = cs["signals"]
        ctx_repair = cs.get("repairs", 0) / _s
        ctx_success = cs.get("successes", 0) / _s
        lane_repair = state.repair_rate
        lane_success = state.success_rate

        repair_div = ctx_repair - lane_repair   # positive = worse than lane
        success_div = lane_success - ctx_success  # positive = worse than lane

        # Negative divergence = context is BETTER than lane → light tier
        repair_advantage = lane_repair - ctx_repair  # positive = context has less repair
        success_advantage = ctx_success - lane_success  # positive = context more successful

        # Check for negative divergence (context outperforms lane) → create _light
        max_advantage = max(repair_advantage, success_advantage)
        if max_advantage > self._DIVERGENCE_THRESHOLD:
            light_name = f"{bucket}_light"
            if light_name not in state.context_stats:
                parent_stats = dict(cs)
                state.context_stats[light_name] = {
                    k: v * self._INHERITANCE_FACTOR
                    for k, v in parent_stats.items()
                }
                _adv_axis = "repair" if repair_advantage >= success_advantage else "success"
                state.bucket_metadata[light_name] = {
                    "origin": bucket,
                    "tier": "light",
                    "created_at": time.time(),
                    "reason": f"{_adv_axis}_advantage={max_advantage:+.2f}",
                }
                logger.info(
                    "dynamic_bucket: created %r (tier=light) from %r (advantage=%.2f)",
                    light_name, bucket, max_advantage,
                )

        # Check for positive divergence (context underperforms lane) → heavy/medium
        max_div = max(repair_div, success_div)
        if max_div <= self._DIVERGENCE_THRESHOLD:
            return

        if max_div > self._DIVERGENCE_THRESHOLD * 2:
            tier = "heavy"
        else:
            tier = "medium"

        derived_name = f"{bucket}_{tier}"
        if derived_name in state.context_stats:
            return

        parent_stats = dict(cs)
        state.context_stats[derived_name] = {
            k: v * self._INHERITANCE_FACTOR
            for k, v in parent_stats.items()
        }
        _reason_axis = "repair" if repair_div >= success_div else "success"
        state.bucket_metadata[derived_name] = {
            "origin": bucket,
            "tier": tier,
            "created_at": time.time(),
            "reason": f"{_reason_axis}_div={max_div:+.2f}",
        }
        logger.info(
            "dynamic_bucket: created %r (tier=%s) from %r (div=%.2f)",
            derived_name, tier, bucket, max_div,
        )

    @staticmethod
    def _bucket_distance(a: dict[str, float], b: dict[str, float]) -> float:
        """Compute distance between two context stat dicts."""
        if not a.get("signals") or not b.get("signals"):
            return float("inf")
        sa, sb = a["signals"], b["signals"]
        repair_diff = abs(a.get("repairs", 0) / sa - b.get("repairs", 0) / sb)
        success_diff = abs(a.get("successes", 0) / sa - b.get("successes", 0) / sb)
        return repair_diff + success_diff

    _ADJACENT_TIERS = {
        ("light", "medium"), ("medium", "light"),
        ("medium", "heavy"), ("heavy", "medium"),
    }

    @classmethod
    def _parse_tier_bucket(cls, name: str) -> tuple[str, str]:
        """Parse '{base}_{tier}' → (base, tier). Returns ('', '') for non-tier buckets."""
        for tier in ("heavy", "medium", "light"):
            if name.endswith(f"_{tier}"):
                return name[: -(len(tier) + 1)], tier
        return "", ""

    def _merge_similar_buckets(self, state: ExecutionBucketState) -> None:
        """Merge dynamic buckets that have converged (distance < ε).

        Only merges adjacent tiers from the same base bucket:
        single_file_medium + single_file_heavy → OK
        single_file_light + multi_file_heavy   → blocked
        """
        dynamic_buckets = [
            b for b in state.context_stats
            if b not in self._STATIC_BUCKETS
        ]
        if len(dynamic_buckets) < 2:
            return

        merged = set()
        for i, a_name in enumerate(dynamic_buckets):
            if a_name in merged:
                continue
            a_base, a_tier = self._parse_tier_bucket(a_name)
            for b_name in dynamic_buckets[i + 1:]:
                if b_name in merged:
                    continue
                b_base, b_tier = self._parse_tier_bucket(b_name)
                # Only merge adjacent tiers from same base
                if a_base != b_base or not a_base:
                    continue
                if (a_tier, b_tier) not in self._ADJACENT_TIERS:
                    continue
                a = state.context_stats.get(a_name, {})
                b = state.context_stats.get(b_name, {})
                if self._bucket_distance(a, b) < self._MERGE_THRESHOLD:
                    if b.get("signals", 0) > a.get("signals", 0):
                        a_name, b_name = b_name, a_name
                        a, b = b, a
                    for k in ("signals", "successes", "repairs", "budget_fails"):
                        a[k] = a.get(k, 0) + b.get(k, 0)
                    state.context_stats[a_name] = a
                    del state.context_stats[b_name]
                    state.bucket_metadata.pop(b_name, None)
                    merged.add(b_name)
                    logger.debug(
                        "dynamic_bucket: merged %r into %r (distance < %.2f)",
                        b_name, a_name, self._MERGE_THRESHOLD,
                    )

    # Eviction priority: light first, then medium, heavy last (most informative)
    _TIER_EVICTION_PRIORITY = {"light": 0, "medium": 1, "heavy": 2, "": 3}

    def _enforce_bucket_cap(self, state: ExecutionBucketState) -> None:
        """Enforce MAX_CONTEXT_BUCKETS by evicting lowest-value dynamic buckets.

        Eviction priority: light > medium > heavy (heavy preserved longest).
        Within same tier: lowest signal count evicted first.
        """
        while len(state.context_stats) > self._MAX_CONTEXT_BUCKETS:
            dynamic = []
            for name, cs in state.context_stats.items():
                if name in self._STATIC_BUCKETS:
                    continue
                _, tier = self._parse_tier_bucket(name)
                priority = self._TIER_EVICTION_PRIORITY.get(tier, 3)
                dynamic.append((name, priority, cs.get("signals", 0)))
            if not dynamic:
                break
            # Sort: lowest priority first (light=0), then lowest signals
            dynamic.sort(key=lambda x: (x[1], x[2]))
            victim_name = dynamic[0][0]
            # Merge victim into its origin (or lane-level if no origin)
            meta = state.bucket_metadata.get(victim_name, {})
            origin = meta.get("origin", "")
            if origin and origin in state.context_stats:
                target = state.context_stats[origin]
                victim = state.context_stats[victim_name]
                for k in ("signals", "successes", "repairs", "budget_fails"):
                    target[k] = target.get(k, 0) + victim.get(k, 0)
            del state.context_stats[victim_name]
            state.bucket_metadata.pop(victim_name, None)
            logger.debug(
                "dynamic_bucket: evicted %r (cap=%d)", victim_name, self._MAX_CONTEXT_BUCKETS
            )

    def get_lane_state(self, lane: str) -> ExecutionBucketState:
        """Return state for a lane; falls back to a fresh state."""
        with self._lock:
            state = self._states.get(lane)
            if state is None:
                return ExecutionBucketState(lane=lane)
            # Return a copy to prevent external mutation
            return ExecutionBucketState(
                lane=state.lane,
                signal_count=state.signal_count,
                success_count=state.success_count,
                repair_count=state.repair_count,
                budget_exhaust_count=state.budget_exhaust_count,
                verify_fail_count=state.verify_fail_count,
                rollback_count=state.rollback_count,
                fallback_count=state.fallback_count,
                total_repair_rounds=state.total_repair_rounds,
                recovery_success_count=state.recovery_success_count,
                recovery_total_count=state.recovery_total_count,
                style_stats={k: dict(v) for k, v in state.style_stats.items()},
                context_stats={k: dict(v) for k, v in state.context_stats.items()},
                bucket_metadata={k: dict(v) for k, v in state.bucket_metadata.items()},
                last_updated=state.last_updated,
            )

    def get_recent_failure_patterns(
        self, lane: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent failure signals for a lane (newest first)."""
        with self._lock:
            signals = self._recent_signals.get(lane, [])
            failures = [s for s in signals if not s.get("success", False)]
            return list(reversed(failures[-limit:]))

    def get_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all lane states.

        This is the canonical export format for persistence.
        Restore with :meth:`load_state`.
        """
        with self._lock:
            return {
                lane: state.to_dict()
                for lane, state in self._states.items()
            }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Restore lane states from a previously persisted dict."""
        with self._lock:
            for lane, data in state_dict.items():
                if lane not in self._states or not isinstance(data, dict):
                    continue
                try:
                    state = self._states[lane]
                    state.signal_count = float(data.get("signal_count", 0))
                    state.success_count = float(data.get("success_count", 0))
                    state.repair_count = float(data.get("repair_count", 0))
                    state.budget_exhaust_count = float(data.get("budget_exhaust_count", 0))
                    state.verify_fail_count = float(data.get("verify_fail_count", 0))
                    state.rollback_count = float(data.get("rollback_count", 0))
                    state.fallback_count = float(data.get("fallback_count", 0))
                    state.total_repair_rounds = int(data.get("total_repair_rounds", 0))
                    state.recovery_success_count = float(data.get("recovery_success_count", 0))
                    state.recovery_total_count = float(data.get("recovery_total_count", 0))
                    # Restore style_stats
                    raw_styles = data.get("style_stats")
                    if isinstance(raw_styles, dict):
                        state.style_stats = {
                            k: {"signals": float(v.get("signals", 0)), "successes": float(v.get("successes", 0))}
                            for k, v in raw_styles.items()
                            if isinstance(v, dict)
                        }
                    # Restore context_stats (forward-compatible: missing keys → 0)
                    raw_ctx = data.get("context_stats")
                    if isinstance(raw_ctx, dict):
                        state.context_stats = {}
                        for k, v in raw_ctx.items():
                            if not isinstance(v, dict):
                                continue
                            state.context_stats[k] = {
                                "signals": float(v.get("signals", 0)),
                                "successes": float(v.get("successes", 0)),
                                "repairs": float(v.get("repairs", 0)),
                                "budget_fails": float(v.get("budget_fails", 0)),
                                "verify_fails": float(v.get("verify_fails", 0)),
                                "with_symbol": float(v.get("with_symbol", 0)),
                                "with_symbol_success": float(v.get("with_symbol_success", 0)),
                                "with_tests": float(v.get("with_tests", 0)),
                                "with_tests_success": float(v.get("with_tests_success", 0)),
                            }
                    # Restore bucket_metadata
                    raw_bm = data.get("bucket_metadata")
                    if isinstance(raw_bm, dict):
                        state.bucket_metadata = {
                            k: dict(v) for k, v in raw_bm.items()
                            if isinstance(v, dict)
                        }
                    state.last_updated = float(data.get("last_updated", 0.0))
                except Exception:
                    pass  # Keep existing state on per-lane error

    def get_style_summary(self, lane: str) -> dict[str, dict[str, float]]:
        """Return per-style success rates for a lane.

        Returns {style: {signals, successes, success_rate}} sorted by signal count desc.
        """
        with self._lock:
            state = self._states.get(lane)
            if not state:
                return {}
            result = {}
            for style, stats in state.style_stats.items():
                s = stats.get("signals", 0)
                result[style] = {
                    "signals": s,
                    "successes": stats.get("successes", 0),
                    "success_rate": stats.get("successes", 0) / s if s > 0 else 0.0,
                }
            return dict(sorted(result.items(), key=lambda x: -x[1]["signals"]))

    def get_recovery_summary(self) -> dict[str, dict[str, float]]:
        """Return fallback recovery stats per lane.

        Returns {lane: {total, successes, recovery_rate}}.
        Only includes lanes with at least one fallback signal.
        """
        with self._lock:
            result = {}
            for lane, state in self._states.items():
                if state.recovery_total_count > 0:
                    result[lane] = {
                        "total": state.recovery_total_count,
                        "successes": state.recovery_success_count,
                        "recovery_rate": state.recovery_success_rate,
                    }
            return result

    def get_confident_stats(self, lane: str) -> Optional[dict[str, Any]]:
        """Return execution stats only if signal count exceeds lane confidence threshold.

        Returns None when insufficient signals (cold-start protection).
        When returned, all rates are confidence-gated — they won't activate
        until enough evidence accumulates per lane.

        Threshold: planner=4, main_agent=5, fast_path=10 (weighted signals).
        """
        with self._lock:
            state = self._states.get(lane)
            if not state:
                return None
            min_signals = _EXEC_MIN_SIGNALS.get(lane, 5)
            if state.signal_count < min_signals:
                return None
            return {
                "lane": lane,
                "signal_count": state.signal_count,
                "success_rate": state.success_rate,
                "repair_rate": state.repair_rate,
                "avg_repair_rounds": state.avg_repair_rounds,
                "budget_exhaust_rate": state.budget_exhaust_rate,
                "verify_fail_rate": state.verify_fail_rate,
                "recovery_success_rate": state.recovery_success_rate,
                "confident": True,
            }

    # Context-lane blending: context_weight = min(ctx_signals / _CTX_FULL_WEIGHT_AT, _CTX_MAX_WEIGHT)
    # Below _CTX_MIN_SIGNALS → pure lane rates; above _CTX_FULL_WEIGHT_AT → max context weight
    _CTX_MIN_SIGNALS: float = 3.0
    _CTX_FULL_WEIGHT_AT: float = 10.0
    _CTX_MAX_WEIGHT: float = 0.7     # Never 100% context — always blend lane as regularizer

    def get_execution_bias(
        self,
        lane: str,
        context_bucket: str = "",
    ) -> dict[str, float]:
        """Return score adjustments derived from execution history.

        Returns empty dict when insufficient signals (no bias applied).

        Blending policy (context ↔ lane partial sharing):
        - context signals < 3  → 100% lane rates
        - context signals 3-10 → linearly increasing context weight (30%-70%)
        - context signals ≥ 10 → 70% context + 30% lane (never 100% context)

        This prevents data sparsity from causing wild swings while still
        allowing context-specific learning to dominate when sufficient data exists.

        Data-driven context refinement:
        If a heuristic bucket (e.g. "single_file") has repair_rate > 60%
        (much worse than lane average), the effective context is promoted to
        "complex_edit" — signaling that the heuristic underestimates difficulty.
        """
        lane_stats = self.get_confident_stats(lane)
        if lane_stats is None:
            return {}

        # Lane-level rates (baseline)
        l_repair = lane_stats["repair_rate"]
        l_budget = lane_stats["budget_exhaust_rate"]
        l_verify = lane_stats["verify_fail_rate"]
        l_success = lane_stats["success_rate"]

        # Blended rates start as lane-level
        repair_rate = l_repair
        budget_rate = l_budget
        verify_rate = l_verify
        success_rate = l_success
        _ctx_source = "lane"
        _ctx_weight = 0.0

        if context_bucket:
            with self._lock:
                state = self._states.get(lane)
                if state:
                    # Check derived tier buckets: prefer heavy > medium > light > base
                    _effective_bucket = context_bucket
                    for _tier in ("heavy", "medium", "light"):
                        _candidate = f"{context_bucket}_{_tier}"
                        _cand_cs = state.context_stats.get(_candidate)
                        if _cand_cs and _cand_cs.get("signals", 0) >= self._CTX_MIN_SIGNALS:
                            _effective_bucket = _candidate
                            break  # Use the most informative tier with sufficient data

                    cs = state.context_stats.get(_effective_bucket)
                    if cs and cs.get("signals", 0) >= self._CTX_MIN_SIGNALS:
                        _s = cs["signals"]
                        c_repair = cs.get("repairs", 0) / _s
                        c_budget = cs.get("budget_fails", 0) / _s
                        c_success = cs.get("successes", 0) / _s

                        # Progressive blending
                        _ctx_weight = min(
                            _s / self._CTX_FULL_WEIGHT_AT,
                            self._CTX_MAX_WEIGHT,
                        )
                        _lane_weight = 1.0 - _ctx_weight

                        repair_rate = _ctx_weight * c_repair + _lane_weight * l_repair
                        budget_rate = _ctx_weight * c_budget + _lane_weight * l_budget
                        success_rate = _ctx_weight * c_success + _lane_weight * l_success
                        _ctx_source = _effective_bucket
                        if _effective_bucket != context_bucket:
                            _ctx_source = f"{context_bucket}→{_effective_bucket}"

        bias: dict[str, float] = {}

        if repair_rate > 0.5:
            bias["repair_penalty"] = -0.03
        if budget_rate > 0.3:
            bias["budget_penalty"] = -0.02
        if verify_rate > 0.4:
            bias["verify_penalty"] = -0.02
        if success_rate > 0.8:
            bias["success_bonus"] = 0.02
        if lane_stats.get("recovery_success_rate", 0) > 0.6:
            bias["recovery_confidence"] = 0.01

        # Context feature micro-adjustments (from with_symbol / with_tests stats)
        if context_bucket and _ctx_source != "lane":
            with self._lock:
                state = self._states.get(lane)
                if state:
                    _eff_cs = state.context_stats.get(_effective_bucket if '_effective_bucket' in dir() else context_bucket) or {}
                    # Symbol-targeted tasks: if success rate with symbols < without → penalty
                    _ws = _eff_cs.get("with_symbol", 0)
                    if _ws >= 3.0:
                        _ws_rate = _eff_cs.get("with_symbol_success", 0) / _ws
                        _total_rate = _eff_cs.get("successes", 0) / max(_eff_cs.get("signals", 1), 1)
                        if _ws_rate < _total_rate - 0.15:
                            bias["symbol_difficulty"] = -0.01
                    # Tasks with tests: if test-present runs fail more → signal harder verification
                    _wt = _eff_cs.get("with_tests", 0)
                    if _wt >= 3.0:
                        _wt_rate = _eff_cs.get("with_tests_success", 0) / _wt
                        _total_rate = _eff_cs.get("successes", 0) / max(_eff_cs.get("signals", 1), 1)
                        if _wt_rate < _total_rate - 0.15:
                            bias["test_difficulty"] = -0.01

        if bias:
            bias["_context_source"] = 0.0  # metadata (ignored in sum)
            logger.debug(
                "execution_bias: lane=%s ctx=%s source=%s weight=%.2f bias=%s",
                lane, context_bucket, _ctx_source, _ctx_weight, bias,
            )

        return bias

    def get_total_signals(self) -> float:
        """Return total weighted signal count across all lanes."""
        with self._lock:
            return sum(s.signal_count for s in self._states.values())


# ---------------------------------------------------------------------------
# StrategyPolicyLearner — reward-based strategy selection policy
# ---------------------------------------------------------------------------

def compute_reward(signal: ExecutionLearningSignal) -> float:
    """Compute scalar reward from an execution signal.

    Reward range: approximately [-2, +2].
    Components: success/fail base ± repair cost ± failure severity
                ± speed bonus ± alignment bonus ± replan penalty.

    P8: alignment_score and termination_decision contribute to reward
    when available (alignment_score >= 0).
    """
    r = 1.0 if signal.success else -1.0
    r -= 0.2 * min(signal.repair_rounds, 5)
    if signal.had_test_failure or signal.had_semantic_failure:
        r -= 0.5
    if signal.had_no_progress:
        r -= 0.3
    if signal.had_budget_failure:
        r -= 0.3
    if signal.success and signal.repair_rounds == 0:
        r += 0.2  # fast success bonus
    if signal.is_fallback and signal.success:
        r += 0.3  # recovery bonus

    # P8: Alignment-based reward shaping
    if signal.alignment_score >= 0:
        # Alignment bonus: high alignment → better reward even on "success"
        # This differentiates "clean success" from "barely passed"
        r += 0.5 * signal.alignment_score  # max +0.5

        # Replan penalty: each replan = strategy wasn't good enough
        r -= 0.2 * min(signal.replan_count, 3)

        # Termination decision bonus/penalty
        if signal.termination_decision == "SUCCESS":
            r += 0.1  # clean termination
        elif signal.termination_decision == "STOP":
            r -= 0.2  # forced stop

    return max(-2.0, min(2.0, r))


class StrategyPolicyLearner:
    """Reward-based strategy selection policy using Q-learning.

    Learns P(strategy | context_state) from execution rewards.
    Uses softmax policy with ε-greedy exploration.

    State: (context_bucket, has_symbol_target, has_tests)
    Action: strategy name
    Q-update: Q[s][a] ← Q[s][a] + α × (reward - Q[s][a])

    Thread-safe via threading.Lock.
    """

    _ALPHA: float = 0.1       # Learning rate
    _TAU: float = 0.8         # Softmax temperature (lower = more exploitative)
    _Q_CLIP: float = 2.0      # Q-value clamp range [-clip, +clip]
    _MIN_SIGNALS_FOR_POLICY: int = 5  # Minimum signals before policy activates
    _GAMMA: float = 0.7       # Discount factor for trajectory credit propagation
    _MAX_TRAJECTORY: int = 5  # Maximum trajectory length for backward propagation

    # Strategy lifecycle thresholds
    _LIFECYCLE_MIN_TRIALS: int = 5
    _PROMOTE_SUCCESS_RATE: float = 0.7
    _PROMOTE_AVG_REWARD: float = 0.3
    _DEPRECATE_SUCCESS_RATE: float = 0.3
    _DEPRECATE_AVG_REWARD: float = 0.0
    _DEPRECATED_SCORE_SCALE: float = 0.2   # soft pruning: score *= 0.2
    _EVAL_INTERVAL: int = 10               # evaluate every N updates
    # Strategies that are always "active" (never deprecated)
    _PROTECTED_STRATEGIES: frozenset = frozenset({"generic_create"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Q-table: {state_key: {strategy: q_value}}
        self._q_table: dict[str, dict[str, float]] = {}
        # Signal counts per state for exploration decay
        self._state_counts: dict[str, int] = {}
        # Per-strategy performance tracking: {strategy: {trials, successes, total_reward}}
        self._strategy_perf: dict[str, dict[str, float]] = {}
        # Strategy lifecycle status: {strategy: "candidate"|"active"|"deprecated"}
        self._strategy_status: dict[str, str] = {}
        # Context-specific status: {strategy: {context: status}}
        # Falls back to global _strategy_status when context-specific not available
        self._context_strategy_status: dict[str, dict[str, str]] = {}
        # Per-strategy per-context performance: {strategy: {context: {trials, successes, total_reward}}}
        self._context_strategy_perf: dict[str, dict[str, dict[str, float]]] = {}
        # Total updates counter for periodic evaluation
        self._total_updates: int = 0
        # Failure→strategy stats: {(failure_type, context): {strategy: {trials, successes}}}
        self._failure_strategy_stats: dict[str, dict[str, dict[str, float]]] = {}
        # Planner quality stats: {strategy: {trials, total_score}}
        self._planner_strategy_stats: dict[str, dict[str, float]] = {}
        # EMA-smoothed baselines for stable threshold tuning
        self._ema_context_baselines: dict[str, dict[str, float]] = {}
        self._ema_global_baseline: dict[str, float] = {"success_rate": 0.5, "avg_reward": 0.0}

    @staticmethod
    def _make_state_key(
        context_bucket: str,
        has_symbol_target: bool = False,
        has_tests: bool = False,
        scope: str = "small",
    ) -> str:
        """Create a hashable state key including task scope."""
        return f"{context_bucket}|sym={has_symbol_target}|test={has_tests}|scope={scope}"

    def update(
        self,
        signal: ExecutionLearningSignal,
        strategy: str,
    ) -> None:
        """Update Q-value for (state, strategy) using the execution reward."""
        if not strategy:
            return
        reward = compute_reward(signal)
        state_key = self._make_state_key(
            signal.context_bucket,
            signal.has_symbol_target,
            signal.has_tests,
            signal.scope,
        )
        with self._lock:
            q_row = self._q_table.setdefault(state_key, {})
            old_q = q_row.get(strategy, 0.0)
            new_q = old_q + self._ALPHA * (reward - old_q)
            q_row[strategy] = max(-self._Q_CLIP, min(self._Q_CLIP, new_q))
            self._state_counts[state_key] = self._state_counts.get(state_key, 0) + 1

            # Per-strategy performance tracking (global) — extended fields
            perf = self._strategy_perf.setdefault(
                strategy, {"trials": 0.0, "successes": 0.0, "total_reward": 0.0,
                           "total_replans": 0.0, "total_repairs": 0.0, "failure_counts": {}}
            )
            perf["trials"] += 1
            if signal.success:
                perf["successes"] += 1
            perf["total_reward"] += reward
            perf["total_repairs"] = perf.get("total_repairs", 0) + signal.repair_rounds
            if signal.is_fallback:
                perf["total_replans"] = perf.get("total_replans", 0) + 1
            if signal.failure_class:
                fc = perf.setdefault("failure_counts", {})
                fc[signal.failure_class] = fc.get(signal.failure_class, 0) + 1

            # Per-strategy per-context performance tracking — extended
            if signal.context_bucket and signal.context_bucket != "unknown":
                ctx_perf = self._context_strategy_perf.setdefault(strategy, {})
                cp = ctx_perf.setdefault(
                    signal.context_bucket, {"trials": 0.0, "successes": 0.0, "total_reward": 0.0,
                                            "total_replans": 0.0, "total_repairs": 0.0, "failure_counts": {}}
                )
                cp["trials"] += 1
                if signal.success:
                    cp["successes"] += 1
                cp["total_reward"] += reward
                cp["total_repairs"] = cp.get("total_repairs", 0) + signal.repair_rounds
                if signal.is_fallback:
                    cp["total_replans"] = cp.get("total_replans", 0) + 1
                if signal.failure_class:
                    fc = cp.setdefault("failure_counts", {})
                    fc[signal.failure_class] = fc.get(signal.failure_class, 0) + 1

            # Failure→strategy stats — extended with reward/repair
            if signal.failure_class and signal.failure_class != "":
                _fkey = f"{signal.failure_class}|{signal.context_bucket or 'global'}"
                _frow = self._failure_strategy_stats.setdefault(_fkey, {})
                _fs = _frow.setdefault(strategy, {"trials": 0.0, "successes": 0.0,
                                                   "total_reward": 0.0, "total_repairs": 0.0,
                                                   "total_replans": 0.0})
                _fs["trials"] += 1
                if signal.success:
                    _fs["successes"] += 1
                _fs["total_reward"] = _fs.get("total_reward", 0) + reward
                _fs["total_repairs"] = _fs.get("total_repairs", 0) + signal.repair_rounds
                if signal.is_fallback:
                    _fs["total_replans"] = _fs.get("total_replans", 0) + 1

            # Periodic lifecycle evaluation
            self._total_updates += 1
            if self._total_updates % self._EVAL_INTERVAL == 0:
                self._evaluate_strategies()

            logger.info(
                "strategy_perf: strategy=%s context=%s success=%s reward=%.2f "
                "Q=%.3f→%.3f trials=%d success_rate=%.0f%% status=%s",
                strategy, signal.context_bucket, signal.success, reward,
                old_q, q_row[strategy], int(perf["trials"]),
                (perf["successes"] / perf["trials"] * 100) if perf["trials"] > 0 else 0,
                self._strategy_status.get(strategy, "candidate"),
            )

    def get_strategy_performance(self) -> dict[str, dict[str, float]]:
        """Return per-strategy performance summary for analysis/logging.

        Returns {strategy: {trials, successes, success_rate, avg_reward}}.
        """
        with self._lock:
            result = {}
            for strat, perf in self._strategy_perf.items():
                t = perf.get("trials", 0)
                result[strat] = {
                    "trials": t,
                    "successes": perf.get("successes", 0),
                    "success_rate": perf.get("successes", 0) / t if t > 0 else 0.0,
                    "avg_reward": perf.get("total_reward", 0) / t if t > 0 else 0.0,
                }
            return dict(sorted(result.items(), key=lambda x: -x[1]["trials"]))

    def get_strategy_report(self) -> dict[str, Any]:
        """Comprehensive strategy performance report for analysis.

        Returns:
        - global: per-strategy summary (trials, success_rate, avg_reward, status)
        - by_context: strategy × context matrix
        - overlap_pairs: strategies with suspiciously similar performance
        - watchlist: strategies near promotion/deprecation thresholds
        - thresholds: current tuned thresholds per context
        """
        with self._lock:
            report: dict[str, Any] = {}

            # A. Global strategy report — extended
            def _summarize_perf(perf: dict[str, Any], status: str) -> dict[str, Any]:
                t = perf.get("trials", 0)
                return {
                    "trials": int(t),
                    "success_rate": round(perf.get("successes", 0) / t, 3) if t > 0 else 0.0,
                    "avg_reward": round(perf.get("total_reward", 0) / t, 3) if t > 0 else 0.0,
                    "avg_replan_count": round(perf.get("total_replans", 0) / t, 2) if t > 0 else 0.0,
                    "avg_repair_rounds": round(perf.get("total_repairs", 0) / t, 2) if t > 0 else 0.0,
                    "failure_distribution": dict(perf.get("failure_counts", {})),
                    "status": status,
                }

            global_report = {}
            for strat, perf in self._strategy_perf.items():
                global_report[strat] = _summarize_perf(
                    perf, self._strategy_status.get(strat, "candidate")
                )
            report["global"] = global_report

            # B. Context × strategy matrix — extended
            by_context: dict[str, dict[str, Any]] = {}
            for strat, ctx_map in self._context_strategy_perf.items():
                strat_ctx: dict[str, Any] = {}
                for ctx, cp in ctx_map.items():
                    ctx_status = (self._context_strategy_status.get(strat, {}).get(ctx)
                                  or self._strategy_status.get(strat, "candidate"))
                    strat_ctx[ctx] = _summarize_perf(cp, ctx_status)
                if strat_ctx:
                    by_context[strat] = strat_ctx
            report["by_context"] = by_context

            # C. Overlap detection
            overlap_pairs = []
            strats = list(self._strategy_perf.keys())
            for i, a in enumerate(strats):
                pa = self._strategy_perf[a]
                ta = pa.get("trials", 0)
                if ta < 10:
                    continue
                sra = pa.get("successes", 0) / ta
                ara = pa.get("total_reward", 0) / ta
                for b in strats[i + 1:]:
                    pb = self._strategy_perf[b]
                    tb = pb.get("trials", 0)
                    if tb < 10:
                        continue
                    srb = pb.get("successes", 0) / tb
                    arb = pb.get("total_reward", 0) / tb
                    if abs(sra - srb) < 0.05 and abs(ara - arb) < 0.10:
                        overlap_pairs.append({"a": a, "b": b,
                                              "sr_diff": round(abs(sra - srb), 3),
                                              "ar_diff": round(abs(ara - arb), 3)})
            report["overlap_pairs"] = overlap_pairs

            # D. Watchlist (near thresholds) — extended
            promotion_watch = []
            deprecation_watch = []
            for strat, perf in self._strategy_perf.items():
                t = perf.get("trials", 0)
                if t < 3 or strat in self._PROTECTED_STRATEGIES:
                    continue
                sr = perf.get("successes", 0) / t
                ar = perf.get("total_reward", 0) / t
                arep = perf.get("total_replans", 0) / t if t > 0 else 0
                arpr = perf.get("total_repairs", 0) / t if t > 0 else 0
                fc = perf.get("failure_counts", {})
                dom_fail = max(fc, key=fc.get) if fc else ""
                status = self._strategy_status.get(strat, "candidate")
                entry = {"strategy": strat, "trials": int(t),
                         "success_rate": round(sr, 3), "avg_reward": round(ar, 3),
                         "avg_replan_count": round(arep, 2), "avg_repair_rounds": round(arpr, 2),
                         "dominant_failure": dom_fail}
                if status != "active" and sr >= 0.6 and ar >= 0.2:
                    promotion_watch.append(entry)
                if status != "deprecated" and sr <= 0.4 and ar <= 0.1:
                    deprecation_watch.append(entry)
            report["watchlist"] = {"promotion": promotion_watch, "deprecation": deprecation_watch}

            # D2. Failure→strategy comparison
            failure_comparison: dict[str, dict[str, Any]] = {}
            for fkey, strat_map in self._failure_strategy_stats.items():
                fc_entry = {}
                for s, stats in strat_map.items():
                    t = stats.get("trials", 0)
                    fc_entry[s] = {
                        "trials": int(t),
                        "success_rate": round(stats.get("successes", 0) / t, 3) if t > 0 else 0.0,
                        "avg_reward": round(stats.get("total_reward", 0) / t, 3) if t > 0 else 0.0,
                        "avg_replan_count": round(stats.get("total_replans", 0) / t, 2) if t > 0 else 0.0,
                        "avg_repair_rounds": round(stats.get("total_repairs", 0) / t, 2) if t > 0 else 0.0,
                    }
                if fc_entry:
                    failure_comparison[fkey] = dict(
                        sorted(fc_entry.items(), key=lambda x: -x[1].get("success_rate", 0))
                    )
            report["failure_strategy_comparison"] = failure_comparison

            # E. Current tuned thresholds per context (using EMA baselines)
            observed_ctx = self._compute_context_baselines()
            ema_ctx = self._ema_context_baselines
            ema_gb = self._ema_global_baseline
            thresholds = {}
            for ctx in set(list(observed_ctx.keys()) + list(ema_ctx.keys())):
                psr, par, dsr, dar = self._get_tuned_thresholds(ctx, ema_ctx, ema_gb)
                obs = observed_ctx.get(ctx, {})
                ema = ema_ctx.get(ctx, {})
                thresholds[ctx] = {
                    "observed_baseline_success": round(obs.get("success_rate", 0), 3),
                    "ema_baseline_success": round(ema.get("success_rate", 0), 3),
                    "observed_baseline_reward": round(obs.get("avg_reward", 0), 3),
                    "ema_baseline_reward": round(ema.get("avg_reward", 0), 3),
                    "promote_success": round(psr, 3),
                    "promote_reward": round(par, 3),
                    "deprecate_success": round(dsr, 3),
                    "deprecate_reward": round(dar, 3),
                }
            report["thresholds"] = thresholds

            # F. Failure→strategy mapping (learned)
            failure_map = {}
            for fkey, strat_map in self._failure_strategy_stats.items():
                fm = {}
                for s, stats in strat_map.items():
                    t = stats.get("trials", 0)
                    fm[s] = {
                        "trials": int(t),
                        "success_rate": round(stats.get("successes", 0) / t, 3) if t > 0 else 0.0,
                    }
                if fm:
                    failure_map[fkey] = dict(sorted(fm.items(),
                        key=lambda x: -x[1].get("success_rate", 0)))
            report["failure_strategy_map"] = failure_map

            return report

    # ── Failure→strategy prior + planner quality ─────────────────────

    _PRIOR_LAMBDA: float = 0.3       # weight for log P(s|f,ctx) in combined score
    _PLANNER_LAMBDA: float = 0.2     # weight for planner_score in combined score
    _PRIOR_ALPHA: float = 1.0        # Laplace smoothing parameter
    _PRIOR_EPSILON: float = 1e-6     # log(0) prevention

    def get_failure_strategy_prior(
        self, failure_type: str, context_bucket: str, strategy: str,
        num_strategies: int = 4,
    ) -> float:
        """Return smoothed P(strategy | failure_type, context_bucket).

        Uses Laplace smoothing: P = (successes + α) / (total_trials + α × N).
        Returns uniform prior (1/N) when no data available.
        """
        with self._lock:
            key = f"{failure_type}|{context_bucket or 'global'}"
            strat_map = self._failure_strategy_stats.get(key, {})
            if not strat_map:
                return 1.0 / max(num_strategies, 1)
            total_trials = sum(s.get("trials", 0) for s in strat_map.values())
            stats = strat_map.get(strategy, {})
            successes = stats.get("successes", 0)
            alpha = self._PRIOR_ALPHA
            return (successes + alpha) / (total_trials + alpha * num_strategies)

    def update_planner_quality(
        self, strategy: str, success: bool, replan_count: int, repair_rounds: int,
    ) -> None:
        """Record planner quality for a strategy execution."""
        score = (1.0 if success else 0.0) - 0.2 * replan_count - 0.1 * repair_rounds
        with self._lock:
            ps = self._planner_strategy_stats.setdefault(
                strategy, {"trials": 0.0, "total_score": 0.0}
            )
            ps["trials"] += 1
            ps["total_score"] += score

    def get_planner_score(self, strategy: str) -> float:
        """Return average planner quality score for a strategy."""
        with self._lock:
            ps = self._planner_strategy_stats.get(strategy, {})
            t = ps.get("trials", 0)
            return ps.get("total_score", 0) / t if t > 0 else 0.0

    def get_combined_replan_scores(
        self,
        failure_type: str,
        context_bucket: str,
        candidates: list[str],
        has_symbol_target: bool = False,
        has_tests: bool = False,
        scope: str = "small",
    ) -> dict[str, float]:
        """Return combined scores for replan strategy selection.

        score = Q + λ1×log(P) + λ2×planner_score + exploration

        Returns empty dict when insufficient data for all components.
        """
        import math

        q_scores = self.get_policy_scores(
            context_bucket, has_symbol_target, has_tests, candidates, scope=scope
        )

        # P1: Cold-start bias from _COLD_PRIORS (domain knowledge).
        # When q_scores is empty (insufficient signals), all candidates get
        # uniform P=1/N, making Q+logP identical.  _COLD_PRIORS.success_rate
        # breaks the tie based on expert priors (e.g. symbol_guided_create=0.85
        # vs generic_create=0.80).
        _cold_bias: dict[str, float] = {}
        if not q_scores:
            try:
                from external_llm.editor._editor_core.common.strategy_priors import _COLD_PRIORS
                for s in candidates:
                    _prior = _COLD_PRIORS.get(s, _COLD_PRIORS.get("generic_create", {}))
                    _cold_bias[s] = (_prior.get("success_rate", 0.5) - 0.5) * 0.1
            except ImportError:
                pass

        result = {}
        for s in candidates:
            q = q_scores.get(s, 0.0)
            p = self.get_failure_strategy_prior(
                failure_type, context_bucket, s, len(candidates)
            )
            ps = self.get_planner_score(s)
            cb = _cold_bias.get(s, 0.0)

            log_p = self._PRIOR_LAMBDA * math.log(p + self._PRIOR_EPSILON)
            planner_contrib = self._PLANNER_LAMBDA * ps
            combined = q + log_p + planner_contrib + cb
            result[s] = round(combined, 4)

            logger.info(
                "combined_score: strategy=%s Q=%.4f P=%.3f logP=%.4f "
                "planner=%.3f planner_contrib=%.4f cold_bias=%.4f combined=%.4f",
                s, q, p, log_p, ps, planner_contrib, cb, combined,
            )

        return result

    def is_deprecated(self, strategy: str, context_bucket: str = "") -> bool:
        """Check if a strategy is deprecated.

        Checks context-specific status first, then falls back to global.
        A strategy can be deprecated globally but active in a specific context.
        """
        with self._lock:
            # Context-specific check first
            if context_bucket:
                ctx_map = self._context_strategy_status.get(strategy, {})
                ctx_status = ctx_map.get(context_bucket)
                if ctx_status is not None:
                    return ctx_status == "deprecated"
            # Fall back to global status
            return self._strategy_status.get(strategy) == "deprecated"

    def get_strategy_status(self, strategy: str, context_bucket: str = "") -> str:
        """Return lifecycle status with context-aware fallback.

        Priority: context-specific → global → "candidate".
        """
        with self._lock:
            if context_bucket:
                ctx_map = self._context_strategy_status.get(strategy, {})
                ctx_status = ctx_map.get(context_bucket)
                if ctx_status is not None:
                    return ctx_status
            return self._strategy_status.get(strategy, "candidate")

    def _evaluate_strategies(self) -> None:
        """Evaluate all strategies and update lifecycle status.

        Called periodically (every _EVAL_INTERVAL updates) under self._lock.

        Rules:
        - trials < MIN_TRIALS → stay as "candidate" (no judgement)
        - success_rate ≥ 0.7 AND avg_reward ≥ 0.3 → "active"
        - success_rate ≤ 0.3 AND avg_reward ≤ 0.0 → "deprecated"
        - deprecated + (later exploration shows improvement) → "candidate" (recovery)
        - protected strategies (generic_create) → always "active"
        """
        for strat, perf in self._strategy_perf.items():
            t = perf.get("trials", 0)
            if t < self._LIFECYCLE_MIN_TRIALS:
                if strat not in self._strategy_status:
                    self._strategy_status[strat] = "candidate"
                continue

            sr = perf.get("successes", 0) / t if t > 0 else 0.0
            ar = perf.get("total_reward", 0) / t if t > 0 else 0.0
            current = self._strategy_status.get(strat, "candidate")

            if strat in self._PROTECTED_STRATEGIES:
                self._strategy_status[strat] = "active"
                continue

            if sr >= self._PROMOTE_SUCCESS_RATE and ar >= self._PROMOTE_AVG_REWARD:
                if current != "active":
                    self._strategy_status[strat] = "active"
                    logger.info(
                        "strategy_lifecycle: %s promoted to active "
                        "(success=%.0f%% reward=%.2f trials=%d)",
                        strat, sr * 100, ar, int(t),
                    )
            elif sr <= self._DEPRECATE_SUCCESS_RATE and ar <= self._DEPRECATE_AVG_REWARD:
                if current != "deprecated":
                    self._strategy_status[strat] = "deprecated"
                    logger.info(
                        "strategy_lifecycle: %s deprecated "
                        "(success=%.0f%% reward=%.2f trials=%d)",
                        strat, sr * 100, ar, int(t),
                    )
            elif current == "deprecated" and sr > self._DEPRECATE_SUCCESS_RATE:
                self._strategy_status[strat] = "candidate"
                logger.info(
                    "strategy_lifecycle: %s recovered to candidate "
                    "(success=%.0f%% reward=%.2f)",
                    strat, sr * 100, ar,
                )

        # Update EMA baselines, then use smoothed values for threshold tuning
        observed_ctx = self._compute_context_baselines()
        observed_global = self._compute_global_baseline()
        self._update_ema_baselines(observed_ctx, observed_global)
        ctx_baselines = self._ema_context_baselines
        global_baseline = self._ema_global_baseline
        for strat, ctx_map in self._context_strategy_perf.items():
            if strat in self._PROTECTED_STRATEGIES:
                continue
            ctx_statuses = self._context_strategy_status.setdefault(strat, {})
            for ctx, cp in ctx_map.items():
                ct = cp.get("trials", 0)
                if ct < self._LIFECYCLE_MIN_TRIALS:
                    continue
                csr = cp.get("successes", 0) / ct if ct > 0 else 0.0
                car = cp.get("total_reward", 0) / ct if ct > 0 else 0.0
                old_status = ctx_statuses.get(ctx, "candidate")

                # Adaptive thresholds based on context difficulty
                prom_sr, prom_ar, depr_sr, depr_ar = self._get_tuned_thresholds(
                    ctx, ctx_baselines, global_baseline
                )

                if csr >= prom_sr and car >= prom_ar:
                    if old_status != "active":
                        ctx_statuses[ctx] = "active"
                        logger.info(
                            "strategy_lifecycle: %s@%s promoted to active "
                            "(success=%.0f%% reward=%.2f thresh=%.0f%%)",
                            strat, ctx, csr * 100, car, prom_sr * 100,
                        )
                elif csr <= depr_sr and car <= depr_ar:
                    if old_status != "deprecated":
                        ctx_statuses[ctx] = "deprecated"
                        logger.info(
                            "strategy_lifecycle: %s@%s deprecated "
                            "(success=%.0f%% reward=%.2f thresh=%.0f%%)",
                            strat, ctx, csr * 100, car, depr_sr * 100,
                        )
                elif old_status == "deprecated" and csr > depr_sr:
                    ctx_statuses[ctx] = "candidate"

    # EMA smoothing constants
    _EMA_ALPHA_CTX: float = 0.2     # context baseline EMA rate (more responsive)
    _EMA_ALPHA_GLOBAL: float = 0.1  # global baseline EMA rate (more stable)
    _BLEND_TARGET_TRIALS: float = 20.0  # trials at which context baseline is fully trusted

    # Threshold tuning constants
    _TUNE_K: float = 0.5           # sensitivity to context-global difference
    _PROMOTE_SR_MIN: float = 0.55
    _PROMOTE_SR_MAX: float = 0.85
    _DEPRECATE_SR_MIN: float = 0.15
    _DEPRECATE_SR_MAX: float = 0.45
    _PROMOTE_AR_MIN: float = 0.10
    _PROMOTE_AR_MAX: float = 0.50
    _DEPRECATE_AR_MIN: float = -0.30
    _DEPRECATE_AR_MAX: float = 0.15

    def _update_ema_baselines(
        self,
        observed_ctx: dict[str, dict[str, float]],
        observed_global: dict[str, float],
    ) -> None:
        """Update EMA-smoothed baselines from observed values.

        Pipeline: observed → sample-based blending with global → EMA smoothing.
        """
        # Update global EMA
        a_g = self._EMA_ALPHA_GLOBAL
        for key in ("success_rate", "avg_reward"):
            old = self._ema_global_baseline.get(key, observed_global.get(key, 0.0))
            new = observed_global.get(key, old)
            self._ema_global_baseline[key] = old * (1 - a_g) + new * a_g

        # Update per-context EMA with sample-based blending
        a_c = self._EMA_ALPHA_CTX
        for ctx, obs in observed_ctx.items():
            # Sample-based blending: trust context more with more trials
            ctx_trials = 0.0
            for _strat, ctx_map in self._context_strategy_perf.items():
                cp = ctx_map.get(ctx)
                if cp:
                    ctx_trials += cp.get("trials", 0)
            blend_w = min(1.0, ctx_trials / self._BLEND_TARGET_TRIALS)
            blended = {}
            for key in ("success_rate", "avg_reward"):
                ctx_val = obs.get(key, 0.0)
                global_val = self._ema_global_baseline.get(key, 0.0)
                blended[key] = blend_w * ctx_val + (1.0 - blend_w) * global_val

            # EMA update
            ema = self._ema_context_baselines.get(ctx)
            if ema is None:
                self._ema_context_baselines[ctx] = dict(blended)
            else:
                for key in ("success_rate", "avg_reward"):
                    old = ema.get(key, blended[key])
                    ema[key] = old * (1 - a_c) + blended[key] * a_c

    def _compute_context_baselines(self) -> dict[str, dict[str, float]]:
        """Compute per-context average success_rate and avg_reward across all strategies."""
        baselines: dict[str, dict[str, float]] = {}
        # Aggregate all strategies' context perf
        ctx_totals: dict[str, dict[str, float]] = {}
        for _strat, ctx_map in self._context_strategy_perf.items():
            for ctx, cp in ctx_map.items():
                agg = ctx_totals.setdefault(ctx, {"trials": 0.0, "successes": 0.0, "reward": 0.0})
                agg["trials"] += cp.get("trials", 0)
                agg["successes"] += cp.get("successes", 0)
                agg["reward"] += cp.get("total_reward", 0)
        for ctx, agg in ctx_totals.items():
            t = agg["trials"]
            if t >= self._LIFECYCLE_MIN_TRIALS:
                baselines[ctx] = {
                    "success_rate": agg["successes"] / t,
                    "avg_reward": agg["reward"] / t,
                }
        return baselines

    def _compute_global_baseline(self) -> dict[str, float]:
        """Compute global average success_rate and avg_reward across all strategies."""
        total_t = sum(p.get("trials", 0) for p in self._strategy_perf.values())
        if total_t < self._LIFECYCLE_MIN_TRIALS:
            return {"success_rate": 0.5, "avg_reward": 0.0}
        total_s = sum(p.get("successes", 0) for p in self._strategy_perf.values())
        total_r = sum(p.get("total_reward", 0) for p in self._strategy_perf.values())
        return {
            "success_rate": total_s / total_t,
            "avg_reward": total_r / total_t,
        }

    def _get_tuned_thresholds(
        self,
        context: str,
        ctx_baselines: dict[str, dict[str, float]],
        global_baseline: dict[str, float],
    ) -> tuple[float, float, float, float]:
        """Return (promote_sr, promote_ar, deprecate_sr, deprecate_ar) tuned for context.

        Easy context → higher thresholds (harder to promote, easier to deprecate).
        Hard context → lower thresholds (easier to promote, harder to deprecate).
        Falls back to fixed thresholds when context baseline unavailable.
        """
        cb = ctx_baselines.get(context)
        if cb is None:
            return (self._PROMOTE_SUCCESS_RATE, self._PROMOTE_AVG_REWARD,
                    self._DEPRECATE_SUCCESS_RATE, self._DEPRECATE_AVG_REWARD)

        gb = global_baseline
        sr_diff = cb["success_rate"] - gb["success_rate"]
        ar_diff = cb["avg_reward"] - gb["avg_reward"]

        prom_sr = max(self._PROMOTE_SR_MIN, min(self._PROMOTE_SR_MAX,
                      self._PROMOTE_SUCCESS_RATE + self._TUNE_K * sr_diff))
        prom_ar = max(self._PROMOTE_AR_MIN, min(self._PROMOTE_AR_MAX,
                      self._PROMOTE_AVG_REWARD + self._TUNE_K * ar_diff))
        depr_sr = max(self._DEPRECATE_SR_MIN, min(self._DEPRECATE_SR_MAX,
                      self._DEPRECATE_SUCCESS_RATE + self._TUNE_K * sr_diff))
        depr_ar = max(self._DEPRECATE_AR_MIN, min(self._DEPRECATE_AR_MAX,
                      self._DEPRECATE_AVG_REWARD + self._TUNE_K * ar_diff))

        return prom_sr, prom_ar, depr_sr, depr_ar

    def update_trajectory(
        self,
        steps: list[tuple["ExecutionLearningSignal", str]],
    ) -> None:
        """Update Q-values using backward credit propagation over a trajectory.

        Each step is (signal, strategy). Computes discounted return:
            R_t = r_t + γ × R_{t+1}

        For single-step trajectories, falls back to immediate reward (same as update()).
        For multi-step (replan/fallback), earlier steps receive partial credit
        for the eventual outcome.

        Parameters
        ----------
        steps:
            List of (ExecutionLearningSignal, strategy_name) in chronological order.
            Truncated to _MAX_TRAJECTORY if longer.
        """
        if not steps:
            return
        # Truncate
        steps = steps[-self._MAX_TRAJECTORY:]

        # Single-step: use immediate reward (no propagation needed)
        if len(steps) == 1:
            self.update(steps[0][0], steps[0][1])
            return

        # Compute immediate rewards
        rewards = [compute_reward(sig) for sig, _ in steps]

        # Backward propagation: R_t = r_t + γ × R_{t+1}
        returns = [0.0] * len(steps)
        returns[-1] = rewards[-1]
        for t in range(len(steps) - 2, -1, -1):
            returns[t] = rewards[t] + self._GAMMA * returns[t + 1]

        # Clamp returns
        returns = [max(-self._Q_CLIP, min(self._Q_CLIP, r)) for r in returns]

        # Update Q for each step
        for (signal, strategy), R_t in zip(steps, returns, strict=False):
            if not strategy:
                continue
            state_key = self._make_state_key(
                signal.context_bucket,
                signal.has_symbol_target,
                signal.has_tests,
                signal.scope,
            )
            with self._lock:
                q_row = self._q_table.setdefault(state_key, {})
                old_q = q_row.get(strategy, 0.0)
                new_q = old_q + self._ALPHA * (R_t - old_q)
                q_row[strategy] = max(-self._Q_CLIP, min(self._Q_CLIP, new_q))
                self._state_counts[state_key] = self._state_counts.get(state_key, 0) + 1

            logger.debug(
                "policy_trajectory: state=%s strategy=%s r=%.2f R=%.2f Q=%.3f→%.3f",
                state_key, strategy, compute_reward(signal), R_t, old_q,
                q_row[strategy],
            )

    def get_policy_scores(
        self,
        context_bucket: str,
        has_symbol_target: bool = False,
        has_tests: bool = False,
        strategies: Optional[list[str]] = None,
        scope: str = "small",
    ) -> dict[str, float]:
        """Return Q-based score adjustments for each strategy.

        Returns empty dict when insufficient signals (cold-start).
        Includes ε-greedy exploration bonus for under-explored strategies.

        Returned values are additive adjustments to existing strategy scores,
        not raw probabilities — this preserves compatibility with the existing
        scoring pipeline (strategy_memory + execution_bias + policy_score).
        """
        state_key = self._make_state_key(context_bucket, has_symbol_target, has_tests, scope)
        with self._lock:
            n = self._state_counts.get(state_key, 0)
            if n < self._MIN_SIGNALS_FOR_POLICY:
                return {}

            q_row = self._q_table.get(state_key, {})
            if not q_row:
                return {}

            # Compute softmax probabilities with exploration bonus
            import math
            all_strategies = strategies or list(q_row.keys())
            q_values = [q_row.get(s, 0.0) for s in all_strategies]

            # Exploration bonus: under-explored strategies get a boost
            # ε = max(0.1, 1/√n) ensures new strategies get tried
            _epsilon = max(0.1, 1.0 / (n ** 0.5)) if n > 0 else 1.0
            _per_strat_counts = {}
            for sig_state, sig_row in self._q_table.items():
                if sig_state == state_key:
                    for s_name in sig_row:
                        _per_strat_counts[s_name] = _per_strat_counts.get(s_name, 0) + 1
            explore_bonuses = []
            for s in all_strategies:
                sc = _per_strat_counts.get(s, 0)
                # UCB-style bonus: higher for less-explored strategies
                bonus = _epsilon * (1.0 / (1 + sc) ** 0.5) if sc < n else 0.0
                explore_bonuses.append(bonus)

            # Softmax on Q + exploration bonus
            adjusted_q = [q + eb for q, eb in zip(q_values, explore_bonuses, strict=False)]
            max_q = max(adjusted_q) if adjusted_q else 0.0
            exp_vals = [math.exp((q - max_q) / self._TAU) for q in adjusted_q]
            sum_exp = sum(exp_vals) or 1.0
            probs = [e / sum_exp for e in exp_vals]

            # Convert probabilities to score adjustments
            avg_prob = 1.0 / len(all_strategies) if all_strategies else 0.5
            scores = {}
            for s, p in zip(all_strategies, probs, strict=False):
                raw_score = round((p - avg_prob) * 0.10, 4)
                # Soft pruning: deprecated strategies get scaled down
                if self._strategy_status.get(s) == "deprecated":
                    raw_score = round(raw_score * self._DEPRECATED_SCORE_SCALE, 4)
                scores[s] = raw_score

            return scores

    def get_summary(self) -> dict[str, Any]:
        """Return JSON-serialisable summary for persistence."""
        with self._lock:
            return {
                "q_table": {k: dict(v) for k, v in self._q_table.items()},
                "state_counts": dict(self._state_counts),
                "strategy_perf": {k: dict(v) for k, v in self._strategy_perf.items()},
                "strategy_status": dict(self._strategy_status),
                "context_strategy_status": {
                    k: dict(v) for k, v in self._context_strategy_status.items()
                },
                "context_strategy_perf": {
                    strat: {ctx: dict(cp) for ctx, cp in ctx_map.items()}
                    for strat, ctx_map in self._context_strategy_perf.items()
                },
                "ema_context_baselines": {k: dict(v) for k, v in self._ema_context_baselines.items()},
                "ema_global_baseline": dict(self._ema_global_baseline),
                "failure_strategy_stats": {
                    k: {s: dict(v) for s, v in sm.items()}
                    for k, sm in self._failure_strategy_stats.items()
                },
                "planner_strategy_stats": {
                    k: dict(v) for k, v in self._planner_strategy_stats.items()
                },
                "total_updates": self._total_updates,
            }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Restore from persisted state."""
        with self._lock:
            raw_q = state_dict.get("q_table")
            if isinstance(raw_q, dict):
                self._q_table = {
                    k: {sk: float(sv) for sk, sv in v.items()}
                    for k, v in raw_q.items()
                    if isinstance(v, dict)
                }
            raw_counts = state_dict.get("state_counts")
            if isinstance(raw_counts, dict):
                self._state_counts = {
                    k: int(v) for k, v in raw_counts.items()
                }
            raw_perf = state_dict.get("strategy_perf")
            if isinstance(raw_perf, dict):
                self._strategy_perf = {
                    k: {
                        "trials": float(v.get("trials", 0)),
                        "successes": float(v.get("successes", 0)),
                        "total_reward": float(v.get("total_reward", 0)),
                    }
                    for k, v in raw_perf.items()
                    if isinstance(v, dict)
                }
            raw_status = state_dict.get("strategy_status")
            if isinstance(raw_status, dict):
                self._strategy_status = {
                    k: str(v) for k, v in raw_status.items()
                    if v in ("candidate", "active", "deprecated")
                }
            raw_ctx_status = state_dict.get("context_strategy_status")
            if isinstance(raw_ctx_status, dict):
                self._context_strategy_status = {
                    strat: {
                        ctx: str(s) for ctx, s in ctx_map.items()
                        if s in ("candidate", "active", "deprecated")
                    }
                    for strat, ctx_map in raw_ctx_status.items()
                    if isinstance(ctx_map, dict)
                }
            raw_ctx_perf = state_dict.get("context_strategy_perf")
            if isinstance(raw_ctx_perf, dict):
                self._context_strategy_perf = {
                    strat: {
                        ctx: {
                            "trials": float(cp.get("trials", 0)),
                            "successes": float(cp.get("successes", 0)),
                            "total_reward": float(cp.get("total_reward", 0)),
                        }
                        for ctx, cp in ctx_map.items()
                        if isinstance(cp, dict)
                    }
                    for strat, ctx_map in raw_ctx_perf.items()
                    if isinstance(ctx_map, dict)
                }
            raw_ema_ctx = state_dict.get("ema_context_baselines")
            if isinstance(raw_ema_ctx, dict):
                self._ema_context_baselines = {
                    k: {sk: float(sv) for sk, sv in v.items()}
                    for k, v in raw_ema_ctx.items()
                    if isinstance(v, dict)
                }
            raw_ema_global = state_dict.get("ema_global_baseline")
            if isinstance(raw_ema_global, dict):
                self._ema_global_baseline = {
                    k: float(v) for k, v in raw_ema_global.items()
                }
            raw_ps = state_dict.get("planner_strategy_stats")
            if isinstance(raw_ps, dict):
                self._planner_strategy_stats = {
                    k: {"trials": float(v.get("trials", 0)), "total_score": float(v.get("total_score", 0))}
                    for k, v in raw_ps.items() if isinstance(v, dict)
                }
            raw_fs = state_dict.get("failure_strategy_stats")
            if isinstance(raw_fs, dict):
                self._failure_strategy_stats = {
                    k: {
                        s: {"trials": float(v.get("trials", 0)), "successes": float(v.get("successes", 0))}
                        for s, v in sm.items() if isinstance(v, dict)
                    }
                    for k, sm in raw_fs.items() if isinstance(sm, dict)
                }
            self._total_updates = int(state_dict.get("total_updates", 0))

    # ── Model-aware knowledge transfer ─────────────────────────────────

    def get_transferable_state(self) -> dict[str, Any]:
        """Return knowledge that transfers across models.

        Transferable: structural insights about which strategies work for which
        failures/contexts. NOT transferable: absolute Q-values, trial counts,
        planner quality (all model-dependent).
        """
        with self._lock:
            return {
                # Failure→strategy mapping: "EXEC_FAIL → symbol_guided works"
                "failure_strategy_stats": {
                    k: {s: dict(v) for s, v in sm.items()}
                    for k, sm in self._failure_strategy_stats.items()
                },
                # Context difficulty baselines: "multi_file is harder"
                "ema_context_baselines": {
                    k: dict(v) for k, v in self._ema_context_baselines.items()
                },
                "ema_global_baseline": dict(self._ema_global_baseline),
                # Strategy lifecycle: "test_aware is active, X is deprecated"
                "strategy_status": dict(self._strategy_status),
                "context_strategy_status": {
                    k: dict(v) for k, v in self._context_strategy_status.items()
                },
            }

    def load_transferable_state(self, state: dict[str, Any]) -> None:
        """Load transferable knowledge from another model's experience.

        Only loads structural insights. Does NOT load Q-values, trial counts,
        or planner quality — those must be learned fresh per model.

        Lifecycle statuses are downgraded: "active" → "candidate" (must re-earn
        active status with the new model), "deprecated" stays deprecated
        (structural insight: this strategy doesn't work for this context).
        """
        with self._lock:
            # Failure→strategy prior (transferable: approach effectiveness)
            raw_fs = state.get("failure_strategy_stats")
            if isinstance(raw_fs, dict) and not self._failure_strategy_stats:
                self._failure_strategy_stats = {
                    k: {
                        s: {"trials": float(v.get("trials", 0)),
                            "successes": float(v.get("successes", 0)),
                            "total_reward": float(v.get("total_reward", 0)),
                            "total_repairs": float(v.get("total_repairs", 0)),
                            "total_replans": float(v.get("total_replans", 0))}
                        for s, v in sm.items() if isinstance(v, dict)
                    }
                    for k, sm in raw_fs.items() if isinstance(sm, dict)
                }
                logger.info("transfer: loaded failure_strategy_stats from prior model")

            # Context baselines (transferable: difficulty assessment)
            raw_ema = state.get("ema_context_baselines")
            if isinstance(raw_ema, dict) and not self._ema_context_baselines:
                self._ema_context_baselines = {
                    k: {sk: float(sv) for sk, sv in v.items()}
                    for k, v in raw_ema.items() if isinstance(v, dict)
                }
                logger.info("transfer: loaded ema_context_baselines from prior model")

            raw_gb = state.get("ema_global_baseline")
            if isinstance(raw_gb, dict):
                self._ema_global_baseline = {
                    k: float(v) for k, v in raw_gb.items()
                }

            # Lifecycle: deprecated stays, active → candidate (must re-earn)
            raw_status = state.get("strategy_status")
            if isinstance(raw_status, dict) and not self._strategy_status:
                for strat, status in raw_status.items():
                    if status == "deprecated":
                        self._strategy_status[strat] = "deprecated"
                    # active → don't transfer (new model must prove it)
                logger.info("transfer: loaded deprecated strategies from prior model")

            raw_ctx_status = state.get("context_strategy_status")
            if isinstance(raw_ctx_status, dict) and not self._context_strategy_status:
                for strat, ctx_map in raw_ctx_status.items():
                    if not isinstance(ctx_map, dict):
                        continue
                    for ctx, status in ctx_map.items():
                        if status == "deprecated":
                            self._context_strategy_status.setdefault(strat, {})[ctx] = "deprecated"


# ---------------------------------------------------------------------------
# AdaptiveLearnerHub — unified learning for tool/patch/context/routing/prompt
# ---------------------------------------------------------------------------

class MiniQLearner:
    """Lightweight Q-table learner for a single decision domain.

    Shared pattern: state×action → Q-value, with EMA update and persistence.
    Thread-safe via external lock (caller must hold).
    """

    def __init__(self, name: str, alpha: float = 0.1, min_signals: int = 5):
        self.name = name
        self._alpha = alpha
        self._min_signals = min_signals
        self._q: dict[str, dict[str, float]] = {}        # {state: {action: Q}}
        self._counts: dict[str, dict[str, int]] = {}     # {state: {action: count}}
        self._perf: dict[str, dict[str, float]] = {}     # {action: {trials, successes, total_reward}}

    def update(self, state: str, action: str, reward: float) -> None:
        q_row = self._q.setdefault(state, {})
        old_q = q_row.get(action, 0.0)
        q_row[action] = old_q + self._alpha * (reward - old_q)

        c_row = self._counts.setdefault(state, {})
        c_row[action] = c_row.get(action, 0) + 1

        perf = self._perf.setdefault(action, {"trials": 0.0, "successes": 0.0, "total_reward": 0.0})
        perf["trials"] += 1
        if reward > 0:
            perf["successes"] += 1
        perf["total_reward"] += reward

    def get_scores(self, state: str, actions: list[str]) -> dict[str, float]:
        """Return Q-based scores. Empty if insufficient data."""
        c_row = self._counts.get(state, {})
        total = sum(c_row.values())
        if total < self._min_signals:
            return {}
        q_row = self._q.get(state, {})
        return {a: round(q_row.get(a, 0.0), 4) for a in actions}

    def get_best(self, state: str, actions: list[str], default: str = "") -> str:
        """Return best action by Q, or default if insufficient data."""
        scores = self.get_scores(state, actions)
        if not scores:
            return default
        return max(actions, key=lambda a: scores.get(a, float("-inf")))

    def get_report(self) -> dict[str, Any]:
        result = {}
        for action, perf in sorted(self._perf.items(), key=lambda x: -x[1].get("trials", 0)):
            t = perf.get("trials", 0)
            result[action] = {
                "trials": int(t),
                "success_rate": round(perf.get("successes", 0) / t, 3) if t > 0 else 0.0,
                "avg_reward": round(perf.get("total_reward", 0) / t, 3) if t > 0 else 0.0,
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": {k: dict(v) for k, v in self._q.items()},
            "counts": {k: dict(v) for k, v in self._counts.items()},
            "perf": {k: dict(v) for k, v in self._perf.items()},
        }

    def load_dict(self, d: dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        raw_q = d.get("q")
        if isinstance(raw_q, dict):
            self._q = {k: {ak: float(av) for ak, av in v.items()}
                       for k, v in raw_q.items() if isinstance(v, dict)}
        raw_c = d.get("counts")
        if isinstance(raw_c, dict):
            self._counts = {k: {ak: int(av) for ak, av in v.items()}
                           for k, v in raw_c.items() if isinstance(v, dict)}
        raw_p = d.get("perf")
        if isinstance(raw_p, dict):
            self._perf = {k: {"trials": float(v.get("trials", 0)),
                              "successes": float(v.get("successes", 0)),
                              "total_reward": float(v.get("total_reward", 0))}
                         for k, v in raw_p.items() if isinstance(v, dict)}


class AdaptiveLearnerHub:
    """Unified learning hub for 5 decision domains.

    1. tool_learner:    MAIN_AGENT tool selection (state=phase, action=tool)
    2. patch_learner:   Patch/repair method (state=failure_class|file_type, action=method)
    3. context_learner: Context budget (state=task_type, action=context_config)
    4. routing_learner: Lane routing (state=request_features, action=lane)
    5. prompt_learner:  Prompt variant (state=strategy, action=prompt_variant)

    Thread-safe. Persistence via get_summary() / load_state().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tool_learner = MiniQLearner("tool", alpha=0.1, min_signals=10)
        self.patch_learner = MiniQLearner("patch", alpha=0.15, min_signals=5)
        self.context_learner = MiniQLearner("context", alpha=0.1, min_signals=8)
        self.routing_learner = MiniQLearner("routing", alpha=0.1, min_signals=10)
        self.prompt_learner = MiniQLearner("prompt", alpha=0.1, min_signals=5)

    # ── 1. Tool selection learning ──────────────────────────────

    def record_tool_usage(
        self, phase: str, tool_name: str, success: bool, context_bucket: str = "",
    ) -> None:
        """Record tool usage outcome in MAIN_AGENT lane."""
        reward = 0.5 if success else -0.3
        state = f"{context_bucket}|{phase}" if context_bucket else phase
        with self._lock:
            self.tool_learner.update(state, tool_name, reward)

    # ── 2. Patch method learning ────────────────────────────────

    def record_patch_result(
        self, failure_class: str, file_ext: str, method: str, success: bool,
        repair_rounds: int = 0,
    ) -> None:
        """Record patch/repair method outcome."""
        reward = 1.0 if success else -0.5
        reward -= 0.1 * repair_rounds
        state = f"{failure_class}|{file_ext}"
        with self._lock:
            self.patch_learner.update(state, method, reward)

    # ── 3. Context budget learning ──────────────────────────────

    def record_context_result(
        self, task_type: str, context_config: str, success: bool,
        plan_quality: float = 0.0,
    ) -> None:
        """Record context configuration outcome.

        context_config: e.g. "callers=10,depth=1" or "callers=20,depth=2"
        """
        reward = _compute_reward(success, plan_quality, base_success=0.8, base_fail=-0.4)
        with self._lock:
            self.context_learner.update(task_type, context_config, reward)

    # ── 4. Routing learning ─────────────────────────────────────

    def record_routing_result(
        self, request_features: str, lane: str, success: bool,
        was_fallback: bool = False,
    ) -> None:
        """Record lane routing outcome.

        request_features: e.g. "python|single_file|symbol" or "css|multi_file"
        """
        reward = 0.8 if success else -0.5
        if was_fallback and success:
            reward += 0.3  # bonus for successful fallback recovery
        with self._lock:
            self.routing_learner.update(request_features, lane, reward)

    # ── 5. Prompt variant learning ──────────────────────────────

    def record_prompt_result(
        self, strategy: str, variant: str, success: bool,
        plan_quality: float = 0.0,
    ) -> None:
        """Record prompt variant A/B result.

        variant: e.g. "default", "strict_constraints", "example_driven"
        """
        reward = _compute_reward(success, plan_quality, base_success=1.0, base_fail=-0.5)
        with self._lock:
            self.prompt_learner.update(strategy, variant, reward)

    # ── Persistence ─────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tool": self.tool_learner.to_dict(),
                "patch": self.patch_learner.to_dict(),
                "context": self.context_learner.to_dict(),
                "routing": self.routing_learner.to_dict(),
                "prompt": self.prompt_learner.to_dict(),
            }

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        with self._lock:
            for name, learner in [
                ("tool", self.tool_learner),
                ("patch", self.patch_learner),
                ("context", self.context_learner),
                ("routing", self.routing_learner),
                ("prompt", self.prompt_learner),
            ]:
                raw = state.get(name)
                if isinstance(raw, dict):
                    learner.load_dict(raw)



# ---------------------------------------------------------------------------
# Integration entry points
# ---------------------------------------------------------------------------

def build_learning_signal_from_execution_metadata(
    metadata: dict[str, Any],
    *,
    success: bool,
    bucket: Optional[str] = None,
) -> Optional[LearningSignal]:
    """Build a ``LearningSignal`` from execution metadata.

    Reads keys already written to plan.metadata / spec.metadata by existing
    pipeline stages.  Caller merges plan + spec metadata before passing.

    Parameters
    ----------
    metadata:
        Combined metadata dict (plan.metadata merged with spec.metadata).
    success:
        Whether the overall execution succeeded.
    bucket:
        Override bucket; derived from metadata when None.

    Returns
    -------
    ``LearningSignal`` or ``None`` if insufficient information.
    """
    try:
        return _build_signal_inner(metadata, success=success, bucket=bucket)
    except Exception:
        logger.debug(
            "build_learning_signal_from_execution_metadata: error", exc_info=True
        )
        return None


def _build_signal_inner(
    metadata: dict[str, Any],
    success: bool,
    bucket: Optional[str],
) -> Optional[LearningSignal]:
    # Pre-execution strategy selection metadata
    pre_sel = metadata.get("pre_execution_strategy_selection") or {}
    selected_strategy = pre_sel.get("selected_strategy", "generic_create")

    # Derive weight profile from ranking (first entry) when available
    ranking = pre_sel.get("strategy_ranking", [])
    selected_profile = (
        ranking[0].get("selected_weight_profile", "DEFAULT")
        if ranking else
        pre_sel.get("weight_source_profile", "DEFAULT")
    )

    # Graph impact
    gi = metadata.get("graph_impact") or {}
    graph_impact_level = gi.get("impact_level", "low")

    # Repair / contract signals
    repair_attempts = int(metadata.get("repair_attempts", 0))
    repair_burden   = metadata.get("repair_burden", "none") or "none"
    contract_viol   = bool(metadata.get("contract_violation", False))
    semantic_fails  = list(metadata.get("semantic_failures", []) or [])
    budget_fail     = bool(metadata.get("budget_failure", False))

    # Derive bucket if not provided
    if not bucket:
        has_strict = bool(
            metadata.get("has_strict_reference") or
            metadata.get("reference_files")
        )
        rb_ctx = bool(metadata.get("reference_bound_context"))
        bucket = resolve_bucket(has_strict, rb_ctx, graph_impact_level)

    return LearningSignal(
        bucket=bucket,
        selected_weight_profile=selected_profile or "DEFAULT",
        selected_strategy=selected_strategy or "generic_create",
        success=success,
        repair_attempts=repair_attempts,
        repair_burden=repair_burden,
        contract_violation=contract_viol,
        semantic_failures=semantic_fails,
        budget_failure=budget_fail,
        graph_impact_level=graph_impact_level,
    )


def update_weights_from_monitor_result(
    weight_learner: WeightLearner,
    result_dict: dict[str, Any],
) -> bool:
    """Apply weight update from an external monitor evaluation result.

    Accepts a dict in the format produced by self_impl_monitor or similar
    evaluation tools.  Expected keys (all optional):

    - ``status``          "success" | "partial_success" | "failed"
    - ``success``         bool (overrides ``status`` if present)
    - ``failure_reasons`` list[str] — semantic failure descriptions
    - ``semantic_failures`` list[str] — alias for failure_reasons
    - ``metadata``        dict — plan.metadata / spec.metadata keys

    Returns True if an update was applied.
    """
    try:
        status = result_dict.get("status", "")
        success_val = bool(
            result_dict.get("success") or
            (status in ("success", "partial_success"))
        )
        meta = dict(result_dict.get("metadata") or {})
        semantic_fails = list(
            result_dict.get("failure_reasons") or
            result_dict.get("semantic_failures") or
            []
        )
        if semantic_fails:
            meta.setdefault("semantic_failures", semantic_fails)

        sig = build_learning_signal_from_execution_metadata(
            meta, success=success_val
        )
        if sig is None:
            return False

        weight_learner.update(sig)
        return True
    except Exception:
        logger.debug(
            "update_weights_from_monitor_result: error", exc_info=True
        )
        return False
