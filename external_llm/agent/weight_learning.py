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

Update deltas are intentionally small (0.01-0.03) and always followed by
clamp → normalise to keep weights in [_W_MIN, _W_MAX] and sum == 1.0.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
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

# Bottleneck kinds that constitute a budget failure — shared structural
# vocabulary with self_impl_monitor's serialized ``budget_failure`` bool.
_BUDGET_FAILURE_KINDS: frozenset[str] = frozenset({"max_turns", "cost_limit", "auto_cancel"})

# Axis weights that should relax toward DEFAULT baseline on clean success
_DEFAULT_CONTRACT_BASELINE: float = 0.20
_DEFAULT_REPAIR_BASELINE:   float = 0.30
_RELAX_THRESHOLD:           float = 0.05   # relax only if > baseline + threshold


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
        return dict.fromkeys(_AXES, equal)
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
    A. contract violation (structured flag):
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
             contract -= 0.5xSMALL, repair -= 0.5xSMALL

    Returns a dict {axis: delta} where positive means "increase this weight".
    """
    delta: dict[str, float] = dict.fromkeys(_AXES, 0.0)

    # ── A: Contract / semantic failure ────────────────────────────────────
    # Structured flag only (set by the monitor's reference-contract evaluator).
    # Free-text keyword sniffing is forbidden — semantic_failures is
    # controlled vocabulary appended by the producer, not a match target.
    if signal.contract_violation:
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
    4. Re-balance: distribute the sum residual from step 3 across every axis
       that still has room in the needed direction (water-filling). Each pass
       spreads the residual evenly over the unpinned axes; an axis that hits a
       bound drops out and the remainder carries to the next pass. Because 1.0
       always lies within [_AXES·_W_MIN, _AXES·_W_MAX], total room always covers
       the residual, so this converges in at most len(_AXES) passes.
    """
    updated = {k: weights.get(k, 0.0) + delta.get(k, 0.0) for k in _AXES}
    normed  = _normalize_weights(updated)
    clamped = _clamp_weights(normed)

    deficit = round(1.0 - sum(clamped.values()), 9)
    for _ in range(len(_AXES)):
        if abs(deficit) <= 1e-9:
            break
        if deficit > 0:
            # Need to ADD weight: axes with upward room below _W_MAX.
            flexible = [k for k in _AXES if clamped[k] < _W_MAX - 1e-12]
        else:
            # Need to REMOVE weight: axes with downward room above _W_MIN.
            flexible = [k for k in _AXES if clamped[k] > _W_MIN + 1e-12]
        if not flexible:
            break  # defensive: every axis pinned (infeasible bounds)
        share = deficit / len(flexible)
        absorbed = 0.0
        for k in flexible:
            new_val = max(_W_MIN, min(_W_MAX, clamped[k] + share))
            absorbed += new_val - clamped[k]
            clamped[k] = round(new_val, 9)
        deficit = round(deficit - absorbed, 9)

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

    def get_learned_weights(self, bucket: str) -> dict[str, float]:  # live via AdaptiveLearnerHub.tool_learner (run_store lazy import)
        """Return the current learned (possibly still equal to static) weights."""
        return dict(self.get_bucket_state(bucket).weights)

    def get_effective_weights(  # live via AdaptiveLearnerHub.tool_learner (run_store lazy import)
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

    def get_summary(self) -> dict[str, Any]:  # live via AdaptiveLearnerHub.tool_learner (run_store lazy import)
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

    def load_state(self, state_dict: dict[str, Any]) -> None:  # live via AdaptiveLearnerHub.tool_learner (run_store lazy import)
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
            with suppress(ValueError, TypeError, KeyError):  # Keep existing state on per-bucket error; silently continue
                restored = _normalize_weights(_clamp_weights({
                    k: float(raw_weights.get(k, self._states[bucket].weights.get(k, 0.0)))
                    for k in _AXES
                }))
                self._states[bucket].weights      = restored
                self._states[bucket].signal_count = int(data.get("signal_count", 0))
                self._states[bucket].last_updated = float(data.get("last_updated", 0.0))


# ---------------------------------------------------------------------------
# AdaptiveLearnerHub — unified learning for tool selection
# ---------------------------------------------------------------------------

class MiniQLearner:
    """Lightweight Q-table learner for a single decision domain.

    Shared pattern: statexaction → Q-value, with EMA update and persistence.
    Thread-safe via external lock (caller must hold).
    """

    def __init__(self, name: str, alpha: float = 0.1):
        self.name = name
        self._alpha = alpha
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


class AdaptiveLearnerHub:  # live cross-module edge (run_store.py lazy import)
    """Unified learning hub for tool selection (MAIN_AGENT lane).

    The patch/context/routing/prompt learners were removed as dead weight
    (their record_* entry points had no callers); only ``tool_learner``
    remains live.

    Thread-safe. Persistence via get_summary() / load_state().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tool_learner = MiniQLearner("tool", alpha=0.1)

    # ── Tool selection learning ──────────────────────────────

    def record_tool_usage(  # live cross-module edge (run_store.py lazy import)
        self, phase: str, tool_name: str, success: bool, context_bucket: str = "",
    ) -> None:
        """Record tool usage outcome in MAIN_AGENT lane."""
        reward = 0.5 if success else -0.3
        state = f"{context_bucket}|{phase}" if context_bucket else phase
        with self._lock:
            self.tool_learner.update(state, tool_name, reward)

    # ── Persistence ─────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:  # live cross-module edge (run_store.py lazy import)
        with self._lock:
            return {
                "tool": self.tool_learner.to_dict(),
            }

    def load_state(self, state: dict[str, Any]) -> None:  # live cross-module edge (run_store.py lazy import)
        if not isinstance(state, dict):
            return
        with self._lock:
            for name, learner in [
                ("tool", self.tool_learner),
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


def infer_budget_failure(result: dict[str, Any]) -> bool:
    """Infer budget failure from a monitor result dict.

    The structured ``budget_failure`` bool serialized by self_impl_monitor
    ``save_result`` is authoritative (presence-wins); bottleneck-kind matching
    is the legacy fallback for pre-structured result JSONs.
    """
    maybe = result.get("budget_failure")
    if isinstance(maybe, bool):
        return maybe
    bottlenecks = result.get("bottlenecks") or []
    for b in bottlenecks:
        if not isinstance(b, dict):
            continue
        kind = str(b.get("kind", ""))
        if kind in _BUDGET_FAILURE_KINDS:
            return True
    return False


def infer_repair_attempts(result: dict[str, Any]) -> int:
    """Resolve repair attempts from a monitor result dict.

    Presence-wins: a direct ``repair_attempts`` int (now emitted by
    self_impl_monitor save_result, counted structurally from failed
    verification calls) is authoritative.  For legacy result JSON without
    the counter this remains a documented estimation: semantic failures
    imply retries and partial_success implies one retry.
    """
    maybe = result.get("repair_attempts")
    if isinstance(maybe, int):
        return maybe

    semantic_failures = result.get("semantic_failures") or []
    evaluated_status = str(result.get("evaluated_status") or result.get("status") or "")

    if semantic_failures:
        return 2
    if evaluated_status == "partial_success":
        return 1
    return 0


def repair_burden_label(repair_attempts: int, budget_failure: bool, success: bool) -> str:
    """Map repair effort onto the ordinal repair_burden vocabulary."""
    if budget_failure or not success:
        return "high"
    if repair_attempts <= 0:
        return "none"
    if repair_attempts == 1:
        return "low"
    return "medium"


def update_weights_from_monitor_result(  # live dev-CLI edge (auto_learning_loop.py)
    weight_learner: WeightLearner,
    result_dict: dict[str, Any],
) -> bool:
    """Apply weight update from an external monitor evaluation result.

    Accepts a dict in the format produced by self_impl_monitor or similar
    evaluation tools.  Expected keys (all optional):

    - ``status``          "success" | "partial_success" | "failed"
    - ``success``         bool (overrides ``status`` if present)
    - ``contract_violation`` bool — structured flag emitted by
      self_impl_monitor save_result (SSOT for rule A; no text sniffing)
    - ``budget_failure`` bool — structured flag emitted by self_impl_monitor
      save_result; derived from bottleneck kinds when absent
    - ``bottlenecks`` list[dict] — legacy fallback for budget_failure
    - ``repair_attempts`` int — direct hint; estimated when absent
    - ``repair_burden`` str — "none"|"low"|"medium"|"high"; derived from
      repair_attempts/budget_failure/success when absent
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
        # Forward the monitor's structured evaluation flags (top-level keys in
        # the self_impl_monitor result JSON).  Metadata wins when both present.
        meta.setdefault("contract_violation", bool(result_dict.get("contract_violation")))
        # Budget failure: structured bool first, bottleneck-kind derivation as
        # legacy fallback (mirrors auto_learning_loop.infer_budget_failure).
        meta.setdefault("budget_failure", infer_budget_failure(result_dict))
        # Repair effort: the monitor's save_result now serializes a direct
        # repair_attempts counter (counted from failed verification calls);
        # estimate only for legacy result JSON that predates the counter.
        if "repair_attempts" not in meta:
            meta["repair_attempts"] = infer_repair_attempts(result_dict)
        if "repair_burden" not in meta:
            meta["repair_burden"] = repair_burden_label(
                meta["repair_attempts"],
                meta["budget_failure"],
                success_val,
            )

        sig = build_learning_signal_from_execution_metadata(
            meta, success=success_val
        )
        if sig is None:
            return False

        weight_learner.update(sig)
    except Exception:
        logger.debug(
            "update_weights_from_monitor_result: error", exc_info=True
        )
        return False
    else:
        return True
