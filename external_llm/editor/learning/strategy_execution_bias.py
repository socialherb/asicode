"""strategy_execution_bias.py — Per-strategy execution fitness scoring.

Reads recent run history from run_store and computes per-strategy
execution bias based on 4 axes:

  1. Success rate    — how often this strategy succeeded recently
  2. Repair burden   — avg repair rounds when this strategy was used
  3. Runtime gate    — compile/lint gate failure rate
  4. Context affinity — request keyword match (create vs modify vs test)

Result: ``{strategy: {"net_bias": float, ...}}`` for unified_prioritize().

Complements PlannerPolicyAdapter:
  - execution_bias  = what fits (bonus/penalty from execution outcomes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context Affinity — structural strategy fitness
# ---------------------------------------------------------------------------


def _compute_context_affinity(
    strategy: str,
    request: str,
    intent_tags: Optional[list[str]],
    spec: Any,
) -> float:
    """Score how well a strategy fits the current request context.

    Uses structured spec attributes (intent, request_type, estimated_scope)
    rather than keyword matching against raw text.

    Returns a value in [-0.10, +0.15].
    """
    intent = ""
    request_type = ""
    scope = ""

    if spec:
        intent = str(getattr(spec, "intent", "") or "").lower()
        request_type = str(getattr(spec, "request_type", "") or "").lower()
        scope = str(getattr(spec, "estimated_scope", "") or "").lower()

    # Also consider intent_tags as structured classification signals
    tags_lower = {t.lower() for t in (intent_tags or [])}
    if not intent and tags_lower:
        # Derive intent from tags using category mapping (no priority-ordered scan).
        # Each IntentResult taxonomy category maps to expected tag values.
        _INTENT_FROM_TAGS: dict[str, set] = {
            "create":   {"create", "feature", "extend"},
            "fix":      {"fix", "bugfix", "bug"},
            "modify":   {"modify", "update"},
            "refactor": {"refactor", "restructure"},
            "test":     {"test", "add_test"},
        }
        for cat, cat_tags in _INTENT_FROM_TAGS.items():
            if cat_tags & tags_lower:
                intent = cat
                break

    score = 0.0

    # Strategy-intent structural alignment (classified values, not keywords)
    if strategy == "symbol_edit" and intent in ("modify", "fix", "update", "bugfix"):
        score += 0.08
    elif strategy == "minimal_patch" and scope in ("single_symbol", "small", "tiny"):
        score += 0.08
    elif strategy == "refactor" and intent in ("refactor", "restructure", "create"):
        score += 0.08
    elif strategy == "test_first" and ("test" in intent or "test" in request_type):
        score += 0.10

    # Anti-affinity: scope mismatch
    if strategy == "minimal_patch" and scope in ("large",):
        score -= 0.06
    elif strategy == "refactor" and scope in ("small", "tiny"):
        score -= 0.05

    return round(max(-0.10, min(0.15, score)), 4)


# ---------------------------------------------------------------------------
# Per-Strategy Execution Stats from run_store
# ---------------------------------------------------------------------------

def _get_strategy_runs(
    run_store: Any,
    strategy: str,
    limit: int = 20,
    current_planner_model: str = "",
) -> list[Any]:
    """Extract recent runs where this strategy was used.

    Primary: InMemoryRunStore (session-scoped).
    Fallback: UnifiedStore (SQLite) when in-session data is sparse (<3 records).
    This enables cross-session strategy bias persistence.

    Model weighting:
        same-model (planner_model matches current) → transfer_weight=1.0
        cross-model (different planner_model, non-empty) → transfer_weight=0.8
        cross-language (different language) → transfer_weight=0.7
    """
    runs: list[Any] = []
    if run_store is not None:
        try:
            all_runs = run_store.list_runs(limit=limit * 3)  # oversample then filter
            for r in all_runs:
                # Check P11 selected_strategy field first
                if getattr(r, "selected_strategy", "") == strategy:
                    runs.append(r)
                    continue
                # Fallback: check candidate_feedback
                feedback = getattr(r, "candidate_feedback", None) or {}
                if isinstance(feedback, dict) and feedback.get("selected_strategy") == strategy:
                    runs.append(r)
                if len(runs) >= limit:
                    break
        except Exception:
            pass

    # Cross-session fallback: supplement from UnifiedStore when in-session data is sparse
    if len(runs) < 3:
        try:
            from external_llm.editor.learning.unified_store import get_unified_store
            store = get_unified_store()

            # Primary: language-specific strategy name (same-language records)
            unified_same = store.get_strategy_runs(
                strategy=strategy, language="python", limit=limit,
            )
            in_session_count = len(runs)
            for ur in unified_same:
                # Apply model transfer weight: same-model=1.0, cross-model=0.8
                rec_planner = getattr(ur, "planner_model", "") or ""
                if current_planner_model and rec_planner and rec_planner != current_planner_model:
                    runs.append(_UnifiedRunProxy(ur, transfer_weight=0.8))
                else:
                    runs.append(_UnifiedRunProxy(ur))

            # Secondary: abstract strategy cross-language (if still sparse)
            if len(runs) < 3:
                try:
                    from external_llm.editor.cross_language.strategy_abstraction import (
                        abstract_strategy as to_abstract,
                    )
                    abstract = to_abstract("python", strategy)
                    abstract_val = abstract.value if hasattr(abstract, "value") else str(abstract)
                    if abstract_val and abstract_val != "unknown":
                        cross_runs = store.get_runs_by_abstract_strategy(
                            abstract_strategy=abstract_val,
                            language="python",  # same-language first, then cross
                            limit=limit,
                        )
                        # Apply transfer weight: cross-language=0.7, cross-model=0.8
                        for ur in cross_runs:
                            rec_planner = getattr(ur, "planner_model", "") or ""
                            if ur.language != "python":
                                runs.append(_UnifiedRunProxy(ur, transfer_weight=0.7))
                            elif ur not in runs:
                                if current_planner_model and rec_planner and rec_planner != current_planner_model:
                                    runs.append(_UnifiedRunProxy(ur, transfer_weight=0.8))
                                else:
                                    runs.append(_UnifiedRunProxy(ur))
                except Exception:
                    pass

            store.close()
            added = len(runs) - in_session_count
            if added > 0:
                logger.debug(
                    "[EXEC_BIAS] %s: %d in-session + %d unified runs (planner=%s)",
                    strategy, in_session_count, added, current_planner_model or "any",
                )
        except Exception:
            pass

    return runs[:limit]


class _UnifiedRunProxy:
    """Thin proxy adapting UnifiedRunRecord to the RunRecord interface
    expected by _success_rate / _avg_repair_rounds / _runtime_gate_fail_rate.

    transfer_weight < 1.0 is applied to cross-language records so that
    foreign-language evidence contributes less than same-language evidence.
    """

    __slots__ = (
        "_transfer_weight",
        "final_failure_class",
        "final_status",
        "repair_rounds_attempted",
        "selected_strategy",
    )

    def __init__(self, ur: Any, transfer_weight: float = 1.0) -> None:
        raw_status = ur.final_status or ("success" if ur.success else "failed")
        # Apply transfer weight: downgrade cross-language successes slightly
        if transfer_weight < 1.0 and raw_status == "success":
            import random
            raw_status = "success" if random.random() < transfer_weight else "failed"
        self.final_status = raw_status
        self.repair_rounds_attempted = getattr(ur, "repair_rounds", 0) or 0
        fc = getattr(ur, "final_failure_class", None)
        self.final_failure_class = fc or ""
        self.selected_strategy = getattr(ur, "strategy", "") or ""
        self._transfer_weight = transfer_weight


def _success_rate(runs: list[Any]) -> float:
    if not runs:
        return 0.5  # neutral prior
    successes = sum(
        1 for r in runs
        if getattr(r, "final_status", "") == "success"
    )
    return successes / len(runs)


def _avg_repair_rounds(runs: list[Any]) -> float:
    if not runs:
        return 0.0
    total = sum(getattr(r, "repair_rounds_attempted", 0) for r in runs)
    return total / len(runs)


def _runtime_gate_fail_rate(runs: list[Any]) -> float:
    """Rate of compile/lint gate failures."""
    if not runs:
        return 0.0
    gate_fails = sum(
        1 for r in runs
        if getattr(r, "final_failure_class", "") in (
            "compile_error", "lint_error", "syntax_error",
        )
    )
    return gate_fails / len(runs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class StrategyExecutionBias:
    """Per-strategy execution bias with breakdown."""
    success_rate_bonus: float = 0.0
    repair_rate_penalty: float = 0.0
    runtime_gate_penalty: float = 0.0
    context_affinity: float = 0.0
    net_bias: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_rate_bonus": round(self.success_rate_bonus, 4),
            "repair_rate_penalty": round(self.repair_rate_penalty, 4),
            "runtime_gate_penalty": round(self.runtime_gate_penalty, 4),
            "context_affinity": round(self.context_affinity, 4),
            "net_bias": round(self.net_bias, 4),
            "sample_count": self.sample_count,
        }


@dataclass
class ExecutionBiasResult:
    """Collection of per-strategy execution biases."""
    biases: dict[str, StrategyExecutionBias] = field(default_factory=dict)

    def net_biases(self) -> dict[str, float]:
        """Return just the net_bias values per strategy."""
        return {k: v.net_bias for k, v in self.biases.items()}

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.biases.items()}


def compute_execution_bias_by_strategy(
    run_store: Any,
    strategies: list[str],
    request: str = "",
    intent_tags: Optional[list[str]] = None,
    context_bucket: str = "",
    spec: Any = None,
    history_limit: int = 20,
    current_planner_model: str = "",
) -> ExecutionBiasResult:
    """Compute per-strategy execution bias from run history + context.

    Args:
        run_store: InMemoryRunStore instance.
        strategies: Available strategy names.
        request: User request text (for context affinity).
        intent_tags: Intent classification tags.
        context_bucket: Execution context bucket (e.g., "product_extend:medium").
        spec: ResolvedExecutionSpec (for context affinity).
        history_limit: Max recent runs to examine per strategy.

    Returns:
        ExecutionBiasResult with per-strategy biases.
    """
    result = ExecutionBiasResult()

    for s in strategies:
        runs = _get_strategy_runs(run_store, s, limit=history_limit,
                                  current_planner_model=current_planner_model)

        sr = _success_rate(runs)
        avg_repair = _avg_repair_rounds(runs)
        gate_fail = _runtime_gate_fail_rate(runs)
        affinity = _compute_context_affinity(s, request, intent_tags, spec)

        success_bonus = round(sr * 0.30, 4)
        repair_penalty = round(avg_repair * 0.08, 4)
        runtime_penalty = round(gate_fail * 0.25, 4)

        net = success_bonus - repair_penalty - runtime_penalty + affinity
        net = round(max(-0.50, min(0.50, net)), 4)

        bias = StrategyExecutionBias(
            success_rate_bonus=success_bonus,
            repair_rate_penalty=repair_penalty,
            runtime_gate_penalty=runtime_penalty,
            context_affinity=affinity,
            net_bias=net,
            sample_count=len(runs),
        )
        result.biases[s] = bias

    if any(b.sample_count > 0 for b in result.biases.values()):
        logger.info(
            "[EXEC_BIAS] per-strategy: %s",
            {k: f"{v.net_bias:+.3f}({v.sample_count})" for k, v in result.biases.items()},
        )

    return result
