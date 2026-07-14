"""primitive_learning_scorer.py — Phase F.1: Primitive Priority Scoring.

Uses learned outcome data to prioritize which missing primitives
to fill first during reconstruction.

Cold-start smoothing ensures sane scores with sparse data.
"""
from __future__ import annotations

import logging

from external_llm.editor.learning.primitive_learning_models import PrimitiveLearningKey, PrimitiveOutcomeRecord
from external_llm.editor.learning.primitive_learning_store import PrimitiveLearningStore

logger = logging.getLogger(__name__)

# Laplace smoothing: add virtual observations to prevent 0-scores
_SMOOTH_USES = 2
_SMOOTH_PASS = 1
_SMOOTH_IMPROVED = 1

# Default priority for cold-start (no data at all)
_DEFAULT_PRIORITY: dict[str, float] = {
    "validate": 0.80,
    "branch_on_failure": 0.75,
    "lookup": 0.70,
    "create_entity": 0.65,
    "persist_state": 0.60,
    "produce_output": 0.55,
    "authorize": 0.50,
    "input_bind": 0.45,
    "update_entity": 0.40,
    "delete_entity": 0.35,
    "list_or_query": 0.30,
    "delegate_action": 0.20,
}


def score_missing_primitives(
    store: PrimitiveLearningStore,
    context_bucket: str,
    action_type: str,
    missing_primitives: list[str],
    entity: str = "",
) -> list[tuple[str, float]]:
    """Score and rank missing primitives by learned priority.

    Returns [(primitive_name, score)] sorted by score descending.
    Higher score = should be filled first.
    """
    scores: list[tuple[str, float]] = []

    for prim in missing_primitives:
        score = _compute_score(store, context_bucket, action_type, prim, entity)
        scores.append((prim, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if scores:
        logger.debug(
            "[PRIM_SCORE] ctx=%s action=%s top: %s",
            context_bucket, action_type,
            [(p, round(s, 3)) for p, s in scores[:5]],
        )

    return scores


def _compute_score(
    store: PrimitiveLearningStore,
    context_bucket: str,
    action_type: str,
    primitive: str,
    entity: str,
) -> float:
    """Compute priority score for one primitive.

    Score = weighted combination of:
    - pass_rate (0.4)
    - improvement_rate (0.3)
    - avg_coverage_gain (0.2)
    - default_priority (0.1)

    With Laplace smoothing for sparse data.
    """
    # Try exact key
    key = PrimitiveLearningKey(
        context_bucket=context_bucket,
        action_type=action_type,
        primitive=primitive,
        entity=entity,
    )
    rec = store.get(key)

    # Try without entity
    if rec is None:
        key_no_entity = PrimitiveLearningKey(
            context_bucket=context_bucket,
            action_type=action_type,
            primitive=primitive,
        )
        rec = store.get(key_no_entity)

    # Try global primitive stats (across all contexts)
    if rec is None:
        all_recs = store.lookup_by_primitive(primitive)
        if all_recs:
            rec = _merge_records(list(all_recs.values()))

    default = _DEFAULT_PRIORITY.get(primitive, 0.3)

    if rec is None or rec.uses == 0:
        return default

    # Laplace-smoothed rates
    pass_rate = (rec.pass_count + _SMOOTH_PASS) / (rec.uses + _SMOOTH_USES)
    improvement_rate = (rec.improved_count + _SMOOTH_IMPROVED) / (rec.uses + _SMOOTH_USES)
    avg_cov_gain = rec.avg_coverage_gain

    # Normalize coverage gain to 0-1 range (typical gain is 0.0-0.5)
    cov_score = min(1.0, max(0.0, avg_cov_gain * 2.0))

    # Weighted combination
    score = (
        0.4 * pass_rate
        + 0.3 * improvement_rate
        + 0.2 * cov_score
        + 0.1 * default
    )

    return round(min(1.0, max(0.0, score)), 4)


def _merge_records(records: list[PrimitiveOutcomeRecord]) -> PrimitiveOutcomeRecord:
    """Merge multiple records into one aggregate."""
    merged = PrimitiveOutcomeRecord()
    for r in records:
        merged.uses += r.uses
        merged.chosen_count += r.chosen_count
        merged.improved_count += r.improved_count
        merged.pass_count += r.pass_count
        merged.total_coverage_delta += r.total_coverage_delta
        merged.total_sem_delta += r.total_sem_delta
        merged.total_contract_delta += r.total_contract_delta
    return merged
