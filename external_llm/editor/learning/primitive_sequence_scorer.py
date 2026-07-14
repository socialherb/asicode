"""primitive_sequence_scorer.py — Phase F.2: Next-Primitive Recommendation.

Given the current primitive and candidate next primitives,
scores each candidate using learned transition data.

Score = 0.5 * transition_success_rate + 0.3 * frequency + 0.2 * avg_gain
Cold-start: falls back to default ordering.
"""
from __future__ import annotations
from external_llm.editor.learning.primitive_sequence_store import PrimitiveSequenceStore
# Laplace smoothing
_SMOOTH_USES = 2
_SMOOTH_SUCCESS = 1

# Virtual token
_START = "__START__"


def score_next_primitives(
    store: PrimitiveSequenceStore,
    current_primitive: str,
    candidates: list[str],
    action_type: str = "",
) -> list[tuple[str, float]]:
    """Score candidate primitives as the next step after current_primitive.

    Returns [(primitive, score)] sorted by score descending.
    """
    scores: list[tuple[str, float]] = []

    for cand in candidates:
        score = _compute_transition_score(store, current_primitive, cand)
        scores.append((cand, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def recommend_sequence(
    store: PrimitiveSequenceStore,
    action_type: str,
    missing_primitives: list[str],
) -> list[str]:
    """Recommend an ordering of missing primitives based on learned patterns.

    Strategy:
    1. Check if any learned sequence pattern matches (subset of missing)
    2. If yes, use the best pattern's order
    3. If no, build order greedily using transition scores
    """
    if not missing_primitives:
        return []

    missing_set = set(missing_primitives)

    # Strategy 1: Find matching sequence pattern
    best_patterns = store.get_best_patterns(action_type, top_k=10)
    for pattern in best_patterns:
        # Check if pattern sequence is a superset or subset match
        pattern_relevant = [p for p in pattern.sequence if p in missing_set]
        if len(pattern_relevant) >= len(missing_set) * 0.5 and pattern.success_rate > 0.5:
            # Use this pattern's order, filling in any extras at the end
            ordered = list(pattern_relevant)
            for m in missing_primitives:
                if m not in ordered:
                    ordered.append(m)
            return ordered

    # Strategy 2: Greedy ordering using transition scores
    return _greedy_order(store, missing_primitives)


def _greedy_order(store: PrimitiveSequenceStore, primitives: list[str]) -> list[str]:
    """Build an ordering greedily using transition scores."""
    if len(primitives) <= 1:
        return list(primitives)

    remaining = set(primitives)
    ordered: list[str] = []
    current = _START

    while remaining:
        # Score all remaining as next step
        candidates = list(remaining)
        scores = score_next_primitives(store, current, candidates)

        if scores:
            best = scores[0][0]
        else:
            best = candidates[0]

        ordered.append(best)
        remaining.discard(best)
        current = best

    return ordered


def _compute_transition_score(
    store: PrimitiveSequenceStore,
    from_prim: str,
    to_prim: str,
) -> float:
    """Compute score for the from→to transition."""
    t = store.get_transition(from_prim, to_prim)

    if t is None or t.uses == 0:
        # Cold start: use default (0.3 = neutral)
        return 0.3

    # Smoothed rates
    success_rate = (t.success_count + _SMOOTH_SUCCESS) / (t.uses + _SMOOTH_USES)
    frequency = min(1.0, t.uses / 10.0)  # Normalize: 10+ uses = 1.0
    avg_gain = min(1.0, max(0.0, t.avg_coverage_gain * 3.0))  # Normalize

    score = 0.5 * success_rate + 0.3 * frequency + 0.2 * avg_gain
    return round(min(1.0, max(0.0, score)), 4)
