"""primitive_sequence_updater.py — Phase F.2: Record Primitive Sequences.

Extracts the ordered sequence of applied primitives from Phase F results
and records transitions + patterns in the sequence store.
"""
from __future__ import annotations

import logging
from typing import Any

from external_llm.editor.learning.primitive_sequence_store import PrimitiveSequenceStore

logger = logging.getLogger(__name__)

# Virtual start/end tokens for sequence boundaries
_START = "__START__"
_END = "__END__"


def update_primitive_sequences(
    store: PrimitiveSequenceStore,
    applied_primitives: list[str],
    action_type: str,
    success: bool,
    coverage_delta: float = 0.0,
) -> dict[str, Any]:
    """Record transitions and sequence pattern from one reconstruction.

    Applied primitives should be in the order they were actually filled.
    """
    result = {
        "transitions_recorded": 0,
        "sequence_recorded": False,
    }

    if not applied_primitives:
        return result

    # Deduplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for p in applied_primitives:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    # Record transitions: START→first, A→B, ..., last→END
    chain = [_START, *deduped, _END]
    per_transition_gain = coverage_delta / max(len(deduped), 1)

    for i in range(len(chain) - 1):
        store.record_transition(
            from_prim=chain[i],
            to_prim=chain[i + 1],
            success=success,
            coverage_gain=per_transition_gain,
        )
        result["transitions_recorded"] += 1

    # Record full sequence pattern
    store.record_sequence(
        action_type=action_type,
        sequence=deduped,
        success=success,
        coverage_gain=coverage_delta,
    )
    result["sequence_recorded"] = True

    logger.debug(
        "[SEQ_UPDATE] action=%s seq=%s success=%s transitions=%d",
        action_type, deduped, success, result["transitions_recorded"],
    )

    return result
