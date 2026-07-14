"""primitive_sequence_store.py — Phase F.2: Primitive Sequence Storage.

Stores transition frequencies (A→B) and full sequence patterns.
Indexed by action_type for fast lookup during reconstruction.
"""
from __future__ import annotations
from typing import Any, Optional

from external_llm.editor.learning.primitive_sequence_models import PrimitiveSequencePattern, PrimitiveTransition
class PrimitiveSequenceStore:
    """Store for primitive transition and sequence data."""

    def __init__(self):
        # (from_prim, to_prim) → PrimitiveTransition
        self._transitions: dict[tuple[str, str], PrimitiveTransition] = {}
        # (action_type, sequence_tuple) → PrimitiveSequencePattern
        self._patterns: dict[tuple[str, tuple[str, ...]], PrimitiveSequencePattern] = {}

    def record_transition(
        self,
        from_prim: str,
        to_prim: str,
        success: bool,
        coverage_gain: float = 0.0,
    ) -> None:
        """Record a single A→B transition."""
        key = (from_prim, to_prim)
        if key not in self._transitions:
            self._transitions[key] = PrimitiveTransition(
                from_prim=from_prim, to_prim=to_prim,
            )
        t = self._transitions[key]
        t.uses += 1
        if success:
            t.success_count += 1
        t.total_coverage_gain += coverage_gain

    def record_sequence(
        self,
        action_type: str,
        sequence: list[str],
        success: bool,
        coverage_gain: float = 0.0,
    ) -> None:
        """Record a full primitive sequence."""
        if not sequence:
            return
        seq_tuple = tuple(sequence)
        key = (action_type, seq_tuple)
        if key not in self._patterns:
            self._patterns[key] = PrimitiveSequencePattern(
                action_type=action_type, sequence=seq_tuple,
            )
        p = self._patterns[key]
        p.uses += 1
        if success:
            p.success_count += 1
        p.total_coverage_gain += coverage_gain

    def get_transition(self, from_prim: str, to_prim: str) -> Optional[PrimitiveTransition]:
        """Get transition stats for A→B."""
        return self._transitions.get((from_prim, to_prim))

    def get_outgoing(self, from_prim: str) -> list[PrimitiveTransition]:
        """Get all transitions from a given primitive."""
        return [t for (f, _), t in self._transitions.items() if f == from_prim]

    def get_best_patterns(self, action_type: str, top_k: int = 5) -> list[PrimitiveSequencePattern]:
        """Get top-k patterns for an action type, sorted by success rate."""
        patterns = [p for (at, _), p in self._patterns.items() if at == action_type]
        patterns.sort(key=lambda p: (p.success_rate, p.uses), reverse=True)
        return patterns[:top_k]

    @property
    def total_transitions(self) -> int:
        return len(self._transitions)

    @property
    def total_patterns(self) -> int:
        return len(self._patterns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transitions": {
                f"{f}→{t}": v.to_dict()
                for (f, t), v in self._transitions.items()
            },
            "patterns": [
                p.to_dict()
                for p in sorted(
                    self._patterns.values(),
                    key=lambda x: x.uses, reverse=True,
                )[:50]  # Keep top 50
            ],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        for _k, v in data.get("transitions", {}).items():
            f, t = v.get("from", ""), v.get("to", "")
            if f and t:
                self._transitions[(f, t)] = PrimitiveTransition(
                    from_prim=f, to_prim=t,
                    uses=v.get("uses", 0),
                    success_count=int(v.get("success_rate", 0) * v.get("uses", 0)),
                    total_coverage_gain=v.get("avg_gain", 0) * v.get("uses", 0),
                )
        for p in data.get("patterns", []):
            seq = tuple(p.get("sequence", []))
            at = p.get("action_type", "")
            if seq and at:
                self._patterns[(at, seq)] = PrimitiveSequencePattern(
                    action_type=at, sequence=seq,
                    uses=p.get("uses", 0),
                    success_count=int(p.get("success_rate", 0) * p.get("uses", 0)),
                )
