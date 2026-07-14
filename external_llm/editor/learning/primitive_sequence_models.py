"""primitive_sequence_models.py — Phase F.2: Primitive Sequence Data Models.

Captures transition patterns between primitives (A→B) and
full sequence patterns for action types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrimitiveTransition:
    """Statistics for a single A→B primitive transition."""
    from_prim: str
    to_prim: str
    uses: int = 0
    success_count: int = 0       # Final pass after this transition
    total_coverage_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.uses if self.uses > 0 else 0.0

    @property
    def avg_coverage_gain(self) -> float:
        return self.total_coverage_gain / self.uses if self.uses > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_prim,
            "to": self.to_prim,
            "uses": self.uses,
            "success_rate": round(self.success_rate, 3),
            "avg_gain": round(self.avg_coverage_gain, 4),
        }


@dataclass
class PrimitiveSequencePattern:
    """A recorded full primitive sequence for an action type."""
    action_type: str              # "login", "create", "send", etc.
    sequence: tuple[str, ...]     # Ordered primitives applied
    uses: int = 0
    success_count: int = 0
    total_coverage_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.uses if self.uses > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "sequence": list(self.sequence),
            "uses": self.uses,
            "success_rate": round(self.success_rate, 3),
        }
