"""synthesis_planner.py — Phase F.3: Plan Function Synthesis.

Determines the optimal primitive sequence for a given action,
using learned patterns when available, with fallback to defaults.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
# Default primitive sequences per action type (fallback when no learning data)
_DEFAULT_SEQUENCES: dict[str, list[str]] = {
    "login": ["lookup", "validate", "branch_on_failure", "authorize", "produce_output"],
    "signup": ["input_bind", "create_entity", "persist_state", "produce_output"],
    "register": ["input_bind", "create_entity", "persist_state", "produce_output"],
    "create": ["input_bind", "create_entity", "persist_state", "produce_output"],
    "send": ["input_bind", "create_entity", "persist_state", "produce_output"],
    "upload": ["input_bind", "create_entity", "persist_state", "produce_output"],
    "get": ["lookup", "produce_output"],
    "list": ["list_or_query", "paginate"],
    "update": ["lookup", "input_bind", "update_entity", "persist_state", "produce_output"],
    "delete": ["lookup", "delete_entity", "persist_state", "produce_output"],
}


@dataclass
class SynthesisPlan:
    """Plan for synthesizing a function from primitives."""
    action_name: str
    action_type: str
    entity: str
    sequence: list[str]
    params: list[str] = field(default_factory=list)
    source: str = ""  # "learned" | "default" | "fallback"
    decorator: str = ""  # e.g., '@router.post("/login")'

    def summary(self) -> dict[str, Any]:
        return {
            "action": self.action_name,
            "type": self.action_type,
            "entity": self.entity,
            "sequence": self.sequence,
            "source": self.source,
        }


def plan_synthesis(
    action_name: str,
    action_type: str,
    entity: str = "",
    params: Optional[list[str]] = None,
    sequence_store: Any = None,
    missing_primitives: Optional[list[str]] = None,
    decorator: str = "",
) -> SynthesisPlan:
    """Build a synthesis plan for a function.

    Priority:
    1. Learned sequence from sequence_store (if available + good success rate)
    2. Missing primitives as-is (if provided)
    3. Default sequence for action_type
    """
    sequence: list[str] = []
    source = "fallback"

    # Strategy 1: Learned sequence
    if sequence_store:
        try:
            from external_llm.editor.learning.primitive_sequence_scorer import recommend_sequence
            if missing_primitives:
                learned = recommend_sequence(sequence_store, action_type, missing_primitives)
                if learned:
                    sequence = learned
                    source = "learned"
            else:
                best = sequence_store.get_best_patterns(action_type, top_k=1)
                if best and best[0].success_rate >= 0.5:
                    sequence = list(best[0].sequence)
                    source = "learned"
        except Exception:
            pass

    # Strategy 2: Missing primitives
    if not sequence and missing_primitives:
        sequence = list(missing_primitives)
        source = "missing"

    # Strategy 3: Default
    if not sequence:
        sequence = list(_DEFAULT_SEQUENCES.get(action_type, ["produce_output"]))
        source = "default"

    return SynthesisPlan(
        action_name=action_name,
        action_type=action_type,
        entity=entity,
        sequence=sequence,
        params=list(params or []),
        source=source,
        decorator=decorator,
    )
