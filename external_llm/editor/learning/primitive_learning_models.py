"""primitive_learning_models.py — Phase F.1: Primitive Learning Data Models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrimitiveLearningKey:
    """Key for indexing primitive learning records."""
    context_bucket: str    # e.g., "create|auth|medium"
    action_type: str       # e.g., "login", "create", "send"
    primitive: str         # e.g., "validate", "persist_state"
    entity: str = ""       # Optional: "User", "Message"

    def to_str(self) -> str:
        # Use "::" as field separator (context_bucket may contain "|")
        parts = [self.context_bucket, self.action_type, self.primitive]
        if self.entity:
            parts.append(self.entity)
        return "::".join(parts)

    @classmethod
    def from_str(cls, s: str) -> "PrimitiveLearningKey":
        # Support both "::" (new) and "|" (legacy) separators
        if "::" in s:
            parts = s.split("::")
        else:
            # Legacy format: context_bucket may contain "|"
            # Try to find action_type and primitive by scanning from the end
            parts = s.split("|")
            if len(parts) > 4:
                # context_bucket had internal "|" — recombine
                # Format: bucket_part1|bucket_part2|...|action_type|primitive[|entity]
                # We know action_type and primitive are the last 2-3 fields
                entity = ""
                primitive = parts[-1]
                action_type = parts[-2]
                # Validate: if primitive looks like a known primitive, use this split
                _KNOWN = {"validate", "lookup", "create_entity", "persist_state",
                          "produce_output", "branch_on_failure", "authorize",
                          "input_bind", "update_entity", "delete_entity",
                          "list_or_query", "delegate_action"}
                if primitive in _KNOWN:
                    context_bucket = "|".join(parts[:-2])
                elif len(parts) > 5 and parts[-1] not in _KNOWN and parts[-2] in _KNOWN:
                    # entity is last, primitive is second-to-last
                    entity = parts[-1]
                    primitive = parts[-2]
                    action_type = parts[-3]
                    context_bucket = "|".join(parts[:-3])
                else:
                    context_bucket = parts[0] if parts else ""
                    action_type = parts[1] if len(parts) > 1 else ""
                    primitive = parts[2] if len(parts) > 2 else ""
                    entity = parts[3] if len(parts) > 3 else ""
                return cls(context_bucket=context_bucket, action_type=action_type,
                           primitive=primitive, entity=entity)
        return cls(
            context_bucket=parts[0] if len(parts) > 0 else "",
            action_type=parts[1] if len(parts) > 1 else "",
            primitive=parts[2] if len(parts) > 2 else "",
            entity=parts[3] if len(parts) > 3 else "",
        )


@dataclass
class PrimitiveOutcomeRecord:
    """Accumulated outcome statistics for one primitive in one context."""
    uses: int = 0
    chosen_count: int = 0       # Times reconstruction was chosen over raw
    improved_count: int = 0     # Times coverage actually improved
    pass_count: int = 0         # Times final verdict was PASS
    total_coverage_delta: float = 0.0
    total_sem_delta: float = 0.0
    total_contract_delta: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.uses if self.uses > 0 else 0.0

    @property
    def improvement_rate(self) -> float:
        return self.improved_count / self.uses if self.uses > 0 else 0.0

    @property
    def avg_coverage_gain(self) -> float:
        return self.total_coverage_delta / self.uses if self.uses > 0 else 0.0

    @property
    def avg_contract_gain(self) -> float:
        return self.total_contract_delta / self.uses if self.uses > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uses": self.uses,
            "chosen": self.chosen_count,
            "improved": self.improved_count,
            "passed": self.pass_count,
            "pass_rate": round(self.pass_rate, 3),
            "improvement_rate": round(self.improvement_rate, 3),
            "avg_coverage_gain": round(self.avg_coverage_gain, 4),
            "avg_contract_gain": round(self.avg_contract_gain, 4),
        }


@dataclass
class PrimitiveStrategyStats:
    """Stats for a specific repair strategy used for a primitive."""
    strategy_name: str     # e.g., "c2_insert_verify_call"
    uses: int = 0
    success_count: int = 0
    total_gain: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.uses if self.uses > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "uses": self.uses,
            "success_rate": round(self.success_rate, 3),
            "avg_gain": round(self.total_gain / self.uses, 4) if self.uses > 0 else 0.0,
        }
