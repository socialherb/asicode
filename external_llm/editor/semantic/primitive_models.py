"""primitive_models.py — Phase F: Semantic Primitive Data Models.

Defines 12 generic semantic primitives and their associated structures.
These primitives are domain-independent behavioral building blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticPrimitive:
    """Definition of a single semantic primitive."""
    name: str                # e.g., "create_entity", "validate", "persist_state"
    category: str            # "data" | "control" | "io" | "structure"
    description: str = ""
    required_for_actions: list[str] = field(default_factory=list)
    # Actions that typically require this primitive: ["create", "login", "send", ...]
    typical_signals: list[str] = field(default_factory=list)
    # Code patterns that indicate this primitive is present


@dataclass
class PrimitiveMatch:
    """Detection result for one primitive in one action."""
    primitive: str           # Primitive name
    present: bool = False
    confidence: float = 0.0  # 0.0-1.0
    evidence: str = ""       # What code/pattern was found
    missing_reason: str = "" # Why it's considered missing


@dataclass
class PrimitiveSequence:
    """Primitive analysis for a single action (function/endpoint)."""
    action_name: str         # Function name: "login", "send_message", "upload_video"
    action_type: str         # "create" | "login" | "send" | "upload" | "list" | "update" | "delete"
    entity: str = ""         # Primary entity: "User", "Message", "Video"
    file_path: str = ""
    present: list[PrimitiveMatch] = field(default_factory=list)
    missing: list[PrimitiveMatch] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.present) + len(self.missing)
        return len(self.present) / total if total > 0 else 1.0

    @property
    def missing_names(self) -> list[str]:
        return [m.primitive for m in self.missing]

    @property
    def present_names(self) -> list[str]:
        return [p.primitive for p in self.present]


@dataclass
class PrimitiveIR:
    """Intermediate Representation: all actions with their primitive analysis."""
    sequences: list[PrimitiveSequence] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    @property
    def overall_coverage(self) -> float:
        if not self.sequences:
            return 1.0
        return sum(s.coverage for s in self.sequences) / len(self.sequences)

    @property
    def all_missing(self) -> list[str]:
        result: list[str] = []
        for s in self.sequences:
            for m in s.missing:
                if m.primitive not in result:
                    result.append(m.primitive)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "actions": len(self.sequences),
            "coverage": round(self.overall_coverage, 4),
            "missing_primitives": self.all_missing,
            "entities": self.entities,
            "sequences": [
                {
                    "action": s.action_name,
                    "type": s.action_type,
                    "entity": s.entity,
                    "coverage": round(s.coverage, 2),
                    "present": s.present_names,
                    "missing": s.missing_names,
                }
                for s in self.sequences
            ],
        }


@dataclass
class ReconstructionCandidate:
    """A reconstructed candidate produced by primitive-based reconstruction."""
    patches: list[dict[str, Any]] = field(default_factory=list)
    fragments: list[dict[str, Any]] = field(default_factory=list)
    applied_primitives: list[str] = field(default_factory=list)
    confidence: float = 0.0
    primitive_coverage_estimate: float = 0.0
    notes: list[str] = field(default_factory=list)
