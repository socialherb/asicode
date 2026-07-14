"""semantic_contract_models.py — Phase C.1: Semantic Contract Data Models.

Defines the core data structures for semantic contracts:
- SemanticContract: behavioral specification ("what must hold")
- SemanticViolation: a single contract breach
- SemanticEvaluationResult: per-contract evaluation
- SemanticContractReport: aggregate report across all contracts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SemanticContract:
    """Behavioral contract for a code capability.

    Unlike expectations (Phase A) which check "does X exist?",
    contracts check "does X behave correctly?" — ordering, data flow,
    branching, and output semantics.
    """
    name: str
    applies_to: list[str] = field(default_factory=list)
    # Tags this contract matches: ["auth.login", "create", "video.upload"]

    requires: list[str] = field(default_factory=list)
    # Behavioral requirements: ["entity_creation", "persistence", "user_lookup"]

    ordering: list[str] = field(default_factory=list)
    # Execution order constraints: ["user_lookup -> password_verification -> token_generation"]

    binding_rules: list[str] = field(default_factory=list)
    # Data flow rules: ["message.content from input", "entity must derive from input"]

    branch_rules: list[str] = field(default_factory=list)
    # Control flow rules: ["if verification fails, token_generation must not execute"]

    output_rules: list[str] = field(default_factory=list)
    # Output semantics: ["output_must_reference_created_entity"]

    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticViolation:
    """A single semantic contract violation."""
    contract_name: str
    rule_type: str      # "requires" | "ordering" | "binding" | "branch" | "output"
    rule: str           # The violated rule string
    severity: str       # "low" | "medium" | "high"
    message: str        # Human-readable description
    evidence: Optional[dict[str, Any]] = None
    # Evidence: {"function": "login", "missing_call": "verify_password", ...}


@dataclass
class SemanticEvaluationResult:
    """Evaluation of a single contract against traced execution."""
    contract_name: str
    passed: bool
    violations: list[SemanticViolation] = field(default_factory=list)
    score: float = 1.0  # 0.0 ~ 1.0

    def high_violations(self) -> list[SemanticViolation]:
        return [v for v in self.violations if v.severity == "high"]


@dataclass
class SemanticContractReport:
    """Aggregate report across all evaluated contracts."""
    results: list[SemanticEvaluationResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def has_critical_violation(self) -> bool:
        return any(
            v.severity == "high"
            for r in self.results
            for v in r.violations
        )

    @property
    def all_violations(self) -> list[SemanticViolation]:
        return [v for r in self.results for v in r.violations]

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 4),
            "has_critical": self.has_critical_violation,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": len(self.results),
            "results": [
                {
                    "contract": r.contract_name,
                    "passed": r.passed,
                    "score": round(r.score, 4),
                    "violations": [
                        {
                            "type": v.rule_type,
                            "rule": v.rule,
                            "severity": v.severity,
                            "message": v.message,
                        }
                        for v in r.violations
                    ],
                }
                for r in self.results
            ],
        }
