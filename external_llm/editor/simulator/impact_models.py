"""
Impact simulation models for P2.1 Impact Simulator.

Defines data structures for representing impact analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImpactReport:
    """Complete impact analysis report for a candidate plan."""

    target_symbols: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    direct_callers: list[str] = field(default_factory=list)
    indirect_callers: list[str] = field(default_factory=list)
    affected_tests: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    impacted_files: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactConfig:
    """Configuration for impact simulation."""

    caller_depth: int = 2
    include_tests: bool = True
    include_dependencies: bool = True
    max_nodes: int = 100
    # Risk estimation weights (can be overridden)
    weight_direct_callers: float = 0.25
    weight_indirect_callers: float = 0.10
    weight_dependencies: float = 0.15
    weight_impacted_files: float = 0.20
    penalty_missing_tests: float = 1.0
    max_risk_score: float = 5.0
