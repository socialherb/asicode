"""
Verification Set Builder: combines multiple test discovery sources into
a structured primary/secondary/fallback verification plan.

Consumes:
- SymbolAwareTestTarget list (P8-3)
- TestCoverageCandidate list (P9-2)
- ImpactSet (P9-1)
- CompositeRisk (P8-2)
- Scope level hint

Produces:
- VerificationSet with primary/secondary/fallback groups
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Group assignment thresholds
PRIMARY_SCORE_THRESHOLD = 0.7     # score >= this → primary
SECONDARY_SCORE_THRESHOLD = 0.3   # score >= this → secondary
# Below SECONDARY → fallback

# Scope level → max targets per group
SCOPE_LIMITS = {
    "minimal": {"primary": 3, "secondary": 0, "fallback": 0},
    "narrow": {"primary": 5, "secondary": 3, "fallback": 0},
    "standard": {"primary": 7, "secondary": 5, "fallback": 3},
    "broad": {"primary": 10, "secondary": 8, "fallback": 5},
}


@dataclass
class VerificationTarget:
    """A single test target with provenance."""
    test_path: str
    priority_score: float = 0.0
    source: str = "heuristic"  # "symbol_aware" | "dependency_graph" | "impact_fallback" | "heuristic"
    reason_codes: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    matched_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_path": self.test_path,
            "priority_score": round(self.priority_score, 3),
            "source": self.source,
            "reason_codes": self.reason_codes[:3],
        }


@dataclass
class VerificationGroup:
    """A group of test targets at a specific verification tier."""
    name: str = "primary"  # "primary" | "secondary" | "fallback"
    targets: list[VerificationTarget] = field(default_factory=list)
    scope_level: str = "standard"
    rationale: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [t.test_path for t in self.targets]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_count": len(self.targets),
            "scope_level": self.scope_level,
            "rationale": self.rationale[:3],
            "top_targets": [t.to_dict() for t in self.targets[:3]],
        }


@dataclass
class VerificationSet:
    """Complete verification plan with tiered test groups."""
    primary: VerificationGroup = field(default_factory=lambda: VerificationGroup(name="primary"))
    secondary: VerificationGroup = field(default_factory=lambda: VerificationGroup(name="secondary"))
    fallback: VerificationGroup = field(default_factory=lambda: VerificationGroup(name="fallback"))
    selected_scope_level: str = "standard"
    composite_risk_level: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_target_count(self) -> int:
        return len(self.primary.targets) + len(self.secondary.targets) + len(self.fallback.targets)

    def flatten_paths(self, include_groups: Optional[list[str]] = None) -> list[str]:
        """
        Flatten to ordered path list for TestRunner.

        Args:
            include_groups: Which groups to include. Default: based on scope level.
                           ["primary"] for minimal, ["primary", "secondary"] for standard, etc.
        """
        if include_groups is None:
            if self.selected_scope_level == "minimal":
                include_groups = ["primary"]
            elif self.selected_scope_level == "narrow":
                include_groups = ["primary", "secondary"]
            else:
                include_groups = ["primary", "secondary", "fallback"]

        paths = []
        seen: set[str] = set()
        for group_name in include_groups:
            group = getattr(self, group_name, None)
            if group:
                for t in group.targets:
                    if t.test_path not in seen:
                        paths.append(t.test_path)
                        seen.add(t.test_path)
        return paths

    def to_summary(self) -> dict[str, Any]:
        return {
            "scope_level": self.selected_scope_level,
            "composite_risk_level": self.composite_risk_level,
            "primary_count": len(self.primary.targets),
            "secondary_count": len(self.secondary.targets),
            "fallback_count": len(self.fallback.targets),
            "total_target_count": self.total_target_count,
            "flattened_target_count": len(self.flatten_paths()),
        }


class VerificationSetBuilder:
    """
    Builds a structured VerificationSet from multiple test discovery sources.

    Group assignment rules:
    - primary: direct_symbol match, strong module_import, dep graph direct_import (score >= 0.7)
    - secondary: same_module, same_package, caller-impact adjacency (score >= 0.3)
    - fallback: filename heuristic, broad package tests, residual (score < 0.3)

    CompositeRisk escalation:
    - critical/high: include secondary and fallback groups
    - medium: include primary and secondary
    - low: primary mainly

    Never raises — returns minimal VerificationSet on error.
    """

    def build(
        self,
        symbol_targets: Optional[list] = None,
        dep_graph_candidates: Optional[list] = None,
        impact_set=None,
        composite_risk=None,
        scope_level_hint: str = "standard",
        gsg_pack_hints: Optional[dict] = None,
        patch_risk: Optional[object] = None,
    ) -> "VerificationSet":
        """
        Build verification set from available sources.

        Args:
            symbol_targets: List[SymbolAwareTestTarget] from P8-3
            dep_graph_candidates: List[TestCoverageCandidate] from P9-2
            impact_set: ImpactSet from P9-1
            composite_risk: CompositeRisk from P8-2
            scope_level_hint: "minimal" | "narrow" | "standard" | "broad"
            gsg_pack_hints: Optional dict from pack_to_planning_priorities (P4)
        """
        vset = VerificationSet()

        try:
            # Determine effective scope level from risk + hint
            risk_level = getattr(composite_risk, 'level', 'unknown') if composite_risk else 'unknown'
            vset.composite_risk_level = risk_level
            vset.selected_scope_level = self._resolve_scope_level(scope_level_hint, risk_level)

            # P4: GSG pack hints can further influence scope
            if gsg_pack_hints:
                _pack_scope = gsg_pack_hints.get("scope_hint", "")
                if _pack_scope == "broad" and vset.selected_scope_level != "broad":
                    # Only escalate, never narrow down
                    if vset.selected_scope_level in ("standard", "narrow", "minimal"):
                        vset.selected_scope_level = "standard" if vset.selected_scope_level == "narrow" else vset.selected_scope_level
                        # If pack says broad AND risk is not low, escalate to broad
                        if risk_level in ("high", "critical", "unknown"):
                            vset.selected_scope_level = "broad"
                _caution = gsg_pack_hints.get("caution_symbols", [])
                if _caution:
                    vset.metadata["gsg_caution_symbols"] = _caution[:5]

            # P9: patch risk-driven scope and metadata
            if patch_risk is not None:
                try:
                    from external_llm.editor.verification.impact_verification_mapper import (
                        map_verification_scope,
                        risk_to_verification_metadata,
                    )
                    _risk_scope = map_verification_scope(patch_risk)
                    # Escalate scope if risk warrants it (never narrow down)
                    _scope_order = ["minimal", "narrow", "standard", "broad"]
                    _current_idx = _scope_order.index(vset.selected_scope_level) if vset.selected_scope_level in _scope_order else 2
                    _risk_idx = _scope_order.index(_risk_scope) if _risk_scope in _scope_order else 2
                    if _risk_idx > _current_idx:
                        vset.selected_scope_level = _risk_scope
                    # Store risk metadata
                    vset.metadata["patch_risk"] = risk_to_verification_metadata(patch_risk)
                except Exception as exc:
                    logger.debug("P9 patch_risk integration failed: %s", exc)

            # Collect all candidates into unified format
            unified: dict[str, VerificationTarget] = {}

            # 1. Symbol-aware targets (highest priority source)
            if symbol_targets:
                for st in symbol_targets:
                    path = getattr(st, 'test_path', None) or str(st)
                    score = getattr(st, 'priority_score', 0.0)

                    t = unified.setdefault(path, VerificationTarget(test_path=path))
                    t.priority_score = max(t.priority_score, score)
                    t.source = "symbol_aware"
                    codes = getattr(st, 'reason_codes', [])
                    for c in codes:
                        if c not in t.reason_codes:
                            t.reason_codes.append(c)
                    syms = getattr(st, 'matched_symbols', [])
                    for s in syms:
                        if s not in t.matched_symbols:
                            t.matched_symbols.append(s)

            # 2. Dependency graph candidates
            if dep_graph_candidates:
                for dc in dep_graph_candidates:
                    path = getattr(dc, 'test_path', None) or str(dc)
                    score = getattr(dc, 'coverage_score', 0.0)

                    t = unified.setdefault(path, VerificationTarget(test_path=path))
                    t.priority_score = max(t.priority_score, score)
                    if t.source == "heuristic":
                        t.source = "dependency_graph"
                    codes = getattr(dc, 'reason_codes', [])
                    for c in codes:
                        if c not in t.reason_codes:
                            t.reason_codes.append(c)
                    files = getattr(dc, 'matched_files', [])
                    for f in files:
                        if f not in t.matched_files:
                            t.matched_files.append(f)

            # 3. Impact set fallback (files that are tests)
            if impact_set:
                for f in getattr(impact_set, 'impacted_files', []):
                    if self._is_test_file(f) and f not in unified:
                        t = VerificationTarget(
                            test_path=f,
                            priority_score=0.2,
                            source="impact_fallback",
                            reason_codes=["IMPACT_SET_TEST_FILE"],
                        )
                        unified[f] = t

            # P9: inject tests from patch risk impacted files
            if patch_risk is not None:
                _risk_files = getattr(patch_risk, "impacted_files", [])
                for f in _risk_files:
                    if self._is_test_file(f) and f not in unified:
                        t = VerificationTarget(
                            test_path=f,
                            priority_score=0.35,
                            source="patch_risk_impact",
                            reason_codes=["PATCH_RISK_IMPACTED_FILE"],
                        )
                        unified[f] = t

            # Assign to groups
            limits = SCOPE_LIMITS.get(vset.selected_scope_level, SCOPE_LIMITS["standard"])

            ranked = sorted(unified.values(), key=lambda t: -t.priority_score)

            for target in ranked:
                if target.priority_score >= PRIMARY_SCORE_THRESHOLD:
                    if len(vset.primary.targets) < limits["primary"]:
                        vset.primary.targets.append(target)
                    elif len(vset.secondary.targets) < limits["secondary"]:
                        vset.secondary.targets.append(target)
                elif target.priority_score >= SECONDARY_SCORE_THRESHOLD:
                    if len(vset.secondary.targets) < limits["secondary"]:
                        vset.secondary.targets.append(target)
                    elif len(vset.fallback.targets) < limits["fallback"]:
                        vset.fallback.targets.append(target)
                else:
                    if len(vset.fallback.targets) < limits["fallback"]:
                        vset.fallback.targets.append(target)

            # Set group metadata
            vset.primary.scope_level = vset.selected_scope_level
            vset.secondary.scope_level = vset.selected_scope_level
            vset.fallback.scope_level = vset.selected_scope_level

            vset.primary.rationale.append(
                f"Top {len(vset.primary.targets)} targets (score >= {PRIMARY_SCORE_THRESHOLD})"
            )
            vset.secondary.rationale.append(
                f"{len(vset.secondary.targets)} targets (score >= {SECONDARY_SCORE_THRESHOLD})"
            )
            vset.fallback.rationale.append(
                f"{len(vset.fallback.targets)} residual targets"
            )

        except Exception as e:
            logger.debug("VerificationSetBuilder failed: %s", e)
            # Return empty set

        return vset

    def _resolve_scope_level(self, hint: str, risk_level: str) -> str:
        """Resolve effective scope level from hint and risk."""
        # Risk can escalate but not reduce scope
        if risk_level == "critical":
            return "broad"
        if risk_level == "high" and hint in ("minimal", "narrow"):
            return "standard"
        if risk_level == "high" and hint == "standard":
            return "broad"
        return hint

    @staticmethod
    def _is_test_file(path: str) -> bool:
        import os
        name = os.path.basename(path)
        return name.startswith("test_") or name.endswith("_test.py")
