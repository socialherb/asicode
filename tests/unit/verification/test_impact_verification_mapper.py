"""Tests for impact_verification_mapper.py — P9 verification expansion."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from external_llm.editor.verification.impact_verification_mapper import (
    expand_verification_targets,
    map_impacted_tests,
    map_verification_scope,
    risk_to_verification_metadata,
)


@dataclass
class PatchRiskEstimate:
    overall_risk: str = "low"
    risk_score: float = 0.0
    impacted_symbols: list[str] = field(default_factory=list)
    caution_symbols: list[str] = field(default_factory=list)
    impacted_files: list[str] = field(default_factory=list)
    verification_scope_hint: str = ""


@dataclass
class _MockTestTarget:
    test_path: str = ""
    priority_score: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    matched_symbols: list[str] = field(default_factory=list)
    matched_files: list[str] = field(default_factory=list)


class TestMapVerificationScope:
    def test_none_returns_standard(self):
        assert map_verification_scope(None) == "standard"

    def test_low_returns_narrow(self):
        risk = PatchRiskEstimate(overall_risk="low")
        assert map_verification_scope(risk) == "narrow"

    def test_medium_returns_standard(self):
        risk = PatchRiskEstimate(overall_risk="medium")
        assert map_verification_scope(risk) == "standard"

    def test_high_returns_broad(self):
        risk = PatchRiskEstimate(overall_risk="high")
        assert map_verification_scope(risk) == "broad"

    def test_critical_returns_broad(self):
        risk = PatchRiskEstimate(overall_risk="critical")
        assert map_verification_scope(risk) == "broad"


class TestMapImpactedTests:
    def test_none_risk_returns_same_list(self):
        tests = [_MockTestTarget(test_path="t.py", priority_score=0.5)]
        result = map_impacted_tests(None, tests)
        assert result is tests
        assert result[0].priority_score == 0.5

    def test_empty_tests_returns_empty(self):
        risk = PatchRiskEstimate(impacted_symbols=["foo"])
        assert map_impacted_tests(risk, []) == []

    def test_impacted_symbol_boosts_score(self):
        risk = PatchRiskEstimate(impacted_symbols=["target_func"])
        t = _MockTestTarget(
            test_path="test_a.py", priority_score=0.5,
            matched_symbols=["target_func"],
        )
        map_impacted_tests(risk, [t])
        assert t.priority_score == pytest.approx(0.7, abs=0.01)
        assert "IMPACTED_SYMBOL_MATCH" in t.reason_codes

    def test_caution_symbol_boosts_score(self):
        risk = PatchRiskEstimate(caution_symbols=["dispatch"])
        t = _MockTestTarget(
            test_path="test_a.py", priority_score=0.5,
            matched_symbols=["dispatch"],
        )
        map_impacted_tests(risk, [t])
        assert t.priority_score >= 0.6
        assert "CAUTION_SYMBOL_PROXIMITY" in t.reason_codes

    def test_impacted_file_boosts_score(self):
        risk = PatchRiskEstimate(impacted_files=["core.py"])
        t = _MockTestTarget(
            test_path="test_core.py", priority_score=0.3,
        )
        map_impacted_tests(risk, [t])
        assert t.priority_score >= 0.4
        assert "IMPACTED_FILE_MATCH" in t.reason_codes

    def test_score_capped_at_1(self):
        risk = PatchRiskEstimate(
            impacted_symbols=["func"],
            caution_symbols=["func"],
            impacted_files=["a.py"],
        )
        t = _MockTestTarget(
            test_path="test_a.py", priority_score=0.9,
            matched_symbols=["func"],
            matched_files=["a.py"],
        )
        map_impacted_tests(risk, [t])
        assert t.priority_score <= 1.0

    def test_no_match_no_boost(self):
        risk = PatchRiskEstimate(impacted_symbols=["other"])
        t = _MockTestTarget(
            test_path="test_x.py", priority_score=0.5,
            matched_symbols=["unrelated"],
        )
        map_impacted_tests(risk, [t])
        assert t.priority_score == 0.5
        assert "IMPACTED_SYMBOL_MATCH" not in t.reason_codes


class TestExpandVerificationTargets:
    def test_none_returns_empty(self):
        assert expand_verification_targets(None) == {}

    def test_includes_symbols_and_files(self):
        risk = PatchRiskEstimate(
            impacted_symbols=["foo", "bar"],
            impacted_files=["a.py"],
            caution_symbols=["foo"],
        )
        result = expand_verification_targets(risk)
        assert "foo" in result["extra_symbols"]
        assert "a.py" in result["extra_files"]
        assert "foo" in result["caution_symbols"]
        assert result["scope_hint"] in ("narrow", "standard", "broad")

    def test_scope_hint_from_risk(self):
        risk = PatchRiskEstimate(overall_risk="high")
        result = expand_verification_targets(risk)
        assert result["scope_hint"] == "broad"


class TestRiskToVerificationMetadata:
    def test_none_returns_empty(self):
        assert risk_to_verification_metadata(None) == {}

    def test_populated_metadata(self):
        risk = PatchRiskEstimate(
            overall_risk="high",
            risk_score=0.55,
            impacted_files=["a.py"],
            impacted_symbols=["func"],
            caution_symbols=["func"],
            verification_scope_hint="broad",
        )
        meta = risk_to_verification_metadata(risk)
        assert meta["level"] == "high"
        assert meta["score"] == 0.55
        assert "a.py" in meta["impacted_files"]
        assert "func" in meta["caution_symbols"]


class TestVerificationSetBuilderIntegration:
    """Test that VerificationSetBuilder.build() accepts patch_risk."""

    def test_build_with_no_risk_works(self):
        from external_llm.editor.verification.verification_set_builder import VerificationSetBuilder
        builder = VerificationSetBuilder()
        vset = builder.build()
        assert vset is not None

    def test_build_with_risk_sets_metadata(self):
        from external_llm.editor.verification.verification_set_builder import VerificationSetBuilder
        builder = VerificationSetBuilder()
        risk = PatchRiskEstimate(
            overall_risk="high",
            risk_score=0.55,
            impacted_files=["a.py"],
            impacted_symbols=["func"],
        )
        vset = builder.build(patch_risk=risk)
        assert "patch_risk" in vset.metadata
        assert vset.metadata["patch_risk"]["level"] == "high"

    def test_build_risk_escalates_scope(self):
        from external_llm.editor.verification.verification_set_builder import VerificationSetBuilder
        builder = VerificationSetBuilder()
        risk = PatchRiskEstimate(overall_risk="critical")
        vset = builder.build(scope_level_hint="narrow", patch_risk=risk)
        # Critical risk should escalate scope
        assert vset.selected_scope_level in ("standard", "broad")
