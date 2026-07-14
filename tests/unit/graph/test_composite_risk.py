"""Tests for CompositeRisk model and CompositeRiskEvaluator."""
from external_llm.agent.execution_thresholds import THRESHOLDS
from external_llm.graph.composite_risk import (
    CompositeRisk,
    CompositeRiskEvaluator,
)

CRITICAL_THRESHOLD = THRESHOLDS.risk_critical_score
HIGH_THRESHOLD = THRESHOLDS.risk_high_score
MEDIUM_THRESHOLD = THRESHOLDS.risk_medium_score


class TestCompositeRisk:
    def test_default_is_low(self):
        r = CompositeRisk()
        assert r.level == "low"
        assert r.score == 0

    def test_to_dict(self):
        r = CompositeRisk(level="high", score=5, reason_codes=["A", "B"])
        d = r.to_dict()
        assert d["level"] == "high"
        assert d["score"] == 5

    def test_to_summary(self):
        r = CompositeRisk(level="critical", score=8, reason_codes=["X"])
        s = r.to_summary()
        assert s["level"] == "critical"
        assert "X" in s["reason_codes"]


class TestCompositeRiskEvaluator:
    def test_all_clear_is_low(self):
        """No risk signals → low."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(graph_confidence=0.9, resolved_symbol_count=3)
        assert r.level == "low"

    def test_low_confidence_structural_is_high(self):
        """Low confidence + structural op → high score."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(graph_confidence=0.2, is_structural_op=True)
        assert r.score >= 3
        assert "LOW_CONFIDENCE_STRUCTURAL" in r.reason_codes

    def test_unresolved_symbols_add_risk(self):
        """Unresolved symbols >= 2 → +2."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(unresolved_symbol_count=3)
        assert r.score >= 2
        assert "UNRESOLVED_SYMBOLS" in r.reason_codes

    def test_wide_impact_adds_risk(self):
        """Impact files >= 6 → +2."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(impact_file_count=8)
        assert r.score >= 2
        assert "WIDE_IMPACT" in r.reason_codes

    def test_high_caller_count_adds_risk(self):
        """Caller count >= 10 → +2."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(caller_count=12)
        assert r.score >= 2
        assert "HIGH_CALLER_COUNT" in r.reason_codes

    def test_structural_op_adds_risk(self):
        """Structural operation → +1."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(operation_kind="RENAME_SYMBOL")
        assert "STRUCTURAL_OP" in r.reason_codes

    def test_multiple_safety_issues_add_risk(self):
        """2+ safety issues → +2."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(safety_issue_codes=["A", "B", "C"])
        assert r.score >= 2
        assert "MULTIPLE_SAFETY_ISSUES" in r.reason_codes

    def test_precise_low_impact_reduces_risk(self):
        """High confidence + resolved + low impact → -2."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(
            graph_confidence=0.9,
            resolved_symbol_count=3,
            impact_file_count=1,
        )
        assert "PRECISE_LOW_IMPACT" in r.reason_codes
        assert r.score == 0  # -2 + -1 clamped to 0

    def test_all_resolved_reduces_risk(self):
        """All resolved + none unresolved → -1."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(resolved_symbol_count=2, unresolved_symbol_count=0)
        assert "ALL_RESOLVED" in r.reason_codes

    def test_critical_level(self):
        """Combined high signals → critical."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(
            graph_confidence=0.2,
            is_structural_op=True,          # +3
            unresolved_symbol_count=3,       # +2
            impact_file_count=8,             # +2
            caller_count=12,                 # +2
        )
        assert r.level == "critical"
        assert r.score >= CRITICAL_THRESHOLD

    def test_medium_level(self):
        """Moderate signals → medium."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(
            unresolved_symbol_count=2,       # +2
        )
        assert r.level == "medium"

    def test_high_level(self):
        """Several risk signals → high."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(
            impact_file_count=7,             # +2
            caller_count=11,                 # +2
        )
        assert r.level == "high"

    def test_evaluate_from_metadata(self):
        """evaluate_from_metadata extracts signals correctly."""
        e = CompositeRiskEvaluator()
        r = e.evaluate_from_metadata(
            graph_context={
                "graph_confidence": 0.3,
                "resolved_symbols": [{"name": "foo"}],
                "unresolved_symbols": ["bar", "baz"],
                "impact_files": [f"f{i}.py" for i in range(7)],
                "callers": {"foo": [{"symbol": f"c{i}"} for i in range(5)]},
            },
            is_structural_op=True,
        )
        assert r.level in ("high", "critical")
        assert r.score >= HIGH_THRESHOLD

    def test_evaluate_from_metadata_empty(self):
        """Empty metadata → low risk."""
        e = CompositeRiskEvaluator()
        r = e.evaluate_from_metadata()
        assert r.level == "low"

    def test_error_returns_low(self):
        """Evaluator errors → safe default (low)."""
        e = CompositeRiskEvaluator()
        # Pass broken input
        r = e.evaluate(graph_confidence="not_a_number")
        assert r.level == "low"

    def test_score_clamped_to_zero(self):
        """Score cannot go below 0."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(
            graph_confidence=0.95,
            resolved_symbol_count=5,
            unresolved_symbol_count=0,
            impact_file_count=1,
        )
        assert r.score >= 0

    def test_source_signals_recorded(self):
        """Source signals stored in result."""
        e = CompositeRiskEvaluator()
        r = e.evaluate(graph_confidence=0.5, caller_count=3)
        assert r.source_signals["graph_confidence"] == 0.5
        assert r.source_signals["caller_count"] == 3

    def test_deterministic(self):
        """Same inputs → same result."""
        e = CompositeRiskEvaluator()
        r1 = e.evaluate(graph_confidence=0.3, is_structural_op=True, impact_file_count=7)
        r2 = e.evaluate(graph_confidence=0.3, is_structural_op=True, impact_file_count=7)
        assert r1.score == r2.score
        assert r1.level == r2.level
