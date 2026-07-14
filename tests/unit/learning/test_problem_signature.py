"""Tests for ProblemSignature."""
from external_llm.editor.learning.problem_signature import ProblemSignature, build_problem_signature


class TestProblemSignature:
    def test_default(self):
        s = ProblemSignature()
        assert s.symbol == ""
        assert s.risk_level == "unknown"

    def test_key(self):
        s = ProblemSignature(failure_type="test_failure", module="mod.agent", symbol="Foo")
        assert "test_failure" in s.key
        assert "mod.agent" in s.key

    def test_fingerprint_deterministic(self):
        s1 = ProblemSignature(failure_type="test_failure", symbol="Foo")
        s2 = ProblemSignature(failure_type="test_failure", symbol="Foo")
        assert s1.fingerprint == s2.fingerprint

    def test_fingerprint_different(self):
        s1 = ProblemSignature(failure_type="test_failure")
        s2 = ProblemSignature(failure_type="apply_failed")
        assert s1.fingerprint != s2.fingerprint

    def test_similarity_identical(self):
        s = ProblemSignature(
            failure_type="test_failure", module="mod", symbol="Foo",
            risk_level="high", operation_kind="edit", request_type="refactor",
        )
        assert s.similarity_score(s) == 1.0

    def test_similarity_partial(self):
        s1 = ProblemSignature(failure_type="test_failure", module="mod")
        s2 = ProblemSignature(failure_type="test_failure", module="other")
        assert 0.0 < s1.similarity_score(s2) < 1.0

    def test_similarity_none(self):
        s1 = ProblemSignature(failure_type="test_failure", risk_level="high")
        s2 = ProblemSignature(failure_type="apply_failed", risk_level="low")
        assert s1.similarity_score(s2) == 0.0

    def test_to_dict(self):
        s = ProblemSignature(symbol="Foo", failure_type="test_failure")
        d = s.to_dict()
        assert d["symbol"] == "Foo"


class TestBuildProblemSignature:
    def test_from_symbols(self):
        sig = build_problem_signature(changed_symbols=["PlannerAgent.create_plan"])
        assert sig.symbol == "PlannerAgent.create_plan"
        assert sig.module == "PlannerAgent"

    def test_from_graph_context(self):
        sig = build_problem_signature(
            graph_context={
                "primary_files": ["external_llm/agent/planner.py"],
                "impact_files": [f"f{i}.py" for i in range(7)],
            }
        )
        assert sig.module == "external_llm.agent.planner"
        assert sig.impact_size == "large"

    def test_from_composite_risk(self):
        from unittest.mock import MagicMock
        risk = MagicMock()
        risk.level = "critical"
        sig = build_problem_signature(composite_risk=risk)
        assert sig.risk_level == "critical"

    def test_empty_input(self):
        sig = build_problem_signature()
        assert sig.symbol == ""
        assert sig.failure_type == ""
