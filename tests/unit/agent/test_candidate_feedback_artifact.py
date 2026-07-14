"""
Unit tests for candidate selection feedback artifact (Phase 4.1).
"""

import pytest

from external_llm.agent.operation_models import (
    CandidateSelectionFeedback,
    ExecutionResult,
    OperationPlan,
    PlanMode,
    build_candidate_feedback_from_plan,
    enrich_candidate_feedback_with_execution,
)
from external_llm.agent.run_store import RunRecord


def test_candidate_selection_feedback_dataclass():
    """Test that CandidateSelectionFeedback can be instantiated and serialized."""
    feedback = CandidateSelectionFeedback(
        request="test request",
        candidate_count=3,
        selected_candidate_id="cand_1",
        selected_strategy="minimal_patch",
        selected_score=0.85,
        selected_source="planner",
        rejected_candidates=[
            {"id": "cand_2", "source": "variant", "score": 0.7},
            {"id": "cand_3", "source": "fallback", "score": 0.5},
        ],
        ranking_explanations=["explanation 1"],
        strategy_distribution=["minimal_patch", "refactor"],
        graph_used=True,
        impact_simulation_enabled=False,
        graph_repo_root="/some/path",
    )
    # Check fields
    assert feedback.request == "test request"
    assert feedback.candidate_count == 3
    assert feedback.selected_candidate_id == "cand_1"
    assert feedback.selected_strategy == "minimal_patch"
    assert feedback.selected_score == 0.85
    assert feedback.selected_source == "planner"
    assert len(feedback.rejected_candidates) == 2
    assert feedback.ranking_explanations == ["explanation 1"]
    assert feedback.strategy_distribution == ["minimal_patch", "refactor"]
    assert feedback.graph_used is True
    assert feedback.impact_simulation_enabled is False
    assert feedback.graph_repo_root == "/some/path"
    # Serialization
    dict_repr = feedback.to_dict()
    assert isinstance(dict_repr, dict)
    assert dict_repr["request"] == "test request"
    assert dict_repr["candidate_count"] == 3
    # Ensure no missing keys
    expected_keys = {
        "request", "candidate_count", "selected_candidate_id", "selected_strategy",
        "selected_score", "selected_source", "rejected_candidates", "ranking_explanations",
        "strategy_distribution", "graph_used", "impact_simulation_enabled",
        "graph_repo_root", "execution_run_id", "execution_success", "final_status",
        "final_failure_class", "verification_summary", "proof_summary", "rolled_back",
        "repair_attempted", "metadata",
        # Phase 5.3: strategy switching metadata
        "switched_from_strategy", "switched_to_strategy", "switch_reason", "switch_hop",
        "previous_strategies_tried", "switch_chain", "switch_memory_score",
        "switch_memory_explanations",
        # Phase 5.5: multi-hop strategy chain metadata
        "strategy_chain", "strategy_chain_score", "strategy_chain_depth",
        "strategy_chain_initial_strategy", "strategy_chain_final_strategy",
        "is_multi_hop_strategy_chain",
    }
    assert set(dict_repr.keys()) == expected_keys


def test_build_candidate_feedback_from_plan_with_selection():
    """Test building feedback from a plan containing candidate selection metadata."""
    # Create a minimal plan with candidate_selection metadata
    plan = OperationPlan(
        operations=[],
        mode=PlanMode.EDIT,
        metadata={
            "candidate_selection": {
                "candidate_count": 2,
                "selected_candidate_id": "cand_a",
                "selected_candidate_source": "planner",
                "selected_candidate_score": 0.9,
                "selected_strategy": "refactor",
                "ranking_explanations": ["good strategy"],
                "strategy_distribution": ["refactor", "minimal_patch"],
                "rejected_candidates": [
                    {"id": "cand_b", "source": "variant", "score": 0.6}
                ],
                "graph_used": False,
                "impact_simulation_enabled": True,
                "graph_repo_root": None,
            }
        }
    )
    feedback = build_candidate_feedback_from_plan(plan, request="original")
    assert feedback.request == "original"
    assert feedback.candidate_count == 2
    assert feedback.selected_candidate_id == "cand_a"
    assert feedback.selected_strategy == "refactor"
    assert feedback.selected_score == 0.9
    assert feedback.selected_source == "planner"
    assert len(feedback.rejected_candidates) == 1
    assert feedback.rejected_candidates[0]["id"] == "cand_b"
    assert feedback.ranking_explanations == ["good strategy"]
    assert feedback.strategy_distribution == ["refactor", "minimal_patch"]
    assert feedback.graph_used is False
    assert feedback.impact_simulation_enabled is True
    assert feedback.graph_repo_root is None
    # Execution fields should be None/empty
    assert feedback.execution_run_id is None
    assert feedback.execution_success is None
    assert feedback.final_status is None
    assert feedback.final_failure_class is None
    assert feedback.verification_summary == {}
    assert feedback.proof_summary == {}
    assert feedback.rolled_back is None
    assert feedback.repair_attempted is None


def test_build_candidate_feedback_from_plan_without_selection():
    """Test building feedback from a plan without candidate selection metadata."""
    plan = OperationPlan(operations=[], mode=PlanMode.EDIT, metadata={})
    feedback = build_candidate_feedback_from_plan(plan, request="test")
    # Should return empty artifact with request set
    assert feedback.request == "test"
    assert feedback.candidate_count == 0
    assert feedback.selected_candidate_id is None
    assert feedback.selected_strategy is None
    assert feedback.selected_score is None
    assert feedback.selected_source is None
    assert feedback.rejected_candidates == []
    assert feedback.ranking_explanations == []
    assert feedback.strategy_distribution == []
    assert feedback.graph_used is None
    assert feedback.impact_simulation_enabled is None
    assert feedback.graph_repo_root is None


def test_enrich_candidate_feedback_with_execution():
    """Test enriching feedback with execution outcome."""
    feedback = CandidateSelectionFeedback(request="test")
    exec_result = ExecutionResult(
        success=True,
        final_status="completed",
        final_failure_class=None,
        run_id="run_123",
        verification={
            "success": True,
            "blocking_reasons": [],
            "warnings": ["minor"],
        },
        proof={
            "modified_files": ["file.py"],
            "rolled_back": False,
        },
        repair={},  # empty dict means no repair attempted
    )
    enriched = enrich_candidate_feedback_with_execution(feedback, exec_result)
    # Same object (mutated)
    assert enriched is feedback
    assert feedback.execution_run_id == "run_123"
    assert feedback.execution_success is True
    assert feedback.final_status == "completed"
    assert feedback.final_failure_class is None
    assert feedback.verification_summary == {
        "success": True,
        "blocking_reasons": [],
        "warnings": ["minor"],
    }
    assert feedback.proof_summary == {
        "modified_files": ["file.py"],
        "rolled_back": False,
    }
    assert feedback.rolled_back is False
    assert feedback.repair_attempted is False


def test_enrich_candidate_feedback_with_execution_repair_attempted():
    """Test repair_attempted detection when repair dict is non-empty."""
    feedback = CandidateSelectionFeedback(request="test")
    exec_result = ExecutionResult(
        success=False,
        final_status="verification_failed",
        final_failure_class="semantic_error",
        run_id="run_456",
        verification={},
        proof={"rolled_back": True},
        repair={"rounds": 1},  # non-empty dict indicates repair attempted
    )
    enrich_candidate_feedback_with_execution(feedback, exec_result)
    assert feedback.repair_attempted is True
    assert feedback.rolled_back is True


def test_run_record_includes_candidate_feedback():
    """Test that RunRecord has candidate_feedback field and accepts dict."""
    # This test ensures backward compatibility: RunRecord can be instantiated without candidate_feedback
    record = RunRecord(
        run_id="run_1",
        timestamp=123.456,
        plan_mode="edit",
        operation_count=0,
        completed=0,
        failed=0,
        skipped=0,
        final_status="completed",
        final_failure_class=None,
        final_blocking_reasons=[],
        final_warning_reasons=[],
        semantic_gate_passed=True,
        semantic_gate_failed_reasons=[],
        plan_acceptance_passed=True,
        plan_acceptance_failed_checks=[],
        repair_attempted=False,
        repair_rounds_attempted=0,
        repair_improved=False,
        semantic_issue_codes=[],
        dependency_issues=[],
        completed_ids=[],
        failed_ids=[],
        skipped_ids=[],
        skipped_reasons={},
        proof={},
        candidate_feedback={"test": "value"},
    )
    assert record.candidate_feedback == {"test": "value"}
    # Default value when omitted (should be empty dict)
    record2 = RunRecord(
        run_id="run_2",
        timestamp=123.456,
        plan_mode="edit",
        operation_count=0,
        completed=0,
        failed=0,
        skipped=0,
        final_status="completed",
        final_failure_class=None,
        final_blocking_reasons=[],
        final_warning_reasons=[],
        semantic_gate_passed=True,
        semantic_gate_failed_reasons=[],
        plan_acceptance_passed=True,
        plan_acceptance_failed_checks=[],
        repair_attempted=False,
        repair_rounds_attempted=0,
        repair_improved=False,
        semantic_issue_codes=[],
        dependency_issues=[],
        completed_ids=[],
        failed_ids=[],
        skipped_ids=[],
        skipped_reasons={},
        proof={},
    )
    assert record2.candidate_feedback == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
