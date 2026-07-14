"""
Unit tests for strategy outcome memory (Phase 4.2).
"""
import time

import pytest

from external_llm.agent.operation_models import (
    CandidateSelectionFeedback,
    StrategyOutcomeMemory,
    StrategyOutcomeStats,
    classify_request_type,
)
from external_llm.agent.run_store import InMemoryRunStore, RunRecord


def test_strategy_outcome_stats_dataclass():
    """Test that StrategyOutcomeStats can be instantiated and serialized."""
    stats = StrategyOutcomeStats(
        strategy="minimal_patch",
        selected_count=5,
        success_count=3,
        failure_count=2,
        verification_failure_count=1,
        rollback_count=0,
        repair_attempt_count=1,
        avg_selected_score=0.75,
        recent_requests=["fix bug", "refactor"],
        request_type_counts={"bugfix": 3, "structural": 2},
        metadata={"extra": "data"},
    )
    assert stats.strategy == "minimal_patch"
    assert stats.selected_count == 5
    assert stats.success_count == 3
    assert stats.failure_count == 2
    assert stats.verification_failure_count == 1
    assert stats.rollback_count == 0
    assert stats.repair_attempt_count == 1
    assert stats.avg_selected_score == 0.75
    assert stats.recent_requests == ["fix bug", "refactor"]
    assert stats.request_type_counts == {"bugfix": 3, "structural": 2}
    assert stats.metadata == {"extra": "data"}
    # Serialization
    dict_repr = stats.to_dict()
    assert dict_repr["strategy"] == "minimal_patch"
    assert dict_repr["selected_count"] == 5
    assert dict_repr["success_count"] == 3
    assert dict_repr["failure_count"] == 2
    assert dict_repr["verification_failure_count"] == 1
    assert dict_repr["rollback_count"] == 0
    assert dict_repr["repair_attempt_count"] == 1
    assert dict_repr["avg_selected_score"] == 0.75
    assert dict_repr["recent_requests"] == ["fix bug", "refactor"]
    assert dict_repr["request_type_counts"] == {"bugfix": 3, "structural": 2}
    assert dict_repr["metadata"] == {"extra": "data"}


def test_strategy_outcome_memory_dataclass():
    """Test that StrategyOutcomeMemory can be instantiated and serialized."""
    stats1 = StrategyOutcomeStats(strategy="minimal_patch", selected_count=3)
    stats2 = StrategyOutcomeStats(strategy="refactor", selected_count=2)
    memory = StrategyOutcomeMemory(
        strategies={"minimal_patch": stats1, "refactor": stats2},
        total_runs_considered=5,
        generated_at="2026-03-13T12:00:00",
        metadata={"source": "test"},
    )
    assert len(memory.strategies) == 2
    assert memory.total_runs_considered == 5
    assert memory.generated_at == "2026-03-13T12:00:00"
    assert memory.metadata == {"source": "test"}
    # Serialization
    dict_repr = memory.to_dict()
    assert dict_repr["total_runs_considered"] == 5
    assert dict_repr["generated_at"] == "2026-03-13T12:00:00"
    assert dict_repr["metadata"] == {"source": "test"}
    assert "minimal_patch" in dict_repr["strategies"]
    assert dict_repr["strategies"]["minimal_patch"]["selected_count"] == 3
    assert dict_repr["strategies"]["refactor"]["selected_count"] == 2


def test_classify_request_type_deprecated():
    """classify_request_type is deprecated — always returns 'unknown'.

    Request type classification is now done by SpecResolver LLM
    and stored in spec.request_type.
    """
    assert classify_request_type("fix bug in function") == "unknown"
    assert classify_request_type("refactor this code") == "unknown"
    assert classify_request_type("") == "unknown"
    assert classify_request_type("random text") == "unknown"


def test_build_strategy_outcome_memory_empty_store():
    """Test aggregation when store has no runs with candidate feedback."""
    store = InMemoryRunStore(max_runs=10)
    memory = store.build_strategy_outcome_memory()
    assert memory.total_runs_considered == 0
    assert memory.strategies == {}
    assert memory.generated_at is not None
    assert "limit" in memory.metadata


def test_build_strategy_outcome_memory_with_feedback():
    """Test aggregation across multiple runs with different strategies."""
    store = InMemoryRunStore(max_runs=10)
    # Add run records with candidate_feedback
    for i in range(5):
        feedback = CandidateSelectionFeedback(
            request=f"fix bug {i}",
            selected_strategy="minimal_patch" if i % 2 == 0 else "refactor",
            selected_score=0.8 if i % 2 == 0 else 0.6,
            execution_success=(i != 2),  # third run fails
            final_status="completed" if i != 2 else "verification_failed",
            rolled_back=(i == 3),
            repair_attempted=(i == 4),
        ).to_dict()
        record = RunRecord(
            run_id=f"run-{i:06d}",
            timestamp=time.time() + i,
            plan_mode="EDIT",
            operation_count=1,
            completed=1,
            failed=0,
            skipped=0,
            final_status="success",
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
            candidate_feedback=feedback,
            request_type="bugfix",
        )
        store.add_run(record)
    memory = store.build_strategy_outcome_memory(limit=10)
    # Should have two strategies
    assert len(memory.strategies) == 2
    assert "minimal_patch" in memory.strategies
    assert "refactor" in memory.strategies
    # Counts: minimal_patch runs: i=0,2,4 (3 runs), refactor: i=1,3 (2 runs)
    mp_stats = memory.strategies["minimal_patch"]
    assert mp_stats.selected_count == 3
    # successes: i=0 success, i=2 failure, i=4 success -> 2 successes, 1 failure
    assert mp_stats.success_count == 2
    assert mp_stats.failure_count == 1
    # verification_failure_count: i=2 final_status verification_failed
    assert mp_stats.verification_failure_count == 1
    # rollback_count: i=3 rolled_back True (but i=3 is refactor, not minimal_patch) => 0
    assert mp_stats.rollback_count == 0
    # repair_attempt_count: i=4 repair_attempted True
    assert mp_stats.repair_attempt_count == 1
    # avg_selected_score: scores 0.8, 0.8, 0.8? Wait i=0,2,4 all minimal_patch, selected_score 0.8 each
    assert mp_stats.avg_selected_score == pytest.approx(0.8)
    # request_type_counts: all requests contain "bug" -> bugfix
    assert mp_stats.request_type_counts == {"bugfix": 3}
    # recent_requests length limited to default 5, but we have 3 entries
    assert len(mp_stats.recent_requests) == 3
    # refactor stats
    rf_stats = memory.strategies["refactor"]
    assert rf_stats.selected_count == 2
    assert rf_stats.success_count == 2  # i=1 success, i=3 success (since i!=2)
    assert rf_stats.failure_count == 0
    assert rf_stats.verification_failure_count == 0
    assert rf_stats.rollback_count == 1  # i=3 rolled_back True
    assert rf_stats.repair_attempt_count == 0  # i=4 is minimal_patch
    assert rf_stats.avg_selected_score == pytest.approx(0.6)
    assert rf_stats.request_type_counts == {"bugfix": 2}
    # total runs considered: 5
    assert memory.total_runs_considered == 5


def test_build_strategy_outcome_memory_missing_fields():
    """Test graceful handling of missing candidate_feedback fields."""
    store = InMemoryRunStore(max_runs=10)
    # Feedback missing selected_strategy -> should be skipped
    feedback = {"request": "fix bug", "execution_success": True}
    record = RunRecord(
        run_id="run-000001",
        timestamp=time.time(),
        plan_mode="EDIT",
        operation_count=1,
        completed=1,
        failed=0,
        skipped=0,
        final_status="success",
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
        candidate_feedback=feedback,
    )
    store.add_run(record)
    memory = store.build_strategy_outcome_memory()
    assert memory.total_runs_considered == 0
    # Feedback with selected_strategy but missing other fields
    feedback2 = {"selected_strategy": "minimal_patch", "request": "fix bug"}
    record2 = RunRecord(
        run_id="run-000002",
        timestamp=time.time() + 1,
        plan_mode="EDIT",
        operation_count=1,
        completed=1,
        failed=0,
        skipped=0,
        final_status="success",
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
        candidate_feedback=feedback2,
    )
    store.add_run(record2)
    memory2 = store.build_strategy_outcome_memory()
    assert memory2.total_runs_considered == 1
    stats = memory2.strategies["minimal_patch"]
    assert stats.selected_count == 1
    assert stats.success_count == 0  # missing execution_success
    assert stats.failure_count == 0
    assert stats.verification_failure_count == 0
    assert stats.rollback_count == 0
    assert stats.repair_attempt_count == 0
    assert stats.avg_selected_score == 0.0
    # request_type not set on RunRecord → falls back to "unknown"
    assert stats.request_type_counts == {"unknown": 1}


def test_build_strategy_outcome_memory_limit():
    """Test that limit parameter restricts number of runs considered."""
    store = InMemoryRunStore(max_runs=10)
    for i in range(10):
        feedback = CandidateSelectionFeedback(
            request=f"req {i}",
            selected_strategy="minimal_patch",
            selected_score=0.5,
            execution_success=True,
        ).to_dict()
        record = RunRecord(
            run_id=f"run-{i:06d}",
            timestamp=time.time() + i,
            plan_mode="EDIT",
            operation_count=1,
            completed=1,
            failed=0,
            skipped=0,
            final_status="success",
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
            candidate_feedback=feedback,
            request_type="bugfix",
        )
        store.add_run(record)
    memory = store.build_strategy_outcome_memory(limit=3)
    # Should only consider 3 most recent runs (newest first)
    assert memory.total_runs_considered == 3
    assert memory.strategies["minimal_patch"].selected_count == 3
    # recent_requests should be limited to default 5, but we have only 3
    assert len(memory.strategies["minimal_patch"].recent_requests) == 3
    # Ensure the requests are the most recent ones (i=9,8,7)
    reqs = memory.strategies["minimal_patch"].recent_requests
    assert "req 9" in reqs[0]
    assert "req 8" in reqs[1]
    assert "req 7" in reqs[2]


def test_recent_requests_truncation():
    """Test that recent_requests are truncated to recent_requests_limit."""
    store = InMemoryRunStore(max_runs=10)
    for i in range(10):
        feedback = CandidateSelectionFeedback(
            request=f"req {i}",
            selected_strategy="minimal_patch",
            execution_success=True,
        ).to_dict()
        record = RunRecord(
            run_id=f"run-{i:06d}",
            timestamp=time.time() + i,
            plan_mode="EDIT",
            operation_count=1,
            completed=1,
            failed=0,
            skipped=0,
            final_status="success",
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
            candidate_feedback=feedback,
            request_type="bugfix",
        )
        store.add_run(record)
    memory = store.build_strategy_outcome_memory(recent_requests_limit=3)
    stats = memory.strategies["minimal_patch"]
    # Should keep only 3 most recent requests (i=9,8,7)
    assert len(stats.recent_requests) == 3
    assert stats.recent_requests == ["req 9", "req 8", "req 7"]


def test_get_strategy_summary_for_request():
    """Test request-aware summary filtering."""
    store = InMemoryRunStore(max_runs=10)
    # Add runs with different request types
    feedback1 = CandidateSelectionFeedback(
        request="fix bug in function",
        selected_strategy="minimal_patch",
        execution_success=True,
    ).to_dict()
    feedback2 = CandidateSelectionFeedback(
        request="refactor code",
        selected_strategy="refactor",
        execution_success=True,
    ).to_dict()
    feedback3 = CandidateSelectionFeedback(
        request="another bug fix",
        selected_strategy="minimal_patch",
        execution_success=False,
    ).to_dict()
    _request_types = ["bugfix", "refactor", "bugfix"]
    for i, (fb, rt) in enumerate(zip([feedback1, feedback2, feedback3], _request_types, strict=False)):
        record = RunRecord(
            run_id=f"run-{i:06d}",
            timestamp=time.time() + i,
            plan_mode="EDIT",
            operation_count=1,
            completed=1,
            failed=0,
            skipped=0,
            final_status="success",
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
            candidate_feedback=fb,
            request_type=rt,
        )
        store.add_run(record)
    # request_type is now passed explicitly (not inferred from text)
    summary = store.get_strategy_summary_for_request("fix bug", limit=10, request_type="bugfix")
    assert summary["request_type"] == "bugfix"
    # Only strategies that have been used for bugfix requests: minimal_patch (2 runs)
    assert set(summary["strategies"].keys()) == {"minimal_patch"}
    assert summary["strategies"]["minimal_patch"]["selected_count"] == 2
    assert summary["strategies"]["minimal_patch"]["success_count"] == 1
    assert summary["strategies"]["minimal_patch"]["failure_count"] == 1
    assert summary["total_runs_considered"] == 3
    # Test with refactor request type
    summary2 = store.get_strategy_summary_for_request("refactor something", limit=10, request_type="refactor")
    assert summary2["request_type"] == "refactor"
    assert set(summary2["strategies"].keys()) == {"refactor"}
    assert summary2["strategies"]["refactor"]["selected_count"] == 1
    # Unknown request type
    summary3 = store.get_strategy_summary_for_request("random text", limit=10, request_type="unknown")
    assert summary3["request_type"] == "unknown"
    # No strategies have been used for unknown request type
    assert summary3["strategies"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
