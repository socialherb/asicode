"""Tests for enums.py — shared enum definitions (Complexity, Scope, EstimatedScope)."""

from __future__ import annotations

from external_llm.agent.enums import (
    Complexity,
    EstimatedScope,
    Scope,
    estimated_scope_to_score,
    scope_to_score,
)


class TestComplexity:
    def test_values(self):
        assert Complexity.LOW == "LOW"
        assert Complexity.MEDIUM == "MEDIUM"
        assert Complexity.HIGH == "HIGH"

    def test_members(self):
        assert set(Complexity.__members__) == {"LOW", "MEDIUM", "HIGH"}


class TestScope:
    def test_values(self):
        assert Scope.SINGLE_FILE == "single_file"
        assert Scope.MULTI_FILE == "multi_file"
        assert Scope.PROJECT_WIDE == "project_wide"

    def test_members(self):
        assert set(Scope.__members__) == {"SINGLE_FILE", "MULTI_FILE", "PROJECT_WIDE"}


class TestEstimatedScope:
    def test_values(self):
        assert EstimatedScope.TINY == "tiny"
        assert EstimatedScope.SMALL == "small"
        assert EstimatedScope.MEDIUM == "medium"
        assert EstimatedScope.LARGE == "large"


class TestEstimatedScopeToScore:
    """Coverage for estimated_scope_to_score() — all branches."""

    def test_tiny(self):
        assert estimated_scope_to_score(EstimatedScope.TINY) == 0.0

    def test_small(self):
        assert estimated_scope_to_score(EstimatedScope.SMALL) == 0.1

    def test_medium(self):
        assert estimated_scope_to_score(EstimatedScope.MEDIUM) == 0.4

    def test_large(self):
        assert estimated_scope_to_score(EstimatedScope.LARGE) == 0.7

    def test_all_have_mapping(self):
        """Every EstimatedScope member must have a mapping."""
        for scope in EstimatedScope:
            score = estimated_scope_to_score(scope)
            assert isinstance(score, float)


class TestScopeToScore:
    """Coverage for scope_to_score() — all branches."""

    def test_single_file(self):
        assert scope_to_score(Scope.SINGLE_FILE) == 0.1

    def test_multi_file(self):
        assert scope_to_score(Scope.MULTI_FILE) == 0.4

    def test_project_wide(self):
        assert scope_to_score(Scope.PROJECT_WIDE) == 0.7

    def test_all_have_mapping(self):
        """Every Scope member must have a mapping."""
        for scope in Scope:
            score = scope_to_score(scope)
            assert isinstance(score, float)


class TestRoundTrip:
    """Score mappings should be monotonic."""

    def test_estimated_scope_monotonic(self):
        scores = [estimated_scope_to_score(s) for s in EstimatedScope]
        assert scores == sorted(scores)

    def test_scope_monotonic(self):
        scores = [scope_to_score(s) for s in Scope]
        assert scores == sorted(scores)
