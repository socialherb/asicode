"""Tests for PatternExtractor."""
from external_llm.editor.learning.pattern_extractor import PatternExtractor, StrategyPattern


class TestStrategyPattern:
    def test_to_dict(self):
        p = StrategyPattern(context_key="k", best_strategy="refactor", success_rate=0.8, sample_count=5)
        d = p.to_dict()
        assert d["best_strategy"] == "refactor"


class TestPatternExtractor:
    def test_extract_basic_pattern(self):
        extractor = PatternExtractor()
        experiences = [
            {"problem_signature": {"failure_type": "test_failure", "module": "mod"}, "strategy_used": "update_callers", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": "mod"}, "strategy_used": "update_callers", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": "mod"}, "strategy_used": "update_callers", "success": True},
        ]
        patterns = extractor.extract_patterns(experiences, min_samples=3)
        assert len(patterns) > 0
        assert patterns[0].best_strategy == "update_callers"

    def test_min_samples_filter(self):
        extractor = PatternExtractor()
        experiences = [
            {"problem_signature": {"failure_type": "test_failure"}, "strategy_used": "x", "success": True},
        ]
        patterns = extractor.extract_patterns(experiences, min_samples=3)
        assert len(patterns) == 0

    def test_best_strategy_by_success_rate(self):
        extractor = PatternExtractor()
        experiences = [
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "a", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "a", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "a", "success": False},
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "b", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "b", "success": True},
            {"problem_signature": {"failure_type": "test_failure", "module": ""}, "strategy_used": "b", "success": True},
        ]
        patterns = extractor.extract_patterns(experiences, min_samples=3)
        # "b" has 100% success, "a" has 66%
        general = [p for p in patterns if p.context_key == "test_failure|*"]
        assert len(general) > 0
        assert general[0].best_strategy == "b"

    def test_get_strategy_for_context_specific(self):
        extractor = PatternExtractor()
        patterns = [
            StrategyPattern(context_key="test_failure|mod.agent", best_strategy="update_callers", success_rate=0.9, sample_count=5),
            StrategyPattern(context_key="test_failure|*", best_strategy="refactor", success_rate=0.7, sample_count=10),
        ]
        result = extractor.get_strategy_for_context(patterns, "test_failure", "mod.agent")
        assert result == "update_callers"  # specific match

    def test_get_strategy_for_context_general(self):
        extractor = PatternExtractor()
        patterns = [
            StrategyPattern(context_key="test_failure|*", best_strategy="refactor", success_rate=0.7, sample_count=10),
        ]
        result = extractor.get_strategy_for_context(patterns, "test_failure", "other_module")
        assert result == "refactor"  # general match

    def test_no_match(self):
        extractor = PatternExtractor()
        result = extractor.get_strategy_for_context([], "test_failure", "mod")
        assert result is None
