"""Tests for StrategyPrioritizer."""
from external_llm.editor.learning.pattern_extractor import StrategyPattern
from external_llm.editor.learning.problem_signature import ProblemSignature
from external_llm.editor.learning.strategy_prioritizer import PrioritizationResult, StrategyPrioritizer


class TestPrioritizationResult:
    def test_to_dict(self):
        r = PrioritizationResult(strategies=["a", "b"], confidence=0.8, top_recommended="a")
        d = r.to_dict()
        assert d["top_recommended"] == "a"


class TestStrategyPrioritizer:
    def test_no_patterns_unchanged(self):
        p = StrategyPrioritizer()
        result = p.prioritize(strategies=["a", "b", "c"])
        assert result.strategies == ["a", "b", "c"]
        assert result.matched_pattern_count == 0

    def test_pattern_boosts_strategy(self):
        p = StrategyPrioritizer()
        patterns = [
            StrategyPattern(context_key="test_failure|*", best_strategy="c", success_rate=0.9, sample_count=5),
        ]
        result = p.prioritize(
            strategies=["a", "b", "c"],
            preference_scores={"a": 0.5, "b": 0.4, "c": 0.3},
            patterns=patterns,
        )
        # "c" should be boosted
        assert result.adjustments.get("c", 0) > 0
        assert result.top_recommended == "c"

    def test_pattern_reorders(self):
        p = StrategyPrioritizer()
        patterns = [
            StrategyPattern(context_key="test_failure|*", best_strategy="c", success_rate=0.9, sample_count=5),
        ]
        result = p.prioritize(
            strategies=["a", "b", "c"],
            preference_scores={"a": 0.5, "b": 0.4, "c": 0.45},  # c is close to a
            patterns=patterns,
        )
        # c should now be above a (0.45 + 0.2 = 0.65 > 0.5)
        assert result.strategies[0] == "c"

    def test_with_experience_store(self, tmp_path):
        from external_llm.editor.learning.experience_store import ExperienceRecord, ExperienceStore

        store = ExperienceStore(store_path=str(tmp_path / "exp.json"))
        for _ in range(5):
            store.record(ExperienceRecord(
                strategy_used="update_callers",
                success=True,
                problem_signature={"failure_type": "test_failure", "module": "mod"},
            ))

        sig = ProblemSignature(failure_type="test_failure", module="mod")
        p = StrategyPrioritizer()
        result = p.prioritize(
            strategies=["symbol_edit", "update_callers", "refactor"],
            preference_scores={"symbol_edit": 0.5, "update_callers": 0.3, "refactor": 0.4},
            problem_signature=sig,
            experience_store=store,
        )
        assert result.matched_pattern_count > 0

    def test_low_confidence_no_boost(self):
        p = StrategyPrioritizer()
        patterns = [
            StrategyPattern(context_key="x|*", best_strategy="c", success_rate=0.2, sample_count=5),
        ]
        result = p.prioritize(
            strategies=["a", "c"],
            preference_scores={"a": 0.5, "c": 0.3},
            patterns=patterns,
        )
        # Low success rate → no boost
        assert result.adjustments.get("c", 0) == 0

    def test_penalty_for_bad_alternatives(self):
        p = StrategyPrioritizer()
        patterns = [
            StrategyPattern(
                context_key="x|*", best_strategy="c", success_rate=0.9, sample_count=5,
                alternatives=[("a", 0.1)],  # "a" has very low success
            ),
        ]
        result = p.prioritize(
            strategies=["a", "c"],
            preference_scores={"a": 0.5, "c": 0.3},
            patterns=patterns,
        )
        assert result.adjustments.get("a", 0) < 0
