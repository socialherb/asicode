"""Tests for execution_thresholds.py — update/reset + ThresholdAdapter."""
from __future__ import annotations

import pytest

from external_llm.agent.execution_thresholds import (
    THRESHOLD_ADAPTER,
    THRESHOLDS,
    ExecutionThresholds,
    ThresholdAdapter,
    reset_thresholds,
    update_thresholds,
)


@pytest.fixture(autouse=True)
def _restore_defaults():
    """Ensure every test starts and ends with factory defaults."""
    reset_thresholds()
    THRESHOLD_ADAPTER.reset()
    yield
    reset_thresholds()
    THRESHOLD_ADAPTER.reset()


# ── update_thresholds / reset_thresholds ────────────────────────────────────

class TestUpdateThresholds:

    def test_update_single_field(self):
        defaults = ExecutionThresholds()
        update_thresholds(lineage_hard_reject_ratio=0.55)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == 0.55
        assert T.semantic_pass_score == defaults.semantic_pass_score  # unchanged

    def test_update_multiple_fields(self):
        update_thresholds(
            semantic_pass_score=0.50,
            surgical_similarity_threshold=0.90,
        )
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.semantic_pass_score == 0.50
        assert T.surgical_similarity_threshold == 0.90

    def test_invalid_field_ignored(self):
        """Unknown fields should be silently ignored."""
        update_thresholds(nonexistent_field=999, lineage_hard_reject_ratio=0.60)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == 0.60
        assert not hasattr(T, "nonexistent_field")

    def test_empty_update_noop(self):
        """Empty overrides should not change any values."""
        defaults = ExecutionThresholds()
        update_thresholds()
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == defaults.lineage_hard_reject_ratio
        assert T.semantic_pass_score == defaults.semantic_pass_score

    def test_reset_restores_defaults(self):
        update_thresholds(lineage_hard_reject_ratio=0.10)
        reset_thresholds()
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == ExecutionThresholds().lineage_hard_reject_ratio

    def test_inprocess_mutation_visible_via_cached_import(self):
        """Modules that imported THRESHOLDS before update_thresholds() was called
        should see the updated value immediately — same object, in-place mutation.

        This is the key correctness guarantee of the mutable singleton design:
        ``from X import THRESHOLDS`` binds a reference, not a snapshot.
        """
        # Simulate a module that captured THRESHOLDS at import time
        cached_ref = THRESHOLDS

        update_thresholds(semantic_pass_score=0.99)

        # The cached reference and the re-imported name both see the change
        assert cached_ref.semantic_pass_score == 0.99
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.semantic_pass_score == 0.99
        # And they are the same object
        assert cached_ref is T

    def test_reset_preserves_singleton_identity(self):
        """reset_thresholds() must mutate the singleton in-place, not replace it."""
        ref_before = THRESHOLDS
        reset_thresholds()
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T is ref_before


# ── ThresholdAdapter ────────────────────────────────────────────────────────

class TestThresholdAdapter:

    def test_no_adaptation_before_min_observations(self):
        adapter = ThresholdAdapter()
        original = THRESHOLDS.lineage_hard_reject_ratio
        # One observation short of the batch threshold
        for _ in range(THRESHOLDS.adapter_min_observations - 1):
            adapter.observe("lineage_hard_reject_ratio", 0.65, was_correct=False)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == original

    def test_adaptation_after_enough_observations(self):
        adapter = ThresholdAdapter()
        original = THRESHOLDS.lineage_hard_reject_ratio  # 0.70
        # Interleave correct/incorrect so both are in the same batch.
        # 5 correct at 0.75 + 5 incorrect at 0.50 → 50% correct < 80% gate → adapts
        for _ in range(5):
            adapter.observe("lineage_hard_reject_ratio", 0.75, was_correct=True)
            adapter.observe("lineage_hard_reject_ratio", 0.50, was_correct=False)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        # avg_correct=0.75, avg_incorrect=0.50 → suggested=0.625
        # EMA: 0.70 * 0.9 + 0.625 * 0.1 = 0.6925
        assert T.lineage_hard_reject_ratio != original
        assert abs(T.lineage_hard_reject_ratio - 0.6925) < 0.01

    def test_no_adaptation_when_mostly_correct(self):
        adapter = ThresholdAdapter()
        original = THRESHOLDS.lineage_hard_reject_ratio
        # 8 correct + 2 incorrect = 80% correct → at/above 80% gate → no adapt
        for _ in range(8):
            adapter.observe("lineage_hard_reject_ratio", 0.75, was_correct=True)
        for _ in range(2):
            adapter.observe("lineage_hard_reject_ratio", 0.50, was_correct=False)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        assert T.lineage_hard_reject_ratio == original

    def test_mixed_observations_adapts_toward_boundary(self):
        adapter = ThresholdAdapter()
        # 5 correct at 0.60 + 5 incorrect at 0.30 → 50% correct < 80% → adapts
        for _ in range(5):
            adapter.observe("semantic_pass_score", 0.60, was_correct=True)
            adapter.observe("semantic_pass_score", 0.30, was_correct=False)
        from external_llm.agent.execution_thresholds import THRESHOLDS as T
        # avg_correct=0.60, avg_incorrect=0.30 → suggested=0.45
        # EMA: 0.40 * 0.9 + 0.45 * 0.1 = 0.405
        assert abs(T.semantic_pass_score - 0.405) < 0.01

    def test_unknown_field_ignored(self):
        adapter = ThresholdAdapter()
        # Should not raise
        for _ in range(25):
            adapter.observe("nonexistent_threshold", 0.5, was_correct=False)
        assert "nonexistent_threshold" not in adapter.observation_counts

    def test_buffer_cleared_after_adaptation(self):
        """After a full batch triggers adaptation, the buffer resets to zero.
        This prevents unbounded growth and keeps each adaptation independent.
        """
        adapter = ThresholdAdapter()
        for _ in range(THRESHOLDS.adapter_min_observations):
            adapter.observe("lineage_hard_reject_ratio", 0.50, was_correct=False)
        assert adapter.observation_counts.get("lineage_hard_reject_ratio", 0) == 0

    def test_second_batch_adapts_from_updated_threshold(self):
        """Second batch's EMA starts from the value produced by the first batch."""
        adapter = ThresholdAdapter()
        # First batch: mixed → adapts
        for _ in range(5):
            adapter.observe("lineage_hard_reject_ratio", 0.75, was_correct=True)
            adapter.observe("lineage_hard_reject_ratio", 0.50, was_correct=False)
        after_first = THRESHOLDS.lineage_hard_reject_ratio
        assert after_first != 0.70  # adapted

        # Second batch: same pattern → EMA continues from after_first
        for _ in range(5):
            adapter.observe("lineage_hard_reject_ratio", 0.75, was_correct=True)
            adapter.observe("lineage_hard_reject_ratio", 0.50, was_correct=False)
        after_second = THRESHOLDS.lineage_hard_reject_ratio
        # EMA: after_first * 0.9 + 0.625 * 0.1
        expected = round(after_first * 0.9 + 0.625 * 0.1, 4)
        assert abs(after_second - expected) < 0.001

    def test_reset_clears_observations(self):
        adapter = ThresholdAdapter()
        for _ in range(5):
            adapter.observe("lineage_hard_reject_ratio", 0.5, was_correct=True)
        adapter.reset()
        assert adapter.observation_counts == {}

    def test_global_adapter_instance_exists(self):
        """THRESHOLD_ADAPTER should be importable and functional."""
        assert isinstance(THRESHOLD_ADAPTER, ThresholdAdapter)
        THRESHOLD_ADAPTER.observe("semantic_pass_score", 0.45, was_correct=True)
        assert THRESHOLD_ADAPTER.observation_counts.get("semantic_pass_score", 0) == 1

    def test_all_correct_no_adaptation(self):
        """Line 298: When all observations are correct, _adapt returns early (no incorrect_vals)."""
        adapter = ThresholdAdapter()
        original = THRESHOLDS.lineage_hard_reject_ratio
        # All observations correct → incorrect_vals is empty → early return
        for _ in range(THRESHOLDS.adapter_min_observations):
            adapter.observe("lineage_hard_reject_ratio", 0.85, was_correct=True)
        # Threshold unchanged because all observations were correct
        assert THRESHOLDS.lineage_hard_reject_ratio == original
        assert adapter.observation_counts.get("lineage_hard_reject_ratio", 0) == 0

    def test_adapt_unknown_field_returns_early(self):
        """Line 311: _adapt with name not on THRESHOLDS returns early at 'if current is None'."""
        adapter = ThresholdAdapter()
        # _adapt is private but we call it directly to exercise the guard
        # that handles getattr returning None for unknown threshold names
        adapter._adapt("does_not_exist_on_thresholds", [(0.5, False)])
        # Should not raise — just returns early at line 311
        assert True
