"""Unit tests for context_utils.py — build_context_key + adaptive_distill_threshold."""

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.context_utils import (
    adaptive_distill_threshold,
    build_context_key,
)


class TestBuildContextKey:
    """Tests for build_context_key().

    Must be called identically in operation_executor and strategy_selector.
    """

    def test_failure_class_with_mode(self):
        key = build_context_key(failure_class="syntax", mode="repair")
        assert key == "syntax:repair"

    def test_failure_class_without_mode(self):
        key = build_context_key(failure_class="import_error")
        assert key == "import_error"

    def test_empty_failure_with_mode(self):
        key = build_context_key(failure_class="", mode="plan")
        assert key == "plan"

    def test_both_empty(self):
        key = build_context_key(failure_class="", mode="")
        assert key == "unknown"

    def test_mode_with_special_chars(self):
        key = build_context_key(failure_class="", mode="read-file")
        assert key == "read-file"

    def test_failure_class_with_empty_mode(self):
        key = build_context_key(failure_class="timeout", mode="")
        assert key == "timeout"

    def test_known_failure_classes_roundtrip(self):
        """This pattern is used identically in executor/selector — verify roundtrip."""
        cases = [
            ("syntax", "repair"),
            ("import", "resolve"),
            ("semantic", "replan"),
            ("permission", ""),
            ("", "fallback"),
            ("", ""),
        ]
        for fc, mode in cases:
            k = build_context_key(failure_class=fc, mode=mode)
            if fc:
                assert fc in k
            if mode and not fc:
                assert k == mode
            elif not fc and not mode:
                assert k == "unknown"


class TestAdaptiveDistillThreshold:
    """Tests for adaptive_distill_threshold().

    Thresholds from config:
      DISTILL_SPARSE_LIMIT, DISTILL_MODERATE_LIMIT
      DISTILL_THRESHOLD_SPARSE, DISTILL_THRESHOLD_MODERATE, DISTILL_THRESHOLD_CONFIDENT
    """

    def test_below_sparse_limit(self):
        th = adaptive_distill_threshold(0)
        assert th == _cfg.scores.DISTILL_THRESHOLD_SPARSE

    def test_sparse_edge_at_limit(self):
        # At DISTILL_SPARSE_LIMIT - 1 → still sparse
        limit = _cfg.scores.DISTILL_SPARSE_LIMIT
        th = adaptive_distill_threshold(limit - 1)
        assert th == _cfg.scores.DISTILL_THRESHOLD_SPARSE

    def test_moderate_range_low(self):
        limit = _cfg.scores.DISTILL_SPARSE_LIMIT
        th = adaptive_distill_threshold(limit)
        assert th == _cfg.scores.DISTILL_THRESHOLD_MODERATE

    def test_moderate_range_mid(self):
        limit = _cfg.scores.DISTILL_SPARSE_LIMIT
        moderate_limit = _cfg.scores.DISTILL_MODERATE_LIMIT
        mid = (limit + moderate_limit) // 2
        th = adaptive_distill_threshold(mid)
        assert th == _cfg.scores.DISTILL_THRESHOLD_MODERATE

    def test_moderate_edge_at_limit(self):
        moderate_limit = _cfg.scores.DISTILL_MODERATE_LIMIT
        th = adaptive_distill_threshold(moderate_limit - 1)
        assert th == _cfg.scores.DISTILL_THRESHOLD_MODERATE

    def test_confident_at_limit(self):
        moderate_limit = _cfg.scores.DISTILL_MODERATE_LIMIT
        th = adaptive_distill_threshold(moderate_limit)
        assert th == _cfg.scores.DISTILL_THRESHOLD_CONFIDENT

    def test_confident_high(self):
        th = adaptive_distill_threshold(10_000)
        assert th == _cfg.scores.DISTILL_THRESHOLD_CONFIDENT

    def test_monotonic_decreasing(self):
        """Threshold should decrease (or stay same) as sample_count increases."""
        sparse = adaptive_distill_threshold(0)
        moderate = adaptive_distill_threshold(_cfg.scores.DISTILL_SPARSE_LIMIT)
        confident = adaptive_distill_threshold(_cfg.scores.DISTILL_MODERATE_LIMIT)
        # Sparse > Moderate > Confident (high threshold when scarce)
        assert sparse >= moderate >= confident

    def test_negative_count_returns_sparse(self):
        th = adaptive_distill_threshold(-1)
        assert th == _cfg.scores.DISTILL_THRESHOLD_SPARSE
