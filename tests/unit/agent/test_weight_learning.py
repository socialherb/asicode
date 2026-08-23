"""Tests for weight_learning — adaptive weight learning system."""

import threading

import pytest

from external_llm.agent.weight_learning import (
    # Constants
    _ALL_BUCKETS,
    _AXES,
    _BUCKET_PROFILE_MAP,
    _BURDEN_RANK,
    _DELTA_MEDIUM,
    _DELTA_SMALL,
    _MIN_SIGNALS_FOR_LEARNING,
    _STATIC_BASE,
    _W_MAX,
    _W_MIN,
    BUCKET_DEFAULT,
    BUCKET_GRAPH_HEAVY,
    BUCKET_STRICT_REFERENCE,
    # Classes
    LearningSignal,
    WeightBucketState,
    WeightLearner,
    # Functions
    _apply_weight_delta,
    _blend_weights,
    _clamp_weights,
    _compute_weight_delta,
    _normalize_weights,
    infer_budget_failure,
    infer_repair_attempts,
    repair_burden_label,
    resolve_bucket,
    update_weights_from_monitor_result,
)


# Helper: create LearningSignal with all required fields
def _learn_sig(**overrides) -> LearningSignal:
    params = {
        "bucket": BUCKET_DEFAULT,
        "selected_weight_profile": "DEFAULT",
        "selected_strategy": "generic_create",
        "success": True,
        "repair_attempts": 0,
        "repair_burden": "none",
        "contract_violation": False,
        "semantic_failures": [],
        "budget_failure": False,
        "graph_impact_level": "low",
    }
    params.update(overrides)
    return LearningSignal(**params)


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_axes_defined(self):
        assert len(_AXES) == 5
        assert "success" in _AXES

    def test_buckets_defined(self):
        assert BUCKET_DEFAULT in _ALL_BUCKETS
        assert BUCKET_GRAPH_HEAVY in _ALL_BUCKETS
        assert BUCKET_STRICT_REFERENCE in _ALL_BUCKETS

    def test_burden_rank(self):
        assert _BURDEN_RANK["none"] == 0
        assert _BURDEN_RANK["high"] == 3

    def test_static_base_has_all_profiles(self):
        for bucket in _ALL_BUCKETS:
            profile = _BUCKET_PROFILE_MAP.get(bucket)
            assert profile is not None
            assert profile in _STATIC_BASE


# ============================================================================
# resolve_bucket
# ============================================================================


class TestResolveBucket:
    def test_default(self):
        assert resolve_bucket() == BUCKET_DEFAULT

    def test_strict_reference(self):
        assert resolve_bucket(has_strict_reference=True) == BUCKET_STRICT_REFERENCE

    def test_reference_bound_context(self):
        assert resolve_bucket(reference_bound_context=True) == BUCKET_STRICT_REFERENCE

    def test_graph_heavy(self):
        assert resolve_bucket(graph_impact_level="high") == BUCKET_GRAPH_HEAVY

    def test_priority_reference_over_graph(self):
        assert resolve_bucket(has_strict_reference=True, graph_impact_level="high") == BUCKET_STRICT_REFERENCE

    def test_priority_graph_over_default(self):
        assert resolve_bucket(graph_impact_level="high") == BUCKET_GRAPH_HEAVY

    def test_low_graph_default(self):
        assert resolve_bucket(graph_impact_level="low") == BUCKET_DEFAULT

    def test_medium_graph_default(self):
        assert resolve_bucket(graph_impact_level="medium") == BUCKET_DEFAULT


# ============================================================================
# _normalize_weights
# ============================================================================


class TestNormalizeWeights:
    def test_normalize_sum_to_one(self):
        w = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        result = _normalize_weights(w)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_normalize_empty_provides_equal(self):
        """When total <= 0, return equal weights."""
        result = _normalize_weights({"success": 0.0, "repair": 0.0, "contract": 0.0, "complexity": 0.0, "cost": 0.0})
        expected = 1.0 / 5
        for k in _AXES:
            assert result[k] == pytest.approx(expected, abs=1e-6)

    def test_normalize_negative_total(self):
        """Negative total also triggers equal-weight fallback."""
        result = _normalize_weights({"success": -1.0, "repair": 0.0, "contract": 0.0, "complexity": 0.0, "cost": 0.0})
        expected = 1.0 / 5
        for k in _AXES:
            assert result[k] == pytest.approx(expected, abs=1e-6)

    def test_normalize_rounds_to_6_decimals(self):
        w = {"success": 0.3, "repair": 0.2, "contract": 0.3, "complexity": 0.1, "cost": 0.1}
        result = _normalize_weights(w)
        for v in result.values():
            assert len(str(v).split(".")[1]) <= 6

    def test_normalize_preserves_only_axes(self):
        w = {"success": 1.0, "repair": 1.0, "contract": 1.0, "complexity": 1.0, "cost": 1.0, "extra": 999}
        result = _normalize_weights(w)
        for k in _AXES:
            assert k in result
        # normalize only outputs axis keys; "extra" is stripped after normalization
        # because _normalize_weights creates a new dict from _AXES keys


# ============================================================================
# _clamp_weights
# ============================================================================


class TestClampWeights:
    def test_clamp_low_values(self):
        result = _clamp_weights({"success": 0.0, "repair": 0.01, "contract": 0.02, "complexity": 0.03, "cost": 0.04})
        for v in result.values():
            assert v >= _W_MIN

    def test_clamp_high_values(self):
        result = _clamp_weights({"success": 1.0, "repair": 0.8, "contract": 0.6, "complexity": 0.7, "cost": 0.9})
        for v in result.values():
            assert v <= _W_MAX

    def test_clamp_within_range(self):
        w = {"success": 0.15, "repair": 0.20, "contract": 0.25, "complexity": 0.10, "cost": 0.30}
        result = _clamp_weights(w)
        assert result == w

    def test_clamp_returns_all_axes(self):
        result = _clamp_weights({"success": 0.1, "repair": 0.1, "contract": 0.1, "complexity": 0.1, "cost": 0.6})
        for k in _AXES:
            assert k in result


# ============================================================================
# _blend_weights
# ============================================================================


class TestBlendWeights:
    def test_blend_half_half(self):
        learned = {"success": 0.4, "repair": 0.3, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        static = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 0.5, 0.5)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_blend_pure_learned(self):
        learned = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        static = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 1.0, 0.0)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_blend_pure_static(self):
        learned = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        static = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 0.0, 1.0)
        assert result == _normalize_weights(static)

    def test_blend_normalized(self):
        learned = {"success": 10.0, "repair": 0.0, "contract": 0.0, "complexity": 0.0, "cost": 0.0}
        static = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 1.0, 1.0)
        assert abs(sum(result.values()) - 1.0) < 1e-6


# ============================================================================
# _compute_weight_delta
# ============================================================================


class TestComputeWeightDelta:
    """Test all 4 rule groups (A, B, C, D) in _compute_weight_delta."""

    def make_signal(self, **overrides) -> LearningSignal:
        return _learn_sig(**overrides)

    def make_weights(self, **overrides) -> dict:
        w = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        w.update(overrides)
        return w

    def test_rule_a_contract_violation(self):
        signal = self.make_signal(contract_violation=True)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["contract"] >= _DELTA_MEDIUM
        assert delta["repair"] >= _DELTA_SMALL
        assert delta["success"] <= -_DELTA_SMALL

    def test_rule_a_structured_flag_with_semantic_failures(self):
        """Rule A fires on the structured flag; semantic_failures ride along."""
        signal = self.make_signal(semantic_failures=["used POST instead of GET"], contract_violation=True)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["contract"] >= _DELTA_MEDIUM
        assert delta["repair"] >= _DELTA_SMALL
        assert delta["success"] <= -_DELTA_SMALL

    def test_rule_a_free_text_semantic_failure_does_not_fire(self):
        """Free-text 'contract' in semantic_failures must NOT fire rule A —
        the structured contract_violation flag is the SSOT (no keyword sniffing)."""
        signal = self.make_signal(semantic_failures=["Contract Breach"], contract_violation=False, success=False)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["contract"] == 0
        assert delta["repair"] == 0
        assert delta["success"] == 0

    def test_rule_b_high_burden(self):
        signal = self.make_signal(repair_burden="high")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["repair"] >= _DELTA_MEDIUM

    def test_rule_b_medium_burden(self):
        signal = self.make_signal(repair_burden="medium")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["repair"] >= _DELTA_SMALL

    def test_rule_b_low_burden_no_delta(self):
        signal = self.make_signal(repair_burden="low")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["repair"] == 0.0

    def test_rule_b_graph_heavy_high_burden(self):
        """Graph-heavy bucket with high burden also lifts complexity."""
        signal = self.make_signal(bucket=BUCKET_GRAPH_HEAVY, repair_burden="high")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["repair"] >= _DELTA_MEDIUM
        assert delta["complexity"] >= _DELTA_SMALL

    def test_rule_b_graph_heavy_medium_burden(self):
        signal = self.make_signal(bucket=BUCKET_GRAPH_HEAVY, repair_burden="medium")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["repair"] >= _DELTA_SMALL
        assert delta["complexity"] >= _DELTA_SMALL

    def test_rule_c_graph_heavy_failure(self):
        signal = self.make_signal(success=False, graph_impact_level="high")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["complexity"] >= _DELTA_MEDIUM
        assert delta["cost"] >= _DELTA_SMALL
        assert delta["success"] <= -_DELTA_SMALL

    def test_rule_d_clean_success(self):
        signal = self.make_signal(success=True, repair_attempts=0, contract_violation=False)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["success"] >= _DELTA_SMALL

    def test_rule_d_relax_contract(self):
        """Clean success with contract above baseline → relax contract."""
        self.make_weights(contract=0.30)  # > 0.20 + 0.05
        signal = self.make_signal(success=True, repair_attempts=0, contract_violation=False)
        delta = _compute_weight_delta(signal, self.make_weights(contract=0.30))
        assert delta["contract"] < 0

    def test_rule_d_relax_repair(self):
        """Clean success with repair above baseline → relax repair."""
        self.make_weights(repair=0.40)  # > 0.30 + 0.05
        signal = self.make_signal(success=True, repair_attempts=0, contract_violation=False)
        delta = _compute_weight_delta(signal, self.make_weights(repair=0.40))
        assert delta["repair"] < 0

    def test_rule_d_no_relax_when_below_baseline(self):
        """Clean success but contract at baseline → no negative delta."""
        signal = self.make_signal(success=True, repair_attempts=0, contract_violation=False)
        delta = _compute_weight_delta(signal, self.make_weights(contract=0.20, repair=0.30))
        # contract=0.20 is exactly baseline, not > baseline + threshold
        # So no relaxation
        assert delta.get("contract", 0) >= 0 or delta.get("repair", 0) >= 0

    def test_clean_success_with_repair_no_relax(self):
        """Clean success with repair attempts → no relaxation (rule D needs repair_attempts==0)."""
        signal = self.make_signal(success=True, repair_attempts=1, contract_violation=False)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["success"] == 0.0  # Not clean success

    def test_clean_success_with_contract_violation_no_relax(self):
        signal = self.make_signal(success=True, repair_attempts=0, contract_violation=True)
        delta = _compute_weight_delta(signal, self.make_weights())
        # Rule A fires for contract violation, rule D does not fire
        assert delta["success"] <= -_DELTA_SMALL  # from rule A
        assert delta["contract"] >= _DELTA_MEDIUM  # from rule A

    def test_all_rules_compose(self):
        """Multiple rules can fire on the same signal."""
        signal = self.make_signal(
            success=False,
            repair_burden="high",
            contract_violation=True,
            graph_impact_level="high",
        )
        delta = _compute_weight_delta(signal, self.make_weights())
        # Rule A: contract++, repair+, success--
        # Rule B: repair++
        # Rule C: complexity++, cost+, success--
        assert delta["contract"] > 0
        assert delta["repair"] > 0
        assert delta["complexity"] > 0
        assert delta["cost"] > 0
        assert delta["success"] < 0

    def test_zero_delta_default(self):
        """Default weight with no signals firing → all zero."""
        signal = self.make_signal(success=False, repair_burden="none")
        delta = _compute_weight_delta(signal, self.make_weights())
        assert all(abs(v) < 1e-9 for v in delta.values())


# ============================================================================
# _apply_weight_delta
# ============================================================================


class TestApplyWeightDelta:
    def test_apply_simple_delta(self):
        weights = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        delta = {"success": 0.01, "repair": 0.0, "contract": 0.02, "complexity": 0.0, "cost": 0.0}
        result = _apply_weight_delta(weights, delta)
        assert abs(sum(result.values()) - 1.0) < 1e-6
        for v in result.values():
            assert _W_MIN <= v <= _W_MAX + 1e-9

    def test_apply_large_delta(self):
        """Large delta that pushes beyond bounds is clamped."""
        weights = {"success": 0.50, "repair": 0.20, "contract": 0.15, "complexity": 0.10, "cost": 0.05}
        delta = {"success": 0.5, "repair": 0, "contract": 0, "complexity": 0, "cost": 0}
        result = _apply_weight_delta(weights, delta)
        for v in result.values():
            assert _W_MIN <= v <= _W_MAX + 1e-9
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_apply_negative_delta(self):
        weights = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        delta = {"success": -0.05, "repair": 0.0, "contract": 0.0, "complexity": 0.0, "cost": 0.0}
        result = _apply_weight_delta(weights, delta)
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert result["success"] >= _W_MIN

    def test_apply_multi_axis_residual_stays_on_simplex(self):
        """Residual from clamping MANY pinned axes must still sum to 1.0.

        Regression: the old re-balancer dumped the entire deficit onto ONE axis
        and broke on the first fit. With 3 axes pushed high and 2 pushed low
        (all clamping), no single axis had room for the whole residual, so the
        simplex invariant drifted (sum ≈ 1.386). The fix distributes the
        residual across every axis that still has room.
        """
        weights = dict.fromkeys(_AXES, 0.2)
        delta = {
            "success": 0.4,
            "repair": 0.4,
            "contract": 0.4,
            "complexity": -0.4,
            "cost": -0.4,
        }
        result = _apply_weight_delta(weights, delta)
        assert abs(sum(result.values()) - 1.0) < 1e-6
        for v in result.values():
            assert _W_MIN - 1e-9 <= v <= _W_MAX + 1e-9

    def test_apply_preserves_all_axes(self):
        weights = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        delta = {"success": 0.01, "repair": 0.0, "contract": 0.02, "complexity": 0.0, "cost": 0.0}
        result = _apply_weight_delta(weights, delta)
        for k in _AXES:
            assert k in result


# ============================================================================
# Data classes — serialization
# ============================================================================


class TestLearningSignal:
    def test_to_dict(self):
        sig = _learn_sig()
        d = sig.to_dict()
        assert d["bucket"] == BUCKET_DEFAULT
        assert d["success"] is True
        assert "semantic_failures" in d

    def test_to_dict_semantic_failures_copy(self):
        failures = ["contract issue"]
        sig = _learn_sig(semantic_failures=failures)
        d = sig.to_dict()
        assert d["semantic_failures"] == ["contract issue"]
        failures.append("extra")  # modify original
        assert d["semantic_failures"] == ["contract issue"]  # not affected


class TestWeightBucketState:
    def test_to_dict(self):
        state = WeightBucketState(
            weights={"success": 0.3, "repair": 0.2, "contract": 0.3, "complexity": 0.1, "cost": 0.1},
            signal_count=5,
            last_updated=123.0,
        )
        d = state.to_dict()
        assert d["signal_count"] == 5
        assert d["weights"]["success"] == 0.3

    def test_default_initialization(self):
        state = WeightBucketState(
            weights={"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        )
        assert state.signal_count == 0
        assert state.last_updated == 0.0


# ============================================================================
# WeightLearner
# ============================================================================


class TestWeightLearner:
    def test_initial_state(self):
        learner = WeightLearner()
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 0
        assert abs(sum(state.weights.values()) - 1.0) < 1e-6

    def test_initial_state_all_buckets(self):
        learner = WeightLearner()
        for bucket in _ALL_BUCKETS:
            state = learner.get_bucket_state(bucket)
            assert state.signal_count == 0
            assert len(state.weights) == len(_AXES)

    def test_get_bucket_state_fallback(self):
        """Unknown bucket falls back to default."""
        learner = WeightLearner()
        state = learner.get_bucket_state("nonexistent")
        assert state.signal_count == 0

    def test_get_learned_weights(self):
        learner = WeightLearner()
        w = learner.get_learned_weights(BUCKET_DEFAULT)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        assert len(w) == len(_AXES)

    def test_get_effective_weights_below_threshold(self):
        """Signal count < MIN → pure static."""
        learner = WeightLearner()
        static = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        _eff, source, n = learner.get_effective_weights(BUCKET_DEFAULT, static)
        assert source == "static"
        assert n == 0

    def test_get_effective_weights_above_threshold(self):
        """Signal count >= MIN → blended."""
        learner = WeightLearner()
        # Manually increase signal count
        bucket = BUCKET_DEFAULT
        learner._states[bucket].signal_count = _MIN_SIGNALS_FOR_LEARNING
        static = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        _eff, source, n = learner.get_effective_weights(BUCKET_DEFAULT, static)
        assert source == "blended"
        assert n >= _MIN_SIGNALS_FOR_LEARNING

    def test_get_effective_weights_static_below_min(self):
        learner = WeightLearner()
        learner._states[BUCKET_DEFAULT].signal_count = _MIN_SIGNALS_FOR_LEARNING - 1
        static = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        _eff, source, _n = learner.get_effective_weights(BUCKET_DEFAULT, static)
        assert source == "static"

    def test_update_increases_signal_count(self):
        learner = WeightLearner()
        sig = _learn_sig(
            success=True,
            repair_attempts=1,
            repair_burden="medium",
        )
        learner.update(sig)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 1

    def test_update_unknown_bucket_skipped(self):
        learner = WeightLearner()
        sig = _learn_sig(
            bucket="nonexistent",
            success=True,
        )
        # Should not raise error
        learner.update(sig)
        # Default bucket should be unaffected
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 0

    def test_update_zero_delta_skipped(self):
        """Signal that produces no delta → no update."""
        learner = WeightLearner()
        # A signal that fires no rules
        sig = _learn_sig(
            success=False,
            repair_burden="none",
        )
        learner.update(sig)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        # For this signal, no rules fire → all deltas are 0 → signal_count not incremented
        assert state.signal_count == 0

    def test_multiple_updates(self):
        learner = WeightLearner()
        sig = _learn_sig(
            success=True,
            repair_attempts=1,
            repair_burden="medium",
        )
        for _ in range(3):
            learner.update(sig)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 3

    def test_get_summary(self):
        learner = WeightLearner()
        summary = learner.get_summary()
        for bucket in _ALL_BUCKETS:
            assert bucket in summary
            assert "weights" in summary[bucket]
            assert "signal_count" in summary[bucket]
            assert "base_profile" in summary[bucket]

    def test_load_state(self):
        learner = WeightLearner()
        summary = learner.get_summary()
        # Modify and reload
        summary[BUCKET_DEFAULT]["signal_count"] = 42
        learner.load_state(summary)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 42

    def test_load_state_ignores_unknown_bucket(self):
        learner = WeightLearner()
        learner.load_state({"unknown_bucket": {"weights": {"success": 0.5}, "signal_count": 5}})
        # Should not crash — unknown bucket is silently skipped

    def test_load_state_handles_bad_weights(self):
        learner = WeightLearner()
        learner.load_state({BUCKET_DEFAULT: {"weights": "not_a_dict", "signal_count": 5}})
        # Bad weights → skipped, keep current state
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 0

    def test_get_summary_returns_copy(self):
        learner = WeightLearner()
        summary = learner.get_summary()
        summary[BUCKET_DEFAULT]["signal_count"] = 999
        original = learner.get_bucket_state(BUCKET_DEFAULT)
        assert original.signal_count != 999


# ============================================================================
# WeightLearner advanced tests
# ============================================================================


class TestWeightLearnerAdvanced:
    def test_update_changes_weights(self):
        """Signal with contract_violation increases contract weight."""
        learner = WeightLearner()
        initial = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        sig = _learn_sig(contract_violation=True, success=True, repair_attempts=1)
        learner.update(sig)
        updated = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        assert updated != initial  # contract_violation signal increases contract weight

    def test_get_effective_weights_higher_tier(self):
        """Signal count >= 10 → 60/40 blend."""
        learner = WeightLearner()
        learner._states[BUCKET_DEFAULT].signal_count = 10
        static = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        _eff, source, _n = learner.get_effective_weights(BUCKET_DEFAULT, static)
        assert source == "blended"

    def test_load_state_preserves_invariants(self):
        """Loaded weights remain normalised (sum=1.0)."""
        learner = WeightLearner()
        bad_weights = {"success": 2.0, "repair": 0.0, "contract": 3.0, "complexity": -1.0, "cost": 0.0}
        learner.load_state({BUCKET_DEFAULT: {"weights": bad_weights, "signal_count": 5}})
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert abs(sum(state.weights.values()) - 1.0) < 1e-6

    def test_load_state_partial_weights(self):
        """Partial weights dictionary fills missing keys from current state."""
        learner = WeightLearner()
        partial = {"success": 0.5}
        learner.load_state({BUCKET_DEFAULT: {"weights": partial, "signal_count": 3}})
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert abs(sum(state.weights.values()) - 1.0) < 1e-6

    def test_update_separate_buckets_independent(self):
        """Updating one bucket does not affect another."""
        learner = WeightLearner()
        learner.get_learned_weights(BUCKET_DEFAULT)["success"]
        sig = _learn_sig(bucket=BUCKET_GRAPH_HEAVY, success=True, repair_attempts=1, repair_burden="medium")
        learner.update(sig)
        # Re-check default bucket weights (should not change since signal was for graph_heavy)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 0  # No update to default bucket

    def test_get_effective_weights_unknown_bucket(self):
        learner = WeightLearner()
        static = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        eff, source, _n = learner.get_effective_weights("unknown", static)
        assert source == "static"
        assert len(eff) == len(_AXES)


# ============================================================================
# Thread safety (moved from test_execution_learning.py when the 2-layer
# ExecutionLearner/StrategyPolicyLearner subsystem was removed)
# ============================================================================


class TestWeightLearnerThreadSafety:
    """WeightLearner.update must be safe under concurrent access."""

    def test_weight_learner_concurrent_updates(self):
        learner = WeightLearner()
        errors = []

        def worker(n: int):
            try:
                for _ in range(n):
                    sig = LearningSignal(
                        bucket="default",
                        selected_weight_profile="DEFAULT",
                        selected_strategy="generic_create",
                        success=False,
                        repair_attempts=2,
                        repair_burden="medium",
                        contract_violation=True,
                        semantic_failures=["contract drift"],
                        budget_failure=False,
                        graph_impact_level="low",
                    )
                    learner.update(sig)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(30,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = learner.get_bucket_state("default")
        assert state.signal_count == 120


# ============================================================================
# update_weights_from_monitor_result (monitor → learner bridge)
# ============================================================================


class TestUpdateFromMonitorResult:
    """Rule A contract signals must flow from the monitor's STRUCTURED
    contract_violation flag — never from free-text keyword sniffing.  Budget /
    repair signals are forwarded or derived structurally (never by sniffing)."""

    def test_structured_contract_violation_fires_rule_a(self):
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        result = {
            "status": "failed",
            "evaluated_status": "failed",
            "semantic_failures": ["used POST instead of GET", "missing endpoint: /api/x"],
            "contract_violation": True,  # emitted by self_impl_monitor save_result
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        assert after > before
        assert learner.get_bucket_state(BUCKET_DEFAULT).signal_count == 1

    def test_legacy_dict_without_flag_does_not_fire_rule_a(self):
        """A pre-structured result dict (no contract_violation key) must NOT
        trigger rule A from semantic_failures text — the contract axis stays
        put.  (Rule B may fire from the derived repair burden: that is the
        documented estimation path, not keyword sniffing.)"""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)
        result = {
            "status": "failed",
            "evaluated_status": "failed",
            "semantic_failures": ["used POST instead of GET"],
        }
        assert update_weights_from_monitor_result(learner, result) is True
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        after = state.weights
        # Rule A stays silent: contract axis must NOT rise (the small decrease
        # is simplex normalization from rule B firing on the repair axis).
        assert after["contract"] < before["contract"]
        assert after["repair"] > before["repair"]
        assert state.signal_count == 1

    def test_metadata_nested_flag_is_honoured(self):
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        result = {
            "status": "failed",
            "metadata": {"contract_violation": True, "semantic_failures": ["contract drift"]},
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["contract"]
        assert after > before

    def test_budget_failure_structured_flag_fires_rule_b(self):
        """budget_failure=True (monitor save_result) must reach the signal and
        fire rule B via the derived high repair burden."""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        result = {
            "status": "failed",
            "evaluated_status": "failed",
            "budget_failure": True,  # emitted by self_impl_monitor save_result
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        assert after > before
        assert learner.get_bucket_state(BUCKET_DEFAULT).signal_count == 1

    def test_bottleneck_kind_fallback_derives_budget_failure(self):
        """Pre-structured JSON without budget_failure: budget-class bottleneck
        kinds derive it (legacy fallback, not text sniffing)."""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        result = {
            "status": "failed",
            "bottlenecks": [{"kind": "max_turns", "severity": "critical"}],
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        assert after > before

    def test_metadata_repair_keys_win_over_derivation(self):
        """Explicit metadata repair keys must win over the derivation (without
        them a failed result derives 'high' and fires rule B)."""
        learner = WeightLearner()
        result = {
            "status": "failed",
            "metadata": {"repair_burden": "low", "repair_attempts": 1},
        }
        assert update_weights_from_monitor_result(learner, result) is True
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        # low (rank 1) → rule B silent; nothing else fires → zero-delta skip.
        assert state.signal_count == 0

    def test_repair_hint_blocks_clean_success_rule_d(self):
        """success + repair_attempts>0 must NOT fire rule D (clean success):
        the gate reads the forwarded repair hint, not the 0 default.  Rule D
        would ADD to the success axis; the observed small decrease is pure
        simplex normalization from rule B (medium burden)."""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["success"]
        result = {
            "status": "success",
            "metadata": {"repair_attempts": 2, "repair_burden": "medium"},
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["success"]
        assert after < before

    def test_monitor_direct_repair_counter_flows_to_signal(self):
        """save_result now emits a top-level repair_attempts counter (counted
        structurally from failed verification calls).  It must reach the
        signal: rule B fires on the derived medium burden and rule D (clean
        success) is blocked even though the status is success."""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        result = {
            "status": "success",
            "evaluated_status": "success",
            "repair_attempts": 2,  # emitted by self_impl_monitor save_result
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["repair"]
        assert after > before
        assert learner.get_bucket_state(BUCKET_DEFAULT).signal_count == 1

    def test_monitor_direct_repair_counter_zero_keeps_clean_success(self):
        """A clean run (counter == 0, no contract violation) must still fire
        rule D's clean-success bonus — the real counter replaces the old
        estimation for new result JSON."""
        learner = WeightLearner()
        before = learner.get_learned_weights(BUCKET_DEFAULT)["success"]
        result = {
            "status": "success",
            "evaluated_status": "success",
            "repair_attempts": 0,  # emitted by self_impl_monitor save_result
        }
        assert update_weights_from_monitor_result(learner, result) is True
        after = learner.get_learned_weights(BUCKET_DEFAULT)["success"]
        assert after > before
        assert learner.get_bucket_state(BUCKET_DEFAULT).signal_count == 1


class TestMonitorSignalInference:
    """Shared monitor-result inference helpers — SSOT used by both the loop
    (summarize_result_row) and the consumer (update_weights_from_monitor_result).
    Presence-wins: structured bools beat legacy derivation."""

    def test_infer_budget_failure_structured_bool_wins(self):
        result = {"budget_failure": False, "bottlenecks": [{"kind": "max_turns"}]}
        assert infer_budget_failure(result) is False

    def test_infer_budget_failure_bottleneck_fallback(self):
        for kind in ("max_turns", "cost_limit", "auto_cancel"):
            assert infer_budget_failure({"bottlenecks": [{"kind": kind}]}) is True

    def test_infer_budget_failure_clean(self):
        assert infer_budget_failure({"bottlenecks": [{"kind": "replan"}]}) is False
        assert infer_budget_failure({}) is False
        assert infer_budget_failure({"bottlenecks": [{"kind": "max_turns"}]}) is True

    def test_infer_repair_attempts_hint_and_heuristics(self):
        assert infer_repair_attempts({"repair_attempts": 3}) == 3
        assert infer_repair_attempts({"semantic_failures": ["x"]}) == 2
        assert infer_repair_attempts({"evaluated_status": "partial_success"}) == 1
        assert infer_repair_attempts({"status": "success"}) == 0

    def test_repair_burden_label_matrix(self):
        assert repair_burden_label(0, False, True) == "none"
        assert repair_burden_label(1, False, True) == "low"
        assert repair_burden_label(2, False, True) == "medium"
        assert repair_burden_label(0, True, True) == "high"
        assert repair_burden_label(0, False, False) == "high"
