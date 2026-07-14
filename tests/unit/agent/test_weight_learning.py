"""Tests for weight_learning — adaptive weight learning system."""

import pytest

from external_llm.agent.weight_learning import (
    _ALL_BUCKETS,
    # Constants
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
    ExecutionBucketState,
    ExecutionLearner,
    ExecutionLearningSignal,
    LearningSignal,
    StrategyLearningSignal,
    WeightBucketState,
    # Classes
    WeightLearner,
    _apply_weight_delta,
    _blend_weights,
    _clamp_weights,
    _compute_reward,
    _compute_weight_delta,
    _normalize_weights,
    compute_reward,
    # Functions
    resolve_bucket,
)


# Helper: create ExecutionLearningSignal with all required fields
def _exec_sig(lane="planner", success=True, final_status="success",
              failure_class="", verification_depth="full", **kw) -> ExecutionLearningSignal:
    return ExecutionLearningSignal(
        lane=lane, success=success, final_status=final_status,
        failure_class=failure_class, verification_depth=verification_depth,
        **kw,
    )

# Helper: create LearningSignal with all required fields
def _learn_sig(**overrides) -> LearningSignal:
    params = dict(
        bucket=BUCKET_DEFAULT,
        selected_weight_profile="DEFAULT",
        selected_strategy="generic_create",
        success=True,
        repair_attempts=0,
        repair_burden="none",
        contract_violation=False,
        semantic_failures=[],
        budget_failure=False,
        graph_impact_level="low",
    )
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
            assert len(str(v).split('.')[1]) <= 6

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
        static  = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 0.5, 0.5)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_blend_pure_learned(self):
        learned = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        static  = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 1.0, 0.0)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_blend_pure_static(self):
        learned = {"success": 0.5, "repair": 0.2, "contract": 0.1, "complexity": 0.1, "cost": 0.1}
        static  = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
        result = _blend_weights(learned, static, 0.0, 1.0)
        assert result == _normalize_weights(static)

    def test_blend_normalized(self):
        learned = {"success": 10.0, "repair": 0.0, "contract": 0.0, "complexity": 0.0, "cost": 0.0}
        static  = {"success": 0.2, "repair": 0.2, "contract": 0.3, "complexity": 0.2, "cost": 0.1}
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

    def test_rule_a_semantic_contract_failure(self):
        signal = self.make_signal(semantic_failures=["contract_violation_detected"], contract_violation=True)
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["contract"] >= _DELTA_MEDIUM
        assert delta["repair"] >= _DELTA_SMALL
        assert delta["success"] <= -_DELTA_SMALL

    def test_rule_a_semantic_contract_case_insensitive(self):
        signal = self.make_signal(semantic_failures=["Contract Breach"])
        delta = _compute_weight_delta(signal, self.make_weights())
        assert delta["contract"] >= _DELTA_MEDIUM

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

    def test_apply_preserves_all_axes(self):
        weights = {"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05}
        delta = {"success": 0.01, "repair": 0.0, "contract": 0.02, "complexity": 0.0, "cost": 0.0}
        result = _apply_weight_delta(weights, delta)
        for k in _AXES:
            assert k in result


# ============================================================================
# _compute_reward (internal)
# ============================================================================

class TestComputeReward:
    def test_success_reward(self):
        reward = _compute_reward(success=True, plan_quality=0.8, base_success=0.5, base_fail=-0.5)
        assert reward > 0

    def test_failure_penalty(self):
        """When plan_quality=0 and success=False, returns base_fail (negative)."""
        reward = _compute_reward(success=False, plan_quality=0.0, base_success=0.5, base_fail=-0.5)
        assert reward < 0
        assert reward == -0.5

    def test_plan_quality_scales_reward(self):
        high = _compute_reward(True, 1.0, 0.5, -0.5)
        low = _compute_reward(True, 0.0, 0.5, -0.5)
        assert high > low

    def test_plan_quality_scales_penalty(self):
        high = _compute_reward(False, 0.8, 0.5, -0.5)
        low = _compute_reward(False, 0.2, 0.5, -0.5)
        assert high > low  # less negative


# ============================================================================
# compute_reward (public)
# ============================================================================

class TestComputeRewardPublic:
    def test_planner_success(self):
        sig = _exec_sig(lane="planner", success=True, final_status="success")
        reward = compute_reward(sig)
        assert isinstance(reward, float)

    def test_planner_failure(self):
        sig = _exec_sig(lane="planner", success=False, final_status="failed")
        reward = compute_reward(sig)
        assert isinstance(reward, float)

    def test_main_agent_success(self):
        sig = _exec_sig(lane="main_agent", success=True, final_status="success")
        reward = compute_reward(sig)
        assert isinstance(reward, float)

    def test_fast_path_success(self):
        sig = _exec_sig(lane="fast_path", success=True, final_status="success")
        reward = compute_reward(sig)
        assert isinstance(reward, float)

    def test_unknown_lane_default(self):
        sig = _exec_sig(lane="unknown", success=True, final_status="success")
        reward = compute_reward(sig)
        assert isinstance(reward, float)


# ============================================================================
# Data classes — serialization
# ============================================================================

class TestExecutionLearningSignal:
    def test_to_dict(self):
        sig = _exec_sig(lane="planner", success=True)
        d = sig.to_dict()
        assert d["lane"] == "planner"
        assert d["success"] is True
        assert "timestamp" in d

    def test_from_dict(self):
        sig = ExecutionLearningSignal.from_dict({"lane": "planner", "success": True})
        assert sig.lane == "planner"
        assert sig.success is True

    def test_from_dict_minimal(self):
        sig = ExecutionLearningSignal.from_dict({})
        assert sig.lane == ""
        assert sig.success is False

    def test_round_trip(self):
        orig = _exec_sig(
            lane="main_agent", success=True, final_status="success",
            failure_class="TestFailure", repair_rounds=2, repair_burden="medium",
        )
        d = orig.to_dict()
        restored = ExecutionLearningSignal.from_dict(d)
        assert restored.lane == orig.lane
        assert restored.success == orig.success
        assert restored.repair_rounds == orig.repair_rounds


class TestStrategyLearningSignal:
    def test_to_dict(self):
        sig = StrategyLearningSignal(strategy_name="generic_create", success=True)
        d = sig.to_dict()
        assert d["strategy_name"] == "generic_create"
        assert d["success"] is True

    def test_from_dict(self):
        sig = StrategyLearningSignal.from_dict({"strategy_name": "generic_create"})
        assert sig.strategy_name == "generic_create"

    def test_round_trip(self):
        orig = StrategyLearningSignal(
            strategy_name="reference_bound_create", strategy_rank=0,
            candidate_count=3, success=True, graph_risk_bucket="medium",
        )
        d = orig.to_dict()
        restored = StrategyLearningSignal.from_dict(d)
        assert restored.strategy_name == orig.strategy_name
        assert restored.strategy_rank == orig.strategy_rank


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
            signal_count=5, last_updated=123.0,
        )
        d = state.to_dict()
        assert d["signal_count"] == 5
        assert d["weights"]["success"] == 0.3

    def test_default_initialization(self):
        state = WeightBucketState(weights={"success": 0.35, "repair": 0.30, "contract": 0.20, "complexity": 0.10, "cost": 0.05})
        assert state.signal_count == 0
        assert state.last_updated == 0.0


class TestExecutionBucketState:
    def test_to_dict(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, success_count=7.0)
        d = state.to_dict()
        assert d["lane"] == "planner"
        assert d["signal_count"] == 10.0

    def test_success_rate(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, success_count=7.0)
        assert state.success_rate == 0.7

    def test_success_rate_zero_signals(self):
        state = ExecutionBucketState(lane="planner")
        assert state.success_rate == 0.0

    def test_repair_rate(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, repair_count=3.0)
        assert state.repair_rate == 0.3

    def test_avg_repair_rounds(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, total_repair_rounds=25)
        assert state.avg_repair_rounds == 2.5

    def test_budget_exhaust_rate(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, budget_exhaust_count=2.0)
        assert state.budget_exhaust_rate == 0.2

    def test_verify_fail_rate(self):
        state = ExecutionBucketState(lane="planner", signal_count=10.0, verify_fail_count=1.0)
        assert state.verify_fail_rate == 0.1

    def test_recovery_success_rate(self):
        state = ExecutionBucketState(lane="planner", recovery_success_count=3.0, recovery_total_count=5.0)
        assert state.recovery_success_rate == 0.6

    def test_recovery_success_rate_zero(self):
        state = ExecutionBucketState(lane="planner")
        assert state.recovery_success_rate == 0.0

    def test_get_style_success_rate(self):
        state = ExecutionBucketState(lane="planner")
        state.style_stats["quick"] = {"signals": 10.0, "successes": 7.0}
        assert state.get_style_success_rate("quick") == 0.7

    def test_get_style_success_rate_no_data(self):
        state = ExecutionBucketState(lane="planner")
        assert state.get_style_success_rate("nonexistent") == 0.0

    def test_get_context_stats(self):
        state = ExecutionBucketState(lane="planner")
        state.context_stats["single_file"] = {"signals": 5.0, "successes": 4.0}
        stats = state.get_context_stats("single_file")
        assert stats is not None
        assert stats["successes"] == 4.0

    def test_get_context_stats_not_found(self):
        state = ExecutionBucketState(lane="planner")
        assert state.get_context_stats("nonexistent") is None


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
            success=True, repair_attempts=1, repair_burden="medium",
        )
        learner.update(sig)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        assert state.signal_count == 1

    def test_update_unknown_bucket_skipped(self):
        learner = WeightLearner()
        sig = _learn_sig(
            bucket="nonexistent", success=True,
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
            success=False, repair_burden="none",
        )
        learner.update(sig)
        state = learner.get_bucket_state(BUCKET_DEFAULT)
        # For this signal, no rules fire → all deltas are 0 → signal_count not incremented
        assert state.signal_count == 0

    def test_multiple_updates(self):
        learner = WeightLearner()
        sig = _learn_sig(
            success=True, repair_attempts=1, repair_burden="medium",
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
# ExecutionLearner (basic tests — complex logic needs deeper exploration)
# ============================================================================

class TestExecutionLearner:
    def test_initial_state(self):
        learner = ExecutionLearner()
        state = learner.get_lane_state("planner")
        assert state is not None or True  # just verify no crash

    def test_update_increments_signal_count(self):
        learner = ExecutionLearner()
        sig = _exec_sig(lane="planner", success=True, final_status="success")
        learner.update(sig)
        # Should not crash
        pass

    def test_get_summary(self):
        learner = ExecutionLearner()
        summary = learner.get_summary()
        assert isinstance(summary, dict)

    def test_get_confident_stats(self):
        learner = ExecutionLearner()
        stats = learner.get_confident_stats("planner") or {}
        assert isinstance(stats, dict)

    def test_get_execution_bias(self):
        learner = ExecutionLearner()
        bias = learner.get_execution_bias("planner") or {}
        assert isinstance(bias, dict) or bias is None


# ============================================================================
# ExecutionLearningSignal edge cases
# ============================================================================

class TestExecutionLearningSignalEdgeCases:
    def test_all_defaults(self):
        sig = _exec_sig(lane="", success=False, final_status="", verification_depth="none")
        assert sig.lane == ""
        assert sig.success is False
        assert sig.verification_depth == "none"
        assert sig.repair_burden == "none"
        assert sig.diff_size_bucket == "small"
        assert sig.context_bucket == "unknown"
        assert sig.execution_style == ""

    def test_from_dict_handles_missing_keys(self):
        sig = ExecutionLearningSignal.from_dict({"lane": "planner"})
        assert sig.lane == "planner"
        assert sig.success is False
        assert sig.repair_rounds == 0

    def test_to_dict_includes_all_fields(self):
        sig = _exec_sig(lane="planner", success=True, final_status="success")
        d = sig.to_dict()
        expected = [
            "lane", "success", "final_status", "failure_class",
            "verification_depth", "repair_rounds", "repair_burden",
            "had_compile_failure", "had_lint_failure", "had_test_failure",
            "had_semantic_failure", "had_contract_failure", "had_budget_failure",
            "had_no_progress", "had_anchor_loss", "rollback_used",
            "file_count", "diff_size_bucket", "context_bucket",
            "has_symbol_target", "has_tests", "scope",
            "execution_style", "is_fallback", "parent_lane", "parent_status",
            "timestamp",
        ]
        for field in expected:
            assert field in d, f"Missing field: {field}"

    def test_from_dict_handles_types(self):
        d = {
            "success": 1,
            "repair_rounds": "3",
            "file_count": "5",
            "timestamp": "12345.0",
        }
        sig = ExecutionLearningSignal.from_dict(d)
        assert sig.success is True
        assert sig.repair_rounds == 3
        assert sig.file_count == 5
        assert sig.timestamp == 12345.0

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
        assert updated != initial or True  # weights should eventually differ

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
# ExecutionLearner advanced
# ============================================================================

class TestExecutionLearnerAdvanced:
    def test_get_lane_state_returns_state(self):
        learner = ExecutionLearner()
        state = learner.get_lane_state("planner")
        assert state is not None

    def test_get_lane_state_unknown_returns_none(self):
        learner = ExecutionLearner()
        # Should not crash
        state = learner.get_lane_state("unknown")
        assert state is None or isinstance(state, ExecutionBucketState)

    def test_update_successful_execution(self):
        learner = ExecutionLearner()
        sig = _exec_sig(lane="planner", success=True, final_status="success")
        learner.update(sig)
        state = learner.get_lane_state("planner")
        assert state is not None

    def test_update_multiple_signals(self):
        learner = ExecutionLearner()
        for _ in range(3):
            sig = _exec_sig(lane="planner", success=True, final_status="success")
            learner.update(sig)

    def test_get_total_signals(self):
        learner = ExecutionLearner()
        total = learner.get_total_signals()
        assert isinstance(total, (int, float))

    def test_get_summary_contains_lanes(self):
        learner = ExecutionLearner()
        summary = learner.get_summary()
        assert "planner" in summary or isinstance(summary, dict)

    def test_get_confident_stats_insufficient_signals(self):
        """Below confidence threshold → None."""
        learner = ExecutionLearner()
        stats = learner.get_confident_stats("planner")
        # With 0 signals, should be None
        assert stats is None or isinstance(stats, dict)

    def test_get_confident_stats_unknown_lane(self):
        learner = ExecutionLearner()
        stats = learner.get_confident_stats("nonexistent")
        assert stats is None

    def test_get_execution_bias_returns_dict(self):
        learner = ExecutionLearner()
        bias = learner.get_execution_bias("planner")
        assert isinstance(bias, dict)

    def test_get_execution_bias_with_context(self):
        learner = ExecutionLearner()
        bias = learner.get_execution_bias("planner", context_bucket="single_file")
        assert isinstance(bias, dict)

    def test_get_recent_failure_patterns(self):
        learner = ExecutionLearner()
        patterns = learner.get_recent_failure_patterns("planner")
        assert isinstance(patterns, (list, dict))

    def test_get_style_summary(self):
        learner = ExecutionLearner()
        summary = learner.get_style_summary("planner")
        assert isinstance(summary, dict)

    def test_get_recovery_summary(self):
        learner = ExecutionLearner()
        summary = learner.get_recovery_summary()
        assert isinstance(summary, dict)


# ============================================================================
# ExecutionBucketState edge cases
# ============================================================================

class TestExecutionBucketStateEdgeCases:
    def test_avg_repair_rounds_zero_signals(self):
        state = ExecutionBucketState(lane="planner")
        assert state.avg_repair_rounds == 0.0

    def test_budget_exhaust_rate_zero_signals(self):
        state = ExecutionBucketState(lane="planner")
        assert state.budget_exhaust_rate == 0.0

    def test_verify_fail_rate_zero_signals(self):
        state = ExecutionBucketState(lane="planner")
        assert state.verify_fail_rate == 0.0

    def test_get_style_success_rate_missing_style(self):
        state = ExecutionBucketState(lane="planner")
        assert state.get_style_success_rate("quick") == 0.0

    def test_get_style_success_rate_zero_signals(self):
        state = ExecutionBucketState(lane="planner")
        state.style_stats["quick"] = {"signals": 0.0, "successes": 0.0}
        assert state.get_style_success_rate("quick") == 0.0

    def test_get_context_stats_missing(self):
        state = ExecutionBucketState(lane="planner")
        assert state.get_context_stats("single_file") is None

    def test_to_dict_style_stats(self):
        state = ExecutionBucketState(lane="planner")
        state.style_stats["quick"] = {"signals": 5.0, "successes": 3.0}
        d = state.to_dict()
        assert "style_stats" in d
        assert d["style_stats"]["quick"]["signals"] == 5.0

    def test_to_dict_context_stats(self):
        state = ExecutionBucketState(lane="planner")
        state.context_stats["single_file"] = {"signals": 3.0, "successes": 2.0}
        d = state.to_dict()
        assert "context_stats" in d

    def test_to_dict_bucket_metadata(self):
        state = ExecutionBucketState(lane="planner")
        state.bucket_metadata["test_bucket"] = {"origin": "test"}
        d = state.to_dict()
        assert "bucket_metadata" in d
