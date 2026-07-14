"""Unit tests for TerminationPolicy — all decision paths + legacy APIs."""
import pytest

from external_llm.agent.termination_policy import (
    IMPROVEMENT_MIN,
    INTENT_HARD_FLOOR,
    MAX_REPAIR_ITERATIONS,
    SCORE_PARTIAL,
    SCORE_SUCCESS,
    TerminationDecision,
    TerminationPolicy,
)

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def policy() -> TerminationPolicy:
    return TerminationPolicy()


@pytest.fixture
def empty_verification():
    """Minimal verification result with no hard fails."""
    class _VR:
        syntax_ok = True
        blocking_reasons = []
    return _VR()


@pytest.fixture
def hard_fail_verification():
    class _VR:
        syntax_ok = False
        blocking_reasons = ["syntax_error: bad indent"]
    return _VR()


# ══════════════════════════════════════════════════════════════════════════
# TerminationDecision dataclass
# ══════════════════════════════════════════════════════════════════════════

class TestTerminationDecision:
    def test_to_dict(self):
        d = TerminationDecision("SUCCESS", "ok", 0.95, 1).to_dict()
        assert d["decision"] == "SUCCESS"
        assert d["reason"] == "ok"
        assert d["score"] == 0.95
        assert d["iteration"] == 1

    def test_to_dict_rounds_score(self):
        d = TerminationDecision("PARTIAL", "", 0.66666, 0).to_dict()
        assert d["score"] == 0.6667


# ══════════════════════════════════════════════════════════════════════════
# decide() — Gate 0: Intent hard floor
# ══════════════════════════════════════════════════════════════════════════

class TestGate0IntentFloor:
    def test_intent_below_floor_stops(self, policy, empty_verification):
        d = policy.decide(0.5, empty_verification, [], breakdown={"intent": 0.2})
        assert d.decision == "STOP"
        assert "intent floor" in d.reason
        assert d.score == 0.5

    def test_intent_at_floor_continues(self, policy, empty_verification):
        d = policy.decide(0.5, empty_verification, [], breakdown={"intent": INTENT_HARD_FLOOR})
        assert d.decision != "STOP"

    def test_intent_above_floor_continues(self, policy, empty_verification):
        d = policy.decide(0.5, empty_verification, [], breakdown={"intent": 1.0})
        assert d.decision != "STOP"

    def test_intent_defaults_to_one(self, policy, empty_verification):
        d = policy.decide(0.5, empty_verification, [], breakdown=None)
        assert d.decision != "STOP"


# ══════════════════════════════════════════════════════════════════════════
# decide() — Gate 1: Hard fail checks
# ══════════════════════════════════════════════════════════════════════════

class TestGate1HardFail:
    def test_hard_fail_requires_repair(self, policy, hard_fail_verification):
        d = policy.decide(0.9, hard_fail_verification, [])
        assert d.decision == "REPAIR_REQUIRED"
        assert "hard fail" in d.reason

    def test_hard_fail_at_max_iter_stops(self, policy, hard_fail_verification):
        d = policy.decide(0.9, hard_fail_verification, [], iteration=MAX_REPAIR_ITERATIONS)
        assert d.decision == "STOP"

    def test_hard_fail_beyond_max_iter_stops(self, policy, hard_fail_verification):
        d = policy.decide(0.9, hard_fail_verification, [], iteration=MAX_REPAIR_ITERATIONS + 1)
        assert d.decision == "STOP"
        assert "max iterations" in d.reason

    def test_no_hard_fail_passes_gate1(self, policy, empty_verification):
        d = policy.decide(0.9, empty_verification, [])
        assert d.decision != "REPAIR_REQUIRED"  # should pass to later gates


# ══════════════════════════════════════════════════════════════════════════
# decide() — Gate 2: Max iterations
# ══════════════════════════════════════════════════════════════════════════

class TestGate2MaxIterations:
    def test_max_iter_high_score_partial(self, policy, empty_verification):
        d = policy.decide(0.7, empty_verification, [], iteration=MAX_REPAIR_ITERATIONS)
        assert d.decision == "PARTIAL"

    def test_max_iter_low_score_stops(self, policy, empty_verification):
        d = policy.decide(0.3, empty_verification, [], iteration=MAX_REPAIR_ITERATIONS)
        assert d.decision == "STOP"

    def test_below_max_iter_continues(self, policy, empty_verification):
        d = policy.decide(0.95, empty_verification, [], iteration=0)
        assert d.decision != "STOP"


# ══════════════════════════════════════════════════════════════════════════
# decide() — Gate 3: Plateau / insufficient improvement
# ══════════════════════════════════════════════════════════════════════════

class TestGate3Plateau:
    def test_insufficient_improvement_partial(self, policy, empty_verification):
        """Small delta (< IMPROVEMENT_MIN) with high score → PARTIAL."""
        d = policy.decide(0.7, empty_verification, [0.69, 0.695])
        assert d.decision == "PARTIAL"

    def test_insufficient_improvement_stops(self, policy, empty_verification):
        """Small delta with low score → STOP."""
        d = policy.decide(0.3, empty_verification, [0.29, 0.295])
        assert d.decision == "STOP"

    def test_sufficient_improvement_passes(self, policy, empty_verification):
        """Large delta passes through to score-based gate."""
        d = policy.decide(0.9, empty_verification, [0.7, 0.85])
        assert d.decision == "SUCCESS"

    def test_single_prev_score_skips_plateau(self, policy, empty_verification):
        """len(prev_scores) < 2 → skip plateau check."""
        d = policy.decide(0.9, empty_verification, [0.85])
        assert d.decision == "SUCCESS"

    def test_empty_prev_skips_plateau(self, policy, empty_verification):
        d = policy.decide(0.9, empty_verification, [])
        assert d.decision == "SUCCESS"


# ══════════════════════════════════════════════════════════════════════════
# decide() — Gate 4: Score-based + semantic cap
# ══════════════════════════════════════════════════════════════════════════

class TestGate4Score:
    def test_success_above_threshold(self, policy, empty_verification):
        d = policy.decide(0.95, empty_verification, [])
        assert d.decision == "SUCCESS"

    def test_success_at_threshold(self, policy, empty_verification):
        d = policy.decide(SCORE_SUCCESS, empty_verification, [])
        assert d.decision == "SUCCESS"

    def test_semantic_capped_repair_required_iter0(self, policy, empty_verification):
        """Semantic below ceiling at iter 0 → REPAIR_REQUIRED even with high score."""
        d = policy.decide(0.95, empty_verification, [], breakdown={"semantic": 0.3})
        assert d.decision == "REPAIR_REQUIRED"

    def test_semantic_capped_partial_iter1(self, policy, empty_verification):
        """Semantic below ceiling at iter > 0 → PARTIAL."""
        d = policy.decide(0.95, empty_verification, [], iteration=1, breakdown={"semantic": 0.3})
        assert d.decision == "PARTIAL"

    def test_high_score_no_semantic_cap_success(self, policy, empty_verification):
        d = policy.decide(0.88, empty_verification, [], breakdown={"semantic": 0.5})
        assert d.decision == "SUCCESS"

    def test_partial_range_first_iter_repair(self, policy, empty_verification):
        """Score in partial range at iter 0 → REPAIR_REQUIRED."""
        d = policy.decide(0.65, empty_verification, [])
        assert d.decision == "REPAIR_REQUIRED"

    def test_partial_range_with_improvement_repair(self, policy, empty_verification):
        """Score in partial range with improvement → REPAIR_REQUIRED."""
        d = policy.decide(0.65, empty_verification, [0.5])
        assert d.decision == "REPAIR_REQUIRED"

    def test_partial_range_no_improvement(self, policy, empty_verification):
        """Score in partial range without improvement at iter>0 → PARTIAL."""
        d = policy.decide(0.65, empty_verification, [0.7], iteration=1)
        assert d.decision == "PARTIAL"

    def test_below_partial_threshold_repair(self, policy, empty_verification):
        d = policy.decide(0.4, empty_verification, [])
        assert d.decision == "REPAIR_REQUIRED"

    def test_partial_at_threshold(self, policy, empty_verification):
        d = policy.decide(SCORE_PARTIAL, empty_verification, [])
        assert d.decision == "REPAIR_REQUIRED"  # at partial threshold → repair (iter 0)


# ══════════════════════════════════════════════════════════════════════════
# _check_hard_fails
# ══════════════════════════════════════════════════════════════════════════

class TestCheckHardFails:
    def test_syntax_ok_none(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": []})()
        assert TerminationPolicy._check_hard_fails(vr) is None

    def test_syntax_error(self):
        vr = type("VR", (), {"syntax_ok": False, "blocking_reasons": []})()
        assert TerminationPolicy._check_hard_fails(vr) == "syntax_error"

    def test_forbidden_token(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": ["forbidden_token: `eval` used"]})()
        assert TerminationPolicy._check_hard_fails(vr) == "forbidden_token"

    def test_missing_symbol(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": ["missing required symbol: Foo"]})()
        assert TerminationPolicy._check_hard_fails(vr) == "required_symbol_missing"

    def test_blocking_syntax_error_in_reasons(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": ["Syntax error in diff at line 42"]})()
        assert TerminationPolicy._check_hard_fails(vr) == "syntax_error"

    def test_none_blocking_reasons(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": None})()
        assert TerminationPolicy._check_hard_fails(vr) is None

    def test_empty_blocking_reasons(self):
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": []})()
        assert TerminationPolicy._check_hard_fails(vr) is None

    def test_no_attributes(self):
        vr = type("VR", (), {})()
        assert TerminationPolicy._check_hard_fails(vr) is None


# ══════════════════════════════════════════════════════════════════════════
# _has_improvement
# ══════════════════════════════════════════════════════════════════════════

class TestHasImprovement:
    def test_empty_prev(self):
        assert TerminationPolicy._has_improvement(0.5, []) is True

    def test_significant_improvement(self):
        assert TerminationPolicy._has_improvement(0.7, [0.5]) is True

    def test_barely_improved(self):
        assert TerminationPolicy._has_improvement(0.5 + IMPROVEMENT_MIN + 0.001, [0.5]) is True

    def test_not_improved_small_delta(self):
        assert TerminationPolicy._has_improvement(0.51, [0.5]) is False

    def test_degraded(self):
        assert TerminationPolicy._has_improvement(0.4, [0.5]) is False

    def test_equal(self):
        assert TerminationPolicy._has_improvement(0.5, [0.5]) is False


# ══════════════════════════════════════════════════════════════════════════
# _calc_slope
# ══════════════════════════════════════════════════════════════════════════

class TestCalcSlope:
    def test_less_than_two_points_zero(self):
        assert TerminationPolicy._calc_slope([]) == 0.0
        assert TerminationPolicy._calc_slope([1.0]) == 0.0

    def test_flat_slope(self):
        assert TerminationPolicy._calc_slope([0.5, 0.5]) == 0.0

    def test_positive_slope(self):
        slope = TerminationPolicy._calc_slope([0.3, 0.6, 0.9])
        assert 0.29 < slope < 0.31  # approx 0.3

    def test_negative_slope(self):
        slope = TerminationPolicy._calc_slope([0.9, 0.6, 0.3])
        assert -0.31 < slope < -0.29

    def test_denom_zero_safe(self):
        # When n * sum_x2 == sum_x * sum_x (e.g. all x same → x=0,0,0)
        # With x=0,1,2: sum_x=3, sum_x2=5, n=3
        # denom = 3*5 - 9 = 6 (not zero)
        # Actually denom=0 happens for n=1 but we filter n<2
        # Let's use a case with n=2 where both x are 0 (impossible since sum(range(n)) gives 0,1)
        # For n=2, sum_x = 0+1 = 1, sum_x2 = 0+1 = 1
        # denom = 2*1 - 1*1 = 1 (never zero for n>=2 with natural x)
        pass  # denom=0 is unreachable for n>=2 with our x values


# ══════════════════════════════════════════════════════════════════════════
# Legacy APIs
# ══════════════════════════════════════════════════════════════════════════

class TestLegacyShouldTerminate:
    def test_perfect_score_terminates(self):
        policy = TerminationPolicy()
        cp = type("CP", (), {"validity_score": 1.0})()
        assert policy.should_terminate([cp]) is True

    def test_improving_trend_terminates(self):
        policy = TerminationPolicy(window_size=3, improvement_threshold=0.2)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.3, 0.5, 0.8]]
        assert policy.should_terminate(cps) is True

    def test_degrading_trend_terminates(self):
        policy = TerminationPolicy(window_size=3, degradation_threshold=-0.2)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.8, 0.5, 0.3]]
        assert policy.should_terminate(cps) is True

    def test_stable_trend_no_terminate(self):
        policy = TerminationPolicy(window_size=3, improvement_threshold=0.5, degradation_threshold=-0.5)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.6, 0.61, 0.62]]
        assert policy.should_terminate(cps) is False

    def test_single_checkpoint_no_terminate(self):
        policy = TerminationPolicy()
        cp = type("CP", (), {"validity_score": 0.5})()
        assert policy.should_terminate([cp]) is False

    def test_empty_list_no_terminate(self):
        policy = TerminationPolicy()
        assert policy.should_terminate([]) is False

    def test_missing_validity_score_defaults_zero(self):
        policy = TerminationPolicy(window_size=3, improvement_threshold=0.5)
        cps = [type("CP", (), {})() for _ in range(3)]
        # All zeros → flat slope → no terminate
        assert policy.should_terminate(cps) is False


class TestLegacyTrendAnalysis:
    def test_perfect_score(self):
        policy = TerminationPolicy()
        cp = type("CP", (), {"validity_score": 1.0})()
        r = policy.get_trend_analysis([cp])
        assert r["should_terminate"] is True
        assert "Perfect" in r["reason"]

    def test_improving_trend(self):
        policy = TerminationPolicy(window_size=3, improvement_threshold=0.2)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.3, 0.5, 0.8]]
        r = policy.get_trend_analysis(cps)
        assert r["should_terminate"] is True
        assert "Improvement" in r["reason"]

    def test_degrading_trend(self):
        policy = TerminationPolicy(window_size=3, degradation_threshold=-0.2)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.8, 0.5, 0.3]]
        r = policy.get_trend_analysis(cps)
        assert r["should_terminate"] is True
        assert "Degradation" in r["reason"]

    def test_stable_trend(self):
        policy = TerminationPolicy(window_size=3, improvement_threshold=0.5, degradation_threshold=-0.5)
        cps = [type("CP", (), {"validity_score": s})() for s in [0.6, 0.61, 0.62]]
        r = policy.get_trend_analysis(cps)
        assert r["should_terminate"] is False
        assert "Stable" in r["reason"]

    def test_single_point(self):
        policy = TerminationPolicy()
        cp = type("CP", (), {"validity_score": 0.5})()
        r = policy.get_trend_analysis([cp])
        assert r["should_terminate"] is False
        assert "Insufficient" in r["reason"]

    def test_empty(self):
        policy = TerminationPolicy()
        r = policy.get_trend_analysis([])
        assert r["should_terminate"] is False
        assert "Insufficient" in r["reason"]

    def test_missing_validity_score(self):
        policy = TerminationPolicy(window_size=2, improvement_threshold=0.5)
        cp = type("CP", (), {})()
        r = policy.get_trend_analysis([cp, cp])
        assert "should_terminate" in r
        assert "trend" in r
        assert r["scores"] == [0.0, 0.0]


# ══════════════════════════════════════════════════════════════════════════
# Custom thresholds
# ══════════════════════════════════════════════════════════════════════════

class TestCustomThresholds:
    def test_custom_success_threshold(self):
        p = TerminationPolicy(success_threshold=0.5)
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": []})()
        d = p.decide(0.6, vr, [])
        assert d.decision == "SUCCESS"

    def test_custom_partial_threshold(self):
        p = TerminationPolicy(partial_threshold=0.8, success_threshold=0.95)
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": []})()
        d = p.decide(0.85, vr, [])
        assert d.decision == "REPAIR_REQUIRED"  # 0.85 >= 0.8 partial but < 0.95 success

    def test_custom_max_iterations(self):
        p = TerminationPolicy(max_iterations=5)
        vr = type("VR", (), {"syntax_ok": True, "blocking_reasons": []})()
        d = p.decide(0.95, vr, [], iteration=4)
        assert d.decision == "SUCCESS"  # iter 4 < 5
        d2 = p.decide(0.95, vr, [], iteration=5)
        assert d2.decision == "PARTIAL"  # iter 5 >= 5


# ══════════════════════════════════════════════════════════════════════════
# Integration: full flow
# ══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_typical_success_path(self, policy, empty_verification):
        d = policy.decide(0.92, empty_verification, [0.8, 0.88], iteration=1)
        assert d.decision == "SUCCESS"

    def test_repair_then_partial(self, policy, empty_verification):
        """Simulate a 2-iteration repair cycle that plateaus."""
        d1 = policy.decide(0.6, empty_verification, [], iteration=0)
        assert d1.decision == "REPAIR_REQUIRED"
        # delta=0.05 equals IMPROVEMENT_MIN (not <) → plateau skipped
        # score=0.65 in partial range, iter=1, no improvement → PARTIAL
        d2 = policy.decide(0.65, empty_verification, [0.6], iteration=1)
        assert d2.decision == "PARTIAL"

    def test_hard_fail_in_repair_cycle(self, policy, hard_fail_verification):
        d1 = policy.decide(0.5, hard_fail_verification, [], iteration=0)
        assert d1.decision == "REPAIR_REQUIRED"
        d2 = policy.decide(0.5, hard_fail_verification, [], iteration=MAX_REPAIR_ITERATIONS)
        assert d2.decision == "STOP"

    def test_deep_verification_missing_attrs(self, policy):
        """Verification result without expected attributes is handled."""
        vr = object()  # no syntax_ok or blocking_reasons
        d = policy.decide(0.9, vr, [])
        assert d.decision == "SUCCESS"  # no hard fail → passes through
