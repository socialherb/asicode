"""Tests for the Python primitive learning system (Phase F.1 / F.2).

Covers:
- PrimitiveLearningKey: to_str, from_str with "::" separator, legacy "|" compat
- PrimitiveLearningStore: update, get, iter_typed, to_dict/load_dict
- update_primitive_learning: records outcomes correctly
- score_missing_primitives: Laplace smoothing, ranking
- PrimitiveSequenceStore: record transitions, record sequences, to_dict/load_dict
- update_primitive_sequences: records START->A->B->END
- recommend_sequence: pattern match, greedy fallback
- score_next_primitives
"""
from __future__ import annotations

import pytest

from external_llm.editor.learning.primitive_learning_models import (
    PrimitiveLearningKey,
    PrimitiveOutcomeRecord,
    PrimitiveStrategyStats,
)
from external_llm.editor.learning.primitive_learning_scorer import (
    _DEFAULT_PRIORITY,
    score_missing_primitives,
)
from external_llm.editor.learning.primitive_learning_store import PrimitiveLearningStore
from external_llm.editor.learning.primitive_learning_updater import update_primitive_learning
from external_llm.editor.learning.primitive_sequence_models import (
    PrimitiveSequencePattern,
    PrimitiveTransition,
)
from external_llm.editor.learning.primitive_sequence_scorer import (
    recommend_sequence,
    score_next_primitives,
)
from external_llm.editor.learning.primitive_sequence_store import PrimitiveSequenceStore
from external_llm.editor.learning.primitive_sequence_updater import update_primitive_sequences

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_key(
    ctx: str = "create|auth|medium",
    action: str = "login",
    prim: str = "validate",
    entity: str = "",
) -> PrimitiveLearningKey:
    return PrimitiveLearningKey(
        context_bucket=ctx,
        action_type=action,
        primitive=prim,
        entity=entity,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveLearningKey
# ═════════════════════════════════════════════════════════════════════════════

class TestPrimitiveLearningKey:

    def test_to_str_no_entity(self):
        key = _make_key(ctx="create|auth|medium", action="login", prim="validate")
        s = key.to_str()
        assert s == "create|auth|medium::login::validate"

    def test_to_str_with_entity(self):
        key = _make_key(entity="User")
        s = key.to_str()
        assert s == "create|auth|medium::login::validate::User"

    def test_from_str_double_colon(self):
        key = PrimitiveLearningKey.from_str("create|auth|medium::login::validate")
        assert key.context_bucket == "create|auth|medium"
        assert key.action_type == "login"
        assert key.primitive == "validate"
        assert key.entity == ""

    def test_from_str_double_colon_with_entity(self):
        key = PrimitiveLearningKey.from_str("create|auth|medium::login::validate::User")
        assert key.context_bucket == "create|auth|medium"
        assert key.action_type == "login"
        assert key.primitive == "validate"
        assert key.entity == "User"

    def test_from_str_legacy_pipe_simple(self):
        """Legacy format with no pipes in context_bucket."""
        key = PrimitiveLearningKey.from_str("create|login|validate")
        assert key.context_bucket == "create"
        assert key.action_type == "login"
        assert key.primitive == "validate"

    def test_from_str_legacy_pipe_with_known_primitive(self):
        """Legacy format where context_bucket contains pipes, last is a known primitive."""
        key = PrimitiveLearningKey.from_str("create|auth|medium|login|validate")
        assert key.primitive == "validate"
        assert key.action_type == "login"
        assert key.context_bucket == "create|auth|medium"

    def test_roundtrip(self):
        original = _make_key(entity="Message")
        s = original.to_str()
        restored = PrimitiveLearningKey.from_str(s)
        assert restored.context_bucket == original.context_bucket
        assert restored.action_type == original.action_type
        assert restored.primitive == original.primitive
        assert restored.entity == original.entity


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveOutcomeRecord
# ═════════════════════════════════════════════════════════════════════════════

class TestPrimitiveOutcomeRecord:

    def test_pass_rate(self):
        r = PrimitiveOutcomeRecord(uses=10, pass_count=7)
        assert r.pass_rate == pytest.approx(0.7, abs=1e-3)

    def test_improvement_rate(self):
        r = PrimitiveOutcomeRecord(uses=10, improved_count=3)
        assert r.improvement_rate == pytest.approx(0.3, abs=1e-3)

    def test_avg_coverage_gain(self):
        r = PrimitiveOutcomeRecord(uses=4, total_coverage_delta=0.2)
        assert r.avg_coverage_gain == pytest.approx(0.05, abs=1e-4)

    def test_zero_uses(self):
        r = PrimitiveOutcomeRecord()
        assert r.pass_rate == 0.0
        assert r.improvement_rate == 0.0
        assert r.avg_coverage_gain == 0.0

    def test_to_dict(self):
        r = PrimitiveOutcomeRecord(uses=5, pass_count=3, improved_count=2, chosen_count=4)
        d = r.to_dict()
        assert d["uses"] == 5
        assert d["passed"] == 3
        assert d["improved"] == 2
        assert d["chosen"] == 4
        assert "pass_rate" in d
        assert "improvement_rate" in d


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveStrategyStats
# ═════════════════════════════════════════════════════════════════════════════

class TestPrimitiveStrategyStats:

    def test_success_rate(self):
        s = PrimitiveStrategyStats(strategy_name="c2_insert_verify_call", uses=10, success_count=8)
        assert s.success_rate == pytest.approx(0.8, abs=1e-3)

    def test_to_dict(self):
        s = PrimitiveStrategyStats(strategy_name="test", uses=4, success_count=2, total_gain=0.4)
        d = s.to_dict()
        assert d["strategy"] == "test"
        assert d["success_rate"] == pytest.approx(0.5, abs=1e-3)
        assert d["avg_gain"] == pytest.approx(0.1, abs=1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveLearningStore
# ═════════════════════════════════════════════════════════════════════════════

class TestPrimitiveLearningStore:

    def test_update_and_get(self):
        store = PrimitiveLearningStore()
        key = _make_key()
        store.update(key, chosen=True, improved=True, passed=True, coverage_delta=0.1)
        rec = store.get(key)
        assert rec is not None
        assert rec.uses == 1
        assert rec.chosen_count == 1
        assert rec.improved_count == 1
        assert rec.pass_count == 1
        assert rec.total_coverage_delta == pytest.approx(0.1, abs=1e-4)

    def test_update_accumulates(self):
        store = PrimitiveLearningStore()
        key = _make_key()
        store.update(key, chosen=True, improved=True, passed=True, coverage_delta=0.1)
        store.update(key, chosen=False, improved=False, passed=True, coverage_delta=0.05)
        rec = store.get(key)
        assert rec.uses == 2
        assert rec.chosen_count == 1  # only first was chosen
        assert rec.pass_count == 2
        assert rec.total_coverage_delta == pytest.approx(0.15, abs=1e-4)

    def test_get_missing_returns_none(self):
        store = PrimitiveLearningStore()
        key = _make_key(prim="nonexistent")
        assert store.get(key) is None

    def test_iter_typed(self):
        store = PrimitiveLearningStore()
        k1 = _make_key(prim="validate")
        k2 = _make_key(prim="lookup")
        store.update(k1, chosen=True, improved=True, passed=True)
        store.update(k2, chosen=False, improved=False, passed=False)
        items = list(store.iter_typed())
        assert len(items) == 2
        prims = {k.primitive for k, _ in items}
        assert prims == {"validate", "lookup"}

    def test_lookup_by_primitive(self):
        store = PrimitiveLearningStore()
        k1 = _make_key(ctx="a", action="x", prim="validate")
        k2 = _make_key(ctx="b", action="y", prim="validate")
        k3 = _make_key(ctx="a", action="x", prim="lookup")
        store.update(k1, chosen=True, improved=True, passed=True)
        store.update(k2, chosen=True, improved=True, passed=True)
        store.update(k3, chosen=True, improved=True, passed=True)
        result = store.lookup_by_primitive("validate")
        assert len(result) == 2

    def test_strategy_stats(self):
        store = PrimitiveLearningStore()
        key = _make_key(prim="validate")
        store.update(key, chosen=True, improved=True, passed=True, strategy_name="c2_insert_verify_call")
        stats = store.get_strategy_stats("validate")
        assert "c2_insert_verify_call" in stats
        assert stats["c2_insert_verify_call"].uses == 1

    def test_to_dict_and_load_dict(self):
        store = PrimitiveLearningStore()
        key = _make_key(prim="validate")
        store.update(key, chosen=True, improved=True, passed=True, coverage_delta=0.1, strategy_name="strat_a")
        d = store.to_dict()
        assert "records" in d
        assert "strategy_stats" in d

        # Load into a new store
        store2 = PrimitiveLearningStore()
        store2.load_dict(d)
        rec = store2.get(key)
        assert rec is not None
        assert rec.uses == 1
        assert rec.pass_count == 1
        stats = store2.get_strategy_stats("validate")
        assert "strat_a" in stats

    def test_total_records_and_uses(self):
        store = PrimitiveLearningStore()
        k1 = _make_key(prim="validate")
        k2 = _make_key(prim="lookup")
        store.update(k1, chosen=True, improved=True, passed=True)
        store.update(k1, chosen=True, improved=True, passed=True)
        store.update(k2, chosen=True, improved=True, passed=True)
        assert store.total_records == 2
        assert store.total_uses == 3


# ═════════════════════════════════════════════════════════════════════════════
# update_primitive_learning
# ═════════════════════════════════════════════════════════════════════════════

class TestUpdatePrimitiveLearning:

    def test_not_attempted_returns_early(self):
        store = PrimitiveLearningStore()
        result = update_primitive_learning(
            store=store,
            primitive_ir_summary={},
            reconstruction_meta={"attempted": False},
            semantic_before=0.5, semantic_after=0.5,
            contract_before=0.5, contract_after=0.5,
            final_pass=True,
            context_bucket="test",
        )
        assert result["updated"] is False
        assert result["records_updated"] == 0

    def test_records_correctly(self):
        store = PrimitiveLearningStore()
        result = update_primitive_learning(
            store=store,
            primitive_ir_summary={
                "sequences": [
                    {"type": "login", "entity": "User", "missing": ["validate", "lookup"]}
                ],
            },
            reconstruction_meta={
                "attempted": True,
                "chosen": "reconstructed",
                "raw_coverage": 0.5,
                "reconstructed_coverage": 0.7,
                "missing_primitives": ["validate", "lookup"],
                "applied_primitives": ["validate"],
                "primitive_ir": {
                    "sequences": [{"missing": ["validate", "lookup"]}]
                },
            },
            semantic_before=0.4, semantic_after=0.5,
            contract_before=0.3, contract_after=0.4,
            final_pass=True,
            context_bucket="create|auth|medium",
        )
        assert result["updated"] is True
        assert result["records_updated"] == 2
        assert result["coverage_delta"] == pytest.approx(0.2, abs=1e-4)

        # Check that validate was recorded as chosen+improved (was in applied_primitives)
        key_val = PrimitiveLearningKey(
            context_bucket="create|auth|medium",
            action_type="login",
            primitive="validate",
            entity="User",
        )
        rec_val = store.get(key_val)
        assert rec_val is not None
        assert rec_val.chosen_count == 1
        assert rec_val.improved_count == 1

        # lookup was missing but NOT in applied_primitives → chosen=False
        key_lk = PrimitiveLearningKey(
            context_bucket="create|auth|medium",
            action_type="login",
            primitive="lookup",
            entity="User",
        )
        rec_lk = store.get(key_lk)
        assert rec_lk is not None
        assert rec_lk.chosen_count == 0
        assert rec_lk.improved_count == 0

    def test_raw_chosen_not_reconstructed(self):
        """When raw was chosen, chosen=False for all primitives."""
        store = PrimitiveLearningStore()
        result = update_primitive_learning(
            store=store,
            primitive_ir_summary={
                "sequences": [
                    {"type": "create", "entity": "", "missing": ["validate"]}
                ],
            },
            reconstruction_meta={
                "attempted": True,
                "chosen": "raw",
                "raw_coverage": 0.6,
                "reconstructed_coverage": 0.5,
                "missing_primitives": ["validate"],
                "applied_primitives": ["validate"],
                "primitive_ir": {"sequences": [{"missing": ["validate"]}]},
            },
            semantic_before=0.5, semantic_after=0.5,
            contract_before=0.5, contract_after=0.5,
            final_pass=False,
            context_bucket="test",
        )
        assert result["updated"] is True
        key = PrimitiveLearningKey(
            context_bucket="test", action_type="create", primitive="validate",
        )
        rec = store.get(key)
        assert rec.chosen_count == 0  # raw was chosen
        assert rec.pass_count == 0   # final_pass=False


# ═════════════════════════════════════════════════════════════════════════════
# score_missing_primitives
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreMissingPrimitives:

    def test_cold_start_returns_defaults(self):
        """With no data, returns default priorities."""
        store = PrimitiveLearningStore()
        scores = score_missing_primitives(
            store, "ctx", "login", ["validate", "lookup", "delegate_action"],
        )
        assert len(scores) == 3
        # validate has highest default priority (0.80)
        assert scores[0][0] == "validate"
        assert scores[1][0] == "lookup"
        assert scores[2][0] == "delegate_action"

    def test_learned_data_affects_ranking(self):
        store = PrimitiveLearningStore()
        # Give lookup very good stats
        key_lk = _make_key(prim="lookup")
        for _ in range(10):
            store.update(key_lk, chosen=True, improved=True, passed=True, coverage_delta=0.2)
        # Give validate poor stats
        key_val = _make_key(prim="validate")
        for _ in range(10):
            store.update(key_val, chosen=True, improved=False, passed=False, coverage_delta=-0.05)

        scores = score_missing_primitives(
            store, "create|auth|medium", "login", ["validate", "lookup"],
        )
        # lookup should rank higher than validate with these stats
        assert scores[0][0] == "lookup"

    def test_laplace_smoothing_prevents_zero(self):
        """Even with 0/1 record, score should not be 0 due to smoothing."""
        store = PrimitiveLearningStore()
        key = _make_key(prim="validate")
        store.update(key, chosen=False, improved=False, passed=False, coverage_delta=0.0)
        scores = score_missing_primitives(
            store, "create|auth|medium", "login", ["validate"],
        )
        assert len(scores) == 1
        # With Laplace smoothing: pass_rate = (0+1)/(1+2) = 0.333, improvement_rate = (0+1)/(1+2) = 0.333
        # Score should be > 0
        assert scores[0][1] > 0

    def test_scores_sorted_descending(self):
        store = PrimitiveLearningStore()
        scores = score_missing_primitives(
            store, "ctx", "create",
            ["delegate_action", "validate", "lookup", "persist_state"],
        )
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i + 1][1]

    def test_unknown_primitive_gets_default_score(self):
        store = PrimitiveLearningStore()
        scores = score_missing_primitives(
            store, "ctx", "login", ["unknown_primitive_xyz"],
        )
        assert len(scores) == 1
        # Unknown primitives get default 0.3
        assert scores[0][1] == pytest.approx(0.3, abs=1e-3)

    def test_global_fallback(self):
        """If no exact context match, falls back to global primitive stats."""
        store = PrimitiveLearningStore()
        # Record data in a different context
        key_other = PrimitiveLearningKey(
            context_bucket="other_ctx", action_type="other_action",
            primitive="validate",
        )
        for _ in range(5):
            store.update(key_other, chosen=True, improved=True, passed=True, coverage_delta=0.15)

        scores = score_missing_primitives(
            store, "new_ctx", "new_action", ["validate"],
        )
        assert len(scores) == 1
        # Should use global stats (not cold-start default).
        # The learned score differs from the raw default (0.80) because it uses
        # weighted combination of smoothed rates + coverage gain + default.
        # With good data: pass_rate=(5+1)/(5+2)=0.857, improvement=(5+1)/(5+2)=0.857
        # avg_cov_gain=0.15, cov_score=min(1, 0.15*2)=0.3
        # Score = 0.4*0.857 + 0.3*0.857 + 0.2*0.3 + 0.1*0.80 = 0.74
        # Key: the score IS different from cold-start default, confirming fallback happened.
        assert scores[0][1] != _DEFAULT_PRIORITY["validate"]
        assert scores[0][1] > 0.5  # still a reasonable positive score


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveSequenceStore
# ═════════════════════════════════════════════════════════════════════════════

class TestPrimitiveSequenceStore:

    def test_record_transition(self):
        store = PrimitiveSequenceStore()
        store.record_transition("validate", "lookup", success=True, coverage_gain=0.1)
        t = store.get_transition("validate", "lookup")
        assert t is not None
        assert t.uses == 1
        assert t.success_count == 1
        assert t.total_coverage_gain == pytest.approx(0.1, abs=1e-4)

    def test_record_transition_accumulates(self):
        store = PrimitiveSequenceStore()
        store.record_transition("validate", "lookup", success=True, coverage_gain=0.1)
        store.record_transition("validate", "lookup", success=False, coverage_gain=0.0)
        t = store.get_transition("validate", "lookup")
        assert t.uses == 2
        assert t.success_count == 1

    def test_get_transition_missing(self):
        store = PrimitiveSequenceStore()
        assert store.get_transition("a", "b") is None

    def test_get_outgoing(self):
        store = PrimitiveSequenceStore()
        store.record_transition("validate", "lookup", success=True)
        store.record_transition("validate", "persist_state", success=True)
        store.record_transition("lookup", "persist_state", success=True)
        outgoing = store.get_outgoing("validate")
        assert len(outgoing) == 2

    def test_record_sequence(self):
        store = PrimitiveSequenceStore()
        store.record_sequence("login", ["validate", "lookup", "persist_state"], success=True, coverage_gain=0.3)
        patterns = store.get_best_patterns("login")
        assert len(patterns) == 1
        assert patterns[0].uses == 1
        assert list(patterns[0].sequence) == ["validate", "lookup", "persist_state"]

    def test_record_sequence_accumulates(self):
        store = PrimitiveSequenceStore()
        seq = ["validate", "lookup"]
        store.record_sequence("login", seq, success=True)
        store.record_sequence("login", seq, success=True)
        patterns = store.get_best_patterns("login")
        assert len(patterns) == 1
        assert patterns[0].uses == 2

    def test_record_empty_sequence_noop(self):
        store = PrimitiveSequenceStore()
        store.record_sequence("login", [], success=True)
        assert store.total_patterns == 0

    def test_get_best_patterns_sorted(self):
        store = PrimitiveSequenceStore()
        # Pattern A: 2 uses, 2 successes (rate=1.0)
        store.record_sequence("login", ["validate", "lookup"], success=True)
        store.record_sequence("login", ["validate", "lookup"], success=True)
        # Pattern B: 3 uses, 1 success (rate=0.33)
        store.record_sequence("login", ["lookup", "validate"], success=True)
        store.record_sequence("login", ["lookup", "validate"], success=False)
        store.record_sequence("login", ["lookup", "validate"], success=False)
        patterns = store.get_best_patterns("login", top_k=2)
        assert len(patterns) == 2
        # First should be the one with higher success rate
        assert patterns[0].success_rate >= patterns[1].success_rate

    def test_total_transitions_and_patterns(self):
        store = PrimitiveSequenceStore()
        store.record_transition("a", "b", success=True)
        store.record_transition("b", "c", success=True)
        store.record_sequence("x", ["a", "b", "c"], success=True)
        assert store.total_transitions == 2
        assert store.total_patterns == 1

    def test_to_dict_and_load_dict(self):
        store = PrimitiveSequenceStore()
        store.record_transition("validate", "lookup", success=True, coverage_gain=0.1)
        store.record_sequence("login", ["validate", "lookup"], success=True, coverage_gain=0.2)

        d = store.to_dict()
        assert "transitions" in d
        assert "patterns" in d

        store2 = PrimitiveSequenceStore()
        store2.load_dict(d)
        t = store2.get_transition("validate", "lookup")
        assert t is not None
        assert t.uses == 1
        patterns = store2.get_best_patterns("login")
        assert len(patterns) == 1


# ═════════════════════════════════════════════════════════════════════════════
# PrimitiveTransition / PrimitiveSequencePattern models
# ═════════════════════════════════════════════════════════════════════════════

class TestSequenceModels:

    def test_transition_success_rate(self):
        t = PrimitiveTransition(from_prim="a", to_prim="b", uses=10, success_count=7)
        assert t.success_rate == pytest.approx(0.7, abs=1e-3)

    def test_transition_avg_coverage_gain(self):
        t = PrimitiveTransition(from_prim="a", to_prim="b", uses=4, total_coverage_gain=0.2)
        assert t.avg_coverage_gain == pytest.approx(0.05, abs=1e-4)

    def test_transition_zero_uses(self):
        t = PrimitiveTransition(from_prim="a", to_prim="b")
        assert t.success_rate == 0.0
        assert t.avg_coverage_gain == 0.0

    def test_transition_to_dict(self):
        t = PrimitiveTransition(from_prim="validate", to_prim="lookup", uses=5, success_count=3)
        d = t.to_dict()
        assert d["from"] == "validate"
        assert d["to"] == "lookup"
        assert d["uses"] == 5

    def test_pattern_success_rate(self):
        p = PrimitiveSequencePattern(action_type="login", sequence=("validate", "lookup"), uses=10, success_count=8)
        assert p.success_rate == pytest.approx(0.8, abs=1e-3)

    def test_pattern_to_dict(self):
        p = PrimitiveSequencePattern(
            action_type="login",
            sequence=("validate", "lookup"),
            uses=5,
            success_count=3,
        )
        d = p.to_dict()
        assert d["action_type"] == "login"
        assert d["sequence"] == ["validate", "lookup"]


# ═════════════════════════════════════════════════════════════════════════════
# update_primitive_sequences
# ═════════════════════════════════════════════════════════════════════════════

class TestUpdatePrimitiveSequences:

    def test_records_transitions_and_pattern(self):
        store = PrimitiveSequenceStore()
        result = update_primitive_sequences(
            store,
            applied_primitives=["validate", "lookup"],
            action_type="login",
            success=True,
            coverage_delta=0.2,
        )
        # START->validate, validate->lookup, lookup->END = 3 transitions
        assert result["transitions_recorded"] == 3
        assert result["sequence_recorded"] is True
        assert store.total_transitions == 3
        assert store.total_patterns == 1

        # Check the transitions were actually recorded
        t_start = store.get_transition("__START__", "validate")
        assert t_start is not None
        assert t_start.success_count == 1

        t_mid = store.get_transition("validate", "lookup")
        assert t_mid is not None

        t_end = store.get_transition("lookup", "__END__")
        assert t_end is not None

    def test_deduplicates_applied(self):
        store = PrimitiveSequenceStore()
        result = update_primitive_sequences(
            store,
            applied_primitives=["validate", "validate", "lookup"],
            action_type="login",
            success=True,
        )
        # After dedup: [validate, lookup] → START->validate, validate->lookup, lookup->END = 3
        assert result["transitions_recorded"] == 3

    def test_empty_primitives_noop(self):
        store = PrimitiveSequenceStore()
        result = update_primitive_sequences(
            store, applied_primitives=[], action_type="login", success=True,
        )
        assert result["transitions_recorded"] == 0
        assert result["sequence_recorded"] is False

    def test_single_primitive(self):
        store = PrimitiveSequenceStore()
        result = update_primitive_sequences(
            store,
            applied_primitives=["validate"],
            action_type="create",
            success=False,
        )
        # START->validate, validate->END = 2 transitions
        assert result["transitions_recorded"] == 2
        assert result["sequence_recorded"] is True
        t = store.get_transition("__START__", "validate")
        assert t.success_count == 0  # success=False


# ═════════════════════════════════════════════════════════════════════════════
# score_next_primitives
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreNextPrimitives:

    def test_cold_start_returns_default(self):
        store = PrimitiveSequenceStore()
        scores = score_next_primitives(store, "validate", ["lookup", "persist_state"])
        assert len(scores) == 2
        # Cold start → all get 0.3
        assert scores[0][1] == pytest.approx(0.3, abs=1e-3)
        assert scores[1][1] == pytest.approx(0.3, abs=1e-3)

    def test_learned_transitions_affect_ranking(self):
        store = PrimitiveSequenceStore()
        # validate->lookup: 10 uses, 9 successes
        for _ in range(9):
            store.record_transition("validate", "lookup", success=True, coverage_gain=0.1)
        store.record_transition("validate", "lookup", success=False, coverage_gain=0.0)
        # validate->persist_state: 2 uses, 0 successes
        store.record_transition("validate", "persist_state", success=False)
        store.record_transition("validate", "persist_state", success=False)

        scores = score_next_primitives(store, "validate", ["lookup", "persist_state"])
        # lookup should rank higher
        assert scores[0][0] == "lookup"
        assert scores[0][1] > scores[1][1]

    def test_sorted_descending(self):
        store = PrimitiveSequenceStore()
        scores = score_next_primitives(store, "a", ["x", "y", "z"])
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i + 1][1]


# ═════════════════════════════════════════════════════════════════════════════
# recommend_sequence
# ═════════════════════════════════════════════════════════════════════════════

class TestRecommendSequence:

    def test_empty_returns_empty(self):
        store = PrimitiveSequenceStore()
        assert recommend_sequence(store, "login", []) == []

    def test_pattern_match(self):
        store = PrimitiveSequenceStore()
        # Record a successful pattern
        for _ in range(3):
            store.record_sequence("login", ["validate", "lookup", "persist_state"], success=True)
        result = recommend_sequence(store, "login", ["validate", "lookup", "persist_state"])
        # Should match the pattern and return in that order
        assert result == ["validate", "lookup", "persist_state"]

    def test_pattern_partial_match(self):
        store = PrimitiveSequenceStore()
        for _ in range(3):
            store.record_sequence("login", ["validate", "lookup", "persist_state"], success=True)
        # Ask for a subset
        result = recommend_sequence(store, "login", ["validate", "persist_state"])
        # Pattern has 2/2 relevant primitives (>=50% match) → uses pattern order
        assert "validate" in result
        assert "persist_state" in result

    def test_greedy_fallback(self):
        """No matching pattern → uses greedy transition-based ordering."""
        store = PrimitiveSequenceStore()
        # Record transitions so validate->lookup is preferred
        for _ in range(5):
            store.record_transition("__START__", "validate", success=True, coverage_gain=0.1)
            store.record_transition("validate", "lookup", success=True, coverage_gain=0.1)
        result = recommend_sequence(store, "login", ["lookup", "validate"])
        # Should prefer validate first (better START->validate transition)
        assert result[0] == "validate"
        assert result[1] == "lookup"

    def test_single_primitive(self):
        store = PrimitiveSequenceStore()
        result = recommend_sequence(store, "login", ["validate"])
        assert result == ["validate"]

    def test_no_data_returns_all(self):
        """With no learned data, still returns all primitives in some order."""
        store = PrimitiveSequenceStore()
        result = recommend_sequence(store, "create", ["persist_state", "validate", "lookup"])
        assert len(result) == 3
        assert set(result) == {"persist_state", "validate", "lookup"}
