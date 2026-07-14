"""Unit tests for strategy_execution_bias — per-strategy execution fitness scoring."""

import time

import pytest

from external_llm.agent.run_store import InMemoryRunStore, RunRecord
from external_llm.editor.learning.strategy_execution_bias import (
    ExecutionBiasResult,
    StrategyExecutionBias,
    compute_execution_bias_by_strategy,
)

STRATEGIES = ["symbol_edit", "minimal_patch", "refactor", "test_first"]


@pytest.fixture()
def populated_store() -> InMemoryRunStore:
    """Create a run_store with synthetic history.

    symbol_edit:   4/5 success, 0 repair
    refactor:      1/3 success, 3 repair avg, 2 compile_error
    minimal_patch: 2/3 success, 0 repair
    test_first:    no history
    """
    store = InMemoryRunStore()

    def _make(strategy, success, failure_class=None, repair_rounds=0):
        return RunRecord(
            run_id=store.create_run_id(),
            timestamp=time.time(),
            plan_mode="operation_plan",
            operation_count=3,
            completed=3 if success else 0,
            failed=0 if success else 2,
            skipped=0,
            final_status="success" if success else "failed",
            final_failure_class=failure_class,
            final_blocking_reasons=[],
            final_warning_reasons=[],
            semantic_gate_passed=True,
            semantic_gate_failed_reasons=[],
            plan_acceptance_passed=True,
            plan_acceptance_failed_checks=[],
            repair_attempted=repair_rounds > 0,
            repair_rounds_attempted=repair_rounds,
            repair_improved=False,
            semantic_issue_codes=[],
            dependency_issues=[],
            completed_ids=[],
            failed_ids=[],
            skipped_ids=[],
            selected_strategy=strategy,
        )

    # symbol_edit: 4 success, 1 fail
    for i in range(5):
        store.add_run(_make("symbol_edit", i < 4, "anchor_loss" if i == 4 else None))

    # refactor: 1 success, 2 fail + high repair
    for i in range(3):
        store.add_run(_make(
            "refactor", i == 0,
            "compile_error" if i > 0 else None,
            repair_rounds=3,
        ))

    # minimal_patch: 2 success, 1 fail
    for i in range(3):
        store.add_run(_make("minimal_patch", i < 2, "no_progress" if i == 2 else None))

    return store


# ---------------------------------------------------------------------------
# Scenario A: modify-existing request
# ---------------------------------------------------------------------------

class TestScenarioA_ModifyExisting:
    REQUEST = "Fix authentication middleware bug in existing auth service"
    TAGS = ["auth", "fix"]
    BUCKET = "product_improve:small"

    @pytest.fixture()
    def result(self, populated_store) -> ExecutionBiasResult:
        return compute_execution_bias_by_strategy(
            run_store=populated_store,
            strategies=STRATEGIES,
            request=self.REQUEST,
            intent_tags=self.TAGS,
            context_bucket=self.BUCKET,
        )

    def test_ordering_symbol_edit_best(self, result: ExecutionBiasResult):
        nb = result.net_biases()
        assert nb["symbol_edit"] > nb["minimal_patch"]
        assert nb["minimal_patch"] > nb["refactor"]

    def test_refactor_negative(self, result: ExecutionBiasResult):
        assert result.net_biases()["refactor"] < 0

    def test_spread_meaningful(self, result: ExecutionBiasResult):
        nb = result.net_biases()
        spread = max(nb.values()) - min(nb.values())
        assert spread > 0.1

    def test_bias_fields_present(self, result: ExecutionBiasResult):
        for s in STRATEGIES:
            b = result.biases[s]
            assert isinstance(b, StrategyExecutionBias)
            assert hasattr(b, "success_rate_bonus")
            assert hasattr(b, "repair_rate_penalty")
            assert hasattr(b, "runtime_gate_penalty")
            assert hasattr(b, "context_affinity")
            assert hasattr(b, "net_bias")

    def test_net_bias_clamped(self, result: ExecutionBiasResult):
        for s, b in result.biases.items():
            assert -0.50 <= b.net_bias <= 0.50, f"{s} net_bias {b.net_bias} out of clamp range"

    def test_no_history_strategy_reasonable(self, result: ExecutionBiasResult):
        """test_first has no runs — should have neutral-ish bias."""
        b = result.biases["test_first"]
        assert abs(b.net_bias) < 0.30


# ---------------------------------------------------------------------------
# Scenario B: create-new request
# ---------------------------------------------------------------------------

class TestScenarioB_CreateNew:
    REQUEST = "Create new payment gateway module with Stripe integration"
    TAGS = ["create", "payment"]
    BUCKET = "product_create:large"

    @pytest.fixture()
    def result_modify(self, populated_store) -> ExecutionBiasResult:
        return compute_execution_bias_by_strategy(
            run_store=populated_store,
            strategies=STRATEGIES,
            request="Fix authentication middleware bug in existing auth service",
            intent_tags=["auth", "fix"],
        )

    @pytest.fixture()
    def result_create(self, populated_store) -> ExecutionBiasResult:
        return compute_execution_bias_by_strategy(
            run_store=populated_store,
            strategies=STRATEGIES,
            request=self.REQUEST,
            intent_tags=self.TAGS,
            context_bucket=self.BUCKET,
        )

    def test_refactor_improves_for_create(
        self,
        result_modify: ExecutionBiasResult,
        result_create: ExecutionBiasResult,
    ):
        nb_m = result_modify.net_biases()
        nb_c = result_create.net_biases()
        assert nb_c["refactor"] > nb_m["refactor"]

    def test_minimal_patch_worsens_for_create(
        self,
        result_modify: ExecutionBiasResult,
        result_create: ExecutionBiasResult,
    ):
        nb_m = result_modify.net_biases()
        nb_c = result_create.net_biases()
        assert nb_c["minimal_patch"] <= nb_m["minimal_patch"]

    def test_to_dict_round_trip(self, result_create: ExecutionBiasResult):
        d = result_create.to_dict()
        for s in STRATEGIES:
            assert s in d
            assert "net_bias" in d[s]
            assert "success_rate_bonus" in d[s]
