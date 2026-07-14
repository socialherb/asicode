"""Tests for ExecutionGraphAdvisor."""
from unittest.mock import MagicMock

from external_llm.graph.execution_graph_advisor import (
    HIGH_CALLER_THRESHOLD,
    ExecutionGraphAdvisor,
    GraphRefactorContext,
    GraphRepairHints,
    GraphVerificationScope,
)


def _make_facade(callers=None, callees=None, related=None):
    facade = MagicMock()
    facade.get_callers = MagicMock(return_value=callers or [])
    facade.get_callees = MagicMock(return_value=callees or [])
    facade.get_related_symbols = MagicMock(return_value=related or [])
    return facade


def _make_graph_context(
    confidence=0.8,
    resolved=None,
    unresolved=None,
    impact_files=None,
    primary_files=None,
    callers=None,
    callees=None,
):
    return {
        "graph_confidence": confidence,
        "resolved_symbols": resolved or [],
        "unresolved_symbols": unresolved or [],
        "impact_files": impact_files or [],
        "primary_files": primary_files or [],
        "callers": callers or {},
        "callees": callees or {},
    }


class TestRepairHints:
    def test_high_confidence_prefers_symbol_focused(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(
            confidence=0.9,
            resolved=[{"name": "foo", "file_path": "a.py"}],
        )
        hints = advisor.get_repair_hints(graph_context=gc)
        assert hints.prefer_symbol_focused_repair is True

    def test_high_caller_count_conservative(self):
        advisor = ExecutionGraphAdvisor()
        callers = {"foo": [{"symbol": f"caller_{i}"} for i in range(12)]}
        gc = _make_graph_context(callers=callers)
        hints = advisor.get_repair_hints(target_symbol="foo", graph_context=gc)
        assert hints.high_breakage_risk is True
        assert hints.prefer_conservative_repair is True
        assert hints.caller_count >= HIGH_CALLER_THRESHOLD

    def test_wide_impact_conservative(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(impact_files=[f"f{i}.py" for i in range(7)])
        hints = advisor.get_repair_hints(graph_context=gc)
        assert hints.prefer_conservative_repair is True

    def test_no_graph_returns_defaults(self):
        advisor = ExecutionGraphAdvisor()
        hints = advisor.get_repair_hints()
        assert hints.prefer_symbol_focused_repair is False
        assert hints.high_breakage_risk is False

    def test_facade_fallback_high_callers(self):
        callers = [MagicMock() for _ in range(11)]
        facade = _make_facade(callers=callers)
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        hints = advisor.get_repair_hints(target_symbol="foo")
        assert hints.high_breakage_risk is True

    def test_facade_fallback_low_impact(self):
        facade = _make_facade(callers=[], callees=[MagicMock()])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        hints = advisor.get_repair_hints(target_symbol="foo")
        assert hints.prefer_symbol_focused_repair is True

    def test_to_dict_contains_all_fields(self):
        hints = GraphRepairHints(prefer_symbol_focused_repair=True, caller_count=5, reason="test")
        d = hints.to_dict()
        assert d["prefer_symbol_focused_repair"] is True
        assert d["caller_count"] == 5
        assert d["reason"] == "test"
        assert "high_breakage_risk" in d
        assert "prefer_conservative_repair" in d
        assert "impact_file_count" in d


class TestSafetyIssues:
    def test_high_caller_warning(self):
        advisor = ExecutionGraphAdvisor()
        callers = {"foo": [{"symbol": f"c{i}"} for i in range(12)]}
        gc = _make_graph_context(callers=callers)
        issues = advisor.get_safety_issues(graph_context=gc)
        codes = [i.code for i in issues]
        assert "HIGH_CALLER_SYMBOL_EDIT" in codes

    def test_wide_impact_warning(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(impact_files=[f"f{i}.py" for i in range(8)])
        issues = advisor.get_safety_issues(graph_context=gc)
        codes = [i.code for i in issues]
        assert "WIDE_IMPACT_REFACTOR" in codes

    def test_low_confidence_structural_edit(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.2)
        issues = advisor.get_safety_issues(
            graph_context=gc,
            operation_kind="MODIFY_SYMBOL",
        )
        codes = [i.code for i in issues]
        assert "LOW_CONFIDENCE_STRUCTURAL_EDIT" in codes

    def test_unresolved_structural_edit(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(unresolved=["a", "b", "c"])
        issues = advisor.get_safety_issues(
            graph_context=gc,
            operation_kind="RENAME_SYMBOL",
        )
        codes = [i.code for i in issues]
        assert "UNRESOLVED_SYMBOL_STRUCTURAL_EDIT" in codes

    def test_no_issues_without_graph(self):
        advisor = ExecutionGraphAdvisor()
        issues = advisor.get_safety_issues()
        assert len(issues) == 0

    def test_no_structural_warning_for_read(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.2)
        issues = advisor.get_safety_issues(
            graph_context=gc,
            operation_kind="READ_SYMBOL",
        )
        codes = [i.code for i in issues]
        assert "LOW_CONFIDENCE_STRUCTURAL_EDIT" not in codes

    def test_facade_fallback_high_callers(self):
        callers = [MagicMock() for _ in range(11)]
        facade = _make_facade(callers=callers)
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        issues = advisor.get_safety_issues(target_symbols=["foo"])
        codes = [i.code for i in issues]
        assert "HIGH_CALLER_SYMBOL_EDIT" in codes

    def test_no_issues_below_threshold(self):
        advisor = ExecutionGraphAdvisor()
        callers = {"foo": [{"symbol": f"c{i}"} for i in range(3)]}
        gc = _make_graph_context(callers=callers, impact_files=["a.py", "b.py"])
        issues = advisor.get_safety_issues(graph_context=gc)
        assert len(issues) == 0


class TestVerificationScope:
    def test_scope_from_graph_context(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(
            resolved=[{"name": "foo"}, {"name": "bar"}],
            primary_files=["a.py"],
            impact_files=["a.py", "b.py", "tests/test_a.py"],
            callers={"foo": [{"symbol": "baz", "file": "c.py"}]},
        )
        scope = advisor.get_verification_scope(graph_context=gc)
        assert "foo" in scope.symbols
        assert "baz" in scope.symbols
        assert "tests/test_a.py" in scope.test_targets
        assert scope.scope_reason != ""

    def test_scope_empty_without_graph(self):
        advisor = ExecutionGraphAdvisor()
        scope = advisor.get_verification_scope()
        assert len(scope.symbols) == 0
        assert len(scope.files) == 0

    def test_test_targets_from_impact_files(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(
            impact_files=["src/foo.py", "tests/test_foo.py", "tests/unit/test_bar.py"],
        )
        scope = advisor.get_verification_scope(graph_context=gc)
        assert "tests/test_foo.py" in scope.test_targets
        assert "tests/unit/test_bar.py" in scope.test_targets
        assert "src/foo.py" not in scope.test_targets

    def test_target_files_included(self):
        advisor = ExecutionGraphAdvisor()
        scope = advisor.get_verification_scope(target_files=["extra.py"])
        assert "extra.py" in scope.files

    def test_to_dict(self):
        scope = GraphVerificationScope(symbols=["a"], files=["b.py"], scope_reason="test")
        d = scope.to_dict()
        assert d["symbols"] == ["a"]
        assert d["files"] == ["b.py"]
        assert d["scope_reason"] == "test"
        assert "test_targets" in d

    def test_facade_fallback_collects_callers(self):
        caller_edge = MagicMock()
        caller_edge.caller_symbol = "bar"
        caller_edge.caller_file = "b.py"
        facade = _make_facade(callers=[caller_edge])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        scope = advisor.get_verification_scope(target_symbols=["foo"])
        assert "foo" in scope.symbols
        assert "bar" in scope.symbols
        assert "b.py" in scope.files


class TestRefactorContext:
    def test_refactor_context_high_callers(self):
        callers = [MagicMock(caller_file=f"f{i}.py") for i in range(12)]
        facade = _make_facade(callers=callers, callees=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        ctx = advisor.get_refactor_context("MyFunc")
        assert ctx.caller_count == 12
        assert ctx.risk_level == "high"

    def test_refactor_context_low_callers(self):
        callers = [MagicMock(caller_file="a.py")]
        facade = _make_facade(callers=callers, callees=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        ctx = advisor.get_refactor_context("MyFunc")
        assert ctx.risk_level == "low"

    def test_refactor_context_no_facade(self):
        advisor = ExecutionGraphAdvisor()
        ctx = advisor.get_refactor_context("MyFunc")
        assert ctx.risk_level == "unknown"
        assert ctx.caller_count == 0

    def test_refactor_context_collects_impact_files(self):
        callers = [MagicMock(caller_file="a.py"), MagicMock(caller_file="b.py")]
        callees = [MagicMock(callee_file="c.py")]
        facade = _make_facade(callers=callers, callees=callees)
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        ctx = advisor.get_refactor_context("MyFunc")
        assert "a.py" in ctx.impact_files
        assert "c.py" in ctx.impact_files

    def test_refactor_context_medium_risk(self):
        callers = [MagicMock(caller_file=f"f{i}.py") for i in range(4)]
        facade = _make_facade(callers=callers, callees=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        ctx = advisor.get_refactor_context("MyFunc")
        assert ctx.risk_level == "medium"

    def test_to_dict(self):
        ctx = GraphRefactorContext(target_symbol="foo", risk_level="high")
        d = ctx.to_dict()
        assert d["risk_level"] == "high"
        assert d["target_symbol"] == "foo"
        assert "caller_count" in d
        assert "callee_count" in d
        assert "impact_files" in d
        assert "related_symbol_count" in d

    def test_available_property(self):
        advisor_no_graph = ExecutionGraphAdvisor()
        assert advisor_no_graph.available is False

        facade = _make_facade()
        advisor_with_graph = ExecutionGraphAdvisor(graph_facade=facade)
        assert advisor_with_graph.available is True


class TestTestTargets:
    def test_test_targets_from_graph_context(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(
            impact_files=["src/module.py", "tests/test_module.py"],
        )
        targets = advisor.get_test_targets(graph_context=gc)
        assert "tests/test_module.py" in targets
        assert "src/module.py" not in targets

    def test_no_targets_without_graph(self):
        advisor = ExecutionGraphAdvisor()
        targets = advisor.get_test_targets()
        assert targets == []

    def test_test_targets_delegates_to_scope(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(
            impact_files=["a.py", "test_b.py", "tests/c.py"],
        )
        targets = advisor.get_test_targets(graph_context=gc)
        assert "test_b.py" in targets
        assert "tests/c.py" in targets
        assert "a.py" not in targets


class TestDataclassSerialize:
    def test_repair_hints_to_dict(self):
        hints = GraphRepairHints(prefer_symbol_focused_repair=True, caller_count=5, reason="test")
        d = hints.to_dict()
        assert d["prefer_symbol_focused_repair"] is True
        assert d["caller_count"] == 5

    def test_verification_scope_to_dict(self):
        scope = GraphVerificationScope(symbols=["a"], files=["b.py"], scope_reason="test")
        d = scope.to_dict()
        assert d["symbols"] == ["a"]

    def test_refactor_context_to_dict(self):
        ctx = GraphRefactorContext(target_symbol="foo", risk_level="high")
        d = ctx.to_dict()
        assert d["risk_level"] == "high"


class TestGracefulFallback:
    """Ensure all methods are robust to errors and None inputs."""

    def test_repair_hints_facade_exception(self):
        facade = MagicMock()
        facade.get_callers.side_effect = RuntimeError("graph error")
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        hints = advisor.get_repair_hints(target_symbol="foo")
        # Should return defaults without raising
        assert isinstance(hints, GraphRepairHints)
        assert hints.high_breakage_risk is False

    def test_safety_issues_facade_exception(self):
        facade = MagicMock()
        facade.get_callers.side_effect = RuntimeError("graph error")
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        issues = advisor.get_safety_issues(target_symbols=["foo"])
        # Should return empty without raising
        assert issues == []

    def test_verification_scope_facade_exception(self):
        facade = MagicMock()
        facade.get_callers.side_effect = RuntimeError("graph error")
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        scope = advisor.get_verification_scope(target_symbols=["foo"])
        # Should return defaults without raising
        assert isinstance(scope, GraphVerificationScope)

    def test_refactor_context_facade_exception(self):
        facade = MagicMock()
        facade.get_callers.side_effect = RuntimeError("graph error")
        advisor = ExecutionGraphAdvisor(graph_facade=facade)
        ctx = advisor.get_refactor_context("foo")
        # Should return defaults without raising; risk_level stays "unknown"
        assert isinstance(ctx, GraphRefactorContext)
        assert ctx.risk_level == "unknown"

    def test_all_none_inputs(self):
        advisor = ExecutionGraphAdvisor()
        assert isinstance(advisor.get_repair_hints(), GraphRepairHints)
        assert advisor.get_safety_issues() == []
        assert isinstance(advisor.get_verification_scope(), GraphVerificationScope)
        assert isinstance(advisor.get_refactor_context("x"), GraphRefactorContext)
        assert advisor.get_test_targets() == []


class TestAdvisorMemoization:
    def test_memo_caches_repair_hints(self):
        """Same inputs → memoized result."""
        callers = [MagicMock() for _ in range(12)]
        facade = _make_facade(callers=callers)
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        h1 = advisor.get_repair_hints(target_symbol="foo")
        h2 = advisor.get_repair_hints(target_symbol="foo")

        # Should be same object (memoized)
        assert h1 is h2
        assert facade.get_callers.call_count == 1  # called only once

    def test_memo_different_symbols(self):
        """Different symbols → separate cache entries."""
        facade = _make_facade(callers=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        advisor.get_repair_hints(target_symbol="foo")
        advisor.get_repair_hints(target_symbol="bar")

        assert facade.get_callers.call_count == 2

    def test_memo_clear(self):
        """clear_memo() resets memoization."""
        facade = _make_facade(callers=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        advisor.get_repair_hints(target_symbol="foo")
        advisor.clear_memo()
        advisor.get_repair_hints(target_symbol="foo")

        assert facade.get_callers.call_count == 2

    def test_memo_stats(self):
        """get_memo_stats() reports hits and misses."""
        facade = _make_facade(callers=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        advisor.get_repair_hints(target_symbol="foo")  # miss
        advisor.get_repair_hints(target_symbol="foo")  # hit

        stats = advisor.get_memo_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_memo_works_with_graph_context(self):
        """Memoization works when using graph_context instead of facade."""
        advisor = ExecutionGraphAdvisor()  # no facade
        gc = _make_graph_context(confidence=0.9, resolved=[{"name": "foo"}])

        h1 = advisor.get_repair_hints(graph_context=gc)
        h2 = advisor.get_repair_hints(graph_context=gc)
        assert h1 is h2

    def test_memo_safety_issues(self):
        """get_safety_issues() is memoized."""
        callers = [MagicMock() for _ in range(12)]
        facade = _make_facade(callers=callers)
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        i1 = advisor.get_safety_issues(target_symbols=["foo"])
        i2 = advisor.get_safety_issues(target_symbols=["foo"])
        assert i1 is i2
        assert facade.get_callers.call_count == 1

    def test_memo_verification_scope(self):
        """get_verification_scope() is memoized — second call returns same object."""
        caller_edge = MagicMock()
        caller_edge.caller_symbol = "bar"
        caller_edge.caller_file = "b.py"
        facade = _make_facade(callers=[caller_edge])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        s1 = advisor.get_verification_scope(target_symbols=["foo"])
        count_after_first = facade.get_callers.call_count
        s2 = advisor.get_verification_scope(target_symbols=["foo"])
        # Memoization: second call must not invoke facade again
        assert s1 is s2
        assert facade.get_callers.call_count == count_after_first

    def test_memo_refactor_context(self):
        """get_refactor_context() is memoized."""
        callers = [MagicMock(caller_file="a.py") for _ in range(3)]
        facade = _make_facade(callers=callers, callees=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        ctx1 = advisor.get_refactor_context("MyFunc")
        ctx2 = advisor.get_refactor_context("MyFunc")
        assert ctx1 is ctx2
        assert facade.get_callers.call_count == 1

    def test_memo_stats_initial(self):
        """Fresh advisor has zero memo stats."""
        advisor = ExecutionGraphAdvisor()
        stats = advisor.get_memo_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

    def test_memo_hit_rate(self):
        """Hit rate is computed correctly."""
        facade = _make_facade(callers=[])
        advisor = ExecutionGraphAdvisor(graph_facade=facade)

        advisor.get_repair_hints(target_symbol="foo")  # miss
        advisor.get_repair_hints(target_symbol="foo")  # hit

        stats = advisor.get_memo_stats()
        assert stats["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# TestExecutionPolicy — new get_execution_policy() method
# ---------------------------------------------------------------------------

class TestExecutionPolicy:
    def test_no_graph_context_trusted(self):
        advisor = ExecutionGraphAdvisor()
        policy = advisor.get_execution_policy(operation_kind="MODIFY_SYMBOL", graph_context=None)
        assert policy["mode"] == "trusted"
        assert policy["block_structural_edit"] is False
        assert policy["requires_anchor_read"] is False
        assert policy["fallback_reason"] is None

    def test_high_confidence_structural_trusted(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.9, unresolved=[])
        policy = advisor.get_execution_policy(
            operation_kind="MODIFY_SYMBOL", graph_context=gc
        )
        assert policy["mode"] == "trusted"
        assert policy["block_structural_edit"] is False
        assert policy["requires_anchor_read"] is False

    def test_low_confidence_structural_requires_anchor(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.3, unresolved=[])
        policy = advisor.get_execution_policy(
            operation_kind="MODIFY_SYMBOL", graph_context=gc
        )
        assert policy["mode"] == "conservative"
        assert policy["requires_anchor_read"] is True
        assert policy["block_structural_edit"] is False

    def test_blocked_policy_for_very_low_confidence_with_unresolved(self):
        # "blocked" mode was removed — pre-edit blocking is redundant with downstream defenses.
        # Very low confidence + unresolved symbols now results in "conservative" mode
        # with block_structural_edit=False (rely on anchor validation + acceptance criteria).
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.1, unresolved=["sym_a"])
        policy = advisor.get_execution_policy(
            operation_kind="MODIFY_SYMBOL", graph_context=gc
        )
        assert policy["mode"] == "conservative"
        assert policy["block_structural_edit"] is False  # never block — downstream defenses
        assert policy["force_conservative_mode"] is True
        assert policy["requires_anchor_read"] is True

    def test_non_structural_op_not_blocked(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.1, unresolved=["sym_a"])
        policy = advisor.get_execution_policy(
            operation_kind="READ_SYMBOL", graph_context=gc
        )
        assert policy["block_structural_edit"] is False
        assert policy["requires_anchor_read"] is False

    def test_insert_after_symbol_is_structural(self):
        # INSERT_AFTER_SYMBOL is structural → requires_anchor_read=True in conservative mode.
        # block_structural_edit is always False (blocked mode was removed).
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.1, unresolved=["sym_a"])
        policy = advisor.get_execution_policy(
            operation_kind="INSERT_AFTER_SYMBOL", graph_context=gc
        )
        assert policy["block_structural_edit"] is False  # never block
        assert policy["requires_anchor_read"] is True    # structural op in conservative mode

    def test_policy_contains_required_keys(self):
        advisor = ExecutionGraphAdvisor()
        policy = advisor.get_execution_policy(graph_context=None)
        for key in (
            "mode", "requires_anchor_read", "force_conservative_mode",
            "block_structural_edit", "fallback_reason",
            "graph_confidence", "unresolved_count",
        ):
            assert key in policy, f"Missing key: {key}"

    def test_guarded_with_mid_confidence(self):
        advisor = ExecutionGraphAdvisor()
        gc = _make_graph_context(confidence=0.6, unresolved=[])
        policy = advisor.get_execution_policy(
            operation_kind="MODIFY_SYMBOL", graph_context=gc
        )
        assert policy["mode"] == "guarded"
        assert policy["requires_anchor_read"] is True
        assert policy["block_structural_edit"] is False
