"""
Tests for operation_models.
"""

import pytest

from external_llm.agent.operation_models import (
    AssertionTier,
    ExecutorState,
    FILE_WRITING_KINDS,
    FailureClass,
    GroundingSummary,
    IntentAssertion,
    IntentAssertionKind,
    Operation,
    OperationKind,
    OperationPlan,
    READ_ONLY_KINDS,
    PlanMode,
    PlanPolicy,
    SemanticFitAction,
    SemanticFitVerdict,
    SkipReason,
    _require_symbol_for_kinds,
    _safe_get_metadata,
    _shared_logic_helper,
    infer_tier,
    is_dependency_blocking_sentinel,
    is_dependency_related_skip,
    is_propagation_kind,
    normalize_failure_class,
    op_intent_is_clearly_additive,
    op_intent_is_clearly_removal,
    parse_skip_reason,
)


def test_operation_creation():
    """Test basic operation creation."""
    op = Operation(
        id="op1",
        kind=OperationKind.READ_SYMBOL,
        path="some/file.py",
        symbol="some_function",
        intent="Read the function definition",
    )
    assert op.id == "op1"
    assert op.kind == OperationKind.READ_SYMBOL
    assert op.path == "some/file.py"
    assert op.symbol == "some_function"
    assert op.intent == "Read the function definition"
    assert op.depends_on == []
    assert op.acceptance == []
    assert op.context_hints == {}
    assert op.metadata == {}


def test_operation_validation():
    """Test operation validation based on kind."""
    # read_symbol requires symbol (or path if symbol is None)
    with pytest.raises(ValueError, match="read_symbol: missing path"):
        Operation(id="op1", kind=OperationKind.READ_SYMBOL)

    # read_symbol requires symbol even when path is specified
    with pytest.raises(ValueError, match="read_symbol: missing symbol"):
        Operation(id="op1b", kind=OperationKind.READ_SYMBOL, path="x.py")

    # modify_symbol requires symbol
    with pytest.raises(ValueError, match="modify_symbol: missing symbol"):
        Operation(id="op2", kind=OperationKind.MODIFY_SYMBOL, path="x.py")

    # insert_after_symbol requires symbol
    with pytest.raises(ValueError, match="insert_after_symbol: missing symbol"):
        Operation(id="op3", kind=OperationKind.INSERT_AFTER_SYMBOL, path="x.py")

    # update_callers requires symbol
    with pytest.raises(ValueError, match="update_callers: missing symbol"):
        Operation(id="op4", kind=OperationKind.UPDATE_CALLERS, path="x.py")

    # Operations that don't require symbols should pass
    op5 = Operation(id="op5", kind=OperationKind.SUMMARIZE_ANALYSIS)
    assert op5.symbol is None


def test_operation_plan_creation():
    """Test operation plan creation."""
    ops = [
        Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="some/file.py", symbol="func1"),
        Operation(id="op2", kind=OperationKind.MODIFY_SYMBOL, path="some/file.py", symbol="func1"),
    ]
    plan = OperationPlan(operations=ops, mode=PlanMode.EDIT, description="Test plan")
    assert len(plan.operations) == 2
    assert plan.mode == PlanMode.EDIT
    assert plan.description == "Test plan"


def test_operation_plan_validation():
    """Test plan validation logic."""
    # Duplicate IDs
    ops = [
        Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="some/file.py", symbol="func1"),
        Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="some/file.py", symbol="func1"),  # same ID
    ]
    plan = OperationPlan(operations=ops)
    warnings = plan.validate()
    assert "Duplicate operation IDs found" in warnings

    # Unknown dependency
    ops2 = [
        Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="some/file.py", symbol="func1"),
        Operation(id="op2", kind=OperationKind.MODIFY_SYMBOL, path="some/file.py", symbol="func1",
                  depends_on=["op3"]),  # op3 doesn't exist
    ]
    plan2 = OperationPlan(operations=ops2)
    warnings2 = plan2.validate()
    assert any("depends on unknown operation" in w for w in warnings2)

    # summarize_analysis position is guaranteed by depends_on, not validate()
    # (43c4d8f2: removed list order dependency — deleted summarize_analysis position check)
    ops3 = [
        Operation(id="op1", kind=OperationKind.SUMMARIZE_ANALYSIS),
        Operation(id="op2", kind=OperationKind.READ_SYMBOL, path="some/file.py", symbol="func1"),
    ]
    plan3 = OperationPlan(operations=ops3)
    warnings3 = plan3.validate()
    assert isinstance(warnings3, list)  # validate() returns list without position check


def test_executor_state():
    """Test executor state operations."""
    state = ExecutorState()
    assert state.current_op is None
    assert state.force_finish is False

    # Mark completed
    state.mark_completed("op1", {"result": "success"})
    assert "op1" in state.completed_ops
    assert state.completed_ops["op1"]["result"] == "success"
    assert state.current_op is None

    # Mark failed
    state.mark_failed("op2", "some error")
    assert "op2" in state.failed_ops
    assert state.failed_ops["op2"] == "some error"
    assert state.current_op is None

    # Visited symbols
    state.add_visited_symbol("func1", "file.py")
    assert state.has_visited_symbol("func1")
    assert state.has_visited_symbol("func1", "file.py")
    assert not state.has_visited_symbol("func2")
    assert not state.has_visited_symbol("func1", "other.py")

    # Read files
    state.add_read_file("file.py", "content")
    assert state.get_read_file("file.py") == "content"
    assert state.get_read_file("nonexistent") is None


# ══════════════════════════════════════════════════════════════════════════
# infer_tier
# ══════════════════════════════════════════════════════════════════════════

def test_infer_tier_existence():
    assert infer_tier(IntentAssertionKind.SYMBOL_EXISTS) == AssertionTier.EXISTENCE
    assert infer_tier(IntentAssertionKind.SYMBOL_NOT_REMOVED) == AssertionTier.EXISTENCE


def test_infer_tier_change():
    assert infer_tier(IntentAssertionKind.SYMBOL_HAS_PARAM) == AssertionTier.CHANGE
    assert infer_tier(IntentAssertionKind.IMPORT_EXISTS) == AssertionTier.CHANGE
    assert infer_tier(IntentAssertionKind.GUARD_IN_SCOPE) == AssertionTier.CHANGE
    assert infer_tier(IntentAssertionKind.ENUM_MEMBER_EXISTS) == AssertionTier.CHANGE
    assert infer_tier(IntentAssertionKind.SYMBOL_CONTAINS) == AssertionTier.CHANGE


def test_infer_tier_state():
    assert infer_tier(IntentAssertionKind.SYMBOL_CALLS) == AssertionTier.STATE
    assert infer_tier(IntentAssertionKind.SYMBOL_REFERENCES) == AssertionTier.STATE


def test_infer_tier_from_string():
    """infer_tier also accepts a raw string (not an enum member)."""
    assert infer_tier("symbol_exists") == AssertionTier.EXISTENCE
    assert infer_tier("symbol_calls") == AssertionTier.STATE


# ══════════════════════════════════════════════════════════════════════════
# IntentAssertion
# ══════════════════════════════════════════════════════════════════════════

def test_intent_assertion_without_tier():
    """When tier is None, __post_init__ calls infer_tier."""
    a = IntentAssertion(kind=IntentAssertionKind.SYMBOL_EXISTS, target_file="f.py")
    assert a.tier == AssertionTier.EXISTENCE


def test_intent_assertion_with_explicit_tier():
    a = IntentAssertion(kind=IntentAssertionKind.SYMBOL_CALLS, target_file="f.py",
                        tier=AssertionTier.CHANGE)
    assert a.tier == AssertionTier.CHANGE


# ══════════════════════════════════════════════════════════════════════════
# SemanticFitVerdict
# ══════════════════════════════════════════════════════════════════════════

def test_semantic_fit_verdict_to_dict():
    v = SemanticFitVerdict(
        semantic_fit="strong",
        action=SemanticFitAction.ACCEPT,
        confidence=0.95,
        reason="Matches user intent",
        reexplore_guidance={"notes": "none"},
    )
    d = v.to_dict()
    assert d["semantic_fit"] == "strong"
    assert d["action"] == "ACCEPT"
    assert d["confidence"] == 0.95
    assert d["reason"] == "Matches user intent"


# ══════════════════════════════════════════════════════════════════════════
# op_intent_is_clearly_additive / _shared_logic_helper / _safe_get_metadata
# ══════════════════════════════════════════════════════════════════════════

def test_op_intent_additive_none():
    assert op_intent_is_clearly_additive(None) is False


def test_op_intent_additive_by_kind():
    op = Operation(id="op1", kind=OperationKind.INSERT_AFTER_SYMBOL, path="x.py", symbol="f")
    assert op_intent_is_clearly_additive(op) is True


def test_op_intent_additive_removal_kind_false():
    op = Operation(id="op1", kind=OperationKind.DELETE_FILE, path="x.py")
    assert op_intent_is_clearly_additive(op) is False


def test_op_intent_additive_by_hint():
    op = Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="x.py", symbol="f",
                   metadata={"action_hint": "append"})
    assert op_intent_is_clearly_additive(op) is True


def test_op_intent_additive_default_false():
    op = Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="x.py", symbol="f")
    assert op_intent_is_clearly_additive(op) is False


def test_shared_logic_helper_none():
    assert _shared_logic_helper(None, set(), set(), set()) is False


def test_safe_get_metadata_none():
    assert _safe_get_metadata(None) == ""


def test_safe_get_metadata_empty_key():
    op = Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="x.py", symbol="f",
                   metadata={"action_hint": "add"})
    md = _safe_get_metadata(op, key="")
    assert md == {"action_hint": "add"}


def test_safe_get_metadata_missing_key():
    op = Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="x.py", symbol="f")
    assert _safe_get_metadata(op, "nonexistent", "fallback") == "fallback"


# ══════════════════════════════════════════════════════════════════════════
# _require_symbol_for_kinds
# ══════════════════════════════════════════════════════════════════════════

def test_require_symbol_for_kinds_raises():
    class FakeOp:
        kind = OperationKind.MODIFY_SYMBOL
        symbol = None
    with pytest.raises(ValueError, match="Symbol is required"):
        _require_symbol_for_kinds(FakeOp(), {OperationKind.MODIFY_SYMBOL})


def test_require_symbol_for_kinds_ok():
    class FakeOp:
        kind = OperationKind.MODIFY_SYMBOL
        symbol = "func1"
    _require_symbol_for_kinds(FakeOp(), {OperationKind.MODIFY_SYMBOL})  # no raise


# ══════════════════════════════════════════════════════════════════════════
# op_intent_is_clearly_removal
# ══════════════════════════════════════════════════════════════════════════

def test_op_intent_removal_by_kind():
    op = Operation(id="op1", kind=OperationKind.DELETE_FILE, path="x.py")
    assert op_intent_is_clearly_removal(op) is True


def test_op_intent_removal_by_action_class():
    op = Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="x.py", symbol="f",
                   action_class="delete")
    assert op_intent_is_clearly_removal(op) is True


def test_op_intent_removal_by_intent():
    op = Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="x.py", symbol="f",
                   intent="Remove unused import")
    assert op_intent_is_clearly_removal(op) is True


def test_op_intent_removal_by_semantic_family():
    op = Operation(id="op1", kind=OperationKind.MODIFY_SYMBOL, path="x.py", symbol="f",
                   metadata={"semantic_change_family": "dead_code_removal"})
    assert op_intent_is_clearly_removal(op) is True


def test_op_intent_removal_false():
    op = Operation(id="op1", kind=OperationKind.READ_SYMBOL, path="x.py", symbol="f")
    assert op_intent_is_clearly_removal(op) is False


# ══════════════════════════════════════════════════════════════════════════
# is_propagation_kind
# ══════════════════════════════════════════════════════════════════════════

def test_is_propagation_kind():
    assert is_propagation_kind(OperationKind.UPDATE_CALLERS) is True
    assert is_propagation_kind(OperationKind.READ_SYMBOL) is False




# ══════════════════════════════════════════════════════════════════════════
# normalize_failure_class
# ══════════════════════════════════════════════════════════════════════════

def test_normalize_failure_class_none():
    assert normalize_failure_class(None) == FailureClass.UNKNOWN


def test_normalize_failure_class_empty():
    assert normalize_failure_class("") == FailureClass.UNKNOWN


def test_normalize_failure_class_exact():
    assert normalize_failure_class("syntax_error") == FailureClass.SYNTAX_ERROR


def test_normalize_failure_class_unknown():
    assert normalize_failure_class("bogus") == FailureClass.UNKNOWN


# ══════════════════════════════════════════════════════════════════════════
# parse_skip_reason / is_dependency_related_skip / is_dependency_blocking_sentinel
# ══════════════════════════════════════════════════════════════════════════

def test_parse_skip_reason_empty():
    assert parse_skip_reason("") is None


def test_parse_skip_reason_unknown():
    assert parse_skip_reason("bogus_reason") is None


def test_parse_skip_reason_known():
    assert parse_skip_reason("unschedulable") == SkipReason.UNSCHEDULABLE


def test_parse_skip_reason_with_detail():
    assert parse_skip_reason("blocked_by_failed_dependency:op1") == SkipReason.BLOCKED_BY_FAILED_DEPENDENCY


def test_is_dependency_related_skip_true():
    assert is_dependency_related_skip("unschedulable") is True


def test_is_dependency_related_skip_false():
    assert is_dependency_related_skip("gate_already_satisfied") is False


def test_is_dependency_related_skip_unknown():
    assert is_dependency_related_skip("bogus_reason") is False


def test_is_dependency_blocking_sentinel_true():
    assert is_dependency_blocking_sentinel("dependency_blocked_operations_present") is True


def test_is_dependency_blocking_sentinel_false():
    assert is_dependency_blocking_sentinel("") is False
    assert is_dependency_blocking_sentinel("some_other_reason") is False


# ══════════════════════════════════════════════════════════════════════════
# GroundingSummary
# ══════════════════════════════════════════════════════════════════════════

def test_grounding_summary_from_spec_meta():
    gs = GroundingSummary.from_spec_meta(
        meta={"grounding_confidence": "0.85", "exploration_confidence": "0.7"},
        spec=None,
    )
    assert gs.grounding_confidence == 0.85
    assert gs.intent_files == []


def test_grounding_summary_to_dict():
    gs = GroundingSummary(grounding_confidence=0.9, source="resolver")
    d = gs.to_dict()
    assert d["grounding_confidence"] == 0.9
    assert d["source"] == "resolver"


# ══════════════════════════════════════════════════════════════════════════
# PlanPolicy
# ══════════════════════════════════════════════════════════════════════════

def test_plan_policy_to_dict():
    pp = PlanPolicy(kind="analysis_only", requires_code_changes=False, confidence=0.3, why="testing")
    d = pp.to_dict()
    assert d["kind"] == "analysis_only"
    assert d["requires_code_changes"] is False
    assert d["confidence"] == 0.3
    assert d["why"] == "testing"


# ══════════════════════════════════════════════════════════════════════════
# Operation.validate — additional paths
# ══════════════════════════════════════════════════════════════════════════

def test_operation_validate_anchor_edit():
    """ANCHOR_EDIT without path fails at post_init."""
    with pytest.raises(ValueError, match="anchor_edit: missing path"):
        Operation(id="op1", kind=OperationKind.ANCHOR_EDIT)


def test_operation_validate_anchor_edit_missing_pattern():
    """Missing anchor_pattern/lineno fails at post_init."""
    with pytest.raises(ValueError, match="anchor_edit requires anchor_pattern"):
        Operation(id="op1", kind=OperationKind.ANCHOR_EDIT, path="x.py")


def test_operation_validate_anchor_edit_invalid_mode():
    """Invalid edit_mode fails at post_init."""
    with pytest.raises(ValueError, match="invalid edit_mode"):
        Operation(id="op1", kind=OperationKind.ANCHOR_EDIT, path="x.py",
                  anchor_pattern="foo", edit_mode="invalid")


def test_operation_validate_anchor_edit_valid():
    Operation(id="op1", kind=OperationKind.ANCHOR_EDIT, path="x.py",
                    anchor_pattern="foo", edit_mode="replace_line").validate()


def test_operation_validate_insert_after_line():
    """INSERT_AFTER_LINE without path fails at post_init."""
    with pytest.raises(ValueError, match="insert_after_line: missing path"):
        Operation(id="op1", kind=OperationKind.INSERT_AFTER_LINE)


def test_operation_validate_read_file_segment():
    """READ_FILE_SEGMENT without path fails at post_init."""
    with pytest.raises(ValueError, match="read_file_segment: missing path"):
        Operation(id="op1", kind=OperationKind.READ_FILE_SEGMENT)


def test_operation_validate_run_scanner():
    """RUN_SCANNER without scanner_name fails at post_init."""
    with pytest.raises(ValueError, match="run_scanner requires metadata"):
        Operation(id="op1", kind=OperationKind.RUN_SCANNER, path="x.py")


def test_operation_validate_run_scanner_with_metadata_paths():
    """metadata.paths can substitute for missing path."""
    err = Operation(id="op1", kind=OperationKind.RUN_SCANNER,
                    metadata={"paths": ["."], "scanner_name": "dead_block_scanner"}).validate()
    assert err is None


def test_operation_validate_run_scanner_with_scanner_name():
    err = Operation(id="op1", kind=OperationKind.RUN_SCANNER, path="x.py",
                    metadata={"scanner_name": "dead_block_scanner"}).validate()
    assert err is None


# ══════════════════════════════════════════════════════════════════════════
# SSOT frozenset contract — guards the str-Enum value-vs-name landmine
# (commit c10897a2).  OperationKind(str, Enum) members compare equal to their
# LOWERCASE value, never their uppercase NAME.  Lane validators/normalizers
# must test membership via these derived SSOT frozensets, not inline
# uppercase-NAME string tuples (which are always-False → dead code).
# ══════════════════════════════════════════════════════════════════════════

def test_read_only_kinds_includes_run_scanner():
    """RUN_SCANNER was added to OP_KIND_POLICY as read_only — the derived
    SSOT frozenset must auto-include it (guards the divergence sub-bug where
    hand-maintained uppercase tuples forgot RUN_SCANNER)."""
    assert OperationKind.RUN_SCANNER in READ_ONLY_KINDS
    assert OperationKind.READ_SYMBOL in READ_ONLY_KINDS


def test_str_enum_lowercase_value_matches_frozenset():
    """Production contract: a real OperationKind member AND its lowercase
    value string both match the SSOT frozenset, because str-Enum members
    hash/compare equal to their value."""
    assert OperationKind.READ_SYMBOL in READ_ONLY_KINDS
    assert "read_symbol" in READ_ONLY_KINDS


def test_uppercase_name_does_not_match_frozenset():
    """LANDMINE guard: an uppercase NAME string ('READ_SYMBOL') does NOT
    match the frozenset, because members compare equal to their lowercase
    VALUE, not their NAME.  This documents WHY comparing op.kind against
    uppercase-NAME tuples is a dead-code bug — the regression fixed in
    c10897a2."""
    assert "READ_SYMBOL" not in READ_ONLY_KINDS
    assert "MODIFY_SYMBOL" not in FILE_WRITING_KINDS


def test_read_only_kinds_disjoint_from_file_writing():
    """No kind may be simultaneously read-only and file-writing."""
    assert READ_ONLY_KINDS.isdisjoint(FILE_WRITING_KINDS)


def test_file_writing_kinds_covers_modify_create_delete():
    assert OperationKind.MODIFY_SYMBOL in FILE_WRITING_KINDS
    assert OperationKind.CREATE_FILE in FILE_WRITING_KINDS
    assert OperationKind.DELETE_FILE in FILE_WRITING_KINDS


def test_operation_validate_insert_after_symbol_with_eof_fallback():
    """INSERT_AFTER_SYMBOL with _symbol_hallucinated_eof_fallback does not require symbol."""
    op = Operation(id="op1", kind=OperationKind.INSERT_AFTER_SYMBOL, path="x.py",
                   metadata={"_symbol_hallucinated_eof_fallback": True})
    err = op.validate()
    assert err is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
