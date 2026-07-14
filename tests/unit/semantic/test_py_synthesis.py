"""Tests for flow synthesis and code template system (Phase F.3).

Covers:
- primitive_code_templates: each template generates valid Python
- synthesis_planner: plan_synthesis with different action types
- flow_synthesizer: synthesize_function / synthesize_from_ir
- output_comparator: compare_candidates decision logic
- primitive_reconstructor: reconstruct_from_ir basic behavior
"""
from __future__ import annotations

import ast

import pytest

from external_llm.editor.semantic.flow_synthesizer import (
    synthesize_from_ir,
    synthesize_function,
)
from external_llm.editor.semantic.output_comparator import (
    ComparisonResult,
    compare_candidates,
)
from external_llm.editor.semantic.primitive_code_templates import (
    _TEMPLATE_BUILDERS,
    CodeTemplate,
    get_code_template,
)
from external_llm.editor.semantic.primitive_models import (
    PrimitiveIR,
    PrimitiveMatch,
    PrimitiveSequence,
    ReconstructionCandidate,
)
from external_llm.editor.semantic.primitive_reconstructor import reconstruct_from_ir
from external_llm.editor.semantic.synthesis_planner import (
    _DEFAULT_SEQUENCES,
    SynthesisPlan,
    plan_synthesis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sequence(
    action_name: str,
    action_type: str,
    entity: str = "Item",
    present: list[str] | None = None,
    missing: list[str] | None = None,
    file_path: str = "",
) -> PrimitiveSequence:
    return PrimitiveSequence(
        action_name=action_name,
        action_type=action_type,
        entity=entity,
        file_path=file_path,
        present=[PrimitiveMatch(primitive=p, present=True, confidence=1.0) for p in (present or [])],
        missing=[PrimitiveMatch(primitive=m, present=False, confidence=0.0) for m in (missing or [])],
    )


def _make_ir(
    sequences: list[PrimitiveSequence] | None = None,
    entities: list[str] | None = None,
) -> PrimitiveIR:
    return PrimitiveIR(
        sequences=sequences or [],
        entities=entities or [],
    )


def _assert_valid_python(code: str, label: str = "") -> None:
    """Assert that code is valid Python via ast.parse."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Invalid Python{f' ({label})' if label else ''}: {e}\n---\n{code[:500]}")


# ===========================================================================
# 1. primitive_code_templates
# ===========================================================================


class TestCodeTemplates:
    """Each template builder produces a CodeTemplate; body is valid Python when wrapped."""

    # All 12 primitives must have a builder registered
    EXPECTED_PRIMITIVES = [
        "lookup", "validate", "branch_on_failure", "authorize",
        "create_entity", "input_bind", "update_entity", "delete_entity",
        "persist_state", "list_or_query", "produce_output", "delegate_action",
    ]

    def test_all_primitives_registered(self):
        for prim in self.EXPECTED_PRIMITIVES:
            assert prim in _TEMPLATE_BUILDERS, f"{prim} not in _TEMPLATE_BUILDERS"

    @pytest.mark.parametrize("prim", EXPECTED_PRIMITIVES)
    def test_template_returns_code_template(self, prim):
        ctx = {"entity": "User", "entity_lower": "user", "params": ["name", "email"]}
        tmpl = get_code_template(prim, ctx)
        assert isinstance(tmpl, CodeTemplate)
        assert tmpl.primitive == prim

    @pytest.mark.parametrize("prim", EXPECTED_PRIMITIVES)
    def test_template_body_is_valid_python(self, prim):
        ctx = {
            "entity": "User",
            "entity_lower": "user",
            "params": ["name", "email"],
            "identifier": "name",
            "password_param": "password",
            "action_type": "create",
            "_sequence": [],
        }
        tmpl = get_code_template(prim, ctx)
        if tmpl.body:
            # Wrap in a function so indented code is valid
            wrapped = "def _test():\n" + "\n".join(
                f"    {line}" if line.strip() else "" for line in tmpl.body.splitlines()
            )
            _assert_valid_python(wrapped, label=prim)

    def test_unknown_primitive_returns_todo(self):
        tmpl = get_code_template("nonexistent_primitive", {})
        assert tmpl.primitive == "nonexistent_primitive"
        assert "TODO" in tmpl.body

    def test_lookup_infers_identifier_from_params(self):
        tmpl = get_code_template("lookup", {"entity": "User", "entity_lower": "user", "params": ["user_id", "name"]})
        assert "user_id" in tmpl.body
        assert "user_id" in tmpl.variables_consumed

    def test_lookup_uses_first_param_as_fallback(self):
        tmpl = get_code_template("lookup", {"entity": "Item", "entity_lower": "item", "params": ["some_val"]})
        assert "some_val" in tmpl.body

    def test_validate_login_action(self):
        tmpl = get_code_template("validate", {"entity_lower": "user", "password_param": "pw", "action_type": "login"})
        assert "verify_password" in tmpl.body
        assert "pw" in tmpl.body

    def test_validate_non_login_action(self):
        tmpl = get_code_template("validate", {"entity_lower": "item", "action_type": "create"})
        # Template is framework-neutral: raises ValueError, not domain-specific HTTPException(400)
        assert "item" in tmpl.body
        assert "ValueError" in tmpl.body

    def test_branch_on_failure_merged_when_validate_in_sequence(self):
        tmpl = get_code_template("branch_on_failure", {"_sequence": ["validate", "branch_on_failure"]})
        assert tmpl.body == ""  # No-op when validate already handles branching

    def test_branch_on_failure_standalone(self):
        tmpl = get_code_template("branch_on_failure", {"_sequence": [], "entity_lower": "result"})
        assert "result" in tmpl.body

    def test_create_entity_with_params(self):
        tmpl = get_code_template("create_entity", {"entity": "Post", "entity_lower": "post", "params": ["title", "body"]})
        assert "Post(" in tmpl.body
        assert "title=title" in tmpl.body

    def test_create_entity_no_params(self):
        tmpl = get_code_template("create_entity", {"entity": "Post", "entity_lower": "post", "params": []})
        assert "Post()" in tmpl.body

    def test_persist_state_sqlalchemy(self):
        tmpl = get_code_template("persist_state", {"entity_lower": "user", "persist_style": "sqlalchemy", "action_type": "create"})
        assert "db.add" in tmpl.body
        assert "db.commit" in tmpl.body

    def test_persist_state_delete(self):
        tmpl = get_code_template("persist_state", {"entity_lower": "user", "persist_style": "sqlalchemy", "action_type": "delete"})
        assert "db.commit" in tmpl.body

    def test_persist_state_memory_fallback(self):
        tmpl = get_code_template("persist_state", {"entity_lower": "item", "persist_style": "memory", "action_type": "create"})
        assert "append" in tmpl.body

    def test_produce_output_login(self):
        tmpl = get_code_template("produce_output", {"action_type": "login"})
        assert "access_token" in tmpl.body

    def test_produce_output_delete(self):
        tmpl = get_code_template("produce_output", {"action_type": "delete", "entity_lower": "item"})
        assert "deleted" in tmpl.body

    def test_produce_output_default_create(self):
        tmpl = get_code_template("produce_output", {"action_type": "create", "entity_lower": "item"})
        assert "created" in tmpl.body

    def test_input_bind_with_params(self):
        tmpl = get_code_template("input_bind", {"params": ["name", "email"]})
        assert "data" in tmpl.body
        assert "name" in tmpl.body

    def test_input_bind_no_params(self):
        tmpl = get_code_template("input_bind", {"params": []})
        assert tmpl.body == ""

    def test_delegate_action_empty(self):
        tmpl = get_code_template("delegate_action", {})
        assert tmpl.body == ""


# ===========================================================================
# 2. synthesis_planner
# ===========================================================================


class TestSynthesisPlanner:
    """plan_synthesis builds SynthesisPlan from action info."""

    def test_default_sequences_has_expected_keys(self):
        expected = {"login", "signup", "register", "create", "send", "upload", "get", "list", "update", "delete"}
        assert expected == set(_DEFAULT_SEQUENCES.keys())

    @pytest.mark.parametrize("action_type", list(_DEFAULT_SEQUENCES.keys()))
    def test_plan_uses_default_sequence(self, action_type):
        plan = plan_synthesis(action_name=f"do_{action_type}", action_type=action_type, entity="Item")
        assert plan.sequence == _DEFAULT_SEQUENCES[action_type]
        assert plan.source == "default"

    def test_plan_unknown_action_falls_back_to_produce_output(self):
        plan = plan_synthesis(action_name="custom_action", action_type="unknown_type")
        assert plan.sequence == ["produce_output"]
        assert plan.source == "default"

    def test_plan_with_missing_primitives(self):
        plan = plan_synthesis(
            action_name="login",
            action_type="login",
            missing_primitives=["validate", "authorize"],
        )
        assert plan.sequence == ["validate", "authorize"]
        assert plan.source == "missing"

    def test_plan_preserves_params(self):
        plan = plan_synthesis(
            action_name="create_user",
            action_type="create",
            entity="User",
            params=["name", "email"],
        )
        assert plan.params == ["name", "email"]
        assert plan.entity == "User"

    def test_plan_decorator(self):
        plan = plan_synthesis(
            action_name="login",
            action_type="login",
            decorator='@router.post("/login")',
        )
        assert plan.decorator == '@router.post("/login")'

    def test_plan_summary(self):
        plan = plan_synthesis(action_name="get_user", action_type="get", entity="User")
        s = plan.summary()
        assert s["action"] == "get_user"
        assert s["type"] == "get"
        assert s["entity"] == "User"
        assert s["source"] == "default"
        assert isinstance(s["sequence"], list)


# ===========================================================================
# 3. flow_synthesizer: synthesize_function
# ===========================================================================


class TestSynthesizeFunction:
    """synthesize_function generates valid Python from a SynthesisPlan."""

    def test_create_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="create_user",
            action_type="create",
            entity="User",
            sequence=["input_bind", "create_entity", "persist_state", "produce_output"],
            params=["name", "email"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "create_user")
        assert "def create_user(" in code
        assert "name" in code

    def test_login_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="login",
            action_type="login",
            entity="User",
            sequence=["lookup", "validate", "branch_on_failure", "authorize", "produce_output"],
            params=["username", "password"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "login")
        assert "def login(" in code
        assert "access_token" in code

    def test_get_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="get_item",
            action_type="get",
            entity="Item",
            sequence=["lookup", "produce_output"],
            params=["item_id"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "get_item")
        assert "def get_item(" in code

    def test_delete_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="delete_post",
            action_type="delete",
            entity="Post",
            sequence=["lookup", "delete_entity", "persist_state", "produce_output"],
            params=["post_id"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "delete_post")

    def test_list_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="list_items",
            action_type="list",
            entity="Item",
            sequence=["list_or_query"],
            params=[],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "list_items")

    def test_update_action_generates_valid_python(self):
        plan = SynthesisPlan(
            action_name="update_item",
            action_type="update",
            entity="Item",
            sequence=["lookup", "input_bind", "update_entity", "persist_state", "produce_output"],
            params=["item_id", "title"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        _assert_valid_python(code, "update_item")

    def test_empty_sequence_returns_none(self):
        plan = SynthesisPlan(
            action_name="noop",
            action_type="unknown",
            entity="X",
            sequence=[],
            source="fallback",
        )
        code = synthesize_function(plan)
        assert code is None

    def test_only_delegate_returns_none(self):
        """delegate_action produces no body, so the function has no body lines."""
        plan = SynthesisPlan(
            action_name="delegate",
            action_type="delegate",
            entity="X",
            sequence=["delegate_action"],
            source="fallback",
        )
        code = synthesize_function(plan)
        assert code is None

    def test_decorator_is_included(self):
        plan = SynthesisPlan(
            action_name="login",
            action_type="login",
            entity="User",
            sequence=["lookup", "produce_output"],
            params=["username"],
            source="default",
            decorator='@router.post("/login")',
        )
        code = synthesize_function(plan)
        assert code is not None
        assert '@router.post("/login")' in code
        _assert_valid_python(code, "login with decorator")

    def test_contains_fastapi_import(self):
        plan = SynthesisPlan(
            action_name="get_user",
            action_type="get",
            entity="User",
            sequence=["lookup", "produce_output"],
            params=["user_id"],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        assert "from fastapi import" in code

    def test_docstring_included(self):
        plan = SynthesisPlan(
            action_name="create_item",
            action_type="create",
            entity="Item",
            sequence=["create_entity", "produce_output"],
            params=[],
            source="default",
        )
        code = synthesize_function(plan)
        assert code is not None
        assert "Auto-generated" in code
        assert "Item" in code


# ===========================================================================
# 4. flow_synthesizer: synthesize_from_ir
# ===========================================================================


class TestSynthesizeFromIR:
    """synthesize_from_ir generates functions for low-coverage sequences."""

    def test_generates_function_for_low_coverage_sequence(self):
        seq = _make_sequence(
            action_name="create_user",
            action_type="create",
            entity="User",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        ir = _make_ir(sequences=[seq], entities=["User"])
        results = synthesize_from_ir(ir)
        assert "create_user" in results
        _assert_valid_python(results["create_user"], "create_user from IR")

    def test_skips_high_coverage_sequence(self):
        seq = _make_sequence(
            action_name="get_item",
            action_type="get",
            entity="Item",
            present=["lookup", "produce_output"],  # 100% coverage
            missing=[],
        )
        ir = _make_ir(sequences=[seq])
        results = synthesize_from_ir(ir)
        assert "get_item" not in results

    def test_skips_at_coverage_threshold(self):
        """A sequence with coverage >= 0.9 should be skipped."""
        seq = _make_sequence(
            action_name="get_item",
            action_type="get",
            entity="Item",
            present=["lookup", "produce_output", "validate", "branch_on_failure",
                      "authorize", "create_entity", "persist_state", "list_or_query", "input_bind"],
            missing=["delegate_action"],  # 9/10 = 0.9 coverage
        )
        ir = _make_ir(sequences=[seq])
        results = synthesize_from_ir(ir)
        assert "get_item" not in results

    def test_multiple_sequences(self):
        seq1 = _make_sequence(
            action_name="create_post",
            action_type="create",
            entity="Post",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        seq2 = _make_sequence(
            action_name="login",
            action_type="login",
            entity="User",
            present=["lookup"],
            missing=["validate", "authorize", "produce_output"],
        )
        ir = _make_ir(sequences=[seq1, seq2], entities=["Post", "User"])
        results = synthesize_from_ir(ir)
        assert "create_post" in results
        assert "login" in results
        for name, code in results.items():
            _assert_valid_python(code, name)

    def test_empty_ir_returns_empty(self):
        ir = _make_ir()
        results = synthesize_from_ir(ir)
        assert results == {}

    def test_scope_filters_out_of_scope_sequences(self):
        """Sequences whose action_name is not in the scope set are skipped."""
        seq_in = _make_sequence(
            action_name="create_post",
            action_type="create",
            entity="Post",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        seq_out = _make_sequence(
            action_name="unrelated_helper",
            action_type="create",
            entity="Helper",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        ir = _make_ir(sequences=[seq_in, seq_out], entities=["Post", "Helper"])
        results = synthesize_from_ir(ir, scope={"create_post"})
        assert "create_post" in results
        assert "unrelated_helper" not in results

    def test_scope_none_preserves_legacy_behaviour(self):
        seq = _make_sequence(
            action_name="create_user",
            action_type="create",
            entity="User",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        ir = _make_ir(sequences=[seq], entities=["User"])
        results = synthesize_from_ir(ir, scope=None)
        assert "create_user" in results

    def test_scope_empty_skips_everything(self):
        seq = _make_sequence(
            action_name="create_user",
            action_type="create",
            entity="User",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        ir = _make_ir(sequences=[seq], entities=["User"])
        results = synthesize_from_ir(ir, scope=set())
        assert results == {}


# ===========================================================================
# 5. output_comparator: compare_candidates
# ===========================================================================


class TestOutputComparator:
    """compare_candidates decides between raw and reconstructed."""

    def test_no_applied_primitives_chooses_raw(self):
        ir = _make_ir(sequences=[_make_sequence("a", "create", present=["input_bind"], missing=["persist_state"])])
        candidate = ReconstructionCandidate()
        result = compare_candidates(ir, candidate)
        assert result.chosen == "raw"
        assert "no reconstruction" in result.reason

    def test_large_improvement_chooses_reconstructed(self):
        ir = _make_ir(sequences=[
            _make_sequence("a", "create", present=["input_bind"], missing=["create_entity", "persist_state", "produce_output"]),
        ])
        candidate = ReconstructionCandidate(
            applied_primitives=["create_entity", "persist_state", "produce_output"],
            primitive_coverage_estimate=1.0,
        )
        result = compare_candidates(ir, candidate)
        assert result.chosen == "reconstructed"
        assert result.improvement >= 0.1

    def test_multiple_primitives_filled_chooses_reconstructed(self):
        """Even with small improvement, filling 2+ primitives triggers reconstructed."""
        ir = _make_ir(sequences=[
            _make_sequence("a", "create", present=["input_bind", "create_entity", "persist_state"],
                           missing=["produce_output", "validate"]),
        ])
        # coverage: 3/5 = 0.6, candidate adds 2 primitives
        candidate = ReconstructionCandidate(
            applied_primitives=["produce_output", "validate"],
            primitive_coverage_estimate=0.65,  # small improvement from 0.6
        )
        result = compare_candidates(ir, candidate)
        assert result.chosen == "reconstructed"

    def test_small_improvement_single_primitive_chooses_raw(self):
        ir = _make_ir(sequences=[
            _make_sequence("a", "create",
                           present=["input_bind", "create_entity", "persist_state"],
                           missing=["produce_output"]),
        ])
        # coverage: 3/4 = 0.75, candidate bumps to 0.8 (+0.05)
        candidate = ReconstructionCandidate(
            applied_primitives=["produce_output"],
            primitive_coverage_estimate=0.8,
        )
        result = compare_candidates(ir, candidate)
        assert result.chosen == "raw"
        assert "too small" in result.reason

    def test_comparison_result_to_dict(self):
        result = ComparisonResult(
            chosen="reconstructed",
            reason="test",
            raw_coverage=0.5,
            reconstructed_coverage=0.9,
            improvement=0.4,
        )
        d = result.to_dict()
        assert d["chosen"] == "reconstructed"
        assert d["raw_coverage"] == 0.5
        assert d["improvement"] == 0.4

    def test_contract_estimate_calculation(self):
        ir = _make_ir(sequences=[
            _make_sequence("a", "create", present=[], missing=["create_entity", "persist_state"]),
        ])
        candidate = ReconstructionCandidate(
            applied_primitives=["create_entity", "persist_state"],
            primitive_coverage_estimate=0.8,
        )
        result = compare_candidates(ir, candidate, contract_score=0.5)
        # 0.5 + 2 * 0.05 = 0.6
        assert abs(result.reconstructed_contract_estimate - 0.6) < 0.01

    def test_contract_estimate_capped_at_1(self):
        ir = _make_ir(sequences=[
            _make_sequence("a", "create", present=[], missing=["a", "b", "c"]),
        ])
        candidate = ReconstructionCandidate(
            applied_primitives=["a", "b", "c"],
            primitive_coverage_estimate=0.8,
        )
        result = compare_candidates(ir, candidate, contract_score=0.95)
        assert result.reconstructed_contract_estimate <= 1.0


# ===========================================================================
# 6. primitive_reconstructor: reconstruct_from_ir
# ===========================================================================


class TestPrimitiveReconstructor:
    """reconstruct_from_ir dispatches repair for missing primitives."""

    def test_high_coverage_skips_reconstruction(self):
        """IR with coverage >= 0.95 returns immediately with confidence=1.0."""
        seq = _make_sequence(
            "get_user", "get",
            entity="User",
            present=["lookup", "produce_output"],
            missing=[],
        )
        ir = _make_ir(sequences=[seq])
        # Use a dummy repo_root -- repair functions won't be called
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        assert result.confidence == 1.0
        assert "no reconstruction needed" in result.notes[0]

    def test_returns_reconstruction_candidate(self):
        """Even when repairs fail, the result is a valid ReconstructionCandidate."""
        seq = _make_sequence(
            "create_item", "create",
            entity="Item",
            present=["input_bind"],
            missing=["create_entity", "persist_state", "produce_output"],
        )
        ir = _make_ir(sequences=[seq], entities=["Item"])
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        assert isinstance(result, ReconstructionCandidate)
        # Coverage estimate should be computed even if repairs fail
        assert result.primitive_coverage_estimate >= 0.0

    def test_authorize_and_input_bind_return_none(self):
        """authorize and input_bind handlers return None (placeholder)."""
        seq = _make_sequence(
            "some_action", "create",
            entity="Item",
            present=["create_entity"],
            missing=["authorize", "input_bind"],
        )
        ir = _make_ir(sequences=[seq], entities=["Item"])
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        assert isinstance(result, ReconstructionCandidate)
        # These primitives should NOT appear in applied_primitives
        assert "authorize" not in result.applied_primitives
        assert "input_bind" not in result.applied_primitives

    def test_dependency_check_persist_requires_create(self):
        """persist_state depends on create_entity; without it, persist is skipped."""
        seq = _make_sequence(
            "save_item", "create",
            entity="Item",
            present=[],
            missing=["persist_state"],  # create_entity not present or being filled
        )
        ir = _make_ir(sequences=[seq], entities=["Item"])
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        # persist_state should be skipped due to unsatisfied dependency
        assert "persist_state" not in result.applied_primitives

    def test_dedup_across_sequences(self):
        """Same primitive across different sequences is applied once globally."""
        seq1 = _make_sequence(
            "action_a", "create", entity="Item",
            present=["create_entity"], missing=["produce_output"],
        )
        seq2 = _make_sequence(
            "action_b", "create", entity="Item",
            present=["create_entity"], missing=["produce_output"],
        )
        ir = _make_ir(sequences=[seq1, seq2], entities=["Item"])
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        # applied_primitives should have at most one "produce_output"
        assert result.applied_primitives.count("produce_output") <= 1

    def test_empty_ir_returns_high_confidence(self):
        """Empty IR has overall_coverage=1.0, so reconstruction is skipped."""
        ir = _make_ir()
        result = reconstruct_from_ir(ir, repo_root="/tmp/fake")
        assert result.confidence == 1.0


# ===========================================================================
# Integration-like: plan_synthesis -> synthesize_function round-trip
# ===========================================================================


class TestPlanToSynthesisRoundtrip:
    """End-to-end: plan_synthesis -> synthesize_function -> valid Python."""

    @pytest.mark.parametrize("action_type", list(_DEFAULT_SEQUENCES.keys()))
    def test_all_default_action_types_produce_valid_code(self, action_type):
        plan = plan_synthesis(
            action_name=f"test_{action_type}",
            action_type=action_type,
            entity="Thing",
            params=["name", "value"],
        )
        code = synthesize_function(plan)
        # "list" only uses list_or_query which has a return, so it should produce code.
        # All default sequences should produce at least some body.
        if code is not None:
            _assert_valid_python(code, f"roundtrip:{action_type}")
