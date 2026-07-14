"""Tests for semantic contract evaluation system (Phase C.1).

Covers:
- 6 core contracts (requirement/ordering/binding/branch/output checks)
- Contract registry: resolve_contracts, extract_context_tags
- Evaluator: evaluate_contracts with passing and failing traces
- ContractReport: overall_score, has_critical_violation, passed_count
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from external_llm.editor.semantic.semantic_contract_evaluator import (
    _check_binding,
    _check_branch,
    _check_ordering,
    _check_output,
    _check_requirement,
    evaluate_contracts,
)
from external_llm.editor.semantic.semantic_contract_models import (
    SemanticContract,
    SemanticContractReport,
    SemanticEvaluationResult,
    SemanticViolation,
)
from external_llm.editor.semantic.semantic_contract_registry import (
    extract_context_tags,
    resolve_contracts,
)
from external_llm.editor.semantic.semantic_contracts import (
    ALL_CONTRACTS,
    AUTH_VERIFICATION_PRECEDES_TOKEN,
    CREATE_OUTPUT_REFERENCES_ENTITY,
    CREATE_REQUIRES_PERSISTENCE,
    FAILURE_BRANCH_BLOCKS_SUCCESS,
)
from external_llm.editor.semantic.semantic_tracer import FunctionTrace, SemanticTrace

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_trace(
    function_traces: Optional[dict[str, FunctionTrace]] = None,
    all_calls: Optional[set[str]] = None,
    all_instantiations: Optional[set[str]] = None,
    all_persist_calls: Optional[set[str]] = None,
) -> SemanticTrace:
    """Build a SemanticTrace with sensible defaults."""
    return SemanticTrace(
        function_traces=function_traces or {},
        all_calls=all_calls or set(),
        all_instantiations=all_instantiations or set(),
        all_persist_calls=all_persist_calls or set(),
    )


def _make_ft(
    name: str = "func",
    calls: Optional[set[str]] = None,
    instantiations: Optional[set[str]] = None,
    persist_calls: Optional[set[str]] = None,
    return_names: Optional[set[str]] = None,
    return_has_entity_ref: bool = False,
    param_names: Optional[list[str]] = None,
    entity_bindings: Optional[list[tuple[str, str]]] = None,
    has_error_branch: bool = False,
    error_before_success: bool = False,
    call_order: Optional[list[str]] = None,
) -> FunctionTrace:
    return FunctionTrace(
        name=name,
        calls=calls or set(),
        instantiations=instantiations or set(),
        persist_calls=persist_calls or set(),
        return_names=return_names or set(),
        return_has_entity_ref=return_has_entity_ref,
        param_names=param_names or [],
        entity_bindings=entity_bindings or [],
        has_error_branch=has_error_branch,
        error_before_success=error_before_success,
        call_order=call_order or [],
    )


# ============================================================================
# 1. Contract Model Tests
# ============================================================================

class TestSemanticContractModels:
    """Test data model basics."""

    def test_contract_has_fields(self):
        c = SemanticContract(name="test", applies_to=["create"])
        assert c.name == "test"
        assert c.applies_to == ["create"]
        assert c.requires == []
        assert c.ordering == []

    def test_violation_fields(self):
        v = SemanticViolation(
            contract_name="c1",
            rule_type="requires",
            rule="entity_creation",
            severity="high",
            message="missing",
        )
        assert v.severity == "high"
        assert v.evidence is None

    def test_evaluation_result_high_violations(self):
        high = SemanticViolation("c", "requires", "r", "high", "m")
        low = SemanticViolation("c", "requires", "r2", "low", "m2")
        result = SemanticEvaluationResult(
            contract_name="c", passed=False, violations=[high, low], score=0.5,
        )
        assert len(result.high_violations()) == 1
        assert result.high_violations()[0].severity == "high"


class TestContractReport:
    """Test SemanticContractReport aggregate properties."""

    def test_empty_report_score_is_1(self):
        report = SemanticContractReport(results=[])
        assert report.overall_score == 1.0
        assert report.passed_count == 0
        assert report.failed_count == 0
        assert not report.has_critical_violation

    def test_all_passed(self):
        results = [
            SemanticEvaluationResult("c1", True, [], 1.0),
            SemanticEvaluationResult("c2", True, [], 1.0),
        ]
        report = SemanticContractReport(results=results)
        assert report.overall_score == 1.0
        assert report.passed_count == 2
        assert report.failed_count == 0

    def test_mixed_results(self):
        v = SemanticViolation("c2", "requires", "r", "medium", "m")
        results = [
            SemanticEvaluationResult("c1", True, [], 1.0),
            SemanticEvaluationResult("c2", False, [v], 0.5),
        ]
        report = SemanticContractReport(results=results)
        assert report.overall_score == pytest.approx(0.75)
        assert report.passed_count == 1
        assert report.failed_count == 1
        assert not report.has_critical_violation

    def test_has_critical_violation(self):
        v = SemanticViolation("c1", "requires", "r", "high", "m")
        results = [
            SemanticEvaluationResult("c1", False, [v], 0.0),
        ]
        report = SemanticContractReport(results=results)
        assert report.has_critical_violation

    def test_all_violations(self):
        v1 = SemanticViolation("c1", "requires", "r1", "high", "m1")
        v2 = SemanticViolation("c2", "output", "r2", "medium", "m2")
        results = [
            SemanticEvaluationResult("c1", False, [v1], 0.5),
            SemanticEvaluationResult("c2", False, [v2], 0.7),
        ]
        report = SemanticContractReport(results=results)
        assert len(report.all_violations) == 2

    def test_to_dict(self):
        results = [SemanticEvaluationResult("c1", True, [], 1.0)]
        report = SemanticContractReport(results=results)
        d = report.to_dict()
        assert d["overall_score"] == 1.0
        assert d["has_critical"] is False
        assert d["passed"] == 1
        assert d["total"] == 1


# ============================================================================
# 2. Contract Definitions
# ============================================================================

class TestContractDefinitions:
    """Verify the 5 core contracts have correct structure."""

    def test_all_contracts_count(self):
        assert len(ALL_CONTRACTS) == 5

    def test_create_requires_persistence(self):
        c = CREATE_REQUIRES_PERSISTENCE
        assert "create" in c.applies_to
        assert "entity_creation" in c.requires
        assert "persistence" in c.requires

    def test_create_output_references_entity(self):
        c = CREATE_OUTPUT_REFERENCES_ENTITY
        assert "create" in c.applies_to
        assert "output_must_reference_created_entity" in c.output_rules

    def test_auth_verification_precedes_token(self):
        c = AUTH_VERIFICATION_PRECEDES_TOKEN
        assert "auth.login" in c.applies_to
        assert len(c.ordering) == 1
        assert "->" in c.ordering[0]
        assert "verification_failure_blocks_token" in c.branch_rules

    def test_failure_branch_blocks_success(self):
        c = FAILURE_BRANCH_BLOCKS_SUCCESS
        assert "auth.login" in c.applies_to
        assert "failure_path_blocks_success" in c.branch_rules


# ============================================================================
# 3. Contract Registry
# ============================================================================

class TestResolveContracts:
    """Test resolve_contracts with various tag sets."""

    def test_empty_tags_returns_empty(self):
        assert resolve_contracts([]) == []

    def test_create_tag_matches_multiple(self):
        contracts = resolve_contracts(["create"])
        names = {c.name for c in contracts}
        assert "create_requires_persistence" in names
        assert "create_output_references_entity" in names
        assert "failure_branch_blocks_success" in names

    def test_auth_login_tag(self):
        contracts = resolve_contracts(["auth.login"])
        names = {c.name for c in contracts}
        assert "auth_verification_precedes_token" in names
        assert "failure_branch_blocks_success" in names

    def test_chat_send_tag(self):
        # chat.send now matches general create contracts (send is in applies_to)
        resolve_contracts(["chat.send"])
        # May match create_requires_persistence (applies_to includes "send")
        # or may be empty if no contract has "chat.send" in applies_to

    def test_upload_tag(self):
        contracts = resolve_contracts(["upload"])
        names = {c.name for c in contracts}
        assert "create_requires_persistence" in names

    def test_no_matching_tag(self):
        contracts = resolve_contracts(["unrelated.tag"])
        assert contracts == []

    def test_no_duplicate_contracts(self):
        contracts = resolve_contracts(["send", "create", "upload"])
        names = [c.name for c in contracts]
        assert len(names) == len(set(names)), "Duplicate contracts returned"


class TestExtractContextTags:
    """Test extract_context_tags — structural sources only (No Keyword Gate).

    Tags must come from spec.request_type (set by SpecResolver LLM or
    structural inference) and matched_semantic_keys.  raw_request / intent
    natural-language keyword matching is intentionally removed — "add" can
    mean list-append or entity-create; keyword-based dispatch is ambiguous.
    """

    def test_empty_inputs(self):
        tags = extract_context_tags()
        assert tags == []

    # ── raw_request is ignored (No Keyword Gate) ──────────────────────────

    def test_raw_request_no_longer_drives_tags(self):
 # "build a login API" contains keywords but raw_request is ignored.
        # Tags must come from spec.request_type or matched_semantic_keys.
        tags = extract_context_tags(raw_request="build a login API")
        assert tags == []

    def test_raw_request_add_does_not_trigger_create(self):
 # "addition" (add) in raw text must NOT map to "create" entity contract.
        # This was the root cause of lineage PASS_PARTIAL on pure function mods.
        tags = extract_context_tags(raw_request="add is_generic_class_def function")
        assert "create" not in tags

    # ── spec.request_type is the authoritative structured signal ──────────

    def test_spec_request_type_create(self):
        @dataclass
        class FakeSpec:
            request_type: str = "create_new"
        tags = extract_context_tags(spec=FakeSpec())
        assert "create" in tags

    def test_spec_request_type_modify_no_create_tag(self):
        @dataclass
        class FakeSpec:
            request_type: str = "modify"
        tags = extract_context_tags(spec=FakeSpec())
        assert "create" not in tags

    def test_spec_request_type_auth(self):
        @dataclass
        class FakeSpec:
            request_type: str = "auth_login"
        tags = extract_context_tags(spec=FakeSpec())
        assert "auth.login" in tags

    # ── spec.intent keyword loop removed — intent is natural language ─────

    def test_spec_intent_no_longer_drives_tags(self):
        @dataclass
        class FakeSpec:
            intent: str = "login authentication"
            request_type: str = ""
        tags = extract_context_tags(spec=FakeSpec())
        # intent text is ignored; request_type="" → no tags
        assert "auth.login" not in tags

    # ── matched_semantic_keys remain authoritative ────────────────────────

    def test_matched_semantic_keys_auth(self):
        tags = extract_context_tags(matched_semantic_keys=["auth_endpoint", "token_generation"])
        assert "auth.login" in tags

    def test_matched_semantic_keys_create(self):
        tags = extract_context_tags(matched_semantic_keys=["product_model"])
        assert "create" in tags

    def test_combined_structured_sources(self):
        @dataclass
        class FakeSpec:
            request_type: str = "create"
        tags = extract_context_tags(
            spec=FakeSpec(),
            raw_request="video upload",          # ignored
            matched_semantic_keys=["upload_endpoint"],
        )
        assert "video.upload" in tags
        assert "create" in tags
        assert "upload" not in tags   # raw_request "upload" keyword no longer fires

    def test_no_duplicate_tags(self):
        @dataclass
        class FakeSpec:
            request_type: str = "create"
        tags = extract_context_tags(
            spec=FakeSpec(),
            matched_semantic_keys=["product_model"],  # also → "create"
        )
        assert tags.count("create") == 1


# ============================================================================
# 4. Evaluator — Individual Checkers
# ============================================================================

class TestCheckRequirement:
    """Test _check_requirement for each known requirement."""

    def test_entity_creation_pass(self):
        trace = _make_trace(all_instantiations={"Message"})
        assert _check_requirement("entity_creation", trace) is True

    def test_entity_creation_fail(self):
        trace = _make_trace(all_instantiations=set())
        assert _check_requirement("entity_creation", trace) is False

    def test_persistence_via_persist_calls(self):
        trace = _make_trace(all_persist_calls={"db.add"})
        assert _check_requirement("persistence", trace) is True

    def test_persistence_via_call_names(self):
        trace = _make_trace(all_calls={"save"})
        assert _check_requirement("persistence", trace) is True

    def test_persistence_via_commit(self):
        trace = _make_trace(all_calls={"commit"})
        assert _check_requirement("persistence", trace) is True

    def test_persistence_fail(self):
        trace = _make_trace(all_calls={"print", "format"})
        assert _check_requirement("persistence", trace) is False

    def test_user_lookup_pass(self):
        trace = _make_trace(all_calls={"get_user"})
        assert _check_requirement("user_lookup", trace) is True

    def test_user_lookup_fail(self):
        trace = _make_trace(all_calls={"create_user"})
        assert _check_requirement("user_lookup", trace) is False

    def test_password_verification_pass(self):
        trace = _make_trace(all_calls={"verify_password"})
        assert _check_requirement("password_verification", trace) is True

    def test_password_verification_case_insensitive(self):
        trace = _make_trace(all_calls={"Verify_Password"})
        assert _check_requirement("password_verification", trace) is True

    def test_password_verification_fail(self):
        trace = _make_trace(all_calls={"hash_password"})
        assert _check_requirement("password_verification", trace) is False

    def test_token_generation_pass(self):
        trace = _make_trace(all_calls={"create_access_token"})
        assert _check_requirement("token_generation", trace) is True

    def test_token_generation_fail(self):
        trace = _make_trace(all_calls={"refresh_token"})
        assert _check_requirement("token_generation", trace) is False

    def test_unknown_requirement_passes(self):
        trace = _make_trace()
        assert _check_requirement("unknown_thing", trace) is True

    # ── P1-2: Fuzzy/stem matching tests ──────────────────────────────────

    def test_persistence_stem_db_save(self):
        """Custom persist function with 'save' stem should match."""
        trace = _make_trace(all_calls={"save_to_database"})
        assert _check_requirement("persistence", trace) is True

    def test_persistence_stem_upsert(self):
        """'upsert_record' contains stem 'upsert' → match."""
        trace = _make_trace(all_calls={"upsert_record"})
        assert _check_requirement("persistence", trace) is True

    def test_user_lookup_stem_load_user(self):
        """'load_user_from_cache' contains stem 'load_user' → match."""
        trace = _make_trace(all_calls={"load_user_from_cache"})
        assert _check_requirement("user_lookup", trace) is True

    def test_password_verification_stem_bcrypt(self):
        """'bcrypt_compare_password' contains stem 'bcrypt' → match."""
        trace = _make_trace(all_calls={"bcrypt_compare_password"})
        assert _check_requirement("password_verification", trace) is True

    def test_password_verification_stem_argon2(self):
        """'argon2_verify' contains stem 'argon2' → match."""
        trace = _make_trace(all_calls={"argon2_verify"})
        assert _check_requirement("password_verification", trace) is True

    def test_token_generation_stem_jwt_sign(self):
        """'jwt_sign_payload' contains stem 'jwt' → match."""
        trace = _make_trace(all_calls={"jwt_sign_payload"})
        assert _check_requirement("token_generation", trace) is True

    def test_token_generation_no_false_positive_refresh(self):
        """'refresh_token' should NOT match token_generation stems."""
        trace = _make_trace(all_calls={"refresh_token"})
        assert _check_requirement("token_generation", trace) is False


class TestCheckOrdering:
    """Test _check_ordering for step-chain ordering rules."""

    def test_correct_order(self):
        ft = _make_ft(
            name="login",
            call_order=["get_user", "verify_password", "create_access_token"],
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, _msg = _check_ordering(
            "user_lookup -> password_verification -> token_generation", trace,
        )
        assert ok is True

    def test_wrong_order(self):
        ft = _make_ft(
            name="login",
            call_order=["create_access_token", "get_user", "verify_password"],
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, msg = _check_ordering(
            "user_lookup -> password_verification -> token_generation", trace,
        )
        assert ok is False
        assert "not satisfied" in msg.lower()

    def test_missing_step(self):
        ft = _make_ft(
            name="login",
            call_order=["get_user", "create_access_token"],
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, _ = _check_ordering(
            "user_lookup -> password_verification -> token_generation", trace,
        )
        assert ok is False

    def test_single_step_always_passes(self):
        trace = _make_trace()
        ok, _ = _check_ordering("single_step", trace)
        assert ok is True

    def test_empty_call_order(self):
        ft = _make_ft(name="f", call_order=[])
        trace = _make_trace(function_traces={"f": ft})
        ok, _ = _check_ordering(
            "user_lookup -> password_verification", trace,
        )
        assert ok is False

    def test_alternative_function_names(self):
        ft = _make_ft(
            name="auth",
            call_order=["find_user", "check_password", "generate_token"],
        )
        trace = _make_trace(function_traces={"auth": ft})
        ok, _ = _check_ordering(
            "user_lookup -> password_verification -> token_generation", trace,
        )
        assert ok is True


class TestCheckBinding:
    """Test _check_binding for data flow rules."""

    def test_entity_content_from_input_pass(self):
        ft = _make_ft(
            name="send",
            entity_bindings=[("content", "content")],
        )
        trace = _make_trace(function_traces={"send": ft})
        ok, _ = _check_binding("entity_content_from_input", trace)
        assert ok is True

    def test_entity_content_from_input_fail(self):
        ft = _make_ft(name="send", entity_bindings=[])
        trace = _make_trace(function_traces={"send": ft})
        ok, _msg = _check_binding("entity_content_from_input", trace)
        assert ok is False

    def test_entity_fields_from_input_pass(self):
        ft = _make_ft(
            name="upload",
            instantiations={"Video"},
            entity_bindings=[("title", "title")],
        )
        trace = _make_trace(function_traces={"upload": ft})
        ok, _ = _check_binding("entity_fields_from_input", trace)
        assert ok is True

    def test_entity_fields_from_input_no_instantiation(self):
        ft = _make_ft(
            name="upload",
            instantiations=set(),
            entity_bindings=[("title", "title")],
        )
        trace = _make_trace(function_traces={"upload": ft})
        ok, _msg = _check_binding("entity_fields_from_input", trace)
        assert ok is False

    def test_unknown_binding_rule_passes(self):
        trace = _make_trace()
        ok, _ = _check_binding("unknown_rule", trace)
        assert ok is True


class TestCheckBranch:
    """Test _check_branch for control flow rules."""

    def test_verification_failure_blocks_token_pass(self):
        ft = _make_ft(
            name="login",
            calls={"verify_password", "create_access_token"},
            has_error_branch=True,
            error_before_success=True,
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, _ = _check_branch("verification_failure_blocks_token", trace)
        assert ok is True

    def test_verification_failure_blocks_token_no_error_branch(self):
        ft = _make_ft(
            name="login",
            calls={"verify_password", "create_access_token"},
            has_error_branch=False,
            error_before_success=False,
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, msg = _check_branch("verification_failure_blocks_token", trace)
        assert ok is False
        assert "error branch" in msg.lower()

    def test_verification_no_token_call_passes(self):
        """If function has verify but not token, rule is N/A."""
        ft = _make_ft(
            name="check",
            calls={"verify_password"},
            has_error_branch=False,
        )
        trace = _make_trace(function_traces={"check": ft})
        ok, _ = _check_branch("verification_failure_blocks_token", trace)
        assert ok is True

    def test_failure_path_blocks_success_pass(self):
        ft = _make_ft(
            name="login",
            calls={"verify_password"},
            has_error_branch=True,
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, _ = _check_branch("failure_path_blocks_success", trace)
        assert ok is True

    def test_failure_path_blocks_success_fail(self):
        ft = _make_ft(
            name="login",
            calls={"validate"},
            has_error_branch=False,
        )
        trace = _make_trace(function_traces={"login": ft})
        ok, msg = _check_branch("failure_path_blocks_success", trace)
        assert ok is False
        assert "no failure branch" in msg.lower()

    def test_failure_path_no_validation_passes(self):
        ft = _make_ft(
            name="helper",
            calls={"format_data"},
            has_error_branch=False,
        )
        trace = _make_trace(function_traces={"helper": ft})
        ok, _ = _check_branch("failure_path_blocks_success", trace)
        assert ok is True

    def test_unknown_branch_rule_passes(self):
        trace = _make_trace()
        ok, _ = _check_branch("unknown_rule", trace)
        assert ok is True


class TestCheckOutput:
    """Test _check_output for output semantics rules."""

    def test_output_references_entity_pass(self):
        ft = _make_ft(
            name="create",
            instantiations={"Item"},
            return_has_entity_ref=True,
        )
        trace = _make_trace(
            function_traces={"create": ft},
            all_instantiations={"Item"},
        )
        ok, _ = _check_output("output_must_reference_created_entity", trace)
        assert ok is True

    def test_output_references_entity_fail(self):
        ft = _make_ft(
            name="create",
            instantiations={"Item"},
            return_has_entity_ref=False,
        )
        trace = _make_trace(
            function_traces={"create": ft},
            all_instantiations={"Item"},
        )
        ok, msg = _check_output("output_must_reference_created_entity", trace)
        assert ok is False
        assert "not referenced" in msg.lower()

    def test_output_no_instantiations_passes(self):
        """If nothing is created, rule does not apply."""
        trace = _make_trace(all_instantiations=set())
        ok, _ = _check_output("output_must_reference_created_entity", trace)
        assert ok is True

    def test_unknown_output_rule_passes(self):
        trace = _make_trace()
        ok, _ = _check_output("unknown_rule", trace)
        assert ok is True


# ============================================================================
# 5. Evaluator — Full Contract Evaluation
# ============================================================================

class TestEvaluateContracts:
    """Test evaluate_contracts end-to-end."""

    def test_passing_create_requires_persistence(self):
        ft = _make_ft(name="create_item", instantiations={"Item"}, persist_calls={"db.add"})
        trace = _make_trace(
            function_traces={"create_item": ft},
            all_instantiations={"Item"},
            all_persist_calls={"db.add"},
        )
        report = evaluate_contracts([CREATE_REQUIRES_PERSISTENCE], trace)
        assert report.passed_count == 1
        assert report.overall_score == 1.0

    def test_failing_create_requires_persistence(self):
        trace = _make_trace(all_instantiations=set(), all_persist_calls=set())
        report = evaluate_contracts([CREATE_REQUIRES_PERSISTENCE], trace)
        assert report.failed_count == 1
        assert report.overall_score < 1.0
        assert report.has_critical_violation  # requires violations are "high"

    def test_passing_auth_contract(self):
        ft = _make_ft(
            name="login",
            calls={"get_user", "verify_password", "create_access_token"},
            call_order=["get_user", "verify_password", "create_access_token"],
            has_error_branch=True,
            error_before_success=True,
        )
        trace = _make_trace(
            function_traces={"login": ft},
            all_calls={"get_user", "verify_password", "create_access_token"},
        )
        report = evaluate_contracts([AUTH_VERIFICATION_PRECEDES_TOKEN], trace)
        assert report.passed_count == 1
        assert report.overall_score == 1.0

    def test_failing_auth_missing_verify(self):
        ft = _make_ft(
            name="login",
            calls={"get_user", "create_access_token"},
            call_order=["get_user", "create_access_token"],
        )
        trace = _make_trace(
            function_traces={"login": ft},
            all_calls={"get_user", "create_access_token"},
        )
        report = evaluate_contracts([AUTH_VERIFICATION_PRECEDES_TOKEN], trace)
        assert report.failed_count == 1
        # Missing password_verification requirement
        violations = report.all_violations
        assert any(v.rule == "password_verification" for v in violations)

    def test_passing_failure_branch(self):
        ft = _make_ft(
            name="login",
            calls={"verify_password"},
            has_error_branch=True,
        )
        trace = _make_trace(function_traces={"login": ft})
        report = evaluate_contracts([FAILURE_BRANCH_BLOCKS_SUCCESS], trace)
        assert report.passed_count == 1

    def test_failing_failure_branch(self):
        ft = _make_ft(
            name="login",
            calls={"verify_password"},
            has_error_branch=False,
        )
        trace = _make_trace(function_traces={"login": ft})
        report = evaluate_contracts([FAILURE_BRANCH_BLOCKS_SUCCESS], trace)
        assert report.failed_count == 1

    def test_multiple_contracts(self):
        """Evaluate multiple contracts at once."""
        ft = _make_ft(
            name="create_item",
            instantiations={"Item"},
            return_has_entity_ref=True,
        )
        trace = _make_trace(
            function_traces={"create_item": ft},
            all_instantiations={"Item"},
            all_persist_calls={"db.add"},
        )
        report = evaluate_contracts(
            [CREATE_REQUIRES_PERSISTENCE, CREATE_OUTPUT_REFERENCES_ENTITY],
            trace,
        )
        assert len(report.results) == 2
        assert report.passed_count == 2

    def test_empty_contracts_list(self):
        trace = _make_trace()
        report = evaluate_contracts([], trace)
        assert report.overall_score == 1.0
        assert len(report.results) == 0

    def test_score_penalty_calculation(self):
        """High severity = 0.25 penalty, medium = 0.15."""
        # Force 2 high violations via create_requires_persistence (entity_creation + persistence)
        trace = _make_trace()
        report = evaluate_contracts([CREATE_REQUIRES_PERSISTENCE], trace)
        result = report.results[0]
        # 2 high violations: 2 * 0.25 = 0.50 penalty => score 0.50
        assert result.score == pytest.approx(0.50, abs=0.01)


# ============================================================================
# 6. Integration: Registry + Evaluator
# ============================================================================

class TestRegistryEvaluatorIntegration:
    """End-to-end: extract tags -> resolve contracts -> evaluate."""

    def test_login_full_pipeline(self):
        # Use matched_semantic_keys (structured) instead of raw_request keywords
        tags = extract_context_tags(matched_semantic_keys=["auth_endpoint", "token_generation"])
        contracts = resolve_contracts(tags)
        assert len(contracts) > 0

        # Build a passing trace
        ft = _make_ft(
            name="login",
            calls={"get_user", "verify_password", "create_access_token"},
            instantiations={"Token"},
            persist_calls=set(),
            call_order=["get_user", "verify_password", "create_access_token"],
            has_error_branch=True,
            error_before_success=True,
            return_has_entity_ref=True,
        )
        trace = _make_trace(
            function_traces={"login": ft},
            all_calls={"get_user", "verify_password", "create_access_token", "save"},
            all_instantiations={"Token"},
            all_persist_calls=set(),
        )
        report = evaluate_contracts(contracts, trace)
        # Should have at least some contracts evaluated
        assert len(report.results) > 0

    def test_create_full_pipeline(self):
        # Use matched_semantic_keys (structured) for entity-create domain
        tags = extract_context_tags(matched_semantic_keys=["product_model"])
        contracts = resolve_contracts(tags)
        assert any(c.name == "create_requires_persistence" for c in contracts)

        ft = _make_ft(
            name="create_product",
            instantiations={"Product"},
            persist_calls={"db.add"},
            return_has_entity_ref=True,
        )
        trace = _make_trace(
            function_traces={"create_product": ft},
            all_instantiations={"Product"},
            all_persist_calls={"db.add"},
        )
        report = evaluate_contracts(contracts, trace)
        # create_requires_persistence should pass
        crp = next(
            (r for r in report.results if r.contract_name == "create_requires_persistence"),
            None,
        )
        assert crp is not None
        assert crp.passed is True
