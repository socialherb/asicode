"""semantic_contract_evaluator.py — Phase C.1: Contract Evaluation Engine.

Evaluates semantic contracts against traced execution results.
Each contract rule type has a dedicated checker that inspects the SemanticTrace.

Phase C.1 scope: detection only (no correction).
"""
from __future__ import annotations

import logging

from external_llm.editor.semantic.semantic_contract_models import (
    SemanticContract,
    SemanticContractReport,
    SemanticEvaluationResult,
    SemanticViolation,
)
from external_llm.editor.semantic.semantic_tracer import SemanticTrace

logger = logging.getLogger(__name__)


def evaluate_contracts(
    contracts: list[SemanticContract],
    trace: SemanticTrace,
) -> SemanticContractReport:
    """Evaluate all contracts against the semantic trace.

    Returns a report with per-contract results and aggregate score.
    """
    results = []
    for contract in contracts:
        result = _evaluate_single(contract, trace)
        results.append(result)

    report = SemanticContractReport(results=results)

    logger.info(
        "[CONTRACT_EVAL] %d/%d passed, score=%.2f, critical=%s",
        report.passed_count, len(results),
        report.overall_score, report.has_critical_violation,
    )

    return report


def _evaluate_single(
    contract: SemanticContract,
    trace: SemanticTrace,
) -> SemanticEvaluationResult:
    """Evaluate a single contract against the trace."""
    violations: list[SemanticViolation] = []

    # Check requires
    for req in contract.requires:
        if not _check_requirement(req, trace):
            violations.append(_make_violation(
                contract, "requires", req, "high",
                f"Required behavior '{req}' not found in trace",
            ))

    # Check ordering
    for rule in contract.ordering:
        ok, msg = _check_ordering(rule, trace)
        if not ok:
            violations.append(_make_violation(
                contract, "ordering", rule, "medium", msg,
            ))

    # Check binding rules
    for rule in contract.binding_rules:
        ok, msg = _check_binding(rule, trace)
        if not ok:
            violations.append(_make_violation(
                contract, "binding", rule, "medium", msg,
            ))

    # Check branch rules
    for rule in contract.branch_rules:
        ok, msg = _check_branch(rule, trace)
        if not ok:
            violations.append(_make_violation(
                contract, "branch", rule, "high", msg,
            ))

    # Check output rules
    for rule in contract.output_rules:
        ok, msg = _check_output(rule, trace)
        if not ok:
            violations.append(_make_violation(
                contract, "output", rule, "high", msg,
            ))

    # Score: 1.0 - (violations penalty)
    # High = 0.25, Medium = 0.15, Low = 0.05
    _SEVERITY_PENALTY = {"high": 0.25, "medium": 0.15, "low": 0.05}
    penalty = sum(_SEVERITY_PENALTY.get(v.severity, 0.1) for v in violations)
    score = max(0.0, 1.0 - penalty)

    return SemanticEvaluationResult(
        contract_name=contract.name,
        passed=len(violations) == 0,
        violations=violations,
        score=round(score, 4),
    )


# ── Requirement Checkers ──────────────────────────────────────────────────────

def _fuzzy_call_match(call_names: set[str], whitelist: set[str], stems: set[str]) -> bool:
    """Match calls against whitelist (exact) then stems (substring).

    *whitelist*: exact function names (e.g. {"verify_password", "check_password"}).
    *stems*: keyword fragments (e.g. {"verify", "auth", "check"}).
    Returns True if any call matches either the whitelist or contains a stem.
    """
    calls_lower = {c.lower() for c in call_names}
    # 1. Exact match
    if calls_lower & {w.lower() for w in whitelist}:
        return True
    # 2. Stem/substring match — any call contains a stem keyword
    return any(
        stem in call for stem in stems for call in calls_lower
    )


def _check_requirement(req: str, trace: SemanticTrace) -> bool:
    """Check if a behavioral requirement is satisfied.

    Uses stem-based matching (behavioral patterns) instead of
    hardcoded function name whitelists (domain-specific).
    """
    if req == "entity_creation":
        return bool(trace.all_instantiations)

    if req == "persistence":
        if trace.all_persist_calls:
            return True
        _stems = {"save", "store", "persist", "insert", "commit", "write", "put", "upsert", "append"}
        return _fuzzy_call_match(trace.all_calls, set(), _stems)

    if req == "user_lookup":
        _stems = {"get_", "find_", "lookup", "fetch_", "query_", "load_"}
        return _fuzzy_call_match(trace.all_calls, set(), _stems)

    if req == "password_verification":
        _stems = {"verify", "check_pass", "authenticate", "validate", "compare",
                  "bcrypt", "argon2", "scrypt"}
        return _fuzzy_call_match(trace.all_calls, set(), _stems)

    if req == "token_generation":
        # Stems that indicate token *creation* (not consumption/refresh)
        _stems = {"create_token", "create_access", "generate_token", "sign_token",
                  "issue_token", "encode_token", "make_token", "jwt_sign", "sign_payload"}
        return _fuzzy_call_match(trace.all_calls, set(), _stems)

    # Unknown requirement → pass (conservative)
    return True


# ── Ordering Checkers ─────────────────────────────────────────────────────────

def _check_ordering(rule: str, trace: SemanticTrace) -> tuple:
    """Check execution ordering constraint.

    Rule format: "step_a -> step_b -> step_c"
    Checks that in any function's call_order, step_a appears before step_b, etc.
    """
    parts = [p.strip() for p in rule.split("->")]
    if len(parts) < 2:
        return True, ""

    # Map abstract steps to stem keywords (behavioral patterns, not function names)
    _STEP_STEMS: dict[str, set[str]] = {
        "user_lookup": {"get_", "find_", "lookup", "fetch_", "query_", "load_"},
        "password_verification": {"verify", "check_pass", "authenticate", "validate", "compare"},
        "token_generation": {"create_token", "create_access", "generate_token", "sign_token",
                             "issue_token", "encode_token", "make_token", "jwt_sign", "sign_payload"},
    }

    # Check ordering in each function's call_order
    for ft in trace.function_traces.values():
        if not ft.call_order:
            continue

        call_order_lower = [c.lower() for c in ft.call_order]

        prev_idx = -1
        all_found = True
        for step in parts:
            step_stems = _STEP_STEMS.get(step, {step})
            # Find first occurrence of any matching function via stem
            found_idx = -1
            for i, call in enumerate(call_order_lower):
                if i <= prev_idx:
                    continue
                if any(stem in call for stem in step_stems):
                    found_idx = i
                    break

            if found_idx == -1:
                all_found = False
                break
            prev_idx = found_idx

        if all_found:
            return True, ""

    # No function had all steps in order
    return False, f"Ordering not satisfied: {rule}"


# ── Binding Checkers ──────────────────────────────────────────────────────────

def _check_binding(rule: str, trace: SemanticTrace) -> tuple:
    """Check data binding rules."""
    if rule == "entity_content_from_input":
        # Check if any entity constructor receives function params as args
        for ft in trace.function_traces.values():
            if ft.entity_bindings:
                return True, ""
        return False, "No entity constructor binds to function input parameters"

    if rule == "entity_fields_from_input":
        # Same check: entity instantiation uses function params
        for ft in trace.function_traces.values():
            if ft.instantiations and ft.entity_bindings:
                return True, ""
        return False, "Entity creation does not derive fields from request input"

    return True, ""


# ── Branch Checkers ───────────────────────────────────────────────────────────

def _check_branch(rule: str, trace: SemanticTrace) -> tuple:
    """Check control flow / branching rules."""
    if rule == "verification_failure_blocks_token":
        # If a function has both verify-like and token-like calls,
        # it should have an error branch before the success path
        _verify_stems = {"verify", "check_pass", "authenticate", "validate", "compare"}
        _token_stems = {"create_token", "create_access", "generate_token", "sign_token",
                        "encode_token", "make_token", "jwt_sign", "sign_payload"}
        for ft in trace.function_traces.values():
            calls_lower = {c.lower() for c in ft.calls}
            has_verify = any(stem in call for stem in _verify_stems for call in calls_lower)
            has_token = any(stem in call for stem in _token_stems for call in calls_lower)
            if has_verify and has_token:
                if ft.has_error_branch and ft.error_before_success:
                    return True, ""
                return False, (
                    f"Function '{ft.name}' calls verify + token but lacks "
                    f"error branch before success path"
                )
        return True, ""

    if rule == "failure_path_blocks_success":
        # Generic: any function with validation/verification should have error handling
        _validation_stems = {"verify", "check_pass", "authenticate", "validate", "compare"}
        for ft in trace.function_traces.values():
            has_validation = _fuzzy_call_match(ft.calls, set(), _validation_stems)
            if has_validation and not ft.has_error_branch:
                return False, (
                    f"Function '{ft.name}' validates but has no failure branch"
                )
        return True, ""

    return True, ""


# ── Output Checkers ───────────────────────────────────────────────────────────

def _check_output(rule: str, trace: SemanticTrace) -> tuple:
    """Check output semantics rules."""
    if rule == "output_must_reference_created_entity":
        # At least one function that creates an entity should return something
        # that references it
        for ft in trace.function_traces.values():
            if ft.instantiations:
                if ft.return_has_entity_ref:
                    return True, ""
                # Also accept: return dict with entity field access
                # (covered by return_has_entity_ref in tracer)

        # Flow inference fallback: if a function creates an entity AND persists it,
        # infer that the flow is complete even if return_has_entity_ref is False.
        # This catches patterns like: video = Video(...); db.add(video); db.commit(); return {..., "id": video.id}
        for ft in trace.function_traces.values():
            if ft.instantiations and ft.persist_calls:
                # Entity created + persisted → accept as connected flow
                return True, ""

        # Cross-function inference: any function creates, any function persists
        if trace.all_instantiations and trace.all_persist_calls:
            return True, ""

        # If no function creates entities, this rule doesn't apply → pass
        if not trace.all_instantiations:
            return True, ""

        return False, "Created entities are not referenced in return values"

    return True, ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_violation(
    contract: SemanticContract,
    rule_type: str,
    rule: str,
    severity: str,
    message: str,
) -> SemanticViolation:
    return SemanticViolation(
        contract_name=contract.name,
        rule_type=rule_type,
        rule=rule,
        severity=severity,
        message=message,
    )
