"""semantic_rewrite_planner.py — Phase C.3: Violation → Rewrite Plan.

Analyzes remaining contract violations after C.2 and generates
AST-based rewrite operations to resolve them.

Only generates plans for violations that C.2 couldn't fix:
- Ordering issues (calls exist but in wrong order)
- Wrong call arguments (call exists but args are wrong)
- Static/mock returns (return exists but doesn't reference entity)
- Misplaced statements (statement exists but in wrong position)
"""
from __future__ import annotations

import logging

from external_llm.editor.semantic.semantic_contract_models import (
    SemanticContractReport,
    SemanticEvaluationResult,
    SemanticViolation,
)
from external_llm.editor.semantic.semantic_rewrite_models import RewriteOperation, RewriteOpType, RewritePlan
from external_llm.editor.semantic.semantic_tracer import SemanticTrace

logger = logging.getLogger(__name__)


# ── Ordering definitions ──────────────────────────────────────────────────────

# Contract → expected call ordering within target functions
_ORDERING_SPECS: dict[str, dict[str, list[str]]] = {
    "auth_verification_precedes_token": {
        # func_name → desired order of calls
        "login": ["get_user", "verify_password", "create_access_token"],
        "authenticate": ["get_user", "verify_password", "create_access_token"],
        "sign_in": ["get_user", "verify_password", "create_access_token"],
    },
}

# Contract → call arg fixes
_ARG_FIX_SPECS: dict[str, dict[str, list[str]]] = {
    "auth_verification_precedes_token": {
        # call_name → correct args
        "verify_password": ["password", "user.hashed_password"],
    },
}

# Entity variable names per context
_ENTITY_VARS: dict[str, str] = {
    "Message": "message",
    "Video": "video",
    "User": "user",
    "Product": "product",
    "Item": "item",
}


def build_rewrite_plans(
    report: SemanticContractReport,
    trace: SemanticTrace,
) -> list[RewritePlan]:
    """Generate rewrite plans for remaining violations.

    Returns one RewritePlan per file that needs modification.
    """
    plans_by_file: dict[str, RewritePlan] = {}
    failed = [r for r in report.results if not r.passed]

    for eval_result in failed:
        for violation in eval_result.violations:
            ops = _violation_to_ops(eval_result, violation, trace)
            for op, file_path in ops:
                if file_path not in plans_by_file:
                    plans_by_file[file_path] = RewritePlan(file_path=file_path)
                plans_by_file[file_path].operations.append(op)

    plans = list(plans_by_file.values())
    total_ops = sum(len(p.operations) for p in plans)
    if total_ops:
        logger.info("[C.3 PLANNER] %d plans, %d total ops", len(plans), total_ops)
    return plans


def _violation_to_ops(
    eval_result: SemanticEvaluationResult,
    violation: SemanticViolation,
    trace: SemanticTrace,
) -> list[tuple]:
    """Convert a violation to (RewriteOperation, file_path) pairs."""
    contract = eval_result.contract_name
    rule_type = violation.rule_type
    rule = violation.rule

    results: list[tuple] = []

    if rule_type == "ordering":
        results.extend(_plan_ordering(contract, rule, trace))
    elif rule_type == "binding":
        results.extend(_plan_binding(contract, rule, trace))
    elif rule_type == "output":
        results.extend(_plan_output(contract, rule, trace))
    elif rule_type == "branch":
        results.extend(_plan_branch(contract, rule, trace))

    return results


def _plan_ordering(
    contract: str, rule: str, trace: SemanticTrace,
) -> list[tuple]:
    """Generate REORDER_CALLS + MOVE_STATEMENT ops for ordering violations."""
    results = []
    specs = _ORDERING_SPECS.get(contract, {})

    for func_name, desired_order in specs.items():
        ft = trace.function_traces.get(func_name)
        if not ft or not ft.file_path:
            continue

        # Check which calls from desired_order are present
        present = [c for c in desired_order if c in ft.calls or c.lower() in {x.lower() for x in ft.calls}]
        if len(present) < 2:
            continue

        # Check if they're out of order
        call_positions = []
        for c in present:
            for i, co in enumerate(ft.call_order):
                if co == c or co.lower() == c.lower():
                    call_positions.append(i)
                    break

        if call_positions == sorted(call_positions):
            continue  # Already in order

        results.append((
            RewriteOperation(
                op_type=RewriteOpType.REORDER_CALLS,
                target_function=func_name,
                payload={"order": present},
                description=f"Reorder calls in {func_name}: {' → '.join(present)}",
                contract_name=contract,
            ),
            ft.file_path,
        ))

        # Also try MOVE_STATEMENT for individual misplaced calls
        for i, c in enumerate(present[:-1]):
            next_c = present[i + 1]
            c_idx = None
            next_idx = None
            for j, co in enumerate(ft.call_order):
                if co == c and c_idx is None:
                    c_idx = j
                if co == next_c and next_idx is None:
                    next_idx = j
            if c_idx is not None and next_idx is not None and c_idx > next_idx:
                results.append((
                    RewriteOperation(
                        op_type=RewriteOpType.MOVE_STATEMENT,
                        target_function=func_name,
                        payload={"call_name": c, "before": next_c},
                        description=f"Move {c} before {next_c} in {func_name}",
                        contract_name=contract,
                    ),
                    ft.file_path,
                ))

    return results


def _plan_binding(
    contract: str, rule: str, trace: SemanticTrace,
) -> list[tuple]:
    """Generate REPLACE_CALL_ARGS ops for binding violations."""
    results = []
    arg_specs = _ARG_FIX_SPECS.get(contract, {})

    for call_name, correct_args in arg_specs.items():
        # Find functions that call this
        for ft in trace.function_traces.values():
            if call_name not in ft.calls and call_name.lower() not in {c.lower() for c in ft.calls}:
                continue
            if not ft.file_path:
                continue

            results.append((
                RewriteOperation(
                    op_type=RewriteOpType.REPLACE_CALL_ARGS,
                    target_function=ft.name,
                    payload={"call_name": call_name, "new_args": correct_args},
                    description=f"Fix {call_name} args to {correct_args}",
                    contract_name=contract,
                ),
                ft.file_path,
            ))

    return results


def _plan_output(
    contract: str, rule: str, trace: SemanticTrace,
) -> list[tuple]:
    """Generate REWRITE_RETURN ops for output violations."""
    results = []

    for ft in trace.function_traces.values():
        if not ft.file_path:
            continue
        if not ft.instantiations:
            continue
        if ft.return_has_entity_ref:
            continue  # Already references entity

        # Build entity-referencing return expression
        entity_cls = next(iter(ft.instantiations))
        var_name = _ENTITY_VARS.get(entity_cls, entity_cls.lower())

        new_return = f'{{"id": getattr({var_name}, "id", 1), "name": getattr({var_name}, "name", "")}}'

        results.append((
            RewriteOperation(
                op_type=RewriteOpType.REWRITE_RETURN,
                target_function=ft.name,
                payload={"new_return": new_return},
                description=f"Rewrite return to reference {entity_cls} in {ft.name}",
                contract_name=contract,
            ),
            ft.file_path,
        ))

    return results


def _plan_branch(
    contract: str, rule: str, trace: SemanticTrace,
) -> list[tuple]:
    """Branch violations are primarily handled by C.2 insert_error_branch.

    C.3 handles the case where verify exists but error branch is after
    the success path (needs MOVE_STATEMENT).
    """
    results = []

    for ft in trace.function_traces.values():
        if not ft.file_path:
            continue
        if ft.has_error_branch and not ft.error_before_success:
            # Error branch exists but is after success path → need to move it
            # This is complex; skip for now, handled by C.2 in most cases
            pass

    return results
