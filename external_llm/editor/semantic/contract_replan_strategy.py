"""contract_replan_strategy.py — Contract failure → replan strategy selection.

Contract evaluation failures determine which replan strategy to use.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
class ContractFailureType(str, Enum):
    MISSING_PERSISTENCE = "missing_persistence"
    MISSING_ENTITY_REFERENCE = "missing_entity_reference"
    MISSING_FAILURE_BRANCH = "missing_failure_branch"
    BROKEN_ENTITY_FLOW = "broken_entity_flow"
    AUTH_TOKEN_MISSING = "auth_token_missing"


# Contract name → failure type mapping
_CONTRACT_FAILURE_MAP = {
    "create_requires_persistence": ContractFailureType.MISSING_PERSISTENCE,
    "entity_flow_connectivity": ContractFailureType.BROKEN_ENTITY_FLOW,
    "failure_branch_blocks_success": ContractFailureType.MISSING_FAILURE_BRANCH,
    "create_output_references_entity": ContractFailureType.MISSING_ENTITY_REFERENCE,
    "auth_verification_precedes_token": ContractFailureType.AUTH_TOKEN_MISSING,
}


# Failure type → planner replan hint
_REPLAN_HINTS = {
    ContractFailureType.MISSING_PERSISTENCE: (
        "Previous contract failure: entity created but NOT persisted.\n"
        "Ensure persistence is wired correctly: session/connection opened, "
        "entity added/saved, and changes committed after creation."
    ),
    ContractFailureType.BROKEN_ENTITY_FLOW: (
        "Previous contract failure: entity flow disconnected.\n"
        "Ensure complete flow: handler → create entity → persist → return entity/id.\n"
        "Do NOT return generic {'status': 'ok'} without entity reference."
    ),
    ContractFailureType.MISSING_ENTITY_REFERENCE: (
        "Previous contract failure: created entity not referenced in response.\n"
        "Return the created entity or its id in the response body."
    ),
    ContractFailureType.MISSING_FAILURE_BRANCH: (
        "Previous contract failure: no error handling branch.\n"
        "Add try/except or validation checks before success path."
    ),
    ContractFailureType.AUTH_TOKEN_MISSING: (
        "Previous contract failure: auth flow incomplete.\n"
        "Ensure: user lookup → password verify → token generation (in order)."
    ),
}


def classify_contract_failures(
    failed_contract_names: list[str],
) -> list[ContractFailureType]:
    """Classify contract name → failure type."""
    out = []
    for name in failed_contract_names:
        ft = _CONTRACT_FAILURE_MAP.get(name)
        if ft:
            out.append(ft)
    return out


def build_contract_replan_hints(
    failures: list[ContractFailureType],
) -> str:
    """Contract failure types → planner prompt hints generation."""
    hints = []
    seen = set()
    for f in failures:
        h = _REPLAN_HINTS.get(f)
        if h and f not in seen:
            hints.append(h)
            seen.add(f)
    if not hints:
        return ""
    return "\n\n## Contract Failure Guard\n" + "\n".join(f"- {h}" for h in hints)


def build_contract_feedback_for_planner(
    contract_report: dict[str, Any],
) -> str:
    """Contract evaluation report → planner feedback text.

    Uses phase_orchestrator C.1 results for subsequent plan/replan.
    """
    if not contract_report:
        return ""

    results = contract_report.get("results", [])
    failed = [r["contract"] for r in results if not r.get("passed", True)]

    if not failed:
        return ""

    failures = classify_contract_failures(failed)
    return build_contract_replan_hints(failures)
