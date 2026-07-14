"""semantic_contracts.py — Phase C.1: Core Semantic Contract Definitions.

5 foundational contracts that cover general behavioral patterns:
1. create_requires_persistence — create action must persist
2. create_output_references_entity — create output must reference what was created
3. auth_verification_precedes_token — auth must verify before granting token
4. failure_branch_blocks_success — failure path must not reach success output
5. entity_flow_connectivity — entity lifecycle: create → persist → return

Phase 2 reform: removed domain-specific contracts:
- message_send_creates_and_returns (chat.send domain)
- upload_creates_entity_from_input (video.upload domain)
These were special cases of create_requires_persistence.
"""
from __future__ import annotations

from external_llm.editor.semantic.semantic_contract_models import SemanticContract

# ── 1. Create requires persistence ────────────────────────────────────────────
CREATE_REQUIRES_PERSISTENCE = SemanticContract(
    name="create_requires_persistence",
    applies_to=["create", "upload", "send", "register", "signup"],
    requires=["entity_creation", "persistence"],
    metadata={"description": "Create actions must persist the created entity"},
)

# ── 2. Create output references entity ────────────────────────────────────────
CREATE_OUTPUT_REFERENCES_ENTITY = SemanticContract(
    name="create_output_references_entity",
    applies_to=["create", "upload", "send"],
    output_rules=["output_must_reference_created_entity"],
    metadata={"description": "Create response must reference the created entity"},
)

# ── 3. Auth verification precedes token ───────────────────────────────────────
AUTH_VERIFICATION_PRECEDES_TOKEN = SemanticContract(
    name="auth_verification_precedes_token",
    applies_to=["auth.login", "auth.authenticate", "auth.sign_in"],
    requires=["user_lookup", "password_verification", "token_generation"],
    ordering=["user_lookup -> password_verification -> token_generation"],
    branch_rules=["verification_failure_blocks_token"],
    metadata={"description": "Auth must verify credentials before issuing token"},
)

# ── 4. Failure branch blocks success ──────────────────────────────────────────
FAILURE_BRANCH_BLOCKS_SUCCESS = SemanticContract(
    name="failure_branch_blocks_success",
    applies_to=["auth.login", "auth.authenticate", "create", "update", "delete"],
    branch_rules=["failure_path_blocks_success"],
    metadata={"description": "Failure paths must not reach success output"},
)

# ── 5. Entity flow connectivity ───────────────────────────────────────────────
ENTITY_FLOW_CONNECTIVITY = SemanticContract(
    name="entity_flow_connectivity",
    applies_to=["create", "upload", "send", "register"],
    requires=["entity_creation", "persistence"],
    output_rules=["output_must_reference_created_entity"],
    metadata={
        "description": "Entity lifecycle must be complete: create → persist → return",
        "flow_contract": "true",
    },
)


ALL_CONTRACTS = [
    CREATE_REQUIRES_PERSISTENCE,
    CREATE_OUTPUT_REFERENCES_ENTITY,
    AUTH_VERIFICATION_PRECEDES_TOKEN,
    FAILURE_BRANCH_BLOCKS_SUCCESS,
    ENTITY_FLOW_CONNECTIVITY,
]
