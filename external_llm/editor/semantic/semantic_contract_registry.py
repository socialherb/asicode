"""semantic_contract_registry.py — Phase C.1: Contract Resolution.

Resolves which contracts apply to a given execution context
based on context tags derived from intent/spec/request.
Intent-aware: soft/skip persistence contracts in specific domains like auth.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.editor.semantic.semantic_contract_models import SemanticContract
from external_llm.editor.semantic.semantic_contracts import ALL_CONTRACTS

logger = logging.getLogger(__name__)


# ── Intent-aware contract policy ─────────────────────────────────────

# Contracts that require DB persistence — soft-handled in auth/token domains
_PERSISTENCE_CONTRACTS = {
    "create_requires_persistence",
    "entity_flow_connectivity",
}

# auth domain tags
_AUTH_TAGS = {"auth.login", "auth.signup", "auth.authenticate", "auth.sign_in"}

# DB usage hint tag (e.g., plan generates database.py)
_DB_HINT_TAGS = {"db_wiring", "database"}


def _is_auth_domain(tags_set: set[str]) -> bool:
    """Determine if auth domain."""
    return bool(tags_set & _AUTH_TAGS)


def _has_db_hint(tags_set: set[str]) -> bool:
    """Check if DB usage hint present."""
    return bool(tags_set & _DB_HINT_TAGS)


def resolve_contracts(context_tags: list[str]) -> list[SemanticContract]:
    """Find all contracts that match the given context tags.

    Intent-aware: exclude persistence contracts in auth domains when no
    DB usage is detected, to prevent false positives.
    """
    if not context_tags:
        return []

    tags_set = set(context_tags)
    is_auth = _is_auth_domain(tags_set)
    has_db = _has_db_hint(tags_set)
    matched: list[SemanticContract] = []
    seen: set[str] = set()
    skipped: list[str] = []

    for contract in ALL_CONTRACTS:
        if contract.name in seen:
            continue

        if not any(tag in tags_set for tag in contract.applies_to):
            continue

        # Intent-aware policy: auth without explicit DB → skip persistence contracts
        if contract.name in _PERSISTENCE_CONTRACTS and is_auth and not has_db:
            skipped.append(contract.name)
            continue

        matched.append(contract)
        seen.add(contract.name)

    if skipped:
        logger.info(
            "[CONTRACT] skipped %d contracts for auth domain (no DB hint): %s",
            len(skipped), skipped,
        )

    logger.info(
        "[CONTRACT] resolved %d contracts for tags %s: %s",
        len(matched), context_tags, [c.name for c in matched],
    )
    return matched


def extract_context_tags(
    spec: Any = None,
    raw_request: str = "",
    matched_semantic_keys: Optional[list[str]] = None,
) -> list[str]:
    """Extract context tags from execution context.

    Tags are derived from STRUCTURED sources only — no keyword matching on
    natural language text (raw_request / intent).  Keyword-based dispatch
    violates the No-Keyword-Gate principle: "추가" can mean list-append or
    entity-create; "생성" can mean file-create or object-instantiation.
    The structured spec.request_type (set by LLM or structural inference in
    SpecResolver) and matched_semantic_keys are the authoritative signals.
    """
    tags: list[str] = []

    # ── Source 1: spec.request_type (structured, set by SpecResolver LLM or
    #   structural inference: new_files→create, new_files+files→extend, files→modify)
    if spec is not None:
        request_type = str(getattr(spec, "request_type", "") or "").lower()
        if "create" in request_type or "new" in request_type:
            if "create" not in tags:
                tags.append("create")
        # auth/login domain via request_type
        if any(k in request_type for k in ("login", "auth", "signup")):
            if "auth.login" not in tags:
                tags.append("auth.login")

    # ── Source 2: matched_semantic_keys (structured keys from semantic analysis)
    if matched_semantic_keys:
        _KEY_TAG_MAP = {
            "auth_endpoint": "auth.login",
            "token_generation": "auth.login",
            "password_handling": "auth.login",
            "user_model": "auth.login",
            "send_message_endpoint": "chat.send",
            "message_model": "chat.send",
            "upload_endpoint": "video.upload",
            "video_model": "video.upload",
            "product_model": "create",
            "cart_or_order": "create",
        }
        for key in matched_semantic_keys:
            tag = _KEY_TAG_MAP.get(key)
            if tag and tag not in tags:
                tags.append(tag)

    return tags
