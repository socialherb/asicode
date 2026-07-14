"""semantic_gap_analyzer.py — Phase D: Semantic Gap Extraction.

Analyzes remaining contract violations after C.2/C.3 and extracts
precise "what is missing" gaps that can be resolved by minimal generation.

Gap types:
- missing_entity: a model/class doesn't exist at all
- missing_persistence: create action lacks persist call
- missing_output_reference: return doesn't reference created entity
- missing_minimal_schema: no request/response schema
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.editor.semantic.semantic_contract_models import SemanticContractReport, SemanticViolation
from external_llm.editor.semantic.semantic_tracer import SemanticTrace

logger = logging.getLogger(__name__)


@dataclass
class SemanticGap:
    """A single semantic gap — something that must be generated."""
    gap_type: str          # "missing_entity" | "missing_persistence" | "missing_output_reference" | "missing_minimal_schema"
    entity: str            # Target entity name (e.g., "User", "Message")
    target_role: str       # "model" | "service" | "route" | "schema"
    reason: str            # Which contract/violation triggered this
    priority: int = 0      # Lower = higher priority (entity=0, schema=1, persist=2, output=3)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Context → Entity mapping ─────────────────────────────────────────────────

_TAG_ENTITY: dict[str, str] = {
    "auth.login": "User",
    "auth.authenticate": "User",
    "auth.signup": "User",
    "chat.send": "Message",
    "chat.send_message": "Message",
    "chat.create_message": "Message",
    "video.upload": "Video",
    "video.create": "Video",
    "create": "",        # Generic — determined from trace
    "upload": "",
    "send": "",
}

# Violation rule → gap type
_RULE_GAP_MAP: dict[str, str] = {
    "entity_creation": "missing_entity",
    "persistence": "missing_persistence",
    "output_must_reference_created_entity": "missing_output_reference",
    "entity_content_from_input": "missing_entity",
    "entity_fields_from_input": "missing_entity",
}


def extract_semantic_gaps(
    report: SemanticContractReport,
    trace: SemanticTrace,
    context_tags: Optional[list[str]] = None,
) -> list[SemanticGap]:
    """Extract semantic gaps from remaining contract violations.

    Only extracts gaps for "missing" type violations — not ordering/branch
    issues (those are C.2/C.3's domain).
    """
    gaps: list[SemanticGap] = []
    seen: set[str] = set()
    tags = context_tags or []

    # Determine expected entity from context
    expected_entity = _resolve_entity(tags, trace)

    for result in report.results:
        if result.passed:
            continue

        for violation in result.violations:
            gap = _violation_to_gap(violation, result.contract_name, expected_entity, trace, tags)
            if gap is None:
                continue

            # Dedup
            key = f"{gap.gap_type}:{gap.entity}"
            if key in seen:
                continue
            seen.add(key)

            # Skip if already satisfied in trace
            if _gap_already_satisfied(gap, trace):
                continue

            gaps.append(gap)

    # Sort by priority
    gaps.sort(key=lambda g: g.priority)

    if gaps:
        logger.info(
            "[GAP] extracted %d gaps: %s",
            len(gaps), [(g.gap_type, g.entity) for g in gaps],
        )

    return gaps


def _resolve_entity(tags: list[str], trace: SemanticTrace) -> str:
    """Determine the expected entity name from context."""
    for tag in tags:
        entity = _TAG_ENTITY.get(tag, "")
        if entity:
            return entity

    # Fallback: first class in trace
    if trace.all_classes:
        return next(iter(trace.all_classes))

    return "Item"


def _violation_to_gap(
    violation: SemanticViolation,
    contract_name: str,
    expected_entity: str,
    trace: SemanticTrace,
    tags: list[str],
) -> Optional[SemanticGap]:
    """Convert a violation to a semantic gap, if applicable."""
    rule = violation.rule
    rule_type = violation.rule_type

    # Only handle "requires" and "output" violations
    if rule_type not in ("requires", "output", "binding"):
        return None

    gap_type = _RULE_GAP_MAP.get(rule)
    if not gap_type:
        return None

    entity = expected_entity

    # Refine entity from contract context
    if "auth" in contract_name:
        entity = "User"
    elif "message" in contract_name or "chat" in contract_name:
        entity = "Message"
    elif "upload" in contract_name or "video" in contract_name:
        entity = "Video"
    elif "product" in contract_name or "ecommerce" in contract_name:
        entity = "Product"

    # Determine target role
    _GAP_ROLE = {
        "missing_entity": "model",
        "missing_persistence": "service_or_route",
        "missing_output_reference": "route",
        "missing_minimal_schema": "schema",
    }
    _GAP_PRIORITY = {
        "missing_entity": 0,
        "missing_minimal_schema": 1,
        "missing_persistence": 2,
        "missing_output_reference": 3,
    }

    return SemanticGap(
        gap_type=gap_type,
        entity=entity,
        target_role=_GAP_ROLE.get(gap_type, "model"),
        reason=f"contract:{contract_name}",
        priority=_GAP_PRIORITY.get(gap_type, 5),
    )


def _gap_already_satisfied(gap: SemanticGap, trace: SemanticTrace) -> bool:
    """Check if the gap is already satisfied by current code."""
    if gap.gap_type == "missing_entity":
        return gap.entity in trace.all_classes

    if gap.gap_type == "missing_persistence":
        return bool(trace.all_persist_calls)

    if gap.gap_type == "missing_output_reference":
        # Check if any function returns entity ref
        for ft in trace.function_traces.values():
            if ft.return_has_entity_ref:
                return True
        return False

    return False
