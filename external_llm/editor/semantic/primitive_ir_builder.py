"""primitive_ir_builder.py — Phase F: Build Primitive IR from Draft + Detection.

Combines draft_parser output, primitive_detector results, and existing
contract/flow data into a unified PrimitiveIR.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.editor.semantic.draft_parser import parse_draft
from external_llm.editor.semantic.primitive_detector import detect_primitives
from external_llm.editor.semantic.primitive_models import PrimitiveIR, PrimitiveSequence
from external_llm.editor.semantic.semantic_tracer import SemanticTrace

logger = logging.getLogger(__name__)


def build_primitive_ir(
    file_paths: list[str],
    repo_root: str = ".",
    trace: Optional[SemanticTrace] = None,
    contract_report: Any = None,
    context_tags: Optional[list[str]] = None,
    scope: Optional[set[str]] = None,
) -> PrimitiveIR:
    """Build PrimitiveIR from files.

    1. Parse draft (actions, entities, roles)
    2. Detect primitives for each action
    3. Assemble into PrimitiveIR

    When ``scope`` is provided, only actions whose bare name is in the set
    are analysed — their primitive sequences shape the IR and its coverage.
    Actions outside the scope are skipped entirely so that Phase F's
    ``overall_coverage`` reflects only the change surface the plan actually
    touches (plus call-graph neighbours). ``scope=None`` preserves the legacy
    behaviour of analysing every action found in ``file_paths``.
    """
    # Build trace if not provided
    if trace is None:
        from external_llm.editor.semantic.semantic_tracer import extract_trace_from_files
        trace = extract_trace_from_files(file_paths, repo_root)

    # Parse draft
    draft = parse_draft(file_paths, repo_root, trace, context_tags)

    # ── context_tags as primary gate ───────────────────────────────────────
    # infer_action_type() is a WEAK PRIOR derived from function names.
    # spec.request_type (reflected in context_tags via extract_context_tags)
    # is the AUTHORITATIVE signal.  We only apply non-trivial action contracts
    # (create/upload/send/delete/update/login) when context_tags actually
    # confirms that domain — otherwise a utility function like add_member()
    # or new_parser() would trigger entity-creation contract checking on a
    # pure modify task, recreating a keyword-gate false positive.
    _tags_set = set(context_tags or [])
    # Map: action_type → the context tag that must be present to confirm it
    _ACTION_CONFIRMATION: dict[str, str] = {
        "create": "create",
        "login": "auth.login",
        "upload": "upload",
        "send": "send",
    }
    # Detect primitives for each action
    sequences: list[PrimitiveSequence] = []
    scope_skipped = 0
    for action in draft.actions:
        # Scope gate: when the caller restricts Phase F to plan-relevant
        # symbols, actions outside that scope never enter the IR.  This
        # prevents overall_coverage and missing-primitive learning signals
        # from being diluted by unrelated helpers in the same file.
        if scope is not None and action.name not in scope:
            scope_skipped += 1
            continue
        # Skip trivial/internal functions
        if action.action_type == "unknown" and not action.has_decorator:
            continue
        # Gate: action types that carry heavy contract requirements (create/
        # login/upload/send) are only analyzed when context_tags confirms the
        # domain.  Without confirmation, infer_action_type is too weak a signal
        # to justify applying those contracts.
        required_ctx_tag = _ACTION_CONFIRMATION.get(action.action_type)
        if required_ctx_tag and required_ctx_tag not in _tags_set:
            logger.debug(
                "[PRIM_IR_GATE] skipping action '%s' (type=%s): context_tags %s "
                "does not confirm required tag '%s' — infer_action_type is hint only",
                action.name, action.action_type, list(_tags_set), required_ctx_tag,
            )
            continue
        seq = detect_primitives(action, trace, contract_report)
        sequences.append(seq)

    if scope_skipped:
        logger.debug(
            "[PRIM_IR] scope gate skipped %d out-of-scope actions", scope_skipped,
        )

    ir = PrimitiveIR(
        sequences=sequences,
        context_tags=list(context_tags or draft.context_tags),
        entities=list(draft.entities),
    )

    logger.info(
        "[PRIM_IR] %d sequences, coverage=%.2f, missing=%s",
        len(sequences), ir.overall_coverage, ir.all_missing,
    )

    return ir
