"""primitive_reconstructor.py — Phase F: Primitive-Based Reconstruction.

Takes a PrimitiveIR with missing primitives and produces a
ReconstructionCandidate by orchestrating existing repair engines
(Phase C.2 insert, C.3 rewrite, D fragment generation).

Key principle: dispatch by PRIMITIVE NAME, not domain name.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.editor.semantic.primitive_models import PrimitiveIR, PrimitiveSequence, ReconstructionCandidate

logger = logging.getLogger(__name__)


def reconstruct_from_ir(
    ir: PrimitiveIR,
    repo_root: str,
    trace: Any = None,
    file_paths: Optional[list[str]] = None,
    context_tags: Optional[list[str]] = None,
    learning_store: Any = None,
    context_bucket: str = "",
    sequence_store: Any = None,
) -> ReconstructionCandidate:
    """Reconstruct missing primitives using existing engines.

    Dispatches by primitive name to the appropriate repair mechanism.
    Returns a candidate with patches/fragments that could fill gaps.
    """
    # Build trace from sequence file paths if not provided
    if trace is None:
        from external_llm.editor.semantic.semantic_tracer import extract_trace_from_files
        _files = file_paths or [s.file_path for s in ir.sequences if s.file_path]
        trace = extract_trace_from_files(_files, repo_root)

    candidate = ReconstructionCandidate()

    if ir.overall_coverage >= 0.95:
        candidate.confidence = 1.0
        candidate.primitive_coverage_estimate = ir.overall_coverage
        candidate.notes.append("coverage sufficient, no reconstruction needed")
        return candidate

    # ── Dependency-aware ordering ─────────────────────────────────
    # Primitives must be filled in dependency order.
    # e.g., create_entity before persist_state before produce_output
    _FILL_ORDER = [
        "lookup",
        "validate",
        "branch_on_failure",
        "authorize",
        "create_entity",
        "input_bind",
        "update_entity",
        "delete_entity",
        "persist_state",
        "list_or_query",
        "produce_output",
        "delegate_action",
    ]
    _ORDER_MAP = {name: i for i, name in enumerate(_FILL_ORDER)}

    # ── Dependency graph: primitive A requires B to exist ─────────
    _DEPENDENCIES: dict[str, list[str]] = {
        "persist_state": ["create_entity"],      # Must create before persisting
        "produce_output": ["create_entity"],      # Output references created entity
        "branch_on_failure": ["validate"],         # Branch needs validation check
    }

    # Collect all unique missing primitives across sequences, deduped per (action, primitive)
    applied: set[str] = set()          # "action:primitive" dedup key
    global_filled: set[str] = set()    # primitive names filled globally (for dedup reporting)

    # Build learned priority map if store available
    _learned_priority: dict[str, float] = {}
    if learning_store and context_bucket:
        try:
            from external_llm.editor.learning.primitive_learning_scorer import score_missing_primitives
            [m.primitive for s in ir.sequences for m in s.missing]
            for seq in ir.sequences:
                scored = score_missing_primitives(
                    learning_store, context_bucket, seq.action_type,
                    [m.primitive for m in seq.missing], seq.entity,
                )
                for prim, score in scored:
                    _learned_priority[prim] = max(_learned_priority.get(prim, 0), score)
        except Exception:
            pass

    # Build sequence-recommended order if store available
    _seq_recommended: dict[str, list[str]] = {}
    if sequence_store:
        try:
            from external_llm.editor.learning.primitive_sequence_scorer import recommend_sequence
            for seq in ir.sequences:
                missing_names = [m.primitive for m in seq.missing]
                if missing_names:
                    recommended = recommend_sequence(sequence_store, seq.action_type, missing_names)
                    _seq_recommended[seq.action_name] = recommended
        except Exception:
            pass

    for seq in ir.sequences:
        # 3-layer ordering: dependency → sequence → learned priority
        _seq_order = _seq_recommended.get(seq.action_name, [])
        _seq_rank = {name: i for i, name in enumerate(_seq_order)}

        ordered_missing = sorted(
            seq.missing,
            key=lambda m: (
                _ORDER_MAP.get(m.primitive, 99),           # L1: dependency safety
                _seq_rank.get(m.primitive, 99),            # L2: sequence pattern
                -_learned_priority.get(m.primitive, 0.5),  # L3: learned priority
            ),
        )

        for miss in ordered_missing:
            key = f"{seq.action_name}:{miss.primitive}"
            if key in applied:
                continue

            # Check dependency: skip if dependency not present and not being filled
            deps = _DEPENDENCIES.get(miss.primitive, [])
            dep_satisfied = True
            for dep in deps:
                dep_present = any(p.primitive == dep for p in seq.present)
                dep_filled = f"{seq.action_name}:{dep}" in applied
                if not dep_present and not dep_filled:
                    dep_satisfied = False
                    break

            if not dep_satisfied:
                continue

            repair = _dispatch_repair(miss.primitive, seq, ir, repo_root, trace)
            if repair:
                candidate.patches.extend(repair.get("patches", []))
                candidate.fragments.extend(repair.get("fragments", []))
                # Only add to applied_primitives if not already globally filled
                if miss.primitive not in global_filled:
                    candidate.applied_primitives.append(miss.primitive)
                    global_filled.add(miss.primitive)
                candidate.notes.append(
                    f"{miss.primitive} → {repair.get('method', 'unknown')} for {seq.action_name}"
                )
                applied.add(key)

    # Estimate coverage improvement
    total_required = sum(len(s.present) + len(s.missing) for s in ir.sequences)
    filled = len(applied)  # Count unique action:primitive pairs
    original_present = sum(len(s.present) for s in ir.sequences)
    candidate.primitive_coverage_estimate = (
        (original_present + filled) / total_required if total_required > 0 else 1.0
    )
    candidate.confidence = min(0.9, candidate.primitive_coverage_estimate)

    if candidate.applied_primitives:
        logger.info(
            "[RECONSTRUCT] %d primitives filled: %s, coverage=%.2f→%.2f",
            filled, candidate.applied_primitives,
            ir.overall_coverage, candidate.primitive_coverage_estimate,
        )

    return candidate


# ── Primitive → Repair Dispatch ───────────────────────────────────────────────

def _dispatch_repair(
    prim_name: str,
    seq: PrimitiveSequence,
    ir: PrimitiveIR,
    repo_root: str,
    trace: Any,
) -> Optional[dict[str, Any]]:
    """Route a missing primitive to the appropriate repair mechanism."""
    dispatch = {
        "create_entity": _repair_create_entity,
        "persist_state": _repair_persist_state,
        "produce_output": _repair_produce_output,
        "branch_on_failure": _repair_branch_on_failure,
        "validate": _repair_validate,
        "lookup": _repair_lookup,
        "authorize": _repair_authorize,
        "input_bind": _repair_input_bind,
    }

    handler = dispatch.get(prim_name)
    if handler:
        return handler(seq, ir, repo_root, trace)
    return None


def _repair_create_entity(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """Missing create_entity → use Phase D fragment generation."""
    entity = seq.entity or (ir.entities[0] if ir.entities else "Item")
    try:
        from external_llm.editor.semantic.fragment_integrator import integrate_fragments
        from external_llm.editor.semantic.generation_planner import build_generation_plan
        from external_llm.editor.semantic.semantic_gap_analyzer import SemanticGap

        gap = SemanticGap(
            gap_type="missing_entity",
            entity=entity,
            target_role="model",
            reason=f"primitive:create_entity for {seq.action_name}",
        )

        if trace is None:
            from external_llm.editor.semantic.semantic_tracer import extract_trace_from_files
            trace = extract_trace_from_files([], repo_root)

        frags = build_generation_plan([gap], trace, repo_root)
        if frags:
            result = integrate_fragments(frags, repo_root)
            return {
                "method": "phase_d_entity_generation",
                "fragments": result.get("applied", []),
                "patches": [],
            }
    except Exception as e:
        logger.debug("create_entity repair failed: %s", e)
    return None


def _repair_persist_state(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """DISABLED (Phase 5) — was injecting persistence code."""
    return None


def _repair_produce_output(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """DISABLED (Phase 5) — was injecting return entity code."""
    return None


def _repair_branch_on_failure(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """DISABLED (Phase 5) — was injecting HTTPException guards."""
    return None


def _repair_validate(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """DISABLED (Phase 5) — was injecting verify_password calls."""
    return None


def _repair_lookup(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """DISABLED (Phase 5) — was injecting get_user calls."""
    return None


def _repair_authorize(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """Missing authorize → token generation is typically present via other repairs.
    This is a placeholder for future token flow insertion."""
    return None


def _repair_input_bind(seq: PrimitiveSequence, ir: PrimitiveIR, repo_root: str, trace: Any) -> Optional[dict]:
    """Missing input_bind → entity creation with params usually handles this.
    Covered by create_entity repair."""
    return None
