"""primitive_learning_updater.py — Phase F.1: Update Primitive Learning from Execution.

Takes Phase F results + final verdict and updates the primitive learning store.
"""
from __future__ import annotations

import logging
from typing import Any

from external_llm.editor.learning.primitive_learning_models import PrimitiveLearningKey
from external_llm.editor.learning.primitive_learning_store import PrimitiveLearningStore

logger = logging.getLogger(__name__)


def update_primitive_learning(
    store: PrimitiveLearningStore,
    primitive_ir_summary: dict[str, Any],
    reconstruction_meta: dict[str, Any],
    semantic_before: float,
    semantic_after: float,
    contract_before: float,
    contract_after: float,
    final_pass: bool,
    context_bucket: str,
) -> dict[str, Any]:
    """Update primitive learning store from one execution.

    Returns summary of what was updated.
    """
    result = {
        "updated": False,
        "context_bucket": context_bucket,
        "records_updated": 0,
        "coverage_delta": 0.0,
        "semantic_delta": 0.0,
        "contract_delta": 0.0,
    }

    if not reconstruction_meta.get("attempted"):
        return result

    chosen = reconstruction_meta.get("chosen", "raw")
    is_chosen = chosen == "reconstructed"
    raw_cov = reconstruction_meta.get("raw_coverage", 0.0)
    recon_cov = reconstruction_meta.get("reconstructed_coverage", 0.0)
    coverage_delta = recon_cov - raw_cov
    improved = coverage_delta > 0.01
    sem_delta = semantic_after - semantic_before
    contract_delta = contract_after - contract_before

    result["coverage_delta"] = round(coverage_delta, 4)
    result["semantic_delta"] = round(sem_delta, 4)
    result["contract_delta"] = round(contract_delta, 4)

    # Extract sequences from IR summary
    sequences = primitive_ir_summary.get("sequences", [])
    missing_all = reconstruction_meta.get("missing_primitives", [])
    filled = reconstruction_meta.get("applied_primitives", [])
    filled_set = set(filled)

    # Extract strategy mapping from reconstruction notes
    strategy_map = _extract_strategy_map(reconstruction_meta)

    count = 0
    for seq_info in sequences:
        action_type = seq_info.get("type", "unknown")
        entity = seq_info.get("entity", "")
        missing_prims = seq_info.get("missing", [])

        for prim in missing_prims:
            key = PrimitiveLearningKey(
                context_bucket=context_bucket,
                action_type=action_type,
                primitive=prim,
                entity=entity,
            )

            was_filled = prim in filled_set
            strategy_name = strategy_map.get(prim, "")

            store.update(
                key=key,
                chosen=is_chosen and was_filled,
                improved=improved and was_filled,
                passed=final_pass,
                coverage_delta=coverage_delta / max(len(missing_all), 1) if was_filled else 0.0,
                sem_delta=sem_delta / max(len(missing_all), 1) if was_filled else 0.0,
                contract_delta=contract_delta / max(len(missing_all), 1) if was_filled else 0.0,
                strategy_name=strategy_name,
            )
            count += 1

    result["updated"] = count > 0
    result["records_updated"] = count

    if count > 0:
        logger.info(
            "[PRIM_LEARN] updated %d records, ctx=%s, chosen=%s, "
            "coverage=%.2f→%.2f, pass=%s",
            count, context_bucket, chosen,
            raw_cov, recon_cov, final_pass,
        )

    return result


def _extract_strategy_map(meta: dict[str, Any]) -> dict[str, str]:
    """Extract primitive→strategy_name mapping from reconstruction metadata."""
    # Parse from the primitive_ir summary or notes
    strategy_map: dict[str, str] = {}

    ir = meta.get("primitive_ir", {})
    for seq in ir.get("sequences", []):
        for prim in seq.get("missing", []):
            # Default strategy names based on primitive
            _DEFAULT_STRATEGIES = {
                "validate": "c2_insert_verify_call",
                "lookup": "c2_insert_user_lookup",
                "branch_on_failure": "c2_insert_error_branch",
                "persist_state": "c2_insert_persistence",
                "create_entity": "d_fragment_generation",
                "produce_output": "c2_fix_return_entity",
                "authorize": "c2_insert_token",
            }
            strategy_map.setdefault(prim, _DEFAULT_STRATEGIES.get(prim, ""))

    return strategy_map
