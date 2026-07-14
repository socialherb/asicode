"""
GSG Safety helpers — lightweight, focused GSG execution policy.

GSG (Global Symbol Graph) is a hypothesis generator, not ground truth.
This module provides:
  - classify_graph_confidence()   — confidence → policy mode
  - build_gsg_execution_policy()  — graph_context → actionable policy dict


Thresholds are imported from self_planning_policy to avoid duplication.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

from .execution_thresholds import THRESHOLDS as _THRESHOLDS

# Removed: GRAPH_BLOCKED_CONFIDENCE_THRESHOLD (blocked mode eliminated)

# Operation kinds treated as structural / risky (graph-dependent targeting)
STRUCTURAL_OP_KINDS = frozenset({
    "MODIFY_SYMBOL",
    "modify_symbol",
    "INSERT_AFTER_SYMBOL",
    "insert_after_symbol",
    "RENAME_SYMBOL",
    "MOVE_SYMBOL",
    "replace_symbol_body",
    "UPDATE_CALLERS",
    "update_callers",
})


def classify_graph_confidence(conf: float, unresolved_count: int = 0) -> str:
    """Classify graph confidence into a safety mode.

    Policy ladder (highest risk → lowest):
      "conservative" — conf < 0.4 OR unresolved >= 2  → enforce anchor read
      "guarded"      — 0.4 <= conf < 0.8             → anchor read recommended
      "trusted"      — conf >= 0.8 AND unresolved == 0 → proceed normally

    Note: "blocked" mode was removed — pre-edit blocking is redundant with
    existing defense layers (read_symbol not-found, validate_gsg_anchor,
    post-apply sanity check, acceptance criteria).
    """
    if conf < _THRESHOLDS.graph_low_confidence or unresolved_count >= _THRESHOLDS.graph_unresolved_symbol_count:
        return "conservative"
    if conf < _THRESHOLDS.graph_high_confidence:
        return "guarded"
    return "trusted"


def build_gsg_execution_policy(
    graph_context: Optional[dict[str, Any]],
    is_structural_op: bool = False,
) -> dict[str, Any]:
    """Build an actionable execution policy from graph context.

    When graph_context is None (graph not available), returns a trusted policy
    so existing non-graph paths are completely unaffected.

    Args:
        graph_context:   dict from spec.metadata["graph_context"], or None
        is_structural_op: True for modify_symbol, insert_after_symbol, etc.

    Returns dict with:
        mode                  — "trusted"|"guarded"|"conservative"|"blocked"
        requires_anchor_read  — True if anchor verification must pass before edit
        force_conservative_mode — True to use conservative repair strategy
        block_structural_edit — True to abort structural edit immediately
        fallback_reason       — non-None string when block_structural_edit is True
        graph_confidence      — raw float from graph_context
        unresolved_count      — raw int from graph_context
    """
    if not graph_context:
        # No graph data → preserve existing behavior, no regression
        return {
            "mode": "trusted",
            "requires_anchor_read": False,
            "force_conservative_mode": False,
            "block_structural_edit": False,
            "fallback_reason": None,
            "graph_confidence": 0.0,
            "unresolved_count": 0,
        }

    conf = float(graph_context.get("graph_confidence", 0.0))
    unresolved = len(graph_context.get("unresolved_symbols", []))
    mode = classify_graph_confidence(conf, unresolved)

    requires_anchor_read = is_structural_op and mode in ("guarded", "conservative")
    force_conservative = mode == "conservative"

    if requires_anchor_read:
        logger.debug(
            "GSG policy: mode=%s conf=%.2f unresolved=%d requires_anchor=%s",
            mode, conf, unresolved, requires_anchor_read,
        )

    return {
        "mode": mode,
        "requires_anchor_read": requires_anchor_read,
        "force_conservative_mode": force_conservative,
        "block_structural_edit": False,  # never block — rely on downstream defenses
        "fallback_reason": None,
        "graph_confidence": conf,
        "unresolved_count": unresolved,
    }



