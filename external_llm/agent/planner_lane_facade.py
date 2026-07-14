"""Single choke-point for the (permanently-disabled) PLANNER lane.

Design intent — BLAST-RADIUS ISOLATION
--------------------------------------
The PLANNER lane (``external_llm/editor/_editor_core/lane/``) is permanently
disabled at routing time: ``task_router.DeterministicClassifier.decide_flow``
always returns ``MAIN_AGENT`` (see its module docstring and the comment at the
end of ``_build_*_decision``). The lane code is therefore dead at runtime but
still imported by ``agent_loop.py`` at module top-level for two symbols:

* ``PlannerAgent``        (lane.planner_agent)
* ``OperationExecutor``   (lane.operation_executor)

Because those two imports are top-level, deleting the lane directory in a
future cleanup would break ``import external_llm.agent.agent_loop`` — and
therefore break ``design_chat_loop`` and every live agent entry point.

This module concentrates the entire lane dependency surface into ONE place so
that a future "delete the PLANNER lane" change is a single-file edit here (or
 outright deletion of this facade) instead of a scattered multi-file fix.

Behaviour
---------
* On successful lane import → the real symbols are re-exported unchanged.
* On lane import failure (``ImportError``) → both symbols resolve to ``None``.
  ``AgentLoop._init_hybrid_components`` already wraps instantiation in a
  ``try/except`` with a ``_hybrid_init_failed`` guard, so a ``None`` symbol
  degrades gracefully to the MAIN_AGENT-only path instead of crashing import.
* Type hints in ``agent_loop`` are safe under ``from __future__ import
  annotations`` (never evaluated at runtime), so ``Optional[OperationExecutor]``
remains valid even when ``OperationExecutor`` is ``None`` at runtime.

Do NOT import lane symbols directly from ``agent_loop`` (or any other live
module). Import them from here instead.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from external_llm.editor._editor_core.lane.operation_executor import OperationExecutor  # type: ignore
    from external_llm.editor._editor_core.lane.planner_agent import PlannerAgent  # type: ignore
    _LANE_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - exercised only when lane is removed
    PlannerAgent = None  # type: ignore[assignment,misc]
    OperationExecutor = None  # type: ignore[assignment,misc]
    _LANE_AVAILABLE = False
    logger.debug(
        "PLANNER lane unavailable (permanently disabled / removed): %s. "
        "MAIN_AGENT-only path in effect.",
        _exc,
    )


__all__ = ["OperationExecutor", "PlannerAgent", "is_planner_lane_available"]


def is_planner_lane_available() -> bool:
    """Return ``True`` iff the (disabled) PLANNER lane can still be imported.

    Live code should not branch on this for behaviour; it exists for diagnostics
    and tests. The routing layer guarantees PLANNER is never selected at
    runtime regardless of this flag.
    """
    return _LANE_AVAILABLE
