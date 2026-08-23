"""
Fast-path detection and execution mixin for AgentLoop.

Extracted from agent_loop.py to keep that file manageable.
AgentLoop inherits FastPathMixin, so all methods have full access to
self.config, self.registry, etc.
"""

from __future__ import annotations

from typing import Any  # f821-protected


def _is_trivial_request(request: str) -> bool:
    """Lightweight triviality check — fallback when RouteDecision unavailable.

    Uses keyword membership (no regex). RouteDecision.task_kind==MICRO_EDIT
    is the primary decision path — this only runs as safety net.
    """
    req = str(request or "").strip().lower()
    if not req:
        return False

    # Variable-name-like requests (e.g. "_max_tokens", "400_000")
    if (
        len(req) < 40
        and "_" in req
        and req.replace("_", "").isalnum()
        and any(c.isascii() and c.isalpha() for c in req)
    ):
        return True  # looks like a code reference

    # Trivial edit patterns (set-based, no regex)
    _trivial_triggers = {"header", "typo", "spelling", "rename"}
    words = set(req.split())
    if _trivial_triggers & words:
        return True

    # "only change X" / "constant" — 2+ word phrases
    if "only change" in req or "only modify" in req:
        return True
    return "constant" in req and len(req.split()) < 10


class FastPathMixin:
    """
    Mixin providing fast-path detection and execution for AgentLoop.

    Requires the host class to expose:
      - self.config       (AgentConfig)
      - self.registry     (ToolRegistry)
      - AgentTurn / ToolResult types (imported in agent_loop.py)
    """

    # Host-class attributes (provided by AgentLoop, not set here). Pure typing
    # scaffolding — AgentLoop.__init__ owns the runtime values.
    config: Any
    registry: Any

    # ------------------------------------------------------------------
    # Trivial / read-only classifiers
    # ------------------------------------------------------------------

    def _is_trivial_edit_request(self, request: str) -> bool:
        """True when request is trivial enough to skip planning and self-review.

        Primary: RouteDecision.task_kind == MICRO_EDIT (LLM-based, language-neutral).
        Fallback: regex check when no route decision is available.
        """
        route = getattr(self.config, "route_decision", None)
        if route is not None:
            from .task_router import TaskKind

            return getattr(route, "task_kind", None) == TaskKind.MICRO_EDIT
        return _is_trivial_request(request)
