"""handoff_observer.py — Planner→Developer handoff observability.

Tracks the causal chain: intent → enrichment → developer request → execution result.
All functions are pure logging — no execution logic changes.

Events:
  [PLANNER_HANDOFF]    — planner → developer state just before handoff
  [DEV_REQUEST]        — final payload delivered to developer
  [DEV_RESULT]         — developer execution result
  [SYMBOL_RESOLUTION]  — symbol existence + action taken
  [INTENT_DIFF]        — transformation/loss tracking during enrichment
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level accumulator for per-run JSON export. Bounded by a deque cap:
# the log_* writers fire on EVERY editor operation (create_file, symbol-modify,
# auto-correction, …) in the shared editor lane — including the long-lived
# webapp server path — yet reset_events()/get_events() have NO callers, so an
# unbounded list accumulated every operation's handoff across all runs for the
# whole process lifetime (a slow OOM under a server processing many edits).
# The deque retains the most-recent events for potential inspection while
# bounding memory at O(maxlen). get_events()/reset_events() are deque-compatible
# (list(deque) iterates; deque.clear() mutates in place), so existing callers
# and tests are unaffected.
_RUN_EVENTS_MAX = 1000
_run_events: deque = deque(maxlen=_RUN_EVENTS_MAX)


def reset_events() -> None:
    """Clear accumulated events (call at run start)."""
    _run_events.clear()


def get_events() -> list[dict[str, Any]]:
    """Return accumulated events for JSON serialization."""
    return list(_run_events)


# ── 1. PLANNER_HANDOFF ──────────────────────────────────────────────────

def log_planner_handoff(
    op_id: str,
    kind: str,
    path: str,
    symbol: str,
    original_intent: str,
    enriched_intent: str,
    final_op_intent: str,
    change_spec_summary: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log the planner→developer handoff state."""
    event = {
        "event": "planner_handoff",
        "op_id": op_id,
        "kind": kind,
        "path": path,
        "symbol": symbol,
        "original_intent": original_intent,
        "enriched_intent": enriched_intent,
        "final_op_intent": final_op_intent,
        "change_spec_summary": change_spec_summary or [],
        "metadata": metadata or {},
    }
    _run_events.append(event)

    # Console summary
    _orig_preview = _truncate(original_intent, 80)
    _enrich_preview = _truncate(enriched_intent, 120)
    _final_preview = _truncate(final_op_intent, 120)
    logger.info(
        "[PLANNER_HANDOFF] op=%s kind=%s path=%s symbol=%s\n"
        "  original_intent: %s\n"
        "  enriched_intent: %s\n"
        "  final_op_intent: %s\n"
        "  change_spec: %s",
        op_id, kind, path, symbol,
        _orig_preview, _enrich_preview, _final_preview,
        change_spec_summary or "(none)",
    )


# ── 2. DEV_REQUEST ──────────────────────────────────────────────────────

def log_developer_request(
    op_id: str,
    effective_intent: str,
    handler: str,
    execution_mode: str,
    file_exists: bool,
    symbol_exists: bool,
    model: str = "",
    provider: str = "",
    context_summary: Optional[dict[str, Any]] = None,
) -> None:
    """Log what the developer LLM actually receives."""
    event = {
        "event": "developer_request_built",
        "op_id": op_id,
        "effective_intent": effective_intent,
        "model": model,
        "provider": provider,
        "handler": handler,
        "execution_mode": execution_mode,
        "file_exists": file_exists,
        "symbol_exists": symbol_exists,
        "context_summary": context_summary or {},
    }
    _run_events.append(event)

    _intent_preview = _truncate(effective_intent, 150)
    _ctx = context_summary or {}
    logger.info(
        "[DEV_REQUEST] op=%s handler=%s mode=%s file_exists=%s symbol_exists=%s\n"
        "  effective_intent: %s\n"
        "  context: file=%d lines, symbol=%d lines, ~%d tokens",
        op_id, handler, execution_mode, file_exists, symbol_exists,
        _intent_preview,
        _ctx.get("file_lines", 0),
        _ctx.get("symbol_lines", 0),
        _ctx.get("prompt_tokens_estimate", 0),
    )


# ── 3. DEV_RESULT ───────────────────────────────────────────────────────

def log_developer_result(
    op_id: str,
    status: str,
    strategy: str,
    changed_lines: int = 0,
    content_length: int = 0,
    failure_class: str = "",
    failure_reason: str = "",
    retry_count: int = 0,
    created_file: bool = False,
) -> None:
    """Log developer execution result."""
    event = {
        "event": "developer_result_summary",
        "op_id": op_id,
        "status": status,
        "strategy": strategy,
        "changed_lines": changed_lines,
        "content_length": content_length,
        "created_file": created_file,
        "failure": {
            "class": failure_class,
            "reason": failure_reason,
            "retry_count": retry_count,
        },
    }
    _run_events.append(event)

    if status == "success":
        if created_file:
            logger.info(
                "[DEV_RESULT] op=%s status=success strategy=%s created_file=True content=%d chars",
                op_id, strategy, content_length,
            )
        else:
            logger.info(
                "[DEV_RESULT] op=%s status=success strategy=%s changed=%d lines",
                op_id, strategy, changed_lines,
            )
    else:
        logger.info(
            "[DEV_RESULT] op=%s status=%s strategy=%s failure=%s reason=%s retries=%d",
            op_id, status, strategy, failure_class,
            _truncate(failure_reason, 80), retry_count,
        )


# ── 4. SYMBOL_RESOLUTION ────────────────────────────────────────────────

def log_symbol_resolution(
    op_id: str,
    symbol: str,
    exists: bool,
    resolution_strategy: str,
    action: str,
) -> None:
    """Log symbol existence check and resolution action."""
    event = {
        "event": "symbol_resolution",
        "op_id": op_id,
        "symbol": symbol,
        "exists": exists,
        "resolution_strategy": resolution_strategy,
        "action": action,
    }
    _run_events.append(event)

    logger.info(
        "[SYMBOL_RESOLUTION] op=%s symbol=%s exists=%s strategy=%s action=%s",
        op_id, symbol, exists, resolution_strategy, action,
    )


# ── 5. INTENT_DIFF ──────────────────────────────────────────────────────

def log_intent_diff(
    op_id: str,
    lost_items: Optional[list[str]] = None,
    added_items: Optional[list[str]] = None,
    preserved: Optional[list[str]] = None,
) -> None:
    """Log what changed between enriched intent and developer effective intent."""
    event = {
        "event": "intent_diff",
        "op_id": op_id,
        "lost_items": lost_items or [],
        "added_items": added_items or [],
        "preserved": preserved or [],
    }
    _run_events.append(event)

    if lost_items:
        logger.warning(
            "[INTENT_DIFF] op=%s LOST=%s added=%s preserved=%d items",
            op_id, lost_items, added_items or [], len(preserved or []),
        )
    else:
        logger.debug(
            "[INTENT_DIFF] op=%s no loss. added=%s preserved=%d items",
            op_id, added_items or [], len(preserved or []),
        )


# ── 6. OP_AUTO_CORRECT ──────────────────────────────────────────────────

def log_auto_correction(
    op_id: str,
    original_kind: str,
    original_symbol: str,
    corrected_kind: str,
    corrected_symbol: str = "",
    corrected_anchor: str = "",
    action: str = "keep",
    rationale: str = "",
    confidence: float = 1.0,
    resolution_facts: Optional[dict[str, Any]] = None,
) -> None:
    """Log operation auto-correction decision."""
    event = {
        "event": "op_auto_correct",
        "op_id": op_id,
        "original_kind": original_kind,
        "original_symbol": original_symbol,
        "corrected_kind": corrected_kind,
        "corrected_symbol": corrected_symbol,
        "corrected_anchor": corrected_anchor,
        "action": action,
        "rationale": rationale,
        "confidence": confidence,
        "resolution_facts": resolution_facts or {},
    }
    _run_events.append(event)

    if action == "keep":
        logger.debug(
            "[OP_AUTO_CORRECT] op=%s action=keep kind=%s symbol=%s",
            op_id, original_kind, original_symbol,
        )
    else:
        logger.info(
            "[OP_AUTO_CORRECT] op=%s original=%s symbol=%s "
            "corrected=%s action=%s rationale=%s confidence=%.2f",
            op_id, original_kind, original_symbol,
            corrected_kind, action, rationale, confidence,
        )


# ── Helpers ──────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text for console display."""
    if not text:
        return "(empty)"
    text = text.replace("\n", " ↵ ")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def compute_intent_diff(
    enriched_intent: str,
    effective_intent: str,
) -> dict[str, list[str]]:
    """Compute what was lost/added between enriched and effective intent.

    Uses keyword extraction: check if key phrases from enriched
    are present in effective.
    """
    _MARKERS = [
        "📌 CHANGE CONTRACT",
        "📌 ROUTER WIRING",
        "📌 AUTH FLOW",
        "📌 MODIFY CHECKLIST",
        "⚠️ MANDATORY",
        "COMPLETENESS RULE",
        "add_field", "add_validation", "rewrite_condition",
        "include_router", "verify_password", "create_access_token",
        "__post_init__", "len(", ".lower()",
    ]

    lost = []
    preserved = []
    for marker in _MARKERS:
        if marker in enriched_intent:
            if marker in effective_intent:
                preserved.append(marker)
            else:
                lost.append(marker)

    added = []
    for marker in _MARKERS:
        if marker not in enriched_intent and marker in effective_intent:
            added.append(marker)

    return {"lost": lost, "added": added, "preserved": preserved}
