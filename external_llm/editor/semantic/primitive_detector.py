"""primitive_detector.py — Phase F: Primitive Presence/Absence Detection.

For each action, detects which of the 12 semantic primitives are
present or missing based on trace data and code analysis.

Detection is conservative: uncertain → missing.

Signal sets are derived from ``primitive_registry.typical_signals`` so that
adding a signal in the registry automatically propagates to detection.
"""
from __future__ import annotations
from typing import Any, Optional

from external_llm.editor.semantic.draft_parser import DraftAction
from external_llm.editor.semantic.primitive_models import PrimitiveMatch, PrimitiveSequence
from external_llm.editor.semantic.primitive_registry import (
    ALL_PRIMITIVES,
    PRIMITIVE_MAP,
    get_required_primitives,
)
from external_llm.editor.semantic.semantic_tracer import FunctionTrace, SemanticTrace
# ── Signal extraction helpers ────────────────────────────────────────────────
# Registry typical_signals contain pattern-like strings (e.g. "raise.*ValueError",
# ".delete(", "ClassName(").  We extract clean keywords suitable for substring /
# exact matching in the detectors.



def _extract_signal_keywords(primitive_name: str) -> frozenset[str]:
    """Return a frozenset of lowercase keywords derived from a primitive's typical_signals.

    Rules:
    - Signals containing regex metacharacters (``.*``) or starting with
      ``raise``/``return`` are skipped (structural/regex patterns).
    - Compound patterns like ``db.add`` are split on ``.`` and each token is
      cleaned; multi-word tokens (with spaces or ``=``) are dropped.
    - Leading dots, trailing parens/brackets/underscores are stripped.
    - The result is a flat set of atomic keywords suitable for exact or
      substring matching.
    """
    prim = PRIMITIVE_MAP.get(primitive_name)
    if not prim:
        return frozenset()
    keywords: set[str] = set()
    for sig in prim.typical_signals:
        # Skip regex-like patterns (e.g. "raise.*ValueError", "return.*entity")
        if ".*" in sig or sig.startswith("raise") or sig.startswith("return"):
            continue
        # Skip structural code patterns with spaces/assignments
        if " " in sig.strip() and "=" in sig:
            continue
        # For compound patterns like "db.add", "session.add", "query.all()",
        # only the action part (last segment) is a useful signal keyword.
        # For qualifier-only patterns like "service.", "handler." (trailing dot),
        # the qualifier itself is the keyword.
        # For simple tokens like "cache", "throttle", keep as-is.
        stripped = sig.strip()
        if "." in stripped:
            segments = stripped.split(".")
            # Filter empty segments (from trailing dots like "service.")
            non_empty = [s for s in segments if s.strip().strip("([])").rstrip("(").strip()]
            if non_empty:
                # Use only the last meaningful segment
                parts = [non_empty[-1]]
            else:
                continue
        else:
            parts = [stripped]
        for part in parts:
            # Strip leading/trailing punctuation: parens, brackets
            clean = part.strip().strip("([])").rstrip("(").strip()
            if not clean or "=" in clean or " " in clean:
                continue
            keywords.add(clean.lower())
    return frozenset(keywords)


# Receiver prefixes that indicate persistence layer vs. cache layer.
# Used to disambiguate `obj.delete()` — a cache receiver is NOT delete_entity.
_PERSIST_RECEIVERS: frozenset[str] = frozenset({
    "db", "session", "repo", "repository", "store", "engine",
    "conn", "connection", "cursor", "manager", "dao", "mapper",
})
_CACHE_RECEIVERS: frozenset[str] = frozenset({
    "cache", "redis", "memcache", "memcached", "memory", "ttl_cache",
})
_QUERY_RECEIVERS: frozenset[str] = frozenset({
    "query", "queryset", "q", "filter", "select", "cursor",
    "db", "session", "repo", "repository", "manager",
})


def _receiver_of(call: str) -> str:
    """Extract the receiver (object) from a dotted call string.

    "db.session.delete" → "session"
    "cache.get"         → "cache"
    "validate"          → ""
    """
    parts = call.split(".")
    if len(parts) >= 2:
        return parts[-2].lower()
    return ""


# Pre-compute signal sets at module load (once).
_VALIDATE_SIGNALS = _extract_signal_keywords("validate")
_LOOKUP_SIGNALS = _extract_signal_keywords("lookup")
_UPDATE_SIGNALS = _extract_signal_keywords("update_entity")
_DELETE_SIGNALS = _extract_signal_keywords("delete_entity")
_PERSIST_SIGNALS = _extract_signal_keywords("persist_state")
_LIST_SIGNALS = _extract_signal_keywords("list_or_query")
_AUTHORIZE_SIGNALS = _extract_signal_keywords("authorize")
_DELEGATE_SIGNALS = _extract_signal_keywords("delegate_action")
_PAGINATE_SIGNALS = _extract_signal_keywords("paginate")
_CACHE_SIGNALS = _extract_signal_keywords("cache")
_RATE_LIMIT_SIGNALS = _extract_signal_keywords("rate_limit")
_TRANSFORM_SIGNALS = _extract_signal_keywords("transform")


def detect_primitives(
    action: DraftAction,
    trace: SemanticTrace,
    contract_report: Any = None,
) -> PrimitiveSequence:
    """Detect primitive presence/absence for a single action."""
    ft = trace.function_traces.get(action.name)

    required = get_required_primitives(action.action_type)
    required_names = {p.name for p in required}

    present: list[PrimitiveMatch] = []
    missing: list[PrimitiveMatch] = []

    for prim in ALL_PRIMITIVES:
        if prim.name not in required_names:
            continue

        match = _detect_single(prim.name, action, ft, trace)
        if match.present:
            present.append(match)
        else:
            missing.append(match)

    return PrimitiveSequence(
        action_name=action.name,
        action_type=action.action_type,
        entity=action.entity,
        file_path=action.file_path,
        present=present,
        missing=missing,
    )


def _detect_single(
    prim_name: str,
    action: DraftAction,
    ft: Optional[FunctionTrace],
    trace: SemanticTrace,
) -> PrimitiveMatch:
    """Detect a single primitive for an action."""
    detector = _DETECTORS.get(prim_name, _detect_unknown)
    return detector(prim_name, action, ft, trace)


# ── Per-Primitive Detectors ───────────────────────────────────────────────────

def _detect_input_bind(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check if function params are bound to entity or logic."""
    if ft and ft.entity_bindings:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence=f"bindings={ft.entity_bindings[:3]}")
    # Fallback: if params are used in calls
    if action.params and action.calls:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.6, evidence="params used in calls")
    if action.params:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.4, evidence="params exist")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no params or bindings")


def _detect_validate(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for validation/verification calls."""
    signals = _VALIDATE_SIGNALS
    if ft:
        calls_lower = {c.lower() for c in ft.calls}
        if calls_lower & signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence="validate call found")
    # Check action calls
    for call in action.calls:
        bare = call.split(".")[-1].lower()
        if bare in signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no validation call found")


def _detect_lookup(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for entity lookup/retrieval."""
    signals = _LOOKUP_SIGNALS
    if ft:
        calls_lower = {c.lower() for c in ft.calls}
        for lf in signals:
            if any(lf in c for c in calls_lower):
                return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence="lookup call found")
    for call in action.calls:
        bare = call.split(".")[-1].lower()
        if any(lf in bare for lf in signals):
            return PrimitiveMatch(primitive=name, present=True, confidence=0.7, evidence=f"call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no lookup call found")


def _detect_create_entity(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for entity instantiation."""
    if ft and ft.instantiations:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence=f"instantiations={list(ft.instantiations)[:3]}")
    # Check calls for class-like names
    for call in action.calls:
        bare = call.split(".")[-1]
        if bare and bare[0].isupper() and bare in trace.all_classes:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"class call={bare}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no entity instantiation found")


def _detect_update_entity(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for entity update patterns."""
    signals = _UPDATE_SIGNALS
    if ft:
        for c in ft.calls:
            if c.lower() in signals or ".update(" in c.lower():
                return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"update call={c}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no update pattern found")


def _detect_delete_entity(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for entity deletion."""
    signals = _DELETE_SIGNALS
    if ft:
        for c in ft.calls:
            bare = c.split(".")[-1].lower()
            if bare in signals:
                recv = _receiver_of(c)
                # cache.delete() is a cache primitive, not entity deletion
                if recv and recv in _CACHE_RECEIVERS:
                    continue
                return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"delete call={c}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no delete pattern found")


def _detect_persist_state(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for persistence patterns."""
    if ft and ft.persist_calls:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence=f"persist={list(ft.persist_calls)[:3]}")
    signals = _PERSIST_SIGNALS
    for call in action.calls:
        bare = call.split(".")[-1].lower()
        if bare in signals:
            recv = _receiver_of(call)
            # Boost confidence when receiver is a known persistence object
            conf = 0.85 if (recv and recv in _PERSIST_RECEIVERS) else 0.7
            return PrimitiveMatch(primitive=name, present=True, confidence=conf, evidence=f"call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no persistence pattern found")


def _detect_list_or_query(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for list/query patterns."""
    signals = _LIST_SIGNALS
    if ft:
        for c in ft.calls:
            bare = c.split(".")[-1].lower()
            recv = _receiver_of(c)
            if bare in signals or ".all(" in c.lower():
                conf = 0.85 if (recv and recv in _QUERY_RECEIVERS) else 0.8
                return PrimitiveMatch(primitive=name, present=True, confidence=conf, evidence=f"list call={c}")
    if ft and ft.return_names:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.5, evidence="has return")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no list/query pattern")


def _detect_authorize(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for authorization/token generation."""
    signals = _AUTHORIZE_SIGNALS
    if ft:
        calls_lower = {c.lower() for c in ft.calls}
        if calls_lower & signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence="token generation found")
    for call in action.calls:
        bare = call.split(".")[-1].lower()
        if bare in signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no auth/token pattern")


def _detect_branch_on_failure(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for failure branching."""
    if ft and ft.has_error_branch and ft.error_before_success:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence="error branch before success")
    if ft and ft.has_error_branch:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.6, evidence="error branch exists")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no failure branch found")


def _detect_produce_output(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check if output references actual processed result."""
    if ft and ft.return_has_entity_ref:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence="return references entity")
    if ft and ft.return_names:
        # Check if any return name is a local variable (not literal)
        return PrimitiveMatch(primitive=name, present=True, confidence=0.5, evidence=f"return_names={list(ft.return_names)[:3]}")
    if action.has_return:
        return PrimitiveMatch(primitive=name, present=True, confidence=0.3, evidence="has return statement")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no output referencing result")


def _detect_delegate_action(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for delegation to service/repository."""
    signals = _DELEGATE_SIGNALS
    for call in action.calls:
        parts = call.split(".")
        if len(parts) >= 2 and parts[0].lower() in signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"delegate={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no delegation pattern")


def _detect_paginate(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for pagination/offset/limit patterns."""
    signals = _PAGINATE_SIGNALS
    for p in action.params:
        if p.lower() in signals:
            return PrimitiveMatch(primitive=name, present=True, confidence=0.9, evidence=f"param={p}")
    for call in action.calls:
        if any(s in call.lower() for s in signals):
            return PrimitiveMatch(primitive=name, present=True, confidence=0.7, evidence=f"call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no pagination pattern")


def _detect_cache(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for cache lookup/store patterns."""
    signals = _CACHE_SIGNALS
    for call in action.calls:
        call.split(".")[-1].lower()
        recv = _receiver_of(call)
        if any(s in call.lower() for s in signals):
            conf = 0.9 if (recv and recv in _CACHE_RECEIVERS) else 0.8
            return PrimitiveMatch(primitive=name, present=True, confidence=conf, evidence=f"call={call}")
    if ft:
        for call in ft.calls:
            call.split(".")[-1].lower()
            recv = _receiver_of(call)
            if any(s in call.lower() for s in signals):
                conf = 0.85 if (recv and recv in _CACHE_RECEIVERS) else 0.7
                return PrimitiveMatch(primitive=name, present=True, confidence=conf, evidence=f"trace_call={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no cache pattern")


def _detect_rate_limit(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for rate limiting patterns."""
    signals = _RATE_LIMIT_SIGNALS
    for call in action.calls:
        if any(s in call.lower() for s in signals):
            return PrimitiveMatch(primitive=name, present=True, confidence=0.8, evidence=f"call={call}")
    if ft and ft.has_error_branch:
        for call in ft.calls:
            if any(s in call.lower() for s in signals):
                return PrimitiveMatch(primitive=name, present=True, confidence=0.7, evidence=f"trace={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no rate limit pattern")


def _detect_transform(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    """Check for data transformation/serialization patterns."""
    signals = _TRANSFORM_SIGNALS
    for call in action.calls:
        if any(s in call.lower() for s in signals):
            return PrimitiveMatch(primitive=name, present=True, confidence=0.7, evidence=f"call={call}")
    if ft:
        for call in ft.calls:
            if any(s in call.lower() for s in signals):
                return PrimitiveMatch(primitive=name, present=True, confidence=0.6, evidence=f"trace={call}")
    return PrimitiveMatch(primitive=name, present=False, missing_reason="no transform pattern")


def _detect_unknown(name: str, action: DraftAction, ft: Optional[FunctionTrace], trace: SemanticTrace) -> PrimitiveMatch:
    return PrimitiveMatch(primitive=name, present=False, missing_reason="unknown primitive")


_DETECTORS = {
    # Base 12
    "input_bind": _detect_input_bind,
    "validate": _detect_validate,
    "lookup": _detect_lookup,
    "create_entity": _detect_create_entity,
    "update_entity": _detect_update_entity,
    "delete_entity": _detect_delete_entity,
    "persist_state": _detect_persist_state,
    "list_or_query": _detect_list_or_query,
    "authorize": _detect_authorize,
    "branch_on_failure": _detect_branch_on_failure,
    "produce_output": _detect_produce_output,
    "delegate_action": _detect_delegate_action,
    # Extended 4 (#9)
    "paginate": _detect_paginate,
    "cache": _detect_cache,
    "rate_limit": _detect_rate_limit,
    "transform": _detect_transform,
}
