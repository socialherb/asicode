"""
Routing intent utilities.

`classify_routing_intent` (LLM-based) has been removed — routing intent is now
derived deterministically from ``IntentResult`` produced by ``IntentResolver``
(TaskRouter Stage 0), which already ran before planning.

Signal priority:
  1. ``IntentResult.lane_hint == "read_only"``  — IntentResolver's explicit routing
     decision (covers pure explanation/question requests only).
  2. ``IntentResult.intent_type == "question"``  — fallback for lane_hint absent.
  3. Default: ``explore_and_edit``              — never block a legitimate edit.

NOTE: ``lane_hint == "clarify"`` maps to ``explore_and_edit`` (not ``read_only``).
Clarification means the edit intent is present but needs more info — the agent
should proceed to the CLARIFY verdict path, not block execution entirely.
"""
from __future__ import annotations

from typing import Literal
import logging

RoutingIntent = Literal["read_only", "clarification_needed", "explore_and_edit"]

# Canonical forms for LLM-output normalization.
# Keys are possible LLM spellings; values are the canonical RoutingIntent.
# Used to absorb label drift without silently changing behavior.
_ROUTING_INTENT_NORMALIZE: dict = {
    "read_only": "read_only", "readonly": "read_only", "read-only": "read_only",
    "read only": "read_only", "ReadOnly": "read_only",
    "explore_and_edit": "explore_and_edit", "explore-and-edit": "explore_and_edit",
    "exploreandedit": "explore_and_edit",
    # IntentResolver lane_hint values — all are edit intents
    "planner": "explore_and_edit", "main_agent": "explore_and_edit",
    # IntentResolver intent_type values — all are edit intents, except "question"
    # which is handled separately (routing_intent_from_intent_result maps it to read_only)
    "bugfix": "explore_and_edit", "feature": "explore_and_edit",
    "refactor": "explore_and_edit", "exploration": "explore_and_edit",
    "modify": "explore_and_edit", "extend": "explore_and_edit",
    "create": "explore_and_edit",
    # IntentResult default / classification-failure sentinel (IntentResolver emits
    # intent_type="unknown" on every failure path: LLM-unavailable, parse failure,
    # empty request — intent_models.py:41). NOT label drift: it is an internal
    # first-class sentinel, not an LLM-output spelling variation. All failure paths
    # also set lane_hint ∈ {"planner","main_agent"} → routing already resolves to
    # explore_and_edit, so this entry changes ONLY the drift-warning noise, not behavior.
    "unknown": "explore_and_edit",
    "question": "question",  # recognized (handled in routing_intent_from_intent_result)
}


def normalize_routing_label(label: str) -> str:
    """Normalize LLM-output label to canonical form.

    Handles variations like "read-only", "readonly", "ReadOnly" →
    "read_only". Unrecognized labels are logged (as a drift signal) and
    returned as-is so the caller can apply its own fallback.
    """
    from external_llm.languages._normalize import normalize_key
    _key = normalize_key(label)
    _canonical = _ROUTING_INTENT_NORMALIZE.get(_key)
    if _canonical is not None:
        return _canonical
    # Unrecognized label — log drift warning, pass through unchanged
    logger = logging.getLogger(__name__)
    logger.warning(
        "[LABEL_DRIFT] normalize_routing_label: unrecognized label=%r — "
        "passing through unchanged. Consider adding to _ROUTING_INTENT_NORMALIZE.",
        label,
    )
    return label


def is_non_edit_intent(intent: RoutingIntent) -> bool:
    """True when the agent must not run normal patch/write execution modes."""
    return intent != "explore_and_edit"


# ``IntentResolver`` metadata sources that mean "classification never happened".
# All three are failure paths, not classifications: minimal_fallback (LLM call
# raised / no client), llm_parse_failed (response was not parseable JSON),
# empty_request (nothing to classify). See intent_resolver.py:791-830.
_UNDETERMINED_INTENT_SOURCES = frozenset({
    "minimal_fallback", "llm_parse_failed", "empty_request",
})


def intent_is_undetermined(intent_result: object) -> bool:
    """True when intent resolution never produced a real classification.

    ``routing_intent_from_intent_result`` maps every failure path to
    ``explore_and_edit`` so a legitimate edit is never blocked — that default is
    correct for *permission*. It is wrong for *expectation*: the caller cannot
    conclude "the user asked for an edit" from a result the resolver failed to
    produce, and a gate that fires on an unproven premise is a false-positive
    machine. This predicate separates the two so only permission keeps the
    edit-leaning default.

    Concretely it distinguishes a real ``intent_type="unknown"`` classification
    from the failure paths that also emit ``"unknown"`` — the sentinel alone
    cannot tell them apart (see ``_ROUTING_INTENT_NORMALIZE["unknown"]``), which
    is why the metadata source is the signal.

    ``None`` (router never attached an ``IntentResult``) is undetermined too.
    """
    if intent_result is None:
        return True
    _meta = getattr(intent_result, "metadata", None)
    if isinstance(_meta, dict):
        return str(_meta.get("source", "")) in _UNDETERMINED_INTENT_SOURCES
    return False


def routing_intent_from_intent_result(intent_result: object) -> RoutingIntent:
    """Derive RoutingIntent from IntentResult without an LLM call.

    IntentResolver emits ``lane_hint="read_only"`` for pure read/explain requests
    (no code change intended).  ``lane_hint="clarify"`` means edit intent is present
    but vague — route to ``explore_and_edit`` so the Semantic-Fit Judge can surface
    a CLARIFY verdict instead of blocking execution outright.

    Labels from IntentResult are normalized via ``normalize_routing_label()`` to
    absorb LLM output variation (e.g. "read-only", "readonly", "ReadOnly").
    """
    if intent_result is None:
        return "explore_and_edit"
    _lane = normalize_routing_label(getattr(intent_result, "lane_hint", "") or "")
    _type = normalize_routing_label(getattr(intent_result, "intent_type", "") or "")
    # Pure read-only: explanation/question only
    if _lane == "read_only" or _type == "question":
        return "read_only"

    # Clarify means edit intent exists but is vague → allow Semantic-Fit Judge to decide
    if _lane == "clarify":
        return "explore_and_edit"

    # Default: allow edit/exploration
    return "explore_and_edit"
