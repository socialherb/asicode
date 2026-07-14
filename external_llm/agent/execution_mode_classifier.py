"""
execution_mode_classifier.py — Request intent analysis for optimal mode selection.

Selects the LLM response *format* (ExecuteMode: strict_json / intelligent /
plan_json / normal / legacy) for the REST API /edit/run endpoint. This is NOT
a routing/lane classifier — it only picks the output format. Lane/agent routing
is decided upstream by task_router + intent_resolver (see CLASSIFIERS.md).

Extracted from main.py. Provides:
- analyze_request_for_optimal_mode(): top-level entry point
- validate_instruction_target_file(): safety guard for LLM instruction JSON
"""

from __future__ import annotations

import enum
import logging
import re
import threading
import weakref
from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg


class ExecuteMode(str, enum.Enum):
    """Execution mode for external LLM requests.

    Replaces scattered string literals with a typed enum.
    """
    NORMAL = "normal"
    STRICT_JSON = "strict_json"
    PLAN_JSON = "plan_json"
    INTELLIGENT = "intelligent"
    LEGACY = "legacy"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ExecuteMode"]:
        """Fuzzy match: allow any case variation and hyphen/underscore variants."""
        if not isinstance(value, str):
            return None
        _normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == _normalized:
                return member
        return None


# Mapping for LLM output → canonical mode name.
# Keys are all plausible LLM output strings (lowercased, normalized).
_LLM_MODE_ALIASES: dict = {
    # strict_json variants
    "strict_json": ExecuteMode.STRICT_JSON,
    "strict json": ExecuteMode.STRICT_JSON,
    "strictjson": ExecuteMode.STRICT_JSON,
    "strict-json": ExecuteMode.STRICT_JSON,
    # intelligent variants
    "intelligent": ExecuteMode.INTELLIGENT,
    # plan_json variants
    "plan_json": ExecuteMode.PLAN_JSON,
    "plan json": ExecuteMode.PLAN_JSON,
    "planjson": ExecuteMode.PLAN_JSON,
    "plan-json": ExecuteMode.PLAN_JSON,
    # normal variants
    "normal": ExecuteMode.NORMAL,
    # legacy variants
    "legacy": ExecuteMode.LEGACY,
}


logger = logging.getLogger(__name__)

# Per-client lock map: keyed by client object (weak reference) to avoid
# mutating shared timeout across concurrent requests.  Uses WeakKeyDictionary
# so entries auto-clean when the client is garbage-collected, preventing
# unbounded growth and id()-reuse collisions.
_client_timeout_locks = weakref.WeakKeyDictionary()
_client_timeout_locks_lock = threading.Lock()


# Embedding-based backstop for line-level edit intent. The structural cue for
# strict_json is a *number* (a line position), so semantic routing only fires
# when a digit is present — this just lifts the reliance on a hardcoded
# position-keyword list ("line"/"라인"/"줄"), letting other phrasings/languages
#("line/row 5", "row 10", "position 20") route correctly. No-op when embeddings are
# unavailable (see external_llm/agent/semantic_intent.py).
_MODE_INTENT_EXAMPLES = {
    "line_edit": [
        "add a comment on line 42",
        "insert at line 10",
        "modify the line at position 15",
        "edit row 7",
        "줄 42에 주석을 추가",
        "라인 10에 삽입",
        "행 7을 수정",
        "위치 20의 줄을 편집",
    ],
    "other": [
        "fix the bug",
        "refactor this function",
        "add a new feature",
        "optimize the code",
        "rename the variable",
        "update the documentation",
        "버그를 수정",
        "기능을 추가",
        "코드를 최적화",
        "함수를 리팩터링",
    ],
}

_mode_matcher = None
_mode_matcher_lock = threading.Lock()


def _get_mode_matcher():
    """Lazily build the line-edit semantic matcher (singleton)."""
    global _mode_matcher
    if _mode_matcher is not None:
        return _mode_matcher
    with _mode_matcher_lock:
        if _mode_matcher is None:
            from external_llm.agent.semantic_intent import SemanticIntentMatcher
            _mode_matcher = SemanticIntentMatcher(
                _MODE_INTENT_EXAMPLES,
                threshold=_cfg.scores.SEMANTIC_INTENT_MIN,
                margin=_cfg.scores.SEMANTIC_INTENT_MARGIN,
                name="mode-line-edit",
            )
        return _mode_matcher


def _has_digit(text: str) -> bool:
    """True if *text* contains any decimal digit (the line-number anchor)."""
    return any(ch.isdigit() for ch in text)


def _has_number_after_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Check if any keyword in *keywords* is immediately followed by a number.

    Handles both ``"line 42"`` and ``"line42"`` forms.  No regex needed.
    """
    for kw in keywords:
        idx = text.find(kw)
        if idx < 0:
            continue
        # Scan the remainder after the keyword — first non-space char must be a digit.
        rest = text[idx + len(kw):].lstrip()
        if rest and rest[0].isdigit():
            return True
        # The keyword may be embedded in a longer word; keep scanning.
        # Fall through to check for later occurrences (handles "online 42")
        # — the prefix check above ensures the first space-delimited neighbor
        # is a number. That's good enough for our use case.
    return False


def analyze_request_for_optimal_mode(prompt: str, target_file: Optional[str]) -> str:
    """
    Analyze prompt to recommend optimal execution mode.
    Returns: ExecuteMode value string ("normal", "strict_json", "intelligent", "plan_json", "legacy").

    Uses LLM-based intent classification when available, falls back to structural matching.
    """
    # Try LLM-based intent classification first (if external LLM is available)
    llm_decision = _analyze_intent_with_llm_if_available(prompt, target_file)
    if llm_decision:
        return llm_decision.value

    # Fallback to structural analysis
    return _analyze_intent_with_keywords(prompt, target_file)


def _analyze_intent_with_keywords(prompt: str, target_file: Optional[str]) -> str:
    """Keyword-based intent classification as fallback when LLM is unavailable.

    - Line-number references -> strict_json
    - Explicit "legacy" -> legacy
    - Default -> normal (TaskRouter handles structural routing)
    """
    prompt_lower = prompt.lower()

    # Legacy mode: only when explicitly requested
    if "legacy" in prompt_lower:
        return "legacy"

    # Check for keyword followed by digits (the number is the actual cue, not the keyword).
    if _has_number_after_keyword(prompt_lower, ("line",)):
        return "strict_json"

    # Semantic backstop: a line-level edit phrased with a position word not in the
    # hardcoded list above. Require a digit so the line-number anchor strict_json
    # depends on is still present, avoiding false routes from generic edit verbs.
    if _has_digit(prompt) and _get_mode_matcher().matches(prompt, "line_edit"):
        return "strict_json"

    # Default: "normal" — downstream TaskRouter handles complexity-based routing
    return "normal"


def _analyze_intent_with_llm_if_available(prompt: str, target_file: Optional[str]) -> Optional[ExecuteMode]:
    """
    Use LLM to analyze intent and determine optimal mode.
    Returns ``None`` if LLM is not available or fails.

    NOTE: Uses a very short timeout (5s) to avoid blocking /edit/run requests.
    Falls through to keyword-based analysis on any failure.
    """
    try:
        # Check if external LLM is configured
        from external_llm.intelligent_service import create_intelligent_service_from_env

        # Use a short timeout so intent analysis never blocks the request for long
        _INTENT_TIMEOUT_SEC = 5

        # Try to create a minimal service for intent analysis
        service = create_intelligent_service_from_env(None, None)
        if not service:
            return None

        # Use the LLM to analyze intent. Prompt is in English so the
        # classifier works for requests in any language — the LLM
        # understands the user's request regardless of locale.
        # Escape curly braces to prevent .format() KeyError when prompt contains { or }.
        # Python dict/JSON literals in user requests are common in coding tasks.
        _prompt_safe = prompt[:500].replace("{", "{{").replace("}", "}}")
        _target_safe = (target_file or "(unspecified)").replace("{", "{{").replace("}", "}}")
        system_prompt = f"""You classify coding requests into the optimal execution mode.

Modes:
1. strict_json - Precise line-level edit (insert/modify at a specific position, add a comment, etc.)
2. intelligent - Multi-file feature work that requires project analysis (new feature, UI/editor work, plan synthesis)
3. plan_json - Multi-file feature work with a structural plan (new feature, API endpoint, new module)
4. normal - Single-file modification (bug fix, refactor, optimization, typo)
5. legacy - Legacy diff format (explicit request only)

Guidance:
- strict_json: explicit position references like "add a comment on line 42" or "insert at line 10"
- intelligent: multi-file work that needs project understanding (e.g. "add a line-number indicator to the editor", "implement login flow")
- plan_json: multi-file feature work without deep analysis (e.g. "add login endpoint", "create a new module")
- normal: single-file modification (bug fix, refactor, error handling)
- legacy: only when the user explicitly asks for legacy diff format

Request: "{_prompt_safe}"
Target file: {_target_safe}

Pick exactly one mode (strict_json, intelligent, plan_json, normal, legacy).
Output only the mode name — no explanation."""

        # Use the LLM service's client for analysis with a short timeout.
        # We temporarily mutate client.timeout; guard with a per-client lock so
        # concurrent requests on the same shared client don't race each other.
        from external_llm.client import LLMMessage, effective_content

        client = service.llm_service.client
        with _client_timeout_locks_lock:
            if client not in _client_timeout_locks:
                _client_timeout_locks[client] = threading.Lock()
            client_lock = _client_timeout_locks[client]

        with client_lock:
            orig_timeout = getattr(client, "timeout", 120)
            try:
                client.timeout = _INTENT_TIMEOUT_SEC
                response = client.chat(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content="Analyze the intent and pick the optimal mode."),
                    ],
                    model=service.model or "gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=_cfg.tokens.INTENT_CLASSIFY,
                )
            finally:
                client.timeout = orig_timeout

        # GLM-5.2/DeepSeek Reasoner may emit the mode name only in
        # reasoning_content with empty content — effective_content recovers it
        # so the alias/fuzzy lookup sees the real answer instead of collapsing
        # to the keyword heuristic fallback.
        response_text = effective_content(response).strip().lower()

        # Parse response: use normalized alias lookup instead of substring matching.
        # The LLM is instructed to output exactly one mode name. We strip whitespace
        # and punctuation, then look up in ``_LLM_MODE_ALIASES``.
        _cleaned = response_text.strip(".!? \t\n").strip()
        mode = _LLM_MODE_ALIASES.get(_cleaned)
        if mode is not None:
            return mode

        # Fallback: fuzzy search — find any known alias in the response text.
        # Word boundaries (\b) prevent inflected/embedded false positives, e.g.
        # "abnormal" no longer matches the "normal" alias. Dict order is
        # preserved so the most specific alias wins. Handles LLM commentary.
        for alias, mode in _LLM_MODE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", response_text):
                return mode

    except Exception as e:
        logger.debug(f"LLM intent analysis failed (falling back to keywords): {e}")

    return None


def validate_instruction_target_file(instruction: dict, expected_target_file: str) -> None:
    """
    Safety guard: reject LLM instruction JSON that targets a different file
    than the UI/server-selected target.
    """
    exp = (expected_target_file or "").strip()
    if not exp:
        return

    got = ""
    if isinstance(instruction, dict):
        got = str(instruction.get("target_file") or "").strip()

    # If LLM did not emit target_file, do not reject here.
    if not got:
        return

    if got != exp:
        raise ValueError(f"instruction target_file mismatch: got={got!r}, expected={exp!r}")
