from __future__ import annotations

import errno
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RecoveryAction(str, Enum):

    RETRY_SAME = "retry_same"
    # The call itself was malformed — a required argument was missing
    # ("'code' is required").  The recovery is to re-issue the call WITH the
    # argument supplied — never to retry the identical call (that is the
    # blind-retry failure mode RETRY_SAME exists to avoid) nor to switch tool
    # (the tool was the right choice; the arguments were wrong).
    FIX_ARGS = "fix_args"
    SWITCH_TOOL = "switch_tool"
    READ_FIRST = "read_first"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class FailureClassification:

    action: RecoveryAction
    reason: str


class FailureClassifier:

    def classify(self, tool_name: str, result) -> FailureClassification:
        # Priority 0: the handler's own structured verdict.
        #
        # The write tools already compute exactly what went wrong and publish it
        # as ``metadata["failure_class"]`` (the canonical ``FailureClass`` enum,
        # ~57 sites). Reading the error STRING instead threw that away and then
        # got it wrong: ``_TEXT_FILE_MISSING``'s bare ``"not found"`` matched
        # ``old_string not found in <path>`` — edit_text's single most common
        # failure — and classified it "file missing" / SWITCH_TOOL. The advice
        # that reached the model ("switch to a different tool") contradicted the
        # tool's own error text ("re-read the file and include 2-3 lines of
        # surrounding context"), and poisoned the persistent recall store:
        # ``edit_text::file missing`` was this repo's #1 pattern with every
        # recorded path present on disk.
        classification = _classify_by_failure_class(result)
        if classification:
            return classification

        error = getattr(result, "error", None)

        if error is None:
            return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="generic failure")

        # Priority 1: Python exception type hierarchy (locale-independent, most reliable)
        classification = _classify_by_type(error)
        if classification:
            return classification

        # Priority 2: structured error code (numeric errno or string code)
        classification = _classify_by_code(error)
        if classification:
            return classification

        # Priority 3: normalized text (last resort — explicit fallback, not primary logic)
        return _classify_by_text(str(error), tool_name)


# ── Keyword sets for structured classification ─────────────────────
# Error codes are underscore-delimited identifiers.  We split by ``_``
# and check set membership — no regex, no substring false positives.
_CODE_IDEMPOTENT_WORDS = frozenset({"already", "duplicate", "idempotent"})
_CODE_FILE_MISSING_WORDS = frozenset({"not_found", "missing", "enoent"})
_CODE_TRANSIENT_WORDS = frozenset({"timeout", "transient", "unavailable"})

# Text phrases checked via ``in`` (lowercased).  Kept intentionally
# narrow: only phrases unambiguous across frameworks and not detectable
# by type or code inspection.
_TEXT_ALREADY_APPLIED = ("already applied", "already exists")
_TEXT_CONTEXT_MISMATCH = ("context mismatch", "does not apply", "hunk")
# Deliberately NOT a bare "not found".  In an edit tool the overwhelmingly
# common "not found" is the SEARCH TEXT, not the file — the file was opened
# successfully several steps earlier.  The write tools spell a genuinely absent
# file "File not found: <path>" and the read tools "Path not found or outside
# repo", so the filesystem sense stays fully covered while
# ``old_string not found`` / ``anchor text not found`` no longer collide with
# it.  A genuinely missing file also usually arrives as FileNotFoundError and
# is settled by ``_classify_by_type`` before any text is inspected.
_TEXT_FILE_MISSING = ("file not found", "path not found", "no such file", "does not exist")
_TEXT_TRANSIENT = ("timeout", "timed out", "connection", "temporarily unavailable")
# Missing required argument — the handler returned its own arg validation
# unwrapped ("'code' is required", "Command is required"), i.e. without
# metadata["failure_class"], so it reaches this text tier.  The single phrase
# "is required" covers every handler emission in the tree (write-tool
# validation, plan_compiler, design_chat_loop tool handlers, ui_tools) and
# mirrors the failure-log tier's ("is required" → invalid_args) rule so both
# tiers agree on the same errors.
_TEXT_MISSING_ARGS = ("is required",)

# Tools that locate an edit site by matching text they were given.  Reaching a
# "not found" in one of these means the SEARCH failed, so the recovery is to
# re-read the file and quote it exactly — never to switch tools.  Used only in
# the text fallback, for the handful of edit paths that return without setting
# ``metadata["failure_class"]``.
_TEXT_MATCHING_EDIT_TOOLS = frozenset({
    "edit_text", "anchor_edit", "edit_file", "modify_symbol",
    "apply_patch", "edit_ast", "write_plan",
})

# ``FailureClass`` value → recovery, for the classes a shipping tool actually
# publishes.  Deliberately partial: a class with no unambiguous single recovery
# (syntax_invalid_after_edit, structural_gate_violation — all "your edit was rejected",
# none of which the five RecoveryActions express)
# is left out and falls through to the text tiers exactly as before.  Reasons
# reuse the existing vocabulary wherever it fits, because ``reason`` is the
# persistence key in ``failure_pattern_store`` and a new spelling starts a new
# counter from zero.
_ACTION_BY_FAILURE_CLASS: dict[str, tuple[RecoveryAction, str]] = {
    # the search/anchor text did not match — re-read and quote exactly
    "search_string_mismatch":    (RecoveryAction.READ_FIRST, "patch context mismatch"),
    "anchor_miss":               (RecoveryAction.READ_FIRST, "patch context mismatch"),
    "anchor_loss":               (RecoveryAction.READ_FIRST, "patch context mismatch"),
    "multiline_mismatch":        (RecoveryAction.READ_FIRST, "patch context mismatch"),
    "anchor_multiline_pattern":  (RecoveryAction.READ_FIRST, "patch context mismatch"),
    "patch_apply_failed":        (RecoveryAction.READ_FIRST, "patch context mismatch"),
    # matched in more than one place — more surrounding context is needed
    "anchor_not_unique":         (RecoveryAction.READ_FIRST, "patch context mismatch"),
    # the edit was already there
    "already_equal":             (RecoveryAction.SKIP, "patch already applied"),
    "already_satisfied":         (RecoveryAction.SKIP, "patch already applied"),
    "no_effect":                 (RecoveryAction.SKIP, "patch already applied"),
    "no_op_edit":                (RecoveryAction.SKIP, "patch already applied"),
    # design_chat_loop's byte-identical apply_patch hard gate (its only issuer)
    "no_effective_change":       (RecoveryAction.SKIP, "patch already applied"),
    # genuinely absent on disk
    "file_not_found":            (RecoveryAction.SWITCH_TOOL, "file missing"),
    "missing_path":              (RecoveryAction.SWITCH_TOOL, "file missing"),
    # environment, not the call
    "timeout":                   (RecoveryAction.RETRY_SAME, "transient failure"),
    "api_connection_error":      (RecoveryAction.RETRY_SAME, "transient failure"),
}


def _classify_by_failure_class(result) -> Optional[FailureClassification]:
    """Classify from ``result.metadata["failure_class"]`` when the handler set one.

    Normalised through ``operation_models.normalize_failure_class`` so legacy /
    unknown spellings collapse to ``UNKNOWN`` and fall through rather than being
    trusted.  The import is lazy: ``operation_models`` is a large module and
    importing it at module scope would invert the dependency direction for a
    classifier that several light callers import.
    """
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("failure_class")
    if not raw:
        return None
    try:
        from .operation_models import normalize_failure_class
        value = normalize_failure_class(raw).value
    except Exception:  # pragma: no cover - operation_models always importable
        value = str(raw).lower()
    mapped = _ACTION_BY_FAILURE_CLASS.get(value)
    if mapped is None:
        return None
    action, reason = mapped
    return FailureClassification(action=action, reason=reason)


# ── Type-based classification ─────────────────────────────────────────────────

_FILE_MISSING_TYPES = (FileNotFoundError, IsADirectoryError, NotADirectoryError)

_TRANSIENT_TYPES = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


def _classify_by_type(error) -> Optional[FailureClassification]:
    """Classify by Python exception type — locale-independent, no string parsing."""
    if isinstance(error, _FILE_MISSING_TYPES):
        return FailureClassification(action=RecoveryAction.SWITCH_TOOL, reason="file missing")
    if isinstance(error, _TRANSIENT_TYPES):
        return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="transient failure")
    if isinstance(error, PermissionError):
        return FailureClassification(action=RecoveryAction.ABORT, reason="permission denied")
    return None


# ── Code-based classification ─────────────────────────────────────────────────

def _has_code_keyword(code: str, keywords: frozenset[str]) -> bool:
    """Check if any keyword appears as an underscore-delimited token.

    Splits by ``_`` to guarantee token-level matching — ``"timeout"`` in
    ``"timeouterror"`` or ``"not_found"`` in ``"inot_foundry"`` is not
    treated as a match.
    """
    return bool(keywords & set(code.lower().split("_")))


def _classify_by_code(error) -> Optional[FailureClassification]:
    """Classify by structured error code — works across frameworks and locales."""
    code = (
        getattr(error, "code", None)
        or getattr(error, "error_code", None)
        or getattr(error, "errno", None)
    )
    if code is None:
        return None

    if isinstance(code, int):
        if code == errno.ENOENT:
            return FailureClassification(action=RecoveryAction.SWITCH_TOOL, reason="file missing")
        if code == errno.EACCES:
            return FailureClassification(action=RecoveryAction.ABORT, reason="permission denied")
        if code in (errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED):
            return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="transient failure")

    if isinstance(code, str):
        if _has_code_keyword(code, _CODE_IDEMPOTENT_WORDS):
            return FailureClassification(action=RecoveryAction.SKIP, reason="patch already applied")
        if _has_code_keyword(code, _CODE_FILE_MISSING_WORDS):
            return FailureClassification(action=RecoveryAction.SWITCH_TOOL, reason="file missing")
        if _has_code_keyword(code, _CODE_TRANSIENT_WORDS):
            return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="transient failure")

    return None


# ── Text-based classification (last resort) ───────────────────────────────────

def _has_text_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Check if any phrase appears in lowercased *text*.

    Uses substring matching (not word boundaries) because the phrase
    tokens (“already applied”, “not found”) are long enough that
    subword false positives (e.g. “not found” in “noteworthy founded”)
    are extremely unlikely in error messages.
    """
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


def _classify_by_text(error_str: str, tool_name: str = "") -> FailureClassification:
    """Last-resort text classification.

    Uses substring matching on lowercased text.  Keywords are long
    enough that false positives are extremely unlikely in error messages.

    *tool_name* disambiguates the one phrase that is genuinely ambiguous across
    tools: a bare "not found".  For a text-matching edit tool it means the
    search string, for everything else it may mean the path — see
    ``_TEXT_MATCHING_EDIT_TOOLS``.
    """
    if _has_text_phrase(error_str, _TEXT_ALREADY_APPLIED):
        return FailureClassification(action=RecoveryAction.SKIP, reason="patch already applied")

    if _has_text_phrase(error_str, _TEXT_CONTEXT_MISMATCH):
        return FailureClassification(action=RecoveryAction.READ_FIRST, reason="patch context mismatch")

    # Explicit filesystem phrasing wins for every tool, including the edit
    # tools — they spell a genuinely absent file "File not found: <path>".
    if _has_text_phrase(error_str, _TEXT_FILE_MISSING):
        return FailureClassification(action=RecoveryAction.SWITCH_TOOL, reason="file missing")

    # Only now is a bare "not found" ambiguous, and only for the edit tools is
    # it decidable: they had already opened the file, so what was not found is
    # the text they were asked to match.
    if tool_name in _TEXT_MATCHING_EDIT_TOOLS and "not found" in error_str.lower():
        return FailureClassification(action=RecoveryAction.READ_FIRST, reason="patch context mismatch")

    if _has_text_phrase(error_str, _TEXT_TRANSIENT):
        return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="transient failure")

    # Missing required argument — the error text names the argument the call
    # lacked.  Retrying the identical call is the blind-retry failure mode; the
    # recovery is to re-issue the call with the argument supplied.  Last in the
    # cascade ("is required" is the most general phrase): any more specific
    # signal above it wins when both appear.
    if _has_text_phrase(error_str, _TEXT_MISSING_ARGS):
        return FailureClassification(action=RecoveryAction.FIX_ARGS, reason="missing required argument")

    return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="generic failure")
