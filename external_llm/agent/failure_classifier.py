from __future__ import annotations

import errno
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RecoveryAction(str, Enum):

    RETRY_SAME = "retry_same"
    SWITCH_TOOL = "switch_tool"
    READ_FIRST = "read_first"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class FailureClassification:

    action: RecoveryAction
    reason: str


class FailureClassifier:

    def classify(self, tool_name: str, args: dict, result) -> FailureClassification:
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
        return _classify_by_text(str(error))


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
_TEXT_FILE_MISSING = ("not found", "no such file")
_TEXT_TRANSIENT = ("timeout", "timed out", "connection", "temporarily unavailable")


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


def _classify_by_text(error_str: str) -> FailureClassification:
    """Last-resort text classification.

    Uses substring matching on lowercased text.  Keywords are long
    enough that false positives are extremely unlikely in error messages.
    """
    if _has_text_phrase(error_str, _TEXT_ALREADY_APPLIED):
        return FailureClassification(action=RecoveryAction.SKIP, reason="patch already applied")

    if _has_text_phrase(error_str, _TEXT_CONTEXT_MISMATCH):
        return FailureClassification(action=RecoveryAction.READ_FIRST, reason="patch context mismatch")

    if _has_text_phrase(error_str, _TEXT_FILE_MISSING):
        return FailureClassification(action=RecoveryAction.SWITCH_TOOL, reason="file missing")

    if _has_text_phrase(error_str, _TEXT_TRANSIENT):
        return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="transient failure")

    return FailureClassification(action=RecoveryAction.RETRY_SAME, reason="generic failure")
