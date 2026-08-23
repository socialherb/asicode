"""
Shared status / failure-classification enums for the agent.

Trimmed from the full planner-lane operation schema (P-PCLa): the
Operation/OperationPlan/ExecutorState/... models were dead in production
(0 consumers since the planner-lane removal) and have been deleted.
Only the enums still consumed by production remain:

* ``OpStatus``            — agent loop / _shared_utils terminal statuses
* ``FailureClass``        — failure_classifier + write_tools_patch_mixin
* ``normalize_failure_class`` — canonical string → FailureClass mapping
"""

from __future__ import annotations

import enum


class OpStatus(str, enum.Enum):
    """Canonical op/plan result status codes.

    Using ``str, enum.Enum`` means each member compares equal to its string
    value (e.g. ``OpStatus.ERROR == "error"``), so existing dict comparisons
    like ``result["status"] == "success"`` keep working unchanged. Note that
    "success" itself has no OpStatus member — success paths return plain
    strings or richer verdict objects, never an OpStatus value.
    """

    FAILED = "failed"
    ERROR = "error"
    NOT_FOUND = "not_found"
    VERIFICATION_FAILED = "verification_failed"
    EXECUTION_ERROR = "execution_error"
    PREFLIGHT_FAILED = "preflight_failed"


class FailureClass(str, enum.Enum):
    """Canonical failure classification codes.

    Using ``str, enum.Enum`` preserves JSON serialisation and dict key equality
    with legacy plain-string values.  ``normalize_failure_class()`` maps unknown
    or legacy strings to ``UNKNOWN`` so callers never receive a raw string.
    """

    # ── edit / patch failures ──────────────────────────────────────────────
    SEARCH_STRING_MISMATCH = "search_string_mismatch"
    NO_DIFF_GENERATED = "no_diff_generated"
    NO_EFFECT = "no_effect"
    NO_EFFECTIVE_PROGRESS = "no_effective_progress"
    # The spelling DesignChatLoop._apply_no_effective_progress_gate actually
    # publishes for a byte-identical apply_patch — the ONLY issuer in the tree.
    # (``no_effective_progress`` above is an alias with no issuer; it is kept
    # only so old persisted patterns keep normalising to a real member.)
    NO_EFFECTIVE_CHANGE = "no_effective_change"
    NO_OP_EDIT = "no_op_edit"
    MODIFY_FAILED = "modify_failed"
    INSERT_FAILED = "insert_failed"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    WRITE_ERROR = "write_error"
    READ_ERROR = "read_error"

    # ── structural / AST failures ─────────────────────────────────────────
    SYNTAX_ERROR = "syntax_error"
    SYNTAX_ERROR_AFTER_PATCH = "syntax_error_after_patch"
    SYNTAX_INVALID_AFTER_EDIT = "syntax_invalid_after_edit"
    AST_OP_FAILED = "ast_op_failed"
    STRUCTURAL_GATE_VIOLATION = "structural_gate_violation"
    BLAST_RADIUS_VIOLATION = "blast_radius_violation"
    OVERBROAD_EDIT = "overbroad_edit"
    DEAD_CODE_INTRODUCED = "dead_code_introduced"
    DECORATOR_DELETION = "decorator_deletion"
    REGRESSION_PURE_DELETION = "regression_pure_deletion"
    REGION_RELOCATION_FAILED = "region_relocation_failed"
    EXTRACTION_ISOMORPHISM_FAILED = "extraction_isomorphism_failed"
    EXTRACTION_EQUIVALENCE_FAILED = "extraction_equivalence_failed"

    # ── anchor / target resolution ────────────────────────────────────────
    ANCHOR_MISS = "anchor_miss"
    ANCHOR_LOSS = "anchor_loss"
    # The LLM passed a multiline (``\n``-joined) anchor_pattern. The exact
    # matcher (``pattern in line``) can never match a single file line, so the
    # call would fall through to a fuzzy fallback and fail opaquely. Rejected
    # up front by detect_multiline_anchor() with an actionable error. Distinct
    # from ANCHOR_MISS (pattern genuinely absent) so the repair ladder can
    # steer the caller to use the first line + context_after, not retry-blind.
    ANCHOR_MULTILINE_PATTERN = "anchor_multiline_pattern"
    # A multiline anchor_pattern was accepted (it has non-empty lines) but a
    # later pattern line does NOT match the corresponding file line, or the
    # pattern extends past EOF. Emitted by anchor_shared.resolve_multiline_anchor
    # AFTER the first line matched, so it is distinct from ANCHOR_MISS (first
    # line absent) and ANCHOR_MULTILINE_PATTERN (rejected up front): the caller
    # had the right first line but a wrong/extra follow-on line, so the repair
    # hint is "re-read and provide the exact block" rather than "first line
    # wrong" or "use a single line".
    MULTILINE_MISMATCH = "multiline_mismatch"
    ANCHOR_NOT_UNIQUE = "anchor_not_unique"
    # The LLM-provided code_snippet for insert_before/insert_after contains a
    # copy of code already present around the anchor (the "fragment duplication"
    # failure mode). Pre-detected BEFORE the insert so the caller gets an
    # immediate, actionable failure instead of an opaque post-write syntax
    # error. Distinct from SYNTAX_INVALID_AFTER_EDIT (a generic post-write
    # syntax break) so the repair ladder can tell the caller "re-read and
    # provide only the NEW lines".
    FRAGMENT_DUPLICATION = "fragment_duplication"
    TARGET_NOT_FOUND = "target_not_found"
    FILE_NOT_FOUND = "file_not_found"
    MISSING_PATH = "missing_path"
    INVALID_LINE_RANGE = "invalid_line_range"
    # Missing required argument — published only by tool_failure_log's
    # _ERROR_PATTERNS fallback ("is required" text, e.g. "'code' is required"),
    # which fires when a write-tool handler returns an arg-validation error
    # unwrapped (no metadata["failure_class"]). Distinct from MISSING_PATH
    # (an absent path arg) and BAD_OP_SPEC (a malformed op spec) so the class
    # stays an exact label, not a drift-collapsed UNKNOWN.
    INVALID_ARGS = "invalid_args"

    # ── semantic / verification failures ─────────────────────────────────
    SEMANTIC_VERIFICATION_FAILED = "semantic_verification_failed"
    SEMANTIC_VERIFY_FAILED = "semantic_verify_failed"  # legacy alias
    VERIFICATION_FAILED = "verification_failed"
    INTENT_ASSERTION_FAILED = "intent_assertion_failed"
    # Target file on disk does not parse — assertion verifier could not
    # evaluate any structural check.  Distinct from INTENT_ASSERTION_FAILED
    # so the strategy ladder routes "broken disk" → revert/replan with
    # disk-aware context, not "cause_anchor_not_found" misdiagnosis.
    TARGET_FILE_SYNTAX_BROKEN = "target_file_syntax_broken"
    PRESUPPOSITION_VIOLATED = "presupposition_violated"
    EXTRACTION_VERIFY_FAILED = "extraction_verify_failed"
    MODULE_IMPORT_GATE = "module_import_gate"

    # ── name / import errors ──────────────────────────────────────────────
    STRUCTURAL_HALLUCINATION = "structural_hallucination"
    F821_UNDEFINED_NAME = "f821_undefined_name"
    UNDEFINED_NAME = "undefined_name"
    NAME_REFERENCE_ERROR = "name_reference_error"
    INVALID_IMPORT_STMT = "invalid_import_stmt"
    INVALID_IMPORT_MODULE_PATH = "invalid_import_module_path"

    # ── plan / op-level failures ──────────────────────────────────────────
    BAD_OP_SPEC = "bad_op_spec"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_FAILED = "execution_failed"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    MIXED_FAILURE = "mixed_failure"
    SEMANTIC_GATE_FAILED = "semantic_gate_failed"
    PLAN_ACCEPTANCE_FAILED = "plan_acceptance_failed"
    SIGNATURE_CHANGED = "signature_changed"
    LINT_ERROR = "lint_error"
    TIMEOUT = "timeout"
    API_CONNECTION_ERROR = "api_connection_error"
    MODIFY_INTENT_WITHOUT_EDIT_OP = "modify_intent_without_edit_op"
    # ── acceptance / alignment ────────────────────────────────────────────
    ACCEPTANCE_FAILED = "acceptance_failed"
    ALIGNMENT_REJECTED = "alignment_rejected"

    # ── no-op / satisfied ────────────────────────────────────────────────
    ALREADY_SATISFIED = "already_satisfied"
    ALREADY_EQUAL = "already_equal"

    # ── sentinel ─────────────────────────────────────────────────────────
    UNKNOWN = "unknown"
    UNSPECIFIED = "unspecified"
    NONE = "none"

    # ── structural / parse (legacy aliases) ──────────────────────────────
    AST_FAILED = "ast_failed"
    INSERT_POSITION_UNKNOWN = "insert_position_unknown"
    SYMBOL_NOT_FOUND = "symbol_not_found"


# Pre-built lookup for O(1) normalisation.
_FAILURE_CLASS_BY_VALUE: dict[str, FailureClass] = {fc.value: fc for fc in FailureClass}


def normalize_failure_class(value: str | None) -> FailureClass:
    """Map a raw failure_class string (or None) to ``FailureClass``.

    Unknown / legacy strings and None both return ``FailureClass.UNKNOWN``
    so callers never have to handle raw strings defensively.
    """
    if not value:
        return FailureClass.UNKNOWN
    # Case-insensitive to tolerate NO_PROGRESS-style uppercase legacy values.
    return _FAILURE_CLASS_BY_VALUE.get(value) or _FAILURE_CLASS_BY_VALUE.get(value.lower(), FailureClass.UNKNOWN)
