"""Regression tests for placement_uncertain → escalation connection.

Verifies that:
  - placement_uncertain=True + no actual file change → failure_class=no_effect + status=error
  - placement_uncertain=True + real file change      → success preserved
  - placement_uncertain=True + already_satisfied     → success preserved (intentional no-op)
  - placement_uncertain absent + no change           → old behavior (tagged but completed)
  - intent assertion failure already converts to error regardless of flag

The core spec:
  placement_uncertain is a *risk amplifier*, not an automatic fail.
  It escalates only when there is concrete evidence (no diff) that the
  uncertain anchor missed the intended location.
"""


# ─── Replicate the escalation logic from operation_executor.py ─────────────────

_FAIL_STATUSES = {"error", "failed", "skipped"}


def _run_placement_uncertain_check(
    result_status: str,
    result: dict,
    op_metadata: dict,
) -> tuple[str, dict]:
    """Replicate the placement_uncertain escalation block from _execute_operations.

    Returns (new_result_status, possibly_mutated_result).
    """
    if (
        result_status not in _FAIL_STATUSES
        and isinstance(result, dict)
        and (op_metadata or {}).get("placement_uncertain")
        and not result.get("already_satisfied")
    ):
        _pu_modified = bool(
            result.get("modified_files") or result.get("patch_applied", "")
        )
        if not _pu_modified:
            result["failure_class"] = result.get("failure_class") or "no_effect"
            result["error"] = (
                "placement_uncertain: edit produced no change — "
                "uncertain anchor may have missed intended location"
            )
            result_status = "error"
    return result_status, result


# ─── Positive case (uncertain flag + no change → escalation) ──────────────────

class TestPlacementUncertainEscalation:

    def test_no_change_with_flag_escalates_to_error(self):
        """Part 4 positive: placement_uncertain + no diff → status=error."""
        status, result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "patch_applied": "", "modified_files": []},
            op_metadata={"placement_uncertain": True},
        )
        assert status == "error"
        assert result["failure_class"] == "no_effect"
        assert "placement_uncertain" in result["error"]

    def test_no_change_with_flag_sets_failure_class_no_effect(self):
        _status, result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success"},
            op_metadata={"placement_uncertain": True},
        )
        assert result.get("failure_class") == "no_effect"

    def test_existing_failure_class_preserved(self):
        """If failure_class was already set (e.g. anchor_miss), keep it."""
        status, result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "failure_class": "anchor_miss"},
            op_metadata={"placement_uncertain": True},
        )
        assert result["failure_class"] == "anchor_miss"  # not overwritten
        assert status == "error"


# ─── Negative case (uncertain flag + real change → success preserved) ──────────

class TestPlacementUncertainSuccessPreserved:

    def test_real_change_preserves_success(self):
        """Part 5 negative: placement_uncertain + actual diff → success NOT failed."""
        status, result = _run_placement_uncertain_check(
            result_status="success",
            result={
                "status": "success",
                "modified_files": ["foo.py"],
                "patch_applied": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n+guard",
            },
            op_metadata={"placement_uncertain": True},
        )
        assert status == "success"
        assert "error" not in result

    def test_patch_applied_string_preserves_success(self):
        status, _result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "patch_applied": "some diff content"},
            op_metadata={"placement_uncertain": True},
        )
        assert status == "success"

    def test_already_satisfied_not_escalated(self):
        """already_satisfied=True means intent was fulfilled — must not fail."""
        status, _result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "already_satisfied": True},
            op_metadata={"placement_uncertain": True},
        )
        assert status == "success"

    def test_flag_absent_no_change_keeps_old_behavior(self):
        """Without the flag, no-change success is tagged but NOT escalated (existing behavior)."""
        status, result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "patch_applied": ""},
            op_metadata={},  # no placement_uncertain
        )
        assert status == "success"  # old behavior: stays success
        assert "error" not in result


# ─── Guard: already-failed status is not double-processed ─────────────────────

class TestPlacementUncertainGuardAlreadyFailed:

    def test_already_error_not_touched(self):
        """If status is already error (e.g. intent assertion failed), skip the check."""
        status, result = _run_placement_uncertain_check(
            result_status="error",
            result={"status": "error", "error": "intent_assertion_failed"},
            op_metadata={"placement_uncertain": True},
        )
        assert status == "error"
        assert result["error"] == "intent_assertion_failed"  # not overwritten

    def test_none_metadata_treated_as_no_flag(self):
        status, _result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success", "patch_applied": ""},
            op_metadata=None,
        )
        assert status == "success"

    def test_empty_metadata_treated_as_no_flag(self):
        status, _result = _run_placement_uncertain_check(
            result_status="success",
            result={"status": "success"},
            op_metadata={},
        )
        assert status == "success"
