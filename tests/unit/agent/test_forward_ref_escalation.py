"""Tests for placement contract escalation (guard_add + placement_contract).

Verifies:
  1. _classify_semantic_issues classifies insertion_scope_warning correctly
  2. get_repair_hint returns "force_body_only" for insertion_scope_warning/placement_violation
  3. escalation logic: placement_contract → blocking for guard_add
  4. Non-guard_add ops are NOT escalated even with insertion_scope_warning
  5. guard_add WITHOUT placement_contract → fallback escalation on insertion_scope_warning
  6. placement_violation from PCL verification → blocking error
"""
from dataclasses import dataclass, field
from typing import Optional

from external_llm.agent.placement_contract import build_after_assignment_contract

# ─── Replicate SemanticVerificationResult (minimal) ───────────────────────────

@dataclass
class _VerResult:
    verification_passed: bool = True
    verification_errors: list[str] = field(default_factory=list)
    verification_warnings: list[str] = field(default_factory=list)
    issue_codes: list[str] = field(default_factory=list)
    primary_issue: Optional[str] = None
    has_blocking_issue: bool = False
    has_warning_only: bool = False


# ─── Inline _classify_semantic_issues (mirrors semantic_verifier.py) ─────────

def _classify(ver: _VerResult) -> dict:
    issue_codes = []
    blocking_issues = {
        'ast_parse_failed', 'symbol_removed', 'signature_changed',
        'class_shape_changed', 'symbol_kind_changed', 'receiver_scope_mismatch',
        'dead_code_inserted', 'partial_implementation', 'incomplete_return_path',
        'insertion_scope_warning', 'placement_violation',
    }
    for error in ver.verification_errors:
        for prefix in ['ast_parse_failed', 'symbol_removed', 'signature_changed',
                       'class_shape_changed', 'symbol_kind_changed',
                       'receiver_scope_mismatch', 'dead_code_after_return',
                       'dead_code_after_raise', 'partial_impl_unused_cache',
                       'partial_impl_stub_block', 'incomplete_return_path',
                       'insertion_scope_warning', 'placement_violation']:
            if error.startswith(prefix):
                code = {
                    'dead_code_after_return': 'dead_code_inserted',
                    'dead_code_after_raise': 'dead_code_inserted',
                    'partial_impl_unused_cache': 'partial_implementation',
                    'partial_impl_stub_block': 'partial_implementation',
                }.get(prefix, prefix)
                issue_codes.append(code)
                break
        else:
            issue_codes.append('unknown_semantic_issue')
    for warning in ver.verification_warnings:
        if warning.startswith('insertion_scope_warning'):
            issue_codes.append('insertion_scope_warning')
        elif warning.startswith('placement_warning'):
            issue_codes.append('placement_warning')
        elif warning.startswith('decorators_changed') or warning.startswith('class_bases_changed'):
            issue_codes.append('decorators_changed')
        elif warning.startswith('unused_parameter'):
            issue_codes.append('unused_parameter')
        elif warning.startswith('wrong_return_shape'):
            issue_codes.append('wrong_return_shape')
    issue_codes = list(set(issue_codes))
    has_blocking = any(c in blocking_issues for c in issue_codes)
    has_warning_only = len(issue_codes) > 0 and not has_blocking
    primary = None
    if has_blocking:
        for code in issue_codes:
            if code in blocking_issues:
                primary = code
                break
    elif has_warning_only:
        primary = issue_codes[0] if issue_codes else None
    return {
        "semantic_issue_codes": issue_codes,
        "semantic_primary_issue": primary,
        "semantic_has_blocking_issue": has_blocking,
        "semantic_has_warning_only": has_warning_only,
    }


# ─── Inline get_repair_hint (relevant paths only) ─────────────────────────────

def _get_repair_hint(ver: _VerResult) -> Optional[str]:
    errors = ver.verification_errors or []
    warnings = ver.verification_warnings or []
    for error in errors:
        if error.startswith('ast_parse_failed'):
            return "force_body_only"
        elif error.startswith('symbol_removed'):
            return "skip_python_precise"
        elif error.startswith('signature_changed'):
            return "force_body_only"
        elif error.startswith('insertion_scope_warning'):
            return "force_body_only"
        elif error.startswith('placement_violation'):
            return "force_body_only"
    if not errors:
        for warning in warnings:
            if warning.startswith('unused_parameter'):
                return "force_body_only"
    return None


# ─── Escalation logic (mirrors repair_engine.py) ─────────────────────────────

def _apply_escalation(ver: _VerResult, op_meta: dict) -> _VerResult:
    """Replicate escalation logic from _run_semantic_verify_and_map.

    Two paths:
      1. PCL verification (placement_contract present)
      2. Fallback (guard_add + insertion_scope_warning, no placement_contract)
    """
    pcl_raw = op_meta.get("placement_contract")
    pcl_handled = False

    # PCL path: placement_contract → blocking if is_blocking
    if pcl_raw:
        from external_llm.agent.placement_contract import get_placement_contract
        pcl = get_placement_contract(op_meta)
        if pcl and pcl.is_blocking:
            # Simulate placement violation detection
            # In real code, verify_placement_contract runs on actual source
            # Here we check if insertion_scope_warning is present (proxy for bad placement)
            fwd_warnings = [
                w for w in list(ver.verification_warnings)
                if w.startswith("insertion_scope_warning")
            ]
            if fwd_warnings:
                for fw in fwd_warnings:
                    ver.verification_warnings.remove(fw)
                    pcl_error = f"placement_violation: {fw}"
                    ver.verification_errors.append(pcl_error)
                ver.verification_passed = False
                ver.has_blocking_issue = True
                if "placement_violation" not in ver.issue_codes:
                    ver.issue_codes.append("placement_violation")
                ver.primary_issue = "placement_violation"
                pcl_handled = True

    # Fallback: guard_add without PCL
    if not pcl_handled and op_meta.get("edit_kind") == "guard_add":
        fwd_warnings = [
            w for w in list(ver.verification_warnings)
            if w.startswith("insertion_scope_warning")
        ]
        if fwd_warnings:
            for fw in fwd_warnings:
                ver.verification_warnings.remove(fw)
                ver.verification_errors.append(fw)
            ver.verification_passed = False
            ver.has_blocking_issue = True
            if "insertion_scope_warning" not in ver.issue_codes:
                ver.issue_codes.append("insertion_scope_warning")
            ver.primary_issue = "insertion_scope_warning"

    return ver


# ─── Tests: _classify_semantic_issues ─────────────────────────────────────────

class TestClassifyInsertionScopeWarning:

    def test_warning_classifies_as_insertion_scope(self):
        ver = _VerResult(verification_warnings=[
            "insertion_scope_warning: forward ref 'candidates'"
        ])
        result = _classify(ver)
        assert "insertion_scope_warning" in result["semantic_issue_codes"]

    def test_warning_in_blocking_set(self):
        """insertion_scope_warning in warnings → has_blocking_issue=True via blocking_issues set."""
        ver = _VerResult(verification_warnings=[
            "insertion_scope_warning: forward ref 'candidates'"
        ])
        result = _classify(ver)
        assert result["semantic_has_blocking_issue"] is True

    def test_warning_in_error_already_blocking(self):
        """Once escalated to errors, still classifies as blocking."""
        ver = _VerResult(verification_errors=[
            "insertion_scope_warning: forward ref 'candidates'"
        ])
        result = _classify(ver)
        assert result["semantic_has_blocking_issue"] is True
        assert result["semantic_primary_issue"] == "insertion_scope_warning"

    def test_no_insertion_scope_warning_no_effect(self):
        ver = _VerResult()
        result = _classify(ver)
        assert "insertion_scope_warning" not in result["semantic_issue_codes"]
        assert result["semantic_has_blocking_issue"] is False

    def test_placement_violation_classifies_as_blocking(self):
        """placement_violation in errors → has_blocking_issue=True."""
        ver = _VerResult(verification_errors=[
            "placement_violation: FAIL relaxed: target not found"
        ])
        result = _classify(ver)
        assert result["semantic_has_blocking_issue"] is True
        assert result["semantic_primary_issue"] == "placement_violation"


# ─── Tests: get_repair_hint ────────────────────────────────────────────────────

class TestRepairHintInsertionScope:

    def test_error_returns_force_body_only(self):
        ver = _VerResult(
            verification_passed=False,
            verification_errors=["insertion_scope_warning: forward ref 'candidates'"],
        )
        assert _get_repair_hint(ver) == "force_body_only"

    def test_placement_violation_returns_force_body_only(self):
        ver = _VerResult(
            verification_passed=False,
            verification_errors=["placement_violation: FAIL relaxed: target not found"],
        )
        assert _get_repair_hint(ver) == "force_body_only"

    def test_warning_only_no_hint(self):
        """In warning-only state (before escalation), get_repair_hint returns None."""
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'candidates'"],
        )
        assert _get_repair_hint(ver) is None

    def test_other_error_not_affected(self):
        ver = _VerResult(
            verification_passed=False,
            verification_errors=["symbol_removed: foo"],
        )
        assert _get_repair_hint(ver) == "skip_python_precise"


# ─── Tests: escalation logic ───────────────────────────────────────────────────

class TestForwardRefEscalation:

    def test_placement_contract_escalates(self):
        """placement_contract + insertion_scope_warning → placement_violation blocking."""
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'candidates'"],
        )
        contract = build_after_assignment_contract(
            "if not candidates: return None", ["candidates"],
        )
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)

        assert ver.verification_passed is False
        assert ver.has_blocking_issue is True
        assert ver.primary_issue == "placement_violation"
        assert any(e.startswith("placement_violation") for e in ver.verification_errors)
        assert not any(w.startswith("insertion_scope_warning") for w in ver.verification_warnings)

    def test_guard_add_no_contract_fallback_escalation(self):
        """guard_add WITHOUT placement_contract → fallback escalation on insertion_scope_warning."""
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'candidates'"],
        )
        op_meta = {"edit_kind": "guard_add"}  # no placement_contract
        ver = _apply_escalation(ver, op_meta)

        assert ver.verification_passed is False
        assert ver.has_blocking_issue is True
        assert ver.primary_issue == "insertion_scope_warning"

    def test_non_guard_add_no_escalation(self):
        """Non-guard_add op is NOT escalated even if insertion_scope_warning present."""
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'x'"],
        )
        op_meta = {"edit_kind": "surgical_edit"}
        ver = _apply_escalation(ver, op_meta)

        assert ver.verification_passed is True
        assert ver.has_blocking_issue is False

    def test_no_insertion_scope_warning_no_escalation(self):
        """placement_contract but no forward ref warning → no escalation."""
        ver = _VerResult(
            verification_warnings=["decorators_changed: foo was @cached"],
        )
        contract = build_after_assignment_contract("assert x", ["x"])
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)

        assert ver.verification_passed is True
        assert ver.has_blocking_issue is False

    def test_escalation_sets_issue_code(self):
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'candidates'"],
        )
        contract = build_after_assignment_contract(
            "if not candidates: return None", ["candidates"],
        )
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)
        assert "placement_violation" in ver.issue_codes

    def test_empty_op_meta_no_escalation(self):
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'x'"],
        )
        ver = _apply_escalation(ver, {})
        assert ver.verification_passed is True

    def test_multiple_forward_refs_all_moved(self):
        """Multiple insertion_scope_warning entries all get escalated."""
        ver = _VerResult(
            verification_warnings=[
                "insertion_scope_warning: forward ref 'candidates'",
                "insertion_scope_warning: forward ref 'result'",
                "decorators_changed: something",
            ],
        )
        contract = build_after_assignment_contract(
            "if not candidates: return None", ["candidates", "result"],
        )
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)

        assert ver.verification_passed is False
        # Both moved to errors as placement_violation
        assert sum(1 for e in ver.verification_errors if e.startswith("placement_violation")) == 2
        # decorators_changed stays in warnings
        assert any("decorators_changed" in w for w in ver.verification_warnings)
        # No insertion_scope_warning remains in warnings
        assert not any(w.startswith("insertion_scope_warning") for w in ver.verification_warnings)

    def test_repair_hint_after_escalation(self):
        """After escalation, get_repair_hint should return force_body_only."""
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'candidates'"],
        )
        contract = build_after_assignment_contract(
            "if not candidates: return None", ["candidates"],
        )
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)
        assert _get_repair_hint(ver) == "force_body_only"

    def test_non_blocking_contract_no_escalation(self):
        """Contract with escalation='warning' does not block."""
        from external_llm.agent.placement_contract import PlacementContract, PlacementRepair
        ver = _VerResult(
            verification_warnings=["insertion_scope_warning: forward ref 'x'"],
        )
        contract = PlacementContract(
            kind="after_anchor",
            repair=PlacementRepair(escalation="warning"),
        )
        op_meta = {
            "edit_kind": "guard_add",
            "placement_contract": contract.to_dict(),
        }
        ver = _apply_escalation(ver, op_meta)
        # Non-blocking contract → falls through to fallback
        # Fallback: guard_add → still escalates insertion_scope_warning
        assert ver.has_blocking_issue is True
