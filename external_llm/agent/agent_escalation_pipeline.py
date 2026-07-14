"""Post-execution plan-validity checks for the PLANNER lane (Mixin).

The ANALYZE_FIRST re-resolution pipeline (SpecResolver-based _resolve_spec_and_judge,
_run_analyze_first_escalation, and their helpers) was removed with the SpecResolver
subsystem. The PLANNER lane now runs only against a prebuilt spec produced by Design
Chat analysis. What remains is post-execution diagnostic metadata recording (ATM
blocking reasons, Phase-1 contract violations) and a pass-through _EscalationOutcome
so callers continue to compile unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .agent_loop_types import _EscalationOutcome

logger = logging.getLogger(__name__)


class EscalationPipelineMixin:
    """Post-execution plan-validity checks for the PLANNER lane.

    The ANALYZE_FIRST re-resolution pipeline (_run_analyze_first_escalation,
    _resolve_spec_and_judge, and their escalation helpers — _escalation_direct_fix,
    _escalation_replan_and_execute, _collect_escalation_anchors,
    _try_auto_expand_phase1_reads, _protect_structural_guidance) was removed with
    the SpecResolver subsystem. The PLANNER lane now runs only against a prebuilt
    spec produced by Design Chat analysis. What remains here is the diagnostic
    metadata recording (ATM blocking reasons, Phase-1 contract violations) and a
    pass-through _EscalationOutcome so callers continue to compile unchanged.
    """

    def _handle_analyze_first_escalation(
        self,
        request: str,
        context: str,
        op_plan: Any,
        exec_result: dict,
        spec: Any,
        atm_no_edit: bool,
        is_read_only_intent: bool,
        escalation_attempt: int,
        llm_hints: dict,
        pending_guidance: Optional[dict],
        prev_spec_fingerprint: Optional[frozenset],
        completed: int,
        failed: int,
        exec_status: str,
        exec_detail: str,
    ) -> "_EscalationOutcome":
        """Post-execution plan-validity checks: record ATM/Phase-1 contract state.

        Previously this gated an ANALYZE_FIRST re-resolution (the SpecResolver
        pipeline). With SpecResolver removed the re-resolution is gone; what
        remains is the diagnostic metadata recording and a pass-through
        _EscalationOutcome.
        """

        # analyze_then_modify with no edit ops -> record blocking reason.
        _plan_meta_post: dict = getattr(op_plan, "metadata", None) or {}
        _blocking_reasons: list = exec_result.get("final_blocking_reasons") or []
        if atm_no_edit:
            if "modify_intent_without_edit_op" not in _blocking_reasons:
                _blocking_reasons = [*list(_blocking_reasons), "modify_intent_without_edit_op"]
            logger.info("[PLAN_VALIDITY] ATM no-edit → recorded blocking reason")

        _no_edit_on_modify = (
            "modify_intent_without_edit_op" in _blocking_reasons
        )

        # Phase 1 analyze_then_modify with clarification_needed + empty targets.
        _phase1_clar_needed = False
        _phase1_ep = getattr(op_plan, "execution_phase", "__missing__")
        _phase1_rp2 = getattr(op_plan, "requires_phase2", "__missing__")
        _phase1_ep_type = type(_phase1_ep).__name__
        _phase1_eq = (_phase1_ep == "phase1_analysis")
        _is_phase1_plan = (
            _phase1_ep == "phase1_analysis"
            and _phase1_rp2 is True
        )
        if not _is_phase1_plan:
            logger.info(
                "[PLAN_VALIDITY_DIAG] exec_phase=%r (type=%s, eq_to_str=%s) "
                "requires_phase2=%r atm_no_edit=%s",
                _phase1_ep, _phase1_ep_type, _phase1_eq,
                _phase1_rp2, atm_no_edit,
            )
        if _is_phase1_plan and not _no_edit_on_modify:
            _exec_state = exec_result.get("state") if isinstance(exec_result, dict) else None
            if _exec_state is not None:
                _fs = getattr(_exec_state, "fix_spec", None)
                if _fs is not None:
                    _fs_targets = _fs.get("targets") or []
                    _fs_clar = _fs.get("clarification_needed")
                    if not _fs_targets and _fs_clar:
                        _phase1_clar_needed = True
                        logger.info(
                            "[PLAN_VALIDITY] Phase 1 analyze_then_modify produced "
                            "clarification_needed with empty targets"
                        )

        #── Phase contract validation: phase1_expected_outputs missing detect ────
        # If phase_contract.phase1_expected_outputs is specified, verify the
        # corresponding output exists in the actual execution state; log a warning if missing.
        _phase_contract_violation = False
        if _is_phase1_plan:
            _pc = getattr(op_plan, "phase_contract", None) or {}
            _expected_outputs: list = _pc.get("phase1_expected_outputs") or []
            if _expected_outputs:
                _exec_state = exec_result.get("state") if isinstance(exec_result, dict) else None
                _missing_outputs: list = []
                # phase1_expected_outputs can name either (a) a top-level
                # ExecutorState attribute (e.g. ``fix_spec``) or (b) a nested
                # key inside ``state.fix_spec`` (e.g. ``scaffold_extraction_verdict``,
                # which is populated by the LLM into the fix_spec dict, not as a
                # separate attribute).  Check both locations before declaring missing.
                for _eo in _expected_outputs:
                    _has_output = False
                    if _exec_state is not None:
                        _attr_val = getattr(_exec_state, _eo, None)
                        if _attr_val not in (None, "", [], {}):
                            _has_output = True
                        else:
                            _fs_dict = getattr(_exec_state, "fix_spec", None)
                            if isinstance(_fs_dict, dict):
                                _nested_val = _fs_dict.get(_eo)
                                if _nested_val not in (None, "", [], {}):
                                    _has_output = True
                    else:
                        _has_output = bool(exec_result.get(_eo))
                    if not _has_output:
                        _missing_outputs.append(_eo)
                if _missing_outputs:
                    _phase_contract_violation = True
                    # Record violation in plan metadata so downstream Phase 2
                    # replan can inspect the root cause.
                    _plan_meta_post["_phase_contract_violation"] = True
                    _plan_meta_post["_phase_contract_missing_outputs"] = list(_missing_outputs)
                    _plan_meta_post["_phase_contract_expected_outputs"] = list(_expected_outputs)
                    logger.warning(
                        "[PHASE_CONTRACT] Phase 1 expected output(s) missing: %s. "
                        "Expected: %s. This may cause downstream failures in Phase 2 replan.",
                        _missing_outputs, _expected_outputs,
                    )
                    if not _phase1_clar_needed:
                        _fs2 = getattr(_exec_state, "fix_spec", {}) if _exec_state else {}
                        if not (_fs2 or {}).get("targets"):
                            _phase1_clar_needed = True
                            logger.info(
                                "[PHASE_CONTRACT] No fix_spec targets + missing expected "
                                "outputs → recorded clarification_needed gate"
                            )
        return _EscalationOutcome(
            exec_result=exec_result,
            op_plan=op_plan,
            spec=spec,
            completed=completed,
            failed=failed,
            exec_status=exec_status,
            exec_detail=exec_detail,
        )
