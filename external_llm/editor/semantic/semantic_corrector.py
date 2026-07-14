"""semantic_corrector.py — Phase C.2: Contract Violation Reporter.

Phase 5 reform: converted from code-injection corrector to report-only.

Previously this module generated and injected deterministic code patches
(db.add/commit, HTTPException guards, entity creation, etc.) based on
contract violations.  This was "hidden code generation" — the repair
engine was making domain decisions (persistence style, error format,
entity structure) that should be made by the LLM or developer.

Now: detects violations and returns a report.  No file modification.
Violation reports are fed back to the LLM via the execution result so
it can fix issues in the next iteration.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.editor.semantic.semantic_contract_models import (
    SemanticContractReport,
)
from external_llm.editor.semantic.semantic_tracer import SemanticTrace

logger = logging.getLogger(__name__)


def apply_contract_repairs(
    report: SemanticContractReport,
    trace: SemanticTrace,
    repo_root: str,
    context_tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Report contract violations without modifying files.

    Returns dict with violation metadata for LLM feedback.
    No code is injected or modified.
    """
    result: dict[str, Any] = {
        "attempted": False,
        "patches_applied": [],
        "patches_skipped": [],
        "files_modified": [],
        "before_score": report.overall_score,
        "after_score": report.overall_score,
        "resolved_contracts": [],
        "remaining_violations": [],
    }

    failed_results = [r for r in report.results if not r.passed]
    if not failed_results:
        return result

    result["attempted"] = True

    # Report violations without fixing them
    for eval_result in failed_results:
        violation_rules = [v.rule for v in eval_result.violations]
        result["remaining_violations"].append({
            "contract": eval_result.contract_name,
            "rules": violation_rules,
            "severity": max(
                (v.severity for v in eval_result.violations),
                key=lambda s: {"high": 3, "medium": 2, "low": 1}.get(s, 0),
                default="medium",
            ),
        })

    logger.info(
        "[C.2 REPORT] %d contract violations detected (no repair applied): %s",
        len(failed_results),
        [r["contract"] for r in result["remaining_violations"]],
    )

    return result
