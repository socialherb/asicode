"""unified_run_record.py — Shared run record model for cross-language learning.

Provides a language-neutral ``UnifiedRunRecord`` that captures the
intersection of Python ``agent.run_store.RunRecord`` and
TS ``ts_vm.learning.models.RunRecord``.  Adapters convert between
language-specific records and the unified format.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class UnifiedRunRecord:
    """Language-neutral execution run record.

    Stored in the unified SQLite store and consumed by the
    cross-language transfer engine.
    """
    run_id: str
    timestamp: float
    language: str               # "python" | "typescript" | "go" | ...
    request: str
    intent: str                 # abstract or language-specific intent
    strategy: str               # language-specific strategy name
    success: bool
    reward: float
    repair_rounds: int = 0
    affected_files: int = 1
    error_types: list[str] = field(default_factory=list)
    context_key: str = ""       # computed
    abstract_strategy: str = "" # language-agnostic strategy (STRUCTURED_EDIT, etc.)
    # Model attribution
    planner_model: str = ""      # model name used as planner (e.g. "deepseek-chat")
    developer_model: str = ""    # model name used as developer (e.g. "claude-sonnet-4-6")
    model_role: str = ""         # 'planner' | 'developer' | 'unified'
    # Test & token metrics (Phase 3 reward signal)
    test_pass_count: int = 0
    test_fail_count: int = 0
    total_tokens: int = 0
    # Outcome details
    final_status: str = ""
    final_failure_class: Optional[str] = None
    completed_ops: int = 0
    failed_ops: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.context_key:
            scope = "single" if self.affected_files <= 1 else "multi"
            self.context_key = f"{self.intent}:{scope}"

    @property
    def scope(self) -> str:
        return "single" if self.affected_files <= 1 else "multi"


# ── Adapters ────────────────────────────────────────────────────────


def from_python_run_record(
    record: Any,
    request: str = "",
) -> UnifiedRunRecord:
    """Convert Python ``agent.run_store.RunRecord`` to unified format."""
    success = getattr(record, "final_status", "") == "success"
    strategy = getattr(record, "selected_strategy", "") or ""
    intent = getattr(record, "plan_mode", "") or ""

    completed = getattr(record, "completed", 0) or 0
    failed = getattr(record, "failed", 0) or 0
    repair_rounds = getattr(record, "repair_rounds_attempted", 0) or 0

    # Count affected files from completed + failed op IDs
    affected = completed + failed

    return UnifiedRunRecord(
        run_id=getattr(record, "run_id", "") or str(uuid.uuid4())[:8],
        timestamp=getattr(record, "timestamp", 0.0) or time.time(),
        language="python",
        request=request,
        intent=intent,
        strategy=strategy,
        success=success,
        reward=getattr(record, "shaped_reward", 0.0) or (1.0 if success else -0.5),
        repair_rounds=repair_rounds,
        affected_files=max(affected, 1),
        error_types=list(getattr(record, "semantic_issue_codes", []) or []),
        final_status=getattr(record, "final_status", ""),
        final_failure_class=getattr(record, "final_failure_class", None),
        completed_ops=completed,
        failed_ops=failed,
        metadata={
            "constraint_mode": getattr(record, "constraint_mode", ""),
        },
    )


def from_ts_run_record(
    record: Any,
) -> UnifiedRunRecord:
    """Convert TS ``ts_vm.learning.models.RunRecord`` to unified format."""
    return UnifiedRunRecord(
        run_id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        language="typescript",
        request=getattr(record, "request", ""),
        intent=getattr(record, "intent", ""),
        strategy=getattr(record, "strategy", ""),
        success=getattr(record, "success", False),
        reward=getattr(record, "reward", 0.0),
        repair_rounds=getattr(record, "repair_rounds", 0),
        affected_files=getattr(record, "affected_files", 1) or 1,
        error_types=list(getattr(record, "error_types", []) or []),
    )
