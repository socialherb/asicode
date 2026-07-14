"""learning_sink.py — Single entry point for all language execution results.

Every language executor (Python, TypeScript, Go, ...) calls ``record_execution()``
to write learning data into unified_runs.db.  Language is a *dimension* of the
record, not a separate system.

Adding a new language:
    1. Call ``record_execution(language="go", strategy="...", ...)``
    2. Add strategy mappings to ``cross_language/strategy_abstraction.py``
    3. Done — no new stores, no new adapters.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level lock to serialize record_execution() calls.
# get_unified_store() creates a new UnifiedStore instance on every call,
# which loads all records from disk.  Concurrent inserts from different
# threads can cause _rewrite_all() (compaction) to clobber each other's
# appends, silently losing records.
_SINK_LOCK = threading.Lock()


def _to_abstract(language: str, strategy: str) -> str:
    """Map a language-specific strategy name to its abstract equivalent."""
    try:
        from external_llm.editor.cross_language.strategy_abstraction import abstract_strategy
        result = abstract_strategy(language, strategy)
        return result.value if hasattr(result, "value") else str(result)
    except Exception:
        return ""


def record_execution(
    language: str,
    strategy: str,
    intent: str,
    success: bool,
    reward: float,
    repair_rounds: int = 0,
    affected_files: int = 1,
    run_id: str = "",
    request: str = "",
    final_status: str = "",
    final_failure_class: Optional[str] = None,
    completed_ops: int = 0,
    failed_ops: int = 0,
    error_types: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    planner_model: str = "",
    developer_model: str = "",
    model_role: str = "",
    test_pass_count: int = 0,
    test_fail_count: int = 0,
    total_tokens: int = 0,
) -> None:
    """Record an execution result from any language executor.

    Best-effort — never raises. Failures are logged at DEBUG level.
    """
    try:
        from external_llm.editor.learning.unified_run_record import UnifiedRunRecord
        from external_llm.editor.learning.unified_store import get_unified_store

        abstract = _to_abstract(language, strategy)
        scope = "single" if affected_files <= 1 else "multi"
        context_key = f"{intent}:{scope}" if intent else scope

        record = UnifiedRunRecord(
            run_id=run_id or str(uuid.uuid4())[:12],
            timestamp=time.time(),
            language=language,
            request=request,
            intent=intent,
            strategy=strategy,
            success=success,
            reward=reward,
            repair_rounds=repair_rounds,
            affected_files=max(affected_files, 1),
            error_types=list(error_types or []),
            context_key=context_key,
            abstract_strategy=abstract,
            planner_model=planner_model or "",
            developer_model=developer_model or "",
            model_role=model_role or "",
            test_pass_count=test_pass_count or 0,
            test_fail_count=test_fail_count or 0,
            total_tokens=total_tokens or 0,
            final_status=final_status or ("success" if success else "failed"),
            final_failure_class=final_failure_class,
            completed_ops=completed_ops,
            failed_ops=failed_ops,
            metadata=metadata or {},
        )

        with _SINK_LOCK:
            store = get_unified_store()
            store.insert(record)
            store.close()

        logger.debug(
            "[SINK] %s/%s (%s) → success=%s reward=%.2f repair=%d",
            language, strategy, abstract, success, reward, repair_rounds,
        )
    except Exception as exc:
        logger.debug("[SINK] record_execution failed: %s", exc)


def update_strategy(run_id: str, language: str, strategy: str) -> None:
    """Update strategy field after post-execution enrichment (e.g. policy trace).

    Also recomputes abstract_strategy so the abstract index stays consistent.
    """
    try:
        from external_llm.editor.learning.unified_store import get_unified_store
        abstract = _to_abstract(language, strategy)
        with _SINK_LOCK:
            store = get_unified_store()
            store.update_strategy(run_id, strategy, abstract_strategy=abstract)
            store._rewrite_all()
            store.close()
        logger.debug("[SINK] update_strategy run_id=%s strategy=%s abstract=%s",
                     run_id, strategy, abstract)
    except Exception as exc:
        logger.debug("[SINK] update_strategy failed: %s", exc)
