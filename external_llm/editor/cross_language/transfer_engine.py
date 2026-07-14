"""transfer_engine.py — Cross-language strategy recording.

Records execution outcomes to the shared JSONL-backed UnifiedStore so that
both Python and TS/JS engines accumulate a common learning history. The
recorded data is consumed by agent query tools and telemetry.

Note: the cross-language *transfer blending* read path (transfer_scores /
rank_with_transfer / suggest_strategies) has been removed — it had no
production callers. Recording remains so the shared store keeps growing;
revive the blending logic from git history if strategy transfer is wired
into ranking later.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from external_llm.editor.cross_language.models import (
    AbstractStrategy,
    Language,
)
from external_llm.editor.cross_language.strategy_abstraction import (
    abstract_strategy,
)
from external_llm.editor.learning.unified_store import UnifiedStore, get_unified_store

logger = logging.getLogger(__name__)


class TransferEngine:
    """Records execution outcomes to the shared cross-language store.

    Usage::

        engine = TransferEngine.from_shared_dir()

        # After a TS execution, record to shared store
        engine.record(
            language="typescript",
            intent="modify_function",
            strategy="symbol_edit",
            success=True, reward=0.85,
        )

        # Convenience: record Python outcome
        engine.record_python_outcome(
            action_type="login", primitive="validate",
            success=True, coverage_delta=0.15,
        )

        engine.save()  # no-op for JSONL-backed store
    """

    def __init__(self, shared_store: UnifiedStore):
        self._store = shared_store

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def from_shared_dir(cls) -> "TransferEngine":
        """Create an engine backed by the default (global) UnifiedStore.

        The cross-language store is intentionally global and shared across every
        task/session so Python and TS/JS pipelines accumulate a common learning
        history at ``~/.asicode/learning/run_history.jsonl`` — there is no
        per-task store, so this factory takes no path argument.
        """
        return cls(shared_store=get_unified_store())

    # ── Record outcomes ─────────────────────────────────────────────

    def record(
        self,
        language: str | Language,
        intent: str,
        strategy: str,
        success: bool,
        reward: float,
        repair_rounds: int = 0,
        affected_files: int = 1,
        error_types: Optional[list[str]] = None,
    ) -> None:
        """Record an execution result to the shared cross-language store."""
        from external_llm.editor.cross_language.strategy_abstraction import abstract_intent as to_abstract_intent
        from external_llm.editor.learning.unified_run_record import UnifiedRunRecord

        lang = Language(language) if isinstance(language, str) else language
        abs_strat = abstract_strategy(lang, strategy)
        abs_intent = to_abstract_intent(lang, intent)
        scope = "single" if affected_files <= 1 else "multi"
        context_key = f"{abs_intent.value}:{scope}"

        record = UnifiedRunRecord(
            run_id=f"cross-{len(list(self._store.iter_all())) + 1}",
            timestamp=__import__("time").time(),
            language=lang.value,
            request="",
            intent=intent,
            strategy=strategy,
            success=success,
            reward=reward,
            repair_rounds=repair_rounds,
            affected_files=affected_files,
            error_types=list(error_types or []),
            context_key=context_key,
            abstract_strategy=abs_strat.value if abs_strat != AbstractStrategy.UNKNOWN else "",
            final_status="success" if success else "failed",
        )
        self._store.insert(record)
        logger.debug(
            "Recorded cross-language: %s/%s -> %s (reward=%.3f)",
            record.language, record.strategy,
            record.abstract_strategy, reward,
        )

    def record_python_outcome(
        self,
        *,
        action_type: str = "",
        primitive: str = "",
        success: bool = False,
        coverage_delta: float = 0.0,
        repair_rounds: int = 0,
    ) -> None:
        """Record a Python learning outcome (convenience wrapper)."""
        intent = action_type or primitive or "unknown"
        strategy = primitive or action_type or "unknown"
        reward = 1.0 if success else -1.0
        if success and coverage_delta > 0:
            reward = min(1.0, reward + coverage_delta * 0.5)

        self.record(
            language="python",
            intent=intent,
            strategy=strategy,
            success=success,
            reward=reward,
            repair_rounds=repair_rounds,
        )

    # ── Persistence helpers ──────────────────────────────────────

    def load(self) -> bool:
        """No-op for JSONL-backed stores (already loaded at init).

        Kept for API compatibility.
        """
        return True

    def save(self) -> bool:
        """No-op for JSONL-backed stores (auto-persisted on insert).

        Kept for API compatibility.
        """
        return True

    def summary(self) -> dict[str, Any]:
        """Quick summary for debugging."""
        py_count = self._store.count("python")
        ts_count = self._store.count("typescript")
        return {
            "total": py_count + ts_count,
            "python": py_count,
            "typescript": ts_count,
            "store_path": self._store._path if self._store._path != ":memory:" else ":memory:",
        }

    @property
    def store(self) -> UnifiedStore:
        return self._store
