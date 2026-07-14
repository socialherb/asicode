"""primitive_learning_store.py — Phase F.1: Primitive Learning Storage.

In-memory store with serialization for primitive outcome records.
Thread-safe via simple dict operations (no concurrent writes expected).
"""
from __future__ import annotations
from typing import Any, Optional

from external_llm.editor.learning.primitive_learning_models import (
    PrimitiveLearningKey,
    PrimitiveOutcomeRecord,
    PrimitiveStrategyStats,
)
class PrimitiveLearningStore:
    """Store for primitive learning records."""

    def __init__(self):
        self._records: dict[str, PrimitiveOutcomeRecord] = {}
        # key_str → PrimitiveOutcomeRecord
        self._strategy_stats: dict[str, dict[str, PrimitiveStrategyStats]] = {}
        # primitive → {strategy_name → stats}

    def update(
        self,
        key: PrimitiveLearningKey,
        chosen: bool,
        improved: bool,
        passed: bool,
        coverage_delta: float = 0.0,
        sem_delta: float = 0.0,
        contract_delta: float = 0.0,
        strategy_name: str = "",
    ) -> None:
        """Update record for a primitive learning key."""
        k = key.to_str()
        if k not in self._records:
            self._records[k] = PrimitiveOutcomeRecord()

        rec = self._records[k]
        rec.uses += 1
        if chosen:
            rec.chosen_count += 1
        if improved:
            rec.improved_count += 1
        if passed:
            rec.pass_count += 1
        rec.total_coverage_delta += coverage_delta
        rec.total_sem_delta += sem_delta
        rec.total_contract_delta += contract_delta

        # Strategy stats
        if strategy_name:
            prim = key.primitive
            if prim not in self._strategy_stats:
                self._strategy_stats[prim] = {}
            if strategy_name not in self._strategy_stats[prim]:
                self._strategy_stats[prim][strategy_name] = PrimitiveStrategyStats(
                    strategy_name=strategy_name,
                )
            ss = self._strategy_stats[prim][strategy_name]
            ss.uses += 1
            if improved:
                ss.success_count += 1
            ss.total_gain += coverage_delta

    def get(self, key: PrimitiveLearningKey) -> Optional[PrimitiveOutcomeRecord]:
        """Get record for exact key."""
        return self._records.get(key.to_str())

    def lookup_by_primitive(self, primitive: str) -> dict[str, PrimitiveOutcomeRecord]:
        """Find all records for a specific primitive across all contexts."""
        result = {}
        for k, rec in self._records.items():
            key = PrimitiveLearningKey.from_str(k)
            if key.primitive == primitive:
                result[k] = rec
        return result

    def iter_typed(self):
        """Iterate records as (PrimitiveLearningKey, PrimitiveOutcomeRecord) pairs.

        Safe typed iteration — no string split guessing.
        """
        for k, rec in self._records.items():
            yield PrimitiveLearningKey.from_str(k), rec

    def get_strategy_stats(self, primitive: str) -> dict[str, PrimitiveStrategyStats]:
        """Get strategy stats for a primitive."""
        return self._strategy_stats.get(primitive, {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize store."""
        return {
            "records": {k: v.to_dict() for k, v in self._records.items()},
            "strategy_stats": {
                prim: {sn: ss.to_dict() for sn, ss in strats.items()}
                for prim, strats in self._strategy_stats.items()
            },
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Deserialize store."""
        for k, v in data.get("records", {}).items():
            self._records[k] = PrimitiveOutcomeRecord(
                uses=v.get("uses", 0),
                chosen_count=v.get("chosen", 0),
                improved_count=v.get("improved", 0),
                pass_count=v.get("passed", 0),
                total_coverage_delta=v.get("total_coverage_delta", 0.0),
                total_sem_delta=v.get("total_sem_delta", 0.0),
                total_contract_delta=v.get("total_contract_delta", 0.0),
            )
        for prim, strats in data.get("strategy_stats", {}).items():
            self._strategy_stats[prim] = {}
            for sn, sv in strats.items():
                self._strategy_stats[prim][sn] = PrimitiveStrategyStats(
                    strategy_name=sv.get("strategy", sn),
                    uses=sv.get("uses", 0),
                    success_count=sv.get("success_count", 0),
                    total_gain=sv.get("total_gain", 0.0),
                )

    @property
    def total_records(self) -> int:
        return len(self._records)

    @property
    def total_uses(self) -> int:
        return sum(r.uses for r in self._records.values())
