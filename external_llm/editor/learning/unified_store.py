"""unified_store.py — JSONL-backed unified run store for cross-language learning.

Single store that accepts ``UnifiedRunRecord`` from all language pipelines.
Provides cross-language queries with age-based decay and size-capped compaction.

Replaces the legacy SQLite-backed store.  Data is stored at
``~/.asicode/learning/run_history.jsonl`` — one JSON object per line.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Iterator
from typing import Any, Optional

from external_llm.common.atomic_io import atomic_write_jsonl
from external_llm.editor.learning.unified_run_record import UnifiedRunRecord

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECORDS = 5000
DEFAULT_DECAY_TAU = 14 * 86400  # 14 days


def _record_to_dict(r: UnifiedRunRecord) -> dict:
    return {
        "run_id": r.run_id,
        "timestamp": r.timestamp,
        "language": r.language,
        "request": r.request,
        "intent": r.intent,
        "strategy": r.strategy,
        "success": r.success,
        "reward": r.reward,
        "repair_rounds": r.repair_rounds,
        "affected_files": r.affected_files,
        "error_types": r.error_types,
        "context_key": r.context_key,
        "abstract_strategy": r.abstract_strategy,
        "planner_model": r.planner_model,
        "developer_model": r.developer_model,
        "model_role": r.model_role,
        "final_status": r.final_status,
        "final_failure_class": r.final_failure_class,
        "completed_ops": r.completed_ops,
        "failed_ops": r.failed_ops,
        "metadata": r.metadata,
        "test_pass_count": r.test_pass_count,
        "test_fail_count": r.test_fail_count,
        "total_tokens": r.total_tokens,
    }


def _dict_to_record(d: dict) -> UnifiedRunRecord:
    return UnifiedRunRecord(
        run_id=d.get("run_id", ""),
        timestamp=d.get("timestamp", 0.0),
        language=d.get("language", ""),
        request=d.get("request", ""),
        intent=d.get("intent", ""),
        strategy=d.get("strategy", ""),
        success=bool(d.get("success", False)),
        reward=d.get("reward", 0.0),
        repair_rounds=d.get("repair_rounds", 0),
        affected_files=d.get("affected_files", 1),
        error_types=list(d.get("error_types", []) or []),
        context_key=d.get("context_key", ""),
        abstract_strategy=d.get("abstract_strategy", ""),
        planner_model=d.get("planner_model", ""),
        developer_model=d.get("developer_model", ""),
        model_role=d.get("model_role", ""),
        final_status=d.get("final_status", ""),
        final_failure_class=d.get("final_failure_class"),
        completed_ops=d.get("completed_ops", 0),
        failed_ops=d.get("failed_ops", 0),
        metadata=dict(d.get("metadata", {}) or {}),
        test_pass_count=d.get("test_pass_count", 0),
        test_fail_count=d.get("test_fail_count", 0),
        total_tokens=d.get("total_tokens", 0),
    )


class UnifiedStore:
    """JSONL-backed unified learning store.

    All records are held in memory for fast queries.  Writes append to the
    JSONL file.  Compaction (``_maybe_compact``) rewrites the file when the
    record count exceeds ``max_records``.
    """

    def __init__(
        self,
        path: str = ":memory:",
        max_records: int = DEFAULT_MAX_RECORDS,
        decay_tau: float = DEFAULT_DECAY_TAU,
    ):
        self._path = path
        self._max_records = max_records
        self._decay_tau = decay_tau
        self._records: list[UnifiedRunRecord] = []

        if path != ":memory:":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._load()

    # ── File I/O ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all records from the JSONL file into memory.

        Resilient to partial corruption: a malformed or non-object line (e.g.
        console output accidentally redirected into the file, or a record
        truncated by a crash during a prior write) is skipped with a warning
        rather than discarding the whole file. Only a totally unreadable file
        starts empty.

        Self-healing: if any lines were skipped, the file is atomically rewritten
        from the valid in-memory state (see :meth:`_heal_file`) so the corruption
        does not persist on disk or re-warn on every subsequent load — normal
        compaction only fires past ``max_records``, so without this a small
        amount of garbage would survive indefinitely. Idempotent: a clean file is
        never rewritten.
        """
        if not os.path.isfile(self._path):
            return
        skipped = 0
        try:
            with open(self._path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        skipped += 1
                        logger.warning(
                            "unified_store: skipping unparseable line %d: %s",
                            lineno, exc,
                        )
                        continue
                    if not isinstance(data, dict):
                        skipped += 1
                        logger.warning(
                            "unified_store: skipping non-object line %d", lineno,
                        )
                        continue
                    if data.get("_revoked"):
                        continue
                    self._records.append(_dict_to_record(data))
        except OSError as exc:
            logger.debug("unified_store: load failed (%s), starting empty", exc)
            self._records = []
            return
        if skipped:
            self._heal_file(skipped)

    def _heal_file(self, skipped: int) -> None:
        """Atomically rewrite the store file from the valid in-memory records.

        Called after :meth:`_load` dropped corrupted/non-object lines, so the
        garbage is removed in one atomic rename (``os.replace``). Unlike
        :meth:`_rewrite_all`, this rewrites even when no records survived — an
        all-garbage file becomes empty rather than accumulating noise forever.
        Uses the same atomic primitive as compaction, so it inherits the same
        crash-safety and (lack of) cross-process guarantees.
        """
        try:
            atomic_write_jsonl(
                self._path,
                (_record_to_dict(r) for r in self._records),
                default=str,
            )
            logger.info(
                "unified_store: healed %s — dropped %d corrupt/non-object line(s), "
                "kept %d record(s)",
                self._path, skipped, len(self._records),
            )
        except Exception as exc:
            logger.debug("unified_store: heal rewrite failed (%s)", exc)

    def _append_to_file(self, d: dict) -> None:
        if self._path == ":memory:":
            return
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, default=str) + "\n")
        except Exception as exc:
            logger.debug("unified_store: append failed (%s)", exc)

    def _rewrite_all(self) -> None:
        """Rewrite the entire JSONL file from in-memory state.

        Atomic: all lines are written to a sibling temp file then renamed into
        place via ``os.replace``, so a crash mid-rewrite never truncates the
        existing store (readers see either the old file or the fully-written new
        one). See :func:`external_llm.common.atomic_io.atomic_write_jsonl`.
        """
        if self._path == ":memory:" or not self._records:
            return
        try:
            atomic_write_jsonl(
                self._path,
                (_record_to_dict(r) for r in self._records),
                default=str,
            )
        except Exception as exc:
            logger.debug("unified_store: rewrite failed (%s)", exc)

    # ── Mutation ──────────────────────────────────────────────────────

    def insert(self, record: UnifiedRunRecord) -> None:
        """Insert a unified run record."""
        self._records.append(record)
        self._append_to_file(_record_to_dict(record))
        self._maybe_compact()

    def update_strategy(
        self, run_id: str, strategy: str, abstract_strategy: str = "",
    ) -> None:
        """Update the strategy fields for an existing record (in-memory only).

        The change is durable only after the next compaction.  If the process
        crashes before compaction, the original strategy is preserved on disk.
        """
        if not strategy:
            return
        for r in self._records:
            if r.run_id == run_id:
                r.strategy = strategy
                if abstract_strategy:
                    r.abstract_strategy = abstract_strategy
                break

    def clear(self) -> None:
        """Remove all records (in-memory and on disk)."""
        self._records = []
        if self._path != ":memory:" and os.path.isfile(self._path):
            try:
                os.remove(self._path)
            except Exception:
                pass

    def close(self) -> None:
        """No-op for JSONL backend (kept for API compatibility)."""
        pass

    # ── Compaction ────────────────────────────────────────────────────

    def _maybe_compact(self) -> None:
        if len(self._records) <= self._max_records:
            return
        excess = len(self._records) - self._max_records
        self._records = self._records[excess:]
        logger.debug("Compacted %d old unified records (kept %d)", excess, self._max_records)
        self._rewrite_all()

    # ── Query helpers ─────────────────────────────────────────────────

    def _filter(
        self,
        *,
        language: Optional[str] = None,
        strategy: Optional[str] = None,
        abstract_strategy: Optional[str] = None,
        exclude_language: Optional[str] = None,
        planner_model: Optional[str] = None,
        developer_model: Optional[str] = None,
        context_key: Optional[str] = None,
    ) -> list[UnifiedRunRecord]:
        """Filter records by criteria (AND logic)."""
        result = self._records
        if language is not None:
            result = [r for r in result if r.language == language]
        if strategy is not None:
            result = [r for r in result if r.strategy == strategy]
        if abstract_strategy is not None:
            result = [r for r in result if r.abstract_strategy == abstract_strategy]
        if exclude_language is not None:
            result = [r for r in result if r.language != exclude_language]
        if planner_model is not None:
            result = [r for r in result if r.planner_model == planner_model]
        if developer_model is not None:
            result = [r for r in result if r.developer_model == developer_model]
        if context_key is not None:
            result = [r for r in result if r.context_key == context_key]
        return result

    # ── Public query API (UnifiedStore original) ───────────────────────

    def get_recent(
        self,
        language: Optional[str] = None,
        limit: int = 50,
    ) -> list[UnifiedRunRecord]:
        """Get recent records, optionally filtered by language."""
        filtered = self._filter(language=language)
        return list(reversed(filtered))[:limit]

    def _weighted_success_rate(self, filtered: list) -> tuple[float, int]:
        """Compute age-decayed success rate from a pre-filtered record list."""
        if not filtered:
            return 0.0, 0
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for r in filtered[-200:]:  # limit to 200 most recent
            w = math.exp(-(now - r.timestamp) / self._decay_tau)
            weighted_sum += (1.0 if r.success else 0.0) * w
            weight_total += w
        rate = weighted_sum / weight_total if weight_total > 0 else 0.0
        return rate, len(filtered)

    def success_rate(
        self,
        strategy: str,
        language: Optional[str] = None,
    ) -> tuple[float, int]:
        """Get (success_rate, count) for a strategy with age-based decay."""
        filtered = self._filter(strategy=strategy, language=language)
        return self._weighted_success_rate(filtered)

    def cross_language_success_rate(
        self,
        strategy: str,
        exclude_language: str,
    ) -> tuple[float, int]:
        """Get success rate from OTHER languages for a strategy."""
        filtered = self._filter(strategy=strategy, exclude_language=exclude_language)
        return self._weighted_success_rate(filtered)

    def get_strategy_runs(
        self,
        strategy: str,
        language: Optional[str] = None,
        limit: int = 50,
    ) -> list[UnifiedRunRecord]:
        """Get recent runs for a specific (language-specific) strategy."""
        filtered = self._filter(strategy=strategy, language=language)
        return list(reversed(filtered))[:limit]

    def get_runs_by_abstract_strategy(
        self,
        abstract_strategy: str,
        language: Optional[str] = None,
        limit: int = 50,
    ) -> list[UnifiedRunRecord]:
        """Get runs by abstract strategy — language-agnostic cross-language query.

        When language is given, same-language records are returned first for
        transfer-penalty calculation.
        """
        candidates = [r for r in self._records if r.abstract_strategy == abstract_strategy]
        if language is None:
            return list(reversed(candidates))[:limit]
        same = [r for r in candidates if r.language == language]
        cross = [r for r in candidates if r.language != language]
        # Most-recent-first within each group
        same.sort(key=lambda r: r.timestamp, reverse=True)
        cross.sort(key=lambda r: r.timestamp, reverse=True)
        merged = same + cross
        return merged[:limit]

    def get_runs_by_model(
        self,
        planner_model: Optional[str] = None,
        developer_model: Optional[str] = None,
        limit: int = 50,
    ) -> list[UnifiedRunRecord]:
        """Get runs filtered by model name."""
        filtered = self._filter(
            planner_model=planner_model,
            developer_model=developer_model,
        )
        # Same-model records first (both planner and developer match)
        same = []
        rest = []
        for r in reversed(filtered):
            pm = r.planner_model or ""
            dm = r.developer_model or ""
            if (not planner_model or pm == planner_model) and \
               (not developer_model or dm == developer_model):
                same.append(r)
            else:
                rest.append(r)
        merged = same + rest
        return merged[:limit]

    def get_model_stats(self) -> dict[str, Any]:
        """Return per-model success rate and repair statistics."""
        buckets: dict[str, dict] = {}
        for r in self._records:
            pm = r.planner_model or ""
            dm = r.developer_model or ""
            if not pm and not dm:
                continue
            key = f"{pm or '?'}/{dm or '?'}"
            if key not in buckets:
                buckets[key] = {
                    "planner_model": pm,
                    "developer_model": dm,
                    "total": 0,
                    "successes": 0,
                    "repair_sum": 0.0,
                    "reward_sum": 0.0,
                }
            b = buckets[key]
            b["total"] += 1
            if r.success:
                b["successes"] += 1
            b["repair_sum"] += r.repair_rounds
            b["reward_sum"] += r.reward

        result: dict[str, Any] = {}
        for key, b in buckets.items():
            total = b["total"]
            result[key] = {
                "planner_model": b["planner_model"],
                "developer_model": b["developer_model"],
                "total": total,
                "success_rate": round((b["successes"] or 0) / total, 3) if total else 0.0,
                "avg_repair_rounds": round(b["repair_sum"] / total, 2) if total else 0.0,
                "avg_reward": round(b["reward_sum"] / total, 3) if total else 0.0,
            }
        return result

    def count(self, language: Optional[str] = None) -> int:
        if language is None:
            return len(self._records)
        return sum(1 for r in self._records if r.language == language)

    # ── CrossLanguageStore-compatible query API ─────────────────────────

    def load_strategy_scores(
        self,
        abstract_strategy: str,
        context_key: Optional[str] = None,
        exclude_language: Optional[str] = None,
        limit: int = 200,
    ) -> list[tuple[str, float, float]]:
        """Load records for an abstract strategy.

        Returns: [(language, reward, decayed_reward), ...]
        """
        candidates = self._filter(
            abstract_strategy=abstract_strategy,
            exclude_language=exclude_language,
            context_key=context_key,
        )
        now = time.time()
        results = []
        for r in reversed(candidates[-limit:]):
            age = now - r.timestamp
            decayed = r.reward * math.exp(-age / self._decay_tau)
            results.append((r.language, r.reward, decayed))
        return results

    def aggregate_strategy_score(
        self,
        abstract_strategy: str,
        context_key: Optional[str] = None,
        exclude_language: Optional[str] = None,
        limit: int = 200,
    ) -> tuple[float, int]:
        """Get aggregated (mean decayed reward, count) for an abstract strategy."""
        rows = self.load_strategy_scores(
            abstract_strategy, context_key, exclude_language, limit)
        if not rows:
            return 0.0, 0
        total = sum(r[2] for r in rows)
        return total / len(rows), len(rows)

    def load_context_scores(
        self,
        context_key: str,
        exclude_language: Optional[str] = None,
        limit: int = 200,
    ) -> dict[str, tuple[float, int]]:
        """Load scores for all strategies in a context.

        Returns: {abstract_strategy: (mean_decayed_reward, count)}
        """
        candidates = self._filter(
            exclude_language=exclude_language,
            context_key=context_key,
        )
        now = time.time()
        buckets: dict[str, list[float]] = {}
        for r in reversed(candidates[-limit:]):
            age = now - r.timestamp
            decayed = r.reward * math.exp(-age / self._decay_tau)
            buckets.setdefault(r.abstract_strategy, []).append(decayed)
        return {
            strat: (sum(rewards) / len(rewards), len(rewards))
            for strat, rewards in buckets.items()
        }

    # ── Tool-specific helpers ─────────────────────────────────────────

    def get_plan_id_outcomes(self) -> dict[str, dict[str, Any]]:
        """Return ``{plan_id: outcome_row}`` for placement_label_join.

        Only records whose ``metadata`` has a non-empty ``plan_id`` are indexed.
        When multiple records share a plan_id, the most recent wins.
        """
        out: dict[str, dict[str, Any]] = {}
        for r in self._records:
            plan_id = r.metadata.get("plan_id") if isinstance(r.metadata, dict) else None
            if not plan_id or not isinstance(plan_id, str):
                continue
            # Most-recent-wins: later records overwrite.
            out[plan_id] = {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "success": 1 if r.success else 0,
                "final_status": r.final_status or "",
                "final_failure_class": r.final_failure_class,
                "completed_ops": r.completed_ops or 0,
                "failed_ops": r.failed_ops or 0,
            }
        return out

    def iter_all(self) -> Iterator[UnifiedRunRecord]:
        """Iterate over all records (for tools that need full scan)."""
        yield from self._records

    def get_strategy_stats(self, limit: int = 30, exclude: tuple = ()) -> dict[str, dict]:
        """Per-strategy success stats for kp_correctness_verify.

        Returns: {strategy: {"ok": count, "total": count, "reward_sum": float}}
        """
        stats: dict[str, dict] = {}
        for r in reversed(self._records[-limit * 10:]):  # oversample
            if r.strategy in exclude or not r.strategy:
                continue
            if r.strategy not in stats:
                stats[r.strategy] = {"ok": 0, "total": 0, "reward_sum": 0.0}
            stats[r.strategy]["total"] += 1
            if r.success:
                stats[r.strategy]["ok"] += 1
            stats[r.strategy]["reward_sum"] += r.reward
            if sum(s["total"] for s in stats.values()) >= limit:
                break
        return stats


def get_unified_store(
    project_root: Optional[str] = None,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> UnifiedStore:
    """Get the unified store for a project.

    JSONL is stored at ``{project_root}/.asicode/learning/run_history.jsonl``.
    Falls back to ``~/.asicode/learning/run_history.jsonl``.
    """
    if project_root:
        base_dir = os.path.join(project_root, ".asicode")
    else:
        base_dir = os.path.join(os.path.expanduser("~"), ".asicode")
    jsonl_path = os.path.join(base_dir, "learning", "run_history.jsonl")
    return UnifiedStore(jsonl_path, max_records=max_records)
