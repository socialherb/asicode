"""
In-memory run store for execution results and repair memory.

Provides deterministic FIFO storage of recent run records and repair memories
to support P14: RunStore + Execution Memory + Repair Learning.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.operation_models import (
    StrategyOutcomeMemory,
    StrategyOutcomeStats,
    SwitchOutcomeMemory,
    SwitchOutcomeStats,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = _cfg.counts.RUN_STORE_TOP_K


@dataclass
class RunRecord:
    """Complete record of a single plan execution run."""
    run_id: str
    timestamp: float
    plan_mode: str
    operation_count: int
    completed: int
    failed: int
    skipped: int
    final_status: str
    final_failure_class: Optional[str]
    final_blocking_reasons: list[str]
    final_warning_reasons: list[str]
    semantic_gate_passed: bool
    semantic_gate_failed_reasons: list[str]
    plan_acceptance_passed: bool
    plan_acceptance_failed_checks: list[str]
    repair_attempted: bool
    repair_rounds_attempted: int
    repair_improved: bool
    semantic_issue_codes: list[str]
    dependency_issues: list[str]
    completed_ids: list[str]
    failed_ids: list[str]
    skipped_ids: list[str]
    skipped_reasons: dict[str, str] = field(default_factory=dict)
    proof: dict[str, Any] = field(default_factory=dict)
    candidate_feedback: dict[str, Any] = field(default_factory=dict)
    # P11: Policy trace fields
    selected_strategy: str = ""
    constraint_mode: str = ""
    selection_temperature: float = 0.0
    temperature_explored: bool = False
    diversity_summary: dict[str, float] = field(default_factory=dict)
    policy_features: dict[str, Any] = field(default_factory=dict)
    shaped_reward: float = 0.0
    # Test & token metrics
    test_pass_count: int = 0
    test_fail_count: int = 0
    total_tokens: int = 0
    # LLM-classified request type (from spec.request_type)
    request_type: str = ""
    # Step 3 (Option B): plan_id stamped at planner time (UUID stored in
    # plan.metadata["plan_id"]).  Persisted into unified_runs.metadata so
    plan_id: str = ""


@dataclass
class StrategyExecutionStats:
    """Decomposition-aware strategy execution statistics."""
    request_type: str
    estimated_scope: str
    used_decomposition: bool
    strategy_name: str
    strategy_chain: Optional[list[str]] = None
    success_count: int = 0
    failure_count: int = 0
    repair_count: int = 0
    switch_count: int = 0


@dataclass
class RepairMemoryEntry:
    """Compact entry for repair memory lookup."""
    run_id: str
    final_failure_class: Optional[str]
    final_status: str
    blocking_reasons: list[str]
    warning_reasons: list[str]
    repair_attempted: bool
    repair_rounds_attempted: int
    repair_improved: bool
    semantic_issue_codes: list[str]


@dataclass
class FailurePatternSummary:
    """Deterministic summary of recent failure patterns for a failure class."""
    failure_class: Optional[str]
    total_runs: int
    repaired_runs: int
    improved_runs: int
    success_after_repair_runs: int
    common_blocking_reasons: list[str]
    common_semantic_issue_codes: list[str]
    common_skipped_reason_prefixes: list[str]
    last_seen_run_id: Optional[str]


class InMemoryRunStore:
    """In-memory FIFO store for run records and repair memories."""

    def __init__(self, max_runs: int = 100, model_name: str = "",
                 write_unified: bool = False):
        self.max_runs = max_runs
        self._runs: dict[str, RunRecord] = {}
        self._run_order: list[str] = []
        self._next_run_id = 1  # deterministic monotonic counter
        # Guard: only write to unified_runs.db in production contexts.
        # Unit tests create InMemoryRunStore() with default write_unified=False,
        # preventing test data from polluting cross-session learning DB.
        # Production code (operation_executor, repair_engine) passes write_unified=True.
        self._write_unified_enabled: bool = write_unified
        # Decomposition-aware strategy execution history
        # Key: (request_type, estimated_scope, used_decomposition, strategy_name)
        self._strategy_exec_stats: dict[tuple, StrategyExecutionStats] = {}
        # Chain history key: (request_type, estimated_scope, used_decomposition, chain_key)
        self._chain_exec_stats: dict[tuple, StrategyExecutionStats] = {}
        # Planner memory signals: contract repair outcomes for bias injection
        self._contract_repair_signals: list[dict] = []
        self._max_contract_signals: int = 200
        # Strategy-level outcome signals: per-operation strategy performance
        self._strategy_outcome_signals: list[dict] = []
        self._max_strategy_signals: int = 200
        # Weight learning state — lazy-initialised on first access
        self._weight_learner: Any = None
        # Execution learning state — lazy-initialised on first access
        self._execution_learner: Any = None
        # Strategy policy learner — lazy-initialised on first access
        self._policy_learner: Any = None
        # Model context is PER-THREAD (threading.local), not shared instance state.
        # The run_store is a process-lifetime singleton shared across concurrent
        # sessions, each running in its own agent_executor thread. A shared instance
        # field would let session B's set_model_context overwrite session A's model
        # mid-run, so A's run-completion telemetry (_write_to_unified_store) and
        # planner bias reads (planner_policy_adapter) would attribute to B's model —
        # silently corrupting per-model strategy learning data. _model_name /
        # _developer_model_name are properties backed by this thread-local; seeding
        # here sets the constructing thread's context (the global singleton is built
        # once on the import thread).
        self._model_ctx = threading.local()
        self._model_name = self._normalize_model_name(model_name)
        self._developer_model_name = ""
        # Adaptive learner hub (tool/patch/context/routing/prompt)
        self._adaptive_hub: Any = None
        # Concurrency: parallel sub-agents share this single run_store instance and
        # each fires the run-completion telemetry hook (_record_p8_strategy_learning,
        # ~12 RMW calls per run) from inside the sub-agent's worker thread. The
        # telemetry writers below do read-modify-write on shared dicts/lists with no
        # atomicity, so concurrent completions lose updates and corrupt streak/history.
        # RLock chosen so a writer may re-enter the same lock (and so a future writer
        # may call another writer/reader safely). These methods fire once per run →
        # contention is negligible; the lock only serializes the RMW critical sections.
        self._telemetry_lock = threading.RLock()
        # P8 minimal: strategy → reward history
        self._strategy_rewards: dict[str, list[dict]] = {}
        self._max_reward_history: int = 50
        # P8.3: failure_type → repair_action → outcome history
        self._repair_outcomes: dict[str, dict[str, list[float]]] = {}
        self._max_repair_history: int = 50
        # P9: EMA-based reward smoothing
        self._strategy_ema: dict[str, float] = {}
        self._strategy_context_ema: dict[str, dict[str, float]] = {}
        self._repair_ema: dict[str, dict[str, float]] = {}  # failure→action→ema
        # Reflection patterns (for P11/P12 evolution pipeline)
        self._reflection_patterns: dict[str, list[dict]] = {}
        self._max_reflection_per_class: int = 50
        # P11: distilled rules (compressed learned policies)
        self._distilled_rules: list[dict] = []
        # P12: dynamic strategies (proposed by MetaStrategyEngine)
        self._dynamic_strategies: dict[str, dict] = {}
        # P12.2: evolution cooldown state
        self._evolution_state = {
            "cooldown_runs": 2,
            "runs_since_last_evolve": 0,
        }
        # multi_strategy_gate: per-state cooldown tracking
        # key: state_key (str) → {"no_gain_count": int, "cooldown_remaining": int}
        self._multi_gate_state: dict[str, dict] = {}
        # P12.2: strategy attribution (strategy → context → ema_reward)
        self._strategy_attribution: dict[str, dict[str, float]] = {}
        # P12.2: strategy versioning
        self._strategy_versions: dict[str, int] = {}  # base_name → latest version
        # P13.1: role-level learning
        self._role_ema: dict[str, float] = {}  # role → ema_reward
        self._role_context_ema: dict[str, dict[str, float]] = {}  # role → context → ema
        # P14/P14.1: difficulty tracking with stabilization
        self._difficulty_state: dict[str, Any] = {
            "current": 0.3,
            "history": [],
            "max_history": 100,
            "ema_signal": 0.5,
            "streak_success": 0,
            "streak_fail": 0,
        }
        # Phase 6: Learned Policy Engine
        self._learned_policy: Any = None
        self._init_learned_policy()
        self._migrate_legacy_state()

    def _migrate_legacy_state(self) -> None:
        """One-time migration from legacy per-file state to strategy_state.json."""
        try:
            from external_llm.editor.learning.strategy_state import read_namespace, write_namespace
        except ImportError:
            return
        import json

        _migrations = (
            ("weights", "weight_state.json"),
            ("policy", "policy_state.json"),
            ("adaptive_hub", "adaptive_hub_state.json"),
            ("execution_state", "execution_state.json"),
        )
        model_dir = self._model_dir()
        for ns_base, filename in _migrations:
            ns = f"{ns_base}/{self._model_name}" if self._model_name else ns_base
            if read_namespace(ns) is not None:
                continue
            legacy = os.path.join(model_dir, filename)
            if not os.path.isfile(legacy):
                continue
            try:
                with open(legacy, encoding="utf-8") as fh:
                    state = json.load(fh)
                if isinstance(state, dict) and state:
                    write_namespace(ns, state)
                    logger.info("run_store: migrated %s -> strategy_state:%s", legacy, ns)
            except Exception:
                logger.debug("run_store: %s migration error", legacy, exc_info=True)
        # Shared knowledge
        if read_namespace("transferable_knowledge") is None:
            shared_path = os.path.join(self._shared_dir(), "transferable_knowledge.json")
            if os.path.isfile(shared_path):
                try:
                    with open(shared_path, encoding="utf-8") as fh:
                        state = json.load(fh)
                    if isinstance(state, dict) and state:
                        write_namespace("transferable_knowledge", state)
                        logger.info("run_store: migrated %s -> strategy_state:transferable_knowledge", shared_path)
                except Exception:
                    logger.debug("run_store: transferable_knowledge migration error", exc_info=True)

    def _init_learned_policy(self) -> None:
        """Initialize Phase 6 LearnedPolicyEngine."""
        try:
            from .learned_policy import LearnedPolicyEngine
            self._learned_policy = LearnedPolicyEngine()
        except (ImportError, AttributeError):
            self._learned_policy = None

    def get_learned_policy(self) -> Any:
        """Return the LearnedPolicyEngine instance."""
        if self._learned_policy is None:
            self._init_learned_policy()
        return self._learned_policy

    def record_semantic_fit_verdict(
        self,
        request: str,
        verdict: dict,
        spec_snapshot: Optional[dict] = None,
        attempt: int = 0,
    ) -> None:
        """Record a semantic-fit judge verdict for future calibration.

        Stored as an append-only list bounded to ~200 entries. Calibration
        code can later join these verdicts to plan outcomes (completed/failed
        ratio) to detect systematic over/under-confidence in a given action.

        Args:
            request:       User request (truncated).
            verdict:       Dict form of SemanticFitVerdict (to_dict()).
            spec_snapshot: Optional compact spec summary (target_files, etc.).
            attempt:       Re-explore attempt index (0 = initial).
        """
        if not hasattr(self, "_semantic_fit_verdicts"):
            self._semantic_fit_verdicts: list[dict] = []
            self._max_semantic_fit_verdicts: int = 200
        try:
            _entry = {
                "ts":             time.time(),
                "request":        (request or "")[:240],
                "attempt":        int(attempt),
                "semantic_fit":   verdict.get("semantic_fit", ""),
                "action":         verdict.get("action", ""),
                "confidence":     float(verdict.get("confidence") or 0.0),
                "reason":         (verdict.get("reason") or "")[:240],
                "reexplore_guidance": verdict.get("reexplore_guidance") or {},
                "spec":           spec_snapshot or {},
            }
            self._semantic_fit_verdicts.append(_entry)
            # Bound: trim oldest half when over limit (amortised O(1))
            if len(self._semantic_fit_verdicts) > self._max_semantic_fit_verdicts:
                _keep = self._max_semantic_fit_verdicts // 2
                self._semantic_fit_verdicts = self._semantic_fit_verdicts[-_keep:]
        except Exception:
            # Learning is best-effort — never let it break the pipeline
            pass

    def recent_semantic_fit_verdicts(self, limit: int = 50) -> list[dict]:
        """Return the N most recent verdicts (newest last)."""
        _verdicts = getattr(self, "_semantic_fit_verdicts", [])
        return list(_verdicts[-limit:])

    def semantic_fit_bias_note(self, window: int = 30) -> str:
        """One-line summary of recent verdict distribution, for prompt self-bias.

        Injected into the judge's user prompt so the LLM can notice if it has
        been systematically skewed toward a single action recently. This is a
        light-touch calibration signal — NOT a directive — and returns an
        empty string when the sample is too small or the distribution is
        balanced. Shape:

            "RECENT VERDICTS (last N=12): ACCEPT 50%, RE_EXPLORE 33%,
             ANALYZE_FIRST 17%, CLARIFY 0% — be cautious of over-skew."

        Returns empty string when: no verdicts, < 8 samples, or the top
        action is < 50% (distribution considered balanced).
        """
        _verdicts = getattr(self, "_semantic_fit_verdicts", [])
        if not _verdicts or len(_verdicts) < 8:
            return ""
        _sample = _verdicts[-window:]
        _n = len(_sample)
        _counts: dict[str, int] = {"ACCEPT": 0, "ANALYZE_FIRST": 0, "RE_EXPLORE": 0, "CLARIFY": 0}
        for _v in _sample:
            _a = str(_v.get("action") or "").upper()
            if _a in _counts:
                _counts[_a] += 1
        _pcts = {k: round(100.0 * v / _n) for k, v in _counts.items()}
        _max_action = max(_pcts, key=_pcts.get)
        _max_pct = _pcts[_max_action]
        # Only emit when there is a meaningful skew — avoids noise
        if _max_pct < 50:
            return ""
        _parts = [f"{k} {v}%" for k, v in _pcts.items() if v > 0]
        return (
            f"RECENT VERDICTS (last N={_n}): {', '.join(_parts)} — "
            f"if you are about to return '{_max_action}' again, check that the "
            f"evidence actually supports it; otherwise prefer the runner-up."
        )

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """Normalize model name for filesystem use."""
        if not name:
            return ""
        return ''.join(c if c.isalnum() or c in '._-' else '_' for c in name.strip())[:50]

    def _model_dir(self) -> str:
        """Return model-specific subdirectory under runs/."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if self._model_name:
            return os.path.join(project_root, "runs", "models", self._model_name)
        return os.path.join(project_root, "runs")

    def _shared_dir(self) -> str:
        """Return shared (model-independent) directory under runs/."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(project_root, "runs", "shared")

    # ── Per-thread model context ─────────────────────────────────
    # ``_model_name`` / ``_developer_model_name`` are properties backed by
    # ``threading.local`` (see __init__) so each session's worker thread sees only
    # its own model. Legacy callers read the raw attribute — e.g.
    # ``getattr(run_store, "_model_name", "")`` in planner_policy_adapter — and the
    # property keeps that working while making the value thread-isolated. Without
    # this, concurrent sessions sharing the singleton store overwrite each other's
    # model context (a read-modify race on instance state).
    @property
    def _model_name(self) -> str:
        return getattr(self._model_ctx, "planner", "")

    @_model_name.setter
    def _model_name(self, value: str) -> None:
        self._model_ctx.planner = value

    @property
    def _developer_model_name(self) -> str:
        return getattr(self._model_ctx, "developer", "")

    @_developer_model_name.setter
    def _developer_model_name(self, value: str) -> None:
        self._model_ctx.developer = value

    @contextmanager
    def model_context_scope(self, planner_model: str = "", developer_model: str = ""):
        """Temporarily bind thread-local model context, restoring the prior value on exit.

        The orchestrator binds a sub-agent's model to its worker thread via this scope
        so run-completion telemetry (_write_to_unified_store) and planner bias reads
        attribute to the correct model rather than the parent session's planner model
        (sub-agents share the singleton run_store but run on distinct worker threads).
        Parallel sub-agents are naturally isolated by ``threading.local``; the
        save/restore matters for sequential mode, which reuses the parent thread and
        would otherwise leak the last sub-agent's model into subsequent parent work.
        """
        _prev_p = getattr(self._model_ctx, "planner", "")
        _prev_d = getattr(self._model_ctx, "developer", "")
        self.set_model_context(planner_model=planner_model, developer_model=developer_model)
        try:
            yield
        finally:
            self.set_model_context(planner_model=_prev_p, developer_model=_prev_d)

    def set_model_context(
        self,
        planner_model: str = "",
        developer_model: str = "",
    ) -> None:
        """Update model names for subsequent record_execution() calls.

        Called per-request so the singleton run_store knows which models are active
        for the CALLING THREAD without requiring per-request instantiation. Backed by
        ``threading.local`` (see ``_model_name``/``_developer_model_name`` properties),
        so concurrent sessions on separate worker threads never observe each other's
        model context.
        """
        self._model_name = self._normalize_model_name(planner_model)
        self._developer_model_name = self._normalize_model_name(developer_model)

    def create_run_id(self) -> str:
        """Generate deterministic monotonic run ID.

        ``_telemetry_lock`` serializes the read-increment RMW: without it two
        concurrent callers (parallel sub-agents share this singleton store) can read
        the same ``_next_run_id`` and produce duplicate IDs, after which ``add_run``
        treats the second run as an "update" of the first and silently merges two
        distinct runs into one record. RLock so it composes with ``add_run``'s own
        acquisition (create_run_id is normally called before add_run, never nested).
        """
        with self._telemetry_lock:
            run_id = f"run-{self._next_run_id:06d}"
            self._next_run_id += 1
        return run_id

    def enrich_run_with_policy_trace(
        self,
        run_id: str,
        selection_metadata: Optional[dict[str, Any]] = None,
        shaped_reward: float = 0.0,
    ) -> None:
        """Attach P11 policy trace to an existing RunRecord.

        Called after execution completes so that learning modules can
        correlate strategy/policy choices with outcomes.
        """
        record = self._runs.get(run_id)
        if record is None:
            return
        if selection_metadata:
            record.selected_strategy = str(selection_metadata.get("selected_strategy", ""))
            record.constraint_mode = str(selection_metadata.get("constraint_mode", ""))
            record.selection_temperature = float(selection_metadata.get("selection_temperature", 0.0))
            record.temperature_explored = bool(selection_metadata.get("temperature_explored", False))
            record.diversity_summary = selection_metadata.get("diversity_summary") or {}
            record.policy_features = selection_metadata.get("policy_features") or {}
        record.shaped_reward = shaped_reward

    def add_run(self, record: RunRecord) -> None:
        """Add a run record, enforcing FIFO eviction if needed.

        ``_telemetry_lock`` serializes the read-check-evict-append RMW against
        concurrent run completions in parallel sub-agents, which share this store
        via ``ThreadPoolExecutor`` (orchestrator passes ``run_store=self._run_store``
        into each in-process sub-agent's operation executor). Without the lock, two
        threads at capacity both read the same ``_run_order[0]`` / ``len(self._runs)``
        and either double-evict (corrupting ``_run_order`` and silently dropping a
        valid run) or raise ``KeyError`` on the second ``del`` — the same class of
        RMW bug ``record_strategy_reward`` / ``update_difficulty`` already guard
        against. ``add_run`` is the ONLY writer to ``_runs`` / ``_run_order``.
        """
        with self._telemetry_lock:
            if record.run_id in self._runs:
                # Update existing: remove from order to re-add at end (most recent)
                self._run_order.remove(record.run_id)
            else:
                # New run: evict oldest if at capacity
                if len(self._runs) >= self.max_runs:
                    oldest_id = self._run_order[0]
                    del self._runs[oldest_id]
                    self._run_order.pop(0)

            self._runs[record.run_id] = record
            self._run_order.append(record.run_id)

        # Write-through to unified cross-language store. External I/O — kept OUTSIDE
        # the lock to minimize hold time; the learning sink has its own concurrency
        # control and only reads ``record`` (never shared mutable store state).
        self._write_to_unified_store(record)

    def _write_to_unified_store(self, record: RunRecord) -> None:
        """Write-through to unified_runs.db via learning_sink (best-effort).

        Only executes when _write_unified_enabled=True (set by production callers).
        Unit-test instances (default write_unified=False) are silently skipped.
        """
        if not self._write_unified_enabled:
            return
        try:
            from external_llm.editor.learning.learning_sink import record_execution
            success = getattr(record, "final_status", "") == "success"
            record_execution(
                language="python",
                strategy=getattr(record, "selected_strategy", "") or "",
                intent=getattr(record, "plan_mode", "") or "",
                success=success,
                reward=getattr(record, "shaped_reward", 0.0) or (1.0 if success else -0.5),
                repair_rounds=getattr(record, "repair_rounds_attempted", 0) or 0,
                affected_files=max(
                    (getattr(record, "completed", 0) or 0) +
                    (getattr(record, "failed", 0) or 0), 1,
                ),
                run_id=getattr(record, "run_id", "") or "",
                final_status=getattr(record, "final_status", "") or "",
                final_failure_class=getattr(record, "final_failure_class", None),
                completed_ops=getattr(record, "completed", 0) or 0,
                failed_ops=getattr(record, "failed", 0) or 0,
                error_types=list(getattr(record, "semantic_issue_codes", []) or []),
                metadata={
                    "constraint_mode": getattr(record, "constraint_mode", ""),
                    # Persist plan_id ONLY when stamped — empty string would still
                    # serialize but adds noise to every legacy run.  Downstream
                    # join CLI treats absence as "no shadow record to correlate".
                    **({"plan_id": record.plan_id} if getattr(record, "plan_id", "") else {}),
                },
                planner_model=self._model_name or "",
                developer_model=getattr(self, "_developer_model_name", "") or "",
                model_role="planner" if not getattr(self, "_developer_model_name", "") else "unified"
                    if self._model_name == getattr(self, "_developer_model_name", "") else "planner",
                test_pass_count=getattr(record, "test_pass_count", 0) or 0,
                test_fail_count=getattr(record, "test_fail_count", 0) or 0,
                total_tokens=getattr(record, "total_tokens", 0) or 0,
            )
        except Exception:
            pass  # non-critical

    def enrich_run_with_test_result(
        self, run_id: str, test_result: Any,
    ) -> None:
        """Attach test results to an existing RunRecord for reward shaping."""
        with self._telemetry_lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            record.test_pass_count = getattr(test_result, "passed_count", 0) or 0
            record.test_fail_count = getattr(test_result, "failed_count", 0) or 0

    def enrich_run_with_tokens(
        self, run_id: str, total_tokens: int,
    ) -> None:
        """Attach cumulative token usage to an existing RunRecord."""
        with self._telemetry_lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            record.total_tokens = (record.total_tokens or 0) + total_tokens

    def flush_unified_store(self, run_id: str) -> None:
        """Update strategy + abstract_strategy in unified_runs.db after policy trace enrichment.

        Called from operation_executor after enrich_run_with_policy_trace() sets
        selected_strategy.  Uses UPDATE (not INSERT) to avoid duplicate records.
        """
        if not self._write_unified_enabled:
            return  # skip disk writes in test contexts (consistent with _write_to_unified_store)
        record = self._runs.get(run_id)
        if record is None or not record.selected_strategy:
            return
        try:
            from external_llm.editor.learning.learning_sink import update_strategy
            update_strategy(run_id, "python", record.selected_strategy)
        except Exception:
            pass  # non-critical

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run record by ID."""
        return self._runs.get(run_id)

    def get_recent_runs(self, limit: int = 10) -> list[RunRecord]:
        """Alias for list_runs — used by candidate_ranker."""
        return self.list_runs(limit=limit)

    def _snapshot_run_order(self) -> list[str]:
        """Snapshot of ``_run_order`` taken under the telemetry lock.

        Readers (list_runs, get_recent_repair_memories, …) run on parallel
        sub-agent threads and may overlap with ``add_run`` evictions on another
        thread (the orchestrator shares one store instance across its
        ThreadPoolExecutor). Iterating ``_run_order`` without the lock while
        ``add_run`` pops/appends yields a torn snapshot, and indexing
        ``_runs[id]`` for an id whose record was just evicted raises
        ``KeyError``. The snapshot is taken under the same lock ``add_run``
        holds; callers iterate it OUTSIDE the lock and read
        ``self._runs.get(run_id)`` defensively — an evicted record is simply
        skipped (it is no longer live), which is the correct behavior.
        """
        with self._telemetry_lock:
            return list(self._run_order)

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """List most recent runs (newest first)."""
        order = self._snapshot_run_order()
        limited_ids = order[-limit:] if limit < len(order) else order
        # Defensive .get(): a concurrent add_run eviction can remove a record
        # whose id is still in this snapshot — skip it (no longer live).
        results: list[RunRecord] = []
        for run_id in reversed(limited_ids):
            record = self._runs.get(run_id)
            if record is not None:
                results.append(record)
        return results

    def build_repair_memory_entry(self, record: RunRecord) -> RepairMemoryEntry:
        """Convert a RunRecord to a compact RepairMemoryEntry."""
        return RepairMemoryEntry(
            run_id=record.run_id,
            final_failure_class=record.final_failure_class,
            final_status=record.final_status,
            blocking_reasons=record.final_blocking_reasons,
            warning_reasons=record.final_warning_reasons,
            repair_attempted=record.repair_attempted,
            repair_rounds_attempted=record.repair_rounds_attempted,
            repair_improved=record.repair_improved,
            semantic_issue_codes=record.semantic_issue_codes,
        )

    def get_recent_repair_memories(self, final_failure_class: Optional[str], limit: int = 5) -> list[RepairMemoryEntry]:
        """
        Retrieve recent repair memory entries filtered by failure class.

        If final_failure_class is provided, returns entries with matching failure class.
        If final_failure_class is None, returns empty list (no matching).

        Results are ordered newest first.
        """
        entries: list[RepairMemoryEntry] = []
        for run_id in reversed(self._snapshot_run_order()):
            if len(entries) >= limit:
                break
            record = self._runs.get(run_id)
            if record is None:
                continue  # evicted concurrently
            if final_failure_class is None:
                # According to spec, if failure_class is None, we should return empty list
                # (or could return recent failed/success_with_warnings, but spec says not to)
                continue
            if record.final_failure_class == final_failure_class:
                entries.append(self.build_repair_memory_entry(record))
        return entries

    def _top_k_strings(self, items: list[str], k: int = _DEFAULT_TOP_K) -> list[str]:
        """Return top k strings by frequency, deterministic tie-breaking."""
        if not items:
            return []
        from collections import Counter
        counter = Counter(items)
        # Sort by frequency desc, then string asc
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, _ in sorted_items[:k]]
    def summarize_failure_pattern(self, final_failure_class: Optional[str], limit: int = 10) -> FailurePatternSummary:
        """Summarize recent failure patterns for a given failure class."""
        if final_failure_class is None:
            return FailurePatternSummary(
                failure_class=None,
                total_runs=0,
                repaired_runs=0,
                improved_runs=0,
                success_after_repair_runs=0,
                common_blocking_reasons=[],
                common_semantic_issue_codes=[],
                common_skipped_reason_prefixes=[],
                last_seen_run_id=None,
            )
        memories = self.get_recent_repair_memories(final_failure_class, limit=limit)
        total_runs = len(memories)
        repaired_runs = sum(1 for m in memories if m.repair_attempted)
        improved_runs = sum(1 for m in memories if m.repair_improved)
        success_after_repair_runs = sum(1 for m in memories if m.repair_attempted and m.final_status == "success")
        # Collect all blocking reasons
        all_blocking_reasons = []
        for m in memories:
            all_blocking_reasons.extend(m.blocking_reasons)
        common_blocking_reasons = self._top_k_strings(all_blocking_reasons, k=3)
        # Collect all semantic issue codes
        all_semantic_issue_codes = []
        for m in memories:
            all_semantic_issue_codes.extend(m.semantic_issue_codes)
        common_semantic_issue_codes = self._top_k_strings(all_semantic_issue_codes, k=3)
        # Collect skipped reason prefixes
        all_skipped_reason_prefixes = []
        for m in memories:
            record = self.get_run(m.run_id)
            if record and record.skipped_reasons:
                for reason in record.skipped_reasons.values():
                    # Extract prefix before first colon
                    if ":" in reason:
                        prefix = reason.split(":", 1)[0]
                        all_skipped_reason_prefixes.append(prefix)
                    else:
                        all_skipped_reason_prefixes.append(reason)
        common_skipped_reason_prefixes = self._top_k_strings(all_skipped_reason_prefixes, k=3)
        last_seen_run_id = memories[0].run_id if memories else None
        return FailurePatternSummary(
            failure_class=final_failure_class,
            total_runs=total_runs,
            repaired_runs=repaired_runs,
            improved_runs=improved_runs,
            success_after_repair_runs=success_after_repair_runs,
            common_blocking_reasons=common_blocking_reasons,
            common_semantic_issue_codes=common_semantic_issue_codes,
            common_skipped_reason_prefixes=common_skipped_reason_prefixes,
            last_seen_run_id=last_seen_run_id,
        )

    def build_strategy_outcome_memory(self, limit: int = 100, recent_requests_limit: int = 5) -> StrategyOutcomeMemory:
        """Aggregate strategy outcomes from stored run records."""
        strategies: dict[str, StrategyOutcomeStats] = {}
        total_considered = 0

        for run_id in reversed(self._snapshot_run_order()):  # newest first
            if limit > 0 and total_considered >= limit:
                break
            record = self._runs.get(run_id)
            if record is None:
                continue  # evicted concurrently
            feedback = record.candidate_feedback
            if not isinstance(feedback, dict):
                continue
            selected_strategy = feedback.get("selected_strategy")
            if not selected_strategy:
                continue
            total_considered += 1

            # Get or create stats
            stats = strategies.get(selected_strategy)
            if stats is None:
                stats = StrategyOutcomeStats(strategy=selected_strategy)
                strategies[selected_strategy] = stats

            # Update counts
            stats.selected_count += 1
            execution_success = feedback.get("execution_success")
            if execution_success is True:
                stats.success_count += 1
            elif execution_success is False:
                stats.failure_count += 1
            final_status = feedback.get("final_status")
            if final_status == "verification_failed":
                stats.verification_failure_count += 1
            rolled_back = feedback.get("rolled_back")
            if rolled_back is True:
                stats.rollback_count += 1
            repair_attempted = feedback.get("repair_attempted")
            if repair_attempted is True:
                stats.repair_attempt_count += 1
            selected_score = feedback.get("selected_score")
            if selected_score is not None and isinstance(selected_score, (int, float)):
                # accumulate for later average
                stats.metadata.setdefault("_score_sum", 0.0)
                stats.metadata.setdefault("_score_count", 0)
                stats.metadata["_score_sum"] += selected_score
                stats.metadata["_score_count"] += 1

            request = feedback.get("request", "")
            if request:
                # Add to recent requests (truncate later)
                stats.recent_requests.append(request)
                # Use stored request_type from RunRecord (LLM-classified)
                req_type = record.request_type or feedback.get("request_type", "unknown")
                stats.request_type_counts[req_type] = stats.request_type_counts.get(req_type, 0) + 1

        # Post-process: compute averages and truncate recent_requests
        for stats in strategies.values():
            score_sum = stats.metadata.get("_score_sum")
            score_count = stats.metadata.get("_score_count")
            if score_count and score_count > 0:
                stats.avg_selected_score = score_sum / score_count
            # Remove internal fields
            stats.metadata.pop("_score_sum", None)
            stats.metadata.pop("_score_count", None)
            # Truncate recent_requests (keep newest first)
            if len(stats.recent_requests) > recent_requests_limit:
                stats.recent_requests = stats.recent_requests[:recent_requests_limit]

        memory = StrategyOutcomeMemory(
            strategies=strategies,
            total_runs_considered=total_considered,
            generated_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            metadata={"limit": limit, "recent_requests_limit": recent_requests_limit}
        )
        return memory

    def get_strategy_summary_for_request(self, request: str, limit: int = 100, request_type: str = "") -> dict[str, Any]:
        """Return a filtered summary of strategy outcomes relevant to the request type."""
        memory = self.build_strategy_outcome_memory(limit=limit)
        req_type = request_type or "unknown"
        filtered = {}
        for strategy, stats in memory.strategies.items():
            # Only include strategies that have been used for this request type at least once
            if req_type in stats.request_type_counts:
                filtered[strategy] = stats
        return {
            "request_type": req_type,
            "strategies": {k: v.to_dict() for k, v in filtered.items()},
            "total_runs_considered": memory.total_runs_considered,
        }

    def get_recent_selected_strategies(self, limit: int = 10) -> list[str]:
        """
        Retrieve recently selected strategies from newest runs.

        Args:
            limit: Maximum number of recent runs to examine.

        Returns:
            List of strategy names in chronological order (newest first).
            Only includes runs where candidate_feedback["selected_strategy"] is present.
        """
        strategies = []
        for run_id in reversed(self._snapshot_run_order()):  # newest first
            if len(strategies) >= limit:
                break
            record = self._runs.get(run_id)
            if record is None:
                continue  # evicted concurrently
            feedback = record.candidate_feedback
            if not isinstance(feedback, dict):
                continue
            selected_strategy = feedback.get("selected_strategy")
            if selected_strategy:
                strategies.append(selected_strategy)
        return strategies

    # ── Planner memory signals — contract repair outcome tracking ─────────────

    def record_planner_memory_signals(self, signals: list[dict]) -> None:
        """Record contract repair outcome signals from a completed execution.

        Only ``contract_repair_outcome`` signals are stored.  FIFO eviction
        keeps the buffer bounded to ``_max_contract_signals`` entries.
        """
        for sig in signals:
            if isinstance(sig, dict) and sig.get("signal_type") == "contract_repair_outcome":
                self._contract_repair_signals.append(sig)
        # FIFO eviction
        if len(self._contract_repair_signals) > self._max_contract_signals:
            self._contract_repair_signals = self._contract_repair_signals[-self._max_contract_signals:]

    def get_recent_planner_memory_signals(self, limit: int = 10) -> list[dict]:
        """Return most recent planner_memory_signals (newest first)."""
        return list(reversed(self._contract_repair_signals[-limit:]))

    def summarize_recent_contract_repairs(self, limit: int = 10) -> dict[str, Any]:
        """Summarize recent contract repair signals for planner bias injection.

        Returns an empty dict when there are no repair-needed signals.

        Strong-bias fields are computed from strict_source signals only.
        Fallback-source signals contribute to general statistics but do NOT
        drive the ``strong_bias_recommended`` flag — they are noisy evidence.
        """
        signals = self.get_recent_planner_memory_signals(limit=limit)
        repair_needed = [s for s in signals if s.get("repair_needed")]
        if not repair_needed:
            return {}
        from collections import Counter

        # Partition by source provenance
        strict_repairs = [s for s in repair_needed if s.get("strict_source")]
        fallback_repairs = [s for s in repair_needed if not s.get("strict_source")]

        # Top categories — computed separately per source
        def _top_cats(src: list[dict]) -> list[str]:
            cats: list[str] = []
            for s in src:
                cats.extend(s.get("dominant_violation_categories") or [])
            return [c for c, _ in Counter(cats).most_common(3)]

        strict_top_cats = _top_cats(strict_repairs)
        fallback_top_cats = _top_cats(fallback_repairs)
        # Combined top categories for general stats (strict first, then fallback)
        all_top_cats = _top_cats(repair_needed)

        strict_count = len(strict_repairs)
        success_count = sum(1 for s in repair_needed if s.get("success_after_repair"))

        # strong_bias_recommended: strict-source evidence must dominate.
        # Requires >= 2 strict repairs AND strict repairs form the majority.
        strong_bias = (
            strict_count >= 2
            and strict_count > len(repair_needed) // 2
        )

        return {
            # General stats (all sources)
            "repair_needed_count": len(repair_needed),
            "total_signals": len(signals),
            "top_violation_categories": all_top_cats,
            "strict_source_heavy": strict_count > len(repair_needed) // 2,
            "success_rate": round(success_count / len(repair_needed), 3),
            # Strict-source breakdown
            "strict_repair_needed_count": strict_count,
            "fallback_repair_needed_count": len(fallback_repairs),
            "strict_top_violation_categories": strict_top_cats,
            "fallback_top_violation_categories": fallback_top_cats,
            # Key flag for callers: drive strong planner bias only on strict evidence
            "strong_bias_recommended": strong_bias,
        }

    # ── Strategy-level outcome signals ────────────────────────────────────────

    def record_strategy_outcome_signals(self, signals: list[dict]) -> None:
        """Record strategy outcome signals from a completed execution.

        Only ``strategy_outcome`` signals are stored.  FIFO eviction keeps the
        buffer bounded to ``_max_strategy_signals`` entries.
        """
        for sig in signals:
            if isinstance(sig, dict) and sig.get("signal_type") == "strategy_outcome":
                self._strategy_outcome_signals.append(sig)
        if len(self._strategy_outcome_signals) > self._max_strategy_signals:
            self._strategy_outcome_signals = self._strategy_outcome_signals[-self._max_strategy_signals:]

    def get_recent_strategy_outcome_signals(self, limit: int = 20) -> list[dict]:
        """Return most recent strategy outcome signals (newest first)."""
        return list(reversed(self._strategy_outcome_signals[-limit:]))

    def summarize_strategy_outcomes(self, limit: int = 20) -> dict[str, Any]:
        """Summarize recent strategy outcomes for planner bias injection.

        Computes three separate views over the same signal pool:

        ``by_initial_strategy``  — grouped by the *original* generation strategy
                                   (e.g. generic_create / reference_bound_create)
        ``by_effective_strategy`` — grouped by what actually ran including +repair
                                    escalations (e.g. generic_create+repair)
        ``by_context``           — grouped by reference_bound context first, then
                                   initial_strategy.  This is the authoritative
                                   axis for ``prefer_reference_bound``.

        ``by_strategy`` is kept as a backward-compat alias for
        ``by_initial_strategy``.

        ``prefer_reference_bound`` is computed from ``reference_bound`` context
        only — non-reference-bound operations do not pollute this flag.
        True when:
          - reference_bound context / generic_create.repair_needed_rate > 0.5
          - AND at least 2 reference_bound generic_create samples exist
          - AND (no rbc data yet OR rbc.avg_repair_burden < gc.avg_repair_burden)

        Returns empty dict on cold start (no signals).
        """
        signals = self.get_recent_strategy_outcome_signals(limit=limit)
        if not signals:
            return {}

        from collections import Counter

        _BURDEN_NUM = {"none": 0, "low": 1, "medium": 2, "high": 3}

        def _empty() -> dict:
            return {"count": 0, "repair_needed_count": 0,
                    "burden_sum": 0, "success_count": 0, "all_cats": []}

        def _update(acc: dict[str, dict], key: str, sig: dict) -> None:
            if key not in acc:
                acc[key] = _empty()
            e = acc[key]
            e["count"] += 1
            if sig.get("repair_needed"):
                e["repair_needed_count"] += 1
            e["burden_sum"] += _BURDEN_NUM.get(sig.get("repair_burden", "none"), 0)
            if sig.get("success"):
                e["success_count"] += 1
            e["all_cats"].extend(sig.get("violation_categories") or [])

        def _compile(acc: dict[str, dict]) -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            for strat, e in acc.items():
                c = e["count"]
                top_cats = [cat for cat, _ in Counter(e["all_cats"]).most_common(3)]
                result[strat] = {
                    "count": c,
                    "repair_needed_rate": round(e["repair_needed_count"] / c, 3),
                    "avg_repair_burden": round(e["burden_sum"] / c, 2),
                    "final_success_rate": round(e["success_count"] / c, 3),
                    "top_failure_categories": top_cats,
                }
            return result

        acc_initial: dict[str, dict] = {}
        acc_effective: dict[str, dict] = {}
        # context → initial_strategy nested accumulator
        acc_ctx: dict[str, dict[str, dict]] = {}

        for sig in signals:
            initial = sig.get("initial_strategy") or "unknown"
            effective = sig.get("effective_strategy") or initial
            ctx_key = "reference_bound" if sig.get("reference_bound") else "non_reference_bound"

            _update(acc_initial, initial, sig)
            _update(acc_effective, effective, sig)
            if ctx_key not in acc_ctx:
                acc_ctx[ctx_key] = {}
            _update(acc_ctx[ctx_key], initial, sig)

        by_initial = _compile(acc_initial)
        by_effective = _compile(acc_effective)
        by_context = {ctx: _compile(inner) for ctx, inner in acc_ctx.items() if inner}

        # prefer_reference_bound: reference_bound context only — non-ref data excluded
        _rb = by_context.get("reference_bound", {})
        gc_rb = _rb.get("generic_create", {})
        rbc_rb = _rb.get("reference_bound_create", {})
        prefer_reference_bound = bool(
            gc_rb.get("repair_needed_rate", 0) > 0.5
            and gc_rb.get("count", 0) >= 2
            and (
                not rbc_rb
                or rbc_rb.get("avg_repair_burden", 99) < gc_rb.get("avg_repair_burden", 0)
            )
        )

        return {
            "by_initial_strategy": by_initial,
            "by_effective_strategy": by_effective,
            "by_context": by_context,
            "by_strategy": by_initial,        # backward-compat alias
            "total_signals": len(signals),
            "prefer_reference_bound": prefer_reference_bound,
        }

    def get_strategy_ranking_inputs(self, limit: int = 20) -> dict[str, Any]:
        """Return ranking-ready inputs for StrategySelector.

        Extracts the ``reference_bound_context`` slice and ``effective_strategy``
        data from :meth:`summarize_strategy_outcomes` into a format that
        :func:`strategy_selector.select_strategy` consumes directly.

        Returns ``{}`` on cold start (no signals in store).
        """
        try:
            summary = self.summarize_strategy_outcomes(limit=limit)
        except Exception:
            return {}  # non-critical — never block execution
        if not summary:
            return {}

        by_ctx = summary.get("by_context", {})
        by_eff = summary.get("by_effective_strategy", {})

        return {
            # Primary axis for reference-bound create_file ranking
            "reference_bound_context": by_ctx.get("reference_bound", {}),
            # Effective-strategy escalation patterns (e.g. generic_create+repair)
            "effective_strategies":    by_eff,
            # Pre-computed flag from summarize_strategy_outcomes
            "prefer_reference_bound":  summary.get("prefer_reference_bound", False),
            "total_signals":           summary.get("total_signals", 0),
        }

    def get_strategy_simulation_inputs(self, limit: int = 20) -> dict[str, Any]:
        """Return enriched simulation inputs for the strategy simulator.

        Extends :meth:`get_strategy_ranking_inputs` with escalation patterns
        and a confidence indicator — these are consumed by
        :func:`strategy_selector.select_strategy` and
        :func:`multi_strategy_gate.decide_gate`.

        Returns ``{}`` on cold start (no signals in store).

        Extra fields vs ``get_strategy_ranking_inputs``
        -----------------------------------------------
        ``gc_escalation_rate``
            How often generic_create needed repair escalation
            (gc_repair_count / gc_count from reference_bound context).
        ``confidence``
            "high" when total_signals >= 5, otherwise "low".
        """
        base = self.get_strategy_ranking_inputs(limit=limit)
        if not base:
            return {}

        eff = base.get("effective_strategies", {})
        gc_repair_eff = eff.get("generic_create+repair", {})
        rb_ctx = base.get("reference_bound_context", {})
        gc = rb_ctx.get("generic_create", {})

        escalation_rate = 0.0
        gc_count = gc.get("count", 0)
        gc_repair_count = gc_repair_eff.get("count", 0)
        if gc_count > 0 and gc_repair_count > 0:
            escalation_rate = round(gc_repair_count / gc_count, 3)

        total_signals = base.get("total_signals", 0)
        base["gc_escalation_rate"] = escalation_rate
        base["confidence"] = "high" if total_signals >= 5 else "low"
        return base

    def record_strategy_execution(
        self,
        request_type: str,
        estimated_scope: str,
        used_decomposition: bool,
        strategy_name: str,
        success: bool,
        repaired: bool = False,
        switched: bool = False,
        strategy_chain: Optional[list[str]] = None,
    ) -> None:
        """Record a strategy execution outcome for future preference scoring."""
        key = (request_type, estimated_scope, used_decomposition, strategy_name)
        if key not in self._strategy_exec_stats:
            self._strategy_exec_stats[key] = StrategyExecutionStats(
                request_type=request_type,
                estimated_scope=estimated_scope,
                used_decomposition=used_decomposition,
                strategy_name=strategy_name,
            )
        stats = self._strategy_exec_stats[key]
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
        if repaired:
            stats.repair_count += 1
        if switched:
            stats.switch_count += 1

        # Track chain history if multi-hop chain provided
        if strategy_chain and len(strategy_chain) >= 2:
            chain_key_str = "->".join(strategy_chain)
            chain_key = (request_type, estimated_scope, used_decomposition, chain_key_str)
            if chain_key not in self._chain_exec_stats:
                self._chain_exec_stats[chain_key] = StrategyExecutionStats(
                    request_type=request_type,
                    estimated_scope=estimated_scope,
                    used_decomposition=used_decomposition,
                    strategy_name=strategy_name,
                    strategy_chain=strategy_chain,
                )
            chain_stats = self._chain_exec_stats[chain_key]
            if success:
                chain_stats.success_count += 1
            else:
                chain_stats.failure_count += 1

    def get_strategy_preference(
        self,
        request_type: str,
        estimated_scope: str,
        used_decomposition: bool,
        strategy_name: str,
    ) -> dict[str, Any]:
        """Return preference score for a strategy given context."""
        key = (request_type, estimated_scope, used_decomposition, strategy_name)
        stats = self._strategy_exec_stats.get(key)
        if stats is None:
            return {
                "preference_score": 0.0,
                "total": 0,
                "success_rate": 0.0,
                "explanations": ["no_history"],
            }
        total = stats.success_count + stats.failure_count
        success_rate = stats.success_count / total if total > 0 else 0.0
        # Simple preference: success_rate * confidence_weight
        # confidence grows with more data (max at 10 samples)
        confidence = min(total / 10.0, 1.0)
        preference_score = max(success_rate * confidence, 0.0)
        explanations = [f"success_rate={success_rate:.3f}", f"total={total}", f"confidence={confidence:.3f}"]
        return {
            "preference_score": round(preference_score, 6),
            "total": total,
            "success_rate": success_rate,
            "explanations": explanations,
        }

    def get_strategy_chain_preference(
        self,
        request_type: str,
        estimated_scope: str,
        used_decomposition: bool,
        strategy_chain: list[str],
    ) -> dict[str, Any]:
        """Return preference score for a multi-hop strategy chain."""
        if not strategy_chain or len(strategy_chain) < 2:
            return {
                "preference_score": 0.0,
                "total": 0,
                "success_rate": 0.0,
                "explanations": ["insufficient_chain_length"],
            }
        chain_key_str = "->".join(strategy_chain)
        chain_key = (request_type, estimated_scope, used_decomposition, chain_key_str)
        stats = self._chain_exec_stats.get(chain_key)
        if stats is None:
            return {
                "preference_score": 0.0,
                "total": 0,
                "success_rate": 0.0,
                "explanations": ["no_chain_history"],
            }
        total = stats.success_count + stats.failure_count
        success_rate = stats.success_count / total if total > 0 else 0.0
        confidence = min(total / 10.0, 1.0)
        preference_score = max(success_rate * confidence, 0.0)
        return {
            "preference_score": round(preference_score, 6),
            "total": total,
            "success_rate": success_rate,
            "explanations": [f"chain={chain_key_str}", f"success_rate={success_rate:.3f}"],
        }

    def get_strategy_usage_count(
        self,
        request_type: str,
        estimated_scope: str,
        used_decomposition: bool,
        strategy_name: str,
    ) -> int:
        """Return total run count (success + failure) for a strategy in given context."""
        key = (request_type, estimated_scope, used_decomposition, strategy_name)
        stats = self._strategy_exec_stats.get(key)
        if stats is None:
            return 0
        return stats.success_count + stats.failure_count

    def get_decomposition_aware_strategy_summary(
        self,
        request_type: str,
        estimated_scope: str,
        used_decomposition: bool,
        candidate_strategies: list[str],
    ) -> dict[str, Any]:
        """Return ordered strategies by preference score for the given context."""
        scores = {}
        for strategy in candidate_strategies:
            pref = self.get_strategy_preference(request_type, estimated_scope, used_decomposition, strategy)
            scores[strategy] = pref["preference_score"]
        ordered = sorted(candidate_strategies, key=lambda s: scores[s], reverse=True)
        return {
            "ordered_strategies": ordered,
            "preference_scores": scores,
            "request_type": request_type,
            "estimated_scope": estimated_scope,
            "used_decomposition": used_decomposition,
        }

    # ── P8 minimal: strategy reward tracking ──────────────────────────────────

    def record_strategy_reward(
        self, strategy: str, reward: float, metadata: Optional[dict] = None,
    ) -> None:
        """Record an alignment-based reward for a strategy execution.

        ``_telemetry_lock`` serializes the read-check-append-trim RMW against
        concurrent completions in parallel sub-agents.
        """
        import time as _time
        entry = {
            "reward": reward,
            "timestamp": _time.time(),
            **(metadata or {}),
        }
        with self._telemetry_lock:
            if strategy not in self._strategy_rewards:
                self._strategy_rewards[strategy] = []
            hist = self._strategy_rewards[strategy]
            hist.append(entry)
            # FIFO eviction
            if len(hist) > self._max_reward_history:
                self._strategy_rewards[strategy] = hist[-self._max_reward_history:]

    def get_strategy_reward_stats(self, strategy: str, window: int = 10) -> dict:
        """Return reward stats for a strategy (P8.2: adaptive weight + UCB input)."""
        hist = self._strategy_rewards.get(strategy, [])
        if not hist:
            return {"count": 0, "recent_avg": 0.0, "recent_success_rate": 0.0}
        recent = hist[-window:]
        rewards = [e["reward"] for e in recent]
        success_count = sum(1 for e in recent if e.get("termination") == "SUCCESS")
        return {
            "count": len(hist),
            "recent_avg": sum(rewards) / len(rewards),
            "recent_success_rate": success_count / len(recent),
        }

    def get_total_strategy_reward_trials(self) -> int:
        """Total reward samples across all strategies."""
        return sum(len(v) for v in self._strategy_rewards.values())

    # ── P9: EMA-based reward smoothing ─────────────────────────────────────────

    def update_strategy_ema(self, strategy: str, reward: float, alpha: float = 0.3) -> None:
        """Update exponential moving average for a strategy."""
        with self._telemetry_lock:
            prev = self._strategy_ema.get(strategy)
            if prev is None:
                self._strategy_ema[strategy] = reward
            else:
                self._strategy_ema[strategy] = round(alpha * reward + (1 - alpha) * prev, 4)

    def get_strategy_ema(self, strategy: str) -> float:
        """Return current EMA for a strategy (0.0 if no data)."""
        return self._strategy_ema.get(strategy, 0.0)

    def update_strategy_context_ema(
        self, strategy: str, context: str, reward: float, alpha: float = 0.3,
    ) -> None:
        """Update context-specific EMA (strategy × context_key)."""
        with self._telemetry_lock:
            ctx = self._strategy_context_ema.setdefault(strategy, {})
            prev = ctx.get(context)
            if prev is None:
                ctx[context] = reward
            else:
                ctx[context] = round(alpha * reward + (1 - alpha) * prev, 4)

    def get_strategy_context_ema(self, strategy: str, context: str) -> Optional[float]:
        """Return context-specific EMA (None if no data for this context)."""
        return self._strategy_context_ema.get(strategy, {}).get(context)

    def update_repair_ema(
        self, failure_class: str, repair_action: str, success: float, alpha: float = 0.3,
    ) -> None:
        """Update EMA for repair action success rate."""
        with self._telemetry_lock:
            fc_map = self._repair_ema.setdefault(failure_class, {})
            prev = fc_map.get(repair_action)
            if prev is None:
                fc_map[repair_action] = success
            else:
                fc_map[repair_action] = round(alpha * success + (1 - alpha) * prev, 4)

    def get_repair_ema(self, failure_class: str, repair_action: str) -> Optional[float]:
        """Return EMA success rate for a repair action (None if no data)."""
        return self._repair_ema.get(failure_class, {}).get(repair_action)

    # ── P11: Policy distillation ───────────────────────────────────────────────

    def distill_rules(self) -> int:
        """P11.1: Extract high-confidence EMA patterns into fast-path rules.

        Uses adaptive threshold — stricter with few samples, relaxed with many.
        """
        from .context_utils import adaptive_distill_threshold
        self._distilled_rules = []
        for strategy, ctx_map in self._strategy_context_ema.items():
            # Count total samples for this strategy across all contexts
            _total_samples = len(self._strategy_rewards.get(strategy, []))
            _threshold = adaptive_distill_threshold(_total_samples)
            for ctx, val in ctx_map.items():
                if val >= _threshold:
                    self._distilled_rules.append({
                        "context": ctx,
                        "preferred_strategy": strategy,
                        "confidence": round(val, 4),
                    })
        if self._distilled_rules:
            logger.info(
                "P11.1 distilled %d rules: %s",
                len(self._distilled_rules),
                [(r["context"], r["preferred_strategy"], r["confidence"])
                 for r in self._distilled_rules[:5]],
            )
        return len(self._distilled_rules)

    def get_distilled_rules(self) -> list[dict]:
        """Return all distilled rules for debugging."""
        return list(self._distilled_rules)

    def compress_memory(self, max_per_class: int | None = None) -> int:
        """P11: Trim reflection patterns to prevent unbounded growth.

        Keeps only the most recent max_per_class patterns per failure_class.
        Returns total patterns removed.

        Default limit comes from ``self._max_reflection_per_class`` (single source
        of truth). Callers may override explicitly.
        """
        if max_per_class is None:
            max_per_class = self._max_reflection_per_class
        removed = 0
        for fc in list(self._reflection_patterns.keys()):
            plist = self._reflection_patterns[fc]
            if len(plist) > max_per_class:
                removed += len(plist) - max_per_class
                self._reflection_patterns[fc] = plist[-max_per_class:]
        if removed:
            logger.info("P11 memory compression: removed %d stale patterns", removed)
        return removed

    # ── P12: Dynamic strategy registry ─────────────────────────────────────────

    def register_dynamic_strategy(self, strategy: dict) -> None:
        """P12.2: Register with versioning — base_name@vN."""
        base_name = strategy.get("name", "")
        if not base_name:
            return
        ver = self._strategy_versions.get(base_name, 0) + 1
        self._strategy_versions[base_name] = ver
        strategy["version"] = ver
        full_name = f"{base_name}@v{ver}"
        strategy["full_name"] = full_name
        strategy["base_name"] = base_name
        strategy["name"] = full_name  # unify: name == full_name for EMA/UCB key match
        self._dynamic_strategies[full_name] = strategy
        logger.info(
            "P12.2 register: %s (v%d, src=%s, offset=%.3f)",
            base_name, ver, strategy.get("source", "?"),
            strategy.get("base_score_offset", 0.0),
        )

    def get_dynamic_strategies(self) -> list[dict]:
        """Return all active dynamic strategies."""
        return list(self._dynamic_strategies.values())

    def prune_dynamic_strategies(self, min_trials: int = 20, max_avg_reward: float = 0.2) -> list[str]:
        """P12: Remove underperforming dynamic strategies. Returns pruned names."""
        try:
            from external_llm.editor._editor_core.lane.meta_strategy_engine import MetaStrategyEngine
        except ImportError:
            # lane is excluded from the public build — degrade gracefully.
            logger.debug("prune_dynamic_strategies: lane not available, skipping")
            return []
        prunable = MetaStrategyEngine.identify_prunable(
            self._strategy_rewards, min_trials=min_trials, max_avg_reward=max_avg_reward,
        )
        pruned = []
        for name in prunable:
            if name in self._dynamic_strategies:
                del self._dynamic_strategies[name]
                pruned.append(name)
                logger.info("P12 pruned dynamic strategy: %s", name)
        return pruned

    # ── P12.2: Cooldown ──────────────────────────────────────────────────────

    def mark_run(self) -> None:
        """Increment runs-since-last-evolve counter."""
        self._evolution_state["runs_since_last_evolve"] += 1

    def should_evolve(self) -> bool:
        """Check if evolution cooldown has elapsed."""
        st = self._evolution_state
        if st["runs_since_last_evolve"] < st["cooldown_runs"]:
            return False
        total_patterns = sum(len(v) for v in self._reflection_patterns.values())
        return total_patterns >= 3

    def _mark_evolved(self) -> None:
        self._evolution_state["runs_since_last_evolve"] = 0

    # ── P12.2: Attribution ────────────────────────────────────────────────────

    def update_attribution(self, strategy: str, context: str, reward: float, alpha: float = 0.3) -> None:
        """Track which context a strategy performs best in."""
        with self._telemetry_lock:
            ctx_map = self._strategy_attribution.setdefault(strategy, {})
            prev = ctx_map.get(context)
            if prev is None:
                ctx_map[context] = reward
            else:
                ctx_map[context] = round(alpha * reward + (1 - alpha) * prev, 4)

    def get_strategy_best_context(self, strategy: str) -> Optional[str]:
        """Return the context where this strategy has the highest EMA reward."""
        ctx_map = self._strategy_attribution.get(strategy, {})
        if not ctx_map:
            return None
        return max(ctx_map.items(), key=lambda x: x[1])[0]

    # ── P13.1: Role-level learning ───────────────────────────────────────────

    def update_role_ema(self, role: str, reward: float, alpha: float = 0.3) -> None:
        """Update role-level EMA (averages over strategies)."""
        with self._telemetry_lock:
            prev = self._role_ema.get(role)
            if prev is None:
                self._role_ema[role] = reward
            else:
                self._role_ema[role] = round(alpha * reward + (1 - alpha) * prev, 4)

    def get_role_ema(self, role: str) -> float:
        """Return current EMA for a role (0.0 if no data)."""
        return self._role_ema.get(role, 0.0)

    def update_role_context_ema(
        self, role: str, context: str, reward: float, alpha: float = 0.3,
    ) -> None:
        """Update role × context EMA."""
        with self._telemetry_lock:
            ctx = self._role_context_ema.setdefault(role, {})
            prev = ctx.get(context)
            if prev is None:
                ctx[context] = reward
            else:
                ctx[context] = round(alpha * reward + (1 - alpha) * prev, 4)

    def get_role_context_ema(self, role: str, context: str) -> Optional[float]:
        """Return context-specific EMA for a role (None if no data)."""
        return self._role_context_ema.get(role, {}).get(context)

    # ── P14: Difficulty tracking ──────────────────────────────────────────────

    def update_difficulty(self, alignment: float, replan_count: int = 0, repair_rounds: int = 0) -> float:
        """P14.1: Stabilized difficulty update with EMA + streak logic.

        Guards the multi-field read-modify-write on ``_difficulty_state`` with
        ``_telemetry_lock`` because parallel sub-agents call this concurrently from
        their run-completion hook. Without the lock, concurrent completions read the
        same ``prev_ema``/``old``/``streak`` and the later writer wins — losing updates
        and corrupting streak/history accounting.
        """
        with self._telemetry_lock:
            signal = alignment - 0.2 * replan_count - 0.1 * repair_rounds
            st = self._difficulty_state

            # EMA signal smoothing (alpha=0.3)
            prev_ema = st["ema_signal"]
            st["ema_signal"] = round(0.3 * signal + 0.7 * prev_ema, 4)
            ema = st["ema_signal"]

            # Streak tracking
            if signal > 0.6:
                st["streak_success"] = st["streak_success"] + 1
                st["streak_fail"] = 0
            elif signal < 0.4:
                st["streak_fail"] = st["streak_fail"] + 1
                st["streak_success"] = 0
            else:
                st["streak_success"] = max(0, st["streak_success"] - 1)
                st["streak_fail"] = max(0, st["streak_fail"] - 1)

            # Stabilized step
            old = st["current"]
            if ema > 0.75 and st["streak_success"] >= 2:
                step = 0.04
            elif ema > 0.65:
                step = 0.02
            elif ema < 0.35 and st["streak_fail"] >= 2:
                step = -0.04
            elif ema < 0.45:
                step = -0.02
            else:
                step = 0.0

            new = round(max(0.15, min(0.95, old + step)), 2)
            st["current"] = new

            hist = st["history"]
            hist.append({"difficulty": new, "signal": round(signal, 3), "ema": round(ema, 3)})
            if len(hist) > st["max_history"]:
                st["history"] = hist[-st["max_history"]:]

            logger.debug(
                "P14.1 difficulty: old=%.2f signal=%.3f ema=%.3f new=%.2f streak=+%d/-%d",
                old, signal, ema, new, st["streak_success"], st["streak_fail"],
            )
            return new

    def get_current_difficulty(self) -> float:
        return self._difficulty_state["current"]

    # NOTE: P15 long-horizon memory subsystem removed — writer(update_long_horizon) and
    # both readers (get_long_horizon_summary, get_frequent_failure_types) had zero callers;
    # _long_horizon state was referenced only by those dead methods. Removed together as a
    # pure dead cluster (prune_versions below is LIVE — called by evolve_strategies).

    # ── P12.2: Version pruning ────────────────────────────────────────────────

    def prune_versions(self, max_per_base: int = 2) -> int:
        """Keep only top N versions per base strategy name (by EMA reward)."""
        groups: dict[str, list[str]] = {}
        for full_name, s in self._dynamic_strategies.items():
            base = s.get("name", full_name)
            groups.setdefault(base, []).append(full_name)

        removed = 0
        for base, full_names in groups.items():
            if len(full_names) <= max_per_base:
                continue
            scored = sorted(
                full_names,
                key=lambda fn: self._strategy_ema.get(fn, 0.0),
                reverse=True,
            )
            for fn in scored[max_per_base:]:
                self._dynamic_strategies.pop(fn, None)
                removed += 1
                logger.info("P12.2 version pruned: %s (keeping top %d of %s)", fn, max_per_base, base)
        return removed

    # ── Evolution cycle ───────────────────────────────────────────────────────

    def evolve_strategies(self, llm_client=None, llm_model: str = "") -> None:
        """P12.2: Full evolution cycle with cooldown + validation + version pruning."""
        if not self.should_evolve():
            logger.debug("P12.2 evolve_skip: runs_since=%d < cooldown=%d",
                         self._evolution_state["runs_since_last_evolve"],
                         self._evolution_state["cooldown_runs"])
            return

        all_patterns = []
        for plist in self._reflection_patterns.values():
            all_patterns.extend(plist)
        if not all_patterns:
            return

        try:
            from external_llm.editor._editor_core.lane.meta_strategy_engine import MetaStrategyEngine
        except ImportError:
            # lane is excluded from the public build — degrade gracefully.
            logger.debug("evolve_strategies: lane not available, skipping")
            return
        engine = MetaStrategyEngine(llm_client=llm_client, llm_model=llm_model)

        existing = list(self._dynamic_strategies.values())
        _existing_names = set(self._dynamic_strategies.keys())
        proposals = engine.propose_strategies(
            all_patterns, existing_strategies=existing,
            existing_names=_existing_names,
        )

        _MAX_DYNAMIC = _cfg.counts.RUN_STORE_MAX_DYNAMIC
        registered = 0
        for p in proposals:
            if len(self._dynamic_strategies) >= _MAX_DYNAMIC:
                break
            self.register_dynamic_strategy(p)
            registered += 1

        # Prune underperformers + old versions
        self.prune_dynamic_strategies()
        self.prune_versions()
        self._mark_evolved()

        if registered:
            logger.info(
                "P12.2 evolution: +%d strategies, %d active, %d patterns",
                registered, len(self._dynamic_strategies), len(all_patterns),
            )

    # ── P8.3: repair outcome learning ──────────────────────────────────────────

    def record_repair_outcome(
        self, failure_class: str, repair_action: str, success: bool,
    ) -> None:
        """Record whether a repair action succeeded for a given failure class."""
        if not failure_class or not repair_action:
            return
        with self._telemetry_lock:
            if failure_class not in self._repair_outcomes:
                self._repair_outcomes[failure_class] = {}
            if repair_action not in self._repair_outcomes[failure_class]:
                self._repair_outcomes[failure_class][repair_action] = []
            hist = self._repair_outcomes[failure_class][repair_action]
            hist.append(1.0 if success else 0.0)
            if len(hist) > self._max_repair_history:
                self._repair_outcomes[failure_class][repair_action] = hist[-self._max_repair_history:]

    def get_repair_ranking(self, failure_class: str, min_samples: int = 3) -> list[tuple[str, float]]:
        """Return repair actions ranked by success rate for a failure class.

        Returns list of (repair_action, avg_success_rate) sorted descending.
        Only includes actions with >= min_samples.
        """
        methods = self._repair_outcomes.get(failure_class, {})
        scored: list[tuple[str, float]] = []
        for action, hist in methods.items():
            if len(hist) < min_samples:
                continue
            recent = hist[-10:]
            avg = sum(recent) / len(recent)
            scored.append((action, avg))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ── Switch outcome memory ─────────────────────────────────────────────────

    def build_switch_outcome_memory(self, limit: int = 100) -> SwitchOutcomeMemory:
        """Aggregate strategy switch transitions from stored run records."""
        transitions: dict[str, SwitchOutcomeStats] = {}
        total_considered = 0

        for run_id in reversed(self._snapshot_run_order()):  # newest first
            if limit > 0 and total_considered >= limit:
                break
            record = self._runs.get(run_id)
            if record is None:
                continue  # evicted concurrently
            feedback = record.candidate_feedback
            if not isinstance(feedback, dict):
                continue
            from_strategy = feedback.get("switched_from_strategy") or feedback.get("from_strategy")
            selected_strategy = feedback.get("switched_to_strategy") or feedback.get("selected_strategy")
            if not from_strategy or not selected_strategy or from_strategy == selected_strategy:
                continue
            total_considered += 1
            trans_key = f"{from_strategy}->{selected_strategy}"
            if trans_key not in transitions:
                transitions[trans_key] = SwitchOutcomeStats(
                    from_strategy=from_strategy,
                    to_strategy=selected_strategy,
                )
            stats = transitions[trans_key]
            stats.attempted_count += 1
            execution_success = feedback.get("execution_success")
            if execution_success is True:
                stats.success_count += 1
            elif execution_success is False:
                stats.failure_count += 1
            if feedback.get("rolled_back") is True:
                stats.rollback_count += 1
            if feedback.get("repair_attempted") is True:
                stats.repair_attempt_count += 1
            selected_score = feedback.get("selected_score")
            if selected_score is not None and isinstance(selected_score, (int, float)):
                stats.metadata.setdefault("_score_sum", 0.0)
                stats.metadata.setdefault("_score_count", 0)
                stats.metadata["_score_sum"] += selected_score
                stats.metadata["_score_count"] += 1
            request = feedback.get("request", "")
            if request:
                req_type = record.request_type or feedback.get("request_type", "unknown")
                stats.request_type_counts[req_type] = stats.request_type_counts.get(req_type, 0) + 1
                if execution_success is True:
                    stats.request_type_success_counts[req_type] = stats.request_type_success_counts.get(req_type, 0) + 1

        # Post-process: compute avg scores
        for stats in transitions.values():
            score_count = stats.metadata.get("_score_count")
            if score_count and score_count > 0:
                stats.avg_selected_score = stats.metadata["_score_sum"] / score_count
            stats.metadata.pop("_score_sum", None)
            stats.metadata.pop("_score_count", None)

        return SwitchOutcomeMemory(
            transitions=transitions,
            total_switch_runs_considered=total_considered,
            generated_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
        )

    def get_transition_preference(
        self,
        from_strategy: str,
        to_strategy: str,
        request: str = "",
    ) -> dict[str, Any]:
        """Return preference score for a strategy transition."""
        memory = self.build_switch_outcome_memory()
        trans_key = f"{from_strategy}->{to_strategy}"
        stats = memory.transitions.get(trans_key)
        exploration_bonus = 0.12  # fixed exploration bonus for unknown transitions
        if stats is None:
            return {
                "preference_score": exploration_bonus,
                "total": 0,
                "explanations": ["no_transition_history", f"exploration_bonus={exploration_bonus:.3f}"],
            }
        total = stats.attempted_count
        success_rate = stats.success_count / total if total > 0 else 0.0
        confidence = min(total / 10.0, 1.0)
        preference_score = max(success_rate * confidence, 0.0)
        return {
            "preference_score": round(preference_score, 6),
            "total": total,
            "success_rate": success_rate,
            "explanations": [f"success_rate={success_rate:.3f}", f"total={total}"],
        }

    def get_switch_summary_for_request(
        self,
        from_strategy: str,
        request: str = "",
        request_type: str = "",
    ) -> dict[str, Any]:
        """Return ordered switch targets for a given from_strategy and request."""
        memory = self.build_switch_outcome_memory()
        req_type = request_type or "unknown"
        transition_scores: dict[str, float] = {}
        ordered_targets = []

        for trans_key, stats in memory.transitions.items():
            if not trans_key.startswith(f"{from_strategy}->"):
                continue
            to_strategy = stats.to_strategy
            total = stats.attempted_count
            success_rate = stats.success_count / total if total > 0 else 0.0
            confidence = min(total / 10.0, 1.0)
            score = max(success_rate * confidence, 0.0)
            transition_scores[to_strategy] = round(score, 6)
            ordered_targets.append(to_strategy)

        ordered_targets.sort(key=lambda s: transition_scores.get(s, 0.0), reverse=True)
        return {
            "ordered_targets": ordered_targets,
            "transition_scores": transition_scores,
            "request_type": req_type,
            "from_strategy": from_strategy,
        }

    # ── Weight learning ────────────────────────────────────────────────────────

    def _get_weight_learner(self) -> Any:
        """Return (lazy-initialised) WeightLearner instance, or None on import error."""
        if self._weight_learner is None:
            try:
                from .weight_learning import WeightLearner
                learner = WeightLearner()
                self._load_weight_learner_state(learner)
                self._weight_learner = learner
            except ImportError:
                logger.debug("run_store: weight_learning module not available")
        return self._weight_learner

    def record_weight_learning_signal(self, signal_dict: dict) -> None:
        """Record a weight learning signal and update the bucket's learned weights.

        ``signal_dict`` must conform to the ``LearningSignal.to_dict()`` schema.
        Silently ignores unknown/missing fields.
        """
        learner = self._get_weight_learner()
        if learner is None:
            return
        try:
            from .weight_learning import LearningSignal
            _REQUIRED = (
                "bucket", "selected_weight_profile", "selected_strategy",
                "success", "repair_attempts", "repair_burden",
                "contract_violation", "semantic_failures",
                "budget_failure", "graph_impact_level",
            )
            kwargs = {k: signal_dict[k] for k in _REQUIRED if k in signal_dict}
            # Fill missing required fields with safe defaults
            kwargs.setdefault("bucket", "default")
            kwargs.setdefault("selected_weight_profile", "DEFAULT")
            kwargs.setdefault("selected_strategy", "generic_create")
            kwargs.setdefault("success", False)
            kwargs.setdefault("repair_attempts", 0)
            kwargs.setdefault("repair_burden", "none")
            kwargs.setdefault("contract_violation", False)
            kwargs.setdefault("semantic_failures", [])
            kwargs.setdefault("budget_failure", False)
            kwargs.setdefault("graph_impact_level", "low")
            sig = LearningSignal(**kwargs)
            learner.update(sig)
            self._save_weight_learner_state()
        except Exception:
            logger.debug("record_weight_learning_signal: error", exc_info=True)

    def get_total_weight_learning_signals(self) -> int:
        """Return the total number of recorded weight learning signals across all buckets.

        Used by ``strategy_selector`` to determine cold-start state:
        - Returns 0  → no learning experience yet (cold start)
        - Returns >0 → at least one signal recorded; learning is active

        Reads directly from the in-memory ``WeightLearner`` state (which is
        restored from ``runs/weight_state.json`` on first access after a restart).
        """
        learner = self._get_weight_learner()
        if learner is None:
            return 0
        try:
            summary = learner.get_summary()
            return sum(
                v.get("signal_count", 0)
                for v in summary.values()
                if isinstance(v, dict)
            )
        except Exception:
            logger.debug("get_total_weight_learning_signals: error", exc_info=True)
            return 0

    # ── Weight learner persistence ─────────────────────────────────────────────

    def _load_weight_learner_state(self, learner: Any) -> None:
        """Restore bucket states into *learner* from strategy_state.json.

        Reads from the ``weights/{model}`` namespace.  Silently no-ops when
        the namespace is absent or contains malformed data so that a fresh
        process always starts correctly even after corruption.
        """
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            ns = f"weights/{self._model_name}" if self._model_name else "weights"
            state = read_namespace(ns)
            if isinstance(state, dict):
                learner.load_state(state)
                logger.debug("run_store: restored weight learner state from %s", ns)
        except Exception as exc:
            logger.debug("run_store: could not restore weight state (%s) — starting fresh", exc)

    def _save_weight_learner_state(self) -> None:
        """Persist the current learner state to strategy_state.json.

        Uses the namespace ``weights/{model}`` to keep model-specific weight
        states separated within the consolidated file.
        """
        learner = self._weight_learner
        if learner is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import write_namespace
            state = learner.get_summary()
            ns = f"weights/{self._model_name}" if self._model_name else "weights"
            write_namespace(ns, state)
            logger.debug("run_store: weight learner state saved to %s", ns)
        except Exception as exc:
            logger.debug("run_store: _save_weight_learner_state failed: %s", exc)

    # ── Execution learning (lane-common) ────────────────────────────────────────

    def _get_execution_learner(self) -> Any:
        """Return (lazy-initialised) ExecutionLearner instance, or None on import error."""
        if self._execution_learner is None:
            try:
                from .weight_learning import ExecutionLearner
                learner = ExecutionLearner()
                self._load_execution_learner_state(learner)
                self._execution_learner = learner
            except ImportError:
                logger.debug("run_store: ExecutionLearner not available")
        return self._execution_learner

    def record_execution_learning_signal(self, signal_dict: dict) -> None:
        """Record a lane-common execution learning signal.

        ``signal_dict`` must conform to ``ExecutionLearningSignal.to_dict()`` schema.
        Always records (no cold-start gate) — execution signals are valuable from the start.
        The in-memory learner is always updated (this feeds ``get_execution_bias``,
        which ``strategy_selector`` consumes at runtime); the aggregated state is
        persisted to strategy_state.json only when _write_unified_enabled=True.
        """
        learner = self._get_execution_learner()
        if learner is None:
            return
        try:
            from .weight_learning import ExecutionLearningSignal
            sig = ExecutionLearningSignal.from_dict(signal_dict)
            learner.update(sig)
            if not self._write_unified_enabled:
                return  # skip disk writes in test contexts
            self._save_execution_learner_state()
        except Exception:
            logger.debug("record_execution_learning_signal: error", exc_info=True)

    def get_execution_lane_state(self, lane: str) -> dict[str, Any]:
        """Return the current execution learning state for a lane.

        Returns empty dict when execution learning is unavailable.
        """
        learner = self._get_execution_learner()
        if learner is None:
            return {}
        try:
            return learner.get_lane_state(lane).to_dict()
        except Exception:
            logger.debug("get_execution_lane_state: error", exc_info=True)
            return {}

    def get_execution_learning_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all lane execution states."""
        learner = self._get_execution_learner()
        if learner is None:
            return {}
        try:
            return learner.get_summary()
        except Exception:
            logger.debug("get_execution_learning_summary: error", exc_info=True)
            return {}

    def get_total_execution_signals(self) -> int:
        """Return the total number of recorded execution signals across all lanes."""
        learner = self._get_execution_learner()
        if learner is None:
            return 0
        try:
            return learner.get_total_signals()
        except Exception:
            logger.debug("get_total_execution_signals: error", exc_info=True)
            return 0

    def get_execution_bias(
        self, lane: str = "planner", context_bucket: str = ""
    ) -> dict[str, float]:
        """Return confidence-gated execution bias for strategy scoring.

        When ``context_bucket`` is provided, uses context-specific rates if
        sufficient data exists; otherwise falls back to lane-level rates.
        Returns empty dict when insufficient signals — no bias applied.
        """
        learner = self._get_execution_learner()
        if learner is None:
            return {}
        try:
            return learner.get_execution_bias(lane, context_bucket=context_bucket)
        except Exception:
            logger.debug("get_execution_bias: error", exc_info=True)
            return {}

    # ── Strategy policy learning ──────────────────────────────────────────────

    def _get_policy_learner(self) -> Any:
        """Return (lazy-initialised) StrategyPolicyLearner, or None.

        On first access for a new model: loads model-specific state if available,
        otherwise loads transferable knowledge from shared directory (warm-start).

        Double-checked locking under ``_telemetry_lock`` guards the lazy init against
        concurrent first-access from parallel sub-agent completion hooks (without it,
        two threads could both pass the ``is None`` check and double-init/double-load
        from disk).
        """
        if self._policy_learner is None:
            with self._telemetry_lock:
                if self._policy_learner is None:
                    try:
                        from .weight_learning import StrategyPolicyLearner
                        learner = StrategyPolicyLearner()
                        self._policy_learner = learner  # Set BEFORE loading to prevent recursion
                        self._load_policy_learner_state(learner)
                        # If no model-specific state was loaded, try shared knowledge
                        _has_data = bool(learner._q_table or learner._strategy_perf)
                        if not _has_data:
                            self.load_shared_knowledge()
                    except ImportError:
                        logger.debug("run_store: StrategyPolicyLearner not available")
        return self._policy_learner

    def update_strategy_policy(self, signal_dict: dict, strategy: str) -> None:
        """Update strategy policy Q-values from a single execution signal."""
        learner = self._get_policy_learner()
        if learner is None or not strategy:
            return
        with self._telemetry_lock:
            try:
                from .weight_learning import ExecutionLearningSignal
                sig = ExecutionLearningSignal.from_dict(signal_dict)
                learner.update(sig, strategy)
                self._save_policy_learner_state()
            except Exception:
                logger.debug("update_strategy_policy: error", exc_info=True)

    def update_strategy_policy_trajectory(
        self, steps: list[tuple]
    ) -> None:
        """Update policy with backward credit propagation over a multi-step trajectory.

        ``steps``: list of (signal_dict, strategy_name) in chronological order.
        Single-step lists fall back to immediate reward (same as update_strategy_policy).
        """
        learner = self._get_policy_learner()
        if learner is None or not steps:
            return
        with self._telemetry_lock:
            try:
                from .weight_learning import ExecutionLearningSignal
                parsed = [
                    (ExecutionLearningSignal.from_dict(sd), strat)
                    for sd, strat in steps
                    if strat
                ]
                if parsed:
                    learner.update_trajectory(parsed)
                    self._save_policy_learner_state()
            except Exception:
                logger.debug("update_strategy_policy_trajectory: error", exc_info=True)

    def get_strategy_report(self) -> dict[str, Any]:
        """Return comprehensive strategy performance report."""
        learner = self._get_policy_learner()
        if learner is None:
            return {}
        try:
            return learner.get_strategy_report()
        except Exception:
            logger.debug("get_strategy_report: error", exc_info=True)
            return {}

    def is_strategy_deprecated(self, strategy: str, context_bucket: str = "") -> bool:
        """Check if a strategy has been deprecated (context-aware).

        Checks context-specific status first, then global.
        """
        learner = self._get_policy_learner()
        if learner is None:
            return False
        try:
            return learner.is_deprecated(strategy, context_bucket=context_bucket)
        except Exception:
            return False  # non-critical — never block execution

    def get_policy_scores(
        self,
        context_bucket: str,
        has_symbol_target: bool = False,
        has_tests: bool = False,
        strategies: Optional[list[str]] = None,
        scope: str = "small",
    ) -> dict[str, float]:
        """Return policy-learned score adjustments for strategies."""
        learner = self._get_policy_learner()
        if learner is None:
            return {}
        try:
            return learner.get_policy_scores(
                context_bucket, has_symbol_target, has_tests, strategies, scope=scope
            )
        except Exception:
            logger.debug("get_policy_scores: error", exc_info=True)
            return {}

    def _load_policy_learner_state(self, learner: Any) -> None:
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            ns = f"policy/{self._model_name}" if self._model_name else "policy"
            state = read_namespace(ns)
            if isinstance(state, dict):
                learner.load_state(state)
        except Exception as exc:
            logger.debug("run_store: could not restore policy state (%s)", exc)

    def _save_policy_learner_state(self) -> None:
        learner = self._policy_learner
        if learner is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import batch_write_namespaces

            # Batch policy + transferable state into one atomic RMW cycle
            ns = f"policy/{self._model_name}" if self._model_name else "policy"
            ns_map: dict[str, object] = {ns: learner.get_summary()}

            transferable = learner.get_transferable_state()
            if transferable:
                ns_map["transferable_knowledge"] = transferable

            batch_write_namespaces(ns_map)
        except Exception as exc:
            logger.debug("run_store: _save_policy_learner_state failed: %s", exc)

    # ── Shared knowledge (transferable across models) ─────────────────────────
    # NOTE: there is intentionally no save_shared_knowledge() here. Transferable
    # knowledge is persisted by _save_policy_learner_state(), which batches the
    # model-specific policy namespace AND "transferable_knowledge" into ONE
    # atomic RMW cycle (batch_write_namespaces). A standalone writer here would
    # bypass that atomic batch and open a lost-update window between the two
    # writes (insight A29). load_shared_knowledge() below is the cold-start
    # fallback that seeds a fresh model from the shared namespace.

    def load_shared_knowledge(self) -> None:
        """Load transferable knowledge from strategy_state.json."""
        learner = self._get_policy_learner()
        if learner is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            state = read_namespace("transferable_knowledge")
            if isinstance(state, dict):
                learner.load_transferable_state(state)
                logger.info("run_store: loaded shared knowledge from strategy_state.json")
        except Exception as exc:
            logger.debug("load_shared_knowledge failed: %s", exc)

    # ── Adaptive learner hub (tool/patch/context/routing/prompt) ──────────────

    def _get_adaptive_hub(self) -> Any:
        """Return (lazy-initialised) AdaptiveLearnerHub."""
        if self._adaptive_hub is None:
            try:
                from .weight_learning import AdaptiveLearnerHub
                hub = AdaptiveLearnerHub()
                self._load_adaptive_hub_state(hub)
                self._adaptive_hub = hub
            except ImportError:
                logger.debug("run_store: AdaptiveLearnerHub not available")
        return self._adaptive_hub


    def _load_adaptive_hub_state(self, hub: Any) -> None:
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            ns = f"adaptive_hub/{self._model_name}" if self._model_name else "adaptive_hub"
            state = read_namespace(ns)
            if isinstance(state, dict):
                hub.load_state(state)
        except Exception as exc:
            logger.debug("run_store: could not restore adaptive hub (%s)", exc)

    def _save_adaptive_hub_state(self) -> None:
        hub = self._adaptive_hub
        if hub is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import write_namespace
            state = hub.get_summary()
            ns = f"adaptive_hub/{self._model_name}" if self._model_name else "adaptive_hub"
            write_namespace(ns, state)
        except Exception as exc:
            logger.debug("run_store: _save_adaptive_hub_state failed: %s", exc)

    @contextmanager
    def batch_adaptive_signals(self):
        """Batch multiple record_* calls into one persistence write.

        During the block, in-memory hub updates proceed normally but disk
        persistence is deferred to block exit — turning N consecutive
        record_* calls (which all write the same ``adaptive_hub`` namespace)
        into a single read-merge-write cycle instead of N.
        """
        self._hub_save_suspended = True
        try:
            yield
        finally:
            self._hub_save_suspended = False
            self._save_adaptive_hub_state()

    # Convenience methods for signal recording
    def record_tool_usage(self, phase: str, tool_name: str, success: bool,
                          context_bucket: str = "") -> None:
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_tool_usage(phase, tool_name, success, context_bucket)
            if not getattr(self, "_hub_save_suspended", False):
                self._save_adaptive_hub_state()

    def record_patch_result(self, failure_class: str, file_ext: str,
                            method: str, success: bool, repair_rounds: int = 0) -> None:
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_patch_result(failure_class, file_ext, method, success, repair_rounds)
            if not getattr(self, "_hub_save_suspended", False):
                self._save_adaptive_hub_state()

    def record_context_result(self, task_type: str, context_config: str,
                              success: bool, plan_quality: float = 0.0) -> None:
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_context_result(task_type, context_config, success, plan_quality)
            if not getattr(self, "_hub_save_suspended", False):
                self._save_adaptive_hub_state()

    def record_routing_result(self, request_features: str, lane: str,
                              success: bool, was_fallback: bool = False) -> None:
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_routing_result(request_features, lane, success, was_fallback)
            if not getattr(self, "_hub_save_suspended", False):
                self._save_adaptive_hub_state()

    def record_prompt_result(self, strategy: str, variant: str,
                             success: bool, plan_quality: float = 0.0) -> None:
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_prompt_result(strategy, variant, success, plan_quality)
            if not getattr(self, "_hub_save_suspended", False):
                self._save_adaptive_hub_state()

    # ── Execution learner persistence ──────────────────────────────────────────

    def _load_execution_learner_state(self, learner: Any) -> None:
        """Restore lane states into *learner* from strategy_state.json."""
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            ns = f"execution_state/{self._model_name}" if self._model_name else "execution_state"
            state = read_namespace(ns)
            if isinstance(state, dict):
                learner.load_state(state)
                logger.debug("run_store: restored execution learner state from %s", ns)
        except Exception as exc:
            logger.debug("run_store: could not restore execution state (%s) — starting fresh", exc)

    def _save_execution_learner_state(self) -> None:
        """Persist the current execution learner state to strategy_state.json."""
        learner = self._execution_learner
        if learner is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import write_namespace
            state = learner.get_summary()
            ns = f"execution_state/{self._model_name}" if self._model_name else "execution_state"
            write_namespace(ns, state)
        except Exception as exc:
            logger.debug("run_store: _save_execution_learner_state failed: %s", exc)

    # -------------------------------------------------------------------------
    # multi_strategy_gate: cooldown state management
    # -------------------------------------------------------------------------

    def get_multi_gate_state(self, state_key: str) -> Any:
        """Return cooldown state for the given state_key.

        Returns a simple namespace-like dict with fields:
            no_gain_count      — consecutive MULTI decisions with no quality gain
            cooldown_remaining — runs left to suppress MULTI (0 = no cooldown)
        """
        try:
            from external_llm.editor._editor_core.lane.multi_strategy_gate import MultiGateCooldownState
        except ImportError:
            # lane is excluded from the public build — degrade gracefully.
            logger.debug("get_multi_gate_state: lane not available")
            return None
        s = self._multi_gate_state.get(state_key)
        if s is None:
            return MultiGateCooldownState(state_key=state_key)
        return MultiGateCooldownState(
            state_key=state_key,
            no_gain_count=s.get("no_gain_count", 0),
            cooldown_remaining=s.get("cooldown_remaining", 0),
        )

    def record_multi_gate_outcome(
        self,
        state_key: str,
        was_multi: bool,
        realized_gain: float,
        repair_rounds: int = 0,
    ) -> None:
        """Update cooldown state based on gate outcome.

        If was_multi and realized_gain <= 0: increment no_gain_count.
        If no_gain_count >= _COOLDOWN_NO_GAIN_LIMIT: set cooldown_remaining.
        If was_multi and realized_gain > 0: reset no_gain_count.
        Each call decrements cooldown_remaining by 1 (cooldown tick).
        """
        try:
            from external_llm.editor._editor_core.lane.multi_strategy_gate import _COOLDOWN_NO_GAIN_LIMIT, _COOLDOWN_RUNS
        except ImportError:
            # lane is excluded from the public build — degrade gracefully.
            logger.debug("record_multi_gate_outcome: lane not available, skipping")
            return
        s = self._multi_gate_state.setdefault(
            state_key, {"no_gain_count": 0, "cooldown_remaining": 0}
        )
        # Tick down cooldown regardless
        if s["cooldown_remaining"] > 0:
            s["cooldown_remaining"] -= 1

        if was_multi:
            if realized_gain <= 0.0:
                s["no_gain_count"] += 1
                if s["no_gain_count"] >= _COOLDOWN_NO_GAIN_LIMIT:
                    s["cooldown_remaining"] = _COOLDOWN_RUNS
                    s["no_gain_count"] = 0
                    logger.debug(
                        "multi_gate: cooldown activated for state=%s (no_gain=%d)",
                        state_key, _COOLDOWN_NO_GAIN_LIMIT,
                    )
            else:
                # MULTI produced gain — reset no_gain streak
                s["no_gain_count"] = 0
