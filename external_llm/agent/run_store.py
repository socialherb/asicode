"""
In-memory run store — adaptive tool-usage learning hub and per-thread model
context.

Kept surface (the only production-used parts of the original P14 run-store):
  * ``record_tool_usage`` / ``batch_adaptive_signals`` — adaptive learner hub
    recording with debounced persistence (agent_loop / design_chat_loop).
  * ``model_context_scope`` / ``set_model_context`` — per-thread model binding
    so hub persistence lands in the correct per-model namespace (orchestrator /
    webapp session handler).

The P14 execution-record surface (RunRecord, repair memory, weight/execution/
policy learners, unified-store write-through) had no production callers and
was removed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


# Adaptive-hub persistence debounce (see batch_adaptive_signals). Bounds how
# much learning data a crash can lose while still collapsing the per-tool-call
# write storm: at most this many signals, or this many seconds, between writes.
_HUB_FLUSH_INTERVAL_S = 5.0
_HUB_FLUSH_MAX_PENDING = 25


class InMemoryRunStore:
    """Adaptive tool-usage learning hub store with per-thread model context."""

    def __init__(self, model_name: str = ""):
        # Adaptive-hub write batching (see batch_adaptive_signals). Thread-local:
        # this store is shared across concurrent subagent loops, so batching
        # depth/pending state must never be a plain instance field.
        self._hub_batch_state = threading.local()
        # Model context is PER-THREAD (threading.local), not shared instance state.
        # The run_store is a process-lifetime singleton shared across concurrent
        # sessions, each running in its own agent_executor thread. A shared instance
        # field would let session B's set_model_context overwrite session A's model
        # mid-run, so hub persistence would attribute to B's model — silently
        # corrupting per-model learning data. _model_name / _developer_model_name
        # are properties backed by this thread-local; seeding here sets the
        # constructing thread's context (the global singleton is built once on the
        # import thread).
        self._model_ctx = threading.local()
        self._model_name = self._normalize_model_name(model_name)
        self._developer_model_name = ""
        # Adaptive learner hub (tool usage). Hubs keyed by persistence namespace
        # (see _adaptive_hub_namespace). One entry per model context this process
        # touches — bounded by the number of distinct models in a session, so a
        # handful; sub-agents reuse the parent's entry when they share its model.
        self._adaptive_hubs: dict[str, Any] = {}
        self._adaptive_hub_lock = threading.Lock()
        self._migrate_legacy_state()

    def _migrate_legacy_state(self) -> None:
        """One-time migration from legacy per-file adaptive-hub state."""
        import json

        from external_llm.editor.learning.strategy_state import read_namespace, write_namespace

        ns = f"adaptive_hub/{self._model_name}" if self._model_name else "adaptive_hub"
        if read_namespace(ns) is not None:
            return
        legacy = os.path.join(self._model_dir(), "adaptive_hub_state.json")
        if not os.path.isfile(legacy):
            return
        try:
            with open(legacy, encoding="utf-8") as fh:
                state = json.load(fh)
            if isinstance(state, dict) and state:
                write_namespace(ns, state)
                logger.info("run_store: migrated %s -> strategy_state:%s", legacy, ns)
        except Exception:
            logger.debug("run_store: %s migration error", legacy, exc_info=True)

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

    # ── Per-thread model context ─────────────────────────────────
    # ``_model_name`` / ``_developer_model_name`` are properties backed by
    # ``threading.local`` (see __init__) so each session's worker thread sees only
    # its own model. The property keeps thread isolation while internal callers
    # (``_adaptive_hub_namespace``) read the value transparently. Without this,
    # concurrent sessions sharing the singleton store overwrite each other's model
    # context (a read-modify race on instance state).
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
        so adaptive-hub persistence attributes to the correct model rather than the
        parent session's planner model (sub-agents share the singleton run_store but
        run on distinct worker threads). Parallel sub-agents are naturally isolated by
        ``threading.local``; the save/restore matters for sequential mode, which
        reuses the parent thread and would otherwise leak the last sub-agent's model
        into subsequent parent work.
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
        """Update model names for the CALLING THREAD.

        Called per-request so the singleton run_store knows which models are active
        for the calling thread without requiring per-request instantiation. Backed by
        ``threading.local`` (see ``_model_name``/``_developer_model_name`` properties),
        so concurrent sessions on separate worker threads never observe each other's
        model context.
        """
        self._model_name = self._normalize_model_name(planner_model)
        self._developer_model_name = self._normalize_model_name(developer_model)

    # ── Adaptive learner hub (tool usage) ─────────────────────────

    def _get_adaptive_hub(self) -> Any:
        """Return the lazy-initialised AdaptiveLearnerHub for this thread's model.

        Keyed by :meth:`_adaptive_hub_namespace`, NOT a single instance field.
        The namespace is thread-local (per-model) while the hub used to be shared
        instance state, so the two disagreed: the hub was loaded once under
        whichever namespace the first caller's thread had — in practice the
        parent session's generic ``adaptive_hub``, since agent_loop constructs
        the store with no model_name — and then a sub-agent thread inside
        ``model_context_scope`` saved that same shared object to
        ``adaptive_hub/<model>``. That per-model namespace was therefore
        WRITE-ONLY (the load had already happened elsewhere), and every
        sub-agent flush copied the parent's whole blob into it, after which the
        two drifted. Keying the cache the same way the load and save are keyed
        removes the split at the source.

        The lock is required for the same reason the telemetry RMW paths have
        one: parallel sub-agents share this singleton. Two threads racing the
        lazy init would each build a hub, one would overwrite the other in the
        map, and any signals already recorded into the loser were silently
        dropped — the loser is never saved, since saving reads the map.
        Double-checked so the common (already-built) path stays lock-free.
        """
        ns = self._adaptive_hub_namespace()
        hub = self._adaptive_hubs.get(ns)
        if hub is not None:
            return hub
        with self._adaptive_hub_lock:
            hub = self._adaptive_hubs.get(ns)
            if hub is not None:
                return hub
            # LIVE cross-module edge — do NOT treat as optional/dead code.
            # This lazy import is the production adapter-learning channel:
            # agent_loop/design_chat_loop -> record_tool_usage() ->
            # _get_adaptive_hub() -> AdaptiveLearnerHub (MiniQLearner EMA),
            # persisted via _save_adaptive_hub_state() under the
            # adaptive_hub/{model} namespace. The method-local import is
            # deliberate (keep the store importable without the learning
            # stack), but it defeats static scanners: vulture and
            # public_dead_code cannot see the cross-module edge through it
            # and will report weight_learning.py as "production-unintegrated
            # legacy" — that is a FALSE POSITIVE (P1, twice).
            # weight_learning imports only stdlib — the import cannot fail,
            # so the old except ImportError fallback was dead code.
            from .weight_learning import AdaptiveLearnerHub
            hub = AdaptiveLearnerHub()
            self._load_adaptive_hub_state(hub, ns)
            self._adaptive_hubs[ns] = hub
            return hub

    def _adaptive_hub_namespace(self) -> str:
        """Persistence namespace for the CURRENT thread's model context.

        Single source of truth for the key, so the cache, the load and the save
        can no longer disagree about which hub they are talking about — they
        previously derived it independently, which is exactly how the split
        below arose.
        """
        return f"adaptive_hub/{self._model_name}" if self._model_name else "adaptive_hub"

    def _load_adaptive_hub_state(self, hub: Any, ns: str) -> None:
        try:
            from external_llm.editor.learning.strategy_state import read_namespace
            state = read_namespace(ns)
            if isinstance(state, dict):
                hub.load_state(state)
        except Exception as exc:
            logger.debug("run_store: could not restore adaptive hub (%s)", exc)

    def _save_adaptive_hub_state(self) -> None:
        ns = self._adaptive_hub_namespace()
        hub = self._adaptive_hubs.get(ns)
        if hub is None:
            return
        try:
            from external_llm.editor.learning.strategy_state import write_namespace
            write_namespace(ns, hub.get_summary())
        except Exception as exc:
            logger.debug("run_store: _save_adaptive_hub_state failed: %s", exc)

    @contextmanager
    def batch_adaptive_signals(self):
        """Batch multiple record_* calls into fewer persistence writes.

        Each record_* call otherwise re-serialises the WHOLE adaptive-hub
        namespace (~94 KB) and fsyncs it. In a MAIN_AGENT run that is one full
        write per tool call — measured at 11 writes / 1.3 MB / 41 ms for a
        12-turn run, i.e. most of the loop's own (non-LLM) wall clock.

        Inside the block, writes are debounced rather than fully suspended:
        state is still flushed every ``_HUB_FLUSH_INTERVAL_S`` or every
        ``_HUB_FLUSH_MAX_PENDING`` signals, so a crash mid-run loses at most a
        bounded slice of learning data instead of the entire session. A final
        flush happens on exit if anything is still pending.

        Thread-local and re-entrant. The store is a process-lifetime singleton
        that the orchestrator hands to CONCURRENT subagent loops, so batching
        state must not live in an instance field — one thread entering the
        block would otherwise suspend writes for every other thread, and its
        exit would re-enable them mid-batch. Nesting is depth-counted so an
        inner block cannot end the outer one's batch.
        """
        tls = self._hub_batch_state
        depth = getattr(tls, "depth", 0)
        if depth == 0:
            tls.pending = 0
            tls.last_flush = time.monotonic()
        tls.depth = depth + 1
        try:
            yield
        finally:
            tls.depth = getattr(tls, "depth", 1) - 1
            if tls.depth <= 0:
                tls.depth = 0
                if getattr(tls, "pending", 0):
                    tls.pending = 0
                    self._save_adaptive_hub_state()

    def _persist_hub_signal(self) -> None:
        """Write hub state now, or defer it when inside batch_adaptive_signals.

        Single choke point for the record_* family so the batching policy lives
        in one place instead of being re-implemented at each call site.
        """
        tls = self._hub_batch_state
        if getattr(tls, "depth", 0) <= 0:
            self._save_adaptive_hub_state()
            return
        pending = getattr(tls, "pending", 0) + 1
        now = time.monotonic()
        if (pending >= _HUB_FLUSH_MAX_PENDING
                or now - getattr(tls, "last_flush", now) >= _HUB_FLUSH_INTERVAL_S):
            tls.pending = 0
            tls.last_flush = now
            self._save_adaptive_hub_state()
        else:
            tls.pending = pending

    def record_tool_usage(self, phase: str, tool_name: str, success: bool,
                          context_bucket: str = "") -> None:
        """Record a tool-usage signal into the adaptive hub (batched persist)."""
        hub = self._get_adaptive_hub()
        if hub:
            hub.record_tool_usage(phase, tool_name, success, context_bucket)
            self._persist_hub_signal()
