"""
ProactiveRunner — top-level coordinator for the autonomous agent system.

Wires together:
  TriggerEngine → TriggerPolicy → AutonomousTaskQueue
                                        ↓
                             _execute_task() (daemon thread)
                                        ↓
                             PushManager.broadcast() → SSE → browser

Module-level registry:
  _runners: Dict[repo_root → ProactiveRunner]

  get_or_create_runner(repo_root, ...)  — get or create a runner for a repo
  update_runner_model_tier(repo_root, tier)  — live model tier update
  make_stream_callback_interceptor(repo_root, original_cb)  — wraps AgentLoop
      stream_callback to forward stall/complete signals to TriggerEngine

LLM invoke bridge:
  ProactiveRunner.llm_invoke_fn is an Optional[Callable] injected by main.py.
  Signature: (repo_root, request_text, source_file) → Dict
  When None, SUGGEST/AUTO_FIX tasks return a stub result.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Optional

from external_llm.agent.config.thresholds import config as _cfg

logger = logging.getLogger(__name__)

# ── Module-level runner registry ─────────────────────────────────────────────
# Bounded LRU (OrderedDict, cap = config.counts.AUTONOMOUS_RUNNER_MAX). Each
# entry owns a drain daemon thread + TriggerEngine schedule timers, so the
# registry must be capped — an unbounded one leaks threads (not just memory)
# in long-lived multi-repo webapp processes. On overflow the LRU entry is
# evicted and stop()'d (thread + timers torn down). See get_or_create_runner.
_runners: "OrderedDict[str, ProactiveRunner]" = OrderedDict()
_runners_lock = threading.Lock()


def get_or_create_runner(
    repo_root: str,
    model_tier: str = "small",
    push_manager=None,
    llm_invoke_fn: Optional[Callable] = None,
) -> "ProactiveRunner":
    """
    Get the ProactiveRunner for repo_root, creating and starting one if absent.

    The registry is a bounded LRU (cap = config.counts.AUTONOMOUS_RUNNER_MAX).
    Access promotes the entry to most-recently-used; on overflow the
    least-recently-used runner is evicted and stop()'d (its drain daemon thread
    + TriggerEngine schedule timers are torn down).

    Args:
        repo_root:      Repository root path (key for deduplication)
        model_tier:     "none" | "small" | "strong" — controls action granularity
        push_manager:   PushManager instance (uses global singleton if None)
        llm_invoke_fn:  Optional callable to run LLM tasks. Injected by main.py.
    """
    evicted: Optional["ProactiveRunner"] = None
    with _runners_lock:
        if repo_root not in _runners:
            from external_llm.editor.agent.autonomous.push_manager import get_push_manager
            pm = push_manager or get_push_manager()
            runner = ProactiveRunner(
                repo_root=repo_root,
                model_tier=model_tier,
                push_manager=pm,
                llm_invoke_fn=llm_invoke_fn,
            )
            runner.start()
            _runners[repo_root] = runner
            logger.info(
                "ProactiveRunner created for repo=%s (model_tier=%s)", repo_root, model_tier
            )
            # LRU eviction: bound the per-repo registry. Each entry owns a drain
            # daemon thread + TriggerEngine schedule timers, so an unbounded
            # registry leaks threads (not just memory) in long-lived multi-repo
            # webapp processes.
            max_runners = _cfg.counts.AUTONOMOUS_RUNNER_MAX
            if max_runners > 0 and len(_runners) > max_runners:
                evicted_repo, evicted = _runners.popitem(last=False)
                logger.warning(
                    "ProactiveRunner registry cap (%d) exceeded; evicting LRU runner "
                    "for repo=%s (drain thread + engine timers will be torn down)",
                    max_runners, evicted_repo,
                )
        else:
            # Update llm_invoke_fn if provided (called again after model selection)
            runner = _runners[repo_root]
            if llm_invoke_fn is not None:
                runner.llm_invoke_fn = llm_invoke_fn
            if model_tier and model_tier != "none":
                runner.policy.model_tier = model_tier
            _runners.move_to_end(repo_root)  # promote to most-recently-used
        result = _runners[repo_root]
    # Tear down the evicted runner OUTSIDE _runners_lock to avoid holding the
    # registry lock during engine timer cancellation (lock-ordering convention;
    # mirrors app_state.cancel_session, which performs queue.put outside the
    # session lock). stop() sets _running=False (drain thread exits on next
    # sleep) and cancels all TriggerEngine schedule timers.
    if evicted is not None:
        try:
            evicted.stop()
        except Exception:
            logger.warning("Failed to stop evicted ProactiveRunner", exc_info=True)
    return result


def update_runner_model_tier(repo_root: str, model_tier: str) -> None:
    """Live-update model tier for an existing runner (no restart needed)."""
    with _runners_lock:
        runner = _runners.get(repo_root)
    if runner:
        runner.policy.model_tier = model_tier
        logger.info("ProactiveRunner model_tier updated: %s → %s", repo_root, model_tier)


def update_runner_features(repo_root: str, features_csv: str) -> None:
    """
    Live-update enabled features for an existing runner.

    features_csv: comma-separated feature names, e.g.
        "file_review,test_analysis,agent_stall"
    Empty string or None → all features enabled (default).
    """
    with _runners_lock:
        runner = _runners.get(repo_root)
    if not runner:
        return
    from external_llm.editor.agent.autonomous.trigger_policy import TriggerPolicy
    if not features_csv or features_csv.strip() == "all":
        runner.policy.enabled_features = set(TriggerPolicy._ALL_FEATURES)
    else:
        requested = {f.strip() for f in features_csv.split(",") if f.strip()}
        runner.policy.enabled_features = requested & TriggerPolicy._ALL_FEATURES
    logger.info(
        "ProactiveRunner features updated: %s → %s",
        repo_root, runner.policy.enabled_features,
    )


def make_stream_callback_interceptor(
    repo_root: str,
    original_cb: Optional[Callable],
) -> Callable:
    """
    Wrap the AgentLoop stream_callback to intercept agent state events.

    The returned callback:
      1. Forwards every event to original_cb unchanged (zero behavior change)
      2. Extracts fail_loop_detected / complete / error → routes to TriggerEngine

    Usage in main.py (inside _run_agent):
        config.stream_callback = make_stream_callback_interceptor(
            repo_root, _stream_cb
        )
    """

    def _interceptor(event_name: str, data: dict[str, Any]) -> None:
        # Always forward first — never break existing behavior
        if original_cb:
            try:
                original_cb(event_name, data)
            except Exception:
                pass  # non-critical — never block execution

        # Route relevant events to TriggerEngine
        if event_name in ("fail_loop_detected", "complete", "error"):
            with _runners_lock:
                runner = _runners.get(repo_root)
            if runner and runner._engine:
                try:
                    runner._engine.notify_agent_event(event_name, data)
                except Exception as exc:
                    logger.debug("interceptor notify_agent_event error: %s", exc)

    return _interceptor


# ── ProactiveRunner ───────────────────────────────────────────────────────────

class ProactiveRunner:
    """
    Main autonomous agent coordinator.

    Lifecycle:
      start() → TriggerEngine.start() + drain daemon thread
      stop()  → TriggerEngine.stop() + drain thread exits on next sleep
    """

    DRAIN_INTERVAL = _cfg.counts.PROACTIVE_DRAIN_INTERVAL_S

    def __init__(
        self,
        repo_root: str,
        model_tier: str = "small",
        push_manager=None,
        llm_invoke_fn: Optional[Callable] = None,
    ):
        self.repo_root = repo_root
        self.llm_invoke_fn = llm_invoke_fn

        from external_llm.editor.agent.autonomous.push_manager import get_push_manager
        from external_llm.editor.agent.autonomous.task_queue import AutonomousTaskQueue
        from external_llm.editor.agent.autonomous.trigger_engine import TriggerEngine
        from external_llm.editor.agent.autonomous.trigger_policy import TriggerPolicy

        self.push = push_manager or get_push_manager()
        self._engine = TriggerEngine(repo_root)
        self.policy = TriggerPolicy(model_tier=model_tier)
        self._queue = AutonomousTaskQueue()

        self._running = False
        self._drain_thread: Optional[threading.Thread] = None
        # Guards _running flag + _drain_thread lifecycle (start/stop). Mirrors
        # TriggerEngine's pattern; prevents concurrent start() from spawning two
        # drain threads and stop()/start() races that orphan a joinable thread.
        self._lifecycle_lock = threading.Lock()

        # Wire TriggerEngine → policy → queue
        self._engine.on_trigger(self._on_trigger)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            self._engine.start()

            self._drain_thread = threading.Thread(
                target=self._drain_loop,
                daemon=True,
                name=f"proactive-drain-{self.repo_root[-24:]}",
            )
            self._drain_thread.start()
            logger.info("ProactiveRunner started (repo=%s, model_tier=%s)",
                        self.repo_root, self.policy.model_tier)

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._running:
                return
            self._running = False
            self._engine.stop()
            logger.info("ProactiveRunner stopped (repo=%s)", self.repo_root)

    # ── Schedule helpers ──────────────────────────────────────────────────────

    # ── Test result forwarding ────────────────────────────────────────────────

    def notify_test_result(self, ok: bool, details: dict[str, Any]) -> None:
        self._engine.notify_test_result(ok, details)

    # ── Internal: trigger → policy → queue ───────────────────────────────────

    def _on_trigger(self, event) -> None:
        """Receive TriggerEvent from engine, evaluate policy, enqueue or push."""
        from external_llm.editor.agent.autonomous.trigger_policy import ActionKind

        decision = self.policy.evaluate(event)

        if decision.kind == ActionKind.IGNORE:
            return

        # ESCALATE: push immediately without queuing (user needs to see it now)
        if decision.kind == ActionKind.ESCALATE:
            self.push.broadcast("proactive_escalation", {
                "message": decision.message,
                "event_kind": event.kind.value,
                "source_file": event.source_file,
                "metadata": event.metadata,
                "timestamp": event.timestamp,
                "priority": decision.priority,
            })
            return

        self._queue.enqueue(event, decision)

    # ── Drain loop ────────────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        """Background daemon thread: drains task queue and spawns executor threads."""
        while self._running:
            task = self._queue.get_nowait()
            if task:
                t = threading.Thread(
                    target=self._execute_task,
                    args=(task,),
                    daemon=True,
                    name=f"proactive-exec-{task.task_id}",
                )
                t.start()
            else:
                time.sleep(self.DRAIN_INTERVAL)

    # ── Task execution ────────────────────────────────────────────────────────

    def _execute_task(self, task) -> None:
        """Execute one autonomous task. Always calls task_done() in finally."""
        from external_llm.editor.agent.autonomous.trigger_policy import ActionKind

        try:
            action = task.action
            event = task.event

            if action.kind == ActionKind.NOTIFY:
                self.push.broadcast("proactive_notification", {
                    "message": action.message,
                    "event_kind": event.kind.value,
                    "source_file": event.source_file,
                    "timestamp": event.timestamp,
                    "priority": action.priority,
                })

            elif action.kind in (ActionKind.SUGGEST, ActionKind.AUTO_FIX):
                # Signal UI that work is starting
                self.push.broadcast("proactive_fix_started", {
                    "task_id": task.task_id,
                    "event_kind": event.kind.value,
                    "source_file": event.source_file,
                    "message": action.message,
                    "action": action.kind.value,
                    "timestamp": time.time(),
                })

                result = self._run_llm_task(event, action)

                self.push.broadcast("proactive_fix_done", {
                    "task_id": task.task_id,
                    "event_kind": event.kind.value,
                    "source_file": event.source_file,
                    "result": result,
                    "action": action.kind.value,
                    "message": action.message,
                    "timestamp": time.time(),
                })

        except Exception as exc:
            logger.warning("ProactiveRunner task %s error: %s", task.task_id, exc, exc_info=True)
            self.push.broadcast("proactive_error", {
                "task_id": task.task_id,
                "error": str(exc),
                "event_kind": task.event.kind.value,
            })
        finally:
            self._queue.task_done(task.task_id)

    def _run_llm_task(self, event, action) -> dict[str, Any]:
        """
        Invoke LLM for SUGGEST / AUTO_FIX.

        If llm_invoke_fn is set (injected by main.py), call it.
        Otherwise return a stub result so the rest of the pipeline still works.
        """
        if not self.llm_invoke_fn:
            return {
                "status": "no_model",
                "message": "No LLM invoke function connected. "
                           "Inject llm_invoke_fn via main.py.",
                "prompt_preview": action.prompt[:200] if action.prompt else "",
            }

        try:
            return self.llm_invoke_fn(
                repo_root=self.repo_root,
                request_text=action.prompt,
                source_file=event.source_file,
            )
        except Exception as exc:
            logger.warning("llm_invoke_fn error: %s", exc)
            return {"status": "error", "error": str(exc)}
