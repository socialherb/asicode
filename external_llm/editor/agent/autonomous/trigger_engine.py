"""
TriggerEngine — event collection hub for the autonomous agent system.

Sources:
  1. AgentLoop interceptor — fail_loop_detected / complete / error callbacks
  2. TestRunner hooks   — test pass / fail notifications
  3. Scheduler          — periodic SCHEDULE events (threading.Timer)

All sources call emit(TriggerEvent) which fans out to registered callbacks.
Callbacks run synchronously in the emitting thread (kept lightweight).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TriggerKind(Enum):
    FILE_MODIFIED       = "file_modified"       # .py/.js/.ts saved
    TEST_FAILED         = "test_failed"          # pytest failure
    TEST_RECOVERED      = "test_recovered"       # failure → pass
    AGENT_STALL         = "agent_stall"          # fail_loop_detected (N consecutive fails)
    AGENT_COMPLETED     = "agent_completed"      # run finished successfully
    AGENT_FAILED        = "agent_failed"         # run finished with error
    IMPORT_ERROR        = "import_error"         # py_compile failure
    INTEGRATION_MISSING = "integration_missing"  # GSG: unlinked new module
    SCHEDULE            = "schedule"             # periodic cron event


@dataclass
class TriggerEvent:
    kind: TriggerKind
    repo_root: str
    source_file: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    severity: int = 0   # 0=info  1=warning  2=error  3=critical


class TriggerEngine:
    """Collects events from multiple sources and routes to registered on_trigger callbacks."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._callbacks: list[Callable[[TriggerEvent], None]] = []
        self._lock = threading.Lock()
        self._schedule_timers: list[threading.Timer] = []
        self._running = False

    # ── Registration ──────────────────────────────────────────────────────────

    def on_trigger(self, callback: Callable[[TriggerEvent], None]) -> None:
        """Register a callback to receive all TriggerEvents."""
        with self._lock:
            self._callbacks.append(callback)

    # ── Emission ──────────────────────────────────────────────────────────────

    def emit(self, event: TriggerEvent) -> None:
        """Emit event to all registered callbacks. Thread-safe."""
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(event)
            except Exception as exc:
                logger.warning("TriggerEngine callback error: %s", exc)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the trigger engine. Thread-safe."""
        with self._lock:
            if self._running:
                return
            self._running = True
        logger.info("TriggerEngine started (repo=%s)", self.repo_root)

    def stop(self) -> None:
        """Stop all sources. Thread-safe."""
        with self._lock:
            self._running = False
            timers = list(self._schedule_timers)
            self._schedule_timers.clear()
        for t in timers:
            try:
                t.cancel()
            except Exception:
                pass
        logger.info("TriggerEngine stopped")

    # ── Agent event hooks ─────────────────────────────────────────────────────

    def notify_agent_event(self, event_name: str, data: dict[str, Any]) -> None:
        """
        Forward stream_callback events from AgentLoop.
        Called by make_stream_callback_interceptor in proactive_runner.py.
        """
        if event_name == "fail_loop_detected":
            self.emit(TriggerEvent(
                kind=TriggerKind.AGENT_STALL,
                repo_root=self.repo_root,
                severity=2,
                metadata=data,
            ))
        elif event_name == "complete":
            status = data.get("status", "")
            if status in ("error", "max_turns"):
                self.emit(TriggerEvent(
                    kind=TriggerKind.AGENT_FAILED,
                    repo_root=self.repo_root,
                    severity=2,
                    metadata=data,
                ))
            else:
                self.emit(TriggerEvent(
                    kind=TriggerKind.AGENT_COMPLETED,
                    repo_root=self.repo_root,
                    severity=0,
                    metadata=data,
                ))

    # ── Test result hooks ─────────────────────────────────────────────────────

    def notify_test_result(self, ok: bool, details: dict[str, Any]) -> None:
        """Called after test runs. ok=True → TEST_RECOVERED, False → TEST_FAILED."""
        self.emit(TriggerEvent(
            kind=TriggerKind.TEST_RECOVERED if ok else TriggerKind.TEST_FAILED,
            repo_root=self.repo_root,
            severity=0 if ok else 2,
            metadata=details,
        ))

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def schedule(self, interval_seconds: float, label: str) -> None:
        """Register a periodic SCHEDULE trigger (fires every interval_seconds)."""

        # holder[0] tracks the currently active timer so _fire can remove it on re-arm
        holder: list[Optional[threading.Timer]] = [None]

        def _fire():
            if not self._running:
                return
            self.emit(TriggerEvent(
                kind=TriggerKind.SCHEDULE,
                repo_root=self.repo_root,
                severity=0,
                metadata={"label": label, "interval": interval_seconds},
            ))
            # Re-arm: remove completed timer from list, register the new one
            next_t = threading.Timer(interval_seconds, _fire)
            next_t.daemon = True
            with self._lock:
                if holder[0] in self._schedule_timers:
                    self._schedule_timers.remove(holder[0])
                self._schedule_timers.append(next_t)
                holder[0] = next_t
            next_t.start()

        t = threading.Timer(interval_seconds, _fire)
        t.daemon = True
        with self._lock:
            self._schedule_timers.append(t)
            holder[0] = t
        t.start()
        logger.info("TriggerEngine: scheduled '%s' every %.0fs", label, interval_seconds)
