"""
AutonomousTaskQueue — priority queue with concurrency limiting and deduplication.

Features:
  - Priority ordering:  priority 0 (critical) served before 2 (normal)
  - Concurrency limit:  MAX_CONCURRENT tasks running simultaneously
  - File deduplication: later event for the same source_file replaces
                        any pending (not yet executing) task for that file
  - Non-blocking get:   get_nowait() returns None when queue empty or at limit
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.editor.agent.autonomous.trigger_engine import TriggerEvent
from external_llm.editor.agent.autonomous.trigger_policy import ActionDecision, ActionKind

logger = logging.getLogger(__name__)


@dataclass(order=True)
class AutonomousTask:
    priority: int                                      # sort key: lower = higher priority
    created_at: float = field(compare=False)
    event: TriggerEvent = field(compare=False)
    action: ActionDecision = field(compare=False)
    task_id: str = field(compare=False, default="")


class AutonomousTaskQueue:
    """Thread-safe priority queue for autonomous tasks."""

    MAX_CONCURRENT = 2   # max tasks executing simultaneously
    # Defense-in-depth cap on pending tasks. Superseded/orphaned tasks are purged
    # by get_nowait() regardless of the concurrency limit (see below), so under
    # normal operation the queue holds at most a handful of *live* tasks. This cap
    # guards against unbounded growth if upstream policy cooldowns are bypassed.
    MAX_PENDING = _cfg.counts.AUTONOMOUS_TASK_QUEUE_MAX

    def __init__(self):
        self._pq: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._running_count = 0
        # source_file → task_id of the pending (not yet running) task for that file
        self._pending_file_map: dict[str, str] = {}
        self._counter = 0
        # Number of tasks rejected because MAX_PENDING was reached (observability).
        self._dropped_count = 0

    def _purge_tombstones_locked(self) -> int:
        """Drop superseded/orphaned (tombstone) tasks from the heap, re-queueing
        the live ones. Caller must hold ``self._lock``.

        A task is a tombstone when its ``source_file`` is tracked in
        ``_pending_file_map`` under a *different* (newer) task_id, or the map
        entry was already cleared (the newer task already ran). Tasks without a
        ``source_file`` are never tombstones. Map entries are not touched here:
        a tombstone never owns the current map slot (it points to a newer task or
        was already deleted), and a live task must remain tracked.

        Returns the number of tombstones purged (observability).
        """
        live: list[AutonomousTask] = []
        purged = 0
        while True:
            try:
                task = self._pq.get_nowait()
            except queue.Empty:
                break
            sf = task.event.source_file
            if sf and self._pending_file_map.get(sf) != task.task_id:
                purged += 1
                continue
            live.append(task)
        for task in live:
            self._pq.put(task)
        if purged:
            logger.debug(
                "Purged %d tombstone task(s) during cap-check (live=%d)",
                purged, len(live),
            )
        return purged

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def enqueue(self, event: TriggerEvent, action: ActionDecision) -> Optional[str]:
        """
        Enqueue a task. Returns task_id or None if ignored or rejected at cap.

        If a pending task already exists for event.source_file, the new task
        replaces it conceptually (old task will be silently dropped on get).

        If the queue is at MAX_PENDING, the new task is rejected (newest-loss:
        an already-loaded backlog should be drained before accepting more) and
        the dropped counter is incremented.
        """
        if action.kind == ActionKind.IGNORE:
            return None

        with self._lock:
            # A supersede (same source_file already pending) does not grow the
            # *live* task count — the old entry becomes a tombstone purged on the
            # next get_nowait(). Exempt it from the cap so the newest event for a
            # file always wins, even under backpressure.
            is_supersede = bool(event.source_file and event.source_file in self._pending_file_map)
            if not is_supersede and self._pq.qsize() >= self.MAX_PENDING:
                # ``qsize()`` counts tombstones (superseded entries not yet reaped
                # by get_nowait()). Under a same-file supersede burst they can
                # inflate qsize far beyond the *live* task count and wrongly
                # reject an unrelated file's first task. Purge tombstones and
                # re-check against the true live count before dropping a new task.
                self._purge_tombstones_locked()
                if self._pq.qsize() >= self.MAX_PENDING:
                    self._dropped_count += 1
                    logger.warning(
                        "AutonomousTaskQueue at cap (%d); dropping new task for %s "
                        "(dropped_total=%d)",
                        self.MAX_PENDING, event.source_file, self._dropped_count,
                    )
                    return None

            self._counter += 1
            task_id = f"auto-{self._counter}"

            task = AutonomousTask(
                priority=action.priority,
                created_at=time.time(),
                event=event,
                action=action,
                task_id=task_id,
            )
            self._pq.put(task)

            if event.source_file:
                self._pending_file_map[event.source_file] = task_id

            logger.debug("Enqueued task %s (%s, priority=%d)", task_id, action.kind.value, action.priority)
            return task_id

    # ── Dequeue ───────────────────────────────────────────────────────────────

    def get_nowait(self) -> Optional[AutonomousTask]:
        """
        Get next task if concurrency limit not reached. Non-blocking.
        Returns None if queue is empty or at MAX_CONCURRENT.

        Superseded/orphaned tasks are purged *regardless* of the concurrency
        limit: the entire queue is scanned each call so stale tasks never
        accumulate while both execution slots are occupied by long-running
        AUTO_FIX tasks. At most one live task is claimed per call (if a slot is
        free); remaining live tasks are re-queued for subsequent calls.
        """
        with self._lock:
            result: Optional[AutonomousTask] = None
            held_live: list[AutonomousTask] = []
            while True:
                try:
                    task = self._pq.get_nowait()
                except queue.Empty:
                    break
                sf = task.event.source_file
                if sf:
                    current = self._pending_file_map.get(sf)
                    if current != task.task_id:
                        # Superseded OR orphaned: either a newer task for the
                        # same file is still pending (current is a newer id), or
                        # a newer task already ran and cleared this map entry
                        # (current is None). In both cases this stale task must
                        # NOT run — purge it WITHOUT claiming a concurrency slot.
                        # This purge runs even when _running_count >= MAX_CONCURRENT
                        # so stale tasks cannot accumulate during long runs.
                        logger.debug(
                            "Dropping superseded task %s for %s (current=%s)",
                            task.task_id, sf, current,
                        )
                        continue
                    del self._pending_file_map[sf]
                # Live task. Claim a slot for the first one if available;
                # hold the rest aside to re-queue after the purge completes.
                if result is None and self._running_count < self.MAX_CONCURRENT:
                    self._running_count += 1
                    result = task
                    # Keep scanning to purge remaining stale tasks.
                    continue
                held_live.append(task)
            # Re-queue live tasks that couldn't run (slot full or already filled).
            for t in held_live:
                self._pq.put(t)
                tsf = t.event.source_file
                if tsf:
                    self._pending_file_map[tsf] = t.task_id
            if result is not None:
                logger.debug(
                    "Dequeued task %s. Running: %d", result.task_id, self._running_count
                )
            return result

    # ── Completion ────────────────────────────────────────────────────────────

    def task_done(self, task_id: str) -> None:
        """Must be called when a task finishes (success or error)."""
        with self._lock:
            self._running_count = max(0, self._running_count - 1)
        logger.debug("Task %s done. Running: %d", task_id, self._running_count)

    # ── Status ────────────────────────────────────────────────────────────────

    def qsize(self) -> int:
        return self._pq.qsize()

    def running_count(self) -> int:
        with self._lock:
            return self._running_count

    def dropped_count(self) -> int:
        """Number of tasks rejected at MAX_PENDING (observability for backpressure)."""
        with self._lock:
            return self._dropped_count
