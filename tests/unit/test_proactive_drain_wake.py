"""Tests for the proactive drain-loop wake signal (P3).

The drain daemon thread used to be a pure poller: ``get_nowait()`` +
``time.sleep(DRAIN_INTERVAL)``. Work enqueued between polls sat idle for up to
one full poll interval (default 1.0s), and ``stop()`` took up to one interval
to take effect. The fix adds a level-triggered ``wake`` event on
``AutonomousTaskQueue``, set by ``enqueue``/``task_done`` (and by
``ProactiveRunner.stop``), so a blocked drain thread returns immediately.

Race-safety contract exercised here: the drain loop runs
``clear → get_nowait → wait`` (clear happens BEFORE the drain), so
  * an enqueue landing after clear is found by get_nowait() itself, and
  * an enqueue/task_done landing after get_nowait() returned None is visible
    to wait() (the event is level-triggered and only cleared at loop top).

The integration tests stretch ``DRAIN_INTERVAL`` to 10s: any pickup within the
few-second test budget is only possible if the wake works (the old poll-only
loop would sleep 10s and the assertions would time out).
"""
from __future__ import annotations

import time

from external_llm.editor.agent.autonomous.proactive_runner import ProactiveRunner
from external_llm.editor.agent.autonomous.task_queue import AutonomousTaskQueue
from external_llm.editor.agent.autonomous.trigger_engine import TriggerEvent, TriggerKind
from external_llm.editor.agent.autonomous.trigger_policy import ActionDecision, ActionKind

NOTIFICATION = "proactive_notification"


def _ev(source_file="A.py"):
    return TriggerEvent(
        kind=TriggerKind.FILE_MODIFIED, repo_root=".", source_file=source_file
    )


def _dec(priority, kind=ActionKind.NOTIFY):
    return ActionDecision(kind=kind, priority=priority)


class _StubPush:
    """Records broadcasts; no SSE machinery, safe for unit tests."""

    def __init__(self):
        self.calls: list[tuple] = []

    def broadcast(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def notification_count(self) -> int:
        return sum(1 for args, _ in self.calls if args and args[0] == NOTIFICATION)


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ── Queue-level wake contract ─────────────────────────────────────────────────


def test_enqueue_sets_wake():
    q = AutonomousTaskQueue()
    assert q.wake.is_set() is False
    q.enqueue(_ev(), _dec(2))
    assert q.wake.is_set() is True


def test_ignored_and_rejected_enqueues_do_not_set_wake():
    q = AutonomousTaskQueue()
    # IGNORE decisions are never enqueued → no wake.
    q.enqueue(_ev(), _dec(2, kind=ActionKind.IGNORE))
    assert q.wake.is_set() is False
    # Rejected at MAX_PENDING → no task was added → no wake.
    q.MAX_PENDING = 0
    q.enqueue(_ev("B.py"), _dec(2))
    assert q.wake.is_set() is False


def test_task_done_sets_wake_after_clear():
    q = AutonomousTaskQueue()
    q.enqueue(_ev(), _dec(2))
    t = q.get_nowait()
    assert t is not None
    q.wake.clear()
    q.task_done(t.task_id)
    assert q.wake.is_set() is True


# ── Runner-level integration (DRAIN_INTERVAL stretched to 10s) ───────────────


class TestDrainWakeIntegration:
    def _make_runner(self, repo: str):
        push = _StubPush()
        runner = ProactiveRunner(repo_root=repo, push_manager=push)
        runner.DRAIN_INTERVAL = 10.0  # old poll-only loop would sleep 10s
        runner.start()
        return runner, push

    def _teardown_runner(self, runner) -> None:
        runner.stop()
        if runner._drain_thread is not None:
            runner._drain_thread.join(timeout=2.0)

    def test_enqueued_task_picked_up_without_poll_interval(self):
        runner, push = self._make_runner("/tmp/pw-enqueue")
        try:
            # Let the drain thread reach its wait() first.
            time.sleep(0.1)
            runner._queue.enqueue(_ev("A.py"), _dec(2, kind=ActionKind.NOTIFY))
            assert _wait_for(lambda: push.notification_count() >= 1), (
                "enqueued task must be executed promptly — the drain loop must "
                "not sleep out the full poll interval while work is pending"
            )
        finally:
            self._teardown_runner(runner)

    def test_slot_free_wake_runs_pending_task_immediately(self):
        runner, push = self._make_runner("/tmp/pw-slotfree")
        try:
            time.sleep(0.1)
            for i in range(3):
                runner._queue.enqueue(_ev(f"f{i}.py"), _dec(2, kind=ActionKind.NOTIFY))
            # Both concurrency slots fill; the 3rd task can only start after a
            # task_done wakes the drain thread. Old poll-only loop: 3rd
            # broadcast ~10s late → this 3s budget would fail.
            assert _wait_for(lambda: push.notification_count() >= 3), (
                "pending task must start as soon as a concurrency slot frees "
                "(task_done must wake the drain thread)"
            )
        finally:
            self._teardown_runner(runner)

    def test_stop_wakes_drain_thread_for_prompt_exit(self):
        runner, _push = self._make_runner("/tmp/pw-stop")
        time.sleep(0.1)  # drain thread is now blocked in wait(10.0)
        t0 = time.monotonic()
        runner.stop()
        elapsed = time.monotonic() - t0
        assert runner._drain_thread is not None
        runner._drain_thread.join(timeout=1.0)
        assert not runner._drain_thread.is_alive(), (
            "drain thread must exit promptly after stop() — stop must wake it "
            "instead of waiting out the poll interval"
        )
        assert elapsed < 1.0, (
            f"stop() must not block on the poll interval (took {elapsed:.2f}s)"
        )
