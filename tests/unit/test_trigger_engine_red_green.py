"""TriggerEngine RED→GREEN: lifecycle + scheduler race regression.

Race fixed in ``TriggerEngine.schedule``'s ``_fire``: the ``_running`` flag was
only checked *outside* the lock, so a ``stop()`` that ran during ``emit()``
(e.g. from a callback) could not stop an in-flight ``_fire`` from re-arming a
fresh timer into the already-cleared list. Consequences:
  * one schedule tick "survives" stop() and resurrects after a later start(),
  * dead timers leak in ``_schedule_timers`` (one per stop-during-tick).

Fix: re-check ``self._running`` inside the lock before re-arming.
"""

from __future__ import annotations

import queue
import threading
import time

from external_llm.editor.agent.autonomous.trigger_engine import (
    TriggerEngine,
    TriggerEvent,
    TriggerKind,
)


def _drain(q: queue.Queue[TriggerEvent]) -> list[TriggerEvent]:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class TestLifecycle:
    def test_start_is_idempotent(self):
        engine = TriggerEngine("/repo")
        engine.start()
        engine.start()  # second start is a no-op
        assert engine._running is True
        engine.stop()
        assert engine._running is False

    def test_stop_cancels_pending_timer(self):
        engine = TriggerEngine("/repo")
        engine.start()
        seen: list[TriggerEvent] = []
        engine.on_trigger(seen.append)
        engine.schedule(10.0, "never")  # far-future tick — must be cancelled
        engine.stop()
        time.sleep(0.2)
        assert seen == []
        assert engine._schedule_timers == []

    def test_schedule_while_stopped_is_inert(self):
        engine = TriggerEngine("/repo")
        seen: list[TriggerEvent] = []
        engine.on_trigger(seen.append)
        engine.schedule(0.02, "inert")  # engine never started
        time.sleep(0.3)
        assert seen == []
        engine.stop()
        assert engine._schedule_timers == []


class TestEmission:
    def test_fanout_preserves_fields(self):
        engine = TriggerEngine("/repo")
        got: list[TriggerEvent] = []
        engine.on_trigger(got.append)
        ev = TriggerEvent(
            kind=TriggerKind.FILE_MODIFIED,
            repo_root="/repo",
            source_file="a.py",
            severity=1,
        )
        engine.emit(ev)
        assert got == [ev]

    def test_callback_error_is_isolated(self):
        engine = TriggerEngine("/repo")
        got: list[TriggerEvent] = []

        def bad(event: TriggerEvent) -> None:
            raise RuntimeError("boom")

        engine.on_trigger(bad)
        engine.on_trigger(got.append)
        engine.emit(TriggerEvent(kind=TriggerKind.FILE_MODIFIED, repo_root="/repo"))
        assert len(got) == 1


class TestAgentAndTestHooks:
    def test_notify_agent_event_routing(self):
        engine = TriggerEngine("/repo")
        got: list[TriggerEvent] = []
        engine.on_trigger(got.append)

        engine.notify_agent_event("fail_loop_detected", {"n": 3})
        assert got[-1].kind is TriggerKind.AGENT_STALL
        assert got[-1].severity == 2
        assert got[-1].metadata == {"n": 3}

        engine.notify_agent_event("complete", {"status": "error"})
        assert got[-1].kind is TriggerKind.AGENT_FAILED
        engine.notify_agent_event("complete", {"status": "max_turns"})
        assert got[-1].kind is TriggerKind.AGENT_FAILED
        engine.notify_agent_event("complete", {"status": "ok"})
        assert got[-1].kind is TriggerKind.AGENT_COMPLETED
        assert got[-1].severity == 0

        n = len(got)
        engine.notify_agent_event("unknown_event", {})
        assert len(got) == n  # unknown event names are ignored

    def test_notify_test_result_routing(self):
        engine = TriggerEngine("/repo")
        got: list[TriggerEvent] = []
        engine.on_trigger(got.append)

        engine.notify_test_result(True, {"passed": 1})
        assert got[-1].kind is TriggerKind.TEST_RECOVERED
        assert got[-1].severity == 0
        assert got[-1].metadata == {"passed": 1}

        engine.notify_test_result(False, {"failed": 1})
        assert got[-1].kind is TriggerKind.TEST_FAILED
        assert got[-1].severity == 2


class TestScheduler:
    def test_periodic_ticks_then_stop(self):
        engine = TriggerEngine("/repo")
        engine.start()
        q: queue.Queue[TriggerEvent] = queue.Queue()
        engine.on_trigger(q.put)
        engine.schedule(0.02, "ticker")

        first = q.get(timeout=3.0)
        assert first.kind is TriggerKind.SCHEDULE
        assert first.repo_root == "/repo"
        assert first.severity == 0
        assert first.metadata == {"label": "ticker", "interval": 0.02}

        second = q.get(timeout=3.0)  # re-arm works
        assert second.kind is TriggerKind.SCHEDULE

        engine.stop()
        time.sleep(0.3)  # >> interval — no tick may survive stop()
        assert _drain(q) == []
        assert engine._schedule_timers == []

    # ── Regression: stop() racing an in-flight tick ─────────────────────────

    def test_stop_during_tick_does_not_leak_timer(self):
        """stop() called from inside emit (e.g. a callback) must not leave a
        re-armed timer behind in _schedule_timers."""
        engine = TriggerEngine("/repo")
        engine.start()
        q: queue.Queue[TriggerEvent] = queue.Queue()

        def cb(event: TriggerEvent) -> None:
            q.put(event)
            if event.kind is TriggerKind.SCHEDULE:
                engine.stop()  # runs inside emit() inside _fire()

        engine.on_trigger(cb)
        engine.schedule(0.02, "race")

        assert q.get(timeout=3.0).kind is TriggerKind.SCHEDULE
        time.sleep(0.3)  # >> interval — a leaked re-arm would fire (and die) here
        assert _drain(q) == [], "no event may arrive after stop()"
        assert engine._schedule_timers == [], "no timer may leak after stop()"

    def test_schedule_does_not_resurrect_after_restart(self):
        """A tick that raced with stop() must not have re-armed a timer that
        fires again after start() — the schedule stays dead until re-scheduled."""
        engine = TriggerEngine("/repo")
        engine.start()
        q: queue.Queue[TriggerEvent] = queue.Queue()

        def cb(event: TriggerEvent) -> None:
            q.put(event)
            if event.kind is TriggerKind.SCHEDULE:
                engine.stop()

        engine.on_trigger(cb)
        interval = 0.3
        engine.schedule(interval, "resurrect")

        first = q.get(timeout=3.0)
        assert first.kind is TriggerKind.SCHEDULE
        engine.start()  # restart while a raced timer would still be pending

        time.sleep(interval + 0.6)  # >> one full leaked interval
        assert _drain(q) == [], "schedule must not resurrect on restart"
        assert engine._schedule_timers == []

    def test_restart_between_stop_and_rearm_does_not_resurrect(self, monkeypatch):
        """A start() that races the in-flight tick's re-arm (stop() already ran
        from the callback) must not let that tick re-arm into the restarted
        engine — the schedule stays dead until re-scheduled.

        Deterministic interleaving: park the timer thread right after emit()
        (i.e. after the callback's stop()), restart from the main thread, then
        release the tick. Without the epoch guard the tick sees _running True
        again and re-arms a fresh timer that fires after the restart.
        """
        engine = TriggerEngine("/repo")
        engine.start()
        q: queue.Queue[TriggerEvent] = queue.Queue()

        orig_emit = TriggerEngine.emit
        gate = threading.Event()
        emit_returned = threading.Event()

        def slow_emit(self, ev):
            try:
                return orig_emit(self, ev)
            finally:
                emit_returned.set()  # emit (incl. callback stop()) finished
                gate.wait(5)  # hold the timer thread before its re-arm check

        monkeypatch.setattr(TriggerEngine, "emit", slow_emit)

        def cb(event: TriggerEvent) -> None:
            q.put(event)
            if event.kind is TriggerKind.SCHEDULE:
                engine.stop()

        engine.on_trigger(cb)
        engine.schedule(0.1, "resurrect")

        try:
            first = q.get(timeout=3.0)
            assert first.kind is TriggerKind.SCHEDULE
            assert emit_returned.wait(3.0)  # timer thread parked before re-arm
            engine.start()  # restart before the re-arm check
            gate.set()  # release the in-flight tick
            time.sleep(0.5)  # a resurrected 0.1s timer would fire here
            assert _drain(q) == [], "schedule must not resurrect on restart"
            assert engine._schedule_timers == []
        finally:
            gate.set()
            engine.stop()
