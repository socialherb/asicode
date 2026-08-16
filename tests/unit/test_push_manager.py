"""Tests for PushManager SSE push→generator cycle.

Regression: the async make_sse_generator must consume the 2-tuple that
push()/broadcast() enqueue, and producers must wake the async consumer promptly.

make_sse_generator is ASYNC (cancellation-aware disconnect handling), so we
drive it via asyncio.run + async for / aclose.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time

import pytest

from external_llm.editor.agent.autonomous.push_manager import PushManager, get_push_manager

# ── Helpers ──────────────────────────────────────────────────────────────

async def _consume_events_async(gen, count: int, timeout: float = 5.0) -> list[dict]:
    """Read *count* SSE events from an async generator, return parsed dicts."""
    events: list[dict] = []
    deadline = time.monotonic() + timeout
    while len(events) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out after {len(events)}/{count} events")
        try:
            chunk = await asyncio.wait_for(gen.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            break
        # SSE format: "event: {type}\ndata: {json}\n\n"
        ev: dict[str, str] = {}
        for line in chunk.strip().split("\n"):
            if line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                ev["data_json"] = line[6:]
        if "event" in ev and "data_json" in ev:
            events.append(ev)
    return events


# NOTE: there is intentionally NO sync ``_consume_events`` wrapper. An async
# generator's suspended state is bound to the event loop it started on, so a
# generator CANNOT be consumed across multiple asyncio.run() calls (each run
# uses a fresh loop). Every test below drives the generator's full lifecycle
# inside ONE asyncio.run coroutine.


# ── Tests ────────────────────────────────────────────────────────────────

class TestPushManagerRoundtrip:
    """push → SSE generator roundtrip.

    Uses a dedicated PushManager instance per test to avoid cross-test pollution.

    Every test drives the generator's FULL lifecycle inside ONE asyncio.run
    coroutine — an async generator's suspended state is bound to the loop it
    started on, so it cannot be consumed across multiple asyncio.run() calls.
    """

    @pytest.fixture(autouse=True)
    def _fresh_manager(self) -> None:
        self.pm = PushManager()

    def test_push_roundtrip(self) -> None:
        """Push a 2-tuple event, verify generator yields valid SSE without crash."""
        pm = self.pm
        client_id = "test_client_1"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                # Consume the handshake event
                handshake = await _consume_events_async(gen, 1, timeout=2.0)
                assert len(handshake) == 1
                assert handshake[0]["event"] == "proactive_connected"

                # Push a real event (2-tuple, as broadcast/push do)
                sent = pm.push(client_id, "test_event", {"msg": "hello"})
                assert sent is True

                # Consume one more event — the 2-tuple unpack must succeed.
                payload = await _consume_events_async(gen, 1, timeout=2.0)
                assert len(payload) == 1
                assert payload[0]["event"] == "test_event"
                assert json.loads(payload[0]["data_json"]) == {"msg": "hello"}
            finally:
                await gen.aclose()

        asyncio.run(_go())

    def test_broadcast_roundtrip(self) -> None:
        """broadcast() (also 2-tuple) followed by generator read."""
        pm = self.pm
        client_id = "test_client_bc"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                await _consume_events_async(gen, 1, timeout=2.0)  # handshake
                count = pm.broadcast("bc_event", {"n": 42})
                assert count == 1
                payload = await _consume_events_async(gen, 1, timeout=2.0)
                assert len(payload) == 1
                assert payload[0]["event"] == "bc_event"
                assert json.loads(payload[0]["data_json"]) == {"n": 42}
            finally:
                await gen.aclose()

        asyncio.run(_go())

    def test_push_after_close_does_not_crash_generator(self) -> None:
        """push() after generator close is benign (unregister already done)."""
        pm = self.pm
        client_id = "test_client_close"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            await _consume_events_async(gen, 1, timeout=2.0)
            await gen.aclose()

        asyncio.run(_go())

        # Push to a now-unregistered client — should return False, not crash
        result = pm.push(client_id, "after_close", {"x": 1})
        assert result is False

    def test_producer_wakes_parked_consumer_promptly(self) -> None:
        """F1/F2 regression: a producer (broadcast) running in a background thread
        must wake an async consumer that is parked on the wake event, without
        waiting for the 15s keepalive timeout. Verifies the cross-thread wakeup
        (loop.call_soon_threadsafe) wiring added when make_sse_generator went async.
        """
        pm = self.pm
        client_id = "test_wake"
        payload = {"ts": 123}
        delivered = threading.Event()
        timing = {"elapsed": None}

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                await _consume_events_async(gen, 1, timeout=2.0)  # handshake

                def producer():
                    time.sleep(0.15)  # ensure consumer is parked on wake event
                    pm.broadcast("wake_test", payload)
                    delivered.set()

                threading.Thread(target=producer, daemon=True).start()
                t0 = time.monotonic()
                events = await _consume_events_async(gen, 1, timeout=3.0)
                timing["elapsed"] = time.monotonic() - t0
                return events
            finally:
                await gen.aclose()

        events = asyncio.run(_go())

        assert delivered.wait(timeout=3.0)
        assert len(events) == 1
        assert events[0]["event"] == "wake_test"
        assert json.loads(events[0]["data_json"]) == payload
        # Must be woken well before the 15s keepalive timeout.
        assert timing["elapsed"] < 5.0, f"consumer took {timing['elapsed']:.2f}s — wakeup wiring broken"

    def test_shutdown_all_delivers_sentinel_when_queue_full(self) -> None:
        """shutdown_all must deliver the None sentinel even when a client's
        queue is full: it drops the oldest pending item to make room (mirrors
        SequencedEventQueue's drop-oldest). Without this the sentinel was
        silently dropped on queue.Full and make_sse_generator never exited on
        graceful teardown."""
        import queue as _q
        pm = self.pm
        client_id = "full-client"
        q = pm.register(client_id)
        # Fill the queue to capacity
        filled = 0
        while True:
            try:
                q.put_nowait(("noop", {"i": filled}))
                filled += 1
            except _q.Full:
                break
        assert q.full()

        pm.shutdown_all()

        # The sentinel must be present despite the queue having been full.
        seen_none = False
        drained = 0
        while True:
            try:
                item = q.get_nowait()
            except _q.Empty:
                break
            drained += 1
            if item is None:
                seen_none = True
        assert seen_none, "shutdown sentinel lost on full queue"
        # One oldest item was evicted to make room for the sentinel, so the
        # total drained equals the original fill count (fill - 1 + sentinel).
        assert drained == filled

    def test_cleanup_stale_keeps_live_idle_generator(self) -> None:
        """A connected-but-idle client (generator running, no events for far
        longer than CLIENT_TTL) must NOT be pruned: last_active is refreshed
        only on event delivery, so liveness comes from the wake_loop binding —
        pruning on last_active alone would silently cut a healthy tab off from
        future broadcasts."""
        pm = self.pm
        client_id = "idle-client"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                await _consume_events_async(gen, 1, timeout=2.0)  # handshake
                # Simulate a long idle stretch: no events delivered, so
                # last_active is stale by far more than CLIENT_TTL.
                with pm._lock:
                    pm._clients[client_id]["last_active"] = time.time() - pm.CLIENT_TTL - 60
                assert pm.cleanup_stale() == 0
                assert client_id in pm._clients
                # Still connected: events must still arrive after the sweep.
                assert pm.push(client_id, "still_alive", {"x": 1}) is True
                payload = await _consume_events_async(gen, 1, timeout=2.0)
                assert len(payload) == 1
                assert payload[0]["event"] == "still_alive"
            finally:
                await gen.aclose()

        asyncio.run(_go())

    def test_cleanup_stale_prunes_abandoned_registration(self) -> None:
        """A client that registered but whose generator never started (no
        wake_loop binding) is pruned once last_active is past CLIENT_TTL."""
        pm = self.pm
        client_id = "abandoned-client"
        q = pm.register(client_id)
        with pm._lock:
            pm._clients[client_id]["last_active"] = time.time() - pm.CLIENT_TTL - 60
        assert pm.cleanup_stale() == 1
        assert client_id not in pm._clients
        # Registration is gone: pushes to the pruned id return False.
        assert pm.push(client_id, "noop", {}) is False
        # The orphaned queue object is inert — nothing else references it.
        assert q is not None

    # ── RED→GREEN: uncovered branches ────────────────────────────────────

    def test_broadcast_drops_event_on_full_queue(self) -> None:
        """broadcast() must not raise when a client queue is full; the event
        is dropped for that client and it is not counted (L151-152)."""
        import queue as _q
        pm = self.pm
        client_id = "full-broadcast"
        q = pm.register(client_id)
        while True:
            try:
                q.put_nowait(("noop", {"i": 0}))
            except _q.Full:
                break
        assert q.full()
        assert pm.broadcast("dropped", {"x": 1}) == 0

    def test_push_returns_false_on_full_queue(self) -> None:
        """push() returns False when the client queue is full (L176-177)."""
        import queue as _q
        pm = self.pm
        client_id = "full-push"
        q = pm.register(client_id)
        while True:
            try:
                q.put_nowait(("noop", {"i": 0}))
            except _q.Full:
                break
        assert q.full()
        assert pm.push(client_id, "dropped", {"x": 1}) is False

    def test_generator_keepalive_and_spurious_wake(self, monkeypatch) -> None:
        """The slow path covers both branches of the wake wait: a wake with an
        empty queue loops back silently (spurious), and the 15s timeout yields
        ': keepalive' (L257-269)."""
        calls = {"n": 0}

        async def _fake_wait_for(awaitable, timeout=None):
            calls["n"] += 1
            # The generator parks on a Task (P-E fix), so close() doesn't
            # apply — cancel it and await the CancelledError instead of
            # leaving it suspended (RuntimeWarning: never awaited).
            awaitable.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await awaitable
            if calls["n"] == 1:
                return  # simulate wake with nothing enqueued → spurious
            raise asyncio.TimeoutError()

        monkeypatch.setattr(
            "external_llm.editor.agent.autonomous.push_manager.asyncio.wait_for",
            _fake_wait_for,
        )
        pm = self.pm
        client_id = "keepalive-client"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                first = await gen.__anext__()
                assert "proactive_connected" in first
                # __anext__ #2: spurious wake (wait_for #1 returns) → loop →
                # keepalive (wait_for #2 times out)
                second = await gen.__anext__()
                assert second == ": keepalive\n\n"
                # __anext__ #3: resumes after the keepalive yield, running the
                # `continue` and timing out again (L259).
                third = await gen.__anext__()
                assert third == ": keepalive\n\n"
            finally:
                await gen.aclose()

        asyncio.run(_go())
        assert calls["n"] == 3

    def test_cancel_while_parked_does_not_strand_wake_coroutine(self) -> None:
        """Regression (P-E): cancelling the consumer while the generator parks
        on wake_event.wait() must not strand the park awaitable.

        asyncio.wait_for (3.12+, timeouts-based) does not cancel its inner
        awaitable on cancellation, so the park must be a Task that the
        generator cancels explicitly before propagating. A bare coroutine
        would stay suspended inside Event.wait()'s waiter list and warn
        "coroutine 'Event.wait' was never awaited" at GC.
        """
        import gc
        import warnings

        async def _go():
            gen = self.pm.make_sse_generator("park-close")
            first = await gen.__anext__()
            assert "proactive_connected" in first
            # Second __anext__ parks on the wake event (empty queue).
            waiter = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.05)  # let the waiter reach the parked await
            waiter.cancel()  # client disconnect: CancelledError at the park
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await waiter
            await asyncio.sleep(0.05)  # let the cancelled park task settle

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(_go())
            gc.collect()

        leaked = [str(w.message) for w in caught if "never awaited" in str(w.message)]
        assert leaked == [], f"leaked coroutines: {leaked}"

    def test_generator_exits_on_shutdown_sentinel(self) -> None:
        """With a non-full queue, shutdown_all's sentinel ends the generator
        and the finally block unregisters the client (L272)."""
        pm = self.pm
        client_id = "shutdown-client"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                first = await gen.__anext__()
                assert "proactive_connected" in first
                pm.shutdown_all()
                with pytest.raises(StopAsyncIteration):
                    await gen.__anext__()
                assert client_id not in pm._clients
            finally:
                await gen.aclose()

        asyncio.run(_go())

    def test_generator_serialization_error_fallback(self) -> None:
        """Un-serializable data (circular reference) falls back to a
        serialization_error event instead of crashing the generator
        (L277-278)."""
        pm = self.pm
        client_id = "serde-client"

        async def _go():
            gen = pm.make_sse_generator(client_id)
            try:
                first = await gen.__anext__()
                assert "proactive_connected" in first
                circular: dict = {}
                circular["self"] = circular
                assert pm.push(client_id, "circular", circular) is True
                ev = await gen.__anext__()
                assert 'data: {"error": "serialization_error"}' in ev
            finally:
                await gen.aclose()

        asyncio.run(_go())



class TestPushManagerSingleton:
    """get_push_manager singleton contract."""

    def test_identical_instances(self) -> None:
        assert get_push_manager() is get_push_manager()




