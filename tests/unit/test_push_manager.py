"""Tests for PushManager SSE push→generator cycle.

Regression: the async make_sse_generator must consume the 2-tuple that
push()/broadcast() enqueue, and producers must wake the async consumer promptly.

make_sse_generator is ASYNC (cancellation-aware disconnect handling), so we
drive it via asyncio.run + async for / aclose.
"""
from __future__ import annotations

import asyncio
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



class TestPushManagerSingleton:
    """get_push_manager singleton contract."""

    def test_identical_instances(self) -> None:
        assert get_push_manager() is get_push_manager()




