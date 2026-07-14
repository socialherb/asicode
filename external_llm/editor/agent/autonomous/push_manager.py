"""
PushManager — SSE-based server→client push for proactive events.

Architecture:
  - Each connected browser tab registers as a "client" with a unique ID
  - ProactiveRunner calls broadcast() from any background thread
  - /agent/proactive/stream SSE generator reads from per-client threading.Queue
  - make_sse_generator is ASYNC (cancellation-aware): a client disconnect
    propagates to its `finally` immediately instead of being blocked for up to
    the 15s keepalive timeout inside a worker thread's queue.get()

Thread safety:
  - register / unregister / broadcast / push are all thread-safe
  - broadcast/push wake the async SSE consumer via loop.call_soon_threadsafe

Client lifecycle:
  - Browser connects to /agent/proactive/stream → register() → gets a Queue
  - Generator runs while browser is connected, yielding SSE events
  - Browser disconnects → generator's finally block → unregister()
  - Stale clients (no activity for CLIENT_TTL seconds) can be purged by cleanup_stale()

Known limitation (deferred):
  Unlike the agent SSE path (SequencedEventQueue + ring buffer + Last-Event-ID
  resume via /agent/attach), the proactive stream has NO reconnection replay: a
  page reload or transient network blip drops in-flight proactive events with no
  resync. A future enhancement would mirror the agent lane (per-client ring +
  ``id:`` lines + a resume endpoint) so proactive insights survive reloads. Not
  built yet because proactive events are ephemeral UI hints (not structural run
  state) and the agent lane already gates anything load-bearing behind
  /agent/pending reconstruction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from external_llm.agent.config.thresholds import config as _cfg

logger = logging.getLogger(__name__)

# Module-level singleton
_push_manager: Optional["PushManager"] = None
_push_manager_lock = threading.Lock()


def get_push_manager() -> "PushManager":
    """Return the global PushManager singleton, creating it if needed."""
    global _push_manager
    with _push_manager_lock:
        if _push_manager is None:
            _push_manager = PushManager()
        return _push_manager


class PushManager:
    """
    Thread-safe registry of SSE client queues.

    Usage in FastAPI endpoint:
        pm = get_push_manager()
        client_id = str(uuid.uuid4())

        return StreamingResponse(
            pm.make_sse_generator(client_id),  # async generator
            media_type="text/event-stream",
        )
    """

    CLIENT_QUEUE_SIZE = _cfg.counts.PUSH_CLIENT_QUEUE_SIZE
    CLIENT_TTL = 3600          # seconds of silence before a client is considered stale

    def __init__(self):
        self._lock = threading.Lock()
        # client_id → {"queue": Queue, "connected_at": float, "last_active": float}
        self._clients: dict[str, dict[str, Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, client_id: str) -> queue.Queue:
        """Register new SSE client. Returns its dedicated event queue."""
        q: queue.Queue = queue.Queue(maxsize=self.CLIENT_QUEUE_SIZE)
        now = time.time()
        with self._lock:
            self._clients[client_id] = {
                "queue": q,
                "connected_at": now,
                "last_active": now,
                # Async wakeup: set by broadcast/push (cross-thread) so the async
                # SSE generator's aget-style wait returns immediately on new work,
                # instead of polling every 15s. The running loop + Event are bound
                # lazily by make_sse_generator on first await.
                "wake_loop": None,
                "wake_event": None,
            }
        logger.info("PushManager: client registered %s (total: %d)", client_id, len(self._clients))
        return q

    def _wake_client(self, info: dict) -> None:
        """Wake a client's async SSE consumer if one is parked (cross-thread safe)."""
        loop = info.get("wake_loop")
        event = info.get("wake_event")
        if loop is None or event is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            # loop closed (client gone) — harmless; generator prunes on exit.
            pass

    def unregister(self, client_id: str) -> None:
        """Unregister client (called when SSE connection closes)."""
        with self._lock:
            self._clients.pop(client_id, None)
        logger.info("PushManager: client unregistered %s (total: %d)", client_id, len(self._clients))

    # ── Push ──────────────────────────────────────────────────────────────────

    def broadcast(self, event_type: str, data: dict[str, Any]) -> int:
        """
        Push event to ALL connected clients.
        Returns number of clients that received the event.
        Silently drops events for full queues.

        Thread-safe: self._lock protects _clients snapshots and per-client
        last_active updates from concurrent register/unregister/cleanup_stale.
        """
        with self._lock:
            clients = list(self._clients.items())

        # Snapshot once; last_active is read under lock by cleanup_stale, but
        # CLIENT_TTL (3600s) makes sub-second skew irrelevant, so a single `now`
        # suffices for all clients instead of a fresh time.time() each.
        now = time.time()
        delivered: list = []
        sent = 0
        for client_id, info in clients:
            q: queue.Queue = info["queue"]
            try:
                q.put_nowait((event_type, data))
                delivered.append(info)
                sent += 1
                self._wake_client(info)
            except queue.Full:
                logger.warning("PushManager: queue full for client %s — dropping %s", client_id, event_type)

        # Batch the last_active refresh into ONE locked pass (was N acquisitions,
        # one per client). info is a reference into self._clients; updating it
        # under lock honors the same contract as register/unregister/cleanup_stale.
        if delivered:
            with self._lock:
                for info in delivered:
                    info["last_active"] = now

        if sent > 0:
            logger.debug("PushManager: broadcast '%s' to %d client(s)", event_type, sent)
        return sent

    def push(self, client_id: str, event_type: str, data: dict[str, Any]) -> bool:
        """Push event to a specific client only. Returns True if delivered."""
        with self._lock:
            info = self._clients.get(client_id)
            if not info:
                return False
            try:
                info["queue"].put_nowait((event_type, data))
                info["last_active"] = time.time()
                delivered = True
            except queue.Full:
                delivered = False
        if delivered:
            self._wake_client(info)
        return delivered

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def cleanup_stale(self) -> int:
        """Remove clients with no activity for > CLIENT_TTL seconds."""
        now = time.time()
        with self._lock:
            stale = [
                cid for cid, info in self._clients.items()
                if now - info.get("last_active", now) > self.CLIENT_TTL
            ]
            for cid in stale:
                del self._clients[cid]
        if stale:
            logger.info("PushManager: pruned %d stale client(s)", len(stale))
        return len(stale)

    # ── SSE generator ─────────────────────────────────────────────────────────

    async def make_sse_generator(self, client_id: str) -> AsyncIterator[str]:
        """
        SSE generator for a registered client.

        Yields SSE-formatted strings. Sends keepalive comment every 15 s.
        Automatically unregisters the client when the generator exits (disconnect).

        ASYNC: being async is load-bearing for disconnect detection. The old sync
        generator blocked inside ``q.get(timeout=15)`` on a worker thread and
        could not observe a client disconnect until the timeout — so a closed tab
        kept a stale client entry alive for up to 15s and the ``finally``/unregister
        was delayed. As an async generator consumed via ``async for``, a disconnect
        cancels the running task and runs ``finally`` immediately. Producers
        (broadcast/push) wake us via ``loop.call_soon_threadsafe`` on new events.
        """
        q = self.register(client_id)
        # Bind the running loop + an asyncio.Event into the client entry so
        # cross-thread producers can wake us. We set them under the lock to pair
        # with _wake_client's read.
        loop = asyncio.get_running_loop()
        wake_event = asyncio.Event()
        with self._lock:
            info = self._clients.get(client_id)
            if info is not None:
                info["wake_loop"] = loop
                info["wake_event"] = wake_event
        try:
            # Initial handshake event
            yield f"event: proactive_connected\ndata: {json.dumps({'client_id': client_id})}\n\n"

            while True:
                # Fast path
                try:
                    item: Optional[tuple] = q.get_nowait()
                except queue.Empty:
                    # Slow path: park on the wake event with a keepalive timeout.
                    wake_event.clear()
                    # Re-check after clear to avoid lost-wakeup (producer enqueued
                    # between our get_nowait and the clear).
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        try:
                            await asyncio.wait_for(wake_event.wait(), timeout=15.0)
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                            continue
                        # Woken — drain whatever landed
                        try:
                            item = q.get_nowait()
                        except queue.Empty:
                            continue  # spurious; loop again
                if item is None:
                    # Explicit shutdown sentinel
                    break

                event_type, data = item
                try:
                    data_json = json.dumps(data, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    data_json = json.dumps({"error": "serialization_error"})
                yield f"event: {event_type}\ndata: {data_json}\n\n"

        finally:
            # Clear the wakeup binding so producers stop trying to wake a dead loop.
            with self._lock:
                info = self._clients.get(client_id)
                if info is not None:
                    info["wake_loop"] = None
                    info["wake_event"] = None
            self.unregister(client_id)

    def shutdown_all(self) -> None:
        """Send shutdown sentinel to all clients (graceful teardown).

        The sentinel is delivered even when a client's queue is full: we evict
        the oldest pending item to make room (mirrors ``SequencedEventQueue``'s
        ``_put_bounded``). Without this, a full queue silently dropped the
        sentinel (``queue.Full`` → ``pass``) and ``make_sse_generator`` never
        saw ``None`` — so the generator stayed alive until the socket died
        instead of exiting on graceful teardown.
        """
        with self._lock:
            clients = list(self._clients.items())
        for _, info in clients:
            q: queue.Queue = info["queue"]
            try:
                q.put_nowait(None)
            except queue.Full:
                # Drop the oldest pending item to guarantee the sentinel lands.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass  # lost the freed slot to a concurrent producer; best-effort
            self._wake_client(info)
