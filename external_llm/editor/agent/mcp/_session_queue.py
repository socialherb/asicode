"""
Shared session plumbing for the MCP HTTP transports (SSE + Streamable-HTTP).

Both transports keep per-session bounded ``queue.Queue`` state guarded by a
single lock, plus a background idle-sweep reaper for clients that vanish
without DELETE.  ``_SessionQueueMixin`` owns that plumbing — registration,
lookup, teardown, enqueue-under-race, sweep, shutdown — and the JSON-RPC
handler dispatch, so the two transports cannot drift: the teardown contract
(pop-before-sentinel; enqueue serialized with pop under the same lock) lives
in exactly one place.

Subclasses provide the transport-specific HTTP handler and must set
``registry`` and ``_handle`` (and bind ``httpd``) before calling
:meth:`_SessionQueueMixin._init_session_plumbing`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Optional

from external_llm.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Upper bound on POST bodies — a malformed/malicious client must not be able
# to balloon the server's memory via an unbounded rfile.read().
_MAX_MESSAGE_BODY_BYTES = 1_048_576  # 1 MiB

# Per-session SSE backlog cap — a client that stops reading its stream must not
# be able to balloon server memory via repeated POSTs (every response carrying
# an id is queued on the session's SSE stream).
_MAX_QUEUED_EVENTS = 1024

# Idle-session reclamation: a session (SSE stream) untouched for this long is
# closed by the background sweep.  Covers clients that vanish without DELETE
# (crash/network drop) — their stream thread would otherwise block on the
# message queue forever.
_SESSION_IDLE_TTL_SECONDS = 30 * 60
_SESSION_SWEEP_INTERVAL_SECONDS = 60

# Type of the injected JSON-RPC handler: (registry, request) -> response dict.
JsonRpcHandler = Callable[[ToolRegistry, dict[str, Any]], dict[str, Any]]


class _SessionQueueMixin:
    """Shared session-queue + lifecycle plumbing for the MCP HTTP transports.

    Requires ``registry`` and ``_handle`` (set by the subclass ``__init__``)
    and ``httpd`` (bound before :meth:`_init_session_plumbing` is called).
    """

    # -- setup ----------------------------------------------------------------

    def _init_session_plumbing(
        self,
        *,
        thread_name: str,
        session_idle_ttl: float = _SESSION_IDLE_TTL_SECONDS,
        sweep_interval: float = _SESSION_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        """Create the session state and start the idle-sweep reaper thread.

        Call at the END of the subclass ``__init__`` (after ``httpd`` is
        bound) so a construction failure cannot leak the daemon thread.
        """
        self._sessions: dict[str, queue.Queue[Optional[str]]] = {}
        self._last_active: dict[str, float] = {}
        self._lock = threading.Lock()
        self._session_idle_ttl = session_idle_ttl
        self._sweep_interval = sweep_interval
        self._sweep_stop = threading.Event()
        # Background reaper for sessions whose client vanished without DELETE.
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop, name=thread_name, daemon=True
        )
        self._sweep_thread.start()

    # -- lifecycle ------------------------------------------------------------

    def serve_forever(self) -> None:
        """Serve until :meth:`shutdown` is called (blocking)."""
        self.httpd.serve_forever(poll_interval=0.5)

    def shutdown(self) -> None:
        """Stop serving, release the listening socket, and close all sessions."""
        self._sweep_stop.set()
        # Wake every stream thread (sentinel) so it can exit — a thread blocked
        # in messages.get() must not outlive the server.
        with self._lock:
            for messages in self._sessions.values():
                try:
                    messages.put_nowait(None)
                except queue.Full:
                    logger.debug("session backlog full at shutdown — dropped")
            self._sessions.clear()
            self._last_active.clear()
        self.httpd.shutdown()
        self.httpd.server_close()

    # -- session plumbing -----------------------------------------------------

    def _new_session(self) -> tuple[str, queue.Queue[Optional[str]]]:
        """Register a fresh session; returns (session_id, message queue)."""
        session_id = uuid.uuid4().hex
        messages: queue.Queue[Optional[str]] = queue.Queue(maxsize=_MAX_QUEUED_EVENTS)
        now = time.monotonic()
        with self._lock:
            self._sessions[session_id] = messages
            self._last_active[session_id] = now
        return session_id, messages

    def _get_session(self, session_id: str) -> Optional[queue.Queue[Optional[str]]]:
        with self._lock:
            messages = self._sessions.get(session_id)
            if messages is not None:
                self._last_active[session_id] = time.monotonic()
            return messages

    def _drop_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_active.pop(session_id, None)

    def _close_session(self, session_id: str) -> None:
        """Tear down a session: pop it, then signal its stream thread to end.

        The pop happens under the lock, before the sentinel, so an in-flight
        POST's enqueue check cannot interleave — the sentinel never lands
        behind a payload on a session nobody reads (see
        :meth:`_enqueue_if_current`).  A full backlog means the consumer is
        gone, but the dict entry is removed either way (later POSTs answer
        404).  Idempotent.
        """
        with self._lock:
            messages = self._sessions.pop(session_id, None)
            self._last_active.pop(session_id, None)
        if messages is not None:
            try:
                messages.put_nowait(None)  # sentinel closes the stream
            except queue.Full:
                logger.debug("MCP session %s backlog full — dropped", session_id[:8])

    def _enqueue_if_current(
        self,
        session_id: str,
        session: queue.Queue[Optional[str]],
        payload: str,
    ) -> str:
        """Queue an SSE payload iff ``session`` is still the registered session.

        The single enqueue path for POSTs racing session teardown: every drop
        (DELETE, idle sweep, shutdown, stream end) removes the session under
        this lock, so the payload is either queued before the sentinel that
        closes the stream (delivered) or the session is already gone and the
        caller answers 409 — never a silent 202 whose response nobody reads.

        Returns "ok" | "gone" | "full".
        """
        with self._lock:
            if self._sessions.get(session_id) is not session:
                return "gone"
            try:
                session.put_nowait(payload)
            except queue.Full:
                return "full"
            else:
                return "ok"

    def _sweep_loop(self) -> None:
        """Background reaper (daemon thread): close sessions idle past the TTL."""
        while not self._sweep_stop.wait(self._sweep_interval):
            self.sweep_idle_sessions()

    def sweep_idle_sessions(self) -> None:
        """Close and drop sessions idle past the TTL (idempotent, thread-safe)."""
        now = time.monotonic()
        with self._lock:
            idle = [
                sid
                for sid, last in self._last_active.items()
                if now - last >= self._session_idle_ttl
            ]
        for sid in idle:
            self._close_session(sid)

    # -- JSON-RPC -------------------------------------------------------------

    def _handle_request(self, request: dict) -> Optional[str]:
        """Run the JSON-RPC handler; returns the response payload, None for notifications."""
        request_id = request.get("id")
        if request_id is None:
            return None  # notification — the 202 response is the only ack
        if self._handle is None:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": "no JSON-RPC handler configured"},
            })
        return json.dumps(self._handle(self.registry, request))
