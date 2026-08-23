"""
MCP Streamable-HTTP transport server — standard library only.

Implements the MCP Streamable-HTTP transport (protocol revision 2025-11-05)
with stdlib ``http.server``, mirroring ``sse_server.py`` (the 2025-03-26
HTTP+SSE transport) so the published core package keeps its zero web-framework
dependency (fastapi/uvicorn live in the optional ``[webapp]`` extra).

Endpoints
---------
POST /mcp
    Single endpoint for the whole protocol (Streamable-HTTP replaces the
    GET /sse + POST /message pair of the older HTTP+SSE transport).

    * ``Accept: application/json`` (no ``text/event-stream``) — JSON mode:
      a request carrying an ``id`` is answered 200 with the JSON-RPC response
      in the body; notifications (no ``id``) are answered 202.
    * ``Accept`` including ``text/event-stream`` — SSE mode: the response is a
      ``200 text/event-stream`` that stays open.  The session id is returned in
      the ``Mcp-Session-Id`` header; the first request's response is delivered
      as the first ``message`` event, and every later POST carrying that
      session id is acknowledged 202 with its response delivered on the open
      stream.  A client that never opens an SSE stream can still use JSON mode
      against the same session id.  A later POST whose session is torn down
      (DELETE / disconnect / idle sweep) while the request is being handled
      answers ``409`` — the tool may already have executed and the client must
      decide whether to retry.

GET /mcp?session_id=<id>
    Re-attach to an existing session's SSE stream (resume path used by some
    clients after a reconnect).

DELETE /mcp?session_id=<id>
    Client-initiated teardown: closes the session's SSE stream.

Threading model: one thread per SSE connection (``ThreadingHTTPServer``,
daemon threads — the process exits cleanly even with open streams).  Session
state is a per-session bounded ``queue.Queue`` guarded by the server's lock —
the same plumbing as the SSE transport, shared via ``_SessionQueueMixin``
(``_session_queue``).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from external_llm.agent.tool_registry import ToolRegistry
from external_llm.editor.agent.mcp._session_queue import (
    _MAX_MESSAGE_BODY_BYTES,
    _SESSION_IDLE_TTL_SECONDS,
    _SESSION_SWEEP_INTERVAL_SECONDS,
    JsonRpcHandler,
    QuietHttpHandler,
    _SessionQueueMixin,
)

logger = logging.getLogger(__name__)

# Transport-wide constants shared with the SSE transport (body-size limit,
# queue cap, idle TTL, JsonRpcHandler type) live in _session_queue; the
# Streamable-HTTP-only limits below stay here.

# Socket timeout for client connections — a stalled client must not pin a
# worker thread forever (ThreadingHTTPServer spawns one thread per connection).
_SOCKET_TIMEOUT_SECONDS = 30

# SSE keep-alive heartbeat interval.  The stream loop blocks on the session
# queue; without periodic writes it can never detect a vanished client (a RST
# only fails the *next* write), so a dead client's session would linger until
# the idle sweep (30 min).  Writing an SSE comment (a no-op event) on a timer
# turns a vanished client into a BrokenPipeError within one interval, and
# touches the session so an *active* client is never swept.
_SSE_HEARTBEAT_INTERVAL_SECONDS = 30.0

# Global cap on concurrent id'd JSON-RPC requests (tool calls) — mirrors the
# stdio transport's BoundedSemaphore(8).  Overflow is answered 503.
_MAX_CONCURRENT_REQUESTS = 8


class StreamableHttpMcpServer(_SessionQueueMixin):
    """Threaded HTTP server exposing asicode tools over MCP Streamable-HTTP.

    Session-queue plumbing (registration, teardown, enqueue-under-race, idle
    sweep, shutdown) is shared with the SSE transport via
    ``_SessionQueueMixin`` (see ``_session_queue``).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        host: str = "127.0.0.1",
        port: int = 8766,
        *,
        handle: JsonRpcHandler | None = None,
        max_concurrent: int = _MAX_CONCURRENT_REQUESTS,
        session_idle_ttl: float = _SESSION_IDLE_TTL_SECONDS,
        sweep_interval: float = _SESSION_SWEEP_INTERVAL_SECONDS,
        heartbeat_interval: float = _SSE_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.registry = registry
        # JSON-RPC handler — injected by server.py's _run_streamable_server so
        # the stdio/SSE/Streamable-HTTP transports share ONE implementation;
        # tests inject fakes.  None (direct construction) degrades to an
        # explicit error.
        self._handle = handle
        self._heartbeat_interval = heartbeat_interval
        self._concurrency_semaphore = threading.BoundedSemaphore(max_concurrent)
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self.host, self.port = self.httpd.server_address[:2]
        logger.info("MCP Streamable-HTTP server bound to %s:%s", self.host, self.port)
        # Session state + idle-sweep reaper (must run after httpd is bound so
        # a construction failure cannot leak the daemon thread).
        self._init_session_plumbing(
            thread_name="mcp-streamable-sweep",
            session_idle_ttl=session_idle_ttl,
            sweep_interval=sweep_interval,
        )

    def _touch(self, session_id: str) -> None:
        with self._lock:
            self._last_active[session_id] = time.monotonic()

    def _handle_request_capped(self, request: dict) -> tuple[bool, str | None]:
        """Handle an id'd request under the global concurrency cap.

        Returns ``(True, payload)`` on success (``payload`` is None for
        notifications, which bypass the cap) and ``(False, None)`` when the
        cap is reached — the caller answers 503.
        """
        if request.get("id") is None:
            return True, None
        if not self._concurrency_semaphore.acquire(blocking=False):
            return False, None
        try:
            return True, self._handle_request(request)
        finally:
            self._concurrency_semaphore.release()


def _make_handler(server: StreamableHttpMcpServer) -> type[BaseHTTPRequestHandler]:
    """Build the request-handler class bound to ``server`` (closure, no globals)."""

    class _StreamableHandler(QuietHttpHandler):
        server_version = "asicode-mcp-streamable/1.0"
        protocol_version = "HTTP/1.1"
        # Socket timeout — a stalled client must not pin a worker thread
        # forever (applies to body reads; SSE streams never read after the
        # response starts, and a slow consumer then fails fast instead).
        timeout = _SOCKET_TIMEOUT_SECONDS

        # SSE streams stay open for the connection's lifetime; a log line per
        # event would spam stderr — route to the module logger (debug level).
        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("MCP Streamable-HTTP %s %s", self.address_string(), format % args)

        # -- helpers ---------------------------------------------------------

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id")

        def _session_id(self) -> str:
            """Session id from the Mcp-Session-Id header, falling back to the query string."""
            sid = (self.headers.get("Mcp-Session-Id") or "").strip()
            if sid:
                return sid
            query = parse_qs(urlparse(self.path).query)
            return query.get("session_id", [""])[0]

        def _json_error(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # Error responses end the exchange: never reuse this connection
            # for a keep-alive read of a client that may already be gone.
            self.send_header("Connection", "close")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _read_body(self) -> dict | None:
            """Read + parse the JSON-RPC body; None on any parse/size failure."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json_error(400, "Invalid Content-Length header")
                return None
            if length < 0 or length > _MAX_MESSAGE_BODY_BYTES:
                self._json_error(
                    413,
                    f"Request body too large (limit {_MAX_MESSAGE_BODY_BYTES} bytes)",
                )
                return None
            raw = self.rfile.read(length) if length else b""
            try:
                request = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                request = None
            if not isinstance(request, dict):
                payload = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                )
                body = payload.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)
                return None
            return request

        def _stream_loop(self, messages: queue.Queue[str | None], session_id: str) -> None:
            """Write queued payloads as SSE ``message`` events until the sentinel.

            Idle periods poll with a heartbeat timeout instead of blocking on
            ``messages.get()`` forever: a vanished client is only detectable at
            the next write, so without a periodic write a dead client's session
            would linger until the idle sweep (30 min).
            """
            try:
                while True:
                    try:
                        data = messages.get(timeout=server._heartbeat_interval)
                    except queue.Empty:
                        # Keep-alive: SSE comment (no-op event).  Fails fast
                        # (BrokenPipeError) if the client has vanished, and
                        # touches the session so an active stream is never
                        # idle-swept.
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        server._touch(session_id)
                        continue
                    if data is None:
                        break
                    self.wfile.write(f"event: message\ndata: {data}\n\n".encode())
                    self.wfile.flush()
                    server._touch(session_id)
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.debug("MCP Streamable-HTTP client disconnected (session %s)", session_id[:8])
            finally:
                # The SSE stream is the session's lifeline: whether it ended
                # via DELETE (sentinel), a client disconnect, or the idle
                # sweep, the session is finished — drop it so a stale session
                # id cannot pile responses onto a stream nobody reads (and
                # vanished clients cannot grow the session dict without bound).
                server._drop_session(session_id)
                # HTTP/1.1 defaults to keep-alive; without this the TCP
                # connection would stay open after the stream ends (DELETE).
                self.close_connection = True

        # -- HTTP verbs --------------------------------------------------------

        def do_OPTIONS(self) -> None:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
            self.send_response(204)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
            if urlparse(self.path).path != "/mcp":
                self._json_error(404, "Not found — POST /mcp")
                return
            request = self._read_body()
            if request is None:
                return  # error response already sent
            accept = self.headers.get("Accept", "")
            wants_sse = "text/event-stream" in accept
            session_id = self._session_id()
            session = server._get_session(session_id) if session_id else None

            if wants_sse:
                if session is None:
                    # New SSE session: open the stream, deliver this request's
                    # response as the first event.
                    session_id, messages = server._new_session()
                    ok, payload = server._handle_request_capped(request)
                    if not ok:
                        # Concurrency cap reached — never open the stream;
                        # answer 503 and drop the just-created session.
                        server._drop_session(session_id)
                        self._json_error(503, "Server busy — concurrency limit reached (retry later)")
                        return
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("Mcp-Session-Id", session_id)
                        self._cors_headers()
                        self.end_headers()
                        if payload is not None:
                            self.wfile.write(f"event: message\ndata: {payload}\n\n".encode())
                            self.wfile.flush()
                        self._stream_loop(messages, session_id)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        # Normal client disconnect mid-stream: the stream loop
                        # already dropped the session (finally).  Do NOT re-raise
                        # — socketserver would print a traceback for a routine
                        # disconnect, spamming the log.
                        server._drop_session(session_id)
                    except Exception:
                        # A failure between _new_session and the stream loop
                        # must not leave the session behind.
                        server._drop_session(session_id)
                        raise
                else:
                    # Existing SSE session: ack 202, deliver on the open stream.
                    ok, payload = server._handle_request_capped(request)
                    if not ok:
                        self._json_error(503, "Server busy — concurrency limit reached (retry later)")
                        return
                    if payload is not None:
                        outcome = server._enqueue_if_current(session_id, session, payload)
                        if outcome == "gone":
                            # Session torn down (DELETE / disconnect / sweep)
                            # while the handler ran — the tool may already have
                            # executed.  Explicit 409 beats a 202 whose response
                            # nobody will read.
                            self._json_error(
                                409,
                                "Session closed — response not delivered; tool may have executed",
                            )
                            return
                        if outcome == "full":
                            self._json_error(
                                503,
                                "Session queue full — SSE stream not being consumed",
                            )
                            return
                    self.send_response(202)
                    self._cors_headers()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                return

            # JSON mode — respond directly in the body.
            ok, payload = server._handle_request_capped(request)
            if not ok:
                self._json_error(503, "Server busy — concurrency limit reached (retry later)")
                return
            if payload is None:
                self.send_response(202)
                self._cors_headers()
                if session_id:
                    self.send_header("Mcp-Session-Id", session_id)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
            # SSE resume path: re-attach to an existing session's stream.
            if urlparse(self.path).path != "/mcp":
                self._json_error(404, "Not found — GET /mcp?session_id=<id>")
                return
            session_id = self._session_id()
            session = server._get_session(session_id)
            if session is None:
                self._json_error(404, "Unknown session_id — POST /mcp first")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors_headers()
            self.end_headers()
            self._stream_loop(session, session_id)

        def do_DELETE(self) -> None:  # noqa: N802 — stdlib/3rd-party dispatch protocol (name is fixed by caller)
            if urlparse(self.path).path != "/mcp":
                self._json_error(404, "Not found")
                return
            session_id = self._session_id()
            # Pop-before-sentinel under the server lock (see _close_session):
            # an in-flight POST's enqueue check cannot interleave, so the
            # sentinel never lands behind a payload on a session nobody reads
            # (see do_POST's 409 path).
            server._close_session(session_id)
            self.send_response(202)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

    return _StreamableHandler
