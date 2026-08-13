"""
MCP HTTP+SSE transport server — standard library only.

Implements the MCP HTTP+SSE transport (protocol version 2025-03-26) with
stdlib ``http.server``, so the published core package keeps its zero
web-framework dependency (fastapi/uvicorn live in the optional ``[webapp]``
extra — see pyproject.toml).

Endpoints
---------
GET /sse
    Establishes the SSE stream.  The first event is ``endpoint``; its data is
    the POST URL for this session (``/message?session_id=<id>``).  Afterwards
    the stream carries ``message`` events with JSON-RPC responses.
POST /message?session_id=<id>
    Client-to-server JSON-RPC requests.  Normally answers ``202 Accepted``; the
    JSON-RPC response (for requests carrying an ``id``) is delivered on the
    session's SSE stream.  Notifications (no ``id``) get no ``message`` event.
    When the session's SSE backlog is full (client stopped reading its stream),
    the POST answers ``503`` instead.  If the session is torn down (DELETE /
    client disconnect) while the request is being handled, the POST answers
    ``409`` — the tool may already have executed and the client must decide
    whether to retry.
DELETE /message?session_id=<id>
    Client-initiated teardown: closes the session's SSE stream.

Threading model: one thread per SSE connection (``ThreadingHTTPServer``,
daemon threads — the process exits cleanly even with open streams).  Session
state is a per-session bounded ``queue.Queue`` guarded by the server's lock;
the plumbing (registration, teardown, idle sweep, shutdown) is shared with the
Streamable-HTTP transport via ``_SessionQueueMixin`` (``_session_queue``).

Limitation: a client that disconnects silently while no event is in flight is
only detected on the next write (BrokenPipe), so its reader thread stays
blocked in ``queue.get()`` until the background idle sweep closes the
session (default TTL 30 min — see ``session_idle_ttl``).  Acceptable for a
local dev server; the daemon-thread model keeps process teardown clean.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from external_llm.agent.tool_registry import ToolRegistry
from external_llm.editor.agent.mcp._session_queue import (
    _MAX_MESSAGE_BODY_BYTES,
    _SESSION_IDLE_TTL_SECONDS,
    _SESSION_SWEEP_INTERVAL_SECONDS,
    JsonRpcHandler,
    _SessionQueueMixin,
)

logger = logging.getLogger(__name__)

# Shared session-queue plumbing (registration, teardown, enqueue-under-race,
# idle sweep, shutdown) lives in _SessionQueueMixin; constants and the
# JsonRpcHandler type live in _session_queue.


class SSEMcpServer(_SessionQueueMixin):
    """Threaded HTTP server exposing asicode tools over MCP's SSE transport.

    Session-queue plumbing (registration, teardown, enqueue-under-race, idle
    sweep, shutdown) is shared with the Streamable-HTTP transport via
    ``_SessionQueueMixin`` (see ``_session_queue``).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        handle: Optional[JsonRpcHandler] = None,
        session_idle_ttl: float = _SESSION_IDLE_TTL_SECONDS,
        sweep_interval: float = _SESSION_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        self.registry = registry
        # JSON-RPC handler — injected by server.py's _run_sse_server so the
        # stdio and SSE transports share ONE implementation; tests inject
        # fakes.  None (direct construction) degrades to an explicit error.
        self._handle = handle
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self.host, self.port = self.httpd.server_address[:2]
        logger.info("MCP SSE server bound to %s:%s", self.host, self.port)
        # Session state + idle-sweep reaper (must run after httpd is bound so
        # a construction failure cannot leak the daemon thread).
        self._init_session_plumbing(
            thread_name="mcp-sse-sweep",
            session_idle_ttl=session_idle_ttl,
            sweep_interval=sweep_interval,
        )



def _make_handler(server: SSEMcpServer) -> type[BaseHTTPRequestHandler]:
    """Build the request-handler class bound to ``server`` (closure, no globals)."""

    class _SseHandler(BaseHTTPRequestHandler):
        server_version = "asicode-mcp-sse/1.0"
        protocol_version = "HTTP/1.1"

        # SSE streams stay open for the connection's lifetime; a log line per
        # event would spam stderr — route to the module logger (debug level).
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("MCP SSE %s %s", self.address_string(), fmt % args)

        # -- helpers ---------------------------------------------------------

        def _cors_headers(self) -> None:
            # Browser-based MCP clients (e.g. IDE webviews) need CORS; the
            # SSE transport is origin-agnostic, so allow all.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _session_id(self) -> str:
            query = parse_qs(urlparse(self.path).query)
            return query.get("session_id", [""])[0]

        def _json_error(self, code: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        # -- HTTP verbs --------------------------------------------------------

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/sse":
                self._json_error(404, "Not found — use GET /sse")
                return
            self._open_stream()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/message":
                self._json_error(404, "Not found — POST to the URL from the 'endpoint' event")
                return
            session_id = self._session_id()
            session = server._get_session(session_id)
            if session is None:
                self._json_error(404, "Unknown session_id — open GET /sse first")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json_error(400, "Invalid Content-Length header")
                return
            if length < 0 or length > _MAX_MESSAGE_BODY_BYTES:
                self._json_error(
                    413,
                    f"Request body too large (limit {_MAX_MESSAGE_BODY_BYTES} bytes)",
                )
                return
            raw = self.rfile.read(length) if length else b""
            try:
                request = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                request = None
            if not isinstance(request, dict):
                payload = json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                })
            else:
                # None for notifications (no id) — the 202 ack is their only reply.
                payload = server._handle_request(request)
            if payload is not None:
                outcome = server._enqueue_if_current(session_id, session, payload)
                if outcome == "gone":
                    # The session was torn down (DELETE / disconnect) while the
                    # handler ran — the tool may already have executed.  An
                    # explicit 409 beats a 202 whose response nobody will read.
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

        def do_DELETE(self) -> None:
            if urlparse(self.path).path != "/message":
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

        # -- SSE stream ---------------------------------------------------------

        def _open_stream(self) -> None:
            session_id, messages = server._new_session()
            endpoint = f"/message?session_id={session_id}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors_headers()
            self.end_headers()
            try:
                # The endpoint event MUST precede every message event (spec).
                self.wfile.write(f"event: endpoint\ndata: {endpoint}\n\n".encode())
                self.wfile.flush()
                while True:
                    data = messages.get()
                    if data is None:
                        break
                    self.wfile.write(f"event: message\ndata: {data}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client went away — nothing to clean up beyond the session.
                logger.debug("MCP SSE client disconnected (session %s)", session_id[:8])
            finally:
                # HTTP/1.1 defaults to keep-alive; without this the TCP
                # connection would stay open after the stream ends (DELETE).
                self.close_connection = True
                server._drop_session(session_id)

    return _SseHandler
