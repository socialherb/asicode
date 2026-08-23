"""
Tests for the MCP HTTP+SSE transport server
(external_llm/editor/agent/mcp/sse_server.py) and the shared JSON-RPC
handler (server.py).

Hermetic: the server binds to 127.0.0.1 with an ephemeral port (port=0) and
all client traffic uses loopback http.client / raw sockets.  No network
access and no claude-agent-sdk required.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.client import HTTPConnection

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.editor.agent.mcp.server import (
    MCP_PROTOCOL_VERSION,
    _handle_jsonrpc,
    list_mcp_tools,
)
from external_llm.editor.agent.mcp.sse_server import SSEMcpServer
from external_llm.repl.collaborate.asi_mcp_adapter import _EXCLUDED_TOOLS


def _make_registry(repo_root: str = ".") -> ToolRegistry:
    return ToolRegistry(repo_root=repo_root, config=AgentConfig())


@pytest.fixture()
def sse_server():
    server = SSEMcpServer(_make_registry(), "127.0.0.1", 0, handle=_handle_jsonrpc)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


class _SseReader:
    """Reads SSE events from a raw socket; every read is timeout-bounded."""

    def __init__(self, sock: socket.socket, initial: bytes = b"") -> None:
        self._sock = sock
        self._sock.settimeout(5)
        self._buf = initial

    def read_event(self) -> tuple[str, str]:
        """Read one SSE event; returns (event_type, data)."""
        while b"\n\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise AssertionError("SSE stream closed unexpectedly")
            self._buf += chunk
        raw, _, self._buf = self._buf.partition(b"\n\n")
        event, data = "message", ""
        for line in raw.split(b"\n"):
            if line.startswith(b"event:"):
                event = line[len(b"event:") :].strip().decode("utf-8")
            elif line.startswith(b"data:"):
                data = line[len(b"data:") :].strip().decode("utf-8")
        return event, data


def _open_sse(host: str, port: int) -> tuple[socket.socket, bytes]:
    """GET /sse over a raw socket; returns (sock, leftover after headers)."""
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(b"GET /sse HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        assert chunk, "connection closed while reading SSE headers"
        buf += chunk
    headers, _, rest = buf.partition(b"\r\n\r\n")
    assert b" 200 " in headers.split(b"\r\n")[0], headers
    assert b"text/event-stream" in headers
    return sock, rest


def _post(
    host: str,
    port: int,
    path: str,
    payload: object | None = None,
    raw: bytes | None = None,
) -> tuple[int, bytes]:
    conn = HTTPConnection(host, port, timeout=5)
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def _open_endpoint(host: str, port: int) -> tuple[str, _SseReader]:
    """Open an SSE connection and return (endpoint-path, reader)."""
    sock, rest = _open_sse(host, port)
    reader = _SseReader(sock, rest)
    event, endpoint = reader.read_event()
    assert event == "endpoint"
    assert endpoint.startswith("/message?session_id=")
    return endpoint, reader


def _start_server(*, handle=_handle_jsonrpc, **kwargs):
    """Start an SSEMcpServer in a daemon thread; returns (server, thread)."""
    server = SSEMcpServer(_make_registry(), "127.0.0.1", 0, handle=handle, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _wait_until(predicate, timeout=5.0):
    """Poll ``predicate`` until truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


# --- SSE transport ---------------------------------------------------------


def test_sse_initialize_and_list_tools_roundtrip(sse_server):
    host, port = sse_server.host, sse_server.port
    endpoint, reader = _open_endpoint(host, port)

    status, _ = _post(
        host,
        port,
        endpoint,
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.initialize", "params": {}},
    )
    assert status == 202
    _, raw = reader.read_event()
    result = json.loads(raw)["result"]
    assert result["serverInfo"]["name"] == "asicode"
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert result["server_name"] == "asicode"  # legacy flat keys preserved

    _post(
        host,
        port,
        endpoint,
        {"jsonrpc": "2.0", "id": 2, "method": "mcp.list_tools", "params": {}},
    )
    _, raw = reader.read_event()
    tools = json.loads(raw)["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "read_file" in names
    assert not (names & _EXCLUDED_TOOLS)
    assert all({"name", "description", "parameters"} <= set(t) for t in tools)


def test_sse_call_tool_read_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "hello.txt").write_text("hello from mcp sse", encoding="utf-8")
    server = SSEMcpServer(_make_registry(str(repo)), "127.0.0.1", 0, handle=_handle_jsonrpc)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint, reader = _open_endpoint(server.host, server.port)
        _post(
            server.host,
            server.port,
            endpoint,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "mcp.call_tool",
                "params": {"name": "read_file", "arguments": {"path": "hello.txt"}},
            },
        )
        _, raw = reader.read_event()
        msg = json.loads(raw)
        assert msg["id"] == 3
        result = msg["result"]
        assert result["isError"] is False
        assert "hello from mcp sse" in result["content"][0]["text"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_sse_call_tool_unknown_tool_is_error(sse_server):
    endpoint, reader = _open_endpoint(sse_server.host, sse_server.port)
    _post(
        sse_server.host,
        sse_server.port,
        endpoint,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "mcp.call_tool",
            "params": {"name": "no_such_tool", "arguments": {}},
        },
    )
    _, raw = reader.read_event()
    result = json.loads(raw)["result"]
    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]


def test_sse_notification_gets_no_message_event(sse_server):
    """Notifications (no id) are acked with 202 but never produce a stream event."""
    endpoint, reader = _open_endpoint(sse_server.host, sse_server.port)
    status, _ = _post(
        sse_server.host,
        sse_server.port,
        endpoint,
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    )
    assert status == 202
    # A request with an id sent afterwards must be the FIRST message event —
    # proving the notification produced none (FIFO queue order).
    _post(
        sse_server.host,
        sse_server.port,
        endpoint,
        {"jsonrpc": "2.0", "id": 9, "method": "mcp.ping", "params": {}},
    )
    _, raw = reader.read_event()
    msg = json.loads(raw)
    assert msg["id"] == 9
    assert msg["result"] == {"status": "ok"}


def test_sse_parse_error_gets_jsonrpc_error_event(sse_server):
    endpoint, reader = _open_endpoint(sse_server.host, sse_server.port)
    _post(sse_server.host, sse_server.port, endpoint, raw=b"{invalid json")
    _, raw = reader.read_event()
    error = json.loads(raw)["error"]
    assert error["code"] == -32700


def test_sse_post_unknown_session_404(sse_server):
    status, body = _post(
        sse_server.host,
        sse_server.port,
        "/message?session_id=deadbeef",
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"},
    )
    assert status == 404
    assert b"Unknown session_id" in body


def test_sse_get_unknown_path_404(sse_server):
    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("GET", "/nope")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_sse_delete_closes_stream(sse_server):
    endpoint, reader = _open_endpoint(sse_server.host, sse_server.port)
    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("DELETE", endpoint)
    resp = conn.getresponse()
    assert resp.status == 202
    resp.read()
    conn.close()
    # The stream must now hit EOF with no further events.
    sock = reader._sock
    sock.settimeout(5)
    chunks = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks += chunk
    assert chunks == b""


def test_sse_options_preflight(sse_server):
    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("OPTIONS", "/sse")
    resp = conn.getresponse()
    assert resp.status == 204
    assert resp.getheader("Access-Control-Allow-Origin") == "*"
    resp.read()
    conn.close()


# --- shared JSON-RPC handler (stdio + SSE) ---------------------------------


def test_handle_jsonrpc_initialize_spec_shape():
    resp = _handle_jsonrpc(
        _make_registry(),
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.initialize", "params": {}},
    )
    result = resp["result"]
    assert result["serverInfo"]["name"] == "asicode"
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert result["server_name"] == "asicode"  # legacy stdio keys preserved
    assert resp["id"] == 1


def test_handle_jsonrpc_unknown_method_acks():
    resp = _handle_jsonrpc(
        _make_registry(),
        {"jsonrpc": "2.0", "id": 7, "method": "mcp.ping", "params": {}},
    )
    assert resp["result"] == {"status": "ok"}


def test_handle_jsonrpc_non_dict_returns_invalid_request():
    resp = _handle_jsonrpc(_make_registry(), [1, 2, 3])
    assert resp["error"]["code"] == -32600


def test_handle_jsonrpc_call_tool_missing_args_is_error():
    resp = _handle_jsonrpc(
        _make_registry(),
        {"jsonrpc": "2.0", "id": 2, "method": "mcp.call_tool", "params": {}},
    )
    assert resp["result"]["isError"] is True


def test_list_mcp_tools_has_schema_shape():
    tools = list_mcp_tools(_make_registry())
    assert tools
    assert set(tools[0]) == {"name", "description", "parameters"}


def test_list_mcp_tools_none_registry_fallback():
    """registry=None falls back to the static AGENT_TOOL_SCHEMAS (L128)."""
    tools = list_mcp_tools(None)
    assert isinstance(tools, list)
    assert all("name" in t for t in tools)


def test_run_sse_server_wires_shared_handler_and_clean_shutdown(monkeypatch):
    """_run_sse_server must inject the shared JSON-RPC handler and shut down
    cleanly on KeyboardInterrupt (Ctrl-C from the CLI)."""
    import external_llm.editor.agent.mcp.sse_server as sse_mod
    from external_llm.editor.agent.mcp import server as mcp_server_mod

    captured: dict = {}

    class FakeServer:
        def __init__(self, registry, host, port, *, handle):
            captured["handle"] = handle
            captured["host"], captured["port"] = host, port

        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            captured["shutdown"] = True

    monkeypatch.setattr(sse_mod, "SSEMcpServer", FakeServer)
    mcp_server_mod._run_sse_server(_make_registry(), "127.0.0.1", 9999)
    assert captured["handle"] is mcp_server_mod._handle_jsonrpc
    assert (captured["host"], captured["port"]) == ("127.0.0.1", 9999)
    assert captured["shutdown"] is True


def test_run_http_server_injects_handler_and_prints_endpoint(capsys):
    """The shared HTTP launcher (behind both wrappers) must construct the
    server with the shared JSON-RPC handler and print the transport's
    endpoint/label on startup."""
    from external_llm.editor.agent.mcp import server as mcp_server_mod

    captured: dict = {}

    class FakeServer:
        def __init__(self, registry, host, port, *, handle):
            captured["handle"] = handle

        def serve_forever(self):
            raise KeyboardInterrupt

        def shutdown(self):
            captured["shutdown"] = True

    mcp_server_mod._run_http_server(
        _make_registry(),
        "127.0.0.1",
        9999,
        server_factory=FakeServer,
        endpoint="/mcp",
        label="Streamable-HTTP",
    )
    err = capsys.readouterr().err
    assert "Streamable-HTTP mode" in err
    assert "http://127.0.0.1:9999/mcp" in err
    assert captured["handle"] is mcp_server_mod._handle_jsonrpc
    assert captured["shutdown"] is True


def test_sse_post_session_queue_full_returns_503(sse_server, monkeypatch):
    """A session whose SSE backlog is full (reader not consuming) must answer
    503 instead of growing an unbounded queue (memory-DoS guard)."""
    import external_llm.editor.agent.mcp._session_queue as session_queue_mod

    monkeypatch.setattr(session_queue_mod, "_MAX_QUEUED_EVENTS", 2)
    session_id, messages = sse_server._new_session()
    assert messages.maxsize == 2
    messages.put("event-1")
    messages.put("event-2")  # backlog full — no reader drains it

    status, body = _post(
        sse_server.host,
        sse_server.port,
        f"/message?session_id={session_id}",
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"},
    )
    assert status == 503
    assert b"queue full" in body.lower()


def test_sse_delete_full_session_drops_without_blocking(sse_server, monkeypatch):
    """DELETE against a full session must not block the handler thread
    (put_nowait, not put); the session is dropped so later POSTs 404."""
    import external_llm.editor.agent.mcp._session_queue as session_queue_mod

    monkeypatch.setattr(session_queue_mod, "_MAX_QUEUED_EVENTS", 2)
    session_id, messages = sse_server._new_session()
    messages.put("event-1")
    messages.put("event-2")  # backlog full — no reader drains it

    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("DELETE", f"/message?session_id={session_id}")
    resp = conn.getresponse()
    assert resp.status == 202
    resp.read()
    conn.close()

    assert sse_server._get_session(session_id) is None
    status, _ = _post(
        sse_server.host,
        sse_server.port,
        f"/message?session_id={session_id}",
        {"jsonrpc": "2.0", "id": 2, "method": "mcp.ping"},
    )
    assert status == 404


def test_sse_post_during_teardown_answers_409_not_silent_202():
    """Race regression: a request whose handler outlives the session (DELETE
    lands mid-flight) must answer an explicit 409 — never a 202 whose response
    is queued onto a stream nobody reads (response lost + duplicate tool run)."""
    executed = threading.Event()
    release = threading.Event()

    def slow_handle(registry, request):
        executed.set()
        release.wait(timeout=5)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"slow": True}}

    server = SSEMcpServer(_make_registry(), "127.0.0.1", 0, handle=slow_handle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.host, server.port
        endpoint, reader = _open_endpoint(host, port)

        outcome: dict = {}

        def post_slow():
            conn = HTTPConnection(host, port, timeout=10)
            conn.request(
                "POST",
                endpoint,
                body=json.dumps({"jsonrpc": "2.0", "id": 77, "method": "mcp.ping", "params": {}}).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            outcome["status"] = resp.status
            outcome["body"] = resp.read()
            conn.close()

        poster = threading.Thread(target=post_slow, daemon=True)
        poster.start()
        assert executed.wait(timeout=5), "handler should be running (tool in flight)"

        # Teardown while the handler is still running.
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("DELETE", endpoint)
        resp = conn.getresponse()
        assert resp.status == 202
        resp.read()
        conn.close()

        release.set()
        poster.join(timeout=5)
        assert outcome["status"] == 409, f"expected 409, got {outcome}"
        assert b"Session closed" in outcome["body"]

        # The stream closed without ever delivering the response.
        reader._sock.settimeout(5)
        chunks = b""
        while True:
            chunk = reader._sock.recv(4096)
            if not chunk:
                break
            chunks += chunk
        assert b"slow" not in chunks
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)


def test_sse_post_after_delete_404(sse_server):
    """DELETE removes the session synchronously — a POST to the same id after
    teardown must 404 instead of enqueueing onto a stream nobody reads."""
    endpoint, _ = _open_endpoint(sse_server.host, sse_server.port)
    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("DELETE", endpoint)
    resp = conn.getresponse()
    assert resp.status == 202
    resp.read()
    conn.close()

    status, body = _post(
        sse_server.host,
        sse_server.port,
        endpoint,
        {"jsonrpc": "2.0", "id": 5, "method": "mcp.ping"},
    )
    assert status == 404
    assert b"Unknown session_id" in body


def test_idle_sessions_swept_after_ttl():
    """A session whose client never sends DELETE is reclaimed by the sweep."""
    server, thread = _start_server(session_idle_ttl=0.5, sweep_interval=0.05)
    try:
        endpoint, reader = _open_endpoint(server.host, server.port)
        session_id = endpoint.split("session_id=")[1]
        assert session_id in server._sessions

        assert _wait_until(lambda: session_id not in server._sessions)

        # The sweep's sentinel closed the stream (EOF → read fails).
        with pytest.raises((AssertionError, OSError)):
            reader.read_event()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_disconnected_client_session_reclaimed_by_sweep():
    """Abrupt close without DELETE: the stream thread stays blocked in
    queue.get(); the sweep must still reclaim the session."""
    server, thread = _start_server(session_idle_ttl=0.5, sweep_interval=0.05)
    try:
        endpoint, reader = _open_endpoint(server.host, server.port)
        session_id = endpoint.split("session_id=")[1]
        assert session_id in server._sessions
        reader._sock.close()  # abrupt teardown — NO DELETE

        assert _wait_until(lambda: session_id not in server._sessions)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_sse_active_session_survives_sweep():
    """POSTing refreshes the idle timestamp — an active session must not be
    reaped while requests keep arriving (guards the _last_active touch)."""
    server, thread = _start_server(session_idle_ttl=0.6, sweep_interval=0.05)
    try:
        endpoint, _ = _open_endpoint(server.host, server.port)
        session_id = endpoint.split("session_id=")[1]

        # Keep POSTing past the TTL: every request must push the reaping point
        # forward (the background sweep fires every 0.05s).
        for _ in range(4):
            time.sleep(0.1)
            status, _ = _post(
                server.host,
                server.port,
                endpoint,
                {"jsonrpc": "2.0", "method": "mcp.ping", "params": {}},
            )
            assert status == 202
        assert session_id in server._sessions

        # Once the POSTs stop, the next sweep reclaims the session.
        assert _wait_until(lambda: session_id not in server._sessions)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_shutdown_drops_all_sessions():
    """shutdown() closes every open session and stops the sweep."""
    server, thread = _start_server()
    endpoint, reader = _open_endpoint(server.host, server.port)
    session_id = endpoint.split("session_id=")[1]
    assert session_id in server._sessions
    try:
        server.shutdown()
        thread.join(timeout=5)
        assert server._sessions == {}
        assert server._last_active == {}
        # The shutdown sentinel closed the stream (EOF → read fails).
        with pytest.raises((AssertionError, OSError)):
            reader.read_event()
    finally:
        reader._sock.close()


# ── RED→GREEN: uncovered branches ────────────────────────────────────────────


def test_sse_post_wrong_path_404(sse_server):
    """POST to a non-/message path answers 404 (L151-152)."""
    status, data = _post(
        sse_server.host,
        sse_server.port,
        "/wrong",
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"},
    )
    assert status == 404
    assert b"endpoint" in data


def test_sse_post_invalid_content_length_400(sse_server):
    """A non-numeric Content-Length on a live session answers 400 (L160-162)."""
    sock, rest = _open_sse(sse_server.host, sse_server.port)
    reader = _SseReader(sock, rest)
    _, endpoint = reader.read_event()
    try:
        conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
        conn.request(
            "POST",
            endpoint,
            body=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "abc"},
        )
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        sock.close()


def test_sse_post_body_too_large_413(sse_server):
    """A body over the size cap on a live session answers 413 (L164-168)."""
    from external_llm.editor.agent.mcp._session_queue import _MAX_MESSAGE_BODY_BYTES

    sock, rest = _open_sse(sse_server.host, sse_server.port)
    reader = _SseReader(sock, rest)
    _, endpoint = reader.read_event()
    try:
        # Send only the headers with an oversized Content-Length; the server
        # must answer 413 without us shipping the (1 MiB+) body.
        conn = socket.create_connection((sse_server.host, sse_server.port), timeout=5)
        conn.sendall(
            f"POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
            f"application/json\r\nContent-Length: {_MAX_MESSAGE_BODY_BYTES + 1}\r\n\r\n".encode()
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
        assert b" 413 " in buf.split(b"\r\n")[0], buf
        conn.close()
    finally:
        sock.close()


def test_sse_error_response_closes_connection(sse_server):
    """An error response must close the connection (no keep-alive reuse).

    Regression: error paths answered 413/404/400 but left the connection in
    HTTP/1.1 keep-alive state, so the server looped back into
    ``rfile.readline`` on a socket the client had closed — a
    ``ConnectionResetError`` that socketserver printed as a traceback
    (log spam for a routine disconnect).
    """
    from external_llm.editor.agent.mcp._session_queue import _MAX_MESSAGE_BODY_BYTES

    # Open a session so we have a real endpoint to POST against.
    sock, rest = _open_sse(sse_server.host, sse_server.port)
    reader = _SseReader(sock, rest)
    _, endpoint = reader.read_event()

    # Send only the headers with an oversized Content-Length; the server must
    # answer 413 AND close the connection (no keep-alive reuse).
    conn = socket.create_connection((sse_server.host, sse_server.port), timeout=5)
    conn.sendall(
        f"POST {endpoint} HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
        f"application/json\r\nContent-Length: {_MAX_MESSAGE_BODY_BYTES + 1}\r\n\r\n".encode()
    )
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    header_block, _, body = buf.partition(b"\r\n\r\n")
    assert b" 413 " in header_block.split(b"\r\n")[0], header_block
    # ``Connection: close`` — the server must not wait for another request
    # on this connection.  Without this, the next readline would block on a
    # socket the client has closed (keep-alive reuse after an error).
    assert b"close" in header_block.lower(), header_block
    # The server closes the connection: read the declared body length, then a
    # subsequent read must return EOF immediately (not hang on a keep-alive
    # read of a connection the client closed, not fail on a reset).
    content_length = int(
        next(
            line.split(b": ", 1)[1]
            for line in header_block.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
    )
    conn.settimeout(5)
    while len(body) < content_length:
        chunk = conn.recv(4096)
        assert chunk, "connection closed before the 413 body completed"
        body += chunk
    rest = conn.recv(4096)
    assert rest == b"", f"expected EOF after 413, got {rest!r}"
    conn.close()
    sock.close()


def test_sse_disconnect_between_requests_is_quiet(sse_server, capsys):
    """RST between requests must not crash the handler or log a traceback.

    ``QuietHttpHandler`` swallows ``ConnectionResetError`` out of the
    keep-alive readline — socketserver would otherwise print a traceback for
    the routine act of a client vanishing between requests.  The server must
    stay fully usable afterwards, and stderr must stay clean.
    """
    import struct

    # Raw-socket keep-alive connection: complete one request (404), then RST
    # the same socket.  The server's worker thread sits in its next
    # rfile.readline() (keep-alive loop); without the QuietHttpHandler guard
    # that readline raises ConnectionResetError which socketserver prints as
    # a traceback.
    sock = socket.create_connection((sse_server.host, sse_server.port), timeout=5)
    sock.sendall(b"GET /wrong HTTP/1.1\r\nHost: localhost\r\n\r\n")
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    assert b" 404 " in buf.split(b"\r\n")[0], buf
    # RST the connection (linger-0 close) so the server's next readline —
    # which it is about to attempt on the keep-alive loop — fails hard.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()

    # Give the server thread time to hit the dead socket.  It must swallow
    # the reset quietly: the worker thread survives AND no traceback reaches
    # stderr (socketserver's default behaviour would print one).
    time.sleep(0.5)

    # The server must still answer new requests (worker thread survived).
    status, body = _post(
        sse_server.host,
        sse_server.port,
        "/message?session_id=nope",
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"},
    )
    assert status == 404
    assert b"Unknown session_id" in body

    # And no traceback was printed by the server's worker thread.
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_sse_delete_wrong_path_404(sse_server):
    """DELETE to a non-/message path answers 404 (L207-208)."""
    conn = HTTPConnection(sse_server.host, sse_server.port, timeout=5)
    conn.request("DELETE", "/wrong")
    resp = conn.getresponse()
    assert resp.status == 404
    resp.read()
    conn.close()


def test_sse_stream_client_disconnect_logged(sse_server, caplog):
    """A client that vanishes mid-stream logs 'disconnected' on the next
    write instead of crashing the stream thread (L241-243)."""
    import logging
    import struct

    sock, rest = _open_sse(sse_server.host, sse_server.port)
    reader = _SseReader(sock, rest)
    event, endpoint = reader.read_event()
    assert event == "endpoint"
    # RST the connection so the server's next write fails hard.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    with caplog.at_level(logging.DEBUG, logger="external_llm.editor.agent.mcp.sse_server"):
        _post(
            sse_server.host,
            sse_server.port,
            endpoint,
            {"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"},
        )
        assert _wait_until(lambda: any("client disconnected" in r.message for r in caplog.records))
