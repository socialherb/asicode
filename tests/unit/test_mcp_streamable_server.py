"""Tests for the MCP Streamable-HTTP transport (protocol 2025-11-05).

Covers the three response modes of ``POST /mcp``:
  * JSON mode  — Accept: application/json → 200 + JSON-RPC body (id'd requests)
                 or 202 (notifications);
  * SSE mode   — Accept including text/event-stream → 200 text/event-stream,
                 session id in Mcp-Session-Id, first response as first event,
                 later POSTs acked 202 with responses on the open stream;
  * teardown   — DELETE /mcp closes the session stream.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
from http.client import HTTPConnection

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.editor.agent.mcp.server import _handle_jsonrpc
from external_llm.editor.agent.mcp.streamable_server import StreamableHttpMcpServer


def _make_registry(repo_root: str = ".") -> ToolRegistry:
    return ToolRegistry(repo_root=repo_root, config=AgentConfig())


@pytest.fixture()
def streamable_server():
    server = StreamableHttpMcpServer(_make_registry(), "127.0.0.1", 0, handle=_handle_jsonrpc)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


def _start_server(*, handle=_handle_jsonrpc, **kwargs):
    """Start a StreamableHttpMcpServer in a daemon thread; returns (server, thread)."""
    server = StreamableHttpMcpServer(_make_registry(), "127.0.0.1", 0, handle=handle, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _wait_until(predicate, timeout=5.0):
    """Poll ``predicate`` until truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def _post(server, body: dict, *, accept: str = "application/json",
          session_id: str = "") -> tuple[int, str, dict]:
    """POST /mcp; returns (status, content_type, headers)."""
    conn = HTTPConnection("127.0.0.1", server.port, timeout=5)
    headers = {"Content-Type": "application/json", "Accept": accept}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    conn.request("POST", "/mcp", body=json.dumps(body), headers=headers)
    resp = conn.getresponse()
    status = resp.status
    ctype = resp.getheader("Content-Type", "")
    hdrs = dict(resp.getheaders())
    body_bytes = resp.read()
    conn.close()
    return status, ctype, hdrs, body_bytes


def _initialize(id_value=1):
    return {
        "jsonrpc": "2.0",
        "id": id_value,
        "method": "mcp.initialize",
        "params": {
            "protocolVersion": "2025-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }


def test_json_mode_roundtrip(streamable_server):
    """JSON mode: id'd request → 200 + JSON-RPC response in the body."""
    status, ctype, _, body = _post(streamable_server, _initialize())
    assert status == 200
    assert ctype.startswith("application/json")
    result = json.loads(body)
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == 1
    assert "serverInfo" in result["result"]


def test_json_mode_notification_gets_202(streamable_server):
    """Notification (no id) → 202 Accepted, no response body."""
    status, _, _, body = _post(streamable_server, {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    })
    assert status == 202
    assert body == b""


def test_sse_mode_opens_stream_and_returns_session_id(streamable_server):
    """SSE mode: 200 text/event-stream + Mcp-Session-Id + first response event."""
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn.request(
        "POST", "/mcp", body=json.dumps(_initialize()),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type", "").startswith("text/event-stream")
    session_id = resp.getheader("Mcp-Session-Id")
    assert session_id, "SSE mode must return Mcp-Session-Id"

    # First event = the initialize response
    import socket as _socket
    _socket.setdefaulttimeout(5)
    try:
        first = resp.readline().decode()
        assert first.startswith("event: message")
        data = resp.readline().decode()
        assert data.startswith("data: ")
        payload = json.loads(data[len("data: "):])
        assert payload["id"] == 1
        assert "serverInfo" in payload["result"]
    finally:
        conn.close()


def test_sse_session_second_post_202_and_stream_delivery(streamable_server):
    """After the stream is open, a POST with the session id is acked 202 and
    the response arrives as the next event on the stream."""
    import http.client
    import socket as _socket

    conn = http.client.HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn.request(
        "POST", "/mcp", body=json.dumps(_initialize()),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    session_id = resp.getheader("Mcp-Session-Id")
    # consume the first event
    resp.readline()  # event
    resp.readline()  # data
    resp.readline()  # blank

    # Second POST with session id → 202
    status, _, _, body = _post(
        streamable_server,
        {"jsonrpc": "2.0", "id": 2, "method": "mcp.list_tools", "params": {}},
        accept="application/json, text/event-stream",
        session_id=session_id,
    )
    assert status == 202, f"expected 202, got {status}: {body!r}"

    # The response arrives on the open stream as the next message event
    _socket.setdefaulttimeout(5)
    try:
        line = resp.readline().decode()
        assert line.startswith("event: message"), f"got {line!r}"
        data = resp.readline().decode()
        payload = json.loads(data[len("data: "):])
        assert payload["id"] == 2
        assert "tools" in payload["result"]
    finally:
        conn.close()


def test_sse_mode_error_response_on_stream(streamable_server):
    """A failed call in SSE mode still delivers its error as a message event."""
    import http.client
    import socket as _socket

    conn = http.client.HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn.request(
        "POST", "/mcp", body=json.dumps({
            "jsonrpc": "2.0", "id": 7,
            "method": "mcp.call_tool",
            "params": {"name": "no_such_tool", "arguments": {}},
        }),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    _socket.setdefaulttimeout(5)
    try:
        resp.readline()  # event: message
        data = resp.readline().decode()
        payload = json.loads(data[len("data: "):])
        assert payload["id"] == 7
        # Unknown tool → result.isError=True (the _dispatch_tool contract),
        # delivered as a message event on the stream.
        assert payload["result"].get("isError") is True
    finally:
        conn.close()


def test_delete_closes_session(streamable_server):
    """DELETE /mcp with the session id terminates the session."""
    import http.client

    # Open an SSE session
    conn = http.client.HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn.request(
        "POST", "/mcp", body=json.dumps(_initialize()),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    session_id = resp.getheader("Mcp-Session-Id")
    resp.readline()
    resp.readline()
    resp.readline()

    # DELETE → 202, stream closes (EOF)
    conn2 = HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn2.request("DELETE", f"/mcp?session_id={session_id}")
    r2 = conn2.getresponse()
    assert r2.status == 202
    r2.read()
    conn2.close()

    import socket as _socket
    _socket.setdefaulttimeout(5)
    try:
        assert resp.read() == b"", "stream should close after DELETE"
    except OSError:
        pass  # connection reset is also acceptable
    finally:
        conn.close()


def test_delete_removes_session_from_server(streamable_server):
    """DELETE must fully remove the session — a stale id then starts a NEW
    session instead of piling responses onto a stream nobody reads (R1)."""
    import http.client
    import socket as _socket

    conn = http.client.HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn.request(
        "POST", "/mcp", body=json.dumps(_initialize()),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    session_id = resp.getheader("Mcp-Session-Id")
    resp.readline()
    resp.readline()
    resp.readline()
    assert session_id in streamable_server._sessions

    conn2 = HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn2.request("DELETE", f"/mcp?session_id={session_id}")
    r2 = conn2.getresponse()
    assert r2.status == 202
    r2.read()
    conn2.close()

    # The stream thread consumes the sentinel and drops the session.
    assert _wait_until(lambda: session_id not in streamable_server._sessions)

    _socket.setdefaulttimeout(5)
    try:
        assert resp.read() == b"", "stream should close after DELETE"
    except OSError:
        pass
    finally:
        conn.close()

    # Re-POST with the stale id → a NEW session (was: dead 202/enqueue loop).
    conn3 = HTTPConnection("127.0.0.1", streamable_server.port, timeout=5)
    conn3.request(
        "POST", "/mcp", body=json.dumps(_initialize(9)),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 "Mcp-Session-Id": session_id},
    )
    resp3 = conn3.getresponse()
    assert resp3.status == 200
    new_sid = resp3.getheader("Mcp-Session-Id")
    assert new_sid and new_sid != session_id
    assert new_sid in streamable_server._sessions
    conn3.close()


def test_idle_sessions_swept_after_ttl():
    """A session whose client never sends DELETE is reclaimed by the sweep (R1)."""
    import http.client
    import socket as _socket

    server, thread = _start_server(session_idle_ttl=0.2, sweep_interval=0.05)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.request(
            "POST", "/mcp", body=json.dumps(_initialize()),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        session_id = resp.getheader("Mcp-Session-Id")
        resp.readline()
        resp.readline()
        resp.readline()
        assert session_id in server._sessions

        assert _wait_until(lambda: session_id not in server._sessions)

        # The sweep's sentinel closed the stream (EOF).
        _socket.setdefaulttimeout(5)
        with contextlib.suppress(OSError):
            assert resp.read() == b""
    finally:
        conn.close()
        server.shutdown()
        thread.join(timeout=5)


def test_disconnected_client_session_reclaimed_by_sweep():
    """Abrupt close without DELETE: the stream thread stays blocked in
    queue.get(); the sweep must still reclaim the session (R1)."""
    import http.client

    server, thread = _start_server(session_idle_ttl=0.2, sweep_interval=0.05)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.request(
            "POST", "/mcp", body=json.dumps(_initialize()),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        session_id = resp.getheader("Mcp-Session-Id")
        resp.readline()
        resp.readline()
        resp.readline()
        assert session_id in server._sessions
        conn.close()  # abrupt teardown — NO DELETE

        assert _wait_until(lambda: session_id not in server._sessions)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_shutdown_drops_all_sessions():
    """shutdown() closes every open session (R1)."""
    import http.client

    server, thread = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.request(
            "POST", "/mcp", body=json.dumps(_initialize()),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        session_id = resp.getheader("Mcp-Session-Id")
        resp.readline()
        resp.readline()
        resp.readline()
        assert session_id in server._sessions

        server.shutdown()
        thread.join(timeout=5)
        assert server._sessions == {}
        assert server._last_active == {}
    finally:
        conn.close()


def test_socket_timeout_set(streamable_server):
    """R3: a stalled client must not pin a worker thread forever."""
    assert streamable_server.httpd.RequestHandlerClass.timeout == 30


def test_sse_post_during_teardown_answers_409_not_silent_202():
    """Race regression: a POST whose handler outlives the session (DELETE lands
    mid-flight) must answer an explicit 409 — never a 202 whose response is
    queued onto a stream nobody reads (response lost + duplicate tool run)."""
    import http.client

    executed = threading.Event()
    release = threading.Event()

    def slow_handle(registry, request):
        if request.get("id") == 77:  # only the mid-flight request is slow
            executed.set()
            release.wait(timeout=5)
        return {"jsonrpc": "2.0", "id": request.get("id"), "result": {"ok": True}}

    server, thread = _start_server(handle=slow_handle)
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    try:
        # Open the SSE session (first request's response is delivered inline).
        conn.request(
            "POST", "/mcp", body=json.dumps(_initialize()),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        session_id = resp.getheader("Mcp-Session-Id")
        resp.readline()  # event: message
        resp.readline()  # data: ...
        resp.readline()  # blank

        outcome: dict = {}

        def post_slow():
            status, _, _, body = _post(
                server,
                {"jsonrpc": "2.0", "id": 77, "method": "mcp.ping", "params": {}},
                accept="application/json, text/event-stream",
                session_id=session_id,
            )
            outcome["status"] = status
            outcome["body"] = body

        poster = threading.Thread(target=post_slow, daemon=True)
        poster.start()
        assert executed.wait(timeout=5), "handler should be running (tool in flight)"

        # Teardown while the handler is still running.
        conn2 = HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn2.request("DELETE", f"/mcp?session_id={session_id}")
        r2 = conn2.getresponse()
        assert r2.status == 202
        r2.read()
        conn2.close()

        release.set()
        poster.join(timeout=5)
        assert outcome["status"] == 409, f"expected 409, got {outcome}"
        assert b"Session closed" in outcome["body"]

        # The stream closed without ever delivering the response.
        import socket as _socket
        _socket.setdefaulttimeout(5)
        # connection reset is also acceptable
        with contextlib.suppress(OSError):
            assert resp.read() == b"", "stream should close after DELETE"
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)
        conn.close()


def test_concurrency_cap_returns_503():
    """R2: over the cap, id'd requests get 503; notifications bypass the cap."""
    entered = threading.Event()
    release = threading.Event()

    def slow_handle(registry, request):
        entered.set()
        release.wait(timeout=5)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}

    server, thread = _start_server(handle=slow_handle, max_concurrent=1)
    try:
        results = {}

        def first_post():
            results["first"] = _post(server, _initialize(1))

        t1 = threading.Thread(target=first_post)
        t1.start()
        assert entered.wait(timeout=5), "first request should occupy the cap slot"

        status, _, _, body = _post(server, _initialize(2))
        assert status == 503, f"expected 503, got {status}: {body!r}"
        assert b"concurrency" in body.lower()

        status2, _, _, _ = _post(server, {
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
        })
        assert status2 == 202

        release.set()
        t1.join(timeout=5)
        assert results["first"][0] == 200
    finally:
        release.set()
        server.shutdown()
        thread.join(timeout=5)


def test_run_streamable_server_wires_shared_handler_and_clean_shutdown(monkeypatch):
    """_run_streamable_server must inject the shared JSON-RPC handler and shut
    down cleanly on KeyboardInterrupt (Ctrl-C from the CLI) — parity twin of
    the SSE launcher test."""
    import external_llm.editor.agent.mcp.streamable_server as streamable_mod
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

    monkeypatch.setattr(streamable_mod, "StreamableHttpMcpServer", FakeServer)
    mcp_server_mod._run_streamable_server(_make_registry(), "127.0.0.1", 9999)
    assert captured["handle"] is mcp_server_mod._handle_jsonrpc
    assert (captured["host"], captured["port"]) == ("127.0.0.1", 9999)
    assert captured["shutdown"] is True
