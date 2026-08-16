"""
Tests for the stdio MCP transport (external_llm/editor/agent/mcp/server.py).

Hermetic: no network and no claude-agent-sdk.  ``_run_stdio_server`` runs in a
thread against in-memory stdin/stdout substitutes, so the tests can prove the
R11-3 concurrency contract:

- fast protocol methods (initialize / ping) stay responsive while a slow
  mcp.call_tool is still running (they are handled inline, tools run on
  bounded worker threads);
- concurrent tool calls execute in parallel;
- tool concurrency is bounded by _MAX_CONCURRENT_TOOL_CALLS;
- stdin EOF drains in-flight tool calls for a bounded time.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Optional

import pytest

from external_llm.editor.agent.mcp import server as mcp_server_mod

# -- in-memory stdin/stdout substitutes -------------------------------------


class _LineFeed:
    """Feeds JSON-RPC lines to the server; close() ends iteration (EOF)."""

    def __init__(self) -> None:
        self._lines: queue.Queue[Optional[str]] = queue.Queue()

    def __iter__(self) -> "_LineFeed":
        return self

    def __next__(self) -> str:
        line = self._lines.get()
        if line is None:
            raise StopIteration
        return line

    def feed(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put(None)


class _Capture:
    """Thread-safe stdout substitute; every response is written as one line."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            self._lines.put(line)
        return len(s)

    def flush(self) -> None:
        pass

    def read_response(self, timeout: float = 5.0) -> dict:
        return json.loads(self._lines.get(timeout=timeout))


class _Result:
    def __init__(self, *, ok: bool, content: str = "", error: str = "") -> None:
        self.ok = ok
        self.content = content
        self.error = error


class _FakeRegistry:
    """ToolRegistry stand-in whose dispatch blocks until ``release`` is set."""

    repo_language = "python"

    def __init__(self, result: Optional[_Result] = None) -> None:
        self.result = result or _Result(ok=True, content="tool-result")
        self.release = threading.Event()
        self.started = threading.Event()
        self.start_count = 0
        self._lock = threading.Lock()

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> _Result:
        with self._lock:
            self.start_count += 1
        self.started.set()
        self.release.wait(timeout=10)
        return self.result

    def get_tool_schemas(self, lang_filter: Optional[str] = None) -> list:
        return []


class _StdioServer:
    """Runs _run_stdio_server in a daemon thread with in-memory stdio."""

    def __init__(self, registry: _FakeRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        self.feed = _LineFeed()
        self.capture = _Capture()
        monkeypatch.setattr("sys.stdin", self.feed)
        monkeypatch.setattr("sys.stdout", self.capture)
        # The in-process SDK MCP server build is not the transport under test
        # and requires claude-agent-sdk — stub it out.
        monkeypatch.setattr(mcp_server_mod, "build_asr_mcp_server", lambda *a, **k: None)
        self.thread = threading.Thread(
            target=mcp_server_mod._run_stdio_server, args=(registry,), daemon=True
        )
        self.thread.start()

    def request(self, method: str, rid: Optional[int] = None, params: Optional[dict] = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if rid is not None:
            payload["id"] = rid
        if params is not None:
            payload["params"] = params
        self.feed.feed(json.dumps(payload))

    def stop(self) -> None:
        self.feed.close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()


def _call_tool(rid: int, name: str = "bash") -> str:
    return json.dumps({
        "jsonrpc": "2.0",
        "id": rid,
        "method": "mcp.call_tool",
        "params": {"name": name, "arguments": {"cmd": "echo hi"}},
    })


# -- tests ------------------------------------------------------------------


def test_ping_responsive_while_slow_tool_runs(monkeypatch):
    """The core R11-3 regression: a long tool call must not block the protocol."""
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.request("mcp.initialize", rid=1)
    server.feed.feed(_call_tool(2))
    server.request("mcp.ping", rid=3)

    # initialize is inline -> first response.
    init_resp = server.capture.read_response()
    assert init_resp["id"] == 1
    assert init_resp["result"]["serverInfo"]["name"] == "asicode"

    # The ping must be answered while the tool is STILL blocked on release.
    ping_resp = server.capture.read_response()
    assert ping_resp["id"] == 3
    assert not registry.release.is_set()  # tool still running -> loop was not blocked

    # Only now release the tool; its response arrives last.
    registry.release.set()
    tool_resp = server.capture.read_response()
    assert tool_resp["id"] == 2
    assert tool_resp["result"] == {
        "content": [{"type": "text", "text": "tool-result"}],
        "isError": False,
    }
    server.stop()


def test_initialize_and_list_tools_are_inline(monkeypatch):
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.request("mcp.initialize", rid=1)
    init_resp = server.capture.read_response()
    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == mcp_server_mod.MCP_PROTOCOL_VERSION

    server.request("mcp.list_tools", rid=2)
    tools_resp = server.capture.read_response()
    assert tools_resp["id"] == 2
    assert tools_resp["result"] == {"tools": []}
    server.stop()


def test_concurrent_tool_calls_run_in_parallel(monkeypatch):
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed(_call_tool(1))
    server.feed.feed(_call_tool(2))

    # Both dispatches must be in flight before either is released.
    assert registry.started.wait(timeout=2)
    deadline = time.monotonic() + 2
    while registry.start_count < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert registry.start_count == 2

    registry.release.set()
    ids = {server.capture.read_response()["id"] for _ in range(2)}
    assert ids == {1, 2}
    server.stop()


def test_tool_concurrency_is_bounded(monkeypatch):
    monkeypatch.setattr(mcp_server_mod, "_MAX_CONCURRENT_TOOL_CALLS", 1)
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed(_call_tool(1))
    server.feed.feed(_call_tool(2))
    server.feed.feed(_call_tool(3))

    # With a cap of 1 only the first tool may be inside dispatch while the
    # semaphore is held (tools 2 and 3 block on it, not on the read loop).
    assert registry.started.wait(timeout=2)
    time.sleep(0.2)  # settle — no second worker may enter dispatch
    assert registry.start_count == 1

    registry.release.set()
    ids = {server.capture.read_response()["id"] for _ in range(3)}
    assert ids == {1, 2, 3}
    server.stop()


def test_eof_drains_in_flight_tool_calls(monkeypatch):
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed(_call_tool(1))
    assert registry.started.wait(timeout=2)

    server.feed.close()  # EOF — reader loop ends, but the tool is still in flight
    time.sleep(0.2)
    assert server.thread.is_alive()  # drain: still waiting for the tool

    registry.release.set()
    server.stop()
    assert server.capture.read_response()["id"] == 1


def test_parse_error_response(monkeypatch):
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed("{not json")
    resp = server.capture.read_response()
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700
    server.stop()


def test_tool_failure_is_error_result(monkeypatch):
    registry = _FakeRegistry(result=_Result(ok=False, error="boom"))
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed(_call_tool(1))
    registry.release.set()
    resp = server.capture.read_response()
    assert resp["id"] == 1
    assert resp["result"]["isError"] is True
    assert resp["result"]["content"][0]["text"] == "boom"
    server.stop()


def test_raising_dispatch_maps_to_internal_error(monkeypatch):
    registry = _FakeRegistry()

    def _boom(tool_name: str, args: dict) -> None:
        raise RuntimeError("kaput")

    registry.dispatch = _boom  # type: ignore[method-assign]
    server = _StdioServer(registry, monkeypatch)

    server.feed.feed(_call_tool(1))
    resp = server.capture.read_response()
    assert resp["id"] == 1
    assert resp["error"]["code"] == -32603
    assert "kaput" in resp["error"]["message"]
    server.stop()


def test_notification_call_tool_still_dispatched(monkeypatch):
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)

    server.request("mcp.call_tool", params={"name": "bash", "arguments": {}})
    assert registry.started.wait(timeout=2)
    registry.release.set()
    resp = server.capture.read_response()
    assert resp["id"] is None  # notifications get an id-null ack, same as before
    assert resp["result"]["isError"] is False
    server.stop()


def test_run_mcp_server_boot_freshness_self_check_logs(monkeypatch, caplog):
    """R12-2: the MCP server logs a scanner freshness self-check at boot.

    A long-lived server keeps executing the scanner code it imported at
    startup; the boot check makes "loaded code != on-disk source" visible in
    the server log (the authoritative per-invocation detector lives in the
    structural-scan tool handler).
    """
    monkeypatch.setattr(mcp_server_mod, "_run_stdio_server", lambda registry: None)
    with caplog.at_level(
        logging.INFO, logger="external_llm.editor.agent.mcp.server",
    ):
        mcp_server_mod.run_mcp_server(_FakeRegistry(), mode="stdio")
    assert "freshness self-check" in caplog.text


def test_run_mcp_server_unknown_mode_exits(monkeypatch):
    """Unknown mode still exits via sys.exit after the boot self-check."""
    with pytest.raises(SystemExit):
        mcp_server_mod.run_mcp_server(_FakeRegistry(), mode="bogus")


# ── RED→GREEN: uncovered branches ────────────────────────────────────────────


class _BoomStdout:
    """stdout stand-in whose write always raises OSError (closed pipe)."""

    def write(self, s: str) -> int:
        raise OSError("stdout closed")

    def flush(self) -> None:
        pass


def test_empty_line_skipped(monkeypatch):
    """A blank line on stdin is skipped, not parsed (L199)."""
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)
    server.feed.feed("")
    server.request("mcp.initialize", rid=1)
    resp = server.capture.read_response()
    assert resp["id"] == 1
    server.stop()


def test_tool_call_handler_raise_maps_to_internal_error(monkeypatch):
    """_run_tool_call's defensive except maps a handler crash to a JSON-RPC
    internal error (L181-182)."""
    registry = _FakeRegistry()

    def _boom(reg, request):
        raise RuntimeError("tool boom")

    monkeypatch.setattr(mcp_server_mod, "_handle_jsonrpc", _boom)
    server = _StdioServer(registry, monkeypatch)
    server.feed.feed(_call_tool(1))
    resp = server.capture.read_response()
    assert resp["id"] == 1
    assert resp["error"]["code"] == -32603
    assert "tool boom" in resp["error"]["message"]
    server.stop()


def test_inline_handler_raise_maps_to_internal_error(monkeypatch):
    """The inline path's defensive except maps a handler crash to a JSON-RPC
    internal error (L228-229 + _rpc_error body)."""
    registry = _FakeRegistry()

    def _boom(reg, request):
        raise RuntimeError("inline boom")

    monkeypatch.setattr(mcp_server_mod, "_handle_jsonrpc", _boom)
    server = _StdioServer(registry, monkeypatch)
    server.request("mcp.ping", rid=1)
    resp = server.capture.read_response()
    assert resp["id"] == 1
    assert resp["error"]["code"] == -32603
    assert "inline boom" in resp["error"]["message"]
    server.stop()


def test_tool_response_write_oserror_logged(monkeypatch):
    """A tool response write on a closed stdout is swallowed, not a crash
    (L185-188)."""
    feed = _LineFeed()
    monkeypatch.setattr("sys.stdin", feed)
    monkeypatch.setattr("sys.stdout", _BoomStdout())
    monkeypatch.setattr(mcp_server_mod, "build_asr_mcp_server", lambda *a, **k: None)
    registry = _FakeRegistry()
    thread = threading.Thread(
        target=mcp_server_mod._run_stdio_server, args=(registry,), daemon=True
    )
    thread.start()
    feed.feed(_call_tool(1))
    assert registry.started.wait(timeout=2)
    registry.release.set()  # dispatch returns → _write hits the OSError
    time.sleep(0.2)
    feed.close()  # EOF → drain (tool thread already done)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_inline_write_oserror_breaks_loop(monkeypatch):
    """An inline response write on a closed stdout ends the read loop
    (L232-234)."""
    feed = _LineFeed()
    monkeypatch.setattr("sys.stdin", feed)
    monkeypatch.setattr("sys.stdout", _BoomStdout())
    monkeypatch.setattr(mcp_server_mod, "build_asr_mcp_server", lambda *a, **k: None)
    thread = threading.Thread(
        target=mcp_server_mod._run_stdio_server, args=(_FakeRegistry(),), daemon=True
    )
    thread.start()
    feed.feed(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "mcp.ping"}))
    thread.join(timeout=5)
    assert not thread.is_alive()  # broke out of the loop without EOF


def test_keyboard_interrupt_exits_promptly(monkeypatch):
    """Ctrl+C exits promptly without draining in-flight calls (L235-238)."""

    class _KbFeed:
        def __iter__(self):
            return self

        def __next__(self):
            raise KeyboardInterrupt()

    monkeypatch.setattr("sys.stdin", _KbFeed())
    monkeypatch.setattr("sys.stdout", _Capture())
    monkeypatch.setattr(mcp_server_mod, "build_asr_mcp_server", lambda *a, **k: None)
    with pytest.raises(KeyboardInterrupt):
        mcp_server_mod._run_stdio_server(_FakeRegistry())


def test_eof_drain_timeout_warns_and_exits(monkeypatch, caplog):
    """EOF with a still-running tool logs a bounded-timeout warning instead
    of waiting forever (L250/L255)."""
    monkeypatch.setattr(mcp_server_mod, "_DRAIN_TIMEOUT_SECONDS", 0.1)
    registry = _FakeRegistry()
    server = _StdioServer(registry, monkeypatch)
    server.feed.feed(_call_tool(1))
    assert registry.started.wait(timeout=2)
    server.feed.close()  # EOF → drain; tool stays blocked on release
    with caplog.at_level(logging.WARNING, logger="external_llm.editor.agent.mcp.server"):
        server.thread.join(timeout=5)
    assert not server.thread.is_alive()
    assert any(
        "in-flight tool call(s) still running" in r.message for r in caplog.records
    )


def test_run_mcp_server_sse_and_http_modes_dispatch(monkeypatch):
    """run_mcp_server routes sse/http modes to their launchers (L105/L107)."""
    calls: list[str] = []
    monkeypatch.setattr(mcp_server_mod, "_run_sse_server", lambda *a, **k: calls.append("sse"))
    monkeypatch.setattr(
        mcp_server_mod, "_run_streamable_server", lambda *a, **k: calls.append("http")
    )
    mcp_server_mod.run_mcp_server(_FakeRegistry(), mode="sse")
    mcp_server_mod.run_mcp_server(_FakeRegistry(), mode="http")
    assert calls == ["sse", "http"]


def test_boot_freshness_self_check_stale_warns(monkeypatch, caplog):
    """Stale scanner sources at boot produce a restart warning (L69)."""

    class _StaleRegistry:
        def verify_loaded_sources(self):
            return ["external_llm/agent/structural_scanners/x.py"]

        def source_versions(self):
            return {}

    monkeypatch.setattr(
        "external_llm.agent.scanner_registry.get_registry", lambda: _StaleRegistry()
    )
    with caplog.at_level(logging.WARNING, logger="external_llm.editor.agent.mcp.server"):
        mcp_server_mod._log_scanner_freshness_at_startup()
    assert "stale source" in caplog.text
