"""
MCP Server — entry point for exposing asicode tools as an MCP server.

Two modes:
  1. SDK in-process MCP (default): built via create_sdk_mcp_server(),
     used internally by CollaborationOrchestrator.
  2. Standalone stdio/SSE MCP server: for external MCP clients, via the
     'asicode mcp start --mode stdio|sse' CLI subcommand.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from external_llm.agent.tool_registry import ToolRegistry
from external_llm.repl.collaborate.asi_mcp_adapter import build_asr_mcp_server

logger = logging.getLogger(__name__)

# MCP protocol version spoken by the hand-rolled stdio/SSE transports.
# 2025-03-26 is the HTTP+SSE transport era revision; newer Streamable-HTTP
# clients negotiate it down during the initialize exchange.
MCP_PROTOCOL_VERSION = "2025-03-26"

# Long-running mcp.call_tool requests run on bounded daemon worker threads so
# the reader loop keeps answering pings / list_tools / initialize while a tool
# executes.  Without this one slow tool would block the whole JSON-RPC
# protocol (R11-3).  The cap prevents a pipelining client from spawning
# unbounded threads.
_MAX_CONCURRENT_TOOL_CALLS = 8

# After stdin EOF the server waits up to this long for in-flight tool calls so
# their responses are not cut off mid-write (some clients close stdin but keep
# reading stdout).  Bounded — a hung tool cannot wedge shutdown forever.
_DRAIN_TIMEOUT_SECONDS = 10.0


def _rpc_error(request: Any, ex: Exception) -> dict:
    """JSON-RPC internal-error response for an unhandled exception."""
    return {
        "jsonrpc": "2.0",
        "id": request.get("id") if isinstance(request, dict) else None,
        "error": {"code": -32603, "message": str(ex)},
    }


def _log_scanner_freshness_at_startup() -> None:
    """Boot self-check: log whether loaded scanner code matches on-disk source.

    R12-2: the authoritative detector is the per-invocation check in
    ``_tool_run_structural_scan`` (it covers every host that dispatches the
    tool). This boot log is the startup signal for the MCP server itself —
    normally clean because the modules were just imported, but it surfaces a
    server started from a checkout whose scanner files were subsequently
    edited before serving traffic, or fingerprinting gaps.
    """
    try:
        from external_llm.agent.scanner_registry import get_registry

        registry = get_registry()
        stale = registry.verify_loaded_sources()
        if stale:
            logger.warning(
                "[SCANNER_REGISTRY] freshness self-check: %d module(s) loaded "
                "with stale source (changed on disk after load) — restart "
                "required: %s",
                len(stale),
                ", ".join(sorted(stale)),
            )
        else:
            logger.info(
                "[SCANNER_REGISTRY] freshness self-check: %d scanner module(s) loaded, all match on-disk source",
                len(registry.source_versions()),
            )
    except Exception:  # pragma: no cover - best-effort diagnostics
        logger.debug(
            "[SCANNER_REGISTRY] freshness self-check unavailable",
            exc_info=True,
        )


def run_mcp_server(
    registry: ToolRegistry,
    mode: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run the asicode MCP server.

    Args:
        registry: Initialized ToolRegistry.
        mode: Server mode ('stdio' for CLI-based, 'sse' for HTTP-based).
        host: Host for SSE mode.
        port: Port for SSE mode.
    """
    _log_scanner_freshness_at_startup()
    if mode == "stdio":
        _run_stdio_server(registry)
    elif mode == "sse":
        _run_sse_server(registry, host, port)
    elif mode in ("http", "streamable-http"):
        _run_streamable_server(registry, host, port)
    else:
        print(
            f"Unknown MCP mode: {mode}. Use 'stdio', 'sse', or 'http'.",
            file=sys.stderr,
        )
        sys.exit(1)


def list_mcp_tools(
    registry: ToolRegistry | None = None,
) -> list[dict[str, Any]]:
    """List all tools exposed by the MCP server."""
    from external_llm.agent.tool_schemas import AGENT_TOOL_SCHEMAS
    from external_llm.repl.collaborate.asi_mcp_adapter import _EXCLUDED_TOOLS as EXCLUDED

    # When registry is available, use its lang-filtered schemas to hide
    # Python-only tools (edit_ast, run_structural_scan) in non-Python repos.
    if registry is not None:
        schemas = registry.get_tool_schemas(lang_filter=registry.repo_language)
    else:
        schemas = AGENT_TOOL_SCHEMAS

    tools = []
    for schema in schemas:
        name = schema["name"]
        if name in EXCLUDED:
            continue
        tools.append(
            {
                "name": name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            }
        )
    return tools


def _run_stdio_server(registry: ToolRegistry) -> None:
    """Run MCP server in stdio mode (read JSON from stdin, write to stdout).

    This is a lightweight MCP server that exchanges JSON-RPC messages
    over stdio, compatible with Claude Code's external MCP server protocol.

    Concurrency (R11-3): fast protocol methods (mcp.initialize /
    mcp.list_tools / mcp.ping, unknown methods, notifications) are handled
    inline in the read loop, so they stay responsive while tools run.
    mcp.call_tool is dispatched on daemon worker threads capped by
    ``_MAX_CONCURRENT_TOOL_CALLS`` — concurrent tool calls from a pipelining
    client no longer serialize, and one long-running tool no longer blocks
    the rest of the protocol.  Tool responses may therefore arrive out of
    request order; JSON-RPC clients match them by id.
    """
    logger.info("Starting asicode MCP server (stdio mode)")

    # Build the MCP server config (in-process)
    build_asr_mcp_server(registry, server_name="asicode")

    stdout_lock = threading.Lock()
    tool_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_TOOL_CALLS)
    in_flight: set[threading.Thread] = set()
    in_flight_lock = threading.Lock()
    counter = itertools.count()

    def _write(response: dict) -> None:
        """Write one response line; the lock keeps concurrent writers atomic."""
        with stdout_lock:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def _run_tool_call(request: dict) -> None:
        """Worker: dispatch mcp.call_tool and write its response."""
        try:
            with tool_slots:
                try:
                    response = _handle_jsonrpc(registry, request)
                except Exception as ex:  # defensive — _handle_jsonrpc never raises
                    response = _rpc_error(request, ex)
                try:
                    _write(response)
                except OSError:
                    # Client's stdout pipe is gone (process teardown) — nothing
                    # left to deliver; the reader loop ends on stdin EOF anyway.
                    logger.debug("MCP stdio: stdout closed while writing tool response")
        finally:
            with in_flight_lock:
                in_flight.discard(threading.current_thread())

    drain = True
    try:
        # Read JSON-RPC requests from stdin, process, write responses to stdout
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    }
                )
                continue

            method = request.get("method", "") if isinstance(request, dict) else ""
            if method == "mcp.call_tool":
                # Long-running dispatch goes off the read loop.  Bounded by the
                # semaphore, so a pipelining client cannot spawn unbounded
                # threads; requests beyond the cap queue on the semaphore while
                # the read loop stays responsive.
                thread = threading.Thread(
                    target=_run_tool_call,
                    args=(request,),
                    name=f"mcp-stdio-tool-{next(counter)}",
                    daemon=True,
                )
                with in_flight_lock:
                    in_flight.add(thread)
                thread.start()
            else:
                try:
                    response = _handle_jsonrpc(registry, request)
                except Exception as ex:  # defensive — _handle_jsonrpc never raises
                    response = _rpc_error(request, ex)
                try:
                    _write(response)
                except OSError:
                    logger.debug("MCP stdio: stdout closed; shutting down")
                    break
    except KeyboardInterrupt:
        # Ctrl+C: exit promptly — do not wait for in-flight tool calls.
        drain = False
        raise
    finally:
        if drain:
            # stdin EOF (or a read error): drain in-flight tool calls for a
            # bounded time so their responses are not cut off mid-write.
            deadline = time.monotonic() + _DRAIN_TIMEOUT_SECONDS
            while True:
                with in_flight_lock:
                    remaining = list(in_flight)
                if not remaining:
                    break
                if time.monotonic() >= deadline:
                    logger.warning(
                        "MCP stdio: %d in-flight tool call(s) still running after %.0fs; exiting without them",
                        len(remaining),
                        _DRAIN_TIMEOUT_SECONDS,
                    )
                    break
                for thread in remaining:
                    thread.join(timeout=0.05)


def _handle_jsonrpc(registry: ToolRegistry, request: Any) -> dict:
    """Handle one JSON-RPC request and return the response dict.

    Shared by the stdio and SSE transports so both expose identical
    protocol semantics (mcp.initialize / mcp.list_tools / mcp.call_tool).
    Never raises — every failure maps to a JSON-RPC error response.
    """
    if not isinstance(request, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")
    try:
        if method == "mcp.list_tools":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": list_mcp_tools(registry)},
            }
        if method == "mcp.call_tool":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _dispatch_tool(registry, tool_name, args),
            }
        if method == "mcp.initialize":
            # Spec-correct shape (protocolVersion / serverInfo) plus the
            # legacy flat keys the original stdio server emitted — additive,
            # so both strict MCP clients and older hand-rolled clients work.
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "asicode", "version": "1.0.0"},
                    "server_name": "asicode",
                    "version": "1.0.0",
                },
            }
    except Exception as ex:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": str(ex)},
        }
    else:
        # Health check or unknown method (notifications included — the stdio
        # transport has always acked every message; SSE only forwards replies
        # for requests with an id).
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"status": "ok"},
        }


def _dispatch_tool(registry: ToolRegistry, tool_name: str, args: dict) -> dict:
    """Dispatch a tool call and return MCP-compatible result."""
    result = registry.dispatch(tool_name, args)
    if result.ok:
        return {
            "content": [
                {"type": "text", "text": result.content or ""},
            ],
            "isError": False,
        }
    return {
        "content": [
            {"type": "text", "text": result.error or "Tool failed"},
        ],
        "isError": True,
    }


def _run_http_server(
    registry: ToolRegistry,
    host: str,
    port: int,
    *,
    server_factory: Callable[..., Any],
    endpoint: str,
    label: str,
) -> None:
    """Shared launcher for the two HTTP transports (SSE, Streamable-HTTP).

    The transports differ only in their server class, the endpoint path and
    the display label; everything else — constructing the server with the
    shared JSON-RPC handler, the startup/shutdown stderr prints and the
    KeyboardInterrupt handling — is identical.
    """
    server = server_factory(registry, host, port, handle=_handle_jsonrpc)
    print(
        f"Starting asicode MCP server ({label} mode) on http://{host}:{port}{endpoint}",
        file=sys.stderr,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"Shutting down asicode MCP server ({label} mode)", file=sys.stderr)
    finally:
        server.shutdown()


def _run_sse_server(registry: ToolRegistry, host: str, port: int) -> None:
    """Run MCP server in SSE mode (HTTP-based).

    Implements the MCP HTTP+SSE transport (protocol 2025-03-26) with the
    standard library (see sse_server.py): clients open ``GET /sse``, receive
    an ``endpoint`` event, then POST JSON-RPC messages whose responses arrive
    as ``message`` events on the stream.  Shares the JSON-RPC handler with
    stdio mode so both transports expose identical protocol semantics.
    """
    from external_llm.editor.agent.mcp.sse_server import SSEMcpServer

    _run_http_server(
        registry,
        host,
        port,
        server_factory=SSEMcpServer,
        endpoint="/sse",
        label="SSE",
    )


def _run_streamable_server(registry: ToolRegistry, host: str, port: int) -> None:
    """Run MCP server in Streamable-HTTP mode (P5-4).

    Implements the MCP Streamable-HTTP transport (protocol 2025-11-05) with
    the standard library (see streamable_server.py): a single ``POST /mcp``
    endpoint serves both JSON and SSE response modes, selected by the
    client's ``Accept`` header.  Shares the JSON-RPC handler with the stdio
    and SSE transports so all three expose identical protocol semantics.
    """
    from external_llm.editor.agent.mcp.streamable_server import StreamableHttpMcpServer

    _run_http_server(
        registry,
        host,
        port,
        server_factory=StreamableHttpMcpServer,
        endpoint="/mcp",
        label="Streamable-HTTP",
    )
