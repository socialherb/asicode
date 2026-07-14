"""
MCP Server — entry point for exposing asicode tools as an MCP server.

Two modes:
  1. SDK in-process MCP (default): built via create_sdk_mcp_server(),
     used internally by CollaborationOrchestrator.
  2. Standalone stdio MCP server: for external MCP clients (future).
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from external_llm.agent.tool_registry import ToolRegistry
from external_llm.repl.collaborate.asi_mcp_adapter import build_asr_mcp_server

logger = logging.getLogger(__name__)


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
    if mode == "stdio":
        _run_stdio_server(registry)
    elif mode == "sse":
        _run_sse_server(registry, host, port)
    else:
        print(f"Unknown MCP mode: {mode}. Use 'stdio' or 'sse'.", file=sys.stderr)
        sys.exit(1)


def list_mcp_tools(
    registry: Optional[ToolRegistry] = None,
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
        tools.append({
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {}),
        })
    return tools


def _run_stdio_server(registry: ToolRegistry) -> None:
    """Run MCP server in stdio mode (read JSON from stdin, write to stdout).

    This is a lightweight MCP server that exchanges JSON-RPC messages
    over stdio, compatible with Claude Code's external MCP server protocol.
    """
    logger.info("Starting asicode MCP server (stdio mode)")

    # Build the MCP server config (in-process)
    build_asr_mcp_server(registry, server_name="asicode")

    # Read JSON-RPC requests from stdin, process, write responses to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id")

            if method == "mcp.list_tools":
                tools = list_mcp_tools(registry)
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools},
                }
            elif method == "mcp.call_tool":
                tool_name = params.get("name", "")
                args = params.get("arguments", {})
                result = _dispatch_tool(registry, tool_name, args)
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }
            elif method == "mcp.initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "server_name": "asicode",
                        "version": "1.0.0",
                        "capabilities": {"tools": {}},
                    },
                }
            else:
                # Health check or unknown method
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"status": "ok"},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as ex:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32603, "message": str(ex)},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


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
    else:
        return {
            "content": [
                {"type": "text", "text": result.error or "Tool failed"},
            ],
            "isError": True,
        }


def _run_sse_server(registry: ToolRegistry, host: str, port: int) -> None:
    """Run MCP server in SSE mode (HTTP-based).

    Not yet implemented. Use stdio mode for now.
    """
    raise NotImplementedError(
        "SSE mode not yet implemented. Use 'stdio' mode."
    )
