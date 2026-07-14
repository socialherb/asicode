"""
MCP (Model Context Protocol) subcommand package.

Provides the 'asicode mcp' subcommand for starting MCP servers
that expose asicode tools to any MCP-compatible client.

Currently supports in-process MCP via Claude Agent SDK.
Future: stdio-mode MCP server for external MCP clients.
"""

from external_llm.editor.agent.mcp.server import list_mcp_tools, run_mcp_server

__all__ = ["list_mcp_tools", "run_mcp_server"]
