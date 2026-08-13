"""
MCP (Model Context Protocol) subcommand package.

Provides the 'asicode mcp' subcommand for starting MCP servers
that expose asicode tools to any MCP-compatible client.

Supports:
  - in-process MCP via Claude Agent SDK (CollaborationOrchestrator),
  - standalone stdio MCP server ('asicode mcp start --mode stdio'),
  - standalone SSE MCP server ('asicode mcp start --mode sse') for
    external MCP clients (Claude Desktop, IDE extensions, ...).
"""

from external_llm.editor.agent.mcp.server import list_mcp_tools, run_mcp_server

__all__ = ["list_mcp_tools", "run_mcp_server"]
