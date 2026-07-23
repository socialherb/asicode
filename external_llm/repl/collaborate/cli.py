"""
CLI entry point for asicode ↔ Claude Code Agent collaboration.

Usage:
    python -m external_llm.repl.collaborate.cli collaborate \
        --task "Analyze this PR for security issues" \
        --verbose

    python -m external_llm.repl.collaborate.cli mcp list
    python -m external_llm.repl.collaborate.cli mcp start
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the collaboration CLI."""
    parser = argparse.ArgumentParser(
        description="asicode x Claude Code Agent Collaboration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s collaborate --task "Review this change for bugs"
  %(prog)s collaborate --task "Find unused code" --no-digest
  %(prog)s collaborate --task "Design the auth module" --file session.log
  %(prog)s mcp list
  %(prog)s mcp start
        """,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Subcommand",
    )

    # ── collaborate subcommand ────────────────────────────────
    collab_parser = subparsers.add_parser(
        "collaborate",
        help="Run a collaboration session with Claude Code Agent",
    )
    collab_parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Task description for Claude Code Agent",
    )
    collab_parser.add_argument(
        "--context", "-c",
        type=str,
        help="Optional additional context",
    )
    collab_parser.add_argument(
        "--model", "-m",
        type=str,
        help=(
            "Claude Code model to use. Default: claude-sonnet-4-20250514 (bundled CLI default). "
            "Examples: claude-sonnet-5, claude-sonnet-4-5, claude-opus-4-5, claude-opus-4-20250514. "
            "Can also set via CLAUDE_MODEL env var."
        ),
    )
    collab_parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Max turns for Claude Code Agent (default: 100)",
    )
    collab_parser.add_argument(
        "--no-digest",
        action="store_true",
        help="Skip asicode preprocessing digest",
    )
    collab_parser.add_argument(
        "--file", "-f",
        type=str,
        help="Output file for session logs",
    )
    collab_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress streaming display (only show verdict)",
    )
    # ── mcp subcommand ────────────────────────────────────────
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="MCP server management",
    )
    mcp_subparsers = mcp_parser.add_subparsers(
        dest="mcp_command",
        required=True,
        help="MCP subcommand",
    )

    mcp_subparsers.add_parser(
        "list",
        help="List all available tools for MCP exposure",
    )

    mcp_start_parser = mcp_subparsers.add_parser(
        "start",
        help="Start MCP server",
    )
    mcp_start_parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="Server mode (default: stdio)",
    )
    mcp_start_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE mode (default: 8765)",
    )
    mcp_start_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for SSE mode (default: 127.0.0.1)",
    )

    mcp_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    if args.command == "collaborate":
        _run_collaborate(args)
    elif args.command == "mcp":
        _run_mcp(args)


def _run_collaborate(args: argparse.Namespace) -> None:
    """Execute the collaborate subcommand."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.repl.collaborate import (
        DEFAULT_COLLAB_MODEL,
        CollaborationOrchestratorConfig,
    )
    from external_llm.repl.collaborate.streaming_display import StreamingDisplay

    # Get repo root (CWD or env)
    repo_root = os.getcwd()

    # Initialize ToolRegistry
    # Trusted local CLI: allow read tools to cross the repo boundary (the bash
    # tool already can). Same trust level applies to the local MCP-server path below.
    agent_config = AgentConfig(unrestricted_read=True)
    registry = ToolRegistry(repo_root=repo_root, config=agent_config)

    # Setup streaming display
    display = StreamingDisplay(
        verbose=args.verbose,
        output_file=args.file,
    )
    model = args.model or DEFAULT_COLLAB_MODEL
    display.print_header(args.task, model=model)

    # Build orchestrator config
    orch_config = CollaborationOrchestratorConfig(
        max_turns_per_iteration=args.max_turns,
        event_callback=display.handle_event if not args.quiet else None,
        repo_root=repo_root,
        model=model,
    )

    try:
        # Run collaboration
        result = asyncio.run(_run_async(registry, orch_config, args))

        # Print summary
        display.print_summary()
        display.flush_log()

        # Print verdict to stdout for scripting
        if args.quiet:
            print(result.verdict.to_dict())

    except KeyboardInterrupt:
        print(f"\n{chr(27)}[33mInterrupted by user{chr(27)}[0m", file=sys.stderr)
        sys.exit(130)
    except Exception as ex:
        logger.exception("Collaboration failed")
        print(f"\n{chr(27)}[31mError: {ex}{chr(27)}[0m", file=sys.stderr)
        sys.exit(1)
    finally:
        display.stop()  # Prevent ticker from covering stderr messages on interrupt/error


async def _run_async(
    registry: Any,
    config: Any,
    args: argparse.Namespace,
) -> Any:
    """Run the async collaboration session."""
    from external_llm.repl.collaborate import CollaborationOrchestrator

    async with CollaborationOrchestrator(registry, config) as orch:
        result = await orch.run(
            task=args.task,
            context=args.context,
            enable_preprocessing=not args.no_digest,
        )
        return result


def _run_mcp(args: argparse.Namespace) -> None:
    """Execute the mcp subcommand."""
    from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
    from external_llm.editor.agent.mcp import list_mcp_tools, run_mcp_server

    repo_root = os.getcwd()

    if args.mcp_command == "list":
        tools = list_mcp_tools()
        print(f"asicode MCP Tools ({len(tools)} total):")
        print("-" * 60)
        for t in tools:
            params = t.get("parameters", {}).get("properties", {})
            param_str = ", ".join(params.keys()) if params else "(no params)"
            print(f"  {t['name']}({param_str})")
            print(f"    {t['description'][:100]}")
            print()

    elif args.mcp_command == "start":
        # Local MCP server serving the user's own MCP client — same trust level as
        # the interactive CLI above, so read tools may cross the repo boundary.
        agent_config = AgentConfig(unrestricted_read=True)
        registry = ToolRegistry(repo_root=repo_root, config=agent_config)

        print(f"Starting asicode MCP server ({args.mode} mode)...")
        run_mcp_server(
            registry=registry,
            mode=args.mode,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
