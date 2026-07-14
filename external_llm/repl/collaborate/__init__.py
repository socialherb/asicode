"""
asicode ↔ Claude Code Agent Collaboration Package.

Provides in-process MCP integration via Claude Agent SDK's @tool decorator
and create_sdk_mcp_server(), plus a CollaborationOrchestrator that manages
the two-way collaboration flow.

Usage:
    from external_llm.repl.collaborate import CollaborationOrchestrator, CollaborationOrchestratorConfig
    async with CollaborationOrchestrator(registry, CollaborationOrchestratorConfig()) as orch:
        verdict = await orch.run("Analyze this PR for security issues")
"""

from .asi_mcp_adapter import (
    build_asr_mcp_server,
    build_collaborate_install_command,
    build_collaborate_install_spec,
    is_claude_sdk_installed,
)
from .claude_session import ClaudeSession
from .collaboration_orchestrator import (
    DEFAULT_COLLAB_MODEL,
    CollaborationOrchestrator,
    CollaborationOrchestratorConfig,
    build_session_handoff,
    format_verdict_for_session,
)
from .verdict import CollaborationVerdict

# Backward-compat alias — the collaboration module's OrchestratorConfig was renamed
# to CollaborationOrchestratorConfig. Prevents confusion with external_llm.agent.orchestrator.OrchestratorConfig (for SubAgent).
OrchestratorConfig = CollaborationOrchestratorConfig

__all__ = [
    "DEFAULT_COLLAB_MODEL",
    "ClaudeSession",
    "CollaborationOrchestrator",
    "CollaborationOrchestratorConfig",
    "CollaborationVerdict",
    "build_asr_mcp_server",
    "build_collaborate_install_command",
    "build_collaborate_install_spec",
    "build_session_handoff",
    "format_verdict_for_session",
    "is_claude_sdk_installed",
]
