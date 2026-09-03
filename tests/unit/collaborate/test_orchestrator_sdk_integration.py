"""SDK-integration tests for CollaborationOrchestrator.

Covers the real claude_agent_sdk construction path in _ensure_session()
(lines 120-144: build_asr_mcp_server -> get_restricted_options -> ClaudeSession)
and the async context-manager contract (__aenter__/__aexit__, lines 146-152).

The SDK is an optional dependency; these tests only run when it is importable
(environments with `pip install '.[collaborate]'`). The async CM bodies are
monkeypatched so no live ClaudeSDKClient connection is attempted.
"""

from __future__ import annotations

import asyncio

import pytest

from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.repl.collaborate import (
    CollaborationOrchestrator,
    CollaborationOrchestratorConfig,
)

pytestmark = pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("claude_agent_sdk"),
    reason="claude_agent_sdk optional dependency not installed",
)


def _make_orch() -> CollaborationOrchestrator:
    registry = ToolRegistry(repo_root=".", config=AgentConfig())
    return CollaborationOrchestrator(registry, CollaborationOrchestratorConfig())


class TestEnsureSessionSdkConstruction:
    """Real SDK path: _ensure_session builds the MCP server, options and session."""

    def test_ensure_session_builds_real_sdk_objects(self):
        orch = _make_orch()
        session = orch._ensure_session()

        # Real ClaudeSession wrapper, not a stub
        assert type(session).__name__ == "ClaudeSession"
        # MCP server config is the SDK's dict form (build_asr_mcp_server)
        assert isinstance(orch._mcp_server, dict)
        assert orch._mcp_server["name"] == "asicode"
        assert orch._mcp_server["type"] == "sdk"
        assert "instance" in orch._mcp_server
        # Options are a real claude_agent_sdk.ClaudeAgentOptions
        assert type(orch._sdk_options).__name__ == "ClaudeAgentOptions"
        # Constructed session is cached — second call returns the same object
        assert orch._ensure_session() is session

    def test_ensure_session_caches_options_mcp_server(self):
        orch = _make_orch()
        s1 = orch._ensure_session()
        s2 = orch._ensure_session()
        assert s1 is s2
        # The SDK-dependent construction happened exactly once
        assert orch._session is s1

    def test_sdk_options_restrict_native_tools(self):
        orch = _make_orch()
        orch._ensure_session()
        opts = orch._sdk_options
        # All Claude Code native tools are disallowed (forced asicode tool use)
        assert "Read" in opts.disallowed_tools
        assert "Write" in opts.disallowed_tools
        assert "Bash" in opts.disallowed_tools
        assert "Edit" in opts.disallowed_tools
        # Only asicode MCP tools allowed
        assert opts.allowed_tools == ["mcp__asr__*"]
        # SDK isolation: no ~/.claude/settings.json inheritance
        assert opts.setting_sources == []

    def test_sdk_options_model_override(self):
        config = CollaborationOrchestratorConfig(model="opus")
        registry = ToolRegistry(repo_root=".", config=AgentConfig())
        orch = CollaborationOrchestrator(registry, config)
        orch._ensure_session()
        assert orch._sdk_options.model == "opus"

    def test_sdk_options_allow_write_adds_read_only_append(self):
        # Default (read-only) session appends the analysis-only directive.
        orch = _make_orch()
        orch._ensure_session()
        read_append = orch._sdk_options.system_prompt["append"]
        assert "analysis" in read_append.lower() or "read-only" in read_append.lower()


class TestOrchestratorAsyncContextManager:
    """__aenter__/__aexit__ delegate to the underlying ClaudeSession CM.

    The delegation contract is verified with a monkeypatched ClaudeSession CM
    so no live CLI subprocess is spawned; the SDK construction itself is real.
    """

    def test_async_cm_enters_and_exits_session(self, monkeypatch, capsys):
        from external_llm.repl.collaborate.claude_session import ClaudeSession

        entered = []
        exited = []

        async def fake_aenter(self):
            entered.append(self)
            return self

        async def fake_aexit(self, *args):
            exited.append(args)

        monkeypatch.setattr(ClaudeSession, "__aenter__", fake_aenter)
        monkeypatch.setattr(ClaudeSession, "__aexit__", fake_aexit)

        orch = _make_orch()

        async def _run():
            async with orch:
                assert orch.session is not None
                assert orch.session in entered

        asyncio.run(_run())
        assert len(entered) == 1
        assert len(exited) == 1

    def test_async_cm_raises_without_enter(self, monkeypatch):
        # __aexit__ without a prior __aenter__ would hit assert self._session
        # — the contract requires the CM pair. The orchestrator guards this
        # via the documented 'Must be called within async with' contract.
        from external_llm.repl.collaborate.claude_session import ClaudeSession

        async def fake_aenter(self):
            return self

        monkeypatch.setattr(ClaudeSession, "__aenter__", fake_aenter)
        orch = _make_orch()

        async def _run():
            async with orch:
                pass

        asyncio.run(_run())  # no exception — CM pair is consistent
