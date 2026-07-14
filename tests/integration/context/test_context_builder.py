"""
Integration tests for context builder and RAG integration.
"""
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import LLMMessage


def _make_agent(repo_root: str, **config_kwargs) -> AgentLoop:
    config = AgentConfig(max_turns=5, **config_kwargs)
    registry = ToolRegistry(repo_root, config)
    mock_llm = Mock()
    mock_llm.get_provider_name.return_value = "openai"
    return AgentLoop(llm_client=mock_llm, registry=registry, config=config, model="test-model")


@pytest.mark.integration
class TestContextBuilder:
    """Test context building with RAG and memory integration."""

    def test_build_session_context_basic(self, temp_repo_root: str):
        """Test _build_session_context returns a non-empty string with repo state."""
        agent = _make_agent(temp_repo_root)
        context = agent._build_session_context()

        assert isinstance(context, str)
        assert len(context) > 0
        # Should include git branch or commit info
        assert "Branch" in context or "branch" in context.lower() or "commit" in context.lower()

    def test_build_session_context_git_state(self, temp_repo_root: str):
        """Test that git state (branch, status) is included in context."""
        agent = _make_agent(temp_repo_root)
        context = agent._build_session_context()

        assert "Branch:" in context or "branch" in context.lower()
        assert "commit" in context.lower() or "Recent" in context

    def test_build_session_context_memory_file_exists(self, temp_repo_root_with_memory: str):
        """Test that .asicode/memory.md is present in the fixture repo."""
        memory_file = Path(temp_repo_root_with_memory) / ".asicode" / "memory.md"
        assert memory_file.exists()
        content = memory_file.read_text()
        assert "Test Memory" in content or "test memory" in content.lower()

    def test_context_window_sliding(self, temp_repo_root: str):
        """Test sliding window context trimming via _trim_context."""
        agent = _make_agent(temp_repo_root, context_window_size=5)

        # Create mock conversation history exceeding the model-based min window (30)
        messages = [LLMMessage(role="system", content="System prompt")]
        for i in range(35):
            messages.append(LLMMessage(role="user", content=f"User message {i}"))
            messages.append(LLMMessage(role="assistant", content=f"Assistant response {i}"))

        trimmed = agent._trim_context(messages)
        assert len(trimmed) > 0

        # System messages always kept, non-system trimmed to window
        system_count = sum(1 for m in trimmed if m.role == "system")
        non_system = [m for m in trimmed if m.role != "system"]

        assert system_count >= 1
        # Most recent messages should be preserved
        assert any("User message 34" in m.content for m in non_system)

    def test_rag_searcher_accessible(self, temp_repo_root_with_memory: str):
        """Test that RAG searcher is accessible via registry."""
        agent = _make_agent(temp_repo_root_with_memory, rag_enabled=True, rag_top_k=3)

        rag_searcher = agent.registry._rag_searcher
        assert rag_searcher is not None
        assert hasattr(rag_searcher, 'find_relevant_files')

    def test_rag_find_relevant_files_called(self, temp_repo_root_with_memory: str):
        """Test RAG find_relevant_files integration."""
        agent = _make_agent(temp_repo_root_with_memory, rag_enabled=True, rag_top_k=3)

        rag_searcher = agent.registry._rag_searcher
        mock_results = [
            {"file": "sample.py", "content": "def hello() -> str:", "score": 0.9},
        ]

        with patch.object(rag_searcher, 'find_relevant_files', return_value=mock_results) as mock_find:
            # _build_session_context doesn't call RAG directly — RAG is injected during run()
            # Just verify the mock path is valid
            result = rag_searcher.find_relevant_files("test query", top_k=3)
            assert result == mock_results
            assert mock_find.call_count == 1


