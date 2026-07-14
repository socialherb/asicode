"""
Pytest fixtures for asicode agent tests.
"""
from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

# Heavy imports deferred to fixture bodies (saves ~300ms on collection).
# from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
# from external_llm.agent.agent_loop import AgentLoop
# from external_llm.agent.orchestrator import FileLockManager


@pytest.fixture(autouse=True)
def _suppress_legacy_shadow_env(monkeypatch):
    """Suppress legacy shadow-logger env var for all tests.

    Prevents fixture-target pollution of ~/.asicode/*.jsonl.
    The shadow module was removed in a cleanup — the env var guard
    is kept for any residual references.
    """
    monkeypatch.setenv("ASICODE_DISABLE_SHADOW", "1")


@pytest.fixture(autouse=True)
def _isolate_runs_dir(tmp_path_factory, monkeypatch):
    """Isolate run artifacts to a per-test temp directory.

    Prevents test runs from leaking into the real .asicode/runs/.

    config.ASICODE_RUNS_DIR is resolved ONCE at import time, and consumers copy
    it by value via ``from config import ASICODE_RUNS_DIR``. Simply setting the
    env var is therefore insufficient (the value is already frozen). We (1)
    set the env var for any future reload, and (2) patch the attribute on
    ``config`` and every already-imported module that captured its own copy.
    Modules imported LATER during the test read config's (already-patched)
    value directly, so they are covered transitively.
    """
    import sys

    target = str(tmp_path_factory.mktemp("asr_runs"))
    monkeypatch.setenv("ASICODE_RUNS_DIR", target)
    # NOTE: use ``in mod.__dict__`` (membership) instead of ``hasattr``.
    # hasattr() triggers a module's lazy ``__getattr__`` — e.g. the
    # ``transformers`` package routes any ``ASICODE_*`` attribute through an
    # Aria-image-processing submodule, which imports torchvision and raises
    # ModuleNotFoundError. __dict__ membership sees only attributes that were
    # actually defined/imported by the module, with no side effects.
    for mod in list(sys.modules.values()):
        if mod is not None and "ASICODE_RUNS_DIR" in mod.__dict__:
            try:
                monkeypatch.setattr(mod, "ASICODE_RUNS_DIR", target)
            except (AttributeError, TypeError):
                pass


@pytest.fixture
def temp_repo_root() -> Generator[str, None, None]:
    """Create a temporary directory as a fake repository root."""
    tmpdir = tempfile.mkdtemp(prefix="asr-test-")
    try:
        # Initialize as a git repo for git operations
        import subprocess
        subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmpdir, capture_output=True)

        # Create a sample Python file for testing
        sample_file = Path(tmpdir) / "sample.py"
        # NOTE: Avoid leading newline so patch hunks starting at line 1 match reliably.
        sample_file.write_text(
            'def hello() -> str:\n'
            '    return "world"\n'
            '\n'
            'class Calculator:\n'
            '    def add(self, a: int, b: int) -> int:\n'
            '        return a + b\n'
        )

        # Commit the file
        subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=tmpdir, capture_output=True)

        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_repo_root_with_memory(temp_repo_root: str) -> Generator[str, None, None]:
    """Create temporary repo root with .asicode/memory.md file."""
    asicode_dir = Path(temp_repo_root) / ".asicode"
    asicode_dir.mkdir(exist_ok=True)
    memory_file = asicode_dir / "memory.md"
    memory_file.write_text("# Test Memory\n\nThis is a test memory file.")
    yield temp_repo_root


@pytest.fixture
def agent_config() -> AgentConfig:
    """Return a basic AgentConfig for testing."""
    from external_llm.agent.tool_registry import AgentConfig
    return AgentConfig(
        max_turns=5,
        run_tests=False,
        run_lint=False,
        auto_test_on_patch=False,
        planning_enabled=False,
        self_review_enabled=False,
        rag_enabled=False,
        parallel_tool_execution_enabled=False,
    )


@pytest.fixture
def mock_llm_client() -> Mock:
    """Return a mock LLM client."""
    client = Mock()

    # Mock provider name - use "openai" to enable native tool calling
    client.get_provider_name.return_value = "openai"
    client.provider = "openai"

    # Mock chat_with_tools method
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_response.tool_calls = []
    mock_response.prompt_tokens = 0
    mock_response.completion_tokens = 0
    mock_response.cache_read_input_tokens = 0
    mock_response.cache_creation_input_tokens = 0
    mock_response.raw_response = None  # For OpenAI format compatibility
    client.chat_with_tools.return_value = mock_response

    # Mock chat method for text mode
    client.chat.return_value = mock_response

    return client


@pytest.fixture
def tool_registry(temp_repo_root: str, agent_config: AgentConfig) -> ToolRegistry:
    """Return a ToolRegistry instance for testing."""
    from external_llm.agent.tool_registry import ToolRegistry
    return ToolRegistry(temp_repo_root, agent_config)


@pytest.fixture
def agent_loop(
    mock_llm_client: Mock, tool_registry: ToolRegistry, agent_config: AgentConfig
) -> AgentLoop:
    """Return an AgentLoop instance with mocked LLM client."""
    from external_llm.agent.agent_loop import AgentLoop
    return AgentLoop(
        llm_client=mock_llm_client,
        registry=tool_registry,
        config=agent_config,
        model="test-model",
    )


@pytest.fixture
def sample_patch() -> str:
    """Return a sample valid patch for testing."""
    return """--- a/sample.py
+++ b/sample.py
@@ -1,6 +1,9 @@
 def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""


@pytest.fixture
def invalid_patch() -> str:
    """Return an invalid patch for testing."""
    return """--- a/sample.py
+++ b/sample.py
@@ -100,6 +100,9 @@ def hello() -> str:
     return "world"

 class Calculator:
+    def __init__(self):
+        self.memory = 0
+
     def add(self, a: int, b: int) -> int:
         return a + b
"""


@pytest.fixture
def file_lock_manager() -> FileLockManager:
    """FileLockManager instance for multi-agent tests."""
    from external_llm.agent.orchestrator import FileLockManager
    return FileLockManager()


@pytest.fixture
def sample_plan_dict() -> dict[str, Any]:
    """Valid ASICODE_PLAN_V1 plan structure."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": '    def add(self, a: int, b: int) -> int:\n        return a + b',
                        "after": '    def add(self, a: int, b: int) -> int:\n        return a + b\n\n    def subtract(self, a: int, b: int) -> int:\n        return a - b'
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_simple_edit_plan_dict() -> dict[str, Any]:
    """Simple edit plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": '    def add(self, a: int, b: int) -> int:\n        return a + b',
                        "after": '    def add(self, a: int, b: int) -> int:\n        return a + b  # Fixed indentation'
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_create_file_plan_dict() -> dict[str, Any]:
    """Create file plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "create_file",
                "path": "new_file.py",
                "content": 'def new_function():\n    return "new"'
            }
        ]
    }


@pytest.fixture
def sample_multi_file_plan_dict() -> dict[str, Any]:
    """Multi-file plan for testing."""
    return {
        "version": "ASICODE_PLAN_V1",
        "operations": [
            {
                "type": "edit_blocks",
                "path": "sample.py",
                "blocks": [
                    {
                        "before": '    def add(self, a: int, b: int) -> int:\n        return a + b',
                        "after": '    def add(self, a: int, b: int) -> int:\n        return a + b  # multi'
                    }
                ]
            },
            {
                "type": "create_file",
                "path": "utils.py",
                "content": 'def helper():\n    return "help"'
            }
        ]
    }
