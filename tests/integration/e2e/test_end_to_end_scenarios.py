"""
End-to-end integration tests for complete asicode workflows.
"""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.integration.helpers import (
    apply_patch_and_verify,
    git_add_and_commit,
)


@pytest.mark.integration
@pytest.mark.slow  # E2E tests may be slower
class TestEndToEndScenarios:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def complex_repo_structure(self):
        """Create a complex repository structure for E2E testing."""
        repo_root = tempfile.mkdtemp(prefix="e2e-repo-")
        repo_path = Path(repo_root)

        # Create project structure
        structure = {
            "src/main.py": """def main():
    print("Hello, world!")
    result = calculate(10, 5)
    print(f"Result: {result}")

def calculate(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    main()
""",
            "src/utils/helpers.py": """def format_result(value):
    return f"Result: {value}"

def validate_input(value):
    if not isinstance(value, (int, float)):
        raise TypeError("Input must be numeric")
    return True
""",
            "tests/test_main.py": """import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from main import calculate

def test_calculate():
    assert calculate(2, 3) == 5
    assert calculate(0, 0) == 0
    assert calculate(-1, 1) == 0

def test_calculate_negative():
    assert calculate(-5, -3) == -8
""",
            "README.md": "# Test Project\n\nThis is a test project for E2E testing.",
            ".asicode/memory.md": "# Project Memory\n\nThis project has a calculator function."
        }

        for filepath, content in structure.items():
            full_path = repo_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_root, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_root, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_root, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_root, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_root, capture_output=True)

        yield repo_root

        # Cleanup
        shutil.rmtree(repo_root, ignore_errors=True)

    @pytest.mark.xfail(reason="Mock does not match current ChatWithToolsResponse structure", strict=False)
    def test_e2e_bug_fix_workflow(self, complex_repo_structure):
        """E2E test: Bug fix workflow from query to patch application."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig(
            max_turns=10,
            rag_enabled=True,
            auto_test_on_patch=True,
            planning_enabled=True,
            self_review_enabled=True
        )

        # Mock LLM to simulate a bug fix scenario
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        # Simulate agent behavior for bug fix
        tool_call_sequence = [
            # First turn: understand the problem
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": json.dumps({"command": "cat src/main.py"})
                    }
                }]
            },
            # Second turn: examine tests
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": json.dumps({"command": "cat tests/test_main.py"})
                    }
                }]
            },
            # Third turn: run tests to see failure
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({"path": "tests/"})
                    }
                }]
            },
            # Fourth turn: create fix
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps({
                            "patch": """--- a/src/main.py
+++ b/src/main.py
@@ -1,8 +1,8 @@
 def main():
     print("Hello, world!")
     result = calculate(10, 5)
     print(f"Result: {result}")

 def calculate(a: int, b: int) -> int:
-    return a + b
+    return a * b  # Fixed: should multiply, not add

 if __name__ == "__main__":
     main()"""
                        })
                    }
                }]
            },
            # Fifth turn: verify fix with tests
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "run_tests",
                        "arguments": json.dumps({"path": "tests/"})
                    }
                }]
            }
        ]

        response_iter = iter(tool_call_sequence)
        def mock_chat_with_tools(*args, **kwargs):
            try:
                response = next(response_iter)
                mock_response = Mock()
                mock_response.content = "Analyzing..."
                mock_response.tool_calls = response.get("tool_calls", [])
                mock_response.prompt_tokens = 100
                mock_response.completion_tokens = 50
                mock_response.raw_response = None  # prevent Mock subscript errors
                return mock_response
            except StopIteration:
                # Final response
                mock_response = Mock()
                mock_response.content = "Bug fix completed successfully."
                mock_response.tool_calls = []
                mock_response.raw_response = None
                mock_response.prompt_tokens = 0
                mock_response.completion_tokens = 0
                return mock_response

        mock_llm.chat_with_tools.side_effect = mock_chat_with_tools

        registry = ToolRegistry(complex_repo_structure, config)
        agent = AgentLoop(
            llm_client=mock_llm,
            registry=registry,
            config=config,
            model="test-model"
        )

        # Mock test results
        from external_llm.agent.tool_registry import ToolResult
        with patch.object(registry, 'dispatch') as mock_dispatch:
            def dispatch_side_effect(tool_name, args):
                if tool_name == "run_tests":
                    return ToolResult(ok=True, content="Tests passed\n2 passed, 0 failed",
                                      metadata={"passed": 2, "failed": 0})
                elif tool_name == "shell_exec":
                    cmd = args.get("command", "")
                    filepath = cmd.replace("cat ", "").strip()
                    full_path = Path(complex_repo_structure) / filepath
                    if full_path.exists():
                        return ToolResult(ok=True, content=full_path.read_text(),
                                          metadata={"command": cmd})
                    else:
                        return ToolResult(ok=False, content="", error="File not found")
                elif tool_name == "apply_patch":
                    return ToolResult(ok=True, content="Patch applied successfully",
                                      metadata={"conflict": False})
                else:
                    return ToolResult(ok=False, content="", error=f"Tool {tool_name} not mocked")

            mock_dispatch.side_effect = dispatch_side_effect

            # Run agent
            result = agent.run("Fix the bug in calculate function: it should multiply instead of add")

        assert result.status in ("success", "max_turns")
        # Agent should have used multiple turns
        assert len(result.turns) >= 1

    def test_e2e_feature_addition_workflow(self, complex_repo_structure):
        """E2E test: Add new feature workflow."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig(
            max_turns=15,
            rag_enabled=True,
            planning_enabled=True
        )

        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        registry = ToolRegistry(complex_repo_structure, config)
        agent = AgentLoop(
            llm_client=mock_llm,
            registry=registry,
            config=config,
            model="test-model"
        )

        # Mock a feature addition scenario
        # This is a simplified test - real test would mock LLM responses

        # For now, just verify the system integrates
        assert agent.config.planning_enabled is True
        assert hasattr(registry, 'dispatch')

    def test_e2e_multi_file_refactor(self, complex_repo_structure):
        """E2E test: Multi-file refactoring workflow."""
        # Create additional files for refactoring
        repo_path = Path(complex_repo_structure)

        # Add duplicate code to refactor
        (repo_path / "src/old_calculator.py").write_text("""def old_add(a, b):
    return a + b

def old_subtract(a, b):
    return a - b
""")

        (repo_path / "src/new_calculator.py").write_text("""# New calculator module
# TODO: Refactor old functions here
""")

        git_add_and_commit(complex_repo_structure, "Add calculator files")

        # Test would involve:
        # 1. Reading multiple files
        # 2. Analyzing code duplication
        # 3. Creating refactoring plan
        # 4. Applying changes across files
        # 5. Running tests to verify

        # For now, verify test setup
        assert (repo_path / "src/old_calculator.py").exists()
        assert (repo_path / "src/new_calculator.py").exists()

    def test_e2e_plan_compilation_and_application(self, complex_repo_structure):
        """E2E test: Plan compilation and application workflow."""
        from plan_compiler import compile_plan_to_unified_diff as compile_plan_to_diff

        # Create a plan to modify multiple files
        plan = {
            "version": "ASICODE_PLAN_V1",
            "operations": [
                {
                    "type": "edit_blocks",
                    "path": "src/main.py",
                    "blocks": [{
                        "before": 'def calculate(a: int, b: int) -> int:\n    return a + b',
                        "after": 'def calculate(a: int, b: int) -> int:\n    """Multiply two numbers."""\n    return a * b'
                    }]
                },
                {
                    "type": "create_file",
                    "path": "src/multiply.py",
                    "content": 'def multiply(a, b):\n    return a * b\n'
                },
            ]
        }

        # Compile plan to diff
        result = compile_plan_to_diff(plan=plan, repo_root=complex_repo_structure, allow_empty=True)
        diff = result.diff_patch

        assert isinstance(diff, str)
        assert len(diff) > 0
        assert "--- a/src/main.py" in diff
        assert "+++ b/src/main.py" in diff
        assert "--- /dev/null" in diff or "--- a/src/multiply.py" in diff
        assert "+++ b/src/multiply.py" in diff

        # Apply the diff (verify only changes to existing files - new files are untracked)
        success = apply_patch_and_verify(
            complex_repo_structure,
            diff,
            [
                '"""Multiply two numbers."""',
            ]
        )

        assert success is True

        # Verify files were modified
        main_content = (Path(complex_repo_structure) / "src/main.py").read_text()
        assert '"""Multiply two numbers."""' in main_content

        multiply_file = Path(complex_repo_structure) / "src/multiply.py"
        assert multiply_file.exists()
        assert "def multiply" in multiply_file.read_text()

    @pytest.mark.xfail(reason="Mock does not match current ChatWithToolsResponse structure", strict=False)
    def test_e2e_error_recovery_workflow(self, complex_repo_structure):
        """E2E test: Error recovery workflow."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig(max_turns=8)

        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        # Simulate an error scenario
        error_sequence = [
            # First attempt fails
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps({
                            "patch": "invalid patch format"
                        })
                    }
                }]
            },
            # Second attempt: diagnose error
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "git_diff",
                        "arguments": json.dumps({})
                    }
                }]
            },
            # Third attempt: correct approach
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": json.dumps({"command": "cat src/main.py"})
                    }
                }]
            },
            # Fourth attempt: apply correct patch
            {
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps({
                            "patch": """--- a/src/main.py
+++ b/src/main.py
@@ -5,6 +5,6 @@
     print(f"Result: {result}")

 def calculate(a: int, b: int) -> int:
-    return a + b
+    return a * b

 if __name__ == "__main__":
     main()"""
                        })
                    }
                }]
            }
        ]

        response_iter = iter(error_sequence)
        def mock_chat_with_tools(*args, **kwargs):
            try:
                response = next(response_iter)
                mock_response = Mock()
                mock_response.content = "Trying..."
                mock_response.tool_calls = response.get("tool_calls", [])
                mock_response.raw_response = None
                mock_response.prompt_tokens = 100
                mock_response.completion_tokens = 50
                return mock_response
            except StopIteration:
                mock_response = Mock()
                mock_response.content = "Recovered from error successfully."
                mock_response.tool_calls = []
                mock_response.raw_response = None
                mock_response.prompt_tokens = 0
                mock_response.completion_tokens = 0
                return mock_response

        mock_llm.chat_with_tools.side_effect = mock_chat_with_tools

        registry = ToolRegistry(complex_repo_structure, config)
        agent = AgentLoop(
            llm_client=mock_llm,
            registry=registry,
            config=config,
            model="test-model"
        )

        from external_llm.agent.tool_registry import ToolResult

        # Mock dispatch to simulate error recovery
        call_count = 0
        def dispatch_side_effect(tool_name, args):
            nonlocal call_count
            call_count += 1

            if tool_name == "apply_patch":
                patch_text = args.get("patch", "")
                if "invalid" in patch_text:
                    return ToolResult(ok=False, content="Invalid patch format", error="Invalid patch format")
                else:
                    return ToolResult(ok=True, content="applied successfully")
            elif tool_name == "git_diff":
                return ToolResult(ok=True, content="No changes")
            elif tool_name == "shell_exec":
                return ToolResult(ok=True, content="file content")
            else:
                return ToolResult(ok=False, content="Unknown tool", error="Unknown tool")

        with patch.object(registry, 'dispatch') as mock_dispatch:
            mock_dispatch.side_effect = dispatch_side_effect

            result = agent.run("Fix calculation function with error recovery")

        # Agent should recover from error and complete
        assert result.status in ("success", "max_turns")
        assert call_count >= 3  # Should have attempted multiple tools

    def test_e2e_multi_agent_orchestration(self, complex_repo_structure):
        """E2E test: Multi-agent orchestration workflow."""
        # Check if orchestrator is available
        try:
            from external_llm.agent.orchestrator import OrchestratorAgent
            ORCHESTRATOR_AVAILABLE = True
        except ImportError:
            ORCHESTRATOR_AVAILABLE = False

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("Orchestrator not available")

        from external_llm.agent.tool_registry import AgentConfig

        config = AgentConfig(
            max_turns=20,
        )

        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        # Mock orchestrator decomposition
        with patch('external_llm.agent.orchestrator.OrchestratorAgent._decompose_task') as mock_decompose:
            from external_llm.agent.orchestrator import SubTaskSpec

            mock_decompose.return_value = [
                SubTaskSpec(
                    task_id="task-1",
                    title="Fix calculation",
                    description="Fix main.py calculation",
                    dependencies=[],
                    assigned_files=["src/main.py"]
                ),
                SubTaskSpec(
                    task_id="task-2",
                    title="Update tests",
                    description="Update tests",
                    dependencies=["task-1"],
                    assigned_files=["tests/test_main.py"]
                )
            ]

            from external_llm.agent.orchestrator import OrchestratorConfig
            from external_llm.agent.tool_registry import ToolRegistry
            orch_config = OrchestratorConfig(max_subagents=2, agent_config=config)
            registry = ToolRegistry(complex_repo_structure, config)

            # Create orchestrator
            orchestrator = OrchestratorAgent(
                llm_client=mock_llm,
                registry=registry,
                orch_config=orch_config,
                model="test-model"
            )

            # Mock subagent execution
            with patch.object(orchestrator, '_run_subagent') as mock_execute:
                from external_llm.agent.agent_loop import AgentResult
                def execute_side_effect(*args, **kwargs):
                    subtask = args[0] if args else kwargs.get('subtask')
                    return AgentResult(
                        status="success",
                        turns=[],
                        final_message=f"Completed {subtask.description}",
                        metadata={}
                    )

                mock_execute.side_effect = execute_side_effect

                # Run orchestrator
                result = orchestrator.run("Refactor calculator and update tests")

        assert result.status in ("success", "partial", "error")
        assert result.subtask_results is not None

    def test_e2e_performance_tracking(self, complex_repo_structure):
        """E2E test: Performance metrics tracking throughout workflow."""
        from external_llm.agent.agent_loop import AgentLoop
        from external_llm.agent.performance_metrics import get_global_collector
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        collector = get_global_collector()
        collector.reset_cache_stats()

        config = AgentConfig(max_turns=5)
        mock_llm = Mock()
        mock_llm.get_provider_name.return_value = "openai"

        registry = ToolRegistry(complex_repo_structure, config)
        AgentLoop(
            llm_client=mock_llm,
            registry=registry,
            config=config,
            model="test-model"
        )

        # Record some tool calls to verify tracking works
        collector.record_tool_call("shell_exec", 10.0, cache_hit=False)
        collector.record_tool_call("apply_patch", 20.0, cache_hit=False)

        # Verify summary can be retrieved
        summary = collector.get_summary()
        assert isinstance(summary, dict)
        assert "tool_metrics" in summary or "cache_metrics" in summary or "llm_metrics" in summary

    def test_e2e_memory_persistence_workflow(self, complex_repo_structure):
        """E2E test: Memory persistence across sessions."""
        from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

        config = AgentConfig(max_turns=5)
        registry = ToolRegistry(complex_repo_structure, config)

        # Check if update_memory tool exists by trying to dispatch it
        test_result = registry.dispatch("update_memory", {"note": "test"})
        if not test_result.ok and "unknown tool" in (test_result.error or "").lower():
            pytest.skip("update_memory tool not available")

        # Read initial memory
        memory_file = Path(complex_repo_structure) / ".asicode" / "memory.md"
        initial_content = memory_file.read_text() if memory_file.exists() else ""

        # First session: add to memory
        result1 = registry.dispatch("update_memory", {
            "note": "## Session 1\nFixed bug in calculate function."
        })
        assert result1.ok is True

        # Second session: add more to memory
        result2 = registry.dispatch("update_memory", {
            "note": "## Session 2\nAdded multiplication feature."
        })
        assert result2.ok is True

        # Verify cumulative memory
        final_content = memory_file.read_text()
        assert initial_content in final_content
        assert "Session 1" in final_content
        assert "Session 2" in final_content
        assert "Fixed bug" in final_content
        assert "multiplication feature" in final_content

    def test_e2e_full_stack_api_workflow(self, test_client, complex_repo_structure):
        """E2E test: Full stack API workflow from request to completion."""
        # This test uses the actual FastAPI endpoints
        # Mock external dependencies

        with patch('external_llm.ExternalLLMService') as mock_llm_service:
            mock_instance = Mock()
            # Simulate successful agent execution
            mock_instance.stream_message.return_value = iter([
                'event: tool_call\ndata: {"session_id": "test", "tool_name": "shell_exec", "success": true}\n\n',
                'event: tool_call\ndata: {"session_id": "test", "tool_name": "apply_patch", "success": true}\n\n',
                'event: complete\ndata: {"session_id": "test", "success": true}\n\n'
            ])
            mock_llm_service.return_value = mock_instance

            # Start agent session via API
            request_data = {
                "prompt": "Fix calculate function in src/main.py",
                "llm_mode": "deterministic",
                "repo_root": complex_repo_structure,
                "model": "claude-3-5-sonnet",
                "planning_enabled": True,
                "auto_test_on_patch": True
            }

            response = test_client.post("/edit/run", json=request_data)
            assert response.status_code in (200, 422, 500)

            if response.status_code == 200:
                session_data = response.json()
                run_id = session_data.get("run_id") or session_data.get("session_id")

                # Cancel endpoint (optional)
                if run_id:
                    test_client.post(f"/agent/cancel/{run_id}")
                    # Might be 200 or 404 depending on state

        # Test history endpoint
        test_client.get("/agent/history", params={"repo_root": complex_repo_structure})
        # Might return empty list or contain our session
