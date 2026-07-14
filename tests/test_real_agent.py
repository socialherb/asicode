#!/usr/bin/env python3
"""
Test actual AgentLoop execution with a small local model.
This tests the full pipeline: prompt → LLM → tool call parsing → tool execution.
"""
import logging
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.providers import OllamaClient


@pytest.mark.skip(reason="직접 실행 전용 — pytest fixture 없는 필수 파라미터(model_name, user_request). "
                          "실행: python3 tests/test_real_agent.py <model> <request>")
def test_real_agent(model_name: str, user_request: str):
    """Test actual AgentLoop execution with a small model."""
    print(f"Testing REAL AgentLoop with model: {model_name}")
    print(f"User request: {user_request}")
    print("-" * 50)

    # Create config
    config = AgentConfig(
        max_turns=5,
        planning_enabled=False,
        self_review_enabled=False,
        auto_test_on_patch=False,
        rag_enabled=False,
        parallel_tool_execution_enabled=True,
    )

    # Create registry
    repo_root = "."  # current directory
    registry = ToolRegistry(repo_root, config)

    # Create LLM client
    llm_client = OllamaClient(api_key=None, base_url="http://127.0.0.1:11434", timeout=90)

    # Create AgentLoop
    agent_loop = AgentLoop(
        llm_client=llm_client,
        registry=registry,
        config=config,
        model=model_name,
        agent_id="test"
    )

    # Check model detection — small model restrictions removed
    is_small = False
    has_native_tools = agent_loop._check_native_tool_support()
    print(f"Model detected as small: {is_small}")
    print(f"Native tool support: {has_native_tools}")

    # Run the agent
    print("\n=== RUNNING AGENT ===")
    try:
        result = agent_loop.run(user_request)

        print(f"\nAgent result status: {result.status}")
        print(f"Final message: {result.final_message[:200]}...")
        print(f"Turns used: {len(result.turns)}")

        if result.turns:
            print("\n=== TOOL EXECUTION HISTORY ===")
            for i, turn in enumerate(result.turns):
                print(f"Turn {i+1}:")
                print(f"  Tool: {turn.tool_name}")
                print(f"  Args: {turn.tool_args}")
                print(f"  Result OK: {turn.tool_result.ok}")
                if turn.tool_result.content:
                    # Truncate long content
                    content = str(turn.tool_result.content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"  Content: {content}")
                if turn.tool_result.error:
                    print(f"  Error: {turn.tool_result.error}")
                print()

        # Check metadata
        if result.metadata:
            print("\n=== METADATA ===")
            for key, value in result.metadata.items():
                if key == "tokens":
                    print(f"  Tokens: {value}")
                elif key == "turns_used":
                    print(f"  Turns used: {value}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nERROR during agent execution: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_real_agent.py <model_name> \"<user_request>\"")
        print("Example: python test_real_agent.py qwen2.5-coder:3b \"Read the first 50 lines of main.py\"")
        sys.exit(1)

    model = sys.argv[1]
    user_request = sys.argv[2]

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("external_llm.client").setLevel(logging.INFO)
    logging.getLogger("external_llm.agent.agent_loop").setLevel(logging.INFO)

    success = test_real_agent(model, user_request)

    if success:
        print("\n✅ TEST PASSED: Agent executed successfully with tool calls")
    else:
        print("\n❌ TEST FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
