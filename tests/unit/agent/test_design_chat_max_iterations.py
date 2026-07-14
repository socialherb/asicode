"""Turn 13122 fix 1: DesignChatResult.hit_max_iterations propagation.

DesignChatLoop's exhaustion path ("[SYSTEM] Tool-call budget exhausted…")
returns is_error=False, so the IPC worker used to map an UNFINISHED task to
status="success". The flag lets the worker report status="max_turns" instead,
matching the in-process AgentLoop path (max_turns_reached signal).
"""
import subprocess

import pytest

from external_llm.client import (
    LLMMessage, LLMResponse, ToolCallRequest, ToolCallResponse,
)
from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


class _StubClient:
    """LLM stub: optionally always requests a tool call (never finishes)."""

    def __init__(self, always_tool_call: bool):
        self.always = always_tool_call

    def chat_with_tools(self, messages, tools, model, **kw):
        if self.always:
            return ToolCallResponse(
                content="", model=model, provider="stub", tokens_used=1,
                finish_reason="tool_calls", raw_response=None,
                tool_calls=[ToolCallRequest(
                    call_id="c1", name="read_file",
                    args={"file_path": "README.md"},
                )],
            )
        return ToolCallResponse(
            content="done, no tools needed", model=model, provider="stub",
            tokens_used=1, finish_reason="stop", raw_response=None,
            tool_calls=[],
        )

    def chat(self, messages, model, **kw):
        return LLMResponse(
            content="final answer after exhaustion", model=model,
            provider="stub", tokens_used=1, finish_reason="stop",
            raw_response=None,
        )


@pytest.fixture
def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("hello\n")
    return str(tmp_path)


def test_exhaustion_sets_hit_max_iterations(_repo):
    """Budget exhausted (every turn asks for a tool, budget=1) ⇒ the result
    carries hit_max_iterations=True with is_error=False — the worker maps this
    to status='max_turns' instead of a false 'success'."""
    reg = ToolRegistry(_repo, AgentConfig())
    loop = DesignChatLoop(_StubClient(True), reg, "stub-model")
    r = loop.respond(
        [LLMMessage(role="user", content="do stuff")], max_tool_iterations=1,
    )
    assert r.hit_max_iterations is True
    assert r.is_error is False
    # The worker's mapping (asi run_subagent_worker): max_turns wins.
    status = ("max_turns" if r.hit_max_iterations
              else "error" if r.is_error else "success")
    assert status == "max_turns"


def test_normal_completion_leaves_flag_false(_repo):
    """A turn that finishes within budget must NOT set the flag."""
    reg = ToolRegistry(_repo, AgentConfig())
    loop = DesignChatLoop(_StubClient(False), reg, "stub-model")
    r = loop.respond(
        [LLMMessage(role="user", content="do stuff")], max_tool_iterations=5,
    )
    assert r.hit_max_iterations is False
    assert r.is_error is False
