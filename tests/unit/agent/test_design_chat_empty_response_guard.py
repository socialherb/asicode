"""Empty-response funnel guard.

All three _respond_impl exit paths — normal termination, max-iterations tail,
tool-loop fallback — can return content="" when the LLM keeps emitting empty
responses after every retry. respond()'s funnel replaces the blank with
_EMPTY_RESPONSE_FALLBACK so the REPL / webapp / subagent summary / session
history never receive an empty turn. is_error stays False (budget exhaustion
with partial progress is not a failure).
"""
import subprocess

import pytest

from external_llm.agent.design_chat_loop import _EMPTY_RESPONSE_FALLBACK, DesignChatLoop
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry
from external_llm.client import (
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolCallResponse,
)


class _StubClient:
    """Scripted stub: per-scenario tool-loop and plain-chat behaviors."""

    def __init__(self, tool_loop=None, plain=None, tool_loop_raise=None):
        self._tool_loop = tool_loop            # callable(messages) -> ToolCallResponse
        self._plain = plain                    # callable(messages) -> LLMResponse
        self._tool_loop_raise = tool_loop_raise
        self.tool_calls = 0
        self.plain_calls = 0

    @staticmethod
    def get_provider_name() -> str:
        return "stub"

    def chat_with_tools(self, messages, tools, model, **kw):
        self.tool_calls += 1
        if self._tool_loop_raise is not None:
            raise self._tool_loop_raise
        return self._tool_loop(messages)

    def chat(self, messages, model, **kw):
        self.plain_calls += 1
        return self._plain(messages)


def _tool_response(content: str, tool_calls=None):
    return ToolCallResponse(
        content=content, model="m", provider="stub", tokens_used=1,
        finish_reason="tool_calls" if tool_calls else "stop",
        raw_response=None, tool_calls=tool_calls or [],
    )


def _plain_response(content: str, raw=None):
    return LLMResponse(
        content=content, model="m", provider="stub", tokens_used=1,
        finish_reason="stop", raw_response=raw,
    )


@pytest.fixture
def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("hello\n")
    return str(tmp_path)


def _run(client, _repo, **kw):
    reg = ToolRegistry(_repo, AgentConfig())
    loop = DesignChatLoop(client, reg, "stub-model")
    return loop.respond([LLMMessage(role="user", content="do stuff")], **kw)


# ── PATH 1: normal termination, empty content twice ─────────────────────────

def test_path1_normal_termination_empty_injects_fallback(_repo):
    """Empty response with no tool calls: one auto-retry (_empty_retried),
    then the funnel guard injects the fallback instead of returning ''."""
    client = _StubClient(
        tool_loop=lambda m: _tool_response(""),
        plain=lambda m: _plain_response("unused"),
    )
    r = _run(client, _repo)
    assert r.content == _EMPTY_RESPONSE_FALLBACK
    assert r.is_error is False
    assert r.hit_max_iterations is False
    assert client.tool_calls == 2  # original + single auto-retry


def test_passthrough_non_empty_response_untouched(_repo):
    """Normal non-empty final answer passes through the funnel unchanged."""
    client = _StubClient(tool_loop=lambda m: _tool_response("all good"))
    r = _run(client, _repo)
    assert r.content == "all good"
    assert r.is_error is False
    assert client.tool_calls == 1


# ── PATH 2: max-iterations tail, final + retry both empty ──────────────────

def test_path2_max_iterations_empty_injects_fallback(_repo):
    """Budget exhausted (every turn asks for a tool, budget=1) and the final
    text-only call AND its retry both return empty ⇒ funnel injects the
    fallback; hit_max_iterations stays True, is_error stays False."""
    client = _StubClient(
        tool_loop=lambda m: _tool_response("", tool_calls=[ToolCallRequest(
            call_id="c1", name="read_file", args={"file_path": "README.md"},
        )]),
        plain=lambda m: _plain_response(""),
    )
    r = _run(client, _repo, max_tool_iterations=1)
    assert r.hit_max_iterations is True
    assert r.is_error is False
    assert r.content == _EMPTY_RESPONSE_FALLBACK
    assert client.tool_calls == 1
    assert client.plain_calls == 2  # final + CRITICAL retry


# ── PATH 3: tool-loop fallback plain chat ───────────────────────────────────

def test_path3_fallback_plain_chat_empty_injects_fallback(_repo):
    """Non-LLM exception in the tool loop degrades to plain chat; if that also
    returns empty, the funnel injects the fallback."""
    client = _StubClient(
        tool_loop_raise=ValueError("boom"),
        plain=lambda m: _plain_response(""),
    )
    r = _run(client, _repo)
    assert r.content == _EMPTY_RESPONSE_FALLBACK
    assert r.is_error is False
    assert client.plain_calls == 1


def test_path3_fallback_uses_reasoning_when_content_empty(_repo):
    """The fallback path must mirror the reasoning fallback of the other exit
    paths: a thinking model's answer in reasoning_content is used instead of
    shipping an empty turn."""
    raw = {"choices": [{"message": {"reasoning_content": "the real answer"}}]}
    client = _StubClient(
        tool_loop_raise=ValueError("boom"),
        plain=lambda m: _plain_response("", raw=raw),
    )
    r = _run(client, _repo)
    assert r.content == "the real answer"
    assert r.reasoning_content == "the real answer"
    assert r.is_error is False
