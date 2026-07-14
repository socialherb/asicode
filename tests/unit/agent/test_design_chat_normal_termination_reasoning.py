"""Normal-termination reasoning_content fallback.

GLM-5.2 (thinking ON) / DeepSeek Reasoner may emit the final answer in
``reasoning_content`` with an empty ``content`` field. Three of the four
termination paths in ``DesignChatLoop._respond_impl`` already fall back to
reasoning_content; this test pins the fourth (the normal no-tool-call
termination path) so the closing summary is never silently swallowed.

Reproduces the reported symptom: reasoning shown (💭 thought), tools ran, but
the REPL returns to prompt with NO final answer because the closing message
landed in reasoning_content and was discarded at the normal termination path.
"""
from __future__ import annotations

from unittest import mock

from external_llm.agent.design_chat_loop import DesignChatLoop, DesignChatResult
from external_llm.client import LLMMessage


def _make_loop_with_reasoning_only_final(reasoning: str) -> DesignChatLoop:
    """Build a DesignChatLoop whose chat_with_tools returns a normal-termination
    response: NO tool calls, EMPTY content, but reasoning_content populated in
    raw_response (the GLM-5.2 / DeepSeek Reasoner "answer in reasoning" shape)."""
    loop = DesignChatLoop.__new__(DesignChatLoop)  # skip heavy __init__
    loop.model = "glm-5.2"
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = []
    loop.registry.repo_language = "python"
    loop.registry.session_plan = None
    loop.registry.config = mock.MagicMock()
    loop.registry.config.cancel_event = None

    _resp = mock.MagicMock()
    _resp.content = ""  # empty — answer lives in reasoning_content
    _resp.tool_calls = None  # no tool calls → normal termination path
    _resp.reasoning_content = reasoning
    _resp.tokens_used = 42
    _resp.prompt_tokens = 30
    _resp.completion_tokens = 12
    _resp.cache_read_input_tokens = 0
    _resp.cache_creation_input_tokens = 0
    _resp.provider = "zai"
    _resp.raw_response = {"choices": [{"message": {"reasoning_content": reasoning}}]}

    loop.llm_client = mock.MagicMock()
    loop.llm_client.chat_with_tools.return_value = _resp

    # _call_llm_with_retry just invokes the passed callable (no retry logic).
    # It is called with _estimated_prompt_tokens / overflow_retry_cb kwargs.
    loop._call_llm_with_retry = lambda fn, **kw: fn()

    return loop


def test_normal_termination_falls_back_to_reasoning_content():
    """When the final turn has empty content but reasoning_content, the closing
    summary must surface from reasoning_content — not be silently dropped."""
    reasoning = "## Done: applied digest offload + centralized confidence normalization in commit 5149a67c"
    loop = _make_loop_with_reasoning_only_final(reasoning)

    result = DesignChatResult()
    msgs = [LLMMessage(role="user", content="let's improve this")]

    with mock.patch("external_llm.agent.design_chat_loop._evict_for_loop", return_value=msgs), \
         mock.patch("external_llm.agent.design_chat_loop._apply_context_hard_cap", return_value=msgs):
        loop._respond_impl(
            msgs, stream_callback=None, reasoning_callback=None,
            max_tool_iterations=5, token_callback=None, result=result,
        )

    # The closing summary must come from reasoning_content, not be empty.
    assert result.content == reasoning, (
        "normal-termination path silently swallowed the GLM-5.2 closing summary "
        "(it was in reasoning_content with empty content)"
    )
    # No wasted retry: with the fallback, content is non-empty so the
    # once-retry does not fire. Before the fix this was 2 (retry + discard).
    assert result.total_llm_calls == 1, "fallback should suppress the futile retry"
    assert result.is_error is False
