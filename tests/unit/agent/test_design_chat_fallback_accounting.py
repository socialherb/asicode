"""Defense-depth parity: the _fallback_plain_chat path in _respond_impl must
accumulate split-token accounting (prompt/completion/cache) into the result —
just like the normal tool-loop path (L1186-1198) and the final-summary path
(L1645-1661) already do.

Previously the fallback only incremented total_llm_calls and set content/error,
silently dropping the fallback call's token consumption from cost/token
accounting. This test pins the invariant: every real LLM call is reflected in
the per-bucket counters.
"""
from __future__ import annotations

from unittest import mock

from external_llm.agent.design_chat_loop import DesignChatLoop, DesignChatResult
from external_llm.client import LLMMessage


def _make_loop_with_fallback_trigger(fallback_tokens: dict) -> DesignChatLoop:
    """Build a DesignChatLoop whose chat_with_tools raises a non-LLMClientError
    (forcing the _fallback_plain_chat path), and whose chat() returns a response
    carrying the given split-token fields."""
    loop = DesignChatLoop.__new__(DesignChatLoop)  # skip heavy __init__
    loop.model = "test-model"
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = []
    loop.registry.repo_language = "python"
    loop.registry.session_plan = None
    loop.registry.config = mock.MagicMock()
    loop.registry.config.cancel_event = None

    # chat_with_tools raises a generic exception (NOT LLMClientError) → fallback
    def _chat_with_tools(*a, **k):
        raise RuntimeError("simulated tool-call failure")

    loop.llm_client = mock.MagicMock()
    loop.llm_client.chat_with_tools.side_effect = _chat_with_tools

    # chat() returns a response with split-token fields populated
    _resp = mock.MagicMock()
    _resp.content = "fallback answer"
    _resp.reasoning_content = ""
    _resp.tokens_used = fallback_tokens.get("tokens_used", 100)
    _resp.prompt_tokens = fallback_tokens.get("prompt_tokens", 80)
    _resp.completion_tokens = fallback_tokens.get("completion_tokens", 20)
    _resp.cache_read_input_tokens = fallback_tokens.get("cache_read_tokens", 5)
    _resp.cache_creation_input_tokens = fallback_tokens.get("cache_creation_tokens", 2)
    _resp.provider = fallback_tokens.get("provider", "test-provider")
    _resp.raw_response = {"choices": [{"message": {}}]}
    loop.llm_client.chat.return_value = _resp

    # _call_llm_with_retry just calls the passed callable (no retry logic)
    loop._call_llm_with_retry = lambda fn: fn()

    return loop


def test_fallback_path_accumulates_split_tokens():
    """The fallback path must accumulate prompt/completion/cache tokens into result."""
    loop = _make_loop_with_fallback_trigger({
        "tokens_used": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 2,
        "provider": "test-provider",
    })

    result = DesignChatResult()
    msgs = [LLMMessage(role="user", content="hello")]

    with mock.patch("external_llm.agent.design_chat_loop._evict_for_loop", return_value=msgs), \
         mock.patch("external_llm.agent.design_chat_loop._apply_context_hard_cap", return_value=msgs):
        loop._respond_impl(
            msgs, stream_callback=None, reasoning_callback=None,
            max_tool_iterations=1, token_callback=None, result=result,
        )

    # The fallback call's tokens must be reflected in the per-bucket counters.
    assert result.total_llm_calls == 1
    assert result.tokens_used == 100
    assert result.prompt_tokens == 80
    assert result.completion_tokens == 20
    assert result.cache_read_tokens == 5
    assert result.cache_creation_tokens == 2
    # last_call_* must also reflect the fallback call (not stay at 0)
    assert result.last_call_prompt_tokens == 80
    assert result.last_call_completion_tokens == 20
    assert result.last_call_cache_read_tokens == 5
    assert result.last_call_cache_creation_tokens == 2
    assert result.provider == "test-provider"
    assert result.content == "fallback answer"
    assert result.is_error is False


def test_fallback_path_tokens_not_silently_dropped():
    """Regression guard: before the fix, fallback tokens were 0 (silently dropped)."""
    loop = _make_loop_with_fallback_trigger({
        "tokens_used": 500,
        "prompt_tokens": 400,
        "completion_tokens": 100,
        "cache_read_tokens": 50,
        "cache_creation_tokens": 10,
    })

    result = DesignChatResult()
    msgs = [LLMMessage(role="user", content="hello")]

    with mock.patch("external_llm.agent.design_chat_loop._evict_for_loop", return_value=msgs), \
         mock.patch("external_llm.agent.design_chat_loop._apply_context_hard_cap", return_value=msgs):
        loop._respond_impl(
            msgs, stream_callback=None, reasoning_callback=None,
            max_tool_iterations=1, token_callback=None, result=result,
        )

    # The key invariant: tokens_used must NOT be 0 when the fallback call
    # consumed tokens. Before the fix this was 0 (silent drop).
    assert result.tokens_used == 500, "fallback tokens were silently dropped from accounting"
    assert result.prompt_tokens == 400
    assert result.completion_tokens == 100
