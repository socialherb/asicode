"""B3: agent_loop's tool path must honor finish_reason="truncated".

B1 normalized Anthropic/Z.AI stop_reason="max_tokens" → "length", and B2 made
the streaming clients report silent network truncation as finish_reason=
"truncated" (dropped tool-call JSON / unbalanced JSON-shaped text). But
``_llm_call_with_tools.call_llm`` only retried on ``"length"`` — a truncated
tool call was therefore silently skipped and the model's intended tool use was
lost, while agent_phase_manager's json-mode path already consumed "truncated".

These tests pin the consumer contract: "truncated" must drive the exact same
recovery chain as "length" — budget-doubling retry (up to 3 attempts), then
clearing partial tool calls while preserving text content.
"""

from __future__ import annotations

from unittest import mock

from external_llm.agent.agent_loop import AgentLoop
from external_llm.client import LLMMessage, ToolCallResponse


def _resp(finish_reason, content="partial", tool_calls=None):
    return ToolCallResponse(
        content=content,
        model="glm-5.2",
        provider="zai",
        tokens_used=10,
        finish_reason=finish_reason,
        raw_response={},
        tool_calls=tool_calls or [],
        is_final=False,
    )


def _make_host(responses):
    """Minimal AgentLoop host: skips __init__, stubs pre-flight plumbing.

    _retry_on_rate_limit is reduced to identity so the test exercises exactly
    call_llm's truncation-retry chain (mirrors test_design_chat_* hosts).
    """
    loop = AgentLoop.__new__(AgentLoop)
    loop.model = "glm-5.2"
    loop.config = mock.MagicMock()
    loop.config.cancel_event = None  # must be falsy — MagicMock truthiness would cancel
    loop.config.thinking_mode = None
    loop.config.stream_callback = None
    loop._context_budget = None
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = []
    loop.registry.repo_language = "python"
    loop._cb = lambda *a, **k: None
    loop._retry_on_rate_limit = lambda fn, *a, **k: fn()
    loop._record_llm_call_both = lambda **k: None
    loop.performance_collector = mock.MagicMock()  # P2: record_agent_result wiring target
    loop.llm_client = mock.MagicMock()
    loop.llm_client.chat_with_tools.side_effect = responses
    return loop


_BASE = 32768  # config.tokens.AGENT_TOOL_CALL


def _call(loop):
    return loop._llm_call_with_tools([LLMMessage(role="user", content="hi")])


def test_truncated_triggers_budget_doubling_retry_then_succeeds():
    """Regression: previously a truncated tool call was silently skipped."""
    loop = _make_host(
        [
            _resp("truncated", tool_calls=[{"id": "t1", "name": "bash", "args": {"command": "rm"}}]),
            _resp("end_turn", content="ok"),
        ]
    )
    result = _call(loop)

    assert loop.llm_client.chat_with_tools.call_count == 2
    # second attempt gets the doubled budget (32768 → 65536)
    assert loop.llm_client.chat_with_tools.call_args_list[1].kwargs["max_tokens"] == _BASE * 2
    assert result["content"] == "ok"
    assert result["finish_reason"] == "end_turn"


def test_truncated_after_three_attempts_clears_tool_calls_preserves_text():
    """Exhausted retries must clear partial tool calls but keep text content."""
    loop = _make_host(
        [
            _resp("truncated", tool_calls=[{"id": "t1", "name": "bash", "args": {"command": "rm"}}]),
            _resp("truncated", tool_calls=[{"id": "t1", "name": "bash", "args": {"command": "rm"}}]),
            _resp("truncated", tool_calls=[{"id": "t1", "name": "bash", "args": {"command": "rm"}}]),
        ]
    )
    with mock.patch("external_llm.agent.agent_loop.get_global_collector") as _gg:
        result = _call(loop)

    assert loop.llm_client.chat_with_tools.call_count == 3
    # budget doubling: 32768 → 65536 → 131072
    assert loop.llm_client.chat_with_tools.call_args_list[1].kwargs["max_tokens"] == _BASE * 2
    assert loop.llm_client.chat_with_tools.call_args_list[2].kwargs["max_tokens"] == _BASE * 4
    # partial tool calls cleared, text preserved
    assert result["tool_calls"] == []
    assert result["content"] == "partial"
    assert result["finish_reason"] == "truncated"
    # P2: the turn-outcome channel must record the truncation storm exactly once
    # (the only signal that can surface it — record_llm_call saw the call succeed)
    assert loop.performance_collector.record_agent_result.call_args_list == [mock.call(truncated=True)]
    # Dual-sink: the dashboard (global collector) receives the same signal.
    _gg.return_value.record_agent_result.assert_called_once_with(truncated=True)


def test_length_retry_chain_regression():
    """The original "length" (max_tokens) recovery must be unchanged."""
    loop = _make_host(
        [
            _resp("length", tool_calls=[{"id": "t1", "name": "bash", "args": {}}]),
            _resp("end_turn", content="ok"),
        ]
    )
    result = _call(loop)

    assert loop.llm_client.chat_with_tools.call_count == 2
    assert loop.llm_client.chat_with_tools.call_args_list[1].kwargs["max_tokens"] == _BASE * 2
    assert result["content"] == "ok"


def test_truncated_text_only_response_is_retried_too():
    """Even without tool calls, truncated text must retry (not be accepted)."""
    loop = _make_host(
        [
            _resp("truncated", content="partial json {"),
            _resp("end_turn", content='{"ok": true}'),
        ]
    )
    result = _call(loop)

    assert loop.llm_client.chat_with_tools.call_count == 2
    assert result["content"] == '{"ok": true}'
    assert result["finish_reason"] == "end_turn"
