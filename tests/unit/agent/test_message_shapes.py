"""Unit tests for provider-agnostic message shape detection.

Pins the contract that ``is_tool_result`` / ``is_tool_call`` recognise ALL
supported provider formats — standard (OpenAI/DeepSeek/Ollama), Anthropic-native
(tool_use/tool_result blocks), and Gemini-native (functionCall/functionResponse
parts). Regression guard for GAP-3 (Gemini was previously invisible to repair /
eviction / sliding-window orphan detection).
"""

from external_llm.agent.message_shapes import (
    _has_raw_blocks,
    _has_raw_key,
    _is_gemini_tool_call,
    _is_gemini_tool_result,
    is_tool_call,
    is_tool_result,
)
from external_llm.client import LLMMessage


def _std_tool_result():
    return LLMMessage(role="tool", content="42", tool_call_id="c1", name="read_file")


def _std_tool_call():
    return LLMMessage(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
    )


def _anthropic_tool_result():
    return LLMMessage(
        role="user",
        content="",
        raw_content=[{"type": "tool_result", "tool_use_id": "tu_1", "content": "42"}],
    )


def _anthropic_tool_call():
    return LLMMessage(
        role="assistant",
        content="",
        raw_content=[{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "tu_1", "name": "grep"}],
    )


def _gemini_tool_result():
    return LLMMessage(
        role="user",
        content="",
        raw_content=[{"functionResponse": {"name": "read_file", "response": {"content": "42"}}}],
    )


def _gemini_tool_call():
    return LLMMessage(
        role="assistant",
        content="",
        raw_content=[{"text": "ok"}, {"functionCall": {"name": "grep", "args": {}}}],
    )


def _plain_user():
    return LLMMessage(role="user", content="hello")


def _plain_assistant():
    return LLMMessage(role="assistant", content="hi")


def test_is_tool_result_all_formats():
    assert is_tool_result(_std_tool_result())
    assert is_tool_result(_anthropic_tool_result())
    assert is_tool_result(_gemini_tool_result()), "GAP-3: Gemini result must be detected"
    assert not is_tool_result(_plain_user())
    assert not is_tool_result(_plain_assistant())


def test_is_tool_call_all_formats():
    assert is_tool_call(_std_tool_call())
    assert is_tool_call(_anthropic_tool_call())
    assert is_tool_call(_gemini_tool_call()), "GAP-3: Gemini call must be detected"
    assert not is_tool_call(_plain_assistant())


def test_is_tool_result_rejects_non_tool_native_messages():
    """A native user/assistant message with NO tool parts must NOT match."""
    plain_native_user = LLMMessage(role="user", content="", raw_content=[{"text": "hi"}])
    plain_native_asst = LLMMessage(role="assistant", content="", raw_content=[{"text": "hi"}])
    assert not is_tool_result(plain_native_user)
    assert not is_tool_call(plain_native_asst)


def test_gemini_helpers_directly():
    assert _is_gemini_tool_result(_gemini_tool_result())
    assert _is_gemini_tool_call(_gemini_tool_call())
    assert not _is_gemini_tool_result(_anthropic_tool_result())
    assert not _is_gemini_tool_call(_anthropic_tool_call())


def test_raw_content_helpers_ignore_non_dict_items():
    """Non-dict entries in ``raw_content`` must be skipped, never crash —
    pins the shared skeleton behind both raw-content helpers."""
    msg = LLMMessage(
        role="user",
        content="",
        raw_content=["junk", {"type": "tool_result", "content": "42"}, None],
    )
    assert _has_raw_blocks(msg, "user", "tool_result")
    assert not _has_raw_key(msg, "user", "tool_result")


def test_raw_content_helpers_enforce_role():
    """Role mismatch short-circuits before ``raw_content`` is inspected."""
    wrong_role = LLMMessage(
        role="assistant",
        content="",
        raw_content=[{"type": "tool_result", "content": "42"}],
    )
    assert not _has_raw_blocks(wrong_role, "user", "tool_result")
    assert not _has_raw_key(wrong_role, "user", "functionResponse")
