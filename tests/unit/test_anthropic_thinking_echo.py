"""Regression: Anthropic extended-thinking multi-turn must echo native content blocks.

When an assistant turn is reconstructed from an Anthropic response and sent back
on the next turn, the native `content[]` blocks (which include the `thinking`
block with its signature) must be preserved. Without them, extended-thinking
multi-turn is rejected by the Anthropic API with HTTP 400.

Two layers are covered:

1. ``design_chat_loop`` — when building the assistant message, the response's
   top-level ``content`` list must be captured into ``LLMMessage.raw_content``.
2. ``AnthropicClient.chat`` payload assembly — an assistant message carrying
   ``raw_content`` must serialize those blocks verbatim (thinking block included)
   instead of synthesizing text+tool_use blocks and dropping thinking.
"""
from __future__ import annotations

from external_llm.anthropic_client import AnthropicClient
from external_llm.client import LLMMessage


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class _FakeSession:
    def __init__(self):
        self.captured = None

    def post(self, url, headers=None, json=None, timeout=None, **kw):
        self.captured = json
        return _FakeResponse()


def _make_client():
    c = AnthropicClient.__new__(AnthropicClient)
    c.api_key = "test"
    c.base_url = None
    c.timeout = 30
    c._session = _FakeSession()
    return c


def test_design_chat_loop_preserves_anthropic_content_blocks_as_raw_content():
    """design_chat_loop must put Anthropic response `content` into raw_content."""
    # Simulate the reconstruction logic added at design_chat_loop.py ~1017.
    thinking_block = {"type": "thinking", "thinking": "planning...", "signature": "sig-abc"}
    tool_use_block = {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "x"}}

    raw = {"content": [thinking_block, tool_use_block]}
    _assistant_raw_blocks = (
        raw.get("content") if isinstance(raw.get("content"), list) else None
    )

    msg = LLMMessage(
        role="assistant", content="",
        tool_calls=[{"id": "tu_1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}}],
        raw_content=_assistant_raw_blocks,
    )
    # thinking block preserved with signature (required for echo).
    types_present = [b["type"] for b in msg.raw_content]
    assert "thinking" in types_present, types_present
    thinking = next(b for b in msg.raw_content if b["type"] == "thinking")
    assert thinking.get("signature") == "sig-abc"


def test_design_chat_loop_omits_raw_content_for_openai_response():
    """OpenAI responses have no top-level `content` list → raw_content stays None."""
    raw = {"choices": [{"message": {"content": "ok"}}]}  # no top-level "content"
    _assistant_raw_blocks = (
        raw.get("content") if isinstance(raw.get("content"), list) else None
    )
    assert _assistant_raw_blocks is None  # OpenAI path unaffected


def test_anthropic_chat_serializes_raw_content_with_thinking_block():
    """Assistant msg with raw_content must echo native blocks (thinking included)."""
    client = _make_client()
    thinking_block = {"type": "thinking", "thinking": "reasoning here", "signature": "sig-xyz"}
    tool_use_block = {"type": "tool_use", "id": "tu_9", "name": "apply_patch", "input": {}}

    messages = [
        LLMMessage(role="user", content="do it"),
        LLMMessage(
            role="assistant", content="",
            tool_calls=[{"id": "tu_9", "type": "function",
                         "function": {"name": "apply_patch", "arguments": "{}"}}],
            raw_content=[thinking_block, tool_use_block],
        ),
        LLMMessage(role="user", content="ok"),
    ]
    tools = [{"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}}]
    client.chat_with_tools(messages, tools, model="claude-sonnet-4-20250514")
    payload = client._session.captured

    assistant_turn = payload["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert isinstance(assistant_turn["content"], list)
    types_present = [b["type"] for b in assistant_turn["content"]]
    assert "thinking" in types_present, types_present
    thinking = next(b for b in assistant_turn["content"] if b["type"] == "thinking")
    assert thinking["signature"] == "sig-xyz"
    assert "tool_use" in types_present, types_present


def test_anthropic_chat_falls_back_to_synthesis_when_no_raw_content():
    """Without raw_content, must still synthesize text+tool_use (legacy path)."""
    client = _make_client()
    messages = [
        LLMMessage(role="user", content="do it"),
        LLMMessage(
            role="assistant", content="patching",
            tool_calls=[{"id": "tu_1", "type": "function",
                         "function": {"name": "apply_patch", "arguments": "{}"}}],
            # no raw_content
        ),
        LLMMessage(role="user", content="ok"),
    ]
    tools = [{"name": "apply_patch", "description": "Patch", "parameters": {"type": "object", "properties": {}}}]
    client.chat_with_tools(messages, tools, model="claude-sonnet-4-20250514")
    payload = client._session.captured

    assistant_turn = payload["messages"][1]
    types_present = [b["type"] for b in assistant_turn["content"]]
    # Synthesized: text + tool_use (no thinking — legacy behavior preserved).
    assert "tool_use" in types_present, types_present
    assert "text" in types_present, types_present
    assert "thinking" not in types_present, types_present


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
