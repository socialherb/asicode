"""ZAIClient GLM-5.2 reasoning_content fallback parity (chat() ↔ chat_with_tools()).

GLM-5.2 (thinking ON) may emit the final answer in ``reasoning_content`` with
an empty ``content`` field. ``chat()`` recovered this via an inline fallback;
``chat_with_tools()`` did NOT — yet ``chat_with_tools`` feeds
intent_resolver / agent_phase_manager / planner_plan_create / orchestrator /
design_chat, all of which read ``resp.content`` directly and would silently
get an empty string (the model's decision lost).

These tests pin the shared ``_apply_glm_reasoning_fallback`` helper and the
critical ``tool_calls`` guard: on a tool-call turn empty ``content`` is NORMAL
(the tool calls ARE the response), so reasoning must NOT be injected there.
"""
from __future__ import annotations

import json


import external_llm.openai_client as oc
from external_llm.client import LLMMessage, LLMResponse, ToolCallResponse
from external_llm.openai_client import ZAIClient


# ── response builders ──────────────────────────────────────────────────────

def _toolcall_resp(*, content="", reasoning="", tool_calls=None):
    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    raw = {
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return ToolCallResponse(
        content=content, model="glm-5.2", provider="zai",
        tokens_used=15, finish_reason="stop",
        raw_response=raw, tool_calls=tool_calls or [], is_final=True,
    )


def _plain_resp(*, content="", reasoning=""):
    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    raw = {
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return LLMResponse(
        content=content, model="glm-5.2", provider="zai",
        tokens_used=15, finish_reason="stop", raw_response=raw,
    )


# ── 1. helper: direct unit tests ───────────────────────────────────────────

def test_helper_recovers_reasoning_when_content_empty_no_tools():
    r = _toolcall_resp(content="", reasoning="## Done: changes applied")
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == "## Done: changes applied"


def test_helper_keeps_content_when_present():
    r = _toolcall_resp(content="real answer", reasoning="thinking...")
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == "real answer"


def test_helper_tool_calls_guard_blocks_injection():
    """A tool-call turn legitimately has empty content — do NOT inject reasoning."""
    tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    r = _toolcall_resp(content="", reasoning="internal reasoning", tool_calls=tc)
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == ""  # unchanged — tool_calls guard fired
    assert out.tool_calls == tc


def test_helper_empty_content_no_reasoning_returns_empty():
    r = _toolcall_resp(content="", reasoning="")
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == ""


def test_helper_no_raw_response_returns_unchanged():
    r = ToolCallResponse(content="", model="glm-5.2", provider="zai",
                         tokens_used=0, finish_reason="stop",
                         raw_response=None, tool_calls=[])
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == ""


def test_helper_plain_llmresponse_recovers_reasoning():
    """chat() returns LLMResponse (no tool_calls attr) — getattr guard never blocks."""
    r = _plain_resp(content="", reasoning="final answer in reasoning")
    out = ZAIClient._apply_glm_reasoning_fallback(r)
    assert out.content == "final answer in reasoning"


# ── 2. full path: chat_with_tools (the bug being fixed) ────────────────────

class _OK:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    @property
    def text(self):
        return json.dumps(self._data)


def _payload_with_reasoning(*, content="", reasoning="", tool_calls=None):
    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _zai_client(monkeypatch, payload):
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    client = ZAIClient(api_key="test")
    monkeypatch.setattr(client._session, "post", lambda *a, **k: _OK(payload))
    return client


def test_chat_with_tools_recovers_reasoning_content(monkeypatch):
    """THE BUG: chat_with_tools now recovers an answer that arrived only in reasoning_content."""
    client = _zai_client(monkeypatch, _payload_with_reasoning(content="", reasoning="resolved intent JSON"))
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2",
    )
    assert resp.content == "resolved intent JSON"


def test_chat_with_tools_tool_calls_turn_not_injected(monkeypatch):
    """Empty content WITH tool calls is a normal tool turn — reasoning stays out of content."""
    tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    client = _zai_client(monkeypatch, _payload_with_reasoning(content="", reasoning="thinking", tool_calls=tc))
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2",
    )
    assert resp.content == ""  # guard held
    assert resp.tool_calls  # tool calls preserved


def test_chat_with_tools_content_present_unchanged(monkeypatch):
    client = _zai_client(monkeypatch, _payload_with_reasoning(content="plain answer", reasoning="ignored"))
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2",
    )
    assert resp.content == "plain answer"


# ── 3. full path: chat() regression ────────────────────────────────────────

def test_chat_recovers_reasoning_content(monkeypatch):
    """chat() fallback preserved after extracting the shared helper."""
    client = _zai_client(monkeypatch, _payload_with_reasoning(content="", reasoning="design answer"))
    resp = client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")
    assert resp.content == "design answer"
