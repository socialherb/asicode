"""OpenAI streaming truncation detection (parity with DeepSeek).

The OpenAI-compatible streaming clients (``_chat_streaming`` and
``_chat_with_tools_streaming``) share the same SSE architecture as DeepSeek's
``chat_with_tools`` but historically lacked the structural-truncation
detection: a response can be silently cut mid-stream yet still arrive with
``finish_reason='stop'``/``'tool_calls'`` (the final delta is dropped but the
terminating chunk is intact). These tests pin the fix that detects unbalanced
JSON delimiters and rewrites ``finish_reason`` to ``'truncated'``, using the
shared ``_count_delimiters`` helper — mirroring providers.py DeepSeek exactly.
"""

from __future__ import annotations

from external_llm import openai_client as oc
from external_llm.openai_client import OpenAIClient


class _FakeResponse:
    """Minimal 200 streaming response stand-in (status checks pass, close is a no-op)."""

    def __init__(self):
        self.status_code = 200
        self.headers: dict = {}

    def close(self) -> None:
        pass


def _make_streaming_client(monkeypatch, events):
    """Return an OpenAIClient whose streaming path yields ``events`` from a mocked SSE stream."""
    client = OpenAIClient(api_key="test")
    monkeypatch.setattr(client, "_request_with_retry", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(oc, "iter_sse_data_events", lambda resp: iter(events))
    return client


# ── _chat_streaming: content truncation ──────────────────────────────────────


def test_chat_streaming_detects_truncated_curly_json(monkeypatch):
    """finish_reason='stop' with unbalanced ``{`` → rewritten to 'truncated'."""
    events = [
        {"choices": [{"delta": {"content": '{"key": "v", "nested":'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_chat_streaming_detects_truncated_square_json(monkeypatch):
    """finish_reason='stop' with unbalanced ``[`` → rewritten to 'truncated'."""
    events = [
        {"choices": [{"delta": {"content": '[1, 2, 3, "four"'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_chat_streaming_balanced_json_stays_stop(monkeypatch):
    """Balanced JSON with finish_reason='stop' is NOT flagged as truncated."""
    events = [
        {"choices": [{"delta": {"content": '{"key": "value", "n": [1, 2, 3]}'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "stop"


def test_chat_streaming_non_json_content_not_checked(monkeypatch):
    """Plain text content (not JSON) is never flagged — only ``{``/``[``-prefixed content is checked."""
    events = [
        {"choices": [{"delta": {"content": "Hello, this is a normal text reply."}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "stop"


def test_chat_streaming_brace_inside_string_does_not_false_trigger(monkeypatch):
    """A ``}`` that appears INSIDE a JSON string literal does not balance a missing outer brace.

    Guards the string-awareness of _count_delimiters at the integration boundary.
    """
    # The object is missing its real closing brace; the only } is INSIDE the
    # string literal and must not count as a closer. A non-string-aware counter
    # would wrongly balance it (false negative). Here open_curly=1, close=0.
    events = [
        {"choices": [{"delta": {"content": '{"msg": "has } stray", "ne'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


# ── _chat_with_tools_streaming: tool-call args truncation ─────────────────────


def _tc_delta(idx, *, id_=None, name=None, arguments=None, finish=None):
    delta: dict = {}
    tc: dict = {"index": idx}
    if id_:
        tc["id"] = id_
    func: dict = {}
    if name:
        func["name"] = name
    if arguments:
        func["arguments"] = arguments
    if func:
        tc["function"] = func
    if tc:
        delta["tool_calls"] = [tc]
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def test_chat_with_tools_streaming_detects_truncated_args(monkeypatch):
    """finish_reason='tool_calls' with unbalanced args JSON → 'truncated' + tool_calls cleared.

    Clearing malformed tool calls mirrors DeepSeek's full_tool_calls_raw.clear(): the
    caller (agent_loop) only retries on 'length', so clearing yields graceful
    degradation (content preserved, no partial tool execution).
    """
    events = [
        _tc_delta(0, id_="call_1", name="write_file", arguments='{"path": "/x"'),
        _tc_delta(0, arguments=', "content":'),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"
    assert resp.tool_calls == []


def test_chat_with_tools_streaming_balanced_args_keeps_tool_calls(monkeypatch):
    """Balanced tool-call args with finish_reason='tool_calls' are preserved."""
    events = [
        _tc_delta(0, id_="call_1", name="read_file", arguments='{"path": "/x/y.py"}'),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"


def test_chat_with_tools_streaming_truncated_content_json(monkeypatch):
    """Tools path also validates text content: truncated JSON content → 'truncated'."""
    events = [
        {"choices": [{"delta": {"content": '{"partial":'}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_chat_with_tools_streaming_partial_args_with_no_arguments_not_flagged(monkeypatch):
    """A tool call whose args are empty (no arguments delta yet) is not flagged."""
    events = [
        _tc_delta(0, id_="call_1", name="noop"),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "tool_calls"
