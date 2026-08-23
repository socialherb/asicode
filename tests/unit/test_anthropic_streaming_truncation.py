"""Anthropic/Z.AI streaming reliability: stop_reason normalization + truncation detection.

Two parity gaps vs the other providers, fixed in anthropic_client.py:

B1 — Anthropic reports truncation as stop_reason="max_tokens", but consumers
(agent_loop, agent_phase_manager) check OpenAI-style finish_reason=="length".
Gemini already normalizes at the provider boundary (providers.py
_normalize_gemini_finish_reason); Anthropic/Z.AI (GLM-5.2 — the default model)
did not, so a token-capped response silently bypassed the retry-on-truncation
logic (budget doubling + partial tool-call clearing in agent_loop).

B2 — Anthropic streaming lacked structural truncation detection: a response
silently cut mid-stream still arrives with stop_reason="end_turn" (the final
content delta is dropped but the terminating chunk arrives intact). Partial
tool-call JSON was quietly swallowed as args={} and JSON-shaped text was not
rechecked. These tests pin the fix that detects unbalanced delimiters via the
shared _count_delimiters helper (parity with providers.py DeepSeek and
openai_client.py) and rewrites finish_reason to "truncated".
"""

from __future__ import annotations

import pytest

from external_llm import anthropic_client as ac
from external_llm.anthropic_client import AnthropicClient
from external_llm.client import LLMMessage


class _FakeResponse:
    """Minimal 200 streaming response stand-in (status checks pass, close no-op)."""

    def __init__(self):
        self.status_code = 200
        self.headers: dict = {}

    def close(self) -> None:
        pass


class _FakeJsonResponse:
    """Non-streaming response stand-in: returns ``data`` from json()."""

    def __init__(self, data):
        self.data = data
        self.status_code = 200
        self.headers: dict = {}

    def json(self):
        return self.data


def _make_streaming_client(monkeypatch, events):
    """AnthropicClient whose streaming path yields ``events`` from a mocked SSE stream."""
    client = AnthropicClient(api_key="test")
    monkeypatch.setattr(client._session, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(ac, "iter_sse_data_events", lambda resp: iter(events))
    return client


_TOOLS = [{"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {}}}]


# ── B1: stop_reason normalization ───────────────────────────────────────────


def test_normalize_max_tokens_to_length():
    assert ac._normalize_anthropic_stop_reason("max_tokens") == "length"


def test_normalize_keeps_other_reasons():
    assert ac._normalize_anthropic_stop_reason("end_turn") == "end_turn"
    assert ac._normalize_anthropic_stop_reason("tool_use") == "tool_use"
    assert ac._normalize_anthropic_stop_reason("stop_sequence") == "stop_sequence"
    assert ac._normalize_anthropic_stop_reason(None) is None


def test_chat_with_tools_normalizes_max_tokens_finish_reason(monkeypatch):
    """Non-streaming chat_with_tools: stop_reason='max_tokens' → finish_reason='length'."""
    client = AnthropicClient(api_key="test")
    data = {
        "content": [{"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "x"}}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    monkeypatch.setattr(client._session, "post", lambda *a, **k: _FakeJsonResponse(data))
    resp = client.chat_with_tools([LLMMessage(role="user", content="hi")], _TOOLS, model="claude-sonnet-4-20250514")
    assert resp.finish_reason == "length"
    assert resp.tool_calls  # partial tool call still surfaced; agent_loop clears it on 'length'


def test_chat_normalizes_max_tokens_finish_reason(monkeypatch):
    """Non-streaming chat: stop_reason='max_tokens' → finish_reason='length'."""
    client = AnthropicClient(api_key="test")
    data = {
        "content": [{"type": "text", "text": "partial answer"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    monkeypatch.setattr(client._session, "post", lambda *a, **k: _FakeJsonResponse(data))
    resp = client.chat([LLMMessage(role="user", content="hi")], model="claude-sonnet-4-20250514")
    assert resp.finish_reason == "length"


def test_chat_with_tools_streaming_normalizes_max_tokens(monkeypatch):
    """Streaming message_delta stop_reason='max_tokens' → finish_reason='length'."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 3}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "length"


# ── B2: _chat_streaming text truncation ─────────────────────────────────────


def test_chat_streaming_detects_truncated_curly_json(monkeypatch):
    """stop_reason='end_turn' with unbalanced ``{`` text → rewritten to 'truncated'."""
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"key": "v", "nested":'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_chat_streaming_detects_truncated_square_json(monkeypatch):
    """stop_reason='end_turn' with unbalanced ``[`` text → rewritten to 'truncated'."""
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '[1, 2, 3, "four"'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_chat_streaming_keeps_balanced_json_end_turn(monkeypatch):
    """Balanced JSON-shaped text with stop_reason='end_turn' stays 'end_turn'."""
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": '{"key": "v", "nested": {"a": 1}}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "end_turn"


def test_chat_streaming_plain_text_untouched(monkeypatch):
    """Non-JSON text (no leading {/[) is never flagged as truncated."""
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "just some prose {"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "end_turn"


# ── B2: _chat_with_tools_streaming tool-call JSON truncation ────────────────


def test_tool_streaming_drops_truncated_input_json(monkeypatch, caplog):
    """Unbalanced tool input_json (final delta dropped) → call dropped + 'truncated'."""
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "x"'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    with caplog.at_level("WARNING", logger="external_llm.anthropic_client"):
        resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"
    assert resp.tool_calls == []  # partial args never executed
    assert "appear truncated" in caplog.text


def test_tool_streaming_keeps_balanced_unparseable_json_legacy(monkeypatch):
    """Balanced-but-unparseable input_json keeps legacy behavior: args={} + end_turn."""
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "x", }'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "end_turn"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].args == {}


def test_tool_streaming_keeps_valid_input_json(monkeypatch):
    """Well-formed tool input_json is appended normally; end_turn preserved."""
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "x"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "end_turn"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].args == {"path": "x"}


# ── Regression: previously-uncovered branches of B1+B2 ──────────────────────
# These pin code paths that the original 12 tests left uncovered. Each maps to
# a real wire: finish_reason=="length" drives agent_loop's budget-doubling
# retry + partial tool-call clearing (agent_loop.py:1020/1028); "truncated"
# drives agent_phase_manager's json-mode retry (agent_phase_manager.py:274).


def test_detect_stream_text_truncation_unit_contract(caplog):
    """Direct contract pins for the shared _detect_stream_text_truncation helper.

    Both _chat_streaming and _chat_with_tools_streaming depend on this helper,
    so its contract is pinned independently of the streaming plumbing.
    """
    # empty / whitespace-only → never truncated
    assert ac._detect_stream_text_truncation("Anthropic", "") is False
    assert ac._detect_stream_text_truncation("Anthropic", "   ") is False
    # leading whitespace is stripped before the {/[ leading check
    assert ac._detect_stream_text_truncation("Anthropic", '  {"a":') is True
    # non-JSON-leading text is never flagged, even with stray braces
    assert ac._detect_stream_text_truncation("Anthropic", "prose with } stray") is False
    # balanced JSON is fine
    assert ac._detect_stream_text_truncation("Anthropic", '{"a": 1}') is False
    # provider label surfaces in the warning so traces are diagnosable
    with caplog.at_level("WARNING", logger="external_llm.anthropic_client"):
        ac._detect_stream_text_truncation("ZAI", '{"k":')
    assert "ZAI" in caplog.text
    assert "unclosed braces" in caplog.text


def test_chat_streaming_normalizes_max_tokens_message_delta(monkeypatch):
    """B1 in the plain _chat_streaming path: message_delta max_tokens → 'length'.

    Previously only the tool-calling streaming path (_chat_with_tools_streaming)
    covered this. A token cap in the non-tool streaming path must still surface as
    finish_reason='length' so agent_loop's budget-doubling retry fires
    (agent_loop.py:1020). Removing the normalization at anthropic_client.py:900
    would silently bypass that retry.
    """
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial answer"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 3}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "length"


def test_tool_streaming_detects_truncated_text_content(monkeypatch):
    """B2 text-imbalance path in _chat_with_tools_streaming (distinct from tool-JSON drop).

    A JSON-shaped TEXT block (not a tool_use block) truncated mid-stream under
    tool-calling mode must rewrite finish_reason to 'truncated'. This branch
    (anthropic_client.py:1167-1172) was previously uncovered — only
    _chat_streaming's copy and the tool-JSON-drop path had tests.
    """
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"results": [{"id": 1,'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "truncated"


def test_truncated_tool_json_does_not_override_length(monkeypatch):
    """Precedence: max_tokens + dropped tool JSON keeps finish_reason='length'.

    anthropic_client.py:1164 guards `if _truncated_tool_json and stop_reason not
    in ("length", "truncated")`. When the server already reported max_tokens
    (→ 'length'), the tool-JSON drop must NOT rewrite it to 'truncated':
    agent_loop's length-retry (budget doubling + partial tool-call clearing) is
    the stronger signal and must win. The dropped call is still withheld so
    partial args are never executed.
    """
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "x"'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
    ]
    client = _make_streaming_client(monkeypatch, events)
    resp = client._chat_with_tools_streaming("http://x/v1/messages", {}, {"model": "m"}, "m", lambda c: None)
    assert resp.finish_reason == "length"
    assert resp.tool_calls == []  # partial args withheld regardless of precedence


def test_normalize_strips_whitespace_around_max_tokens():
    """The .strip() in _normalize_anthropic_stop_reason handles padded payloads."""
    assert ac._normalize_anthropic_stop_reason("  max_tokens  ") == "length"
    assert ac._normalize_anthropic_stop_reason("\tmax_tokens\n") == "length"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
