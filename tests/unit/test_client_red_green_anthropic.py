"""RED→GREEN: external_llm/anthropic_client.py — remaining untested branches.

Covers: GLM error-code parsing, thinking-kwarg injection, the system-prompt
cache splitter (all block layouts + small-chunk merging), chat /
chat_with_tools message shaping (system merge, tool results, native blocks,
images), error statuses and transport failures on every path, both streaming
loops (thinking blocks, deltas, usage, callback failure, transport failure),
and ZAIAnthropicClient cache diagnostics + temperature quantization.
"""
from __future__ import annotations

import json

import pytest
import requests

from external_llm.anthropic_client import (
    AnthropicClient,
    ZAIAnthropicClient,
    _inject_glm_thinking_kwargs,
    _parse_glm_error_code,
)
from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMRateLimitError,
    LLMServerUnavailableError,
)


class _FakeResp:
    def __init__(self, status_code=200, text="", json_value=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._json_value = json_value
        self.headers = headers or {}

    def json(self):
        if self._json_value is not None:
            return self._json_value
        return json.loads(self.text)

    def close(self):
        pass


class _StreamResp:
    def __init__(self, chunks, status_code=200, text=""):
        self.status_code = status_code
        self.text = text
        self._chunks = list(chunks)
        self.closed = False

    def iter_bytes(self):
        yield from self._chunks

    def close(self):
        self.closed = True


def _ok_content(text="hi", stop="end_turn", usage=None, blocks=None):
    if blocks is None:
        blocks = [{"type": "text", "text": text}]
    return {"content": blocks, "stop_reason": stop, "usage": usage or {}}


def _client(monkeypatch, post, cls=AnthropicClient, **kw):
    client = cls(api_key="test", **kw)
    monkeypatch.setattr(client._session, "post", post)
    return client


# ── helpers ──────────────────────────────────────────────────────────────────

def test_parse_glm_error_code_int():
    class _R:
        text = '{"error":{"code":1305,"message":"overloaded"}}'
    assert _parse_glm_error_code(_R()) == 1305


def test_parse_glm_error_code_missing():
    class _R:
        text = '{"error":{"message":"x"}}'
    assert _parse_glm_error_code(_R()) is None


def test_inject_glm_thinking_disabled():
    kw: dict = {}
    _inject_glm_thinking_kwargs(kw, False, "high", True)
    assert kw == {"thinking": {"type": "disabled"}}


def test_inject_glm_thinking_adaptive_with_effort():
    kw: dict = {}
    _inject_glm_thinking_kwargs(kw, None, "high", True)
    assert kw == {"thinking": {"type": "adaptive"}, "effort": "high"}


def test_inject_glm_thinking_enabled_with_effort():
    kw: dict = {}
    _inject_glm_thinking_kwargs(kw, True, "medium", False)
    assert kw == {"thinking": {"type": "enabled"}, "effort": "medium"}


def test_inject_glm_thinking_enabled_no_effort():
    kw: dict = {}
    _inject_glm_thinking_kwargs(kw, True, None, False)
    assert kw == {"thinking": {"type": "enabled"}}


def test_inject_glm_thinking_noop():
    kw: dict = {}
    _inject_glm_thinking_kwargs(kw, None, None, False)
    assert kw == {}


# ── _split_system_with_caching ───────────────────────────────────────────────

def test_split_short_system_single_block():
    out = AnthropicClient._split_system_with_caching("x" * 300)
    assert out == [{"type": "text", "text": "x" * 300}]


def test_split_no_headers_single_block():
    out = AnthropicClient._split_system_with_caching("abc\n" * 200)
    assert len(out) == 1


def test_split_tools_marker_with_next_header_three_block():
    sys = ("intro " * 300) + "## Available Tools\n" + "\n## Next\n" + "tail"
    out = AnthropicClient._split_system_with_caching(sys)
    assert [b["type"] for b in out] == ["text", "text", "text"]
    assert out[0]["cache_control"] == {"type": "ephemeral"}


def test_split_tools_marker_small_chunk_merged_into_marker_block():
    # chunk1 (before marker) < 1000 chars → merged into chunk2 which carries
    # the cache_control marker (rest[0]-marker branch of _merge_small_cached).
    sys = ("intro " * 100) + "## Available Tools\n" + "\n## Next\n" + "tail"
    out = AnthropicClient._split_system_with_caching(sys)
    assert len(out) == 2
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["text"].startswith("intro ")


def test_split_tools_marker_small_chunk_merged_head_marker():
    # No second header after the marker → rest[0] has NO cache_control, so the
    # merge keeps the head chunk's marker (head-marker branch).
    # head (600 chars) + chunk2 → a SINGLE merged block.
    sys = ("intro " * 100) + "## Available Tools\n" + "rest text"
    out = AnthropicClient._split_system_with_caching(sys)
    assert len(out) == 1
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["text"].startswith("intro ")
    assert "rest text" in out[0]["text"]


def test_split_starts_with_header_no_second_single_block():
    sys = "## Title\n" + "x" * 600
    out = AnthropicClient._split_system_with_caching(sys)
    assert len(out) == 1


def test_split_general_three_block():
    sys = ("intro " * 300) + "\n## Sec1\n" + "y" * 100 + "\n## Sec2\n" + "x" * 500
    out = AnthropicClient._split_system_with_caching(sys)
    assert len(out) == 3
    assert out[0]["text"].startswith("intro ")
    assert "## Sec1" in out[1]["text"]
    assert "## Sec2" in out[2]["text"]


def test_split_general_two_block():
    sys = ("intro " * 300) + "\n## Sec2\n" + "x" * 300
    out = AnthropicClient._split_system_with_caching(sys)
    assert len(out) == 2
    assert out[0]["text"].startswith("intro ")


# ── chat() ───────────────────────────────────────────────────────────────────

def test_chat_default_model_and_system_merge(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    msgs = [
        LLMMessage(role="system", content="rules"),
        LLMMessage(role="system", content="extra"),
        LLMMessage(role="user", content="hi"),
    ]
    resp = client.chat(msgs, model="")
    assert resp.content == "hi"
    assert captured["model"] == "claude-sonnet-5"
    assert captured["system"] == [{"type": "text", "text": "rules\n\nextra"}]


def test_chat_always_thinking_model_effort(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat([LLMMessage(role="user", content="hi")], model="claude-opus-4-8",
                reasoning_effort="high")
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "high"


def test_chat_always_thinking_model_thinking_mode_effort(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat([LLMMessage(role="user", content="hi")], model="claude-opus-4-8",
                thinking_mode=False)
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "low"


def test_chat_thinking_mode_true_with_effort(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                thinking_mode=True, reasoning_effort="high")
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "high"


def test_chat_401(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(401, "nope"))
    with pytest.raises(LLMAuthenticationError, match="API key"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


def test_chat_5xx(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(503, "down"))
    with pytest.raises(LLMServerUnavailableError, match="503"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


def test_chat_other_error(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(400, "bad"))
    with pytest.raises(LLMAPIError, match="400"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


def test_chat_no_content_blocks(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(json_value={"content": []}))
    resp = client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")
    assert resp.content == ""


def test_chat_connection_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError, match="Cannot connect"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


def test_chat_timeout_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.Timeout("slow")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError, match="timed out"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


def test_chat_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.RequestException("bad")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMAPIError, match="request failed"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")


# ── chat_with_tools() ────────────────────────────────────────────────────────

def test_chat_with_tools_default_model_and_message_shaping(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    msgs = [
        LLMMessage(role="tool", content="res", tool_call_id="t1"),
        LLMMessage(role="assistant", content="interim", tool_calls=[
            {"id": "tc1", "function": {"name": "f", "arguments": '{"a": 1}'}},
        ]),
        LLMMessage(role="user", content="continue",
                   raw_content=[{"type": "text", "text": "native block"}]),
    ]
    out = client.chat_with_tools(msgs, tools=[{"name": "f", "description": "d"}], model="")
    assert out.content == "hi"
    assert captured["model"] == "claude-sonnet-5"
    assert captured["max_tokens"] == 65536
    api = captured["messages"]
    assert api[0] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "res"}]}
    assert api[1]["content"][0] == {"type": "text", "text": "interim"}
    assert api[1]["content"][1]["type"] == "tool_use"
    assert api[1]["content"][1]["input"] == {"a": 1}
    assert api[2]["content"] == [
        {"type": "text", "text": "native block", "cache_control": {"type": "ephemeral"}},
    ]
    assert captured["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


def test_chat_with_tools_assistant_bad_args_json(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    msgs = [
        LLMMessage(role="assistant", content="", tool_calls=[
            {"id": "tc1", "function": {"name": "f", "arguments": "{bad json"}},
        ]),
        LLMMessage(role="user", content="go"),
    ]
    client.chat_with_tools(msgs, tools=[], model="claude-3-5-sonnet-20241022")
    blocks = captured["messages"][0]["content"]
    assert len(blocks) == 1  # empty content → no text block, only tool_use
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["input"] == {}


def test_chat_with_tools_images(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    msgs = [LLMMessage(role="user", content="look",
                       images=[{"media_type": "image/png", "data": "abc"}])]
    client.chat_with_tools(msgs, tools=[], model="claude-3-5-sonnet-20241022")
    content = captured["messages"][0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
    assert content[1] == {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}}


def test_chat_with_tools_always_thinking(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="claude-opus-4-8", reasoning_effort="high")
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "high"
    assert "temperature" not in captured


def test_chat_with_tools_thinking_true(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="claude-3-5-sonnet-20241022", thinking_mode=True)
    assert captured["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured


def test_chat_with_tools_401(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(401, "nope"))
    with pytest.raises(LLMAuthenticationError):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


def test_chat_with_tools_5xx(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(503, "down"))
    with pytest.raises(LLMServerUnavailableError, match="503"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


def test_chat_with_tools_other_error(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(400, "bad"))
    with pytest.raises(LLMAPIError, match="400"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


def test_chat_with_tools_connection_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


def test_chat_with_tools_timeout_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.Timeout("slow")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError, match="timed out"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


def test_chat_with_tools_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.RequestException("bad")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMAPIError, match="request failed"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")


# ── _chat_streaming ──────────────────────────────────────────────────────────

def _sse(*events):
    return [("data: " + json.dumps(ev) + "\n\n").encode() for ev in events]


def test_chat_streaming_basic_and_usage(monkeypatch):
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10,
                                                          "cache_read_input_tokens": 3,
                                                          "cache_creation_input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
    ]
    calls = []
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                      token_callback=calls.append)
    assert out.content == "Hello"
    assert calls == ["Hel", "lo"]
    assert out.finish_reason == "end_turn"
    assert out.prompt_tokens == 10
    assert out.completion_tokens == 5
    assert out.cache_read_input_tokens == 3
    assert out.raw_response["usage"]["cache_creation_input_tokens"] == 2
    assert resp.closed


def test_chat_streaming_thinking_blocks_and_unknown_block(monkeypatch):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "deep"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig1"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "weird"}},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "ans"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                      token_callback=lambda _c: None)
    assert out.content == "ans"
    blocks = out.raw_response["content"]
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "deep"
    assert blocks[0]["signature"] == "sig1"


def test_chat_streaming_callback_failure_tolerated(monkeypatch):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)

    def _bad(_c):
        raise RuntimeError("ui")
    out = client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                      token_callback=_bad)
    assert out.content == "x"


def test_chat_streaming_429(monkeypatch):
    resp = _FakeResp(429, '{"error":{"code":"1302","message":"rate"}}')
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMRateLimitError):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_chat_streaming_5xx(monkeypatch):
    resp = _FakeResp(503, "down")
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMServerUnavailableError, match="503"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_chat_streaming_post_connection_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_chat_streaming_post_timeout_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.Timeout("slow")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError, match="timed out"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_chat_streaming_iter_request_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.RequestException("wire")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="streaming request failed"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_chat_streaming_iter_unexpected_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise RuntimeError("framing")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


# ── _chat_with_tools_streaming ───────────────────────────────────────────────

def test_tools_streaming_tool_use_and_thinking(monkeypatch):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu1", "name": "f"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": "deep"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "done"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_start", "message": {"usage": {"input_tokens": 7}}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                                 model="claude-3-5-sonnet-20241022",
                                 token_callback=lambda _c: None)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].call_id == "tu1"
    assert out.tool_calls[0].args == {"a": 1}
    assert out.content == "done"
    assert out.prompt_tokens == 7
    assert out.completion_tokens == 3
    raw = out.raw_response["content"]
    assert raw[0] == {"type": "thinking", "thinking": "deep", "signature": "sig"}
    assert raw[-1]["type"] == "tool_use"


def test_tools_streaming_callback_failure_tolerated(monkeypatch):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)

    def _bad(_c):
        raise RuntimeError("ui")
    out = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                                 model="claude-3-5-sonnet-20241022", token_callback=_bad)
    assert out.content == "x"


def test_tools_streaming_429(monkeypatch):
    resp = _FakeResp(429, '{"error":{"code":"1302","message":"rate"}}')
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMRateLimitError):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_5xx(monkeypatch):
    resp = _FakeResp(503, "down")
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMServerUnavailableError, match="503"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_post_connection_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_post_timeout_error(monkeypatch):
    def _boom(*a, **k):
        raise requests.Timeout("slow")
    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMConnectionError, match="timed out"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_iter_connection_error(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.ConnectionError("lost")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMConnectionError):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_iter_timeout_error(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.Timeout("slow")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMConnectionError, match="timed out"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_iter_request_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.RequestException("wire")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="streaming request failed"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


def test_tools_streaming_iter_unexpected_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise RuntimeError("framing")
            yield  # pragma: no cover

        def close(self):
            pass
    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)


# ── ZAIAnthropicClient ───────────────────────────────────────────────────────

def test_zai_get_provider_name():
    assert ZAIAnthropicClient(api_key="k").get_provider_name() == "zai"


def test_zai_chat_cache_diagnostics(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content(usage={"input_tokens": 100,
                                                       "cache_read_input_tokens": 50}))
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    resp = client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")
    assert resp.content == "hi"
    assert captured["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured


def test_zai_chat_no_raw_response_usage(monkeypatch):
    def _post(url, **kw):
        return _FakeResp(json_value={"content": []})
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    resp = client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")
    assert resp.content == ""


def test_zai_chat_with_tools_temperature_quantized(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="glm-4.6", temperature=0.123456)
    assert captured["temperature"] == 0.12


def test_zai_chat_with_tools_temperature_dropped_when_thinking(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="glm-5.2", temperature=0.9)
    assert captured["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured


def test_zai_chat_with_tools_cache_diagnostics(monkeypatch):
    def _post(url, **kw):
        return _FakeResp(json_value=_ok_content(usage={"input_tokens": 200,
                                                       "cache_read_input_tokens": 80}))
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    resp = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                                  model="glm-5.2")
    assert resp.content == "hi"


def test_zai_chat_with_tools_no_raw_response_usage(monkeypatch):
    def _post(url, **kw):
        return _FakeResp(json_value={"content": []})
    client = _client(monkeypatch, _post, cls=ZAIAnthropicClient)
    resp = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                                  model="glm-5.2")
    assert resp.content == ""


# ── remaining branch coverage ────────────────────────────────────────────────

def test_chat_with_tools_always_thinking_thinking_mode_effort(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="claude-opus-4-8", thinking_mode=False)
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "low"


def test_chat_with_tools_thinking_true_with_effort(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_content())
    client = _client(monkeypatch, _post)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                           model="claude-3-5-sonnet-20241022",
                           thinking_mode=True, reasoning_effort="medium")
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["effort"] == "medium"


def test_chat_streaming_message_delta_usage_fields(monkeypatch):
    events = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"input_tokens": 99, "output_tokens": 5,
                   "cache_read_input_tokens": 4, "cache_creation_input_tokens": 1}},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                      token_callback=lambda _c: None)
    assert out.prompt_tokens == 99
    assert out.completion_tokens == 5
    assert out.cache_read_input_tokens == 4
    assert out.raw_response["usage"]["cache_creation_input_tokens"] == 1


def test_chat_streaming_non_dict_delta_wrapped(monkeypatch):
    # A malformed server frame (delta as a string) must surface as a typed
    # LLMAPIError via the catch-all except, not as a raw AttributeError.
    events = [
        {"type": "content_block_delta", "index": 0, "delta": "oops"},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022",
                    token_callback=lambda _c: None)


def test_tools_streaming_non_dict_delta_wrapped(monkeypatch):
    events = [
        {"type": "content_block_delta", "index": 0, "delta": "oops"},
    ]
    resp = _StreamResp(_sse(*events))
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[],
                               model="m", token_callback=lambda _c: None)
