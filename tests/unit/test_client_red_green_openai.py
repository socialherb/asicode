"""RED→GREEN: external_llm/openai_client.py — remaining untested branches.

Covers: _strip_image_parts edge cases, error-body parsing variants,
thinking-mode payload mutation, retry timeout exhaustion, chat /
chat_with_tools error paths and message shaping, both streaming loops
(truncation detection, callback failure, usage/reasoning chunks, transport
failures), ZAIClient thinking kwargs, OpenRouterClient chat wiring.
"""

from __future__ import annotations

import json

import pytest
import requests

import external_llm.openai_client as oc
from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMCancelled,
    LLMMessage,
    LLMQuotaExceededError,
    LLMServerUnavailableError,
)
from external_llm.openai_client import (
    OpenAIClient,
    OpenRouterClient,
    ZAIClient,
    _apply_thinking_mode,
    _reasoning_effort_value,
    _short_error_reason,
    _strip_image_parts,
)


class _FakeResp:
    def __init__(self, status_code=200, text="", json_value=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._json_value = json_value
        self.headers = headers or {}
        self.closed = False

    def json(self):
        if self._json_value is not None:
            return self._json_value
        return json.loads(self.text)

    def close(self):
        self.closed = True


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


def _ok_choices(text="hi", finish="stop", usage=None, tool_calls=None):
    msg = {"content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage or {}}


def _client(monkeypatch, post, cls=OpenAIClient, **kw):
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    client = cls(api_key="test", **kw)
    monkeypatch.setattr(client._session, "post", post)
    return client


# ── _strip_image_parts ───────────────────────────────────────────────────────


def test_strip_image_parts_non_list_messages():
    assert _strip_image_parts({"messages": "nope"}) is None


def test_strip_image_parts_non_data_image_url():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]
    }
    out = _strip_image_parts(payload)
    assert out is not None
    content = out["messages"][0]["content"]
    assert "hello" in content


def test_strip_image_parts_content_without_images_unchanged():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "only text"},
                ],
            }
        ]
    }
    assert _strip_image_parts(payload) is None


# ── _short_error_reason ──────────────────────────────────────────────────────


def test_short_error_reason_string_error():
    assert _short_error_reason('{"error": "oops"}') == "oops"


def test_short_error_reason_invalid_json_after_brace():
    assert _short_error_reason('{"error": {bad') == '{"error": {bad'


def test_short_error_reason_truncation_ellipsis():
    out = _short_error_reason("x" * 300, limit=100)
    assert len(out) == 100 and out.endswith("…")


# ── reasoning/thinking helpers ───────────────────────────────────────────────


def test_reasoning_effort_value_thinking_medium():
    assert _reasoning_effort_value("gpt-5", True) == "medium"


def test_apply_thinking_mode_non_reasoning_override():
    payload: dict = {}
    _apply_thinking_mode(payload, "glm-5.2", None, "high", is_reasoning=False)
    assert payload["reasoning_effort"] == "high"


def test_apply_thinking_mode_reasoning_override_wins():
    payload: dict = {}
    _apply_thinking_mode(payload, "o4-mini", True, "low", is_reasoning=True)
    assert payload["reasoning_effort"] == "low"


# ── _request_with_retry ──────────────────────────────────────────────────────


def test_retry_timeout_exhaustion_raises_server_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise requests.Timeout("slow")

    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMServerUnavailableError, match="timed out"):
        client._request_with_retry("http://x", {}, {}, tag="chat")


def test_retry_connection_exhaustion_raises_server_unavailable(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("refused")

    client = _client(monkeypatch, _boom)
    with pytest.raises(LLMServerUnavailableError, match="Cannot connect"):
        client._request_with_retry("http://x", {}, {}, tag="chat")


def test_retry_cancel_during_backoff(monkeypatch):
    import threading

    ev = threading.Event()
    ev.set()
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(503, "err"))
    client.cancel_event = ev
    with pytest.raises(LLMCancelled):
        client._request_with_retry("http://x", {}, {}, tag="chat")


# ── chat() ───────────────────────────────────────────────────────────────────


def test_chat_default_model(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_choices())

    client = _client(monkeypatch, _post)
    resp = client.chat([LLMMessage(role="user", content="hi")], model="")
    assert resp.content == "hi"
    assert captured["model"] == "gpt-5.6-sol"


def test_chat_401(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(401, "nope"))
    with pytest.raises(LLMAuthenticationError, match="401"):
        client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")


def test_chat_402(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(402, "broke"))
    with pytest.raises(LLMQuotaExceededError, match="402"):
        client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")


def test_chat_no_choices_warns_and_returns_empty(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(json_value={"choices": []}))
    resp = client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")
    assert resp.content == ""


def test_chat_json_timeout_propagates(monkeypatch):
    fake = _FakeResp(json_value=_ok_choices())
    fake.json = lambda: (_ for _ in ()).throw(requests.Timeout("late"))

    def _post(*a, **k):
        return fake

    client = _client(monkeypatch, _post)
    with pytest.raises(requests.Timeout):
        client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")


def test_chat_json_request_exception_wrapped(monkeypatch):
    fake = _FakeResp(json_value=_ok_choices())
    fake.json = lambda: (_ for _ in ()).throw(requests.RequestException("bad"))

    def _post(*a, **k):
        return fake

    client = _client(monkeypatch, _post)
    with pytest.raises(LLMAPIError, match="API request failed"):
        client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")


def test_chat_streaming_path_via_token_callback(monkeypatch):
    calls = []
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4",
        token_callback=calls.append,
    )
    assert out.content == "Hello"
    assert calls == ["Hel", "lo"]
    assert out.prompt_tokens == 4 and out.completion_tokens == 2
    assert resp.closed


def test_chat_streaming_usage_cached_tokens(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":7}}}\n\n',
            b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert out.cache_read_input_tokens == 7


def test_chat_streaming_reasoning_content_accumulated(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"reasoning_content":"deep"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"ans"},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert out.raw_response["choices"][0]["message"]["reasoning_content"] == "deep"


def test_chat_streaming_callback_failure_tolerated(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)

    def _bad(_c):
        raise RuntimeError("ui broke")

    out = client.chat(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4",
        token_callback=_bad,
    )
    assert out.content == "x"


def test_chat_streaming_square_bracket_truncation(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"content":"[1, 2"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat(
        [LLMMessage(role="user", content="hi")],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert out.finish_reason == "truncated"


def test_chat_streaming_non200(monkeypatch):
    resp = _StreamResp([], status_code=400, text="boom")
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="400"):
        client.chat(
            [LLMMessage(role="user", content="hi")],
            model="gpt-4",
            token_callback=lambda _c: None,
        )
    assert resp.closed


def test_chat_streaming_transport_request_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.RequestException("wire broke")
            yield  # pragma: no cover

        def close(self):
            pass

    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="streaming request failed"):
        client.chat(
            [LLMMessage(role="user", content="hi")],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


def test_chat_streaming_unexpected_exception_wrapped(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise RuntimeError("framing bug")
            yield  # pragma: no cover

        def close(self):
            pass

    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat(
            [LLMMessage(role="user", content="hi")],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


# ── chat_with_tools() ────────────────────────────────────────────────────────


def test_chat_with_tools_default_model_and_message_shaping(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_choices())

    client = _client(monkeypatch, _post)
    msgs = [
        LLMMessage(
            role="assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
        ),
        LLMMessage(role="tool", content="res", tool_call_id="c1", name="f"),
    ]
    out = client.chat_with_tools(msgs, tools=[{"name": "f", "description": "d"}], model="")
    assert out.content == "hi"
    assert captured["model"] == "gpt-5.6-sol"
    api_msgs = captured["messages"]
    assert api_msgs[0]["tool_calls"] == msgs[0].tool_calls
    assert api_msgs[0]["content"] == ""
    assert api_msgs[1]["tool_call_id"] == "c1"
    assert api_msgs[1]["name"] == "f"
    assert captured["tools"][0]["function"]["name"] == "f"


def test_chat_with_tools_empty_tools_omits_keys(monkeypatch):
    # Empty tools must omit "tools"/"tool_choice" — same contract as the
    # DeepSeek/Google payload guards; an empty array 400s on some backends.
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_choices())

    client = _client(monkeypatch, _post)
    client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")
    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_chat_with_tools_401(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(401, "nope"))
    with pytest.raises(LLMAuthenticationError, match="401"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")


def test_chat_with_tools_402(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(402, "broke"))
    with pytest.raises(LLMQuotaExceededError, match="402"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")


def test_chat_with_tools_other_error(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(400, "bad"))
    with pytest.raises(LLMAPIError, match="400"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")


def test_chat_with_tools_no_choices(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: _FakeResp(json_value={"choices": []}))
    out = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")
    assert out.tool_calls == [] and out.is_final is True


def test_chat_with_tools_json_timeout_propagates(monkeypatch):
    fake = _FakeResp(json_value=_ok_choices())
    fake.json = lambda: (_ for _ in ()).throw(requests.Timeout("late"))
    client = _client(monkeypatch, lambda *a, **k: fake)
    with pytest.raises(requests.Timeout):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")


def test_chat_with_tools_json_request_exception_wrapped(monkeypatch):
    fake = _FakeResp(json_value=_ok_choices())
    fake.json = lambda: (_ for _ in ()).throw(requests.RequestException("bad"))
    client = _client(monkeypatch, lambda *a, **k: fake)
    with pytest.raises(LLMAPIError, match="API request failed"):
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")


def test_tools_streaming_tool_call_deltas(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\\"a\\":"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")],
        tools=[],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].call_id == "c1"
    assert out.tool_calls[0].name == "f"
    assert out.tool_calls[0].args == {"a": 1}
    assert out.is_final is False


def test_tools_streaming_bad_args_json_falls_back_to_empty(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"\\"unterminated"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")],
        tools=[],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].args == {}


def test_tools_streaming_square_bracket_text_truncation(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"content":"[1, 2"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")],
        tools=[],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert out.finish_reason == "truncated"


def test_tools_streaming_tool_args_truncated_cleared(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\\"a\\":"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    out = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")],
        tools=[],
        model="gpt-4",
        token_callback=lambda _c: None,
    )
    assert out.finish_reason == "truncated"
    assert out.tool_calls == []


def test_tools_streaming_callback_failure_tolerated(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)

    def _bad(_c):
        raise RuntimeError("ui broke")

    out = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")],
        tools=[],
        model="gpt-4",
        token_callback=_bad,
    )
    assert out.content == "x"


def test_tools_streaming_non200(monkeypatch):
    resp = _StreamResp([], status_code=400, text="boom")
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="400"):
        client.chat_with_tools(
            [LLMMessage(role="user", content="hi")],
            tools=[],
            model="gpt-4",
            token_callback=lambda _c: None,
        )
    assert resp.closed


def test_tools_streaming_transport_request_exception(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise requests.RequestException("wire broke")
            yield  # pragma: no cover

        def close(self):
            pass

    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="streaming request failed"):
        client.chat_with_tools(
            [LLMMessage(role="user", content="hi")],
            tools=[],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


def test_tools_streaming_unexpected_exception_wrapped(monkeypatch):
    class _Boom:
        status_code = 200

        def iter_bytes(self):
            raise RuntimeError("framing bug")
            yield  # pragma: no cover

        def close(self):
            pass

    client = _client(monkeypatch, lambda *a, **k: _Boom())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat_with_tools(
            [LLMMessage(role="user", content="hi")],
            tools=[],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


# ── ZAIClient thinking kwargs ────────────────────────────────────────────────


def _zai_client(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value={"choices": []})

    client = _client(monkeypatch, _post, cls=ZAIClient)
    return client, captured


def test_zai_chat_thinking_enabled_with_effort(monkeypatch):
    client, captured = _zai_client(monkeypatch)
    client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2", thinking_mode=True, reasoning_effort="high")
    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "high"


def test_zai_chat_thinking_disabled(monkeypatch):
    client, captured = _zai_client(monkeypatch)
    client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2", thinking_mode=False, reasoning_effort="high")
    assert captured["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured


def test_zai_chat_effort_without_thinking_toggle(monkeypatch):
    client, captured = _zai_client(monkeypatch)
    client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2", reasoning_effort="low")
    assert captured["reasoning_effort"] == "low"


def test_zai_chat_with_tools_thinking_enabled_with_effort(monkeypatch):
    client, captured = _zai_client(monkeypatch)
    client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2", thinking_mode=True, reasoning_effort="high"
    )
    assert captured["thinking"] == {"type": "enabled"}
    assert captured["reasoning_effort"] == "high"


def test_zai_chat_with_tools_effort_without_thinking(monkeypatch):
    client, captured = _zai_client(monkeypatch)
    client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2", reasoning_effort="minimal"
    )
    assert captured["reasoning_effort"] == "minimal"


# ── OpenRouterClient ─────────────────────────────────────────────────────────


def test_openrouter_log_cache_no_cached_tokens():
    OpenRouterClient._log_cache({"usage": {"prompt_tokens_details": {}}})


def test_openrouter_chat_strips_prefix_injects_pref_logs_cache(monkeypatch):
    captured = {}
    resp = _FakeResp(
        json_value=_ok_choices(usage={"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40}})
    )

    def _post(url, **kw):
        captured.update(kw["json"])
        return resp

    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek, OpenRouter")
    client = OpenRouterClient(api_key="sk-or-test")
    monkeypatch.setattr(client._session, "post", _post)
    out = client.chat([LLMMessage(role="user", content="hi")], model="openrouter/deepseek/deepseek-v4-flash")
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert captured["provider"] == {"order": ["DeepSeek", "OpenRouter"]}
    assert out.content == "hi"


def test_openrouter_chat_with_tools_prefix_stripped(monkeypatch):
    captured = {}

    def _post(url, **kw):
        captured.update(kw["json"])
        return _FakeResp(json_value=_ok_choices())

    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    client = OpenRouterClient(api_key="sk-or-test")
    monkeypatch.setattr(client._session, "post", _post)
    client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="openrouter/anthropic/claude-sonnet-4-6"
    )
    assert captured["model"] == "anthropic/claude-sonnet-4-6"


# ── remaining branch coverage ────────────────────────────────────────────────


def test_chat_streaming_non_dict_delta_wrapped(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":"oops"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat(
            [LLMMessage(role="user", content="hi")],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


def test_tools_streaming_non_dict_delta_wrapped(monkeypatch):
    resp = _StreamResp(
        [
            b'data: {"choices":[{"delta":"oops"}]}\n\n',
        ]
    )
    client = _client(monkeypatch, lambda *a, **k: resp)
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        client.chat_with_tools(
            [LLMMessage(role="user", content="hi")],
            tools=[],
            model="gpt-4",
            token_callback=lambda _c: None,
        )


def test_openrouter_log_cache_ptd_without_cached_tokens():
    OpenRouterClient._log_cache({"usage": {"prompt_tokens": 10, "prompt_tokens_details": {"foo": 1}}})


def test_apply_thinking_mode_reasoning_no_toggle_unchanged():
    # Contract pin: reasoning models WITHOUT an explicit thinking toggle must
    # NOT get a reasoning_effort dial injected (provider default applies) —
    # the effort passthrough fix applies to non-reasoning models only.
    payload: dict = {}
    _apply_thinking_mode(payload, "o4-mini", None, "low", is_reasoning=True)
    assert "reasoning_effort" not in payload
