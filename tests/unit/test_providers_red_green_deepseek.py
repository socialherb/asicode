"""RED→GREEN: DeepSeekClient — chat / _chat_streaming / chat_with_tools full
branch coverage via fake sessions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

import external_llm.providers as providers_module
from external_llm import openai_client
from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from external_llm.model_catalog import KNOWN_MODELS
from external_llm.model_registry import vision_capable
from external_llm.providers import DeepSeekClient


def _resp(*, status=200, json_data=None, sse=None, headers=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    if sse is not None:
        r.iter_bytes.return_value = iter(sse)
    return r


def _client(resp=None) -> DeepSeekClient:
    c = DeepSeekClient(api_key="test-key")
    c._session = MagicMock()
    if resp is not None:
        c._session.post.return_value = resp
    return c


def _ok_json(content="hi", finish="stop", usage=None):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage or {"total_tokens": 9, "prompt_tokens": 4, "completion_tokens": 5},
    }


def test_get_provider_name() -> None:
    assert DeepSeekClient(api_key="k").get_provider_name() == "deepseek"


# ── chat (non-streaming) ─────────────────────────────────────────────────────


def test_chat_default_model_and_message_serialization(monkeypatch) -> None:
    """Empty ``model`` falls back to ``DEFAULT_MODEL``, and that id is declared to
    read images — so the attachment must go out as a part, not as OCR text (the OCR
    branch belongs to text-only ids; see
    test_text_only_model_still_folds_images_into_text).
    """
    monkeypatch.setattr(
        providers_module,
        "_images_to_text",
        lambda _imgs: pytest.fail("the default model reads images — OCR must not run"),
    )
    c = _client(_resp(json_data=_ok_json()))
    c.chat(
        [
            LLMMessage(role="system", content="S"),
            LLMMessage(role="user", content="u", images=_image_uri()),
            LLMMessage(role="assistant", content="a", reasoning_content="cot"),
            LLMMessage(role="assistant", content="b"),
        ],
        model="",
        max_tokens=50,
    )
    sent = c._session.post.call_args
    assert "/chat/completions" in sent.args[0]
    payload = sent.kwargs["json"]
    assert payload["model"] == "deepseek-flash"
    assert payload["max_tokens"] == 50
    assert payload["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "u"},
        ],
    }
    assert payload["messages"][2]["reasoning_content"] == "cot"


def test_chat_reasoner_adds_empty_reasoning_content() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="assistant", content="a")], model="deepseek-reasoner")
    msgs = c._session.post.call_args.kwargs["json"]["messages"]
    assert msgs[0]["reasoning_content"] == ""


def test_chat_thinking_modes() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4", thinking_mode=False)
    assert c._session.post.call_args.kwargs["json"]["thinking"] == {"type": "disabled"}

    c2 = _client(_resp(json_data=_ok_json()))
    c2.chat([LLMMessage(role="user", content="x")], model="deepseek-v4", thinking_mode=True, reasoning_effort="max")
    payload = c2._session.post.call_args.kwargs["json"]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"


def test_chat_routes_to_streaming_when_callbacks(monkeypatch) -> None:
    c = _client()
    called = {}

    def fake_stream(url, headers, payload, model, token_callback=None, reasoning_callback=None):
        called["stream"] = (payload, token_callback, reasoning_callback)
        return "STREAMED"

    monkeypatch.setattr(c, "_chat_streaming", fake_stream)
    tc = lambda _c: None  # noqa: E731
    rc = lambda _c: None  # noqa: E731
    out = c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4", token_callback=tc, reasoning_callback=rc)
    assert out == "STREAMED"
    _payload, ptc, prc = called["stream"]
    assert ptc is tc and prc is rc


def test_chat_status_errors() -> None:
    for status, exc in [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (402, LLMQuotaExceededError),
        (429, LLMRateLimitError),
        (503, LLMServerUnavailableError),
        (500, LLMServerUnavailableError),
        (501, LLMAPIError),
        (400, LLMAPIError),
    ]:
        c = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4")


def test_chat_429_retry_after() -> None:
    c = _client(_resp(status=429, headers={"Retry-After": "7"}))
    with pytest.raises(LLMRateLimitError) as ei:
        c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4")
    assert ei.value.retry_after == 7


def test_chat_no_choices() -> None:
    c = _client(_resp(json_data={"choices": []}))
    r = c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4")
    assert r.content == ""


def test_chat_full_usage_extraction() -> None:
    usage = {
        "total_tokens": 20,
        "prompt_tokens": 8,
        "completion_tokens": 12,
        "prompt_cache_hit_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    c = _client(_resp(json_data=_ok_json(content="ans", finish="length", usage=usage)))
    r = c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4")
    assert r.content == "ans" and r.finish_reason == "length"
    assert r.tokens_used == 20 and r.prompt_tokens == 8 and r.completion_tokens == 12
    assert r.cache_read_input_tokens == 5 and r.reasoning_tokens == 3


def test_chat_requests_exceptions() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMServerUnavailableError),
        (requests.Timeout("t"), LLMServerUnavailableError),
        (requests.RequestException("r"), LLMAPIError),
    ]:
        c = _client()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat([LLMMessage(role="user", content="x")], model="deepseek-v4")


# ── _chat_streaming ──────────────────────────────────────────────────────────


def test_chat_streaming_status_errors() -> None:
    for status, exc in [
        (401, LLMAuthenticationError),
        (402, LLMQuotaExceededError),
        (429, LLMRateLimitError),
        (503, LLMServerUnavailableError),
        (500, LLMServerUnavailableError),
        (501, LLMAPIError),
        (400, LLMAPIError),
    ]:
        c = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)


def test_chat_streaming_full_flow() -> None:
    def chunk(d, fin=None):
        ev = {"choices": [{"delta": d}]}
        if fin:
            ev["choices"][0]["finish_reason"] = fin
        return ("data: " + json.dumps(ev) + "\n\n").encode()

    sse = [
        chunk({"content": "Hel"}, None),
        chunk({"reasoning_content": "think"}, None),
        ("data: " + json.dumps({"choices": [], "usage": {"total_tokens": 6}}) + "\n\n").encode(),
        chunk({"content": "lo"}, "stop"),
        (
            "data: "
            + json.dumps(
                {"choices": [{"delta": {}}], "usage": {"total_tokens": 9, "prompt_tokens": 4, "completion_tokens": 5}}
            )
            + "\n\n"
        ).encode(),
    ]
    c = _client(_resp(sse=sse))
    got_tok, got_reason = [], []
    r = c._chat_streaming(
        "u",
        {},
        {"model": "m"},
        "m",
        token_callback=got_tok.append,
        reasoning_callback=got_reason.append,
    )
    assert r.content == "Hello"
    assert got_tok == ["Hel", "lo"]
    assert got_reason == ["think"]
    assert r.finish_reason == "stop"
    assert r.tokens_used == 9 and r.prompt_tokens == 4 and r.completion_tokens == 5
    assert r.raw_response["streamed"] is True
    assert r.raw_response["choices"][0]["message"]["reasoning_content"] == "think"


def test_chat_streaming_iteration_exceptions() -> None:
    def _boom(raise_exc):
        yield b'data: {"choices": [{"delta": {"content": "a"}}]}\n'
        raise raise_exc

    cases = [
        (requests.exceptions.ChunkedEncodingError("c"), LLMServerUnavailableError),
        (requests.RequestException("r"), LLMAPIError),
        (RuntimeError("plain"), LLMAPIError),
    ]
    for exc, expected in cases:
        c = _client(_resp(sse=_boom(exc)))
        with pytest.raises(expected):
            c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)

    # typed LLMClientError passes through
    c = _client(_resp(sse=_boom(LLMRateLimitError("rl"))))
    with pytest.raises(LLMRateLimitError):
        c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)


def test_chat_streaming_post_connection_errors() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMServerUnavailableError),
        (requests.Timeout("t"), LLMServerUnavailableError),
    ]:
        c = _client()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)


# ── chat_with_tools (streaming-only path) ────────────────────────────────────


def _tc_chunk(delta, fin=None, usage=None):
    ev = {"choices": [{"delta": delta}]}
    if fin:
        ev["choices"][0]["finish_reason"] = fin
    if usage:
        ev["usage"] = usage
    return ("data: " + json.dumps(ev) + "\n\n").encode()


def test_chat_with_tools_default_model_and_messages() -> None:
    c = _client(_resp(sse=[_tc_chunk({"content": "ok"}, "stop")]))
    c.chat_with_tools(
        [
            LLMMessage(role="system", content="S"),
            LLMMessage(role="user", content="u", images=_image_uri()),
            LLMMessage(
                role="assistant",
                content=None,
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            ),
            LLMMessage(role="tool", content="res", tool_call_id="c1", name="f"),
        ],
        tools=[{"name": "f", "description": "d", "parameters": {"type": "object"}}],
        model="",
        token_callback=lambda _t: None,
    )
    sent = c._session.post.call_args
    assert "/chat/completions" in sent.args[0]
    payload = sent.kwargs["json"]
    assert payload["model"] == "deepseek-flash"
    assert payload["stream"] is True
    msgs = payload["messages"]
    assert msgs[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "u"},
    ]  # the default model reads images — no OCR text
    assert msgs[2]["tool_calls"] is not None and msgs[2]["content"] == ""
    assert msgs[3]["tool_call_id"] == "c1" and msgs[3]["name"] == "f"
    assert payload["tools"][0]["function"]["name"] == "f"


def test_chat_with_tools_empty_tools_omits_keys() -> None:
    # Empty tools must omit "tools"/"tool_choice" entirely — OpenAI-compatible
    # backends (vLLM, gateways, Anthropic-compat shims) 400 on an empty array.
    # Mirrors GoogleClient's `if gemini_tools:` payload precedent.
    c = _client(_resp(sse=[_tc_chunk({"content": "ok"}, "stop")]))
    c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    payload = c._session.post.call_args.kwargs["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_chat_with_tools_thinking_and_errors() -> None:
    c = _client(_resp(sse=[_tc_chunk({"content": "ok"}, "stop")]))
    c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        thinking_mode=True,
        reasoning_effort="high",
        token_callback=lambda _t: None,
    )
    payload = c._session.post.call_args.kwargs["json"]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"

    for status, exc in [
        (401, LLMAuthenticationError),
        (402, LLMQuotaExceededError),
        (429, LLMRateLimitError),
        (403, LLMAuthenticationError),
        (500, LLMServerUnavailableError),
        (400, LLMAPIError),
    ]:
        c2 = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c2.chat_with_tools(
                [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
            )


def test_chat_with_tools_full_stream() -> None:
    sse = [
        _tc_chunk({"content": "Hel"}),
        _tc_chunk(
            {"tool_calls": [{"index": 0, "id": "tc1", "function": {"name": "get_weather", "arguments": '{"city":'}}]}
        ),
        _tc_chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "seoul"}'}}]}),
        _tc_chunk({"reasoning_content": "cot"}, "tool_calls"),
        (
            "data: "
            + json.dumps(
                {"choices": [{"delta": {}}], "usage": {"total_tokens": 8, "prompt_tokens": 3, "completion_tokens": 5}}
            )
            + "\n\n"
        ).encode(),
    ]
    c = _client(_resp(sse=sse))
    got: list[str] = []
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        token_callback=got.append,
    )
    assert r.content == "Hel" and got == ["Hel"]
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "get_weather"
    assert r.tool_calls[0].args == {"city": "seoul"}
    assert r.tool_calls[0].call_id == "tc1"
    assert r.finish_reason == "tool_calls" and r.is_final is False
    assert r.raw_response["choices"][0]["message"]["reasoning_content"] == "cot"
    assert r.tokens_used == 8 and r.cache_read_input_tokens is None


def test_chat_with_tools_content_truncation_detection() -> None:
    # unclosed braces in content → finish_reason rewritten to "truncated"
    c = _client(_resp(sse=[_tc_chunk({"content": '{"a": 1'}, "stop")]))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.finish_reason == "truncated"

    # unclosed brackets
    c2 = _client(_resp(sse=[_tc_chunk({"content": "[1, 2"}, "stop")]))
    r2 = c2.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r2.finish_reason == "truncated"

    # balanced content stays "stop"
    c3 = _client(_resp(sse=[_tc_chunk({"content": '{"a": 1}'}, "stop")]))
    r3 = c3.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r3.finish_reason == "stop"


def test_chat_with_tools_tool_args_truncation_detection() -> None:
    sse = [
        _tc_chunk(
            {"tool_calls": [{"index": 0, "id": "t", "function": {"name": "f", "arguments": '{"a":'}}]}, "tool_calls"
        ),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.finish_reason == "truncated"
    assert r.tool_calls == []  # malformed calls cleared for retry
    assert r.is_final is True


def test_chat_with_tools_iteration_exceptions() -> None:
    def _boom(raise_exc):
        yield b'data: {"choices": [{"delta": {"content": "a"}}]}\n'
        raise raise_exc

    for exc, expected in [
        (requests.ConnectionError("c"), LLMConnectionError),
        (requests.Timeout("t"), LLMConnectionError),
        (requests.exceptions.ChunkedEncodingError("c"), LLMServerUnavailableError),
        (requests.RequestException("r"), LLMAPIError),
        (RuntimeError("plain"), LLMAPIError),
    ]:
        c = _client(_resp(sse=_boom(exc)))
        with pytest.raises(expected):
            c.chat_with_tools(
                [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
            )

    c = _client(_resp(sse=_boom(LLMRateLimitError("rl"))))
    with pytest.raises(LLMRateLimitError):
        c.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
        )


# ── remaining branch gaps ────────────────────────────────────────────────────


def test_chat_streaming_consumer_loop_exception_is_typed() -> None:
    """A malformed-but-JSON event (choices: 42) trips the loop body; the
    ``except Exception → raise_sse_iteration_failure`` tail must convert it."""
    c = _client(_resp(sse=[b'data: {"choices": 42}\n']))
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)


def test_chat_with_tools_reasoner_and_thinking_disabled() -> None:
    c = _client(_resp(sse=[_tc_chunk({"content": "ok"}, "stop")]))
    c.chat_with_tools(
        [LLMMessage(role="assistant", content="prev"), LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-reasoner",
        thinking_mode=False,
        token_callback=lambda _t: None,
    )
    payload = c._session.post.call_args.kwargs["json"]
    # assistant msg gets empty reasoning_content (reasoner requires it)
    assert payload["messages"][0]["reasoning_content"] == ""
    assert payload["thinking"] == {"type": "disabled"}


def test_chat_with_tools_usage_only_chunk() -> None:
    sse = [
        ("data: " + json.dumps({"choices": [], "usage": {"total_tokens": 3}}) + "\n\n").encode(),
        _tc_chunk({"content": "ok"}, "stop"),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.content == "ok"
    assert r.tokens_used == 3


def test_chat_with_tools_reasoning_callback() -> None:
    sse = [_tc_chunk({"reasoning_content": "cot"}, "stop")]
    c = _client(_resp(sse=sse))
    got: list[str] = []
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        reasoning_callback=got.append,
    )
    assert got == ["cot"]
    assert r.raw_response["choices"][0]["message"]["reasoning_content"] == "cot"


def test_chat_with_tools_empty_args_skipped_in_truncation_check() -> None:
    """A tool call with empty arguments must be skipped by the truncation
    detector (not flagged), leaving finish_reason intact."""
    sse = [
        _tc_chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "t1", "function": {"name": "f", "arguments": ""}},
                    {"index": 1, "id": "t2", "function": {"name": "g", "arguments": '{"ok": 1}'}},
                ]
            },
            "tool_calls",
        ),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.finish_reason == "tool_calls"
    assert len(r.tool_calls) == 2
    assert r.tool_calls[1].args == {"ok": 1}


def test_chat_with_tools_consumer_loop_exception_is_typed() -> None:
    c = _client(_resp(sse=[b'data: {"choices": 42}\n']))
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        c.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
        )


# ── None-null usage robustness ────────────────────────────────────────────────
# OpenAI-compatible streams legitimately carry "usage": null in intermediate
# chunks (until the final include_usage chunk) and usage dicts with explicit
# "completion_tokens_details": null for non-reasoning models. All consumers
# must be None-safe (None-null token field bug class — see 4db12e26/bbb7c463).


def _raw_event(obj) -> bytes:
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def test_chat_with_tools_null_usage_details_no_crash() -> None:
    """usage["completion_tokens_details"] = explicit null must not raise
    (nested .get(K, {}) default only covers key *absence*, not null value)."""
    sse = [
        _tc_chunk({"content": "ok"}, None),
        _tc_chunk(
            {},
            "stop",
            usage={
                "total_tokens": 7,
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "completion_tokens_details": None,
            },
        ),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.content == "ok"
    assert r.tokens_used == 7 and r.prompt_tokens == 3 and r.completion_tokens == 4


def test_chat_with_tools_null_usage_chunks_no_crash() -> None:
    """Intermediate chunks with explicit "usage": null (both choices-present
    and choices-empty shapes) must not clobber a later valid usage object."""
    sse = [
        _tc_chunk({"content": "ok"}, None),  # usage key absent
        _raw_event({"choices": [{"delta": {}}], "usage": None}),
        _raw_event({"choices": [], "usage": None}),
        _tc_chunk(
            {},
            "stop",
            usage={"total_tokens": 9, "prompt_tokens": 4, "completion_tokens": 5},
        ),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.tokens_used == 9 and r.prompt_tokens == 4 and r.completion_tokens == 5


def test_chat_with_tools_stream_ends_after_null_usage() -> None:
    """Aborted stream whose last usage signal was null → usage falls back to
    the initial {} → token fields are None, never an AttributeError."""
    c = _client(
        _resp(
            sse=[
                _tc_chunk({"content": "partial"}, None),
                _raw_event({"choices": [{"delta": {}}], "usage": None}),
                _tc_chunk({}, "stop"),
            ]
        )
    )
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="deepseek-v4", token_callback=lambda _t: None
    )
    assert r.content == "partial"
    assert r.tokens_used is None


def test_chat_streaming_null_usage_chunks_no_crash() -> None:
    """_chat_streaming parity: explicit "usage": null intermediates and null
    completion_tokens_details must not raise either."""
    sse = [
        _raw_event({"choices": [{"delta": {"content": "Hi"}}], "usage": None}),
        _raw_event({"choices": [], "usage": None}),
        _raw_event(
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "total_tokens": 6,
                    "prompt_tokens": 2,
                    "completion_tokens": 4,
                    "completion_tokens_details": None,
                },
            }
        ),
    ]
    c = _client(_resp(sse=sse))
    r = c._chat_streaming("u", {}, {"model": "m"}, "m", token_callback=lambda _c: None)
    assert r.content == "Hi"
    assert r.tokens_used == 6


# ── chat_with_tools streaming gate (P2) ──────────────────────────────────────


def test_chat_with_tools_no_callbacks_uses_non_streaming() -> None:
    """Calls without token/reasoning callbacks must skip SSE entirely: no
    "stream"/"stream_options" payload keys, requests stream=False, and the
    single-shot JSON body is parsed directly (OllamaClient gate parity)."""
    json_data = {
        "choices": [
            {
                "message": {
                    "content": "done",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"total_tokens": 12, "prompt_tokens": 8, "completion_tokens": 4},
    }
    c = _client(_resp(json_data=json_data))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[{"name": "f", "description": "d", "parameters": {"type": "object"}}],
        model="deepseek-v4",
    )
    sent = c._session.post.call_args
    assert "stream" not in sent.kwargs["json"]
    assert "stream_options" not in sent.kwargs["json"]
    assert sent.kwargs["stream"] is False
    c._session.post.return_value.iter_bytes.assert_not_called()
    assert r.content == "done"
    assert r.tool_calls[0].name == "f"
    assert r.tool_calls[0].args == {"x": 1}
    assert r.finish_reason == "tool_calls"
    assert r.is_final is False
    assert r.tokens_used == 12 and r.prompt_tokens == 8 and r.completion_tokens == 4
    assert r.raw_response["streamed"] is False


def test_chat_with_tools_token_callback_still_streams() -> None:
    """token_callback present → SSE path is preserved end-to-end (gate must
    not regress the streaming contract)."""
    seen: list[str] = []
    c = _client(
        _resp(
            sse=[
                _tc_chunk({"content": "he"}, None),
                _tc_chunk(
                    {"content": "y"}, "stop", usage={"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1}
                ),
            ]
        )
    )
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        token_callback=seen.append,
    )
    assert "".join(seen) == "hey"
    assert r.content == "hey"
    payload = c._session.post.call_args.kwargs["json"]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert c._session.post.call_args.kwargs["stream"] is True
    assert r.raw_response["streamed"] is True


def test_chat_with_tools_reasoning_callback_alone_streams() -> None:
    """reasoning_callback alone (no token_callback) is also a live stream
    consumer on DeepSeek — it must keep the SSE path."""
    seen: list[str] = []
    c = _client(
        _resp(
            sse=[
                _raw_event({"choices": [{"delta": {"reasoning_content": "think"}}]}),
                _tc_chunk({"content": "ans"}, "stop"),
            ]
        )
    )
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        reasoning_callback=seen.append,
    )
    assert seen == ["think"]
    assert r.content == "ans"
    assert c._session.post.call_args.kwargs["json"]["stream"] is True


# ── P4: explicit-null objects in OpenAI-compat tool_call deltas ──────────────


def test_chat_with_tools_null_function_in_tool_call_delta() -> None:
    # Gateways / OpenAI-compat shims may emit "function": null on id-only
    # tool_call fragments (deepseek keeps a locally-constructed function dict,
    # so later deltas still merge). .get("function", {}) only defaults on key
    # ABSENCE — an explicit null crashed the accumulation loop.
    sse = [
        _tc_chunk({"tool_calls": [{"index": 0, "id": "tc1", "function": None}]}),
        _tc_chunk({"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]}, "tool_calls"),
    ]
    c = _client(_resp(sse=sse))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="deepseek-v4",
        token_callback=lambda _t: None,
    )
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "f"
    assert r.tool_calls[0].args == {}
    assert r.tool_calls[0].call_id == "tc1"
    assert r.finish_reason == "tool_calls"


# ── image capability axis on the native route ────────────────────────────────
# The native route folded EVERY attachment through _images_to_text(), so the one
# catalog id built to read images (deepseek-v4-flash-vision-exp) never put an
# image on the wire — and because no image part left the process, the gateway
# could not return the 400 that the strip-and-retry net turns into a correction.
# Silent in every direction: no error, no warning, nothing learned. These tests
# pin the repaired axis: one content builder shared by both entry points, the
# declaration table as the decision, and a recovery path for the unverified case.


@pytest.fixture
def learned_rejections(monkeypatch) -> set:
    """Isolate the process-wide (base_url, model) set that the recovery writes."""
    learned: set = set()
    monkeypatch.setattr(openai_client, "_IMAGE_REJECTING_MODELS", learned)
    return learned


def _route_key(model: str) -> tuple[str, str]:
    return (openai_client._norm_base(DeepSeekClient.DEFAULT_BASE_URL), openai_client._bare_model_name(model))


def _image_uri(ocr_text: str = "OCR-BODY") -> list[dict[str, str]]:
    return [{"data": "AAAA", "media_type": "image/png", "ocr_text": ocr_text}]


def test_chat_declared_vision_model_sends_image_parts(monkeypatch) -> None:
    """The vision variant must get image_url parts. _images_to_text is
    booby-trapped so a regression fails loudly instead of quietly succeeding."""

    def _boom(_imgs):
        raise AssertionError("declared-vision model must not be routed through OCR")

    monkeypatch.setattr(providers_module, "_images_to_text", _boom)
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash-vision-exp")
    content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert content[1] == {"type": "text", "text": "u"}


def test_chat_with_tools_declared_vision_model_sends_image_parts(monkeypatch) -> None:
    """Parity with chat(): the tools path historically dropped attachments
    entirely, so both entry points must build the same content for one input."""
    monkeypatch.setattr(providers_module, "_images_to_text", lambda _imgs: pytest.fail("unexpected OCR"))
    c = _client(_resp(sse=[_tc_chunk({"content": "ok"}, "stop")]))
    c.chat_with_tools(
        [LLMMessage(role="user", content="u", images=_image_uri())],
        tools=[],
        model="deepseek-v4-flash-vision-exp",
        token_callback=lambda _t: None,
    )
    content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_text_only_model_still_folds_images_into_text() -> None:
    """The other half of the axis: a declared text-only id keeps its OCR fold
    (its gateway 400s on image_url parts even for a 1x1 PNG)."""
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash")
    content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert content == "[Image 1 — OCR Extracted Text:\nOCR-BODY\n]\nu"


@pytest.mark.parametrize("model_id", KNOWN_MODELS["deepseek"])
def test_route_agrees_with_the_declared_vision_axis(model_id: str) -> None:
    """Catalog-driven contract: whatever the pickers can offer must be
    classified by the native route exactly as the declaration table does, so a
    newly cataloged vision id is covered without touching this test.

    Asked per ROUTE, not per id: the image axis is declared per route
    (model_catalog.ROUTE_VISION), and this client always posts to
    ``DEFAULT_BASE_URL``. Comparing against the route-agnostic answer instead —
    as this test did until 2026-09-11 — passes only while every route agrees, and
    then breaks for the wrong reason the moment one disagrees (V4.1 Flash reads
    images both at the vendor and behind the gateway; the same id is text-only on
    neither route today, but ``deepseek-v4-pro`` will be re-routed to it at the
    vendor on 2026-09-14).
    """
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model=model_id)
    content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert isinstance(content, list) is vision_capable(model_id, DeepSeekClient.DEFAULT_BASE_URL)


def test_400_with_images_retries_stripped_and_learns_route(learned_rejections) -> None:
    """Native image acceptance for a versioned id is not fully measured, so a
    400 must degrade to OCR text instead of being fatal."""
    bad = _resp(status=400, text="unsupported content type 'image_url'")
    c = _client(_resp(json_data=_ok_json()))
    c._session.post.side_effect = [bad, _resp(json_data=_ok_json())]
    out = c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash-vision-exp")
    assert out.content == "hi"
    assert c._session.post.call_count == 2
    assert bad.close.called  # connection released before retrying
    retry_content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    # The wire has no room for the precomputed ocr_text cache, so the strip
    # rebuilds the attachment from the data URI and re-derives its text (the LRU
    # OCR cache absorbs the recomputation in production).
    assert isinstance(retry_content, str)
    assert retry_content.startswith("[Image 1") and retry_content.endswith("\nu")
    assert _route_key("deepseek-v4-flash-vision-exp") in learned_rejections


def test_learned_route_sends_text_on_the_next_call(learned_rejections) -> None:
    """Learning is per process and per route: the next call must not pay the
    failed round trip (nor re-attach an image the route just rejected)."""
    learned_rejections.add(_route_key("deepseek-v4-flash-vision-exp"))
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash-vision-exp")
    assert c._session.post.call_count == 1
    content = c._session.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert isinstance(content, str) and "OCR-BODY" in content


def test_400_without_images_is_not_retried() -> None:
    """Only an image-carrying 400 is recoverable — a text-only id already sent
    no image part, so retrying would repeat the same bad request."""
    c = _client(_resp(status=400, text="bad tool schema"))
    with pytest.raises(LLMAPIError):
        c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash")
    assert c._session.post.call_count == 1


def test_400_image_retry_that_also_fails_does_not_learn_the_route(learned_rejections) -> None:
    """A 400 that survives the strip is not an image problem (context overflow,
    bad schema) — teaching the route would disable vision for the process."""
    c = _client()
    c._session.post.side_effect = [_resp(status=400, text="a"), _resp(status=400, text="b")]
    with pytest.raises(LLMAPIError):
        c.chat([LLMMessage(role="user", content="u", images=_image_uri())], model="deepseek-v4-flash-vision-exp")
    assert c._session.post.call_count == 2
    assert learned_rejections == set()


def test_streaming_400_with_images_retries_as_a_stream(learned_rejections) -> None:
    """The recovery belongs to the request, not to one code path: the streaming
    route must retry with stream=True and keep its callbacks."""
    c = _client()
    c._session.post.side_effect = [_resp(status=400, text="nope"), _resp(sse=[_tc_chunk({"content": "ok"}, "stop")])]
    out = c.chat(
        [LLMMessage(role="user", content="u", images=_image_uri())],
        model="deepseek-v4-flash-vision-exp",
        token_callback=lambda _t: None,
    )
    assert out.content == "ok"
    assert c._session.post.call_count == 2
    assert c._session.post.call_args.kwargs["stream"] is True
    assert isinstance(c._session.post.call_args.kwargs["json"]["messages"][0]["content"], str)


def test_chat_with_tools_400_with_images_retries_stripped(learned_rejections) -> None:
    c = _client()
    c._session.post.side_effect = [_resp(status=400, text="nope"), _resp(sse=[_tc_chunk({"content": "ok"}, "stop")])]
    out = c.chat_with_tools(
        [LLMMessage(role="user", content="u", images=_image_uri())],
        tools=[],
        model="deepseek-v4-flash-vision-exp",
        token_callback=lambda _t: None,
    )
    assert out.content == "ok"
    assert c._session.post.call_count == 2
    assert learned_rejections != set()
