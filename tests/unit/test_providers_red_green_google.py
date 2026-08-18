"""RED→GREEN: GoogleClient (Gemini) — chat / chat_with_tools /
_chat_with_tools_streaming_gemini full branch coverage via fake sessions."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from external_llm.providers import GoogleClient


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


def _client(resp) -> GoogleClient:
    c = GoogleClient(api_key="test-key")
    c._session = MagicMock()
    c._session.post.return_value = resp
    return c


def _ok_json(content="hi", parts=None, finish="STOP", candidates=None):
    if candidates is None:
        candidates = [{"content": {"parts": parts or [{"text": content}]}, "finishReason": finish}]
    return {"candidates": candidates, "usageMetadata": {"totalTokenCount": 7}}


# ── chat ─────────────────────────────────────────────────────────────────────


def test_chat_uses_default_model_when_empty() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="")
    sent = c._session.post.call_args
    assert "/models/gemini-2.5-flash:generateContent" in sent.args[0]


def test_chat_system_instruction_and_text_parts() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat(
        [LLMMessage(role="system", content="SYS"), LLMMessage(role="assistant", content="R")],
        model="gemini-2.5-flash",
    )
    payload = c._session.post.call_args.kwargs["json"]
    assert payload["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert payload["contents"][0] == {"role": "model", "parts": [{"text": "R"}]}


def test_chat_max_tokens_and_thinking_budget_negative_one() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat(
        [LLMMessage(role="user", content="x")],
        model="gemini-2.5-flash",
        max_tokens=500,
        thinking_mode=True,
    )
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert gc["maxOutputTokens"] == 500
    assert gc["thinkingConfig"] == {"thinkingBudget": -1}


def test_chat_thinking_budget_zero_for_non_pro() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash", thinking_mode=False)
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingBudget": 0}


def test_chat_thinking_empty_for_pro_false() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-pro", thinking_mode=False)
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert "thinkingConfig" not in gc


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("high", "high"), ("max", "high"), ("low", "low"), (None, "high"), ("medium", "high")],
)
def test_chat_gemini3_thinking_level(effort, expected) -> None:
    c = _client(_resp(json_data=_ok_json()))
    kwargs = {"thinking_mode": True}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    c.chat([LLMMessage(role="user", content="x")], model="gemini-3-flash", **kwargs)
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": expected}


def test_chat_gemini3_thinking_minimal_when_disabled() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="gemini-3-flash", thinking_mode=False)
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_chat_gemini3_no_thinking_mode_empty_config() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat([LLMMessage(role="user", content="x")], model="gemini-3-flash")
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert "thinkingConfig" not in gc


def test_chat_error_statuses() -> None:
    cases = [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (402, LLMQuotaExceededError),
        (503, LLMServerUnavailableError),
        (500, LLMServerUnavailableError),
        (501, LLMAPIError),  # 501 excluded from >=500 branch
        (400, LLMAPIError),
    ]
    for status, exc in cases:
        c = _client(_resp(status=status, text="err"))
        with pytest.raises(exc):
            c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash")


def test_chat_429_carries_retry_after() -> None:
    c = _client(_resp(status=429, headers={"Retry-After": "5"}))
    with pytest.raises(LLMRateLimitError) as ei:
        c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash")
    assert ei.value.retry_after == 5


def test_chat_no_candidates_returns_empty() -> None:
    c = _client(_resp(json_data={"candidates": []}))
    r = c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash")
    assert r.content == "" and r.tokens_used is None


def test_chat_extracts_usage_and_finish_reason() -> None:
    c = _client(_resp(json_data=_ok_json(content="answer", finish="MAX_TOKENS")))
    r = c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash")
    assert r.content == "answer"
    assert r.finish_reason == "length"
    assert r.tokens_used == 7


def test_chat_requests_exceptions() -> None:
    cases = [
        (requests.ConnectionError("boom"), LLMConnectionError),
        (requests.Timeout("slow"), LLMConnectionError),
        (requests.RequestException("bad"), LLMAPIError),
    ]
    for exc, expected in cases:
        c = GoogleClient(api_key="k")
        c._session = MagicMock()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat([LLMMessage(role="user", content="x")], model="gemini-2.5-flash")


# ── chat_with_tools (non-streaming) ──────────────────────────────────────────


def _tool_msg() -> LLMMessage:
    return LLMMessage(role="tool", content="result", name="get_weather", tool_call_id="t1")


def test_tools_default_model_and_message_kinds() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat_with_tools(
        [
            LLMMessage(role="system", content="SYS"),
            _tool_msg(),
            LLMMessage(role="assistant", content="", raw_content=[{"functionCall": {"name": "f"}}]),
            LLMMessage(role="user", content="plain"),
            LLMMessage(role="user", content="img", images=[{"media_type": "image/jpeg", "data": "QQ=="}]),
        ],
        tools=[{"name": "get_weather", "description": "d", "parameters": {"type": "object"}}],
        model="",
    )
    sent = c._session.post.call_args
    assert "/models/gemini-2.5-flash:generateContent" in sent.args[0]
    payload = sent.kwargs["json"]
    assert payload["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    # tool role → functionResponse
    assert payload["contents"][0]["parts"][0]["functionResponse"]["name"] == "get_weather"
    # raw_content passthrough
    assert payload["contents"][1] == {"role": "model", "parts": [{"functionCall": {"name": "f"}}]}
    # plain text
    assert payload["contents"][2] == {"role": "user", "parts": [{"text": "plain"}]}
    # images → inlineData + text
    parts = payload["contents"][3]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "image/jpeg"
    assert parts[1]["text"] == "img"
    # tools → functionDeclarations
    assert payload["tools"] == [{"functionDeclarations": [
        {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}
    ]}]


def test_tools_generation_config_and_errors() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="gemini-2.5-flash",
        max_tokens=100,
        thinking_mode=True,
    )
    gc = c._session.post.call_args.kwargs["json"]["generationConfig"]
    assert gc["maxOutputTokens"] == 100
    assert gc["thinkingConfig"] == {"thinkingBudget": -1}

    for status, exc in [
        (401, LLMAuthenticationError), (403, LLMAuthenticationError),
        (429, LLMRateLimitError), (402, LLMQuotaExceededError),
        (500, LLMServerUnavailableError), (400, LLMAPIError),
    ]:
        c2 = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c2.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash")


def test_tools_no_candidates() -> None:
    c = _client(_resp(json_data={"candidates": []}))
    r = c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash")
    assert r.tool_calls == [] and r.is_final is True


def test_tools_function_call_extraction() -> None:
    c = _client(_resp(json_data=_ok_json(parts=[
        {"text": "thinking"},
        {"functionCall": {"name": "f", "args": {"a": 1}}},
    ])))
    r = c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash")
    assert r.content == "thinking"
    assert r.is_final is False
    assert r.tool_calls[0].name == "f" and r.tool_calls[0].args == {"a": 1}
    assert r.prompt_tokens is None


def test_tools_requests_exceptions() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMConnectionError),
        (requests.Timeout("t"), LLMConnectionError),
        (requests.RequestException("r"), LLMAPIError),
    ]:
        c = GoogleClient(api_key="k")
        c._session = MagicMock()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash")


# ── _chat_with_tools_streaming_gemini ────────────────────────────────────────


def _stream_client(sse, *, status=200):
    c = GoogleClient(api_key="k")
    c._session = MagicMock()
    c._session.post.return_value = _resp(status=status, sse=sse, text="")
    return c


def test_streaming_post_connection_and_timeout() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMConnectionError),
        (requests.Timeout("t"), LLMConnectionError),
    ]:
        c = GoogleClient(api_key="k")
        c._session = MagicMock()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat_with_tools(
                [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
                token_callback=lambda _c: None,
            )


def test_streaming_status_errors() -> None:
    for status, exc in [
        (401, LLMAuthenticationError), (403, LLMAuthenticationError),
        (429, LLMRateLimitError), (402, LLMQuotaExceededError),
        (500, LLMServerUnavailableError), (400, LLMAPIError),
    ]:
        c = _stream_client([], status=status)
        with pytest.raises(exc):
            c.chat_with_tools(
                [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
                token_callback=lambda _c: None,
            )


def test_streaming_full_flow() -> None:
    chunks = [
        "data: " + __import__("json").dumps({"candidates": [
            {"content": {"parts": [{"text": "hel"}]}, "finishReason": None}
        ], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}}) + "\n\n",
        "data: " + __import__("json").dumps({"candidates": [
            {"content": {"parts": [{"text": "lo"}, {"functionCall": {"name": "f", "args": {"a": 1}}}]}}
        ]}) + "\n\n",
        "data: " + __import__("json").dumps({"candidates": [
            {"content": {"parts": []}, "finishReason": "STOP"}
        ]}) + "\n\n",
    ]
    c = _stream_client([ch.encode() for ch in chunks])
    got: list[str] = []
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
        token_callback=got.append,
    )
    assert r.content == "hello"
    assert got == ["hel", "lo"]
    assert r.tool_calls[0].name == "f"
    assert r.is_final is False
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 3 and r.completion_tokens == 2 and r.tokens_used == 5


def test_streaming_iteration_failures() -> None:
    # ChunkedEncodingError → LLMServerUnavailableError
    def _boom_chunked():
        yield b'data: {"candidates": [{"content": {"parts": [{"text": "a"}]}}]}\n'
        raise requests.exceptions.ChunkedEncodingError("cut")

    c = GoogleClient(api_key="k")
    c._session = MagicMock()
    c._session.post.return_value = _resp(sse=_boom_chunked())
    with pytest.raises(LLMServerUnavailableError):
        c.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
            token_callback=lambda _c: None,
        )

    # plain RequestException → LLMAPIError
    def _boom_req():
        yield b'data: {}\n'
        raise requests.RequestException("net")

    c2 = GoogleClient(api_key="k")
    c2._session = MagicMock()
    c2._session.post.return_value = _resp(sse=_boom_req())
    with pytest.raises(LLMAPIError):
        c2.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
            token_callback=lambda _c: None,
        )

    # typed LLMClientError passes through unchanged
    def _boom_typed():
        yield b'data: {}\n'
        raise LLMRateLimitError("rl")

    c3 = GoogleClient(api_key="k")
    c3._session = MagicMock()
    c3._session.post.return_value = _resp(sse=_boom_typed())
    with pytest.raises(LLMRateLimitError):
        c3.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
            token_callback=lambda _c: None,
        )

    # non-requests Exception → LLMAPIError via raise_sse_iteration_failure
    def _boom_plain():
        yield b'data: {}\n'
        raise RuntimeError("weird")

    c4 = GoogleClient(api_key="k")
    c4._session = MagicMock()
    c4._session.post.return_value = _resp(sse=_boom_plain())
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        c4.chat_with_tools(
            [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
            token_callback=lambda _c: None,
        )


# ── P4: Gemini's documented-nullable candidate.content ──────────────────────


def test_streaming_null_candidate_content_is_skipped() -> None:
    # candidate.content is nullable per the Gemini API (SAFETY / RECITATION
    # and other blocked finish reasons) — the parser must skip, not crash.
    chunks = [
        "data: " + __import__("json").dumps({"candidates": [{"content": None, "finishReason": "SAFETY"}]}) + "\n\n",
    ]
    c = _stream_client([ch.encode() for ch in chunks])
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")], tools=[], model="gemini-2.5-flash",
        token_callback=lambda _c: None,
    )
    assert r.content == ""
