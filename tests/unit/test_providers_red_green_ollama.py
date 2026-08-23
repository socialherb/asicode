"""RED→GREEN: OllamaClient — chat / _num_ctx_for_model / chat_with_tools full
branch coverage via fake sessions. Runtime capability queries (/api/show) are
patched out so tests never touch the network."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

import external_llm.providers as providers_module
from external_llm.client import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from external_llm.providers import OllamaClient


@pytest.fixture(autouse=True)
def _patch_runtime_queries(monkeypatch):
    """Pin /api/show-backed lookups so no test reaches the network."""
    monkeypatch.setattr("external_llm.model_registry.ollama_vision", lambda model, base_url_hint=None: False)
    monkeypatch.setattr("external_llm.ollama_api.query_ollama_num_ctx", lambda model, base_url_hint=None: None)
    monkeypatch.setattr("external_llm.model_registry.get_ollama_num_ctx", lambda model: None)
    monkeypatch.setattr(
        "external_llm.agent._shared_utils.estimate_tokens_from_msgs",
        lambda msgs: 1000,
    )
    monkeypatch.setattr(
        "external_llm.agent._shared_utils.estimate_tokens_from_tool_schemas",
        lambda tools: 500 * len(tools),
    )


def _resp(*, status=200, json_data=None, lines=None, headers=None, text=""):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    if lines is not None:
        r.iter_lines.return_value = iter(lines)
    return r


def _client(resp=None) -> OllamaClient:
    c = OllamaClient()
    c._session = MagicMock()
    if resp is not None:
        c._session.post.return_value = resp
    return c


def _ok_json(content="hi", done_reason="stop", tool_calls=None, pt=3, ct=5):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done_reason": done_reason, "prompt_eval_count": pt, "eval_count": ct}


# ── chat ─────────────────────────────────────────────────────────────────────


def test_chat_default_model_and_options() -> None:
    c = _client(_resp(json_data=_ok_json()))
    c.chat(
        [LLMMessage(role="user", content="x")],
        model="",
        max_tokens=64,
        thinking_mode=True,
        top_p=0.9,  # extra kwargs land in options
    )
    sent = c._session.post.call_args
    assert "/api/chat" in sent.args[0]
    payload = sent.kwargs["json"]
    assert payload["model"] == "qwen2.5-coder:3b"
    assert payload["options"]["num_predict"] == 64
    assert payload["options"]["num_ctx"] == 8192
    assert payload["think"] is True
    assert payload["options"]["top_p"] == 0.9


def test_chat_status_errors() -> None:
    for status, exc in [
        (404, LLMAPIError),
        (401, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMServerUnavailableError),
        (400, LLMAPIError),
    ]:
        c = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c.chat([LLMMessage(role="user", content="x")], model="qwen2.5-coder:3b")


def test_chat_streaming_full_flow() -> None:
    lines = [
        json.dumps({"message": {"thinking": "think..."}}),
        json.dumps({"message": {"content": "Hel"}}),
        "not json at all",
        json.dumps(
            {"message": {"content": "lo"}, "done": True, "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 6}
        ),
        "",
    ]
    c = _client(_resp(lines=lines))
    got_tok: list[str] = []
    got_reason: list[str] = []
    r = c.chat(
        [LLMMessage(role="user", content="x")],
        model="qwen2.5-coder:3b",
        token_callback=got_tok.append,
        reasoning_callback=got_reason.append,
        thinking_mode=True,
    )
    assert r.content == "Hello"
    assert got_tok == ["Hel", "lo"]
    assert got_reason == ["think..."]
    assert r.finish_reason == "stop"
    assert r.prompt_tokens == 4 and r.completion_tokens == 6
    assert r.raw_response["streamed"] is True
    assert r.raw_response["thinking"] == "think..."

    # error line → LLMAPIError
    c2 = _client(_resp(lines=[json.dumps({"error": "model busy"})]))
    with pytest.raises(LLMAPIError, match="model busy"):
        c2.chat([LLMMessage(role="user", content="x")], model="qwen2.5-coder:3b", token_callback=lambda _c: None)


def test_chat_non_streaming_response() -> None:
    c = _client(_resp(json_data=_ok_json(content="plain", pt=10, ct=2)))
    r = c.chat([LLMMessage(role="user", content="x")], model="qwen2.5-coder:3b")
    assert r.content == "plain"
    assert r.finish_reason == "stop"
    assert r.tokens_used == 12
    assert r.prompt_tokens == 10 and r.completion_tokens == 2


def test_chat_requests_exceptions() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMConnectionError),
        (requests.Timeout("t"), LLMConnectionError),
        (requests.RequestException("r"), LLMAPIError),
    ]:
        c = _client()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat([LLMMessage(role="user", content="x")], model="qwen2.5-coder:3b")


# ── _num_ctx_for_model ───────────────────────────────────────────────────────


def test_num_ctx_from_api(monkeypatch) -> None:
    monkeypatch.setattr("external_llm.ollama_api.query_ollama_num_ctx", lambda model, base_url_hint=None: 16384)
    c = _client()
    assert c._num_ctx_for_model("m1") == 16384


def test_num_ctx_registry_override(monkeypatch) -> None:
    monkeypatch.setattr("external_llm.model_registry.get_ollama_num_ctx", lambda model: 4096)
    c = _client()
    assert c._num_ctx_for_model("m1") == 4096


def test_num_ctx_estimation_fallback_with_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "external_llm.agent._shared_utils.estimate_tokens_from_msgs",
        lambda msgs: 9000,  # 9000 + 2048 budget + tools(2*500=1000) > 8192
    )
    c = _client()
    val = c._num_ctx_for_model("m1", messages=[{"role": "user"}], tools=[{}, {}])
    # (9000 + 2048 + 1000) = 12048 → round up to 512 boundary → 12288
    assert val == 12288


def test_num_ctx_estimation_clamps_at_cap_and_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "external_llm.agent._shared_utils.estimate_tokens_from_msgs",
        lambda msgs: 40000,
    )
    c = _client()
    with caplog.at_level("WARNING", logger="external_llm.providers"):
        assert c._num_ctx_for_model("m1", messages=[{}]) == 32768
        # second call → debug path, no duplicate warning
        assert c._num_ctx_for_model("m1", messages=[{}]) == 32768
    warns = [r for r in caplog.records if r.levelno == 30 and "exceeds memory-safe cap" in r.getMessage()]
    assert len(warns) == 1
    providers_module._num_ctx_overshoot_warned.clear()


def test_num_ctx_estimation_failure_falls_back(monkeypatch) -> None:
    def _boom(_msgs):
        raise RuntimeError("est broken")

    monkeypatch.setattr("external_llm.agent._shared_utils.estimate_tokens_from_msgs", _boom)
    c = _client()
    assert c._num_ctx_for_model("m1", messages=[{}]) == 8192


# ── chat_with_tools ──────────────────────────────────────────────────────────


def _tc_line(content=None, tool_calls=None, done=False, done_reason=None):
    msg = {}
    if content is not None:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    ev = {"message": msg}
    if done:
        ev.update({"done": True, "done_reason": done_reason, "prompt_eval_count": 2, "eval_count": 3})
    return json.dumps(ev)


def test_chat_with_tools_default_model_and_message_kinds() -> None:
    c = _client(_resp(lines=[_tc_line(content="ok", done=True, done_reason="stop")]))
    c.chat_with_tools(
        [
            LLMMessage(role="system", content="S"),
            LLMMessage(role="user", content="u"),
            LLMMessage(role="tool", content="res"),
            LLMMessage(
                role="assistant",
                content="prev",
                tool_calls=[
                    # format A: OpenAI-style with string args
                    {"type": "function", "function": {"name": "f1", "arguments": '{"a": 1}'}},
                    # format A: dict args
                    {"type": "function", "function": {"name": "f2", "arguments": {"b": 2}}},
                    # format B: agent_loop normalized
                    {"id": "x", "name": "f3", "args": {"c": 3}},
                    # malformed: non-dict entry → skipped
                    "garbage",
                    # no function/name keys → skipped
                    {"id": "y"},
                ],
            ),
        ],
        tools=[{"name": "f1", "description": "d", "parameters": {}}],
        model="",
        max_tokens=32,
    )
    sent = c._session.post.call_args
    assert "/api/chat" in sent.args[0]
    payload = sent.kwargs["json"]
    assert payload["model"] == "qwen2.5-coder:3b"
    assert payload["options"]["num_predict"] == 32
    assert payload["options"]["num_ctx"] == 8192
    msgs = payload["messages"]
    assert msgs[2] == {"role": "tool", "content": "res"}
    tcs = msgs[3]["tool_calls"]
    assert tcs[0]["function"] == {"name": "f1", "arguments": {"a": 1}}
    assert tcs[1]["function"] == {"name": "f2", "arguments": {"b": 2}}
    assert tcs[2]["function"] == {"name": "f3", "arguments": {"c": 3}}
    assert len(tcs) == 3


def test_chat_with_tools_empty_tools_omits_key() -> None:
    # "tools": [] is rejected by some OpenAI-compatible Ollama frontends
    # (LLamaCpp/OpenAI-compat shims); omit it when no tools are requested.
    c = _client(_resp(lines=[_tc_line(content="ok", done=True, done_reason="stop")]))
    c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")
    payload = c._session.post.call_args.kwargs["json"]
    assert "tools" not in payload


def test_chat_with_tools_status_errors() -> None:
    for status, exc in [
        (404, LLMAPIError),
        (401, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMServerUnavailableError),
        (400, LLMAPIError),
    ]:
        c = _client(_resp(status=status, text="e"))
        with pytest.raises(exc):
            c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")


def test_chat_with_tools_streaming_flow() -> None:
    tc = {"function": {"name": "get_weather", "arguments": {"city": "seoul"}}}
    lines = [
        _tc_line(content="Hel"),
        _tc_line(tool_calls=[tc]),
        _tc_line(content="lo", done=True, done_reason="stop"),
    ]
    c = _client(_resp(lines=lines))
    got: list[str] = []
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="qwen2.5-coder:3b",
        token_callback=got.append,
    )
    assert r.content == "Hello" and got == ["Hel", "lo"]
    assert r.tool_calls[0].name == "get_weather"
    assert r.tool_calls[0].args == {"city": "seoul"}
    assert r.tool_calls[0].call_id == "ollama_0_get_weather"
    assert r.is_final is False
    assert r.prompt_tokens == 2 and r.completion_tokens == 3


def test_chat_with_tools_non_streaming_flow_and_arg_parsing() -> None:
    # string args JSON-parseable
    c = _client(
        _resp(
            json_data=_ok_json(
                tool_calls=[{"function": {"name": "f", "arguments": '{"k": 1}'}}],
            )
        )
    )
    r = c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")
    assert r.tool_calls[0].args == {"k": 1}
    assert r.is_final is False

    # string args NOT parseable → __raw_arguments
    c2 = _client(
        _resp(
            json_data=_ok_json(
                tool_calls=[{"function": {"name": "f", "arguments": "{broken"}}],
            )
        )
    )
    r2 = c2.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")
    assert r2.tool_calls[0].args == {"__raw_arguments": "{broken"}

    # non-dict args → __raw_arguments str
    c3 = _client(
        _resp(
            json_data=_ok_json(
                tool_calls=[{"function": {"name": "f", "arguments": 42}}],
            )
        )
    )
    r3 = c3.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")
    assert r3.tool_calls[0].args == {"__raw_arguments": "42"}

    # no tool calls → is_final True
    c4 = _client(_resp(json_data=_ok_json()))
    r4 = c4.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")
    assert r4.is_final is True and r4.tool_calls == []


def test_chat_with_tools_stream_error_line() -> None:
    c = _client(_resp(lines=[json.dumps({"error": "oom"})]))
    with pytest.raises(LLMAPIError, match="oom"):
        c.chat_with_tools(
            [LLMMessage(role="user", content="x")],
            tools=[],
            model="qwen2.5-coder:3b",
            token_callback=lambda _c: None,
        )


def test_chat_with_tools_requests_exceptions() -> None:
    for exc, expected in [
        (requests.ConnectionError("c"), LLMConnectionError),
        (requests.Timeout("t"), LLMConnectionError),
        (requests.exceptions.ChunkedEncodingError("c"), LLMServerUnavailableError),
        (requests.RequestException("r"), LLMAPIError),
    ]:
        c = _client()
        c._session.post.side_effect = exc
        with pytest.raises(expected):
            c.chat_with_tools([LLMMessage(role="user", content="x")], tools=[], model="qwen2.5-coder:3b")


# ── remaining branch gaps ────────────────────────────────────────────────────


def test_chat_with_tools_think_and_message_edge_cases() -> None:
    c = _client(_resp(lines=[_tc_line(content="ok", done=True, done_reason="stop")]))
    c.chat_with_tools(
        [
            LLMMessage(
                role="assistant",
                content="prev",
                tool_calls=[
                    # format A string args that fail JSON parsing → {}
                    {"type": "function", "function": {"name": "f1", "arguments": "{broken"}},
                    # format B with non-dict args → {}
                    {"id": "x", "name": "f2", "args": "not-a-dict"},
                ],
            ),
        ],
        tools=[],
        model="qwen2.5-coder:3b",
        thinking_mode=True,
        reasoning_effort="high",
    )
    payload = c._session.post.call_args.kwargs["json"]
    assert payload["think"] is True  # non-gpt-oss model → boolean think
    tcs = payload["messages"][0]["tool_calls"]
    assert tcs[0]["function"]["arguments"] == {}  # parse failure → {}
    assert tcs[1]["function"]["arguments"] == {}  # non-dict → {}


def test_chat_with_tools_streaming_skips_non_json_lines() -> None:
    lines = [
        "garbage line",
        "",  # blank keep-alive line
        _tc_line(content="ok", done=True, done_reason="stop"),
    ]
    c = _client(_resp(lines=lines))
    r = c.chat_with_tools(
        [LLMMessage(role="user", content="x")],
        tools=[],
        model="qwen2.5-coder:3b",
        token_callback=lambda _c: None,
    )
    assert r.content == "ok"
