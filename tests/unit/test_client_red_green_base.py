"""RED→GREEN: external_llm/client.py — SSE framing, retry parsing, factory, base LLMClient.

Covers the remaining untested branches: effective_content reasoning fallback,
HTTP-date Retry-After parsing, SSE line/stream edge cases, guard_sse_iteration
failure conversion, the default chat_with_tools implementation, close(), and
every create_llm_client factory branch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import external_llm.client as cc
from external_llm.client import (
    LLMAPIError,
    LLMClient,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    ToolCallResponse,
    _parse_sse_line,
    create_llm_client,
    effective_content,
    guard_sse_iteration,
    iter_sse_data_events,
    parse_retry_after,
    raise_sse_iteration_failure,
)

# ── effective_content ────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, content, raw_response=None):
        self.content = content
        self.raw_response = raw_response


def test_effective_content_falls_back_to_reasoning_content():
    resp = _Resp("", {"choices": [{"message": {"reasoning_content": "  deep think  "}}]})
    assert effective_content(resp) == "deep think"


def test_effective_content_strips_whitespace_only_content():
    resp = _Resp("   ", {"choices": [{"message": {"reasoning_content": "think"}}]})
    assert effective_content(resp) == "think"


def test_effective_content_ignores_non_dict_raw_response():
    resp = _Resp("", "not-a-dict")
    assert effective_content(resp) == ""


# ── parse_retry_after ────────────────────────────────────────────────────────


def test_parse_retry_after_none_headers():
    assert parse_retry_after(None) is None


def test_parse_retry_after_http_date_in_past_returns_none():
    past = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after({"Retry-After": past}) is None


def test_parse_retry_after_http_date_future_clamped():
    future = (datetime.now(timezone.utc) + timedelta(seconds=10)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    wait = parse_retry_after({"Retry-After": future})
    assert wait is not None and 1 <= wait <= 60


# ── _parse_sse_line ──────────────────────────────────────────────────────────


def test_parse_sse_line_strips_carriage_return():
    assert _parse_sse_line(b'data: {"a": 1}\r') == {"a": 1}


def test_parse_sse_line_blank():
    assert _parse_sse_line(b"") is None


def test_parse_sse_line_undecodable_bytes():
    assert _parse_sse_line(b"\xff\xfe\x00\x01") is None


def test_parse_sse_line_non_data_frame():
    assert _parse_sse_line(b"event: ping") is None


def test_parse_sse_line_malformed_json():
    assert _parse_sse_line(b"data: {bad") is None


def test_parse_sse_line_done_sentinel():
    assert _parse_sse_line(b"data: [DONE]") is None


# ── iter_sse_data_events ─────────────────────────────────────────────────────


class _BytesResp:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def iter_bytes(self):
        yield from self._chunks


def test_iter_sse_skips_empty_chunks():
    events = list(iter_sse_data_events(_BytesResp([b"", b'data: {"a": 1}\n', b"", b"\n"])))
    assert events == [{"a": 1}]


def test_iter_sse_aborts_on_oversized_line():
    big = b"data: " + b"x" * (4 * 1024 * 1024 + 1) + b"\n"
    events = list(iter_sse_data_events(_BytesResp([big])))
    assert events == []


def test_iter_sse_aborts_on_oversized_remainder():
    big = b"data: " + b"x" * (4 * 1024 * 1024 + 1)  # no newline
    events = list(iter_sse_data_events(_BytesResp([big])))
    assert events == []


def test_iter_sse_parses_tail_without_trailing_newline():
    events = list(iter_sse_data_events(_BytesResp([b'data: {"a": 1}'])))
    assert events == [{"a": 1}]


def test_iter_sse_multiline_chunk_split():
    events = list(iter_sse_data_events(_BytesResp([b'data: {"a": 1}\ndata: {"b": 2}\n'])))
    assert events == [{"a": 1}, {"b": 2}]


# ── guard_sse_iteration ──────────────────────────────────────────────────────


def test_guard_sse_passes_typed_errors_through():
    def _gen():
        raise LLMRateLimitError("rl")
        yield  # pragma: no cover

    with pytest.raises(LLMRateLimitError):
        list(guard_sse_iteration(_gen()))


def test_guard_sse_converts_unexpected_exception():
    def _gen():
        raise RuntimeError("boom")
        yield  # pragma: no cover

    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        list(guard_sse_iteration(_gen()))


def test_raise_sse_iteration_failure_wraps():
    with pytest.raises(LLMAPIError, match="SSE stream iteration failed"):
        raise_sse_iteration_failure(ValueError("bad"))


# ── LLMClient base ───────────────────────────────────────────────────────────


class _MiniLLM(LLMClient):
    def chat(self, messages, model="", temperature=0.0, max_tokens=None, **kwargs):
        return LLMResponse(content="hi", model=model, provider="mini")

    def get_provider_name(self):
        return "mini"


def test_default_chat_with_tools_wraps_chat():
    client = _MiniLLM(api_key="k")
    out = client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="m")
    assert isinstance(out, ToolCallResponse)
    assert out.content == "hi"
    assert out.tool_calls == []
    assert out.is_final is True


def test_close_closes_session():
    client = _MiniLLM(api_key="k")
    fake_session = MagicMock()
    client._session = fake_session
    client.close()
    fake_session.close.assert_called_once()


def test_close_no_session_is_noop():
    client = _MiniLLM(api_key="k")
    client._session = None
    client.close()  # must not raise


# ── create_llm_client factory ────────────────────────────────────────────────


def test_create_ollama_uses_extended_timeout_by_default():
    client = create_llm_client("ollama", "k")
    assert client.timeout == cc.OLLAMA_LLM_TIMEOUT


def test_create_ollama_explicit_timeout_respected():
    client = create_llm_client("ollama", "k", timeout=123)
    assert client.timeout == 123


def test_create_anthropic():
    from external_llm.anthropic_client import AnthropicClient

    assert isinstance(create_llm_client("anthropic", "k"), AnthropicClient)


def test_create_google():
    from external_llm.providers import GoogleClient

    assert isinstance(create_llm_client("google", "k"), GoogleClient)


def test_create_deepseek():
    from external_llm.providers import DeepSeekClient

    assert isinstance(create_llm_client("deepseek", "k"), DeepSeekClient)


def test_create_ollama_client():
    from external_llm.providers import OllamaClient

    assert isinstance(create_llm_client("ollama", "k"), OllamaClient)


def test_create_opencode_defaults_base_url():
    from external_llm.openai_client import OpenAIClient

    client = create_llm_client("opencode", "k")
    assert isinstance(client, OpenAIClient)
    assert client.base_url == "https://opencode.ai/zen/go/v1"


def test_create_opencode_explicit_base_url_kept():
    client = create_llm_client("opencode", "k", base_url="http://custom:1/v1")
    assert client.base_url == "http://custom:1/v1"


def test_create_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client("nope", "k")


# ── remaining branch coverage ────────────────────────────────────────────────


def test_effective_content_returns_non_empty_content():
    assert effective_content(_Resp("hello")) == "hello"


def test_parse_retry_after_unparseable_value_returns_none():
    assert parse_retry_after({"Retry-After": "not-a-date"}) is None
