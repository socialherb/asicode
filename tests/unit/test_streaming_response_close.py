"""Regression: streaming Response must be closed on a non-200 status.

The OpenAI-compatible streaming clients (OpenAIClient, AnthropicClient) used to
run their status-code checks BEFORE the try/finally that closes the response, so
a non-200 streaming response (or a failure while draining its error body via
``.text``) leaked the underlying connection. The fix folds the status checks
into the same try/finally as the iteration loop, matching the providers.py
pattern. These tests pin that contract.

Only NON-retryable statuses (401/402/generic non-200) reach the caller's status
check — _request_with_retry returns them as-is (or, for Anthropic, post() is
called directly). 429/5xx are retried inside _request_with_retry and never reach
this code path, so they are out of scope here.
"""
from __future__ import annotations

import pytest

from external_llm.anthropic_client import AnthropicClient
from external_llm.client import LLMAPIError, LLMAuthenticationError, LLMQuotaExceededError
from external_llm.openai_client import OpenAIClient


class _FakeStreamResponse:
    """Minimal stand-in for a streaming requests.Response with a tracked close."""

    def __init__(self, status_code: int, text: str = "error body"):
        self.status_code = status_code
        self.text = text
        self.headers: dict = {}
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1

    def iter_lines(self):  # not reached on a non-200 status
        return iter(())


def _patch_post(monkeypatch, client, response):
    """Make ``client._session.post`` return ``response`` for every call."""
    monkeypatch.setattr(client._session, "post", lambda *a, **k: response)


_STREAM_METHODS = ["_chat_streaming", "_chat_with_tools_streaming"]


# ── OpenAIClient ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", _STREAM_METHODS)
def test_openai_streaming_closes_response_on_401(monkeypatch, method):
    client = OpenAIClient(api_key="test")
    fake = _FakeStreamResponse(401)
    _patch_post(monkeypatch, client, fake)
    with pytest.raises(LLMAuthenticationError):
        getattr(client, method)(
            "http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None
        )
    assert fake.close_count == 1


@pytest.mark.parametrize("method", _STREAM_METHODS)
def test_openai_streaming_closes_response_on_402(monkeypatch, method):
    client = OpenAIClient(api_key="test")
    fake = _FakeStreamResponse(402)
    _patch_post(monkeypatch, client, fake)
    with pytest.raises(LLMQuotaExceededError):
        getattr(client, method)(
            "http://x/v1/chat/completions", {}, {"model": "m"}, "m", lambda c: None
        )
    assert fake.close_count == 1


# ── AnthropicClient ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", _STREAM_METHODS)
def test_anthropic_streaming_closes_response_on_401(monkeypatch, method):
    client = AnthropicClient(api_key="test")
    fake = _FakeStreamResponse(401)
    _patch_post(monkeypatch, client, fake)
    with pytest.raises(LLMAuthenticationError):
        getattr(client, method)(
            "http://x/v1/messages", {}, {"model": "m"}, "claude-3-5-sonnet", lambda c: None
        )
    assert fake.close_count == 1


@pytest.mark.parametrize("method", _STREAM_METHODS)
def test_anthropic_streaming_closes_response_on_non200(monkeypatch, method):
    client = AnthropicClient(api_key="test")
    fake = _FakeStreamResponse(418)  # generic non-200 (not 401/429/>=500)
    _patch_post(monkeypatch, client, fake)
    with pytest.raises(LLMAPIError):
        getattr(client, method)(
            "http://x/v1/messages", {}, {"model": "m"}, "claude-3-5-sonnet", lambda c: None
        )
    assert fake.close_count == 1
