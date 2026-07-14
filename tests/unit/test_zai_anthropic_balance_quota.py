"""ZAI's PRIMARY endpoint is the Anthropic-compatible one (ZAIAnthropicClient),
not the OpenAI-compatible failover (ZAIClient). zai reports an EXHAUSTED
ACCOUNT BALANCE as HTTP 429 with code 1113 — a permanent billing failure that
must raise the non-retryable ``LLMQuotaExceededError``, never a retryable
``LLMRateLimitError``.

Commit d0945e58 wired this classification into the OpenAI failover path only;
the primary Anthropic path (all four 429 sites) kept misclassifying balance
exhaustion as a transient rate limit — burning the retry budget and surfacing a
misleading "slow down" / "re-enter your key" instead of "out of credit".

These tests pin the parity fix: the canonical ``is_balance_quota_signal``
helper (single source in client.py, shared by both endpoints) classifies
correctly, and every 429 site in ZAIAnthropicClient raises the right error.
"""
from __future__ import annotations

import pytest

from external_llm.anthropic_client import AnthropicClient, ZAIAnthropicClient
from external_llm.client import (
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    is_balance_quota_signal,
)


# ── canonical helper (client.py) ────────────────────────────────────────────

def test_balance_signal_true_for_zai_code_1113():
    assert is_balance_quota_signal(1113, "") is True


def test_balance_signal_true_for_billing_phrase_without_code():
    assert is_balance_quota_signal(None, "please recharge your account") is True


def test_balance_signal_false_for_genuine_rate_limit_codes():
    # GLM 1305 (server overload) and 1302 (rate limit) are transient.
    assert is_balance_quota_signal(1305, "overloaded") is False
    assert is_balance_quota_signal(1302, "rate limited") is False


def test_balance_signal_false_for_none_and_empty():
    assert is_balance_quota_signal(None, "") is False
    assert is_balance_quota_signal(None, "slow down") is False


# ── end-to-end: ZAIAnthropicClient.chat (primary endpoint) ──────────────────

class _Resp:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        import json as _json
        return _json.loads(self.text)


_BALANCE_BODY = (
    '{"error":{"code":"1113","message":"Insufficient balance or no resource '
    'package. Please recharge."}}'
)
_PHRASE_BODY = '{"error":{"message":"please recharge your account to continue"}}'
_OVERLOAD_BODY = '{"error":{"code":"1305","message":"server overloaded"}}'
_RATELIMIT_BODY = '{"error":{"code":"1302","message":"rate limited"}}'


def _zai_client(monkeypatch, resp):
    client = ZAIAnthropicClient(api_key="test")
    # The Anthropic-compatible client posts directly via _session.post (no
    # internal retry loop — retry lives in upper layers), so a single 429
    # response triggers the classifier immediately.
    monkeypatch.setattr(client._session, "post", lambda *a, **k: resp)
    return client


def test_zai_anthropic_chat_balance_429_raises_quota(monkeypatch):
    """A 429 carrying the zai balance signal must surface as a non-retryable
    quota error, not a rate-limit error."""
    client = _zai_client(monkeypatch, _Resp(429, text=_BALANCE_BODY))
    with pytest.raises(LLMQuotaExceededError):
        client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")


def test_zai_anthropic_chat_balance_phrase_raises_quota(monkeypatch):
    """A 429 whose body carries an unambiguous billing phrase (no numeric code)
    is still a balance/quota error."""
    client = _zai_client(monkeypatch, _Resp(429, text=_PHRASE_BODY))
    with pytest.raises(LLMQuotaExceededError):
        client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")


def test_zai_anthropic_chat_genuine_rate_limit_still_rate_limit(monkeypatch):
    """GLM 1305 (overload) is a genuine transient rate limit — must NOT be
    misclassified as balance/quota."""
    client = _zai_client(monkeypatch, _Resp(429, text=_OVERLOAD_BODY))
    with pytest.raises(LLMRateLimitError):
        client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")


def test_zai_anthropic_chat_rate_limit_error_code_preserved(monkeypatch):
    """The genuine-rate-limit path still attaches the parsed GLM error code."""
    client = _zai_client(monkeypatch, _Resp(429, text=_RATELIMIT_BODY))
    with pytest.raises(LLMRateLimitError) as ei:
        client.chat([LLMMessage(role="user", content="hi")], model="glm-5.2")
    assert ei.value.error_code == 1302


# ── chat_with_tools exercises a different 429 site ──────────────────────────

def test_zai_anthropic_chat_with_tools_balance_429_raises_quota(monkeypatch):
    client = _zai_client(monkeypatch, _Resp(429, text=_BALANCE_BODY))
    with pytest.raises(LLMQuotaExceededError):
        client.chat_with_tools(
            [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2"
        )


# ── plain Anthropic (Claude) regression: genuine 429 unaffected ─────────────

def test_plain_anthropic_genuine_429_still_rate_limit(monkeypatch):
    """Claude never emits balance code 1113, so a Claude 429 stays a rate
    limit — the balance check is a no-op for non-zai providers."""
    client = AnthropicClient(api_key="test")
    monkeypatch.setattr(
        client._session, "post", lambda *a, **k: _Resp(429, text=_OVERLOAD_BODY)
    )
    with pytest.raises(LLMRateLimitError):
        client.chat([LLMMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022")
