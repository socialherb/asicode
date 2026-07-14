"""openai_client must attach the server Retry-After hint to the
LLMRateLimitError it raises, so retry layers above it (agent_loop's
_retry_on_rate_limit, design_chat's _call_llm_with_retry) can honor it instead
of falling back to fixed backoff.
"""
from __future__ import annotations

import logging

import pytest

import external_llm.openai_client as oc
from external_llm.client import (
    RETRY_AFTER_MAX_WAIT,
    LLMMessage,
    LLMRateLimitError,
    LLMServerUnavailableError,
)
from external_llm.openai_client import OpenAIClient, _short_error_reason


class _Resp429:
    status_code = 429
    text = '{"error":{"code":"1305","message":"overloaded"}}'

    def __init__(self, headers):
        self.headers = headers


def _client(monkeypatch, headers):
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)  # no real backoff
    client = OpenAIClient(api_key="test")
    monkeypatch.setattr(client._session, "post", lambda *a, **k: _Resp429(headers))
    return client


def test_retry_after_attached_when_header_present(monkeypatch):
    client = _client(monkeypatch, {"Retry-After": "5"})
    with pytest.raises(LLMRateLimitError) as ei:
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")
    assert ei.value.retry_after == 5


def test_retry_after_none_when_header_absent(monkeypatch):
    client = _client(monkeypatch, {})
    with pytest.raises(LLMRateLimitError) as ei:
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")
    assert ei.value.retry_after is None


def test_retry_after_clamped_to_max(monkeypatch):
    client = _client(monkeypatch, {"Retry-After": "99999"})
    with pytest.raises(LLMRateLimitError) as ei:
        client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")
    assert ei.value.retry_after == RETRY_AFTER_MAX_WAIT


# ── Log/message cleanliness (_short_error_reason) ────────────────────────────


def test_short_error_reason_extracts_code_and_message():
    body = ('{"error":{"code":"1305","message":"The service may be temporarily '
            'overloaded, please try again later"}}')
    assert _short_error_reason(body) == (
        "1305 The service may be temporarily overloaded, please try again later"
    )


def test_short_error_reason_flattens_and_truncates():
    # newlines collapsed to single spaces
    assert _short_error_reason("line one\n  line two") == "line one line two"
    # plain (non-JSON) body passes through, trimmed
    assert _short_error_reason("502 Bad Gateway") == "502 Bad Gateway"
    # over-long is truncated with an ellipsis
    out = _short_error_reason("x" * 500, limit=40)
    assert len(out) == 40 and out.endswith("…")
    # empty stays empty
    assert _short_error_reason("") == ""


def test_transient_retry_logged_at_info_off_the_prompt_and_raise_is_clean(monkeypatch, caplog):
    client = _client(monkeypatch, {})  # no Retry-After -> exercises the plain retry path
    with caplog.at_level(logging.INFO):
        with pytest.raises(LLMRateLimitError) as ei:
            client.chat_with_tools([LLMMessage(role="user", content="hi")], tools=[], model="gpt-4")

    rate_records = [r for r in caplog.records if "rate limited (429)" in r.getMessage()]
    assert rate_records, "expected at least one retry log record"
    # Transient auto-recovering retries log at INFO, not WARNING, so asi's
    # _TerminalInfoFilter (which suppresses INFO from asicode.*/external_llm.*)
    # keeps them off the interactive prompt while the file handler still records.
    assert all(r.levelno == logging.INFO for r in rate_records), \
        [(r.levelname, r.getMessage()) for r in rate_records]
    # And they're terse — no raw JSON envelope.
    assert all("{" not in r.getMessage() for r in rate_records)

    # The raised exception (the only thing that surfaces to the user on give-up)
    # still carries the condensed human-readable reason.
    msg = str(ei.value)
    assert "1305 overloaded" in msg
    assert "{" not in msg


def test_parse_error_code_extracts_glm_error_codes():
    """_parse_error_code must extract 1302/1305 from GLM JSON error body."""

    class _FakeResponse:
        text: str = ""
        headers: dict = {}

        def __init__(self, text: str) -> None:
            self.text = text

    # 1305 — server congestion (string code in JSON)
    r = _FakeResponse('{"error":{"code":"1305","message":"overloaded"}}')
    assert oc._parse_error_code(r) == 1305

    # 1302 — rate limit (integer code in JSON)
    r = _FakeResponse('{"error":{"code":1302,"message":"rate limited"}}')
    assert oc._parse_error_code(r) == 1302

    # No error dict → None
    r = _FakeResponse('{"id":"abc"}')
    assert oc._parse_error_code(r) is None

    # Invalid JSON → None
    r = _FakeResponse("not json")
    assert oc._parse_error_code(r) is None

    # Empty body → None
    r = _FakeResponse("")
    assert oc._parse_error_code(r) is None



# ── _request_with_retry backoff semantics (#1 bug fix) ─────────────────────
# Regression: ``_skipped_backoff`` used to be a STICKY flag — once a 429 with
# Retry-After set it True, exponential backoff was disabled for ALL subsequent
# attempts. So a short Retry-After (e.g. 1s) followed by a 5xx would retry the
# 5xx with ZERO backoff, hammering an already-strained server. The one-shot
# ``_skip_next_backoff`` must suppress backoff for the immediately-following
# attempt only, then resume normal exponential backoff.


class _Resp:
    def __init__(self, status_code, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self):
        import json as _json
        return _json.loads(self.text)


def _tracking_client(monkeypatch, response_sequence):
    """Build an OpenAIClient whose session.post returns ``response_sequence``
    in order, and record every ``time.sleep`` duration (backoff + Retry-After)."""
    sleeps: list[float] = []
    monkeypatch.setattr(oc.time, "sleep", lambda d, *a, **k: sleeps.append(d))
    client = OpenAIClient(api_key="test")
    call_state = {"i": 0}

    def _post(*a, **k):
        idx = call_state["i"]
        call_state["i"] += 1
        return response_sequence[idx]

    monkeypatch.setattr(client._session, "post", _post)
    return client, sleeps


def test_backoff_resumes_after_retry_after_then_5xx(monkeypatch):
    """429(Retry-After=1) -> 500 must apply exponential backoff on the 3rd
    attempt. The legacy sticky ``_skipped_backoff`` would skip it entirely
    (sleeps == [1.0] only); the one-shot fix yields [retry_after, backoff]."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(429, {"Retry-After": "1"}),   # attempt 0 -> retry_after sleep
        _Resp(500),                          # attempt 1 -> backoff skipped (just slept)
        _Resp(500),                          # attempt 2 -> MUST backoff, then give up
    ])
    with pytest.raises(LLMServerUnavailableError):
        client._request_with_retry("http://x", {}, {}, tag="chat")
    # sleeps[0] == 1.0 (Retry-After honored), sleeps[1] == exponential backoff
    # resumed on the 3rd attempt. The bug produced sleeps == [1.0] only.
    assert len(sleeps) >= 2, f"backoff was permanently disabled: {sleeps}"
    assert sleeps[0] == 1.0  # Retry-After
    assert sleeps[1] > 1.0   # resumed exponential backoff (4s base + jitter)


def test_retry_after_skips_only_the_next_backoff(monkeypatch):
    """429(Retry-After=1) -> 200: the second attempt runs immediately (no
    backoff), because we just slept for Retry-After. sleep list == [1.0]."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(429, {"Retry-After": "1"}),
        _Resp(200, text='{"choices":[]}'),
    ])
    resp = client._request_with_retry("http://x", {}, {}, tag="chat")
    assert resp.status_code == 200
    assert sleeps == [1.0], f"expected only the Retry-After sleep: {sleeps}"


def test_plain_5xx_uses_exponential_backoff(monkeypatch):
    """5xx with no preceding Retry-After uses normal exponential backoff on
    every retry (no skip semantics involved)."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(503),
        _Resp(503),
        _Resp(503),
    ])
    with pytest.raises(LLMServerUnavailableError):
        client._request_with_retry("http://x", {}, {}, tag="chat")
    # Two backoff sleeps (retry 1: ~4s, retry 2: ~8s), no Retry-After.
    assert len(sleeps) == 2, f"expected 2 backoff sleeps: {sleeps}"
    # First backoff base is 2**(1+1)=4s, second is 2**(2+1)=8s (both + jitter).
    assert 4.0 <= sleeps[0] <= 6.0
    assert 8.0 <= sleeps[1] <= 12.0


def test_helper_returns_response_on_success(monkeypatch):
    """A 200 response is returned immediately with no retries/sleeps."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(200, text='{"choices":[]}'),
    ])
    resp = client._request_with_retry("http://x", {}, {}, tag="chat")
    assert resp.status_code == 200
    assert sleeps == []


def test_connection_error_retries_with_backoff(monkeypatch):
    """ConnectionError retries with exponential backoff, then raises
    LLMServerUnavailableError after exhausting retries."""
    import requests as _requests

    client, sleeps = _tracking_client(monkeypatch, [])
    # Override post to always raise ConnectionError.
    monkeypatch.setattr(
        client._session, "post",
        lambda *a, **k: (_ for _ in ()).throw(_requests.ConnectionError("boom")),
    )
    with pytest.raises(LLMServerUnavailableError):
        client._request_with_retry("http://x", {}, {}, tag="chat")
    # 2 backoff sleeps between 3 attempts.
    assert len(sleeps) == 2, f"expected 2 backoff sleeps: {sleeps}"


def test_chat_uses_unified_retry_helper(monkeypatch):
    """chat() must route through _request_with_retry (dedup regression):
    a 429-then-success sequence yields exactly one Retry-After sleep."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(429, {"Retry-After": "2"}),
        _Resp(200, text='{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}'),
    ])
    resp = client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")
    assert resp.content == "hi"
    assert sleeps == [2.0]


# ── zai/GLM balance-quota masquerading as 429 ──────────────────────────────
# zai reports an EXHAUSTED ACCOUNT BALANCE as HTTP 429 with code 1113
# ("Insufficient balance or no resource package. Please recharge."). That is NOT
# a transient rate limit — retrying cannot recharge the account, and surfacing
# it as rate-limit/auth misleads the user (pointless retries, wrong message,
# or a "re-enter your API key" prompt for a key that is actually valid).
# The shared 429 handler must detect the balance signal and raise a
# NON-retryable LLMQuotaExceededError instead.

from external_llm.client import LLMQuotaExceededError  # noqa: E402

_ZAI_BALANCE_BODY = (
    '{"error":{"code":"1113","message":"Insufficient balance or no resource '
    'package. Please recharge."}}'
)


def test_is_balance_quota_response_detects_zai_code_1113():
    r = _Resp(429, text=_ZAI_BALANCE_BODY)
    assert oc._is_balance_quota_response(r) is True


def test_is_balance_quota_response_detects_phrase_without_code():
    # A 429 whose body carries an unambiguous billing phrase (no numeric code)
    # is still a balance/quota error.
    r = _Resp(429, text='{"error":{"message":"please recharge your account"}}')
    assert oc._is_balance_quota_response(r) is True


def test_is_balance_quota_response_false_for_genuine_rate_limit():
    # GLM code 1305 (server overload) and 1302 (rate limit) are genuine
    # transient rate limits — must NOT be misclassified as balance/quota.
    r = _Resp(429, text='{"error":{"code":"1305","message":"overloaded"}}')
    assert oc._is_balance_quota_response(r) is False


def test_balance_429_raises_quota_error_not_rate_limit(monkeypatch):
    """A 429 carrying a balance signal must raise LLMQuotaExceededError
    immediately, WITHOUT consuming the retry budget (no sleeps)."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(429, text=_ZAI_BALANCE_BODY),
    ])
    with pytest.raises(LLMQuotaExceededError):
        client._request_with_retry("http://x", {}, {}, tag="chat")
    # No backoff/Retry-After sleep — raised on the first attempt.
    assert sleeps == [], f"balance error must not retry (slept {sleeps})"


def test_balance_429_via_chat_raises_quota_error(monkeypatch):
    """End-to-end: chat() path surfaces the balance failure as a quota error,
    not a rate-limit error — so design_chat's error_type stays non-auth and the
    REPL never offers the misleading "re-enter your API key" prompt."""
    client, sleeps = _tracking_client(monkeypatch, [
        _Resp(429, text=_ZAI_BALANCE_BODY),
    ])
    with pytest.raises(LLMQuotaExceededError):
        client.chat([LLMMessage(role="user", content="hi")], model="glm-4.6")
    assert sleeps == []
