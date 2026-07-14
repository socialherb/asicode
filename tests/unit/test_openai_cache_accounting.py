"""OpenAI-protocol prompt-cache accounting.

Covers three layers:
1. ``_extract_cached_tokens`` — pulls ``usage.prompt_tokens_details.cached_tokens``.
2. Parent ``OpenAIClient`` populates ``resp.cache_read_input_tokens`` on every
   response path (chat / chat_with_tools / streaming).
3. ``ZAIClient._normalize_cache_accounting`` — z.ai is served over BOTH
   protocols; the OpenAI path (subset) is re-normalized at the boundary to the
   separate-accounting shape ``_CACHE_TOKENS_SEPARATE`` expects for "zai".

The normalization is what keeps the ⚡ cache-hit display and cost math correct
when the design-chat failover flips from the Anthropic facade to the OpenAI one.
"""
from __future__ import annotations

import json

import pytest

import external_llm.openai_client as oc
from external_llm.agent._shared_utils import (
    _get_cached_input_rate,
    _get_rates,
    cache_hit_pct,
    estimate_cache_adjusted_cost,
)
from external_llm.client import LLMMessage, ToolCallResponse
from external_llm.openai_client import (
    OpenAIClient,
    OpenRouterClient,
    ZAIClient,
    _extract_cached_tokens,
)

# ── 1. _extract_cached_tokens ────────────────────────────────────────────────


def _usage(prompt=10000, cached=None, completion=50):
    u = {"prompt_tokens": prompt, "completion_tokens": completion,
         "total_tokens": prompt + completion}
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return u


def test_extract_cached_tokens_present():
    assert _extract_cached_tokens(_usage(cached=8500)) == 8500


def test_extract_cached_tokens_absent():
    assert _extract_cached_tokens(_usage(cached=None)) is None


def test_extract_cached_tokens_non_dict_usage():
    assert _extract_cached_tokens(None) is None
    assert _extract_cached_tokens("nope") is None


def test_extract_cached_tokens_zero_is_not_none():
    # 0 cached is a real value (full miss reported), distinct from absent.
    assert _extract_cached_tokens(_usage(cached=0)) == 0


# ── 2. Parent OpenAIClient populates cache field on all paths ────────────────


class _OK:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    @property
    def text(self):
        return json.dumps(self._data)


class _Stream:
    status_code = 200
    headers = {}

    def __init__(self, sse_lines):
        self._lines = sse_lines

    def iter_lines(self):
        for ln in self._lines:
            yield ln.encode()

    def close(self):
        pass

    @property
    def text(self):
        return ""


def _payload(content="hi", finish="stop", usage=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    body = {
        "id": "x", "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage or _usage(),
    }
    return body


def _parent(monkeypatch, resp):
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    client = OpenAIClient(api_key="test")
    monkeypatch.setattr(client._session, "post", lambda *a, **k: resp)
    return client


def test_chat_nonstream_populates_cache_field(monkeypatch):
    client = _parent(monkeypatch, _OK(_payload(usage=_usage(prompt=10000, cached=8500))))
    resp = client.chat([LLMMessage(role="user", content="hi")], model="gpt-4")
    assert resp.cache_read_input_tokens == 8500
    assert resp.prompt_tokens == 10000
    assert resp.completion_tokens == 50


def test_chat_with_tools_nonstream_populates_cache_field(monkeypatch):
    client = _parent(monkeypatch, _OK(_payload(usage=_usage(prompt=10000, cached=8500))))
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="gpt-4",
    )
    assert resp.cache_read_input_tokens == 8500
    # prompt_tokens stays as the OpenAI subset total here (parent = subset).
    assert resp.prompt_tokens == 10000


def test_streaming_populates_cache_field(monkeypatch):
    # Final usage chunk carries prompt_tokens_details.cached_tokens.
    usage_chunk = {
        "usage": {"prompt_tokens": 10000, "completion_tokens": 50,
                  "prompt_tokens_details": {"cached_tokens": 8500}},
    }
    sse = ["data: " + json.dumps(usage_chunk), "data: [DONE]"]
    client = _parent(monkeypatch, _Stream(sse))
    captured = []
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="gpt-4",
        token_callback=captured.append,
    )
    assert resp.cache_read_input_tokens == 8500
    assert resp.prompt_tokens == 10000


# ── 3. ZAIClient._normalize_cache_accounting ─────────────────────────────────


def _resp(prompt=10000, cached=8500, completion=50):
    return ToolCallResponse(
        content="x", model="glm-5.2", provider="zai",
        tokens_used=prompt + completion, finish_reason="stop",
        raw_response={}, tool_calls=[], is_final=True,
        prompt_tokens=prompt, completion_tokens=completion,
        cache_read_input_tokens=cached,
    )


def test_normalize_splits_subset_into_separate():
    resp = ZAIClient._normalize_cache_accounting(_resp(prompt=10000, cached=8500))
    assert resp.cache_read_input_tokens == 8500          # unchanged
    assert resp.prompt_tokens == 10000 - 8500            # uncached-only


def test_normalize_noop_without_cache():
    resp = ZAIClient._normalize_cache_accounting(_resp(prompt=10000, cached=None))
    assert resp.prompt_tokens == 10000
    assert resp.cache_read_input_tokens is None


def test_normalize_guard_when_cached_exceeds_prompt():
    # Malformed: cached > prompt — guard keeps prompt non-negative (no change).
    resp = ZAIClient._normalize_cache_accounting(_resp(prompt=100, cached=5000))
    assert resp.prompt_tokens == 100  # not driven below zero


# ── 4. ZAIClient full path (parent mock + normalization) ─────────────────────


def test_zai_chat_with_tools_normalizes_to_separate(monkeypatch):
    client = ZAIClient(api_key="test")
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        client._session, "post",
        lambda *a, **k: _OK(_payload(usage=_usage(prompt=10000, cached=8500))),
    )
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[], model="glm-5.2",
    )
    # Parent filled cache_read from usage; ZAIClient then normalized prompt.
    assert resp.cache_read_input_tokens == 8500
    assert resp.prompt_tokens == 10000 - 8500


# ── 5. Correctness: normalized zai == subset formula ─────────────────────────


def test_normalize_keeps_hit_pct_meaningful():
    # OpenAI reports prompt=10000 with 8500 cached → true hit ratio is 85%.
    # Without normalization (bug): cache_hit_pct("zai", 10000, 8500) → 45.9%
    # because the separate formula adds cached on top of an already-inclusive prompt.
    assert cache_hit_pct("zai", 10000, 8500) == pytest.approx(45.945, rel=1e-3)
    # After normalization (prompt=uncached=1500) the denominator is 1500+8500=10000:
    assert cache_hit_pct("zai", 1500, 8500) == pytest.approx(85.0)


def test_normalize_cost_matches_subset_formula():
    prompt, cached, completion = 10000, 8500, 50
    uncached = prompt - cached
    in_rate, out_rate = _get_rates("zai", "glm-5.2")
    cached_rate = _get_cached_input_rate("zai", in_rate, "glm-5.2", base_url="https://api.z.ai/paas/v4/chat")

    # Separate-accounting cost on the NORMALIZED (uncached) prompt.
    separate = estimate_cache_adjusted_cost(
        "zai", uncached, completion, cache_read_tok=cached, model="glm-5.2",
        base_url="https://api.z.ai/paas/v4/chat",
    )
    # Equivalent subset re-price done by hand on the raw inclusive prompt.
    subset_manual = (
        prompt * in_rate + completion * out_rate - cached * (in_rate - cached_rate)
    ) / 1_000_000
    assert separate == pytest.approx(subset_manual, rel=1e-9)


# ── 6. OpenRouter regression: subset stays subset (no normalization) ─────────


def test_openrouter_does_not_normalize_prompt(monkeypatch):
    # OpenRouter is subset-accounting (provider "openrouter"); its prompt_tokens
    # must NOT be reduced by cached_tokens — only ZAIClient renormalizes.
    client = OpenRouterClient(api_key="sk-test")
    monkeypatch.setattr(oc.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(
        client._session, "post",
        lambda *a, **k: _OK(_payload(usage=_usage(prompt=10000, cached=8500))),
    )
    resp = client.chat_with_tools(
        [LLMMessage(role="user", content="hi")], tools=[],
        model="deepseek/deepseek-v4-flash",
    )
    assert resp.cache_read_input_tokens == 8500
    assert resp.prompt_tokens == 10000  # unchanged — subset preserved
