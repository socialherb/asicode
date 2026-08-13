"""Context-length 400 in-turn retry must record the POST-trim estimate.

Regression: the re-trim callbacks computed the post-trim estimate but collapsed
it to a bool; the retry wrappers kept passing the PRE-trim (inflated) estimate
to _record_context_overflow on every attempt. A stale estimate disables the
estimate clamp (min(reduced, est*0.85)) — the override then only falls 25% per
400 and hits _MAX_OVERRIDE_REDUCTIONS before reaching the real window. The
callback contract is now ``Callable[[], int | None]`` — the post-trim estimate
(or None = no progress) — and the wrappers feed it back between attempts.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.client import LLMAPIError


def _context_error() -> LLMAPIError:
    return LLMAPIError(
        "upstream 400: maximum context length is 128000 tokens, but you sent 300000"
    )


def test_retry_on_rate_limit_feeds_post_trim_estimate_back():
    """A second 400 in the same turn must be recorded against the POST-trim
    estimate returned by the callback, not the stale pre-trim estimate."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.config = SimpleNamespace(cancel_event=None)
    loop.model = "gpt-4o"
    loop.llm_client = SimpleNamespace(base_url=None)
    loop._cb = lambda *a, **k: None
    loop._record_llm_call_both = lambda **k: None

    calls = {"n": 0, "estimates": []}

    def _fake_record(model, estimated_prompt_tokens=None, **kwargs):
        calls["estimates"].append(estimated_prompt_tokens)

    def _callable():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _context_error()
        return {"content": "ok", "prompt_tokens": 1, "completion_tokens": 1}

    def _trim_cb():
        return 100_000  # post-trim estimate while attempts remain

    with patch("external_llm.agent.agent_loop._record_context_overflow", _fake_record):
        loop._retry_on_rate_limit(
            _callable, _estimated_prompt_tokens=300_000, overflow_retry_cb=_trim_cb,
        )

    # Attempt 1 records the initial pre-trim estimate; attempts 2-3 record the
    # POST-trim size fed back by the callback (old code: [300k, 300k, 300k]).
    assert calls["estimates"] == [300_000, 100_000, 100_000]
    assert calls["n"] == 4


def test_retry_on_rate_limit_none_progress_reraises():
    """The None contract: no trim progress → the original 400 is re-raised."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.config = SimpleNamespace(cancel_event=None)
    loop.model = "gpt-4o"
    loop.llm_client = SimpleNamespace(base_url=None)
    loop._cb = lambda *a, **k: None
    loop._record_llm_call_both = lambda **k: None

    def _callable():
        raise _context_error()

    with (
        patch("external_llm.agent.agent_loop._record_context_overflow"),
        pytest.raises(LLMAPIError),
    ):
        loop._retry_on_rate_limit(_callable, overflow_retry_cb=lambda: None)


def test_design_call_llm_with_retry_feeds_post_trim_estimate_back():
    """Design-chat twin of the agent-side test (DESIGN_CHAT_LLM_MAX_RETRIES=2)."""
    from external_llm.agent.design_chat_loop import DesignChatLoop

    loop = DesignChatLoop.__new__(DesignChatLoop)
    loop.registry = SimpleNamespace(config=SimpleNamespace(cancel_event=None))
    loop.model = "gpt-4o"
    loop.llm_client = None

    calls = {"n": 0, "estimates": []}

    def _fake_record(model, estimated_prompt_tokens=None, **kwargs):
        calls["estimates"].append(estimated_prompt_tokens)

    def fn():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise LLMAPIError("upstream 400: context window is too small")
        return "ok"

    def _trim():
        return 50_000

    with patch("external_llm.agent.design_chat_loop._record_context_overflow", _fake_record):
        loop._call_llm_with_retry(
            fn, _estimated_prompt_tokens=200_000, overflow_retry_cb=_trim,
        )

    # Attempt 1 records the initial estimate; attempt 2 records the post-trim
    # size fed back by the callback (old code: [200k, 200k]).
    assert calls["estimates"] == [200_000, 50_000]
    assert calls["n"] == 3
