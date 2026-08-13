"""P1-(a): context-cap structural collapse must fail fast with a clear error.

When ``ctx_limit - output_reserve - tool_tokens`` falls below
``MIN_USABLE_MESSAGE_BUDGET`` (the _shared_utils structural floor), NO message
trimming can fit the call — it would 400 even with zero chat history. Before
this fix the loops:

  1. sent the guaranteed-400 request anyway,
  2. caught the context-length 400 and re-trimmed messages toward zero,
  3. never fit, burned every retry, and re-raised the raw provider error.

Now the pre-flight guards (``_apply_context_hard_cap`` / ``_llm_call_with_tools``)
raise :class:`ContextWindowCollapseError` immediately, and the in-turn re-trim
callbacks recognize the collapse and skip the doomed trim (return None so the
caller propagates the original 400 at once, without another attempt).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.design_chat_loop import (
    _apply_context_hard_cap,
    _user_facing_llm_error,
)
from external_llm.client import (
    ContextWindowCollapseError,
    LLMAPIError,
    LLMMessage,
)


# ~8k CJK chars → ~8k+ estimated tokens: far above any small-window budget.
def _big_schema() -> list:
    return [{"name": "t", "description": "가" * 8000}]


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        cancel_event=None,
        thinking_mode=None,
        stream_callback=None,
        tokens=SimpleNamespace(AGENT_TOOL_CALL=4096),
    )


# ── design-chat pre-flight guard ────────────────────────────────────────────


def test_apply_context_hard_cap_raises_on_structural_collapse():
    """4096 window + ~8k schema tokens → cap pinned to the 512 floor → the
    guard must raise before the request instead of sending a guaranteed 400."""
    msgs = [LLMMessage(role="system", content="sys"), LLMMessage(role="user", content="hi")]
    with (
        mock.patch("external_llm.agent.design_chat_loop._resolve_context_limit", return_value=4096),
        pytest.raises(ContextWindowCollapseError, match="too small"),
    ):
        _apply_context_hard_cap(msgs, model="tiny-model", tool_schemas=_big_schema())


def test_apply_context_hard_cap_passthrough_when_window_sufficient():
    """A large window must keep the pre-existing trim/no-op behaviour — the
    guard is collapse-only."""
    msgs = [LLMMessage(role="system", content="sys"), LLMMessage(role="user", content="hi")]
    with mock.patch(
        "external_llm.agent.design_chat_loop._resolve_context_limit", return_value=131072
    ):
        out = _apply_context_hard_cap(msgs, model="big-model", tool_schemas=_big_schema())
    assert out == msgs  # 131072 - 4096 reserve - 8k schemas >> floor → untouched


# ── agent-loop pre-flight guard ─────────────────────────────────────────────


def test_agent_loop_preflight_raises_on_collapse_before_request():
    """_llm_call_with_tools with a collapsed budget must raise immediately and
    NEVER send the request (old behaviour: send → 400 → 3 doomed retries)."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.model = "tiny-model"
    loop.config = _tiny_config()
    loop._context_budget = mock.MagicMock()  # truthy → pre-flight guard runs
    loop._context_budget.fit_messages.side_effect = lambda m: m
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = _big_schema()
    loop.registry.repo_language = "python"
    loop._cb = lambda *a, **k: None
    loop._record_llm_call_both = lambda **k: None
    loop.llm_client = mock.MagicMock()

    with (
        mock.patch("external_llm.agent.agent_loop._resolve_context_limit", return_value=4096),
        pytest.raises(ContextWindowCollapseError),
    ):
        loop._llm_call_with_tools([LLMMessage(role="user", content="hi")])

    loop.llm_client.chat_with_tools.assert_not_called()


# ── in-turn retry path (pre-flight absent) must skip the doomed trim ────────


def test_agent_loop_retry_callback_skips_doomed_trim_on_collapse():
    """_context_budget=None disables the pre-flight guard, so the collapse
    surfaces as a provider 400. The re-trim callback must recognize it, skip
    the pointless trim, and return None → the ORIGINAL 400 propagates after
    exactly ONE attempt (old behaviour: trim → still 400 → retry x3 → then
    raise, while the trim kept shrinking the conversation)."""
    loop = AgentLoop.__new__(AgentLoop)
    loop.model = "tiny-model"
    loop.config = _tiny_config()
    loop._context_budget = None
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = _big_schema()
    loop.registry.repo_language = "python"
    loop._cb = lambda *a, **k: None
    loop._record_llm_call_both = lambda **k: None
    loop.llm_client = mock.MagicMock()
    loop.llm_client.chat_with_tools.side_effect = LLMAPIError(
        "upstream 400: maximum context length is 128000 tokens, but you sent 300000"
    )

    with (
        mock.patch("external_llm.agent.agent_loop._resolve_context_limit", return_value=4096),
        mock.patch("external_llm.agent.agent_loop._record_context_overflow"),
        mock.patch("external_llm.agent.agent_loop.preemptive_trim") as _trim,
        pytest.raises(LLMAPIError),
    ):
        loop._llm_call_with_tools([LLMMessage(role="user", content="hi")])

    assert loop.llm_client.chat_with_tools.call_count == 1  # no doomed retry
    _trim.assert_not_called()  # the collapse is recognized BEFORE the trim runs


# ── user-facing error mapping ───────────────────────────────────────────────


def test_user_facing_error_explains_collapse():
    """The collapse error must map to an actionable user message (not the
    generic fallback) when design chat surfaces it."""
    msg = _user_facing_llm_error(
        ContextWindowCollapseError("context window (4096) is too small")
    )
    assert "too small" in msg
    assert "message trimming cannot fix" in msg
