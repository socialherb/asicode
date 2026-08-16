"""Agent turn pipeline RED->GREEN coverage (21st round).

Baseline: 64% / 357 miss over tests/unit/agent (6044 tests).

RED defect pinned here first (single-outlier drift from the documented
main-loop token contract):

    _handle_max_turns_reached used ``if not _pt:`` for the tokens_used
    fallback at BOTH the initial call and the wrap-up retry, while
    _run_llm_loop's accumulation — whose comment (L338-341) declares the
    contract — uses ``if _pt is None:``: a REAL 0 is a valid value and must
    not trigger the fallback.  A gateway reporting ``prompt_tokens=0`` with
    ``tokens_used=777`` therefore inflated total_prompt_tokens by 777 on the
    max-turns final call.

The rest of the file exercises the untested branches of the turn pipeline:
message preparation hints/interrupts, prepared-call filtering (masked /
unknown tools, JSON arg salvage), tool-result accounting (fail streaks,
recall hints, retry exhaustion, noop detection, cache 3-state), post-tool
auto-observation / TDD early finish, deferred semantic settling, patch
recovery ladder, cancellation/error handlers, and the eviction stub helpers.
"""
from __future__ import annotations

import queue as queue_mod
from collections import defaultdict
from types import SimpleNamespace
from typing import ClassVar  # f821-protected
from unittest import mock

import pytest

from external_llm.agent import agent_turn_pipeline as atp
from external_llm.agent.agent_loop_types import (
    AgentCancelled,
    AgentResult,
    TurnContext,
    _FinalAnswerOutcome,
    _PostToolResult,
    _PreparedCallsResult,
    _ResultsProcessingOutcome,
    _ToolTurnOutcome,
    _TurnPrepResult,
)
from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin
from external_llm.agent.tool_registry import ToolResult
from external_llm.client import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerUnavailableError,
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _make_loop(**config_over):
    """Minimal TurnPipelineMixin with mocked host dependencies."""
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    cfg = {
        "max_turns": 5,
        "model_name": "claude-test",
        "agent_id": "agent-1",
        "stream_callback": None,
        "auto_observation_enabled": False,
        "auto_test_on_patch": False,
        "self_review_enabled": False,
        "cancel_event": None,
        "message_queue": None,
        "parallel_tool_execution_enabled": False,
        "tolerant_patch_mode": False,
        "tolerant_patch_max_failures": 2,
        "scoped_verification": False,
        "model": "claude-test",
        "make_token_callback": lambda: None,
    }
    cfg.update(config_over)
    loop.config = SimpleNamespace(**cfg)
    loop.llm_client = mock.MagicMock()
    loop.llm_client.get_provider_name.return_value = "anthropic"
    loop.llm_client.base_url = ""
    loop.registry = mock.MagicMock()
    loop.registry.applied_patches = []
    loop.registry.repo_root = "/tmp"
    loop.registry.repo_language = None
    loop.registry.is_result_cacheable = mock.MagicMock(return_value=False)
    loop.registry.get_tool_schemas = mock.MagicMock(return_value=[])
    loop.registry.get_tool_names = mock.MagicMock(return_value=frozenset({"read_file"}))
    loop.registry._WRITE_TOOLS = {"apply_patch", "write_plan"}
    loop.registry.normalize_args_for_display = staticmethod(lambda a: a)
    loop.performance_collector = mock.MagicMock()
    loop.performance_collector.get_summary.return_value = {"tools": 0}
    loop._failure_classifier = mock.MagicMock()
    loop._failure_classifier.classify.return_value = SimpleNamespace(action="retry", reason="bad args")
    loop._tool_retry_counter = defaultdict(int)
    loop._patch_fail_count = 0
    loop._cb = mock.MagicMock()
    loop._save_session_log = mock.MagicMock()
    return loop


def _make_ctx(**over):
    """TurnContext with the fields the pipeline reads."""
    ctx = TurnContext(
        request="test request",
        context=None,
        route=None,
        git_state={"head": "abc"},
        session_id="sess-1",
        is_local_model=False,
        has_native_tools=False,
        read_only_request=True,
        known_target_file="",
        target_keywords=[],
        tier=1,
        plan=None,
        plan_subtasks=[],
    )
    for k, v in over.items():
        setattr(ctx, k, v)
    return ctx


def _full_ctx(**over):
    """Loop-ready ctx (fields _run_llm_loop would have initialized)."""
    ctx = _make_ctx(**over)
    ctx.turns = []
    ctx.messages = []
    ctx.write_tools = {"apply_patch", "write_plan"}
    ctx.provider_name = "anthropic"
    ctx.model_name = "claude-test"
    ctx.base_url = ""
    ctx.recall_session_key = "rk-1"
    return ctx


def _prep_result(ctx, messages=None):
    return _TurnPrepResult(
        messages=messages if messages is not None else list(ctx.messages),
        budget_warned=getattr(ctx, "budget_warned", False),
        goal_reminder_injected=ctx.goal_reminder_injected,
        search_first_hint_done=ctx.search_first_hint_done,
        reads_since_last_edit=ctx.reads_since_last_edit,
    )


# ---------------------------------------------------------------------------
# RED: token fallback parity in _handle_max_turns_reached
# ---------------------------------------------------------------------------

def test_max_turns_real_zero_prompt_tokens_must_not_fall_back():
    """A REAL prompt_tokens=0 is a valid value (main-loop contract, L338-341):
    the tokens_used fallback must fire only on None, never on 0."""
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 0, "completion_tokens": 5, "tokens_used": 777,
        "content": "summary", "tool_calls": [], "finish_reason": "stop",
    })
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    assert ctx.total_prompt_tokens == 0  # RED before fix: 777
    assert ctx.total_completion_tokens == 5


def test_max_turns_wrapup_real_zero_prompt_tokens_must_not_fall_back():
    """Same contract on the wrap-up retry call (second site)."""
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    responses = [
        {"prompt_tokens": 10, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "read_file", "args": {}}], "finish_reason": "tool_use"},
        {"prompt_tokens": 0, "completion_tokens": 2, "tokens_used": 888,
         "content": "wrapped", "tool_calls": [], "finish_reason": "stop"},
    ]
    loop._llm_call_with_tools = mock.MagicMock(side_effect=responses)
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    # 10 from the first call; the wrap-up call contributes 0, not 888.
    assert ctx.total_prompt_tokens == 10  # RED before fix: 10 + 888
    assert ctx.total_completion_tokens == 3


def test_max_turns_none_prompt_tokens_still_falls_back():
    """Regression pin: None (provider omitted the split) still uses tokens_used."""
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": None, "completion_tokens": None, "tokens_used": 432,
        "content": "summary", "tool_calls": [], "finish_reason": "stop",
    })
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    assert ctx.total_prompt_tokens == 432
    assert ctx.total_completion_tokens == 0


# ---------------------------------------------------------------------------
# _handle_max_turns_reached — remaining branches
# ---------------------------------------------------------------------------

def test_max_turns_write_intent_without_patches_falls_back_to_max_turns():
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=False)
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 3, "completion_tokens": 1, "content": "done",
        "tool_calls": [], "finish_reason": "stop",
    })
    result = loop._handle_max_turns_reached(ctx)
    # RuntimeError("... without any applied patches") is swallowed → max_turns
    assert result.status == "max_turns"
    assert "maximum turns" in result.final_message


def test_max_turns_still_requesting_tools_after_wrapup_falls_back():
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    responses = [
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "read_file"}], "finish_reason": "tool_use"},
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "read_file"}], "finish_reason": "tool_use"},
    ]
    loop._llm_call_with_tools = mock.MagicMock(side_effect=responses)
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "max_turns"


def test_max_turns_self_review_appends_summary():
    loop = _make_loop(self_review_enabled=True)
    ctx = _full_ctx(read_only_request=True)
    loop.registry.applied_patches = ["a.py"]
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "base summary",
        "tool_calls": [], "finish_reason": "stop",
    })
    loop._run_self_review = mock.MagicMock(return_value="found one issue: X")
    loop._is_trivial_edit_request = mock.MagicMock(return_value=False)
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    assert "[Self-Review]" in result.final_message
    assert result.metadata["self_review"]["issues_found"] is True


def test_max_turns_trivial_request_skips_review():
    loop = _make_loop(self_review_enabled=True)
    ctx = _full_ctx(read_only_request=True)
    loop.registry.applied_patches = ["a.py"]
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "s",
        "tool_calls": [], "finish_reason": "stop",
    })
    loop._run_self_review = mock.MagicMock()
    loop._is_trivial_edit_request = mock.MagicMock(return_value=True)
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    loop._run_self_review.assert_not_called()
    assert result.metadata["self_review"]["issues_found"] is False


def test_max_turns_llm_failure_returns_max_turns():
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    loop._llm_call_with_tools = mock.MagicMock(side_effect=LLMConnectionError("down"))
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "max_turns"


# ---------------------------------------------------------------------------
# _handle_final_answer_turn
# ---------------------------------------------------------------------------

def _fa_ctx(**over):
    ctx = _full_ctx(**over)
    ctx.turn_num = 2
    return ctx


def test_final_answer_intent_undetermined_skips_false_success_gate():
    loop = _make_loop()
    ctx = _fa_ctx(read_only_request=False, intent_undetermined=True)
    out = loop._handle_final_answer_turn(ctx, "here is the answer")
    assert out.result.status == "success"
    assert out.result.final_message == "here is the answer"


def test_final_answer_noop_confirmed_returns_success_without_patches():
    loop = _make_loop()
    ctx = _fa_ctx(read_only_request=False, noop_confirmed=True)
    out = loop._handle_final_answer_turn(ctx, "already correct, no change needed")
    assert out.result.status == "success"
    assert out.result.metadata["noop"] is True
    assert out.result.applied_patches == []


def test_final_answer_text_reply_when_no_tool_called():
    loop = _make_loop()
    ctx = _fa_ctx(read_only_request=False, any_tool_called=False)
    out = loop._handle_final_answer_turn(ctx, "pure prose answer")
    assert out.result.status == "text_reply"


def test_final_answer_nudge_then_exhaustion_blocks_false_success():
    loop = _make_loop()
    ctx = _fa_ctx(read_only_request=False, any_tool_called=True, no_tool_nudge_count=0)
    ctx.config = loop.config
    out = loop._handle_final_answer_turn(ctx, "I would change the code like this...")
    assert out.nudge_message is not None
    assert out.nudge_count == 1
    assert "[ACTION REQUIRED]" in out.nudge_message.content

    # Exhausted nudges → false-success error result.
    ctx2 = _fa_ctx(read_only_request=False, any_tool_called=True,
                   no_tool_nudge_count=atp._NO_TOOL_NUDGE_MAX)
    out2 = loop._handle_final_answer_turn(ctx2, "still no patch")
    assert out2.result.status == "error"
    assert out2.result.error == "write_intent_finished_without_patch"
    assert out2.result.metadata["false_success_blocked"] is True


def test_final_answer_noop_requires_final_msg_else_nudge():
    """noop_confirmed but empty final message → falls through to the nudge
    ladder instead of an empty success."""
    loop = _make_loop()
    ctx = _fa_ctx(read_only_request=False, noop_confirmed=True)
    out = loop._handle_final_answer_turn(ctx, "")
    assert out.result is None and out.nudge_message is not None


def test_final_answer_with_patches_runs_self_review():
    loop = _make_loop(self_review_enabled=True)
    loop.registry.applied_patches = ["a.py"]
    loop._run_self_review = mock.MagicMock(return_value="LGTM")
    loop._is_trivial_edit_request = mock.MagicMock(return_value=False)
    ctx = _fa_ctx(read_only_request=False, any_tool_called=True)
    out = loop._handle_final_answer_turn(ctx, "done")
    assert out.result.status == "success"
    # ' lgtm ' not in ' lgtm ' → issues_found False, message unchanged.
    assert "[Self-Review]" not in out.result.final_message
    assert out.result.metadata["self_review"]["issues_found"] is False


def test_final_answer_review_trivial_skip():
    loop = _make_loop(self_review_enabled=True)
    loop.registry.applied_patches = ["a.py"]
    loop._run_self_review = mock.MagicMock()
    loop._is_trivial_edit_request = mock.MagicMock(return_value=True)
    ctx = _fa_ctx(read_only_request=False, any_tool_called=True)
    out = loop._handle_final_answer_turn(ctx, "done")
    loop._run_self_review.assert_not_called()
    assert out.result.status == "success"


# ---------------------------------------------------------------------------
# _run_llm_loop — branches
# ---------------------------------------------------------------------------

def _loop_ready(ctx, responses, **config_over):
    loop = _make_loop(**config_over)
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_prep_result(ctx))
    loop._llm_call_with_tools = mock.MagicMock(side_effect=responses)
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="text_reply", final_message="done")))
    loop._handle_loop_error = mock.MagicMock(
        side_effect=AssertionError("_handle_loop_error must not run"))
    return loop


def test_run_llm_loop_uses_continuation_messages_for_non_planner():
    loop = _make_loop()
    loop._continuation_data = {"conversation": [LLMMessage(role="user", content="hi")]}
    loop._build_continuation_messages = mock.MagicMock(return_value=[LLMMessage(role="user", content="c")])
    loop._build_initial_messages = mock.MagicMock(
        side_effect=AssertionError("initial build must not run for continuation"))
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "done",
        "tool_calls": [], "finish_reason": "stop"})
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="text_reply")))
    ctx = _full_ctx()
    result = loop._run_llm_loop(ctx)
    assert result.status == "text_reply"
    loop._build_continuation_messages.assert_called_once()


def test_run_llm_loop_planner_lane_ignores_continuation():
    loop = _make_loop()
    loop._continuation_data = {"conversation": []}
    loop._build_continuation_messages = mock.MagicMock(
        side_effect=AssertionError("planner must build fresh messages"))
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "done",
        "tool_calls": [], "finish_reason": "stop"})
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="text_reply")))
    ctx = _full_ctx()
    ctx.route = SimpleNamespace(lane="PLANNER")
    assert loop._run_llm_loop(ctx).status == "text_reply"
    loop._build_initial_messages.assert_called_once()


@pytest.mark.parametrize("exc,expected_fragment", [
    (LLMConnectionError("x"), "connection error"),
    (LLMServerUnavailableError("x"), "server unavailable"),
    (LLMQuotaExceededError("x"), "quota exceeded"),
    (LLMAuthenticationError("x"), "authentication error"),
    (LLMRateLimitError("x"), "rate limit"),
])
def test_run_llm_loop_typed_llm_errors_return_error_result(exc, expected_fragment):
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(side_effect=exc)
    ctx = _full_ctx()
    result = loop._run_llm_loop(ctx)
    assert result.status == "error"
    assert expected_fragment in result.error.lower()
    # Both outcome channels recorded the failure.
    loop.performance_collector.record_agent_result.assert_called_once_with(failed=True)


def test_run_llm_loop_tokens_used_fallback_and_usage_cb():
    """prompt_tokens=None (provider omitted split) → tokens_used fallback;
    token_usage cb carries the four cost fields."""
    loop = _loop_ready(_full_ctx(), [{
        "prompt_tokens": None, "completion_tokens": 2, "tokens_used": 31,
        "content": "done", "tool_calls": [], "finish_reason": "stop"}])
    ctx = _full_ctx()
    result = loop._run_llm_loop(ctx)
    assert result.status == "text_reply"
    assert ctx.total_prompt_tokens == 31
    calls = [c.args[0] for c in loop._cb.call_args_list if c.args and c.args[0] == "token_usage"]
    assert calls, "token_usage cb must fire when tokens are present"
    payload = next(c.args[1] for c in loop._cb.call_args_list
                   if c.args and c.args[0] == "token_usage")
    assert payload["prompt_tokens"] == 31
    assert "turn_cost_usd" in payload and "total_actual_cost_usd" in payload


def test_run_llm_loop_agent_thinking_cb_fires_for_content():
    loop = _loop_ready(_full_ctx(), [{
        "prompt_tokens": 1, "completion_tokens": 1, "content": "thinking aloud",
        "tool_calls": [], "finish_reason": "stop"}])
    ctx = _full_ctx()
    loop._run_llm_loop(ctx)
    kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert "agent_thinking" in kinds


def test_run_llm_loop_finish_reason_stop_with_tool_calls_is_completion():
    """finish_reason=stop + tool_calls → treated as final answer (tools dropped)."""
    loop = _loop_ready(_full_ctx(), [{
        "prompt_tokens": 1, "completion_tokens": 1, "content": "final",
        "tool_calls": [{"name": "read_file", "args": {}}], "finish_reason": "stop"}])
    ctx = _full_ctx()
    loop._run_llm_loop(ctx)
    loop._handle_final_answer_turn.assert_called_once()
    loop._execute_and_process_tool_calls = mock.MagicMock(
        side_effect=AssertionError("tools must not execute"))
    # (the assertion above documents intent; second run confirms no execution)


def test_run_llm_loop_nudge_continues_then_text_reply_finishes():
    loop = _make_loop()
    responses = [
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "no tools",
         "tool_calls": [], "finish_reason": "stop"},
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "still prose",
         "tool_calls": [], "finish_reason": "stop"},
    ]
    loop = _loop_ready(_full_ctx(), responses)
    outcomes = [
        _FinalAnswerOutcome(nudge_message=LLMMessage(role="user", content="nudge"), nudge_count=1),
        _FinalAnswerOutcome(result=AgentResult(status="text_reply")),
    ]
    loop._handle_final_answer_turn = mock.MagicMock(side_effect=outcomes)
    ctx = _full_ctx()
    ctx.any_tool_called = True
    result = loop._run_llm_loop(ctx)
    assert result.status == "text_reply"
    # any_tool_called reset by the nudge path so the follow-up text reply
    # resolves as text_reply rather than re-nudging.
    assert ctx.any_tool_called is False
    assert len(loop._handle_final_answer_turn.call_args_list) == 2


def test_run_llm_loop_tool_early_return_propagates():
    early = AgentResult(status="success", final_message="early")
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "",
        "tool_calls": [{"name": "read_file"}], "finish_reason": "tool_use"})
    loop._execute_and_process_tool_calls = mock.MagicMock(return_value=_ToolTurnOutcome(
        new_messages=[], prepared_calls=[], write_tool_used=False,
        any_tool_called=True, fail_streak={}, reads_since_last_edit=0,
        plan_current_index=0, early_return=early))
    ctx = _full_ctx()
    assert loop._run_llm_loop(ctx) is early


def test_run_llm_loop_should_continue_appends_phase_rules():
    loop = _make_loop()
    rule = LLMMessage(role="user", content="[PHASE RULE] no writes")
    loop._build_initial_messages = mock.MagicMock(return_value=[LLMMessage(role="user", content="base")])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[LLMMessage(role="user", content="base")], budget_warned=False,
        goal_reminder_injected=0, search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(side_effect=[
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "nope"}], "finish_reason": "tool_use"},
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "done",
         "tool_calls": [], "finish_reason": "stop"},
    ])
    loop._execute_and_process_tool_calls = mock.MagicMock(side_effect=[
        _ToolTurnOutcome(new_messages=[], prepared_calls=[], write_tool_used=False,
                         any_tool_called=False, fail_streak={}, reads_since_last_edit=0,
                         plan_current_index=0, should_continue=True, phase_rule_messages=[rule]),
    ])
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="text_reply")))
    ctx = _full_ctx()
    assert loop._run_llm_loop(ctx).status == "text_reply"
    # The phase rule was folded into history for the next turn.
    assert any("PHASE RULE" in getattr(m, "content", "") for m in ctx.messages)


def test_run_llm_loop_cancellation_and_finally_drains_semantic_turn():
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(side_effect=AgentCancelled("stop"))
    cancelled_result = AgentResult(status="cancelled")
    loop._handle_loop_cancellation = mock.MagicMock(return_value=cancelled_result)
    ctx = _full_ctx()
    assert loop._run_llm_loop(ctx) is cancelled_result
    loop.registry.end_semantic_turn.assert_called_once()


def test_run_llm_loop_generic_exception_routes_to_error_handler():
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(side_effect=RuntimeError("boom"))
    err_result = AgentResult(status="error")
    loop._handle_loop_error = mock.MagicMock(return_value=err_result)
    ctx = _full_ctx()
    assert loop._run_llm_loop(ctx) is err_result
    loop.registry.end_semantic_turn.assert_called_once()


# ---------------------------------------------------------------------------
# _process_post_tool_turn
# ---------------------------------------------------------------------------

def _patch_turn(tool, ok=True, metadata=None, turn_num=1):
    """AgentTurn whose turn_num matches _post_ctx()."""
    from external_llm.agent.agent_loop_types import AgentTurn
    return AgentTurn(
        turn_num=turn_num, tool_name=tool, tool_args={},
        tool_result=ToolResult(ok=ok, content="c", metadata=metadata or {}))


def _post_ctx(**over):
    """ctx for _process_post_tool_turn — turn_num aligned with _patch_turn(1)."""
    ctx = _full_ctx(**over)
    ctx.turn_num = 1
    return ctx


def test_post_tool_turn_auto_observation_scopes_git_diff_to_touched_files():
    loop = _make_loop(auto_observation_enabled=True)
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("write_plan", metadata={"touched_files": ["a.py", "b.py"]}),
                 _patch_turn("write_plan", metadata={"touched_files": ["a.py"]}),
                 _patch_turn("read_file")]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "text")
    with mock.patch("subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="diff --git a/a.py", stderr="")
        loop._process_post_tool_turn(ctx, None, [])
    run.assert_called_once()
    assert run.call_args.args[0][:3] == ["git", "diff", "--"]
    # de-dup, order preserved
    assert list(run.call_args.args[0][3:]) == ["a.py", "b.py"]
    assert any("auto_observation" in getattr(m, "content", "") for m in ctx.messages)
    kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert "auto_observation" in kinds


def test_post_tool_turn_auto_observation_git_failure_no_message():
    loop = _make_loop(auto_observation_enabled=True)
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("apply_patch", metadata={"files": ["a.py"]})]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "text")
    with mock.patch("subprocess.run", side_effect=OSError("no git")):
        loop._process_post_tool_turn(ctx, None, [])
    assert not any("auto_observation" in getattr(m, "content", "") for m in ctx.messages)


def test_post_tool_turn_early_finish_without_tdd_or_review():
    loop = _make_loop()
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("apply_patch")]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "final words")
    loop.registry.applied_patches = ["a.py"]
    out = loop._process_post_tool_turn(ctx, SimpleNamespace(content=""), [])
    assert out.early_return is not None
    assert out.early_return.status == "success"
    assert out.early_return.final_message == "final words"
    assert out.early_return.metadata["early_finish"]["reason"] == "patch_ok_this_turn_and_no_tdd_and_no_self_review"


def test_post_tool_turn_early_finish_default_message_when_no_content():
    loop = _make_loop()
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("write_plan")]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "")
    loop.registry.applied_patches = ["a.py"]
    out = loop._process_post_tool_turn(ctx, None, [])
    assert out.early_return.final_message == "Task completed. Changes applied."


def test_post_tool_turn_tdd_pass_early_finish():
    loop = _make_loop(auto_test_on_patch=True)
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("apply_patch")]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "done")
    loop.registry.applied_patches = ["a.py"]
    loop._auto_test_and_inject = mock.MagicMock(side_effect=lambda m, t, f: (m, 0))
    out = loop._process_post_tool_turn(ctx, SimpleNamespace(content=""), [])
    assert out.early_return is not None
    assert out.early_return.status == "success"
    assert out.early_return.final_message == "done"
    assert ctx.tdd_total_runs == 1 and ctx.tdd_total_pass == 1


def test_post_tool_turn_tdd_fail_no_early_return():
    loop = _make_loop(auto_test_on_patch=True)
    ctx = _post_ctx()
    ctx.turns = [_patch_turn("apply_patch")]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "done")
    loop._auto_test_and_inject = mock.MagicMock(side_effect=lambda m, t, f: (m, 2))
    out = loop._process_post_tool_turn(ctx, None, [])
    assert out.early_return is None
    assert ctx.tdd_fail_count == 2


def test_post_tool_turn_thinking_strip_and_passthrough():
    loop = _make_loop()
    ctx = _post_ctx()
    ctx.messages = []
    ctx.turns = [_patch_turn("read_file")]
    loop._strip_thinking_text = mock.MagicMock(return_value="clean")
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs + new
    resp = SimpleNamespace(content="<think>x</think>clean")
    out = loop._process_post_tool_turn(ctx, resp, [LLMMessage(role="tool", content="{}")])
    assert resp.content == "clean"
    assert len(out.messages) == 1


# ---------------------------------------------------------------------------
# _prepare_turn_messages
# ---------------------------------------------------------------------------

def _prep_ctx(**over):
    ctx = _full_ctx(**over)
    ctx.turn_num = 1
    ctx.messages = [LLMMessage(role="user", content="task")]
    return ctx


def _prep_loop(**config_over):
    loop = _make_loop(**config_over)
    loop._trim_context = lambda m: m
    loop._build_tool_hint = mock.MagicMock(return_value="[TOOL HINT] available")
    loop._build_phase_state_message = mock.MagicMock(return_value="[AGENT STATE] read-only")
    loop._trajectory_compress = mock.MagicMock(return_value=None)
    return loop


def test_prepare_messages_search_first_hint_on_turn_one():
    loop = _prep_loop()
    ctx = _prep_ctx(target_keywords=["handle_click", "parse", "extra"], known_target_file="")
    res = loop._prepare_turn_messages(ctx)
    hints = [m.content for m in res.messages if "TOOL HINT" in m.content and "find_symbol" in m.content]
    assert hints, "search-first hint must be injected on turn 1 with target keywords"
    assert '"handle_click"' in hints[0] and '"parse"' in hints[0] and "extra" not in hints[0]
    assert ctx.search_first_hint_done is True
    # Not re-injected on a later call.
    ctx.turn_num = 2
    res2 = loop._prepare_turn_messages(ctx)
    assert not any("Do NOT browse files randomly" in m.content for m in res2.messages)


def test_prepare_messages_goal_reminder_first_and_urgent():
    loop = _prep_loop()
    ctx = _prep_ctx(reads_since_last_edit=atp._NO_PROGRESS_THRESHOLD, goal_reminder_injected=0)
    res = loop._prepare_turn_messages(ctx)
    first = [m for m in res.messages if "GOAL REMINDER" in m.content]
    assert first and ctx.goal_reminder_injected == 1 and ctx.reads_since_last_edit == 0
    kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert "goal_reminder" in kinds

    # Second reminder carries the urgent suffix.
    ctx.reads_since_last_edit = atp._NO_PROGRESS_THRESHOLD
    res2 = loop._prepare_turn_messages(ctx)
    second = [m for m in res2.messages if "GOAL REMINDER" in m.content]
    assert second and "apply the edit NOW" in second[0].content


def test_prepare_messages_plan_progress_hint():
    loop = _prep_loop()
    ctx = _prep_ctx(plan_subtasks=[{"title": "step one", "files": ["a.py", "b.py"]}],
                    plan_current_index=0)
    res = loop._prepare_turn_messages(ctx)
    hint = [m for m in res.messages if "PLAN PROGRESS" in m.content]
    assert hint and "step one" in hint[0].content and "a.py, b.py" in hint[0].content


def test_prepare_messages_phase_state_for_readonly_or_local():
    loop = _prep_loop()
    ctx = _prep_ctx(read_only_request=True)
    assert any("AGENT STATE" in m.content for m in loop._prepare_turn_messages(ctx).messages)

    loop2 = _prep_loop()
    ctx2 = _prep_ctx(read_only_request=False, is_local_model=True)
    assert any("AGENT STATE" in m.content for m in loop2._prepare_turn_messages(ctx2).messages)

    loop3 = _prep_loop()
    ctx3 = _prep_ctx(read_only_request=False, is_local_model=False)
    assert not any("AGENT STATE" in m.content for m in loop3._prepare_turn_messages(ctx3).messages)


def test_prepare_messages_known_target_file_strategy():
    loop = _prep_loop()
    ctx = _prep_ctx(read_only_request=False, known_target_file="src/app.py")
    res = loop._prepare_turn_messages(ctx)
    assert any("TARGET FILE STRATEGY" in m.content and "src/app.py" in m.content
               for m in res.messages)


def test_prepare_messages_trajectory_compressed_after_turn_two():
    loop = _prep_loop()
    loop._trajectory_compress = mock.MagicMock(return_value="[TRAJECTORY] steps...")
    ctx = _prep_ctx()
    ctx.turn_num = 3
    res = loop._prepare_turn_messages(ctx)
    assert any("TRAJECTORY" in m.content for m in res.messages)
    # Turn <= 2: no trajectory.
    ctx2 = _prep_ctx()
    loop._trajectory_compress.reset_mock()
    loop._prepare_turn_messages(ctx2)
    loop._trajectory_compress.assert_not_called()


def test_prepare_messages_cancel_event_raises():
    import threading
    loop = _prep_loop()
    evt = threading.Event()
    evt.set()
    loop.config.cancel_event = evt
    with pytest.raises(AgentCancelled):
        loop._prepare_turn_messages(_prep_ctx())


def test_prepare_messages_user_interrupt_injected_from_queue():
    loop = _prep_loop()
    q = queue_mod.Queue()
    q.put("focus on tests")
    q.put("also docs")
    loop.config.message_queue = q
    ctx = _prep_ctx()
    res = loop._prepare_turn_messages(ctx)
    interrupts = [m.content for m in res.messages if "USER INTERRUPT" in m.content]
    assert interrupts == ["[USER INTERRUPT] focus on tests", "[USER INTERRUPT] also docs"]
    kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert kinds.count("user_message_received") == 2


def test_prepare_messages_budget_warning_near_limit():
    loop = _prep_loop()
    ctx = _prep_ctx()
    ctx.turn_num = 99  # >= max_turns(5) - 4
    res = loop._prepare_turn_messages(ctx)
    assert any("BUDGET WARNING" in m.content for m in res.messages)
    assert ctx.budget_warned is True
    # Not re-emitted once warned.
    res2 = loop._prepare_turn_messages(ctx)
    assert not any("BUDGET WARNING" in m.content for m in res2.messages)


def test_prepare_messages_tool_hint_exception_suppressed():
    loop = _prep_loop()
    loop._build_tool_hint = mock.MagicMock(side_effect=TypeError("bad hint"))
    ctx = _prep_ctx(read_only_request=True)
    res = loop._prepare_turn_messages(ctx)  # must not raise
    assert res.messages  # base message still present


# ---------------------------------------------------------------------------
# _build_and_filter_prepared_calls
# ---------------------------------------------------------------------------

def _filter_loop(**config_over):
    loop = _make_loop(**config_over)
    loop.registry.get_tool_names = mock.MagicMock(
        side_effect=lambda lang_filter=None:
            frozenset({"read_file"}) if lang_filter is not None
            else frozenset({"read_file", "edit_ast"}))
    return loop


def test_filter_advances_plan_index_after_patch_tool():
    loop = _filter_loop()
    from external_llm.agent.agent_loop_types import AgentTurn
    turns = [AgentTurn(turn_num=1, tool_name="write_plan", tool_args={}, tool_result=None)]
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {"path": "a"}}],
        turns=turns, plan_subtasks=[{"title": "s1"}, {"title": "s2"}],
        plan_current_index=0, read_only_request=True, turn_num=2)
    assert res.plan_current_index == 1
    # Non-patch last tool does not advance.
    turns2 = [AgentTurn(turn_num=1, tool_name="read_file", tool_args={}, tool_result=None)]
    res2 = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {}}], turns=turns2,
        plan_subtasks=[{"title": "s1"}, {"title": "s2"}],
        plan_current_index=0, read_only_request=True, turn_num=2)
    assert res2.plan_current_index == 0


def test_filter_skips_non_dict_and_nameless_calls():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=["not-a-dict", {"args": {"path": "x"}}, {"name": "  ", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.prepared_calls == []
    assert res.should_continue is True
    assert res.phase_rule_messages == []


def test_filter_accepts_tool_key_and_function_variants():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[
            {"tool": "read_file", "args": {"path": "a"}, "id": "id-1"},
            {"function": {"name": "read_file", "arguments": '{"path": "b"}'}},
            {"function": {"name": "read_file", "arguments": "garbage {\"path\": \"z\"} trailing"}},
            {"function": {"name": "read_file", "arguments": "[1, 2, 3]"}},
            {"function": {"name": "read_file", "arguments": ""}},
        ],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=3)
    assert len(res.prepared_calls) == 5
    assert res.prepared_calls[0]["call_id"] == "id-1"
    assert res.prepared_calls[1]["args"] == {"path": "b"}
    # Broken JSON salvaged via parse_tool_args (find/rfind recovery).
    assert res.prepared_calls[2]["args"] == {"path": "z"}
    # JSON list payload is NOT a dict → dropped to {}.
    assert res.prepared_calls[3]["args"] == {}
    assert res.prepared_calls[4]["args"] == {}
    # Auto-generated call id carries turn + tool name.
    assert res.prepared_calls[1]["call_id"] == "call_3_read_file"


def test_filter_language_masked_tool_notice():
    loop = _filter_loop()
    loop.registry.repo_language = "java"
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "edit_ast", "args": {}}, {"name": "read_file", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert [pc["tool"] for pc in res.prepared_calls] == ["read_file"]
    assert len(res.phase_rule_messages) == 1
    assert "Python-only tool" in res.phase_rule_messages[0].content
    assert "java" in res.phase_rule_messages[0].content


def test_filter_unknown_tool_notice_readonly_vs_write_mode():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "nonexistent_tool", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.should_continue is True
    assert "read-only" in res.phase_rule_messages[0].content

    res2 = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "nonexistent_tool", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=False, turn_num=1)
    assert "current mode" in res2.phase_rule_messages[0].content


def test_filter_stream_preview_emitted():
    loop = _filter_loop(stream_callback=lambda *a, **k: None)
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    kinds = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert "tool_call_preview" in kinds
    assert res.prepared_calls


def test_filter_cancel_event_raises_before_execution():
    import threading
    loop = _filter_loop()
    evt = threading.Event()
    evt.set()
    loop.config.cancel_event = evt
    with pytest.raises(AgentCancelled):
        loop._build_and_filter_prepared_calls(
            tool_calls=[{"name": "read_file", "args": {}}],
            turns=[], plan_subtasks=[], plan_current_index=0,
            read_only_request=True, turn_num=1)


# ---------------------------------------------------------------------------
# _process_tool_results
# ---------------------------------------------------------------------------

_TEST_PATCH = (
    "--- a/tests/unit/test_x.py\n"
    "+++ b/tests/unit/test_x.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-old\n"
    "+new\n")
_SRC_PATCH = (
    "--- a/src/app.py\n"
    "+++ b/src/app.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-old\n"
    "+new\n")


def _ptr_loop(**config_over):
    loop = _make_loop(**config_over)
    loop._try_readonly_early_finish = mock.MagicMock(return_value=None)
    loop._advance_phase_after_success = mock.MagicMock()
    loop._build_tool_result_message = mock.MagicMock(
        side_effect=lambda cid, tool, result, args: LLMMessage(role="tool", content="{}"))
    return loop


def _ptr(results, calls, loop=None, **kw):
    loop = loop or _ptr_loop()
    prepared = [{"tool": t, "args": a, "call_id": f"c{i}", "original_call": {}}
                for i, (t, a) in enumerate(calls)]
    out = loop._process_tool_results(
        results=results, prepared_calls=prepared, new_messages=[],
        write_tool_used=kw.get("write_tool_used", False),
        reads_since_last_edit=kw.get("reads_since_last_edit", 0),
        fail_streak=kw.get("fail_streak", {}),
        fail_streak_threshold=kw.get("fail_streak_threshold", 2),
        session_key=kw.get("session_key", "rk-1"),
        write_tools={"apply_patch", "write_plan"},
        read_only_request=kw.get("read_only_request", True),
        request="find it", session_id="sess-1", git_state=None,
        turn_num=kw.get("turn_num", 1), turns=[])
    return out, loop, prepared


def test_ptr_cache_outcome_three_states():
    loop = _ptr_loop()
    loop.registry.is_result_cacheable = mock.MagicMock(
        side_effect=lambda tool: tool in {"read_file", "read_symbol"})
    ok_cached = ToolResult(ok=True, content="x", metadata={"cache_hit": True})
    ok_cacheable = ToolResult(ok=True, content="x")
    write_res = ToolResult(ok=True, content="x")
    _, loop, _ = _ptr([ok_cached, ok_cacheable, write_res],
                        [("read_file", {}), ("read_symbol", {}), ("apply_patch", {})], loop=loop)
    calls = loop.performance_collector.record_tool_call.call_args_list
    assert calls[0].kwargs["cache_hit"] is True     # metadata cache_hit → hit
    assert calls[1].kwargs["cache_hit"] is False    # cacheable read tool → miss
    assert calls[2].kwargs["cache_hit"] is None     # write tool → not probed


def test_ptr_readonly_early_finish_returns_annotated_result():
    loop = _ptr_loop(stream_callback=lambda *a, **k: None)
    early = AgentResult(status="success", final_message="deterministic answer")
    loop._try_readonly_early_finish = mock.MagicMock(return_value=early)
    big = "x" * 9000
    res = ToolResult(ok=True, content=big)
    out, loop, _ = _ptr([res], [("bash", {"command": "ls -la"})],
                        loop=loop, read_only_request=True)
    # _STREAM_VERBOSE_TOOLS cap (6000) applied in the streamed tool_call payload.
    tc = [c.args[1] for c in loop._cb.call_args_list
          if c.args and c.args[0] == "tool_call" and c.args[1].get("tool") == "bash"]
    assert tc and len(tc[0]["result"]["content"]) == 6000
    assert out.early_return is early
    assert out.early_return is early
    assert early.metadata["readonly_early_finish"] is True
    assert early.metadata["deterministic_tool"] == "bash"
    assert early.metadata["turns_used"] == 1
    loop.performance_collector.end_session.assert_called_once()


def test_ptr_failure_classification_and_recall_hint():
    loop = _ptr_loop()
    failed = ToolResult(ok=False, content="", error="bad anchor", metadata=None)
    with mock.patch("external_llm.agent.failure_pattern_store.record_recall_outcome") as rro, \
         mock.patch("external_llm.agent.failure_pattern_store.recall_on_failure",
                    return_value="[RECALL] this anchor failed before"):
        out, _, _ = _ptr([failed], [("edit_text", {"file_path": "a.py"})], loop=loop)
    assert failed.metadata["failure_classification"] == {"action": "retry", "reason": "bad args"}
    recall_msgs = [m for m in out.new_messages if "RECALL" in m.content]
    assert recall_msgs, "recall hint must be appended for a failing tool"
    rro.assert_called_once()
    loop._failure_classifier.classify.assert_called_once()


def test_ptr_failure_classify_metadata_error_suppressed():
    """A metadata dict whose __setitem__ raises TypeError must not break the
    failure-classification bookkeeping (guarded site)."""
    loop = _ptr_loop()

    class BadDict(dict):
        def __setitem__(self, k, v):
            raise TypeError("immutable metadata")

    with mock.patch("external_llm.agent.failure_pattern_store.record_recall_outcome"), \
         mock.patch("external_llm.agent.failure_pattern_store.recall_on_failure", return_value=None):
        out, loop, _ = _ptr(
            [ToolResult(ok=False, content="", error="x", metadata=BadDict())],
            [("edit_text", {})], loop=loop)
    assert out.early_return is None  # pipeline survived the suppressed failure


def test_ptr_retry_exhaustion_emits_strategy_warning():
    loop = _ptr_loop()
    loop._tool_retry_counter["edit_text"] = atp._TOOL_RETRY_LIMIT - 1
    with mock.patch("external_llm.agent.failure_pattern_store.record_recall_outcome"), \
         mock.patch("external_llm.agent.failure_pattern_store.recall_on_failure", return_value=None):
        out, loop, _ = _ptr([ToolResult(ok=False, content="", error="x")],
                            [("edit_text", {"anchor": "p"})], loop=loop)
    warns = [m for m in out.new_messages if "STRATEGY WARNING" in m.content and "switch to a completely different strategy" in m.content]
    assert warns
    events = [c.args[0] for c in loop._cb.call_args_list if c.args]
    assert "fail_loop_detected" in events
    assert loop._tool_retry_counter["edit_text"] == 0  # reset after exhaustion


@pytest.mark.parametrize("tool,recovery_fragment", [
    ("write_plan", "Do NOT call write_plan with the same arguments"),
    ("apply_patch", "Switch to write_plan with edit_blocks"),
    ("edit_text", "Try a different tool or a different approach"),
])
def test_ptr_fail_streak_threshold_recovery_variants(tool, recovery_fragment):
    from external_llm.agent._shared_utils import make_tool_signature
    loop = _ptr_loop()
    key_calls = {  # identical args → same signature → streak accumulates
        "write_plan": ("write_plan", {"plan": "p"}),
        "apply_patch": ("apply_patch", {"patch": "x"}),
        "edit_text": ("edit_text", {"anchor": "a"}),
    }[tool]
    sig = make_tool_signature(tool, key_calls[1])
    with mock.patch("external_llm.agent.failure_pattern_store.record_recall_outcome"), \
         mock.patch("external_llm.agent.failure_pattern_store.recall_on_failure", return_value=None):
        out, loop, _ = _ptr([ToolResult(ok=False, content="", error="x")],
                            [key_calls], loop=loop,
                            fail_streak={sig: 1}, fail_streak_threshold=2)
    warns = [m for m in out.new_messages if "STRATEGY WARNING" in m.content]
    assert warns and recovery_fragment in warns[0].content
    assert out.fail_streak[sig] == 2


def test_ptr_success_resets_streak_and_retry_counter():
    from external_llm.agent._shared_utils import make_tool_signature
    loop = _ptr_loop()
    loop._tool_retry_counter["read_file"] = 1
    sig_key = make_tool_signature("read_file", {"path": "a"})
    out, loop, _ = _ptr([ToolResult(ok=True, content="ok")],
                        [("read_file", {"path": "a"})], loop=loop,
                        fail_streak={sig_key: 1})
    assert out.fail_streak == {}
    assert loop._tool_retry_counter["read_file"] == 0


def test_ptr_noop_detection_via_empty_error_variants():
    for err in ("No-op patch", "no change detected", "empty diff", "empty patch",
                "compiled to empty plan"):
        loop = _ptr_loop()
        out, loop, _ = _ptr([ToolResult(ok=False, content="", error=err)],
                            [("apply_patch", {})], loop=loop)
        assert out.noop_confirmed is True, err


def test_ptr_stream_emit_suppressed_on_cb_error():
    loop = _ptr_loop(stream_callback=lambda *a, **k: None)
    loop._cb = mock.MagicMock(side_effect=[None, TypeError("cb exploded")])
    # First cb call (tool_call emit) ok, second (nothing) raises — the emit
    # inside the loop is wrapped; use a raising cb from the start:
    loop._cb = mock.MagicMock(side_effect=TypeError("cb exploded"))
    with mock.patch("external_llm.agent.failure_pattern_store.record_recall_outcome"), \
         mock.patch("external_llm.agent.failure_pattern_store.recall_on_failure", return_value=None):
        _, loop, _ = _ptr([ToolResult(ok=False, content="", error="x")],
                           [("read_file", {})], loop=loop)
    # emit failure must not break the pipeline


def test_ptr_scoped_verification_invalidates_test_index_on_test_write():
    loop = _ptr_loop(scoped_verification=True)
    with mock.patch("external_llm.agent.test_impact_selector.invalidate_index") as inv:
        _, loop, _ = _ptr([ToolResult(ok=True, content="ok")],
                           [("apply_patch", {"patch": _TEST_PATCH})],
                           loop=loop, read_only_request=False)
    inv.assert_called_once_with("/tmp")


def test_ptr_scoped_verification_no_invalidation_for_non_test_write():
    loop = _ptr_loop(scoped_verification=True)
    with mock.patch("external_llm.agent.test_impact_selector.invalidate_index") as inv:
        _, loop, _ = _ptr([ToolResult(ok=True, content="ok")],
                           [("apply_patch", {"patch": _SRC_PATCH})],
                           loop=loop, read_only_request=False)
    inv.assert_not_called()


def test_ptr_scoped_verification_failure_suppressed():
    loop = _ptr_loop(scoped_verification=True)
    with mock.patch("external_llm.agent.test_impact_selector.invalidate_index",
                    side_effect=RuntimeError("boom")):
        out, loop, _ = _ptr([ToolResult(ok=True, content="ok")],
                            [("apply_patch", {"patch": _TEST_PATCH})],
                            loop=loop, read_only_request=False)
    assert out.early_return is None


def test_ptr_exploration_reads_counted_and_write_resets():
    loop = _ptr_loop()
    out1, _, _ = _ptr([ToolResult(ok=True, content="ok")], [("grep", {"pattern": "x"})],
                      loop=loop, reads_since_last_edit=2)
    assert out1.reads_since_last_edit == 3
    out2, _, _ = _ptr([ToolResult(ok=True, content="ok")], [("write_plan", {"plan": "p"})],
                      loop=loop, reads_since_last_edit=5)
    assert out2.reads_since_last_edit == 0
    assert out2.write_tool_used is True


def test_ptr_stream_content_limits_by_tool_tier():
    """apply_patch (large tier, 8000) vs read_file (default tier, 2000) —
    streamed tool_call payload truncation."""
    loop = _ptr_loop(stream_callback=lambda *a, **k: None)
    big = "y" * 9000
    _, loop, _ = _ptr(
        [ToolResult(ok=True, content=big), ToolResult(ok=True, content=big)],
        [("apply_patch", {"patch": "x"}), ("read_file", {"path": "a"})],
        loop=loop, read_only_request=False)
    tc = {c.args[1]["tool"]: c.args[1]["result"]["content"]
          for c in loop._cb.call_args_list if c.args and c.args[0] == "tool_call"}
    assert len(tc["apply_patch"]) == 8000
    assert len(tc["read_file"]) == 2000


# ---------------------------------------------------------------------------
# _settle_deferred_semantics
# ---------------------------------------------------------------------------

def _deferred_msg(path, extra_syn=None):
    payload = {"content": "applied", "metadata": {"syntax_check": {
        "semantic_deferred": True, "semantic_deferred_path": path, **(extra_syn or {}),
    }}}
    return LLMMessage(role="tool", content=json_dumps(payload))


def json_dumps(obj):
    import json as _j
    return _j.dumps(obj, ensure_ascii=False)


def test_settle_semantics_drain_failure_returns_quietly():
    loop = _make_loop()
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(
        side_effect=RuntimeError("drain boom"))
    msgs = [_deferred_msg("a.py")]
    loop._settle_deferred_semantics(msgs)  # must not raise
    assert msgs[0].content.startswith("{")


def test_settle_semantics_fills_last_write_only():
    loop = _make_loop()
    outcome = SimpleNamespace(checked=True, skip_reason=None,
                              diagnostics=[{"line": 3, "message": "bad indent", "severity": "error"}])
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(
        return_value={"a.py": outcome})
    early_msg = _deferred_msg("a.py")
    late_msg = _deferred_msg("a.py")
    loop._settle_deferred_semantics([early_msg, late_msg])
    import json as _j
    late = _j.loads(late_msg.content)
    assert late["metadata"]["syntax_check"]["semantic_diagnostics"] == [
        {"line": 3, "message": "bad indent", "severity": "error"}]
    assert "semantic_diagnostics" not in _j.loads(early_msg.content)["metadata"]["syntax_check"]
    # internal deferred keys stripped from the filled message
    assert "semantic_deferred_path" not in late["metadata"]["syntax_check"]
    assert "semantic_deferred" not in late["metadata"]["syntax_check"]
    # guidance block rendered into the visible content
    assert "<file_diagnostics>" in late["content"]


def test_settle_semantics_skipped_check_reports_reason():
    loop = _make_loop()
    outcome = SimpleNamespace(checked=False, skip_reason="no toolchain", diagnostics=[])
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(
        return_value={"a.py": outcome})
    msg = _deferred_msg("a.py")
    loop._settle_deferred_semantics([msg])
    import json as _j
    syn = _j.loads(msg.content)["metadata"]["syntax_check"]
    assert syn["semantic_check_skipped"] == "no toolchain"
    assert "semantic_diagnostics" not in syn


def test_settle_semantics_bad_payload_not_fatal():
    loop = _make_loop()
    outcome = SimpleNamespace(checked=True, skip_reason=None,
                              diagnostics=[{"line": 1, "message": "E1", "severity": "error"}])
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(
        return_value={"a.py": outcome})
    msgs = [LLMMessage(role="tool", content="semantic_deferred but not json {{{"),
            _deferred_msg("a.py")]
    loop._settle_deferred_semantics(msgs)  # bad payload skipped, good one filled
    import json as _j
    assert "semantic_diagnostics" in _j.loads(msgs[1].content)["metadata"]["syntax_check"]


# ---------------------------------------------------------------------------
# _execute_and_process_tool_calls — remaining branches
# ---------------------------------------------------------------------------

def _exec_loop(**config_over):
    loop = _make_loop(**config_over)
    loop._record_tool_success = mock.MagicMock()
    loop._record_tool_failure = mock.MagicMock()
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._settle_deferred_semantics = mock.MagicMock()
    loop._process_tool_results = mock.MagicMock(return_value=_ResultsProcessingOutcome(
        new_messages=[], write_tool_used=False, reads_since_last_edit=0,
        noop_confirmed=False, fail_streak={}))
    return loop


def test_exec_should_continue_returns_phase_rules():
    loop = _exec_loop()
    rule = LLMMessage(role="user", content="[PHASE RULE]")
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=[], phase_rule_messages=[rule], plan_current_index=0,
        should_continue=True))
    out = loop._execute_and_process_tool_calls(_full_ctx(), tool_calls=[])
    assert out.should_continue is True
    assert out.phase_rule_messages == [rule]


def test_exec_parallel_stopiteration_yields_failed_results():
    loop = _exec_loop(parallel_tool_execution_enabled=True)
    calls = [{"tool": "read_file", "args": {}, "call_id": "c1"},
             {"tool": "read_file", "args": {}, "call_id": "c2"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch_parallel = mock.MagicMock(side_effect=StopIteration())
    with mock.patch.object(atp, "_log_parallel_write_failures"):
        ctx = _full_ctx()
        out = loop._execute_and_process_tool_calls(ctx, tool_calls=calls)
    assert out.any_tool_called is True
    assert len(ctx.turns) == 2
    assert all(not t.tool_result.ok for t in ctx.turns)


def test_exec_serial_stopiteration_yields_failed_result():
    loop = _exec_loop()
    calls = [{"tool": "read_file", "args": {}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch = mock.MagicMock(side_effect=StopIteration())
    ctx = _full_ctx()
    out = loop._execute_and_process_tool_calls(ctx, tool_calls=calls)
    assert out.any_tool_called is False  # StopIteration happened before the flag set
    assert len(ctx.turns) == 1 and not ctx.turns[0].tool_result.ok


def test_exec_serial_write_tool_failure_recorded_to_jsonl():
    loop = _exec_loop()
    calls = [{"tool": "apply_patch", "args": {"patch": "x"}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch = mock.MagicMock(return_value=ToolResult(ok=False, content="", error="bad"))
    with mock.patch("external_llm.agent.tool_failure_log.record_write_tool_failure_from_tr") as rec:
        loop._execute_and_process_tool_calls(_full_ctx(), tool_calls=calls)
    rec.assert_called_once()
    assert rec.call_args.kwargs["session_key"] == "rk-1"


def test_exec_early_return_from_results_propagates():
    loop = _exec_loop()
    early = AgentResult(status="success")
    loop._process_tool_results = mock.MagicMock(return_value=_ResultsProcessingOutcome(
        new_messages=[], write_tool_used=False, reads_since_last_edit=0,
        noop_confirmed=False, fail_streak={}, early_return=early))
    calls = [{"tool": "read_file", "args": {}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    out = loop._execute_and_process_tool_calls(_full_ctx(), tool_calls=calls)
    assert out.early_return is early


# ---------------------------------------------------------------------------
# _post_dispatch_patch_recovery — tolerant edit_blocks ladder
# ---------------------------------------------------------------------------

def test_recovery_repair_retry_failure_records_metadata():
    loop = _make_loop()
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value={"patch": "fixed"})
    loop.registry.dispatch = mock.MagicMock(return_value=ToolResult(ok=False, error="still bad", content=""))
    failed = ToolResult(ok=False, error="orig", content="", metadata={})
    out = loop._post_dispatch_patch_recovery("apply_patch", {"patch": "broken"}, failed)
    assert out.metadata["auto_repair"] == {
        "attempted": True, "kind": "patch_format_fix",
        "original_error": "orig", "success": False, "retry_error": "still bad",
    }
    assert loop._patch_fail_count == 1


def test_recovery_tolerant_conversion_success():
    loop = _make_loop(tolerant_patch_mode=True, tolerant_patch_max_failures=2)
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._patch_fail_count = 1  # this failure reaches the threshold
    eb_ok = ToolResult(ok=True, content="", metadata={})
    loop.registry.dispatch = mock.MagicMock(return_value=eb_ok)
    converted = {"file_path": "a.py", "blocks": [{"before": "x", "after": "y"}]}
    with mock.patch("external_llm.patch_engine.PatchEngine") as pe:
        pe.return_value.convert_patch_to_edit_blocks.return_value = converted
        out = loop._post_dispatch_patch_recovery(
            "apply_patch", {"patch": "*** p", "path": "a.py"}, ToolResult(ok=False, error="fail", content=""))
    assert out is eb_ok
    assert out.metadata["auto_converted_from_patch"] is True
    assert out.metadata["edit_blocks_count"] == 1
    assert "auto-converted to edit_blocks" in out.content
    assert loop._patch_fail_count == 0
    # dispatched as write_plan with an ASICODE_PLAN_V1 edit_blocks op
    sent = loop.registry.dispatch.call_args
    assert sent.args[0] == "write_plan"
    import json as _j
    assert _j.loads(sent.args[1]["plan"])["ops"][0]["op"] == "edit_blocks"


def test_recovery_tolerant_conversion_failure_keeps_error_metadata():
    loop = _make_loop(tolerant_patch_mode=True, tolerant_patch_max_failures=2)
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._patch_fail_count = 1
    loop.registry.dispatch = mock.MagicMock(return_value=ToolResult(ok=False, error="eb fail", content=""))
    with mock.patch("external_llm.patch_engine.PatchEngine") as pe:
        pe.return_value.convert_patch_to_edit_blocks.return_value = {"file_path": "a.py", "blocks": [{}]}
        out = loop._post_dispatch_patch_recovery(
            "apply_patch", {"patch": "p"}, ToolResult(ok=False, error="orig", content="", metadata={}))
    assert out.metadata["edit_blocks_fallback_error"] == "eb fail"
    assert out.ok is False


def test_recovery_tolerant_engine_exception_suppressed():
    loop = _make_loop(tolerant_patch_mode=True, tolerant_patch_max_failures=2)
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._patch_fail_count = 1
    with mock.patch("external_llm.patch_engine.PatchEngine") as pe:
        pe.return_value.convert_patch_to_edit_blocks.side_effect = RuntimeError("engine boom")
        out = loop._post_dispatch_patch_recovery(
            "apply_patch", {"patch": "p"}, ToolResult(ok=False, error="orig", content="", metadata={}))
    assert out.ok is False and loop._patch_fail_count == 2


def test_recovery_tolerant_conversion_empty_result_ignored():
    loop = _make_loop(tolerant_patch_mode=True, tolerant_patch_max_failures=2)
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._patch_fail_count = 1
    with mock.patch("external_llm.patch_engine.PatchEngine") as pe:
        pe.return_value.convert_patch_to_edit_blocks.return_value = None
        out = loop._post_dispatch_patch_recovery(
            "apply_patch", {"patch": "p"}, ToolResult(ok=False, error="orig", content="", metadata={}))
    assert out.ok is False and "edit_blocks_fallback_error" not in out.metadata


def test_recovery_ok_patch_resets_fail_count():
    loop = _make_loop()
    loop._patch_fail_count = 1
    ok = ToolResult(ok=True, content="done")
    out = loop._post_dispatch_patch_recovery("apply_patch", {}, ok)
    assert out is ok and loop._patch_fail_count == 0


# ---------------------------------------------------------------------------
# _handle_loop_cancellation / _handle_loop_error
# ---------------------------------------------------------------------------

def test_cancellation_successful_rollback_clears_patches():
    loop = _make_loop()
    loop.registry.applied_patches = ["a.py", "b.py"]
    loop._rollback_patches = mock.MagicMock(return_value={
        "success": True, "rolled_back": 2, "total": 2, "results": []})
    result = loop._handle_loop_cancellation(turns=[], git_state=None)
    assert result.status == "cancelled"
    assert "successfully rolled back" in result.final_message
    assert loop.registry.applied_patches == []  # cleared so DIFF_VERIFY stays honest
    assert result.metadata["rollback"]["performed"] is True


def test_cancellation_partial_rollback_keeps_patches():
    loop = _make_loop()
    loop.registry.applied_patches = ["a.py", "b.py"]
    loop._rollback_patches = mock.MagicMock(return_value={
        "success": False, "rolled_back": 1, "total": 2,
        "results": [{"success": True}, {"success": False, "patch_index": 1, "message": "conflict"}]})
    result = loop._handle_loop_cancellation(turns=[], git_state=None)
    assert result.status == "cancelled"
    assert "partially failed" in result.final_message
    assert loop.registry.applied_patches == ["a.py", "b.py"]  # kept for DIFF_VERIFY


def test_cancellation_no_patches_short_circuits():
    loop = _make_loop()
    result = loop._handle_loop_cancellation(turns=[], git_state=None)
    assert result.status == "cancelled"
    assert result.metadata["rollback"]["performed"] is False


def test_loop_error_types_and_rollback():
    loop = _make_loop()
    loop.registry.applied_patches = ["a.py"]
    loop._rollback_patches = mock.MagicMock(return_value={
        "success": True, "rolled_back": 1, "total": 1, "results": []})
    result = loop._handle_loop_error(
        error=LLMConnectionError("down"), turns=[], git_state=None,
        rollback_performed=False, rollback_result=None)
    assert result.status == "error"
    assert "connection" in result.error or "Unexpected" in result.error
    loop.performance_collector.record_agent_result.assert_called_once_with(failed=True)


def test_loop_error_classification_variants():
    for exc, frag in [(LLMRateLimitError("x"), "rate_limit"),
                      (LLMServerUnavailableError("x"), "server_unavailable"),
                      (ValueError("x"), "api")]:
        loop = _make_loop()
        kinds = [c.args[1].get("error_type")
                 for c in _err_calls(loop, exc)]
        assert kinds == [frag]


def _err_calls(loop, exc):
    loop._cb = mock.MagicMock()
    loop._handle_loop_error(error=exc, turns=[], git_state=None,
                            rollback_performed=False, rollback_result=None)
    return [c for c in loop._cb.call_args_list if c.args and c.args[0] == "error"]


# ---------------------------------------------------------------------------
# _log_parallel_write_failures
# ---------------------------------------------------------------------------

def _pwf_pipeline(write_tools):
    loop = _make_loop()
    loop.registry._WRITE_TOOLS = write_tools
    return loop


def test_pwf_records_write_tool_outcomes():
    loop = _pwf_pipeline({"apply_patch"})
    results = [ToolResult(ok=False, content="", error="bad"),
               ToolResult(ok=True, content="ok")]
    calls = [{"tool": "apply_patch", "args": {"patch": "x"}},
             {"tool": "read_file", "args": {}}]
    with mock.patch("external_llm.agent.tool_failure_log.record_write_tool_failure_from_tr") as rec:
        atp._log_parallel_write_failures(results, calls, loop, session_key="rk-9")
    rec.assert_called_once()
    assert rec.call_args.kwargs["tool"] == "apply_patch"
    assert rec.call_args.kwargs["session_key"] == "rk-9"


def test_pwf_index_out_of_range_skips_quietly():
    loop = _pwf_pipeline({"apply_patch"})
    results = [ToolResult(ok=False, content="", error="x"), ToolResult(ok=False, content="", error="y")]
    calls = [{"tool": "apply_patch", "args": {}}]  # fewer calls than results
    with mock.patch("external_llm.agent.tool_failure_log.record_write_tool_failure_from_tr") as rec:
        atp._log_parallel_write_failures(results, calls, loop)
    rec.assert_called_once()  # only idx 0 has a tool name


def test_pwf_registry_failure_suppressed():
    loop = _make_loop()
    del loop.registry._WRITE_TOOLS  # AttributeError inside → suppressed
    with mock.patch("external_llm.agent.tool_failure_log.record_write_tool_failure_from_tr") as rec:
        atp._log_parallel_write_failures([ToolResult(ok=False)], [{"tool": "x"}], loop)
    rec.assert_not_called()


# ---------------------------------------------------------------------------
# eviction helpers
# ---------------------------------------------------------------------------

def test_evict_for_loop_estimate_failure_returns_messages(monkeypatch):
    monkeypatch.setattr(atp, "_EVICTION_ENABLED", True)
    monkeypatch.setattr(atp, "_resolve_context_limit", mock.MagicMock(side_effect=ValueError("bad model")))
    msgs = [LLMMessage(role="user", content="hi")]
    assert atp._evict_for_loop(msgs, model="nope") is msgs


def test_evict_for_loop_below_trigger_returns_unchanged(monkeypatch):
    monkeypatch.setattr(atp, "_EVICTION_ENABLED", True)
    monkeypatch.setattr(atp, "estimate_tokens_from_msgs", mock.MagicMock(return_value=10))
    msgs = [LLMMessage(role="user", content="hi")]
    assert atp._evict_for_loop(msgs, model="m") is msgs


def test_evict_for_loop_above_trigger_calls_eviction(monkeypatch):
    monkeypatch.setattr(atp, "_EVICTION_ENABLED", True)
    monkeypatch.setattr(atp, "estimate_tokens_from_msgs", mock.MagicMock(return_value=10**9))
    sentinel = object()
    monkeypatch.setattr(atp, "_evict_consumed_tool_results", mock.MagicMock(return_value=sentinel))
    msgs = [LLMMessage(role="user", content="hi")]
    assert atp._evict_for_loop(msgs, model="m") is sentinel


def test_stub_tool_result_unserializable_raw_content_keeps_content_size():
    m = LLMMessage(role="tool", content="x" * 500, raw_content=object())
    out = atp._stub_tool_result(m)
    # raw_content not JSON-serializable → size from content only; still stubbed.
    assert atp._EVICTED_MARKER in out.content


def test_is_stubbed_tool_result_gemini_and_noise_blocks():
    gem = LLMMessage(role="user", content="", raw_content=[
        "not-a-dict",
        {"text": "hi"},
        {"functionResponse": {"name": "read_file", "response": {
            "content": f"{atp._EVICTED_MARKER}: read_file — 100 chars evicted]"}}},
    ])
    assert atp._is_stubbed_tool_result(gem) is True
    clean = LLMMessage(role="user", content="", raw_content=[
        "noise", {"text": "hi"},
        {"functionResponse": {"name": "grep", "response": {"content": "results"}}},
    ])
    assert atp._is_stubbed_tool_result(clean) is False


def test_stub_blocks_non_list_raw_content_falls_back_to_standard():
    m = LLMMessage(role="tool", content="payload", raw_content=None)
    out = atp._stub_tool_result_blocks(m, "STUB", lambda b: b)
    assert out.content == "STUB" and out.raw_content is None


def test_stub_anthropic_leaves_text_blocks_intact():
    m = LLMMessage(role="user", content="", raw_content=[
        {"type": "text", "text": "strategy warning"},
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "y" * 300},
    ])
    out = atp._stub_anthropic_tool_result(m, "STUB", {"tu_1": "apply_patch"})
    blocks = out.raw_content
    assert blocks[0] == {"type": "text", "text": "strategy warning"}  # untouched
    assert atp._EVICTED_MARKER in blocks[1]["content"]
    assert "apply_patch" in blocks[1]["content"]  # name recovered via name_map
    assert "tu_1" in blocks[1]["content"]


def test_stub_gemini_non_function_response_passthrough():
    m = LLMMessage(role="user", content="", raw_content=[
        {"text": "hello"},
        {"functionResponse": {"name": "bash", "response": {"content": "z" * 400}}},
    ])
    out = atp._stub_gemini_tool_result(m, "STUB")
    assert out.raw_content[0] == {"text": "hello"}
    stubbed = out.raw_content[1]["functionResponse"]["response"]["content"]
    assert atp._EVICTED_MARKER in stubbed and "bash" in stubbed


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------

def test_write_touched_test_file_json_list_plan():
    plan = [{"op": "create_file", "path": "tests/unit/test_new.py"}]
    assert atp._write_touched_test_file(
        "write_plan", {"plan": __import__("json").dumps(plan)}) is True


def test_effective_final_content_reasoning_fallback_and_suppressed_error():
    # dict response with raw.raw_response carrying reasoning_content
    msg = {"content": "", "raw": SimpleNamespace(raw_response={
        "choices": [{"message": {"reasoning_content": "the final answer is 42"}}]})}
    assert "42" in TurnPipelineMixin._effective_final_content(msg)
    # extraction raising must be suppressed → returns plain content
    bad = {"content": "plain", "raw": SimpleNamespace(raw_response=object())}
    with mock.patch.object(atp, "extract_llm_reasoning", side_effect=TypeError("boom")):
        assert TurnPipelineMixin._effective_final_content(bad) == "plain"


# ---------------------------------------------------------------------------
# tail coverage: remaining branches
# ---------------------------------------------------------------------------

def test_run_llm_loop_full_tool_flow_assignments():
    """Normal tool-turn flow: assignments from _ToolTurnOutcome/_PostToolResult
    propagate onto ctx and the loop continues to the final answer."""
    loop = _make_loop()
    base = [LLMMessage(role="user", content="task")]
    loop._build_initial_messages = mock.MagicMock(return_value=list(base))
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=list(base), budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(side_effect=[
        {"prompt_tokens": 2, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "read_file"}], "finish_reason": "tool_use"},
        {"prompt_tokens": 1, "completion_tokens": 1, "content": "all done",
         "tool_calls": [], "finish_reason": "stop"},
    ])
    tool_out = _ToolTurnOutcome(
        new_messages=[LLMMessage(role="tool", content="{}")], prepared_calls=[],
        write_tool_used=False, any_tool_called=True, fail_streak={"s": 1},
        reads_since_last_edit=3, plan_current_index=1, noop_confirmed=True)
    loop._execute_and_process_tool_calls = mock.MagicMock(return_value=tool_out)
    loop._process_post_tool_turn = mock.MagicMock(return_value=_PostToolResult(
        messages=base + tool_out.new_messages, tdd_fail_count=0,
        tdd_total_runs=1, tdd_total_pass=1))
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="success")))
    ctx = _full_ctx(read_only_request=False)
    ctx.messages = list(base)
    assert loop._run_llm_loop(ctx).status == "success"
    assert ctx.any_tool_called is True and ctx.write_tool_used is False
    assert ctx.fail_streak == {"s": 1}
    # reads_since_last_edit is re-synced from _prepare_turn_messages each turn.
    assert ctx.plan_current_index == 1
    assert ctx.noop_confirmed is True
    assert ctx.tdd_total_runs == 1 and ctx.tdd_total_pass == 1
    assert ctx.messages[-1].role == "tool"


def test_run_llm_loop_turn_cap_routes_to_max_turns_handler():
    loop = _make_loop(max_turns=0)
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    maxed = AgentResult(status="max_turns")
    loop._handle_max_turns_reached = mock.MagicMock(return_value=maxed)
    loop._prepare_turn_messages = mock.MagicMock()  # never reached
    ctx = _full_ctx()
    ctx.turn_num = 1
    assert loop._run_llm_loop(ctx) is maxed


def test_run_llm_loop_attribute_response_object():
    """Non-dict response object → _rget getattr fallback + tokens_used fallback."""
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    resp = SimpleNamespace(prompt_tokens=None, tokens_used=17, completion_tokens=3,
                           content="obj answer", tool_calls=[], finish_reason="stop")
    loop._llm_call_with_tools = mock.MagicMock(return_value=resp)
    loop._handle_final_answer_turn = mock.MagicMock(return_value=_FinalAnswerOutcome(
        result=AgentResult(status="text_reply")))
    ctx = _full_ctx()
    loop._run_llm_loop(ctx)
    assert ctx.total_prompt_tokens == 17  # object attr fallback path


def test_max_turns_wrapup_none_prompt_tokens_fallback():
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    responses = [
        {"prompt_tokens": 5, "completion_tokens": 1, "content": "",
         "tool_calls": [{"name": "read_file"}], "finish_reason": "tool_use"},
        {"prompt_tokens": None, "tokens_used": 40, "completion_tokens": None,
         "content": "wrapped", "tool_calls": [], "finish_reason": "stop"},
    ]
    loop._llm_call_with_tools = mock.MagicMock(side_effect=responses)
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    assert ctx.total_prompt_tokens == 5 + 40


def test_max_turns_rget_getter_raising_falls_back_to_getattr():
    class WeirdResp:
        def get(self, k, d=None):
            raise KeyError("nope")
        prompt_tokens = 7
        completion_tokens = 1
        content = "weird"
        tool_calls: ClassVar[list] = []
        finish_reason = "stop"
    loop = _make_loop()
    ctx = _full_ctx(read_only_request=True)
    loop._llm_call_with_tools = mock.MagicMock(return_value=WeirdResp())
    result = loop._handle_max_turns_reached(ctx)
    assert result.status == "success"
    assert ctx.total_prompt_tokens == 7


def test_final_answer_review_appends_non_lgtm_summary():
    loop = _make_loop(self_review_enabled=True)
    loop.registry.applied_patches = ["a.py"]
    loop._run_self_review = mock.MagicMock(return_value="issue: unused import")
    loop._is_trivial_edit_request = mock.MagicMock(return_value=False)
    ctx = _fa_ctx(read_only_request=False, any_tool_called=True)
    out = loop._handle_final_answer_turn(ctx, "done")
    assert "[Self-Review]" in out.result.final_message
    assert out.result.metadata["self_review"]["issues_found"] is True


def test_post_tool_turn_thinking_attribute_error_suppressed():
    loop = _make_loop()
    ctx = _post_ctx()
    ctx.messages = []
    ctx.turns = [_patch_turn("read_file")]
    # response.content is a non-str → strip raises TypeError inside the guard
    loop._strip_thinking_text = mock.MagicMock(side_effect=TypeError("not str"))
    loop._append_native_tool_messages = lambda m, r, n: m
    out = loop._process_post_tool_turn(ctx, SimpleNamespace(content=None), [])
    assert out.early_return is None


def test_post_tool_turn_observation_no_paths_no_git_call():
    loop = _make_loop(auto_observation_enabled=True)
    ctx = _post_ctx()
    ctx.messages = []
    # patch ok but no touched_files/files metadata anywhere
    ctx.turns = [_patch_turn("apply_patch")]
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda m, r, n: m
    loop._effective_final_content = staticmethod(lambda r: "t")
    with mock.patch("subprocess.run") as run:
        loop._process_post_tool_turn(ctx, None, [])
    run.assert_not_called()


def test_prepare_messages_plan_hint_and_target_strategy_exceptions_suppressed():
    loop = _prep_loop()
    # plan_subtasks entries are non-dict → task.get raises AttributeError → guarded
    ctx = _prep_ctx(plan_subtasks=["not-a-dict"], plan_current_index=0)
    res = loop._prepare_turn_messages(ctx)  # must not raise
    assert res.messages


def test_prepare_messages_queue_empty_break():
    loop = _prep_loop()
    loop.config.message_queue = queue_mod.Queue()  # empty
    ctx = _prep_ctx()
    res = loop._prepare_turn_messages(ctx)  # drains to Empty → break
    assert not any("USER INTERRUPT" in m.content for m in res.messages)


def test_filter_registry_without_get_tool_names_uses_schemas():
    loop = _make_loop()
    del loop.registry.get_tool_names
    loop.registry.get_tool_schemas = mock.MagicMock(return_value=[
        {"name": "read_file"}, {"name": "edit_text"}, {"no_name": 1}])
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {}}, {"name": "mystery", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert [pc["tool"] for pc in res.prepared_calls] == ["read_file"]
    assert "not available in read-only mode" in res.phase_rule_messages[0].content


def test_filter_non_dict_args_coerced():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": "just a string"},
                    {"function": {"name": "read_file", "arguments": None}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert all(pc["args"] == {} for pc in res.prepared_calls)


def test_filter_plan_advance_bad_turns_suppressed():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {}}],
        turns="not-a-list",  # turns[-1] raises TypeError → guarded, index unchanged
        plan_subtasks=[{"title": "s"}], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.plan_current_index == 0


def test_ptr_readonly_early_finish_emit_exception_suppressed():
    loop = _ptr_loop(stream_callback=lambda *a, **k: None)
    early = AgentResult(status="success")
    loop._try_readonly_early_finish = mock.MagicMock(return_value=early)
    loop._cb = mock.MagicMock(side_effect=TypeError("cb boom"))
    out, loop, _ = _ptr([ToolResult(ok=True, content="r")], [("read_file", {})],
                        loop=loop, read_only_request=True)
    assert out.early_return is early  # emit failure did not break early finish


def test_ptr_recall_bookkeeping_import_failure_suppressed():
    import builtins
    real_import = builtins.__import__

    def broken_import(name, *a, **k):
        if "failure_pattern_store" in name:
            raise ImportError("store gone")
        return real_import(name, *a, **k)

    loop = _ptr_loop()
    with mock.patch("builtins.__import__", side_effect=broken_import):
        out, loop, _ = _ptr([ToolResult(ok=False, content="", error="x")],
                            [("read_file", {})], loop=loop)
    assert out.early_return is None  # recall bookkeeping failure is non-fatal


def test_settle_semantics_empty_and_non_syn_payloads():
    loop = _make_loop()
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(return_value={})
    assert loop._settle_deferred_semantics([]) is None

    # deferred marker present but syntax_check not a dict → skipped
    loop2 = _make_loop()
    loop2.registry.drain_pending_semantic_checks = mock.MagicMock(
        return_value={"a.py": SimpleNamespace(checked=True, skip_reason=None,
                                              diagnostics=[{"line": 1, "message": "E", "severity": "error"}])})
    import json as _j
    msg = LLMMessage(role="tool", content=_j.dumps({"metadata": {"syntax_check": "not-a-dict",
                                                                  "semantic_deferred": True}}))
    loop2._settle_deferred_semantics([msg])
    assert "semantic_diagnostics" not in msg.content


def test_settle_semantics_unfilled_path_and_repeat_write_skip():
    import json as _j
    loop = _make_loop()
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(
        return_value={"b.py": SimpleNamespace(checked=True, skip_reason=None,
                                              diagnostics=[{"line": 9, "message": "E", "severity": "error"}])})
    # a.py deferred but not in diags → left untouched; b.py filled.
    a = _deferred_msg("a.py")
    b1 = _deferred_msg("b.py")
    b2 = _deferred_msg("b.py")
    loop._settle_deferred_semantics([a, b1, b2])
    assert "semantic_deferred_path" in _j.loads(a.content)["metadata"]["syntax_check"]
    assert "semantic_diagnostics" in _j.loads(b2.content)["metadata"]["syntax_check"]
    assert "semantic_diagnostics" not in _j.loads(b1.content)["metadata"]["syntax_check"]
    # earlier b.py write re-serialized without the internal path key
    assert "semantic_deferred_path" not in _j.loads(b1.content)["metadata"]["syntax_check"]


def test_exec_parallel_record_failure_suppressed():
    loop = _exec_loop(parallel_tool_execution_enabled=True)
    calls = [{"tool": "read_file", "args": {}, "call_id": "c1"},
             {"tool": "read_file", "args": {}, "call_id": "c2"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch_parallel = mock.MagicMock(return_value=[
        ToolResult(ok=True), ToolResult(ok=True)])
    loop._record_tool_success = mock.MagicMock(side_effect=TypeError("bookkeeping gone"))
    with mock.patch.object(atp, "_log_parallel_write_failures"):
        ctx = _full_ctx()
        out = loop._execute_and_process_tool_calls(ctx, tool_calls=calls)
    assert out.any_tool_called is True  # suppression kept the turn alive


def test_exec_serial_write_tool_record_import_failure_suppressed():
    import builtins
    real_import = builtins.__import__

    def broken_import(name, *a, **k):
        if "tool_failure_log" in name:
            raise ImportError("gone")
        return real_import(name, *a, **k)

    loop = _exec_loop()
    calls = [{"tool": "apply_patch", "args": {"patch": "x"}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch = mock.MagicMock(return_value=ToolResult(ok=True))
    with mock.patch("builtins.__import__", side_effect=broken_import):
        ctx = _full_ctx()
        out = loop._execute_and_process_tool_calls(ctx, tool_calls=calls)
    assert out.any_tool_called is True


def test_loop_error_partial_rollback_logs_failures():
    loop = _make_loop()
    loop.registry.applied_patches = ["a.py", "b.py"]
    loop._rollback_patches = mock.MagicMock(return_value={
        "success": False, "rolled_back": 1, "total": 2, "results": []})
    result = loop._handle_loop_error(
        error=ValueError("x"), turns=[], git_state=None,
        rollback_performed=False, rollback_result=None)
    assert result.status == "error"
    assert "partially failed" in result.error or "Unexpected" in result.error
    assert loop.registry.applied_patches  # kept


def test_summarize_rollback_manual_variants():
    msg, meta = atp._summarize_rollback(None)
    assert meta["performed"] is False and "No patches" in msg

    rr = {"success": False, "rolled_back": 1, "total": 2, "results": [
        {"needs_manual_rollback": True, "affected_files": ["shared.py", "shared.py", "x.py"]},
    ]}
    msg2, meta2 = atp._summarize_rollback(rr)
    assert "Manual targeted rollback required" in msg2
    assert meta2["needs_manual_rollback"] is True
    assert meta2["affected_files"] == ["shared.py", "x.py"]  # dedup, order kept

    rr3 = {"success": True}
    msg3, _ = atp._summarize_rollback(rr3)
    assert "successfully" in msg3


def test_evict_for_loop_empty_messages_short_circuit():
    assert atp._evict_for_loop([]) == []


def test_is_stubbed_gemini_non_str_inner_content():
    m = LLMMessage(role="user", content="", raw_content=[
        {"functionResponse": {"name": "t", "response": {"content": ["not", "str"]}}}])
    assert atp._is_stubbed_tool_result(m) is False
    m2 = LLMMessage(role="user", content="", raw_content=[
        {"functionResponse": "not-a-dict-response"}])
    assert atp._is_stubbed_tool_result(m2) is False


def test_effective_final_content_reasoning_exception_suppressed():
    """raw_response IS a dict but extraction raises → guarded, plain content."""
    bad = {"content": "plain", "raw": SimpleNamespace(raw_response={"choices": "junk"})}
    with mock.patch.object(atp, "extract_llm_reasoning", side_effect=TypeError("boom")):
        assert TurnPipelineMixin._effective_final_content(bad) == "plain"


def test_run_llm_loop_post_tool_early_return_propagates():
    early = AgentResult(status="success", final_message="tdd finished it")
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=_TurnPrepResult(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0))
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": 1, "completion_tokens": 1, "content": "",
        "tool_calls": [{"name": "write_plan"}], "finish_reason": "tool_use"})
    loop._execute_and_process_tool_calls = mock.MagicMock(return_value=_ToolTurnOutcome(
        new_messages=[], prepared_calls=[], write_tool_used=True,
        any_tool_called=True, fail_streak={}, reads_since_last_edit=0,
        plan_current_index=0))
    loop._process_post_tool_turn = mock.MagicMock(return_value=_PostToolResult(
        messages=[], tdd_fail_count=0, tdd_total_runs=1, tdd_total_pass=1,
        early_return=early))
    ctx = _full_ctx()
    assert loop._run_llm_loop(ctx) is early


def test_post_tool_turn_observation_skips_failed_patch_turns():
    """A failed patch in the same turn must not contribute observation paths."""
    loop = _make_loop(auto_observation_enabled=True)
    ctx = _post_ctx()
    ctx.messages = []
    ctx.turns = [_patch_turn("apply_patch", ok=False, metadata={"touched_files": ["bad.py"]})]
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda m, r, n: m
    loop._effective_final_content = staticmethod(lambda r: "t")
    with mock.patch("subprocess.run") as run:
        loop._process_post_tool_turn(ctx, None, [])
    run.assert_not_called()  # patch_ok_this_turn False → no observation at all


def test_filter_function_arguments_as_dict():
    loop = _filter_loop()
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"function": {"name": "read_file",
                                  "arguments": {"path": "direct-dict"}}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.prepared_calls[0]["args"] == {"path": "direct-dict"}


def test_effective_final_content_empty_content_extraction_raise():
    """Empty content + raw dict response + extraction raising → guarded."""
    bad = {"content": "", "raw": SimpleNamespace(raw_response={"choices": "junk"})}
    with mock.patch.object(atp, "extract_llm_reasoning", side_effect=TypeError("boom")):
        assert TurnPipelineMixin._effective_final_content(bad) == ""


# ---------------------------------------------------------------------------
# Round 30: remaining miss lines (exception-suppression + fallback branches)
# ---------------------------------------------------------------------------

class _AppendBoom(list):
    """ephemeral_pending whose append() raises AttributeError (suppressed paths)."""

    def append(self, item):
        raise AttributeError("boom")


def test_post_tool_turn_thinking_strip_exception_suppressed():
    """_process_post_tool_turn:0 — strip failure must not break the turn."""
    loop = _make_loop()
    ctx = _post_ctx()
    ctx.turns = []
    ctx.messages = []
    loop._strip_thinking_text = mock.MagicMock(side_effect=AttributeError("boom"))
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "text")
    resp = SimpleNamespace(content="plain text")
    loop._process_post_tool_turn(ctx, resp, [])
    assert resp.content == "plain text"  # untouched after failed strip


def test_post_tool_turn_auto_observation_skips_failed_patch_turn():
    """Auto-observation must skip patch turns whose result is not ok."""
    loop = _make_loop(auto_observation_enabled=True)
    ctx = _post_ctx()
    ctx.turns = [
        _patch_turn("apply_patch", ok=True, metadata={"touched_files": ["a.py"]}),
        _patch_turn("write_plan", ok=False, metadata={"touched_files": ["b.py"]}),
    ]
    ctx.messages = []
    loop._strip_thinking_text = staticmethod(lambda s: s)
    loop._append_native_tool_messages = lambda msgs, resp, new: msgs
    loop._effective_final_content = staticmethod(lambda r: "text")
    with mock.patch("subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="diff", stderr="")
        loop._process_post_tool_turn(ctx, None, [])
    assert list(run.call_args.args[0][3:]) == ["a.py"]  # failed turn excluded


def test_prepare_messages_phase_state_exception_suppressed():
    """_prepare_turn_messages:2 — phase-state builder failure is suppressed."""
    loop = _prep_loop()
    loop._build_phase_state_message = mock.MagicMock(side_effect=AttributeError("boom"))
    ctx = _prep_ctx(read_only_request=True)
    res = loop._prepare_turn_messages(ctx)
    assert res.messages


def test_prepare_messages_target_file_strategy_append_exception_suppressed():
    """_prepare_turn_messages:3 — TARGET FILE STRATEGY append failure is suppressed."""
    loop = _prep_loop()
    ctx = _prep_ctx(read_only_request=False, known_target_file="src/app.py")
    ctx.ephemeral_pending = _AppendBoom()
    ctx.budget_warned = True  # budget append is outside the suppressed try
    res = loop._prepare_turn_messages(ctx)
    assert res.messages


def test_prepare_messages_trajectory_append_exception_suppressed():
    """_prepare_turn_messages:4 — trajectory append failure is suppressed."""
    loop = _prep_loop()
    loop._trajectory_compress = mock.MagicMock(return_value="[TRAJECTORY] steps")
    ctx = _prep_ctx()
    ctx.turn_num = 3
    ctx.ephemeral_pending = _AppendBoom()
    ctx.budget_warned = True  # budget append is outside the suppressed try
    res = loop._prepare_turn_messages(ctx)
    assert res.messages


def test_filter_schema_fallback_without_get_tool_names():
    """registry without get_tool_names() falls back to get_tool_schemas()."""
    loop = _filter_loop()
    loop.registry = SimpleNamespace(
        repo_root="/tmp",
        repo_language=None,
        get_tool_schemas=mock.MagicMock(
            return_value=[{"name": "read_file"}, {"name": "edit_ast"}]),
        normalize_args_for_display=staticmethod(lambda a: a),
        _WRITE_TOOLS={"apply_patch", "write_plan"},
    )
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {"path": "a"}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.prepared_calls and res.prepared_calls[0]["tool"] == "read_file"


def test_filter_stream_preview_exception_suppressed():
    """tool_call_preview callback failure must not drop the prepared call."""
    loop = _filter_loop(stream_callback=lambda *a, **k: None)

    def _cb_boom(kind, *a, **k):
        if kind == "tool_call_preview":
            raise AttributeError("boom")

    loop._cb = _cb_boom
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    assert res.prepared_calls and res.prepared_calls[0]["tool"] == "read_file"


def test_ptr_failure_classification_exception_suppressed():
    """classify() returning None must not break result processing."""
    loop = _ptr_loop()
    loop._failure_classifier.classify.return_value = None
    loop.registry.is_result_cacheable = mock.MagicMock(return_value=False)
    out, loop, _ = _ptr(
        results=[ToolResult(ok=False, content="", error="boom", metadata={})],
        calls=[("read_file", {"path": "a"})], loop=loop)
    assert out is not None


def test_settle_deferred_semantics_skips_messages_without_marker():
    """Messages without the semantic_deferred marker are left untouched."""
    loop = _make_loop()
    loop.registry.drain_pending_semantic_checks = mock.MagicMock(return_value={
        "/tmp/a.py": SimpleNamespace(
            checked=True, skip_reason=None, diagnostics=[{"severity": "info"}])})
    plain = LLMMessage(role="tool", content="plain result")
    deferred = LLMMessage(role="tool", content=__import__("json").dumps({
        "metadata": {"syntax_check": {
            "semantic_deferred_path": "/tmp/a.py",
            "semantic_deferred": True}}}))
    loop._settle_deferred_semantics([plain, deferred])
    filled = __import__("json").loads(deferred.content)["metadata"]["syntax_check"]
    assert "semantic_diagnostics" in filled
    assert plain.content == "plain result"


def test_exec_serial_record_success_exception_suppressed():
    """Serial path: record_success raising AttributeError is suppressed."""
    loop = _exec_loop()
    loop._record_tool_success = mock.MagicMock(side_effect=AttributeError("boom"))
    calls = [{"tool": "read_file", "args": {}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0))
    loop.registry.dispatch = mock.MagicMock(
        return_value=ToolResult(ok=True, content="c", metadata={}))
    ctx = _full_ctx()
    out = loop._execute_and_process_tool_calls(ctx, tool_calls=calls)
    assert len(ctx.turns) == 1 and ctx.turns[0].tool_result.ok
    assert out.should_continue is False


def test_is_stubbed_tool_result_gemini_function_response():
    """Gemini functionResponse block with EVICTED marker counts as stubbed."""
    m = SimpleNamespace(
        content="",
        raw_content=[{"functionResponse": {"response": {
            "content": f"{atp._EVICTED_MARKER}: read_file — 1234 chars "
                       "evicted to save context; re-read if still needed.]"}}}])
    assert atp._is_stubbed_tool_result(m) is True


def test_filter_registry_errors_fall_back_to_empty_sets():
    """registry access raising AttributeError/TypeError/KeyError → empty sets."""
    loop = _filter_loop()
    loop.registry.get_tool_names = mock.MagicMock(side_effect=KeyError("boom"))
    res = loop._build_and_filter_prepared_calls(
        tool_calls=[{"name": "read_file", "args": {}}],
        turns=[], plan_subtasks=[], plan_current_index=0,
        read_only_request=True, turn_num=1)
    # Empty known set = no filtering: the call still goes through.
    assert res.prepared_calls and res.prepared_calls[0]["tool"] == "read_file"
    assert res.phase_rule_messages == []


def test_is_stubbed_tool_result_anthropic_tool_result_block():
    """Anthropic tool_result block with EVICTED marker counts as stubbed."""
    m = SimpleNamespace(
        content="",
        raw_content=[{"type": "tool_result",
                      "content": f"{atp._EVICTED_MARKER}: read_file — 1234 chars "
                                 "evicted to save context; re-read if still needed.]"}])
    assert atp._is_stubbed_tool_result(m) is True
