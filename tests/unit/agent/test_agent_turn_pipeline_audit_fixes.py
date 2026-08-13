"""Agent turn pipeline audit fixes (P0-P3).

P0 — provider responses whose token fields are all ``None`` (Gemini
safety-block ``candidates==[]``, OpenAI-compatible ``choices==[]``) used to
TypeError in the main-loop token accumulation — the ONLY one of four identical
idioms without ``or 0`` guards — which routed through ``_handle_loop_error``
and ROLLED BACK every applied patch of the session.

P1 — the parallel dispatch branch (2+ tool calls in one turn) skipped the
serial branch's tool accounting (``_record_tool_success``/``_failure``), the
false-success gate flag (``any_tool_called``), and the apply_patch
auto-repair / tolerant edit_blocks recovery ladder.

P2/P3 — dead params removed from ``_process_tool_results``;
``metadata["tokens"]`` unified behind ``_token_metadata`` (single 11-key set
on every exit path).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from external_llm.agent import agent_turn_pipeline as atp
from external_llm.agent.agent_loop_types import (
    TurnContext,
    _PreparedCallsResult,
    _ResultsProcessingOutcome,
)
from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin, _token_metadata
from external_llm.agent.tool_registry import ToolResult
from external_llm.client import LLMMessage

# ---------------------------------------------------------------------------
# P0: None token fields must not crash the turn loop
# ---------------------------------------------------------------------------


def _make_loop():
    """Minimal TurnPipelineMixin with mocked host dependencies."""
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    loop.config = mock.MagicMock()
    loop.config.max_turns = 5
    loop.config.model_name = "claude-test"
    loop.config.make_token_callback.return_value = None
    loop.llm_client = mock.MagicMock()
    loop.llm_client.get_provider_name.return_value = "anthropic"
    loop.llm_client.base_url = ""
    loop.registry = mock.MagicMock()
    loop.registry.applied_patches = []
    loop._patch_fail_count = 0
    loop._cb = mock.MagicMock()
    return loop


def _make_ctx():
    """Minimal TurnContext; remaining fields are initialized by _run_llm_loop."""
    ctx = TurnContext.__new__(TurnContext)
    ctx.route = None
    ctx.request = "test"
    ctx.context = None
    ctx.tier = 1
    ctx.read_only_request = True
    ctx.has_native_tools = False
    return ctx


def test_none_token_fields_do_not_crash_turn_loop():
    """P0 regression: all-None token fields must not TypeError into a rollback.

    Before the fix the main-loop accumulation did ``ctx.total_prompt_tokens +=
    None`` -> TypeError -> ``_handle_loop_error`` -> rollback of applied
    patches. After the fix the loop completes normally (text_reply) with the
    None fields coerced to 0.
    """
    loop = _make_loop()
    loop._build_initial_messages = mock.MagicMock(return_value=[])
    loop._prepare_turn_messages = mock.MagicMock(return_value=SimpleNamespace(
        messages=[], budget_warned=False, goal_reminder_injected=0,
        search_first_hint_done=False, reads_since_last_edit=0,
    ))
    final_out = SimpleNamespace(
        result=SimpleNamespace(status="text_reply"), nudge_message=None)
    loop._handle_final_answer_turn = mock.MagicMock(return_value=final_out)
    loop._handle_loop_error = mock.MagicMock(
        side_effect=AssertionError("_handle_loop_error must not run for None tokens"))
    loop._llm_call_with_tools = mock.MagicMock(return_value={
        "prompt_tokens": None, "completion_tokens": None, "tokens_used": None,
        "cache_read_input_tokens": None, "cache_creation_input_tokens": None,
        "content": "done", "tool_calls": [], "finish_reason": "stop",
    })

    ctx = _make_ctx()
    result = loop._run_llm_loop(ctx)

    assert result.status == "text_reply"
    # None coerced to 0 — counters stay integer-typed, nothing crashes.
    assert ctx.total_prompt_tokens == 0
    assert ctx.total_completion_tokens == 0
    assert ctx.total_cache_read_tokens == 0
    assert ctx.total_cache_creation_tokens == 0
    assert ctx.last_call_prompt_tokens == 0
    loop._handle_loop_error.assert_not_called()
    loop._handle_final_answer_turn.assert_called_once()


# ---------------------------------------------------------------------------
# P1: parallel branch parity
# ---------------------------------------------------------------------------


def _exec_ctx():
    ctx = TurnContext.__new__(TurnContext)
    ctx.plan_current_index = 0
    ctx.any_tool_called = False
    ctx.write_tool_used = False
    ctx.reads_since_last_edit = 0
    ctx.fail_streak = {}
    ctx.plan_subtasks = []
    ctx.read_only_request = True
    ctx.turn_num = 1
    ctx.turns = []
    ctx.session_id = "s1"
    ctx.git_state = None
    ctx.request = "test"
    ctx.plan = None
    ctx.write_tools = {"apply_patch", "write_plan"}
    return ctx


def test_parallel_branch_wires_accounting_and_recovery():
    """P1: a 2+ tool-call turn must set any_tool_called, record success/failure,
    and run the apply_patch recovery ladder — previously skipped entirely."""
    loop = _make_loop()
    loop.config.parallel_tool_execution_enabled = True
    loop.config.tolerant_patch_mode = False
    loop._record_tool_success = mock.MagicMock()
    loop._record_tool_failure = mock.MagicMock()
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)
    loop._process_tool_results = mock.MagicMock(return_value=_ResultsProcessingOutcome(
        new_messages=[], write_tool_used=False, reads_since_last_edit=0,
        noop_confirmed=False, fail_streak={},
    ))
    loop._settle_deferred_semantics = mock.MagicMock()

    calls = [
        {"tool": "read_file", "args": {"path": "a.txt"}, "call_id": "c1"},
        {"tool": "apply_patch", "args": {"patch": "x"}, "call_id": "c2"},
    ]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0,
    ))
    loop.registry.dispatch_parallel = mock.MagicMock(return_value=[
        ToolResult(ok=True, content="file content"),
        ToolResult(ok=False, content="", error="patch failed"),
    ])

    ctx = _exec_ctx()
    with mock.patch.object(atp, "_log_parallel_write_failures") as _log_pwf:
        outcome = loop._execute_and_process_tool_calls(
            ctx, tool_calls=calls,
        )

    assert outcome.any_tool_called is True
    loop._record_tool_success.assert_called_once_with("read_file", {"path": "a.txt"})
    loop._record_tool_failure.assert_called_once_with("apply_patch", {"patch": "x"})
    _log_pwf.assert_called_once()
    # Recovery ladder ran for the parallel apply_patch failure.
    assert loop._patch_fail_count == 1
    # Every dispatched call landed as an AgentTurn.
    assert [t.tool_name for t in ctx.turns] == ["read_file", "apply_patch"]


def test_serial_branch_uses_same_recovery_helper():
    """P1: serial branch delegates to _post_dispatch_patch_recovery (parity pin)."""
    loop = _make_loop()
    loop.config.parallel_tool_execution_enabled = False
    loop.config.tolerant_patch_mode = False
    loop._record_tool_success = mock.MagicMock()
    loop._record_tool_failure = mock.MagicMock()
    loop._process_tool_results = mock.MagicMock(return_value=_ResultsProcessingOutcome(
        new_messages=[], write_tool_used=False, reads_since_last_edit=0,
        noop_confirmed=False, fail_streak={},
    ))
    loop._settle_deferred_semantics = mock.MagicMock()
    calls = [{"tool": "apply_patch", "args": {"patch": "x"}, "call_id": "c1"}]
    loop._build_and_filter_prepared_calls = mock.MagicMock(return_value=_PreparedCallsResult(
        prepared_calls=calls, phase_rule_messages=[], plan_current_index=0,
    ))
    loop.registry.dispatch = mock.MagicMock(
        return_value=ToolResult(ok=False, content="", error="boom"))
    loop._auto_repair_apply_patch_args = mock.MagicMock(return_value=None)

    ctx = _exec_ctx()
    outcome = loop._execute_and_process_tool_calls(
        ctx, tool_calls=calls,
    )

    assert outcome.any_tool_called is True
    assert loop._patch_fail_count == 1
    loop._record_tool_failure.assert_called_once_with("apply_patch", {"patch": "x"})


def test_post_dispatch_patch_recovery_auto_repair():
    """P1: failed apply_patch with repair args retries once and records metadata."""
    loop = _make_loop()
    ok_result = ToolResult(ok=True, content="applied")
    loop._auto_repair_apply_patch_args = mock.MagicMock(
        return_value={"patch": "fixed", "path": "a.txt"})
    loop.registry.dispatch = mock.MagicMock(return_value=ok_result)

    out = loop._post_dispatch_patch_recovery(
        "apply_patch", {"patch": "broken"}, ToolResult(ok=False, error="orig", content=""),
    )

    assert out is ok_result
    assert out.metadata["auto_repair"] == {
        "attempted": True, "kind": "patch_format_fix",
        "original_error": "orig", "success": True,
    }
    assert loop._patch_fail_count == 0


def test_post_dispatch_patch_recovery_non_patch_passthrough():
    """P1: non-apply_patch results pass through untouched (no dispatch, no count)."""
    loop = _make_loop()
    loop._auto_repair_apply_patch_args = mock.MagicMock()
    r = ToolResult(ok=False, error="nope", content="")
    assert loop._post_dispatch_patch_recovery("read_file", {"path": "a.txt"}, r) is r
    loop._auto_repair_apply_patch_args.assert_not_called()
    assert loop._patch_fail_count == 0


# ---------------------------------------------------------------------------
# P2: _token_metadata SSOT
# ---------------------------------------------------------------------------


def _token_ctx(**over):
    base = {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_cache_read_tokens": 500,
        "total_cache_creation_tokens": 100,
        "last_call_prompt_tokens": 20,
        "last_call_completion_tokens": 10,
        "provider_name": "anthropic",
        "model_name": "claude-test",
        "base_url": "",
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_token_metadata_uniform_field_set():
    """P2: every exit path now emits the SAME 11-key token field set."""
    m = _token_metadata(_token_ctx())
    assert set(m) == {
        "prompt", "completion", "total", "cost_usd", "cache_adjusted_cost_usd",
        "cache_read_tokens", "cache_creation_tokens", "cache_hit_ratio",
        "last_call_prompt", "last_call_completion", "provider",
    }
    assert m["total"] == 150
    assert m["provider"] == "anthropic"
    # cache_hit_ratio matches the standalone helper on the same ctx.
    assert m["cache_hit_ratio"] == atp._cache_hit_ratio(_token_ctx())


def test_token_metadata_zero_tokens_ratio_zero():
    """P2: empty-run ctx (all zeros) yields ratio 0.0 without crashing."""
    m = _token_metadata(_token_ctx(
        total_prompt_tokens=0, total_completion_tokens=0,
        total_cache_read_tokens=0, total_cache_creation_tokens=0,
        last_call_prompt_tokens=0, last_call_completion_tokens=0,
    ))
    assert m["total"] == 0
    assert m["cache_hit_ratio"] == 0.0


# ---------------------------------------------------------------------------
# P3: _build_tool_name_map no longer self-references Gemini functionCall names
# ---------------------------------------------------------------------------


def test_build_tool_name_map_skips_gemini_self_reference():
    """P3: Gemini functionCall parts have no id — a name->name entry would
    only pollute the id->name map, so it must not appear."""
    m = LLMMessage(
        role="assistant", content="",
        raw_content=[
            {"text": "hi"},
            {"functionCall": {"name": "read_file", "args": {}}},
            {"type": "tool_use", "id": "tu_1", "name": "grep"},
        ],
    )
    assert atp._build_tool_name_map([m]) == {"tu_1": "grep"}
