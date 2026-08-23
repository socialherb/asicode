"""F3: tool_call_preview / tool_call SSE payloads must carry the agent identity.

The frontend keys its pending tool-card map by (agent, turn, tool) and renders
a per-agent badge on cards. Without ``agent_id`` in these payloads every event
fell back to ``"main"`` — parallel sub-agents (each with turns restarting at 1)
collided on the same ``T{turn}:{tool}`` keys, so a sub-agent's result could
resolve the main lane's (or another sub-agent's) pending card.

The payload sites are the two emission points in the turn pipeline:
  - ``_build_and_filter_prepared_calls``  → ``tool_call_preview``
  - ``_process_tool_results`` (normal + read-only early-finish) → ``tool_call``
"""

from __future__ import annotations

from unittest import mock

from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin
from external_llm.agent.tool_registry import ToolResult


def _make_loop(agent_id: str = "main"):
    """Minimal TurnPipelineMixin host (mirrors test_agent_turn_pipeline_audit_fixes)."""
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    events: list[tuple[str, dict]] = []
    loop._cb = lambda name, data=None: events.append((name, data or {}))
    loop.config = mock.MagicMock()
    loop.config.stream_callback = True
    loop.config.cancel_event = None  # MagicMock truthiness would cancel
    loop.config.agent_id = agent_id
    loop.config.scoped_verification = False  # skip test-impact index invalidation
    loop.registry = mock.MagicMock()
    loop.registry.repo_language = None
    loop.registry.get_tool_names.return_value = {"grep"}
    loop.registry.normalize_args_for_display.side_effect = lambda a: a
    loop.registry.is_result_cacheable.return_value = False
    loop._tool_retry_counter = {}
    loop.performance_collector = mock.MagicMock()
    loop._advance_phase_after_success = mock.MagicMock()
    loop._build_tool_result_message = mock.MagicMock(return_value="<msg>")
    return loop, events


# ---------------------------------------------------------------------------
# tool_call_preview
# ---------------------------------------------------------------------------


def test_preview_payload_carries_main_agent_id():
    loop, events = _make_loop("main")
    loop._build_and_filter_prepared_calls(
        tool_calls=[{"id": "c1", "name": "grep", "args": {"pattern": "x"}}],
        turns=[],
        plan_subtasks=[],
        plan_current_index=0,
        read_only_request=False,
        turn_num=2,
    )
    previews = [d for n, d in events if n == "tool_call_preview"]
    assert previews, "tool_call_preview must be emitted"
    assert previews[0]["agent_id"] == "main"
    assert previews[0]["turn"] == 2 and previews[0]["tool"] == "grep"


def test_preview_payload_carries_subagent_id():
    loop, events = _make_loop("sub_3")
    loop._build_and_filter_prepared_calls(
        tool_calls=[{"id": "c1", "name": "grep", "args": {"pattern": "x"}}],
        turns=[],
        plan_subtasks=[],
        plan_current_index=0,
        read_only_request=False,
        turn_num=1,
    )
    previews = [d for n, d in events if n == "tool_call_preview"]
    assert previews and previews[0]["agent_id"] == "sub_3"


# ---------------------------------------------------------------------------
# tool_call (normal path)
# ---------------------------------------------------------------------------


def _run_process_tool_results(loop):
    return loop._process_tool_results(
        results=[ToolResult(ok=True, content="hit", error="")],
        prepared_calls=[{"tool": "grep", "args": {"pattern": "x"}, "call_id": "c1"}],
        new_messages=[],
        write_tool_used=False,
        reads_since_last_edit=0,
        fail_streak={},
        fail_streak_threshold=3,
        session_key="sk",
        write_tools=set(),
        read_only_request=False,
        request="r",
        session_id="s",
        git_state=None,
        turn_num=2,
        turns=[],
    )


def test_tool_call_payload_carries_agent_id():
    loop, events = _make_loop("sub_7")
    _run_process_tool_results(loop)
    calls = [d for n, d in events if n == "tool_call"]
    assert calls, "tool_call must be emitted"
    assert calls[0]["agent_id"] == "sub_7"
    assert calls[0]["result"]["ok"] is True
    assert calls[0]["result"]["content"] == "hit"


def test_tool_call_early_finish_payload_carries_agent_id():
    """The read-only early-finish path (single tool_call before AgentResult)."""
    loop, events = _make_loop("main")
    loop._try_readonly_early_finish = mock.MagicMock(return_value=None)  # no early finish
    _run_process_tool_results(loop)
    calls = [d for n, d in events if n == "tool_call"]
    assert calls and calls[0]["agent_id"] == "main"


def test_payload_agent_id_never_falls_back_to_constant():
    """The payload must read the config — not a hard-coded literal."""
    loop, events = _make_loop("lane_b")
    _run_process_tool_results(loop)
    calls = [d for n, d in events if n == "tool_call"]
    assert calls and calls[0]["agent_id"] == "lane_b"
    loop2, events2 = _make_loop("main")
    _run_process_tool_results(loop2)
    calls2 = [d for n, d in events2 if n == "tool_call"]
    assert calls2 and calls2[0]["agent_id"] == "main"
