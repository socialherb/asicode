"""_prepare_turn_messages must pass the SAME tool-schema variant to
_evict_for_loop that the API request uses (agent_loop passes
lang_filter=repo_language). Before the fix it called get_tool_schemas() with
no filter, so a non-Python repo's occupancy budget accounted for masked
python-only tools the request never sends — budget and wire disagreed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from external_llm.agent import agent_turn_pipeline as atp
from external_llm.agent.agent_turn_pipeline import TurnPipelineMixin
from external_llm.client import LLMMessage


def _make_loop():
    loop = TurnPipelineMixin.__new__(TurnPipelineMixin)
    loop.config = mock.MagicMock()
    loop.config.max_turns = 5
    loop.config.cancel_event = None
    loop.config.message_queue = None
    loop.registry = mock.MagicMock()
    loop.registry.get_tool_schemas.return_value = [{"name": "bash"}]
    loop.registry.repo_language = None
    loop._cb = mock.MagicMock()
    loop._trim_context = lambda msgs: msgs
    loop._build_tool_hint = lambda: ""
    return loop


def _make_ctx():
    return SimpleNamespace(
        ephemeral_pending=[], messages=[LLMMessage(role="user", content="hi")],
        turn_num=1, search_first_hint_done=False, target_keywords=[],
        known_target_file=None, reads_since_last_edit=0, goal_reminder_injected=0,
        request="test", plan_subtasks=[], plan_current_index=0,
        read_only_request=False, is_local_model=False, turns=[],
        budget_warned=False, model_name="claude-test",
    )


def test_prepare_turn_messages_forwards_lang_filter(monkeypatch):
    loop = _make_loop()
    captured = {}

    def _fake_evict(msgs, model="", tool_schemas=None, base_url=None):
        captured["tool_schemas"] = tool_schemas
        return msgs

    monkeypatch.setattr(atp, "_evict_for_loop", _fake_evict)
    loop._prepare_turn_messages(_make_ctx())
    loop.registry.get_tool_schemas.assert_called_once_with(lang_filter=None)
    assert captured["tool_schemas"] == [{"name": "bash"}]


def test_prepare_turn_messages_forwards_non_python_lang_filter(monkeypatch):
    loop = _make_loop()
    loop.registry.repo_language = "typescript"  # non-Python → masked variant
    captured = {}

    def _fake_evict(msgs, model="", tool_schemas=None, base_url=None):
        captured["tool_schemas"] = tool_schemas
        return msgs

    monkeypatch.setattr(atp, "_evict_for_loop", _fake_evict)
    loop._prepare_turn_messages(_make_ctx())
    loop.registry.get_tool_schemas.assert_called_once_with(lang_filter="typescript")
    assert captured["tool_schemas"] == [{"name": "bash"}]
