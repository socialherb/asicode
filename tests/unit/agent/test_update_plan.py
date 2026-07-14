"""Tests for the update_plan tool and the design-chat plan completion gate.

Covers:
  - plan_state.validate_plan / open_items / render_plan
  - AgentToolsMixin._tool_update_plan (validation + registry state)
  - DesignChatLoop plan gate: nudge on early exit, honest exit after nudges,
    no gate without a plan, ephemeral plan re-injection
"""
from types import SimpleNamespace

from external_llm.agent.design_chat_loop import DesignChatLoop
from external_llm.agent.plan_state import (
    diff_plans,
    open_items,
    render_plan,
    validate_plan,
)
from external_llm.agent.tool_handlers.agent_tools import AgentToolsMixin
from external_llm.client import LLMMessage

# ---------------------------------------------------------------------------
# plan_state
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan_normalizes(self):
        plan, err = validate_plan("build X", [
            {"title": "step 1"},
            {"title": "step 2", "status": "in_progress"},
        ])
        assert err == ""
        assert plan["goal"] == "build X"
        assert plan["items"][0]["status"] == "pending"
        assert plan["items"][1]["status"] == "in_progress"

    def test_empty_items_rejected(self):
        plan, err = validate_plan("g", [])
        assert plan is None and "non-empty" in err

    def test_invalid_status_rejected(self):
        plan, err = validate_plan("g", [{"title": "a", "status": "wip"}])
        assert plan is None and "invalid" in err

    def test_skipped_without_note_rejected(self):
        plan, err = validate_plan("g", [{"title": "a", "status": "skipped"}])
        assert plan is None and "reason is required" in err

    def test_blocked_with_note_accepted(self):
        plan, err = validate_plan("g", [{"title": "a", "status": "blocked", "note": "needs creds"}])
        assert err == ""
        assert plan["items"][0]["note"] == "needs creds"

    def test_missing_title_rejected(self):
        plan, err = validate_plan("g", [{"status": "pending"}])
        assert plan is None and "title is required" in err

    def test_too_many_items_rejected(self):
        """More than MAX_PLAN_ITEMS (50) is rejected."""
        items = [{"title": f"step {i}"} for i in range(51)]
        plan, err = validate_plan("g", items)
        assert plan is None and "exceeds the 50-item limit" in err

    def test_nondict_item_rejected(self):
        """An item that is not a dict is rejected."""
        plan, err = validate_plan("g", ["not_a_dict"])
        assert plan is None and "must be an object" in err


class TestOpenItemsAndRender:
    def test_open_items_filters_terminal(self):
        plan, _ = validate_plan("g", [
            {"title": "a", "status": "done"},
            {"title": "b", "status": "pending"},
            {"title": "c", "status": "blocked", "note": "x"},
            {"title": "d", "status": "in_progress"},
        ])
        titles = [it["title"] for it in open_items(plan)]
        assert titles == ["b", "d"]

    def test_open_items_none_plan(self):
        assert open_items(None) == []

    def test_render_contains_goal_and_progress(self):
        plan, _ = validate_plan("build X", [
            {"title": "a", "status": "done"},
            {"title": "b"},
        ])
        out = render_plan(plan)
        assert "Goal: build X" in out
        assert "[x] a" in out
        assert "[ ] b" in out
        assert "(1/2 done)" in out

    def test_render_with_note(self):
        """render_plan includes note text when present (line 132)."""
        plan, _ = validate_plan("g", [
            {"title": "a", "note": "waiting on review"},
        ])
        out = render_plan(plan)
        assert "[ ] a — waiting on review" in out


class TestDiffPlans:
    def test_first_call_reports_creation(self):
        plan, _ = validate_plan("g", [{"title": "a"}, {"title": "b"}])
        out = diff_plans(None, plan)
        assert "Plan created: 2 item(s)" in out
        assert "2 open" in out

    def test_status_transition_reported(self):
        prev, _ = validate_plan("g", [{"title": "a", "status": "in_progress"}, {"title": "b"}])
        new, _ = validate_plan("g", [{"title": "a", "status": "done"}, {"title": "b", "status": "in_progress"}])
        out = diff_plans(prev, new)
        assert "'a' in_progress→done" in out
        assert "'b' pending→in_progress" in out
        assert "(1/2 done, 1 open)" in out

    def test_added_and_removed_items(self):
        prev, _ = validate_plan("g", [{"title": "a"}])
        new, _ = validate_plan("g", [{"title": "b"}])
        out = diff_plans(prev, new)
        assert "+'b'" in out and "-'a'" in out

    def test_no_changes(self):
        plan, _ = validate_plan("g", [{"title": "a"}])
        assert "no status changes" in diff_plans(plan, plan)

    def test_long_title_truncated(self):
        long = "x" * 80
        plan, _ = validate_plan("g", [{"title": long}])
        out = diff_plans(None, plan)  # creation path doesn't show titles
        prev, _ = validate_plan("g", [{"title": long}])
        new, _ = validate_plan("g", [{"title": long, "status": "done"}])
        out = diff_plans(prev, new)
        assert long not in out and "…" in out


# ---------------------------------------------------------------------------
# _tool_update_plan handler
# ---------------------------------------------------------------------------

class _FakeHost(AgentToolsMixin):
    def _make_result(self, ok, content, error=None, metadata=None):
        return SimpleNamespace(ok=ok, content=content, error=error, metadata=metadata or {})


class TestUpdatePlanHandler:
    def test_stores_plan_on_registry(self):
        host = _FakeHost()
        res = host._tool_update_plan({"goal": "g", "items": [{"title": "a"}]})
        assert res.ok
        assert host.session_plan["goal"] == "g"
        assert res.metadata["open_items"] == 1

    def test_validation_error_returned(self):
        host = _FakeHost()
        res = host._tool_update_plan({"items": [{"title": "a", "status": "skipped"}]})
        assert not res.ok and "reason is required" in res.error
        assert getattr(host, "session_plan", None) is None

    def test_goal_preserved_across_calls(self):
        host = _FakeHost()
        host._tool_update_plan({"goal": "g", "items": [{"title": "a"}]})
        host._tool_update_plan({"items": [{"title": "a", "status": "done"}]})
        assert host.session_plan["goal"] == "g"

    def test_first_line_is_diff_summary(self):
        host = _FakeHost()
        res1 = host._tool_update_plan({"goal": "g", "items": [{"title": "a"}, {"title": "b"}]})
        assert res1.content.splitlines()[0].startswith("Plan created: 2 item(s)")
        res2 = host._tool_update_plan({"items": [
            {"title": "a", "status": "done"}, {"title": "b", "status": "in_progress"},
        ]})
        first = res2.content.splitlines()[0]
        assert "'a' pending→done" in first and "'b' pending→in_progress" in first
        assert res2.metadata["summary"] == first


# ---------------------------------------------------------------------------
# DesignChatLoop plan completion gate
# ---------------------------------------------------------------------------

def _resp(content):
    return SimpleNamespace(
        tool_calls=[], content=content, raw_response={},
        tokens_used=10, prompt_tokens=8, completion_tokens=2,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        provider="fake",
    )


class _FakeClient:
    """Scripted chat_with_tools: each entry is a () -> response callable."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []  # recorded message lists

    def chat_with_tools(self, messages, tools, model, **kwargs):
        self.calls.append(list(messages))
        return self.scripted.pop(0)()


class _FakeRegistry:
    def __init__(self):
        self.config = SimpleNamespace(cancel_event=None)
        self.session_plan = None
        self._repo_language = None

    @property
    def repo_language(self):
        return self._repo_language

    def get_tool_schemas(self, **kwargs):
        return []


def _make_plan(statuses):
    plan, err = validate_plan("big goal", [
        {"title": f"item {i}", "status": s,
         **({"note": "reason"} if s in ("skipped", "blocked") else {})}
        for i, s in enumerate(statuses)
    ])
    assert err == ""
    return plan


def _run_loop(client, registry):
    loop = DesignChatLoop(llm_client=client, registry=registry, model="test-model")
    return loop.respond([LLMMessage(role="user", content="do the big goal")])


class TestPlanGate:
    def test_no_plan_no_gate(self):
        registry = _FakeRegistry()
        client = _FakeClient([lambda: _resp("all done")])
        result = _run_loop(client, registry)
        assert result.content == "all done"
        assert len(client.calls) == 1

    def test_open_items_trigger_nudge_then_accept(self):
        registry = _FakeRegistry()

        def first():
            registry.session_plan = _make_plan(["done", "pending"])
            return _resp("wrapping up early")

        def second():
            registry.session_plan = _make_plan(["done", "blocked"])
            return _resp("final: item 1 blocked because of missing creds")

        client = _FakeClient([first, second])
        result = _run_loop(client, registry)

        assert result.content.startswith("final:")
        assert len(client.calls) == 2
        # The nudge message was injected into the second call
        assert any("unresolved items" in (m.content or "") for m in client.calls[1])
        # The nudge forbids the narrative preamble and directs action
        assert any("이어서 진행" in (m.content or "") for m in client.calls[1])

    def test_nudges_exhausted_appends_honest_warning(self):
        registry = _FakeRegistry()

        def stop():
            registry.session_plan = _make_plan(["pending", "pending"])
            return _resp("stopping here")

        # MAX_NUDGES == 1: initial call + exactly 1 nudge before honest exit
        client = _FakeClient([stop, stop])
        result = _run_loop(client, registry)

        assert len(client.calls) == 2  # initial + 1 nudge
        assert "stopping here" in result.content
        assert "Unresolved plan items" in result.content
        assert "item 0" in result.content and "item 1" in result.content

    def test_all_terminal_exits_clean(self):
        registry = _FakeRegistry()

        def finish():
            registry.session_plan = _make_plan(["done", "skipped"])
            return _resp("done; item 1 skipped because reason")

        client = _FakeClient([finish])
        result = _run_loop(client, registry)
        assert len(client.calls) == 1
        assert "Unresolved plan items" not in result.content

    def test_stale_plan_reset_at_turn_start(self):
        registry = _FakeRegistry()
        registry.session_plan = _make_plan(["pending"])  # leftover from a previous turn
        client = _FakeClient([lambda: _resp("small answer")])
        result = _run_loop(client, registry)
        # Reset at turn start → no gate, no plan injection
        assert result.content == "small answer"
        assert len(client.calls) == 1
        assert "Current work plan" not in client.calls[0][-1].content
