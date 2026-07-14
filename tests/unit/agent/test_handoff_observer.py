"""Unit tests for handoff_observer.py — 100% coverage."""
from __future__ import annotations

import pytest

from external_llm.agent.handoff_observer import (
    _run_events,
    _truncate,
    compute_intent_diff,
    get_events,
    log_auto_correction,
    log_developer_request,
    log_developer_result,
    log_intent_diff,
    log_planner_handoff,
    log_symbol_resolution,
    reset_events,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_events():
    """Reset module-level events before each test."""
    reset_events()
    yield


# ── reset_events / get_events ────────────────────────────────────────────────

class TestEventLifecycle:
    """Coverage for lines 26, 31 — reset_events and get_events."""

    def test_initial_empty(self):
        assert get_events() == []

    def test_reset_clears(self):
        log_planner_handoff("op1", "edit", "a.py", "foo", "orig", "enrich", "final")
        assert len(get_events()) == 1
        reset_events()
        assert get_events() == []

    def test_get_returns_copy(self):
        log_planner_handoff("op1", "edit", "a.py", "foo", "orig", "enrich", "final")
        events = get_events()
        events.clear()
        assert len(get_events()) == 1  # original list unaffected


# ── log_planner_handoff ──────────────────────────────────────────────────────

class TestLogPlannerHandoff:
    def test_minimal(self):
        log_planner_handoff("op1", "edit", "a.py", "foo", "orig", "enrich", "final")
        assert len(_run_events) == 1
        e = _run_events[0]
        assert e["event"] == "planner_handoff"
        assert e["op_id"] == "op1"
        assert e["change_spec_summary"] == []
        assert e["metadata"] == {}

    def test_with_change_spec_and_metadata(self):
        log_planner_handoff(
            "op2", "add", "b.py", "Bar",
            "orig", "enrich", "final",
            change_spec_summary=["add field x"],
            metadata={"score": 0.9},
        )
        e = _run_events[0]
        assert e["change_spec_summary"] == ["add field x"]
        assert e["metadata"] == {"score": 0.9}


# ── log_developer_request ────────────────────────────────────────────────────

class TestLogDeveloperRequest:
    """Coverage for lines 92-108."""

    def test_minimal(self):
        log_developer_request("op1", "intent", "handler", "mode", True, True)
        e = _run_events[0]
        assert e["event"] == "developer_request_built"
        assert e["model"] == ""
        assert e["context_summary"] == {}

    def test_with_context(self):
        log_developer_request(
            "op1", "intent", "handler", "mode", True, False,
            model="gpt4", provider="openai",
            context_summary={"file_lines": 10, "symbol_lines": 5, "prompt_tokens_estimate": 200},
        )
        e = _run_events[0]
        assert e["model"] == "gpt4"
        assert e["symbol_exists"] is False
        assert e["context_summary"]["file_lines"] == 10


# ── log_developer_result ─────────────────────────────────────────────────────

class TestLogDeveloperResult:
    """Coverage for lines 150-166 — success/created_file/failure branches."""

    def test_success_no_create(self):
        """Line 156-157: status=success, created_file=False."""
        log_developer_result("op1", "success", "apply_patch", changed_lines=5)
        e = _run_events[0]
        assert e["status"] == "success"
        assert e["changed_lines"] == 5
        assert e["created_file"] is False

    def test_success_created_file(self):
        """Line 151-154: status=success, created_file=True."""
        log_developer_result("op1", "success", "create_file", content_length=300, created_file=True)
        e = _run_events[0]
        assert e["status"] == "success"
        assert e["content_length"] == 300
        assert e["created_file"] is True

    def test_failure(self):
        """Line 161-166: status=failure."""
        log_developer_result(
            "op1", "parse_error", "apply_patch",
            failure_class="ParseError",
            failure_reason="unexpected indent",
            retry_count=2,
        )
        e = _run_events[0]
        assert e["status"] == "parse_error"
        assert e["failure"]["class"] == "ParseError"
        assert e["failure"]["retry_count"] == 2


# ── log_symbol_resolution ────────────────────────────────────────────────────

class TestLogSymbolResolution:
    def test_exists_true(self):
        log_symbol_resolution("op1", "Foo.bar", True, "ast", "direct_edit")
        e = _run_events[0]
        assert e["event"] == "symbol_resolution"
        assert e["exists"] is True

    def test_exists_false(self):
        log_symbol_resolution("op1", "Baz", False, "grep", "search_fallback")
        e = _run_events[0]
        assert e["exists"] is False
        assert e["action"] == "search_fallback"


# ── log_intent_diff ──────────────────────────────────────────────────────────

class TestLogIntentDiff:
    """Coverage for lines 204-219 — event construction and lost_items branching."""

    def test_with_lost_items(self):
        """Line 213-216: lost_items present."""
        log_intent_diff("op1", lost_items=["add_field"], added_items=[], preserved=["len("])
        e = _run_events[0]
        assert e["lost_items"] == ["add_field"]
        assert e["preserved"] == ["len("]

    def test_without_lost_items(self):
        """Line 218-222: no lost_items."""
        log_intent_diff("op2", lost_items=[], added_items=["verify_password"], preserved=["add_validation"])
        e = _run_events[0]
        assert e["lost_items"] == []
        assert e["added_items"] == ["verify_password"]


# ── log_auto_correction ──────────────────────────────────────────────────────

class TestLogAutoCorrection:
    def test_action_keep(self):
        log_auto_correction("op1", "edit", "foo", "edit", "foo", action="keep")
        e = _run_events[0]
        assert e["action"] == "keep"

    def test_action_correct(self):
        log_auto_correction(
            "op1", "edit", "foo", "add", "bar", "baz",
            action="correct", rationale="symbol not found",
            confidence=0.85, resolution_facts={"found_in": "b.py"},
        )
        e = _run_events[0]
        assert e["action"] == "correct"
        assert e["corrected_symbol"] == "bar"
        assert e["resolution_facts"] == {"found_in": "b.py"}


# ── _truncate ────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_empty_string(self):
        assert _truncate("") == "(empty)"

    def test_short_string(self):
        assert _truncate("hello") == "hello"

    def test_long_string(self):
        s = "x" * 200
        assert len(_truncate(s, 50)) == 50 + 1  # 50 chars + …

    def test_newlines_replaced(self):
        assert _truncate("a\nb\nc") == "a ↵ b ↵ c"


# ── compute_intent_diff ──────────────────────────────────────────────────────

class TestComputeIntentDiff:
    """Coverage for lines 290-316 — keyword-based intent diff computation."""

    def test_no_changes(self):
        text = "add_field COMPLETENESS RULE"
        result = compute_intent_diff(text, text)
        assert result["lost"] == []
        assert result["added"] == []
        assert "add_field" in result["preserved"]

    def test_lost_marker(self):
        enriched = "add_field COMPLETENESS RULE"
        effective = "add_field"  # missing COMPLETENESS RULE
        result = compute_intent_diff(enriched, effective)
        assert "COMPLETENESS RULE" in result["lost"]
        assert "add_field" in result["preserved"]

    def test_added_marker(self):
        enriched = "verify_password"
        effective = "verify_password len("  # len( is added
        result = compute_intent_diff(enriched, effective)
        assert "len(" in result["added"]
        assert "verify_password" in result["preserved"]

    def test_all_empty(self):
        result = compute_intent_diff("", "")
        assert result == {"lost": [], "added": [], "preserved": []}

    def test_multiple_lost_and_preserved(self):
        enriched = "add_field add_validation ⚠️ MANDATORY"
        effective = "add_field"
        result = compute_intent_diff(enriched, effective)
        assert result["lost"] == ["⚠️ MANDATORY", "add_validation"]
        assert result["preserved"] == ["add_field"]
        assert result["added"] == []


# ── Bounded accumulator (memory-leak guard) ──────────────────────────────────

class TestRunEventsBounded:
    """Regression guard: _run_events must be bounded.

    The log_* writers fire on every editor operation in the shared editor lane
    (including the long-lived webapp server path), but reset_events() and
    get_events() have NO callers — an unbounded list leaked every operation's
    handoff event for the whole process lifetime. The deque cap bounds memory
    while preserving recent-event inspection semantics.
    """

    def test_run_events_is_bounded_deque(self):
        from collections import deque as _deque
        from external_llm.agent.handoff_observer import _RUN_EVENTS_MAX

        assert isinstance(_run_events, _deque)
        assert _run_events.maxlen == _RUN_EVENTS_MAX

    def test_overflow_evicts_oldest(self):
        """Appending beyond the cap drops the oldest, never grows unbounded."""
        from external_llm.agent.handoff_observer import _RUN_EVENTS_MAX

        cap = _RUN_EVENTS_MAX
        # Fill exactly to the cap, then overflow by 10.
        for i in range(cap + 10):
            log_symbol_resolution(f"op{i}", "sym", True, "direct", "execute")
        # Bounded: length never exceeds the cap.
        assert len(_run_events) == cap
        # Oldest 10 were evicted; the first retained is the 11th call (op10).
        assert _run_events[0]["op_id"] == "op10"
        # The newest is retained.
        assert _run_events[-1]["op_id"] == f"op{cap + 9}"

    def test_get_events_returns_capped_recent_in_order(self):
        """get_events() yields the most-recent `maxlen` events, oldest→newest."""
        from external_llm.agent.handoff_observer import _RUN_EVENTS_MAX

        cap = _RUN_EVENTS_MAX
        for i in range(cap + 5):
            log_symbol_resolution(f"op{i}", "sym", True, "direct", "execute")
        events = get_events()
        assert isinstance(events, list)
        assert len(events) == cap
        # Order preserved; first retained is op5 (the 6th call after 5 evicted).
        assert events[0]["op_id"] == "op5"
        assert events[-1]["op_id"] == f"op{cap + 4}"
