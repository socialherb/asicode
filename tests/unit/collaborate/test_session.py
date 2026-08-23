"""
Tests for ClaudeSession event handling and SessionEvent/SessionResult.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from external_llm.repl.collaborate import CollaborationVerdict
from external_llm.repl.collaborate.claude_session import (
    _MAX_RETAINED_EVENTS,
    ClaudeSession,
    SessionEvent,
    SessionResult,
)


class TestSessionEvent:
    """Verify session event dataclass."""

    def test_default_creation(self):
        e = SessionEvent()
        assert e.type == "unknown"
        assert e.content == ""
        assert e.metadata == {}
        assert e.timestamp > 0

    def test_custom_event(self):
        e = SessionEvent(
            type="tool_call",
            content="read_file",
            metadata={"file": "test.py"},
        )
        assert e.type == "tool_call"
        assert e.content == "read_file"
        assert e.metadata["file"] == "test.py"


class TestSessionResult:
    """Verify session result dataclass."""

    def test_default_creation(self):
        v = CollaborationVerdict()
        r = SessionResult(verdict=v)
        assert r.verdict is v
        assert r.events == []
        assert r.tool_calls_count == 0
        assert r.duration_seconds == 0.0
        assert r.error is None

    def test_with_events(self):
        events = [SessionEvent(type="text", content="Hello")]
        r = SessionResult(
            verdict=CollaborationVerdict(status="success"),
            events=events,
            tool_calls_count=5,
            duration_seconds=12.34,
        )
        assert len(r.events) == 1
        assert r.tool_calls_count == 5
        assert r.duration_seconds == 12.34


class TestResultMessageHandling:
    """Verify _handle_result_message accounting and error-preservation."""

    def _session(self) -> ClaudeSession:
        # __init__ does NOT connect — safe to instantiate without async context.
        s = ClaudeSession()
        s._events = []
        return s

    def test_total_tokens_includes_cache_tokens(self):
        """Anthropic-backed usage: input_tokens EXCLUDES cache tokens.

        A cache-heavy session must count cache_creation + cache_read on top of
        input + output, or total_tokens massively underreports consumption.
        """
        s = self._session()
        msg = SimpleNamespace(
            usage={
                "input_tokens": 100,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 8000,
                "output_tokens": 200,
            },
            total_cost_usd=0.01,
            structured_output=None,
            result="",
            is_error=False,
            errors=[],
        )
        s._handle_result_message(msg)
        # 100 + 5000 + 8000 + 200 = 13300, NOT 300 (input+output only)
        assert s._last_total_tokens == 13300

    def test_total_tokens_omits_missing_cache_keys(self):
        """Usage without cache keys must fall back to input+output (no KeyError)."""
        s = self._session()
        msg = SimpleNamespace(
            usage={"input_tokens": 100, "output_tokens": 200},
            total_cost_usd=0.0,
            structured_output=None,
            result="",
            is_error=False,
            errors=[],
        )
        s._handle_result_message(msg)
        assert s._last_total_tokens == 300

    def test_structured_output_attaches_late_error_note(self):
        """A structured verdict arriving WITH a late error must preserve it.

        Parity with the structured_candidate salvage path: result_error lands
        in metadata instead of being silently dropped on the success path.
        """
        s = self._session()
        msg = SimpleNamespace(
            usage=None,
            total_cost_usd=0.0,
            structured_output={
                "status": "success",
                "summary": "done",
                "details": "analysis complete",
            },
            result="",
            is_error=True,
            errors=["Reached maximum budget"],
        )
        verdict = s._handle_result_message(msg)
        assert verdict.status == "success"
        assert verdict.metadata.get("result_error") == "Reached maximum budget"

    def test_structured_output_no_error_keeps_clean_metadata(self):
        """No errors → no result_error key injected (regression guard)."""
        s = self._session()
        msg = SimpleNamespace(
            usage=None,
            total_cost_usd=0.0,
            structured_output={
                "status": "success",
                "summary": "done",
                "details": "ok",
            },
            result="",
            is_error=False,
            errors=[],
        )
        verdict = s._handle_result_message(msg)
        assert verdict.status == "success"
        assert "result_error" not in verdict.metadata


class TestStreamEventHandling:
    """Verify _handle_stream_event robustness for malformed event payloads."""

    def test_none_event_attribute_does_not_raise(self):
        """An .event attribute that is None must not crash the stream loop.

        Previously ``ev = getattr(event, "event", event)`` yielded None when
        the attribute existed but was None, then ``ev.get(...)`` raised
        AttributeError and aborted the entire query.
        """
        s = ClaudeSession()
        s._events = []
        # event.event is explicitly None — the hazardous case.
        s._handle_stream_event(SimpleNamespace(event=None))
        # No exception, no spurious events.
        assert s._events == []

    def test_raw_dict_event_passes_through(self):
        """A raw dict event (no .event attribute) is handled directly."""
        s = ClaudeSession()
        s._events = []
        s._handle_stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hi"},
            }
        )
        assert len(s._events) == 1
        assert s._events[0].type == "text"
        assert s._events[0].content == "hi"

    def test_typed_event_with_dict_payload(self):
        """An event object whose .event is a dict is unwrapped correctly."""
        s = ClaudeSession()
        s._events = []
        s._handle_stream_event(
            SimpleNamespace(
                event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}},
            )
        )
        assert len(s._events) == 1
        assert s._events[0].content == "x"


class TestQueryTimeout:
    """ClaudeSession.query() timeout + interrupt wiring (R11-1)."""

    def test_timeout_returns_failure_and_interrupts_agent(self):
        """A hung receive_response() must time out, interrupt the SDK agent
        subprocess, and return a failure verdict instead of hanging forever.

        Drives query(), which reaches _process_response_stream's SDK import —
        requires claude_agent_sdk (optional dependency), skipped when absent.
        Without the skip this fails as "Session error" (the ImportError path)
        rather than the timeout it means to assert.
        """
        pytest.importorskip("claude_agent_sdk")
        interrupted = {"called": False}

        class _HungClient:
            async def query(self, prompt):
                pass

            async def receive_response(self):
                # SDK hang: stream never completes
                while True:
                    await asyncio.sleep(0.1)
                    yield SimpleNamespace()

            async def interrupt(self):
                interrupted["called"] = True

        s = ClaudeSession(query_timeout=0.05)
        s._client = _HungClient()

        result = asyncio.run(s.query("hello"))

        assert result.verdict.status == "failure"
        assert result.verdict.summary == "Query timed out"
        assert "timed out" in (result.error or "").lower()
        assert interrupted["called"] is True
        # interrupt() emits a status event into the failure result
        assert any(e.type == "status" and e.content == "INTERRUPTED" for e in result.events)

    def test_completes_normally_with_timeout_set(self):
        """A fast stream must complete normally even with a timeout configured.

        Requires claude_agent_sdk (optional dependency) — skipped when absent,
        for the same reason as the timeout test above.
        """
        pytest.importorskip("claude_agent_sdk")

        class _FastClient:
            async def query(self, prompt):
                pass

            async def receive_response(self):
                # empty stream — generator completes immediately
                return
                yield  # pragma: no cover — unreachable, keeps it a generator

        s = ClaudeSession(query_timeout=5.0)
        s._client = _FastClient()

        result = asyncio.run(s.query("hello"))

        assert result.verdict.status == "insufficient_info"  # fallback verdict
        assert result.error is None

    def test_no_timeout_by_default(self):
        """query_timeout defaults to None → wait_for not applied."""
        s = ClaudeSession()
        assert s._query_timeout is None


class TestEventRetentionCap:
    """SessionResult.events must be a bounded ring (most recent), not an
    unbounded accumulation — streamed text deltas can emit hundreds of events
    per single query, and no production consumer reads the full log."""

    def test_retained_events_ring_capped_but_callback_sees_all(self):
        seen: list[SessionEvent] = []
        s = ClaudeSession(event_callback=seen.append)
        total = _MAX_RETAINED_EVENTS + 50
        for i in range(total):
            s._emit_event(SessionEvent(type="text", content=f"chunk-{i}"))
        # Ring cap: only the most recent events are retained, in order.
        assert len(s._events) == _MAX_RETAINED_EVENTS
        assert s._events[0].content == "chunk-50"  # oldest dropped
        assert s._events[-1].content == f"chunk-{total - 1}"  # newest kept
        # The live callback is NOT capped — every event still reaches it.
        assert len(seen) == total
        assert seen[0].content == "chunk-0"
        assert seen[-1].content == f"chunk-{total - 1}"

    def test_under_cap_keeps_all_in_order(self):
        s = ClaudeSession()
        for i in range(5):
            s._emit_event(SessionEvent(type="text", content=f"c{i}"))
        assert [e.content for e in s._events] == ["c0", "c1", "c2", "c3", "c4"]

    def test_stream_deltas_stay_bounded(self):
        """The real emission path (_handle_stream_event → _emit_event) must
        keep the retained list within the cap for a long partial stream."""
        s = ClaudeSession()
        s._events = []
        for i in range(_MAX_RETAINED_EVENTS + 30):
            s._handle_stream_event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": f"t{i}"},
                }
            )
        assert len(s._events) == _MAX_RETAINED_EVENTS
        assert s._events[-1].content == f"t{_MAX_RETAINED_EVENTS + 29}"
