"""
Tests for ClaudeSession event handling and SessionEvent/SessionResult.
"""
from __future__ import annotations

from types import SimpleNamespace

from external_llm.repl.collaborate import CollaborationVerdict
from external_llm.repl.collaborate.claude_session import (
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
        s._handle_stream_event({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hi"},
        })
        assert len(s._events) == 1
        assert s._events[0].type == "text"
        assert s._events[0].content == "hi"

    def test_typed_event_with_dict_payload(self):
        """An event object whose .event is a dict is unwrapped correctly."""
        s = ClaudeSession()
        s._events = []
        s._handle_stream_event(SimpleNamespace(
            event={"type": "content_block_delta",
                   "delta": {"type": "text_delta", "text": "x"}},
        ))
        assert len(s._events) == 1
        assert s._events[0].content == "x"
