"""
RED→GREEN coverage tests for external_llm/repl/collaborate/claude_session.py.

Closes the 56% → 100% gap: context-manager lifecycle (__aenter__/__aexit__),
the stream loop's message-type dispatch (StreamEvent / AssistantMessage /
UserMessage / ResultMessage / SystemMessage), assistant/user block handling,
the verdict salvage ladder (structured_output → captured StructuredOutput
tool input → result text → accumulated text → failure), interrupt
best-effort isolation, and event-callback error isolation.

Uses the REAL claude_agent_sdk dataclasses as message fixtures (the SDK is an
optional dependency — skipped when absent) and monkeypatched ClaudeSDKClient
for lifecycle tests.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
AssistantMessage = claude_agent_sdk.AssistantMessage
ResultMessage = claude_agent_sdk.ResultMessage
StreamEvent = claude_agent_sdk.StreamEvent
SystemMessage = claude_agent_sdk.SystemMessage
TextBlock = claude_agent_sdk.TextBlock
ToolResultBlock = claude_agent_sdk.ToolResultBlock
ToolUseBlock = claude_agent_sdk.ToolUseBlock
UserMessage = claude_agent_sdk.UserMessage

from external_llm.repl.collaborate.claude_session import (
    _MAX_RETAINED_EVENTS,
    ClaudeSession,
    SessionEvent,
)


def _run(coro):
    return asyncio.run(coro)


class _StreamClient:
    """Fake SDK client: records query(), yields canned messages, optional
    receive_response/interrupt failure injection."""

    def __init__(self, messages=(), fail_receive=None, fail_interrupt=None):
        self._messages = list(messages)
        self._fail_receive = fail_receive
        self._fail_interrupt = fail_interrupt
        self.query_calls = 0

    async def query(self, prompt):
        self.query_calls += 1

    async def receive_response(self):
        if self._fail_receive is not None:
            raise self._fail_receive
        for m in self._messages:
            yield m

    async def interrupt(self):
        if self._fail_interrupt is not None:
            raise self._fail_interrupt


class _BlockingClient:
    """Fake SDK client whose stream never completes — models a stuck agent
    subprocess (the exact hazard query_timeout guards). Records whether the
    stream observed cancellation and how often interrupt() ran."""

    def __init__(self):
        self.interrupt_calls = 0
        self.stream_cancelled = False

    async def query(self, prompt):
        pass

    async def receive_response(self):
        try:
            await asyncio.Event().wait()  # never set: stream hangs forever
            yield None  # pragma: no cover
        except (asyncio.CancelledError, GeneratorExit):
            self.stream_cancelled = True
            raise

    async def interrupt(self):
        self.interrupt_calls += 1


class TestContextManagerLifecycle:
    """__aenter__/__aexit__: client construction, partial-message flag,
    connect failure cleanup, disconnect error tolerance."""

    def test_aenter_connects_and_sets_partial_flag(self, monkeypatch):
        calls = {}

        class FakeClient:
            def __init__(self, options=None):
                self.options = options
                calls["constructed"] = True

            async def connect(self):
                calls["connected"] = True

            async def disconnect(self):
                calls["disconnected"] = True

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
        opts = SimpleNamespace()
        s = ClaudeSession(options=opts)

        _run(s.__aenter__())

        assert calls == {"constructed": True, "connected": True}
        assert opts.include_partial_messages is True  # L92 side effect
        assert s._client is not None
        assert s._start_time > 0

    def test_aenter_include_partial_false_skips_flag(self, monkeypatch):
        class FakeClient:
            def __init__(self, options=None):
                self.options = options

            async def connect(self):
                pass

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
        opts = SimpleNamespace()
        s = ClaudeSession(options=opts, include_partial=False)
        _run(s.__aenter__())
        assert not hasattr(opts, "include_partial_messages")

    def test_aenter_no_options_still_connects(self, monkeypatch):
        class FakeClient:
            def __init__(self, options=None):
                self.options = options

            async def connect(self):
                pass

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
        s = ClaudeSession(options=None)  # _include_partial True but no options
        _run(s.__aenter__())
        assert s._client is not None

    def test_aenter_connect_failure_disconnects_and_reraises(self, monkeypatch):
        calls = {}

        class FakeClient:
            def __init__(self, options=None):
                pass

            async def connect(self):
                raise ConnectionError("refused")

            async def disconnect(self):
                calls["disconnected"] = True

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
        s = ClaudeSession()

        with pytest.raises(ConnectionError, match="refused"):
            _run(s.__aenter__())

        assert calls == {"disconnected": True}  # cleanup attempted
        assert s._client is None  # leaked client cleared

    def test_aexit_disconnects_and_clears(self):
        class FakeClient:
            def __init__(self):
                self.disconnect_called = False

            async def disconnect(self):
                self.disconnect_called = True

        c = FakeClient()
        s = ClaudeSession()
        s._client = c
        _run(s.__aexit__(None, None, None))
        assert c.disconnect_called
        assert s._client is None

    def test_aexit_disconnect_error_ignored(self, caplog):
        class FakeClient:
            async def disconnect(self):
                raise OSError("already closed")

        s = ClaudeSession()
        s._client = FakeClient()
        with caplog.at_level(logging.DEBUG):
            _run(s.__aexit__(None, None, None))  # must not raise
        assert s._client is None
        assert "Disconnect error (ignored)" in caplog.text

    def test_aexit_without_client_is_noop(self):
        s = ClaudeSession()
        _run(s.__aexit__(None, None, None))  # _client None → nothing


class TestQueryGuard:
    """query() precondition + timeout-less path + generic failure."""

    def test_query_raises_when_not_connected(self):
        s = ClaudeSession()
        with pytest.raises(RuntimeError, match="not connected"):
            _run(s.query("hello"))

    def test_query_without_timeout_uses_plain_path(self):
        """query_timeout=None → wait_for NOT applied (L154 else branch)."""
        client = _StreamClient()  # empty stream
        s = ClaudeSession(query_timeout=None)
        s._client = client
        result = _run(s.query("hello"))
        assert client.query_calls == 1
        assert result.verdict.status == "insufficient_info"
        assert result.error is None

    def test_query_generic_exception_returns_failure(self, caplog):
        client = _StreamClient(fail_receive=ValueError("boom"))
        s = ClaudeSession()
        s._client = client
        with caplog.at_level(logging.ERROR):
            result = _run(s.query("hello"))
        assert result.verdict.status == "failure"
        assert result.verdict.summary == "Session error"
        assert result.error == "boom"
        assert result.duration_seconds >= 0.0
        assert "query failed" in caplog.text

    def test_best_effort_interrupt_success(self):
        client = _StreamClient()
        s = ClaudeSession()
        s._client = client
        _run(s._best_effort_interrupt())  # interrupt() called, no event assert

    def test_best_effort_interrupt_failure_ignored(self, caplog):
        client = _StreamClient(fail_interrupt=RuntimeError("nope"))
        s = ClaudeSession()
        s._client = client
        with caplog.at_level(logging.DEBUG):
            _run(s._best_effort_interrupt())  # must swallow, not raise
        assert "Interrupt failed (ignored)" in caplog.text

    def test_interrupt_emits_status_event(self):
        client = _StreamClient()
        s = ClaudeSession()
        s._client = client
        _run(s.interrupt())
        assert s._events[-1].type == "status"
        assert s._events[-1].content == "INTERRUPTED"

    def test_interrupt_without_client_is_noop(self):
        s = ClaudeSession()
        _run(s.interrupt())  # _client None → no-op


class TestOuterCancellationSealsSubprocess:
    """Outer cancellation / timeout must not leak the SDK agent subprocess.

    On 3.12+ wait_for does not clean up its inner awaitable on outer
    cancellation, and the cancellation paths never called interrupt() —
    only the TimeoutError branch did — so cancelling the collaboration
    task (collaboration_orchestrator.run) left the agent subprocess
    running in the background burning tokens/CPU. The explicit Task guard
    must cancel the inner task and interrupt the subprocess before
    re-raising. (The GeneratorExit leg of the guard is defensive parity
    with push_manager's sealed keepalive-park pattern; the production
    caller chain here is a plain task, so CancelledError is the realistic
    delivery and what these tests seal.)"""

    def test_outer_task_cancel_interrupts_subprocess(self):
        async def scenario():
            client = _BlockingClient()
            s = ClaudeSession(query_timeout=30.0)
            s._client = client
            task = asyncio.create_task(s.query("hello"))
            await asyncio.sleep(0.05)  # reach the blocking stream
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return client

        client = _run(scenario())
        assert client.interrupt_calls == 1
        assert client.stream_cancelled is True

    def test_timeout_path_still_interrupts_with_task_wrapper(self):
        """The create_task refactor must not regress the TimeoutError
        branch: wait_for(task, ...) must cancel the inner stream on timeout
        and the branch must still interrupt the subprocess."""

        async def scenario():
            client = _BlockingClient()
            s = ClaudeSession(query_timeout=0.05)
            s._client = client
            result = await s.query("hello")
            return client, result

        client, result = _run(scenario())
        assert result.verdict.status == "failure"
        assert "timed out" in result.error
        assert client.interrupt_calls == 1
        assert client.stream_cancelled is True


class TestStreamLoopDispatch:
    """_process_response_stream: every message-type branch (L251-282)."""

    def _collect(self, messages):
        client = _StreamClient(messages)
        s = ClaudeSession()
        s._client = client
        result = _run(s.query("hello"))
        return s, result

    def test_all_message_types_dispatched(self):
        stream = StreamEvent(
            uuid="u1", session_id="s1",
            event={"type": "content_block_delta",
                   "delta": {"type": "text_delta", "text": "part"}},
        )
        assistant = AssistantMessage(
            content=[TextBlock(text="analysis"),
                     ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"})],
            model="claude-sonnet",
        )
        user = UserMessage(content=[
            ToolResultBlock(tool_use_id="t1", content="ok", is_error=False),
        ])
        result = ResultMessage(
            subtype="result", duration_ms=10, duration_api_ms=8,
            is_error=False, num_turns=2, session_id="s1",
            total_cost_usd=0.5,
            usage={"input_tokens": 10, "output_tokens": 5,
                   "cache_creation_input_tokens": 2, "cache_read_input_tokens": 3},
            structured_output={"status": "success", "summary": "done",
                               "confidence": 0.9},
        )
        system = SystemMessage(subtype="system", data={"hook": "x"})

        s, result = self._collect([stream, assistant, user, result, system])

        assert result.verdict.status == "success"
        assert result.verdict.summary == "done"
        assert s._tool_calls_count == 1
        assert s._last_total_tokens == 20  # 10+5+2+3
        assert s._last_cost_usd == 0.5
        types = [e.type for e in s._events]
        assert "text" in types  # partial delta + full TextBlock
        assert types.count("tool_call") == 1  # ToolUseBlock complete event
        assert "tool_result" in types
        assert types.count("verdict") == 1
        # SystemMessage produced nothing
        assert len([e for e in s._events if e.content.startswith("System")]) == 0

    def test_assistant_error_sets_has_error(self):
        assistant = AssistantMessage(
            content=[], model="m", error="billing_error",
        )
        _s, result = self._collect([assistant])
        # No ResultMessage → fallback; has_error → "error" status
        assert result.verdict.status == "error"
        assert result.verdict.summary == "No structured verdict returned"

    def test_result_with_errors_emits_error_event_and_note(self):
        result = ResultMessage(
            subtype="result", duration_ms=1, duration_api_ms=1,
            is_error=True, num_turns=1, session_id="s1",
            errors=["Reached maximum budget"],
            api_error_status=429,
            structured_output={"status": "success", "summary": "done"},
        )
        s, result = self._collect([result])
        error_events = [e for e in s._events if e.type == "error"]
        assert len(error_events) == 1
        assert "Reached maximum budget" in error_events[0].content
        assert result.verdict.status == "success"  # completed work survives late error
        assert result.verdict.metadata["result_error"] == "Reached maximum budget"

    def test_result_errors_not_duplicated_when_has_error(self):
        assistant = AssistantMessage(content=[], model="m", error="rate_limit")
        result = ResultMessage(
            subtype="result", duration_ms=1, duration_api_ms=1,
            is_error=True, num_turns=1, session_id="s1",
            errors=["rate_limit"], api_error_status=429,
            structured_output={"status": "success", "summary": "done"},
        )
        s, _ = self._collect([assistant, result])
        # has_error already True → loop must NOT emit a second error event
        assert len([e for e in s._events if e.type == "error"]) == 0

    def test_system_message_only_falls_back(self):
        system = SystemMessage(subtype="system", data={})
        _s, result = self._collect([system])
        assert result.verdict.status == "insufficient_info"

    def test_fallback_uses_accumulated_text(self):
        assistant = AssistantMessage(
            content=[TextBlock(text="full analysis body")], model="m",
        )
        _s, result = self._collect([assistant])
        assert result.verdict.status == "success"  # full_text non-empty
        assert "full analysis body" in result.verdict.details


class TestStreamEventHandler:
    """_handle_stream_event remaining branches: input_json_delta,
    content_block_start tool_use, content_block_stop."""

    def test_input_json_delta_emits_partial_tool_call(self):
        s = ClaudeSession()
        s._handle_stream_event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'},
        })
        assert len(s._events) == 1
        e = s._events[0]
        assert e.type == "tool_call"
        assert e.content == '{"cmd":'
        assert e.metadata["partial"] is True

    def test_input_json_delta_empty_partial_no_event(self):
        s = ClaudeSession()
        s._handle_stream_event({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": ""},
        })
        assert s._events == []

    def test_content_block_start_tool_use_records_and_emits(self):
        s = ClaudeSession()
        s._handle_stream_event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "t9", "name": "read_file"},
        })
        assert s._tool_names_by_id["t9"] == "read_file"
        assert len(s._events) == 1
        e = s._events[0]
        assert e.type == "tool_call"
        assert e.content == "Starting tool: read_file"
        assert e.metadata["event"] == "start"

    def test_content_block_start_tool_use_without_id(self):
        s = ClaudeSession()
        s._handle_stream_event({
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "bash"},  # no id
        })
        assert s._events[0].content == "Starting tool: bash"
        assert "t" not in s._tool_names_by_id

    def test_content_block_start_non_tool_ignored(self):
        s = ClaudeSession()
        s._handle_stream_event({
            "type": "content_block_start",
            "content_block": {"type": "text", "text": "hi"},
        })
        assert s._events == []

    def test_content_block_stop_is_noop(self):
        s = ClaudeSession()
        s._handle_stream_event({"type": "content_block_stop"})
        assert s._events == []


class TestAssistantMessageBlocks:
    """_handle_assistant_message: TextBlock / ToolUseBlock / StructuredOutput
    capture / ToolResultBlock."""

    def test_text_block_appends_and_emits(self):
        s = ClaudeSession()
        acc = []
        s._handle_assistant_message(
            SimpleNamespace(content=[TextBlock(text="hello")]), acc,
        )
        assert acc == ["hello"]
        assert s._events[0].type == "text"
        assert s._events[0].content == "hello"

    def test_empty_text_block_skipped(self):
        s = ClaudeSession()
        acc = []
        s._handle_assistant_message(
            SimpleNamespace(content=[TextBlock(text="")]), acc,
        )
        assert acc == []
        assert s._events == []

    def test_tool_use_block_counts_and_emits(self):
        s = ClaudeSession()
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"}),
            ]), [],
        )
        assert s._tool_calls_count == 1
        assert s._tool_names_by_id["t1"] == "bash"
        e = s._events[0]
        assert e.type == "tool_call"
        assert e.content == "Tool: bash"
        assert e.metadata["input"] == {"cmd": "ls"}
        assert e.metadata["event"] == "complete"

    def test_tool_use_block_without_id_counts_anyway(self):
        s = ClaudeSession()
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolUseBlock(id="", name="bash", input={}),
            ]), [],
        )
        assert s._tool_calls_count == 1
        assert s._events[0].content == "Tool: bash"

    def test_structured_output_tool_captured_without_count(self):
        s = ClaudeSession()
        payload = {"status": "success", "summary": "v"}
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolUseBlock(id="so1", name="StructuredOutput", input=payload),
            ]), [],
        )
        assert s._structured_candidate == payload
        assert s._tool_calls_count == 0  # not a user-facing tool call
        assert s._events == []  # skipped entirely

    def test_output_alias_tool_captured(self):
        s = ClaudeSession()
        payload = {"status": "failure", "summary": "v"}
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolUseBlock(id="o1", name="output_json", input=payload),
            ]), [],
        )
        assert s._structured_candidate == payload

    def test_structured_output_non_dict_input_not_captured(self):
        s = ClaudeSession()
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolUseBlock(id="so2", name="StructuredOutput", input="nope"),
            ]), [],
        )
        assert s._structured_candidate is None
        assert s._tool_calls_count == 0

    def test_tool_result_block_routes_to_emitter(self):
        s = ClaudeSession()
        s._handle_assistant_message(
            SimpleNamespace(content=[
                ToolResultBlock(tool_use_id="t1", content="done"),
            ]), [],
        )
        assert s._events[0].type == "tool_result"
        assert s._events[0].content == "done"

    def test_no_content_blocks(self):
        s = ClaudeSession()
        s._handle_assistant_message(SimpleNamespace(content=[]), [])
        s._handle_assistant_message(SimpleNamespace(content=None), [])
        assert s._events == []


class TestUserMessageToolResults:
    """_handle_user_message: ToolResultBlock extraction from UserMessage."""

    def test_extracts_tool_result_blocks(self):
        s = ClaudeSession()
        s._handle_user_message(UserMessage(content=[
            ToolResultBlock(tool_use_id="t1", content="ok"),
            TextBlock(text="ignored"),
        ]))
        assert len(s._events) == 1
        assert s._events[0].type == "tool_result"

    def test_non_list_content_returns(self):
        s = ClaudeSession()
        s._handle_user_message(UserMessage(content="plain text"))
        assert s._events == []

    def test_list_without_tool_results_noop(self):
        s = ClaudeSession()
        s._handle_user_message(UserMessage(content=[TextBlock(text="x")]))
        assert s._events == []


class TestEmitToolResult:
    """_emit_tool_result content normalization (str | list | None)."""

    def test_list_content_joined(self):
        s = ClaudeSession()
        s._emit_tool_result(SimpleNamespace(
            tool_use_id="t1", content=[{"type": "text", "text": "a"},
                                       {"type": "text", "text": "b"}],
            is_error=False,
        ))
        assert s._events[0].content == "a\nb"
        assert s._events[0].metadata["tool_use_id"] == "t1"

    def test_list_content_filters_non_dicts(self):
        s = ClaudeSession()
        s._emit_tool_result(SimpleNamespace(
            tool_use_id="t1",
            content=[{"type": "text", "text": "keep"}, "junk", 42],
            is_error=False,
        ))
        assert s._events[0].content == "keep"

    def test_str_content_truncated_to_500(self):
        s = ClaudeSession()
        s._emit_tool_result(SimpleNamespace(
            tool_use_id="t1", content="x" * 1000, is_error=False,
        ))
        assert len(s._events[0].content) == 500

    def test_none_content_becomes_empty(self):
        s = ClaudeSession()
        s._emit_tool_result(SimpleNamespace(
            tool_use_id="t1", content=None, is_error=False,
        ))
        assert s._events[0].content == ""

    def test_is_error_and_missing_id_defaults(self):
        s = ClaudeSession()
        s._emit_tool_result(SimpleNamespace(
            content="fail", is_error=True,  # tool_use_id attribute absent
        ))
        e = s._events[0]
        assert e.metadata["is_error"] is True
        assert e.metadata["tool_name"] == "?"  # unknown id
        assert e.metadata["tool_use_id"] == "?"

    def test_tool_name_resolved_from_capture(self):
        s = ClaudeSession()
        s._tool_names_by_id["t1"] = "bash"
        s._emit_tool_result(SimpleNamespace(
            tool_use_id="t1", content="ok", is_error=None,
        ))
        assert s._events[0].metadata["tool_name"] == "bash"
        assert s._events[0].metadata["is_error"] is False  # None coerced


class TestResultSalvageLadder:
    """_handle_result_message salvage order: structured → captured tool input
    → result text → accumulated text → failure/insufficient."""

    def _result(self, **kw):
        base = {"subtype": "result", "duration_ms": 1, "duration_api_ms": 1,
            "is_error": False, "num_turns": 1, "session_id": "s1"}
        base.update(kw)
        return ResultMessage(**base)

    def test_captured_structured_candidate_with_error_note(self):
        s = ClaudeSession()
        s._structured_candidate = {"status": "success", "summary": "captured"}
        verdict = s._handle_result_message(
            self._result(errors=["budget"], is_error=True), [],
        )
        assert verdict.status == "success"
        assert verdict.summary == "captured"
        assert verdict.metadata["result_error"] == "budget"

    def test_captured_structured_candidate_clean(self):
        s = ClaudeSession()
        s._structured_candidate = {"status": "failure", "summary": "captured"}
        verdict = s._handle_result_message(self._result(), [])
        assert verdict.status == "failure"
        assert "result_error" not in verdict.metadata

    def test_result_text_needs_review(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(
            self._result(result="First line summary\nlong body"), [],
        )
        assert verdict.status == "needs_review"
        assert verdict.summary == "First line summary"
        assert "long body" in verdict.details
        assert verdict.confidence == 0.5

    def test_result_text_with_error_is_failure(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(
            self._result(result="partial work", is_error=True), [],
        )
        assert verdict.status == "failure"

    def test_accumulated_text_fallback_with_error_note(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(
            self._result(errors=["max turns"]), ["analysis text"],
        )
        assert verdict.status == "needs_review"
        assert "max turns" in verdict.summary
        assert verdict.details == "analysis text"
        assert verdict.metadata["result_error"] == "max turns"

    def test_accumulated_text_fallback_without_error_note(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(self._result(), ["analysis text"])
        assert verdict.summary == "Analysis text without structured verdict"
        assert verdict.metadata == {}

    def test_truly_empty_with_error_failure(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(
            self._result(errors=["boom"], is_error=True), [],
        )
        assert verdict.status == "failure"
        assert verdict.summary == "Execution failed"
        assert verdict.details == "boom"
        assert verdict.confidence == 1.0

    def test_truly_empty_error_without_note(self):
        s = ClaudeSession()
        verdict = s._handle_result_message(
            self._result(is_error=True), [],
        )
        assert verdict.status == "failure"
        assert verdict.details == "Unknown error"

    def test_verdict_event_emitted_with_metadata(self):
        s = ClaudeSession()
        s._handle_result_message(
            self._result(result="plain text"), [],
        )
        verdict_events = [e for e in s._events if e.type == "verdict"]
        assert len(verdict_events) == 1
        assert verdict_events[0].content.startswith("Status: needs_review |")
        assert "verdict" in verdict_events[0].metadata


class TestCallbackIsolation:
    """A raising event_callback must not break the session (L550-551)."""

    def test_callback_error_swallowed(self, caplog):
        def _bad_cb(event):
            raise ValueError("callback bug")

        s = ClaudeSession(event_callback=_bad_cb)
        with caplog.at_level(logging.DEBUG):
            s._handle_stream_event({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "ok"},
            })
        # Event still retained internally despite callback failure
        assert len(s._events) == 1
        assert "Event callback error" in caplog.text

    def test_callback_receives_every_event_over_cap(self):
        seen = []
        s = ClaudeSession(event_callback=seen.append)
        for i in range(_MAX_RETAINED_EVENTS + 5):
            s._emit_event(SessionEvent(type="text", content=f"c{i}"))
        assert len(seen) == _MAX_RETAINED_EVENTS + 5
