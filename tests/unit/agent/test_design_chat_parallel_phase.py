"""DesignChatLoop's parallel tool phase must serialize mutating bash.

A ``bash`` call whose command changes filesystem/git state (rm, git commit,
"> file", sed -i, …) is a WRITE, not a pure read: running it in the parallel
read phase alongside other reads / other bash races on shared file/git state.
The fix routes such calls to the serialized write phase via ``_is_mutating`` →
``registry._tool_call_mutates`` (single source of truth shared with cache
invalidation and ``dispatch_parallel``).

These drive the real ``_respond_impl`` multi-tool branch and assert that a
mutating bash never overlaps a concurrent read (max concurrency == 1), while a
read-only bash batch still parallelizes (guards against over-serialization).
"""
from __future__ import annotations

import threading
import time
import types
from typing import ClassVar
from unittest.mock import patch

import pytest

from external_llm.agent.agent_loop_types import AgentCancelled
from external_llm.agent.design_chat_loop import DesignChatLoop, DesignChatResult
from external_llm.client import ToolCallRequest, ToolCallResponse


def _make_tc(name: str, args: dict) -> ToolCallRequest:
    return ToolCallRequest(call_id=f"call_{name}", name=name, args=args)


def _make_response(tool_calls=None, content="ok") -> ToolCallResponse:
    return ToolCallResponse(
        content=content, model="x", provider="openai", tool_calls=tool_calls or [],
    )


class _ConfigStub:
    cancel_event = None


class _RegStub:
    """Minimal registry stub exposing the partition-relevant surface, while
    re-using the REAL ``_tool_call_mutates`` classifier (single source of
    truth) so the test exercises the same predicate the fix wires in."""
    _WRITE_TOOLS: ClassVar[set] = {
        "apply_patch", "write_plan", "edit_ast", "edit_file",
        "edit_text", "modify_symbol", "anchor_edit",
    }
    _SERIAL_TOOLS = frozenset({"ask_user"})
    repo_language = None

    def __init__(self):
        self.session_plan = None

    def _tool_call_mutates(self, name, args):
        # Re-use the REAL classifier (single source of truth) so the test
        # exercises the same predicate logic the fix wires in.
        from external_llm.agent.tool_registry import ToolRegistry
        if name in self._WRITE_TOOLS:
            return True
        if name == "bash":
            return ToolRegistry._bash_command_mutates_files((args or {}).get("command", ""))
        return False

    def _tool_call_is_serial(self, name, args):
        if name == "ask_user":
            return True
        if name == "job":
            return (args or {}).get("action") == "kill"
        return False

    def get_tool_schemas(self, **kw):
        return [{"name": "bash"}, {"name": "find_symbol"}]


class _LLMStub:
    @staticmethod
    def get_provider_name() -> str:
        return "stub"

    def chat(self, *a, **k):
        return _make_response(content="final")


def _drive(loop, tool_calls):
    """Run one ``_respond_impl`` iteration that processes ``tool_calls``.

    Returns (concurrency_state, call_order) observed by the ``_process_tool_call``
    spy, which replaces the real dispatcher so no real tool side effects fire.
    """
    state = {"current": 0, "max": 0}
    order: list[str] = []
    _guard = threading.Lock()

    def _spy_process(tc, cb, result):
        with _guard:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
            order.append(tc.name)
        try:
            time.sleep(0.08)  # widen the overlap window so races are detectable
            return "tool-result"
        finally:
            with _guard:
                state["current"] -= 1

    loop._process_tool_call = _spy_process
    reg = _RegStub()
    reg.config = _ConfigStub()
    loop.registry = reg
    loop.model = "x"
    loop._result_lock = threading.Lock()
    loop.llm_client = _LLMStub()
    loop._build_final_instruction = lambda: "final"

    responses = [
        _make_response(tool_calls=tool_calls),  # iteration 0: emits the batch
        _make_response(tool_calls=[], content="done"),
    ]
    loop._call_llm_with_retry = lambda fn, **kwargs: responses.pop(0)

    result = DesignChatResult()
    with patch("external_llm.agent.design_chat_loop._apply_context_hard_cap",
               lambda msgs, *a, **k: msgs), \
         patch("external_llm.agent.design_chat_loop._strip_tool_messages",
               lambda m: m):
        loop._respond_impl([], None, None, 1, None, result, mode="code")
    return state, order


def test_dcl_serializes_mutating_bash_against_reads():
    """A mutating bash (rm) must run in the serialized write phase, AFTER the
    read phase — never overlapping a concurrent read."""
    tool_calls = [
        _make_tc("bash", {"command": "rm -rf build"}),
        _make_tc("find_symbol", {"name": "x"}),
    ]
    loop = DesignChatLoop.__new__(DesignChatLoop)
    state, _order = _drive(loop, tool_calls)
    assert state["max"] == 1, (
        f"mutating bash overlapped a concurrent read (max={state['max']})")


def test_dcl_keeps_readonly_bash_parallel():
    """Guard against over-serialization: read-only bash (ls, git status) MUST
    still run in the parallel read phase alongside other reads."""
    tool_calls = [
        _make_tc("bash", {"command": "ls -la"}),
        _make_tc("bash", {"command": "git status"}),
        _make_tc("find_symbol", {"name": "x"}),
    ]
    loop = DesignChatLoop.__new__(DesignChatLoop)
    state, _order = _drive(loop, tool_calls)
    assert state["max"] >= 2, (
        f"read-only bash batch did not parallelize (max={state['max']})")


def test_dcl_two_mutating_bash_never_overlap():
    """Two mutating bash calls in one batch must serialize against each other
    (both routed to the write phase, which holds _write_lock)."""
    tool_calls = [
        _make_tc("bash", {"command": "git commit -am a"}),
        _make_tc("bash", {"command": "rm -f out.log"}),
    ]
    loop = DesignChatLoop.__new__(DesignChatLoop)
    state, _order = _drive(loop, tool_calls)
    assert state["max"] == 1, (
        f"two mutating bash overlapped (max={state['max']})")


def _drive_cancel(loop, tool_calls, spy, ev):
    """Drive one _respond_impl tool iteration with a custom spy and a LIVE
    cancel event (parallel-phase cancel coverage)."""
    loop._process_tool_call = spy
    loop._run_store = None
    reg = _RegStub()
    reg.config = types.SimpleNamespace(cancel_event=ev)
    loop.registry = reg
    loop.model = "x"
    loop._result_lock = threading.Lock()
    loop.llm_client = _LLMStub()
    loop._build_final_instruction = lambda: "final"

    responses = [
        _make_response(tool_calls=tool_calls),
        _make_response(tool_calls=[], content="done"),
    ]
    loop._call_llm_with_retry = lambda fn, **kwargs: responses.pop(0)

    result = DesignChatResult()
    with patch("external_llm.agent.design_chat_loop._apply_context_hard_cap",
               lambda msgs, *a, **k: msgs), \
         patch("external_llm.agent.design_chat_loop._strip_tool_messages",
               lambda m: m):
        return loop._respond_impl([], None, None, 1, None, result, mode="code")


def test_dcl_cancel_event_preempts_parallel_tool_phase():
    """ESC during a long-running parallel tool must abort the batch at the poll
    cadence — NOT block until the tool's own finish.

    Regression: future.result() had no timeout, so a multi-minute tool (huge
    grep, web fetch) froze the whole turn after ESC; the cancel only took
    effect at the next loop boundary, after the tool completed."""
    ev = threading.Event()
    finished = {"v": False}

    def _slow_spy(tc, cb, result):
        time.sleep(1.5)
        finished["v"] = True
        return "tool-result"

    loop = DesignChatLoop.__new__(DesignChatLoop)
    out = {}

    def _run():
        try:
            _drive_cancel(loop, [
                _make_tc("find_symbol", {"name": "x"}),
                _make_tc("bash", {"command": "ls -la"}),
            ], _slow_spy, ev)
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # let the tools start in the pool
    ev.set()         # user presses ESC
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"cancel did not preempt the tool phase (exc={out.get('exc')!r})")
    assert not finished["v"], "cancel must abort while the tool is still running"


def test_dcl_agent_cancelled_from_tool_future_propagates():
    """AgentCancelled raised INSIDE a parallel tool must abort the batch —
    NOT degrade into a tool-error message that gets fed back to the LLM."""
    ev = threading.Event()
    release = threading.Event()

    def _cancelling_spy(tc, cb, result):
        release.wait(5)
        raise AgentCancelled("cancelled by user in tool")

    loop = DesignChatLoop.__new__(DesignChatLoop)
    out = {}

    def _run():
        try:
            _drive_cancel(loop, [
                _make_tc("find_symbol", {"name": "x"}),
                _make_tc("bash", {"command": "ls -la"}),
            ], _cancelling_spy, ev)
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # let the tools start and block on release
    release.set()
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"AgentCancelled was swallowed into a tool error (exc={out.get('exc')!r})")


def test_dcl_serial_phase_agent_cancelled_propagates():
    """The serial phase (ask_user) must propagate AgentCancelled from the tool
    instead of converting it to a tool-error message."""
    ev = threading.Event()

    def _spy(tc, cb, result):
        if tc.name == "ask_user":
            raise AgentCancelled("cancelled by user in serial tool")
        return "tool-result"

    loop = DesignChatLoop.__new__(DesignChatLoop)
    with pytest.raises(AgentCancelled):
        _drive_cancel(loop, [
            _make_tc("find_symbol", {"name": "x"}),
            _make_tc("ask_user", {"question": "?"}),
        ], _spy, ev)


def test_dcl_single_tool_phase_agent_cancelled_propagates():
    """The single-tool fast path must also propagate AgentCancelled (parity
    with the parallel phase) instead of converting it to a tool error."""
    ev = threading.Event()

    def _spy(tc, cb, result):
        raise AgentCancelled("cancelled by user in single tool")

    loop = DesignChatLoop.__new__(DesignChatLoop)
    with pytest.raises(AgentCancelled):
        _drive_cancel(loop, [_make_tc("find_symbol", {"name": "x"})], _spy, ev)


def _drive_with_cancel(loop, tool_calls, cancel_event, spy):
    """Drive one ``_respond_impl`` iteration with a custom tool spy and a LIVE
    cancel_event wired into the registry stub's config."""
    reg = _RegStub()
    reg.config = _ConfigStub()
    reg.config.cancel_event = cancel_event
    loop.registry = reg
    loop.model = "x"
    loop._result_lock = threading.Lock()
    loop.llm_client = _LLMStub()
    loop._build_final_instruction = lambda: "final"
    loop._process_tool_call = spy
    responses = [
        _make_response(tool_calls=tool_calls),  # iteration 0: emits the batch
        _make_response(tool_calls=[], content="done"),
    ]
    loop._call_llm_with_retry = lambda fn, **kwargs: responses.pop(0)
    result = DesignChatResult()
    with patch("external_llm.agent.design_chat_loop._apply_context_hard_cap",
               lambda msgs, *a, **k: msgs), \
         patch("external_llm.agent.design_chat_loop._strip_tool_messages",
               lambda m: m):
        loop._respond_impl([], None, None, 1, None, result, mode="code")


def test_dcl_cancel_during_parallel_phase_aborts_wait():
    """ESC while the parallel read phase is waiting on an in-flight tool must
    abort the phase within one poll interval — not after the slowest tool
    finishes. Deterministic: the fast read completes first and SETS the event
    from its own worker; the collect loop's next poll observes it while the
    slow read is still running."""
    cancel_event = threading.Event()
    tool_calls = [
        _make_tc("find_symbol", {"name": "fast"}),
        _make_tc("find_symbol", {"name": "slow"}),
    ]
    loop = DesignChatLoop.__new__(DesignChatLoop)

    def _spy(tc, cb, result):
        if tc.args.get("name") == "fast":
            time.sleep(0.1)
            cancel_event.set()  # fired inside the fast worker thread
            return "tool-result"
        time.sleep(1.0)  # slow — must be abandoned, not waited out
        return "tool-result"

    with pytest.raises(AgentCancelled):
        _drive_with_cancel(loop, tool_calls, cancel_event, _spy)
    assert cancel_event.is_set()


def test_dcl_cancel_before_serial_tool_skips_it():
    """ESC set while the read phase ran must be honored BEFORE the serial
    (ask_user) phase starts — ask_user blocks on human input and cannot be
    interrupted mid-call, so it must never execute after ESC."""
    cancel_event = threading.Event()
    calls: list[str] = []
    tool_calls = [
        _make_tc("find_symbol", {"name": "reader"}),
        _make_tc("ask_user", {"question": "q"}),
    ]
    loop = DesignChatLoop.__new__(DesignChatLoop)

    def _spy(tc, cb, result):
        calls.append(tc.name)
        if tc.name == "find_symbol":
            time.sleep(0.05)
            cancel_event.set()  # set while the read phase is still finishing
            return "tool-result"
        time.sleep(0.5)  # ask_user — must never be reached after ESC
        return "tool-result"

    with pytest.raises(AgentCancelled):
        _drive_with_cancel(loop, tool_calls, cancel_event, _spy)
    assert "ask_user" not in calls, "serial tool ran after ESC"


def test_dcl_cancel_event_preempts_single_tool_phase():
    """ESC during a long-running SINGLE tool must abort the wait at the poll
    cadence — the single-tool path was a bare inline call, so ESC froze the
    whole turn until the tool's own timeout (seconds to minutes)."""
    cancel_event = threading.Event()
    loop = DesignChatLoop.__new__(DesignChatLoop)
    out = {}

    def _slow_spy(tc, cb, result):
        time.sleep(1.5)
        return "tool-result"

    def _run():
        try:
            _drive_with_cancel(
                loop, [_make_tc("find_symbol", {"name": "x"})],
                cancel_event, _slow_spy,
            )
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # let the tool start in the pool
    cancel_event.set()  # user presses ESC
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"cancel did not preempt the single-tool phase (exc={out.get('exc')!r})")


def test_dcl_single_serial_tool_runs_inline_on_calling_thread():
    """A single ask_user call must run INLINE on the calling thread — never on
    the pool.

    Regression: the single-tool branch (added for cancel polling) submitted
    EVERY tool to shared_pool, so ask_user blocked a pool worker on stdin while
    the design thread exited on ESC — the worker's _cli_checkpoint_cb then
    re-wrote terminal state over the live prompt and double-read stdin. The
    batch path had always kept serial tools off the pool (phase 3)."""
    cancel_event = threading.Event()
    seen = {"thread": None}

    def _spy(tc, cb, result):
        seen["thread"] = threading.current_thread().name
        return "tool-result"

    loop = DesignChatLoop.__new__(DesignChatLoop)
    _drive_with_cancel(loop, [_make_tc("ask_user", {"question": "q"})], cancel_event, _spy)
    assert seen["thread"] == threading.main_thread().name, (
        f"ask_user ran on {seen['thread']!r} instead of the calling thread")


def test_dcl_single_serial_tool_skipped_after_cancel():
    """ESC before a single serial tool must skip it — ask_user blocks on human
    input and cannot be interrupted mid-call, so it must never execute after
    ESC (parity with the batch serial phase's entry check)."""
    cancel_event = threading.Event()
    cancel_event.set()
    called = {"v": False}

    def _spy(tc, cb, result):
        called["v"] = True
        return "tool-result"

    loop = DesignChatLoop.__new__(DesignChatLoop)
    with pytest.raises(AgentCancelled):
        _drive_with_cancel(loop, [_make_tc("ask_user", {"question": "q"})], cancel_event, _spy)
    assert not called["v"], "serial tool ran after ESC"


def test_dcl_abandoned_worker_events_suppressed_after_cancel():
    """After ESC, an abandoned pool worker's late stream events must be
    suppressed — the turn is over and must not emit tool lines into it.

    Deterministic: the slow worker only emits AFTER cancel is set, so a
    delivered event proves the guard failed."""
    cancel_event = threading.Event()
    events: list = []
    worker_emitted = threading.Event()

    loop = DesignChatLoop.__new__(DesignChatLoop)
    loop._run_store = None
    reg = _RegStub()
    reg.config = types.SimpleNamespace(cancel_event=cancel_event)
    loop.registry = reg
    loop.model = "x"
    loop._result_lock = threading.Lock()
    loop.llm_client = _LLMStub()
    loop._build_final_instruction = lambda: "final"

    def _spy(tc, cb, result):
        if tc.args.get("name") == "slow":
            cancel_event.wait(5)  # proceed only after ESC
            time.sleep(0.5)  # stay alive past the poll tick → phase aborts first
            cb("design_tool_call", {"tool": "slow", "status": "done"})
            worker_emitted.set()  # signal AFTER the emit attempt, guard or not
            return "tool-result"
        return "tool-result"

    loop._process_tool_call = _spy
    responses = [
        _make_response(tool_calls=[
            _make_tc("find_symbol", {"name": "fast"}),
            _make_tc("find_symbol", {"name": "slow"}),
        ]),
        _make_response(tool_calls=[], content="done"),
    ]
    loop._call_llm_with_retry = lambda fn, **kwargs: responses.pop(0)
    result = DesignChatResult()
    out = {}

    def _run():
        try:
            with patch("external_llm.agent.design_chat_loop._apply_context_hard_cap",
                       lambda msgs, *a, **k: msgs), \
                 patch("external_llm.agent.design_chat_loop._strip_tool_messages",
                       lambda m: m):
                loop._respond_impl([], lambda *a, **k: events.append(a),
                                   None, 1, None, result, mode="code")
            out["exc"] = None
        except AgentCancelled as ac:
            out["exc"] = ac

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.2)  # let the tools start; the slow worker waits on the event
    cancel_event.set()  # user presses ESC
    t.join(timeout=5)
    assert isinstance(out.get("exc"), AgentCancelled), (
        f"cancel did not abort the phase (exc={out.get('exc')!r})")
    # Wait for the abandoned worker's late emit attempt — otherwise the
    # assertion below races the worker and passes vacuously.
    assert worker_emitted.wait(5), "slow worker never reached its emit"
    leaked = [e for e in events
              if isinstance(e, tuple) and e and e[0] == "design_tool_call"]
    assert leaked == [], f"late events from abandoned worker leaked: {leaked}"
