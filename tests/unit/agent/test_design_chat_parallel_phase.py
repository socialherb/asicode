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
from unittest.mock import patch

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
    _WRITE_TOOLS = {
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
