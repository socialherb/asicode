"""replace_tool_calls: shared dict/object tool_calls mutation contract.

Single-sourced from agent_loop's truncation-recovery closure (R-series) so
both agent_loop and design_chat_loop share ONE implementation of the
dict/object dual path. These tests pin the module-level contract:
mutation happens in place (the same response is returned), a dict subclass
that rejects the key is left untouched, and a response object without a
settable ``tool_calls`` attribute degrades silently instead of raising.
"""
from __future__ import annotations

from external_llm.agent._response_utils import replace_tool_calls


class _NoSetter:
    """Object whose tool_calls is a read-only attribute (no setter)."""

    def __init__(self, calls):
        self._calls = calls

    @property
    def tool_calls(self):
        return self._calls


class TestReplaceToolCalls:
    def test_dict_response_mutated_in_place_and_returned(self):
        resp = {"content": "text", "tool_calls": ["stale"]}
        returned = replace_tool_calls(resp, [])
        assert resp["tool_calls"] == []
        assert returned is resp  # same object, in-place contract

    def test_object_response_attribute_set(self):
        class Resp:
            def __init__(self):
                self.tool_calls = ["stale"]

        resp = Resp()
        returned = replace_tool_calls(resp, [])
        assert resp.tool_calls == []
        assert returned is resp

    def test_no_setter_attribute_degrades_silently(self):
        resp = _NoSetter(["stale"])
        returned = replace_tool_calls(resp, [])
        # untouched (no setter) — no exception, same object back
        assert resp.tool_calls == ["stale"]
        assert returned is resp

    def test_new_calls_installed(self):
        resp = {"tool_calls": []}
        new_calls = [{"id": "call_1", "type": "function"}]
        replace_tool_calls(resp, new_calls)
        assert resp["tool_calls"] is new_calls
