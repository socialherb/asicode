"""Regression: native tool-message assembly must not interleave non-tool
messages between an assistant(tool_calls) message and its tool responses.

_process_tool_results injects role="user" strategy/exhaustion warnings into the
same list as the role="tool" results, sometimes *before* a tool result. If
_append_native_tool_messages appended that list wholesale, the wire sequence
became assistant(tool_calls) -> user(warning) -> tool(result), which:
  - OpenAI/DeepSeek reject with HTTP 400 (tool_calls must be immediately
    followed by tool messages), and
  - repair_tool_message_sequence "fixes" by dropping the entire tool exchange.
"""
from __future__ import annotations

import itertools
import types

from external_llm.agent.agent_loop import AgentLoop
from external_llm.agent.context_budget import repair_tool_message_sequence
from external_llm.client import LLMMessage


def _loop(provider: str) -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)  # skip heavy __init__
    loop.llm_client = types.SimpleNamespace(get_provider_name=lambda: provider)
    return loop


def _response_with_tool_call(call_id: str) -> dict:
    raw = types.SimpleNamespace(
        raw_response={
            "choices": [
                {"message": {"tool_calls": [{"id": call_id, "type": "function",
                                             "function": {"name": "apply_patch", "arguments": "{}"}}]}}
            ]
        }
    )
    return {"content": "let me patch", "raw": raw}


def _warn(text: str) -> LLMMessage:
    return LLMMessage(role="user", content=text)


def _tool_result(call_id: str) -> LLMMessage:
    return LLMMessage(role="tool", name="apply_patch", tool_call_id=call_id,
                      content='{"ok": false}')


def test_deepseek_warning_does_not_interleave_assistant_and_tool():
    loop = _loop("deepseek")
    response = _response_with_tool_call("call_1")
    # warning appears BEFORE the tool result, exactly as _process_tool_results emits it
    tool_result_messages = [_warn("[STRATEGY WARNING] stop retrying"), _tool_result("call_1")]

    out = loop._append_native_tool_messages([], response, tool_result_messages)
    roles = [m.role for m in out]

    # assistant(tool_calls) must be immediately followed by the tool message
    assert roles == ["assistant", "tool", "user"], roles
    assert out[0].tool_calls and out[0].tool_calls[0]["id"] == "call_1"
    assert out[1].tool_call_id == "call_1"
    assert "STRATEGY WARNING" in out[2].content

    # And the repair pass must now preserve the exchange (not drop it).
    repaired = repair_tool_message_sequence(out)
    rroles = [m.role for m in repaired]
    assert "assistant" in rroles and "tool" in rroles, rroles


def _assert_roles_alternate(roles):
    """Anthropic/Gemini reject two consecutive same-role (user/model) turns."""
    for a, b in itertools.pairwise(roles):
        assert not (a == b == "user"), f"consecutive user turns: {roles}"


def test_anthropic_warning_folded_into_tool_result_turn_no_malformed_block():
    loop = _loop("anthropic")
    response = {"content": "patching", "raw": types.SimpleNamespace(
        raw_response={"content": [{"type": "tool_use", "id": "call_9", "name": "apply_patch", "input": {}}]})}
    tool_result_messages = [_warn("[STRATEGY WARNING] switch approach"), _tool_result("call_9")]

    out = loop._append_native_tool_messages([], response, tool_result_messages)

    _assert_roles_alternate([m.role for m in out])
    # Exactly one user turn, carrying the tool_result block (correct id, no
    # empty-tool_use_id block) plus the warning as a folded text block.
    user_turns = [m for m in out if m.role == "user"]
    assert len(user_turns) == 1, [m.role for m in out]
    blocks = user_turns[0].raw_content
    tool_use_ids = [b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"]
    assert tool_use_ids == ["call_9"], tool_use_ids
    text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
    assert any("STRATEGY WARNING" in t for t in text_blocks), blocks


def test_gemini_warning_folded_keeps_user_turns_alternating():
    loop = _loop("google")
    response = {"content": "patching", "raw": types.SimpleNamespace(
        raw_response={"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "apply_patch", "args": {}}}]}}]})}
    tool_result_messages = [_warn("[STRATEGY WARNING] switch approach"),
                            LLMMessage(role="tool", name="apply_patch", tool_call_id="x", content='{"ok": false}')]

    out = loop._append_native_tool_messages([], response, tool_result_messages)

    _assert_roles_alternate([m.role for m in out])
    user_turns = [m for m in out if m.role == "user"]
    assert len(user_turns) == 1
    parts = user_turns[0].raw_content
    assert any("functionResponse" in p for p in parts)
    assert any(p.get("text", "").find("STRATEGY WARNING") >= 0 for p in parts), parts


def test_no_warnings_is_unchanged_shape():
    loop = _loop("deepseek")
    response = _response_with_tool_call("call_2")
    out = loop._append_native_tool_messages([], response, [_tool_result("call_2")])
    assert [m.role for m in out] == ["assistant", "tool"]
