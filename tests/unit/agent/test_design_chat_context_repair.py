"""Regression: DCL _apply_context_hard_cap must repair orphaned tool messages
left by count-based preemptive_trim.

preemptive_trim slices ``messages[:1] + messages[-(N+1):]`` without regard for
assistant(tool_calls) <-> role="tool" pairs. When the trim boundary splits such
a pair, the kept tail starts with an orphaned role="tool" message whose
preceding assistant(tool_calls) was dropped. OpenAI/DeepSeek/Anthropic reject
this with HTTP 400 ("orphaned tool_result" / "messages must alternate").

AgentLoop applies repair_tool_message_sequence after its own trim; DCL must do
the same in _apply_context_hard_cap (which feeds the main tool-loop LLM call at
design_chat_loop.py:993 -> the API call at :1010).
"""
import external_llm.agent.design_chat_loop as dcl
from external_llm.client import LLMMessage


# ── helpers ────────────────────────────────────────────────────────────────

def _asst(call_ids: list[str], content: str = "") -> LLMMessage:
    return LLMMessage(
        role="assistant", content=content,
        tool_calls=[
            {"id": cid, "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}
            for cid in call_ids
        ],
    )


def _tool(call_id: str, content: str = "result") -> LLMMessage:
    return LLMMessage(role="tool", content=content, tool_call_id=call_id,
                      name="read_file")


def is_http400_safe(messages: list) -> bool:
    """Independent oracle for the OpenAI/DeepSeek message invariant:

    - No role="tool" message may appear without an immediately-preceding
      assistant(tool_calls) that owns it (i.e. no orphans).
    - Every assistant(tool_calls) must be immediately followed by tool messages
      covering ALL of its tool_call_ids (no missing responses).
    """
    i = 0
    while i < len(messages):
        m = messages[i]
        role = getattr(m, "role", "")
        if role == "tool":
            return False  # orphan — tool not preceded by owning assistant
        if role == "assistant":
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                j = i + 1
                while j < len(messages) and getattr(messages[j], "role", "") == "tool":
                    j += 1
                tool_msgs = messages[i + 1:j]
                if not tool_msgs:
                    return False  # assistant(tool_calls) with no responses
                expected = {t.get("id") for t in tcs
                            if isinstance(t, dict) and t.get("id")}
                actual = {getattr(tm, "tool_call_id", None) for tm in tool_msgs}
                ev = bool(expected)
                av = any(tid for tid in actual)
                if ev and av and expected != actual:
                    return False  # tool_call_id mismatch
                i = j
                continue
        i += 1
    return True


# ── tests ──────────────────────────────────────────────────────────────────

class TestHardCapRepairsOrphanedToolAfterTrim:
    """The core fix: repair_tool_message_sequence runs after preemptive_trim
    inside _apply_context_hard_cap, covering all three call sites (208/993/1415)."""

    def test_decisive_orphan_dropped(self, monkeypatch):
        """preemptive_trim drops the owning assistant but keeps its tool response
        (the orphan). _apply_context_hard_cap must drop it, else HTTP 400."""
        system = LLMMessage(role="system", content="sys")
        orphan = _tool("tc1", "result_for_trimmed_assistant")
        asst2 = _asst(["tc2"])
        tool2 = _tool("tc2", "result_tc2")

        # Simulate count-based preemptive_trim keeping [system, orphan, asst2, tool2]
        # — exactly what happens when the slice boundary splits the tc1 pair.
        monkeypatch.setattr(dcl, "preemptive_trim",
                            lambda msgs, **kw: [system, orphan, asst2, tool2])
        monkeypatch.setattr(dcl, "estimate_tokens_from_msgs", lambda msgs: 10_000_000)

        result = dcl._apply_context_hard_cap(
            [system, orphan, asst2, tool2], model="deepseek-chat")

        assert is_http400_safe(result), (
            f"orphaned tool survived trim — would cause HTTP 400: "
            f"{[getattr(m, 'role', '') for m in result]}")
        roles = [getattr(m, "role", "") for m in result]
        assert roles[0] == "system"
        # the valid tc2 pair survives
        assert "assistant" in roles and "tool" in roles

    def test_split_multi_tool_pair_dropped(self, monkeypatch):
        """assistant(tool_calls=[tc1, tc2]) where only tool(tc1) survives the trim
        (tc2's response dropped). Incomplete tool responses -> HTTP 400; repair
        must drop the whole group."""
        system = LLMMessage(role="system", content="sys")
        asst = _asst(["tc1", "tc2"])
        partial_tool = _tool("tc1", "only-one-of-two")  # tc2's response trimmed away
        followup = LLMMessage(role="user", content="next")

        monkeypatch.setattr(dcl, "preemptive_trim",
                            lambda msgs, **kw: [system, asst, partial_tool, followup])
        monkeypatch.setattr(dcl, "estimate_tokens_from_msgs", lambda msgs: 10_000_000)

        result = dcl._apply_context_hard_cap(
            [system, asst, partial_tool, followup], model="deepseek-chat")

        assert is_http400_safe(result), (
            f"incomplete tool group survived — would cause HTTP 400: "
            f"{[getattr(m, 'role', '') for m in result]}")

    def test_valid_pair_preserved_not_over_dropped(self, monkeypatch):
        """A *valid* assistant(tool_calls)+tool pair after trim must be kept — repair
        must not over-aggressively drop legitimate pairs."""
        system = LLMMessage(role="system", content="sys")
        asst = _asst(["tc1"])
        tool = _tool("tc1", "ok")

        monkeypatch.setattr(dcl, "preemptive_trim",
                            lambda msgs, **kw: [system, asst, tool])
        monkeypatch.setattr(dcl, "estimate_tokens_from_msgs", lambda msgs: 10_000_000)

        result = dcl._apply_context_hard_cap([system, asst, tool], model="deepseek-chat")
        roles = [getattr(m, "role", "") for m in result]
        assert roles == ["system", "assistant", "tool"]

    def test_plain_messages_no_tool_unchanged(self, monkeypatch):
        """No tool messages -> repair is a no-op; plain paths (call sites 208/1415,
        which operate on _strip_tool_messages output) are unaffected."""
        msgs = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="hello"),
            LLMMessage(role="assistant", content="hi"),
        ]
        monkeypatch.setattr(dcl, "preemptive_trim", lambda msgs, **kw: msgs)
        monkeypatch.setattr(dcl, "estimate_tokens_from_msgs", lambda msgs: 10_000_000)

        result = dcl._apply_context_hard_cap(msgs, model="deepseek-chat")
        assert [getattr(m, "role", "") for m in result] == ["system", "user", "assistant"]

    def test_under_cap_no_trim_no_repair(self):
        """When under cap, no trim/repair runs — messages returned as-is."""
        msgs = [LLMMessage(role="system", content="sys"),
                LLMMessage(role="user", content="hi")]
        result = dcl._apply_context_hard_cap(msgs, model="deepseek-chat")
        assert result is msgs or result == msgs


class TestRealPreemptiveTrimOrphan:
    """End-to-end: a real (un-mocked) preemptive_trim that splits a pair is repaired."""

    def test_large_session_split_pair_repaired(self):
        """A long orchestration session where real preemptive_trim naturally splits
        an assistant(tool_calls)<->tool pair must produce a repaired (HTTP-400-safe)
        sequence. deepseek-r1: limit=64000, cap~59904 -> need est>59904 (>~180k chars)."""
        system = LLMMessage(role="system", content="sys")
        big_filler = LLMMessage(role="user", content="Z" * 200_000)  # pushes over cap
        asst1 = _asst(["tc1"], content="analyze")     # trimmed (mid-list)
        tool1 = _tool("tc1", "tool_result_tc1")        # orphan candidate
        asst2 = _asst(["tc2"], content="next")
        tool2 = _tool("tc2", "tool_result_tc2")

        msgs = [system, big_filler, asst1, tool1, asst2, tool2]
        result = dcl._apply_context_hard_cap(msgs, model="deepseek-r1")

        assert is_http400_safe(result), (
            f"real trim produced an orphan that survived — would cause HTTP 400: "
            f"{[getattr(m, 'role', '') for m in result]}")
        roles = [getattr(m, "role", "") for m in result]
        assert roles[0] == "system"
        # the big filler must have been trimmed (otherwise nothing happened)
        assert "Z" * 100 not in (getattr(result[1], "content", "") or "")
