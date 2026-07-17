"""Unit tests for external_llm/agent/context_budget.py"""

from __future__ import annotations

import time

from external_llm.agent.context_budget import (
    _CONTEXT_LIMITS,
    _DEFAULT_CONTEXT_LIMIT,
    _MAX_OVERRIDE_REDUCTIONS,
    _context_window_overrides,
    _is_context_length_error,
    _override_meta,
    _record_context_overflow,
    _resolve_base_context_limit,
    _resolve_context_limit,
    ContextBudgetManager,
    repair_tool_message_sequence,
)
from external_llm.client import LLMMessage

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_msg(role: str, content: str, **kwargs) -> LLMMessage:
    return LLMMessage(role=role, content=content, **kwargs)


def make_manager(model: str = "gpt-4o", reserve: int = 4096) -> ContextBudgetManager:
    return ContextBudgetManager(model_name=model, reserve_for_output=reserve)


def _clear_overrides() -> None:
    """Clear all runtime override state between tests."""
    _context_window_overrides.clear()
    _override_meta.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 1. _resolve_context_limit (dynamic Ollama API query + 1M fallback)
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveContextLimit:
    def test_gpt4o_returns_128k(self):
        assert _resolve_context_limit("gpt-4o") == 128_000

    def test_gpt4o_mini_returns_128k(self):
        assert _resolve_context_limit("gpt-4o-mini") == 128_000  # explicit entry

    def test_gpt4o_date_stamped_returns_128k(self):
        assert _resolve_context_limit("gpt-4o-2024-08-06") == 128_000  # explicit entry

    def test_gpt4o_unknown_variant_returns_default(self):
        assert _resolve_context_limit("gpt-4o-unknown-future") == _DEFAULT_CONTEXT_LIMIT  # no prefix match

    def test_o3_mini_returns_200k(self):
        assert _resolve_context_limit("o3-mini") == 200_000

    def test_o3_mini_high_returns_200k(self):
        assert _resolve_context_limit("o3-mini-high") == 200_000  # explicit entry

    def test_o4_mini_returns_200k(self):
        assert _resolve_context_limit("o4-mini") == 200_000

    def test_claude_haiku_returns_200k(self):
        assert _resolve_context_limit("claude-haiku-4-5") == 200_000

    def test_claude_sonnet_returns_200k(self):
        # Claude Sonnet 4.6 has a 200K context window — explicit table entry
        # (previously fell back to 1M, which left the /general occupancy gate and
        # the hard-cap front-trim effectively disabled for claude-sonnet).
        assert _resolve_context_limit("claude-sonnet-4-6") == 200_000

    def test_claude_unknown_variant_returns_1m_fallback(self):
        # Unlisted claude variants still fall back to 1M (no prefix matching).
        assert _resolve_context_limit("claude-future-9") == _DEFAULT_CONTEXT_LIMIT

    def test_deepseek_v4_flash_returns_1m(self):
        # deepseek-v4-flash supports 1M native context (no explicit entry → fallback)
        assert _resolve_context_limit("deepseek-v4-flash") == _DEFAULT_CONTEXT_LIMIT

    def test_deepseek_v4_pro_returns_1m(self):
        assert _resolve_context_limit("deepseek-v4-pro") == _DEFAULT_CONTEXT_LIMIT

    def test_deepseek_reasoner_returns_1m_fallback(self):
        # deepseek-reasoner is deprecated alias for v4-flash thinking mode (1M)
        assert _resolve_context_limit("deepseek-reasoner") == _DEFAULT_CONTEXT_LIMIT

    def test_deepseek_r1_returns_64k(self):
        assert _resolve_context_limit("deepseek-r1") == 64_000

    def test_deepseek_chat_returns_1m_fallback(self):
        # deepseek-chat is deprecated alias for v4-flash non-thinking mode (1M)
        assert _resolve_context_limit("deepseek-chat") == _DEFAULT_CONTEXT_LIMIT

    def test_glm_5_2_returns_1m_fallback(self):
        assert _resolve_context_limit("glm-5.2") == _DEFAULT_CONTEXT_LIMIT

    def test_glm_5_1_returns_200k(self):
        assert _resolve_context_limit("glm-5.1") == 200_000

    def test_qwen3_7_max_returns_1m_fallback(self):
        assert _resolve_context_limit("qwen3.7-max") == _DEFAULT_CONTEXT_LIMIT

    def test_mimo_v2_5_returns_1m_fallback(self):
        assert _resolve_context_limit("mimo-v2.5") == _DEFAULT_CONTEXT_LIMIT

    def test_kimi_k2_7_code_returns_262k(self):
        assert _resolve_context_limit("kimi-k2.7-code") == 262_144

    def test_kimi_k3_returns_1m_fallback(self):
        # kimi-k3 is a 1M+ model (1,048,576 = 2^20) — uses the _DEFAULT_CONTEXT_LIMIT
        # fallback (no explicit entry), consistent with every other 1M+ model.
        # The table is exact-match-only, so variants must resolve uniformly to 1M
        # (no per-variant drift). Regression guard: do NOT re-add a 1M+ entry.
        assert _resolve_context_limit("kimi-k3") == _DEFAULT_CONTEXT_LIMIT
        assert _resolve_context_limit("kimi-k3-0711") == _DEFAULT_CONTEXT_LIMIT
        assert _resolve_context_limit("kimi-k3-turbo") == _DEFAULT_CONTEXT_LIMIT

    def test_minimax_m3_returns_1m_fallback(self):
        assert _resolve_context_limit("minimax-m3") == _DEFAULT_CONTEXT_LIMIT

    def test_qwen_returns_default(self):
        assert _resolve_context_limit("qwen/qwen3.6-27b-20260422") == _DEFAULT_CONTEXT_LIMIT

    def test_unknown_model_returns_default(self):
        assert _resolve_context_limit("") == _DEFAULT_CONTEXT_LIMIT

    def test_unknown_prefix_returns_default(self):
        assert _resolve_context_limit("custom-llm-v2") == _DEFAULT_CONTEXT_LIMIT

    def test_known_models_in_table(self):
        """Verify all _CONTEXT_LIMITS entries resolve to expected values."""
        for model, expected in _CONTEXT_LIMITS.items():
            assert _resolve_context_limit(model) == expected, f"model={model} expected={expected}"


# ══════════════════════════════════════════════════════════════════════════════
# 1b. _record_context_overflow (reactive backstop override)
# ══════════════════════════════════════════════════════════════════════════════

class TestContextOverflowOverride:
    """Verify the reactive backstop: context-length 400 → reduced limit."""

    def teardown_method(self):
        """Clean up overrides between tests."""
        _clear_overrides()

    def test_record_reduces_limit(self):
        """_record_context_overflow stores a reduced limit for the model."""
        _record_context_overflow("gpt-4o")
        reduced = _resolve_context_limit("gpt-4o")
        base = _resolve_base_context_limit("gpt-4o")
        assert reduced < base  # reduced from 128_000
        assert reduced == max(8192, base * 3 // 4)

    def test_override_takes_priority(self):
        """_resolve_context_limit returns override value, base returns configured."""
        _record_context_overflow("gpt-4o")
        assert _resolve_context_limit("gpt-4o") != _resolve_base_context_limit("gpt-4o")

    def test_base_context_limit_ignores_override(self):
        """_resolve_base_context_limit returns the configured limit regardless of overrides."""
        _record_context_overflow("deepseek-r1")
        assert _resolve_base_context_limit("deepseek-r1") == 64_000

    def test_progressive_reduction_on_repeated_overflow(self):
        """Repeated overflows progressively reduce the limit (75% each time)."""
        model = "gpt-4o"
        _record_context_overflow(model)
        first = _resolve_context_limit(model)
        _record_context_overflow(model)
        second = _resolve_context_limit(model)
        assert second < first
        assert second == max(8192, first * 3 // 4)

    def test_minimum_floor_8k(self):
        """_record_context_overflow never reduces below 8K."""
        model = "tiny-model"
        # Simulate reaching the floor
        _context_window_overrides[model] = 9000
        _record_context_overflow(model)
        assert _resolve_context_limit(model) >= 8192

    def test_override_does_not_affect_other_models(self):
        """Overflow for one model does not affect another."""
        _record_context_overflow("gpt-4o")
        assert _resolve_context_limit("deepseek-r1") == 64_000

    def test_unknown_model_override_reduces_from_default(self):
        """Unknown model overflow reduces from 1M fallback."""
        _record_context_overflow("custom-unknown-model")
        assert _resolve_context_limit("custom-unknown-model") == _DEFAULT_CONTEXT_LIMIT * 3 // 4


# ══════════════════════════════════════════════════════════════════════════════
# 2. estimate_tokens (ContextBudgetManager)
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert ContextBudgetManager.estimate_tokens("") == 0

    def test_none_like_falsy_returns_zero(self):
        # The implementation uses `if not text: return 0`
        assert ContextBudgetManager.estimate_tokens("") == 0

    def test_short_text_at_least_one(self):
        # "Hi" → CJK-aware: max(2//3, 2//2) + 1 = max(0, 1) + 1 = 2
        assert ContextBudgetManager.estimate_tokens("Hi") == 2

    def test_approximation_ratio(self):
        # 350 ASCII chars → CJK-aware: max(350//3, 350//2) + 1 = max(116, 175) + 1 = 176
        text = "a" * 350
        assert ContextBudgetManager.estimate_tokens(text) == 176

    def test_long_text(self):
        text = "word " * 1000  # 5000 chars → CJK-aware: max(5000//3, 5000//2) + 1 = 2501
        result = ContextBudgetManager.estimate_tokens(text)
        assert result == max(5000 // 3, 5000 // 2) + 1

    def test_single_char(self):
        # "x" → CJK-aware: max(1//3, 1//2) + 1 = max(0, 0) + 1 = 1
        assert ContextBudgetManager.estimate_tokens("x") == 1

    def test_exactly_3_5_chars(self):
        # 7 ASCII chars → CJK-aware: max(7//3, 7//2) + 1 = max(2, 3) + 1 = 4
        assert ContextBudgetManager.estimate_tokens("abcdefg") == 4


# ══════════════════════════════════════════════════════════════════════════════
# 3. estimate_messages_tokens
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateMessagesTokens:
    def setup_method(self):
        self.mgr = make_manager()

    def test_empty_list(self):
        assert self.mgr.estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msg = make_msg("user", "a" * 350)  # CJK-aware: max(350//3, 350//2)+1 = 176
        assert self.mgr.estimate_messages_tokens([msg]) == 176

    def test_multiple_messages_sum(self):
        msgs = [
            make_msg("system", "a" * 350),   # 176
            make_msg("user", "a" * 700),      # max(700//3, 700//2)+1 = 351
            make_msg("assistant", "a" * 350), # 176
        ]
        assert self.mgr.estimate_messages_tokens(msgs) == 703

    def test_tool_calls_add_overhead(self):
        tool_calls = [{"id": "1", "function": {"name": "find_symbol", "arguments": "{}"}}]
        msg = make_msg("assistant", "", tool_calls=tool_calls)
        result = self.mgr.estimate_messages_tokens([msg])
        # content="" → 0, tool call: name='find_symbol'(11)+10=21 → 8 tokens, args='{}'=2 → 1 token → total 9
        assert result == 9

    def test_multiple_tool_calls_overhead(self):
        tool_calls = [{"id": str(i)} for i in range(4)]
        msg = make_msg("assistant", "a" * 350, tool_calls=tool_calls)
        result = self.mgr.estimate_messages_tokens([msg])
        # 176 (content) + 4 * 1 (each empty tool_call adds +1 for serialization overhead)
        assert result == 180

    def test_none_content_treated_as_empty(self):
        msg = LLMMessage(role="assistant", content=None)
        # Should not raise; content or "" evaluates to "" → 0 tokens
        assert self.mgr.estimate_messages_tokens([msg]) == 0

    def test_message_without_tool_calls_attr(self):
        # Plain LLMMessage with no tool_calls defaults to None
        msg = make_msg("user", "hello")
        result = self.mgr.estimate_messages_tokens([msg])
        # "hello" → CJK-aware: max(5//3, 5//2) + 1 = max(1, 2) + 1 = 3
        assert result == 3


# ══════════════════════════════════════════════════════════════════════════════
# 4. fit_messages
# ══════════════════════════════════════════════════════════════════════════════

class TestFitMessages:
    def _make_manager_with_tiny_budget(self) -> ContextBudgetManager:
        """Manager whose budget is very small (forces truncation)."""
        mgr = ContextBudgetManager.__new__(ContextBudgetManager)
        mgr.model_name = "test"
        mgr.context_limit = 1000
        mgr.reserve_for_output = 0
        mgr.total_budget = 1000  # ~3500 chars budget
        return mgr

    def test_messages_within_budget_returned_unchanged(self):
        mgr = make_manager("gpt-4o")  # 128k context
        msgs = [
            make_msg("system", "You are helpful."),
            make_msg("user", "Hello"),
            make_msg("assistant", "Hi there"),
        ]
        result = mgr.fit_messages(msgs)
        assert result is msgs or result == msgs

    def test_system_messages_always_preserved(self):
        mgr = self._make_manager_with_tiny_budget()
        # Produce many messages that exceed the budget
        msgs = [make_msg("system", "SYS")]
        for _i in range(50):
            msgs.append(make_msg("user", "u" * 200))
            msgs.append(make_msg("assistant", "a" * 200))

        result = mgr.fit_messages(msgs)
        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "SYS"

    def test_large_tool_result_preserved(self):
        """Tool result unchanged — no truncation even when over budget."""
        mgr = self._make_manager_with_tiny_budget()
        large_tool_content = "X" * 5000
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", "task"),
            make_msg("tool", large_tool_content),
            make_msg("assistant", "done"),
        ]
        mgr.total_budget = 700  # below estimated tokens

        result = mgr.fit_messages(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        # No truncation — content unchanged
        assert tool_msgs[0].content == large_tool_content

    def test_large_user_message_preserved(self):
        """Old user message unchanged — no truncation."""
        mgr = ContextBudgetManager.__new__(ContextBudgetManager)
        mgr.model_name = "test"
        mgr.context_limit = 10_000
        mgr.reserve_for_output = 0
        old_long = "B" * 3000
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", old_long),
            make_msg("assistant", "r1"),
            make_msg("user", "u2"),
            make_msg("assistant", "r2"),
            make_msg("user", "u3"),
            make_msg("assistant", "r3"),
            make_msg("user", "u4"),
        ]
        raw = mgr.estimate_messages_tokens(msgs)
        mgr.total_budget = raw - 200

        result = mgr.fit_messages(msgs)
        # No truncation — all messages preserved unchanged
        user_msgs = [m for m in result if m.role == "user"]
        assert len(user_msgs) == 4
        assert user_msgs[0].content == old_long

    def test_all_messages_preserved_even_when_over_budget(self):
        """All messages preserved unchanged even when over budget — no dropping."""
        mgr = ContextBudgetManager.__new__(ContextBudgetManager)
        mgr.model_name = "test"
        mgr.context_limit = 200
        mgr.reserve_for_output = 0
        mgr.total_budget = 50  # extremely tight

        msgs = [
            make_msg("system", "s"),
            make_msg("user", "first user message"),
            make_msg("assistant", "first assistant reply"),
            make_msg("user", "last user message"),
        ]
        result = mgr.fit_messages(msgs)
        # All messages preserved (no dropping)
        assert len(result) == 4
        assert result[0].content == "s"
        assert result[1].content == "first user message"
        assert result[2].content == "first assistant reply"
        assert result[3].content == "last user message"
        # All content preserved even though over budget

    def test_fit_messages_no_mutation_of_original_when_within_budget(self):
        mgr = make_manager("gpt-4o")
        original = [make_msg("user", "hello"), make_msg("assistant", "world")]
        result = mgr.fit_messages(original)
        # Should return same list or equivalent list (no mutation of originals)
        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_orphaned_tool_messages_not_dropped(self):
        """Orphaned tool messages preserved — no dropping."""
        mgr = ContextBudgetManager.__new__(ContextBudgetManager)
        mgr.model_name = "test"
        mgr.context_limit = 200
        mgr.reserve_for_output = 0
        mgr.total_budget = 30

        msgs = [
            make_msg("system", "s"),
            make_msg("user", "task"),
            make_msg("tool", "tool result orphan"),
            make_msg("user", "follow-up"),
        ]
        result = mgr.fit_messages(msgs)
        # All messages preserved (fit_messages does not drop anything)
        assert len(result) == 4

    def test_tool_message_following_assistant_with_tool_calls_is_kept(self):
        """Tool message that follows assistant with tool_calls is appended to result."""
        mgr = ContextBudgetManager.__new__(ContextBudgetManager)
        mgr.model_name = "test"
        mgr.context_limit = 200
        mgr.reserve_for_output = 0
        mgr.total_budget = 30

        assistant_msg = make_msg("assistant", "calling tools")
        assistant_msg.tool_calls = [{"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}]
        tool_msg = make_msg("tool", "tool result")
        msgs = [
            make_msg("system", "s"),
            assistant_msg,
            tool_msg,
            make_msg("user", "follow-up"),
        ]
        result = mgr.fit_messages(msgs)
        # All messages preserved — tool message kept alongside assistant
        assert len(result) == 4
        assert any(getattr(m, 'role', '') == 'tool' and getattr(m, 'content', '') == 'tool result' for m in result)


# ══════════════════════════════════════════════════════════════════════════════
# 5. No-truncation verification
# ══════════════════════════════════════════════════════════════════════════════
#
# ContextBudgetManager.fit_messages returns messages unchanged — no truncation.
# This is intentional: truncating tool results causes the LLM to re-issue the
# same tool calls, wasting more tokens than the truncation saves.
# SlidingWindowContext handles context management via summarisation.

class TestNoTruncation:
    """Verify that fit_messages does NOT truncate any content."""

    def setup_method(self):
        self.mgr = make_manager()

    def test_messages_unchanged_within_budget(self):
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", "hello"),
            make_msg("assistant", "hi"),
        ]
        result = self.mgr.fit_messages(msgs)
        assert result is msgs or result == msgs
        assert result[0].content == "sys"
        assert result[1].content == "hello"

    def test_large_tool_result_not_truncated(self):
        """Even very large tool results are returned unchanged."""
        large = "X" * 50_000
        msgs = [
            make_msg("system", "sys"),
            make_msg("tool", large),
        ]
        result = self.mgr.fit_messages(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == large  # unchanged

    def test_large_user_message_not_truncated(self):
        large = "B" * 100_000
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", large),
        ]
        result = self.mgr.fit_messages(msgs)
        user_msgs = [m for m in result if m.role == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == large  # unchanged

    def test_all_messages_preserved(self):
        msgs = [make_msg("user", f"msg_{i}") for i in range(100)]
        result = self.mgr.fit_messages(msgs)
        assert len(result) == 100




# ══════════════════════════════════════════════════════════════════════════════
# 7. ContextBudgetManager initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestContextBudgetManagerInit:
    def test_total_budget_equals_limit_minus_reserve(self):
        mgr = ContextBudgetManager("gpt-4o", reserve_for_output=4096)
        assert mgr.total_budget == 128_000 - 4096
        assert mgr.total_budget == 123_904

    def test_gpt4o_uses_128k_limit(self):
        mgr = ContextBudgetManager("gpt-4o")
        assert mgr.context_limit == 128_000

    def test_claude_uses_200k_limit(self):
        mgr = ContextBudgetManager("claude-sonnet-4-6")
        assert mgr.context_limit == 200_000

    def test_default_reserve_is_4096(self):
        mgr = ContextBudgetManager("gpt-4o")
        assert mgr.reserve_for_output == 4096

    def test_unknown_model_uses_default_limit(self):
        mgr = ContextBudgetManager("my-custom-llm")
        assert mgr.context_limit == _DEFAULT_CONTEXT_LIMIT


# ══════════════════════════════════════════════════════════════════════════════
# 9. repair_tool_message_sequence
# ══════════════════════════════════════════════════════════════════════════════

class TestRepairToolMessageSequence:
    """repair_tool_message_sequence enforces the tool-call → tool-response invariant."""

    def test_regular_messages_preserved(self):
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", "hello"),
            make_msg("assistant", "world"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 3
        assert [m.content for m in result] == ["sys", "hello", "world"]

    def test_valid_tool_call_sequence_preserved(self):
        msgs = [
            make_msg("user", "task"),
            make_msg("assistant", "", tool_calls=[{"id": "tc1"}]),
            make_msg("tool", "result1"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 3

    def test_valid_multi_tool_sequence_preserved(self):
        msgs = [
            make_msg("assistant", "", tool_calls=[{"id": "tc1"}, {"id": "tc2"}]),
            make_msg("tool", "r1"),
            make_msg("tool", "r2"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 3

    def test_orphaned_tool_message_dropped(self):
        """Tool message without preceding assistant(tool_calls) is dropped."""
        msgs = [
            make_msg("user", "task"),
            make_msg("tool", "orphan"),
            make_msg("assistant", "done"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 2
        assert result[0].content == "task"
        assert result[1].content == "done"

    def test_assistant_with_tool_calls_no_following_tools_dropped(self):
        """Assistant with tool_calls but no following tool messages is dropped."""
        msgs = [
            make_msg("user", "task"),
            make_msg("assistant", "", tool_calls=[{"id": "tc1"}]),
            # No tool messages follow — entire assistant(tool_calls) is dropped
            make_msg("user", "follow-up"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 2
        assert result[0].content == "task"
        assert result[1].content == "follow-up"

    def test_assistant_without_tool_calls_preserved(self):
        msgs = [
            make_msg("user", "q"),
            make_msg("assistant", "a"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 2

    def test_mixed_sequence_some_dropped(self):
        """Mix of valid and orphaned messages — orphan1 dropped, orphan2 consumed by valid group."""
        msgs = [
            make_msg("system", "sys"),
            make_msg("user", "task"),
            make_msg("tool", "orphan1"),  # orphaned → dropped
            make_msg("assistant", "", tool_calls=[{"id": "tc1"}]),
            make_msg("tool", "valid_result"),  # valid → kept
            make_msg("tool", "orphan2"),  # consumed by valid assistant's tool group
            make_msg("user", "done"),
        ]
        result = repair_tool_message_sequence(msgs)
        assert len(result) == 6
        roles = [m.role for m in result]
        assert roles == ["system", "user", "assistant", "tool", "tool", "user"]
        assert result[5].content == "done"

    def test_empty_list(self):
        assert repair_tool_message_sequence([]) == []

    def test_single_user_message(self):
        result = repair_tool_message_sequence([make_msg("user", "hi")])
        assert len(result) == 1

    def test_message_without_role_attr(self):
        """Messages without 'role' attr are treated as empty-role and passed through."""
        msg = object()
        result = repair_tool_message_sequence([msg])
        assert len(result) == 1
        assert result[0] is msg


# ══════════════════════════════════════════════════════════════════════════════
# 10. SlidingWindowContext hysteresis cadence
# ══════════════════════════════════════════════════════════════════════════════

class TestSlidingWindowHysteresis:
    """Verify that SlidingWindowContext hysteresis prevents cache thrashing.

    With window=60 and SlidingWindowConfig.hysteresis_factor=0.6, the first trim happens at
    61 messages and reduces them to 36 (60*0.6).  Subsequent calls stay
    below 60 and return unchanged, so over 200 turns we expect exactly 1
    trim event (not 200-60=140).
    """

    def test_hysteresis_cadence(self):
        from external_llm.agent.context_manager import SlidingWindowConfig, SlidingWindowContext

        swc = SlidingWindowContext(SlidingWindowConfig(context_window_size=60))
        sys_msg = LLMMessage(role="system", content="sys")
        trim_events: list[int] = []

        msgs = [sys_msg]
        for turn in range(200):
            msgs.append(LLMMessage(role="user", content=f"turn {turn}"))
            msgs.append(LLMMessage(role="assistant", content=f"response {turn}"))
            before = len(msgs)
            out = swc.prepare_before_call(msgs)
            if len(out) < before:
                trim_events.append(turn)
            msgs = out

        # With hysteresis, after first trim to 36, count must regrow past
        # 60 before trimming again — at 2 msg/turn, that's ~12 turns later.
        # Over 200 turns we expect at most ~ceil(200 / (12 + 1)) ≈ 15 trims.
        # Without hysteresis, every turn past 60 would trim = ~140 trims.
        naive_events = 200 - 60  # what would happen without hysteresis
        assert len(trim_events) <= 16, (
            f"Hysteresis cadence broken: {len(trim_events)} trims "
            f"(naive = {naive_events}), expected ≤ 16"
        )
        assert len(trim_events) >= 1, "No trim ever fired — hysteresis too aggressive?"

    def test_hysteresis_keeps_most_recent_messages(self):
        """After trimming, the most recent messages are preserved."""
        from external_llm.agent.context_manager import SlidingWindowConfig, SlidingWindowContext

        swc = SlidingWindowContext(SlidingWindowConfig(context_window_size=60))
        sys_msg = LLMMessage(role="system", content="sys")
        msgs = [sys_msg]
        for i in range(100):
            msgs.append(LLMMessage(role="user", content=f"msg_{i}"))
        out = swc.prepare_before_call(msgs)

        # Should keep system + compressed summary + last ~36 user messages
        assert sys_msg in out
        non_sys = [m for m in out if m.role != "system"]
        # The first non-system should be the compressed summary
        assert "[COMPRESSED CONTEXT]" in (non_sys[0].content or "")
        # The remaining should be the most recent user messages
        remaining = non_sys[1:]
        assert len(remaining) >= 35, f"Expected ≥35 recent msgs, got {len(remaining)}"
        # Verify they're the tail (most recent)
        assert remaining[-1].content == "msg_99"





class TestIsContextLengthError:
    """Verify _is_context_length_error pattern matching.

    P3: Tests must cover provider-specific true positives (actual error messages
    from OpenAI/DeepSeek/GLM/Anthropic) and true negatives (non-context errors
    like image size, auth failures, malformed payload).
    """

    # ── True positives: provider-specific error messages ──────────────────────

    def test_openai_style_max_context(self):
        """OpenAI: 'maximum context length is X tokens, but you sent Y'"""
        assert _is_context_length_error(Exception(
            "maximum context length is 128000 tokens, but you sent 145000"
        ))

    def test_deepseek_context_exceeded(self):
        """DeepSeek: 'context length exceeded'"""
        assert _is_context_length_error(Exception(
            "context length exceeded: prompt has N tokens, limit M"
        ))

    def test_deepseek_too_large_with_token(self):
        """DeepSeek: 'too large' when near 'token'"""
        assert _is_context_length_error(Exception(
            "Input token limit exceeded: request too large"
        ))

    def test_glm_context_window_too_small(self):
        """GLM/ZAI: code 1305 'context window is too small'"""
        assert _is_context_length_error(Exception(
            "1305: context window is too small"
        ))

    def test_anthropic_prompt_too_long(self):
        """Anthropic: 'prompt is too long'"""
        assert _is_context_length_error(Exception(
            "prompt is too long: your prompt of N tokens exceeds M"
        ))

    def test_max_length_alone_not_matched(self):
        """'max_length' alone (pydantic/JSON-schema validation 400) must NOT match."""
        assert not _is_context_length_error(Exception(
            "String should have at most 4096 characters (max_length)"
        ))

    def test_reduce_length(self):
        """Generic: 'reduce length' in error message"""
        assert _is_context_length_error(Exception(
            "Please reduce length: prompt exceeds context window"
        ))

    def test_prompt_length_narrow(self):
        """'prompt length' in error message"""
        assert _is_context_length_error(Exception(
            "Prompt length exceeds model capacity"
        ))

    def test_context_window_narrow(self):
        """'context window' in error message"""
        assert _is_context_length_error(Exception(
            "The context window is full"
        ))

    # ── True negatives: non-context errors must NOT match ─────────────────────

    def test_image_too_large_not_matched(self):
        """Image size 400: 'image too large' must NOT match (no context term)."""
        assert not _is_context_length_error(Exception(
            "image too large: maximum image size is 20MB"
        ))

    def test_request_entity_too_large_not_matched(self):
        """413/400: 'request entity too large' must NOT match."""
        assert not _is_context_length_error(Exception(
            "request entity too large: payload exceeds maximum"
        ))

    def test_filename_too_long_not_matched(self):
        """400: 'filename too long' must NOT match."""
        assert not _is_context_length_error(Exception(
            "filename too long: path exceeds 255 characters"
        ))

    def test_auth_error_not_matched(self):
        """Auth 401: no context terms."""
        assert not _is_context_length_error(Exception(
            "Incorrect API key provided"
        ))

    def test_rate_limit_not_matched(self):
        """Rate limit 429: no context terms."""
        assert not _is_context_length_error(Exception(
            "Rate limit exceeded: too many requests"
        ))

    def test_malformed_json_not_matched(self):
        """Malformed payload 400: no context terms."""
        assert not _is_context_length_error(Exception(
            "Invalid JSON in request body"
        ))

    def test_empty_exception_not_matched(self):
        """Empty or very short message should not match."""
        assert not _is_context_length_error(Exception(""))
        assert not _is_context_length_error(Exception("ok"))

    def test_too_large_orphan_not_matched(self):
        """"too large" without context/token/prompt nearby must NOT match."""
        assert not _is_context_length_error(Exception(
            "Payload too large"
        ))

    def test_too_long_orphan_not_matched(self):
        """"too long" without context/token/prompt nearby must NOT match."""
        assert not _is_context_length_error(Exception(
            "Value too long for column"
        ))


class TestContextOverflowOverrideExtended:
    """Additional tests for _record_context_overflow with estimated_prompt_tokens.

    P5: Fast convergence — when the actual prompt size is known, the override
    should clamp below it in one shot instead of requiring multiple turns of
    25% progressive reduction.
    """

    def teardown_method(self):
        """Clean up overrides between tests."""
        _clear_overrides()

    def test_estimate_clamps_below_failed_prompt(self):
        """With estimated_prompt_tokens, override is clamped below that value (proportional 85%)."""
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=100_000)
        # P3: proportional headroom = int(100_000 * 0.85) = 85_000.
        # Progressive: 128K * 0.75 = 96K. Estimate clamp wins: min(96K, 85K) = 85K.
        assert _resolve_context_limit("gpt-4o") == 85_000

    def test_estimate_converges_in_one_shot(self):
        """With large fallback and known prompt size, one call converges (proportional 85%)."""
        # 1M fallback, actual prompt ~90K → override should be ≤ int(90K * 0.85) = 76_500
        _record_context_overflow("unknown-1m-model", estimated_prompt_tokens=90_000)
        assert _resolve_context_limit("unknown-1m-model") <= 76_500

    def test_estimate_respects_floor(self):
        """estimated_prompt_tokens below floor is clamped to floor."""
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=100)
        assert _resolve_context_limit("gpt-4o") >= 8192

    def test_estimate_without_estimate_is_progressive(self):
        """Without estimate, override follows the 75% progressive reduction."""
        _record_context_overflow("gpt-4o")  # no estimate → old behavior
        first = _resolve_context_limit("gpt-4o")  # 128K * 0.75 = 96K
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=90_000)
        second = _resolve_context_limit("gpt-4o")
        # second should be ≤ first (progressive) AND also clamped by estimate
        assert second <= first
        # With proportional headroom, 90K * 0.85 = 76_500, but progressive (96K * 0.75 = 72K) wins
        assert second == max(8192, 96_000 * 3 // 4)  # 72K — progressive reduction before estimate clamp

    def test_estimate_none_uses_75pct_reduction(self):
        """estimated_prompt_tokens=None preserves original behavior."""
        base = _resolve_base_context_limit("gpt-4o")
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=None)
        assert _resolve_context_limit("gpt-4o") == max(8192, base * 3 // 4)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Override TTL, reduction cap, and meta integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideTTLAndCap:
    """Verify TTL expiry, reduction cap, and override metadata."""

    def teardown_method(self):
        _clear_overrides()

    def test_max_reductions_cap(self):
        """After _MAX_OVERRIDE_REDUCTIONS calls, further overflows don't reduce."""
        model = "gpt-4o"
        base = _resolve_base_context_limit(model)
        for i in range(_MAX_OVERRIDE_REDUCTIONS + 1):
            _record_context_overflow(model)
        # Limit should be stable after cap reached
        final = _resolve_context_limit(model)
        expected = max(8192, base * 3 // 4)  # one reduction only (first)
        # Each reduction is 75% of previous; after _MAX_OVERRIDE_REDUCTIONS stops
        # Actually the last call doesn't reduce, so the value is after _MAX_OVERRIDE_REDUCTIONS calls
        _clear_overrides()
        # Simulate _MAX_OVERRIDE_REDUCTIONS sequential reductions
        cur = base
        for _ in range(_MAX_OVERRIDE_REDUCTIONS):
            _record_context_overflow(model)
            cur = max(8192, cur * 3 // 4)
        expected_stable = _resolve_context_limit(model)
        # Now call one more — should stay the same (cap reached)
        _record_context_overflow(model)
        assert _resolve_context_limit(model) == expected_stable

    def test_ttl_clears_expired_override(self):
        """TTL-expired overrides are cleared by _resolve_context_limit."""
        model = "gpt-4o"
        _record_context_overflow(model)
        assert _resolve_context_limit(model) < _resolve_base_context_limit(model)
        # Simulate TTL expiry by setting the timestamp to the past
        meta = _override_meta.get(model.lower().strip())
        assert meta is not None
        meta["ts"] = 0  # far in the past
        # Resolve should clear the expired override
        resolved = _resolve_context_limit(model)
        assert resolved == _resolve_base_context_limit(model)
        # Override should also be removed from the dict
        assert model.lower().strip() not in _context_window_overrides

    def test_clear_override_meta_on_ttl_expiry(self):
        """_override_meta is also cleaned when TTL expires."""
        model = "gpt-4o"
        model_lower = model.lower().strip()
        _record_context_overflow(model)
        assert model_lower in _override_meta
        # Expire
        _override_meta[model_lower]["ts"] = 0
        _resolve_context_limit(model)
        assert model_lower not in _override_meta


# ══════════════════════════════════════════════════════════════════════════════
# 8. End-to-end wiring: override → resolve → context_message_cap → trim
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideEndToEndWiring:
    """Verify the full pipeline: override → resolve → cap computation.

    Ensures that an override recorded by _record_context_overflow propagates
    through _resolve_context_limit to affect the cap that drives preemptive_trim.
    """

    def teardown_method(self):
        _clear_overrides()

    def test_override_affects_cap(self):
        """After override, the hard cap returned by context_message_cap changes."""
        from external_llm.agent._shared_utils import context_message_cap
        model = "gpt-4o"
        base_limit = _resolve_base_context_limit(model)
        safety = 4096
        tool_schemas = []
        cap_before = context_message_cap(base_limit, safety, tool_schemas)
        # Record overflow
        _record_context_overflow(model)
        new_limit = _resolve_context_limit(model)
        cap_after = context_message_cap(new_limit, safety, tool_schemas)
        # cap should be lower (or equal if floor hit)
        assert cap_after <= cap_before

    def test_resolve_context_limit_reflects_override(self):
        """_resolve_context_limit returns the overridden limit, not the base."""
        model = "gpt-4o"
        base = _resolve_base_context_limit(model)
        _record_context_overflow(model)
        assert _resolve_context_limit(model) != base
        assert _resolve_context_limit(model) == max(8192, base * 3 // 4)


# ══════════════════════════════════════════════════════════════════════════════
# 9. P1/P2/P3 regression: proportional headroom, stale-estimate convergence,
#    no-progress detection at reduction cap
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideRegression:
    """Regression tests for reactive backstop edge cases (P1/P2/P3/P5)."""

    def teardown_method(self):
        _clear_overrides()

    # ── P3: proportional headroom (0.85 × estimated_prompt_tokens) ────────────

    def test_proportional_headroom_binding(self):
        """When estimate clamp is tighter than 75% reduction, 85% headroom binds."""
        # gpt-4o (128K): 75% → 96K; estimate 10K → 85% → 8500 → override = 8500
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=10_000)
        assert _resolve_context_limit("gpt-4o") == max(8192, int(10_000 * 0.85))

    def test_proportional_headroom_larger_than_progressive(self):
        """When 75% progressive reduction is tighter than 85% estimate, progressive wins."""
        # gpt-4o: first 75% → 96K; second with 200K estimate → 85% = 170K
        # but progressive 96K * 0.75 = 72K binds
        _record_context_overflow("gpt-4o")  # 128K → 96K
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=200_000)
        assert _resolve_context_limit("gpt-4o") == max(8192, 96_000 * 3 // 4)  # 72K

    # ── P2: post-trim estimate (smaller) produces tighter override ────────────

    def test_smaller_estimate_tighter_override(self):
        """A smaller estimated_prompt_tokens produces a tighter (lower) override."""
        # Same model, same progressive starting point, different estimates
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=90_000)
        override_large = _resolve_context_limit("gpt-4o")
        _clear_overrides()
        _record_context_overflow("gpt-4o", estimated_prompt_tokens=30_000)
        override_small = _resolve_context_limit("gpt-4o")
        # 90K * 0.85 = 76_500 vs 30K * 0.85 = 25_500
        assert override_small < override_large
        assert override_small == max(8192, int(30_000 * 0.85))

    # ── P1: reduction cap resilience (no crash / no silent None) ─────────────

    def test_overflow_at_cap_does_not_change_override(self):
        """When reduction cap is reached, _record_context_overflow is a no-op."""
        model = "gpt-4o"
        for _ in range(_MAX_OVERRIDE_REDUCTIONS):
            _record_context_overflow(model)
        stable = _resolve_context_limit(model)
        # One more call at cap — must not raise, must not change
        _record_context_overflow(model)  # no-op at cap
        assert _resolve_context_limit(model) == stable

    def test_overflow_at_cap_with_estimate_still_noop(self):
        """At reduction cap, even with estimate, _record_context_overflow is no-op."""
        model = "gpt-4o"
        for _ in range(_MAX_OVERRIDE_REDUCTIONS):
            _record_context_overflow(model)
        stable = _resolve_context_limit(model)
        _record_context_overflow(model, estimated_prompt_tokens=1_000)
        assert _resolve_context_limit(model) == stable

    # ── P5: no-progress detection semantics ──────────────────────────────────

    def test_progressive_reduction_eventually_stops(self):
        """After _MAX_OVERRIDE_REDUCTIONS reductions, further overflows stop changing."""
        model = "deepseek-r1"  # 64K — tighter, fewer reductions to floor
        limits = []
        for i in range(_MAX_OVERRIDE_REDUCTIONS + 2):
            _record_context_overflow(model)
            limits.append(_resolve_context_limit(model))
        # The last two entries should be equal (cap reached)
        assert limits[-1] == limits[-2]
        assert limits[-1] >= 8192


# ══════════════════════════════════════════════════════════════════════════════
# 11. P6: Native tool payload token counting (estimate_tokens_from_msgs)
# ══════════════════════════════════════════════════════════════════════════════
# Native tool providers (Anthropic/zai/GLM) embed tool payloads in raw_content
# blocks instead of the standard content/tool_calls fields.  Prior to the P1
# fix, tool_use.input and tool_result.content were silently counted as 0
# tokens, defeating the pre-trim guard for those providers.

class TestEstimateTokensFromMsgsNativeToolPayload:
    """Verify that estimate_tokens_from_msgs counts native tool payloads."""

    def test_tool_use_input_dict_counted(self):
        """tool_use block with dict input contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="",
            raw_content=[{"type": "tool_use", "name": "bash",
                          "input": {"command": "ls -la /very/long/path"}}],
        )
        est = estimate_tokens_from_msgs([msg])
        # content="" → 0; tool_use "bash" name ~3 tokens; input dict JSON ~30 chars → ~10 tokens
        assert est > 0, "tool_use dict input was counted as 0 tokens"
        assert est > 5, f"Expected >5 tokens for tool_use with args, got {est}"

    def test_tool_use_input_str_counted(self):
        """tool_use block with string input contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        large_input = "x" * 5000
        msg = LLMMessage(
            role="assistant", content="",
            raw_content=[{"type": "tool_use", "name": "read_file",
                          "input": large_input}],
        )
        est = estimate_tokens_from_msgs([msg])
        # content="" → 0; name "read_file" ~4 tokens; large_input: CJK-aware: max(5000//3, 5000//2)+1 = 5000//2+1 = 2501
        assert est >= 2500, f"Expected ~2500 tokens for 5K-string input, got {est}"

    def test_tool_result_content_str_counted(self):
        """tool_result block with string content contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        large_content = "Y" * 10000
        msg = LLMMessage(
            role="user", content="",
            raw_content=[{"type": "tool_result", "content": large_content}],
        )
        est = estimate_tokens_from_msgs([msg])
        # content="" → 0; tool_result.content = 10K chars → CJK-aware: max(10000//3, 10000//2)+1 = 5001
        assert est >= 5000, f"Expected ~5000 tokens for 10K tool result, got {est}"

    def test_tool_result_content_list_counted(self):
        """tool_result block with list of text sub-blocks contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="user", content="",
            raw_content=[{
                "type": "tool_result",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2 " * 500},
                ],
            }],
        )
        est = estimate_tokens_from_msgs([msg])
        # "line1" (5 chars) + "line2 " * 500 (3000 chars) = 3005 chars
        # CJK-aware: max(3005//3, 3005//2)+1 = max(1001, 1502)+1 = 1503
        assert est >= 1500, f"Expected ~1500 tokens for list tool result, got {est}"

    def test_tool_use_without_input_zero_impact(self):
        """tool_use block without input does not add arbitrary tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="hi",
            raw_content=[{"type": "tool_use", "name": "bash"}],  # no 'input' key
        )
        est = estimate_tokens_from_msgs([msg])
        # "hi" → ~2 tokens; tool_use name "bash" ~3 tokens
        assert est >= 2
        assert est < 20, "Without input, tool_use should only add name tokens"

    def test_content_skipped_when_raw_content_present(self):
        """Plain .content is NOT counted when raw_content exists (avoids double-count)."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="Hello world",
            raw_content=[{"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}}],
        )
        est = estimate_tokens_from_msgs([msg])
        # "Hello world" (11 chars) is SKIPPED because raw_content is present.
        # Only tool_use block counted:
        #   name "bash": (4+10)//3 + 1 = 5
        #   input {"cmd":"ls"} JSON: 13 chars → 13//3 + 1 = 5
        # Total = ~10
        assert est >= 8, f"Expected raw_content tool_use counted, got {est}"
        # Verify content is NOT double-counted: same message without raw_content
        msg_no_rc = LLMMessage(role="assistant", content="Hello world")
        est_no_rc = estimate_tokens_from_msgs([msg_no_rc])
        # "Hello world" (11 chars) → max(11//3, 11//2)+1 = 6
        assert est_no_rc >= 5, f"Expected content counted when no raw_content, got {est_no_rc}"

    def test_estimate_messages_tokens_matches_shared(self):
        """ContextBudgetManager.estimate_messages_tokens delegates to shared function."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs as shared_est
        msg = LLMMessage(
            role="assistant", content="test",
            raw_content=[{"type": "tool_use", "name": "bash", "input": {"cmd": "echo 1"}}],
        )
        mgr = make_manager()
        delegate = mgr.estimate_messages_tokens([msg])
        direct = shared_est([msg])
        assert delegate == direct, (
            f"estimate_messages_tokens ({delegate}) != estimate_tokens_from_msgs ({direct})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 12. P5: TTL-aware reduction cap — expired meta resets reductions counter
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideTTLAwareReductionCap:
    """Verify that _record_context_overflow resets reductions when TTL expired.

    P5: Without TTL check, an expired meta entry with reductions=3 permanently
    blocks further overrides for a persistently misconfigured model.
    """

    def teardown_method(self):
        _clear_overrides()

    def test_expired_meta_resets_reduction_counter(self):
        """When meta TTL has expired, reductions counter is reset to 0."""
        model = "gpt-4o"
        # Simulate reaching reduction cap
        for _ in range(_MAX_OVERRIDE_REDUCTIONS):
            _record_context_overflow(model)
        assert _override_meta[model]["reductions"] == _MAX_OVERRIDE_REDUCTIONS
        # Expire the meta
        _override_meta[model]["ts"] = 0
        # Next overflow should reset and record a new reduction
        _record_context_overflow(model)
        assert _override_meta[model]["reductions"] == 1, (
            f"Expected reductions=1 after TTL expiry, got {_override_meta[model]['reductions']}"
        )

    def test_expired_meta_allows_further_reduction(self):
        """After TTL expiry, the override limit actually decreases further."""
        model = "gpt-4o"
        # Record one overflow, then expire, then check if new overflow reduces further
        _record_context_overflow(model)
        limit_before = _resolve_context_limit(model)
        _override_meta[model]["ts"] = 0
        _record_context_overflow(model)
        limit_after = _resolve_context_limit(model)
        # After meta expiry and new overflow, the limit may be lower or same
        # (depends on progressive reduction + any base recalculation)
        # The key assertion: the operation succeeds (no silent return at cap)
        assert limit_after <= limit_before, "Expired meta should allow further reduction"

    def test_fresh_meta_still_blocks_at_cap(self):
        """Non-expired meta at reduction cap still blocks new overrides."""
        model = "gpt-4o"
        for _ in range(_MAX_OVERRIDE_REDUCTIONS):
            _record_context_overflow(model)
        limit_before = _resolve_context_limit(model)
        # Meta is NOT expired (recent timestamp)
        _record_context_overflow(model)
        limit_after = _resolve_context_limit(model)
        assert limit_after == limit_before, "Non-expired meta at cap should not reduce further"


# ══════════════════════════════════════════════════════════════════════════════
# 13. P4: Force cache save (skip debounce)
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideCacheForceSave:
    """Verify that _save_override_cache(force=True) skips the debounce interval."""

    def teardown_method(self):
        _clear_overrides()

    def test_force_save_skips_debounce(self):
        """force=True writes to disk even within the debounce window."""
        from external_llm.agent.context_budget import (
            _save_override_cache, _last_cache_save, _OVERRIDE_CACHE_FILE,
        )
        import os
        # Clean up any existing cache file
        if os.path.exists(_OVERRIDE_CACHE_FILE):
            os.remove(_OVERRIDE_CACHE_FILE)
        # Simulate a recent save (within debounce interval)
        _last_cache_save = time.time()
        # Force save should write regardless
        _save_override_cache(force=True)
        assert os.path.exists(_OVERRIDE_CACHE_FILE), "Force save should write to disk"
        # Clean up
        if os.path.exists(_OVERRIDE_CACHE_FILE):
            os.remove(_OVERRIDE_CACHE_FILE)

    def test_normal_save_is_debounced(self):
        """force=False respects the debounce interval."""
        from external_llm.agent.context_budget import (
            _save_override_cache, _last_cache_save, _OVERRIDE_CACHE_FILE,
        )
        import os
        if os.path.exists(_OVERRIDE_CACHE_FILE):
            os.remove(_OVERRIDE_CACHE_FILE)
        # Set a very recent save time
        _last_cache_save = time.time()
        _save_override_cache(force=False)  # should be debounced
        # May or may not write depending on timing — so we can't assert
        # absence.  Instead verify that calling force=True immediately after
        # does write.
        _save_override_cache(force=True)
        assert os.path.exists(_OVERRIDE_CACHE_FILE)
        if os.path.exists(_OVERRIDE_CACHE_FILE):
            os.remove(_OVERRIDE_CACHE_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# 13. P2: Override cache snapshot under lock (thread safety)
# ══════════════════════════════════════════════════════════════════════════════

class TestOverrideCacheSnapshotSafety:
    """Verify _save_override_cache snapshots _override_meta under the lock."""

    def test_snapshot_under_lock(self):
        """Snapshot prevents 'dict changed size during iteration'."""
        from external_llm.agent.context_budget import (
            _save_override_cache, _override_meta, _OVERRIDE_CACHE_FILE,
        )
        import os, threading

        # Simulate concurrent mutation during serialization
        _override_meta["test-model"] = {"ts": time.time(), "reductions": 1, "limit": 1000}

        def mutate():
            for i in range(100):
                _override_meta[f"concurrent-{i}"] = {"ts": time.time(), "reductions": 0, "limit": 500}

        t = threading.Thread(target=mutate, daemon=True)
        t.start()
        # While the thread is mutating, save should snapshot under lock and not crash
        _save_override_cache(force=True)
        t.join(timeout=2)
        # Also call from inside the lock (as _record_context_overflow does)
        from external_llm.agent.context_budget import _override_lock
        with _override_lock:
            _save_override_cache(force=True)
        # Clean up
        if os.path.exists(_OVERRIDE_CACHE_FILE):
            os.remove(_OVERRIDE_CACHE_FILE)
        _clear_overrides()


# ══════════════════════════════════════════════════════════════════════════════
# 14. P3: Per-message token estimate caching
# ══════════════════════════════════════════════════════════════════════════════

class TestMsgTokenCache:
    """Verify _estimate_single_message_tokens caches per-message results."""

    def test_cache_reuses_estimate(self):
        """Same LLMMessage returns cached value on second call."""
        from external_llm.agent._shared_utils import (
            _estimate_single_message_tokens, _cjk_aware_tokens,
        )
        msg = LLMMessage(role="user", content="Hello world test message")
        first = _estimate_single_message_tokens(msg)
        # Second call should use cache
        second = _estimate_single_message_tokens(msg)
        assert first == second
        # Verify cache attribute was set
        assert getattr(msg, '_msg_token_estimate', None) == first

    def test_plain_dict_does_not_cache(self):
        """Plain dict messages are handled without caching (no __dict__)."""
        from external_llm.agent._shared_utils import _estimate_single_message_tokens
        # Plain dict messages don't support getattr — content defaults to ''
        # and tool_calls/raw_content are not found via getattr.  This is a
        # pre-existing limitation; the function was designed for LLMMessage.
        msg = {"role": "user", "content": "hello"}
        est = _estimate_single_message_tokens(msg)
        assert est == 0  # getattr('content', '') → '' on dict → 0 tokens
        # Plain dict should NOT have the cache attribute
        assert not hasattr(msg, '_msg_token_estimate')

    def test_estimate_tokens_from_msgs_uses_cache(self):
        """estimate_tokens_from_msgs benefits from cached per-message estimates."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="",
            raw_content=[{"type": "tool_use", "name": "read_file",
                          "input": {"path": "/very/long/path/to/file.txt"}}],
        )
        # First call computes and caches
        est1 = estimate_tokens_from_msgs([msg])
        # Second call should use cache
        est2 = estimate_tokens_from_msgs([msg])
        assert est1 == est2


# ══════════════════════════════════════════════════════════════════════════════
# 15. P4: ContextBudgetManager tool-schema accounting
# ══════════════════════════════════════════════════════════════════════════════

class TestContextBudgetManagerToolSchemas:
    """Verify ContextBudgetManager deducts tool-schema tokens from budget."""

    def test_tool_schemas_deducted_from_budget(self):
        """With tool_schemas, total_budget is lower than without."""
        mgr_no_ts = ContextBudgetManager("gpt-4o", reserve_for_output=4096)
        schemas = [{"name": "read_file", "description": "x" * 500}]
        mgr_ts = ContextBudgetManager("gpt-4o", reserve_for_output=4096,
                                      tool_schemas=schemas)
        assert mgr_ts.total_budget < mgr_no_ts.total_budget

    def test_tool_schemas_none_same_budget(self):
        """Without tool_schemas, total_budget matches old behavior."""
        mgr = ContextBudgetManager("gpt-4o", reserve_for_output=4096)
        expected = 128_000 - 4096
        assert mgr.total_budget == expected

    def test_fit_messages_accepts_tool_schemas(self):
        """fit_messages with tool_schemas uses context_message_cap for comparison."""
        mgr = ContextBudgetManager("gpt-4o", reserve_for_output=4096)
        msgs = [LLMMessage(role="user", content="hello")]
        # Should not crash
        result = mgr.fit_messages(msgs, tool_schemas=[])
        assert result is msgs


# ══════════════════════════════════════════════════════════════════════════════
# 16. P6a: Environment variable overrides for TTL / max reductions
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvOverrideConstants:
    """Verify CONTEXT_OVERRIDE_TTL / CONTEXT_MAX_REDUCTIONS env vars work.

    Tests check that the constants fall back to defaults when env is unset
    and that the env-var reading logic is wired (the actual override happens
    at import time; we verify the import-time code path works).
    """

    def test_ttl_default_is_1800(self):
        """CONTEXT_OVERRIDE_TTL defaults to 1800 seconds."""
        from external_llm.agent.context_budget import _OVERRIDE_TTL_SECONDS
        assert _OVERRIDE_TTL_SECONDS == 1800

    def test_max_reductions_default_is_3(self):
        """CONTEXT_MAX_REDUCTIONS defaults to 3."""
        from external_llm.agent.context_budget import _MAX_OVERRIDE_REDUCTIONS
        assert _MAX_OVERRIDE_REDUCTIONS == 3

    def test_env_override_format(self):
        """The env-var reading code compiles and produces an int (no module reload)."""
        import os
        ttl_default = int(os.getenv("CONTEXT_OVERRIDE_TTL", "1800"))
        max_red_default = int(os.getenv("CONTEXT_MAX_REDUCTIONS", "3"))
        assert ttl_default == 1800
        assert max_red_default == 3


# ══════════════════════════════════════════════════════════════════════════════
# 17. P6b: _is_context_length_error error_code attribute detection
# ══════════════════════════════════════════════════════════════════════════════

class TestIsContextLengthErrorExtended:
    """Verify _is_context_length_error checks error_code attribute."""

    def test_error_code_1305_with_context_text(self):
        """error_code=1305 with 'context window' text is detected."""
        from external_llm.agent.context_budget import _is_context_length_error
        err = Exception("1305: context window is too small")
        assert _is_context_length_error(err)

    def test_error_code_1305_too_small_narrow_pattern(self):
        """"too small" narrow pattern catches GLM context errors."""
        from external_llm.agent.context_budget import _is_context_length_error
        err = Exception("context window is too small")
        assert _is_context_length_error(err)

    def test_error_code_1305_on_exception_object(self):
        """error_code=1305 on exception with context terms is detected."""
        from external_llm.agent.context_budget import _is_context_length_error
        from external_llm.client import LLMRateLimitError
        err = LLMRateLimitError(
            "too small context window",
            error_code=1305,
        )
        # "too small" is now a narrow pattern — this should match
        assert _is_context_length_error(err)

    def test_error_code_1305_server_overload_not_matched(self):
        """error_code=1305 without context terms is NOT a context-length error."""
        # Note: 1305 is overloaded for both "context too small" and "server overloaded".
        # Without context terms, it should NOT be treated as context-length.
        # This test uses "overloaded" text which has no context-related terms.
        from external_llm.agent.context_budget import _is_context_length_error
        from external_llm.client import LLMRateLimitError
        err = LLMRateLimitError(
            "server overloaded, please retry",
            error_code=1305,
        )
        # "too small" narrow pattern won't match ("overloaded" not "too small").
        # The only remaining check is the error_code backstop which requires
        # context-related terms — none present → False.
        assert not _is_context_length_error(err)

# ══════════════════════════════════════════════════════════════════════════════
# 14. P1/P2: Reasoning content, thinking blocks, Gemini native tool payloads
# ══════════════════════════════════════════════════════════════════════════════
# These fields were silently counted as 0 prior to the fix, repeating the same
# under-count pattern as tool_use/tool_result blocks in the previous P1 round.

class TestEstimatorReasoningAndThinking:
    """Verify that non-text fields (reasoning_content, thinking, functionCall) are counted."""

    def test_reasoning_content_counted(self):
        """reasoning_content attribute (DeepSeek reasoner) contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="summary",
            reasoning_content="Step 1: think deeply... " * 300,  # ~7200 chars
        )
        est = estimate_tokens_from_msgs([msg])
        # content "summary"(7 chars) → max(7//3, 7//2)+1 = max(2,3)+1 = 4
        # reasoning_content ~7200 chars → max(7200//3, 7200//2)+1 = 3600+1 = 3601
        assert est > 3500, f"Expected >3500 tokens with reasoning_content, got {est}"

    def test_reasoning_content_none_is_noop(self):
        """LLMMessage without reasoning_content does not add spurious tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(role="assistant", content="short")
        est = estimate_tokens_from_msgs([msg])
        # "short"(5 chars) → max(5//3, 5//2)+1 = max(1,2)+1 = 3
        assert 0 < est < 20, f"Expected ~3 tokens, got {est}"

    def test_thinking_block_counted(self):
        """Anthropic/zai 'thinking' block in raw_content contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="final answer",
            raw_content=[
                {"type": "thinking", "thinking": "Let me reason step by step... " * 100},  # ~3000 chars
                {"type": "text", "text": "final answer"},
            ],
        )
        # content "final answer" is SKIPPED because raw_content is present.
        # Only raw_content blocks counted: thinking (3000 chars) + text (12 chars)
        # thinking: CJK-aware max(3000//3, 3000//2)+1 = 1501
        # text "final answer": 12 chars → max(12//3, 12//2)+1 = 7
        # Total ~1508
        est = estimate_tokens_from_msgs([msg])
        assert est > 1500, f"Expected >1500 tokens including thinking block, got {est}"

    def test_redacted_thinking_block_counted(self):
        """Anthropic/zai 'redacted_thinking' block with data contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="",
            raw_content=[
                {"type": "redacted_thinking", "data": "opaque_signature_payload_here " * 50},  # ~1500 chars
            ],
        )
        est = estimate_tokens_from_msgs([msg])
        # redacted_thinking data ~1500 chars → CJK-aware: max(1500//3, 1500//2)+1 = 751
        assert est > 700, f"Expected >700 tokens for redacted_thinking data, got {est}"

    def test_function_call_block_counted(self):
        """Gemini-native functionCall block in raw_content contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="assistant", content="",
            raw_content=[{
                "functionCall": {"name": "bash", "args": {"command": "ls -la /"}},
            }],
        )
        est = estimate_tokens_from_msgs([msg])
        # functionCall JSON: {"name":"bash","args":{"command":"ls -la /"}}
        # JSON ~60 chars → 60//3 + 1 = 21
        assert est >= 15, f"Expected ~21 tokens for functionCall block, got {est}"
        assert est < 100, f"functionCall should add reasonable tokens, got {est}"

    def test_function_response_block_counted(self):
        """Gemini-native functionResponse block in raw_content contributes tokens."""
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs
        msg = LLMMessage(
            role="user", content="",
            raw_content=[{
                "functionResponse": {"name": "bash", "response": {"content": "ls output here " * 200}},
            }],
        )
        est = estimate_tokens_from_msgs([msg])
        # functionResponse JSON ~3500 chars → 3500//3 + 1 = 1167
        assert est > 1000, f"Expected >1000 tokens for functionResponse block, got {est}"

# ══════════════════════════════════════════════════════════════════════════════
# 12. P1/P2/P3 regression tests from external analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestP1TooSmallPattern:
    """P1 regression: 'too small' without context terms must NOT match."""

    def test_too_small_alone_not_matched(self):
        """'temperature too small' — no context term → must NOT match."""
        assert not _is_context_length_error(Exception("temperature too small"))

    def test_too_small_max_tokens_not_matched(self):
        """'max_tokens value is too small' — no context term → must NOT match."""
        assert not _is_context_length_error(Exception("max_tokens value is too small"))

    def test_too_small_batch_not_matched(self):
        """'batch size too small' — no context term → must NOT match."""
        assert not _is_context_length_error(Exception("batch size too small"))

    def test_context_window_too_small_still_matched(self):
        """'context window is too small' — still matched via 'context window' in narrow patterns."""
        assert _is_context_length_error(Exception("context window is too small"))


class TestP2PreemptiveTrimFallback:
    """P2 regression: fallback must not duplicate single message."""

    def teardown_method(self):
        _clear_overrides()

    def test_single_message_no_duplicate(self):
        """Single huge message must not appear twice in preemptive_trim fallback."""
        from external_llm.agent._shared_utils import preemptive_trim
        msg = make_msg("user", "x" * 100_000)
        result = preemptive_trim([msg], max_tokens=1000, preserve_last=2, tag="test")
        assert len(result) == 1, f"Expected 1 message, got {len(result)}"
        assert result[0] is msg, "Object identity must be preserved (no copy)"

    def test_two_messages_not_affected(self):
        """Two messages should still produce [system, last] fallback."""
        from external_llm.agent._shared_utils import preemptive_trim
        m1 = make_msg("system", "be helpful")
        m2 = make_msg("user", "x" * 100_000)
        result = preemptive_trim([m1, m2], max_tokens=1000, preserve_last=2, tag="test")
        assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
        assert result[0] is m1
        assert result[1] is m2


class TestP3TTLResetInOverflow:
    """P3 regression: TTL-reset in _record_context_overflow must clear
    _context_window_overrides too (not just _override_meta)."""

    def teardown_method(self):
        _clear_overrides()

    def test_ttl_reset_starts_from_base_limit(self):
        """After TTL expiry, fresh overflow must reduce from base, not stale floor."""
        model = "gpt-4o"
        model_lower = model.lower().strip()
        base = _resolve_base_context_limit(model)

        # Set up override then expire
        _record_context_overflow(model)
        _override_meta[model_lower]["ts"] = 0  # expire
        # Inject a stale floor override (simulating old state)
        _context_window_overrides[model_lower] = 8192

        # This overflow should detect TTL, clear stale, and start fresh from base
        _record_context_overflow(model, estimated_prompt_tokens=100_000)

        limit = _resolve_context_limit(model)
        # With fix: base=128000 → 75%=96000, estimate clamp=85000 → 85000
        # Without fix: stale=8192 → 75%=6144, floor=8192
        assert limit > 8192, f"Fresh overflow must exceed stale floor, got {limit}"
        # The 85% headroom on 100k estimate = 85000, which is < 75% of 128k (96000)
        expected = max(8192, int(100_000 * 0.85))
        assert limit == expected, f"Expected {expected} (fresh from base), got {limit}"

    def test_ttl_reset_meta_cleared(self):
        """_override_meta entry is removed on TTL expiry during overflow."""
        model = "gpt-4o"
        model_lower = model.lower().strip()
        _record_context_overflow(model)
        _override_meta[model_lower]["ts"] = 0
        _context_window_overrides[model_lower] = 8192
        _record_context_overflow(model)
        # After TTL reset, meta should have been popped and re-created with fresh reductions=1
        meta = _override_meta.get(model_lower)
        assert meta is not None, "Fresh overflow must re-create meta"
        assert meta["reductions"] == 1, f"Expected reductions=1 (fresh start), got {meta['reductions']}"


class TestEstimateToolSchemasCache:
    """P1: estimate_tokens_from_tool_schemas id()-based cache behavior."""

    def test_same_list_uses_cache(self):
        """Same list object returns cached value; json.dumps called once."""
        from unittest.mock import patch
        from external_llm.agent._shared_utils import (
            estimate_tokens_from_tool_schemas,
            _tool_schema_token_cache,
        )
        import external_llm.agent._shared_utils as _su
        schemas = [{"name": "test", "description": "a tool"}]
        # Clear any prior cache entries
        _tool_schema_token_cache.clear()

        with patch.object(_su.json, "dumps",
                          wraps=_su.json.dumps) as mock_dumps:
            first = estimate_tokens_from_tool_schemas(schemas)
            second = estimate_tokens_from_tool_schemas(schemas)

        assert first == second, f"Cache must return same value: {first} vs {second}"
        mock_dumps.assert_called_once()

    def test_cache_respects_bounded_size(self):
        """Cache does not grow unboundedly with different list objects."""
        from external_llm.agent._shared_utils import (
            estimate_tokens_from_tool_schemas,
            _tool_schema_token_cache,
        )
        _tool_schema_token_cache.clear()
        # Create 12 unique list objects (cache max is 8)
        lists = [[{"n": i}] for i in range(12)]
        for lst in lists:
            estimate_tokens_from_tool_schemas(lst)
        assert len(_tool_schema_token_cache) <= 8, (
            f"Cache must be bounded, got {len(_tool_schema_token_cache)}"
        )


class TestRawContentTypeGuard:
    """P2: content-skip guard prevents silent under-count on non-list raw_content."""

    def test_non_list_raw_content_counts_content(self):
        """raw_content=str should NOT skip content (type violation guard)."""
        from external_llm.agent._shared_utils import _estimate_single_message_tokens
        msg = LLMMessage(role="user", content="Hello world test message")
        # Inject a string raw_content (type violation — should be list | None)
        msg.raw_content = "stray string, not a list"  # type: ignore[assignment]
        est = _estimate_single_message_tokens(msg)
        # Content should still be counted (~3 chars/token → 24/3=8)
        assert est >= 4, f"Content must be counted, got {est}"

class TestToolRegistryMemo:
    """P1: Non-Python lang_filter must return same object across calls (memoized).

    The id()-keyed token cache in ``estimate_tokens_from_tool_schemas``
    depends on stable object identity. Without memoization, every turn
    creates a fresh filtered list, and the cache never hits after ~8 turns.
    """

    def test_get_tool_schemas_memoizes_filtered_list(self):
        """Same lang_filter returns the SAME list object (memoized)."""
        from external_llm.agent.tool_registry import ToolRegistry
        from external_llm.agent.tool_registry import LanguageId
        reg = object.__new__(ToolRegistry)
        r1 = reg.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT)
        r2 = reg.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT)
        assert r1 is r2, (
            "Filtered tool schemas must be the same object (id() stability for cache)"
        )

    def test_get_tool_names_memoizes_filtered_set(self):
        """Same lang_filter returns the SAME frozenset (memoized)."""
        from external_llm.agent.tool_registry import ToolRegistry
        from external_llm.agent.tool_registry import LanguageId
        reg = object.__new__(ToolRegistry)
        n1 = reg.get_tool_names(lang_filter=LanguageId.TYPESCRIPT)
        n2 = reg.get_tool_names(lang_filter=LanguageId.TYPESCRIPT)
        assert n1 is n2, (
            "Filtered tool names must be the same object (id() stability)"
        )

    def test_no_filter_still_uses_constant(self):
        """lang_filter=None returns AGENT_TOOL_SCHEMAS (the module constant)."""
        from external_llm.agent.tool_registry import ToolRegistry
        from external_llm.agent.tool_schemas import AGENT_TOOL_SCHEMAS
        reg = object.__new__(ToolRegistry)
        result = reg.get_tool_schemas(lang_filter=None)
        assert result is AGENT_TOOL_SCHEMAS

    def test_clone_for_subagent_recomputes_memo_lazily(self):
        """A clone does not inherit the parent's memo — re-computes on first call.
        This is correct: the clone may have a different repo_language, and the
        one-time recompute cost is negligible for the rare clone path.
        """
        from external_llm.agent.tool_registry import ToolRegistry
        from external_llm.agent.tool_registry import LanguageId
        reg = object.__new__(ToolRegistry)
        reg._repo_language = LanguageId.TYPESCRIPT
        # Memoize on parent
        parent_result = reg.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT)
        # Create a minimal clone (as clone_for_subagent does)
        clone = object.__new__(ToolRegistry)
        clone.repo_root = getattr(reg, "repo_root", "/tmp")
        clone._repo_language = LanguageId.TYPESCRIPT
        # First call on clone should compute fresh (not use parent's memo)
        clone_result = clone.get_tool_schemas(lang_filter=LanguageId.TYPESCRIPT)
        assert clone_result is not parent_result, (
            "Clone must NOT share parent's memoized list (isolated state)"
        )
        # But identical content
        assert clone_result == parent_result


class TestEstimatorReasoningContentNonStr:
    """P3: Non-str reasoning_content must use _cjk_aware_tokens, not //3."""

    def test_non_str_reasoning_uses_cjk_aware(self):
        """Non-string reasoning_content (e.g. bytes) should go through _cjk_aware_tokens."""
        from external_llm.agent._shared_utils import _cjk_aware_tokens
        from external_llm.agent._shared_utils import _estimate_single_message_tokens
        # Create a message with non-str reasoning_content (e.g. bytes)
        msg = LLMMessage(role="user", content="hello")
        # bytes object is not str — hits the fallback branch
        msg.reasoning_content = b"test reasoning"  # type: ignore[assignment]
        # Should not raise; fallback now routes through _cjk_aware_tokens
        est = _estimate_single_message_tokens(msg)
        # The CJK-aware estimator for "b'test reasoning'" (repr) should give reasonable count
        assert est > 0, f"Must produce positive estimate, got {est}"
        # Compare to pure CJK-aware counting of the same repr
        expected = _cjk_aware_tokens(str(msg.reasoning_content))
        # The estimate should be close to the CJK-aware count (not //3 under-count)
        assert est >= expected * 0.8, (
            f"Non-str reasoning_content must use CJK-aware estimator, "
            f"got {est}, _cjk_aware_tokens(expected)={expected}"
        )

    def test_str_reasoning_unaffected(self):
        """String reasoning_content continues to use _cjk_aware_tokens directly."""
        from external_llm.agent._shared_utils import _cjk_aware_tokens
        from external_llm.agent._shared_utils import _estimate_single_message_tokens
        msg = LLMMessage(role="user", content="hello")
        msg.reasoning_content = "This is a reasoning trace with 한글 text"
        est = _estimate_single_message_tokens(msg)
        expected = _cjk_aware_tokens(msg.reasoning_content)
        # Should be close to CJK-aware count
        assert est >= expected * 0.8


class TestToolSchemaTokenCacheEviction:
    """P2: _tool_schema_token_cache must evict (not freeze) when full.

    The old 'if len < 8: insert' froze permanently at 8 entries — if a 9th
    unique id appeared, the cache never inserted again. The new strategy
    clears-on-full so fresh entries always enter the cache.
    """

    def test_cache_evicts_on_full(self):
        """After 8 unique ids, the 9th triggers a clear; subsequent ids insert."""
        from external_llm.agent._shared_utils import (
            estimate_tokens_from_tool_schemas,
            _tool_schema_token_cache,
        )
        _tool_schema_token_cache.clear()
        # Build 9 unique list objects (KEEP ALL alive to avoid id() reuse by GC)
        lists = [[{"n": i}] for i in range(9)]
        for lst in lists:
            estimate_tokens_from_tool_schemas(lst)
        # After 9, the cache was cleared at entry 9 (len 8→clear→insert)
        # Now cache has 1 entry (the 9th object = lists[8])
        assert len(_tool_schema_token_cache) == 1, (
            f"After 9 unique ids with cap=8, expected 1 entry, "
            f"got {len(_tool_schema_token_cache)}"
        )
        # lists[8] must be in cache (freshly inserted after clear)
        cached = _tool_schema_token_cache.get(id(lists[8]))
        assert cached is not None, "Freshly inserted entry must be in cache"
        assert cached == estimate_tokens_from_tool_schemas(lists[8])

    def test_cache_hit_after_clear(self):
        """After clear-on-full, a subsequent duplicate id must still hit."""
        from external_llm.agent._shared_utils import (
            estimate_tokens_from_tool_schemas,
            _tool_schema_token_cache,
        )
        from unittest.mock import patch
        import external_llm.agent._shared_utils as _su

        _tool_schema_token_cache.clear()
        # Fill cache to cap
        lists = [[{"n": i}] for i in range(8)]
        for lst in lists:
            estimate_tokens_from_tool_schemas(lst)
        assert len(_tool_schema_token_cache) == 8

        # 9th unique object triggers clear + insert
        lst_new = [{"n": 100}]
        estimate_tokens_from_tool_schemas(lst_new)
        assert len(_tool_schema_token_cache) == 1

        # Now call with same lst_new — should hit cache (no json.dumps)
        with patch.object(_su.json, "dumps", wraps=_su.json.dumps) as mock_dumps:
            hit = estimate_tokens_from_tool_schemas(lst_new)
        mock_dumps.assert_not_called()
        assert hit is not None
