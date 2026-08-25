"""Dedicated contract tests for ``external_llm/agent/context_manager.py``.

Locks the full behavioral surface of the two context strategies where prior
coverage was indirect (76%):

  * ``SlidingWindowContext`` — trim/hysteresis, orphan tool-result and orphan
    tool-call removal, trajectory summaries, compressed-context categorisation
    (role="tool" JSON and Anthropic-native ``tool_result`` blocks), the
    carry-forward byte cap fallback (no line boundary).
  * ``SessionCompressionContext`` — ``needs_compression`` boundaries,
    ``compress_old_turns`` guards (no client / cancel / empty summary /
    preserve-only / notify=None), background-schedule and ``compact_now``
    lock/thread semantics, ``build_context_messages`` rendering (model
    switches, preserved turns, in-progress terminal, stale tool turns, insight
    failure paths), and the mtime-cached project.md loader.
  * module helpers — ``_compress_failure_notice`` interactive path,
    ``_safe_content`` / ``_extract_topics`` list-content paths.

These are GREEN-lock tests: every assertion encodes the documented contract, so
a failure is a real regression, not a spec guess.
"""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

from external_llm.agent import context_manager as cm
from external_llm.agent.context_manager import (
    _MODULE_COMPRESS_LOCKS,
    SessionCompressionContext,
    SlidingWindowConfig,
    SlidingWindowContext,
    _extract_topics,
    _safe_content,
    _SuppressInfoFilter,
)
from external_llm.client import (
    LLMAuthenticationError,
    LLMMessage,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMResponse,
)

# ── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Isolation: clear the failure-notice latch and the lock registry."""
    with cm._compress_fail_latch_lock:
        cm._compress_fail_latch.clear()
    _MODULE_COMPRESS_LOCKS.clear()
    gc.collect()
    yield
    with cm._compress_fail_latch_lock:
        cm._compress_fail_latch.clear()
    _MODULE_COMPRESS_LOCKS.clear()
    gc.collect()


def _make_session(turns, *, compressed_up_to=0, summary="", session_id="s1"):
    return SimpleNamespace(
        turns=turns,
        compressed_up_to=compressed_up_to,
        compressed_summary=summary,
        session_id=session_id,
        archived_count=0,
    )


def _resp(content="summary text"):
    return LLMResponse(content=content, model="helper", provider="test")


class _FakeClient:
    """Minimal llm_client whose .chat() returns one canned response."""

    def __init__(self, response=None):
        self._response = response or _resp()
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _RaisingClient:
    def chat(self, **kwargs):
        raise RuntimeError("provider down")


# ── _compress_failure_notice: interactive path ─────────────────────────────


class TestCompressFailureNoticeInteractive:
    def test_auth_interactive_wording(self):
        msg = cm._compress_failure_notice(
            "s", "m", LLMAuthenticationError("401"), use_latch=False, context="interactive"
        )
        assert "authentication failed" in msg
        assert "fix the API key" in msg
        assert "try again" in msg

    def test_quota_interactive_wording(self):
        msg = cm._compress_failure_notice(
            "s", "m", LLMQuotaExceededError("402"), use_latch=False, context="interactive"
        )
        assert "quota exceeded" in msg
        assert "restore credits" in msg

    def test_rate_interactive_wording(self):
        msg = cm._compress_failure_notice("s", "m", LLMRateLimitError("429"), use_latch=False, context="interactive")
        assert "rate-limited" in msg
        assert "wait a moment" in msg

    def test_interactive_bypasses_latch(self):
        """use_latch=False reports EVERY attempt (user is diagnosing)."""
        exc = LLMAuthenticationError("401")
        first = cm._compress_failure_notice("s", "m", exc, use_latch=False, context="interactive")
        second = cm._compress_failure_notice("s", "m", exc, use_latch=False, context="interactive")
        assert first is not None
        assert second is not None  # not latched

    def test_background_generic_error_stays_silent(self):
        assert cm._compress_failure_notice("s", "m", RuntimeError("boom"), use_latch=True) is None


# ── SlidingWindowContext.prepare_before_call ───────────────────────────────


def _sys(content="SYS"):
    return LLMMessage(role="system", content=content)


def _user(content):
    return LLMMessage(role="user", content=content)


def _asst(content, tool_calls=None, raw_content=None):
    return LLMMessage(role="assistant", content=content, tool_calls=tool_calls, raw_content=raw_content)


def _tool(name, content, raw_content=None):
    return LLMMessage(role="tool", content=content, name=name, raw_content=raw_content)


def _trim(messages, window=5, hysteresis=0.6):
    cfg = SlidingWindowConfig(context_window_size=window, hysteresis_factor=hysteresis)
    return SlidingWindowContext(cfg).prepare_before_call(messages)


def _roles(result):
    return [getattr(m, "role", "?") for m in result]


class TestSlidingWindowPrepare:
    def test_window_disabled_returns_unchanged(self):
        msgs = [_user("a"), _user("b")]
        cfg = SlidingWindowConfig(context_window_size=0)
        out = SlidingWindowContext(cfg).prepare_before_call(msgs)
        assert out is msgs

    def test_orphan_standard_tool_result_dropped_from_kept(self):
        # Trim boundary lands on a role="tool" message with no preceding assistant.
        msgs = [_sys(), _user("u1"), _asst("a1"), _tool("bash", "t1"), _tool("bash", "t2"), _asst("a2"), _user("u2")]
        out = _trim(msgs, window=5)
        # t2 was trimmed out of `kept` and folded into the summary.
        assert _roles(out) == ["system", "user", "assistant", "user"]
        summary = out[1].content
        assert "bash: t1" in summary and "bash: t2" in summary

    def test_anthropic_mixed_result_keeps_text_blocks(self):
        tr = LLMMessage(
            role="user",
            content="plain",
            raw_content=[
                {"type": "tool_result", "tool_use_id": "c1", "content": "out"},
                {"type": "text", "text": "strategy warning"},
            ],
        )
        # 6 non-system messages, window=5 → trim boundary lands on `tr` (kept[0]).
        msgs = [_sys(), _user("u1"), _asst("a1"), _tool("bash", "t1"), tr, _asst("a2"), _user("u2")]
        out = _trim(msgs, window=5)
        # The mixed message survives with only the text block.
        kept_user = [m for m in out if getattr(m, "role", "") == "user" and m.content == "plain"]
        assert len(kept_user) == 1
        assert kept_user[0].raw_content == [{"type": "text", "text": "strategy warning"}]
        assert _roles(out) == ["system", "user", "user", "assistant", "user"]

    def test_anthropic_pure_tool_result_dropped_whole(self):
        tr = LLMMessage(
            role="user",
            content="",
            raw_content=[
                {"type": "tool_result", "tool_use_id": "c1", "content": "out"},
            ],
        )
        # 6 non-system messages, window=5 → the pure tool_result lands in kept[0]
        # and is dropped whole by the orphan loop.
        msgs = [_sys(), _user("u1"), _asst("a1"), _tool("bash", "t1"), tr, _asst("a2"), _user("u2")]
        out = _trim(msgs, window=5)
        assert _roles(out) == ["system", "user", "assistant", "user"]
        assert "out" in out[1].content  # folded into summary

    def test_orphan_assistant_with_tool_calls_dropped(self):
        a_tc = _asst("will call", tool_calls=[{"id": "c1", "name": "bash"}])
        # hysteresis=1.0 → trim_target=3: kept=[t1, a_tc, u2]. The orphan loop
        # then drops t1 (no preceding assistant) AND a_tc (call without result).
        msgs = [_sys(), _user("u1"), _tool("bash", "t1"), a_tc, _user("u2")]
        out = _trim(msgs, window=3, hysteresis=1.0)
        assert _roles(out) == ["system", "user", "user"]
        assert "bash: t1" in out[1].content
        assert "will call" in out[1].content

    def test_assistant_tool_call_with_following_result_kept(self):
        a_tc = _asst("will call", tool_calls=[{"id": "c1", "name": "bash"}])
        tr = _tool("bash", "result")
        # hysteresis=1.0 → trim_target=3: kept=[a_tc, tr, u2] — a_tc's result
        # immediately follows, so neither orphan loop drops it.
        msgs = [_sys(), _user("u1"), _user("x"), a_tc, tr, _user("u2")]
        out = _trim(msgs, window=3, hysteresis=1.0)
        assert _roles(out) == ["system", "user", "assistant", "tool", "user"]


# ── SlidingWindowContext: token-budget force trim (R3) ─────────────────────


class TestSlidingWindowTokenBudget:
    """prepare_before_call(budget=N) force-compresses the OLDEST messages and
    keeps the NEWEST suffix that fits the token budget."""

    def _heavy(self, n_pairs: int, content: str = "x" * 300):
        msgs = [_sys("SYS")]
        for i in range(n_pairs):
            msgs.append(_user(f"q{i} " + content))
            msgs.append(_asst(f"a{i} " + content))
        return msgs

    def test_budget_trims_to_fit_and_keeps_newest(self):
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs

        msgs = self._heavy(40)
        est_all = estimate_tokens_from_msgs(msgs)
        cfg = SlidingWindowConfig(context_window_size=300)
        swc = SlidingWindowContext(cfg)
        out = swc.prepare_before_call(msgs, budget=est_all // 2)
        est_out = estimate_tokens_from_msgs(out)
        assert est_out <= est_all // 2
        # The newest exchange survives
        assert out[-1].content.startswith("a39")
        # A compact summary block exists
        assert any(getattr(m, "role", "") == "user" and m.content.startswith("[COMPRESSED CONTEXT]") for m in out)

    def test_budget_noop_when_already_fits(self):
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs

        msgs = self._heavy(3)
        est = estimate_tokens_from_msgs(msgs)
        out = SlidingWindowContext(SlidingWindowConfig(context_window_size=300)).prepare_before_call(
            msgs, budget=est + 10_000
        )
        assert len(out) == len(msgs)  # no trim when comfortably under budget

    def test_budget_impossible_returns_none_falls_back_to_count(self):
        msgs = self._heavy(5)
        # Budget far below a single message + summary reserve → impossible.
        swc = SlidingWindowContext(SlidingWindowConfig(context_window_size=2))
        out = swc.prepare_before_call(msgs, budget=10)
        # falls back to count window (window=2 → trim)
        assert len(out) < len(msgs)

    def test_emits_context_trimmed_event(self):
        from external_llm.agent._shared_utils import estimate_tokens_from_msgs

        msgs = self._heavy(40)
        est = estimate_tokens_from_msgs(msgs)
        events: list = []
        swc = SlidingWindowContext(
            SlidingWindowConfig(context_window_size=300), stream_callback=lambda ev, d: events.append(ev)
        )
        swc.prepare_before_call(msgs, budget=est // 2)
        assert "context_trimmed" in events
        assert "agent_working" in events


# ── trajectory_summary ─────────────────────────────────────────────────────


def _turn(num, name, ok):
    return SimpleNamespace(turn_num=num, tool_name=name, tool_result=SimpleNamespace(ok=ok))


class TestTrajectorySummary:
    def test_empty(self):
        assert SlidingWindowContext(SlidingWindowConfig()).trajectory_summary([]) == ""

    def test_format_and_cap_last_10(self):
        turns = [_turn(i, f"tool{i}", i % 2 == 0) for i in range(1, 13)]
        out = SlidingWindowContext(SlidingWindowConfig()).trajectory_summary(turns)
        lines = out.split("\n")
        assert lines[0] == "[TRAJECTORY SUMMARY]"
        assert lines[-1] == "[END TRAJECTORY]"
        # Only the last 10 turns are summarised (3..12).
        assert lines[1] == "3. tool3 ✗"
        assert lines[-2] == "12. tool12 ✓"
        assert not any(ln.startswith(("1. ", "2. ")) for ln in lines)
        # ok=False renders ✗.
        assert "11. tool11 ✗" in out


# ── _build_compressed_message: categorisation + carry-forward ──────────────


def _compress(dropped, config=None):
    sw = SlidingWindowContext(config or SlidingWindowConfig())
    return sw._build_compressed_message(dropped)


def _category_body(out):
    """Content of the summary message."""
    return out.content


class TestCompressedMessageCategorisation:
    def test_tool_non_json_goes_to_errors(self):
        out = _compress([_tool("bash", "plain output 123")])
        body = _category_body(out)
        assert "Failed tool calls" in body
        assert "bash: plain output 123" in body

    def test_tool_ok_false_uses_error_field(self):
        out = _compress([_tool("bash", '{"ok": false, "error": "boom happened"}')])
        body = _category_body(out)
        assert "Failed tool calls" in body
        assert "boom happened" in body

    def test_tool_buckets_by_name(self):
        dropped = [
            _tool("apply_patch", '{"ok": true, "content": "patched"}'),
            _tool("write_plan", '{"ok": true, "content": "planned"}'),
            _tool("find_symbol", '{"ok": true, "content": "sym"}'),
            _tool("find_references", '{"ok": true, "content": "refs"}'),
            _tool("read_file", '{"ok": true, "content": "file"}'),
            _tool("read_symbol", '{"ok": true, "content": "sym2"}'),
            _tool("get_file_outline", '{"ok": true, "content": "outline"}'),
            _tool("bash", '{"ok": true, "content": "ls"}'),
        ]
        body = _category_body(_compress(dropped))
        assert "Applied changes" in body and "apply_patch: patched" in body
        assert "write_plan: planned" in body
        assert "Symbol / search results" in body
        assert "find_symbol: sym" in body and "find_references: refs" in body
        assert "Files read" in body
        assert "read_file: file" in body and "read_symbol: sym2" in body
        assert "get_file_outline: outline" in body
        assert "Other tool calls" in body and "bash: ls" in body

    def test_anthropic_tool_result_non_json_content(self):
        a_tc = _asst("", raw_content=[{"type": "tool_use", "id": "c1", "name": "bash"}])
        tr = LLMMessage(
            role="user",
            content="",
            raw_content=[
                {"type": "tool_result", "tool_use_id": "c1", "content": {"nested": True}},
            ],
        )
        body = _category_body(_compress([a_tc, tr]))
        assert "Failed tool calls" in body
        assert "bash: {'nested': True}" in body  # str() of non-str content, ok defaults False

    def test_anthropic_tool_result_is_error_forces_failure(self):
        tr = LLMMessage(
            role="user",
            content="",
            raw_content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": '{"ok": true, "content": "looks fine"}',
                    "is_error": True,
                },
            ],
        )
        body = _category_body(_compress([tr]))
        assert "Failed tool calls" in body
        assert "looks fine" in body

    def test_anthropic_tool_result_empty_snippet_placeholder(self):
        tr = LLMMessage(
            role="user",
            content="",
            raw_content=[
                {"type": "tool_result", "tool_use_id": "c1", "content": ""},
            ],
        )
        body = _category_body(_compress([tr]))
        assert "[empty result]" in body

    def test_carry_forward_cap_without_line_boundary(self):
        """A single over-long carried line is raw-capped (no newline to cut at)."""
        cfg = SlidingWindowConfig(carry_forward_bytes=40)
        prev = LLMMessage(role="user", content="[COMPRESSED CONTEXT]\n" + "x" * 200)
        body = _category_body(_compress([prev, _user("fresh")], config=cfg))
        assert "Previous summary (carried forward):" in body
        carried = body.split("Previous summary (carried forward):\n", 1)[1]
        carried = carried.rsplit("\n[END COMPRESSED CONTEXT]", 1)[0]
        assert carried == "x" * 40  # raw byte cap, whole line preserved

    def test_carry_forward_cap_cuts_at_last_whole_line(self):
        """When a newline exists inside the cap, truncation keeps whole lines only."""
        cfg = SlidingWindowConfig(carry_forward_bytes=50)
        prev = LLMMessage(role="user", content="[COMPRESSED CONTEXT]\nshort line\n" + "y" * 100)
        body = _category_body(_compress([prev, _user("fresh")], config=cfg))
        carried = body.split("Previous summary (carried forward):\n", 1)[1]
        carried = carried.rsplit("\n[END COMPRESSED CONTEXT]", 1)[0]
        assert carried == "short line"  # cut at the newline, never mid-line

    def test_discussion_buckets(self):
        out = _compress([_user("user thought"), _asst("assistant thought")])
        body = _category_body(out)
        assert "Discussion summary" in body
        assert "[user] user thought" in body
        assert "[assistant] assistant thought" in body


# ── SessionCompressionContext: ABC passthrough + project.md cache ──────────


class TestSessionBasics:
    def test_prepare_before_call_passthrough(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        msgs = [{"role": "user", "content": "x"}]
        assert ctx.prepare_before_call(msgs) is msgs

    def test_trajectory_summary_empty(self):
        assert SessionCompressionContext("/tmp/nonexistent-repo").trajectory_summary([]) == ""

    def test_project_md_missing_file(self, tmp_path):
        ctx = SessionCompressionContext(str(tmp_path))
        calls = []
        ctx._load_project_context_md_fn = lambda root: calls.append(root) or "BODY"
        assert ctx.load_project_context_md() == ""
        assert calls == []  # loader never invoked when file is absent

    def test_project_md_cache_hit(self, tmp_path):
        (tmp_path / ".asicode").mkdir()
        (tmp_path / ".asicode" / "project.md").write_text("A", encoding="utf-8")
        ctx = SessionCompressionContext(str(tmp_path))
        calls = []
        ctx._load_project_context_md_fn = lambda root: calls.append(root) or "BODY"
        assert ctx.load_project_context_md() == "BODY"
        assert ctx.load_project_context_md() == "BODY"  # cache hit
        assert len(calls) == 1

    def test_project_md_rewrite_refetches(self, tmp_path):
        (tmp_path / ".asicode").mkdir()
        proj = tmp_path / ".asicode" / "project.md"
        proj.write_text("A", encoding="utf-8")
        ctx = SessionCompressionContext(str(tmp_path))
        calls = []
        ctx._load_project_context_md_fn = lambda root: calls.append(root) or "BODY"
        ctx.load_project_context_md()
        proj.write_text("AB", encoding="utf-8")  # size change → cache miss
        assert ctx.load_project_context_md() == "BODY"
        assert len(calls) == 2

    def test_project_md_loader_exception(self, tmp_path, caplog):
        (tmp_path / ".asicode").mkdir()
        (tmp_path / ".asicode" / "project.md").write_text("A", encoding="utf-8")
        ctx = SessionCompressionContext(str(tmp_path))
        ctx._load_project_context_md_fn = lambda root: (_ for _ in ()).throw(RuntimeError("boom"))
        with caplog.at_level("WARNING"):
            assert ctx.load_project_context_md() == ""
        assert "Could not load project context" in caplog.text


# ── needs_compression boundaries ───────────────────────────────────────────


def _turns(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"c{i}"} for i in range(n)]


class TestNeedsCompression:
    def test_empty_turns(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        assert ctx.needs_compression(_make_session([])) is False

    def test_boundary_exact_batch_min(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        # defaults: recent_keep=4, batch_min=11 → 15 verbatim: 15-4=11 >= 11
        assert ctx.needs_compression(_make_session(_turns(15))) is True
        # 14 verbatim: 14-4=10 < 11
        assert ctx.needs_compression(_make_session(_turns(14))) is False

    def test_batch_min_override_force_path(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        # FORCE_COMPRESS_MIN_TURNS=3 → 7 verbatim: 7-4=3 >= 3
        assert ctx.needs_compression(_make_session(_turns(7)), batch_min=3) is True
        assert ctx.needs_compression(_make_session(_turns(6)), batch_min=3) is False

    def test_recent_keep_override(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        assert ctx.needs_compression(_make_session(_turns(10)), recent_keep=2, batch_min=5) is True
        assert ctx.needs_compression(_make_session(_turns(6)), recent_keep=2, batch_min=5) is False

    def test_compressed_up_to_offsets_verbatim_window(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        # compressed_up_to=5 of 15 turns → verbatim=10 → 10-4=6 < 11
        assert ctx.needs_compression(_make_session(_turns(15), compressed_up_to=5)) is False
        # compressed_up_to=5 of 16 → verbatim=11 → 7 >= 11? no: 11-4=7 < 11
        assert ctx.needs_compression(_make_session(_turns(16), compressed_up_to=5)) is False
        # compressed_up_to=5 of 20 → verbatim=15 → 11 >= 11
        assert ctx.needs_compression(_make_session(_turns(20), compressed_up_to=5)) is True

    def test_archived_count_cancels_absolute_pointer(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(15), compressed_up_to=5)
        s.archived_count = 5  # pointer refers to archived turns only → verbatim=15
        assert ctx.needs_compression(s) is True


# ── compress_old_turns: guards ─────────────────────────────────────────────


class TestCompressOldTurnsGuards:
    def test_empty_turns_noop(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session([])
        client = _FakeClient()
        ctx.compress_old_turns(s, client, "helper")
        assert client.calls == []
        assert s.compressed_up_to == 0

    def test_cutoff_nonpositive_noop(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(2))  # 2 - 4 < 0
        client = _FakeClient()
        ctx.compress_old_turns(s, client, "helper")
        assert client.calls == []

    def test_all_excluded_returns_without_llm(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [{"role": "user", "content": "x", "exclude_from_compression": True}] * 5
        s = _make_session(turns)
        ctx.compress_old_turns(s, None, "helper", recent_keep=0)
        assert s.compressed_up_to == 0

    def test_no_llm_client_preserves_turns(self, caplog):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        with caplog.at_level("DEBUG"):
            ctx.compress_old_turns(s, None, "helper", recent_keep=2)
        assert s.compressed_up_to == 0  # pointer must NOT advance
        assert "no llm_client" in caplog.text

    def test_preserve_only_advances_without_llm(self):
        """preserve turns are re-inserted verbatim; no summary call needed."""
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [{"role": "user", "content": f"p{i}", "preserve": True} for i in range(5)]
        s = _make_session(turns)
        notified = []
        ctx.compress_old_turns(s, None, "helper", recent_keep=0, notify=notified.append)
        assert s.compressed_up_to == 5  # pointer advances past preserved turns
        assert any("Compressed" in n for n in notified)

    def test_cancel_event_skips_llm_call(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        client = _FakeClient()
        ev = threading.Event()
        ev.set()
        ctx.compress_old_turns(s, client, "helper", recent_keep=2, cancel_event=ev)
        assert client.calls == []
        assert s.compressed_up_to == 0

    def test_previous_summary_included_in_prompt(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6), summary="OLD SUMMARY BODY")
        client = _FakeClient()
        ctx.compress_old_turns(s, client, "helper", recent_keep=2)
        prompt = client.calls[0]["messages"][-1].content
        assert "Previous summary:\nOLD SUMMARY BODY" in prompt
        assert "New conversation turns to incorporate:" in prompt

    def test_success_pops_tool_results_and_notifies(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = _turns(6)
        turns[0]["tool_results"] = [{"name": "bash", "output": "x"}]
        s = _make_session(turns)
        notified = []
        ctx.compress_old_turns(s, _FakeClient(), "helper", recent_keep=2, notify=notified.append)
        assert s.compressed_up_to == 4
        assert "tool_results" not in turns[0]  # popped after successful fold
        assert any("Compressed 4 user/AI turns" in n for n in notified)

    def test_success_logs_when_no_notify(self, caplog):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        with caplog.at_level("INFO"):
            ctx.compress_old_turns(s, _FakeClient(), "helper", recent_keep=2)
        assert any("Compressed 4 user/AI turns" in r.message for r in caplog.records)

    def test_llm_exception_background_silent_preserves_turns(self, caplog):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        notified = []
        with caplog.at_level("DEBUG"):
            ctx.compress_old_turns(s, _RaisingClient(), "helper", recent_keep=2, notify=notified.append)
        assert s.compressed_up_to == 0  # turns preserved on failure
        assert notified == []  # generic errors stay silent in background path
        assert "Failed to compress conversation" in caplog.text

    def test_llm_empty_summary_does_not_advance(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        client = _FakeClient(_resp(content="   "))
        ctx.compress_old_turns(s, client, "helper", recent_keep=2)
        assert s.compressed_up_to == 0
        assert s.compressed_summary == ""


# ── threading entry points ─────────────────────────────────────────────────


class _FakeThread:
    """Synchronous stand-in for threading.Thread (deterministic, no joins)."""

    created: ClassVar[list] = []

    def __init__(self, target=None, daemon=False):
        self.target = target
        self.daemon = daemon
        _FakeThread.created.append(self)

    def start(self):
        self.target()


@pytest.fixture
def fake_thread(monkeypatch):
    _FakeThread.created = []
    monkeypatch.setattr(cm.threading, "Thread", _FakeThread)
    return _FakeThread


class TestScheduleBackgroundCompress:
    def test_not_needed_skips(self, fake_thread):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        ctx.schedule_background_compress(_make_session(_turns(5)), "helper", _FakeClient())
        assert fake_thread.created == []

    def test_force_path_uses_smaller_batch_min(self, fake_thread):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(7))  # 7-4=3 >= FORCE_COMPRESS_MIN_TURNS
        ctx.schedule_background_compress(s, "helper", _FakeClient(), force=True)
        assert len(fake_thread.created) == 1
        assert s.compressed_up_to == 3  # compress_old_turns folded turns[:3]

    def test_force_path_below_min_skips(self, fake_thread):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        ctx.schedule_background_compress(_make_session(_turns(6)), "helper", _FakeClient(), force=True)
        assert fake_thread.created == []

    def test_lock_busy_returns(self, fake_thread):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(20))
        lock = ctx._get_compress_lock(s.session_id)
        assert lock.acquire(blocking=False)
        try:
            ctx.schedule_background_compress(s, "helper", _FakeClient())
        finally:
            lock.release()
        assert fake_thread.created == []

    def test_happy_path_compresses_and_persists(self, fake_thread):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(20))
        persisted = []
        ctx.schedule_background_compress(s, "helper", _FakeClient(), persist=lambda: persisted.append(1))
        assert s.compressed_up_to == 16  # 20 turns, keep 4
        assert persisted == [1]

    def test_run_exception_is_logged_and_lock_released(self, fake_thread, caplog):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(20))

        def bad_persist():
            raise RuntimeError("persist boom")

        with caplog.at_level("ERROR"):
            ctx.schedule_background_compress(s, "helper", _FakeClient(), persist=bad_persist)
        assert "Background compress failed" in caplog.text
        # lock released in finally → a follow-up run proceeds (fresh session:
        # the first run already advanced the pointer past the batch gate)
        s2 = _make_session(_turns(20))
        fake_thread.created = []
        ctx.schedule_background_compress(s2, "helper", _FakeClient())
        assert len(fake_thread.created) == 1


class TestCompactNow:
    def test_empty_turns(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        assert ctx.compact_now(_make_session([]), "helper", _FakeClient()) is False

    def test_lock_busy(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(10))
        lock = ctx._get_compress_lock(s.session_id)
        assert lock.acquire(blocking=False)
        try:
            assert ctx.compact_now(s, "helper", _FakeClient()) is False
        finally:
            lock.release()

    def test_success_persists_and_reports(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(6))
        persisted = []
        assert ctx.compact_now(s, "helper", _FakeClient(), recent_keep=0, persist=lambda: persisted.append(1)) is True
        assert s.compressed_up_to == 6
        assert persisted == [1]

    def test_no_change_returns_false(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(2))
        # recent_keep > turns → cutoff <= 0 → nothing to fold
        assert ctx.compact_now(s, "helper", _FakeClient(), recent_keep=5) is False
        assert s.compressed_up_to == 0

    def test_make_compress_cancel_event_fresh_and_unset(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        e1 = ctx._make_compress_cancel_event()
        e2 = ctx._make_compress_cancel_event()
        assert isinstance(e1, threading.Event) and not e1.is_set()
        assert e1 is not e2


# ── build_context_messages ─────────────────────────────────────────────────


def _build(ctx, session, **kw):
    kw.setdefault("skip_core_prompt", True)
    kw.setdefault("mode", "general")
    return ctx.build_context_messages(session, **kw)


def _contents(msgs):
    return [m["content"] for m in msgs]


class TestBuildContextMessages:
    def test_core_prompt_embedded_by_default(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(2))
        msgs = ctx.build_context_messages(s, mode="general")
        assert msgs[0]["role"] == "system"
        assert "## Available Tools" not in msgs[0]["content"]  # core chunk only
        assert msgs[1] == {"role": "system", "content": "──"}

    def test_general_mode_marker_and_no_project_md(self, tmp_path):
        (tmp_path / ".asicode").mkdir()
        (tmp_path / ".asicode" / "project.md").write_text("PROJ", encoding="utf-8")
        ctx = SessionCompressionContext(str(tmp_path))
        calls = []
        ctx._load_project_context_md_fn = lambda root: calls.append(root) or "PROJ"
        msgs = _build(ctx, _make_session(_turns(2)), mode="general")
        assert calls == []  # general mode never loads project.md
        assert any("[MODE: General Chat]" in c for c in _contents(msgs))

    def test_insights_failure_is_silent(self, tmp_path, monkeypatch):
        import external_llm.agent.design_chat_loop as dcl

        monkeypatch.setattr(dcl, "load_design_insights", lambda repo: (_ for _ in ()).throw(RuntimeError("boom")))
        ctx = SessionCompressionContext(str(tmp_path))
        msgs = _build(ctx, _make_session(_turns(2)), mode="code")
        assert all("DESIGN INSIGHTS" not in c for c in _contents(msgs))

    def test_insights_injected_in_code_mode(self, tmp_path, monkeypatch):
        import external_llm.agent.design_chat_loop as dcl

        monkeypatch.setattr(dcl, "load_design_insights", lambda repo: "INSIGHT-BODY")
        ctx = SessionCompressionContext(str(tmp_path))
        msgs = _build(ctx, _make_session(_turns(2)), mode="code")
        assert any("=== DESIGN INSIGHTS" in c and "INSIGHT-BODY" in c for c in _contents(msgs))

    def test_summary_label_turns_range(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        s = _make_session(_turns(2), compressed_up_to=7, summary="SUM BODY")
        msgs = _build(ctx, s)
        assert any("=== CONVERSATION SUMMARY (turns 1-7) ===" in c for c in _contents(msgs))

    def test_preserved_old_turn_with_digest_rendered(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [
            {"role": "user", "content": "old user", "preserve": True},
            {"role": "assistant", "content": "old asst", "preserve": True, "digest": "DIG: read foo.py"},
            {"role": "user", "content": "new"},
        ]
        s = _make_session(turns, compressed_up_to=2, summary="SUM")
        msgs = _build(ctx, s)
        contents = _contents(msgs)
        assert any(c == "(turn 1) old user" for c in contents)
        assert any(c.startswith("(turn 2) old asst") and "[WORK STATE" in c for c in contents)

    def test_stale_excluded_tool_turns_skipped(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [
            {"role": "tool", "content": "stale tool", "exclude_from_compression": True},
            {"role": "assistant", "content": "normal asst"},
            {"role": "tool", "content": "fresh tool", "exclude_from_compression": True},
            {"role": "user", "content": "current"},
        ]
        msgs = _build(ctx, _make_session(turns))
        contents = _contents(msgs)
        assert all("stale tool" not in c for c in contents)
        assert any(c == "(turn 3) fresh tool" for c in contents)

    def test_in_progress_other_owner_rendered_as_system_label(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [
            {"role": "user", "content": "parallel request", "in_progress": True, "owner": "other"},
            {"role": "user", "content": "mine"},
        ]
        msgs = _build(ctx, _make_session(turns), owner="me")
        contents = _contents(msgs)
        assert any("[IN-PROGRESS IN ANOTHER TERMINAL]" in c and "parallel request" in c for c in contents)
        assert any(c == "(turn 2) mine" for c in contents)

    def test_in_progress_same_owner_renders_normally(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [{"role": "user", "content": "mine", "in_progress": True, "owner": "me"}]
        msgs = _build(ctx, _make_session(turns), owner="me")
        assert any(c == "(turn 1) mine" for c in _contents(msgs))

    def test_model_switch_annotations(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "model": "gpt-a"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2", "model": "gpt-b"},
        ]
        msgs = _build(ctx, _make_session(turns), current_model="gpt-c")
        contents = _contents(msgs)
        assert any("[Model switched: gpt-a → gpt-b]" in c for c in contents)
        assert any("You are now continuing this conversation" in c and "gpt-b" in c for c in contents)

    def test_verbatim_assistant_digest_appended(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [{"role": "assistant", "content": "answer", "digest": "DIG: patched x.py"}]
        msgs = _build(ctx, _make_session(turns))
        contents = _contents(msgs)
        assert any(c.startswith("answer") and "[WORK STATE" in c and "DIG: patched x.py" in c for c in contents)

    def test_non_conversation_role_not_labelled(self):
        ctx = SessionCompressionContext("/tmp/nonexistent-repo")
        turns = [{"role": "system", "content": "raw sys"}]
        msgs = _build(ctx, _make_session(turns))
        assert any(c == "raw sys" for c in _contents(msgs))

    def test_promoted_insights_appended(self, tmp_path, monkeypatch):
        import external_llm.agent.design_chat_loop as dcl

        monkeypatch.setattr(dcl, "load_promoted_insights", lambda repo, q: "PROMO!")
        ctx = SessionCompressionContext(str(tmp_path))
        msgs = _build(ctx, _make_session(_turns(2)), mode="code")
        assert any(c == "PROMO!" for c in _contents(msgs))

    def test_promoted_insights_failure_silent(self, tmp_path, monkeypatch, caplog):
        import external_llm.agent.design_chat_loop as dcl

        monkeypatch.setattr(dcl, "load_promoted_insights", lambda repo, q: (_ for _ in ()).throw(RuntimeError("boom")))
        ctx = SessionCompressionContext(str(tmp_path))
        with caplog.at_level("DEBUG"):
            msgs = _build(ctx, _make_session(_turns(2)), mode="code")
        assert all("PROMO" not in c for c in _contents(msgs))

    def test_non_str_turn_content_skips_promotion(self, tmp_path, monkeypatch):
        """A turn whose content is not a str (structured blocks) must degrade
        to no-promotion instead of crashing the whole context build — the 2c
        ``"\n".join`` raises and the except collapses ``_task_q`` to ""."""
        import external_llm.agent.design_chat_loop as dcl

        calls = []
        monkeypatch.setattr(dcl, "load_promoted_insights", lambda repo, q: calls.append(q) or "X")
        ctx = SessionCompressionContext(str(tmp_path))
        turns = _turns(2)
        turns.append({"role": "user", "content": {"blocks": [{"type": "text"}]}})
        msgs = _build(ctx, _make_session(turns), mode="code")
        assert calls == []
        assert msgs  # build still succeeded


# ── module-level helpers ───────────────────────────────────────────────────


class TestModuleHelpers:
    def test_suppress_filter_blocks_everything(self):
        f = _SuppressInfoFilter()
        assert f.filter(None) is False  # type: ignore[arg-type]

    def test_safe_content_list(self):
        assert _safe_content(SimpleNamespace(content=[{"text": "a"}, {"text": "b"}])) == "a b"

    def test_safe_content_str_strips(self):
        assert _safe_content(SimpleNamespace(content="  hello \n")) == "hello"

    def test_extract_topics_list_content(self):
        msgs = [
            SimpleNamespace(content=[{"text": "caching caching caching"}, {"text": "graph"}]),
            SimpleNamespace(content="caching graph"),
        ]
        assert _extract_topics(msgs) == ["caching", "graph"]

    def test_extract_topics_stopwords_and_short_words(self):
        msgs = [SimpleNamespace(content="the a and graph! Graph, caching...")]
        assert _extract_topics(msgs) == ["graph", "caching"]

    def test_extract_topics_max_keywords(self):
        msgs = [SimpleNamespace(content="alpha beta gamma delta")]
        assert _extract_topics(msgs, max_keywords=2) == ["alpha", "beta"]
