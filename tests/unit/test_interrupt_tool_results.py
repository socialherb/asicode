"""Option B: ESC-interrupted tool-loop results are persisted full-detail and
re-rendered into the next turn's context, then dropped when the next assistant
turn is appended (so the turn after that sees only the digest — symmetric with
normal completed turns).

Covers the three-part chain:
  1. design_session.add_turn(tool_results=...) persists turn["tool_results"]
  2. context_manager.build_context_messages renders them (budget-capped) so the
     resumed model recovers the exact code body it had read/modified
  3. The *next* add_turn(assistant) drops the prior turn's tool_results — they
     exist for exactly one resumed turn. The digest keeps file/tool-level
     metadata for the long term.

Determinism (byte-identical across rebuilds) is asserted because the rendered
text enters the prompt-cache prefix.
"""
from __future__ import annotations

import pytest

from external_llm.agent.context_manager import SessionCompressionContext
from external_llm.agent.interrupt_tool_results import render_interrupt_tool_results
from external_llm.design_session import DesignSessionManager


def _entry(tool, args=None, content="", ok=True):
    return {"tool": tool, "args": args or {}, "content": content, "ok": ok}


# ── render_interrupt_tool_results ───────────────────────────────────────────

class TestRender:
    def test_empty_returns_empty(self):
        assert render_interrupt_tool_results([]) == ""
        assert render_interrupt_tool_results(None) == ""

    def test_renders_tool_args_content_status(self):
        out = render_interrupt_tool_results([
            _entry("read_file", {"path": "foo.py"}, "def bar():\n    return 1"),
            _entry("bash", "ls", "", ok=False),
        ])
        assert "read_file (ok)" in out
        assert "def bar():\n    return 1" in out
        assert "bash (FAIL)" in out
        assert '{"path": "foo.py"}' in out

    def test_byte_identical_for_cache_safety(self):
        trs = [_entry("read_file", {"path": "a.py"}, "code"), _entry("edit_text", {"path": "a.py"}, "ok")]
        assert render_interrupt_tool_results(trs) == render_interrupt_tool_results(trs)

    def test_per_result_budget_caps_content(self):
        out = render_interrupt_tool_results(
            [_entry("read_file", {"path": "x"}, "A" * 20000)],
            per_result_chars=100, total_chars=10_000,
        )
        assert "truncated" in out
        # 100 chars of content + header/args overhead, well under 20000
        assert len(out) < 5000

    def test_total_budget_stops_early(self):
        out = render_interrupt_tool_results(
            [_entry("read_file", {"path": f"f{i}.py"}, "C" * 9000) for i in range(10)],
            per_result_chars=8000, total_chars=15000,
        )
        # Not all 10 fit within the total budget
        assert "of 10" in out
        assert out.count("(ok)") < 10

    def test_args_summarized_not_dumped_raw(self):
        big_args = {"data": "x" * 2000}
        out = render_interrupt_tool_results([_entry("read_file", big_args, "c")])
        # args get capped at MAX_ARGS_CHARS (400)
        idx = out.find("args:")
        assert 0 <= idx
        line_end = out.find("\n", idx)
        assert line_end - idx <= 410


# ── Integration: add_turn persistence ────────────────────────────────────────

@pytest.fixture
def session_mgr(tmp_path):
    return DesignSessionManager(repo_root=str(tmp_path))


class TestPersistAndRender:
    def test_add_turn_persists_tool_results_for_assistant(self, session_mgr):
        trs = [_entry("read_file", {"path": "a.py"}, "code body")]
        session_mgr.add_turn("test", "assistant", "note", tool_results=trs)
        sess = session_mgr.get_or_create("test")
        assert sess.turns[-1].get("tool_results") == trs

    def test_add_turn_ignores_tool_results_for_user(self, session_mgr):
        # Only assistant turns carry interrupt tool_results.
        session_mgr.add_turn("test", "user", "hi", tool_results=[_entry("read_file")])
        sess = session_mgr.get_or_create("test")
        assert "tool_results" not in sess.turns[-1]

    def test_add_turn_no_tool_results_by_default(self, session_mgr):
        session_mgr.add_turn("test", "assistant", "normal completed turn")
        sess = session_mgr.get_or_create("test")
        assert "tool_results" not in sess.turns[-1]


# ── Integration: build_context_messages renders ─────────────────────────────

def _build_ctx(tmp_path, turns, compressed_up_to=0):
    class _S:
        pass
    s = _S()
    s.turns = turns
    s.compressed_up_to = compressed_up_to
    s.compressed_summary = ""
    s.session_id = "test"
    s.chat_mode = "code"
    s.archived_count = 0
    ctx = SessionCompressionContext(str(tmp_path))
    return ctx.build_context_messages(s, skip_core_prompt=True, mode="code")


class TestContextRender:
    def test_interrupted_turn_renders_full_results(self, tmp_path):
        trs = [_entry("read_file", {"path": "foo.py"}, "def bar():\n    return 42")]
        turns = [
            {"role": "user", "content": "go", "model": ""},
            {"role": "assistant", "content": "[Interrupted]", "model": "",
             "tool_results": trs},
        ]
        msgs = _build_ctx(tmp_path, turns)
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert asst, "expected assistant turn"
        assert "def bar():\n    return 42" in asst[-1]["content"]
        assert "Interrupted tool-loop results" in asst[-1]["content"]

    def test_normal_turn_has_no_tool_results_block(self, tmp_path):
        # A turn without tool_results must not get the block (regression guard).
        turns = [
            {"role": "user", "content": "go", "model": ""},
            {"role": "assistant", "content": "done", "model": ""},
        ]
        msgs = _build_ctx(tmp_path, turns)
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert "Interrupted tool-loop results" not in asst[-1]["content"]

    def test_render_byte_identical_across_rebuilds(self, tmp_path):
        # Cache-prefix safety: re-rendering the same session yields identical bytes.
        trs = [_entry("read_file", {"path": "a.py"}, "code")]
        turns = [
            {"role": "user", "content": "go", "model": ""},
            {"role": "assistant", "content": "note", "model": "", "tool_results": trs},
        ]
        a = _build_ctx(tmp_path, turns)
        b = _build_ctx(tmp_path, turns)
        assert a == b


# ── Integration: next add_turn drops prior tool_results ──────────────────────
# tool_results exist for exactly ONE resumed turn. Appending the next assistant
# turn (the resumed turn's own completion) clears them — symmetric with normal
# completed turns whose tool messages are discarded at turn end.

class TestNextTurnDropsPriorToolResults:
    def test_next_assistant_turn_drops_prior_tool_results(self, session_mgr):
        # Turn 0: ESC-interrupted assistant turn with full tool_results.
        trs = [_entry("read_file", {"path": "old.py"}, "old code")]
        session_mgr.add_turn("t1", "assistant", "interrupted note", tool_results=trs)
        sess = session_mgr.get_or_create("t1")
        assert sess.turns[-1].get("tool_results") == trs  # precondition

        # The resumed turn is appended. Appending it clears the prior tool_results.
        session_mgr.add_turn("t1", "assistant", "resumed completion")

        sess = session_mgr.get_or_create("t1")
        # Prior (interrupted) turn no longer carries tool_results.
        assert "tool_results" not in sess.turns[0], \
            "prior turn's tool_results must be dropped once the resumed turn completes"
        # The resumed turn itself carries nothing (normal completion).
        assert "tool_results" not in sess.turns[-1]

    def test_user_turn_between_does_not_drop(self, session_mgr):
        # A bare user turn (not assistant) must NOT clear tool_results — the
        # resume only completes once an assistant turn is recorded. Guards
        # against prematurely dropping before the resumed loop even runs.
        trs = [_entry("read_file", {"path": "x.py"}, "x")]
        session_mgr.add_turn("t1", "assistant", "interrupted", tool_results=trs)
        session_mgr.add_turn("t1", "user", "resume please")
        sess = session_mgr.get_or_create("t1")
        # Still present: the resumed assistant turn hasn't been recorded yet.
        assert sess.turns[0].get("tool_results") == trs

    def test_digest_survives_tool_results_drop(self, session_mgr):
        # digest must survive the tool_results drop (long-term file/tool metadata).
        trs = [_entry("read_file", {"path": "d.py"}, "d code")]
        session_mgr.add_turn(
            "t1", "assistant", "interrupted", digest="read: d.py", tool_results=trs,
        )
        session_mgr.add_turn("t1", "assistant", "resumed")
        sess = session_mgr.get_or_create("t1")
        assert "tool_results" not in sess.turns[0]
        assert sess.turns[0].get("digest") == "read: d.py"
