"""RED→GREEN coverage for repl_impl.py pure helpers (6% → pure surface 100%).

Targets module-level pure functions + _ProgressPrinter's terminal-independent
rendering helpers (static/class methods, state machines). No terminal I/O,
no threads, no subprocesses.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from external_llm.repl import repl_impl
from external_llm.repl.repl_impl import (
    _auto_continue_should_arm,
    _auto_submit_now,
    _build_json_output,
    _build_orchestrator_digest,
    _build_turn_digest,
    _cjk_width,
    _deliver_next_suggestion,
    _dropped_entries,
    _eval_ctrlc_armed,
    _extract_patched_file,
    _format_result,
    _insights_compact_is_noop,
    _json_stream_emit,
    _maybe_arm_auto_submit,
    _notify_above_prompt,
    _parse_auto_arg,
    _result_output_dict,
    _retry_create_svc_with_api_key_prompt,
    _save_key_to_dotenv,
    _size_compact_budget,
    _split_work_state,
    _text_has_hangul,
    _validate_next_suggestion,
    _wrap_cjk,
    _wrap_preserve_code,
)

# ── _cjk_width ────────────────────────────────────────────────────────────────


class TestCjkWidth:
    def test_ascii_is_width_1(self):
        assert _cjk_width("abc") == 3
        assert _cjk_width("") == 0

    def test_cjk_wide_chars_count_2(self):
        assert _cjk_width("가나다") == 6  # Hangul syllables (W)
        assert _cjk_width("한") == 2
        assert _cjk_width("\uff26\uff21") == 4  # fullwidth (F)

    def test_combining_chars_are_zero_width(self):
        # e + combining acute = 1 column
        assert _cjk_width("e\u0301") == 1

    def test_mixed(self):
        assert _cjk_width("a가b") == 4


# ── _wrap_cjk ────────────────────────────────────────────────────────────────


class TestWrapCjk:
    def test_empty_text_returns_single_indented_line(self):
        assert _wrap_cjk("", 20) == [""]

    def test_fits_on_one_line(self):
        assert _wrap_cjk("hello world", 20) == ["hello world"]

    def test_wraps_at_boundary(self):
        out = _wrap_cjk("aaa bbb ccc", 8)
        assert out == ["aaa bbb", "ccc"]

    def test_initial_and_subsequent_indent(self):
        out = _wrap_cjk("aaa bbb ccc", 8, initial_indent=">> ", subsequent_indent=".. ")
        # ">> " (3) + "aaa" (3) + " " + "bbb" (3) = 10 > 8 → break before bbb
        assert out[0] == ">> aaa"
        assert out[1] == ".. bbb"
        assert out[2] == ".. ccc"

    def test_hard_split_long_token(self):
        out = _wrap_cjk("abcdefgh", 5)
        # abcde / fgh — budget = avail - indent = 5
        assert out == ["abcde", "fgh"]

    def test_hard_split_cjk_token(self):
        out = _wrap_cjk("가나다라마", 6)
        # budget 6 → 가나다 (6 cols), then 라마 (4 cols)
        assert out == ["가나다", "라마"]

    def test_hard_split_loop_multiple_pieces(self):
        out = _wrap_cjk("abcdefghij", 4)
        assert out == ["abcd", "efgh", "ij"]

    def test_hard_split_cjk_mixed_with_wrap(self):
        # "가나다라마" (10 cols) with avail 6: hard split into 6/4; then " xyz" wraps
        out = _wrap_cjk("가나다라마 xyz", 6)
        assert out[0] == "가나다"
        assert out[1] == "라마"
        assert out[2] == "xyz"

    def test_token_exactly_fits_budget_no_split(self):
        out = _wrap_cjk("abcde", 5)
        assert out == ["abcde"]

    def test_single_char_word_sequence(self):
        out = _wrap_cjk("a b c d e", 3)
        assert out == ["a b", "c d", "e"]


# ── _split_work_state ────────────────────────────────────────────────────────


class TestSplitWorkState:
    def test_no_block_returns_body_unchanged(self):
        assert _split_work_state("hello\nworld") == ("hello\nworld", "")

    def test_empty_content(self):
        assert _split_work_state("") == ("", "")
        assert _split_work_state(None) == ("", "")

    def test_block_at_end_splits(self):
        body, work = _split_work_state("body text\n[WORK STATE] done\n")
        assert body == "body text"
        assert work == "[WORK STATE] done"

    def test_mid_sentence_mention_not_split(self):
        text = "mention [WORK STATE] mid-sentence"
        assert _split_work_state(text) == (text, "")

    def test_indented_block_still_detected(self):
        body, work = _split_work_state("body\n    [WORK STATE] x")
        assert body == "body"
        assert work == "[WORK STATE] x"

    def test_last_of_multiple_blocks_wins(self):
        body, work = _split_work_state("[WORK STATE] a\nmiddle\n[WORK STATE] b")
        assert body == "[WORK STATE] a\nmiddle"
        assert work == "[WORK STATE] b"

    def test_strips_surrounding_whitespace(self):
        body, work = _split_work_state("  body  \n  [WORK STATE] x  ")
        assert body == "body"
        assert work == "[WORK STATE] x"


# ── _build_turn_digest ───────────────────────────────────────────────────────


class TestBuildTurnDigest:
    def test_none_returns_empty(self):
        assert _build_turn_digest(None) == ""

    def test_empty_tool_results_returns_empty(self):
        assert _build_turn_digest(SimpleNamespace(tool_results=[])) == ""

    def test_with_tool_results_builds_digest(self):
        res = SimpleNamespace(
            tool_results=[
                {"tool": "read_file", "args": {"path": "a.py"}, "content": "", "ok": True},
            ]
        )
        out = _build_turn_digest(res)
        assert isinstance(out, str)
        assert "a.py" in out

    def test_shape_error_returns_empty(self):
        class _Bad:
            @property
            def tool_results(self):
                raise TypeError("boom")

        assert _build_turn_digest(_Bad()) == ""


# ── _build_orchestrator_digest ───────────────────────────────────────────────


def _sub(status="ok", patches=None, final=""):
    return SimpleNamespace(status=status, applied_patches=patches or [], final_message=final)


class TestBuildOrchestratorDigest:
    def test_none_returns_empty(self):
        assert _build_orchestrator_digest(None) == ""

    def test_no_subtasks_returns_empty(self):
        r = SimpleNamespace(status="done", total_turns=0, subtask_results=[])
        assert _build_orchestrator_digest(r) == ""

    def test_total_turns_line(self):
        r = SimpleNamespace(status="done", total_turns=3, subtask_results=[])
        out = _build_orchestrator_digest(r)
        assert "orchestration status: done" in out
        assert "total sub-agent turns: 3" in out

    def test_sub_with_patches_lists_files(self):
        r = SimpleNamespace(
            status="done",
            total_turns=0,
            subtask_results=[
                _sub(status="ok", patches=[{"file": "a.py"}, {"file": "b.py"}]),
            ],
        )
        out = _build_orchestrator_digest(r)
        assert "sub-1 [ok]: patched a.py, b.py" in out

    def test_patches_capped_at_5(self):
        patches = [{"file": f"f{i}.py"} for i in range(7)]
        r = SimpleNamespace(status="done", total_turns=0, subtask_results=[_sub(patches=patches)])
        out = _build_orchestrator_digest(r)
        assert "f0.py" in out and "f5.py" not in out

    def test_patch_non_dict_entry_str(self):
        r = SimpleNamespace(
            status="done",
            total_turns=0,
            subtask_results=[
                _sub(patches=["raw/path.py"]),
            ],
        )
        out = _build_orchestrator_digest(r)
        assert "raw/path.py" in out

    def test_sub_with_final_message(self):
        r = SimpleNamespace(
            status="done",
            total_turns=0,
            subtask_results=[
                _sub(status="fail", final="oops something broke"),
            ],
        )
        out = _build_orchestrator_digest(r)
        assert "sub-1 [fail]: oops something broke" in out

    def test_sub_with_nothing(self):
        r = SimpleNamespace(
            status="done",
            total_turns=0,
            subtask_results=[
                _sub(status="skip", final=""),
            ],
        )
        out = _build_orchestrator_digest(r)
        assert "sub-1 [skip]" in out

    def test_none_sub_skipped(self):
        r = SimpleNamespace(status="done", total_turns=0, subtask_results=[None, _sub(final="hi")])
        out = _build_orchestrator_digest(r)
        assert "sub-1" not in out
        assert "sub-2 [ok]: hi" in out

    def test_exception_returns_empty(self):
        def _gen():
            yield SimpleNamespace(status="x")
            raise ValueError("boom")

        r = SimpleNamespace(status="done", total_turns=0, subtask_results=_gen())
        assert _build_orchestrator_digest(r) == ""


# ── _validate_next_suggestion (remaining branches) ───────────────────────────


class TestValidateNextSuggestion:
    def test_none_variants_suppressed(self):
        for t in ("none", "None", "NONE", "none.", "none!"):
            assert _validate_next_suggestion(t, "user request") is None

    def test_verbatim_echo_suppressed(self):
        req = "please add a regression test for the parser"
        assert _validate_next_suggestion(f"First, the user said: '{req}'", req) is None

    def test_short_request_no_echo_guard(self):
        # < 8 chars → echo guard skipped
        assert _validate_next_suggestion("hi there", "short") == "hi there"

    def test_clean_suggestion_passes(self):
        assert _validate_next_suggestion("다음 단계로 진행", "한국어 요청") == "다음 단계로 진행"

    def test_hangul_request_requires_hangul_reply(self):
        assert _validate_next_suggestion("next step", "한국어 요청") is None

    def test_max_len_enforced(self):
        assert _validate_next_suggestion("x" * 141, "req") is None
        assert _validate_next_suggestion("x" * 140, "req") == "x" * 140

    def test_blank_suppressed(self):
        assert _validate_next_suggestion("   ", "req") is None


# ── _eval_ctrlc_armed ────────────────────────────────────────────────────────


class TestEvalCtrlCArmed:
    def test_not_main_prompt_always_raises(self):
        assert _eval_ctrlc_armed(False, True) == (False, True)
        assert _eval_ctrlc_armed(False, False) == (False, True)

    def test_buffer_has_text_clears(self):
        assert _eval_ctrlc_armed(True, True) == (True, False)

    def test_empty_main_prompt_raises_single_press(self):
        assert _eval_ctrlc_armed(True, False) == (False, True)


# ── _wrap_preserve_code ──────────────────────────────────────────────────────


class TestWrapPreserveCode:
    def test_plain_text_wrapped(self):
        out = _wrap_preserve_code("aaa bbb ccc", width=8)
        assert out == ["aaa bbb", "ccc"]

    def test_code_block_kept_intact(self):
        out = _wrap_preserve_code("```\ndef x():\n    pass\n```", width=8)
        assert out == ["```", "def x():", "    pass", "```"]

    def test_mixed_text_and_code(self):
        out = _wrap_preserve_code("before ```code here``` after", width=10)
        # text parts wrapped, code segment kept as a single line
        assert "```code here```" in out
        assert all(len(x) <= 10 or x.startswith("```") for x in out)

    def test_empty_text(self):
        assert _wrap_preserve_code("", width=8) == []

    def test_whitespace_only_segments_dropped(self):
        out = _wrap_preserve_code("   \n\n  ", width=8)
        assert out == []


# ── insights compact pure helpers ────────────────────────────────────────────


class TestInsightsCompactHelpers:
    def test_noop_same_normalized_text_and_count(self):
        assert _insights_compact_is_noop("a  b\nc", "a b c", [1, 2], [1, 2]) is True

    def test_not_noop_different_text(self):
        assert _insights_compact_is_noop("a b", "a b c", [1], [1]) is False

    def test_not_noop_different_count(self):
        assert _insights_compact_is_noop("a b", "a b", [1], [1, 2]) is False

    def test_empty_equal(self):
        assert _insights_compact_is_noop("", "", [], []) is True

    def test_size_compact_budget_floor(self):
        assert _size_compact_budget("") == 8192
        assert _size_compact_budget("hi") == 8192

    def test_size_compact_budget_scales(self):
        big = "가" * 20000  # 60000 bytes → 30001 tokens (canonical +1)
        assert _size_compact_budget(big) == 30001 + 2048

    def test_size_compact_budget_matches_canonical_estimator(self):
        """SSOT delegation — budget == canonical CJK-aware tokens + slack forall."""
        from external_llm.agent._shared_utils import _cjk_aware_tokens as _canonical

        for c in ["", "hi", "a" * 14000, "안" * 6000, "가" * 20000]:
            expected = max(8192, _canonical(c) + 2048)
            assert _size_compact_budget(c) == expected, f"{c[:20]!r} → {_size_compact_budget(c)} != {expected}"

    def test_dropped_entries_detects_missing(self):
        e1 = SimpleNamespace(header_line="### [a] 1")
        e2 = SimpleNamespace(header_line="### [b] 2")
        e3 = SimpleNamespace(header_line="### [a] 1")
        assert _dropped_entries([e1, e2], [e3]) == [e2]

    def test_dropped_entries_none_dropped(self):
        e1 = SimpleNamespace(header_line="h1")
        assert _dropped_entries([e1], [SimpleNamespace(header_line="h1")]) == []


# ── _extract_patched_file ────────────────────────────────────────────────────


class TestExtractPatchedFile:
    def test_dict_file_key(self):
        assert _extract_patched_file({"file": "a.py"}) == "a.py"

    def test_dict_path_key(self):
        assert _extract_patched_file({"path": "b.py"}) == "b.py"

    def test_dict_empty(self):
        assert _extract_patched_file({}) == ""
        assert _extract_patched_file({"file": ""}) == ""

    def test_empty_string(self):
        assert _extract_patched_file("") == ""
        assert _extract_patched_file(None) == ""

    def test_edit_file_prefix(self):
        assert _extract_patched_file("edit_file:src/x.py:123:456") == "src/x.py"

    def test_edit_text_prefix(self):
        assert _extract_patched_file("edit_text:src/y.py:1:2") == "src/y.py"

    def test_modify_symbol_prefix(self):
        assert _extract_patched_file("modify_symbol:src/z.py:func") == "src/z.py"

    def test_unified_diff_plus_plus(self):
        assert _extract_patched_file("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@") == "x.py"

    def test_diff_git_fallback(self):
        assert _extract_patched_file("diff --git a/old.py b/new.py") == "old.py"

    def test_no_match_returns_empty(self):
        assert _extract_patched_file("random text") == ""


# ── _result_output_dict / _build_json_output / _json_stream_emit ─────────────


def _mk_result(status="done", final="ok", error=None, metadata=None, patches=None, turns=None):
    return SimpleNamespace(
        status=status,
        final_message=final,
        error=error,
        metadata=metadata or {},
        applied_patches=patches or [],
        turns=turns or [],
    )


class TestResultOutputDict:
    def test_basic_done(self):
        r = _mk_result(
            metadata={"tokens": {"prompt": 10, "completion": 20, "total": 30, "cost_usd": 0.01}},
            patches=[{"file": "a.py"}],
            turns=["t1", "t2"],
        )
        d = _result_output_dict(r, 1.5)
        assert d["status"] == "done"
        assert d["output"] == "ok"
        assert d["duration_ms"] == 1500
        assert d["tokens_in"] == 10 and d["tokens_out"] == 20
        assert d["total_tokens"] == 30 and d["cost_usd"] == 0.01
        assert d["patches"] == 1
        assert d["patched_files"] == ["a.py"]
        assert d["turns"] == 2
        assert d["questions"] == []

    def test_int_turns_passthrough(self):
        r = _mk_result(turns=5)
        assert _result_output_dict(r, 0)["turns"] == 5

    def test_turns_fallback_to_metadata(self):
        r = _mk_result(turns=[], metadata={"turns_used": 7})
        assert _result_output_dict(r, 0)["turns"] == 7

    def test_cancelled_fills_error(self):
        r = _mk_result(status="cancelled", error="")
        assert _result_output_dict(r, 0)["error"] == "Request cancelled by user"

    def test_existing_error_kept(self):
        r = _mk_result(status="cancelled", error="custom")
        assert _result_output_dict(r, 0)["error"] == "custom"

    def test_clarification_questions_from_metadata(self):
        r = _mk_result(
            status="clarification_needed",
            metadata={
                "required_clarifications": [
                    {"field": "branch", "reason": "ambiguous"},
                    {"field": "only_field"},
                    "raw string question",
                ]
            },
        )
        q = _result_output_dict(r, 0)["questions"]
        assert q == ["branch: ambiguous", "only_field:", "raw string question"]

    def test_clarification_falls_back_to_final_message(self):
        r = _mk_result(status="clarification_needed", final="what do you mean?")
        assert _result_output_dict(r, 0)["questions"] == ["what do you mean?"]

    def test_clarification_no_fallback(self):
        r = _mk_result(status="clarification_needed", final="")
        assert _result_output_dict(r, 0)["questions"] == []

    def test_patched_files_drops_empties(self):
        r = _mk_result(patches=[{"file": "a.py"}, "junk", {"file": ""}])
        assert _result_output_dict(r, 0)["patched_files"] == ["a.py"]


class TestBuildJsonOutput:
    def test_success_path_prints_json(self, capsys):
        r = _mk_result(final="ok", metadata={"tokens": {"prompt": 1}})
        _build_json_output(r, 0.25)
        d = json.loads(capsys.readouterr().out)
        assert d["status"] == "done" and d["duration_ms"] == 250

    def test_exception_falls_back_to_error_json(self, capsys):
        class _Bad:
            @property
            def status(self):
                raise AttributeError("boom")

        _build_json_output(_Bad(), 0.1)
        d = json.loads(capsys.readouterr().out)
        assert d["status"] == "unexpected_error"
        assert "boom" in d["error"]


class TestJsonStreamEmit:
    def test_dict_payload_merged(self, capsys):
        _json_stream_emit("tool", {"tool": "read_file"})
        d = json.loads(capsys.readouterr().out)
        assert d == {"event": "tool", "tool": "read_file"}

    def test_non_dict_payload_wrapped(self, capsys):
        _json_stream_emit("x", 42)
        d = json.loads(capsys.readouterr().out)
        assert d == {"event": "x", "payload": 42}

    def test_extra_kwargs_merged(self, capsys):
        _json_stream_emit("x", extra=1)
        d = json.loads(capsys.readouterr().out)
        assert d == {"event": "x", "extra": 1}

    def test_none_payload(self, capsys):
        _json_stream_emit("x")
        d = json.loads(capsys.readouterr().out)
        assert d == {"event": "x"}

    def test_write_failure_never_raises(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("broken pipe")

        monkeypatch.setattr(repl_impl.sys.stdout, "write", _boom)
        _json_stream_emit("x")  # must not raise


# ── _ProgressPrinter terminal-independent helpers ────────────────────────────


@pytest.fixture
def printer():
    return repl_impl._ProgressPrinter(verbose=False)


class TestProgressPrinterStatics:
    def test_plain_strips_rich_markup(self):
        p = repl_impl._ProgressPrinter
        assert p._plain("[bold #89b4fa]hi[/bold] [dim]x[/dim]") == "hi x"

    def test_plain_no_markup(self):
        p = repl_impl._ProgressPrinter
        assert p._plain("plain [text") == "plain [text"

    def test_spinner_safe_zero_budget(self):
        p = repl_impl._ProgressPrinter
        assert p._spinner_safe("hi", 0) == ""
        assert p._spinner_safe("hi", -1) == ""

    def test_spinner_safe_collapses_whitespace(self):
        p = repl_impl._ProgressPrinter
        assert p._spinner_safe("a\n\n  b\tc", 50) == "a b c"

    def test_spinner_safe_truncates_cjk(self):
        p = repl_impl._ProgressPrinter
        # 가(2)+나(2)=4, 다음 글자는 6>5 → … 대체 → 정확히 5셀
        assert p._spinner_safe("가나다라마", 5) == "가나…"
        assert p._spinner_safe("가나", 5) == "가나"

    def test_spinner_safe_ascii_truncate(self):
        p = repl_impl._ProgressPrinter
        out = p._spinner_safe("abcdefghij", 6)
        assert out == "abcde…"

    def test_spinner_safe_empty(self):
        p = repl_impl._ProgressPrinter
        assert p._spinner_safe("", 20) == ""
        assert p._spinner_safe(None, 20) == ""

    def test_hex_ansi(self):
        p = repl_impl._ProgressPrinter
        on, off = p._hex_ansi("#89b4fa")
        assert on == "\x1b[38;2;137;180;250m"
        assert off == "\x1b[39m"

    def test_spinner_indent(self):
        p = repl_impl._ProgressPrinter()
        assert p._spinner_indent() == " " * (2 + (repl_impl._SEQ_W + 2))

    def test_fit_row_flattens_and_truncates(self, monkeypatch):
        # Scope the patch: a test-wide shutil.get_terminal_size patch leaks
        # into pytest's progress rendering (it calls get_terminal_size with a
        # `fallback` kwarg) -> TypeError -> xdist worker crash.
        with monkeypatch.context() as m:
            m.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((30, 24)))
            p = repl_impl._ProgressPrinter
            assert p._fit_row("a\rb\nc\td") == "a b c d"
            out = p._fit_row("x" * 100)
            assert out.endswith("…") and len(out) <= 30  # max_cols 29 + …

    def test_fit_row_cjk_wide(self, monkeypatch):
        with monkeypatch.context() as m:
            m.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((30, 24)))
            p = repl_impl._ProgressPrinter
            out = p._fit_row("가" * 20)  # 40 cols > 29 → truncated
            assert out.endswith("…")

    def test_style_row_no_brackets_unchanged(self):
        p = repl_impl._ProgressPrinter
        assert p._style_row("no brackets") == "no brackets"

    def test_style_row_dims_token(self):
        p = repl_impl._ProgressPrinter
        out = p._style_row("  [3]   ✓ bash", "#89b4fa")
        assert out.startswith("  \x1b[2m[3]\x1b[22m")
        assert "\x1b[38;2;137;180;250m✓\x1b[39m" in out

    def test_style_row_no_icon_color(self):
        p = repl_impl._ProgressPrinter
        out = p._style_row("  [1] body")
        assert "\x1b[2m[1]\x1b[22m" in out
        assert "\x1b[38;2;" not in out

    def test_style_row_ellipsis_icon_not_colored(self):
        p = repl_impl._ProgressPrinter
        out = p._style_row("  [1]… body", "#89b4fa")
        # '…' is the first non-space char → skipped (no color)
        assert out.count("\x1b[38;2;") == 0

    def test_shimmer_row_empty(self):
        p = repl_impl._ProgressPrinter
        assert p._shimmer_row("", 1.0) == ""

    def test_shimmer_row_all_space(self):
        p = repl_impl._ProgressPrinter
        assert p._shimmer_row("   ", 1.0) == "   "

    def test_shimmer_row_short_body_untouched(self):
        p = repl_impl._ProgressPrinter
        assert p._shimmer_row("  ◴ ab", 1.0) == "  ◴ ab"  # body len 2 < 4

    def test_shimmer_row_colors_body(self):
        p = repl_impl._ProgressPrinter
        out = p._shimmer_row("  ◴ bash tool running", 1.0)
        assert out.startswith("  ◴ ")
        assert "\x1b[38;2;" in out

    def test_shimmer_row_spaces_not_colored(self):
        import re as _re

        p = repl_impl._ProgressPrinter
        out = p._shimmer_row("  ◴ bash long tool", 5.0)
        # ANSI strip round-trip: body text preserved verbatim (spaces included)
        plain = _re.sub(r"\x1b\[[0-9;]*m", "", out)
        assert plain == "  ◴ bash long tool"
        assert "\x1b[38;2;" in out


class TestRenderLiveLine:
    def test_empty_inflight(self, printer):
        assert printer._render_live_line("○") == ""

    def _term(self, cols):
        return lambda *a, **kw: os.terminal_size((cols, 24))

    def test_single_tool_recent(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", self._term(80))
        printer._inflight = {"a": {"tool": "bash", "hint": "ls -la", "t0": time.perf_counter()}}
        line = printer._render_live_line("○")
        assert "bash" in line
        assert "ls -la" in line
        assert line.startswith("  ")

    def test_single_tool_with_elapsed(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", self._term(80))
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter() - 5}}
        line = printer._render_live_line("○")
        assert "5s" in line

    def test_multiple_tools_extra_count(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", self._term(80))
        t0 = time.perf_counter()
        printer._inflight = {
            "a": {"tool": "read_file", "hint": "", "t0": t0 - 2},
            "b": {"tool": "grep", "hint": "", "t0": t0},
        }
        line = printer._render_live_line("○")
        assert "(+1)" in line

    def test_long_hint_truncated(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", self._term(80))
        printer._inflight = {"a": {"tool": "bash", "hint": "x" * 500, "t0": time.perf_counter()}}
        line = printer._render_live_line("○")
        assert "…" in line or len(line) < 200

    def test_hint_dropped_when_no_room(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", self._term(30))
        printer._inflight = {"a": {"tool": "bash", "hint": "abcdef", "t0": time.perf_counter()}}
        line = printer._render_live_line("○")
        assert "bash" in line


class TestPopInflight:
    def test_exact_call_id_match(self, printer):
        printer._inflight = {"c1": {"tool": "bash"}, "anon-1": {"tool": "grep"}}
        assert printer._pop_inflight("c1") == {"tool": "bash"}
        assert "c1" not in printer._inflight

    def test_none_call_id_pops_oldest_anon(self, printer):
        printer._inflight = {"anon-1": {"tool": "grep"}, "anon-2": {"tool": "read_file"}}
        assert printer._pop_inflight(None) == {"tool": "grep"}

    def test_none_call_id_single_entry(self, printer):
        printer._inflight = {"x": {"tool": "bash"}}
        assert printer._pop_inflight(None) == {"tool": "bash"}
        assert printer._inflight == {}

    def test_none_call_id_multiple_named_no_steal(self, printer):
        printer._inflight = {"x": {"tool": "bash"}, "y": {"tool": "grep"}}
        assert printer._pop_inflight(None) is None
        assert len(printer._inflight) == 2

    def test_unknown_call_id_returns_none(self, printer):
        printer._inflight = {"c1": {"tool": "bash"}}
        assert printer._pop_inflight("nope") is None
        assert "c1" in printer._inflight

    def test_empty_inflight(self, printer):
        assert printer._pop_inflight(None) is None
        assert printer._pop_inflight("x") is None


class TestEmitPlanLinePlain:
    def test_plain_path_prints_joined_text(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._emit_plan_line([("a", "green"), ("b", None)])
        assert capsys.readouterr().out == "  ab\n"

    def test_plain_path_skips_palette(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._emit_plan_line([("x", "nonexistent_key")])
        assert capsys.readouterr().out == "  x\n"


class TestRenderPlanUpdate:
    @pytest.fixture(autouse=True)
    def _plain_mode(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((100, 24)))

    def test_no_items_returns(self, printer, capsys):
        printer._render_plan_update({"items": []}, None)
        assert capsys.readouterr().out == ""

    def test_non_dict_items_filtered(self, printer, capsys):
        printer._render_plan_update({"items": ["junk", {"title": "t", "status": "done"}]}, None)
        out = capsys.readouterr().out
        assert "╭─ Plan" in out
        assert "1/1" in out
        assert "✓ t" in out
        assert "junk" not in out

    def test_basic_plan_with_goal(self, printer, capsys):
        plan = {
            "goal": "Do the thing",
            "items": [
                {"title": "step one", "status": "done"},
                {"title": "step two", "status": "in_progress"},
                {"title": "step three", "status": "pending"},
            ],
        }
        printer._render_plan_update(plan, None)
        out = capsys.readouterr().out
        assert "Goal  Do the thing" in out
        assert "1/3" in out
        assert "✓ step one" in out
        assert "▸ step two" in out
        assert "○ step three" in out
        assert "33%" in out
        assert "2 open" in out

    def test_skipped_blocked_statuses(self, printer, capsys):
        plan = {
            "items": [
                {"title": "a", "status": "done"},
                {"title": "b", "status": "skipped"},
                {"title": "c", "status": "blocked", "note": "why"},
            ]
        }
        printer._render_plan_update(plan, None)
        out = capsys.readouterr().out
        assert "⊘ b" in out
        assert "✖ c" in out
        assert "why" in out
        assert "1 skipped" in out
        assert "1 blocked" in out

    def test_unknown_status_falls_back(self, printer, capsys):
        plan = {"items": [{"title": "a", "status": "weird"}]}
        printer._render_plan_update(plan, None)
        out = capsys.readouterr().out
        assert "○ a" in out

    def test_prev_changed_and_new_items(self, printer, capsys):
        plan = {
            "items": [
                {"title": "a", "status": "done"},
                {"title": "new", "status": "pending"},
            ]
        }
        printer._render_plan_update(plan, {"a": "pending"})
        out = capsys.readouterr().out
        assert "✓ a" in out
        assert "new" in out

    def test_collapse_many_done_items(self, printer, capsys):
        items = [{"title": f"item {i}", "status": "done"} for i in range(20)]
        plan = {"items": items}
        printer._render_plan_update(plan, {it["title"]: "done" for it in items})
        out = capsys.readouterr().out
        assert "20 done (unchanged)" in out
        assert "item 0" not in out

    def test_collapse_keeps_changed_items(self, printer, capsys):
        items = [{"title": f"item {i}", "status": "done"} for i in range(20)]
        items[3]["status"] = "in_progress"
        prev = {f"item {i}": "done" for i in range(20)}
        printer._render_plan_update({"items": items}, prev)
        out = capsys.readouterr().out
        assert "▸ item 3" in out
        assert "19 done (unchanged)" in out

    def test_removed_items_counted(self, printer, capsys):
        plan = {"items": [{"title": "a", "status": "done"}]}
        printer._render_plan_update(plan, {"a": "done", "gone": "pending"})
        out = capsys.readouterr().out
        assert "1 removed" in out

    def test_removed_acc_resets_on_fresh_plan(self, printer, capsys):
        plan = {"items": [{"title": "a", "status": "done"}]}
        printer._render_plan_update(plan, {"a": "done", "gone": "pending"})
        capsys.readouterr()
        printer._render_plan_update(plan, None)  # fresh plan → accumulator cleared
        out = capsys.readouterr().out
        assert "removed" not in out

    def test_alloc_exact_width_sum(self, printer, capsys):
        # 10-cell bar: 3 done + 4 skipped + 2 blocked + 1 open
        plan = {
            "items": [{"title": f"d{i}", "status": "done"} for i in range(3)]
            + [{"title": f"s{i}", "status": "skipped"} for i in range(4)]
            + [{"title": f"b{i}", "status": "blocked"} for i in range(2)]
            + [
                {"title": "o", "status": "pending"},
            ]
        }
        printer._render_plan_update(plan, None)
        out = capsys.readouterr().out
        # bar segments sum to 10 glyphs in the bottom border line
        foot = next(ln for ln in out.splitlines() if ln.startswith("  ╰─"))
        bar_chars = foot.split("╰─ ", 1)[1].split(" ")[0]
        assert len(bar_chars) == 10

    def test_narrow_terminal_stat_clipped(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((40, 24)))
        plan = {
            "items": [{"title": f"item {i}", "status": "done"} for i in range(3)]
            + [{"title": f"s{i}", "status": "skipped"} for i in range(3)]
            + [{"title": f"b{i}", "status": "blocked"} for i in range(3)]
            + [{"title": f"o{i}", "status": "pending"} for i in range(3)]
        }
        printer._render_plan_update(plan, None)
        out = capsys.readouterr().out
        foot = next(ln for ln in out.splitlines() if "╰─" in ln)
        assert foot.endswith("╯")


class TestConsumeLlmTokensStr:
    def test_empty_state(self, printer):
        assert printer._consume_llm_tokens_str() == ""

    def test_miss_only_no_provider(self, printer):
        printer._pending_llm_tokens = 500
        out = printer._consume_llm_tokens_str()
        assert out == " \x1b[2m· ↑500\x1b[22m"
        # consumed once
        assert printer._consume_llm_tokens_str() == ""

    def test_cache_hit_pct_shown(self, printer):
        printer._pending_llm_tokens = 400
        printer._pending_llm_cache_read = 100
        printer._pending_llm_provider = "zai"
        out = printer._consume_llm_tokens_str()
        assert "20% cached" in out  # 100 / (400+100)
        assert "↑500" in out

    def test_subset_provider_miss_only(self, printer):
        printer._pending_llm_tokens = 400
        printer._pending_llm_cache_read = 100
        printer._pending_llm_provider = "openai"
        out = printer._consume_llm_tokens_str()
        assert "25% cached" in out  # 100 / 400

    def test_cache_creation_included_for_separate(self, printer):
        printer._pending_llm_tokens = 400
        printer._pending_llm_cache_read = 100
        printer._pending_llm_cache_creation = 100
        printer._pending_llm_provider = "zai"
        out = printer._consume_llm_tokens_str()
        assert "↑600" in out
        assert "17% cached" in out  # 100/600 = 16.67 → rounds to 17

    def test_exception_path_falls_back(self, printer, monkeypatch):
        import external_llm.agent._shared_utils as _su

        monkeypatch.setattr(_su, "cache_hit_pct", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        printer._pending_llm_tokens = 400
        printer._pending_llm_cache_read = 100
        printer._pending_llm_provider = "zai"
        out = printer._consume_llm_tokens_str()
        assert out == " \x1b[2m· ↑500\x1b[22m"


# ── auto-continue global-state helpers ───────────────────────────────────────


@pytest.fixture
def auto_state():
    saved = {
        "state": dict(repl_impl._auto_continue_state),
        "gen": repl_impl._auto_submit_gen,
        "active": repl_impl._auto_countdown_active,
        "last": repl_impl._last_input_was_auto,
        "sug": repl_impl._next_prompt_suggestion,
        "sug_gen": repl_impl._next_suggestion_gen,
        "session": repl_impl._prompt_session,
        "underline": repl_impl._input_underline,
    }
    yield
    repl_impl._auto_continue_state.update(saved["state"])
    repl_impl._auto_submit_gen = saved["gen"]
    repl_impl._auto_countdown_active = saved["active"]
    repl_impl._last_input_was_auto = saved["last"]
    repl_impl._next_prompt_suggestion = saved["sug"]
    repl_impl._next_suggestion_gen = saved["sug_gen"]
    repl_impl._prompt_session = saved["session"]
    repl_impl._input_underline = saved["underline"]


class TestAutoSubmitHelpers:
    def test_should_arm_already_covered(self):
        assert _auto_continue_should_arm(False, 0, 5, "s") == (False, "off")

    def test_maybe_arm_off(self, auto_state, monkeypatch, capsys):
        timers = []
        monkeypatch.setattr(threading, "Timer", lambda *a, **k: timers.append(a) or object())
        repl_impl._auto_continue_state.update({"on": False, "depth": 0, "cap": 5})
        _maybe_arm_auto_submit()
        assert timers == []

    def test_maybe_arm_no_suggestion(self, auto_state, monkeypatch):
        timers = []
        monkeypatch.setattr(threading, "Timer", lambda *a, **k: timers.append(a) or object())
        repl_impl._auto_continue_state.update({"on": True, "depth": 0, "cap": 5})
        repl_impl._next_prompt_suggestion = ""
        _maybe_arm_auto_submit()
        assert timers == []

    def test_maybe_arm_cap_reached_resets_depth(self, auto_state, monkeypatch):
        timers = []
        seen = []
        monkeypatch.setattr(threading, "Timer", lambda *a, **k: timers.append(a) or object())
        monkeypatch.setattr(repl_impl, "_print", lambda t, c: seen.append((t, c)))
        repl_impl._auto_continue_state.update({"on": True, "depth": 5, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        assert repl_impl._auto_continue_state["depth"] == 0
        assert any("cap reached" in t for t, _ in seen)
        assert timers == []

    def test_maybe_arm_arms_timer(self, auto_state, monkeypatch):
        timers = []
        seen = []

        def _fake_timer(*a, **k):
            t = SimpleNamespace(daemon=True, start=lambda: None)
            timers.append(a)
            return t

        monkeypatch.setattr(threading, "Timer", _fake_timer)
        monkeypatch.setattr(repl_impl, "_print", lambda t, c: seen.append((t, c)))
        repl_impl._auto_continue_state.update({"on": True, "depth": 1, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        assert len(timers) == 1
        assert repl_impl._auto_countdown_active is True
        assert any("2/5" in t for t, _ in seen)

    def test_auto_submit_now_gen_mismatch(self, auto_state):
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = True
        repl_impl._auto_submit_gen = 7
        repl_impl._next_suggestion_gen = 8
        _auto_submit_now(0, 0)  # gen mismatch
        assert repl_impl._last_input_was_auto is False

    def test_auto_submit_now_auto_off(self, auto_state):
        repl_impl._auto_continue_state.update({"on": False})
        repl_impl._input_underline = True
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)
        assert repl_impl._last_input_was_auto is False

    def test_auto_submit_now_no_app(self, auto_state):
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = True
        repl_impl._prompt_session = None
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)
        assert repl_impl._last_input_was_auto is False

    def test_auto_submit_now_success(self, auto_state):
        class _Buf:
            def __init__(self):
                self.text = ""

            def insert_text(self, t):
                self.text = t

            def validate_and_handle(self):
                pass

        buf = _Buf()
        app = SimpleNamespace(is_running=True, current_buffer=buf)
        repl_impl._prompt_session = SimpleNamespace(app=app)
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = True
        repl_impl._next_prompt_suggestion = "sug"
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)
        assert repl_impl._last_input_was_auto is True
        assert buf.text == "sug"
        assert repl_impl._auto_countdown_active is False

    def test_auto_submit_now_buffer_has_text(self, auto_state):
        class _Buf:
            text = "typed"

            def insert_text(self, t):
                pass

            def validate_and_handle(self):
                pass

        app = SimpleNamespace(is_running=True, current_buffer=_Buf())
        repl_impl._prompt_session = SimpleNamespace(app=app)
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = True
        repl_impl._next_prompt_suggestion = "sug"
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)
        assert repl_impl._last_input_was_auto is False

    def test_auto_submit_now_exception(self, auto_state):
        class _Buf:
            text = ""

            def insert_text(self, t):
                pass

            def validate_and_handle(self):
                raise RuntimeError("boom")

        app = SimpleNamespace(is_running=True, current_buffer=_Buf())
        repl_impl._prompt_session = SimpleNamespace(app=app)
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = True
        repl_impl._next_prompt_suggestion = "sug"
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)  # must not raise
        assert repl_impl._last_input_was_auto is False

    def test_notify_above_prompt_success(self, auto_state, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl, "_print", lambda t, c: seen.append((t, c)))
        _notify_above_prompt("hello", "red")
        assert seen == [("hello", "red")]

    def test_notify_above_prompt_swallows_error(self, auto_state, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("print failed")

        monkeypatch.setattr(repl_impl, "_print", _boom)
        _notify_above_prompt("hello", "red")  # must not raise

    def test_deliver_gen_mismatch_discards(self, auto_state):
        _deliver_next_suggestion("sug", gen=999)
        assert repl_impl._next_prompt_suggestion == ""

    def test_deliver_stores_without_session(self, auto_state):
        repl_impl._next_suggestion_gen = 5
        _deliver_next_suggestion("sug", 5)
        assert repl_impl._next_prompt_suggestion == "sug"

    def test_deliver_app_not_running_stores_only(self, auto_state):
        app = SimpleNamespace(is_running=False)
        repl_impl._prompt_session = SimpleNamespace(app=app)
        repl_impl._next_suggestion_gen = 5
        _deliver_next_suggestion("sug", 5)
        assert repl_impl._next_prompt_suggestion == "sug"

    def test_deliver_applies_via_loop(self, auto_state):
        from prompt_toolkit.auto_suggest import Suggestion

        class _Buf:
            def __init__(self):
                self.text = ""
                self.suggestion = None

        buf = _Buf()
        app = SimpleNamespace(
            is_running=True,
            current_buffer=buf,
            invalidate=lambda: None,
            loop=SimpleNamespace(call_soon_threadsafe=lambda fn: fn()),
        )
        repl_impl._prompt_session = SimpleNamespace(app=app)
        repl_impl._input_underline = True
        repl_impl._next_suggestion_gen = 5
        repl_impl._auto_continue_state.update({"on": False})
        _deliver_next_suggestion("sug", 5)
        assert isinstance(buf.suggestion, Suggestion)
        assert buf.suggestion.text == "sug"

    def test_deliver_exception_swallowed(self, auto_state):
        class _App:
            is_running = True

            @property
            def current_buffer(self):
                raise RuntimeError("boom")

            loop = SimpleNamespace(call_soon_threadsafe=lambda fn: fn())

        repl_impl._prompt_session = SimpleNamespace(app=_App())
        repl_impl._input_underline = True
        repl_impl._next_suggestion_gen = 5
        _deliver_next_suggestion("sug", 5)  # must not raise


# ── misc existing-contract re-verification (cheap) ───────────────────────────


class TestMiscContracts:
    def test_format_result_last_call_branch(self):
        r = SimpleNamespace(
            metadata={
                "tokens": {
                    "prompt": 1000,
                    "completion": 2000,
                    "total": 3000,
                    "last_call_prompt": 100,
                    "last_call_completion": 200,
                }
            }
        )
        out = _format_result(r)
        assert "last" in out
        assert "1.0K" in out

    def test_parse_auto_digit_zero_invalid(self):
        assert _parse_auto_arg("0", False) == (
            None,
            None,
            "usage: /auto [N | on | off]  (N = max consecutive auto steps)",
        )

    def test_parse_auto_digit_valid(self):
        assert _parse_auto_arg("3", False) == (True, 3, None)

    def test_text_has_hangul_ranges(self):
        assert _text_has_hangul("가") is True  # AC00-D7A3
        assert _text_has_hangul("ᄀ") is True  # 1100-11FF Jamo
        assert _text_has_hangul("ㄱ") is True  # 3130-318F compat Jamo
        assert _text_has_hangul("abc") is False


# ── 추가 순수 표면 (2차 패스) ─────────────────────────────────────────────────


class TestSetCompleterContext:
    def test_sets_globals(self):
        repl_impl._completer_provider = ""
        repl_impl._completer_model = ""
        repl_impl._set_completer_context("openai", "gpt-4o")
        assert repl_impl._completer_provider == "openai"
        assert repl_impl._completer_model == "gpt-4o"

    def test_empty_values_default(self):
        repl_impl._set_completer_context(None, "")
        assert repl_impl._completer_provider == ""


class TestOrchestratorResultToAgentLike:
    def test_empty(self):
        r = SimpleNamespace(status="success", summary="", subtask_results=[], metadata={})
        out = repl_impl._orchestrator_result_to_agent_like(r)
        assert out.status == "success"
        assert out.error is None
        assert out.turns == 0
        assert out.applied_patches == []

    def test_aggregates_patches_and_turns(self):
        r = SimpleNamespace(
            status="done",
            summary="s",
            subtask_results=[
                SimpleNamespace(applied_patches=[{"file": "a.py"}], turns=3),
                SimpleNamespace(applied_patches=["edit_file:b.py:1:2"], turns=[1, 2]),
                None,
            ],
            metadata={"k": "v"},
        )
        out = repl_impl._orchestrator_result_to_agent_like(r)
        assert out.applied_patches == [{"file": "a.py"}, "edit_file:b.py:1:2"]
        assert out.turns == 5  # 3 + len([1, 2])
        assert out.metadata == {"k": "v"}

    def test_failure_status_error_filled(self):
        r = SimpleNamespace(status="failed", summary="boom", subtask_results=[])
        out = repl_impl._orchestrator_result_to_agent_like(r)
        assert out.error == "boom"

    def test_none_result(self):
        out = repl_impl._orchestrator_result_to_agent_like(None)
        assert out.status == "success"
        assert out.turns == 0


class TestWrapCjkEdge:
    def test_for_else_piece_when_indent_eats_budget(self):
        # ind_w=5, avail=5 → budget clamped to 1 → single-char word fits entirely
        # (for-else `rest = ""` branch)
        assert _wrap_cjk("a", 5, initial_indent=".....") == [".....a"]


class TestConsumeLlmTokensZeroTotal:
    def test_subset_provider_zero_total_returns_empty(self, printer):
        # openai: total = prompt_tok = 0 → `not _total` → ""
        printer._pending_llm_tokens = 0
        printer._pending_llm_cache_read = 50
        printer._pending_llm_provider = "openai"
        assert printer._consume_llm_tokens_str() == ""


class _FakeConsoleFile:
    def reset_bol(self):
        pass


class _FakeConsole:
    def __init__(self):
        self.file = _FakeConsoleFile()
        self.lines = []
        self.width = None

    def print(self, t, end="\n"):
        self.lines.append(getattr(t, "plain", str(t)))


class TestProgressPrinterEvent:
    def _plain(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        monkeypatch.setattr(repl_impl, "_ensure_out_console_imported", lambda: None)

    def _rich(self, monkeypatch):
        fake = _FakeConsole()
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl, "_ensure_out_console_imported", lambda: None)
        monkeypatch.setattr(repl_impl.asi, "_out_console", fake)
        return fake

    def test_event_plain_with_icon(self, printer, monkeypatch, capsys):
        self._plain(monkeypatch)
        printer._event(1.25, "✓", "done", "green")
        out = capsys.readouterr().out
        assert "[  1.2s] ✓ done" in out

    def test_event_plain_no_icon(self, printer, monkeypatch, capsys):
        self._plain(monkeypatch)
        printer._event(0.5, "", "msg", "red")
        out = capsys.readouterr().out
        assert "[  0.5s] msg" in out

    def test_event_rich_path(self, printer, monkeypatch):
        fake = self._rich(monkeypatch)
        printer._event(2.5, "✗", "boom", "red", "red")
        assert any("✗ boom" in ln for ln in fake.lines)

    def test_emit_plan_line_rich(self, printer, monkeypatch):
        fake = self._rich(monkeypatch)
        printer._emit_plan_line([("a", "green"), ("b", None)])
        assert any("ab" in ln for ln in fake.lines)

    def test_render_plan_update_rich(self, printer, monkeypatch):
        fake = self._rich(monkeypatch)
        fake.width = 60
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((100, 24)))
        printer._render_plan_update({"items": [{"title": "t", "status": "done"}]}, None)
        assert any("╭─ Plan" in ln for ln in fake.lines)
        assert any("✓ t" in ln for ln in fake.lines)

    def test_make_spinner(self, printer):
        sp = printer._make_spinner("thinking", repl_impl._C["blue"])
        rendered = sp.render(0)
        assert "thinking" in str(rendered)


class TestRefreshLiveLine:
    def test_empty_inflight_clears_filter(self, printer, monkeypatch):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        printer._refresh_live_line()
        assert printer._live_drawn is False
        assert repl_impl._tool_running_filter.active is False

    def test_inflight_draws_live_line(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter()}}
        try:
            printer._refresh_live_line()
            out = capsys.readouterr().out
            assert "\r\x1b[2K" in out
            assert printer._live_drawn is True
            assert repl_impl._tool_running_filter.active is True
        finally:
            printer._inflight = {}
            printer._live_drawn = False
            repl_impl._tool_running_filter.active = False


class TestToolTicker:
    def test_start_no_tty(self, printer, monkeypatch):
        monkeypatch.setattr(repl_impl.sys.stdout, "isatty", lambda: False)
        printer._start_tool_ticker()
        assert printer._ticker_thread is None

    def test_start_stop_with_isatty(self, printer, monkeypatch):
        monkeypatch.setattr(repl_impl.sys.stdout, "isatty", lambda: True)
        try:
            printer._start_tool_ticker()
            assert printer._ticker_thread is not None
            assert printer._ticker_thread.is_alive()
            first = printer._ticker_thread
            printer._start_tool_ticker()  # already alive → no second thread
            assert printer._ticker_thread is first
            printer._stop_tool_ticker()
            assert printer._ticker_thread is None
        finally:
            printer._stop_tool_ticker()
            printer._ticker_stop = None

    def test_worker_renders_when_conditions_met(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter() - 5}}
        printer._live_drawn = True
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            return calls["n"] > 1  # first: enter loop; second: exit

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        try:
            printer._tool_ticker_worker(threading.Event())
            out = capsys.readouterr().out
            assert "\r\x1b[2K" in out
            import re as _re

            plain = _re.sub(r"\x1b\[[0-9;]*m", "", out)
            assert "bash" in plain
        finally:
            printer._inflight = {}
            printer._live_drawn = False


class TestThinkingTicker:
    def test_start_stop_spinner_running(self, printer, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._spinner_running = True
        try:
            printer._start_thinking_ticker()
            assert printer._think_tick_thread is not None
            assert printer._think_tick_thread.is_alive()
            printer._stop_thinking_ticker()
            assert printer._think_tick_thread is None
        finally:
            printer._stop_thinking_ticker()
            printer._think_tick_stop = None

    def test_start_respawns_spinner(self, printer, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._spinner_running = False
        try:
            printer._start_thinking_ticker()
            assert printer._spinner_running is True
        finally:
            printer._stop_spinner()
            printer._stop_thinking_ticker()

    def test_worker_timer_branch(self, printer, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._spinner_running = True
        printer._think_tick_t0 = time.perf_counter() - 5
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            return calls["n"] > 1

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        try:
            printer._thinking_ticker_worker(threading.Event())
            assert "5.0s" in printer._spinner_msg
        finally:
            printer._spinner_running = False


class TestMute:
    def test_mute_stops_and_ignores(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer.mute()
        assert printer._muted is True
        printer("design_tool_call", {"tool": "bash", "status": "running"})
        assert capsys.readouterr().out == ""


class TestTurnsToIntGuards:
    def test_bool_and_float_shapes(self):
        assert repl_impl._turns_to_int(True) == 0
        assert repl_impl._turns_to_int(3.7) == 3
        assert repl_impl._turns_to_int(None) == 0


class TestInvalidateCancel:
    def test_invalidate_and_cancel(self, auto_state):
        repl_impl._next_prompt_suggestion = "s"
        gen0 = repl_impl._next_suggestion_gen
        repl_impl._invalidate_next_suggestion()
        assert repl_impl._next_prompt_suggestion == ""
        assert repl_impl._next_suggestion_gen == gen0 + 1
        g = repl_impl._auto_submit_gen
        repl_impl._cancel_auto_submit()
        assert repl_impl._auto_submit_gen == g + 1
        assert repl_impl._auto_countdown_active is False


# ── _ProgressPrinter.__call__ 이벤트 디스패처 (plain 모드) ─────────────────────


class TestProgressPrinterCall:
    @staticmethod
    def _strip(out):
        return re.sub(r"\x1b\[[0-9;]*m", "", out)

    @pytest.fixture
    def plain(self, printer, monkeypatch):
        """_RICH off + _print captured + fixed terminal width."""
        prints = []
        monkeypatch.setattr(repl_impl, "_RICH", False)
        monkeypatch.setattr(repl_impl, "_ensure_out_console_imported", lambda: None)
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": prints.append((t, c)))
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((100, 24)))
        return prints

    def test_route_applied(self, printer, plain, capsys):
        printer("route_applied", {"lane": "main", "task_kind": "edit", "confidence": 0.9})
        out = capsys.readouterr().out
        assert "main · edit · 0.90" in out

    def test_rate_limit_retry(self, printer, plain, capsys):
        printer("rate_limit_retry", {"delay": 3, "attempt": 1, "max_retries": 5})
        out = capsys.readouterr().out
        assert "retry in 3s (1/5)" in out

    def test_tdd_events(self, printer, plain, capsys):
        printer("tdd_cycle_pass", {})
        assert "pass" in capsys.readouterr().out
        printer("tdd_cycle_fail", {})
        assert "fail" in capsys.readouterr().out

    def test_fail_loop(self, printer, plain, capsys):
        printer("fail_loop_detected", {})
        assert "switching strategy" in capsys.readouterr().out

    def test_error_event(self, printer, plain, capsys):
        printer("error", {"message": "boom"})
        assert "boom" in capsys.readouterr().out

    def test_cancelled_event(self, printer, plain, capsys):
        printer("cancelled", {"error": "user esc"})
        assert "user esc" in capsys.readouterr().out

    def test_reasoning_accumulates(self, printer, plain):
        printer("reasoning", {"text": "step1"})
        printer("reasoning", {"text": " step2"})
        assert printer._thinking_buffer == "step1 step2"

    def test_agent_thinking(self, printer, plain):
        printer("agent_thinking", {"content": "**bold**\n\nthinking about `x`"})
        assert "thinking about" in printer._spinner_msg

    def test_agent_thinking_dashes_collapsed(self, printer, plain):
        printer("agent_thinking", {"content": "a --- b"})
        assert "a b" in printer._spinner_msg

    def test_turn_start_gt1(self, printer, plain, capsys):
        printer("turn_start", {"turn": 3, "model": "gpt"})
        assert "Turn 3 (gpt)" in capsys.readouterr().out

    def test_turn_start_thinking_summary(self, printer, plain, capsys):
        printer._thinking_displayed = True
        printer._thinking_buffer = "x" * 900
        printer("turn_start", {"turn": 1})
        assert "(thinking 900 chars total)" in capsys.readouterr().out
        assert printer._thinking_buffer == ""

    def test_turn_start_resets_state(self, printer, plain):
        printer._thinking_displayed = True
        printer._thinking_buffer = "abc"
        printer("turn_start", {"turn": 1})
        assert printer._thinking_buffer == ""
        assert printer._thinking_displayed is False

    def test_token_usage_miss_only(self, printer, plain):
        printer(
            "token_usage",
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0.001,
                "provider": "",
            },
        )
        assert any("tok ↑100 ↓50" in t for t, _ in plain)

    def test_token_usage_cache_hit_suffix(self, printer, plain):
        printer(
            "token_usage",
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_prompt_tokens": 30,
                "total_completion_tokens": 40,
                "total_cost_usd": 0.1,
                "provider": "zai",
                "cache_read_tokens": 5,
                "total_cache_read_tokens": 15,
            },
        )
        assert any("↑10 ↓20" in t for t, _ in plain)
        assert any("% cached" in t for t, _ in plain)

    def test_token_usage_actual_cost_no_crash(self, printer, plain):
        printer(
            "token_usage",
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0.1,
                "total_actual_cost_usd": 0.05,
                "provider": "",
            },
        )
        assert len(plain) == 1

    def test_token_usage_null_fields_coerce_to_zero(self, printer, plain):
        # A malformed stream (or a direct handler call) may carry explicit nulls
        # on every token/cost field — the handler must coerce them to 0 rather
        # than TypeError the `+=` accumulation / f-string cost display.
        printer(
            "token_usage",
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_prompt_tokens": None,
                "total_completion_tokens": None,
                "total_cost_usd": None,
                "provider": "",
            },
        )
        assert any("tok ↑0 ↓0" in t for t, _ in plain)

    def test_token_usage_cache_hit_exception(self, printer, plain, monkeypatch):
        import external_llm.agent._shared_utils as _su

        def _boom(*a, **k):
            raise RuntimeError("pct failed")

        monkeypatch.setattr(_su, "cache_hit_pct", _boom)
        printer(
            "token_usage",
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_prompt_tokens": 30,
                "total_completion_tokens": 40,
                "total_cost_usd": 0.1,
                "provider": "zai",
                "cache_read_tokens": 5,
                "total_cache_read_tokens": 15,
            },
        )
        assert any("↑10 ↓20" in t for t, _ in plain)

    def test_design_llm_call_sets_pending(self, printer, plain):
        printer(
            "design_llm_call",
            {
                "prompt_tokens": 10,
                "cache_read_tokens": 5,
                "provider": "zai",
                "cache_creation_tokens": 2,
            },
        )
        assert printer._pending_llm_tokens == 10
        assert printer._pending_llm_cache_read == 5
        assert printer._pending_llm_provider == "zai"
        assert printer._pending_llm_cache_creation == 2

    def test_design_plan_gate(self, printer, plain, capsys):
        printer("design_plan_gate", {"open_items": ["a", "b", "c", "d"], "nudge": 1, "max_nudges": 3})
        out = capsys.readouterr().out
        assert "4 open item(s), nudge 1/3" in out
        assert "a; b; c" in out
        assert "+1 more" in out

    def test_design_plan_gate_no_titles(self, printer, plain, capsys):
        printer("design_plan_gate", {"open_items": [], "nudge": 2})
        assert "0 open item(s)" in capsys.readouterr().out

    def test_design_thinking_stop(self, printer, plain):
        printer("design_thinking_stop", {})
        assert printer._spinner_running is False

    def test_design_thinking_plain(self, printer, plain, capsys):
        printer("design_thinking", {"content": "line one\n\nline two", "elapsed": 1.5})
        out = capsys.readouterr().out
        assert "thought for 1.5s" in out
        assert "line one" in out
        assert "line two" in out

    def test_design_thinking_no_content(self, printer, plain, capsys):
        printer("design_thinking", {"content": ""})
        assert capsys.readouterr().out == ""

    def test_design_thinking_no_elapsed(self, printer, plain, capsys):
        printer("design_thinking", {"content": "x"})
        assert "thinking" in capsys.readouterr().out

    def test_self_review_start(self, printer, plain):
        printer("self_review", {"status": "start", "files_changed": 3})
        assert "self-review" in printer._spinner_msg
        printer("self_review", {"status": "start", "diff_chars": 500})
        assert "500 chars" in printer._spinner_msg

    def test_self_review_clean(self, printer, plain, capsys):
        printer("self_review", {"status": "clean", "elapsed": 0.5})
        assert "LGTM" in capsys.readouterr().out

    def test_self_review_failed(self, printer, plain, capsys):
        printer("self_review", {"status": "failed", "error": "no model"})
        assert "skipped" in capsys.readouterr().out

    def test_self_review_issues_plain(self, printer, plain, capsys):
        printer("self_review", {"status": "issues", "content": "fix this\n\nand that", "elapsed": 1.0})
        out = capsys.readouterr().out
        assert "issues found" in out
        assert "fix this" in out

    def test_scope_violation(self, printer, plain, capsys):
        printer("scope_violation", {"error": "out of scope", "source": "tool"})
        assert "scope violation (tool): out of scope" in capsys.readouterr().out

    def test_verbose_fallback_event(self, monkeypatch, capsys):
        printer = repl_impl._ProgressPrinter(verbose=True)
        monkeypatch.setattr(repl_impl, "_RICH", False)
        monkeypatch.setattr(repl_impl, "_ensure_out_console_imported", lambda: None)
        printer("unknown_event", {"a": 1})
        assert "unknown_event" in capsys.readouterr().out

    def test_silent_events_skipped(self, printer, plain, capsys):
        printer("session_start", {})
        printer("done", {})
        assert capsys.readouterr().out == ""

    def test_tool_call_preview_grep(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "grep", "args": {"pattern": "foo", "path": "src"}})
        out = capsys.readouterr().out
        assert "grep" in out
        assert "'foo'" in out
        assert printer._preview_active is True
        assert printer._call_seq == 1
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_preview_no_args(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "bash"})
        out = capsys.readouterr().out
        assert "bash" in out
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_ok_no_preview(self, printer, plain, capsys):
        printer(
            "tool_call", {"tool": "read_file", "result": {"ok": True, "content": "`a.py` (10 lines) lines 1-10\nbody"}}
        )
        out = capsys.readouterr().out
        assert "✓ read_file" in out
        assert "(10 lines) lines 1-10" in out

    def test_tool_call_ok_verbose(self, printer, plain, capsys):
        printer = repl_impl._ProgressPrinter(verbose=True)
        printer("tool_call", {"tool": "bash", "result": {"ok": True, "content": ""}})
        assert "✓ bash" in capsys.readouterr().out

    def test_tool_call_bash_file_not_found(self, printer, plain, capsys):
        printer("tool_call", {"tool": "bash", "result": {"ok": False, "error": "No such file or directory"}})
        out = capsys.readouterr().out
        assert "No such file" in out

    def test_tool_call_error_generic(self, printer, plain, capsys):
        printer("tool_call", {"tool": "edit_text", "result": {"ok": False, "error": "match failed"}})
        out = capsys.readouterr().out
        assert "✗ edit_text" in out
        assert "match failed" in out

    def test_tool_call_ok_with_preview(self, printer, plain, capsys):
        printer._preview_active = True
        printer("tool_call", {"tool": "bash", "result": {"ok": True, "content": "first line"}})
        out = self._strip(capsys.readouterr().out)
        assert "✓ bash" in out
        assert printer._preview_active is False

    def test_design_tool_call_running(self, printer, plain, capsys):
        printer("design_tool_call", {"tool": "bash", "status": "running", "args": {"command": "ls -la"}})
        out = capsys.readouterr().out
        assert "\r\x1b[2K" in out
        assert len(printer._inflight) == 1
        assert printer._live_drawn is True
        printer._inflight = {}
        printer._live_drawn = False

    def test_design_tool_call_complete(self, printer, plain, capsys):
        printer._inflight = {"c1": {"tool": "bash", "hint": "ls -la", "t0": time.perf_counter() - 1.5}}
        printer("design_tool_call", {"tool": "bash", "status": "complete", "call_id": "c1", "preview": "", "args": {}})
        out = self._strip(capsys.readouterr().out)
        assert "✓ bash" in out
        assert printer._inflight == {}

    def test_design_tool_call_complete_no_inflight(self, printer, plain, capsys):
        printer("design_tool_call", {"tool": "bash", "status": "complete", "preview": "", "args": {}})
        assert "✓ bash" in self._strip(capsys.readouterr().out)

    def test_design_tool_call_plan(self, printer, plain, capsys):
        printer(
            "design_tool_call",
            {
                "tool": "update_plan",
                "status": "complete",
                "preview": "",
                "plan": {"items": [{"title": "t", "status": "done"}]},
                "args": {},
            },
        )
        assert "╭─ Plan" in capsys.readouterr().out

    def test_design_tool_call_read_file_preview(self, printer, plain):
        printer(
            "design_tool_call",
            {"tool": "read_file", "status": "complete", "preview": "`a.py` (5 lines) lines 1-5\nbody", "args": {}},
        )
        assert any("(5 lines) lines 1-5" in t for t, _ in plain)

    def test_design_tool_call_read_symbol_preview(self, printer, plain):
        printer(
            "design_tool_call",
            {
                "tool": "read_symbol",
                "status": "complete",
                "preview": "**function** `foo` defined in `a.py:10`\nbody",
                "args": {},
            },
        )
        assert any("[function] a.py:10" in t for t, _ in plain)

    def test_design_tool_call_generic_preview(self, printer, plain):
        printer("design_tool_call", {"tool": "bash", "status": "complete", "preview": "line one\nline two", "args": {}})
        assert any("line one" in t for t, _ in plain)

    def test_design_tool_call_error_stderr_label(self, printer, plain, capsys):
        printer(
            "design_tool_call", {"tool": "bash", "status": "error", "preview": "[stderr]\nreal error here", "args": {}}
        )
        assert "✗ bash" in self._strip(capsys.readouterr().out)
        assert any("real error here" in t for t, _ in plain)

    def test_design_tool_call_error_no_preview(self, printer, plain, capsys):
        printer("design_tool_call", {"tool": "bash", "status": "error", "preview": "", "args": {}})
        assert "✗ bash" in self._strip(capsys.readouterr().out)

    def test_design_tool_call_switching(self, printer, plain, capsys):
        printer("design_tool_call", {"tool": "bash", "status": "switching", "preview": "moving to agent"})
        assert "agent switch" in capsys.readouterr().out


# ── ticker/thinking 잔여 분기 ────────────────────────────────────────────────


class TestTickerRemainingBranches:
    def test_worker_skips_when_stop_set(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter() - 5}}
        printer._live_drawn = True
        stop = threading.Event()
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                stop.set()
                return False
            return True

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        try:
            printer._tool_ticker_worker(stop)
            assert capsys.readouterr().out == ""
        finally:
            printer._inflight = {}
            printer._live_drawn = False

    def test_worker_skips_pause_and_sub_second(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        repl_impl._esc_watcher_pause.set()
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter()}}
        printer._live_drawn = True
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            return calls["n"] > 2  # body 2회(continue) 후 탈출

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        try:
            printer._tool_ticker_worker(threading.Event())
            assert capsys.readouterr().out == ""
        finally:
            repl_impl._esc_watcher_pause.clear()
            printer._inflight = {}
            printer._live_drawn = False

    def test_thinking_worker_returns_when_spinner_stopped(self, printer, monkeypatch):
        printer._spinner_running = False
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            return False

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        printer._thinking_ticker_worker(threading.Event())
        assert calls["n"] == 1

    def test_spinner_worker_renders_loop(self, printer, monkeypatch, capsys):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        monkeypatch.setattr(repl_impl, "_RICH", False)
        printer._spinner_running = True
        printer._spinner_msg = "hello world"
        calls = {"n": 0}

        def _fake_sleep(t):
            calls["n"] += 1
            if calls["n"] >= 2:
                printer._spinner_running = False

        monkeypatch.setattr(repl_impl.time, "sleep", _fake_sleep)
        try:
            printer._spinner_worker()
            out = capsys.readouterr().err
            plain_out = re.sub(r"\x1b\[[0-9;]*m", "", out)
            assert "hello" in plain_out
            assert "\x1b[38;2;" in out  # shimmer colors
        finally:
            printer._spinner_running = False


# ── auto-continue fire closure + submit 브랜치 ───────────────────────────────


class TestAutoSubmitFire:
    def test_fire_no_app(self, auto_state, monkeypatch):
        captured = {}

        def _fake_timer(*a, **k):
            captured["fn"] = a[1]
            return SimpleNamespace(daemon=True, start=lambda: None)

        monkeypatch.setattr(threading, "Timer", _fake_timer)
        repl_impl._auto_continue_state.update({"on": True, "depth": 0, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        repl_impl._prompt_session = None
        captured["fn"]()  # app None → return

    def test_fire_app_not_running(self, auto_state, monkeypatch):
        captured = {}

        def _fake_timer(*a, **k):
            captured["fn"] = a[1]
            return SimpleNamespace(daemon=True, start=lambda: None)

        monkeypatch.setattr(threading, "Timer", _fake_timer)
        repl_impl._auto_continue_state.update({"on": True, "depth": 0, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        repl_impl._prompt_session = SimpleNamespace(app=SimpleNamespace(is_running=False))
        captured["fn"]()  # not running → return

    def test_fire_app_loop_none(self, auto_state, monkeypatch):
        captured = {}

        def _fake_timer(*a, **k):
            captured["fn"] = a[1]
            return SimpleNamespace(daemon=True, start=lambda: None)

        monkeypatch.setattr(threading, "Timer", _fake_timer)
        repl_impl._auto_continue_state.update({"on": True, "depth": 0, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        repl_impl._prompt_session = SimpleNamespace(app=SimpleNamespace(is_running=True, loop=None))
        captured["fn"]()  # loop None → return

    def test_fire_schedules_submit(self, auto_state, monkeypatch):
        captured = {}
        scheduled = []

        def _fake_timer(*a, **k):
            captured["fn"] = a[1]
            return SimpleNamespace(daemon=True, start=lambda: None)

        monkeypatch.setattr(threading, "Timer", _fake_timer)
        repl_impl._auto_continue_state.update({"on": True, "depth": 0, "cap": 5})
        repl_impl._next_prompt_suggestion = "sug"
        _maybe_arm_auto_submit()
        loop = SimpleNamespace(call_soon_threadsafe=scheduled.append)
        repl_impl._prompt_session = SimpleNamespace(app=SimpleNamespace(is_running=True, loop=loop))
        captured["fn"]()
        assert len(scheduled) == 1

    def test_auto_submit_now_underline_off(self, auto_state):
        repl_impl._auto_continue_state.update({"on": True})
        repl_impl._input_underline = False
        repl_impl._auto_submit_gen = 1
        repl_impl._next_suggestion_gen = 1
        _auto_submit_now(1, 1)
        assert repl_impl._last_input_was_auto is False

    def test_deliver_outer_exception(self, auto_state):
        class _Bad:
            @property
            def app(self):
                raise RuntimeError("boom")

        repl_impl._prompt_session = _Bad()
        repl_impl._next_suggestion_gen = 5
        _deliver_next_suggestion("sug", 5)
        assert repl_impl._next_prompt_suggestion == "sug"


# ── dotenv / provider / repo-root / terminal-config 헬퍼 ─────────────────────


class TestDotenvAndConfigHelpers:
    def test_save_key_creates_env(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": seen.append(t))
        _save_key_to_dotenv(str(tmp_path), "K", "v")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert content == 'K="v"\n'
        assert any("saved" in t for t in seen)

    def test_save_key_updates_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        (tmp_path / ".env").write_text('K="old"\nM="x"\n', encoding="utf-8")
        _save_key_to_dotenv(str(tmp_path), "K", "new")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert content == 'K="new"\nM="x"\n'

    def test_save_key_no_trailing_newline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        (tmp_path / ".env").write_text('M="x"', encoding="utf-8")
        _save_key_to_dotenv(str(tmp_path), "K", "v")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert content == 'M="x"\nK="v"\n'

    def test_save_key_write_failure_cleans_tmp(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": seen.append(t))

        def _boom(src, dst):
            raise OSError("read-only")

        monkeypatch.setattr(os, "replace", _boom)
        _save_key_to_dotenv(str(tmp_path), "K", "v")
        assert not (tmp_path / ".env.tmp").exists()
        assert any("could not persist" in t for t in seen)

    def test_save_key_write_failure_unlink_fails_too(self, tmp_path, monkeypatch):
        def _boom(src, dst):
            raise OSError("read-only")

        def _unlink_boom(p):
            raise OSError("nope")

        monkeypatch.setattr(os, "replace", _boom)
        monkeypatch.setattr(os, "unlink", _unlink_boom)
        _save_key_to_dotenv(str(tmp_path), "K", "v")  # must not raise

    def test_list_provider_model_choices(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_get_ollama_models", lambda timeout=5: ["local1"])
        choices = repl_impl._list_provider_model_choices()
        assert isinstance(choices, list) and choices
        assert ("ollama", "local1") in choices
        assert all(isinstance(c, tuple) and len(c) == 2 for c in choices)

    def test_interactive_setup_non_tty(self, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: False))
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": seen.append(t))
        assert repl_impl._interactive_provider_setup(None) is None

    def test_interactive_setup_no_choices(self, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(repl_impl, "_list_provider_model_choices", lambda: [])
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": seen.append(t))
        assert repl_impl._interactive_provider_setup(None) is None

    def test_interactive_setup_pick(self, monkeypatch, tmp_path):
        env_backup = (os.environ.get("EXTERNAL_LLM_PROVIDER"), os.environ.get("EXTERNAL_LLM_MODEL"))
        try:
            monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: True))
            monkeypatch.setattr(
                repl_impl, "_list_provider_model_choices", lambda: [("openai", "gpt-4o"), ("ollama", "local")]
            )
            monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "2")
            monkeypatch.setattr(repl_impl, "_save_key_to_dotenv", lambda *a, **k: None)
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
            picked = repl_impl._interactive_provider_setup(str(tmp_path))
            assert picked == ("ollama", "local")
            assert os.environ["EXTERNAL_LLM_PROVIDER"] == "ollama"
            assert os.environ["EXTERNAL_LLM_MODEL"] == "local"
        finally:
            if env_backup[0] is None:
                os.environ.pop("EXTERNAL_LLM_PROVIDER", None)
            else:
                os.environ["EXTERNAL_LLM_PROVIDER"] = env_backup[0]
            if env_backup[1] is None:
                os.environ.pop("EXTERNAL_LLM_MODEL", None)
            else:
                os.environ["EXTERNAL_LLM_MODEL"] = env_backup[1]

    def test_interactive_setup_eof_cancels(self, monkeypatch):
        def _raise(prompt):
            raise EOFError

        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(repl_impl, "_list_provider_model_choices", lambda: [("openai", "gpt-4o")])
        monkeypatch.setattr(repl_impl, "_collect_input", _raise)
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        assert repl_impl._interactive_provider_setup(None) is None

    def test_interactive_setup_invalid_selection(self, monkeypatch):
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(repl_impl, "_list_provider_model_choices", lambda: [("openai", "gpt-4o")])
        monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "9")
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        assert repl_impl._interactive_provider_setup(None) is None

    def test_interactive_setup_empty_cancels(self, monkeypatch):
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(repl_impl, "_list_provider_model_choices", lambda: [("openai", "gpt-4o")])
        monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "   ")
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        assert repl_impl._interactive_provider_setup(None) is None

    def test_retry_svc_factory_success(self):
        seen = {}

        def _factory(provider, model, api_key=None):
            seen.update(provider=provider, model=model, api_key=api_key)
            return "svc"

        assert _retry_create_svc_with_api_key_prompt(_factory, "openai", "gpt-4o", api_key="k") == "svc"
        assert seen == {"provider": "openai", "model": "gpt-4o", "api_key": "k"}

    def test_retry_svc_interactive_pick_creates(self, monkeypatch):
        monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EXTERNAL_LLM_MODEL", raising=False)
        calls = []

        def _factory(provider, model, api_key=None):
            calls.append((provider, model, api_key))
            if len(calls) >= 2:
                return "svc"
            return None

        monkeypatch.setattr(repl_impl, "_interactive_provider_setup", lambda repo_root: ("ollama", "local"))
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
        assert _retry_create_svc_with_api_key_prompt(_factory, None, None) == "svc"
        assert len(calls) == 2

    def test_retry_svc_interactive_cancel(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_interactive_provider_setup", lambda repo_root: None)
        assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, None, None) is None

    def test_retry_svc_api_key_prompt(self, monkeypatch, tmp_path):
        env_key = "OPENAI_API_KEY"
        backup = os.environ.get(env_key)
        os.environ.pop(env_key, None)
        calls = []

        def _factory(provider, model, api_key=None):
            calls.append(api_key)
            return "svc" if api_key == "sk-123" else None

        try:
            monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "sk-123")
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
            svc = _retry_create_svc_with_api_key_prompt(_factory, "openai", "gpt-4o", repo_root=str(tmp_path))
            assert svc == "svc"
            assert os.environ[env_key] == "sk-123"
            assert (tmp_path / ".env").exists()
        finally:
            if backup is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = backup

    def test_retry_svc_api_key_rejected(self, monkeypatch):
        env_key = "OPENAI_API_KEY"
        backup = os.environ.get(env_key)
        os.environ.pop(env_key, None)
        try:
            monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "bad")
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
            assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, "openai", "gpt-4o") is None
        finally:
            if backup is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = backup

    def test_retry_svc_api_key_cancelled(self, monkeypatch):
        env_key = "OPENAI_API_KEY"
        backup = os.environ.get(env_key)
        os.environ.pop(env_key, None)
        try:
            monkeypatch.setattr(repl_impl, "_collect_input", lambda p: "")
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
            assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, "openai", "gpt-4o") is None
        finally:
            if backup is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = backup

    def test_retry_svc_unknown_provider(self):
        assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, "mystery-provider", "m") is None


class TestOllamaCacheAndPaths:
    def test_ollama_cache_hit(self, monkeypatch):
        repl_impl._ollama_cache = ["m1"]
        repl_impl._ollama_cache_ts = time.monotonic()
        ran = {"n": 0}
        orig_run = subprocess.run
        try:

            def _fake_run(*a, **k):
                ran["n"] += 1
                return orig_run(*a, **k)

            monkeypatch.setattr(repl_impl.subprocess, "run", _fake_run)
            assert repl_impl._get_ollama_models() == ["m1"]
            assert ran["n"] == 0
        finally:
            repl_impl._ollama_cache = None

    def test_ollama_success_parses_names(self, monkeypatch):
        repl_impl._ollama_cache = None
        result = SimpleNamespace(returncode=0, stdout="NAME\tID\nllama3\tabc\nqwen2\tdef\n")
        monkeypatch.setattr(repl_impl.subprocess, "run", lambda *a, **k: result)
        assert repl_impl._get_ollama_models() == ["llama3", "qwen2"]

    def test_ollama_empty_stdout(self, monkeypatch):
        repl_impl._ollama_cache = None
        result = SimpleNamespace(returncode=0, stdout="")
        monkeypatch.setattr(repl_impl.subprocess, "run", lambda *a, **k: result)
        assert repl_impl._get_ollama_models() == []

    def test_ollama_nonzero_rc(self, monkeypatch):
        repl_impl._ollama_cache = None
        result = SimpleNamespace(returncode=1, stdout="")
        monkeypatch.setattr(repl_impl.subprocess, "run", lambda *a, **k: result)
        assert repl_impl._get_ollama_models() == []

    def test_ollama_exception(self, monkeypatch):
        repl_impl._ollama_cache = None

        def _boom(*a, **k):
            raise FileNotFoundError("ollama not installed")

        monkeypatch.setattr(repl_impl.subprocess, "run", _boom)
        assert repl_impl._get_ollama_models() == []

    def test_resolve_repo_root_arg(self):
        out = repl_impl._resolve_repo_root("some/dir")
        assert out == str(Path("some/dir").resolve())

    def test_resolve_repo_root_git(self, monkeypatch):
        result = SimpleNamespace(returncode=0, stdout="/repo/root\n")
        monkeypatch.setattr(repl_impl.subprocess, "run", lambda *a, **k: result)
        assert repl_impl._resolve_repo_root(None) == "/repo/root"

    def test_resolve_repo_root_git_failure_falls_back(self, monkeypatch):
        result = SimpleNamespace(returncode=1, stdout="")
        monkeypatch.setattr(repl_impl.subprocess, "run", lambda *a, **k: result)
        out = repl_impl._resolve_repo_root(None)
        assert out == str(Path(os.getcwd()).resolve())

    def test_resolve_repo_root_git_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise FileNotFoundError("git missing")

        monkeypatch.setattr(repl_impl.subprocess, "run", _boom)
        out = repl_impl._resolve_repo_root(None)
        assert out == str(Path(os.getcwd()).resolve())

    def test_terminal_config_path_no_tty(self, monkeypatch):
        def _ttyname(fd):
            raise OSError("not a tty")

        monkeypatch.setattr(os, "ttyname", _ttyname)
        assert repl_impl._terminal_config_path("/repo") is None

    def test_terminal_config_path_odd_name(self, monkeypatch):
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(fileno=lambda: 0))
        monkeypatch.setattr(os, "ttyname", lambda fd: "/dev/ttys004")
        assert repl_impl._terminal_config_path("/repo") == "/repo/.asicode/terminals/ttys004.json"

    def test_terminal_config_path_slash_name(self, monkeypatch):
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(fileno=lambda: 0))
        monkeypatch.setattr(os, "ttyname", lambda fd: "/")
        assert repl_impl._terminal_config_path("/repo") is None

    def test_seed_terminal_config_exists_noop(self, tmp_path):
        term = tmp_path / "t.json"
        term.write_text("{}", encoding="utf-8")
        repl_impl._seed_terminal_config(str(term), str(tmp_path / "shared.json"))
        assert term.read_text(encoding="utf-8") == "{}"

    def test_seed_terminal_config_copies_shared(self, tmp_path):
        shared = tmp_path / "shared.json"
        shared.write_text('{"model": "x"}', encoding="utf-8")
        term = tmp_path / "sub" / "t.json"
        repl_impl._seed_terminal_config(str(term), str(shared))
        assert term.read_text(encoding="utf-8") == '{"model": "x"}'

    def test_seed_terminal_config_no_shared(self, tmp_path):
        term = tmp_path / "t.json"
        repl_impl._seed_terminal_config(str(term), str(tmp_path / "missing.json"))
        assert term.read_text(encoding="utf-8") == "{}"

    def test_seed_terminal_config_oserror(self, tmp_path, monkeypatch):
        def _makedirs(p, exist_ok=False):
            raise OSError("denied")

        monkeypatch.setattr(os, "makedirs", _makedirs)
        repl_impl._seed_terminal_config(str(tmp_path / "t.json"), "/nope")  # no raise


# ── collaborate passthrough ──────────────────────────────────────────────────


class TestRunCollaborateSession:
    def test_passes_through(self, monkeypatch):
        import asyncio

        seen = {}

        class _FakeOrch:
            def __init__(self, registry, config):
                seen["registry"] = registry
                seen["config"] = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def run(self, task=None, context=None, enable_preprocessing=True):
                seen["task"] = task
                seen["context"] = context
                seen["preprocess"] = enable_preprocessing
                return "ok"

        monkeypatch.setattr("external_llm.repl.collaborate.CollaborationOrchestrator", _FakeOrch)
        out = asyncio.run(repl_impl._run_collaborate_session("reg", "cfg", "task", "ctx"))
        assert out == "ok"
        assert seen == {"registry": "reg", "config": "cfg", "task": "task", "context": "ctx", "preprocess": True}


# ── 3차 패스: rich 스피너/씽킹/결과 + 잔여 분기 ──────────────────────────────


class TestProgressPrinterRichSpinner:
    def _rich_env(self, monkeypatch):
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=80, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_console", console)
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((100, 24)))
        margin = SimpleNamespace(called=False)
        margin.reset_bol = lambda: setattr(margin, "called", True)
        monkeypatch.setattr(repl_impl.asi, "_margin_stderr", margin)
        return margin

    def test_rich_spinner_lifecycle(self, printer, monkeypatch):
        margin = self._rich_env(monkeypatch)
        printer._start_spinner("working")
        assert printer._spinner_live is not None
        assert printer._spinner_live.is_started
        printer._update_spinner("thinking")
        assert printer._spinner_obj is not None
        printer._suspend_live_for_log()
        assert printer._spinner_live is None
        assert margin.called
        # recreate after log emit
        printer._update_spinner("thinking  ·  1.0s")
        assert printer._spinner_live is not None
        # spinner_obj None + live exists → live.update fallback branch
        printer._spinner_obj = None
        printer._update_spinner("x")
        printer._stop_spinner()
        assert printer._spinner_live is None
        assert printer._spinner_running is False

    def test_design_thinking_rich(self, printer, monkeypatch):
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        printer("design_thinking", {"content": "**bold** thought here", "elapsed": 2.0})
        out = buf.getvalue()
        assert "thought for 2.0s" in out
        assert "thought here" in out

    def test_self_review_issues_rich(self, printer, monkeypatch):
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        printer("self_review", {"status": "issues", "content": "fix the bug", "elapsed": 1.0})
        out = buf.getvalue()
        assert "issues found" in out
        assert "fix the bug" in out

    def test_design_tool_call_complete_rich_margin(self, printer, monkeypatch):
        margin = self._rich_env(monkeypatch)
        printer("design_tool_call", {"tool": "bash", "status": "complete", "preview": "", "args": {}})
        assert margin.called

    def test_design_tool_call_error_rich_margin(self, printer, monkeypatch):
        margin = self._rich_env(monkeypatch)
        printer("design_tool_call", {"tool": "bash", "status": "error", "preview": "", "args": {}})
        assert margin.called

    def test_design_tool_call_running_rich_console_width(self, printer, monkeypatch):
        # _render_plan_update/_preview width path via _RICH True + console.width
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        printer(
            "design_tool_call",
            {
                "tool": "update_plan",
                "status": "complete",
                "preview": "",
                "plan": {"items": [{"title": "t", "status": "done"}]},
                "args": {},
            },
        )
        out = buf.getvalue()
        assert "╭─ Plan" in out


class TestProgressPrinterCallExtra:
    @staticmethod
    def _strip(out):
        return re.sub(r"\x1b\[[0-9;]*m", "", out)

    @pytest.fixture
    def plain(self, printer, monkeypatch):
        prints = []
        monkeypatch.setattr(repl_impl, "_RICH", False)
        monkeypatch.setattr(repl_impl, "_ensure_out_console_imported", lambda: None)
        monkeypatch.setattr(repl_impl, "_print", lambda t, c="": prints.append((t, c)))
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((100, 24)))
        return prints

    def test_routing_intent_starts_spinner(self, printer, plain):
        printer("routing_intent", {})
        assert printer._spinner_running is True
        printer("done", {})
        assert printer._spinner_running is False

    def test_route_decision_updates_spinner(self, printer, plain):
        printer("route_decision", {})
        assert printer._spinner_msg == "routing"

    def test_tool_call_preview_read_symbol_with_path(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "read_symbol", "args": {"name": "foo", "file_path": "a.py"}})
        assert "foo in a.py" in capsys.readouterr().out
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_preview_read_symbol_no_path(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "read_symbol", "args": {"name": "foo"}})
        assert "foo" in capsys.readouterr().out
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_preview_generic_arg(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "read_file", "args": {"path": "a.py"}})
        assert "a.py" in capsys.readouterr().out
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_preview_double_writes_newline(self, printer, plain, capsys):
        printer("tool_call_preview", {"tool": "bash"})
        capsys.readouterr()
        printer("tool_call_preview", {"tool": "bash"})
        out = capsys.readouterr().out
        assert out.startswith("\n")  # second preview: newline before \r
        printer._preview_active = False
        repl_impl._tool_running_filter.active = False

    def test_tool_call_ok_grep_suffix(self, printer, plain, capsys):
        printer(
            "tool_call",
            {"tool": "grep", "result": {"ok": True, "content": "grep: 'foo' in src (3 matches) (context)\nline"}},
        )
        out = capsys.readouterr().out
        assert "(3 matches) (context)" in out

    def test_tool_call_ok_unknown_tool_no_suffix(self, printer, plain, capsys):
        printer("tool_call", {"tool": "mystery", "result": {"ok": True, "content": "body"}})
        assert capsys.readouterr().out == ""

    def test_tool_call_ok_symbol_group_suffix(self, printer, plain, capsys):
        printer("tool_call", {"tool": "find_symbol", "result": {"ok": True, "content": "def foo() defined in a.py:10"}})
        assert "defined in a.py:10" in capsys.readouterr().out

    def test_tool_call_bash_err_with_preview(self, printer, plain, capsys):
        printer._preview_active = True
        printer("tool_call", {"tool": "bash", "result": {"ok": False, "error": "No such file or directory"}})
        out = self._strip(capsys.readouterr().out)
        assert "✓ bash" in out
        assert printer._preview_active is False

    def test_tool_call_err_with_preview(self, printer, plain, capsys):
        printer._preview_active = True
        printer("tool_call", {"tool": "edit_text", "result": {"ok": False, "error": "no match"}})
        out = self._strip(capsys.readouterr().out)
        assert "✗ edit_text" in out
        assert printer._preview_active is False

    def test_design_tool_call_hint_strip(self, printer, plain):
        printer._inflight = {"c1": {"tool": "grep", "hint": "grep x", "t0": time.perf_counter()}}
        printer(
            "design_tool_call",
            {"tool": "grep", "status": "complete", "call_id": "c1", "preview": "grep x\nin : result", "args": {}},
        )
        assert any("result" in t for t, _ in plain)

    def test_design_tool_call_preview_prefix_strip(self, printer, plain):
        # hint "bash" + first line "bash : result" → hint replaced, ": " prefix stripped
        printer._inflight = {"c1": {"tool": "bash", "hint": "bash", "t0": time.perf_counter()}}
        printer(
            "design_tool_call",
            {"tool": "bash", "status": "complete", "call_id": "c1", "preview": "bash : result\nline2", "args": {}},
        )
        assert any("result" in t for t, _ in plain)

    def test_design_thinking_start(self, printer, plain):
        try:
            printer("design_thinking_start", {})
            assert printer._think_tick_thread is not None
        finally:
            printer._stop_thinking_ticker()
            printer._stop_spinner()  # kill the spawned spinner thread (no leak)


class TestShowResult:
    def _plain(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)

    def test_success_plain(self, monkeypatch, capsys):
        self._plain(monkeypatch)
        r = SimpleNamespace(
            status="success", final_message="done\nsecond", error="", applied_patches=[1, 2], turns=["a"], metadata={}
        )
        repl_impl._show_result(r, 1.25)
        out = capsys.readouterr().out
        assert "✓ success" in out
        assert "1 turn" in out
        assert "2 patches" in out
        assert "done" in out

    def test_error_status(self, monkeypatch, capsys):
        self._plain(monkeypatch)
        r = SimpleNamespace(status="error", final_message="", error="boom", applied_patches=[], turns=[], metadata={})
        out = capsys.readouterr().out
        repl_impl._show_result(r, 0.5)
        out = capsys.readouterr().out
        assert "✗ error" in out
        assert "boom" in out

    def test_unknown_status(self, monkeypatch, capsys):
        self._plain(monkeypatch)
        r = SimpleNamespace(status="weird", final_message="", error="", applied_patches=[], turns=[], metadata={})
        repl_impl._show_result(r, 0.5)
        assert "· weird" in capsys.readouterr().out

    def test_diff_summary_hint(self, monkeypatch, capsys):
        self._plain(monkeypatch)
        seen = []
        prints = []
        import external_llm.agent.config.thresholds as _th

        saved = _th.config.display.RUN_DIFF
        object.__setattr__(_th.config.display, "RUN_DIFF", False)
        try:
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": prints.append(t))
            monkeypatch.setattr(repl_impl, "_print_run_change_summary", lambda a, b: seen.append((a, b)) or True)
            r = SimpleNamespace(status="success", final_message="", error="", applied_patches=[], turns=[], metadata={})
            repl_impl._show_result(r, 0.1, repo_root="/repo", baseline={"changed": True})
            assert seen == [("/repo", {"changed": True})]
            assert any("/diff" in t for t in prints)
        finally:
            object.__setattr__(_th.config.display, "RUN_DIFF", saved)

    def test_diff_run_diff_enabled(self, monkeypatch):
        self._plain(monkeypatch)
        calls = []
        monkeypatch.setattr(repl_impl, "_render_run_diff", lambda a, b: calls.append((a, b)))
        import external_llm.agent.config.thresholds as _th

        saved = _th.config.display.RUN_DIFF
        object.__setattr__(_th.config.display, "RUN_DIFF", True)
        try:
            r = SimpleNamespace(status="success", final_message="", error="", applied_patches=[], turns=[], metadata={})
            repl_impl._show_result(r, 0.1, repo_root="/repo", baseline={"changed": True})
            assert calls
        finally:
            object.__setattr__(_th.config.display, "RUN_DIFF", saved)

    def test_diff_summary_exception_swallowed(self, monkeypatch):
        self._plain(monkeypatch)

        def _boom(a, b):
            raise RuntimeError("git failed")

        monkeypatch.setattr(repl_impl, "_print_run_change_summary", _boom)
        r = SimpleNamespace(status="success", final_message="", error="", applied_patches=[], turns=[], metadata={})
        repl_impl._show_result(r, 0.1, repo_root="/repo", baseline={"changed": True})  # no raise

    def test_diff_skipped_on_error_status(self, monkeypatch, capsys):
        self._plain(monkeypatch)
        called = []
        monkeypatch.setattr(repl_impl, "_print_run_change_summary", lambda a, b: called.append(1) or False)
        r = SimpleNamespace(status="error", final_message="", error="x", applied_patches=[], turns=[], metadata={})
        repl_impl._show_result(r, 0.1, repo_root="/repo", baseline={"changed": True})
        assert called == []

    def test_rich_path(self, monkeypatch):
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        r = SimpleNamespace(
            status="already_satisfied",
            final_message="no change needed",
            error="",
            applied_patches=[],
            turns=["a"],
            metadata={},
        )
        repl_impl._show_result(r, 0.5)
        out = buf.getvalue()
        assert "already_satisfied" in out
        assert "no change needed" in out


class TestRetryAndPathExtras:
    def test_retry_svc_interactive_returns_none(self, monkeypatch):
        monkeypatch.delenv("EXTERNAL_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("EXTERNAL_LLM_MODEL", raising=False)
        monkeypatch.setattr(repl_impl, "_interactive_provider_setup", lambda repo_root: None)
        assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, None, None) is None

    def test_retry_svc_api_key_eof(self, monkeypatch):
        env_key = "OPENAI_API_KEY"
        backup = os.environ.get(env_key)
        os.environ.pop(env_key, None)

        def _raise(prompt):
            raise EOFError

        try:
            monkeypatch.setattr(repl_impl, "_collect_input", _raise)
            monkeypatch.setattr(repl_impl, "_print", lambda t, c="": None)
            assert _retry_create_svc_with_api_key_prompt(lambda *a, **k: None, "openai", "m") is None
        finally:
            if backup is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = backup

    def test_terminal_config_path_empty_name(self, monkeypatch):
        monkeypatch.setattr(repl_impl.sys, "stdin", SimpleNamespace(fileno=lambda: 0))
        monkeypatch.setattr(os, "ttyname", lambda fd: "")
        assert repl_impl._terminal_config_path("/repo") is None


class TestTickerPauseOrder:
    def test_worker_pause_then_sub_second(self, printer, monkeypatch, capsys):
        """pause continue(1st) → pause cleared → <1s continue(2nd) → exit(3rd)."""
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((80, 24)))
        repl_impl._esc_watcher_pause.set()
        printer._inflight = {"a": {"tool": "bash", "hint": "", "t0": time.perf_counter()}}
        printer._live_drawn = True
        calls = {"n": 0}

        def _fake_wait(self, timeout=None):
            calls["n"] += 1
            if calls["n"] == 2:
                repl_impl._esc_watcher_pause.clear()
            return calls["n"] > 2

        monkeypatch.setattr(threading.Event, "wait", _fake_wait)
        try:
            printer._tool_ticker_worker(threading.Event())
            assert capsys.readouterr().out == ""
        finally:
            repl_impl._esc_watcher_pause.clear()
            printer._inflight = {}
            printer._live_drawn = False


# ── 4차 패스: 예외 경로 + rich reset_bol + show_result 변형 ──────────────────


class _BolFile:
    """rich Console file with reset_bol + capture."""

    def __init__(self):
        import io as _io

        self._buf = _io.StringIO()
        self.bol = 0

    def reset_bol(self):
        self.bol += 1

    def write(self, s):
        return self._buf.write(s)

    def flush(self):
        return self._buf.flush()

    def isatty(self):
        return False

    def getvalue(self):
        return self._buf.getvalue()


class TestRichExceptionPaths:
    def _rich_env(self, monkeypatch):
        import io as _io

        from rich.console import Console

        console = Console(file=_io.StringIO(), width=80, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_console", console)
        margin = SimpleNamespace(called=False)
        margin.reset_bol = lambda: setattr(margin, "called", True)
        monkeypatch.setattr(repl_impl.asi, "_margin_stderr", margin)
        return margin

    def test_suspend_live_exception_paths(self, printer, monkeypatch):
        self._rich_env(monkeypatch)

        class _BadLive:
            def stop(self):
                raise RuntimeError("stop failed")

        def _bol_boom():
            raise RuntimeError("bol failed")

        printer._spinner_live = _BadLive()
        printer._spinner_obj = object()
        monkeypatch.setattr(repl_impl.asi, "_margin_stderr", SimpleNamespace(reset_bol=_bol_boom))
        printer._suspend_live_for_log()  # must not raise
        assert printer._spinner_live is None
        assert printer._spinner_obj is None

    def test_stop_spinner_exception_path(self, printer, monkeypatch):
        self._rich_env(monkeypatch)

        class _BadLive:
            def stop(self):
                raise RuntimeError("stop failed")

        printer._spinner_live = _BadLive()
        printer._spinner_obj = object()
        printer._stop_spinner()  # must not raise
        assert printer._spinner_live is None
        assert printer._spinner_running is False

    def test_design_thinking_rich_bol_and_blank_lines(self, printer, monkeypatch):
        from rich.console import Console

        f = _BolFile()
        console = Console(file=f, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        printer("design_thinking", {"content": "> quote\n\n**bold** thought here\n\n```\ncode\n```", "elapsed": 2.0})
        assert f.bol > 0
        out = f.getvalue()
        assert "thought for 2.0s" in out
        assert "thought here" in out
        assert "quote" in out

    def test_self_review_issues_rich_bol(self, printer, monkeypatch):
        from rich.console import Console

        f = _BolFile()
        console = Console(file=f, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        printer("self_review", {"status": "issues", "content": "fix the bug", "elapsed": 1.0})
        assert f.bol > 0
        assert "issues found" in f.getvalue()


class TestShowResultVariants:
    def _console(self, monkeypatch):
        import io as _io

        from rich.console import Console

        buf = _io.StringIO()
        console = Console(file=buf, width=60, force_terminal=False)
        monkeypatch.setattr(repl_impl, "_RICH", True)
        monkeypatch.setattr(repl_impl.asi, "_out_console", console)
        return buf

    def test_rich_with_error_tokens_patches(self, monkeypatch):
        buf = self._console(monkeypatch)
        r = SimpleNamespace(
            status="error",
            final_message="msg",
            error="boom",
            applied_patches=[{"file": "a.py"}],
            turns=["a", "b"],
            metadata={"tokens": {"prompt": 10, "completion": 5, "last_call_prompt": 1, "last_call_completion": 2}},
        )
        repl_impl._show_result(r, 1.0)
        out = buf.getvalue()
        assert "boom" in out
        assert "1 patch" in out
        assert "2 turns" in out
        assert "last" in out  # token line from _format_result

    def test_rich_empty_renderable(self, monkeypatch):
        buf = self._console(monkeypatch)
        r = SimpleNamespace(status="success", final_message="", error="", applied_patches=[], turns=[], metadata={})
        repl_impl._show_result(r, 0.5)
        assert "success" in buf.getvalue()

    def test_plain_token_line(self, monkeypatch, capsys):
        monkeypatch.setattr(repl_impl, "_RICH", False)
        r = SimpleNamespace(
            status="success",
            final_message="",
            error="",
            applied_patches=[],
            turns=[],
            metadata={"tokens": {"prompt": 500, "completion": 300}},
        )
        repl_impl._show_result(r, 0.5)
        assert "↑500" in capsys.readouterr().out

    def test_diff_render_exception_swallowed(self, monkeypatch):
        monkeypatch.setattr(repl_impl, "_RICH", False)

        def _boom(a, b):
            raise RuntimeError("diff failed")

        monkeypatch.setattr(repl_impl, "_render_run_diff", _boom)
        import external_llm.agent.config.thresholds as _th

        saved = _th.config.display.RUN_DIFF
        object.__setattr__(_th.config.display, "RUN_DIFF", True)
        try:
            r = SimpleNamespace(status="success", final_message="", error="", applied_patches=[], turns=[], metadata={})
            repl_impl._show_result(r, 0.1, repo_root="/repo", baseline={"changed": True})  # no raise
        finally:
            object.__setattr__(_th.config.display, "RUN_DIFF", saved)


class TestRunReplEntry:
    def test_run_repl_delegates(self, monkeypatch):
        seen = []
        monkeypatch.setattr(repl_impl, "_run_repl_impl", lambda a: seen.append(a))
        repl_impl.run_repl("ARGS")
        assert seen == ["ARGS"]


class TestRunWithCancel:
    """_run_with_cancel: thread orchestration with cancel/esc-watcher/stream-callback."""

    def _run(self, monkeypatch, loop, cancel, **kw):
        monkeypatch.setattr(repl_impl, "_drain_stdin", lambda: None)
        return repl_impl._run_with_cancel(loop, "req", "ctx", cancel, **kw)

    def test_happy_path(self, monkeypatch):
        loop = SimpleNamespace(run=lambda req, context: "ok")
        stop = threading.Event()
        events = []
        out = self._run(
            monkeypatch,
            loop,
            threading.Event(),
            esc_watcher_stop=stop,
            stream_callback=lambda e, d: events.append((e, d)),
        )
        assert out == "ok"
        assert stop.is_set()
        assert events == [("done", {})]

    def test_cancel_path(self, monkeypatch):
        cancel = threading.Event()
        cancel.set()

        def _slow(req, context):
            time.sleep(2)
            return "late"

        loop = SimpleNamespace(run=_slow)
        stop = threading.Event()
        out = self._run(monkeypatch, loop, cancel, esc_watcher_stop=stop)
        assert stop.is_set()
        assert out is None  # worker still running → early return

    def test_exception_propagates(self, monkeypatch):
        class _BoomError(Exception):
            pass

        def _run(req, context):
            raise _BoomError("x")

        loop = SimpleNamespace(run=_run)
        with pytest.raises(_BoomError):
            self._run(monkeypatch, loop, threading.Event())

    def test_callback_failure_swallowed(self, monkeypatch):
        loop = SimpleNamespace(run=lambda req, context: "ok")

        def _cb(e, d):
            raise RuntimeError("cb failed")

        out = self._run(monkeypatch, loop, threading.Event(), stream_callback=_cb)
        assert out == "ok"
