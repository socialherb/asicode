"""RED→GREEN: render_interrupt_tool_results — args/직렬화/예외 경로.

Full-detail renderer for interrupted tool-loop results (Option B).
Coverage gaps closed: args=None → 빈 str, json.dumps 실패 → repr,
콘텐츠 cap 초과 truncation, total_chars 예산 중단.
"""

from __future__ import annotations

from external_llm.agent.interrupt_tool_results import (
    MAX_ARGS_CHARS,
    PER_RESULT_CHARS,
    TOTAL_CHARS,
    render_interrupt_tool_results,
)


class _BadStr:
    """json.dumps(default=str)가 str()을 호출할 때 TypeError를 던지는 객체."""

    def __str__(self):
        raise TypeError("boom")


def test_empty_input_returns_empty_string():
    assert render_interrupt_tool_results([]) == ""
    assert render_interrupt_tool_results(None) == ""  # type: ignore[arg-type]


def test_single_ok_result_renders_header_and_block():
    out = render_interrupt_tool_results(
        [
            {"tool": "read_file", "args": {"path": "a.py"}, "content": "line1", "ok": True},
        ]
    )
    assert "[Interrupted tool-loop results — full detail preserved]" in out
    assert "[1 of 1 tool call(s) shown" in out
    assert "[1] read_file (ok)" in out
    assert 'args: {"path": "a.py"}' in out
    assert "result:\nline1" in out


def test_failed_result_marks_fail_status():
    out = render_interrupt_tool_results(
        [
            {"tool": "apply_patch", "args": {"patch": "x"}, "content": "", "ok": False},
        ]
    )
    assert "[1] apply_patch (FAIL)" in out


def test_args_none_omits_args_line():
    out = render_interrupt_tool_results(
        [
            {"tool": "bash", "args": None, "content": "out", "ok": True},
        ]
    )
    assert "(ok)" in out
    assert "args:" not in out


def test_args_scalar_stringified():
    out = render_interrupt_tool_results(
        [
            {"tool": "t", "args": 42, "content": "", "ok": True},
        ]
    )
    assert "args: 42" in out


def test_args_serialization_failure_falls_back_to_repr():
    bad = {"payload": _BadStr()}
    out = render_interrupt_tool_results(
        [
            {"tool": "t", "args": bad, "content": "", "ok": True},
        ]
    )
    assert "args:" in out
    assert "_BadStr" in out  # repr of the args dict


def test_long_args_truncated_to_max_chars():
    out = render_interrupt_tool_results(
        [
            {"tool": "t", "args": {"big": "x" * (MAX_ARGS_CHARS + 50)}, "content": "", "ok": True},
        ]
    )
    assert "…" in out
    args_line = out.split("args: ")[1].split("\n")[0]
    assert len(args_line.rstrip("…")) <= MAX_ARGS_CHARS + 1


def test_long_content_truncated_with_marker():
    out = render_interrupt_tool_results(
        [
            {"tool": "t", "args": {}, "content": "y" * (PER_RESULT_CHARS + 100), "ok": True},
        ]
    )
    assert "[truncated 100 chars]" in out


def test_total_budget_stops_after_first_block():
    out = render_interrupt_tool_results(
        [{"tool": "t", "args": {}, "content": "z" * 100, "ok": True} for _ in range(5)],
        total_chars=50,
    )
    assert "[1 of 5 tool call(s) shown" in out


def test_per_result_cap_below_total():
    out = render_interrupt_tool_results(
        [{"tool": "t", "args": {}, "content": "q" * 200, "ok": True}],
        per_result_chars=40,
        total_chars=TOTAL_CHARS,
    )
    assert "[truncated 160 chars]" in out


def test_tool_name_defaults_to_question_mark():
    out = render_interrupt_tool_results([{"content": "c", "ok": True}])
    assert "[1] ? (ok)" in out


def test_missing_content_key_treated_as_empty():
    out = render_interrupt_tool_results([{"tool": "t", "args": {}, "ok": True}])
    assert "(ok)" in out
    assert "result:" not in out


def test_budget_accounting_uses_block_length():
    # 전체 예산이 첫 블록보다 작아도 첫 블록은 렌더링된다 (break는 루프 시작에서만).
    out = render_interrupt_tool_results(
        [{"tool": "t", "args": {}, "content": "c", "ok": True}],
        total_chars=1,
    )
    assert out
    assert "[1 of 1 tool call(s) shown; budget" in out


def test_defaults_are_public_constants():
    assert PER_RESULT_CHARS > 0 and TOTAL_CHARS > 0
