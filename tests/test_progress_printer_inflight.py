"""_ProgressPrinter 의 동시 실행(in-flight) 툴 라인 렌더링 회귀 테스트.

배경: 툴은 ThreadPoolExecutor 로 병렬 실행되어 design_tool_call running/complete
이벤트가 섞여 들어온다. 과거에는 in-flight 상태를 단일 스칼라 슬롯으로 추적해,
늦게 도착한 complete 가 '가장 최근에 시작한 툴' 번호로 찍히며 앞선 ○ 라인들이
✓로 갱신되지 못한 채 화면에 남았다(번호 중복/뒤섞임). 이 테스트는 call_id 기반
매칭 + 완료-순서 번호 부여 + 단일 live 라인 모델이 그 버그류를 막는지 고정한다.
"""
import io
import sys

import asi


def _drive(events):
    """events 를 _ProgressPrinter.__call__ 에 흘려보내고, 한 줄짜리 터미널을
    에뮬레이트해 (committed_rows, raw_stream, printer) 를 돌려준다.

    \\r\\x1b[2K 는 현재 행을 비우고, \\n 은 현재 행을 확정(commit)한다.
    색/dim 등 표시용 ANSI 는 어서션 편의를 위해 떼어낸다."""
    printer = asi._ProgressPrinter()
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        for name, data in events:
            printer(name, data)
    finally:
        sys.stdout = real
    raw = buf.getvalue()

    committed = []
    row = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\x1b" and raw[i + 1:i + 2] == "[":
            j = i + 2
            while j < len(raw) and not raw[j].isalpha():
                j += 1
            seq = raw[i:j + 1]
            if seq.endswith("K"):  # \x1b[2K → 현재 행 비우기
                row = []
            # 그 외(색/dim)는 표시 속성 — 어서션엔 무관하므로 버린다
            i = j + 1
            continue
        if ch == "\r":
            i += 1
            continue
        if ch == "\n":
            committed.append("".join(row))
            row = []
            i += 1
            continue
        row.append(ch)
        i += 1
    return committed, raw, printer


def _run(cid, tool):
    return ("design_tool_call", {"call_id": cid, "tool": tool, "status": "running"})


def _done(cid, tool):
    return ("design_tool_call", {"call_id": cid, "tool": tool, "status": "complete", "preview": ""})


def _err(cid, tool):
    return ("design_tool_call", {"call_id": cid, "tool": tool, "status": "error", "preview": "boom"})


def test_interleaved_parallel_tools_number_in_completion_order():
    # 3개 동시 시작 → 완료가 시작과 다른 순서(b, a, c)로 도착.
    committed, _raw, printer = _drive([
        _run("a", "read_file"),
        _run("b", "grep"),
        _run("c", "read_symbol"),
        _done("b", "grep"),
        _done("a", "read_file"),
        _done("c", "read_symbol"),
    ])
    # 확정된 ✓ 라인은 정확히 3개, 고아 ○ 0개
    check_rows = [r for r in committed if r.strip()]
    assert len(check_rows) == 3, check_rows
    assert all("✓" in r for r in check_rows), check_rows
    assert sum(r.count("○") for r in check_rows) == 0, check_rows
    # 번호는 완료 순서대로 1,2,3 — 각 라인이 올바른 툴과 짝지어졌는지
    assert "[1]" in check_rows[0] and "grep" in check_rows[0], check_rows
    assert "[2]" in check_rows[1] and "read_file" in check_rows[1], check_rows
    assert "[3]" in check_rows[2] and "read_symbol" in check_rows[2], check_rows
    # 전부 끝났으면 in-flight 비어 있고 live 라인 내려감
    assert printer._inflight == {}
    assert printer._live_drawn is False


def test_subsecond_tool_flashes_pending_then_commits_check():
    # 1초 미만 순차 툴: ○가 동기로 잠깐 떴다가 ✓로 확정돼야 한다(#2 회귀).
    committed, raw, printer = _drive([_run("a", "read_file"), _done("a", "read_file")])
    assert "○" in raw  # pending 라인이 즉시 그려졌다
    check_rows = [r for r in committed if r.strip()]
    assert len(check_rows) == 1, check_rows
    assert "✓" in check_rows[0] and "○" not in check_rows[0], check_rows
    assert "[1]" in check_rows[0] and "read_file" in check_rows[0]
    assert printer._inflight == {}


def test_missing_call_id_concurrent_does_not_drop_completion():
    # provider가 tool-call id를 안 주는 동시 실행: 두 완료 모두 ✓로 찍히고
    # in-flight 가 끝까지 빈다(완료 유실/유령 live 라인 방지, #1 회귀).
    committed, _raw, printer = _drive([
        _run(None, "read_file"),
        _run(None, "grep"),
        _done(None, "read_file"),
        _done(None, "grep"),
    ])
    check_rows = [r for r in committed if r.strip()]
    assert len(check_rows) == 2, check_rows
    assert sum(r.count("✓") for r in check_rows) == 2, check_rows
    assert sum(r.count("○") for r in check_rows) == 0, check_rows
    assert "[1]" in check_rows[0] and "[2]" in check_rows[1], check_rows
    assert printer._inflight == {}


def test_error_completion_renders_cross_and_drains_inflight():
    committed, _raw, printer = _drive([
        _run("a", "read_file"),
        _run("b", "grep"),
        _err("a", "read_file"),
        _done("b", "grep"),
    ])
    check_rows = [r for r in committed if r.strip()]
    # ✗ 라인 + 에러 상세 + ✓ 라인이 섞여 있으므로 glyph 단위로 검증
    assert sum(r.count("✗") for r in check_rows) == 1, check_rows
    assert sum(r.count("✓") for r in check_rows) == 1, check_rows
    assert sum(r.count("○") for r in check_rows) == 0, check_rows
    assert printer._inflight == {}
    assert printer._live_drawn is False
