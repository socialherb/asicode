"""Regression tests for _ProgressPrinter's concurrent (in-flight) tool line rendering.

Background: tools run in parallel via ThreadPoolExecutor, so design_tool_call
running/complete events arrive interleaved. In the past, in-flight state was
tracked with a single scalar slot, so a late-arriving complete event would be
stamped with the number of the "most recently started tool", leaving earlier
○ lines never updated to ✓ and left stuck on screen (duplicate/scrambled
numbering). This test pins down that call_id-based matching + completion-order
numbering + single live-line model prevents that class of bug.
"""
import io
import sys

import asi


def _drive(events):
    """Feed events through _ProgressPrinter.__call__, emulating a one-line
    terminal, and return (committed_rows, raw_stream, printer).

    \\r\\x1b[2K clears the current row, and \\n commits the current row.
    Display-only ANSI codes (color/dim etc.) are stripped for assertion
    convenience."""
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
            if seq.endswith("K"):  # \x1b[2K → clear current row
                row = []
            # anything else (color/dim) is a display attribute — irrelevant to assertions, discard
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
    # 3 tools start concurrently → completions arrive in a different order than starts (b, a, c).
    committed, _raw, printer = _drive([
        _run("a", "read_file"),
        _run("b", "grep"),
        _run("c", "read_symbol"),
        _done("b", "grep"),
        _done("a", "read_file"),
        _done("c", "read_symbol"),
    ])
    # exactly 3 committed ✓ lines, 0 orphaned ○
    check_rows = [r for r in committed if r.strip()]
    assert len(check_rows) == 3, check_rows
    assert all("✓" in r for r in check_rows), check_rows
    assert sum(r.count("○") for r in check_rows) == 0, check_rows
    # numbers should follow completion order 1,2,3 — verify each line is paired with the correct tool
    assert "[1]" in check_rows[0] and "grep" in check_rows[0], check_rows
    assert "[2]" in check_rows[1] and "read_file" in check_rows[1], check_rows
    assert "[3]" in check_rows[2] and "read_symbol" in check_rows[2], check_rows
    # once everything is done, in-flight should be empty and the live line taken down
    assert printer._inflight == {}
    assert printer._live_drawn is False


def test_subsecond_tool_flashes_pending_then_commits_check():
    # sub-second sequential tool: ○ should flash briefly then commit as ✓ (#2 regression).
    committed, raw, printer = _drive([_run("a", "read_file"), _done("a", "read_file")])
    assert "○" in raw  # the pending line was drawn immediately
    check_rows = [r for r in committed if r.strip()]
    assert len(check_rows) == 1, check_rows
    assert "✓" in check_rows[0] and "○" not in check_rows[0], check_rows
    assert "[1]" in check_rows[0] and "read_file" in check_rows[0]
    assert printer._inflight == {}


def test_missing_call_id_concurrent_does_not_drop_completion():
    # concurrent execution where the provider doesn't supply a tool-call id: both completions
    # should render as ✓ and in-flight should end up empty (guards against dropped completions /
    # ghost live lines, #1 regression).
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
    # ✗ line + error detail + ✓ line are interleaved, so verify at the glyph level
    assert sum(r.count("✗") for r in check_rows) == 1, check_rows
    assert sum(r.count("✓") for r in check_rows) == 1, check_rows
    assert sum(r.count("○") for r in check_rows) == 0, check_rows
    assert printer._inflight == {}
    assert printer._live_drawn is False
