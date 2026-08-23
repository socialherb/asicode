"""
Tests for StreamingDisplay.
"""

from __future__ import annotations

import io
import sys

from external_llm.repl.collaborate.claude_session import SessionEvent
from external_llm.repl.collaborate.streaming_display import (
    StreamingDisplay,
    _display_name,
    _format_tool_input,
    _markdown_lines,
)


class TestStreamingDisplay:
    """Verify streaming display event handling."""

    def test_handle_text_event(self):
        display = StreamingDisplay()
        event = SessionEvent(type="text", content="Hello world")
        # Should not raise
        display.handle_event(event)

    def test_handle_tool_call_event(self):
        display = StreamingDisplay()
        event = SessionEvent(
            type="tool_call",
            content="read_file",
            metadata={"tool_name": "read_file", "input": {"path": "test.py"}},
        )
        display.handle_event(event)

    def test_handle_verdict_event(self):
        display = StreamingDisplay()
        event = SessionEvent(
            type="verdict",
            content="Status: success",
            metadata={
                "verdict": {
                    "status": "success",
                    "summary": "All good",
                    "details": "Everything passed.",
                    "confidence": 0.95,
                    "suggestions": ["Deploy"],
                }
            },
        )
        display.handle_event(event)
        assert display._verdict_summary is not None

    def test_handle_error_event(self):
        display = StreamingDisplay()
        event = SessionEvent(type="error", content="Something broke")
        display.handle_event(event)

    def test_handle_status_event(self):
        display = StreamingDisplay()
        event = SessionEvent(type="status", content="INTERRUPTED")
        display.handle_event(event)

    def test_header_and_summary(self):
        display = StreamingDisplay()
        display.print_header("Test task")
        display.print_summary()

    def test_flush_log_no_file(self):
        display = StreamingDisplay()
        display.flush_log()  # Should not raise

    def test_flush_log_with_file(self, tmp_path):
        log_file = tmp_path / "session.log"
        display = StreamingDisplay(output_file=str(log_file))
        display.handle_event(SessionEvent(type="text", content="Test log entry"))
        display.flush_log()
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test log entry" in content


class TestToolInputFormatting:
    """Verify read_file line-range notation + MCP tool name cleanup."""

    def test_display_name_strips_mcp_prefix(self):
        assert _display_name("mcp__asr__read_file") == "read_file"
        assert _display_name("mcp__asr__find_relevant_files") == "find_relevant_files"
        # Names without prefix, single underscore preserved
        assert _display_name("read_file") == "read_file"
        assert _display_name("StructuredOutput") == "StructuredOutput"

    def test_read_file_shows_line_range(self):
        # Even with mcp__ prefix, recognized as read_file with line range notation
        out = _format_tool_input(
            "mcp__asr__read_file",
            {"path": "asi.py", "start_line": 120, "end_line": 340},
        )
        assert out == "asi.py:120-340"

    def test_read_file_partial_range(self):
        assert _format_tool_input("mcp__asr__read_file", {"path": "a.py", "start_line": 120}) == "a.py:120-end"
        assert _format_tool_input("mcp__asr__read_file", {"path": "a.py", "end_line": 340}) == "a.py:1-340"

    def test_read_file_whole_file_no_range(self):
        # No range args → path only (full read)
        assert _format_tool_input("mcp__asr__read_file", {"path": "a.py"}) == "a.py"

    def test_other_tools_use_key_arg_under_mcp_prefix(self):
        assert _format_tool_input("mcp__asr__grep", {"pattern": "def foo"}) == "def foo"
        assert _format_tool_input("mcp__asr__find_relevant_files", {"query": "streaming"}) == "streaming"

    def test_long_command_not_pre_truncated(self):
        # Width adaptation is the renderer's job (_truncate(hint, hint_avail)
        # in _render_live / _handle_tool_result). _format_tool_input must return
        # the FULL value so wide terminals can show it; the old fixed 72/80
        # caps cut commands short with '…' even when room remained.
        long_cmd = "cd /repo && echo 'section' && cat external_llm/repl/collaborate/streaming_display.py | head"
        assert _format_tool_input("bash", {"command": long_cmd}) == long_cmd
        long_path = "a" * 200 + ".py"
        assert _format_tool_input("read_file", {"path": long_path}) == long_path
        assert (
            _format_tool_input("read_file", {"path": long_path, "start_line": 1, "end_line": 5}) == f"{long_path}:1-5"
        )


class TestTruncateSingleRow:
    """Regression guard: _truncate output must NEVER contain a newline.

    The in-place live line (_render_live) is written with ``\\r\\x1b[2K``,
    which erases only the *last* physical row. A multi-line tool arg that
    survives _truncate renders across several rows, so every 0.25s ticker
    re-render leaves an orphan line behind — ~360 for a 90s command. See the
    _truncate docstring and ``_render_live`` MUST-fit-one-row contract.
    """

    def test_fold_happens_even_when_under_budget(self):
        # The pre-fix bug: _disp_width(s) <= max_cells returned s UNCHANGED,
        # so a short 2-line string kept its newline. The cell budget must NOT
        # gate the fold — a 3-cell "a\nb" passes a 60-cell check yet wraps.
        from external_llm.repl.collaborate.streaming_display import _truncate

        out = _truncate("import sys\nprint('hi')", max_cells=60)
        assert "\n" not in out
        assert out == "import sys print('hi')"

    def test_heredoc_hint_has_no_newline_wide_terminal(self):
        # Exact reproduction: a bash heredoc command spanning 3 logical lines.
        # On a wide terminal the folded width fits, so pre-fix the whole thing
        # (with \n) was returned — wrapping under \r\x1b[2K.
        from external_llm.repl.collaborate.streaming_display import _truncate

        heredoc = "cd /repo && python3 -u - <<'PYEOF'\nimport sys\nprint('hi')"
        out = _truncate(heredoc, max_cells=120)
        assert "\n" not in out
        assert "PYEOF" in out and "import sys" in out

    def test_live_line_render_has_no_newline(self):
        # End-to-end through _render_live: a bash tool_call with a heredoc
        # command must render as a single physical row.
        import os as _os

        from external_llm.repl.collaborate import streaming_display as mod

        heredoc = "cd /repo && timeout 90 python3 -u - <<'PYEOF'\nimport sys\nprint('x')"
        old_cols = mod.shutil.get_terminal_size
        old_out = sys.stdout
        disp = None
        try:
            # Force a wide terminal so the un-folded hint WOULD have wrapped.
            mod.shutil.get_terminal_size = lambda *a, **kw: _os.terminal_size((120, 24))
            fake = io.StringIO()
            fake.isatty = lambda: True
            sys.stdout = fake
            disp = StreamingDisplay()
            disp.handle_event(
                SessionEvent(
                    type="tool_call",
                    content="bash",
                    metadata={"tool_name": "bash", "input": {"command": heredoc}},
                )
            )
            rendered = fake.getvalue()
            # The live line written via \r\x1b[2K must contain no bare newline.
            assert "\n" not in rendered, rendered
        finally:
            if disp is not None:
                disp.stop()
            mod.shutil.get_terminal_size = old_cols
            sys.stdout = old_out


class TestMarkdownRendering:
    """Verify final-answer markdown rendering."""

    def test_markdown_strips_syntax_markers(self):
        md = "### Heading\n\n**bold** and `code`\n\n- item one\n\n---"
        lines = _markdown_lines(md, width=80)
        assert lines is not None
        import re

        plain = "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in lines)
        # Literal markdown symbols are rendered away
        assert "###" not in plain
        assert "**bold**" not in plain
        assert "`code`" not in plain
        assert "Heading" in plain and "bold" in plain and "code" in plain
        assert "•" in plain  # bullet rendered

    def test_markdown_lines_fit_width(self):
        # Every physical line must fit within `width` cells or the gutter breaks.
        # Uses Korean text on purpose: east_asian_width() must classify these
        # characters as double-width ("W"/"F") for the wrap-width check below
        # to be meaningful.
        import re
        import unicodedata

        md = "긴 한국어 텍스트가 " * 20
        lines = _markdown_lines(md, width=40)
        assert lines is not None
        for ln in lines:
            plain = re.sub(r"\x1b\[[0-9;]*m", "", ln)
            w = sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in plain)
            assert w <= 40

    def test_markdown_empty_returns_none(self):
        assert _markdown_lines("", width=80) is None
        assert _markdown_lines("   \n  ", width=80) is None

    def test_verdict_details_render_no_literal_fences(self):
        # E2E: even if verdict details contain a code block, the ```python
        # fence itself must not show up on screen.
        display = StreamingDisplay()
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            display.handle_event(
                SessionEvent(
                    type="verdict",
                    content="",
                    metadata={
                        "verdict": {
                            "status": "needs_review",
                            "summary": "review",
                            "confidence": 0.9,
                            "details": "## Title\n\n```python\nx = 1\n```\n",
                        }
                    },
                )
            )
            display.stop()
        finally:
            sys.stdout = old
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        assert "```python" not in plain
        assert "## Title" not in plain  # header symbol gets rendered away
        assert "x = 1" in plain  # code body is preserved


class TestTickerConcurrency:
    """P1 regression guard — check-and-start serialization of _start_ticker."""

    def test_concurrent_start_ticker_creates_single_thread(self):
        """Exactly one ticker is created even when N threads call
        _start_ticker concurrently.

        handle_event (asyncio callback thread) and print_header (orchestrator
        thread) call _start_ticker from different threads. Without
        _ticker_launch_lock, both threads could observe "no ticker yet",
        each spawn its own thread, and the second one would overwrite
        _ticker_thread, orphaning the first ticker.
        """
        import threading

        from external_llm.repl.collaborate import streaming_display as mod

        RealThread = threading.Thread
        created: list = []

        class RecordingThread(RealThread):  # subclasses the real thread before patching
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                created.append(self)

        mod.threading.Thread = RecordingThread
        fake = io.StringIO()
        fake.isatty = lambda: True
        old = sys.stdout
        sys.stdout = fake
        disp = None
        try:
            disp = StreamingDisplay()
            n = 8
            barrier = threading.Barrier(n)

            def go():
                barrier.wait()
                disp._start_ticker()

            # Callers use real threads (captured) — not recorded by RecordingThread
            callers = [RealThread(target=go) for _ in range(n)]
            for t in callers:
                t.start()
            for t in callers:
                t.join()
            # Exactly one ticker is created even with 8 concurrent calls
            assert len(created) == 1
            assert disp._ticker_thread is created[0]
        finally:
            if disp is not None:
                disp.stop()
            mod.threading.Thread = RealThread
            sys.stdout = old
        for t in created:
            t.join(timeout=1.0)

    def test_start_ticker_idempotent_sequential(self):
        """Consecutive calls from the same thread don't create a second thread."""
        import threading

        from external_llm.repl.collaborate import streaming_display as mod

        RealThread = threading.Thread
        created: list = []

        class RecordingThread(RealThread):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                created.append(self)

        mod.threading.Thread = RecordingThread
        fake = io.StringIO()
        fake.isatty = lambda: True
        old = sys.stdout
        sys.stdout = fake
        disp = None
        try:
            disp = StreamingDisplay()
            disp._start_ticker()
            first = disp._ticker_thread
            disp._start_ticker()
            disp._start_ticker()
            assert disp._ticker_thread is first  # not replaced
            assert len(created) == 1
        finally:
            if disp is not None:
                disp.stop()
            mod.threading.Thread = RealThread
            sys.stdout = old
        for t in created:
            t.join(timeout=1.0)


class TestVerdictConfidenceGuard:
    """P2 regression guard — confidence type guard in _print_verdict."""

    def test_print_verdict_string_confidence_does_not_crash(self):
        # Even a raw dict that skipped from_result_message (a future emitter
        # path) must render without a ':.0%' ValueError.
        display = StreamingDisplay()
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            display._print_verdict({"status": "success", "summary": "ok", "confidence": "0.9"})
            display._print_verdict({"status": "success", "summary": "ok", "confidence": "bad"})
        finally:
            sys.stdout = old
            display.stop()
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        assert "90%" in plain  # "0.9" → 90%

    def test_print_verdict_out_of_range_clamped(self):
        display = StreamingDisplay()
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            display._print_verdict({"status": "success", "summary": "ok", "confidence": 1.5})
        finally:
            sys.stdout = old
            display.stop()
        import re

        plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        assert "100%" in plain
        assert "150%" not in plain


class TestMarkdownTableFoldPatch:
    """Verify the ``rich.markdown.TableElement`` fold patch is applied."""

    def test_table_cell_no_ellipsis(self):
        """Table cells with long content must NOT contain ``"…"`` (fold vs ellipsis)."""
        md = (
            "| Command | Description |\n"
            "|---|---|\n"
            "| `verylongcommandwithoutspaces` | This is a very long description without any spaces that would normally be truncated |\n"
            "| `short` | Short desc |\n"
        )
        lines = _markdown_lines(md, width=40)
        assert lines is not None
        for line in lines:
            assert "…" not in line, f"fold patch failed: ellipsis found in {line!r}"

    def test_markdown_imports_use_shared_ssot_module(self):
        """All ``from rich.markdown import`` sites must go through
        ``external_llm.common.rich_markdown.markdown_cls()``, NOT import
        ``Markdown`` directly — otherwise the fold patch is silently bypassed.
        """
        import subprocess

        proc = subprocess.run(
            ["grep", "-rn", r"from rich\.markdown import", "external_llm", "asi.py", "webapp"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # Only legitimate site: the shared module itself (which uses it inside markdown_cls)
        lines = [line for line in proc.stdout.splitlines() if line.strip() and "rich_markdown.py" not in line]
        assert not lines, (
            "Direct `from rich.markdown import` bypasses the shared module — "
            "use `markdown_cls()` from `external_llm.common.rich_markdown` instead. Found:\n" + "\n".join(lines)
        )
