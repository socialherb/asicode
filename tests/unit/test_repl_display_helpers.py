"""repl_impl display-helper contracts (deep-audit findings).

- ``_stderr_spinner``: the shared stderr spinner extracted from the two
  verbatim closures (``/clear`` compaction, insights compact). Pins that a
  pre-set stop event exits immediately with no output, and that a running
  spinner ticks frames until stopped.
- ``_format_result``: now returns ONLY the dim token line (the former
  ``(main_text, token_line)`` tuple computed a status summary + textwrap of
  up to 8000 chars that its only caller discarded).
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from external_llm.repl import repl_impl

# ── _stderr_spinner ────────────────────────────────────────────────────────


def test_stderr_spinner_exits_immediately_when_stop_preset(capsys):
    """stop already set → the wait(0.1) loop never runs; no output, fast return."""
    stop = threading.Event()
    stop.set()
    t0 = time.monotonic()
    repl_impl._stderr_spinner(stop, t0, "Compacting…")
    assert capsys.readouterr().err == ""


def test_stderr_spinner_ticks_until_stopped(capsys):
    """While running, the helper writes spinner frames with the message."""
    stop = threading.Event()
    t0 = time.monotonic()
    t = threading.Thread(
        target=repl_impl._stderr_spinner, args=(stop, t0, "Compacting…"), daemon=True
    )
    t.start()
    # Wait for at least one tick (0.1s cadence) instead of a fixed sleep — a
    # fixed sleep flakes when the spinner thread is starved under load.
    deadline = time.monotonic() + 2.0
    err = ""
    while time.monotonic() < deadline and "Compacting…" not in err:
        time.sleep(0.02)
        err = capsys.readouterr().err
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert "Compacting…" in err
    assert "\r\033[K" in err  # in-place line redraw


# ── _format_result ─────────────────────────────────────────────────────────


def test_format_result_returns_token_line_only():
    """New contract: a single str (the dim token line), not a tuple."""
    result = SimpleNamespace(
        status="success",
        applied_patches=[{"file": "a.py"}],
        turns=[1],
        final_message="done",
        error=None,
        metadata={
            "tokens": {
                "prompt": 1000, "completion": 200, "total": 1200,
                "last_call_prompt": 500, "last_call_completion": 50,
            }
        },
    )
    line = repl_impl._format_result(result)
    assert isinstance(line, str)
    assert line.startswith("  tok ↑")
    assert "last ↑" in line


def test_format_result_total_branch_without_last_call():
    result = SimpleNamespace(
        status="success", applied_patches=[], turns=[], final_message="",
        error=None,
        metadata={"tokens": {"prompt": 1000, "completion": 200}},
    )
    line = repl_impl._format_result(result)
    assert "total" in line
    assert "last" not in line


def test_format_result_without_metadata_returns_empty():
    result = SimpleNamespace(
        status="success", applied_patches=[], turns=[], final_message="",
        error=None, metadata=None,
    )
    assert repl_impl._format_result(result) == ""
