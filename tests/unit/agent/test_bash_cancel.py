"""Cancellation must reach a `bash` command that is already running.

`cancel_event` was checked once, at dispatch entry. After that the tool sat in a
single blocking wait on the whole budget, so ESC pressed one
second into a command was not observed until the command finished or its budget
expired — 120 s by default, 300 s at the ceiling. Measured before the fix:
`cancel_event` set at t=1.0 s, `bash sleep 12` returned at t=12.0 s.

The CLI hid this rather than showing it. `_run_with_cancel` runs the agent loop
in a daemon thread and returns the prompt ~1 s after ESC, so the user gets their
prompt back promptly — while the subprocess keeps running, keeps its side
effects, and (being in its own session) outlives the agent process entirely.

Three properties are pinned here: the wait observes the event, the process tree
is actually torn down, and the ordinary paths — success, output produced across
several poll slices, timeout-to-background — are unchanged.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

import pytest


def _dispatch(reg, command, timeout=30):
    return reg.dispatch("bash", {"command": command, "timeout": timeout})


@pytest.fixture
def cancel_reg(tool_registry):
    """Registry with a cancel_event the test can set from another thread."""
    tool_registry.config.cancel_event = threading.Event()
    return tool_registry


def _cancel_after(ev: threading.Event, delay: float) -> None:
    threading.Thread(target=lambda: (time.sleep(delay), ev.set()), daemon=True).start()


def test_cancel_interrupts_a_sleeping_command(cancel_reg):
    _cancel_after(cancel_reg.config.cancel_event, 0.6)
    t0 = time.monotonic()
    result = _dispatch(cancel_reg, "sleep 30")
    elapsed = time.monotonic() - t0

    assert elapsed < 5, f"cancel was not observed until {elapsed:.1f}s"
    assert not result.ok
    assert result.error == "Operation cancelled"
    assert (result.metadata or {}).get("cancelled") is True


def test_cancel_kills_the_process_tree(cancel_reg, tmp_path):
    """The command must not survive the cancel and complete its work.

    `start_new_session=True` means nothing on this side reaps it, so without an
    explicit group kill the side effect still lands — the cancel would be
    cosmetic. Asserted through a real filesystem effect rather than a pid check,
    because that is what the user actually cares about.
    """
    marker = tmp_path / "should-not-exist"
    _cancel_after(cancel_reg.config.cancel_event, 0.6)
    result = _dispatch(cancel_reg, f"sleep 3 && touch {marker}")

    assert not result.ok
    time.sleep(4)  # past when the command would have finished
    assert not marker.exists(), "cancelled command ran to completion anyway"


def test_cancel_returns_what_the_command_had_already_printed(cancel_reg):
    """Stopping a command is not a reason to hide its output so far."""
    _cancel_after(cancel_reg.config.cancel_event, 0.8)
    result = _dispatch(cancel_reg, "echo BEFORE-CANCEL; sleep 30")

    assert not result.ok
    assert "BEFORE-CANCEL" in (result.content or ""), (
        f"partial output dropped on cancel: {result.content!r}"
    )
    assert (result.metadata or {}).get("partial_output") is True


def test_a_cleared_event_does_not_cancel(cancel_reg):
    """The poll must read the event, not merely notice that one exists."""
    result = _dispatch(cancel_reg, "echo fine", timeout=10)
    assert result.ok
    assert "fine" in result.content


def test_output_spanning_many_poll_slices_is_complete(cancel_reg):
    """The wait is sliced; the output must not be.

    The pump reads whatever is ready on each poll and accumulates it, so six
    lines emitted over ~2.4 s — roughly ten poll slices — must arrive whole.
    """
    result = _dispatch(
        cancel_reg,
        "for i in 1 2 3 4 5 6; do echo line-$i; sleep 0.4; done",
        timeout=30,
    )
    assert result.ok
    missing = [f"line-{i}" for i in range(1, 7) if f"line-{i}" not in result.content]
    assert not missing, f"sliced wait lost output: {missing} — {result.content!r}"


@pytest.mark.parametrize(
    "raw",
    [b"a\rb\r\nc\n", "\uac00\ub098\ub2e4\r\n".encode(), b"plain\n"],
    ids=["cr-and-crlf", "multibyte-crlf", "plain"],
)
def test_the_pump_decodes_exactly_as_text_mode_did(raw, tmp_path):
    """Reading the pipes ourselves must not change what a command's output IS.

    `communicate(text=True)` performs universal-newline translation: both `\r`
    and `\r\n` become `\n`. Progress bars (pip, docker, curl) are made of `\r`,
    so a decoder that skipped the translation would reshape the output of a
    large fraction of real commands. The pump uses IncrementalNewlineDecoder for
    exactly this, and statefully — asserted at EVERY split point, because the
    interesting case is a `\r\n` or a multibyte sequence straddling two reads.

    Separate from the tool so a future interpreter change fails HERE, naming the
    cause, instead of surfacing as mysteriously reshaped bash output.
    """
    import codecs
    import io

    # The bytes come from a FILE rather than through a text-mode stdin pipe.
    # Writing them to `proc.stdin.buffer` and closing it leaves communicate() to
    # flush an already-closed TextIOWrapper, which raises ValueError — it catches
    # only BrokenPipeError. CPython 3.14 tolerates that; 3.12 does not, and
    # requires-python is >=3.10, so the pipe made this test pass on exactly one
    # interpreter (it was green locally and red on CI's 3.12 — measured both).
    # A file also keeps the write side out of it entirely: feeding a str through
    # the wrapper would newline-translate on the way IN, destroying the very
    # `\r` under test. Only the READ side is what this test is about.
    src = tmp_path / "raw.bin"
    src.write_bytes(raw)
    proc = subprocess.Popen(
        ["cat", str(src)], stdout=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    expected, _ = proc.communicate()

    for cut in range(len(raw) + 1):
        dec = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")("replace"), True,
        )
        got = dec.decode(raw[:cut]) + dec.decode(raw[cut:], True)
        assert got == expected, (
            f"split at byte {cut} decoded to {got!r}, but text=True gives "
            f"{expected!r} — the pump would reshape this command's output"
        )


def test_timeout_still_transitions_to_background(cancel_reg):
    """The budget path is unchanged: exceeding it hands off, never cancels."""
    result = _dispatch(cancel_reg, "echo EARLY; sleep 20", timeout=2)
    assert result.ok, result.error
    job_id = (result.metadata or {}).get("background_job_id")
    assert job_id, f"no background handoff: {result.content!r}"
    assert (result.metadata or {}).get("cancelled") is None
    try:
        time.sleep(0.5)
        out = cancel_reg.dispatch("job", {"action": "output", "job_id": job_id})
        assert "EARLY" in (out.content or ""), (
            "pre-timeout output lost on handoff"
        )
    finally:
        cancel_reg.dispatch("job", {"action": "kill", "job_id": job_id})


def test_cancel_survives_an_already_dead_process(cancel_reg):
    """Racing the process's own exit must not raise out of the tool."""
    ev = cancel_reg.config.cancel_event
    # Set the event first, so the very first poll after a near-instant command
    # can find both a set event and a finished process.
    ev.set()
    result = _dispatch(cancel_reg, "true", timeout=10)
    # Either outcome is legitimate (it may simply have completed first); what
    # must not happen is an exception escaping as a generic tool failure.
    assert result.ok or result.error == "Operation cancelled", result.error
    assert "Command execution failed" not in (result.error or "")


def test_cancel_event_is_read_fresh_each_poll(cancel_reg):
    """A per-turn swap of config.cancel_event must be honoured mid-command.

    The design-chat REPL replaces the event object between turns, so a value
    captured before the wait would leave ESC inert — the same trap the RAG
    indexers document.
    """
    replacement = threading.Event()
    cancel_reg.config.cancel_event = threading.Event()

    def _swap_and_set():
        time.sleep(0.6)
        cancel_reg.config.cancel_event = replacement
        replacement.set()

    threading.Thread(target=_swap_and_set, daemon=True).start()
    t0 = time.monotonic()
    result = _dispatch(cancel_reg, "sleep 30")
    assert time.monotonic() - t0 < 5
    assert result.error == "Operation cancelled"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_no_orphans_remain_after_cancel(cancel_reg):
    """A shell's grandchildren go with it — the kill targets the group."""
    marker = "asicode-cancel-orphan-probe"
    _cancel_after(cancel_reg.config.cancel_event, 0.8)
    t0 = time.monotonic()
    _dispatch(cancel_reg, f"bash -c 'sleep 30 # {marker}' & sleep 30 # {marker}")
    # Without the mid-wait cancel this returns only once the sleeps end, and
    # "no survivors" would then be true for the wrong reason.
    assert time.monotonic() - t0 < 5, "the cancel never interrupted the wait"
    time.sleep(1)
    survivors = subprocess.run(
        ["pgrep", "-f", marker], capture_output=True, text=True,
        check=False,
    ).stdout.split()
    # pgrep matches its own invocation shell in some environments; allow none.
    assert not survivors, f"orphaned processes after cancel: {survivors}"
