"""RED->GREEN regression for the pty harness itself (tests/unit/pty_driver.py).

The v0.2.26 release aborted ``--verify=full`` three times on
``test_repl_stage3_red_green.py::test_session_d_model_api_key_undo_baseline``:
30s ``wait_for`` timeouts with the child alive and re-rendering, only under
the macOS 8-worker xdist full suite (standalone: 7s pass, CI Linux: green).
Root cause: ``wait_for`` only polled the shared ``_buf``, so output delivery
depended entirely on the background drain thread — exactly what load
starves; prompt_toolkit CPR requests also went unanswered without a reading
loop. ``dump()`` has pumped the master fd directly since the v0.2.24
truncation flake (commit 0f64d74b). These tests pin ``wait_for`` to the same
contract: it must observe bytes even when the drain thread is dead.
"""

from __future__ import annotations

import sys
import termios

from tests.unit.pty_driver import PtySession, SpawnPtySession

# Child protocol: "ready" -> wait for the tester to kill the drain thread
# -> (after a grace pause, so the marker lands strictly after drain death)
# print the marker, then block on stdin until close() terminates us.
_CHILD_SRC = (
    "import sys, time\n"
    "print('ready', flush=True)\n"
    "sys.stdin.readline()\n"
    "time.sleep(0.3)\n"
    "print('marker-after-drain-death', flush=True)\n"
    "sys.stdin.readline()\n"
)


def test_spawn_wait_for_reads_master_with_dead_drain(tmp_path):
    """SpawnPtySession.wait_for must not depend on the drain thread.

    With a dead drain and poll-only wait_for the marker never reaches
    ``_buf`` and the 5s wait times out (the v0.2.26 failure mode); with a
    self-pumping wait_for the master fd is read directly and the marker is
    observed within one poll interval.
    """
    sess = SpawnPtySession([sys.executable, "-u", "-c", _CHILD_SRC], cwd=str(tmp_path), timeout=15)
    try:
        sess.wait_for(b"ready", timeout=15)
        # Kill the drain thread: deterministic stand-in for the xdist load
        # that starves it for the whole timeout.
        sess._stop.set()
        sess._drain.join(timeout=2)
        assert not sess._drain.is_alive(), "drain thread must be dead here"
        sess.send(b"go\n")
        buf = sess.wait_for(b"marker-after-drain-death", timeout=5)
        assert b"marker-after-drain-death" in buf
    finally:
        sess.close()


def test_session_wait_for_reads_master_with_dead_drain(monkeypatch):
    """PtySession.wait_for: same contract, in-process slave writes."""
    tty = PtySession(monkeypatch)
    try:
        tty.activate()
        print("ready", flush=True)
        tty.wait_for(b"ready", timeout=5)
        tty._stop.set()
        tty._drain.join(timeout=2)
        assert not tty._drain.is_alive(), "drain thread must be dead here"
        print("marker-after-drain-death", flush=True)
        buf = tty.wait_for(b"marker-after-drain-death", timeout=5)
        assert b"marker-after-drain-death" in buf
    finally:
        tty.close()


# Child protocol: report stdin's termios input flags, then block on stdin.
_TERMIOS_CHILD = (
    "import sys, termios\n"
    "f = termios.tcgetattr(0)\n"
    "print('FLAGS icrnl=%d inlcr=%d igncr=%d' % ("
    "bool(f[0] & termios.ICRNL), bool(f[0] & termios.INLCR),"
    "bool(f[0] & termios.IGNCR)), flush=True)\n"
    "sys.stdin.readline()\n"
)


def test_spawn_slave_cr_translation_cleared(tmp_path):
    """SpawnPtySession must hand the child a slave with ICRNL/INLCR/IGNCR off.

    The kernel applies ICRNL at ENQUEUE time: a ``\r`` written while the
    child sits between prompts (canonical termios restored by
    prompt_toolkit) becomes ``\n`` before the app reads it, and ptk maps
    ``\x0a`` to ControlJ ("insert newline"), not Enter — the burst-submit
    hang. The child reports its own stdin flags, so this pins the device
    state the harness actually delivers (RED on HEAD: icrnl=1 — the pty
    default).
    """
    sess = SpawnPtySession([sys.executable, "-u", "-c", _TERMIOS_CHILD], cwd=str(tmp_path), timeout=15)
    try:
        buf = sess.wait_for(b"FLAGS ", timeout=10)
        assert b"icrnl=0" in buf, f"ICRNL must be cleared: {buf[-120:]!r}"
        assert b"inlcr=0" in buf and b"igncr=0" in buf
    finally:
        sess.close()


def test_session_slave_keeps_cr_translation(monkeypatch):
    """PtySession (in-process) must KEEP ICRNL — the real-terminal contract.

    In-process tests exercise product code that reads stdin canonically
    (``_cli_checkpoint_cb``'s ``sys.stdin.readline()``): with ICRNL on, a
    typed ``\r`` completes the line exactly as it does on a user's
    terminal. The CR-submit race this file's SpawnPtySession contract seals
    only exists across a real prompt_toolkit read lifecycle, so the
    asymmetry is deliberate (see pty_driver._disable_cr_translation).
    """
    tty = PtySession(monkeypatch)
    try:
        attrs = termios.tcgetattr(tty.slave_fd)
        assert attrs[0] & termios.ICRNL, "PtySession must keep ICRNL (canonical reads)"
    finally:
        tty.close()
