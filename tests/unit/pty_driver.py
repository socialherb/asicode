"""In-process pty harness for interactive-terminal coverage tests (layer 1).

repl_impl's terminal-bound functions (``_run_esc_watcher``,
``_cli_checkpoint_cb``, ``_collect_input``) need a real TTY: termios
fcntls, ``select`` on stdin, and prompt_toolkit's Vt100 input/output. A
POSIX pty pair provides a real terminal device inside the pytest process:
``sys.stdin`` (and optionally ``sys.stdout``) are redirected to the pty
slave, and a background thread drains the master side — a minimal
terminal-emulator surrogate (it also answers prompt_toolkit's cursor
position requests so the renderer doesn't wait out its ~1s CPR timeout).

Why in-process (no fork): the three target functions run in the pytest
process with monkeypatched module state (LLM fakes, ``time``/``select``
fakes, module globals). Fork isolation is only needed for
``_run_repl_impl`` / ``run_subagent_worker`` (signal handlers require a
dedicated main thread).

Environment facts this harness accounts for:

* pytest's capture machinery re-wraps ``sys.stdout``/``sys.stderr`` after
  fixture setup, undoing a fixture-time replacement of ``sys.stdout``
  (``sys.stdin`` is untouched and sticks) — call :meth:`PtySession.activate`
  from the test body.
* Python 3.12+ ``threading.Thread`` does not inherit the caller's
  contextvars by default (``sys.flags.thread_inherit_context``), so worker
  threads see prompt_toolkit's shared default ``AppSession`` whose
  ``input``/``output`` are cached on first access — reset those caches per
  test (see ``tests.unit.test_repl_pty_red_green._reset_app_session``).
* prompt_toolkit 3.0.52: ``create_input()`` always returns ``Vt100Input``
  on POSIX (reads via ``os.read(fileno)``); ``create_output()`` returns
  ``Vt100_Output`` when ``sys.stdout.isatty()`` with an ioctl(TIOCGWINSZ)
  based ``get_size`` — setting the pty winsize is sufficient. Non-main
  threads get SIGINT/WINCH handling disabled automatically.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import select
import struct
import subprocess
import sys
import termios
import threading
import time


def _disable_cr_translation(slave_fd: int) -> None:
    """Clear the CR/NL input translations (ICRNL/INLCR/IGNCR) on *slave_fd*.

    Applied to :class:`SpawnPtySession` ONLY (see below for the PtySession
    asymmetry). The default pty slave termios is canonical with ICRNL on.
    prompt_toolkit clears these flags only WHILE a prompt is reading
    (vt100.py raw mode) and restores the saved flags the moment the prompt
    returns — so between prompts the tty is back to canonical+ICRNL. Any
    ``\r`` the harness writes into that window is translated to ``\n`` BY
    THE KERNEL at enqueue time, before the app ever reads it, and
    prompt_toolkit maps ``\x0a`` to ControlJ — this REPL's "insert newline"
    binding — not to Enter. A burst ``text + b"\r"`` then leaves the buffer
    ``text\n`` and the prompt never returns (the full-suite hang with the
    child parked in ``_collect_input``; RED 3/3 burst vs fixed 10/10).

    With the flags cleared, a CR enqueued during the canonical window is
    held in the input queue untranslated — canonical mode is not a line
    delimiter for a bare CR — and drains as ``\r`` through the app's
    non-canonical read, dispatching the Enter submit binding. Real
    terminals rely on exactly this: interactive line editors keep ICRNL
    off so the Enter key stays distinguishable from a literal newline.

    The harness CR/LF contract that follows — "\r" reaches prompt_toolkit
    as Enter; "\n" is the canonical line terminator": the child's own
    canonical-mode reads (``input()`` in the auth-retry prompt,
    ``sys.stdin.readline()`` in ``_cli_checkpoint_cb``) must be fed
    ``text + b"\n"`` by tests (see ``_send_canonical`` in
    test_repl_stage3_red_green.py), because a bare CR no longer completes
    a canonical line. Real user terminals keep ICRNL on in those windows,
    so pressing Enter still works there in production.

    NOT applied to :class:`PtySession` (in-process): its tests exercise
    product code that legitimately reads stdin canonically (the checkpoint
    callback's ``readline()``), and mirroring the real-terminal ICRNL-on
    behavior there preserves that contract — the CR-submit race this
    seals only exists across a real prompt_toolkit read lifecycle, i.e.
    in spawned children.
    """
    with contextlib.suppress(Exception):
        attrs = termios.tcgetattr(slave_fd)
        attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)


def _pump_master_once(
    master_fd: int,
    buf: bytearray,
    lock: threading.Lock,
    *,
    on_data=None,
    select_timeout: float = 0.0,
) -> bool:
    """One select+read pass over *master_fd*.

    Appends any bytes read to *buf* under *lock* and returns True; returns
    False when nothing was readable within *select_timeout* (or the fd
    errored/EOFed). Safe to run while the background drain thread is also
    reading — each byte is consumed exactly once by whichever reader gets it.
    """
    try:
        r, _, _ = select.select([master_fd], [], [], select_timeout)
    except OSError:
        return False
    if not r:
        return False
    try:
        data = os.read(master_fd, 65536)
    except BlockingIOError:
        # select said readable but a concurrent reader (the drain thread)
        # consumed the bytes between our select() and read() — nothing left
        # for us; NOT an error. Requires a non-blocking master fd, else this
        # read would block the caller until new output arrives.
        return False
    except OSError:
        return False
    if not data:
        return False
    with lock:
        buf.extend(data)
    if on_data is not None:
        on_data(data)
    return True
def _pump_master_until_quiet(
    master_fd: int,
    buf: bytearray,
    lock: threading.Lock,
    *,
    timeout: float,
    quiet: float,
    on_data=None,
) -> None:
    """Read *master_fd* until it stays silent for ``quiet`` seconds.

    Bounded by ``timeout``; every chunk is appended to *buf* under *lock*
    (safe to run while the background drain thread is also reading — each
    byte is consumed exactly once by whichever reader gets it).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pump_master_once(master_fd, buf, lock, on_data=on_data,
                                 select_timeout=quiet):
            break  # quiet window / fd error / EOF: snapshot is settled


class PtySession:
    """Redirect ``sys.stdin`` (and optionally ``sys.stdout``) onto a pty slave.

    Redirections are monkeypatch-based, so pytest restores the real streams
    at test teardown. The slave's winsize is preset so prompt_toolkit's
    ioctl-based ``get_size`` reports sane dimensions.
    """

    def __init__(self, monkeypatch, rows: int = 24, cols: int = 80,
                 redirect_stdout: bool = True) -> None:
        self._monkeypatch = monkeypatch
        self._redirect_stdout = redirect_stdout
        self._master, self._slave = os.openpty()
        # Two concurrent readers (background drain thread + wait_for/dump
        # pumps) share this fd: select-then-read is only race-free for a
        # SINGLE reader, so a blocking fd lets one reader hang inside
        # os.read when the other steals the bytes — non-blocking +
        # BlockingIOError handling in every reader instead.
        os.set_blocking(self._master, False)
        self.master_fd = self._master
        self.slave_fd = self._slave
        fcntl.ioctl(
            self._slave, termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr(
            sys, "stdin", os.fdopen(os.dup(self._slave), "r", buffering=1))
        if redirect_stdout:
            monkeypatch.setattr(
                sys, "stdout", os.fdopen(os.dup(self._slave), "w", buffering=1))
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._drain = threading.Thread(
            target=self._drain_loop, name="pty-drain", daemon=True)
        self._drain.start()

    def activate(self) -> None:
        """Re-apply the stream redirections (see module docstring)."""
        monkeypatch = self._monkeypatch
        monkeypatch.setattr(
            sys, "stdin", os.fdopen(os.dup(self._slave), "r", buffering=1))
        if self._redirect_stdout:
            monkeypatch.setattr(
                sys, "stdout", os.fdopen(os.dup(self._slave), "w", buffering=1))

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._master], [], [], 0.1)
            except OSError:
                break
            if not r:
                continue
            try:
                data = os.read(self._master, 65536)
            except BlockingIOError:
                continue  # the pump thread took the bytes; nothing for us
            except OSError:
                break
            if not data:
                break
            with self._lock:
                self._buf.extend(data)
            if b"\x1b[6n" in data:
                self._answer_cpr()

    def _answer_cpr(self) -> None:
        """Reply to prompt_toolkit's cursor-position request (CSI 6n)."""
        with contextlib.suppress(OSError):
            os.write(self._master, b"\x1b[1;1R")

    def send(self, data: bytes) -> None:
        """Write bytes to the pty master — these appear on the app's stdin."""
        view = memoryview(data)
        while view:
            try:
                n = os.write(self._master, view)
            except BlockingIOError:
                time.sleep(0.005)
                continue
            view = view[n:]

    def wait_for(self, needle: bytes, timeout: float = 8.0) -> bytes:
        """Block until ``needle`` appears in drained output; return the buffer.

        Pumps the master fd directly on every poll instead of only watching
        ``_buf``: under load (macOS 8-worker xdist full suite) the background
        drain thread can be starved for the entire timeout while the app has
        long since flushed the bytes — poll-only waiting is what made the
        v0.2.26 ``test_session_d_model_api_key_undo_baseline`` time out 3/3
        with the child alive and re-rendering. Pumping also answers CPR
        inline, so prompt_toolkit never sits out its ~1s cursor timeout
        waiting for a starved drain thread (same cure as :meth:`dump`,
        proven since the v0.2.24 truncation flake).
        """
        deadline = time.monotonic() + timeout
        last = b""
        while time.monotonic() < deadline:
            _pump_master_once(
                self._master, self._buf, self._lock,
                on_data=self._maybe_answer_cpr,
            )
            with self._lock:
                last = bytes(self._buf)
            if needle in last:
                return last
            time.sleep(0.02)
        raise AssertionError(
            f"timed out after {timeout}s waiting for {needle!r} in pty output; "
            f"last {len(last)} bytes: {last[-400:]!r}")

    def dump(self, timeout: float = 5.0, quiet: float = 0.05) -> bytes:
        """All output drained so far (not consumed).

        Pumps the master fd until it stays silent for ``quiet`` seconds
        (bounded by ``timeout``) before snapshotting: the background drain
        thread only reads *eventually*, and on a loaded 2-core CI runner it
        can lag behind the app's final flushed writes — a bare buffer
        snapshot then misses bytes that had already left the process by the
        time the test's join() returned (the v0.2.24
        test_truncation_over_50000_chars flake: the value assertion passed,
        the just-printed truncation notice did not).
        """
        _pump_master_until_quiet(
            self._master, self._buf, self._lock,
            timeout=timeout, quiet=quiet,
            on_data=self._maybe_answer_cpr,
        )
        with self._lock:
            return bytes(self._buf)

    def _maybe_answer_cpr(self, data: bytes) -> None:
        if b"\x1b[6n" in data:
            self._answer_cpr()

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        self._stop.set()
        self._drain.join(timeout=1.0)
        with contextlib.suppress(OSError):
            for fd in (self._master, self._slave):
                os.close(fd)


class SpawnPtySession:
    """Spawn a subprocess with stdin/stdout/stderr on a pty slave and drive
    it from the master side (fork-free child isolation).

    Needed for functions that MUST run in a dedicated main thread with a real
    terminal: ``repl_impl._run_repl_impl`` (prompt_toolkit full-screen prompt,
    ``signal.signal`` handlers, terminal-config TTY-keyed paths). The exec
    boundary means parent-side monkeypatches do NOT cross — the child driver
    re-applies its own fakes (see ``repl_stage2_child.py``).

    The child is placed in its own session (``start_new_session=True``) so the
    pty is its controlling terminal; the parent holds the master and drains it
    with the same CPR-answering loop as :class:`PtySession`.
    """

    def __init__(self, argv, *, cwd=None, env=None, rows: int = 24,
                 cols: int = 80, timeout: float = 45.0) -> None:
        self.timeout = timeout
        self._master, slave = os.openpty()
        # See PtySession.__init__: shared by two concurrent readers — the
        # fd must be non-blocking or a select/read race hangs a reader.
        os.set_blocking(self._master, False)
        with contextlib.suppress(OSError):
            fcntl.ioctl(
                slave, termios.TIOCSWINSZ,
                struct.pack("HHHH", rows, cols, 0, 0),
            )
        # Must precede Popen: a CR the harness writes before the child's
        # first prompt (banner -> prompt gap) would otherwise be ICRNL-
        # translated by the kernel (see _disable_cr_translation).
        _disable_cr_translation(slave)
        child_env = dict(os.environ)
        child_env.update({"TERM": "xterm", "NO_COLOR": "1"})
        if os.environ.get("ASICODE_PTY_DIAG"):
            child_env["PYTHONFAULTHANDLER"] = "1"  # TEMP DIAG: child stack on SIGABRT
        if env:
            child_env.update(env)
        # `coverage run -m pytest` sets COVERAGE_PROCESS_START, which makes
        # coverage's .pth hook auto-start instrumentation in EVERY spawned
        # python process. Our child drivers also start their own Coverage
        # instance (repl_stage2_child.py), so the double instrumentation
        # races on the same data file — observed as non-deterministic
        # "database disk image is malformed" on some suffixed files, which
        # then fail to combine. The child must own coverage exclusively;
        # strip the auto-start vars AFTER merging caller env too.
        child_env.pop("COVERAGE_PROCESS_START", None)
        child_env.pop("COVERAGE_PROCESS_CONFIG", None)
        self._proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave,
            cwd=cwd, env=child_env, close_fds=True, start_new_session=True,
        )
        os.close(slave)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._drain = threading.Thread(
            target=self._drain_loop, name="pty-spawn-drain", daemon=True)
        self._drain.start()

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._master], [], [], 0.1)
            except OSError:
                break
            if not r:
                continue
            try:
                data = os.read(self._master, 65536)
            except BlockingIOError:
                continue  # the pump thread took the bytes; nothing for us
            except OSError:
                break
            if not data:
                break
            with self._lock:
                self._buf.extend(data)
            if b"\x1b[6n" in data:
                with contextlib.suppress(OSError):
                    os.write(self._master, b"\x1b[1;1R")

    def send(self, data: bytes) -> None:
        """Write bytes to the pty master — these appear on the child's stdin."""
        view = memoryview(data)
        while view:
            try:
                n = os.write(self._master, view)
            except BlockingIOError:
                time.sleep(0.005)
                continue
            view = view[n:]

    def wait_for(self, needle: bytes, timeout: float | None = None) -> bytes:
        """Block until ``needle`` appears in drained output; return the buffer.

        Pumps the master fd directly on every poll — same rationale as
        :meth:`PtySession.wait_for`: the background drain thread is exactly
        what load starves (v0.2.26 ``test_session_d_model_api_key_undo_baseline``
        3/3 timeout under the 8-worker xdist full suite, child alive and
        re-rendering), and CPR requests need a reading loop to answer inline.
        """
        deadline = time.monotonic() + (timeout or self.timeout)
        last = b""
        while time.monotonic() < deadline:
            _pump_master_once(
                self._master, self._buf, self._lock,
                on_data=self._maybe_answer_cpr,
            )
            with self._lock:
                last = bytes(self._buf)
            if needle in last:
                return last
            time.sleep(0.02)
        raise AssertionError(
            f"timed out after {timeout or self.timeout}s waiting for {needle!r} "
            f"in pty output; last {len(last)} bytes: {last[-4000:]!r}")

    def _maybe_answer_cpr(self, data: bytes) -> None:
        if b"\x1b[6n" in data:
            with contextlib.suppress(OSError):
                os.write(self._master, b"\x1b[1;1R")

    def dump(self, timeout: float = 5.0, quiet: float = 0.05) -> bytes:
        """All output drained so far (not consumed) — pumps until quiet.

        Same rationale as :meth:`PtySession.dump`: the child's final writes
        can sit unread in the kernel master queue while the drain thread is
        starved; the caller (which usually just waited for the child) must
        not observe a snapshot that predates them.
        """
        _pump_master_until_quiet(
            self._master, self._buf, self._lock,
            timeout=timeout, quiet=quiet, on_data=self._maybe_answer_cpr,
        )
        with self._lock:
            return bytes(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self._proc.wait(timeout=timeout or self.timeout)

    def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            self._proc.terminate()

    def close(self) -> None:
        self._stop.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5)
        self._drain.join(timeout=1.0)
        with contextlib.suppress(OSError):
            os.close(self._master)
