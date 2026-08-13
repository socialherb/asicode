"""
Agent Test Runner

Runs tests in repo_root via LanguageProvider commands (pytest, jest, go test, ...).
Captures stdout/stderr (raw). Optionally streams output lines via callback.

Designed for asicode "Agent Mode" loops.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from external_llm.common.bounded_capture import _BoundedCapture

logger = logging.getLogger(__name__)

# Pytest output prefix constants
_FAILED_PREFIX = 'FAILED '

# Reaping a SIGKILLed group is not part of the test budget, so it gets its own
# small grace rather than the deadline's remainder.
_KILL_REAP_GRACE = 5
# The drain threads exit on EOF, which the group kill guarantees. Bounded anyway:
# a join that cannot be waited out must not become the new way to hang.
_DRAIN_JOIN_GRACE = 2


def _kill_process_group(proc, pgid: int) -> None:
    """SIGKILL *proc*'s whole process group, falling back to the process alone.

    ``start_new_session=True`` put the child in its own group, so the group is
    exactly the test command and everything it spawned — nothing of ours is in
    it. Grandchildren are the point: they inherit the stdout/stderr pipes, and a
    survivor keeps the write end open, which is what defeated the timeout.

    ``pgid`` must be resolved IMMEDIATELY after the Popen, while the child is
    certainly alive. The timeout/cancel path may run AFTER the direct child
    exited and was reaped (a pytest wrapper that spawned xdist workers and
    returned), and a re-resolved ``getpgid`` would then raise
    ProcessLookupError — silently falling back to ``proc.kill()`` on the dead
    leader and orphaning the grandchildren, the leak this function exists to
    prevent. The GROUP survives its leader while any member is alive, so the
    stored pgid stays valid.

    Falls back to ``proc.kill()`` when the group cannot be resolved (already
    reaped, or a platform without process groups) so the direct child still dies.

    Refuses to signal our OWN group, which is the whole safety margin here. The
    group kill is correct only because the Popen above passes
    ``start_new_session=True``; drop that one keyword and ``getpgid(child)``
    returns the agent's own group, so this SIGKILLs the agent — CLI, REPL and
    all. Found the hard way: mutating the Popen to verify the regression test
    killed the shell running it, three times, with no output. A guard that costs
    one ``getpgrp()`` is cheaper than a landmine under a future edit.
    """
    try:
        if pgid == os.getpgrp():
            # Not an assert: in a wedged agent the useful outcome is a dead
            # child, not a traceback from the teardown path.
            logger.error(
                "test_runner: child %s shares our process group (%s) — "
                "start_new_session was not applied; killing the child only",
                proc.pid, pgid,
            )
        else:
            os.killpg(pgid, signal.SIGKILL)
            return
    except (ProcessLookupError, PermissionError, AttributeError, OSError) as exc:
        logger.debug("test_runner: process group kill failed (%s) — killing child", exc)
    try:
        proc.kill()
    except (ProcessLookupError, OSError) as exc:
        logger.debug("test_runner: child already gone: %s", exc)



@dataclass
class TestRunResult:
    ok: bool
    exit_code: int
    duration_ms: int
    stdout: str
    stderr: str
    combined: str
    summary_line: Optional[str] = None
    failing_tests: Optional[list[str]] = None
    first_traceback: Optional[str] = None
    # Structured test results
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    xpassed_count: int = 0
    xfailed_count: int = 0
    failed_test_details: list[dict[str, Any]] = None
    error_test_details: list[dict[str, Any]] = None
    # True when the run was killed at *timeout_sec* rather than finishing.
    # Without this the caller cannot tell "the suite failed" from "we never let
    # the suite finish" — both arrive as ok=False with a nonzero exit code, and
    # the TDD cycle reported the second as the first (partial output, no marker).
    timed_out: bool = False
    # True when the run was killed on a cancel request (cancel_check returned
    # True mid-run) rather than finishing or timing out. Distinct from
    # timed_out so the caller can tell "the user asked us to stop" from "the
    # budget ran out" — the first is NOT a test failure.
    cancelled: bool = False


class TestRunner:
    """
    Minimal pytest runner with streaming.

    stream_callback signature:
        stream_callback(line: str, stream: str, meta: dict)

    - line: text line WITHOUT trailing newline
    - stream: "stdout" | "stderr"
    - meta: dict (attempt, max_attempts, phase, etc.)
    """

    def __init__(
        self,
        repo_root: str,
        *,
        python_executable: Optional[str] = None,
        env_overrides: Optional[dict[str, str]] = None,
        test_command: Optional[list[str]] = None,
    ):
        self.repo_root = str(Path(repo_root).resolve())
        self.python_executable = python_executable or os.environ.get("PYTHON", "python3")
        self.env_overrides = dict(env_overrides or {})
        self.test_command = test_command

    @classmethod
    def from_provider(cls, repo_root, provider, test_args=None):
        """Create a TestRunner from a LanguageProvider's get_test_command()."""
        cmd = provider.get_test_command(repo_root, test_args)
        return cls(repo_root, test_command=cmd)

    def run(
        self,
        *,
        args: Optional[list[str]] = None,
        timeout_sec: int = 120,
        cancel_check: Optional[Callable[[], bool]] = None,
        stream_callback: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> TestRunResult:
        """
        Generic run — uses self.test_command if set, otherwise falls back to pytest.
        Returns raw combined output; structured pytest parsing is only done
        when the command looks like pytest.

        cancel_check: polled during the run; when it returns True the process
        group is killed and the result comes back with cancelled=True.
        """
        cmd = self._build_cmd(args=args)
        return self._run_cmd(cmd, timeout_sec=timeout_sec, cancel_check=cancel_check, stream_callback=stream_callback, meta=meta)

    def run_pytest(
        self,
        *,
        args: Optional[list[str]] = None,
        timeout_sec: int = 300,
        cancel_check: Optional[Callable[[], bool]] = None,
        stream_callback: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> TestRunResult:
        """Run pytest. Delegates to run() with a pytest-friendly default timeout."""
        return self.run(
            args=args,
            timeout_sec=timeout_sec,
            cancel_check=cancel_check,
            stream_callback=stream_callback,
            meta=meta,
        )

    # ── Internal ────────────────────────────────────────────────────────────

    def _run_cmd(
        self,
        cmd: list[str],
        *,
        timeout_sec: int = 120,
        cancel_check: Optional[Callable[[], bool]] = None,
        stream_callback: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> TestRunResult:
        """Core subprocess runner — shared by run() and run_pytest()."""
        meta = dict(meta or {})

        env = os.environ.copy()
        env.update(self.env_overrides)
        # Prevent pytest from wrapping long test names mid-word.
        # Without this, pytest reads the terminal width (often 80-120 cols)
        # and breaks node IDs like "...::TestFoo\n  ::test_bar", which makes
        # the structured output hard to parse and the LLM result confusing.
        env.setdefault("COLUMNS", "200")

        from .config.thresholds import config as _thresholds
        _CAP = _thresholds.tokens.BASH_OUTPUT_MAX_CHARS

        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            # Own session, so the timeout path can signal the whole GROUP. A test
            # command routinely outlives its direct child — pytest-xdist workers,
            # a server/daemon fixture, docker — and those grandchildren inherit
            # these very pipes. Killing only the direct child then left them
            # holding the write end, so the drain threads never saw EOF and the
            # `finally` below blocked forever on a pipe close: measured
            # timeout_sec=1 returning at 15.08 s, i.e. no timeout at all. Same
            # discipline as `bash` (_tool_shell_exec), `grep`
            # (_run_search_bounded) and run_bounded_subprocess, whose docstring
            # names "pytest-spawned server fixtures" as the case it exists for.
            start_new_session=True,
        )
        # Resolve the group NOW, while the child is certainly alive: the
        # timeout/cancel teardown may run after the leader exited and was
        # reaped, and a re-resolved getpgid() would then fail (see
        # _kill_process_group).
        _pgid = os.getpgid(proc.pid)

        def _emit(line: str, stream: str) -> None:
            if stream_callback:
                try:
                    stream_callback(line, stream, meta)
                except Exception as _exc:
                    logger.debug("test stream callback failed: %s", _exc)

        timed_out = False
        stdout_text = ""
        stderr_text = ""
        # Bound before the try: the finally consults them, and a failure between
        # here and the thread starts would otherwise raise NameError over the
        # real exception.
        _t_out = None
        _t_err = None

        try:
            stdout = proc.stdout
            stderr = proc.stderr
            assert stdout is not None and stderr is not None

            import threading as _th

            # Bounded, not a list of every line: a verbose or failing suite
            # prints tens of MB and the previous unbounded append held all of it
            # (measured: 40 MB of output -> +229 MB peak RSS, and a 41.5 MB
            # string handed to the pytest parsers). Head AND tail because both
            # ends carry signal — collection errors at the start, the summary
            # line and FAILED list at the end, which is what every extractor
            # below reads.
            _out_cap = _BoundedCapture(_CAP)
            _err_cap = _BoundedCapture(_CAP)
            _out_lock = _th.Lock()
            _err_lock = _th.Lock()

            def _read_stream(stream, capture, lock, stream_name):
                try:
                    for _raw in iter(stream.readline, ''):
                        _line = _raw.rstrip('\n')
                        with lock:
                            capture.feed(_line + "\n")
                        _emit(_line, stream_name)
                except (ValueError, OSError) as exc:
                    # Pipe closed under us (kill/cancel path): the partial
                    # output already fed to the capture is still the answer.
                    logger.debug(
                        "test_runner: %s drain ended early: %s", stream_name, exc,
                    )

            _t_out = _th.Thread(
                target=_read_stream,
                args=(stdout, _out_cap, _out_lock, "stdout"),
                daemon=True,
            )
            _t_err = _th.Thread(
                target=_read_stream,
                args=(stderr, _err_cap, _err_lock, "stderr"),
                daemon=True,
            )
            _t_out.start()
            _t_err.start()

            # Poll loop instead of one blocking wait: the same loop doubles as
            # the cancel check, so a cancel_event set MID-RUN kills the tests
            # instead of leaving them unowned until the budget expires (same
            # discipline as bash's _capture_bounded poll). Deadline-based, so
            # the timeout contract is unchanged.
            _CANCEL_POLL_S = 0.5
            _deadline = time.monotonic() + timeout_sec
            cancelled = False
            while True:
                try:
                    proc.wait(timeout=min(_CANCEL_POLL_S, max(0.0, _deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= _deadline:
                        timed_out = True
                        break
                    if cancel_check is not None and cancel_check():
                        cancelled = True
                        break

            if timed_out or cancelled:
                _kill_process_group(proc, _pgid)
                if timed_out:
                    _marker = f"[test_runner] TIMEOUT — killed after {timeout_sec}s"
                else:
                    _marker = "[test_runner] CANCELLED — killed on cancel request"
                # Into the CAPTURE, not just the callback. _emit only reaches
                # stream_callback, and the tool handler passes none — so the
                # marker used to vanish and a timed-out run reached the model as
                # an ordinary failure with truncated output ("Tests failed", no
                # mention of a timeout), which it then debugged as a real one.
                with _err_lock:
                    _err_cap.feed(_marker + "\n")
                _emit(_marker, "stderr")
                try:
                    proc.wait(timeout=_KILL_REAP_GRACE)
                except subprocess.TimeoutExpired:
                    # A SIGKILLed process that has not been reaped in 5 s is
                    # stuck in uninterruptible sleep (a hung mount, a wedged
                    # driver). Nothing here can free it, and raising would turn
                    # "the tests timed out" into an exception out of the tool —
                    # the partial output plus timed_out=True is the better
                    # answer. The zombie is the OS's problem.
                    logger.warning(
                        "test_runner: child %s not reaped %ss after SIGKILL",
                        proc.pid, _KILL_REAP_GRACE,
                    )

            _t_out.join(timeout=_DRAIN_JOIN_GRACE)
            _t_err.join(timeout=_DRAIN_JOIN_GRACE)

            with _out_lock:
                stdout_text = _out_cap.text().rstrip("\n")
            with _err_lock:
                stderr_text = _err_cap.text().rstrip("\n")

        finally:
            # Only close what nothing is reading. TextIOWrapper.close() takes the
            # buffer lock, so closing a pipe whose drain thread is still parked in
            # readline() blocks on that thread — the exact deadlock above. The
            # group kill makes a live thread here nearly unreachable (EOF arrives
            # with the group), but "nearly" is not a bound: leaking two fds in
            # that pathological case is strictly better than wedging the agent.
            for _thread, _stream in ((_t_out, proc.stdout), (_t_err, proc.stderr)):
                if _stream is None or (_thread is not None and _thread.is_alive()):
                    continue
                try:
                    _stream.close()
                except (OSError, ValueError, AttributeError) as exc:
                    logger.debug("test_runner: pipe close failed: %s", exc)

        end = time.monotonic()
        exit_code = proc.returncode if proc.returncode is not None else -1

        combined = (stdout_text + "\n" + stderr_text).strip("\n") if (stdout_text or stderr_text) else ""

        # Structured parse only for pytest-compatible output
        is_pytest = any(x in cmd[0] for x in ("pytest", "python")) if cmd else False

        if is_pytest:
            summary_line = self._extract_summary_line(combined)
            failing_tests = self._extract_failing_tests(combined)
            first_traceback = self._extract_first_traceback(combined)
            structured_results = self._parse_pytest_output(combined)
        else:
            summary_line = self._extract_summary_line(combined)
            failing_tests = None
            first_traceback = None
            structured_results = {
                "passed": 0, "failed": 0, "errors": 0,
                "skipped": 0, "xpassed": 0, "xfailed": 0,
                "failed_tests": [], "error_tests": [],
            }

        return TestRunResult(
            ok=(exit_code == 0),
            exit_code=exit_code,
            duration_ms=int((end - start) * 1000),
            stdout=stdout_text,
            stderr=stderr_text,
            combined=combined,
            summary_line=summary_line,
            failing_tests=failing_tests,
            first_traceback=first_traceback,
            passed_count=structured_results.get("passed", 0),
            failed_count=structured_results.get("failed", 0),
            error_count=structured_results.get("errors", 0),
            skipped_count=structured_results.get("skipped", 0),
            xpassed_count=structured_results.get("xpassed", 0),
            xfailed_count=structured_results.get("xfailed", 0),
            failed_test_details=structured_results.get("failed_tests", []),
            error_test_details=structured_results.get("error_tests", []),
            timed_out=timed_out,
            cancelled=cancelled,
        )

    def _build_cmd(self, *, args: Optional[list[str]]) -> list[str]:
        if args is not None:
            return list(args)
        if self.test_command:
            return list(self.test_command)
        # prefer python -m pytest for venv consistency
        return [self.python_executable, "-m", "pytest", "-q"]

    def _extract_summary_line(self, combined: str) -> Optional[str]:
        """Return summary line from combined output (pytest, jest, go test, etc.)."""
        if not combined:
            return None
        lines = combined.splitlines()
        # Scan from bottom for known summary patterns
        for line in reversed(lines):
            s = line.strip()
            if not s:
                continue
            # pytest: "1 passed, 1 failed in 0.15s"
            if any(keyword in s for keyword in _SUMMARY_KEYWORDS):
                return s
            # jest: "Tests: 1 failed, 3 passed, 4 total"
            if s.startswith("Tests:"):
                return s
            # go test: "ok  \tpackage/path\t0.123s"
            if s.startswith(("ok ", "FAIL")):
                return " ".join(s.split())
        return lines[-1] if lines else None

    def _extract_failing_tests(self, combined: str) -> list[str]:
        """
        Best-effort parse failing test nodeids from pytest output.

        This is intentionally conservative. We'll improve later via FailureContext.
        """
        fails: list[str] = []
        if not combined:
            return fails

        for line in combined.splitlines():
            s = line.strip()
            # Pytest -q often shows: "FAILED tests/test_x.py::test_name - ..."
            if s.startswith(_FAILED_PREFIX):
                # "FAILED " + nodeid + (" - ...")
                nodeid = s.removeprefix(_FAILED_PREFIX).split(" - ", 1)[0].strip()
                if nodeid and nodeid not in fails:
                    fails.append(nodeid)

        return fails

    def _parse_pytest_output(self, combined: str) -> dict[str, Any]:
        """Parse pytest output into structured counts and test details."""
        import re

        result: dict[str, Any] = {
            "passed": 0, "failed": 0, "errors": 0,
            "skipped": 0, "xpassed": 0, "xfailed": 0,
            "failed_tests": [], "error_tests": [],
        }
        if not combined:
            return result

        # Parse count from summary line: "N keyword" patterns
        summary = self._extract_summary_line(combined)
        if summary:
            _count_keys = [
                ("passed",  "passed"),
                ("failed",  "failed"),
                ("errors",  "errors"),
                ("error",   "errors"),   # pytest sometimes uses singular
                ("skipped", "skipped"),
                ("xpassed", "xpassed"),
                ("xfailed", "xfailed"),
            ]
            for keyword, key in _count_keys:
                m = re.search(r"(\d+)\s+" + keyword, summary)
                if m:
                    result[key] = max(result[key], int(m.group(1)))

        # Collect traceback blocks per test (between "____ name ____" markers)
        # so we can attach assertion messages to individual failing tests.
        _tb_by_test: dict[str, str] = {}
        _current_tb_test: Optional[str] = None
        _current_tb_lines: list[str] = []

        def _tb_head_tail(lines: list[str], max_lines: int = 50) -> str:
            """Return full traceback — head+tail truncation removed per commit 320365fa."""
            return "\n".join(lines)

        _tb_section = False
        for line in combined.splitlines():
            # "____ test_name ____" marks the start of a new failure section
            m = re.match(r"_{4,}\s+(.+?)\s+_{4,}", line)
            if m:
                if _current_tb_test and _current_tb_lines:
                    _tb_by_test[_current_tb_test] = _tb_head_tail(_current_tb_lines, 50)
                _current_tb_test = m.group(1).strip()
                _current_tb_lines = []
                _tb_section = True
                continue
            if _tb_section and _current_tb_test is not None:
                # Stop at next "=====" separator or "----- " short separator
                if line.startswith(("=" * 5, "-" * 5)):
                    if _current_tb_test and _current_tb_lines:
                        _tb_by_test[_current_tb_test] = _tb_head_tail(_current_tb_lines, 50)
                    _current_tb_test = None
                    _current_tb_lines = []
                    _tb_section = False
                else:
                    _current_tb_lines.append(line)
        if _current_tb_test and _current_tb_lines:
            _tb_by_test[_current_tb_test] = _tb_head_tail(_current_tb_lines, 50)

        # Failed test details from "FAILED ... - ..." lines
        _lines_combined = combined.splitlines()
        for line in _lines_combined:
            s = line.strip()
            if s.startswith(_FAILED_PREFIX):
                rest = s.removeprefix(_FAILED_PREFIX)
                _parts = rest.split(" - ", 1)
                test_id = _parts[0].strip()
                if not test_id or test_id.startswith("::"):
                    continue
                # Extract error type and message from " - " suffix
                error_type = ""
                message = ""
                if len(_parts) > 1:
                    err_msg = _parts[1].strip()
                    # Format: "ErrorType: message" or just "message"
                    _colon = err_msg.find(": ")
                    if _colon > 0:
                        error_type = err_msg[:_colon].strip()
                        message = err_msg[_colon + 2:].strip()
                    else:
                        error_type = ""
                        message = err_msg
                traceback = _tb_by_test.get(
                    test_id.split("::")[-1] if "::" in test_id else test_id,
                    "",
                )
                result["failed_tests"].append({
                    "test_id": test_id,
                    "name": test_id.split("::")[-1] if "::" in test_id else test_id,
                    "error_type": error_type,
                    "message": message,
                    "traceback": traceback,
                })

        # Error test details from "ERROR ..." lines
        seen_errors: set = set()
        for line in _lines_combined:
            s = line.strip()
            if s.startswith("ERROR "):
                rest = s[len("ERROR "):].strip()
                test_id = rest.split(" - ", 1)[0].strip()
                if test_id and test_id not in seen_errors:
                    seen_errors.add(test_id)
                    # Extract error message for ERROR lines too
                    err_msg = ""
                    if " - " in rest:
                        err_msg = rest.split(" - ", 1)[1].strip()
                    result["error_tests"].append({
                        "test_id": test_id,
                        "name": test_id.split("::")[-1] if "::" in test_id else test_id,
                        "error_type": "ERROR",
                        "message": err_msg,
                    })

        return result

    def _extract_first_traceback(self, combined: str) -> Optional[str]:
        if not combined:
            return None

        lines = combined.splitlines()
        # pytest traceback often includes "E   ..." lines or "Traceback (most recent call last):"
        start_idx = None
        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                start_idx = i
                break
        if start_idx is None:
            # fallback: first "E   " block
            for i, line in enumerate(lines):
                if line.lstrip().startswith("E   "):
                    start_idx = max(0, i - 10)
                    break

        if start_idx is None:
            return None

        excerpt = lines[start_idx : start_idx + 120]
        return "\n".join(excerpt).rstrip("\n")

_SUMMARY_KEYWORDS = (
    'PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS',  # pytest
    'PASS', 'FAIL',  # jest/go test
)

