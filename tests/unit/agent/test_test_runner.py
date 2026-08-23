"""Unit tests for the Agent TestRunner (external_llm/agent/test_runner.py).

Covers the pure output-parsing helpers exhaustively (summary/failing/traceback
extraction, structured pytest parse), the command builder, the provider factory,
and the core subprocess runner via a real trivial subprocess + a timeout path.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time

from external_llm.agent import test_runner as _tr
from external_llm.agent.config.thresholds import config as _thresholds

# Each bulk line is a 7-digit counter, a space, _BULK_PAD 'x' and a newline.
_BULK_PAD = 90
_BULK_LINE_LEN = 7 + 1 + _BULK_PAD + 1


def _bulk_line_count(cap: int) -> tuple[int, int]:
    """Lines needed to overrun *cap*, and the char count they produce.

    Retention is head+tail, i.e. up to ``2 * cap``. Sizing production at merely
    "past the cap" makes the bound assertion pass vacuously — the first draft
    emitted 235 KB against a 120 KB retention and proved nothing. 10x leaves the
    two an order of magnitude apart.
    """
    n_lines = (cap * 10) // _BULK_LINE_LEN
    return n_lines, n_lines * _BULK_LINE_LEN


def _pid_alive(pid: int) -> bool:
    """True while *pid* exists.

    ``os.kill(pid, 0)`` rather than a ``ps`` subprocess: the caller polls this in
    a loop, and shelling out per poll is both slower and one more thing that can
    fail for a reason unrelated to what is being asserted.
    """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # alive, just not ours to signal
    return True


# ── TestRunResult defaults ──────────────────────────────────────────────────


class TestRunResultDefaults:
    def test_failed_test_details_default_is_none(self):
        # dataclass field without factory → shared None sentinel (documented quirk)
        r = _tr.TestRunResult(ok=True, exit_code=0, duration_ms=0, stdout="", stderr="", combined="")
        assert r.failed_test_details is None
        assert r.error_test_details is None
        assert r.summary_line is None and r.failing_tests is None

    def test_count_defaults_zero(self):
        r = _tr.TestRunResult(ok=True, exit_code=0, duration_ms=0, stdout="", stderr="", combined="")
        assert (r.passed_count, r.failed_count, r.error_count) == (0, 0, 0)
        assert (r.skipped_count, r.xpassed_count, r.xfailed_count) == (0, 0, 0)


# ── __init__ / from_provider ────────────────────────────────────────────────


class TestInitAndFactory:
    def test_init_defaults(self):
        tr = _tr.TestRunner("/tmp")
        assert str(tr.repo_root).endswith("tmp")
        assert tr.test_command is None
        assert tr.python_executable  # non-empty default

    def test_init_with_overrides(self):
        tr = _tr.TestRunner("/tmp", python_executable="/x/python", env_overrides={"FOO": "1"}, test_command=["echo"])
        assert tr.python_executable == "/x/python"
        assert tr.env_overrides == {"FOO": "1"}
        assert tr.test_command == ["echo"]

    def test_from_provider(self):
        class _Prov:
            def get_test_command(self, root, args):
                return ["pytest", "-x", *(args or [])]

        tr = _tr.TestRunner.from_provider("/tmp", _Prov(), test_args=["-q"])
        assert tr.test_command == ["pytest", "-x", "-q"]


# ── _build_cmd ──────────────────────────────────────────────────────────────


class TestBuildCmd:
    def test_args_override_everything(self):
        tr = _tr.TestRunner("/tmp", test_command=["pytest"])
        assert tr._build_cmd(args=["custom", "cmd"]) == ["custom", "cmd"]

    def test_test_command_used_when_no_args(self):
        tr = _tr.TestRunner("/tmp", test_command=["jest"])
        assert tr._build_cmd(args=None) == ["jest"]

    def test_default_pytest_command(self):
        tr = _tr.TestRunner("/tmp")
        cmd = tr._build_cmd(args=None)
        assert cmd == [tr.python_executable, "-m", "pytest", "-q"]


# ── _extract_summary_line ───────────────────────────────────────────────────


class TestExtractSummaryLine:
    def setup_method(self):
        self.tr = _tr.TestRunner("/tmp")

    def test_empty_returns_none(self):
        assert self.tr._extract_summary_line("") is None

    def test_pytest_summary(self):
        out = "collected 3 items\n...\n2 passed, 1 failed in 0.15s\n"
        assert self.tr._extract_summary_line(out) == "2 passed, 1 failed in 0.15s"

    def test_jest_tests_line(self):
        out = "...\nTests: 1 failed, 3 passed, 4 total\n"
        assert self.tr._extract_summary_line(out) == "Tests: 1 failed, 3 passed, 4 total"

    def test_go_test_ok(self):
        out = "...\nok  \tpkg/path\t0.123s\n"
        # go test: leading "ok " → returns whitespace-collapsed line
        assert self.tr._extract_summary_line(out) == "ok pkg/path 0.123s"

    def test_go_test_fail(self):
        out = "...\nFAIL\tpkg/path\t0.5s [build failed]\n"
        assert self.tr._extract_summary_line(out).startswith("FAIL")

    def test_fallback_last_line(self):
        out = "no recognized keyword anywhere\njust some line\n"
        assert self.tr._extract_summary_line(out) == "just some line"


# ── _extract_failing_tests ──────────────────────────────────────────────────


class TestExtractFailingTests:
    def setup_method(self):
        self.tr = _tr.TestRunner("/tmp")

    def test_empty_returns_empty(self):
        assert self.tr._extract_failing_tests("") == []

    def test_parses_failed_lines(self):
        out = (
            "FAILED tests/test_a.py::test_one - AssertionError: x\n"
            "FAILED tests/test_b.py::TestC::test_two - ValueError: y\n"
        )
        fails = self.tr._extract_failing_tests(out)
        assert fails == ["tests/test_a.py::test_one", "tests/test_b.py::TestC::test_two"]

    def test_dedup(self):
        out = "FAILED t.py::test_x - E\nFAILED t.py::test_x - E\n"
        assert self.tr._extract_failing_tests(out) == ["t.py::test_x"]

    def test_strips_message_suffix(self):
        out = "FAILED t.py::test_x - Some Error: with detail\n"
        assert self.tr._extract_failing_tests(out) == ["t.py::test_x"]


# ── _parse_pytest_output ────────────────────────────────────────────────────


class TestParsePytestOutput:
    def setup_method(self):
        self.tr = _tr.TestRunner("/tmp")

    def test_empty_returns_zeros(self):
        r = self.tr._parse_pytest_output("")
        assert r == {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "xpassed": 0,
            "xfailed": 0,
            "failed_tests": [],
            "error_tests": [],
        }

    def test_count_parsing(self):
        out = "...\n10 passed, 2 failed, 3 skipped, 1 xfailed, 1 xpassed in 1.0s\n"
        r = self.tr._parse_pytest_output(out)
        assert r["passed"] == 10
        assert r["failed"] == 2
        assert r["skipped"] == 3
        assert r["xfailed"] == 1
        assert r["xpassed"] == 1

    def test_singular_error_keyword(self):
        # pytest sometimes prints "1 error in ..."
        out = "...\n1 error in 0.5s\n"
        r = self.tr._parse_pytest_output(out)
        assert r["errors"] == 1

    def test_failed_test_detail_with_error_type(self):
        out = "FAILED tests/test_a.py::test_one - AssertionError: boom\n2 passed, 1 failed in 0.1s\n"
        r = self.tr._parse_pytest_output(out)
        assert len(r["failed_tests"]) == 1
        ft = r["failed_tests"][0]
        assert ft["test_id"] == "tests/test_a.py::test_one"
        assert ft["name"] == "test_one"
        assert ft["error_type"] == "AssertionError"
        assert ft["message"] == "boom"

    def test_failed_test_detail_message_only(self):
        out = "FAILED t.py::test_x - no colon here\n1 failed in 0.1s\n"
        r = self.tr._parse_pytest_output(out)
        ft = r["failed_tests"][0]
        assert ft["error_type"] == ""
        assert ft["message"] == "no colon here"

    def test_error_tests_parsing(self):
        out = "ERROR tests/test_e.py::test_setup - setup err\nERROR tests/test_e.py::test_setup2\n1 error in 0.1s\n"
        r = self.tr._parse_pytest_output(out)
        assert len(r["error_tests"]) == 2
        assert r["error_tests"][0]["test_id"] == "tests/test_e.py::test_setup"
        assert r["error_tests"][0]["message"] == "setup err"
        assert r["error_tests"][1]["message"] == ""

    def test_error_tests_dedup(self):
        out = "ERROR t.py::test_x - e\nERROR t.py::test_x - e\n1 error in 0.1s\n"
        r = self.tr._parse_pytest_output(out)
        assert len(r["error_tests"]) == 1

    def test_traceback_block_attached(self):
        out = (
            "____ test_one ____\n"
            "line of tb\n"
            "E   AssertionError: x\n"
            "========================= short test summary =========================\n"
            "FAILED tests/test_a.py::test_one - AssertionError: x\n"
            "1 failed in 0.1s\n"
        )
        r = self.tr._parse_pytest_output(out)
        ft = r["failed_tests"][0]
        assert "line of tb" in ft["traceback"]
        assert "AssertionError" in ft["traceback"]


# ── _extract_first_traceback ────────────────────────────────────────────────


class TestExtractFirstTraceback:
    def setup_method(self):
        self.tr = _tr.TestRunner("/tmp")

    def test_empty_returns_none(self):
        assert self.tr._extract_first_traceback("") is None

    def test_traceback_marker(self):
        out = "header\nTraceback (most recent call last):\n  File x\nE   Error\n"
        tb = self.tr._extract_first_traceback(out)
        assert "Traceback (most recent call last):" in tb
        assert "Error" in tb

    def test_e_prefix_fallback(self):
        # No 'Traceback' marker, but 'E   ' present → fallback window
        out = "noise\nnoise2\nE   ValueError: bad\n"
        tb = self.tr._extract_first_traceback(out)
        assert tb is not None
        assert "ValueError" in tb

    def test_no_marker_no_eprefix_returns_none(self):
        assert self.tr._extract_first_traceback("just some output\nno markers\n") is None


# ── _run_cmd (real subprocess) ──────────────────────────────────────────────


class TestRunCmd:
    def test_real_subprocess_success(self):
        tr = _tr.TestRunner(".")
        # cmd[0] contains 'python' → is_pytest path triggers structured parse
        r = tr._run_cmd([sys.executable, "-c", "print('5 passed in 0.5s')"])
        assert r.ok is True
        assert r.exit_code == 0
        assert r.passed_count == 5
        assert "5 passed" in (r.summary_line or "")

    def test_real_subprocess_failure_exit_code(self):
        tr = _tr.TestRunner(".")
        # python -c that exits 1 and prints a failure summary
        r = tr._run_cmd([sys.executable, "-c", "import sys; print('1 failed in 0.1s'); sys.exit(1)"])
        assert r.ok is False
        assert r.exit_code == 1
        assert r.failed_count == 1

    def test_stream_callback_invoked(self):
        tr = _tr.TestRunner(".")
        seen: list[str] = []
        tr._run_cmd(
            [sys.executable, "-c", "print('hello'); print('world')"],
            stream_callback=lambda line, stream, meta: seen.append(line),
        )
        assert any("hello" in s for s in seen)
        assert any("world" in s for s in seen)

    def test_meta_dict_passed_to_callback(self):
        tr = _tr.TestRunner(".")
        metas: list[dict] = []
        tr._run_cmd(
            [sys.executable, "-c", "print('x')"],
            stream_callback=lambda line, stream, meta: metas.append(meta),
            meta={"attempt": 1},
        )
        assert metas and metas[0].get("attempt") == 1

    def test_timeout_kills_process(self):
        tr = _tr.TestRunner(".")
        start = time.monotonic()
        r = tr._run_cmd([sys.executable, "-c", "import time; time.sleep(30)"], timeout_sec=2)
        elapsed = time.monotonic() - start
        # killed well before the 30s sleep
        assert elapsed < 15
        assert r.ok is False  # exit_code != 0 after kill

    def test_cancel_check_kills_running_process(self):
        tr = _tr.TestRunner(".")
        start = time.monotonic()
        r = tr._run_cmd(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_sec=30,
            cancel_check=lambda: True,
        )
        elapsed = time.monotonic() - start
        # Cancelled on the first poll (0.5 s), not after the 30 s budget.
        assert elapsed < 10
        assert r.cancelled is True
        assert r.timed_out is False
        assert r.ok is False
        assert r.exit_code != 0
        assert "CANCELLED" in r.combined

    def test_timeout_still_wins_with_cancel_check_present(self):
        tr = _tr.TestRunner(".")
        r = tr._run_cmd(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_sec=1,
            cancel_check=lambda: False,
        )
        assert r.timed_out is True
        assert r.cancelled is False
        assert "TIMEOUT" in r.combined

    def test_normal_run_not_cancelled(self):
        tr = _tr.TestRunner(".")
        r = tr._run_cmd(
            [sys.executable, "-c", "print('2 passed in 0.2s')"],
            timeout_sec=30,
            cancel_check=lambda: False,
        )
        assert r.cancelled is False
        assert r.timed_out is False
        assert r.ok is True

    def test_env_overrides_applied(self):
        tr = _tr.TestRunner(".", env_overrides={"MY_TEST_VAR": "abc123"})
        r = tr._run_cmd([sys.executable, "-c", "import os; print(os.environ.get('MY_TEST_VAR', ''))"])
        assert "abc123" in r.stdout

    def test_columns_env_set(self):
        tr = _tr.TestRunner(".")
        r = tr._run_cmd([sys.executable, "-c", "import os; print(os.environ.get('COLUMNS', ''))"])
        assert r.stdout.strip() == "200"


# ── run() / run_pytest() integration ────────────────────────────────────────


class TestRunEntryPoints:
    def test_run_uses_test_command(self):
        tr = _tr.TestRunner(".", test_command=[sys.executable, "-c", "print('3 passed in 0.1s')"])
        r = tr.run()
        assert r.ok and r.passed_count == 3

    def test_run_with_args_override(self):
        tr = _tr.TestRunner(".", test_command=["should-not-be-used"])
        r = tr.run(args=[sys.executable, "-c", "print('1 passed in 0.1s')"])
        assert r.ok and r.passed_count == 1

    def test_run_pytest_delegates_to_run(self):
        tr = _tr.TestRunner(".", test_command=[sys.executable, "-c", "print('7 passed in 0.1s')"])
        r = tr.run_pytest()
        assert r.ok and r.passed_count == 7

    def test_non_pytest_command_skips_structured_parse(self):
        # cmd[0] has no 'pytest'/'python' → is_pytest False → counts stay 0
        tr = _tr.TestRunner("/bin")
        r = tr._run_cmd(["echo", "1 passed"])
        assert r.passed_count == 0  # not parsed as pytest
        assert r.combined == "1 passed"


# ── Timeout teardown: process group, reporting, output bound ────────────────


class TestTimeoutTeardown:
    """The three defects a test command with a surviving grandchild exposed.

    ``test_timeout_kills_process`` above covers the childless case, which always
    worked: SIGKILL on the direct child ends it and the pipes reach EOF. The
    interesting shape is the one real suites produce — pytest-xdist workers, a
    server/daemon fixture, docker — where a grandchild inherits stdout/stderr and
    outlives the child that spawned it.
    """

    @staticmethod
    def _spawner(tmp_path, pidfile, sleep_s=60):
        """A command that leaks a grandchild holding the inherited pipes."""
        script = tmp_path / "spawner.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c',\n"
            f"    \"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid()));\"\n"
            f'    "time.sleep({sleep_s})"])\n'
            f"time.sleep({sleep_s})\n"
        )
        return [sys.executable, str(script)]

    def test_timeout_is_honoured_when_a_grandchild_holds_the_pipes(self, tmp_path):
        """The budget must bound the call, not the orphan's lifetime.

        Regression: the direct child died at the deadline but the grandchild kept
        the write end of stdout open, so the drain threads never saw EOF and the
        `finally` blocked in ``TextIOWrapper.close()`` on the buffer lock those
        threads held. Measured before the fix: ``timeout_sec=1`` returned at
        15.08 s — the grandchild's full sleep, i.e. no timeout at all.
        """
        pidfile = tmp_path / "gc.pid"
        tr = _tr.TestRunner(str(tmp_path), test_command=self._spawner(tmp_path, pidfile, 60))

        start = time.monotonic()
        r = tr._run_cmd(tr.test_command, timeout_sec=1)
        elapsed = time.monotonic() - start

        # Generous vs the 1 s budget (kill + reap + joins), but far under the
        # 60 s that "waited for the orphan" would take. A regression here does
        # not shave the margin, it blows straight past it.
        assert elapsed < 15, f"timeout not honoured: returned after {elapsed:.2f}s"
        assert r.timed_out is True
        assert r.ok is False

    def test_timeout_kills_the_whole_process_group(self, tmp_path):
        """Grandchildren must not outlive the run that spawned them."""
        pidfile = tmp_path / "gc.pid"
        tr = _tr.TestRunner(str(tmp_path), test_command=self._spawner(tmp_path, pidfile, 60))
        tr._run_cmd(tr.test_command, timeout_sec=2)

        # The grandchild writes its pid before sleeping; give the write a moment.
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "grandchild never started — test would prove nothing"
        pid = int(pidfile.read_text())

        # SIGKILL delivery to the group is not instantaneous.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                break
            time.sleep(0.05)
        else:
            os.kill(pid, signal.SIGKILL)  # don't leak it out of the test run
            raise AssertionError(f"grandchild {pid} survived the timeout kill")

    def test_timeout_is_named_in_the_output(self, tmp_path):
        """A killed run must not read as a finished, failing one.

        Regression: the marker went only to ``stream_callback``, which the tool
        handler does not pass — so a timed-out run reached the model as ordinary
        partial output and got debugged as a real test failure.
        """
        tr = _tr.TestRunner(
            str(tmp_path),
            test_command=[sys.executable, "-c", "print('collected 3 items', flush=True); import time; time.sleep(30)"],
        )
        r = tr._run_cmd(tr.test_command, timeout_sec=1)

        assert r.timed_out is True
        assert "TIMEOUT" in r.combined
        assert "TIMEOUT" in r.stderr
        # ...and what the command did print is still there, not discarded.
        assert "collected 3 items" in r.combined

    def test_no_timeout_leaves_the_flag_clear(self, tmp_path):
        """The flag must mean "killed", not "exited nonzero"."""
        tr = _tr.TestRunner(str(tmp_path), test_command=[sys.executable, "-c", "import sys; sys.exit(1)"])
        r = tr._run_cmd(tr.test_command, timeout_sec=30)
        assert r.timed_out is False
        assert r.ok is False  # a real failure, distinguishable from a timeout
        assert "TIMEOUT" not in r.combined

    def test_output_is_bounded(self, tmp_path):
        """A verbose suite must not be held whole in memory.

        Regression: every line was appended to an unbounded list. Measured
        before the fix: 40 MB of output cost +229 MB of peak RSS and produced a
        41.5 MB string for the pytest parsers to walk.
        """
        cap = _thresholds.tokens.BASH_OUTPUT_MAX_CHARS
        n_lines, produced = _bulk_line_count(cap)
        tr = _tr.TestRunner(
            str(tmp_path),
            test_command=[
                sys.executable,
                "-c",
                f"import sys\nfor i in range({n_lines}): sys.stdout.write(f'{{i:07d}} ' + 'x'*{_BULK_PAD} + '\\n')",
            ],
        )
        r = tr._run_cmd(tr.test_command, timeout_sec=60)

        # Retention is head+tail, so the bound is 2*cap — anything under that and
        # over `produced` would pass vacuously, hence the sizing precondition in
        # _bulk_line_count.
        assert len(r.stdout) < 2 * cap + 200, f"retained {len(r.stdout)} chars for a {cap} cap"
        assert len(r.stdout) < produced // 4, "nothing was actually elided"
        assert "chars dropped (middle)" in r.stdout

    def test_bounded_output_keeps_head_and_tail(self, tmp_path):
        """Both ends carry signal: collection errors lead, the summary trails."""
        cap = _thresholds.tokens.BASH_OUTPUT_MAX_CHARS
        n_lines, _ = _bulk_line_count(cap)
        tr = _tr.TestRunner(
            str(tmp_path),
            test_command=[
                sys.executable,
                "-c",
                "import sys\n"
                "print('collected 999 items')\n"
                f"for i in range({n_lines}): sys.stdout.write(f'{{i:07d}} ' + 'x'*{_BULK_PAD} + '\\n')\n"
                "print('1 failed, 998 passed in 12.34s')\n",
            ],
        )
        r = tr._run_cmd(tr.test_command, timeout_sec=60)

        assert "collected 999 items" in r.combined  # head survived
        assert "1 failed, 998 passed in 12.34s" in r.combined  # tail survived
        # ...and the tail is still what the extractors read.
        assert r.summary_line == "1 failed, 998 passed in 12.34s"
        assert r.passed_count == 998 and r.failed_count == 1


class TestKillProcessGroupSafety:
    """``_kill_process_group`` must never signal the group the agent is in.

    The group kill is correct only because ``_run_cmd``'s Popen passes
    ``start_new_session=True``. Without it ``getpgid(child)`` is our OWN group,
    so the SIGKILL takes the agent down with the test command. This is not
    hypothetical: mutating the Popen to verify the regression tests above killed
    the shell running them three times, silently.

    ``os.killpg`` is stubbed rather than exercised: a test that proves the guard
    by actually firing an unguarded SIGKILL takes the whole pytest session with
    it when the guard regresses, reporting nothing. Recording the call is the
    only form of this assertion that can survive its own failure.
    """

    def test_refuses_to_signal_our_own_group(self, monkeypatch):
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(_tr.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

        killed: list[bool] = []

        class _SameGroupProc:
            pid = os.getpid()  # our pid → the resolved pgid is our own group

            def kill(self):
                killed.append(True)

        # The pgid is resolved at Popen time; pass our own group to stand in
        # for a child that (incorrectly) shares it — the guard must refuse.
        _tr._kill_process_group(_SameGroupProc(), os.getpgrp())

        assert calls == [], f"killpg called on our own group: {calls}"
        assert killed == [True], "fell through without killing the child either"

    def test_signals_the_group_when_the_child_owns_one(self, monkeypatch, tmp_path):
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(_tr.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            # Pre-resolve the pgid exactly as _run_cmd does at Popen time.
            _tr._kill_process_group(proc, os.getpgid(proc.pid))
            assert len(calls) == 1
            pgid, sig = calls[0]
            assert sig == signal.SIGKILL
            assert pgid == os.getpgid(proc.pid) != os.getpgrp()
        finally:
            proc.kill()  # the stub never actually signalled it
            proc.wait(timeout=10)
            for _s in (proc.stdout, proc.stderr):
                if _s is not None:
                    _s.close()

    def test_pgid_none_falls_back_to_child_kill(self, monkeypatch):
        """A leader reaped before pgid resolution must not crash the group kill.

        ``_run_cmd`` resolves the pgid immediately after Popen; under parallel
        load an instantly-exiting command can be reaped before ``getpgid``,
        which then raises ProcessLookupError. The degraded value is None —
        there is no group to signal — and ``_kill_process_group`` must fall
        through to killing the (already-dead) child without raising.
        """
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(_tr.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
        killed: list[bool] = []

        class _DeadLeaderProc:
            pid = 999999999  # not resolvable

            def kill(self):
                killed.append(True)

        # Must not raise TypeError (killpg(None, SIGKILL)) nor ProcessLookupError.
        _tr._kill_process_group(_DeadLeaderProc(), None)

        assert calls == [], f"killpg must not be called with None: {calls}"
        assert killed == [True], "must still attempt the direct child kill"

    def test_run_cmd_survives_instant_exit_race(self, monkeypatch, tmp_path):
        """Popen→getpgid race (ProcessLookupError) must not crash _run_cmd.

        Regression for the parallel-suite flake: an `echo`-fast command reaped
        between Popen return and os.getpgid made the whole _run_cmd raise
        ProcessLookupError instead of returning the captured output. The
        resolution is wrapped so a reaped leader yields pgid=None, and the
        group-kill falls back to the direct child.
        """
        monkeypatch.setattr(
            _tr.os,
            "getpgid",
            lambda pid: (_ for _ in ()).throw(ProcessLookupError(pid)),
        )
        tr = _tr.TestRunner(str(tmp_path))
        r = tr._run_cmd(["echo", "1 passed"])
        assert r.combined == "1 passed"
        assert r.passed_count == 0  # 'echo' is not a pytest command
