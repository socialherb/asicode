"""Regression tests for run_bounded_subprocess — the mandatory-timeout +
process-group-kill helper in common/subprocess_utils.py (single source of truth).

Before this helper existed, recovery paths invoked subprocess.run with no
timeout. A pytest that dropped into --pdb / input(), or a pip install hitting a
network stall, would hang the agent loop forever (TimeoutExpired is a
SubprocessError, not OSError, so it escaped the surrounding except handlers
only AFTER the hang). These tests pin the two invariants the fix restores:

  1. A stalled command never hangs — the helper returns within ~timeout, not
     after the child's natural (possibly infinite) lifetime.
  2. The whole process group is torn down (start_new_session=True + killpg),
     so grandchildren (pytest-spawned server fixtures) don't survive as orphans.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

import external_llm.agent.tool_handlers.git_tools as gt_mod
from external_llm.common.subprocess_utils import run_bounded_subprocess

# The helper now lives in common/subprocess_utils.py; git_tools binds it as
# _run_bounded_subprocess. Test the SSOT directly, plus the git_tools binding
# (the live production call path) so a re-localized shadow copy is caught.
_HELPERS = [
    pytest.param(run_bounded_subprocess, id="common"),
    pytest.param(gt_mod._run_bounded_subprocess, id="git_tools"),
]


def test_git_tools_helper_is_the_shared_implementation():
    """git_tools must bind the shared helper, not a re-localized copy."""
    assert gt_mod._run_bounded_subprocess is run_bounded_subprocess


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="process-group kill + POSIX shell semantics; the agent itself requires bash",
)


@pytest.mark.parametrize("run", _HELPERS)
class TestRunBoundedSubprocess:
    def test_normal_execution_returns_completed_process(self, run):
        """A fast command returns a CompletedProcess with stdout/stderr populated."""
        cp = run(["echo", "hello-world"], timeout=10)
        assert cp.returncode == 0
        assert "hello-world" in cp.stdout

    def test_nonzero_returncode_preserved(self, run):
        """The child's real exit code flows through unchanged."""
        cp = run(["sh", "-c", "exit 7"], timeout=10)
        assert cp.returncode == 7

    def test_stderr_captured(self, run):
        """stderr is collected separately from stdout."""
        cp = run(["sh", "-c", "echo out; echo err 1>&2"], timeout=10)
        assert "out" in cp.stdout
        assert "err" in cp.stderr

    def test_input_forwarded_to_stdin(self, run):
        """input= reaches the child's stdin (the git-apply path relies on this)."""
        cp = run(["cat"], timeout=10, input="piped-payload")
        assert cp.stdout == "piped-payload"

    def test_timeout_returns_sentinel_quickly(self, run):
        """A stalled command does NOT block for its natural lifetime.

        The helper must return roughly at `timeout`, not after `sleep 30`. This
        is the core regression: pre-fix, this call would hang 30s (or forever
        for a command waiting on input).
        """
        start = time.monotonic()
        cp = run(["sleep", "30"], timeout=1)
        elapsed = time.monotonic() - start
        # Returns near the 1s timeout, plus a small grace for killpg + drain.
        assert elapsed < 10, f"helper hung for {elapsed:.1f}s instead of killing at timeout"
        assert cp.returncode == -9, "sentinel returncode signals timeout-kill"
        assert "timeout" in cp.stderr, "stderr must explain the abort"

    def test_timeout_kills_the_launched_process(self, run, tmp_path):
        """start_new_session=True + killpg terminates the child, not just
        abandons it. We exec `sleep` over the shell so the recorded PID *is* the
        process the helper spawned — if it survives, the group kill failed.
        """
        marker = tmp_path / "pid.txt"
        cp = run(
            ["bash", "-c", f"echo $$ > {marker}; exec sleep 30"],
            timeout=1,
        )
        assert cp.returncode == -9
        assert marker.exists(), "child never recorded its PID (spawn failed?)"
        pid = int(marker.read_text().strip())
        # Allow the SIGKILL to propagate, then poll (bounded) until the child is
        # fully reaped — a zombie still answers os.kill(pid, 0), so only a
        # successful raise proves the kill. A fixed sleep flakes when the
        # reaper is slow under load.
        deadline = time.monotonic() + 2.0
        reaped = False
        while time.monotonic() < deadline and not reaped:
            try:
                os.kill(pid, 0)  # signal 0 = liveness check; raises if reaped
            except (ProcessLookupError, OSError):
                reaped = True
            else:
                time.sleep(0.05)
        assert reaped, f"pid {pid} still alive 2s after SIGKILL (killpg failed?)"

    def test_timeout_kills_grandchildren_when_leader_exits_first(self, run, tmp_path):
        """The direct child can exit well BEFORE the timeout — e.g.
        ``bash -c "sleep 30 & echo started"`` returns as soon as the
        backgrounding is done, leaving only the grandchild. The group kill must
        then target the GROUP (which survives its leader while any member is
        alive), not a re-resolved ``getpgid(proc.pid)``: ``communicate()``
        reaps the exited leader, so getpgid raises ProcessLookupError and the
        silently-skipped kill orphans the grandchild — the leak the helper
        exists to prevent.
        """
        marker = tmp_path / "grandchild_pid.txt"
        cp = run(
            ["bash", "-c", f"sleep 30 & echo $! > {marker}; echo started"],
            timeout=1,
        )
        assert cp.returncode == -9
        if not marker.exists():
            pytest.skip("grandchild did not record PID before timeout (race); retry")
        gpid = int(marker.read_text().strip())
        deadline = time.monotonic() + 2.0
        reaped = False
        while time.monotonic() < deadline and not reaped:
            try:
                os.kill(gpid, 0)
            except (ProcessLookupError, OSError):
                reaped = True
            else:
                time.sleep(0.05)
        assert reaped, f"pid {gpid} still alive 2s after SIGKILL (leader-reaped killpg failed?)"

    def test_timeout_with_early_leader_exit_keeps_output_and_stays_fast(self, run):
        """Regression for the leader-exits-first shape: the pre-fix path let
        the grandchild hold the pipes, the post-kill drain burned its full 5 s
        grace, and the ``except Exception`` then BLANKED the output the command
        had already printed — the caller saw an empty, failed-looking result
        for a command that had answered. Post-fix the stored-pid killpg works,
        the drain returns immediately, and the partial output survives.
        """
        start = time.monotonic()
        cp = run(["bash", "-c", "sleep 30 & echo started"], timeout=1)
        elapsed = time.monotonic() - start
        assert cp.returncode == -9
        assert "started" in cp.stdout, f"partial stdout lost on timeout: {cp.stdout!r}"
        # Pre-fix this returned in ~6 s (1 s budget + 5 s failed second drain);
        # the teardown must stay near the budget.
        assert elapsed < 4, f"teardown dragged on for {elapsed:.1f}s (kill was skipped?)"

    def test_timeout_kills_grandchildren(self, run, tmp_path):
        """A backgrounded child of the shell is in the same process group and
        must be torn down too — otherwise pytest-spawned server fixtures (or any
        `cmd &` grandchild) leak as orphans.
        """
        marker = tmp_path / "grandchild_pid.txt"
        cp = run(
            ["bash", "-c", f"sleep 30 & echo $! > {marker}; wait"],
            timeout=1,
        )
        assert cp.returncode == -9
        # The grandchild writes its PID before the parent blocks on `wait`.
        if not marker.exists():
            pytest.skip("grandchild did not record PID before timeout (race); retry")
        gpid = int(marker.read_text().strip())
        deadline = time.monotonic() + 2.0
        reaped = False
        while time.monotonic() < deadline and not reaped:
            try:
                os.kill(gpid, 0)
            except (ProcessLookupError, OSError):
                reaped = True
            else:
                time.sleep(0.05)
        assert reaped, f"pid {gpid} still alive 2s after SIGKILL (killpg failed?)"


# ── inherit_output mode: live-streaming launches with the same hard bound ────


class TestInheritOutputMode:
    """inherit_output=True: the child shares the real terminal (no pipes) while
    keeping the mandatory timeout + process-group kill. Added for the
    auto-learning monitor (tools tree), whose subprocess must stream live
    output but used a bare timeout-less run that hung the loop forever."""

    def test_inherit_mode_rejects_input(self):
        # stdin is inherited in this mode, so input= has no pipe to go through.
        with pytest.raises(ValueError, match="inherit_output"):
            run_bounded_subprocess(["cat"], timeout=5, input="x", inherit_output=True)

    def test_inherit_mode_runs_to_completion(self, capfd):
        cp = run_bounded_subprocess(["echo", "inherit-ok"], timeout=10, inherit_output=True)
        assert cp.returncode == 0
        # Nothing was captured — output went to the real stdout instead.
        assert cp.stdout == ""
        assert cp.stderr == ""
        assert "inherit-ok" in capfd.readouterr().out

    def test_inherit_mode_nonzero_exit_preserved(self):
        cp = run_bounded_subprocess(["sh", "-c", "exit 7"], timeout=10, inherit_output=True)
        assert cp.returncode == 7

    def test_inherit_mode_timeout_kills_and_reports(self):
        start = time.monotonic()
        cp = run_bounded_subprocess(["sleep", "30"], timeout=1, inherit_output=True)
        elapsed = time.monotonic() - start
        assert elapsed < 10, f"inherit mode hung for {elapsed:.1f}s instead of killing at timeout"
        assert cp.returncode == -9
        assert "timeout" in cp.stderr, "stderr must explain the abort even when not captured"

    def test_inherit_mode_timeout_kills_grandchildren(self, tmp_path):
        """The process-group kill works identically in inherit mode — a hung
        monitor's grandchildren must not survive as orphans."""
        marker = tmp_path / "grandchild_pid.txt"
        cp = run_bounded_subprocess(
            ["bash", "-c", f"sleep 30 & echo $! > {marker}; wait"],
            timeout=1,
            inherit_output=True,
        )
        assert cp.returncode == -9
        if not marker.exists():
            pytest.skip("grandchild did not record PID before timeout (race); retry")
        gpid = int(marker.read_text().strip())
        deadline = time.monotonic() + 2.0
        reaped = False
        while time.monotonic() < deadline and not reaped:
            try:
                os.kill(gpid, 0)
            except (ProcessLookupError, OSError):
                reaped = True
            else:
                time.sleep(0.05)
        assert reaped, f"pid {gpid} still alive 2s after SIGKILL (inherit-mode killpg failed?)"


# ── source-level guard: the bare-subprocess.run regression must not return ──


def test_no_timeoutless_subprocess_run_in_target_files():
    """Static guard: no target file may reintroduce a bare blocking subprocess call.

    A future edit could quietly drop the timeout for a new call site. AST-scan
    every HTTP-path file that spawns subprocesses and fail if any blocking call
    (run/check_output/check_call/call) lacks a timeout= keyword. Popen is allowed
    because the main bash path pairs it with communicate(timeout=).
    """
    import ast

    targets = [
        "external_llm/common/subprocess_utils.py",
        "external_llm/agent/tool_handlers/git_tools.py",
        "external_llm/intelligent_service.py",
        "webapp/ui/ui_tools.py",
        "diff_apply.py",
    ]
    blocking = {
        "subprocess.run",
        "subprocess.check_output",
        "subprocess.check_call",
        "subprocess.call",
    }
    offenders = []
    for path in targets:
        # The public (CLI-only) snapshot ships without webapp/ — guard only
        # the targets present in this tree.
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src, path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "subprocess"
            ):
                continue
            call = f"subprocess.{func.attr}"
            if call not in blocking:
                continue
            if not any(kw.arg == "timeout" for kw in node.keywords):
                offenders.append(f"{path}:{node.lineno} ({call})")
    assert not offenders, "timeout-less blocking subprocess call reintroduced (hang risk): " + ", ".join(offenders)
