"""Tests for the macOS capability shims prepended to agent commands.

macOS ships BSD userland and lacks several GNU tools that LLMs emit
frequently (``timeout``, ``tac``, ``nproc``, ``shuf``, ``gtimeout``...).
Without a shim the command dies with "command not found".
``_apply_shell_shims`` prepends a conditional shell function for each, guarded
by ``command -v <name>`` so it is a complete no-op on hosts where the real
binary exists (Linux, GNU coreutils). These tests execute the shim through
real bash (the same path the bash tool uses) so the contract is verified
end-to-end, and are written to pass on BOTH macOS (shim active) and Linux
(real binary, shim inert).

Design principle under test: shims are added ONLY when they can produce output
identical to the GNU original (``tac``/``nproc``/``shuf``/``gtimeout``/
``realpath``). GNU-vs-BSD-incompatible tools (``gsed``, ``gstat``) get an
explanatory error stub instead — aliasing them to BSD tools would silently
corrupt output.
"""

import os
import shutil
import subprocess

import pytest

from external_llm.agent.tool_handlers.git_tools import (
    _SHELL_SHIM_PRELUDE,
    _apply_shell_shims,
)

_BASH = shutil.which("bash")


def _run(cmd, env=None):
    """Run *cmd* through the shim + bash, exactly like the bash tool."""
    p = subprocess.run(
        _apply_shell_shims(cmd),
        shell=True,
        executable=_BASH,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


# ── (1) prelude is prepended & is itself silent ──────────────────────────────


def test_prelude_prepended():
    out = _apply_shell_shims("echo hi")
    # timeout is the first shim block (it may be preceded by a comment line).
    assert "command -v timeout" in out
    assert out.rstrip().endswith("echo hi")


def test_prelude_alone_is_silent():
    # The function definition must produce no stdout/stderr on its own.
    p = subprocess.run(
        _SHELL_SHIM_PRELUDE,
        shell=True,
        executable=_BASH,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.stdout == "" and p.stderr == ""


# ── (2) functional behaviour — passes on macOS (shim) AND Linux (real timeout)


@pytest.mark.slow
def test_normal_completion():
    rc, out, _ = _run("timeout 2 bash -c 'echo hi'")
    assert rc == 0
    assert out == "hi"


def test_timeout_returns_124():
    # GNU coreutils contract: 124 when the time limit is reached.
    rc, _, _ = _run("timeout 1 sleep 5")
    assert rc == 124


@pytest.mark.skipif(
    shutil.which("timeout") is not None,
    reason=(
        "the shim's own `timeout` only exists when a real one does NOT: the "
        "prelude is guarded by `if ! command -v timeout`. Where coreutils "
        "provides /usr/bin/timeout (any Linux CI runner) the shim function is "
        "never defined, so this would assert on GNU coreutils' usage output "
        "instead of on the shim — measured there as rc=125 with only "
        "\"Try 'timeout --help' for more information.\" on stderr, no "
        '"missing operand" line at all. Pinning another project\'s message '
        "format buys no coverage of ours, so skip rather than branch."
    ),
)
def test_missing_command():
    rc, _, err = _run("timeout 5")
    assert rc == 1
    assert "missing command" in err


@pytest.mark.slow
def test_pipe_and_redirection_preserved():
    # The gradlew-shaped pattern: timeout N CMD 2>&1 | tail -1
    rc, out, _ = _run("timeout 2 bash -c 'echo line1; echo line2' 2>&1 | tail -1")
    assert rc == 0
    assert out == "line2"


# ── (2b) GNU option forms — the duration is NOT always "$1" ──────────────────
#
# The shim used to bind dur="$1" unconditionally, so every option form shifted
# the real duration into the command slot: `timeout -k 2 5 echo hi` ran `2 5
# echo hi` and returned 127 with the command never executed. That is the silent
# -failure-in-a-pipeline hazard the whole prelude exists to prevent, because
# `timeout -k 5 300 pytest ... | tail -20` then yields an empty tail that the
# agent reads as a pass. Every shape below is one an LLM emits routinely; they
# hold for real GNU coreutils too, so this table is not shim-only.


@pytest.mark.slow
@pytest.mark.parametrize(
    "invocation",
    [
        "timeout -k 2 5 echo hi",
        "timeout -k2 5 echo hi",
        "timeout --kill-after=2 5 echo hi",
        "timeout -s KILL 5 echo hi",
        "timeout -s TERM 5 echo hi",
        "timeout --signal=KILL 5 echo hi",
        "timeout --preserve-status 5 echo hi",
        "timeout --foreground 5 echo hi",
        "timeout -k 2 -s KILL 5 echo hi",
        "timeout 5s echo hi",  # GNU duration suffix, no option
    ],
)
def test_option_forms_still_run_the_command(invocation):
    rc, out, err = _run(invocation)
    assert out == "hi", f"{invocation!r} did not run the command (stderr: {err!r})"
    assert rc == 0, f"{invocation!r} → rc={rc}"


@pytest.mark.slow
def test_option_forms_still_time_out():
    """An option prefix must not cost the 124 verdict."""
    rc, _, _ = _run("timeout -k 1 1 sleep 5")
    assert rc == 124


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("timeout") is not None,
    reason=(
        "same reason as test_missing_command: where coreutils provides a real "
        "timeout the shim function is never defined, so this asserts GNU's "
        "escalation semantics rather than ours. The exit codes diverge only "
        "once the child TRAPS TERM and the KILL escalation is forced — the "
        "plain-timeout cases above stay 124 on both platforms, which is why "
        "only this one needs the guard. Measured on ubuntu-latest / coreutils: "
        "rc=-9, i.e. the KILL reaches the invoking bash itself rather than "
        "surfacing as 124. Pinning that buys no coverage of our shim."
    ),
)
def test_kill_after_escalates_to_sigkill():
    """`-k` is honoured, not merely parsed and dropped.

    The child traps TERM and keeps running, so only the KILL escalation can end
    it. Without `-k` handling this hangs until the outer test timeout; with it,
    the run ends promptly and still reports 124.

    Promptness is asserted, not just assumed: the escalation must take the
    whole process GROUP down (`set -m` + `kill -- -pid`), so the wrapper's
    children die with it. A pid-only kill orphans the child's `sleep`, which
    keeps the caller's pipes open until it finishes on its own — measured as a
    2 s timeout turning into a 31 s hang, invisible to the rc assertion alone.
    """
    import time as _time

    _start = _time.monotonic()
    rc, _, _ = _run("timeout -k 1 1 bash -c 'trap \"\" TERM; sleep 30'")
    _elapsed = _time.monotonic() - _start
    assert rc == 124
    # 1 s TERM grace + 1 s KILL grace + reap slack; the pre-fix orphan held the
    # pipes for the full 30 s of its own lifetime.
    assert _elapsed < 5, f"run dragged on for {_elapsed:.1f}s (group kill failed?)"


@pytest.mark.slow
def test_option_forms_survive_a_pipeline():
    """The regression's real-world shape: an option form heading a pipeline."""
    rc, out, _ = _run("timeout -k 2 5 bash -c 'echo line1; echo line2' 2>&1 | tail -1")
    assert rc == 0
    assert out == "line2"


# ── (3) no-op when a real `timeout` exists (Linux / coreutils) ───────────────


def test_shim_does_not_shadow_real_timeout():
    """When `timeout` is on PATH, the shim must NOT define its function, so the
    real binary is used instead. Verified by planting a fake `timeout`."""
    import tempfile

    d = tempfile.mkdtemp()
    fake = os.path.join(d, "timeout")
    with open(fake, "w") as f:
        f.write('#!/bin/bash\necho FAKE-TIMEOUT "$@"\n')
    os.chmod(fake, 0o755)
    try:
        env = dict(os.environ)
        env["PATH"] = d + os.pathsep + env["PATH"]
        rc, out, _ = _run("timeout 5 echo hi", env=env)
        assert rc == 0
        assert "FAKE-TIMEOUT" in out, "shim shadowed a real timeout"
    finally:
        os.remove(fake)
        os.rmdir(d)


# ── (4) extended shims: tac / nproc / shuf / gtimeout ────────────────────────
# These have identical semantics to the GNU originals, so the functional tests
# pass regardless of whether the shim or the real binary is active on the host.


def test_tac_reverses_lines():
    rc, out, _ = _run("printf '1\\n2\\n3\\n' | tac")
    assert rc == 0
    assert out == "3\n2\n1"


def test_nproc_returns_positive_int():
    rc, out, _ = _run("nproc")
    assert rc == 0
    assert out.isdigit() and int(out) >= 1


def test_shuf_permutes_without_loss():
    # Shuffling then sorting must recover the original multiset.
    rc, out, _ = _run("printf 'a\\nb\\nc\\n' | shuf | sort")
    assert rc == 0
    assert out == "a\nb\nc"


def test_shuf_n_limits_count():
    rc, out, _ = _run("printf 'a\\nb\\nc\\nd\\n' | shuf -n 2 | wc -l | tr -d ' '")
    assert rc == 0
    assert out == "2"


def test_gtimeout_normal_completion():
    rc, out, _ = _run("gtimeout 2 echo hi")
    assert rc == 0
    assert out == "hi"


def test_gtimeout_returns_124():
    rc, _, _ = _run("gtimeout 1 sleep 5")
    assert rc == 124


# ── (5) gsed / gstat error stubs — only when the g-variant is absent ─────────
# These GNU variants CANNOT be aliased to BSD sed/stat (different -i / -c
# semantics would silently corrupt output), so the shim emits a clear install
# hint and returns 127. Skipped on hosts that happen to have the real tool.


@pytest.mark.skipif(shutil.which("gsed") is not None, reason="real gsed present")
def test_gsed_stub_when_absent():
    rc, _, err = _run("gsed s/x/y/")
    assert rc == 127
    assert "gsed" in err and "brew" in err


@pytest.mark.skipif(shutil.which("gstat") is not None, reason="real gstat present")
def test_gstat_stub_when_absent():
    rc, _, err = _run("gstat -c %s /etc/hosts")
    assert rc == 127
    assert "gstat" in err and "brew" in err


# ── (6) structural: every shim is guarded by `command -v` (no-op on Linux) ───


def test_all_shims_guarded_by_command_v():
    """Each shim must be wrapped in `if ! command -v <name>` so the whole
    prelude is inert wherever the real binary lives on PATH."""
    for name in ("timeout", "gtimeout", "tac", "nproc", "shuf", "realpath", "gsed", "gstat"):
        assert f"command -v {name}" in _SHELL_SHIM_PRELUDE, f"{name} missing no-op guard"


# ── timeout's verdict must not depend on a race ─────────────────────────────
# `timeout` used to decide "did we time out?" by asking whether the watchdog
# subshell was still alive. The watchdog has nothing to do after `kill -TERM`,
# so it normally exits before `wait` resumes the caller — but only normally.
# Lose that race and a timed-out command reports 143 (128+SIGTERM) instead of
# 124, so every caller keying on 124 sees a generic signal death. It surfaced
# once in an 11,918-test suite run: rare enough to look like noise, real enough
# to matter on a loaded machine.
#
# Asserted deterministically instead of by repetition: keeping the watchdog
# alive one extra beat past its kill is exactly the state the old code
# mis-read, and it is reproducible on demand. A correct implementation cannot
# care how long the watchdog lives.


def _widened_prelude() -> str:
    """The shipped shim, with the watchdog left alive past its own kill."""
    from external_llm.agent.tool_handlers.git_tools import _SHELL_SHIM_PRELUDE

    # Anchored on the watchdog subshell's closing line rather than on the kill
    # itself: the kill is now followed by the optional `-k` escalation block, so
    # "one beat past its kill" means one beat past the whole subshell body.
    widened = _SHELL_SHIM_PRELUDE.replace(
        "      fi ) &",
        "      fi; sleep 0.3 ) &",
    )
    assert widened != _SHELL_SHIM_PRELUDE, (
        "the watchdog line moved — update this test's widening patch, do not "
        "delete it: without the widening the assertion below passes vacuously"
    )
    return widened


@pytest.mark.parametrize("name", ["timeout", "gtimeout"])
def test_timeout_reports_124_even_when_the_watchdog_outlives_its_kill(name):
    rc = subprocess.run(
        _widened_prelude() + f"\n{name} 1 sleep 5",
        shell=True,
        executable=_BASH,
        capture_output=True,
        text=True,
        check=False,
    ).returncode
    assert rc == 124, (
        f"{name} returned {rc} (143 = SIGTERM leaked through) — the timeout "
        "verdict is racing the watchdog's exit instead of reading a fact"
    )


def test_timeout_still_reports_the_command_status_when_it_finishes_first():
    """The other direction: no timeout means the command's own exit code."""
    for script, expected in (("gtimeout 5 true", 0), ("gtimeout 5 sh -c 'exit 7'", 7)):
        rc, _, _ = _run(script)
        assert rc == expected, f"{script!r} -> {rc}, expected {expected}"


def test_timeout_marker_files_are_not_left_behind(tmp_path):
    """The marker is an implementation detail; it must not accumulate in TMPDIR."""
    env = dict(os.environ, TMPDIR=str(tmp_path))
    rc, _, _ = _run("gtimeout 1 sleep 5", env=env)
    assert rc == 124
    leftovers = list(tmp_path.glob("asi_timeout.*"))
    assert not leftovers, f"marker files left behind: {leftovers}"
