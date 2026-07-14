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
        shell=True, executable=_BASH,
        capture_output=True, text=True, env=env,
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
        _SHELL_SHIM_PRELUDE, shell=True, executable=_BASH,
        capture_output=True, text=True,
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


# ── (3) no-op when a real `timeout` exists (Linux / coreutils) ───────────────

def test_shim_does_not_shadow_real_timeout():
    """When `timeout` is on PATH, the shim must NOT define its function, so the
    real binary is used instead. Verified by planting a fake `timeout`."""
    import tempfile
    d = tempfile.mkdtemp()
    fake = os.path.join(d, "timeout")
    with open(fake, "w") as f:
        f.write("#!/bin/bash\necho FAKE-TIMEOUT \"$@\"\n")
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
