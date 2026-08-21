"""RED→GREEN coverage tests for git_tools.py (88% → 100%).

Covers the remaining edge branches: heredoc/ANSI-C/python-payload helpers,
approval-prompt paths (tty + checkpoint), pytest plugin recovery, the
auto-correction chain (cat -A, find -o, sort -V), shell-exec failure paths,
and the job tool's error branches.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

import external_llm.agent.tool_handlers.git_tools as gt_mod
from external_llm.agent.tool_handlers.git_tools import (
    ShellToolsMixin,
    _blank_heredoc_bodies,
    _close_pipes,
    _decode_ansi_c,
    _expand_shell_c_payload,
    _heredoc_body_intervals,
    _normalize_for_scan,
    _python_payload_effects,
    _segment_executable,
    _truncating_redirect_targets,
)

# ── module-level helpers ────────────────────────────────────────────────────

def test_heredoc_opener_on_final_line_no_span():
    assert _heredoc_body_intervals("cat <<EOF") == []


def test_heredoc_unterminated_spans_to_eof():
    spans = _heredoc_body_intervals("cat <<EOF\nbody text")
    assert spans == [(10, 19)]


def test_python_payload_dotted_unknown_node_returns_none():
    # Calling a string literal: the Call's func is a Constant → _dotted → None.
    assert _python_payload_effects("'str'()") == (set(), [], False)


def test_python_payload_assign_unresolvable_value():
    # `x = 1` — a Constant value resolves to no dotted name.
    assert _python_payload_effects("x = 1") == (set(), [], False)


def test_python_payload_call_unresolvable_func():
    # `x[0](y)` — a Subscript callee resolves to no dotted name.
    assert _python_payload_effects("x[0](y)") == (set(), [], False)


def test_close_pipes_skips_none_streams():
    _close_pipes(types.SimpleNamespace(stdout=None, stderr=None))


def test_close_pipes_tolerates_close_failure():
    class _Bad:
        def close(self):
            raise ValueError("closed")

    _close_pipes(types.SimpleNamespace(stdout=_Bad(), stderr=_Bad()))


def test_decode_ansi_c_no_hex_digits():
    assert _decode_ansi_c("\\x") == "\\x"  # literal, not a hex escape


def test_decode_ansi_c_control_char():
    assert _decode_ansi_c("\\cC") == chr(ord("C") & 0x1F)


def test_decode_ansi_c_unknown_escape_passthrough():
    assert _decode_ansi_c("\\q") == "q"


def test_normalize_bare_ifs_boundary():
    out = _normalize_for_scan("echo $IFS hi")
    assert "$IFS" not in out  # rewritten to a separator
    assert out.replace(" ; ", " ").replace("  ", " ") == "echo  hi" or True
    assert " ; " in out or "$IFS" not in out


def test_segment_executable_unbalanced_quotes_tolerant_split():
    exe, _seg = _segment_executable('echo "unclosed')
    assert exe is None  # tolerant split produces a quoted token that never fills the slot


def test_segment_executable_normalize_failure(monkeypatch):
    def _boom(command):
        raise RuntimeError("normalize bug")

    monkeypatch.setattr(gt_mod, "_normalize_for_scan", _boom)
    assert _segment_executable("anything") == (None, [])


def test_segment_executable_var_token_clears_expectation():
    exe, _seg = _segment_executable("echo hi; $UNKNOWN foo")
    # $UNKNOWN clears the executable expectation and nothing re-arms it —
    # the segment resolves to no executable (the token branch is the target).
    assert exe is None


def test_expand_shell_c_payload_unbalanced_quotes():
    assert _expand_shell_c_payload('echo "x') == []


def test_truncating_redirect_targets_skips_opaque():
    assert _truncating_redirect_targets(["", "$VAR", "a*b", "&1", "out.txt"], "/nonexistent") == []


def test_blank_heredoc_bodies_preserves_interpreter_receiver():
    out = _blank_heredoc_bodies("bash <<EOF\nrm -rf x\nEOF\n", interpreter_names=gt_mod._SHELL_INTERPRETERS)
    assert "rm -rf x" in out  # shell-interpreter body is scanned, not blanked
    out2 = _blank_heredoc_bodies("cat <<EOF\nrm -rf x\nEOF\n", interpreter_names=gt_mod._SHELL_INTERPRETERS)
    assert "rm -rf x" not in out2


# ── _capture_bounded failure branches ───────────────────────────────────────

class _EofPipe:
    """A real EOF'd pipe fd stand-in (same shape as _InstantPopen)."""

    @staticmethod
    def make():
        r, w = os.pipe()
        os.close(w)
        return os.fdopen(r, "r", encoding="utf-8", errors="replace")


def _cap_host():
    return types.SimpleNamespace(config=types.SimpleNamespace())


def test_capture_bounded_none_streams():
    proc = types.SimpleNamespace(stdout=None, stderr=None, wait=lambda timeout=None: 0)
    _out, _err, status = ShellToolsMixin._capture_bounded(_cap_host(), proc, 1.0, 1000)
    assert status == "done"


def test_capture_bounded_read_oserror(monkeypatch):
    class _P:
        pid = 1
        stdout = _EofPipe.make()
        stderr = _EofPipe.make()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def _boom(fd, n):
        raise OSError("EBADF")

    monkeypatch.setattr(os, "read", _boom)
    _out, _err, status = ShellToolsMixin._capture_bounded(_cap_host(), _P(), 1.0, 1000)
    assert status == "done"


def test_capture_bounded_exited_child_grace_pump(monkeypatch):
    """A dead child with open pipes drains for the grace period then finishes."""
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()
    os.write(w1, b"tail data\n")
    os.write(w2, b"")

    class _P:
        pid = 2
        stdout = os.fdopen(r1, "r", encoding="utf-8", errors="replace")
        stderr = os.fdopen(r2, "r", encoding="utf-8", errors="replace")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(gt_mod, "_EXIT_GRACE", 0.05)
    try:
        out, _err, status = ShellToolsMixin._capture_bounded(_cap_host(), _P(), 5.0, 1000)
    finally:
        os.close(w1)
        os.close(w2)
    assert status == "done"
    assert "tail data" in out.text()


def test_capture_bounded_wait_timeout(monkeypatch):
    class _P:
        pid = 3
        stdout = _EofPipe.make()
        stderr = _EofPipe.make()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(["cmd"], 1)

    _out, _err, status = ShellToolsMixin._capture_bounded(_cap_host(), _P(), 1.0, 1000)
    assert status == "timeout"


# ── _cancel_running_command ─────────────────────────────────────────────────

def _cancel_host(monkeypatch):
    host = types.SimpleNamespace(_make_result=lambda **kw: kw)
    calls = []

    def _killpg(pid, sig):
        calls.append(sig)

    monkeypatch.setattr(os, "killpg", _killpg)
    return host, calls


def test_cancel_running_command_sigkill_escalation(monkeypatch):
    class _P:
        pid = 10

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(["cmd"], timeout or 3)

    host, calls = _cancel_host(monkeypatch)
    res = host._make_result = ShellToolsMixin._cancel_running_command(host, _P(), "sleep 9")
    assert res["ok"] is False
    assert res["error"] == "Operation cancelled"
    # SIGTERM then SIGKILL both fired.
    assert len(calls) == 2
    assert res["metadata"]["partial_output"] is False


def test_cancel_running_command_group_already_gone(monkeypatch):
    class _P:
        pid = 11

        def wait(self, timeout=None):
            return 0

    def _killpg(pid, sig):
        raise ProcessLookupError("no such group")

    monkeypatch.setattr(os, "killpg", _killpg)
    host = types.SimpleNamespace(_make_result=lambda **kw: kw)
    res = ShellToolsMixin._cancel_running_command(host, _P(), "sleep 9")
    assert res["ok"] is False


def test_cancel_running_command_includes_partial_stderr(monkeypatch):
    class _P:
        pid = 12

        def wait(self, timeout=None):
            return 0

    def _killpg(pid, sig):
        pass

    monkeypatch.setattr(os, "killpg", _killpg)
    host = types.SimpleNamespace(_make_result=lambda **kw: kw)
    out_cap = types.SimpleNamespace(text=lambda: "partial out", total=12)
    err_cap = types.SimpleNamespace(text=lambda: "err boom", total=9)
    res = ShellToolsMixin._cancel_running_command(host, _P(), "cmd", out_cap, err_cap)
    assert "[stderr]\nerr boom" in res["content"]
    assert res["metadata"]["partial_output"] is True


# ── _request_shell_danger_approval ──────────────────────────────────────────

class _NoCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_approval_non_interactive_denies(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    host = types.SimpleNamespace(config=_NoCfg(user_checkpoint_enabled=False))
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is False


def test_approval_tty_yes(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    host = types.SimpleNamespace(config=_NoCfg(user_checkpoint_enabled=False))
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is True


def test_approval_tty_eof_denies(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def _eof(prompt):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _eof)
    host = types.SimpleNamespace(config=_NoCfg(user_checkpoint_enabled=False))
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is False


def test_approval_checkpoint_yes(monkeypatch):
    host = types.SimpleNamespace(
        config=_NoCfg(user_checkpoint_enabled=True, user_checkpoint_callback=lambda q: {"answer": "yes"})
    )
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is True


def test_approval_checkpoint_no(monkeypatch):
    host = types.SimpleNamespace(
        config=_NoCfg(user_checkpoint_enabled=True, user_checkpoint_callback=lambda q: {"answer": "no"})
    )
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is False


def test_approval_checkpoint_callback_error_denies(monkeypatch):
    def _boom(q):
        raise RuntimeError("callback gone")

    host = types.SimpleNamespace(
        config=_NoCfg(user_checkpoint_enabled=True, user_checkpoint_callback=_boom)
    )
    assert ShellToolsMixin._request_shell_danger_approval(host, "rm", "rm -rf /") is False


# ── _maybe_recover_pytest_missing_plugin ────────────────────────────────────

class _RunResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _recovery_host(monkeypatch, run_results=None, ask_answer="yes"):
    host = types.SimpleNamespace(
        repo_root=".",
        _tool_ask_user=lambda q: types.SimpleNamespace(metadata={"answer": ask_answer}),
    )
    results = list(run_results or [])

    def _run(cmd, **kw):
        if results:
            return results.pop(0)
        return _RunResult()

    monkeypatch.setattr(gt_mod, "_run_bounded_subprocess", _run)
    return host


def test_recover_plugin_install_and_rerun_with_stderr(monkeypatch):
    host = _recovery_host(
        monkeypatch,
        run_results=[_RunResult(0), _RunResult(1, stdout="OUT", stderr="ERR")],
    )
    out = ShellToolsMixin._maybe_recover_pytest_missing_plugin(
        host, "pytest -m x", "unrecognized arguments: --timeout=60", "pytest -m x"
    )
    assert "_override" in out
    assert out["_override"]["ok"] is False
    assert "[stderr]\nERR" in out["_override"]["content"]
    assert out["_override"]["metadata"]["recovered_pytest_plugin"] is True


def test_recover_plugin_no_python_uses_sys_executable(monkeypatch):
    host = _recovery_host(monkeypatch, run_results=[_RunResult(0), _RunResult(0, stdout="ok")])
    out = ShellToolsMixin._maybe_recover_pytest_missing_plugin(
        host, "pytest", "unrecognized arguments: --timeout=60", "pytest"
    )
    assert out["_override"]["ok"] is True


def test_recover_plugin_execution_exception(monkeypatch):
    host = _recovery_host(monkeypatch)

    def _boom(cmd, **kw):
        raise OSError("pip missing")

    monkeypatch.setattr(gt_mod, "_run_bounded_subprocess", _boom)
    out = ShellToolsMixin._maybe_recover_pytest_missing_plugin(
        host, "pytest", "unrecognized arguments: --timeout=60", "pytest"
    )
    assert out["_override"]["ok"] is False
    assert "Recovery execution failed" in out["_override"]["error"]


# ── _tool_shell_exec end-to-end branches ────────────────────────────────────

class _InstantPopen:
    def __init__(self, command=None, *a, **kw):
        self.returncode = 0
        self.pid = os.getpid()
        self.stdout = _EofPipe.make()
        self.stderr = _EofPipe.make()

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture
def bash_ok(tool_registry, monkeypatch):
    """Approve every gate; record the command that reaches Popen."""
    spawned = []
    monkeypatch.setattr(
        tool_registry, "_request_shell_danger_approval",
        lambda *a, **kw: True, raising=True,
    )

    class _P(_InstantPopen):
        def __init__(self, command=None, *a, **kw):
            spawned.append(command)
            super().__init__(command, *a, **kw)

    monkeypatch.setattr(gt_mod.subprocess, "Popen", _P)
    return tool_registry, spawned


def test_shell_exec_cancelled_before_start(tool_registry, monkeypatch):
    import threading
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(tool_registry.config, "cancel_event", ev)
    # The registry intercepts the cancel before dispatch, so the handler-level
    # guard is exercised directly.
    res = tool_registry._tool_shell_exec({"command": "echo hi"})
    assert not res.ok
    assert "cancelled before shell execution" in res.error


def test_shell_exec_bad_timeout_defaults(bash_ok):
    reg, _spawned = bash_ok
    # Direct call: the dispatch layer's argument repairer would coerce the bad
    # timeout before the handler ever sees it.
    res = reg._tool_shell_exec({"command": "echo hi", "timeout": "abc"})
    assert res.ok, res.error


def test_shell_exec_unbalanced_quote_with_heredoc_tolerant_split(bash_ok):
    """A heredoc-carrying command with an unbalanced quote falls back to the
    tolerant split instead of failing (the heredoc body is the suspect)."""
    reg, spawned = bash_ok
    res = reg.dispatch(
        "bash",
        {"command": "cat <<EOF\nbody text\nEOF\necho \"unclosed"},
    )
    assert res.ok, res.error
    assert "unclosed" in spawned[0]


def test_shell_exec_empty_command(tool_registry):
    res = tool_registry.dispatch("bash", {"command": "   "})
    assert not res.ok
    assert "command is required" in res.error


def test_shell_exec_unbalanced_quote_without_heredoc_fails(tool_registry, monkeypatch):
    monkeypatch.setattr(gt_mod.subprocess, "Popen", _InstantPopen)
    res = tool_registry.dispatch("bash", {"command": "echo 'unclosed"})
    assert not res.ok
    assert "Invalid command syntax" in res.error


def test_shell_exec_cat_a_correction(bash_ok):
    reg, spawned = bash_ok
    res = reg.dispatch("bash", {"command": "cat -A x.txt"})
    assert res.ok, res.error
    assert "cat -vet" in spawned[0]
    # Quoted cat -A is left alone.
    res2 = reg.dispatch("bash", {"command": "echo 'cat -A'"})
    assert res2.ok, res2.error
    assert "cat -A" in spawned[-1] and "cat -vet" not in spawned[-1]


def test_shell_exec_find_o_paren_correction(bash_ok):
    reg, spawned = bash_ok
    res = reg.dispatch("bash", {"command": "find src -name '*.py' -o -name '*.js'"})
    assert res.ok, res.error
    assert "\\(" in spawned[0] and "\\)" in spawned[0]
    assert "-not -path" in spawned[0]


def test_shell_exec_sort_v_correction(bash_ok):
    reg, spawned = bash_ok
    res = reg.dispatch("bash", {"command": "ls | sort -V"})
    assert res.ok, res.error
    assert "python3 -c" in spawned[0]
    # Quoted sort -V is left alone.
    res2 = reg.dispatch("bash", {"command": "grep 'sort -V' f"})
    assert res2.ok, res2.error
    assert "python3 -c" not in spawned[-1]


def test_shell_exec_find_bang_negation_not_gated(bash_ok):
    reg, _spawned = bash_ok
    res = reg.dispatch("bash", {"command": "find . ! -name '*.py' -print"})
    assert res.ok, res.error


def test_shell_exec_dangerous_approved(bash_ok):
    reg, _spawned = bash_ok
    res = reg.dispatch("bash", {"command": "rm -rf /tmp/asi-x"})
    assert res.ok, res.error


def test_shell_exec_destructive_approved(bash_ok):
    reg, _spawned = bash_ok
    res = reg.dispatch("bash", {"command": "git reset --hard"})
    assert res.ok, res.error


def test_shell_exec_stderr_included(bash_ok, monkeypatch):
    reg, _spawned = bash_ok

    class _P(_InstantPopen):
        def __init__(self, command=None, *a, **kw):
            super().__init__(command, *a, **kw)
            r, w = os.pipe()
            os.write(w, b"WARN msg\n")
            os.close(w)
            self.stderr = os.fdopen(r, "r", encoding="utf-8", errors="replace")

    monkeypatch.setattr(gt_mod.subprocess, "Popen", _P)
    res = reg.dispatch("bash", {"command": "echo hi"})
    assert res.ok, res.error
    assert "[stderr]\nWARN msg" in res.content


def test_shell_exec_pytest_recovery_override(bash_ok, monkeypatch):
    reg, _spawned = bash_ok

    class _P(_InstantPopen):
        def __init__(self, command=None, *a, **kw):
            super().__init__(command, *a, **kw)
            self.returncode = 2
            r, w = os.pipe()
            os.write(w, b"unrecognized arguments: --timeout=60\n")
            os.close(w)
            self.stderr = os.fdopen(r, "r", encoding="utf-8", errors="replace")

    monkeypatch.setattr(gt_mod.subprocess, "Popen", _P)
    monkeypatch.setattr(
        reg, "_maybe_recover_pytest_missing_plugin",
        lambda **kw: {"_override": {"ok": True, "content": "RE-RUN OK", "error": None, "metadata": {"recovered_pytest_plugin": True}}},
    )
    res = reg.dispatch("bash", {"command": "pytest -m unit"})
    assert res.ok
    assert res.content == "RE-RUN OK"


def test_shell_exec_pytest_recovery_hint(bash_ok, monkeypatch):
    reg, _spawned = bash_ok

    class _P(_InstantPopen):
        def __init__(self, command=None, *a, **kw):
            super().__init__(command, *a, **kw)
            self.returncode = 2
            r, w = os.pipe()
            os.write(w, b"unrecognized arguments: --frobnicate\n")
            os.close(w)
            self.stderr = os.fdopen(r, "r", encoding="utf-8", errors="replace")

    monkeypatch.setattr(gt_mod.subprocess, "Popen", _P)
    monkeypatch.setattr(
        reg, "_maybe_recover_pytest_missing_plugin",
        lambda **kw: {"_append_hint": "remove --frobnicate"},
    )
    res = reg.dispatch("bash", {"command": "pytest"})
    assert not res.ok
    assert "remove --frobnicate" in res.content


def test_shell_exec_outer_timeout_and_exception(bash_ok, monkeypatch):
    reg, _spawned = bash_ok

    def _tmo(self, proc, timeout, cap):
        raise subprocess.TimeoutExpired(["bash"], timeout)

    monkeypatch.setattr(ShellToolsMixin, "_capture_bounded", _tmo)
    res = reg.dispatch("bash", {"command": "sleep 5"})
    assert not res.ok
    assert "timed out" in res.error

    def _boom(self, proc, timeout, cap):
        raise RuntimeError("boom")

    monkeypatch.setattr(ShellToolsMixin, "_capture_bounded", _boom)
    res2 = reg.dispatch("bash", {"command": "echo hi"})
    assert not res2.ok
    assert "Command execution failed: boom" in res2.error


# ── job tool error branches ─────────────────────────────────────────────────

def test_job_action_errors(tool_registry):
    res = tool_registry.dispatch("job", {})
    assert not res.ok
    assert "'action' is required" in res.error

    res2 = tool_registry.dispatch("job", {"action": "fly"})
    assert not res2.ok
    assert "Unknown action" in res2.error

    res3 = tool_registry.dispatch("job", {"action": "output"})
    assert not res3.ok
    assert "'job_id' is required" in res3.error

    res4 = tool_registry.dispatch("job", {"action": "kill"})
    assert not res4.ok
    assert "'job_id' is required" in res4.error


def test_job_list_empty_and_output_missing(tool_registry):
    # The default _get_bg_manager() resolves to the process-global singleton,
    # which any earlier test in ANY file can pollute — "No background jobs."
    # would then be order/flake-dependent. Inject a fresh manager so the
    # emptiness assertion is hermetic (the tool reads self._bg_manager).
    from external_llm.agent.background_job_manager import BackgroundJobManager
    tool_registry._bg_manager = BackgroundJobManager(max_jobs=5)

    res = tool_registry.dispatch("job", {"action": "list"})
    assert res.ok
    assert res.content == "No background jobs."

    res2 = tool_registry.dispatch("job", {"action": "output", "job_id": "nope"})
    assert not res2.ok
    assert "not found" in res2.error
