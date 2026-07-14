"""Unit tests for the missing-pytest-plugin auto-recovery in ShellToolsMixin.

Covers `_maybe_recover_pytest_missing_plugin` and its integration point in
`_tool_shell_exec`. pytest emits `unrecognized arguments:` for uninstalled
entry-point plugins (--timeout, --cov, ...); the recovery layer diagnoses,
asks the user, installs, and re-runs — or appends a hint when install is not
possible / declined.
"""

import external_llm.agent.tool_handlers.git_tools as gt_mod
from external_llm.agent.tool_handlers.git_tools import _PYTEST_CMD_RE, ShellToolsMixin
from external_llm.agent.tool_registry import ToolResult


class _FakeReg(ShellToolsMixin):
    """Minimal registry stub: only the collaborators the recovery touches."""

    def __init__(self, ask_answer="no"):
        self.repo_root = "/tmp"
        self.config = None
        self._ask_answer = ask_answer
        self.pip_called = None
        self.rerun_called = None

    def _tool_ask_user(self, args):
        return ToolResult(ok=True, content="", metadata={"answer": self._ask_answer})

    def _fake_subprocess_run(self, cmd, **kwargs):
        if "pip install" in cmd:
            self.pip_called = cmd
            return _RunResult(returncode=0, stdout="Successfully installed", stderr="")
        if "pytest" in cmd:
            self.rerun_called = cmd
            return _RunResult(returncode=0, stdout="1 passed in 0.5s", stderr="")
        raise AssertionError(f"unexpected subprocess call: {cmd}")


class _RunResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _PatchedRun:
    """Context manager that monkeypatches gt_mod._run_bounded_subprocess.

    Patches the bounded-subprocess helper (not raw subprocess.run/Popen) so the
    test is independent of the helper's internal implementation (run vs Popen).
    """

    def __init__(self, fake_fn):
        self._fake_fn = fake_fn
        self._orig = None

    def __enter__(self):
        self._orig = gt_mod._run_bounded_subprocess
        gt_mod._run_bounded_subprocess = self._fake_fn
        return self

    def __exit__(self, *exc):
        gt_mod._run_bounded_subprocess = self._orig


# ── pytest command detection ──────────────────────────────────────────────


def test_pytest_cmd_detection():
    """_PYTEST_CMD_RE matches pytest invocations, not lookalikes."""
    positive = [
        "python3 -m pytest tests/ --timeout=60",
        "python -m pytest -x",
        "pytest tests/unit/",
        "py.test foo",
    ]
    negative = [
        "echo 'pytest' | grep",   # not a head segment
        "python3 test_runner.py",  # not pytest
        "pip install pytest",      # pytest token but not -m pytest
    ]
    for cmd in positive:
        assert _PYTEST_CMD_RE.search(cmd), f"should match: {cmd}"
    for cmd in negative:
        assert not _PYTEST_CMD_RE.search(cmd), f"should NOT match: {cmd}"


# ── non-intervention guards ───────────────────────────────────────────────


def test_non_pytest_command_no_intervention():
    """A non-pytest command with 'unrecognized arguments' is left alone."""
    reg = _FakeReg()
    r = reg._maybe_recover_pytest_missing_plugin(
        command="git status --bogus",
        stderr="error: unrecognized arguments: --bogus",
        original_command="git status --bogus",
    )
    assert r is None


def test_pytest_without_unrecognized_arguments_no_intervention():
    """A normal pytest failure (no usage error) is not a recovery target."""
    reg = _FakeReg()
    r = reg._maybe_recover_pytest_missing_plugin(
        command="python3 -m pytest tests/",
        stderr="==== 1 failed in 1.2s ====",
        original_command="python3 -m pytest tests/",
    )
    assert r is None


# ── unmapped options → hint only ──────────────────────────────────────────


def test_unmapped_option_returns_append_hint():
    """Options not in _PYTEST_PLUGIN_OPTIONS get a removal hint, no ask_user."""
    reg = _FakeReg()
    reg._ask_answer = "SHOULD_NOT_ASK"  # would explode if ask_user fired
    r = reg._maybe_recover_pytest_missing_plugin(
        command="python3 -m pytest --frobnicate",
        stderr="pytest: error: unrecognized arguments: --frobnicate",
        original_command="python3 -m pytest --frobnicate",
    )
    assert r is not None
    assert "_append_hint" in r
    assert reg.pip_called is None
    assert reg.rerun_called is None


# ── user approval → install + rerun ───────────────────────────────────────


def test_user_approves_installs_and_reruns():
    """Approval triggers pip install then re-execution; result overrides original."""
    reg = _FakeReg(ask_answer="yes")
    with _PatchedRun(lambda *a, **kw: reg._fake_subprocess_run(a[0], **kw)):
        r = reg._maybe_recover_pytest_missing_plugin(
            command="python3 -m pytest tests/ --timeout=60",
            stderr="pytest: error: unrecognized arguments: --timeout=60",
            original_command="python3 -m pytest tests/ --timeout=60",
        )
    assert "pytest-timeout" in reg.pip_called
    assert "--timeout=60" in reg.rerun_called
    assert "_override" in r
    assert r["_override"]["ok"] is True
    assert "1 passed" in r["_override"]["content"]
    assert r["_override"]["metadata"]["recovered_pytest_plugin"] is True
    assert r["_override"]["metadata"]["installed_packages"] == ["pytest-timeout"]


def test_user_declines_appends_hint_no_install():
    """Decline → hint appended, no pip/rerun."""
    reg = _FakeReg(ask_answer="no")
    r = reg._maybe_recover_pytest_missing_plugin(
        command="python3 -m pytest --timeout=60",
        stderr="pytest: error: unrecognized arguments: --timeout=60",
        original_command="python3 -m pytest --timeout=60",
    )
    assert reg.pip_called is None
    assert reg.rerun_called is None
    assert "_append_hint" in r


def test_pip_install_failure_returns_override_error():
    """A failed pip install returns an _override error result."""
    reg = _FakeReg(ask_answer="yes")
    reg._fake_subprocess_run = lambda cmd, **kw: _RunResult(
        returncode=1, stdout="", stderr="permission denied"
    )
    with _PatchedRun(lambda *a, **kw: reg._fake_subprocess_run(a[0], **kw)):
        r = reg._maybe_recover_pytest_missing_plugin(
            command="python3 -m pytest --timeout=60",
            stderr="pytest: error: unrecognized arguments: --timeout=60",
            original_command="python3 -m pytest --timeout=60",
        )
    assert "_override" in r
    assert r["_override"]["ok"] is False
    assert "pip install failed" in r["_override"]["error"]


def test_ask_user_exception_falls_back_to_decline():
    """If ask_user raises, recovery degrades to a decline hint (no crash)."""
    reg = _FakeReg(ask_answer="yes")
    reg._tool_ask_user = lambda args: (_ for _ in ()).throw(RuntimeError("no checkpoint"))
    r = reg._maybe_recover_pytest_missing_plugin(
        command="python3 -m pytest --timeout=60",
        stderr="pytest: error: unrecognized arguments: --timeout=60",
        original_command="python3 -m pytest --timeout=60",
    )
    assert reg.pip_called is None  # did not install
    assert "_append_hint" in r
