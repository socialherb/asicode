"""Tests for scripts/git_commit_wrapper.py (pre-commit parallel-session race).

The gate's contract (see the module docstring):
  * transparent passthrough on success and on any non-race failure;
  * on "files were modified by this hook": compare ONLY the out-of-scope
    slice of the diff (files OUTSIDE this commit's staged set). If it
    changed, a parallel session wrote during the run → wait for the tree to
    settle and retry the gate ONCE (strictly bounded — never loops);
  * if the out-of-scope diff did NOT change, the failure is a genuine hook
    modification of its own staged files → report as-is, never retry.
"""

import importlib.util
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "git_commit_wrapper.py"
_spec = importlib.util.spec_from_file_location("git_commit_wrapper", _SCRIPT)
w = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(w)  # type: ignore[union-attr]

DIFF = ("git", "diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules")
TOPLEVEL = ("git", "rev-parse", "--show-toplevel")
STAGED = ("git", "diff", "--cached", "--name-only", "-z")
# _precommit_argv() resolves the binary (shutil.which, then -m pre_commit
# fallback, then --color=always on a tty) — key responses on the exact argv.
PRECOMMIT = tuple(w._precommit_argv())


def _ok(out: bytes = b"", err: bytes = b"", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def _fail(out: bytes = b"", err: bytes = b"", rc: int = 1) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


class FakeGit:
    """Scriptable runner: exact-args responses, records (args, cwd)."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.responses: dict[tuple, object] = {}

    def __call__(self, args: list[str], cwd: str) -> subprocess.CompletedProcess:
        self.calls.append((list(args), str(cwd)))
        key = tuple(args)
        resp = self.responses.get(key)
        if resp is None:
            return _fail(err=b"fake runner: no response for %r" % (tuple(args),))
        if isinstance(resp, subprocess.CompletedProcess):
            return resp
        return resp(list(args), str(cwd))

    def pre_commit_calls(self) -> int:
        return len(
            [c for c in self.calls if any(
                "pre_commit" in a or "pre-commit" in a for a in c[0])]
        )


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=r, check=True)
    (r / "a.txt").write_text("x1\n")
    (r / "b.txt").write_text("y1\n")
    subprocess.run(["git", "add", "a.txt", "b.txt"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=r, check=True)
    return r


# --- marker detection --------------------------------------------------------
def test_race_marker_detected():
    assert w._is_race_failure("- files were modified by this hook\n")
    assert w._is_race_failure("x\n- files were modified by this hook\ny")
    assert not w._is_race_failure("F821 undefined name 'z'")


def test_precommit_argv_resolves_binary():
    argv = w._precommit_argv()
    assert argv[-1] == "run"
    assert len(argv) in (2, 4)  # [pre, run] or [python, -m, pre_commit, run]


# --- scope helpers ------------------------------------------------------------
def test_scope_pathspec_empty_and_excludes():
    assert w._scope_pathspec(()) == []
    assert w._scope_pathspec(("a.py", "b/c.py")) == [
        "--", ":(top)",
        ":(exclude,top,literal)a.py",
        ":(exclude,top,literal)b/c.py",
    ]


def test_staged_paths_real_repo(repo):
    (repo / "a.txt").write_text("x2\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    assert w._staged_paths(str(repo), w._default_runner) == ("a.txt",)
    subprocess.run(["git", "commit", "-qm", "t"], cwd=repo, check=True)
    assert w._staged_paths(str(repo), w._default_runner) == ()


def test_staged_paths_handles_non_ascii_names(repo):
    # -z: --name-only alone C-quotes non-ASCII paths and would drop them.
    (repo / "한글.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "한글.py"], cwd=repo, check=True)
    assert w._staged_paths(str(repo), w._default_runner) == ("한글.py",)


# --- snapshot mirrors pre-commit's signal ------------------------------------
def test_snapshot_tracks_worktree_and_index_changes(repo):
    (repo / "a.txt").write_text("x2\n")
    rc, s_worktree = w._snapshot(str(repo), w._default_runner)
    assert rc == 0 and b"x2" in s_worktree and s_worktree != b""
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    rc, s_after_stage = w._snapshot(str(repo), w._default_runner)
    assert rc == 0 and s_after_stage == b""  # staged content drops out


def test_snapshot_scoped_excludes_staged_paths(repo):
    (repo / "a.txt").write_text("x2\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    (repo / "b.txt").write_text("y2\n")  # outside the staged set
    rc, s = w._snapshot(str(repo), w._default_runner, exclude=("a.txt",))
    assert rc == 0 and b"b.txt" in s and b"a.txt" not in s


# --- settle ------------------------------------------------------------------
def test_wait_for_settle_returns_when_stable(repo):
    assert w._wait_for_settle(
        str(repo), w._default_runner, timeout=5.0, interval=0.01, stable_samples=2
    )


def test_wait_for_settle_times_out_while_tree_keeps_changing(repo):
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            (repo / "a.txt").write_text(f"churn{i}\n")
            i += 1
            time.sleep(0.02)

    t = threading.Thread(target=churn)
    t.start()
    try:
        assert not w._wait_for_settle(
            str(repo), w._default_runner, timeout=0.3, interval=0.02, stable_samples=2
        )
    finally:
        stop.set()
        t.join()


# --- run_hooks(): success passthrough ----------------------------------------
def test_run_hooks_success_passthrough_no_retry(tmp_path, capsys):
    top = str(tmp_path / "repo")
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(b"")
    fg.responses[DIFF] = _ok(b"")
    fg.responses[PRECOMMIT] = _ok(b"F821 baseline-diff........................Passed\n")
    rc = w.run_hooks([], cwd=top, runner=fg)
    assert rc == 0
    assert fg.pre_commit_calls() == 1
    assert "Passed" in capsys.readouterr().out


def test_run_hooks_not_a_repo(tmp_path):
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _fail(rc=128, err=b"fatal: not a git repository\n")
    assert w.run_hooks([], cwd=str(tmp_path), runner=fg) == 1
    assert fg.pre_commit_calls() == 0


def test_run_hooks_ordinary_failure_passthrough_no_retry(tmp_path, capsys):
    top = str(tmp_path / "repo")
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(b"")
    fg.responses[DIFF] = _ok(b"")
    fg.responses[PRECOMMIT] = _fail(err=b"F821 undefined name 'z'\n")
    rc = w.run_hooks([], cwd=top, runner=fg)
    assert rc == 1
    assert fg.pre_commit_calls() == 1  # no retry on non-race failures
    assert "F821" in capsys.readouterr().err


# --- run_hooks(): race retry contract ----------------------------------------
SCOPE = ("a.py",)
SCOPED_DIFF = (
    "git", "diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules",
    *w._scope_pathspec(SCOPE),
)
RACE_ERR = b"- files were modified by this hook\n"


def _race_fake(top, staged=SCOPE, diff_after=b"CHANGED\n"):
    """FakeGit wired for the race path: staged scope, before==clean,
    after==changed, settle stable on the changed value."""
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(("".join(staged) + "\0").encode())
    state = {"n": 0}

    def scoped_diff_resp(args, cwd):
        state["n"] += 1
        return _ok(diff_after if state["n"] >= 2 else b"")

    fg.responses[SCOPED_DIFF] = scoped_diff_resp
    return fg, state


def test_run_hooks_race_retries_once_then_success(tmp_path):
    """Out-of-scope change during the run → exactly one retry → success."""
    top = str(tmp_path / "repo")
    fg, _ = _race_fake(top)
    commit_rc = {"n": 0}

    def pre_resp(args, cwd):
        commit_rc["n"] += 1
        if commit_rc["n"] == 1:
            return _fail(err=RACE_ERR)
        return _ok(b"Passed\n")

    fg.responses[PRECOMMIT] = pre_resp
    rc = w.run_hooks(
        [],
        cwd=top,
        runner=fg,
        settle_timeout=5.0,
        settle_interval=0.01,
        settle_stable_samples=2,
    )
    assert rc == 0
    assert fg.pre_commit_calls() == 2


def test_run_cmd_timeout_aborts_hung_command():
    """A command that outlives its cap must raise TimeoutExpired (B1)."""
    with pytest.raises(subprocess.TimeoutExpired):
        w._run_cmd(["sleep", "5"], ".", timeout=0.2)


def test_run_hooks_precommit_timeout_aborts_with_warning(tmp_path, capsys):
    """A hung pre-commit must abort the gate with a warning, not block forever."""
    top = str(tmp_path / "repo")
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(b"")
    fg.responses[DIFF] = _ok(b"")

    def hung(args, cwd):
        raise subprocess.TimeoutExpired(cmd=args, timeout=w._CMD_TIMEOUT)

    fg.responses[PRECOMMIT] = hung
    rc = w.run_hooks([], cwd=top, runner=fg)
    assert rc == 1
    assert fg.pre_commit_calls() == 1  # timeout is terminal — no race retry
    assert "did not finish within" in capsys.readouterr().err


def test_run_hooks_precommit_timeout_on_race_retry_aborts(tmp_path, capsys):
    """Race path: the retry run hanging is also bounded and reported."""
    top = str(tmp_path / "repo")
    fg, _ = _race_fake(top)
    commit_rc = {"n": 0}

    def pre_resp(args, cwd):
        if commit_rc["n"] == 0:
            commit_rc["n"] += 1
            return _fail(err=RACE_ERR)
        raise subprocess.TimeoutExpired(cmd=args, timeout=w._CMD_TIMEOUT)

    fg.responses[PRECOMMIT] = pre_resp
    rc = w.run_hooks(
        [],
        cwd=top,
        runner=fg,
        settle_timeout=5.0,
        settle_interval=0.01,
        settle_stable_samples=2,
    )
    assert rc == 1
    assert fg.pre_commit_calls() == 2
    assert "did not finish within" in capsys.readouterr().err


def test_run_hooks_race_genuine_hook_modification_no_retry(tmp_path, capsys):
    """Marker but NO out-of-scope change → genuine hook edit of its own
    staged files → report as-is, never retry."""
    top = str(tmp_path / "repo")
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(b"a.py\0")
    fg.responses[SCOPED_DIFF] = _ok(b"")  # out-of-scope diff unchanged
    fg.responses[PRECOMMIT] = _fail(err=RACE_ERR)
    rc = w.run_hooks([], cwd=top, runner=fg)
    assert rc == 1
    assert fg.pre_commit_calls() == 1
    err = capsys.readouterr().err
    assert "genuine hook modification" in err
    assert "files were modified by this hook" in err  # passthrough preserved


def test_run_hooks_race_retry_also_fails_reports_second_failure(tmp_path, capsys):
    top = str(tmp_path / "repo")
    fg, _ = _race_fake(top)
    fg.responses[PRECOMMIT] = _fail(err=RACE_ERR)
    rc = w.run_hooks(
        [],
        cwd=top,
        runner=fg,
        settle_timeout=5.0,
        settle_interval=0.01,
        settle_stable_samples=2,
    )
    assert rc == 1  # bounded: exactly one retry, then report
    assert fg.pre_commit_calls() == 2
    err = capsys.readouterr().err
    assert "retrying once" in err
    assert "failed again after the retry" in err


def test_run_hooks_race_conflict_notice_counts_as_evidence(tmp_path):
    """Out-of-scope diff UNCHANGED but a stash/pop conflict notice present:
    pre-commit's rollback erased the parallel write before the comparison —
    the notice alone is race evidence → retry once."""
    top = str(tmp_path / "repo")
    fg = FakeGit()
    fg.responses[TOPLEVEL] = _ok(top.encode() + b"\n")
    fg.responses[STAGED] = _ok(b"a.py\0")
    fg.responses[SCOPED_DIFF] = _ok(b"")  # diff unchanged (rollback restored it)
    commit_rc = {"n": 0}

    def pre_resp(args, cwd):
        commit_rc["n"] += 1
        if commit_rc["n"] == 1:
            return _fail(
                err=RACE_ERR
                + b"[WARNING] Stashed changes conflicted with hook auto-fixes... "
                + b"Rolling back fixes...\n"
            )
        return _ok(b"Passed\n")

    fg.responses[PRECOMMIT] = pre_resp
    rc = w.run_hooks(
        [],
        cwd=top,
        runner=fg,
        settle_timeout=5.0,
        settle_interval=0.01,
        settle_stable_samples=2,
    )
    assert rc == 0
    assert fg.pre_commit_calls() == 2


def test_run_hooks_race_surfaces_conflict_warning(tmp_path, capsys):
    """pre-commit's stash/pop conflict notice (rollback clobbered a
    concurrent write) must stay visible even though run 1 is retried."""
    top = str(tmp_path / "repo")
    fg, _ = _race_fake(top)
    commit_rc = {"n": 0}

    def pre_resp(args, cwd):
        commit_rc["n"] += 1
        if commit_rc["n"] == 1:
            return _fail(
                err=RACE_ERR
                + b"[WARNING] Stashed changes conflicted with hook auto-fixes... "
                + b"Rolling back fixes...\n"
            )
        return _ok(b"Passed\n")

    fg.responses[PRECOMMIT] = pre_resp
    rc = w.run_hooks(
        [],
        cwd=top,
        runner=fg,
        settle_timeout=5.0,
        settle_interval=0.01,
        settle_stable_samples=2,
    )
    assert rc == 0
    assert fg.pre_commit_calls() == 2
    err = capsys.readouterr().err
    assert "conflicted with hook auto-fixes" in err  # surfaced, not swallowed


# --- install/uninstall hook ---------------------------------------------------
def test_install_hook_writes_executable_gate(repo, monkeypatch, capsys):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "git_commit_wrapper.py").write_text("")
    monkeypatch.chdir(repo)
    assert w._install_hook() == 0
    hook = repo / ".git" / "hooks" / "pre-commit"
    content = hook.read_text(encoding="utf-8")
    assert "--run-hooks" in content
    assert str(repo / "scripts" / "git_commit_wrapper.py") in content
    assert os.access(hook, os.X_OK)
    assert w._install_hook() == 0  # idempotent
    assert "installed" in capsys.readouterr().out


def test_install_hook_replaces_precommit_generated_hook(repo, monkeypatch):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "git_commit_wrapper.py").write_text("")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "# File generated by pre-commit: https://pre-commit.com\n"
        "exec pre-commit\n"
    )
    monkeypatch.chdir(repo)
    assert w._install_hook() == 0  # regenerable standard hook → safe to replace
    assert "--run-hooks" in hook.read_text(encoding="utf-8")


def test_install_hook_refuses_foreign_hook_and_force_overwrites(repo, monkeypatch):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "git_commit_wrapper.py").write_text("")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexec pre-commit\n")
    monkeypatch.chdir(repo)
    assert w._install_hook() == 1  # refuses to clobber
    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\nexec pre-commit\n"
    assert w._install_hook(force=True) == 0  # --force overwrites
    assert "--run-hooks" in hook.read_text(encoding="utf-8")


def test_install_hook_requires_script_in_repo(repo, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    assert w._install_hook() == 1
    assert "not found" in capsys.readouterr().err


def test_installed_hook_shell_quotes_paths_and_pins_interpreter(repo, monkeypatch):
    """A repo path containing a space must not break the generated hook, and
    the pinned INSTALL_PYTHON (with a python3 fallback) is what keeps the hook
    working from GUIs/IDEs that run with a minimal PATH — mirrors pre-commit's
    own hook.

    The hook is EXECUTED, not just parsed: an unquoted path with a space is
    still valid bash (`WRAPPER=/a/dir with space/...` parses as the
    `VAR=value command args` form), so `bash -n` passes and only a real run
    catches it."""
    weird = repo / "dir with space"
    (weird / "scripts").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=weird, check=True)
    (weird / "scripts" / "git_commit_wrapper.py").write_text(
        "print('WRAPPER_REACHED')\n", encoding="utf-8"
    )
    monkeypatch.chdir(weird)
    assert w._install_hook() == 0
    hook = weird / ".git" / "hooks" / "pre-commit"
    content = hook.read_text(encoding="utf-8")
    assert "INSTALL_PYTHON=" in content and "exec python3" in content
    cp = subprocess.run([str(hook)], capture_output=True, text=True, check=False)
    assert "WRAPPER_REACHED" in cp.stdout, (cp.stdout, cp.stderr)


def test_precommit_argv_falls_back_to_module_when_binary_absent(monkeypatch):
    """A venv/pipx pre-commit stays reachable when the hook runs with a
    minimal PATH where the console script is not visible."""
    monkeypatch.setattr(w.shutil, "which", lambda _n: None)
    monkeypatch.setattr(w, "_module_available", lambda _n: True)
    argv = w._precommit_argv()
    assert argv[:3] == [w.sys.executable, "-m", "pre_commit"]
    assert argv[3] == "run"


def test_precommit_argv_last_resort_is_the_bare_name(monkeypatch):
    """Neither binary nor module: return the bare name so the caller's OSError
    path prints the install hint rather than a confusing ModuleNotFoundError."""
    monkeypatch.setattr(w.shutil, "which", lambda _n: None)
    monkeypatch.setattr(w, "_module_available", lambda _n: False)
    assert w._precommit_argv() == ["pre-commit", "run"]


def test_module_available_reports_real_importability():
    assert w._module_available("os") is True
    assert w._module_available("definitely_not_a_module_xyz") is False


def test_snapshot_returncode_is_part_of_identity(repo):
    """A failed `git diff` — e.g. a parallel session holding .git/index.lock —
    must not compare equal to a clean "no changes" snapshot, which would read
    as "no concurrent writer" and suppress the retry."""
    ok = w._snapshot(str(repo), w._default_runner)
    assert ok == (0, b"")
    assert w._snapshot(str(repo), lambda a, c: _fail(rc=128)) != ok


def test_race_marker_survives_ansi_colour():
    """_precommit_argv() adds --color=always on a tty, so the marker arrives
    wrapped in ANSI codes. Pins that they wrap the phrase rather than split it
    (verified against pre-commit 4.6.0 output)."""
    assert w._is_race_failure("\x1b[2m- files were modified by this hook\x1b[m")


def test_uninstall_hook_removes_only_ours(repo, monkeypatch):
    (repo / "scripts").mkdir()
    (repo / "scripts" / "git_commit_wrapper.py").write_text("")
    hook = repo / ".git" / "hooks" / "pre-commit"
    monkeypatch.chdir(repo)
    assert w._uninstall_hook() == 0  # not installed → no-op
    assert w._install_hook() == 0
    assert hook.exists()
    assert w._uninstall_hook() == 0
    assert not hook.exists()
    hook.write_text("#!/bin/sh\nexec pre-commit\n")
    assert w._uninstall_hook() == 1  # foreign → refuse
    assert hook.exists()
