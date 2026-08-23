#!/usr/bin/env python3
"""Race-resilient pre-commit gate — closes the parallel-session write race.

Background
----------
pre-commit's ``commands/run.py`` captures ``git diff --no-ext-diff
--no-textconv --ignore-submodules`` when the hook run starts and re-compares
it after every hook. Any tracked-file write by a PARALLEL session during the
run changes the diff, so the run fails with the false positive
"files were modified by this hook". The per-file incremental hooks (commit
0f9747cb) shrank that window to a few hundred milliseconds; this gate closes
it at the hook layer:

  1. run the real pre-commit gate (``pre-commit run``);
  2. if it fails with the files-modified marker AND a tracked file OUTSIDE
     this commit's staged set changed during the run, a concurrent writer is
     the cause: wait for the tree to settle (bounded, default 15 s) and retry
     the gate ONCE;
  3. otherwise the failure is genuine — a hook really did modify this
     commit's files: report it as-is. Never retry without evidence, and the
     retry is capped at one attempt, so this never loops.

Why the evidence is scoped to files outside the staged set
----------------------------------------------------------
The obvious signal — "the whole-tree diff changed during the run" — does NOT
discriminate, because a hook that legitimately rewrites a staged file changes
that same diff. Measured: a formatter hook in a single-process repo with no
concurrent writer at all yields marker=True AND before != after, so a
whole-tree comparison takes the retry branch every time and the "genuine
failure" branch is unreachable for the very case it exists to catch.

pre-commit hands each hook only the commit's staged files, so a hook's own
edits land INSIDE the staged set, while a parallel session writing an
unrelated tracked file lands OUTSIDE it. Comparing only the out-of-scope
slice of the diff (``:(exclude,top,literal)`` pathspecs) separates the two.

Residual cases are bounded and fail toward one extra gate run, never toward
a false pass: a hook that writes outside its own file list retries once and
fails identically, and a parallel write to a file this commit stages is
reported as-is (that content is genuinely in flux, so auto-retrying would be
wrong). Note also that pre-commit's stash/pop roundtrip for pre-existing
unstaged changes does NOT by itself trip the marker — measured with pure
reader hooks: stash, run, restore, zero marker occurrences. When a parallel
write lands on a file that ALSO has pre-existing unstaged changes, the pop
conflicts and pre-commit's rollback restores the pre-run state — erasing the
write before the out-of-scope comparison sees it. The stash/pop CONFLICT
notice ("Rolling back") is therefore treated as race evidence on its own
(measured: dirty tree + concurrent write to the same file → marker +
"Rolling back" + diff back to before).

This is installed as ``.git/hooks/pre-commit`` so EVERY commit in the repo
goes through it automatically (repo-local, no per-user setup)::

    python3 scripts/git_commit_wrapper.py --install-hook
    python3 scripts/git_commit_wrapper.py --uninstall-hook

NOTE: ``pre-commit install`` overwrites ``.git/hooks/pre-commit`` — re-run
``--install-hook`` after re-installing pre-commit's own hook.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence

RACE_MARKER = "files were modified by this hook"
SETTLE_TIMEOUT = 15.0
SETTLE_INTERVAL = 0.2
SETTLE_STABLE_SAMPLES = 2
# Beyond this many staged files, encoding them all as pathspec excludes risks
# the argv limit — fall back to a whole-tree comparison. That is the strictly
# conservative direction: at worst one extra gate run, never a false pass.
MAX_SCOPE_EXCLUDES = 2000

# Hard cap per subprocess: a hung git/pre-commit (index.lock contention, network
# FS, deadlocked hook) must abort the commit LOUDLY rather than block every
# commit in the repo — this gate runs on EVERY commit.
_CMD_TIMEOUT = 300.0

# pre-commit's own change signal (commands/run.py _get_diff) — mirror exactly.
_DIFF_ARGS = ["diff", "--no-ext-diff", "--no-textconv", "--ignore-submodules"]
# pre-commit's stash/pop rollback notices. When these appear, pre-commit's
# restore clobbered a concurrent write to a file that also had pre-existing
# unstaged changes — real data loss the user must see, so that output is not
# suppressed as transient race noise.
_CONFLICT_MARKERS = ("conflicted", "Rolling back")
_PREFIX = "  [commit-wrapper] "
_HOOK_MARKER = "git_commit_wrapper.py"
# INSTALL_PYTHON mirrors pre-commit's own generated hook: git hooks can run
# from IDEs/GUIs with a minimal PATH where `python3` is absent, so pin the
# installing interpreter and keep `python3` only as the fallback.
_HOOK_TEMPLATE = (
    "#!/usr/bin/env bash\n"
    "# Race-resilient pre-commit gate — installed by scripts/git_commit_wrapper.py\n"
    "# --install-hook. Retries ONCE when a PARALLEL session writes a tracked file\n"
    "# OUTSIDE this commit's staged set during the hook run and trips pre-commit's\n"
    '# "files were modified by this hook" false positive. NOTE: `pre-commit install`\n'
    "# overwrites this file — re-run `--install-hook` afterwards.\n"
    "INSTALL_PYTHON={python}\n"
    "WRAPPER={wrapper}\n"
    'if [ -x "$INSTALL_PYTHON" ]; then\n'
    '    exec "$INSTALL_PYTHON" "$WRAPPER" --run-hooks "$@"\n'
    "fi\n"
    'exec python3 "$WRAPPER" --run-hooks "$@"\n'
)

Runner = Callable[[list[str], str], subprocess.CompletedProcess]


def _run_cmd(argv: list[str], cwd: str, timeout: float | None = _CMD_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a command, capturing output. OSError guard: required by the repo's
    unguarded-subprocess gate (a missing binary must not crash silently).

    ``timeout`` (default ``_CMD_TIMEOUT``) caps the wait so a hung git or
    pre-commit — index.lock contention, a network FS, a deadlocked hook — raises
    ``subprocess.TimeoutExpired`` (loud, non-zero) instead of blocking every
    commit in the repo indefinitely. ``None`` disables the cap.
    """
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        # Guard required by the repo's unguarded-subprocess gate (a missing
        # binary must not crash silently). Enrich so callers see WHICH binary
        # failed to start — a bare FileNotFoundError hides argv[0].
        raise OSError(f"{argv[0]!r} failed to start: {exc}") from exc


def _default_runner(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Raw command runner (no prefix) — git callers pass ["git", ...] explicitly."""
    return _run_cmd(list(args), cwd)


def _passthrough(cp: subprocess.CompletedProcess) -> None:
    if cp.stdout:
        sys.stdout.buffer.write(cp.stdout)
    if cp.stderr:
        sys.stderr.buffer.write(cp.stderr)
    sys.stdout.flush()
    sys.stderr.flush()


def _output_text(cp: subprocess.CompletedProcess) -> str:
    return (cp.stdout + cp.stderr).decode("utf-8", errors="replace")


def _warn(msg: str) -> None:
    sys.stderr.write(_PREFIX + msg + "\n")
    sys.stderr.flush()


def _toplevel(cwd: str, runner: Runner) -> str | None:
    cp = runner(["git", "rev-parse", "--show-toplevel"], cwd)
    if cp.returncode != 0:
        return None
    return cp.stdout.decode("utf-8", errors="replace").strip()


def _staged_paths(cwd: str, runner: Runner) -> tuple[str, ...]:
    """Repo-relative paths in the index — exactly the files pre-commit hands
    to its hooks, i.e. the scope a hook's own edits are confined to.

    ``-z`` is mandatory: ``--name-only`` C-quotes non-ASCII paths, which would
    silently drop Korean filenames from the scope and misread a hook's own
    edit as a concurrent write.

    Returns an empty tuple when the scope is unusable (git failed, nothing is
    staged, or too many paths to encode as pathspecs); callers then fall back
    to comparing the whole tree.
    """
    cp = runner(["git", "diff", "--cached", "--name-only", "-z"], cwd)
    if cp.returncode != 0:
        return ()
    raw = cp.stdout.decode("utf-8", errors="replace")
    paths = tuple(p for p in raw.split("\0") if p)
    if len(paths) > MAX_SCOPE_EXCLUDES:
        return ()
    return paths


def _scope_pathspec(exclude: Sequence[str]) -> list[str]:
    """Pathspecs selecting everything EXCEPT ``exclude`` (repo-root relative).

    ``top`` makes the paths root-relative regardless of cwd; ``literal`` stops
    git from treating ``*``/``?``/``[`` in a filename as a glob.
    """
    if not exclude:
        return []
    return ["--", ":(top)"] + [":(exclude,top,literal)" + p for p in exclude]


def _snapshot(cwd: str, runner: Runner, exclude: Sequence[str] = ()) -> tuple[int, bytes]:
    """Working-tree diff (pre-commit's own signal, see _DIFF_ARGS), restricted
    to files outside ``exclude``.

    The returncode is part of the snapshot identity so that a failed ``git
    diff`` — e.g. a parallel session holding ``.git/index.lock`` — is not
    silently indistinguishable from "no changes at all".
    """
    cp = runner(["git", *_DIFF_ARGS, *_scope_pathspec(exclude)], cwd)
    return cp.returncode, cp.stdout


def _is_race_failure(text: str) -> bool:
    return RACE_MARKER in text


def _has_conflict_notice(text: str) -> bool:
    return any(m in text for m in _CONFLICT_MARKERS)


def _wait_for_settle(
    cwd: str,
    runner: Runner,
    timeout: float,
    interval: float,
    stable_samples: int,
    exclude: Sequence[str] = (),
) -> bool:
    """Wait until the out-of-scope diff is stable across consecutive samples.

    Scoped to ``exclude`` for the same reason as the evidence check: we are
    waiting for the PARALLEL writer to stop, not for our own hooks' edits.
    """
    deadline = time.monotonic() + timeout
    prev: tuple[int, bytes] | None = None
    stable = 0
    while time.monotonic() < deadline:
        cur = _snapshot(cwd, runner, exclude)
        if cur == prev:
            stable += 1
            if stable >= stable_samples:
                return True
        else:
            stable = 0
        prev = cur
        time.sleep(interval)
    return False


def _module_available(name: str) -> bool:
    """Whether ``name`` is importable by THIS interpreter. Split out so the
    resolution branches stay testable: a pipx-installed pre-commit lives in
    its own venv and is deliberately NOT importable here, so the fallback
    branch would otherwise never be exercised on a developer machine."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _precommit_argv() -> list[str]:
    """Resolve the pre-commit CLI.

    Prefer the PATH binary, then ``-m pre_commit`` in this interpreter — that
    fallback keeps a venv/pipx install reachable when the hook runs with a
    minimal PATH (pre-commit's own generated hook pins INSTALL_PYTHON for the
    same reason). If neither resolves, return the bare name so the caller's
    OSError path prints the install hint.
    """
    pre = shutil.which("pre-commit")
    if pre:
        argv = [pre, "run"]
    elif _module_available("pre_commit"):
        argv = [sys.executable, "-m", "pre_commit", "run"]
    else:
        return ["pre-commit", "run"]
    if sys.stdout.isatty():
        # We capture through a pipe, so pre-commit would auto-disable colour;
        # force it back on when our own stdout is a terminal. Verified: the
        # ANSI wrappers do not break RACE_MARKER detection.
        argv.append("--color=always")
    return argv


def run_hooks(
    args: list[str],
    cwd: str | None = None,
    runner: Runner | None = None,
    settle_timeout: float = SETTLE_TIMEOUT,
    settle_interval: float = SETTLE_INTERVAL,
    settle_stable_samples: int = SETTLE_STABLE_SAMPLES,
) -> int:
    """Run the pre-commit gate with one bounded retry on the write race.

    Called by the installed ``.git/hooks/pre-commit`` (``--run-hooks``).
    ``args`` are git's hook arguments — ignored (the pre-commit hook receives
    none).
    """
    cwd = os.getcwd() if cwd is None else str(cwd)
    runner = runner or _default_runner
    top = _toplevel(cwd, runner)
    if top is None:
        _warn("not inside a git repository")
        return 1
    cmd = _precommit_argv()
    # The index cannot change while the hook runs (hooks cannot stage), so the
    # scope is stable for the whole call.
    scope = _staged_paths(cwd, runner)
    before = _snapshot(cwd, runner, scope)
    try:
        cp = runner(cmd, cwd)
    except subprocess.TimeoutExpired:
        _warn(
            f"pre-commit did not finish within {_CMD_TIMEOUT:.0f}s — a hook is hung "
            "(deadlock or network fetch); commit aborted. Investigate before retrying."
        )
        return 1
    except OSError:
        _warn("pre-commit not found on PATH — install it (`pipx install pre-commit`)")
        return 1
    if cp.returncode == 0:
        _passthrough(cp)
        return 0
    text = _output_text(cp)
    if _is_race_failure(text):
        after = _snapshot(cwd, runner, scope)
        # A stash/pop CONFLICT notice counts as race evidence even when the
        # out-of-scope diff looks unchanged: pre-commit's rollback restores
        # the pre-run state, erasing the parallel write before this
        # comparison (measured: dirty tree + concurrent write to the same
        # file → marker + "Rolling back" + diff back to before).
        if after == before and not _has_conflict_notice(text):
            _warn(
                f"pre-commit reported {RACE_MARKER!r} and nothing outside this "
                "commit's staged files changed during the run — this is a genuine "
                "hook modification, not a parallel-session race. Reporting as-is "
                "(`git add` the hook's changes and commit again)."
            )
            _passthrough(cp)
            return cp.returncode
        if _has_conflict_notice(text):
            # Data loss, not transient noise — always surface it.
            _passthrough(cp)
        _warn(
            f"pre-commit reported {RACE_MARKER!r} and a tracked file OUTSIDE this "
            "commit's staged set changed during the run (or pre-commit's "
            "stash/pop rollback notice appeared — a concurrent write was rolled "
            "back). A PARALLEL session is writing. Waiting for the tree to "
            "settle, then retrying once (strictly bounded — never loops)."
        )
        settled = _wait_for_settle(
            cwd,
            runner,
            settle_timeout,
            settle_interval,
            settle_stable_samples,
            scope,
        )
        if not settled:
            _warn(f"tree did not settle within {settle_timeout:.0f}s; retrying once anyway.")
        try:
            cp = runner(cmd, cwd)
        except subprocess.TimeoutExpired:
            _warn(
                f"pre-commit did not finish within {_CMD_TIMEOUT:.0f}s — a hook is "
                "hung (deadlock or network fetch); commit aborted. Investigate before "
                "retrying."
            )
            return 1
        except OSError:
            _warn("pre-commit not found on PATH — install it (`pipx install pre-commit`)")
            return 1
        _passthrough(cp)
        if cp.returncode != 0:
            _warn(
                "pre-commit failed again after the retry. If another session is "
                "still writing, re-run the commit once it finishes; if a hook "
                "genuinely modified files, `git add` them and commit again."
            )
        return cp.returncode
    _passthrough(cp)
    return cp.returncode


def _top_for_cli(runner: Runner) -> str | None:
    top = _toplevel(os.getcwd(), runner)
    if top is None:
        print("commit-wrapper: not inside a git repository", file=sys.stderr)
    return top


def _hook_path(top: str, runner: Runner) -> str | None:
    # --git-path honours core.hooksPath (verified on git 2.39): installing into
    # a hardcoded .git/hooks would be silently inert when it is set.
    cp = runner(["git", "rev-parse", "--git-path", "hooks/pre-commit"], top)
    if cp.returncode != 0:
        return None
    p = cp.stdout.decode("utf-8", errors="replace").strip()
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.join(top, p)
    return os.path.abspath(p)


def _install_hook(runner: Runner | None = None, force: bool = False) -> int:
    runner = runner or _default_runner
    top = _top_for_cli(runner)
    if top is None:
        return 1
    script_in_top = os.path.join(top, "scripts", "git_commit_wrapper.py")
    if not os.path.exists(script_in_top):
        print(
            f"commit-wrapper: error: {script_in_top} not found",
            file=sys.stderr,
        )
        return 1
    if shutil.which("pre-commit") is None and importlib.util.find_spec("pre_commit") is None:
        print(
            "commit-wrapper: warning: pre-commit not found on PATH — the gate "
            "will fail with a clear error until it is installed",
            file=sys.stderr,
        )
    hook_path = _hook_path(top, runner)
    if hook_path is None:
        print("commit-wrapper: error: cannot resolve hooks/pre-commit", file=sys.stderr)
        return 1
    if os.path.exists(hook_path):
        with open(hook_path, encoding="utf-8", errors="replace") as f:
            existing = f.read()
        generated = "generated by pre-commit" in existing
        if _HOOK_MARKER not in existing and not generated:
            if not force:
                print(
                    f"commit-wrapper: error: {hook_path} is neither this wrapper's "
                    "hook nor a pre-commit-generated hook — refusing to overwrite "
                    "(use --force)",
                    file=sys.stderr,
                )
                return 1
            print(
                f"commit-wrapper: warning: overwriting foreign hook at {hook_path} (--force)",
                file=sys.stderr,
            )
    content = _HOOK_TEMPLATE.format(
        python=shlex.quote(sys.executable),
        wrapper=shlex.quote(script_in_top),
    )
    with open(hook_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(hook_path, 0o755)
    print(f"commit-wrapper: installed race-resilient pre-commit gate at {hook_path}")
    print(
        "commit-wrapper: every `git commit` in this repo now retries once on the "
        "parallel-session race. NOTE: `pre-commit install` overwrites this hook — "
        "re-run --install-hook afterwards. Uninstall: --uninstall-hook"
    )
    return 0


def _uninstall_hook(runner: Runner | None = None) -> int:
    runner = runner or _default_runner
    top = _top_for_cli(runner)
    if top is None:
        return 1
    hook_path = _hook_path(top, runner)
    if hook_path is None:
        print("commit-wrapper: error: cannot resolve hooks/pre-commit", file=sys.stderr)
        return 1
    if not os.path.exists(hook_path):
        print("commit-wrapper: hook not installed")
        return 0
    with open(hook_path, encoding="utf-8", errors="replace") as f:
        existing = f.read()
    if _HOOK_MARKER not in existing:
        print(
            f"commit-wrapper: error: {hook_path} is not this wrapper's hook — not touching it",
            file=sys.stderr,
        )
        return 1
    os.remove(hook_path)
    print(f"commit-wrapper: removed {hook_path} — run `pre-commit install` to restore the standard pre-commit gate")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    doc = (__doc__ or "").strip()
    if not argv:
        print(doc)
        return 2
    cmd = argv[0]
    if cmd == "--run-hooks":
        return run_hooks(argv[1:])
    if cmd == "--install-hook":
        return _install_hook(force="--force" in argv[1:])
    if cmd == "--uninstall-hook":
        return _uninstall_hook()
    if cmd in ("--help", "-h"):
        print(doc)
        return 0
    print(f"commit-wrapper: unknown option {cmd!r} (see --help)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
