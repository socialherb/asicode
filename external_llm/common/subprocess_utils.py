"""Bounded subprocess execution with a mandatory timeout + process-group cleanup.

Single source of truth for :func:`run_bounded_subprocess`. Previously this
helper was duplicated (and at risk of drift) in ``intelligent_service.py`` and
``git_tools.py``; both now import it from here.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
from collections.abc import Callable

# How often a tool blocked on a subprocess re-checks for cancellation. Small
# enough that ESC feels immediate, large enough that the wakeups are free next
# to the command itself (4/s while one runs, none otherwise).
#
# Shared rather than per-tool: `bash` and `grep` are the two waits long enough
# to strand a user (a 300s ceiling and a 120s one), and a cancel that feels
# instant in one tool and laggy in the next is worse than either.
CANCEL_POLL_INTERVAL = 0.25


def cancel_probe(config) -> Callable[[], bool]:
    """A zero-arg predicate reading ``config.cancel_event`` FRESH each call.

    Not a captured event: the design-chat REPL swaps ``config.cancel_event``
    per turn (asi.py), so a value read once before a long wait goes stale and
    the poll then watches an event nobody will ever set — the same trap the RAG
    indexers document. Returns a callable so the polling loop stays free of the
    config object.
    """

    def _probe() -> bool:
        _ev = getattr(config, "cancel_event", None)
        return _ev is not None and _ev.is_set()

    return _probe


def run_bounded_subprocess(
    cmd,
    *,
    timeout: int = 120,
    shell: bool = False,
    executable: str | None = None,
    cwd: str | None = None,
    input: str | None = None,
    env: dict[str, str] | None = None,
    inherit_output: bool = False,
) -> subprocess.CompletedProcess:
    """``subprocess.run`` with a mandatory timeout and full process-group cleanup.

    Guarantees the agent never blocks indefinitely on a subprocess — e.g.
    ``pytest`` dropping into ``--pdb`` / ``input()``, or a build/network stall.
    A bare ``subprocess.run`` (no timeout) hangs forever in that case; and
    since ``TimeoutExpired`` is a ``SubprocessError`` (not ``OSError``), it
    escapes the surrounding ``except Exception`` only *after* the hang — by
    then the agent loop is wedged.

    Mirrors the safety discipline of ``git_tools._tool_shell_exec``:
    ``start_new_session=True`` + ``killpg`` on timeout, so grandchildren
    (pytest-spawned server fixtures) are torn down too, not orphaned. Returns a
    ``CompletedProcess`` (returncode=-9 + a trailing note on timeout) so callers
    keep their existing ``.returncode`` / ``.stdout`` / ``.stderr`` access and
    degrade gracefully.

    With ``inherit_output=True`` the child shares this process's stdin/stdout/
    stderr instead of piping — for long-running launches whose output must
    stream live to the user (e.g. ``tools/auto_learning_loop.py`` watching the
    monitor). The timeout + process-group kill guarantees are IDENTICAL; the
    returned ``CompletedProcess`` simply carries empty stdout/stderr (nothing
    was captured). ``input=`` is incompatible with this mode (stdin is
    inherited, not a pipe) and raises ``ValueError``.
    """
    if inherit_output and input is not None:
        raise ValueError("input= cannot be combined with inherit_output=True (stdin is inherited, not a pipe)")
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        executable=executable,
        cwd=cwd,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=None if inherit_output else subprocess.PIPE,
        stderr=None if inherit_output else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        env=env,
    )
    try:
        if inherit_output:
            # No pipes to drain: wait() is the bounded form (the gate's
            # wait(timeout=) shape). Output is already on the real terminal.
            proc.wait(timeout=timeout)
            stdout = stderr = ""
        else:
            stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Kill the whole process group (start_new_session created one) so
        # grandchildren are terminated too, not orphaned. The child is a
        # session leader, so its pgid == its pid; do NOT re-resolve the group
        # via os.getpgid() here — `communicate()` reaps a direct child that
        # exited early (e.g. `bash -c "sleep 45 & ..."`), and a dead leader
        # makes getpgid() raise ProcessLookupError, which the suppress() below
        # would swallow: the kill is silently skipped and the grandchildren
        # survive as orphans. killpg() on the stored pid targets the GROUP,
        # which outlives its leader while any member is still alive.
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        if inherit_output:
            # No pipes to reap/drain — just collect the exit status so the
            # child does not linger as a zombie. A failure to reap is logged
            # (best-effort) rather than silently suppressed.
            try:
                proc.wait(timeout=5)
            except Exception as reap_exc:
                logging.getLogger(__name__).debug("post-timeout reap failed for pid %s: %s", proc.pid, reap_exc)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=-9,
                stdout="",
                stderr=f"[aborted: exceeded {timeout}s timeout]",
            )
        # Reap the killed process and drain partial output to avoid zombies.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            # Preserve the partial output buffered before the timeout rather
            # than blanking it: half a traceback is still evidence, and losing
            # it turns a timeout into a misleading empty result.
            # text=True above: exc.stdout is str | None; decode defensively
            # when the typeshed union (bytes | str | None) is wider.
            _out, _err = exc.stdout or "", exc.stderr or ""
            stdout = _out.decode("utf-8", "replace") if isinstance(_out, bytes) else _out
            stderr = _err.decode("utf-8", "replace") if isinstance(_err, bytes) else _err
        _note = f"\n[aborted: exceeded {timeout}s timeout]"
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-9,
            stdout=stdout or "",
            stderr=(stderr or "") + _note,
        )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )
