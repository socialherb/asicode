"""Bounded subprocess execution with a mandatory timeout + process-group cleanup.

Single source of truth for :func:`run_bounded_subprocess`. Previously this
helper was duplicated (and at risk of drift) in ``intelligent_service.py`` and
``git_tools.py``; both now import it from here.
"""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Optional


def run_bounded_subprocess(
    cmd,
    *,
    timeout: int = 120,
    shell: bool = False,
    executable: Optional[str] = None,
    cwd: Optional[str] = None,
    input: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> "subprocess.CompletedProcess":
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
    """
    proc = subprocess.Popen(
        cmd, shell=shell, executable=executable, cwd=cwd,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        start_new_session=True, env=env,
    )
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole process group (start_new_session created one) so
        # grandchildren are terminated too, not orphaned.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        # Reap the killed process and drain partial output to avoid zombies.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            stdout, stderr = "", ""
        _note = f"\n[aborted: exceeded {timeout}s timeout]"
        return subprocess.CompletedProcess(
            args=cmd, returncode=-9,
            stdout=stdout or "", stderr=(stderr or "") + _note,
        )
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode,
        stdout=stdout or "", stderr=stderr or "",
    )
