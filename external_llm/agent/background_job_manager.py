"""
Background Job Manager for asicode Agent.

Manages long-running shell commands that exceed the bash tool's timeout.
Commands are automatically transitioned from blocking to background execution,
allowing the agent to continue working while the job runs.

Design:
  - Thread-safe (Lock-protected job registry)
  - Auto-cleanup of completed/failed jobs (lazy eviction)
  - Configurable max concurrent jobs to limit resource usage
  - Integrates with _tool_shell_exec via a simple transition at timeout
"""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Optional

from ..client import interruptible_sleep

logger = logging.getLogger(__name__)

# Cap per accumulated output stream. The reaper drains pipes every tick, so a
# chatty long-running job (server logs, verbose build) would otherwise grow
# the in-RAM buffer without bound. Keep the TAIL — the most recent output is
# what a "what is this job doing" query needs.
_OUTPUT_BUF_CAP = 2 * 1024 * 1024  # 2 MiB
_TRUNCATION_MARKER = "…[oldest output truncated]…\n"
# Max completed-job results preserved in the ring-buffer after reaping/eviction.
# Once the ring is full, the oldest entry is evicted (FIFO).  This gives the
# agent a window to retrieve job output after a job is cleaned up.
_REAPED_RESULTS_MAX = 20
# Per-stream output kept per reaped job. Generous because the whole point of
# the ring is that this is the ONLY surviving copy of a finished job's result:
# 20 jobs x 2 streams x 32 KiB ≈ 1.3 MiB worst case, which is nothing next to
# the 2 MiB a single *live* job's buffer is already allowed.
_REAPED_OUTPUT_CAP = 32 * 1024


def _cap_tail(buf: str) -> str:
    if len(buf) <= _OUTPUT_BUF_CAP:
        return buf
    return _TRUNCATION_MARKER + buf[-_OUTPUT_BUF_CAP:]


_MALLOC_NOISE_TOKEN = "MallocStackLogging"


def strip_malloc_noise(text: str) -> str:
    """Drop macOS libmalloc stack-logging chatter from captured stderr.

    A forked child inherits the parent's malloc stack-logging state and
    libmalloc writes its status lines to fd 2 *before* exec — so they land in
    the captured stderr pipe and masquerade as command output. Unsetting the
    ``MallocStackLogging*`` env vars on the child does not suppress them when
    logging was enabled on the parent by some route other than the
    environment, so filtering is the load-bearing mitigation.

    Applied to the *accumulated* buffer at read time rather than to each pipe
    chunk: a non-blocking read can split a noise line in half, and re-filtering
    the whole buffer lets the straddled line self-heal once the rest arrives.
    """
    if not text or _MALLOC_NOISE_TOKEN not in text:
        return text
    return "".join(
        line for line in text.splitlines(keepends=True)
        if _MALLOC_NOISE_TOKEN not in line
    )


# NOTE: the partial-output hand-over used to be excavated from CPython's private
# `_fileobj2output` after `communicate(timeout=...)` raised — the only place the
# bytes it had already consumed still existed. The bash tool now reads the pipes
# itself into a bounded capture (`_BoundedCapture`), so the data is in hand and
# the hand-over is a plain assignment to `_recovered_stdout` / `_recovered_stderr`.
# `read_output` below is unchanged and still the consumer of that pair.


class BackgroundJobInfo:
    """Immutable snapshot of a background job's state."""

    def __init__(self, job_id: str, command: str, pid: Optional[int],
                 status: str, elapsed: float, stdout: str, stderr: str):
        self.job_id = job_id
        self.command = command
        self.pid = pid
        self.status = status  # "running", "completed", "failed", "killed"
        self.elapsed = elapsed
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self) -> str:
        return (
            f"BackgroundJobInfo(id={self.job_id!r}, cmd={self.command[:60]!r}, "
            f"status={self.status!r}, pid={self.pid}, elapsed={self.elapsed:.1f}s)"
        )


class BackgroundJob:
    """Internal mutable job state (not exposed to callers directly)."""

    def __init__(self, job_id: str, command: str, proc: subprocess.Popen,
                 start_time: float):
        self.job_id = job_id
        self.command = command
        self.proc = proc
        self.start_time = start_time
        self.status: str = "running"
        self._lock = threading.Lock()
        # Resolve the process group ONCE, here, while the child is certainly
        # alive (start() registers the job immediately after the Popen). A
        # later kill() must NOT re-resolve: the leader may have exited and been
        # reaped by then — getpgid() raises ProcessLookupError, the fallback
        # proc.kill() then no-ops on the dead leader, and the grandchildren are
        # orphaned. The GROUP, which killpg targets, outlives its leader while
        # any member (e.g. `cmd &` children) is still alive, so the stored
        # pgid stays valid. Job spawners use start_new_session=True, making
        # this equal to proc.pid.
        try:
            self._pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            # Leader already gone / no process groups: fall back to the pid —
            # with start_new_session=True the two are identical anyway.
            self._pgid = proc.pid
        # Accumulated stdout/stderr buffers — read_output() drains into these
        # so that pipe data is never lost between calls.  get_info() reads the
        # accumulated buffer.  The reaper tick also drains periodically to
        # prevent pipe-full deadlock (Bug #3). Tail-capped at _OUTPUT_BUF_CAP.
        self._stdout_buf: str = ""
        self._stderr_buf: str = ""
        # Serializes drains: the reaper thread and an agent thread calling
        # get_info()/list_jobs() concurrently would otherwise interleave fd
        # reads (chunk-order corruption) and race the non-atomic `buf +=`.
        # Separate from self._lock on purpose — kill() holds self._lock for
        # up to ~6 s (SIGTERM wait), and output drains must not block on that.
        self._io_lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def read_output(self) -> tuple[str, str]:
        """Read available stdout/stderr (non-blocking) and accumulate in buffer.

        Returns the *new* data read this call (not the entire accumulated
        buffer).  Use :attr:`_stdout_buf` / :attr:`_stderr_buf` for the full
        accumulated output.
        """
        with self._io_lock:
            stdout = ""
            stderr = ""

            # Prepend partially-read data recovered from communicate()'s internal
            # buffer after a TimeoutExpired (Bug #2).  This data lives in ad-hoc
            # attributes on the proc object and is consumed exactly once (first
            # drain after background transition loses communicate's read-ahead).
            #
            # getattr with a default, per stream, NOT attribute access on both
            # inside one try: the two streams are independent, and fetching them
            # as a pair meant an absent `_recovered_stderr` (set only when
            # non-empty) threw away an already-fetched `_recovered_stdout`.
            # Measured: the attribute held "PRE-1\nPRE-2\nPRE-3\n" while this
            # method returned "" and the model saw only post-transition output.
            recovered_stdout = getattr(self.proc, "_recovered_stdout", "")
            recovered_stderr = getattr(self.proc, "_recovered_stderr", "")
            if recovered_stdout:
                stdout += recovered_stdout
                self.proc._recovered_stdout = ""
            if recovered_stderr:
                stderr += recovered_stderr
                self.proc._recovered_stderr = ""

            if self.proc.stdout:
                with contextlib.suppress(UnicodeDecodeError, OSError):  # binary output / closed pipe
                    stdout += self._read_fd(self.proc.stdout)
            if self.proc.stderr:
                with contextlib.suppress(UnicodeDecodeError, OSError):
                    stderr += self._read_fd(self.proc.stderr)
            self._stdout_buf = _cap_tail(self._stdout_buf + stdout)
            self._stderr_buf = _cap_tail(self._stderr_buf + stderr)
            return stdout, stderr

    @staticmethod
    def _read_fd(fd) -> str:
        """Read all available data from a file descriptor (non-blocking)."""
        import fcntl
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        try:
            data = fd.read()
        except (OSError, ValueError, TypeError):
            # TypeError is the "no data available yet" signal on CPython < 3.14.
            # A non-blocking read that would block makes the raw layer return
            # None; a text-mode pipe then feeds that None to the incremental
            # decoder, which fails with "can't concat NoneType to bytes"
            # (measured: 3.12.13 raises TypeError, 3.14.3 raises BlockingIOError).
            # Only 3.14+ raises an OSError subclass, so on every supported
            # version below it this except clause is what keeps an empty pipe
            # from propagating out of get_info() and failing the job_output tool.
            return ""
        else:
            return data if data else ""
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

    def poll_status(self) -> str:
        """Update and return current status.

        Thread-safe: wraps status in self._lock to avoid TOCTOU with kill().
        """
        with self._lock:
            if self.status in ("completed", "failed", "killed"):
                return self.status
            ret = self.proc.poll()
            if ret is None:
                self.status = "running"
            elif ret == 0:
                self.status = "completed"
            else:
                self.status = "failed"
            return self.status

    def kill(self) -> None:
        """Terminate the process tree."""
        with self._lock:
            if self.status in ("completed", "failed", "killed"):
                return
            try:
                # Kill the process group to catch children. Uses the pgid
                # resolved at registration (see __init__) — re-resolving via
                # getpgid() here fails once the leader exited and was reaped,
                # silently skipping the kill and orphaning the grandchildren.
                import signal
                os.killpg(self._pgid, signal.SIGTERM)
                # Give it a moment, then SIGKILL
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self._pgid, signal.SIGKILL)
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            "Job %s did not die after SIGKILL — process still alive", self.job_id,
                        )
                        return  # Don't set status to "killed" if still alive
            except (ProcessLookupError, PermissionError, OSError):
                # Already dead or no permission
                self.proc.kill()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning("Job %s did not die after SIGKILL (fallback path)", self.job_id)
                    return  # Don't set status to "killed" if still alive
            except Exception as e:
                logger.warning("Failed to kill job %s: %s", self.job_id, e)
                return
            self.status = "killed"


class BackgroundJobManager:
    """Thread-safe manager for background shell jobs.

    Usage:
        mgr = BackgroundJobManager(max_jobs=5)
        job_id = mgr.start(command, proc)
        info = mgr.get_info(job_id)
        summary = mgr.list_jobs()
        mgr.kill(job_id)
        mgr.cleanup()  # reap completed/failed jobs
    """

    def __init__(self, max_jobs: int = 5, reap_interval: int = 30):
        self.max_jobs = max_jobs
        self._reap_interval = reap_interval
        self._jobs: dict[str, BackgroundJob] = OrderedDict()
        # Ring buffer for completed/killed job results: preserves output after
        # the job is removed from _jobs by cleanup() or _evict_over_capacity.
        # Maps job_id -> BackgroundJobInfo.  Bounded at _REAPED_RESULTS_MAX
        # (oldest entries evicted first via popitem(last=False)).
        self._reaped_results: "OrderedDict[str, BackgroundJobInfo]" = OrderedDict()
        self._lock = threading.Lock()
        self._last_reap: float = 0.0
        self._reaper_timer: Optional[threading.Timer] = None
        self._reaper_active: bool = False

    @staticmethod
    def _reaped_tail(buf: Optional[str]) -> str:
        """Keep the LAST ``_REAPED_OUTPUT_CAP`` chars of *buf*, marking elision.

        Tail, never head: a job is backgrounded precisely because it is long,
        and a long command's answer — the test summary, the build verdict, the
        final line of a script — is at the END. A leading slice preserves the
        boilerplate and drops the result, which is worse than useless because
        it looks like a successful retrieval. (Same reasoning, and the same
        direction, as :func:`_cap_tail` for live buffers.)
        """
        buf = buf or ""
        if len(buf) <= _REAPED_OUTPUT_CAP:
            return buf
        return _TRUNCATION_MARKER + buf[-_REAPED_OUTPUT_CAP:]

    def _snapshot_job_locked(self, job_id: str, job: BackgroundJob) -> BackgroundJobInfo:
        """Build a BackgroundJobInfo from a job without I/O (caller must hold _lock).

        Uses the already-accumulated output buffers; does NOT call poll_status()
        or read_output() to avoid I/O under the lock.
        """
        return BackgroundJobInfo(
            job_id=job_id,
            command=job.command,
            pid=job.proc.pid,
            status=job.status,
            elapsed=job.elapsed,
            stdout=self._reaped_tail(job._stdout_buf),
            stderr=self._reaped_tail(strip_malloc_noise(job._stderr_buf)),
        )

    def _store_reaped_locked(self, job_id: str, info: BackgroundJobInfo) -> None:
        """Insert a reaped job into the ring buffer, evicting oldest if over cap.

        Caller MUST hold ``self._lock``.
        """
        self._reaped_results[job_id] = info
        self._reaped_results.move_to_end(job_id)
        while len(self._reaped_results) > _REAPED_RESULTS_MAX:
            self._reaped_results.popitem(last=False)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, command: str, proc: subprocess.Popen) -> str:
        """Register a background job and return its job_id.

        If max_jobs is exceeded, finished (completed/failed/killed) jobs are
        evicted first; if every slot is occupied by a running job, the oldest
        running job is killed.

        max_jobs is a HARD bound: the new job is inserted *and* any
        over-capacity eviction is decided in a single ``self._lock``
        acquisition, so concurrent starters can never each observe a
        post-eviction (lower) count and collectively insert past the cap.
        The potentially-blocking process kill (up to ~6 s) still runs
        *outside* the lock so ``get_info`` / ``list_jobs`` / ``kill`` are
        not blocked.
        """
        job_id = uuid.uuid4().hex[:12]

        # Start the periodic reaper on first registration (lazy: managers that
        # never register a job — common in tests — never spawn a thread).
        self._ensure_reaper()

        with self._lock:
            job = BackgroundJob(job_id, command, proc, time.monotonic())
            self._jobs[job_id] = job
            # Atomically bring the registry back down to max_jobs. Finished
            # jobs are dropped inline (non-blocking); running jobs selected
            # for killing are popped now and killed after we release the lock.
            kill_victims = self._evict_over_capacity_locked(job_id)
            logger.info(
                "Background job started: id=%s cmd=%.200s pid=%d",
                job_id, command, proc.pid,
            )

        # Kill outside the lock — may block up to ~6 s.
        for victim_id, victim_job, victim_cmd in kill_victims:
            victim_job.kill()
            # Drain any remaining output (final pipe flush after kill).
            try:
                victim_job.read_output()
            except Exception:
                # Best-effort: the snapshot below still stores whatever was
                # already buffered. Logged because a failure here is the
                # difference between a retrievable result and a silently
                # empty one.
                logger.debug(
                    "Final drain failed for evicted job %s", victim_id, exc_info=True
                )
            # Snapshot the victim's final state into the ring buffer so
            # get_info() can still retrieve output after eviction.
            with self._lock:
                if victim_id not in self._reaped_results:
                    info = self._snapshot_job_locked(victim_id, victim_job)
                    self._store_reaped_locked(victim_id, info)
            logger.warning(
                "Killed oldest job to enforce max_jobs=%d: id=%s cmd=%.200s",
                self.max_jobs, victim_id, victim_cmd,
            )

        return job_id

    def get_info(self, job_id: str) -> Optional[BackgroundJobInfo]:
        """Get a snapshot of job state, or None if not found.

        Fallback: if the job has been reaped (removed from ``_jobs`` by
        cleanup or capacity eviction), searches the ring buffer of recent
        results (``_reaped_results``), which preserves the final state
        including stdout/stderr tail.

        I/O (``poll_status``, ``read_output``) is performed *outside*
        ``self._lock`` so that long-running pipe reads do not block
        other callers (``kill``, ``list_jobs``, etc.).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                # Not in active registry — check the reaped-results ring.
                reaped = self._reaped_results.get(job_id)
                if reaped is not None:
                    return reaped
                return None
            # Snapshot mutable fields under the lock, then release
            command = job.command
            pid = job.proc.pid

        # ── I/O outside the lock ──
        status = job.poll_status()
        # drain pipe → accumulates into _stdout_buf/_stderr_buf
        with contextlib.suppress(UnicodeDecodeError, OSError):  # binary output / closed pipe
            job.read_output()

        if status in ("completed", "failed", "killed"):
            # final drain (process is dead, pipe is flushing)
            with contextlib.suppress(UnicodeDecodeError, OSError):
                job.read_output()

        stdout = job._stdout_buf
        stderr = strip_malloc_noise(job._stderr_buf)

        return BackgroundJobInfo(
            job_id=job_id,
            command=command,
            pid=pid,
            status=status,
            elapsed=job.elapsed,
            stdout=stdout or "",
            stderr=stderr or "",
        )

    def wait_for_completion(self, job_id: str, timeout: float = 120.0,
                                poll_interval: float = 1.0,
                                cancel_event: Optional[Any] = None) -> Optional[BackgroundJobInfo]:
        """Wait for a background job to finish (completed/failed/killed).

        Polls at *poll_interval* seconds until the job terminates or
        *timeout* seconds elapse.  Returns the final BackgroundJobInfo
        on completion, or None if the job is not found.

        If the timeout expires while the job is still running, returns
        the current snapshot (status == "running").

        If *cancel_event* is set mid-wait (e.g. the user pressed ESC while a
        ``job(action=output, wait_timeout=...)`` call was blocking the turn),
        the wait is abandoned at the next poll tick and the current snapshot
        is returned so the caller can surface the cancellation — a raw
        ``time.sleep`` here would otherwise ignore ESC for the whole budget.
        """
        deadline = time.monotonic() + timeout
        while True:
            info = self.get_info(job_id)
            if info is None:
                return None
            if info.status != "running":
                return info
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return info  # timeout — return current snapshot
            if interruptible_sleep(min(poll_interval, remaining), cancel_event):
                return info  # cancelled — return current snapshot

    def list_jobs(self, include_completed: bool = True) -> list[BackgroundJobInfo]:
        """List all tracked jobs, lazily reaping old completed ones first.

        I/O (``poll_status``, ``read_output``) is performed *outside*
        ``self._lock`` so that pipe reads do not block other callers.
        """
        self._maybe_reap()
        with self._lock:
            # Snapshot immutable fields under the lock
            snapshots = [
                (job_id, job, job.command, job.proc.pid)
                for job_id, job in list(self._jobs.items())
            ]

        # ── I/O outside the lock ──
        infos = []
        for job_id, job, command, pid in snapshots:
            status = job.poll_status()
            if not include_completed and status in ("completed", "failed", "killed"):
                continue
            # drain pipe → accumulates into buffer
            with contextlib.suppress(UnicodeDecodeError, OSError):  # binary output / closed pipe
                job.read_output()
            stdout = job._stdout_buf
            stderr = strip_malloc_noise(job._stderr_buf)
            infos.append(BackgroundJobInfo(
                job_id=job_id,
                command=command,
                pid=pid,
                status=status,
                elapsed=job.elapsed,
                stdout=(stdout or "")[:200],
                stderr=(stderr or "")[:200],
            ))
        return infos

    def kill(self, job_id: str) -> Optional[str]:
        """Kill a specific job. Returns the final status string, or None if not found.

        ``job.kill()`` may block up to 6 seconds (SIGTERM wait + SIGKILL fallback),
        so it is deliberately called *outside* ``self._lock`` to avoid blocking
        the entire manager for other callers (``get_info``, ``list_jobs``, etc.).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
        job.kill()  # outside lock — may block
        with self._lock:
            logger.info("Background job killed: id=%s cmd=%.200s", job_id, job.command)
            return self._jobs[job_id].poll_status() if job_id in self._jobs else "killed"

    def _drain_finished(self) -> None:
        """Read any still-buffered pipe output for finished jobs. Lock-free.

        MUST run before :meth:`cleanup` snapshots a job, because
        ``_snapshot_job_locked`` deliberately performs no I/O — it copies
        ``_stdout_buf``, and that buffer is only filled by ``read_output()``.
        A job nobody polled between its exit and the reaper tick therefore had
        its entire output still sitting unread in the OS pipe, and the ring
        preserved an empty string: a "successful" retrieval of nothing, for
        precisely the walk-away workflow the ring exists to serve.

        The job list is snapshotted under the lock and the reads happen
        outside it, matching ``get_info``/``list_jobs``.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            try:
                if job.poll_status() != "running":
                    job.read_output()  # final drain — process is dead, pipe flushing
            except Exception:
                # Never let one unreadable pipe abort the sweep for the rest,
                # but do say so: an undrained job is reaped with empty output,
                # which reads as a successful retrieval of nothing.
                logger.debug("Final drain failed before reap", exc_info=True)

    def cleanup(self) -> int:
        """Remove completed/failed/killed jobs from the registry. Returns count removed.

        Before removal, each job's final state is snapshotted into a bounded
        ring buffer (``_reaped_results``) so callers can still retrieve recent
        job output via ``get_info()`` after cleanup.
        """
        self._drain_finished()
        with self._lock:
            before = len(self._jobs)
            # Snapshot completed/failed/killed jobs BEFORE removal so their
            # final output is preserved in the ring buffer.
            for jid, j in list(self._jobs.items()):
                if j.poll_status() != "running":
                    info = self._snapshot_job_locked(jid, j)
                    self._store_reaped_locked(jid, info)
            self._jobs = OrderedDict(
                (jid, j) for jid, j in self._jobs.items()
                if j.poll_status() == "running"
            )
            removed = before - len(self._jobs)
            if removed:
                logger.debug("Cleaned up %d background job(s)", removed)
            return removed

    def get(self, job_id: str) -> Optional[BackgroundJob]:
        """Direct access to internal job (for integration use)."""
        with self._lock:
            return self._jobs.get(job_id)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _evict_over_capacity_locked(self, keep_job_id: str) -> list:
        """Reduce ``_jobs`` to ``max_jobs``, returning running jobs to kill.

        Caller MUST hold ``self._lock``.  Finished (completed/failed/killed)
        jobs are removed inline; when only running jobs remain over capacity,
        the oldest running job — never the just-inserted ``keep_job_id`` — is
        popped and returned for killing outside the lock.

        Because the insert that created ``keep_job_id`` and this eviction
        happen under a single lock acquisition, concurrent starters cannot
        observe an intermediate (sub-max) count and pile on:
        ``len(_jobs) <= max_jobs`` is an invariant visible to every lock
        holder.
        """
        victims: list = []
        while len(self._jobs) > self.max_jobs:
            # Prefer dropping a finished job (no kill needed).
            evicted_finished = False
            for jid, job in list(self._jobs.items()):
                if jid == keep_job_id:
                    continue
                if job.poll_status() != "running":
                    # Snapshot before removal so output is preserved.
                    info = self._snapshot_job_locked(jid, job)
                    self._store_reaped_locked(jid, info)
                    del self._jobs[jid]
                    logger.debug("Evicted completed job: id=%s", jid)
                    evicted_finished = True
                    break
            if evicted_finished:
                continue
            # All other jobs are running — kill the oldest one we can.
            killed_one = False
            for jid, job in list(self._jobs.items()):
                if jid != keep_job_id:
                    # Victim will be killed outside the lock.  Don't snapshot
                    # yet — the status is still "running" here.  The caller
                    # (start()) snapshots after the kill so the ring buffer
                    # gets the correct "killed" status.
                    del self._jobs[jid]
                    victims.append((jid, job, job.command))
                    killed_one = True
                    break
            if not killed_one:
                # Only keep_job_id remains (e.g. max_jobs <= 0); cannot evict
                # further without dropping the just-started job. Stop to avoid
                # an infinite loop.
                break
        return victims

    def _maybe_reap(self) -> None:
        """Periodically reap completed jobs based on interval.

        The timestamp check is lock-protected so only one thread triggers
        cleanup per interval.  cleanup() is called *outside* the lock to
        avoid re-entrant deadlock (cleanup acquires self._lock internally).
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_reap < self._reap_interval:
                return
            self._last_reap = now
        self.cleanup()

    # ── Background reaper (periodic zombie / stale-job cleanup) ────────────

    def _ensure_reaper(self) -> None:
        """Lazily start a daemon reaper on first job registration.

        Without a periodic reaper, a job the agent never queries again (no
        ``get_info`` / ``list_jobs`` / ``start``) is never ``poll()``-ed, so a
        finished subprocess stays a zombie and its ``_jobs`` entry lingers for
        the whole process lifetime (bounded only by ``max_jobs`` eviction on
        the next ``start()``). The reaper bounds this to ~``reap_interval``.
        """
        with self._lock:
            if self._reaper_active:
                return
            self._reaper_active = True
            self._schedule_reap_locked()

    def _schedule_reap_locked(self) -> None:
        """Schedule the next reap tick. Caller MUST hold ``self._lock``."""
        t = threading.Timer(self._reap_interval, self._reap_tick)
        t.daemon = True
        self._reaper_timer = t
        t.start()

    def _reap_tick(self) -> None:
        try:
            self._maybe_reap()
        except Exception:
            logger.debug("Background reaper tick failed", exc_info=True)

        # Drain pipes for all running jobs to prevent pipe-full deadlock
        # (Bug #3).  If a child process fills the OS pipe buffer (~64 KB) and
        # nobody reads, write() blocks indefinitely, making the job appear
        # stuck forever.  Periodic draining keeps the pipe clear even when
        # the agent has not called get_info() / job_output recently.
        try:
            with self._lock:
                jobs = list(self._jobs.items())
            for _job_id, job in jobs:
                if job.poll_status() == "running":
                    with contextlib.suppress(UnicodeDecodeError, OSError):  # binary output / closed pipe
                        job.read_output()
        except Exception:
            logger.debug("Reaper pipe drain failed", exc_info=True)

        with self._lock:
            if self._reaper_active:
                self._schedule_reap_locked()

    def shutdown(self) -> None:
        """Stop the background reaper and cancel the pending tick.

        Safe to call multiple times. Does not touch tracked jobs.
        """
        with self._lock:
            self._reaper_active = False
            t = self._reaper_timer
            self._reaper_timer = None
        if t is not None:
            t.cancel()


# Module-level singleton for shared use across tool instances
_global_bg_manager: Optional[BackgroundJobManager] = None
_global_bg_manager_lock = threading.Lock()


def get_global_background_job_manager(max_jobs: int = 5) -> BackgroundJobManager:
    """Get or create the global BackgroundJobManager singleton.

    ToolRegistry clones (subagents) share this singleton so that
    background jobs survive subagent lifecycle.
    """
    global _global_bg_manager
    if _global_bg_manager is None:
        with _global_bg_manager_lock:
            if _global_bg_manager is None:
                _global_bg_manager = BackgroundJobManager(max_jobs=max_jobs)
    return _global_bg_manager
