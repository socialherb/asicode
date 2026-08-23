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
from typing import Any

from ..client import interruptible_sleep
from ..common.bounded_capture import _BoundedCapture

logger = logging.getLogger(__name__)

# Cap per accumulated output stream. The reaper drains pipes every tick, so a
# chatty long-running job (server logs, verbose build) would otherwise grow
# the in-RAM buffer without bound.  The buffers are _BoundedCapture instances
# (head+tail, chunk-fed — no O(n) re-materialisation per drain): the most
# recent output is what a "what is this job doing" query needs, and the head
# is what _truncate_bash_output's leading half (the failing command) needs.
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


_MALLOC_NOISE_TOKEN = "MallocStackLogging"


def _tail_preview(buf: str) -> str:
    """Keep the LAST 200 chars of *buf* for list_jobs() previews.

    Tail, never head — a long command's answer (test summary, build verdict)
    is at the END; a leading slice preserves boilerplate and drops the result.
    Mirrors :func:`_reaped_tail` (which applies a different cap because it
    feeds the preserved-output ring buffer rather than a one-line preview).
    """
    return buf[-200:] if len(buf) > 200 else buf


def _close_job_pipes(job: BackgroundJob) -> None:
    """Close a job's pipe read ends — final resource reclaim for a dropped job.

    Used when a stale victim is evicted from ``_stale_jobs`` without waiting
    for its process to die (or on shutdown).  Closing the read ends makes the
    orphan's next write hit EPIPE instead of blocking forever on a full pipe;
    the capture buffers themselves are freed with the job object.
    """
    for _fd in (job.proc.stdout, job.proc.stderr):
        if _fd is not None:
            try:
                _fd.close()
            except Exception:
                # A concurrently-closed fd (reaper drain) is the normal cause;
                # the eviction already gave up on this job either way.
                logger.debug("Pipe close failed for dropped job %s", job.job_id, exc_info=True)


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
    return "".join(line for line in text.splitlines(keepends=True) if _MALLOC_NOISE_TOKEN not in line)


# NOTE: the partial-output hand-over used to be excavated from CPython's private
# `_fileobj2output` after `communicate(timeout=...)` raised — the only place the
# bytes it had already consumed still existed. The bash tool now reads the pipes
# itself into a bounded capture (`_BoundedCapture`), so the data is in hand and
# the hand-over is a plain assignment to `_recovered_stdout` / `_recovered_stderr`.
# `read_output` below is unchanged and still the consumer of that pair.


class BackgroundJobInfo:
    """Immutable snapshot of a background job's state."""

    def __init__(
        self,
        job_id: str,
        command: str,
        pid: int | None,
        status: str,
        elapsed: float,
        stdout: str,
        stderr: str,
        stdout_total: int = 0,
        stderr_total: int = 0,
    ):
        self.job_id = job_id
        self.command = command
        self.pid = pid
        self.status = status  # "running", "completed", "failed", "killed"
        self.elapsed = elapsed
        self.stdout = stdout
        self.stderr = stderr
        # Characters the streams ACTUALLY produced, per the capture's `total`
        # counter — deliberately separate from len(stdout)/len(stderr): a live
        # buffer elides its middle past _OUTPUT_BUF_CAP and the reaped ring
        # tails to _REAPED_OUTPUT_CAP, so the surviving text undercounts.  The
        # render-time truncation notice names these totals (what the process
        # printed), not what happened to survive.
        self.stdout_total = stdout_total
        self.stderr_total = stderr_total

    def __repr__(self) -> str:
        return (
            f"BackgroundJobInfo(id={self.job_id!r}, cmd={self.command[:60]!r}, "
            f"status={self.status!r}, pid={self.pid}, elapsed={self.elapsed:.1f}s)"
        )


class BackgroundJob:
    """Internal mutable job state (not exposed to callers directly)."""

    def __init__(self, job_id: str, command: str, proc: subprocess.Popen, start_time: float):
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
        # prevent pipe-full deadlock (Bug #3).  _BoundedCapture caps each at
        # _OUTPUT_BUF_CAP, keeping head AND tail: the tail is what a live
        # "what is this job doing" query needs, the head is what a later
        # truncation of the rendered result keeps as its leading half.
        self._stdout_buf = _BoundedCapture(_OUTPUT_BUF_CAP)
        self._stderr_buf = _BoundedCapture(_OUTPUT_BUF_CAP)
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
        buffer).  Use :attr:`_stdout_buf` / :attr:`_stderr_buf` (each a
        ``_BoundedCapture``; read via ``.text()``) for the accumulated output.
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
                self.proc._recovered_stdout = ""  # type: ignore[attr-defined]  # dynamic Popen attr
            if recovered_stderr:
                stderr += recovered_stderr
                self.proc._recovered_stderr = ""  # type: ignore[attr-defined]  # dynamic Popen attr

            if self.proc.stdout:
                with contextlib.suppress(UnicodeDecodeError, OSError):  # binary output / closed pipe
                    stdout += self._read_fd(self.proc.stdout)
            if self.proc.stderr:
                with contextlib.suppress(UnicodeDecodeError, OSError):
                    stderr += self._read_fd(self.proc.stderr)
            self._stdout_buf.feed(stdout)
            self._stderr_buf.feed(stderr)
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

        The lock is acquired *non-blocking*: kill() holds ``self._lock`` for
        up to ~6 s (SIGTERM wait + SIGKILL fallback), and ``cleanup()`` /
        ``_evict_over_capacity_locked`` call this while holding the manager
        lock.  A blocking acquire there would freeze the entire manager
        (list_jobs / get_info / start) behind one mid-teardown job.  On lock
        contention we return the last-known status — slightly stale, but
        non-blocking; the next poll resolves it.
        """
        if not self._lock.acquire(blocking=False):
            # kill() is mid-teardown; report the cached status rather than
            # block (up to ~6 s) on the job lock.
            return self.status
        try:
            if self.status in ("completed", "failed", "killed", "killing"):
                # "killing" is set by _evict_over_capacity_locked for a victim
                # whose kill is in flight outside the lock; never regress it
                # back to "running" from a stale poll.
                return self.status
            ret = self.proc.poll()
            if ret is None:
                self.status = "running"
            elif ret == 0:
                self.status = "completed"
            else:
                self.status = "failed"
            return self.status
        finally:
            self._lock.release()

    def _settle_status(self) -> None:
        """Resolve status after a kill attempt that could not finish.

        The transient ``"killing"`` marker (set by over-capacity eviction)
        must never outlive a kill() attempt: if the process is dead,
        classify its real rc (0 -> completed, <0 -> killed, >0 -> failed);
        if it is still alive, revert to "running" so the job stays honest
        and an eviction victim gets re-tracked for reaping instead of
        freezing the ring as a permanent "killing" entry.
        """
        try:
            ret = self.proc.poll()
        except Exception:
            logger.warning("poll() failed while settling status for job %s", self.job_id, exc_info=True)
            return  # keep the current status — the next reap tick retries
        if ret is None:
            self.status = "running"
        elif ret == 0:
            self.status = "completed"
        elif ret < 0:
            self.status = "killed"  # died from a signal (typically our SIGKILL)
        else:
            self.status = "failed"

    def kill(self) -> None:
        """Terminate the process tree."""
        with self._lock:
            if self.status in ("completed", "failed", "killed"):
                return
            # Resolve the real exit status FIRST.  status is only refreshed by
            # poll_status(), which the reaper calls on a 30s tick — a process
            # that exited between ticks still reads "running" here.  Killing it
            # would then take the ProcessLookupError fallback (no-op on a dead
            # leader) and force status="killed", destroying the real outcome:
            # a build that succeeded (rc=0) and one that failed (rc=7) both
            # read "killed" afterwards, permanently (poll_status() early-returns
            # on terminal states).  A pre-poll preserves the truth.
            ret = self.proc.poll()
            if ret is not None:
                self.status = "completed" if ret == 0 else "failed"
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
                            "Job %s did not die after SIGKILL — process still alive",
                            self.job_id,
                        )
                        # Don't claim "killed" for a live process — and never
                        # leave the transient "killing" marker (eviction
                        # victims) set once the kill attempt has returned.
                        self._settle_status()
                        return
            except (ProcessLookupError, PermissionError, OSError):
                # Already dead or no permission.  This is also the R2 TOCTOU
                # residual of the F1 pre-poll: the leader can exit between
                # poll() and killpg(), and the fallback wait() then returns
                # the REAL exit code.  Honor it (rc==0 -> completed, rc>0 ->
                # failed) instead of stamping "killed" unconditionally — a
                # build that succeeded and one that failed must stay
                # distinguishable even in this narrow window.  rc<0 (signal
                # death, typically our SIGKILL) stays "killed".
                try:
                    self.proc.kill()  # no-op on an already-dead leader
                    rc = self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning("Job %s did not die after SIGKILL (fallback path)", self.job_id)
                    # Same convergence contract as the main path: never return
                    # with the transient "killing" marker still set.
                    self._settle_status()
                    return
                except (ProcessLookupError, ChildProcessError):
                    rc = self.proc.returncode  # leader already reaped — use recorded rc
                if rc is None:
                    self.status = "killed"  # no recorded exit code — assume killed
                elif rc == 0:
                    self.status = "completed"
                elif rc < 0:
                    self.status = "killed"  # died from a signal (our SIGKILL)
                else:
                    self.status = "failed"
                return
            except Exception as e:
                logger.warning("Failed to kill job %s: %s", self.job_id, e)
                # Same convergence contract: settle so the victim never freezes
                # as a permanent "killing" ring entry.
                self._settle_status()
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
        self._reaped_results: OrderedDict[str, BackgroundJobInfo] = OrderedDict()
        # Eviction victims whose kill could not finish (process still alive,
        # e.g. D-state after SIGKILL).  They are out of _jobs — the slot was
        # freed — but not dead, so the reaper tracks them here and converges
        # the ring placeholder to the real status when they finally die.
        # Deliberately NOT re-inserted into _jobs: that would break the
        # max_jobs hard bound and re-evict the same victim on every start.
        self._stale_jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        self._last_reap: float = 0.0
        self._reaper_timer: threading.Timer | None = None
        self._reaper_active: bool = False

    @staticmethod
    def _reaped_tail(buf: str | None) -> str:
        """Keep the LAST ``_REAPED_OUTPUT_CAP`` chars of *buf*, marking elision.

        Tail, never head: a job is backgrounded precisely because it is long,
        and a long command's answer — the test summary, the build verdict, the
        final line of a script — is at the END. A leading slice preserves the
        boilerplate and drops the result, which is worse than useless because
        it looks like a successful retrieval. (Same reasoning, and the same
        direction, as :func:`_tail_preview` and the live buffers' head+tail
        capture for live reads.)
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
            stdout=self._reaped_tail(job._stdout_buf.text()),
            stderr=self._reaped_tail(strip_malloc_noise(job._stderr_buf.text())),
            stdout_total=job._stdout_buf.total,
            stderr_total=job._stderr_buf.total,
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
                job_id,
                command,
                proc.pid,
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
                logger.debug("Final drain failed for evicted job %s", victim_id, exc_info=True)
            # Overwrite the "killing" placeholder (pre-snapshotted under the
            # lock in _evict_over_capacity_locked) with the true outcome after
            # the kill: the final state, or — when the kill could not finish
            # and the process is still alive — a "running" snapshot plus
            # stale re-tracking (R3: the placeholder must never freeze).
            with self._lock:
                info = self._snapshot_job_locked(victim_id, victim_job)
                self._store_reaped_locked(victim_id, info)
                if victim_job.status not in ("completed", "failed", "killed"):
                    self._stale_jobs[victim_id] = victim_job
                    # F4: _stale_jobs must stay bounded like every other
                    # registry.  A SIGKILL-surviving victim (D-state) can
                    # outlive the whole session, and without a cap each
                    # over-capacity start would pile up one Popen + two pipes
                    # + the output buffers forever.  FIFO: the oldest
                    # un-converged victim is dropped, its pipes closed so a
                    # later write EPIPEs instead of blocking the orphan.
                    while len(self._stale_jobs) > self.max_jobs:
                        _oldest_id, _oldest_job = next(iter(self._stale_jobs.items()))
                        del self._stale_jobs[_oldest_id]
                        _close_job_pipes(_oldest_job)
                        logger.warning(
                            "Dropped oldest stale job %s (cap %d): still alive after SIGKILL",
                            _oldest_id,
                            self.max_jobs,
                        )
                    logger.warning(
                        "Evicted job %s still alive after kill — re-tracking for reap: cmd=%.200s",
                        victim_id,
                        victim_cmd,
                    )
            logger.warning(
                "Killed oldest job to enforce max_jobs=%d: id=%s cmd=%.200s",
                self.max_jobs,
                victim_id,
                victim_cmd,
            )

        return job_id

    def get_info(self, job_id: str) -> BackgroundJobInfo | None:
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
        # drain pipe → accumulates into _stdout_buf/_stderr_buf.
        # read_output() swallows UnicodeDecodeError/OSError internally.
        job.read_output()

        if status in ("completed", "failed", "killed"):
            # final drain (process is dead, pipe is flushing)
            job.read_output()

        stdout = job._stdout_buf.text()
        stderr = strip_malloc_noise(job._stderr_buf.text())

        return BackgroundJobInfo(
            job_id=job_id,
            command=command,
            pid=pid,
            status=status,
            elapsed=job.elapsed,
            stdout=stdout or "",
            stderr=stderr or "",
            stdout_total=job._stdout_buf.total,
            stderr_total=job._stderr_buf.total,
        )

    def wait_for_completion(
        self, job_id: str, timeout: float = 120.0, poll_interval: float = 1.0, cancel_event: Any | None = None
    ) -> BackgroundJobInfo | None:
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

        A transient ``None`` (a job that is being evicted over capacity and
        is mid-kill) is tolerated for a short grace window instead of being
        reported as "not found" — the eviction placeholder in
        ``_reaped_results`` normally covers this, but a caller that started
        polling before the placeholder was written needs the retry.  The
        grace never outlives *timeout*: a job that is still not found when
        the deadline arrives returns ``None`` at the deadline.
        """
        deadline = time.monotonic() + timeout
        none_since: float | None = None
        while True:
            info = self.get_info(job_id)
            if info is None:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    return None  # timeout — job still not found
                if none_since is None:
                    none_since = now
                if now - none_since >= min(3.0, remaining):
                    return None  # genuinely not found — give up
                if interruptible_sleep(min(0.1, poll_interval), cancel_event):
                    return None  # cancelled — abandon
                continue
            none_since = None
            if info.status not in ("running", "killing"):
                # "killing" is the transient eviction placeholder — the final
                # status lands right after the kill completes outside the
                # lock, so keep polling instead of returning it as final.
                return info
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return info  # timeout — return current snapshot
            if interruptible_sleep(min(poll_interval, remaining), cancel_event):
                return info  # cancelled — return current snapshot

    def list_jobs(self, include_completed: bool = True) -> list[BackgroundJobInfo]:
        """List all tracked jobs, lazily reaping old completed ones first.

        The list merges two registries: active jobs in ``_jobs`` and the
        ``_reaped_results`` ring.  Reaped jobs (cleanup / capacity eviction /
        stale convergence) stay retrievable via :meth:`get_info`, so dropping
        them from the list would hide final states a caller is still waiting
        on — the merged ring keeps ``action=list`` a complete picture.

        I/O (``poll_status``, ``read_output``) is performed *outside*
        ``self._lock`` so that pipe reads do not block other callers.

        Preview fields are the TAIL of each buffer (``_tail_preview``): a
        running job's verdict — build result, test summary — arrives last,
        so a head slice would freeze the preview on boilerplate.  This makes
        ``action=list`` a usable progress signal without a separate
        ``action=output`` call.
        """
        self._maybe_reap()
        with self._lock:
            # Snapshot immutable fields under the lock
            snapshots = [(job_id, job, job.command, job.proc.pid) for job_id, job in list(self._jobs.items())]
            reaped = list(self._reaped_results.values())

        # ── I/O outside the lock ──
        infos = []
        for job_id, job, command, pid in snapshots:
            status = job.poll_status()
            if not include_completed and status in ("completed", "failed", "killed"):
                continue
            # drain pipe → accumulates into buffer (read_output() swallows
            # UnicodeDecodeError/OSError internally)
            job.read_output()
            stdout = job._stdout_buf.text()
            stderr = strip_malloc_noise(job._stderr_buf.text())
            infos.append(
                BackgroundJobInfo(
                    job_id=job_id,
                    command=command,
                    pid=pid,
                    status=status,
                    elapsed=job.elapsed,
                    stdout=_tail_preview(stdout or ""),
                    stderr=_tail_preview(stderr or ""),
                    stdout_total=job._stdout_buf.total,
                    stderr_total=job._stderr_buf.total,
                )
            )

        # ── Reaped-results ring (already-snapshotted; no I/O needed) ──
        # Terminal entries respect include_completed; the transient
        # "killing"/"running" placeholders of evicted victims are NOT
        # terminal, so they stay listed for include_completed=False too.
        # Dedup against active ids: no current path creates the overlap, but
        # a duplicate entry in a job list is a silent correctness trap.
        active_ids = {job_id for job_id, *_ in snapshots}
        if include_completed:
            infos.extend(r for r in reaped if r.job_id not in active_ids)
        else:
            infos.extend(
                r for r in reaped if r.status not in ("completed", "failed", "killed") and r.job_id not in active_ids
            )
        return infos

    def kill(self, job_id: str) -> str | None:
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
            self._jobs = OrderedDict((jid, j) for jid, j in self._jobs.items() if j.poll_status() == "running")
            removed = before - len(self._jobs)
            if removed:
                logger.debug("Cleaned up %d background job(s)", removed)
            return removed

    def _reap_stale(self) -> None:
        """Converge stale victims: evicted jobs whose kill could not finish.

        Such a job lives only in ``_stale_jobs`` plus a "running" ring
        placeholder.  Every reap tick, settle its status: when the process
        finally dies (e.g. a D-state process that SIGKILL could not
        interrupt), snapshot the REAL terminal status into the ring,
        replacing the placeholder — otherwise the ring would serve
        "running" forever.  While still alive, keep draining its pipes so a
        stuck-but-productive process cannot deadlock on a full pipe buffer.
        Deliberately no kill retry: SIGKILL was already delivered, and a
        process that survived it is uninterruptible.
        """
        with self._lock:
            stale = list(self._stale_jobs.items())
        for jid, job in stale:
            try:
                job._settle_status()
                if job.status in ("completed", "failed", "killed"):
                    job.read_output()  # final drain — process is dead, pipe flushing
                    with self._lock:
                        if jid in self._stale_jobs:  # still tracked
                            del self._stale_jobs[jid]
                            info = self._snapshot_job_locked(jid, job)
                            self._store_reaped_locked(jid, info)
                            logger.debug("Stale job %s converged to %s", jid, job.status)
                else:
                    job.read_output()  # still alive — keep the pipes drained
            except Exception:
                # Never let one unreadable pipe abort the sweep for the rest;
                # the next tick retries.
                logger.debug("Stale-job reap failed for %s", jid, exc_info=True)

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

        A running victim is pre-snapshotted into ``_reaped_results`` (status
        ``"killing"``) *before* the lock is released: the caller kills it
        outside the lock for up to ~6 s, and without this placeholder the
        victim would be in neither registry during that window — get_info()
        would answer "not found" for a job that is alive and dying.
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
                    # Pre-snapshot NOW, under the lock, with an explicit
                    # "killing" marker.  The victim leaves _jobs immediately
                    # but the actual kill runs outside the lock (up to ~6 s);
                    # the placeholder keeps get_info()/wait_for_completion()
                    # from reporting "not found" in that window.  The caller
                    # overwrites it with the final status after the kill.
                    job.status = "killing"
                    info = self._snapshot_job_locked(jid, job)
                    self._store_reaped_locked(jid, info)
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
        self._reap_stale()

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

        Safe to call multiple times. Does not touch tracked (active) jobs;
        stale re-tracking is dropped — the reaper that would converge it is
        stopped, so the dict would otherwise pin Popen + pipes forever.
        """
        with self._lock:
            self._reaper_active = False
            t = self._reaper_timer
            self._reaper_timer = None
            # F4: no reaper tick will ever converge these again — drop them
            # now (pipes closed so an orphan's write EPIPEs instead of
            # blocking).  Safe against a mid-settle _reap_stale: its
            # `jid in self._stale_jobs` re-check under this lock no-ops.
            for _job in self._stale_jobs.values():
                _close_job_pipes(_job)
            self._stale_jobs.clear()
        if t is not None:
            t.cancel()


# Module-level singleton for shared use across tool instances
_global_bg_manager: BackgroundJobManager | None = None
_global_bg_manager_lock = threading.Lock()


def get_global_background_job_manager(max_jobs: int = 5) -> BackgroundJobManager:
    """Get or create the global BackgroundJobManager singleton.

    ToolRegistry clones (subagents) share this singleton so that
    background jobs survive subagent lifecycle.

    Only the FIRST call's *max_jobs* takes effect — the singleton is created
    once.  Later calls with a different value log a warning instead of
    silently ignoring the caller's intent.
    """
    global _global_bg_manager
    if _global_bg_manager is None:
        with _global_bg_manager_lock:
            if _global_bg_manager is None:
                _global_bg_manager = BackgroundJobManager(max_jobs=max_jobs)
    elif max_jobs != _global_bg_manager.max_jobs:
        logger.warning(
            "get_global_background_job_manager(max_jobs=%d) ignored: singleton already created with max_jobs=%d",
            max_jobs,
            _global_bg_manager.max_jobs,
        )
    return _global_bg_manager
