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

import logging
import os
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)

# Cap per accumulated output stream. The reaper drains pipes every tick, so a
# chatty long-running job (server logs, verbose build) would otherwise grow
# the in-RAM buffer without bound. Keep the TAIL — the most recent output is
# what a "what is this job doing" query needs.
_OUTPUT_BUF_CAP = 2 * 1024 * 1024  # 2 MiB
_TRUNCATION_MARKER = "…[oldest output truncated]…\n"


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


def recover_communicate_partial(proc) -> None:
    """Salvage output ``communicate(timeout=...)`` consumed before TimeoutExpired.

    ``communicate()`` reads from the pipes into an internal buffer and raises
    ``TimeoutExpired`` when the deadline passes — the data read so far lives in
    ``proc._fileobj2output`` and is NOT recoverable from the raw pipe fd (those
    bytes were already consumed). Stash it on ``proc._recovered_stdout`` /
    ``_recovered_stderr`` so :meth:`BackgroundJob.read_output` prepends it on
    the first drain after a timeout→background transition.

    ``_fileobj2output`` maps file object → LIST of raw bytes chunks (CPython
    appends each selector read; verified on 3.14). Treating the value as bytes
    silently no-ops via AttributeError — join the chunks instead, isinstance-
    guarding each so a future CPython switch to str chunks stays correct.
    Best-effort: relies on a CPython private attribute, so any failure
    degrades to the pre-recovery behavior (partial output lost).
    """
    def _join_chunks(chunks) -> str:
        return "".join(
            c.decode("utf-8", errors="replace") if isinstance(c, bytes) else c
            for c in (chunks or [])
            if isinstance(c, (bytes, str))
        )

    try:
        partial_out = _join_chunks(proc._fileobj2output.get(proc.stdout))
        partial_err = _join_chunks(proc._fileobj2output.get(proc.stderr))
        if partial_out:
            proc._recovered_stdout = partial_out
        if partial_err:
            proc._recovered_stderr = partial_err
    except (AttributeError, TypeError, ValueError):
        pass


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
            try:
                recovered_stdout = self.proc._recovered_stdout
                recovered_stderr = self.proc._recovered_stderr
                if recovered_stdout:
                    stdout += recovered_stdout
                    self.proc._recovered_stdout = ""
                if recovered_stderr:
                    stderr += recovered_stderr
                    self.proc._recovered_stderr = ""
            except AttributeError:
                pass

            if self.proc.stdout:
                try:
                    stdout += self._read_fd(self.proc.stdout)
                except Exception:
                    pass
            if self.proc.stderr:
                try:
                    stderr += self._read_fd(self.proc.stderr)
                except Exception:
                    pass
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
            return data if data else ""
        except (OSError, ValueError):
            return ""
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
                # Kill the process group to catch children
                import signal
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                # Give it a moment, then SIGKILL
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
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
        self._lock = threading.Lock()
        self._last_reap: float = 0.0
        self._reaper_timer: Optional[threading.Timer] = None
        self._reaper_active: bool = False

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
            logger.warning(
                "Killed oldest job to enforce max_jobs=%d: id=%s cmd=%.200s",
                self.max_jobs, victim_id, victim_cmd,
            )

        return job_id

    def get_info(self, job_id: str) -> Optional[BackgroundJobInfo]:
        """Get a snapshot of job state, or None if not found.

        I/O (``poll_status``, ``read_output``) is performed *outside*
        ``self._lock`` so that long-running pipe reads do not block
        other callers (``kill``, ``list_jobs``, etc.).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Snapshot mutable fields under the lock, then release
            command = job.command
            pid = job.proc.pid

        # ── I/O outside the lock ──
        status = job.poll_status()
        try:
            job.read_output()  # drain pipe → accumulates into _stdout_buf/_stderr_buf
        except Exception:
            pass

        if status in ("completed", "failed", "killed"):
            try:
                job.read_output()  # final drain (process is dead, pipe is flushing)
            except Exception:
                pass

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
                                poll_interval: float = 1.0) -> Optional[BackgroundJobInfo]:
        """Wait for a background job to finish (completed/failed/killed).

        Polls at *poll_interval* seconds until the job terminates or
        *timeout* seconds elapse.  Returns the final BackgroundJobInfo
        on completion, or None if the job is not found.

        If the timeout expires while the job is still running, returns
        the current snapshot (status == "running").
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
            time.sleep(min(poll_interval, remaining))
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
            try:
                job.read_output()  # drain pipe → accumulates into buffer
            except Exception:
                pass
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

    def cleanup(self) -> int:
        """Remove completed/failed/killed jobs from the registry. Returns count removed."""
        with self._lock:
            before = len(self._jobs)
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
                    try:
                        job.read_output()
                    except Exception:
                        pass
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
