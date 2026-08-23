"""Regression guard for BackgroundJobManager.

Covers the max_jobs hard-bound fix: ``start()`` must insert the new job and
decide any over-capacity eviction in a SINGLE ``self._lock`` acquisition, so
concurrent starters cannot each observe a post-eviction (lower) count and
collectively insert past the cap. Before the fix, ``_evict_if_needed`` ran in
a separate lock hold from the insertion; a scheduling preemption between the
two could let many threads pass eviction at a sub-max count and then all
insert, transiently blowing ``len(_jobs)`` far past ``max_jobs``.

These tests also guard the behavioural contract: finished jobs are evicted
before killing a running one, the kill still happens outside the manager
lock, and the public API (get_info / list_jobs / kill / cleanup) is intact.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

from external_llm.agent import background_job_manager as bjm
from external_llm.agent.background_job_manager import BackgroundJobManager


class _FakeProc:
    """Minimal subprocess.Popen stand-in.

    ``poll()`` returning ``None`` keeps the job "running"; returning ``0``
    marks it completed. A blocking ``wait()`` simulates the real SIGTERM/SIGKILL
    teardown (up to ~6 s in production) that widens the race window the
    hard-bound fix closes.
    """

    _next_pid = 2_000_000

    def __init__(
        self,
        *,
        done: bool = False,
        kill_delay: float = 0.0,
        returncode: int = 0,
        wait_returncode: int | None = None,
        stuck: bool = False,
    ):
        _FakeProc._next_pid += 1
        self.pid = _FakeProc._next_pid
        self.stdout = None
        self.stderr = None
        self._done = done
        self._kill_delay = kill_delay
        self._returncode = returncode
        # Simulates a leader that ALREADY exited (its real rc) before the
        # kill attempt — the F1/R2 TOCTOU window.  wait() must then report
        # the real exit code and kill() must NOT overwrite it.  None → the
        # process dies FROM our SIGKILL (real signal death, rc = -9).
        self._wait_returncode = wait_returncode
        # Simulates a process that SURVIVES SIGKILL (uninterruptible D-state):
        # kill() is a no-op and wait() raises TimeoutExpired forever — the R3
        # sticky-"killing" window.  Tests flip this off to let the process
        # finally die (with a chosen rc) so the reaper can converge.
        self._stuck = stuck

    def poll(self):
        return self._returncode if self._done else None

    def kill(self):
        if self._stuck:
            return  # SIGKILL delivered but the process is in D-state — survives
        self._done = True
        if self._wait_returncode is None:
            self._returncode = -9  # killed by our SIGKILL
        else:
            self._returncode = self._wait_returncode  # already dead — rc preserved

    def wait(self, timeout=None):
        if self._stuck:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        if self._kill_delay:
            time.sleep(self._kill_delay)
        return self._returncode


@pytest.fixture(autouse=True)
def _force_kill_through_fake_proc(monkeypatch):
    """Route BackgroundJob.kill() through the fake proc by making os.getpgid
    raise ProcessLookupError, so tests never touch a real OS process group."""
    monkeypatch.setattr(bjm.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    # Silence the per-kill warning spam.
    monkeypatch.setattr(bjm.logger, "level", 40)


def _peak_job_count(mgr, stop):
    """Sample len(_jobs) under the lock until `stop` is set; return the peak."""
    peak = 0
    while not stop[0]:
        with mgr._lock:
            peak = max(peak, len(mgr._jobs))
        time.sleep(0.0003)
    with mgr._lock:
        return max(peak, len(mgr._jobs))


@pytest.mark.parametrize("max_jobs", [1, 2, 4])
def test_max_jobs_is_hard_bound_under_concurrent_start(max_jobs):
    """Concurrent start() with a blocking kill must never let len(_jobs)
    exceed max_jobs — the count visible to any lock holder stays bounded."""
    mgr = BackgroundJobManager(max_jobs=max_jobs, reap_interval=9999.0)
    try:
        for _ in range(max_jobs):
            mgr.start("pre", _FakeProc(kill_delay=0.15))

        stop = [False]
        mon = threading.Thread(target=lambda: None, daemon=True)  # placeholder

        def worker(i):
            mgr.start(f"c{i}", _FakeProc(kill_delay=0.15))

        peak_box = [0]

        def monitor():
            peak_box[0] = _peak_job_count(mgr, stop)

        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stop[0] = True
        mon.join(timeout=2.0)

        assert peak_box[0] <= max_jobs, f"max_jobs={max_jobs} violated: observed peak={peak_box[0]}"
    finally:
        mgr.shutdown()


def test_finished_job_evicted_before_killing_runner():
    """At capacity, a finished job must be evicted in preference to killing a
    running job (and the just-started job must never be the victim)."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        done_id = mgr.start("done", _FakeProc(done=True))
        run_id = mgr.start("run", _FakeProc(done=False))
        new_id = mgr.start("new", _FakeProc(done=False))

        ids = set(mgr._jobs.keys())
        assert done_id not in ids, "finished job should have been evicted"
        assert run_id in ids, "running job must not have been killed"
        assert new_id in ids, "just-started job must survive"
        assert len(ids) == 2
    finally:
        mgr.shutdown()


def test_public_api_contract():
    """get_info / list_jobs / kill / cleanup remain functional."""
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        jid = mgr.start("job", _FakeProc(done=False))

        info = mgr.get_info(jid)
        assert info is not None
        assert info.job_id == jid
        assert info.status == "running"

        listed = mgr.list_jobs()
        assert any(j.job_id == jid for j in listed)

        final = mgr.kill(jid)
        assert final == "killed"

        removed = mgr.cleanup()
        assert isinstance(removed, int)
    finally:
        mgr.shutdown()


def test_get_info_unknown_returns_none():
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        assert mgr.get_info("nope") is None
        assert mgr.kill("nope") is None
    finally:
        mgr.shutdown()


# ── Output accumulation / recovery (real subprocesses) ───────────────────────


def _real_proc(script: str):
    import subprocess

    return subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )


def test_output_survives_intermediate_reads():
    """Regression (Bug #1): output drained by list_jobs()/an early get_info()
    must still be returned by a later get_info() — reads accumulate instead of
    consuming."""
    proc = _real_proc("echo PHASE1; sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        jid = mgr.start("phased", proc)
        deadline = time.monotonic() + 5
        while "PHASE1" not in mgr.get_info(jid).stdout:  # early drain
            assert time.monotonic() < deadline, "PHASE1 never arrived"
            time.sleep(0.05)
        mgr.list_jobs()  # drains again — historically discarded the data
        assert "PHASE1" in mgr.get_info(jid).stdout, "accumulated output lost by an intermediate drain"
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_pre_timeout_output_rides_the_handover_into_the_job_buffer():
    """Regression (Bug #2): what the command printed before the timeout must
    reach the job's accumulated buffer.

    The bash tool now reads the pipes itself into a bounded capture, so the
    hand-over is a plain assignment rather than an excavation of CPython's
    private ``_fileobj2output`` — but the contract the CONSUMER implements is
    unchanged, and it is the consumer that had the defect. Set here the way
    ``_tool_shell_exec`` sets it.

    Production invariant the fixture must respect: by the time
    ``_capture_bounded`` hands over, the recovered text has ALREADY been
    consumed from the pipes, so recovered data and post-transition pipe data
    are DISJOINT. The child therefore prints distinct text (PIPE_OUT/PIPE_ERR)
    to the real pipes; reusing the recovered string would make the count
    assertion below racy — a second copy can land in the pipe on either side
    of the first drain depending on scheduling.
    """
    proc = _real_proc("echo PIPE_OUT; echo PIPE_ERR >&2; sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        proc._recovered_stdout = "EARLY_OUT\n"
        proc._recovered_stderr = "EARLY_ERR\n"

        jid = mgr.start("timed-out", proc)
        info = mgr.get_info(jid)
        assert "EARLY_OUT" in info.stdout
        assert "EARLY_ERR" in info.stderr
        # Recovered data is consumed exactly once — not duplicated on re-read.
        info2 = mgr.get_info(jid)
        assert info2.stdout.count("EARLY_OUT") == 1
        # Post-transition pipe output still rides into the same buffer
        # (disjoint from the recovered text, as in production).
        deadline = time.monotonic() + 5
        while True:
            cur = mgr.get_info(jid)
            if "PIPE_OUT" in cur.stdout and "PIPE_ERR" in cur.stderr:
                break
            assert time.monotonic() < deadline, "post-transition output never arrived"
            time.sleep(0.05)
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_output_buffer_head_and_tail_under_cap():
    """The accumulated buffer (a _BoundedCapture) keeps head AND tail under
    the cap: the stream START survives a >cap stream, not just the tail —
    _truncate_bash_output's leading half (pytest's failing command) is only
    as good as what the buffer retained.  RED before F5: the old tail-only
    _cap_tail string dropped the head entirely."""
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        jid = mgr.start("big", _FakeProc())
        job = mgr._jobs[jid]  # get() removed — internal job reached directly
        job._stdout_buf.feed("HEAD_MARKER\n" + "A" * (bjm._OUTPUT_BUF_CAP * 2) + "\nTAIL_MARKER")
        info = mgr.get_info(jid)
        assert info.stdout.startswith("HEAD_MARKER\n"), "stream head lost"
        assert info.stdout.endswith("TAIL_MARKER")
        assert "chars dropped" in info.stdout, "elision not announced"
    finally:
        mgr.shutdown()


def _noisy_proc(script: str):
    """Popen whose child emits real macOS libmalloc stack-logging chatter.

    ``MallocStackLogging=1`` makes libmalloc write its status lines to fd 2
    before exec, so they land in the captured stderr pipe exactly as they do
    when the parent has stack logging enabled by some non-environment route.
    """
    import os
    import subprocess

    env = os.environ.copy()
    env["MallocStackLogging"] = "1"
    return subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
        env=env,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="libmalloc noise is macOS-only")
def test_malloc_noise_stripped_from_background_job_output():
    """Regression: the blocking bash path filtered MallocStackLogging lines but
    the timeout→background path did not, so the noise reappeared in every
    `job output` / `job list` result. Both consumption points must filter."""
    proc = _noisy_proc("echo REAL_OUT; echo REAL_ERR >&2; sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        jid = mgr.start("noisy", proc)
        deadline = time.monotonic() + 5
        while "REAL_ERR" not in mgr.get_info(jid).stderr:
            assert time.monotonic() < deadline, "REAL_ERR never arrived"
            time.sleep(0.05)

        # The raw buffer must actually contain noise, else the test is vacuous.
        assert "MallocStackLogging" in mgr._jobs[jid]._stderr_buf.text(), (
            "no libmalloc noise produced — test would pass vacuously"
        )

        info = mgr.get_info(jid)
        assert "MallocStackLogging" not in info.stderr
        assert "REAL_ERR" in info.stderr, "real stderr destroyed by the filter"
        assert "REAL_OUT" in info.stdout

        listed = next(j for j in mgr.list_jobs() if j.job_id == jid)
        assert "MallocStackLogging" not in listed.stderr
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_strip_malloc_noise_is_line_exact():
    """Only whole noise lines are dropped; a chunk-straddled noise line
    self-heals because filtering runs on the accumulated buffer, not on the
    per-read chunk."""
    s = bjm.strip_malloc_noise
    assert s("") == ""
    assert s("clean\noutput\n") == "clean\noutput\n"
    assert s("a\nsh(1) MallocStackLogging: z\nb\n") == "a\nb\n"
    assert s("keep\r\nsh(2) MallocStackLogging: q\r\n") == "keep\r\n"
    assert s("no trailing newline") == "no trailing newline"
    # Half a noise line has no token yet, so it survives this drain...
    half = "sh(1) MallocStack"
    assert s(half) == half
    # ...and is removed once the remainder lands in the same buffer.
    assert s(half + "Logging: x\nkeep\n") == "keep\n"


# ── Reaped-result ring buffer ─────────────────────────────────────────────
#
# `bash` hands a timed-out command to this manager and returns a Job ID for the
# agent to poll later. The periodic reaper deletes finished jobs, so without a
# retention window the agent's own result answered "not found". Two things have
# to hold for the ring to be worth anything: the output must be READ from the
# pipe before the snapshot, and what survives must be the TAIL.


def _real_job(mgr, script: str, drain: bool = False) -> str:
    """Start a real subprocess job and wait for it to exit.

    With ``drain=False`` the pipe is deliberately NOT read while waiting —
    get_info()/list_jobs() drain as a side effect and would mask the
    walk-away workflow (agent starts a job, goes off, comes back after the
    reaper ran). Only safe for payloads that fit in the ~64 KiB OS pipe buffer.

    ``drain=True`` mirrors what the production reaper does every tick
    (``_reap_tick`` drains running jobs precisely to avoid pipe-full deadlock);
    required for payloads larger than the pipe buffer, which would otherwise
    block in ``write()`` and never exit.
    """
    import subprocess as _sp

    proc = _sp.Popen(["bash", "-c", script], stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
    job_id = mgr.start(script, proc)
    job = mgr._jobs[job_id]  # get() removed — white-box tests reach the internal job directly
    deadline = time.monotonic() + 30
    while proc.poll() is None and time.monotonic() < deadline:
        if drain:
            job.read_output()
        time.sleep(0.02)
    return job_id


def test_reaped_job_output_survives_when_never_polled_before_cleanup():
    """`_snapshot_job_locked` does no I/O, so whatever `read_output()` had not
    yet pulled out of the pipe was simply absent. Cleanup must drain first, or
    the ring preserves an empty string — a retrieval that *looks* successful
    while having lost the entire result."""
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        job_id = _real_job(mgr, "sleep 0.2; echo BUILD_OK_FINAL")
        assert mgr.cleanup() == 1
        info = mgr.get_info(job_id)
        assert info is not None, "reaped job vanished entirely"
        assert info.status == "completed"
        assert "BUILD_OK_FINAL" in info.stdout, f"output lost: {info.stdout!r}"
    finally:
        mgr.shutdown()


def test_reaped_output_keeps_the_tail_not_the_head():
    """A job is backgrounded because it is long, and a long command's answer is
    at the END. A leading slice preserves boilerplate and drops the verdict."""
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        job_id = _real_job(
            mgr,
            "for i in $(seq 1 4000); do echo line-$i-xxxxxxxxxxxxxxxxxxxx; done; echo FINAL_ANSWER_42",
            drain=True,  # payload exceeds the OS pipe buffer
        )
        mgr.cleanup()
        info = mgr.get_info(job_id)
        assert info is not None
        assert len(info.stdout) > bjm._REAPED_OUTPUT_CAP, "test payload too small to elide"
        assert "FINAL_ANSWER_42" in info.stdout, "the tail (the answer) was dropped"
        assert info.stdout.startswith(bjm._TRUNCATION_MARKER), "elision not announced"
        assert "line-1-" not in info.stdout, "kept the head instead of the tail"
    finally:
        mgr.shutdown()


def test_reaped_ring_is_bounded_and_evicts_oldest():
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        first = _real_job(mgr, "echo one")
        mgr.cleanup()
        for _ in range(bjm._REAPED_RESULTS_MAX):
            _real_job(mgr, "echo x")
            mgr.cleanup()
        assert len(mgr._reaped_results) <= bjm._REAPED_RESULTS_MAX
        assert mgr.get_info(first) is None, "oldest entry was not evicted"
    finally:
        mgr.shutdown()


def test_short_output_is_preserved_verbatim():
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        job_id = _real_job(mgr, "echo hello")
        mgr.cleanup()
        info = mgr.get_info(job_id)
        assert info.stdout == "hello\n"
        assert not info.stdout.startswith(bjm._TRUNCATION_MARKER)
    finally:
        mgr.shutdown()


# ── the salvage was inert for a stdout-only command ────────────────────────
# The test above passes and always did, because its script writes to BOTH
# streams. That is the one shape in which the defect cannot fire: the producer
# set each attribute only when its stream was non-empty, and `read_output`
# fetched the pair with plain attribute access, stderr second — so an absent
# `_recovered_stderr` raised AttributeError and discarded the already-fetched
# stdout. A command that writes to stdout and not to stderr is the common case,
# so the feature was inert in normal use while its own test stayed green. Both
# ends are fixed, and both are still pinned: the producer always writes the
# pair, and the consumer below survives one of them being missing outright.


@pytest.mark.parametrize(
    "script,marker,stream",
    [
        ("echo ONLY_OUT; sleep 30", "ONLY_OUT", "stdout"),
        ("echo ONLY_ERR >&2; sleep 30", "ONLY_ERR", "stderr"),
    ],
    ids=["stdout-only", "stderr-only"],
)
def test_partial_output_survives_when_one_stream_is_silent(script, marker, stream):
    proc = _real_proc(script)
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        proc._recovered_stdout = marker + "\n" if stream == "stdout" else ""
        proc._recovered_stderr = marker + "\n" if stream == "stderr" else ""

        jid = mgr.start("timed-out", proc)
        info = mgr.get_info(jid)
        assert marker in getattr(info, stream), (
            f"pre-timeout {stream} was salvaged and then dropped: stdout={info.stdout!r} stderr={info.stderr!r}"
        )
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_a_missing_recovery_attribute_does_not_discard_the_other():
    """The consumer must tolerate one name being absent entirely.

    The asymmetry was the whole bug — a reader that fetched the pair with plain
    attribute access had no way to tell "nothing on this stream" from "attribute
    missing", and threw away the stdout it had already fetched. Asserted against
    the reader rather than any one producer, so it holds for whatever sets it.
    """
    proc = _real_proc("sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        proc._recovered_stdout = "ONLY_OUT\n"
        assert not hasattr(proc, "_recovered_stderr")

        jid = mgr.start("half-recovered", proc)
        info = mgr.get_info(jid)
        assert "ONLY_OUT" in info.stdout, f"a missing _recovered_stderr discarded the recovered stdout: {info.stdout!r}"
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_read_output_tolerates_a_missing_recovery_attribute():
    """The reader is defensive on its own, not only because the producer is.

    Either end alone would let the other reintroduce the coupling, so the
    contract is pinned from both sides. Simulated directly rather than through
    a timeout, because the producer no longer creates this state.
    """
    proc = _real_proc("sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        jid = mgr.start("half-recovered", proc)
        job = mgr._jobs[jid]
        job.proc._recovered_stdout = "SALVAGED\n"
        # _recovered_stderr deliberately absent — the old shape.
        if hasattr(job.proc, "_recovered_stderr"):
            del job.proc._recovered_stderr
        out, _err = job.read_output()
        assert "SALVAGED" in out, "a missing stderr attribute discarded stdout"
        assert "SALVAGED" in job._stdout_buf.text()
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_empty_nonblocking_pipe_reads_as_empty_not_typeerror():
    """A would-block read must return "", on every supported interpreter.

    Below CPython 3.14 a non-blocking TEXT-mode pipe with nothing buffered
    signals "would block" by letting the raw layer return None; the incremental
    decoder then rejects that with ``TypeError: can't concat NoneType to bytes``.
    TypeError is neither OSError nor ValueError, so it escaped _read_fd's except
    clause AND read_output()'s suppress(), propagated out of get_info(), and
    surfaced to the model as a failed job_output tool call — on what is simply
    an idle job. 3.14 raises BlockingIOError (an OSError subclass) for the same
    condition, which is exactly why the dev interpreter never reproduced it, so
    the TypeError is injected rather than provoked through a real empty pipe.
    """
    r_fd, w_fd = os.pipe()

    class _WouldBlockPipe:
        """Text-mode pipe that is open and empty, spelled the pre-3.14 way."""

        def fileno(self):
            return r_fd  # real fd: _read_fd toggles O_NONBLOCK on it

        def read(self):
            raise TypeError("can't concat NoneType to bytes")

    try:
        assert bjm.BackgroundJob._read_fd(_WouldBlockPipe()) == ""
    finally:
        os.close(r_fd)
        os.close(w_fd)


# ── F1: kill() must preserve the real exit status of an already-exited job ──


def test_kill_preserves_real_exit_status_of_finished_job():
    """kill() on a job whose process already exited must report the real
    outcome (completed/failed), not overwrite it with "killed".

    status is only refreshed by poll_status() (reaper tick = 30s), so a
    finished-but-unpolled job still reads "running".  Before the fix, kill()
    took the ProcessLookupError fallback (no-op on the dead leader) and
    forced status="killed", destroying rc=0 vs rc=7 — permanently, because
    poll_status() early-returns on terminal states.
    """
    mgr = BackgroundJobManager(max_jobs=4, reap_interval=9999.0)
    try:
        ok_id = mgr.start("ok", _FakeProc(done=True, returncode=0))
        bad_id = mgr.start("bad", _FakeProc(done=True, returncode=7))

        # Neither job was ever polled — both still read "running" internally
        # (get_info() would poll and flip them, so inspect the raw state).
        assert mgr._jobs[ok_id].status == "running"
        assert mgr._jobs[bad_id].status == "running"

        assert mgr.kill(ok_id) == "completed"
        assert mgr.kill(bad_id) == "failed"
    finally:
        mgr.shutdown()


# ── R2: the kill() fallback must preserve the real rc (TOCTOU residual) ──


def test_kill_fallback_preserves_real_exit_code_in_toctou_window():
    """R2 hardening: F1's pre-poll closes the wide window, but the leader can
    still exit BETWEEN poll() and killpg().  The fallback wait() then returns
    the REAL exit code — it must be preserved (rc=7 -> "failed"), not
    overwritten with "killed", or the rc distinction F1 protects is destroyed
    again in the narrow window.
    """
    mgr = BackgroundJobManager(max_jobs=4, reap_interval=9999.0)
    try:
        # wait_returncode=7 simulates the leader exiting with rc=7 after the
        # pre-poll saw it "running" but before killpg() failed with ESRCH.
        jid = mgr.start("j", _FakeProc(wait_returncode=7))

        # Job still reads "running" — the pre-poll (poll() -> None) does not
        # catch it; the fallback must.
        assert mgr._jobs[jid].status == "running"

        assert mgr.kill(jid) == "failed", "TOCTOU rc=7 must survive as 'failed'"
        assert mgr._jobs[jid].status == "failed"
    finally:
        mgr.shutdown()


def test_kill_fallback_signal_death_stays_killed():
    """A process we actually SIGKILL in the fallback (rc = -9, signal death)
    must still read "killed", not "failed" — rc<0 maps to the killed label,
    mirroring the main-path semantics."""
    mgr = BackgroundJobManager(max_jobs=4, reap_interval=9999.0)
    try:
        jid = mgr.start("j", _FakeProc())
        assert mgr.kill(jid) == "killed"
        assert mgr._jobs[jid].status == "killed"
    finally:
        mgr.shutdown()


# ── R3: the "killing" placeholder must never become a terminal state ──


def test_stuck_eviction_victim_does_not_freeze_as_killing():
    """R3: an eviction victim whose process survives SIGKILL (D-state) must
    not freeze in the ring as "killing" forever.  kill() settles to the
    honest status, start() re-tracks the victim, and the reaper converges
    the ring to the real outcome when the process finally dies."""
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    try:
        proc = _FakeProc(stuck=True)
        victim_id = mgr.start("victim", proc)
        mgr.start("new", _FakeProc())  # evicts the victim — the kill cannot finish

        assert victim_id not in mgr._jobs, "victim should have been evicted"
        assert victim_id in mgr._stale_jobs, "stuck victim was not re-tracked"
        info = mgr.get_info(victim_id)
        assert info is not None and info.status == "running", f"victim frozen as {info.status if info else None!r}"

        # The process finally dies from the SIGKILL that was delivered while
        # it was stuck — the reaper must converge the ring to the REAL
        # outcome instead of serving the placeholder forever.
        proc._stuck = False
        proc._done = True
        proc._returncode = -9
        mgr._reap_stale()

        assert victim_id not in mgr._stale_jobs, "victim never converged"
        final = mgr.get_info(victim_id)
        assert final is not None and final.status == "killed", (
            f"expected killed, got {final.status if final else None!r}"
        )
        waited = mgr.wait_for_completion(victim_id, timeout=2.0, poll_interval=0.01)
        assert waited is not None and waited.status == "killed"
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("rc,expected", [(-9, "killed"), (0, "completed"), (7, "failed")])
def test_stale_victim_converges_to_real_exit_code(rc, expected):
    """R3: the reaper's convergence classifies the victim's REAL exit code —
    signal death -> killed, 0 -> completed, positive -> failed."""
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    try:
        proc = _FakeProc(stuck=True)
        victim_id = mgr.start("victim", proc)
        mgr.start("new", _FakeProc())
        assert victim_id in mgr._stale_jobs

        proc._stuck = False
        proc._done = True
        proc._returncode = rc
        mgr._reap_stale()

        final = mgr.get_info(victim_id)
        assert final is not None and final.status == expected, (
            f"rc={rc}: expected {expected}, got {final.status if final else None!r}"
        )
    finally:
        mgr.shutdown()


# ── F4: _stale_jobs must be bounded like every other registry ───────────────


def test_stale_jobs_capped_at_max_jobs_fifo():
    """F4: a SIGKILL-surviving victim (D-state) can outlive the whole session,
    and without a cap every over-capacity start piles up one Popen + two pipes
    + buffers forever.  FIFO: the oldest un-converged victim is dropped first,
    and the ring still serves its placeholder (visibility is not lost)."""
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    try:
        j1 = mgr.start("v1", _FakeProc(stuck=True))
        j2 = mgr.start("v2", _FakeProc(stuck=True))
        mgr.start("v3", _FakeProc(stuck=True))  # evicts j2; its kill also sticks
        with mgr._lock:
            assert len(mgr._stale_jobs) == 1, f"_stale_jobs grew past max_jobs: {list(mgr._stale_jobs)}"
            assert list(mgr._stale_jobs) == [j2], "FIFO: oldest (j1) must drop, j2 kept"
        # The dropped victim's last snapshot stays retrievable via the ring.
        info = mgr.get_info(j1)
        assert info is not None and info.status == "running"
    finally:
        mgr.shutdown()


def test_shutdown_clears_stale_jobs():
    """F4: shutdown stops the reaper that would converge stale victims — the
    dict must not keep pinning Popen + pipes after the manager is gone."""
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    mgr.start("v1", _FakeProc(stuck=True))
    mgr.start("v2", _FakeProc(stuck=True))
    assert len(mgr._stale_jobs) == 1
    mgr.shutdown()
    assert mgr._stale_jobs == {}, "stale tracking survived shutdown"


def test_direct_kill_of_stuck_job_stays_running_not_killing():
    """R3: a DIRECT kill of a process that survives SIGKILL must keep the job
    honest — "running", still tracked in _jobs, reaped when it finally dies.
    It must never read "killing" (an eviction-only placeholder) or a bogus
    "killed"."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        jid = mgr.start("stuck", _FakeProc(stuck=True))
        assert mgr.kill(jid) == "running", "stuck job must stay 'running'"
        assert jid in mgr._jobs, "stuck job must remain tracked"
        assert mgr._jobs[jid].status == "running"
        assert jid not in mgr._stale_jobs, "direct kill must not stale-track"
    finally:
        mgr.shutdown()


# ── F3: poll_status() must not block behind kill()'s job-lock hold ──


def test_poll_status_does_not_block_when_job_lock_is_held():
    """poll_status() must return the cached status immediately when kill()
    holds the job lock (up to ~6 s), instead of blocking the caller.

    cleanup()/list_jobs()/get_info() call poll_status(); a blocking acquire
    would freeze the whole manager behind one mid-teardown job.
    """
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        job_id = mgr.start("j", _FakeProc())
        job = mgr._jobs[job_id]  # get() removed — internal job reached directly

        job._lock.acquire()  # simulate kill() mid-teardown
        try:
            box: list = []
            t = threading.Thread(target=lambda: box.append(job.poll_status()))
            t.daemon = True
            t.start()
            t.join(timeout=0.5)
            assert not t.is_alive(), "poll_status blocked on the job lock"
            assert box == ["running"]  # cached status returned
        finally:
            job._lock.release()
    finally:
        mgr.shutdown()


# ── F2: an eviction victim stays visible (get_info/wait) during the kill ──


def test_eviction_victim_stays_visible_during_kill():
    """An over-capacity victim is pre-snapshotted (status="killing") before
    its ~seconds-long kill runs outside the lock — get_info() and
    wait_for_completion() must never report "not found" for it.
    """
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    try:
        victim_id = mgr.start("victim", _FakeProc(kill_delay=0.5))
        # Second start evicts the victim; the kill takes 0.5 s (kill_delay).
        start_box: list = []

        def _second_start():
            start_box.append(mgr.start("new", _FakeProc()))

        t = threading.Thread(target=_second_start)
        t.daemon = True
        t.start()

        # Wait until the victim has been popped from _jobs but the kill is
        # still in flight (kill_delay=0.5 widens the window).
        deadline = time.monotonic() + 3.0
        while victim_id in mgr._jobs and time.monotonic() < deadline:
            time.sleep(0.005)
        assert victim_id not in mgr._jobs, "victim should have been evicted"

        info = mgr.get_info(victim_id)
        assert info is not None, "victim invisible during eviction kill"
        assert info.status == "killing", f"expected killing, got {info.status}"

        # wait_for_completion must tolerate the transient state and resolve
        # to the final "killed" status rather than bailing with None.
        final = mgr.wait_for_completion(victim_id, timeout=5.0, poll_interval=0.02)
        assert final is not None, "wait_for_completion saw 'not found' mid-kill"
        assert final.status == "killed"

        t.join(timeout=3.0)
        assert not t.is_alive()
    finally:
        mgr.shutdown()


def test_wait_for_completion_nonexistent_job_respects_timeout():
    """R1 regression: the transient-None grace in wait_for_completion() must
    never outlive the caller's timeout budget.  Before the fix, a nonexistent
    job id always waited the full 3 s grace — a short timeout=0.2 was
    silently stretched to ~3 s, stalling the serial job tool mid-turn.
    """
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        start = time.monotonic()
        info = mgr.wait_for_completion("nonexistent", timeout=0.2, poll_interval=0.02)
        elapsed = time.monotonic() - start
        assert info is None
        assert elapsed < 1.0, f"grace outlived timeout: {elapsed:.2f}s"
    finally:
        mgr.shutdown()


def test_wait_for_completion_nonexistent_job_grace_is_capped():
    """Complementary guard: with a generous timeout the not-found grace is
    still capped at 3 s — a genuinely absent job must not block a long-waiting
    caller beyond the cap, and must not loop until the full timeout."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        start = time.monotonic()
        info = mgr.wait_for_completion("nonexistent", timeout=10.0, poll_interval=0.02)
        elapsed = time.monotonic() - start
        assert info is None
        assert 2.5 <= elapsed < 4.0, f"grace cap broken: {elapsed:.2f}s"
    finally:
        mgr.shutdown()


# ── F4: list_jobs() preview shows the TAIL of the output ──


def test_list_jobs_preview_is_tail_not_head():
    """list_jobs() preview must show the END of the output (where a long
    command's verdict — build result, test summary — lands), not the head.
    A head slice freezes the preview on boilerplate while the job runs.
    """
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        job_id = mgr.start("j", _FakeProc())
        job = mgr._jobs[job_id]  # get() removed — internal job reached directly
        job._stdout_buf.feed("P" * 500 + "\nBUILD-SUCCEEDED\n")

        infos = mgr.list_jobs()
        assert len(infos) == 1
        assert "BUILD-SUCCEEDED" in infos[0].stdout, "preview dropped the tail"
        assert not infos[0].stdout.startswith("P" * 200), "preview kept the head"
    finally:
        mgr.shutdown()


# ── F5-followup: info carries the captures' TRUE totals ──


def test_info_reports_true_output_totals_beyond_cap():
    """BackgroundJobInfo must report how many characters the streams ACTUALLY
    produced (capture ``total``), not how many survived the bounded capture —
    the render-time truncation notice names this number, and a job that wrote
    more than _OUTPUT_BUF_CAP would otherwise be described by the size of the
    elided remainder."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        job_id = mgr.start("j", _FakeProc())
        job = mgr._jobs[job_id]
        job._stdout_buf.feed("P" * (bjm._OUTPUT_BUF_CAP * 2 + 100_000))

        info = mgr.get_info(job_id)
        assert info.stdout_total == bjm._OUTPUT_BUF_CAP * 2 + 100_000
        assert len(info.stdout) < info.stdout_total, "capped text must not be mistaken for the true output size"

        listed = mgr.list_jobs()
        assert listed[0].stdout_total == bjm._OUTPUT_BUF_CAP * 2 + 100_000
    finally:
        mgr.shutdown()


# ── C1: list_jobs() merges the reaped-results ring ──


def test_list_jobs_includes_reaped_jobs_after_cleanup():
    """A finished job moved to the reaped-results ring by cleanup() must
    still appear in list_jobs() — get_info() answers it by id, so dropping
    it from the list breaks the 'list all tracked jobs' contract."""
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        done_id = mgr.start("done", _FakeProc(done=True, returncode=0))
        run_id = mgr.start("run", _FakeProc(done=False))

        removed = mgr.cleanup()
        assert removed == 1
        assert done_id not in mgr._jobs
        assert done_id in mgr._reaped_results  # sanity: ring holds the final state

        listed = mgr.list_jobs()  # include_completed=True is the default
        by_id = {j.job_id: j for j in listed}
        assert done_id in by_id, "reaped job missing from list_jobs()"
        assert by_id[done_id].status == "completed"
        assert run_id in by_id
    finally:
        mgr.shutdown()


def test_list_jobs_includes_evicted_jobs_after_capacity_eviction():
    """Capacity eviction also moves finished jobs to the ring; the list
    must keep showing them."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        done_id = mgr.start("done", _FakeProc(done=True))
        mgr.start("run", _FakeProc(done=False))
        mgr.start("new", _FakeProc(done=False))  # evicts "done" over capacity

        assert done_id not in mgr._jobs
        assert done_id in mgr._reaped_results

        listed = mgr.list_jobs()
        assert done_id in {j.job_id for j in listed}
    finally:
        mgr.shutdown()


def test_list_jobs_include_completed_false_excludes_reaped_terminal():
    """include_completed=False must also filter terminal entries coming from
    the reaped ring, not just the active registry."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        done_id = mgr.start("done", _FakeProc(done=True))
        run_id = mgr.start("run", _FakeProc(done=False))
        mgr.start("new", _FakeProc(done=False))  # evicts "done"

        listed = mgr.list_jobs(include_completed=False)
        ids = {j.job_id for j in listed}
        assert done_id not in ids, "terminal reaped job must be filtered out"
        assert run_id in ids
        assert len(listed) == 2  # run + new (both running)
    finally:
        mgr.shutdown()


def test_list_jobs_include_completed_false_keeps_stale_running_placeholder():
    """A stuck evicted victim is re-tracked in the ring as a 'running'
    placeholder (R3) — it is NOT terminal, so include_completed=False must
    still list it."""
    proc = _FakeProc(stuck=True)
    mgr = BackgroundJobManager(max_jobs=1, reap_interval=9999.0)
    try:
        victim_id = mgr.start("victim", proc)
        mgr.start("new", _FakeProc())  # evicts the victim — kill cannot finish

        assert victim_id in mgr._reaped_results
        listed = mgr.list_jobs(include_completed=False)
        assert victim_id in {j.job_id for j in listed}
    finally:
        proc._stuck = False
        mgr.shutdown()


def test_list_jobs_deduplicates_active_and_reaped_overlap():
    """Defensive: if a job_id somehow exists in BOTH registries, the list
    must not show it twice (no current path creates the overlap, but a
    duplicate entry is a silent correctness trap for callers)."""
    mgr = BackgroundJobManager(max_jobs=2, reap_interval=9999.0)
    try:
        jid = mgr.start("j", _FakeProc(done=False))
        with mgr._lock:
            mgr._store_reaped_locked(jid, mgr._snapshot_job_locked(jid, mgr._jobs[jid]))

        listed = mgr.list_jobs()
        assert [j.job_id for j in listed].count(jid) == 1
    finally:
        mgr.shutdown()


# ── misc: get_global_background_job_manager warns on max_jobs mismatch ──


def test_global_manager_max_jobs_mismatch_warns(monkeypatch):
    """A later get_global_background_job_manager(max_jobs=...) call with a
    different value is silently ignored today (singleton); it must at least
    log a warning instead of hiding the caller's intent."""
    monkeypatch.setattr(bjm, "_global_bg_manager", None)
    warnings: list = []
    monkeypatch.setattr(bjm.logger, "warning", lambda *a, **k: warnings.append(a))

    m1 = bjm.get_global_background_job_manager(max_jobs=2)
    m2 = bjm.get_global_background_job_manager(max_jobs=8)

    assert m2 is m1  # singleton preserved
    assert m1.max_jobs == 2  # first call's value wins
    assert warnings, "max_jobs mismatch must be logged"
