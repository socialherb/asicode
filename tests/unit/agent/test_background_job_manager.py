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

    def __init__(self, *, done: bool = False, kill_delay: float = 0.0):
        _FakeProc._next_pid += 1
        self.pid = _FakeProc._next_pid
        self.stdout = None
        self.stderr = None
        self._done = done
        self._kill_delay = kill_delay

    def poll(self):
        return 0 if self._done else None

    def kill(self):
        self._done = True

    def wait(self, timeout=None):
        if self._kill_delay:
            time.sleep(self._kill_delay)
        return 0


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
        peak = max(peak, len(mgr._jobs))
    return peak


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

        assert peak_box[0] <= max_jobs, (
            f"max_jobs={max_jobs} violated: observed peak={peak_box[0]}"
        )
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
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
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
        assert "PHASE1" in mgr.get_info(jid).stdout, (
            "accumulated output lost by an intermediate drain"
        )
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
    ``_tool_shell_exec`` sets it."""
    proc = _real_proc("echo EARLY_OUT; echo EARLY_ERR >&2; sleep 30")
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
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()


def test_output_buffer_tail_cap():
    """The accumulated buffer keeps the most recent output under the cap."""
    old = "A" * bjm._OUTPUT_BUF_CAP
    grown = old + "TAIL_END"
    capped = bjm._cap_tail(grown)
    assert capped.endswith("TAIL_END")
    assert capped.startswith(bjm._TRUNCATION_MARKER)
    assert len(capped) <= bjm._OUTPUT_BUF_CAP + len(bjm._TRUNCATION_MARKER)


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
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        start_new_session=True, env=env,
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
        assert "MallocStackLogging" in mgr.get(jid)._stderr_buf, (
            "no libmalloc noise produced — test would pass vacuously"
        )

        info = mgr.get_info(jid)
        assert "MallocStackLogging" not in info.stderr
        assert "REAL_ERR" in info.stderr, "real stderr destroyed by the filter"
        assert "REAL_OUT" in info.stdout

        listed = [j for j in mgr.list_jobs() if j.job_id == jid][0]
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
    job = mgr.get(job_id)
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
            "for i in $(seq 1 4000); do echo line-$i-xxxxxxxxxxxxxxxxxxxx; done; "
            "echo FINAL_ANSWER_42",
            drain=True,   # payload exceeds the OS pipe buffer
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
            f"pre-timeout {stream} was salvaged and then dropped: "
            f"stdout={info.stdout!r} stderr={info.stderr!r}"
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
        assert "ONLY_OUT" in info.stdout, (
            f"a missing _recovered_stderr discarded the recovered stdout: "
            f"{info.stdout!r}"
        )
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
        assert "SALVAGED" in job._stdout_buf
    finally:
        proc.kill()
        proc.wait()
        mgr.shutdown()
