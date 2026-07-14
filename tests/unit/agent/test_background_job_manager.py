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


def test_recover_communicate_partial_after_timeout():
    """Regression (Bug #2): output consumed by communicate() before
    TimeoutExpired must be salvaged into the job's accumulated buffer.
    _fileobj2output holds a LIST of bytes chunks — the original fix treated it
    as bytes and silently no-opped via AttributeError."""
    import subprocess
    proc = _real_proc("echo EARLY_OUT; echo EARLY_ERR >&2; sleep 30")
    mgr = BackgroundJobManager(max_jobs=5, reap_interval=9999.0)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=1)
        bjm.recover_communicate_partial(proc)
        assert getattr(proc, "_recovered_stdout", "") == "EARLY_OUT\n"
        assert getattr(proc, "_recovered_stderr", "") == "EARLY_ERR\n"

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
