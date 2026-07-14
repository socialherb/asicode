"""Tests for the IPC shutdown-sentinel lifecycle and the task-before-shutdown
ordering invariant (Bug1 fix in subagent_ipc.poll_for_task).

These pin the behaviour of the new shutdown machinery added to prevent worker
leaks, and guard the race fix: a pending task.json must NEVER be silently
dropped when a shutdown sentinel is also present (the cancel/exception →
_cleanup_ipc_workers path used to drop it because the poll loop checked
shutdown before task).
"""
from __future__ import annotations

import json
import os

from external_llm.agent.subagent_ipc import (
    SubagentTask,
    build_subagent_prompt,
    check_shutdown_sentinel,
    derive_applied_patches,
    derive_unassigned_changes,
    partition_changed_files,
    poll_for_task,
    write_shutdown_sentinel,
    write_task,
)


def _task(task_id: str = "worker-1") -> SubagentTask:
    return SubagentTask(task_id=task_id, title="t", description="d")


def test_shutdown_sentinel_makes_poll_exit(tmp_path):
    """A shutdown sentinel causes poll_for_task to return None (worker exits)."""
    repo = str(tmp_path)
    write_shutdown_sentinel(repo, "worker-1")
    assert check_shutdown_sentinel(repo, "worker-1") is True
    result = poll_for_task(repo, "worker-1", poll_interval_s=0.01, timeout_s=2.0)
    assert result is None


def test_shutdown_sentinel_is_fire_once(tmp_path):
    """The sentinel is consumed on read so a relaunched worker isn't poisoned."""
    repo = str(tmp_path)
    write_shutdown_sentinel(repo, "worker-1")
    poll_for_task(repo, "worker-1", poll_interval_s=0.01, timeout_s=2.0)
    # After being honoured once, the sentinel file must be gone.
    assert check_shutdown_sentinel(repo, "worker-1") is False


def test_write_task_clears_stale_sentinel(tmp_path):
    """write_task removes any leftover shutdown.json so a reused worker starts."""
    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    os.makedirs(d)
    # Poison the directory with a stale sentinel from a previous run.
    with open(os.path.join(d, "shutdown.json"), "w") as f:
        json.dump({"shutdown": True}, f)
    write_task(repo, _task("worker-1"))
    assert check_shutdown_sentinel(repo, "worker-1") is False
    assert os.path.isfile(os.path.join(d, "task.json"))


def test_pending_task_wins_over_shutdown(tmp_path):
    """Bug1 invariant: a queued task is picked up even when a shutdown sentinel
    is also present — never silently dropped."""
    repo = str(tmp_path)
    # Queue a task, THEN write the shutdown sentinel (simulating a cancel that
    # races with a not-yet-picked-up task in a reusable worker).
    write_task(repo, _task("worker-1"))
    write_shutdown_sentinel(repo, "worker-1")
    result = poll_for_task(repo, "worker-1", poll_interval_s=0.01, timeout_s=2.0)
    # The task must win — it is returned, not dropped via the sentinel.
    assert result is not None
    assert result.task_id == "worker-1"


def test_poll_picks_up_task_and_deletes_it(tmp_path):
    """Baseline: poll reads the task and deletes task.json so it isn't re-read."""
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    result = poll_for_task(repo, "worker-1", poll_interval_s=0.01, timeout_s=2.0)
    assert result is not None and result.task_id == "worker-1"
    task_path = os.path.join(repo, ".asicode", "subagents", "worker-1", "task.json")
    assert not os.path.isfile(task_path)


# ── Regression tests for atomicity / epoch / backoff hardening ────────────
import threading

from external_llm.agent.subagent_ipc import (
    SubagentResult,
    _atomic_write,
    _next_backoff,
    wait_for_result,
    write_result,
)


def test_poll_acquisition_is_atomic_no_double_dispatch(tmp_path):
    """Two workers polling the SAME directory must NOT both acquire the task.

    Guards the read-then-unlink TOCTOU bug: with the legacy `open`+`unlink`
    split, two threads could both read task.json before either unlinked it.
    The rename-based acquisition (`os.rename` removes the source atomically)
    makes exactly one worker win — this is the only correct cross-process
    exclusion primitive and is deterministic (POSIX rename is atomic).
    """
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    results = [None, None]
    barrier = threading.Barrier(2)

    def worker(idx):
        barrier.wait()  # release both threads simultaneously
        results[idx] = poll_for_task(
            repo, "worker-1", poll_interval_s=0.02, timeout_s=0.5,
        )

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start(); t2.start(); t1.join(); t2.join()

    winners = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly one winner, got {results}"
    assert winners[0].task_id == "worker-1"


def test_poll_quarantines_malformed_task_no_spin(tmp_path):
    """A malformed task.json must be quarantined (.bad), not re-read forever.

    Guards the log-spam / infinite-retry bug where a corrupt task.json kept the
    worker spinning + warning every poll until timeout.
    """
    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    os.makedirs(d)
    # Write a malformed task.json directly (bypass write_task).
    with open(os.path.join(d, "task.json"), "w") as f:
        f.write("{ this is not valid json")

    result = poll_for_task(repo, "worker-1", poll_interval_s=0.02, timeout_s=0.4)
    assert result is None  # malformed → no task returned

    bad_files = [n for n in os.listdir(d) if n.endswith(".bad")]
    assert len(bad_files) == 1, f"malformed task must be quarantined, found {bad_files}"
    # And task.json must be gone (renamed into quarantine).
    assert not os.path.isfile(os.path.join(d, "task.json"))


def test_wait_for_result_rejects_stale_result_via_epoch(tmp_path):
    """wait_for_result must ignore a result whose epoch ≠ dispatched epoch.

    Guards the agent_id-reuse misattribution: a leftover result.json from a
    previous run must not short-circuit wait_for_result for a new task.
    """
    repo = str(tmp_path)
    # Dispatch a task (write_task assigns epoch E1 + writes expected.json).
    write_task(repo, _task("worker-1"))
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    # Drop a STALE result with a wrong epoch (simulating a previous run).
    stale = SubagentResult(task_id="worker-1", status="success", epoch=999)
    _atomic_write(os.path.join(d, "result.json"),
                  __import__("json").dumps(stale.to_dict()))

    r = wait_for_result(repo, "worker-1", poll_interval_s=0.02, timeout_s=0.3)
    assert r is None, "stale result (epoch mismatch) must NOT be returned"

    # Now write a FRESH result with the matching epoch → must resolve.
    # Read expected epoch from the sidecar to echo it exactly.
    import json as _json
    with open(os.path.join(d, "expected.json")) as f:
        expected_epoch = _json.load(f)["epoch"]
    fresh = SubagentResult(task_id="worker-1", status="success", epoch=expected_epoch)
    write_result(repo, fresh)
    r2 = wait_for_result(repo, "worker-1", poll_interval_s=0.02, timeout_s=1.0)
    assert r2 is not None and r2.epoch == expected_epoch


def test_wait_for_result_deletes_stale_result_file(tmp_path):
    """A rejected stale result.json must be unlinked so the poll loop doesn't
    re-read (and re-log) the same rejected file every cycle until timeout.

    Without the unlink, a leftover stale file spins the loop until the full
    timeout, accumulating a duplicate "ignoring stale result" warning on every
    poll cycle. The unlink lets the loop fall back to the clean "no result yet"
    (FileNotFoundError) wait state.
    """
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    result_path = os.path.join(d, "result.json")
    # Drop a STALE result with a wrong epoch (simulating a leftover from a
    # previous run that clear_result missed).
    import json as _json
    stale = SubagentResult(task_id="worker-1", status="success", epoch=999)
    _atomic_write(result_path, _json.dumps(stale.to_dict()))
    assert os.path.exists(result_path)

    r = wait_for_result(repo, "worker-1", poll_interval_s=0.02, timeout_s=0.3)
    assert r is None, "stale result (epoch mismatch) must NOT be returned"
    # The stale file must have been removed so the poll loop does not re-read
    # the same rejected file on every subsequent cycle.
    assert not os.path.exists(result_path), (
        "stale result.json must be deleted after epoch-mismatch rejection"
    )


def test_wait_for_result_accepts_legacy_zero_epoch_result(tmp_path):
    """Backward compat: a result with no epoch (0) is accepted (no validation)."""
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    legacy = SubagentResult(task_id="worker-1", status="success", epoch=0)
    _atomic_write(os.path.join(d, "result.json"),
                  __import__("json").dumps(legacy.to_dict()))
    r = wait_for_result(repo, "worker-1", poll_interval_s=0.02, timeout_s=1.0)
    assert r is not None and r.status == "success"


def test_atomic_write_concurrent_no_corruption(tmp_path):
    """Concurrent _atomic_write to the same path must not corrupt content.

    Guards the fixed `.tmp` name bug: N writers sharing `path.tmp` would
    truncate each other's buffer. With mkstemp each writer has a unique temp,
    so the final file is exactly one complete payload.
    """
    import json as _json
    path = str(tmp_path / "out.json")
    n = 8
    payloads = [_json.dumps({"i": i, "pad": "x" * 8000}) for i in range(n)]
    barrier = threading.Barrier(n)
    errors = []

    def writer(i):
        barrier.wait()
        try:
            _atomic_write(path, payloads[i])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    with open(path) as f:
        final = f.read()
    # Must be exactly one complete payload (no interleaving / truncation).
    assert final in payloads, "concurrent writes corrupted the file"


def test_poll_uses_exponential_backoff(tmp_path, monkeypatch):
    """Idle polling must back off, not hammer the FS at a fixed interval.

    Guards the fixed-0.5s busy-poll regression: over a 6s idle window the poll
    count with backoff (~5) must be far below the no-backoff count (~12).
    Simulated time advances with each sleep so the wall-clock deadline logic in
    wait_for_result is exercised deterministically.
    """
    import external_llm.agent.subagent_ipc as ipc

    clock = [0.0]
    sleeps = []

    def fake_monotonic():
        return clock[0]

    def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s  # advance the simulated clock by the slept amount

    monkeypatch.setattr(ipc.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(ipc.time, "sleep", fake_sleep)

    # No result present → wait_for_result idles until timeout.
    r = wait_for_result(str(tmp_path), "ghost", poll_interval_s=0.5, timeout_s=6.0)
    assert r is None
    # No backoff → ~12 polls; backoff (cap 3.0) → ~5 polls.
    assert len(sleeps) <= 8, f"backoff not applied: {len(sleeps)} sleeps"
    # And the interval must have grown beyond the start value.
    assert max(sleeps) > 0.5


def test_next_backoff_caps():
    assert _next_backoff(0.5) > 0.5
    # Repeated application caps at the module cap (3.0).
    v = 0.5
    for _ in range(20):
        v = _next_backoff(v)
    assert v == 3.0


def test_write_task_reuse_writes_expected_epoch_to_task_dir(tmp_path):
    """Reuse-path epoch gap: write_task(worker_id=W) must deposit expected.json
    in the TASK's own directory, not the worker's poll directory.

    In reuse mode, task.json is written to the worker dir (so the worker picks it
    up) but result.json is keyed by task.task_id.  wait_for_result(agent_id=
    task.task_id) reads expected.json from the TASK dir, so the epoch nonce must
    live there too — otherwise expected_epoch=0 → validation silently skipped and
    a stale result from a previous run on this task_id is misattributed.

    This is the reuse-path gap the write_task task_dir fix closes; the non-reuse
    path is already covered by test_wait_for_result_rejects_stale_result_via_epoch.
    """
    import json as _json
    repo = str(tmp_path)
    # Reuse path: worker_id differs from task.task_id.
    write_task(repo, _task("task-99"), worker_id="reusable-W")
    worker_dir = os.path.join(repo, ".asicode", "subagents", "reusable-W")
    task_dir = os.path.join(repo, ".asicode", "subagents", "task-99")

    # task.json went to the WORKER dir (so the reused worker polls and finds it).
    assert os.path.isfile(os.path.join(worker_dir, "task.json"))
    # expected.json must be in the TASK dir (paired with result.json), NOT worker.
    assert os.path.isfile(os.path.join(task_dir, "expected.json")), \
        "expected.json must live in the task dir for epoch validation to work"
    assert not os.path.isfile(os.path.join(worker_dir, "expected.json")), \
        "expected.json leaked into the worker dir (reuse-path bug)"

    # And the epoch there must match the dispatched task's epoch.
    # (task.json itself lives in the worker dir in reuse mode.)
    with open(os.path.join(worker_dir, "task.json")) as f:
        dispatched_epoch = _json.load(f)["epoch"]
    with open(os.path.join(task_dir, "expected.json")) as f:
        assert _json.load(f)["epoch"] == dispatched_epoch

    # End-to-end: a stale result on the task dir must be rejected via epoch.
    stale = SubagentResult(task_id="task-99", status="success", epoch=dispatched_epoch + 1)
    _atomic_write(os.path.join(task_dir, "result.json"), _json.dumps(stale.to_dict()))
    r = wait_for_result(repo, "task-99", poll_interval_s=0.02, timeout_s=0.3)
    assert r is None, "stale result must be rejected via epoch even on the reuse path"

    # A fresh result with the matching epoch resolves.
    fresh = SubagentResult(task_id="task-99", status="success", epoch=dispatched_epoch)
    write_result(repo, fresh)
    r2 = wait_for_result(repo, "task-99", poll_interval_s=0.02, timeout_s=1.0)
    assert r2 is not None and r2.epoch == dispatched_epoch


# ── Engine-asymmetry fixes: the IPC worker must mirror the in-process path ──
# A (critical): the worker must forward predecessor_context + original_request,
#   not just task.description — otherwise dependent IPC subtasks run "blind".
# B: applied_patches derived from git diff so cross-verification / summaries
#   have file attribution (DCL does not track patches itself).


def test_build_subagent_prompt_uses_description_when_no_context():
    """Independent task (no predecessor): prompt == description."""
    task = SubagentTask(task_id="dev_1", title="t", description="Do thing X")
    assert build_subagent_prompt(task) == "Do thing X"


def test_build_subagent_prompt_prefers_predecessor_context():
    """Dependent task: predecessor_context (a superset of description) wins.

    This is the core fix — the orchestrator stores the richly-built task_text
    (description + assigned-files hint + predecessor results + shared memory)
    in ``predecessor_context``; using bare ``description`` dropped all of it.
    """
    rich = (
        "[Orchestration progress]\n[dev_1: done | Files: a.py]\n\n"
        "[Predecessor task status]\n[dev_1: built foundation]\n\n"
        "[Current task]\nWrite tests\n\n[Assigned files: test_a.py]"
    )
    task = SubagentTask(
        task_id="dev_2", title="t", description="Write tests",
        predecessor_context=rich,
    )
    assert build_subagent_prompt(task) == rich
    # And it must NOT degrade to the bare description.
    assert build_subagent_prompt(task) != "Write tests"


def test_build_subagent_prompt_wraps_original_request_goal():
    """original_request (the overall user goal) is wrapped around the task."""
    task = SubagentTask(
        task_id="dev_1", title="t", description="Implement parser",
        original_request="Build a JSON parser library",
    )
    out = build_subagent_prompt(task)
    assert "[Original request goal]" in out
    assert "Build a JSON parser library" in out
    assert "[This sub-agent's task]" in out
    assert "Implement parser" in out


def test_build_subagent_prompt_no_double_wrap_when_goal_embedded():
    """If the original_request text already appears in the context, don't duplicate."""
    task = SubagentTask(
        task_id="dev_1", title="t",
        description="Continue the goal",
        original_request="THE GOAL",
        predecessor_context="Context mentioning THE GOAL already",
    )
    out = build_subagent_prompt(task)
    # No wrapper injected because "THE GOAL" is already present.
    assert "[Original request goal]" not in out
    assert out == "Context mentioning THE GOAL already"


def test_build_subagent_prompt_mirrors_in_process_wrapping(monkeypatch):
    """The wrapping format must match orchestrator._run_subagent exactly:
    '[Original request goal]\\n{goal}\\n\\n[This sub-agent's task]\\n{body}'."""
    task = SubagentTask(
        task_id="dev_1", title="t", description="BODY",
        original_request="GOAL",
    )
    expected = "[Original request goal]\nGOAL\n\n[This sub-agent's task]\nBODY"
    assert build_subagent_prompt(task) == expected


def test_derive_applied_patches_from_git_diff(tmp_path):
    """Patches are derived from `git diff --name-only` scoped to assigned_files."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    # Committed baseline.
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    # Working-tree change to a.py only.
    (repo / "a.py").write_text("x = 11\n")

    patches = derive_applied_patches(str(repo), ["a.py"])
    assert patches == [{"file": "a.py"}]


def test_derive_applied_patches_scoped_to_assigned_files(tmp_path):
    """Unassigned files are excluded from the patch list."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 1\n")
    (repo / "b.py").write_text("y = 2\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    # Both changed, but only a.py is assigned.
    (repo / "a.py").write_text("x = 11\n")
    (repo / "b.py").write_text("y = 22\n")

    patches = derive_applied_patches(str(repo), ["a.py"])
    assert patches == [{"file": "a.py"}]
    assert {"file": "b.py"} not in patches


def test_derive_applied_patches_no_changes(tmp_path):
    """Clean working tree → empty patch list (not an error)."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 1\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)

    assert derive_applied_patches(str(repo), ["a.py"]) == []


def test_derive_applied_patches_non_git_returns_empty(tmp_path):
    """A non-git directory degrades gracefully to [] (no exception)."""
    assert derive_applied_patches(str(tmp_path), ["a.py"]) == []


def test_derive_applied_patches_empty_assigned_files(tmp_path):
    """Empty assigned_files → whole-repo `git diff --name-only` (worker fallback)."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 1\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 11\n")

    # No scoping → reports the changed file (matches sequential-mode fallback).
    patches = derive_applied_patches(str(repo), [])
    assert patches == [{"file": "a.py"}]


def test_derive_applied_patches_untracked_new_file(tmp_path):
    """New (untracked) files are included in patches — 'git diff --name-only' would miss them."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    # Committed baseline with tracked.py only.
    (repo / "tracked.py").write_text("x = 1\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    # Modify tracked + create brand-new file.
    (repo / "tracked.py").write_text("x = 11\n")
    (repo / "brandnew.py").write_text("y = 2\n")

    patches = derive_applied_patches(str(repo), ["tracked.py", "brandnew.py"])
    assert {"file": "tracked.py"} in patches
    assert {"file": "brandnew.py"} in patches


def test_partition_changed_files_out_of_scope_detected(tmp_path):
    """B5: out-of-scope writes are surfaced (not silently dropped)."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 1\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    # In-scope change (a.py) + out-of-scope change (verify-repo/stats.py).
    (repo / "a.py").write_text("x = 11\n")
    (repo / "verify-repo").mkdir()
    (repo / "verify-repo" / "stats.py").write_text("z = 3\n")

    in_scope, out_scope = partition_changed_files(str(repo), ["a.py"])
    assert in_scope == [{"file": "a.py"}]
    assert {"file": "verify-repo/stats.py"} in out_scope
    # The standalone helper agrees with the partition.
    assert derive_unassigned_changes(str(repo), ["a.py"]) == out_scope


def test_partition_changed_files_empty_assignment_is_all_in_scope(tmp_path):
    """Empty assigned_files → everything in-scope, out-of-scope empty (legacy)."""
    import subprocess as _sp

    repo = tmp_path
    _sp.run(["git", "init", "-q"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.email", "t@t.t"], cwd=str(repo), check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 1\n")
    _sp.run(["git", "add", "."], cwd=str(repo), check=True)
    _sp.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
    (repo / "a.py").write_text("x = 11\n")

    in_scope, out_scope = partition_changed_files(str(repo), [])
    assert in_scope == [{"file": "a.py"}]
    assert out_scope == []


def test_partition_changed_files_rename_both_positions_no_ghost(tmp_path, monkeypatch):
    """Rename/copy markers in EITHER porcelain column must consume the OLD field.

    porcelain -z emits ``"XY NEW\\0OLD\\0"`` for renames/copies, where R/C can
    land in the Y (worktree) column as well as X (staged): ``"R  NEW\\0OLD\\0"``
    (staged), ``" R NEW\\0OLD\\0"`` (worktree rename), ``" C NEW\\0OLD\\0"``
    (worktree copy). A parser that only checks the X column parses the trailing
    OLD field as a standalone status line and chops its first 3 chars (where the
    status codes live), producing a ghost path (e.g. ``"old file.py"`` -> ``"
    file.py"``). All three forms must yield exactly the NEW path, nothing else.
    """
    import subprocess

    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)

    crafted = (
        "R  new_x.py\x00old_x.py\x00"   # X-position rename (staged)
        " R new_y.py\x00old_y.py\x00"   # Y-position rename (worktree) — the regression
        " C new_c.py\x00old_c.py\x00"   # Y-position copy (worktree)
    )

    class _Fake:
        stdout = crafted

    def _fake_run(cmd, *a, **k):
        return _Fake()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    in_scope, out_scope = partition_changed_files(
        str(repo), ["new_x.py", "new_y.py", "new_c.py"],
    )
    all_reported = sorted(p["file"] for p in in_scope) + sorted(p["file"] for p in out_scope)
    # Exactly the three NEW paths — no OLD fields, no 3-char-chopped ghosts
    # ("old_x.py"[3:] == "_x.py", etc.).
    assert all_reported == ["new_c.py", "new_x.py", "new_y.py"]


def test_subagent_result_carries_unassigned_changes():
    """SubagentResult round-trips the unassigned_changes field."""
    from external_llm.agent.subagent_ipc import SubagentResult

    r = SubagentResult(
        task_id="t1", status="success",
        applied_patches=[{"file": "a.py"}],
        unassigned_changes=[{"file": "verify-repo/stats.py"}],
        epoch=42,
    )
    d = r.to_dict()
    assert d["unassigned_changes"] == [{"file": "verify-repo/stats.py"}]
    # from_dict tolerates the new field (forward compat).
    r2 = SubagentResult.from_dict(d)
    assert r2.unassigned_changes == [{"file": "verify-repo/stats.py"}]
    # from_dict tolerates ABSENCE of the field (backward compat with old workers).
    r3 = SubagentResult.from_dict({"task_id": "t1", "status": "success"})
    assert r3.unassigned_changes == []


# ─────────────────────────────────────────────────────────────────────────────
# Cancel sentinel (mid-task abort) — distinct from the shutdown sentinel.
# Shutdown = "exit the poll loop between tasks"; cancel = "abort the task you
# are running RIGHT NOW".  The worker's cancel watcher thread polls cancel.json
# DURING task execution and sets the local cancel_event so DesignChatLoop aborts
# at its next turn boundary.
# ─────────────────────────────────────────────────────────────────────────────


def test_cancel_sentinel_roundtrip(tmp_path):
    """write_cancel_sentinel / check_cancel_sentinel are a matched pair."""
    from external_llm.agent.subagent_ipc import (
        check_cancel_sentinel, write_cancel_sentinel,
    )
    repo = str(tmp_path)
    assert check_cancel_sentinel(repo, "worker-1") is False
    write_cancel_sentinel(repo, "worker-1")
    assert check_cancel_sentinel(repo, "worker-1") is True
    # Distinct from the shutdown sentinel — writing cancel does NOT set shutdown.
    assert check_shutdown_sentinel(repo, "worker-1") is False


def test_write_task_clears_stale_cancel_sentinel(tmp_path):
    """A leftover cancel.json from a previous (cancelled) task must not poison a
    fresh dispatch — otherwise the worker's watcher aborts the new task the
    instant it starts.  write_task clears it (belt-and-braces with the watcher's
    own fire-once consume)."""
    from external_llm.agent.subagent_ipc import (
        check_cancel_sentinel, write_cancel_sentinel,
    )
    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    os.makedirs(d)
    write_cancel_sentinel(repo, "worker-1")
    assert check_cancel_sentinel(repo, "worker-1") is True
    write_task(repo, _task("worker-1"))
    assert check_cancel_sentinel(repo, "worker-1") is False
    assert os.path.isfile(os.path.join(d, "task.json"))


def test_cancel_sentinel_does_not_exit_poll_loop(tmp_path):
    """cancel.json alone must NOT make poll_for_task return None.

    poll_for_task honors only the shutdown sentinel (between tasks).  cancel is
    consumed by the worker's SEPARATE watcher thread DURING a task — so a
    cancel.json with no task and no shutdown should leave the poll loop waiting
    (here: timing out) rather than exiting.  This proves the two signals are
    distinct and cancel cannot be mistaken for shutdown by the poll path."""
    from external_llm.agent.subagent_ipc import write_cancel_sentinel
    repo = str(tmp_path)
    write_cancel_sentinel(repo, "worker-1")
    # No task, no shutdown — poll must time out (return None after timeout),
    # NOT exit immediately.  A short timeout keeps the test fast.
    import time
    t0 = time.monotonic()
    result = poll_for_task(repo, "worker-1", poll_interval_s=0.02, timeout_s=0.3)
    elapsed = time.monotonic() - t0
    assert result is None
    # If cancel.json were honored as a shutdown, poll would exit in <<0.3s.
    assert elapsed >= 0.25, f"poll exited too fast ({elapsed:.2f}s) — cancel misread as shutdown?"
    # cancel.json is still present (the poll path does not consume it).
    from external_llm.agent.subagent_ipc import check_cancel_sentinel
    assert check_cancel_sentinel(repo, "worker-1") is True


# ── _is_process_alive: cross-platform pid liveness probe ─────────────────────


def test_is_process_alive_true_for_self():
    from external_llm.agent.subagent_ipc import _is_process_alive
    assert _is_process_alive(os.getpid()) is True


def test_is_process_alive_false_for_dead_pid():
    from external_llm.agent.subagent_ipc import _is_process_alive
    # A pid essentially guaranteed not to be a live process on the test host.
    assert _is_process_alive(2**31 - 1) is False


# ── B3: orphan self-exit when the spawning process is gone ───────────────────


def test_poll_for_task_exits_when_parent_gone(tmp_path, monkeypatch):
    """B3: if expected_parent_pid is set and the spawning process has died
    (e.g. orchestrator SIGKILL'd), poll_for_task self-exits immediately
    instead of polling forever for a shutdown.json that will never arrive
    (the writer is dead). On POSIX the check uses os.getppid() (immune to
    PID-reuse race); on Windows it falls back to _is_process_alive (cross-
    platform kernel probe, has theoretical PID-reuse race)."""
    import time as _t
    from external_llm.agent import subagent_ipc as _ipc_mod
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    monkeypatch.setattr(_ipc_mod, "_is_process_alive", lambda pid: False)

    t0 = _t.monotonic()
    result = poll_for_task(
        repo, "worker-1",
        poll_interval_s=0.01, timeout_s=5.0,
        expected_parent_pid=9999,  # the (now-dead) orchestrator pid
    )
    elapsed = _t.monotonic() - t0
    assert result is None
    assert elapsed < 1.0, "worker did NOT self-exit on orphan — polled the full timeout"


def test_poll_for_task_polls_when_parent_alive(tmp_path, monkeypatch):
    """B3 converse: when the parent is still alive, the orphan check must NOT
    fire and the worker polls until the timeout (no false-positive exit)."""
    import time as _t
    from external_llm.agent import subagent_ipc as _ipc_mod
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    monkeypatch.setattr(_ipc_mod, "_is_process_alive", lambda pid: True)
    # On POSIX the orphan check uses getppid() — match expected_parent_pid so
    # the check does NOT fire (the parent is "still alive").  Keep the
    # _is_process_alive monkeypatch for Windows (unchanged path).
    monkeypatch.setattr(os, "getppid", lambda: 4242)

    t0 = _t.monotonic()
    result = poll_for_task(
        repo, "worker-1",
        poll_interval_s=0.01, timeout_s=0.3,
        expected_parent_pid=4242,  # still alive — must NOT early-exit
    )
    elapsed = _t.monotonic() - t0
    assert result is None
    assert elapsed >= 0.25, "worker exited early despite parent still alive"


def test_poll_for_task_no_orphan_check_when_pid_unset(tmp_path, monkeypatch):
    """B3 opt-in: when expected_parent_pid is None (default), the orphan check
    is disabled — backwards compatible with any caller that doesn't opt in."""
    import time as _t
    from external_llm.agent import subagent_ipc as _ipc_mod
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    # Even if the "parent" looks dead, with the check disabled we poll to timeout.
    monkeypatch.setattr(_ipc_mod, "_is_process_alive", lambda pid: False)

    t0 = _t.monotonic()
    result = poll_for_task(
        repo, "worker-1", poll_interval_s=0.01, timeout_s=0.3,
        # expected_parent_pid omitted → opt-out
    )
    elapsed = _t.monotonic() - t0
    assert result is None
    assert elapsed >= 0.25, "opt-out path exited early; orphan check should be disabled"


# ── malformed-JSON structural robustness (TypeError from from_dict) ──────────
# A result.json/task.json that is syntactically valid JSON but missing a
# required dataclass field (e.g. task_id) makes Subagent*.from_dict raise
# TypeError. Before the fix this TypeError was outside the poll/read except
# tuples and either crashed the worker (poll_for_task) or crashed the whole
# orchestration (wait_for_result), and in poll_for_task also left an orphaned
# task.json.claimed.<pid> file. Both except clauses now catch TypeError.


def test_poll_for_task_quarantines_structurally_malformed_task(tmp_path):
    """Bug 3: a structurally-malformed task.json (missing required `task_id`)
    raises TypeError in SubagentTask.from_dict. The fix catches it alongside
    JSONDecodeError/ValueError so the malformed task is quarantined to .bad
    (not left as an orphaned .claimed file) and the worker keeps polling
    instead of crashing."""
    import glob
    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    os.makedirs(d)
    # Valid JSON, but missing the required `task_id` / `description` fields
    # → SubagentTask.from_dict → TypeError.
    with open(os.path.join(d, "task.json"), "w") as f:
        json.dump({"title": "no task_id here"}, f)
    # A shutdown sentinel lets the poll exit cleanly right after quarantining
    # (otherwise it would poll the full timeout waiting for a re-write).
    write_shutdown_sentinel(repo, "worker-1")

    result = poll_for_task(repo, "worker-1", poll_interval_s=0.01, timeout_s=2.0)
    # No crash; the worker moved on (here: honoured the shutdown → None).
    assert result is None
    # The malformed task was quarantined, not left as an orphaned .claimed file.
    assert glob.glob(os.path.join(d, "task.json.claimed.*.bad")), \
        "malformed task.json was not quarantined"
    assert not os.path.isfile(os.path.join(d, "task.json")), \
        "claimed (malformed) task.json should have been renamed away"


def test_wait_for_result_tolerates_structurally_malformed_result(tmp_path):
    """Bug 2: a structurally-malformed result.json (missing required `task_id`)
    raises TypeError in SubagentResult.from_dict. Before the fix this TypeError
    escaped wait_for_result's except tuple and propagated, crashing the entire
    orchestration on a single bad file. Now it is treated like a not-yet-ready
    result (caught + polled past) so the call degrades to a graceful timeout
    (returns None) instead of raising."""
    from external_llm.agent.subagent_ipc import wait_for_result
    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "agent-7")
    os.makedirs(d)
    # Valid JSON, but missing the required `task_id` and `status` fields
    # → SubagentResult.from_dict → TypeError.
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({"final_message": "no task_id/status here"}, f)

    result = wait_for_result(
        repo, "agent-7",
        poll_interval_s=0.02, timeout_s=0.4,
    )
    # Must NOT raise; a permanently-malformed result degrades to timeout (None).
    assert result is None

# ── PermissionError / jitter / timeout passthrough ─────────────────────────


def test_wait_for_result_tolerates_permission_error(tmp_path, monkeypatch):
    """On Windows (and occasionally Linux with antivirus locks), wait_for_result
    may encounter a PermissionError when opening result.json.  The except tuple
    already catches PermissionError, so the poll loop keeps running and the call
    degrades to a graceful timeout (returns None) instead of crashing."""
    import builtins as _builtins
    from external_llm.agent.subagent_ipc import wait_for_result

    repo = str(tmp_path)
    d = os.path.join(repo, ".asicode", "subagents", "agent-7")
    os.makedirs(d)
    # Write a valid result.json so open() is reached by the poll loop.
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({"task_id": "agent-7", "status": "success"}, f)

    real_open = _builtins.open

    def _failing_open(*args, **kwargs):
        if len(args) > 0 and "result.json" in str(args[0]):
            raise PermissionError("Simulated Windows file-lock denial")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(_builtins, "open", _failing_open)

    result = wait_for_result(
        repo, "agent-7",
        poll_interval_s=0.02, timeout_s=0.4,
    )
    # Must NOT raise; the PermissionError is swallowed so the poll loop
    # continues and eventually times out.
    assert result is None


def test_next_backoff_includes_jitter():
    """_next_backoff adds random jitter (uniform 0…0.2×interval) so the
    returned value is >= interval * _POLL_BACKOFF_FACTOR and <= _POLL_BACKOFF_CAP_S.
    Multiple calls produce different values — the function is non-deterministic."""
    from external_llm.agent.subagent_ipc import (
        _next_backoff,
        _POLL_BACKOFF_FACTOR,
        _POLL_BACKOFF_CAP_S,
    )

    interval = 1.0
    results = [_next_backoff(interval) for _ in range(100)]

    for r in results:
        assert r >= interval * _POLL_BACKOFF_FACTOR, (
            f"backoff {r} < interval * factor ({interval * _POLL_BACKOFF_FACTOR})"
        )
        assert r <= _POLL_BACKOFF_CAP_S, (
            f"backoff {r} > cap ({_POLL_BACKOFF_CAP_S})"
        )

    # With jitter the values should not all be identical.
    assert len(set(results)) > 1, (
        "100 consecutive calls produced identical values — "
        "jitter appears to be missing"
    )


def test_derive_applied_patches_uses_custom_timeout(tmp_path):
    """derive_applied_patches passes the timeout_s parameter through to
    subprocess.run so the git-status call respects the caller's deadline."""
    from unittest.mock import patch as _patch
    from external_llm.agent.subagent_ipc import derive_applied_patches

    with _patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = ""

        derive_applied_patches(str(tmp_path), ["a.py"], timeout_s=15.0)

        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 15.0, (
            f"expected timeout=15.0, got {kwargs.get('timeout')}"
        )


# ── Worker heartbeat: cross-process liveness for wait_for_result ────────────
import time as _time


def test_write_heartbeat_writes_json(tmp_path):
    """write_heartbeat creates heartbeat.json with a wall-clock ts and pid."""
    from external_llm.agent.subagent_ipc import write_heartbeat, _heartbeat_path

    repo = str(tmp_path)
    path = write_heartbeat(repo, "worker-1", pid=12345)
    assert os.path.isfile(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pid"] == 12345
    assert data["agent_id"] == "worker-1"
    assert data["ts"] > 0
    # Lives in the task's OWN directory (same as result.json).
    assert path == _heartbeat_path(
        os.path.join(repo, ".asicode", "subagents", "worker-1")
    )


def test_read_heartbeat_age_s_none_when_absent(tmp_path):
    """No heartbeat.json → read_heartbeat_age_s returns None (not 'dead')."""
    from external_llm.agent.subagent_ipc import read_heartbeat_age_s

    assert read_heartbeat_age_s(str(tmp_path), "ghost") is None


def test_read_heartbeat_age_s_returns_positive_age(tmp_path):
    """A fresh heartbeat yields a small, positive age."""
    from external_llm.agent.subagent_ipc import write_heartbeat, read_heartbeat_age_s

    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=1)
    age = read_heartbeat_age_s(repo, "worker-1")
    assert age is not None and 0.0 <= age < 5.0


def test_read_heartbeat_age_s_none_for_invalid_payload(tmp_path):
    """A corrupt/zero-ts heartbeat yields None rather than a misleading age."""
    from external_llm.agent.subagent_ipc import read_heartbeat_age_s, _subagent_dir

    repo = str(tmp_path)
    d = _subagent_dir(repo, "worker-1")
    with open(os.path.join(d, "heartbeat.json"), "w") as f:
        f.write("{not valid json")
    assert read_heartbeat_age_s(repo, "worker-1") is None
    # Zero timestamp → also None (treated as "no usable heartbeat").
    with open(os.path.join(d, "heartbeat.json"), "w") as f:
        json.dump({"ts": 0, "pid": 1}, f)
    assert read_heartbeat_age_s(repo, "worker-1") is None


def test_wait_for_result_presumes_dead_on_stale_heartbeat(tmp_path):
    """A stale heartbeat makes wait_for_result return None immediately instead
    of burning the full timeout (the crash-detection win)."""
    from external_llm.agent.subagent_ipc import _subagent_dir

    repo = str(tmp_path)
    d = _subagent_dir(repo, "worker-1")
    # Write a heartbeat that is 200s old.
    with open(os.path.join(d, "heartbeat.json"), "w") as f:
        json.dump({"ts": _time.time() - 200, "pid": 1}, f)

    t0 = _time.monotonic()
    # stale threshold 100s, but timeout 5s — dead detection must trip first.
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.02, timeout_s=5.0, heartbeat_stale_s=100.0,
    )
    elapsed = _time.monotonic() - t0
    assert r is None, "stale heartbeat must cause presumed-dead return"
    assert elapsed < 2.0, (
        f"dead detection should bail within ~1 poll, took {elapsed:.1f}s"
    )


def test_wait_for_result_no_heartbeat_uses_normal_timeout(tmp_path):
    """Absent heartbeat (legacy/just-started worker) must NOT be treated as dead;
    the normal timeout governs and the call blocks for the full duration."""
    repo = str(tmp_path)
    t0 = _time.monotonic()
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.05, timeout_s=0.4, heartbeat_stale_s=0.1,
    )
    elapsed = _time.monotonic() - t0
    assert r is None
    # Should have waited the full timeout, not bailed on a non-existent heartbeat.
    assert elapsed >= 0.35, f"expected ~full timeout, bailed early after {elapsed:.2f}s"


def test_wait_for_result_fresh_heartbeat_does_not_bail(tmp_path):
    """A fresh heartbeat proves liveness; the call keeps waiting and returns the
    result once it arrives (no false-positive dead declaration)."""
    from external_llm.agent.subagent_ipc import write_heartbeat

    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=1)
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.02, timeout_s=1.0, heartbeat_stale_s=100.0,
    )
    # No result written, heartbeat fresh → should wait the full timeout.
    assert r is None


def test_clear_result_clears_heartbeat(tmp_path):
    """clear_result removes a stale heartbeat so the next wait_for_result does
    not see a leftover from the previous task and falsely flag the worker dead."""
    from external_llm.agent.subagent_ipc import (
        write_heartbeat, clear_result, _heartbeat_path, _subagent_dir,
    )

    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=1)
    hb = _heartbeat_path(_subagent_dir(repo, "worker-1"))
    assert os.path.isfile(hb)
    clear_result(repo, "worker-1")
    assert not os.path.isfile(hb), "clear_result must remove heartbeat.json"


# ── Startup-failure timeout (2-stage timeout, opt-in) ───────────────────────
def test_wait_for_result_startup_timeout_no_heartbeat(tmp_path):
    """With startup_timeout_s set and NO heartbeat ever appearing, the call bails
    at the startup deadline instead of waiting the full timeout."""
    repo = str(tmp_path)
    t0 = _time.monotonic()
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.02, timeout_s=10.0,
        startup_timeout_s=0.3, heartbeat_stale_s=0.0,
    )
    elapsed = _time.monotonic() - t0
    assert r is None, "startup timeout with no heartbeat must bail"
    assert 0.25 <= elapsed < 2.0, (
        f"expected ~startup deadline, took {elapsed:.2f}s"
    )


def test_wait_for_result_startup_timeout_inert_after_heartbeat(tmp_path):
    """Once a heartbeat has been seen the startup guard is inert — a heartbeat
    that appears right at the deadline keeps the call alive to the full timeout."""
    from external_llm.agent.subagent_ipc import write_heartbeat

    repo = str(tmp_path)
    # Fresh heartbeat present → startup guard must NOT fire even with a tiny
    # startup deadline; the call should wait for the result.
    write_heartbeat(repo, "worker-1", pid=1)
    t0 = _time.monotonic()
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.02, timeout_s=0.4,
        startup_timeout_s=0.1, heartbeat_stale_s=100.0,
    )
    elapsed = _time.monotonic() - t0
    assert r is None  # no result written
    # Heartbeat was fresh → startup guard inert → waited the full timeout, not
    # the 0.1s startup deadline.
    assert elapsed >= 0.35, f"startup guard fired despite fresh heartbeat ({elapsed:.2f}s)"
def test_wait_for_result_startup_timeout_inert_with_liveness_disabled(tmp_path):
    """Regression: heartbeat_stale_s<=0 (liveness disabled) + startup_timeout_s>0
    must NOT falsely abandon a live worker that IS writing heartbeats.

    heartbeat_ever_seen used to be flipped only inside the heartbeat_stale_s>0
    liveness block, so with liveness disabled it stayed False forever and the
    startup guard fired on a healthy worker once the deadline elapsed. The guard
    now probes heartbeat directly when liveness is off, so a fresh heartbeat
    keeps the call alive to the full timeout.
    """
    from external_llm.agent.subagent_ipc import write_heartbeat

    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=1)
    t0 = _time.monotonic()
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.02, timeout_s=0.4,
        startup_timeout_s=0.1, heartbeat_stale_s=0.0,
    )
    elapsed = _time.monotonic() - t0
    assert r is None  # no result written
    # Fresh heartbeat present + liveness disabled → startup guard must stay inert
    # → waited the full timeout, not the 0.1s startup deadline.
    assert elapsed >= 0.35, (
        f"startup guard fired despite fresh heartbeat (liveness disabled) "
        f"after {elapsed:.2f}s"
    )


def test_wait_for_result_startup_timeout_disabled_by_default(tmp_path):
    """Default startup_timeout_s=0 → disabled: a missing heartbeat never bails
    early, the call waits the full timeout (backward-compatible, safe for legacy
    workers that do not write heartbeats)."""
    repo = str(tmp_path)
    t0 = _time.monotonic()
    r = wait_for_result(
        repo, "worker-1",
        poll_interval_s=0.05, timeout_s=0.4,
        heartbeat_stale_s=0.1,  # also disabled-ish: no heartbeat → no stale trip
    )
    elapsed = _time.monotonic() - t0
    assert r is None
    assert elapsed >= 0.35, f"default-off startup guard should wait full timeout ({elapsed:.2f}s)"


# ── Infinite-poll safety net: idle warning + max_poll_s cap (poll_for_task) ──
def test_poll_for_task_idle_warn_emits_warning(tmp_path, caplog):
    """After idle_warn_s of no task, poll_for_task emits a WARNING diagnostic."""
    import logging as _lg
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    with caplog.at_level(_lg.WARNING, logger="external_llm.agent.subagent_ipc"):
        poll_for_task(
            repo, "worker-1",
            poll_interval_s=0.01, timeout_s=0.3, idle_warn_s=0.1,
        )
    assert any("idle" in rec.message for rec in caplog.records), (
        "expected an idle warning after idle_warn_s of no task"
    )


def test_poll_for_task_max_poll_s_caps_infinite_poll(tmp_path):
    """timeout_s=None is bounded by max_poll_s so a worker cannot poll forever
    when the orchestrator is alive but never dispatches / shuts down."""
    import time as _t
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    t0 = _t.monotonic()
    r = poll_for_task(
        repo, "worker-1",
        poll_interval_s=0.01, timeout_s=None,
        max_poll_s=0.3, idle_warn_s=0.0,
    )
    elapsed = _t.monotonic() - t0
    assert r is None
    assert 0.25 <= elapsed < 2.0, (
        f"expected max_poll_s cap (~0.3s), took {elapsed:.2f}s"
    )


def test_poll_for_task_max_poll_s_inert_for_finite_timeout(tmp_path):
    """A finite timeout_s governs the call; max_poll_s is NOT applied (it only
    bounds the timeout_s=None path) so a small max_poll_s cannot truncate a
    larger timeout_s."""
    import time as _t
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, ".asicode", "subagents", "worker-1"))
    t0 = _t.monotonic()
    r = poll_for_task(
        repo, "worker-1",
        poll_interval_s=0.05, timeout_s=0.4,
        max_poll_s=0.1,  # smaller than timeout_s — must be ignored
        idle_warn_s=0.0,
    )
    elapsed = _t.monotonic() - t0
    assert r is None
    assert elapsed >= 0.35, (
        f"finite timeout_s should govern, but exited after {elapsed:.2f}s "
        f"(max_poll_s=0.1 wrongly applied?)"
    )


# ── B2: TOCTOU-safe stale-result quarantine ──────────────────────────────────


def test_quarantine_and_recheck_adopts_fresh_raced_write(tmp_path):
    """If a fresh result (matching epoch) is on disk at quarantine-rename time,
    it is adopted rather than dropped — the TOCTOU fix.

    Models the race: the poll loop read an OLD-epoch result, but by the time the
    stale-removal runs, the worker has renamed in a FRESH result. os.replace
    captures the fresh file and the epoch re-check adopts it.
    """
    from external_llm.agent.subagent_ipc import _quarantine_and_recheck
    d = os.path.join(str(tmp_path), ".asicode", "subagents", "worker-1")
    os.makedirs(d, exist_ok=True)
    result_path = os.path.join(d, "result.json")
    fresh = SubagentResult(task_id="worker-1", status="success", epoch=777)
    _atomic_write(result_path, json.dumps(fresh.to_dict()))

    adopted = _quarantine_and_recheck(result_path, 777, "worker-1")
    assert adopted is not None and adopted.epoch == 777
    assert not os.path.exists(result_path), (
        "adopted file removed from result_path (returned in-memory, not lost)"
    )


def test_quarantine_and_recheck_drops_genuinely_stale(tmp_path):
    """A genuinely stale result (epoch mismatch at rename time) is discarded."""
    from external_llm.agent.subagent_ipc import _quarantine_and_recheck
    d = os.path.join(str(tmp_path), ".asicode", "subagents", "worker-1")
    os.makedirs(d, exist_ok=True)
    result_path = os.path.join(d, "result.json")
    stale = SubagentResult(task_id="worker-1", status="success", epoch=999)
    _atomic_write(result_path, json.dumps(stale.to_dict()))

    adopted = _quarantine_and_recheck(result_path, 777, "worker-1")
    assert adopted is None
    assert not os.path.exists(result_path), "stale file cleaned up after quarantine check"


def test_quarantine_and_recheck_handles_missing_file(tmp_path):
    """If the result file vanished before the rename, return None (no crash)."""
    from external_llm.agent.subagent_ipc import _quarantine_and_recheck
    d = os.path.join(str(tmp_path), ".asicode", "subagents", "worker-1")
    os.makedirs(d, exist_ok=True)
    result_path = os.path.join(d, "result.json")  # never created
    assert _quarantine_and_recheck(result_path, 777, "worker-1") is None


# ── F1: heartbeat soft-timeout extension ─────────────────────────────────────


def test_wait_for_result_soft_timeout_extends_on_fresh_heartbeat(tmp_path):
    """A fresh heartbeat extends the deadline to max_timeout_s (once): a result
    landing AFTER timeout_s but BEFORE max_timeout_s is adopted, not abandoned."""
    import threading
    import time as _time
    from external_llm.agent.subagent_ipc import write_heartbeat, write_result
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    d = os.path.join(repo, ".asicode", "subagents", "worker-1")
    with open(os.path.join(d, "expected.json")) as f:
        expected_epoch = json.load(f)["epoch"]
    write_heartbeat(repo, "worker-1", pid=123)  # fresh → worker alive

    def _delayed():
        _time.sleep(1.2)  # lands after timeout_s=0.4, before max_timeout_s=3.0
        write_result(repo, SubagentResult(task_id="worker-1", status="success", epoch=expected_epoch))
    threading.Thread(target=_delayed, daemon=True).start()

    r = wait_for_result(
        repo, "worker-1", poll_interval_s=0.05, timeout_s=0.4, max_timeout_s=3.0,
    )
    assert r is not None and r.epoch == expected_epoch, (
        "fresh heartbeat should extend deadline past timeout_s"
    )


def test_wait_for_result_no_extension_without_max_timeout(tmp_path):
    """Without max_timeout_s (> timeout_s), a fresh heartbeat does NOT extend —
    timeout_s remains the hard deadline (default behavior preserved)."""
    from external_llm.agent.subagent_ipc import write_heartbeat
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    write_heartbeat(repo, "worker-1", pid=123)
    r = wait_for_result(repo, "worker-1", poll_interval_s=0.05, timeout_s=0.4)
    assert r is None, "no max_timeout_s → hard timeout_s governs even with fresh heartbeat"


def test_wait_for_result_stale_heartbeat_still_kills_under_soft_timeout(tmp_path):
    """Under soft timeout, a STALE heartbeat still bails early — the extension
    only rewards a FRESH (alive) worker, never a dead one."""
    from external_llm.agent.subagent_ipc import write_heartbeat
    repo = str(tmp_path)
    write_task(repo, _task("worker-1"))
    # Write a heartbeat, then backdate it well past the stale threshold.
    write_heartbeat(repo, "worker-1", pid=123)
    hb_path = os.path.join(repo, ".asicode", "subagents", "worker-1", "heartbeat.json")
    json.dump({"ts": 1.0, "pid": 123, "agent_id": "worker-1"}, open(hb_path, "w"))
    r = wait_for_result(
        repo, "worker-1", poll_interval_s=0.02, timeout_s=1.0, max_timeout_s=5.0,
        heartbeat_stale_s=1.0,
    )
    assert r is None, "stale heartbeat must still bail early even with max_timeout_s set"


# ── F3: heartbeat progress hints ─────────────────────────────────────────────


def test_write_heartbeat_carries_progress_hints(tmp_path):
    """write_heartbeat records turn/last_tool so the orchestrator can show
    progress ("turn 5, run_tests") instead of only elapsed time."""
    from external_llm.agent.subagent_ipc import write_heartbeat, read_heartbeat_state
    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=42, turn=5, last_tool="run_tests")
    st = read_heartbeat_state(repo, "worker-1")
    assert st is not None
    assert st["turn"] == 5
    assert st["last_tool"] == "run_tests"
    assert st["pid"] == 42
    assert st["ts"] > 0


def test_write_heartbeat_progress_defaults(tmp_path):
    """Legacy call (no turn/last_tool) writes zeros/empty — backward compatible."""
    from external_llm.agent.subagent_ipc import write_heartbeat, read_heartbeat_state
    repo = str(tmp_path)
    write_heartbeat(repo, "worker-1", pid=1)
    st = read_heartbeat_state(repo, "worker-1")
    assert st["turn"] == 0
    assert st["last_tool"] == ""


def test_read_heartbeat_state_none_when_absent(tmp_path):
    from external_llm.agent.subagent_ipc import read_heartbeat_state
    assert read_heartbeat_state(str(tmp_path), "worker-1") is None


# ── Turn 13114: idle heartbeat for Terminal-launch worker liveness ────────────


def test_write_worker_idle_heartbeat_writes_json(tmp_path):
    """write_worker_idle_heartbeat writes worker.heartbeat.json (NOT the task
    heartbeat.json) into the worker's OWN poll dir, marking state=idle."""
    from external_llm.agent.subagent_ipc import (
        write_worker_idle_heartbeat, _IDLE_HEARTBEAT_FILENAME,
    )
    import json as _json
    repo = tmp_path
    path = write_worker_idle_heartbeat(str(repo), "worker-1", pid=4242)
    assert path is not None
    hb_file = repo / ".asicode" / "subagents" / "worker-1" / _IDLE_HEARTBEAT_FILENAME
    assert hb_file.exists()
    data = _json.loads(hb_file.read_text())
    assert data["pid"] == 4242
    assert data["worker_id"] == "worker-1"
    assert data["state"] == "idle"
    assert data["ts"] > 0


def test_read_worker_idle_heartbeat_age_none_when_absent(tmp_path):
    """No heartbeat ⇒ None (inconclusive, NOT dead). Callers must keep
    optimistic reuse for legacy / pre-heartbeat workers."""
    from external_llm.agent.subagent_ipc import read_worker_idle_heartbeat_age
    assert read_worker_idle_heartbeat_age(str(tmp_path), "worker-1") is None


def test_read_worker_idle_heartbeat_age_returns_positive_age(tmp_path):
    """An existing heartbeat returns its age in seconds (positive)."""
    from external_llm.agent.subagent_ipc import (
        write_worker_idle_heartbeat, read_worker_idle_heartbeat_age,
    )
    import time as _time
    write_worker_idle_heartbeat(str(tmp_path), "worker-1", pid=1)
    # Backdate the ts by rewriting with an old timestamp.
    import json as _json
    hb = tmp_path / ".asicode" / "subagents" / "worker-1" / "worker.heartbeat.json"
    hb.write_text(_json.dumps({"ts": _time.time() - 100.0, "pid": 1}))
    age = read_worker_idle_heartbeat_age(str(tmp_path), "worker-1")
    assert age is not None and age >= 99.0


def test_write_worker_exited_heartbeat_overwrites_idle_state(tmp_path):
    """Fix 1: the exited marker is written to the SAME file as the idle
    heartbeat (worker.heartbeat.json), flipping state idle→exited so
    _claim_reusable_worker can drop the worker regardless of heartbeat age."""
    from external_llm.agent.subagent_ipc import (
        read_worker_idle_heartbeat_age,
        read_worker_idle_heartbeat_state,
        write_worker_exited_heartbeat,
        write_worker_idle_heartbeat,
    )
    repo = str(tmp_path)
    assert read_worker_idle_heartbeat_state(repo, "w1") is None
    p_idle = write_worker_idle_heartbeat(repo, "w1", pid=123)
    assert p_idle is not None
    assert read_worker_idle_heartbeat_state(repo, "w1") == "idle"
    p_exit = write_worker_exited_heartbeat(repo, "w1", pid=123)
    assert p_exit == p_idle  # same file, overwritten in place
    assert read_worker_idle_heartbeat_state(repo, "w1") == "exited"
    # Age still readable off the exited marker (fresh).
    age = read_worker_idle_heartbeat_age(repo, "w1")
    assert age is not None and age < 5.0


def test_poll_for_task_self_exits_when_orchestrator_pid_dead(tmp_path):
    """Fix 1: on the macOS Terminal launch path getppid() is the login shell,
    so the orphan check never fired when the orchestrator died. The explicit
    orchestrator_pid probe (--orch-pid) must self-exit promptly on a dead pid
    instead of polling until timeout_s/max_poll_s."""
    import subprocess as _sp
    import time as _time
    from external_llm.agent.subagent_ipc import poll_for_task

    proc = _sp.Popen(["sleep", "0.1"])
    dead_pid = proc.pid
    proc.wait()
    t0 = _time.monotonic()
    r = poll_for_task(
        repo_root=str(tmp_path), agent_id="w1", poll_interval_s=0.1,
        timeout_s=30.0, orchestrator_pid=dead_pid,
    )
    assert r is None
    assert (_time.monotonic() - t0) < 5.0, "must exit on the dead-pid probe, not the 30s timeout"


def test_poll_for_task_keeps_polling_when_orchestrator_pid_alive(tmp_path):
    """Fix 1 negative case: a LIVE orchestrator pid must not trip the probe —
    the normal timeout governs."""
    import os as _os
    import time as _time
    from external_llm.agent.subagent_ipc import poll_for_task

    t0 = _time.monotonic()
    r = poll_for_task(
        repo_root=str(tmp_path), agent_id="w1", poll_interval_s=0.1,
        timeout_s=0.5, orchestrator_pid=_os.getpid(),
    )
    assert r is None
    assert (_time.monotonic() - t0) >= 0.5  # ran to the timeout, no false orphan


def test_write_worker_idle_heartbeat_observability_fields(tmp_path):
    """Fix 4: the idle heartbeat carries tasks_served/last_task_id/uptime_s so
    --json-stream consumers and diagnostics can see worker-pool health."""
    import json as _json
    from external_llm.agent.subagent_ipc import write_worker_idle_heartbeat

    write_worker_idle_heartbeat(
        str(tmp_path), "w1", pid=1,
        tasks_served=7, last_task_id="dev_3", uptime_s=123.456,
    )
    data = _json.loads(
        (tmp_path / ".asicode" / "subagents" / "w1" / "worker.heartbeat.json").read_text()
    )
    assert data["state"] == "idle"
    assert data["tasks_served"] == 7
    assert data["last_task_id"] == "dev_3"
    assert data["uptime_s"] == 123.5
    # Defaults stay backward-compatible (callers that do not track stats).
    write_worker_idle_heartbeat(str(tmp_path), "w2", pid=1)
    data2 = _json.loads(
        (tmp_path / ".asicode" / "subagents" / "w2" / "worker.heartbeat.json").read_text()
    )
    assert data2["tasks_served"] == 0 and data2["last_task_id"] == ""


def test_path_matches_scope_prefix_and_guard():
    """Fix 2: scope matching is prefix-aware for directory entries (unexpanded
    over-cap dirs), with an os.sep guard so 'a.txt' does not match 'a.txt.bak'."""
    import os as _os
    from external_llm.agent.subagent_ipc import _path_matches_scope

    scope = {_os.path.normpath(p) for p in ["src/utils", "top.py"]}
    assert _path_matches_scope("src/utils/a.py", scope)
    assert _path_matches_scope("src/utils/sub/b.py", scope)
    assert _path_matches_scope("top.py", scope)
    assert not _path_matches_scope("top.py.bak", scope)
    assert not _path_matches_scope("src/utils2/x.py", scope)


def test_partition_changed_files_prefix_matches_unexpanded_dir(tmp_path):
    """Fix 2: a bare directory entry in assigned_files (left unexpanded past
    _MAX_DIR_EXPANSION_FILES) must still classify files UNDER it as in-scope —
    exact-path comparison used to flag every such edit as a violation."""
    import subprocess as _sp
    from external_llm.agent.subagent_ipc import partition_changed_files

    _sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "big" / "sub").mkdir(parents=True)
    (tmp_path / "big" / "a.py").write_text("x")
    (tmp_path / "big" / "sub" / "b.py").write_text("x")
    (tmp_path / "outside.py").write_text("x")

    in_scope, out_scope = partition_changed_files(str(tmp_path), ["big"])
    assert {d["file"] for d in in_scope} == {"big/a.py", "big/sub/b.py"}
    assert {d["file"] for d in out_scope} == {"outside.py"}


# ── Heartbeat reader/writer dedup invariants ──────────────────────────────
# These pin the refactor that hoisted the duplicated open/parse/except skeleton
# of the four heartbeat readers into _load_heartbeat_json, and the two idle
# writers into _write_idle_heartbeat. The goal was to eliminate drift: the
# exception tuple used to be copy-pasted across four readers, so adding a new
# exception type to one (and forgetting the others) would make liveness
# judgements asymmetric. The tests below lock the single-source-of-truth and
# the reader/writer mkdir asymmetry so the dedup cannot silently regress.


def test_heartbeat_readers_share_single_exception_semantics(tmp_path):
    """DRIFT LOCK: all four heartbeat readers must collapse the SAME failure
    modes to None. If one reader is ever given a bespoke except clause again
    (bypassing _load_heartbeat_json), the others would disagree — a worker
    could read as 'alive' via one reader and 'dead' via another. We feed every
    reader identical broken input and assert identical (None) results."""
    from external_llm.agent.subagent_ipc import (
        _subagent_dir, _heartbeat_path, _IDLE_HEARTBEAT_FILENAME,
        read_heartbeat_state, read_heartbeat_age_s,
        read_worker_idle_heartbeat_state, read_worker_idle_heartbeat_age,
    )

    repo = str(tmp_path)

    def _write_both(payload_text):
        d = _subagent_dir(repo, "w1")
        with open(_heartbeat_path(d), "w") as f:
            f.write(payload_text)
        with open(os.path.join(d, _IDLE_HEARTBEAT_FILENAME), "w") as f:
            f.write(payload_text)

    # DRIFT LOCK targets the shared except tuple in _load_heartbeat_json
    # (FileNotFoundError, JSONDecodeError, ValueError, TypeError,
    # PermissionError, OSError). Each input below trips one of those at the
    # load stage, so ALL readers must collapse it to None — if any reader
    # re-acquired a bespoke except clause the asymmetry would surface here.
    # (Non-dict VALID JSON like [] is intentionally excluded: the load stage
    # succeeds on it, and readers historically diverged — read_heartbeat_state
    # returns the value as-is while others raise AttributeError on .get —
    # which the refactor preserves, not unifies.)
    for broken in ("{not valid json", "", "{trailing,", '   '):
        _write_both(broken)
        results = {
            "state": read_heartbeat_state(repo, "w1"),
            "age": read_heartbeat_age_s(repo, "w1"),
            "idle_state": read_worker_idle_heartbeat_state(repo, "w1"),
            "idle_age": read_worker_idle_heartbeat_age(repo, "w1"),
        }
        assert all(v is None for v in results.values()), (
            "reader asymmetry for payload=%r: %r — a reader stopped collapsing "
            "a failure mode to None (drift re-introduced)" % (broken, results)
        )


def test_heartbeat_age_collapses_bad_ts_to_none(tmp_path):
    """Both age readers must collapse a non-numeric / null / <=0 ts to None.

    Historically float(data.get('ts', 0)) ran INSIDE the broad except, so a
    string or null ts raised ValueError/TypeError that was swallowed. Hoisting
    the load into _load_heartbeat_json must not leak those exceptions —
    _heartbeat_age_from restores the None-collapsing semantics. This guards
    that helper directly (the idle age reader had no prior ts-edge coverage)."""
    from external_llm.agent.subagent_ipc import (
        _subagent_dir, _heartbeat_path, _IDLE_HEARTBEAT_FILENAME,
        read_heartbeat_age_s, read_worker_idle_heartbeat_age,
    )

    repo = str(tmp_path)
    ts_payloads = ['{"ts": "abc", "pid": 1}', '{"ts": null, "pid": 1}',
                   '{"ts": 0, "pid": 1}', '{"ts": -1, "pid": 1}']
    for payload in ts_payloads:
        d = _subagent_dir(repo, "w1")
        for fname in (_heartbeat_path(d), os.path.join(d, _IDLE_HEARTBEAT_FILENAME)):
            with open(fname, "w") as f:
                f.write(payload)
        assert read_heartbeat_age_s(repo, "w1") is None, payload
        assert read_worker_idle_heartbeat_age(repo, "w1") is None, payload


def test_heartbeat_readers_do_not_create_directory(tmp_path):
    """Reader/writer mkdir asymmetry must be preserved: the readers use
    _subagent_dir_path (no mkdir) so probing liveness of a worker that never
    started does NOT litter an empty .asicode/subagents/<id>/ dir. Only the
    writers (_write_idle_heartbeat -> _subagent_dir) may create it."""
    from external_llm.agent.subagent_ipc import (
        read_heartbeat_state, read_heartbeat_age_s,
        read_worker_idle_heartbeat_state, read_worker_idle_heartbeat_age,
    )

    repo = str(tmp_path)
    worker_dir = os.path.join(repo, ".asicode", "subagents", "never-started")
    assert not os.path.exists(worker_dir)

    # Every reader probes a worker that has no directory at all.
    read_heartbeat_state(repo, "never-started")
    read_heartbeat_age_s(repo, "never-started")
    read_worker_idle_heartbeat_state(repo, "never-started")
    read_worker_idle_heartbeat_age(repo, "never-started")

    assert not os.path.exists(worker_dir), (
        "a heartbeat reader created the subagent dir — the reader/writer "
        "mkdir asymmetry was broken (readers must use _subagent_dir_path)"
    )

    # Sanity: a writer DOES create it.
    from external_llm.agent.subagent_ipc import write_worker_idle_heartbeat
    write_worker_idle_heartbeat(repo, "never-started", pid=1)
    assert os.path.isdir(worker_dir)
