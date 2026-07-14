"""Regression tests for ``AutonomousTaskQueue`` file-deduplication.

Covers a defect fixed in this change: when two tasks target the same
``source_file`` and the *newer* task has higher priority (lower number),
the priority queue dequeued the newer task first and cleared the dedup
map entry for that file. The older, superseded task then saw a ``None``
map entry, fell through the dedup check, and ran anyway — defeating the
documented guarantee ("a later event for the same source_file replaces
any pending task for that file") and consuming a concurrency slot.

Root cause: the dedup predicate had two branches — drop when
``current != task_id`` (newer pending), delete when
``current == task_id`` — but the ``current is None`` case (newer already
ran and cleared the entry) was unhandled and fell through to run the
stale task. The fix collapses the predicate to a single rule: a task
with a ``source_file`` runs *only* when it is still the newest pending
task for that file (``current == task_id``); everything else is dropped.
"""

from external_llm.editor.agent.autonomous.task_queue import (
    AutonomousTaskQueue,
)
from external_llm.editor.agent.autonomous.trigger_engine import (
    TriggerEvent,
    TriggerKind,
)
from external_llm.editor.agent.autonomous.trigger_policy import (
    ActionDecision,
    ActionKind,
)


def _ev(source_file=None):
    return TriggerEvent(
        kind=TriggerKind.FILE_MODIFIED, repo_root=".", source_file=source_file
    )


def _dec(priority, kind=ActionKind.AUTO_FIX):
    return ActionDecision(kind=kind, priority=priority)


def _drain(q):
    """Dequeue all runnable tasks, marking each done."""
    out = []
    while True:
        t = q.get_nowait()
        if t is None:
            break
        out.append((t.task_id, t.action.priority, t.event.source_file))
        q.task_done(t.task_id)
    return out


def test_dedup_when_newer_task_has_higher_priority():
    """Newer higher-priority task supersedes the older one — older must NOT run.

    Before the fix, the older task (auto-1) leaked through and ran because the
    newer task (auto-2) had already been dequeued first and cleared the map
    entry, leaving ``current is None`` for the stale task.
    """
    q = AutonomousTaskQueue()
    q.enqueue(_ev("X.py"), _dec(2))  # older, normal
    q.enqueue(_ev("X.py"), _dec(0))  # newer, critical (supersedes)

    got = _drain(q)
    assert got == [("auto-2", 0, "X.py")]


def test_dedup_three_same_file_descending_priority():
    """Only the newest task (auto-3) must run; older siblings are superseded."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("Y.py"), _dec(2))  # auto-1
    q.enqueue(_ev("Y.py"), _dec(1))  # auto-2
    q.enqueue(_ev("Y.py"), _dec(0))  # auto-3 (newest)

    got = _drain(q)
    assert got == [("auto-3", 0, "Y.py")]


def test_dedup_classic_same_priority_newer_wins():
    """Equal-priority dedup: the newer of two same-file tasks wins."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("Z.py"), _dec(2))
    q.enqueue(_ev("Z.py"), _dec(2))

    got = _drain(q)
    assert got == [("auto-2", 2, "Z.py")]


def test_different_files_both_run():
    """Unrelated files are independent — both tasks run."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("A.py"), _dec(2))
    q.enqueue(_ev("B.py"), _dec(2))

    got = sorted(_drain(q))
    assert got == [("auto-1", 2, "A.py"), ("auto-2", 2, "B.py")]


def test_no_source_file_tasks_are_never_deduped():
    """Tasks without a source_file bypass dedup entirely."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev(None), _dec(2))
    q.enqueue(_ev(None), _dec(2))

    got = _drain(q)
    assert len(got) == 2


def test_concurrency_limit_blocks_third_running_task():
    """MAX_CONCURRENT (2) slots enforced; 3rd task waits until one finishes."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("A.py"), _dec(2))
    q.enqueue(_ev("B.py"), _dec(2))
    q.enqueue(_ev("C.py"), _dec(2))

    t1 = q.get_nowait()
    t2 = q.get_nowait()
    assert t1 is not None and t2 is not None
    assert q.get_nowait() is None  # at limit

    q.task_done(t1.task_id)
    t3 = q.get_nowait()
    assert t3 is not None
    q.task_done(t2.task_id)
    q.task_done(t3.task_id)


def test_supersede_does_not_leak_map_or_queue():
    """After draining, the dedup map and underlying queue are fully drained."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("X.py"), _dec(2))
    q.enqueue(_ev("X.py"), _dec(0))  # supersedes auto-1

    _drain(q)
    assert q._pending_file_map == {}
    assert q.qsize() == 0


def test_ignore_action_not_enqueued():
    """IGNORE decisions return None and are never enqueued."""
    q = AutonomousTaskQueue()
    tid = q.enqueue(_ev("X.py"), _dec(2, kind=ActionKind.IGNORE))
    assert tid is None
    assert q.get_nowait() is None


def test_task_done_clamps_at_zero():
    """Extra task_done() calls never drive the count negative."""
    q = AutonomousTaskQueue()
    q.enqueue(_ev("X.py"), _dec(2))
    t = q.get_nowait()
    q.task_done(t.task_id)
    q.task_done(t.task_id)  # spurious second done
    assert q.running_count() == 0


# ── Purge/concurrency-decoupling regression tests ───────────────────────────
#
# These guard the fix for a latent unbounded-growth defect: the original
# get_nowait() returned None immediately when _running_count >= MAX_CONCURRENT,
# *before* entering the purge loop. So while both execution slots were busy
# (long AUTO_FIX runs), superseded/orphaned tasks enqueued for the same file
# were never reaped and accumulated indefinitely. The fix moves purge ahead of
# the concurrency check so stale tasks are always reaped; a live task found at
# the limit is re-queued for the next drain iteration.


def test_superseded_purged_while_slots_saturated():
    """Both slots busy → enqueue many same-file tasks → all but newest purged.

    Before the fix this test FAILED: get_nowait() returned None without purging,
    leaving all 10 tasks in the underlying queue (qsize stayed 10).
    """
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 100  # headroom — we're testing purge, not the cap

    # Occupy both execution slots with different files.
    q.enqueue(_ev("A.py"), _dec(2))
    q.enqueue(_ev("B.py"), _dec(2))
    t1 = q.get_nowait()
    t2 = q.get_nowait()
    assert t1 is not None and t2 is not None
    assert q.running_count() == 2

    # Burst: 10 tasks for the SAME file, each superseding the previous.
    # Only the last (newest task_id) is "live"; the other 9 are superseded.
    for _ in range(10):
        q.enqueue(_ev("C.py"), _dec(2))
    assert q.qsize() == 10  # physically present, purge hasn't run yet

    # One get_nowait() call must purge all 9 superseded tasks even though we're
    # at the concurrency limit (returns None, but the purge side-effect ran).
    result = q.get_nowait()
    assert result is None  # still at limit — no slot claimed
    # Only the single live C.py task remains (re-queued, waiting for a slot).
    assert q.qsize() == 1


def test_live_task_requeued_then_runs_when_slot_frees():
    """A live task re-queued at the concurrency limit runs once a slot frees."""
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 100

    q.enqueue(_ev("A.py"), _dec(2))
    q.enqueue(_ev("B.py"), _dec(2))
    t1 = q.get_nowait()
    t2 = q.get_nowait()

    # A third, distinct-file live task cannot run yet (at limit).
    q.enqueue(_ev("C.py"), _dec(2))
    assert q.get_nowait() is None  # re-queued, not purged (different file)
    assert q.qsize() == 1

    # Free a slot — the re-queued C.py task must now run.
    q.task_done(t1.task_id)
    t3 = q.get_nowait()
    assert t3 is not None
    assert t3.event.source_file == "C.py"
    q.task_done(t2.task_id)
    q.task_done(t3.task_id)


def test_cap_rejects_newest_at_max_pending():
    """Enqueue beyond MAX_PENDING rejects the newest task (newest-loss)."""
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 3
    # Fill to cap with distinct files (all live, none superseded).
    ids = [q.enqueue(_ev(f"f{i}.py"), _dec(2)) for i in range(3)]
    assert all(ids)
    assert q.qsize() == 3

    # 4th enqueue must be rejected.
    rejected = q.enqueue(_ev("f3.py"), _dec(2))
    assert rejected is None
    assert q.qsize() == 3  # unchanged
    assert q.dropped_count() == 1

    # A 5th rejection increments the counter further.
    q.enqueue(_ev("f4.py"), _dec(2))
    assert q.dropped_count() == 2


def test_cap_does_not_interfere_with_dedup():
    """Same-file superseding enqueues must not be blocked by the cap: they don't
    grow the *live* task count (the old entry is superseded, not duplicated)."""
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 2
    q.enqueue(_ev("X.py"), _dec(2))
    q.enqueue(_ev("Y.py"), _dec(2))
    assert q.qsize() == 2

    # Same-file supersede: physically adds a 3rd PQ entry, but only one is live.
    q.enqueue(_ev("X.py"), _dec(0))  # supersedes the first X.py
    assert q.dropped_count() == 0  # NOT rejected — cap counts PQ entries, and
    # the cap check happens at enqueue before we know it's a supersede. This is
    # acceptable: the superseded entry is purged on the next get_nowait().
    got = _drain(q)
    # Newest X.py (priority 0) and Y.py (priority 2) run; old X.py is purged.
    assert ("auto-3", 0, "X.py") in got
    assert ("auto-2", 2, "Y.py") in got
    assert len(got) == 2


def test_purge_keeps_queue_bounded_under_same_file_burst():
    """Integration: slots saturated + large same-file burst → queue stays tiny.

    Before the fix, qsize grew to the burst size (e.g. 50) and only drained
    once a slot freed. Now purge runs on every get_nowait() regardless of
    concurrency, so qsize collapses to 1 immediately.
    """
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 200

    q.enqueue(_ev("A.py"), _dec(2))
    q.enqueue(_ev("B.py"), _dec(2))
    q.get_nowait()
    q.get_nowait()
    assert q.running_count() == 2

    for _ in range(50):
        q.enqueue(_ev("Z.py"), _dec(2))
    assert q.qsize() == 50

    # A single drain probe collapses the backlog to the 1 live task.
    assert q.get_nowait() is None
    assert q.qsize() == 1
    assert q.dropped_count() == 0  # cap never hit


# ── Cap vs. tombstone-inflation regression tests ───────────────────────────
#
# Guards a defect where the enqueue cap check used ``qsize()``, which counts
# tombstones (superseded entries not yet reaped by get_nowait()). A same-file
# supersede burst (e.g. editor autosave flood between drain cycles) inflated
# qsize far beyond the *live* task count, so the next *different* file's first
# enqueue was wrongly rejected — the defense-in-depth cap was killing live work
# via a false positive. The fix purges tombstones at the cap decision point and
# re-checks against the true live count.


def test_cap_purges_tombstones_before_rejecting_new_file():
    """Tombstones from a same-file supersede burst must NOT count against
    MAX_PENDING and drop an unrelated file's first task.

    Before the fix: qsize == 6 (1 live + 5 tombstones) >= MAX_PENDING(4), so
    Y.py was rejected (dropped_count == 1) even though only ONE live task
    existed.
    """
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 4
    # One live task for X.py.
    q.enqueue(_ev("X.py"), _dec(2))
    # Burst of supersedes for the SAME file: each is cap-exempt (is_supersede),
    # so all enqueue. Physically qsize grows, but only the newest is live; the
    # rest are tombstones not yet reaped by get_nowait().
    for _ in range(5):
        q.enqueue(_ev("X.py"), _dec(2))
    assert q.qsize() == 6  # 1 live + 5 tombstones, past the cap

    # An unrelated file's FIRST task must still be accepted: purging the 5
    # tombstones leaves a true live count of 1, well under cap.
    tid = q.enqueue(_ev("Y.py"), _dec(2))
    assert tid is not None
    assert q.dropped_count() == 0
    # Tombstones really were purged — qsize collapsed to the 2 live tasks.
    assert q.qsize() == 2

    # Both live tasks survive to run; old X.py tombstones never run.
    got = _drain(q)
    assert len(got) == 2
    files = {row[2] for row in got}
    assert files == {"X.py", "Y.py"}
    # Only the newest X.py task (auto-6) ran, not any superseded sibling.
    x_rows = [row for row in got if row[2] == "X.py"]
    assert x_rows == [("auto-6", 2, "X.py")]


def test_cap_still_rejects_when_genuinely_full_of_live_tasks():
    """After purging tombstones, a queue genuinely full of LIVE tasks must
    still reject new enqueues — the fix must not weaken the cap.

    Guards the over-acceptance mutation where purge runs but the re-check is
    skipped (always accept after purge): here purge finds no tombstones, live
    count stays at the cap, so the new task must still be dropped.
    """
    q = AutonomousTaskQueue()
    q.MAX_PENDING = 3
    # Fill with distinct live files (no tombstones).
    for i in range(3):
        assert q.enqueue(_ev(f"f{i}.py"), _dec(2)) is not None
    assert q.qsize() == 3

    # 4th distinct file: purge finds no tombstones, live count still 3 >= cap.
    assert q.enqueue(_ev("g.py"), _dec(2)) is None
    assert q.dropped_count() == 1
    assert q.qsize() == 3  # unchanged — nothing purged, nothing added
