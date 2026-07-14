"""Tests for the per-repo persistent failure-pattern store."""
from __future__ import annotations

import time
import types

import pytest

from external_llm.agent.failure_classifier import RecoveryAction
from external_llm.agent.failure_pattern_store import (
    FailurePatternStore,
    get_store,
    recall_on_failure,
    record_recall_outcome,
    get_recall_counts,
    reset_recall_counts,
    reset_recall_session,
    _decay_weight,
    _FLUSH_INTERVAL,
    _MAX_PATHS,
)


@pytest.fixture
def store(tmp_path):
    return FailurePatternStore(tmp_path)


@pytest.fixture(autouse=True)
def _reset_recall_counters():
    """Recall efficacy counters are process-global; isolate every test."""
    reset_recall_counts()
    reset_recall_session()
    yield
    reset_recall_counts()
    reset_recall_session()


# ── record / count ───────────────────────────────────────────────────────────

def test_record_increments_count(store):
    assert store.record("apply_patch", "patch context mismatch") >= 1
    assert store.effective_count("apply_patch", "patch context mismatch") == 1
    store.record("apply_patch", "patch context mismatch")
    assert store.effective_count("apply_patch", "patch context mismatch") == 2


def test_distinct_reasons_kept_separately(store):
    store.record("apply_patch", "context mismatch")
    store.record("apply_patch", "file missing")
    assert store.effective_count("apply_patch", "context mismatch") == 1
    assert store.effective_count("apply_patch", "file missing") == 1


# ── persistence ──────────────────────────────────────────────────────────────

def test_persists_across_instances(tmp_path):
    s1 = FailurePatternStore(tmp_path)
    s1.record("edit_text", "old_string not found")
    s1.flush()  # batching defers disk writes; flush forces them before cross-instance read
    assert (tmp_path / ".asicode" / "failure_patterns.json").exists()
    # New instance sees the prior record.
    s2 = FailurePatternStore(tmp_path)
    assert s2.effective_count("edit_text", "old_string not found") == 1


def test_corrupt_file_treated_as_empty(tmp_path):
    p = tmp_path / ".asicode"
    p.mkdir()
    (p / "failure_patterns.json").write_text("{not valid json", encoding="utf-8")
    s = FailurePatternStore(tmp_path)
    # Must not raise; behaves as empty.
    assert s.top_patterns() == []
    assert s.effective_count("any", "thing") == 0


# ── recall ───────────────────────────────────────────────────────────────────

def test_recall_silent_below_threshold(store):
    store.record("apply_patch", "context mismatch")
    store.record("apply_patch", "context mismatch")
    assert store.recall_for("apply_patch", "context mismatch") == ""  # count=2 < 3


def test_recall_fires_at_threshold(store):
    for _ in range(3):
        store.record("apply_patch", "context mismatch")
    hint = store.recall_for("apply_patch", "context mismatch")
    assert hint.startswith("[RECALL]")
    assert "3" in hint
    assert "change approach" in hint


def test_recall_scoped_to_reason(store):
    for _ in range(5):
        store.record("apply_patch", "context mismatch")
    # Same tool, different reason → not recurrent.
    assert store.recall_for("apply_patch", "file missing") == ""


# ── recall_on_failure (unified hook for both agent loops) ────────────────────


def _mk_result(error):
    """Build a minimal ToolResult-shaped object the classifier can read."""
    return types.SimpleNamespace(error=error, ok=False, content="", metadata={})


def test_recall_on_failure_first_observation_silent(tmp_path):
    reset_recall_session()
    # First observation of a recallable reason → recorded, but below threshold → "".
    hint = recall_on_failure("read_file", {}, _mk_result(FileNotFoundError("x")), tmp_path)
    assert hint == ""
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 1


def test_recall_on_failure_fires_when_recurrent(tmp_path):
    reset_recall_session()
    store = get_store(tmp_path)
    # Pre-seed 2 observations directly (bypass the dedup set).
    store.record("apply_patch", "patch context mismatch")
    store.record("apply_patch", "patch context mismatch")
    store.flush()
    # recall_on_failure adds the 3rd observation and surfaces the hint.
    hint = recall_on_failure("apply_patch", {}, _mk_result(ValueError("hunk rejected")), tmp_path)
    assert hint.startswith("[RECALL]")
    assert "apply_patch" in hint
    assert "patch context mismatch" in hint


def test_recall_on_failure_dedup_within_run(tmp_path):
    reset_recall_session()
    result = _mk_result(FileNotFoundError("nope"))
    # First call in run "A" records (0 → 1).
    assert recall_on_failure("read_file", {}, result, tmp_path, session_key="A") == ""
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 1
    # Second call in the SAME run (same session_key) is deduped: no increment,
    # no hint — retries within one run must not inflate the cross-run count.
    assert recall_on_failure("read_file", {}, result, tmp_path, session_key="A") == ""
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 1
    # A NEW run (different session_key) re-records (1 → 2): per-run scoping
    # means a fresh run may observe the same failure again.
    assert recall_on_failure("read_file", {}, result, tmp_path, session_key="B") == ""
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 2


def test_recall_on_failure_rehints_in_new_run_at_threshold(tmp_path):
    """Regression: a long-lived process must keep re-firing the recall hint in
    each NEW run once the cross-run count reaches threshold.  A process-global
    dedup set with no session_key would silence the hint after the very first
    observation and never deliver it to later conversations that need it."""
    reset_recall_session()
    store = get_store(tmp_path)
    result = _mk_result(ValueError("hunk rejected"))
    # Three distinct runs (A, B, C): each records once → count climbs 1, 2, 3.
    assert recall_on_failure("apply_patch", {}, result, tmp_path, session_key="A") == ""
    assert recall_on_failure("apply_patch", {}, result, tmp_path, session_key="B") == ""
    hint = recall_on_failure("apply_patch", {}, result, tmp_path, session_key="C")
    assert hint.startswith("[RECALL]")  # count == 3 → fires
    # A FOURTH new run re-surfaces the hint again — not silenced by process state.
    hint2 = recall_on_failure("apply_patch", {}, result, tmp_path, session_key="D")
    assert hint2.startswith("[RECALL]")
    assert store.effective_count("apply_patch", "patch context mismatch") == 4


def test_recall_on_failure_dedup_scoped_per_repo(tmp_path, tmp_path_factory):
    reset_recall_session()
    other_repo = tmp_path_factory.mktemp("other_repo")
    result = _mk_result(FileNotFoundError("nope"))
    # Same (tool, reason) in two different repos → each records once.
    recall_on_failure("read_file", {}, result, tmp_path)
    recall_on_failure("read_file", {}, result, other_repo)
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 1
    assert get_store(other_repo).effective_count("read_file", "file missing") == 1


def test_recall_on_failure_skips_non_recallable_reason(tmp_path):
    reset_recall_session()
    # A bare RuntimeError with no recognizable markers → "generic failure",
    # which is in _NON_RECALLABLE_REASONS → not recorded, no hint.
    assert recall_on_failure("read_file", {}, _mk_result(RuntimeError("zzz")), tmp_path) == ""
    assert get_store(tmp_path).effective_count("read_file", "generic failure") == 0


def test_recall_on_failure_classifies_raised_exception(tmp_path):
    reset_recall_session()
    # dispatch raised (result is None); the exc is classified via a shim so the
    # type hierarchy still applies → FileNotFoundError → "file missing".
    assert recall_on_failure("read_file", {}, None, tmp_path, exc=FileNotFoundError("x")) == ""
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 1


def test_recall_on_failure_no_repo_root_returns_empty(tmp_path):
    reset_recall_session()
    result = _mk_result(FileNotFoundError("x"))
    assert recall_on_failure("read_file", {}, result, None) == ""
    assert recall_on_failure("read_file", {}, result, "") == ""
    # Nothing recorded against a falsy root.
    assert get_store(tmp_path).effective_count("read_file", "file missing") == 0


def test_recall_on_failure_never_raises(tmp_path):
    reset_recall_session()
    # A non-object result with no .error attribute → classified as "generic
    # failure" → non-recallable → "". Must not raise.
    assert recall_on_failure("read_file", {}, {"error": "x"}, tmp_path) == ""
    assert recall_on_failure("read_file", {}, object(), tmp_path) == ""


# ── decay ────────────────────────────────────────────────────────────────────

def test_decay_weight_halves_over_half_life():
    now = 1_000_000.0
    assert _decay_weight(now, now) == pytest.approx(1.0)
    half = _decay_weight(now - 7 * 24 * 3600, now)
    assert 0.45 < half < 0.55


def test_decay_reduces_effective_count(store, monkeypatch):
    # Record with old timestamp → effective count decays below raw count.
    store.record("t", "r")
    store.record("t", "r")
    store.record("t", "r")
    raw = store._load()["t::r"]["count"]
    # Force last_seen far into the past.
    with store._lock:
        data = store._load()
        data["t::r"]["last_seen"] = time.time() - 365 * 24 * 3600  # 1 year
        store._cache = data
        store._save(data)
    eff = store.effective_count("t", "r")
    assert eff < raw


# ── registry ─────────────────────────────────────────────────────────────────

def test_get_store_caches_per_repo(tmp_path):
    a = get_store(tmp_path)
    b = get_store(tmp_path)
    assert a is b


def test_disabled_store_is_noop(tmp_path):
    s = FailurePatternStore(tmp_path, enabled=False)
    assert s.record("t", "r") == 0
    assert s.recall_for("t", "r") == ""


# ── bounded pruning ──────────────────────────────────────────────────────────

def test_bounded_prune_on_save(tmp_path):
    s = FailurePatternStore(tmp_path)
    # Exceed the cap with low-count patterns.
    for i in range(250):
        s.record(f"tool{i}", f"reason{i}")
    data = s._load()
    assert len(data) <= 250
    # Pruning keeps it from growing without bound; high-count survives.
    for _ in range(5):
        s.record("keeper", "hot")
    top = {p["key"] for p in s.top_patterns(limit=5)}
    assert any("keeper" in k for k in top)
def test_bounded_cap_enforced_when_all_patterns_stale(tmp_path):
    """The hard cap is enforced even when every pattern is a stale singleton
    (soft-prune keeps none of them).

    The existing ``test_bounded_prune_on_save`` only exercises the
    ``len(keep) > _MAX_PATTERNS`` branch (fresh singletons all survive
    soft-prune).  This test forces the *other* branch — every pattern decayed
    below the keep threshold — which previously fell through to
    ``merged = keep or merged`` and wrote the full over-capacity set, breaking
    the "bounded" contract.
    """
    import json

    from external_llm.agent.failure_pattern_store import _MAX_PATTERNS
    s = FailurePatternStore(tmp_path)
    # Record more than the cap, all distinct single occurrences.
    for i in range(_MAX_PATTERNS + 30):
        s.record(f"tool{i}", f"reason{i}")
    # Poison every pattern to a year-old singleton so soft-prune keeps NONE.
    with s._lock:
        data = s._load()
        stale = time.time() - 365 * 24 * 3600
        for v in data.values():
            v["count"] = 1
            v["last_seen"] = stale
        s._save(data, merge=False)
    # Read the persisted file directly, bypassing the in-memory cache.
    raw = json.loads(
        (tmp_path / ".asicode" / "failure_patterns.json").read_text("utf-8")
    )
    assert len(raw["patterns"]) <= _MAX_PATTERNS, (
        f"store exceeded cap: {len(raw['patterns'])} > {_MAX_PATTERNS}"
    )


# ── clear / drop (merge-on-write regression) ─────────────────────────────────
# _save() merges disk data on top of in-memory data for cross-process safety.
# clear()/drop() must bypass that merge (merge=False), otherwise the deleted
# keys are silently resurrected from disk and the operation becomes a no-op.

def test_clear_removes_all_patterns_from_disk(tmp_path):
    s = FailurePatternStore(tmp_path)
    s.record("apply_patch", "context mismatch")
    s.record("edit_text", "old_string not found")
    s.flush()
    assert s.store_size() == 2

    s.clear()

    # In-memory view is empty.
    assert s.store_size() == 0
    assert s.top_patterns() == []
    # Crucially, the *disk* file is also empty — this is what the merge=False
    # guard protects. Without it, _save() re-reads disk and resurrects all keys.
    disk = s._read_from_disk_unsafe()
    assert disk == {}, f"clear() left patterns on disk: {list(disk)}"


def test_drop_removes_specific_pattern_without_resurrecting(tmp_path):
    s = FailurePatternStore(tmp_path)
    s.record("apply_patch", "context mismatch")
    s.record("edit_text", "old_string not found")
    s.flush()
    assert s.store_size() == 2

    top = s.top_patterns(limit=10)
    assert len(top) == 2
    removed = s.drop(1)  # drop highest-ranked pattern
    assert removed is not None
    _tool, _reason = removed
    assert _tool == top[0]["tool"]
    assert _reason == top[0]["reason"]

    # The dropped key must not reappear (merge-on-write would resurrect it).
    remaining = {p["key"] for p in s.top_patterns(limit=10)}
    assert top[0]["key"] not in remaining
    assert len(remaining) == 1
    disk = s._read_from_disk_unsafe()
    assert top[0]["key"] not in disk, f"dropped key resurrected on disk: {top[0]['key']}"


def test_drop_out_of_range_returns_false(tmp_path):
    s = FailurePatternStore(tmp_path)
    s.record("t", "r")
    s.flush()
    assert s.drop(0) is None   # 1-based; 0 is invalid
    assert s.drop(99) is None  # out of range
    assert s.store_size() == 1  # unchanged


def test_drop_index_follows_top_patterns_score_order_not_insertion(tmp_path):
    """drop(N) must remove the N-th pattern shown by top_patterns() (score
    order), NOT the N-th-inserted pattern.

    Regression guard: an earlier drop() indexed into raw dict insertion order,
    which only coincided with top_patterns() when all patterns had equal scores.
    With divergent counts, insertion order != score order, so drop(1) silently
    removed the wrong (low-count) pattern. This test forces that divergence.
    """
    s = FailurePatternStore(tmp_path)
    # Insert a LOW-count pattern first, then a HIGH-count pattern second.
    # top_patterns() must rank the high-count one as #1.
    for _ in range(2):
        s.record("low_tool", "low_reason")
    for _ in range(8):
        s.record("high_tool", "high_reason")
    s.flush()

    top = s.top_patterns(limit=10)
    assert top[0]["tool"] == "high_tool"  # highest effective score is #1

    removed = s.drop(1)  # must remove the #1 displayed (high_tool)
    assert removed is not None
    _tool, _reason = removed
    assert _tool == "high_tool"
    assert _reason == "high_reason"

    remaining = [f"{p['tool']}::{p['reason']}" for p in s.top_patterns(limit=10)]
    assert "high_tool::high_reason" not in remaining, (
        "drop(1) removed the wrong pattern — it did not target top_patterns()[0]"
    )
    assert remaining == ["low_tool::low_reason"]


# ── drop_key ────────────────────────────────────────────────────────────────


def test_drop_key_removes_by_exact_key(store):
    """drop_key removes the pattern matching the exact tool::reason key."""
    store.record("tool_a", "reason_a")
    store.record("tool_b", "reason_b")
    store.flush()

    result = store.drop_key("tool_a::reason_a")
    assert result is not None
    assert result == ("tool_a", "reason_a")

    remaining = [f"{p['tool']}::{p['reason']}" for p in store.top_patterns(limit=10)]
    assert "tool_a::reason_a" not in remaining
    assert "tool_b::reason_b" in remaining


def test_drop_key_unknown_key_returns_none(store):
    """drop_key returns None for a non-existent key."""
    store.record("tool_a", "reason_a")
    result = store.drop_key("nonexistent::nope")
    assert result is None


def test_drop_key_leaves_other_patterns_intact(store):
    """drop_key removes only the targeted pattern, keeping all others."""
    for i in range(5):
        store.record(f"tool_{i}", f"reason_{i}")
    store.flush()

    store.drop_key("tool_2::reason_2")
    remaining = store.top_patterns(limit=10)
    assert len(remaining) == 4
    for p in remaining:
        assert p["key"] != "tool_2::reason_2"


def test_drop_key_persists_across_instances(tmp_path):
    """drop_key persists — a new store instance sees the deletion."""
    s1 = FailurePatternStore(tmp_path)
    s1.record("tool_a", "reason_a")
    s1.record("tool_b", "reason_b")
    s1.flush()

    s1.drop_key("tool_a::reason_a")
    s1.flush()  # ensure written to disk

    s2 = FailurePatternStore(tmp_path)
    remaining = s2.top_patterns(limit=10)
    assert len(remaining) == 1
    assert remaining[0]["key"] == "tool_b::reason_b"


# ── pending-preservation on drop/drop_key ────────────────────────────────────
# record() uses write-behind batching (_FLUSH_INTERVAL=5), so up to 4 records
# can exist only in-memory (_pending). Earlier bugs silently discarded pending
# records when drop/drop_key forced a cache-reload (self._cache = None) because
# the pending data was in the old cache object that got thrown away.
# The fix: call _flush_unsafe() before self._cache = None.
#
# These tests use drop_key() with an explicit key so they control exactly which
# pattern is removed, regardless of decay-based ranking.

def test_drop_key_preserves_pending_records(store):
    """drop_key() must not lose pending records when removing a different key."""
    store.record("committed", "reason_a")
    store.flush()

    # Pending record.
    store.record("pending", "reason_b")

    # Drop the committed key by exact key.
    removed = store.drop_key("committed::reason_a")
    assert removed is not None

    remaining = store.top_patterns(limit=10)
    remaining_keys = {p["key"] for p in remaining}
    assert "pending::reason_b" in remaining_keys, (
        f"pending record lost after drop_key(); keys={remaining_keys}"
    )

    # Disk view after explicit flush confirms the pending record wasn't lost.
    store.flush()
    s2 = FailurePatternStore(store.repo_root)
    s2_keys = {p["key"] for p in s2.top_patterns(limit=10)}
    assert "pending::reason_b" in s2_keys


def test_drop_index_with_pending_does_not_lose_pending(store):
    """drop() by index must preserve pending records — but note that the pending
    record may itself be the one removed if it ranks #1."""
    store.record("low_priority", "old")
    for _ in range(3):
        store.record("high_priority", "important")
    store.flush()

    # Pending record (slightly lower count, but most recent).
    store.record("pending", "survivor")

    # Drop by index 1 removes the top-ranked pattern. Since pending was just
    # recorded and high_priority has count=3 vs pending's count=1, high_priority
    # still outranks it. After the _flush_unsafe in drop(), pending is saved
    # to disk and the intended removal happens correctly.
    # Before the fix, pending disappeared entirely because _flush_unsafe wasn't
    # called and the cache discard threw away the unflushed record.
    removed = store.drop(1)
    assert removed is not None, "drop(1) should succeed"

    # high_priority should be the one removed (highest count).
    assert removed[0] == "high_priority", (
        f"drop(1) removed {removed} but expected high_priority"
    )

    # The pending record should still be present.
    remaining = store.top_patterns(limit=10)
    remaining_keys = {p["key"] for p in remaining}
    assert "pending::survivor" in remaining_keys, (
        f"pending record lost after drop(1); remaining keys: {remaining_keys}"
    )


# ── count inflation regression ───────────────────────────────────────────────
# _save() previously set baseline to pre-write disk_data instead of post-write
# merged, causing the delta-merge to double-count across successive flushes.
# E.g. 23 observations → persisted 53 (2.3× inflation).


def test_count_does_not_inflate_across_flush_boundaries(tmp_path):
    """record() across multiple flush boundaries must not inflate persisted count.

    Regression: baseline lag caused delta to include already-persisted increments.
    """
    s = FailurePatternStore(tmp_path)
    total = _FLUSH_INTERVAL * 4  # 20 observations → 4 flush boundaries
    assert _FLUSH_INTERVAL == 5, "test assumes FLUSH_INTERVAL=5"

    for i in range(total):
        s.record("tool", "reason")
        # Ensure flush boundaries by explicitly flushing mid-way.
        if i > 0 and i % (_FLUSH_INTERVAL + 1) == 0:
            s.flush()

    s.flush()  # final sync

    # Read from a fresh instance to get authoritative disk state.
    s2 = FailurePatternStore(tmp_path)
    persisted = s2.effective_count("tool", "reason")

    # Within decay tolerance (negligible within test time), must be ~total.
    assert abs(persisted - total) <= 1, (
        f"persisted count {persisted} should approximate observations {total}; "
        f"inflation likely due to baseline bug"
    )


def test_count_preserved_across_recall_flush_boundary(tmp_path):
    """Regression: recall_for() must NOT advance _baseline beyond disk state.

    The bug: recall_for() re-reads disk, merges with pending records, then
    sets _baseline to the *merged* count.  On the next flush, _merge_max
    computes delta = mem_c - baseline = 0 for those pending records,
    silently losing them.

    Sequence under test:
        1. Flush 2 observations to disk (disk=2).
        2. Record 3 more (pending=3, mem=5).
        3. recall_for() — with fix: baseline stays at disk(2) not merged(5).
        4. flush() — delta = 5-2 = 3, persisted = 2+3 = 5.
           With bug: baseline=5, delta=0, persisted=2.
    """
    s = FailurePatternStore(tmp_path)

    # Phase 1: 2 observations flushed to disk.
    s.record("tool", "reason")
    s.record("tool", "reason")
    s.flush()
    assert s._path.stat().st_size > 0, "disk file should exist after flush"

    # Phase 2: 3 more observations (pending, not flushed).
    for _ in range(3):
        s.record("tool", "reason")

    # Force TTL expiry so recall_for re-reads disk (default 2.0s).
    s._last_read_ts = 0.0

    # Trigger recall_for — this is where the bug manifests.
    hint = s.recall_for("tool", "reason", min_count=1)
    assert "RECALL" in hint, "should recall at >=5 total"

    # Flush: with correct baseline, disk becomes ~5.
    s.flush()

    # Read from fresh instance.
    s2 = FailurePatternStore(tmp_path)
    persisted = s2.effective_count("tool", "reason")

    # With decay negligible (<1s), count should be ~5.
    msg = (
        f"expected >=4, got {persisted}; "
        f"recall_for() baseline bug likely lost pending observations"
    )
    assert persisted >= 4, msg


def test_count_not_inflated_by_repeated_recall(tmp_path):
    """Sanity: repeated recall_for() must not inflate count.

    Even with correct baseline (disk_data), confirm that recall_for does not
    accidentally amplify the count on repeated calls.
    """
    s = FailurePatternStore(tmp_path)

    s.record("tool", "reason")
    s.flush()

    for _ in range(5):
        s._last_read_ts = 0.0
        hint = s.recall_for("tool", "reason", min_count=1)
        assert "RECALL" in hint

    s.flush()
    s2 = FailurePatternStore(tmp_path)
    c = s2.effective_count("tool", "reason")
    assert c <= 2, f"count inflated to {c} by repeated recall_for"


def test_decay_not_lost_by_delta_merge(tmp_path):
    """Regression: decay must survive _merge_max delta merge.

    The bug: record() applies decay (count = old*w + 1), but _merge_max
    delta approach computed disk_c + max(0, mem_c - bl_c) which becomes
    disk_c + 0 when mem_c < bl_c (decay reduced the count), silently
    losing the decay and writing the pre-decay disk count back.
    """
    import time, json
    from external_llm.agent.failure_pattern_store import FailurePatternStore

    now = time.time()
    old_ts = now - 30 * 86400  # 30 days ago

    # Seed disk with a 30-day-old entry (count=10, not yet decayed for today).
    s = FailurePatternStore(tmp_path)
    s._path.parent.mkdir(parents=True, exist_ok=True)
    s._path.write_text(json.dumps({
        "patterns": {
            "tool::reason": {
                "count": 10.0, "first_seen": old_ts, "last_seen": old_ts,
                "tool": "tool", "reason": "reason",
            }
        },
        "version": 1, "updated": now,
    }))

    # Fresh session: load → decay → observe → flush.
    s2 = FailurePatternStore(tmp_path)
    s2._load()
    s2.record("tool", "reason")
    s2.flush()

    # Fresh read: count must reflect decay (10*w + 1 ≈ 1.5), not the raw 10.
    s3 = FailurePatternStore(tmp_path)
    c = s3.effective_count("tool", "reason")
    assert 1 <= c <= 4, (
        f"decay lost by delta merge: expected ~1.5, got {c}; "
        f"disk likely retains pre-decay count"
    )


# ── cross-process decay merge (regression: P1) ──────────────────────────────
def test_cross_process_decay_preserves_other_process_increments(tmp_path):
    """Multi-process: decay + external disk bump must preserve both.

    Regression: _merge_max decay branch used ``merged_count = mem_c``,
    discarding increments written by another process after baseline capture.

    Scenario: both processes load bl_c=10; we decay to 4; other process
    adds +3 and flushes (disk_c=13).  Merge must produce 4 + (13-10) = 7,
    not 4 (which would lose the other process's +3).
    """
    import time
    from external_llm.agent.failure_pattern_store import FailurePatternStore

    s = FailurePatternStore(tmp_path)
    baseline: dict[str, int] = {"tool::reason": 10}
    now = time.time()
    ts_old = now - 100

    disk_data = {
        "tool::reason": {"count": 13, "last_seen": now, "first_seen": ts_old,
                         "tool": "tool", "reason": "reason"},
    }
    our_data = {
        "tool::reason": {"count": 4, "last_seen": now, "first_seen": ts_old,
                         "tool": "tool", "reason": "reason"},
    }

    merged = s._merge_max(disk_data, our_data, baseline=baseline)
    merged_count = merged["tool::reason"]["count"]
    expected = 4 + max(0, 13 - 10)  # our decay + their increment
    assert merged_count == expected, (
        f"expected {expected} (decayed + external delta), got {merged_count}; "
        f"with bug this would be 4 (other process's +3 lost)"
    )


def test_cross_process_decay_single_process_unchanged(tmp_path):
    """Single-process behaviour: decay branch must produce same result as before.

    When disk_c == bl_c (no other process touched the file), the formula
    ``mem_c + max(0, disk_c - bl_c)`` must reduce to ``mem_c`` — the same
    as the original ``merged_count = mem_c``.
    """
    import time
    from external_llm.agent.failure_pattern_store import FailurePatternStore

    s = FailurePatternStore(tmp_path)
    baseline: dict[str, int] = {"tool::reason": 10}
    now = time.time()
    ts_old = now - 100

    # disk / baseline match — single-process scenario
    disk_data = {
        "tool::reason": {"count": 10, "last_seen": now, "first_seen": ts_old,
                         "tool": "tool", "reason": "reason"},
    }
    our_data = {
        "tool::reason": {"count": 4, "last_seen": now, "first_seen": ts_old,
                         "tool": "tool", "reason": "reason"},
    }

    merged = s._merge_max(disk_data, our_data, baseline=baseline)
    merged_count = merged["tool::reason"]["count"]
    assert merged_count == 4, (
        f"single-process: expected 4 (decayed count preserved), got {merged_count}"
    )


# ── ① action-shaped recall advice ────────────────────────────────────────────

def test_recall_advice_reflects_action(store):
    """The classifier's RecoveryAction must shape the advice, not a one-size generality."""
    store.record("edit_text", "file missing")
    store.record("edit_text", "file missing")
    store.record("edit_text", "file missing")
    switch = store.recall_for("edit_text", "file missing", action=RecoveryAction.SWITCH_TOOL)
    read = store.recall_for("edit_text", "file missing", action=RecoveryAction.READ_FIRST)
    abort = store.recall_for("edit_text", "file missing", action=RecoveryAction.ABORT)
    skip = store.recall_for("edit_text", "file missing", action=RecoveryAction.SKIP)
    assert "switch to a different tool" in switch
    assert "read the target first" in read
    assert "permission/environment" in abort
    assert "already applied" in skip
    # And each differs from the others (not all the generic string).
    assert len({switch, read, abort, skip}) == 4


def test_recall_advice_falls_back_to_generic_when_action_unknown(store):
    store.record("edit_text", "file missing")
    store.record("edit_text", "file missing")
    store.record("edit_text", "file missing")
    none_hint = store.recall_for("edit_text", "file missing", action=None)
    obj_hint = store.recall_for("edit_text", "file missing", action=object())
    assert "change approach" in none_hint
    assert "change approach" in obj_hint


def test_recall_on_failure_forwards_action(tmp_path):
    """recall_on_failure must forward the classifier's action to recall_for."""
    s = FailurePatternStore(tmp_path)
    for _ in range(3):
        s.record("edit_text", "file missing")
    s.flush()
    reset_recall_session()
    # FileNotFoundError → classifier maps to SWITCH_TOOL / "file missing"
    err = types.SimpleNamespace(error=FileNotFoundError("nope"), ok=False)
    hint = recall_on_failure("edit_text", {"path": "x.py"}, err, tmp_path, session_key="run-a")
    assert hint
    assert "switch to a different tool" in hint


# ── ② per-path breakdown ─────────────────────────────────────────────────────

def test_record_accumulates_per_path_breakdown(store):
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="b.py")
    entry = store._load()["edit_text::file missing"]
    assert entry["paths"] == {"a.py": 2.0, "b.py": 1.0}


def test_record_normalizes_absolute_path_to_repo_relative(tmp_path):
    s = FailurePatternStore(tmp_path)
    s.record("edit_text", "file missing", path=str(tmp_path / "src" / "mod.py"))
    entry = s._load()["edit_text::file missing"]
    assert entry["paths"] == {"src/mod.py": 1.0}


def test_recall_hint_names_top_path(store):
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="b.py")
    hint = store.recall_for("edit_text", "file missing", action=RecoveryAction.SWITCH_TOOL)
    assert "Mostly on `a.py`" in hint
    assert "75% of 2 affected paths" in hint


def test_recall_hint_single_path_says_concentrated(store):
    store.record("edit_text", "file missing", path="only.py")
    store.record("edit_text", "file missing", path="only.py")
    store.record("edit_text", "file missing", path="only.py")
    hint = store.recall_for("edit_text", "file missing", action=RecoveryAction.SWITCH_TOOL)
    assert "concentrated on `only.py`" in hint


def test_key_not_fragmented_by_path(store):
    """Different paths must NOT split the key — the threshold still triggers."""
    store.record("edit_text", "file missing", path="a.py")
    store.record("edit_text", "file missing", path="b.py")
    store.record("edit_text", "file missing", path="c.py")
    # 3 observations across 3 files → key count == 3 → recall fires.
    assert store.recall_for("edit_text", "file missing", action=RecoveryAction.SWITCH_TOOL)


def test_paths_bounded_to_top_n(store):
    for i in range(_MAX_PATHS + 3):
        store.record("edit_text", "file missing", path=f"f{i}.py")
    entry = store._load()["edit_text::file missing"]
    assert len(entry["paths"]) == _MAX_PATHS


def test_merge_max_reconciles_paths_across_processes(tmp_path):
    """Cross-process merge must not drop the per-path breakdown."""
    s1 = FailurePatternStore(tmp_path)
    s1.record("edit_text", "file missing", path="shared.py")
    s1.flush()
    s2 = FailurePatternStore(tmp_path)
    s2.record("edit_text", "file missing", path="shared.py")
    s2.record("edit_text", "file missing", path="only_s2.py")
    s2.flush()
    entry = FailurePatternStore(tmp_path)._load()["edit_text::file missing"]
    assert entry["paths"]["shared.py"] == 2.0
    assert entry["paths"]["only_s2.py"] == 1.0


# ── ③ recall efficacy counters ───────────────────────────────────────────────

def _recurring_failure(tmp_path, session_key, tool="apply_patch", reason="patch context mismatch"):
    s = FailurePatternStore(tmp_path)
    for _ in range(3):
        s.record(tool, reason)
    s.flush()
    err = types.SimpleNamespace(error=ValueError("hunk context mismatch"), ok=False)
    return recall_on_failure(tool, {"path": "x.py"}, err, tmp_path, session_key=session_key)


def test_recall_counter_fired_increments_on_hint(tmp_path):
    hint = _recurring_failure(tmp_path, session_key="run-1")
    assert hint
    assert get_recall_counts()["fired"] == 1


def test_recall_counter_helped_after_success(tmp_path):
    _recurring_failure(tmp_path, session_key="run-1")
    record_recall_outcome(ok=True, session_key="run-1")
    counts = get_recall_counts()
    assert counts == {"fired": 1, "helped": 1, "ignored": 0}


def test_recall_counter_ignored_after_failure(tmp_path):
    _recurring_failure(tmp_path, session_key="run-1")
    record_recall_outcome(ok=False, session_key="run-1")
    counts = get_recall_counts()
    assert counts == {"fired": 1, "helped": 0, "ignored": 1}


def test_recall_outcome_noop_when_no_marker_pending(tmp_path):
    # No recall fired for this run → settling is a no-op.
    record_recall_outcome(ok=True, session_key="never-fired")
    assert get_recall_counts() == {"fired": 0, "helped": 0, "ignored": 0}


def test_recall_outcome_noop_without_session_key(tmp_path):
    _recurring_failure(tmp_path, session_key="run-1")
    record_recall_outcome(ok=True, session_key="")  # empty key → no-op
    assert get_recall_counts()["helped"] == 0


def test_recall_marker_settled_once_not_double_counted(tmp_path):
    _recurring_failure(tmp_path, session_key="run-1")
    record_recall_outcome(ok=True, session_key="run-1")
    record_recall_outcome(ok=True, session_key="run-1")  # marker already gone
    assert get_recall_counts()["helped"] == 1


def test_reset_recall_counts_clears_counters(tmp_path):
    _recurring_failure(tmp_path, session_key="run-1")
    record_recall_outcome(ok=True, session_key="run-1")
    snap = reset_recall_counts()
    assert snap["fired"] == 1
    assert get_recall_counts() == {"fired": 0, "helped": 0, "ignored": 0}
