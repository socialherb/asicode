"""
Per-repo persistent failure-pattern store with recall.

Closes the gap that ``FailureClassifier`` classifies each failure in isolation
but never *accumulates* across runs: the same (tool, reason) failure happening
repeatedly in a repo is invisible to future turns. This store persists a
recency-decayed frequency table to ``.asicode/failure_patterns.json`` so the
agent can be nudged — "this exact failure has happened N times here before,
change approach" — at the moment a failure recurs.

Design notes
------------
* **Bounded**: caps total distinct patterns (LRU-ish prune of stale, low-count
  entries on save) so a noisy repo can't grow the file unbounded.
* **Decay**: ``recency_weight`` down-weights old observations so a transient
  spate of failures doesn't dominate forever after the underlying bug is fixed.
* **Thread-safe**: a module-level registry of per-repo locks guards the JSON
  file (mirrors ``FileLockManager``'s per-path mutual exclusion rationale).
* **Recall is pull-based**: callers ask ``recall_for(tool, reason)`` and get ""
  when the pattern isn't yet "known bad" (below threshold) — so recall never
  fires on first occurrence and never spams.

This store is the failure-side analogue of ``insights_manager`` (design
lessons) and ``learned_policy`` (strategy Q-values): a per-repo memory the
agent accumulates automatically.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

_STORE_RELPATH = ".asicode/failure_patterns.json"
_MAX_PATTERNS = 200          # bound on distinct (tool, reason) keys
_DECAY_HALF_LIFE_SEC = 7 * 24 * 3600  # ~1 week → weight halves
_PRUNE_BELOW_COUNT = 2       # drop stale low-count patterns on save
_RECALL_MIN_COUNT = 3        # a pattern must recur this often before recall fires
# Reasons that are too broad to produce meaningful recall.  "generic failure"
# and "transient failure" are fallback classifications from FailureClassifier
# that conflate unrelated errors — including them would fire "a retry is likely
# to fail the same way" for completely different failures that happen to share
# the same coarse reason string.  See failure_classifier.py for the full list
# of possible reason values.
_NON_RECALLABLE_REASONS = frozenset({"generic failure", "transient failure"})
_FLUSH_INTERVAL = 5          # batch disk writes: flush every N records
_READ_REFRESH_TTL: float = 2.0  # seconds between disk re-reads in recall_for
# Per-repo-path locks (same pattern as FileLockManager): the JSON file is shared
# across concurrent sub-agents / sessions in a shared checkout.  LRU eviction
# parallels _stores so both registries stay bounded together.
_MAX_LOCKS = 100
_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_locks_guard = threading.Lock()


def _get_lock(store_path: str) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(store_path)
        if lk is None:
            if len(_locks) >= _MAX_LOCKS:
                _locks.popitem(last=False)
            lk = threading.Lock()
            _locks[store_path] = lk
        else:
            _locks.move_to_end(store_path)
        return lk


def _decay_weight(last_seen_ts: float, now: float) -> float:
    """Exponential decay in [0,1]; ~0.5 after one half-life."""
    import math
    age = max(0.0, now - last_seen_ts)
    return math.pow(0.5, age / _DECAY_HALF_LIFE_SEC)


# ── Recall-hint shaping ──────────────────────────────────────────────────────
# Maps a RecoveryAction enum NAME (a stable identifier string) → concrete advice
# the LLM can act on.  FailureClassifier already computes the action when it
# classifies a failure; recall_on_failure forwards it here so the hint says
# "switch to a different tool" / "read the target first" instead of an identical
# three-way generality for every reason.  Keyed by .name (not .value) so the
# mapping is decoupled from the enum's wire string and we avoid importing the
# enum at module top (it is lazy-imported inside recall_on_failure to prevent an
# import cycle).  Unknown / None actions fall back to the generic advice.
_ACTION_ADVICE_BY_NAME = {
    "SWITCH_TOOL": "switch to a different tool",
    "READ_FIRST": "read the target first (it may have moved or changed)",
    "SKIP": "skip this — the change is likely already applied",
    "ABORT": "stop — this is a permission/environment issue that won't resolve by retrying",
}
_RECALL_ADVICE_GENERIC = (
    "change approach (different tool, read the target first, or re-fetch fresh context)"
)

# Candidate arg keys that carry a tool call's target path.  We accumulate a
# bounded per-path breakdown under the (tool, reason) key so the recall hint can
# name WHERE the failure concentrates ("recurred 4x, mostly on `foo/bar.py`")
# WITHOUT fragmenting the key itself — keeping the key at (tool, reason) means
# the recurrence threshold still triggers even when every failure hits a
# different file.  Path counts are display-only metadata.
_PATH_ARG_KEYS = ("path", "file_path", "filepath", "filename", "target")
_MAX_PATHS = 5  # keep the top-N paths per pattern (bounded memory)


def _advice_for_action(action) -> str:
    """Concrete advice for a RecoveryAction enum value, else the generic fallback."""
    name = getattr(action, "name", None)
    return _ACTION_ADVICE_BY_NAME.get(name or "") or _RECALL_ADVICE_GENERIC


def _extract_target_path(args) -> str:
    """Best-effort extraction of a tool call's target path from its args dict."""
    if not isinstance(args, dict):
        return ""
    for k in _PATH_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _normalize_path(path: str, repo_root: Path) -> str:
    """Shorten an absolute path to repo-relative for display; pass through on failure."""
    if not path:
        return ""
    try:
        p = Path(path)
        if p.is_absolute():
            try:
                return str(p.relative_to(repo_root))
            except ValueError:
                return str(p)
        return str(p)
    except (ValueError, OSError):
        return path


def _merge_paths(a, b) -> dict:
    """Per-path max merge of two ``{path: count}`` dicts (cross-process reconciliation)."""
    out: dict = {}
    for src in (a, b):
        if isinstance(src, dict):
            for pk, pc in src.items():
                if isinstance(pc, (int, float)) and pc:
                    out[pk] = max(out.get(pk, 0), pc)
    return out


def _format_path_breakdown(paths: dict) -> str:
    """Render the per-path breakdown as a compact suffix for the recall hint."""
    if not isinstance(paths, dict) or not paths:
        return ""
    ranked = sorted(paths.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(c for _, c in ranked if isinstance(c, (int, float)))
    top_label, top_count = ranked[0]
    if len(ranked) <= 1 or not total:
        return f" This is concentrated on `{top_label}`."
    share = (top_count / total) if total else 0.0
    return (
        f" Mostly on `{top_label}` "
        f"({round(share * 100)}% of {len(ranked)} affected paths)."
    )


class FailurePatternStore:
    """Persistent, bounded, recency-decayed failure-frequency accumulator.

    .. caution::
       Cross-process deletion limitation: ``clear()`` and ``drop()`` write
       ``merge=False`` to avoid resurrecting just-deleted keys from a stale
       disk snapshot.  However, a *different* process that holds unwritten
       pending records (its ``_dirty_count > 0``) may later ``flush()`` with
       ``merge=True`` (the default), resurrecting the deleted keys.  In
       single-process CLI use this is not an issue; for multi-process
       coordination a generation counter in the JSON file would be needed.
    """

    def __init__(self, repo_root: str | Path, *, enabled: bool = True) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.enabled = enabled and bool(repo_root)
        self._path = self.repo_root / _STORE_RELPATH
        self._lock = _get_lock(str(self._path)) if self.enabled else None
        self._cache: dict[str, dict] | None = None  # lazy load
        self._dirty_count = 0  # pending writes since last flush
        self._last_read_ts: float = 0.0  # last disk-read timestamp (recall_for TTL)
        self._baseline: dict[str, int] = {}  # per-key count at last disk read (for delta merge)

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict]:
        if not self.enabled:
            return {}
        if self._cache is not None:
            return self._cache
        data = self._read_from_disk_unsafe()
        self._baseline = {k: v.get("count", 0) for k, v in (data or {}).items()}
        self._cache = data
        return data

    def _save(self, data: dict[str, dict], *, merge: bool = True) -> None:
        if not self.enabled:
            return
        if merge:
            # Cross-process safety: re-read the file BEFORE writing and merge
            # our in-memory entries on top.  This avoids losing entries written
            # by another process since the last _load() call.  The merge is
            # always done under the same thread lock, but since the file lock
            # is process-local, concurrent processes can still race on the
            # read-modify-write cycle.  Merging on every write mitigates this
            # by ensuring no entries are lost.
            disk_data = self._read_from_disk_unsafe()
            merged = self._merge_max(disk_data, data, baseline=self._baseline)
        else:
            # Explicit deletion (clear/drop): do NOT merge disk data here —
            # otherwise the deleted keys would be resurrected by the merge,
            # silently turning the operation into a no-op.  The caller has
            # already loaded the authoritative in-memory view and pruned it.
            merged = dict(data)
        self._cache = merged
        # Bound growth: prune stale, low-count patterns when at capacity.
        if len(merged) > _MAX_PATTERNS:
            now = time.time()
            keep: dict[str, dict] = {}
            for k, v in merged.items():
                raw = v.get("count", 0)
                w = _decay_weight(v.get("last_seen", now), now)
                eff = raw * w
                if raw >= _PRUNE_BELOW_COUNT or eff > 0.2:
                    keep[k] = v
            # Hard cap: enforce ``len <= _MAX_PATTERNS`` unconditionally.
            # Two triggers:
            #   (a) ``len(keep) > _MAX_PATTERNS`` — soft-prune wasn't enough.
            #   (b) ``not keep`` — every pattern is a stale singleton, so the
            #       soft-prune kept nothing.  Without this branch the fallback
            #       ``merged = keep or merged`` would write the full over-capacity
            #       set, violating the "bounded" contract in the class docstring
            #       (a noisy repo could grow the file past _MAX_PATTERNS once all
            #       its patterns decay below the keep threshold).
            if len(keep) > _MAX_PATTERNS or not keep:
                pool = keep if keep else merged
                scored = sorted(
                    pool.items(),
                    key=lambda kv: kv[1].get("count", 0) * _decay_weight(kv[1].get("last_seen", now), now),
                )
                keep = dict(scored[-_MAX_PATTERNS:])
            merged = keep
        # Update baseline AFTER merge+prune so the next flush computes the
        # correct delta.  Baseline must reflect what we just wrote to disk
        # (merged), not the pre-write disk_data, otherwise the delta will
        # double-count the previous flush's increments.
        self._baseline = {k: v.get("count", 0) for k, v in merged.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"patterns": merged, "version": 1, "updated": time.time()},
            ensure_ascii=False, indent=2, allow_nan=False,
        )
        # Atomic write: temp file in same dir + rename.
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except OSError as e:
            logger.debug("failure_pattern_store: could not persist (%s)", e)

    def _read_from_disk_unsafe(self) -> dict[str, dict]:
        """Fresh read from the JSON file, bypassing the in-memory cache.

        Called under ``self._lock`` so thread-safe, but the file itself is
        not locked across processes — callers must accept eventual consistency.

        Updates ``_last_read_ts`` so ``recall_for()`` TTL avoids redundant re-reads.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("patterns"), dict):
                self._last_read_ts = time.time()
                return obj["patterns"]
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        self._last_read_ts = time.time()
        return {}

    # ── public API ───────────────────────────────────────────────────────────

    @staticmethod
    def _key(tool_name: str, reason: str) -> str:
        return f"{tool_name or '?'}::{reason or '?'}"

    def record(self, tool_name: str, reason: str, *, path: str = "") -> int:
        """Record one observation; return the new (decayed) effective count.

        *path*, when provided, accumulates a bounded per-path breakdown under
        this (tool, reason) key so the recall hint can name *where* the failure
        recurs.  The key stays at (tool, reason) — the path is display-only
        metadata and never fragments the recurrence threshold.
        """
        if not self.enabled or not self._lock:
            return 0
        key = self._key(tool_name, reason)
        with self._lock:
            data = self._load()
            entry = data.get(key, {"count": 0, "first_seen": time.time(), "last_seen": time.time()})
            now = time.time()
            # Apply decay to the stored count before incrementing so the figure
            # reflects recency, not raw history.
            w = _decay_weight(entry.get("last_seen", now), now)
            entry["count"] = round(entry.get("count", 0) * w, 3) + 1.0
            entry["last_seen"] = now
            entry["tool"] = tool_name
            entry["reason"] = reason
            if path:
                norm = _normalize_path(path, self.repo_root)
                if norm:
                    paths = entry.get("paths")
                    if not isinstance(paths, dict):
                        paths = {}
                    paths[norm] = float(paths.get(norm, 0)) + 1.0
                    # Bound: keep the top-N paths by count so a repo with many
                    # distinct files can't grow the entry unbounded.
                    if len(paths) > _MAX_PATHS:
                        paths = dict(
                            sorted(paths.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_PATHS]
                        )
                    entry["paths"] = paths
            data[key] = entry
            self._cache = data
            # Write-behind: batch disk writes so failure-flood runs don't pay
            # synchronous I/O on every single observation.  The in-memory cache
            # is always immediately up-to-date; the disk is at most
            # _FLUSH_INTERVAL-1 records behind.
            self._dirty_count += 1
            if self._dirty_count >= _FLUSH_INTERVAL:
                self._save(data)
                self._dirty_count = 0
            return round(entry["count"])

    def flush(self) -> None:
        """Force-pending writes to disk.

        Safe to call from read-paths that need cross-process visibility, or
        from ``atexit`` / signal handlers for crash safety.
        """
        if not self.enabled or not self._lock:
            return
        with self._lock:
            self._flush_unsafe()

    def _flush_unsafe(self) -> None:
        """Flush pending writes under lock (caller must hold ``self._lock``).

        Extracted so ``drop()`` / ``drop_key()`` can persist pending records
        before discarding the cache (``self._cache = None``) without
        deadlocking on ``self._lock``.
        """
        if self._dirty_count > 0:
            self._save(self._cache if self._cache is not None else {})
            self._dirty_count = 0

    def effective_count(self, tool_name: str, reason: str) -> int:
        if not self.enabled or not self._lock:
            return 0
        with self._lock:
            data = self._load()
            entry = data.get(self._key(tool_name, reason))
            if not entry:
                return 0
            now = time.time()
            return round(self._effective_unsafe(entry, now))

    def recall_for(
        self, tool_name: str, reason: str, *,
        action=None, min_count: int = _RECALL_MIN_COUNT,
    ) -> str:
        """Return a compact recall hint if this failure is recurrent, else "".

        *action* is the ``RecoveryAction`` enum value the classifier computed for
        this failure (forwarded by ``recall_on_failure``).  It shapes the advice
        — e.g. ``SWITCH_TOOL`` → "switch to a different tool" — so the hint gives
        a concrete next step instead of an identical generality for every reason.
        ``None``/unknown falls back to the generic advice.

        Does NOT flush pending writes — the write-behind batching
        (``_FLUSH_INTERVAL``) accumulates records and flushes asynchronously
        so failure-flood runs don't pay synchronous I/O per observation.

        Reads from disk (read-only refresh) so cross-process observations
        are visible, but does NOT write — the next batched ``_save`` or
        ``atexit`` flush will persist our pending records.

        Excludes reasons in ``_NON_RECALLABLE_REASONS`` (``"generic failure"``,
        ``"transient failure"``) because they are too broad — unrelated errors
        share the same coarse reason string, causing false-positive recall.
        """
        if not self.enabled or not self._lock:
            return ""
        if reason in _NON_RECALLABLE_REASONS:
            return ""
        with self._lock:
            # Read-only refresh with TTL: avoid re-reading the JSON file on every
            # recall call during a failure flood.  Within _READ_REFRESH_TTL seconds
            # of the last disk read, use the in-memory cache as-is.
            if time.time() - self._last_read_ts > _READ_REFRESH_TTL:
                disk_data = self._read_from_disk_unsafe()
                if disk_data:
                    cache = self._cache or {}
                    self._cache = self._merge_max(disk_data, cache, baseline=self._baseline)
                    # Baseline MUST reflect what's on disk (disk_data), not the merged
                    # result.  Otherwise the next flush computes delta = mem_c - merged_c
                    # = 0 for pending observations, silently losing them.  See
                    # test_count_preserved_across_recall_flush_boundary.
                    self._baseline = {k: v.get("count", 0) for k, v in disk_data.items()}
            data = self._cache or {}
            entry = data.get(self._key(tool_name, reason))
            if not entry:
                return ""
            now = time.time()
            n = round(self._effective_unsafe(entry, now))
            paths = entry.get("paths") if isinstance(entry.get("paths"), dict) else {}
        if n < min_count:
            return ""
        advice = _advice_for_action(action)
        hint = (
            f"[RECALL] `{tool_name}` failing with \"{reason}\" has recurred "
            f"{n}× in this repo. A retry is likely to fail the same way — {advice}."
        )
        hint += _format_path_breakdown(paths)
        return hint

    def store_size(self) -> int:
        """Return the number of distinct patterns currently stored."""
        if not self.enabled or not self._lock:
            return 0
        with self._lock:
            return len(self._load())

    def clear(self) -> None:
        """Remove all patterns from the store."""
        if not self.enabled or not self._lock:
            return
        with self._lock:
            self._save({}, merge=False)
            self._cache = {}
            self._dirty_count = 0  # no pending writes after clearing

    def prune(self, threshold: float = 1.0) -> int:
        """Remove all patterns whose effective score is below *threshold*.

        Returns the number of patterns removed.  Safe to call from CLI — uses
        ``merge=False`` (explicit deletion) so cross-process observations are
        not resurrected.  Uses ``_ranked_keys()`` sorting (same as
        ``top_patterns()``) so the user sees consistent ordering.

        A threshold of ``0.0`` is a no-op (only zero-effective patterns match).
        """
        if not self.enabled or not self._lock:
            return 0
        with self._lock:
            self._flush_unsafe()  # persist pending writes before cache discard
            self._cache = None  # force fresh disk read
            data = self._load()
            if not data:
                return 0
            now = time.time()
            to_drop: list[str] = []
            for k, entry in data.items():
                eff = self._effective_unsafe(entry, now)
                if eff < threshold:
                    to_drop.append(k)
            if not to_drop:
                return 0
            for k in to_drop:
                data.pop(k, None)
            self._cache = data
            self._save(data, merge=False)
            self._dirty_count = 0
            return len(to_drop)

    def drop(self, idx: int) -> tuple[str, str] | None:
        """Drop the pattern at 1-based index from ``top_patterns()``.

        The index follows the ``top_patterns()`` display order (effective score,
        descending) — NOT raw dict insertion order — so ``drop(N)`` removes the
        N-th pattern the user sees listed. Returns ``(tool, reason)`` of the
        removed pattern on success, or ``None`` if the index is out of range or
        the store is disabled.
        """
        if not self.enabled or not self._lock:
            return None
        with self._lock:
            self._flush_unsafe()  # persist pending writes before cache discard
            self._cache = None  # force fresh disk read (minimize cross-process entry loss)
            data = self._load()
            ranked = self._ranked_keys(data)
            if not (1 <= idx <= len(ranked)):
                return None
            key = ranked[idx - 1][0]
            return self._drop_key_unsafe(key)  # saved, no pending writes

    def drop_key(self, key: str) -> tuple[str, str] | None:
        """Drop a pattern by its exact key (e.g. ``"apply_patch::context mismatch"``).

        Returns ``(tool, reason)`` of the removed pattern on success, or ``None``
        if the key doesn't exist or the store is disabled.
        """
        if not self.enabled or not self._lock:
            return None
        with self._lock:
            self._flush_unsafe()  # persist pending writes before cache discard
            self._cache = None  # force fresh disk read
            return self._drop_key_unsafe(key)

    def _drop_key_unsafe(self, key: str) -> tuple[str, str] | None:
        """Drop by key under lock (caller must hold ``self._lock``)."""
        data = self._load()
        entry = data.get(key)
        if entry is None:
            return None
        tool = entry.get("tool", "")
        reason = entry.get("reason", "")
        data.pop(key, None)
        self._cache = data
        self._save(data, merge=False)
        self._dirty_count = 0
        return (tool, reason)

    def _ranked_keys(self, data: dict[str, dict]) -> list[tuple[str, float]]:
        """Return pattern keys ordered by effective score (descending).

        Returns ``list[(key, effective_count)]`` so callers (``drop()``,
        ``top_patterns()``) can avoid recomputing the effective score.

        Single source of truth for the ``top_patterns()`` display order so that
        ``drop(idx)`` removes exactly the N-th pattern the user sees listed,
        regardless of dict insertion order. Ties keep insertion order (stable
        sort). Must be called under ``self._lock``.
        """
        now = time.time()
        scored = [(k, self._effective_unsafe(data[k], now)) for k in data]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    @staticmethod
    def _effective_unsafe(entry: dict, now: float) -> float:
        """Compute the decayed effective count for an entry.

        Must only be called under ``self._lock`` (or with unchanging data).
        Extracted so ``effective_count()``, ``recall_for()``, and
        ``top_patterns()`` share the same formula.
        """
        return entry.get("count", 0) * _decay_weight(entry.get("last_seen", now), now)

    @staticmethod
    def _merge_max(disk_data: dict[str, dict], memory_data: dict[str, dict],
                   baseline: dict[str, int] | None = None) -> dict[str, dict]:
        """Merge two dicts, taking max(count), max(last_seen), min(first_seen) per key.

        The default ``{**disk_data, **memory_data}`` merge lets a stale in-memory
        count overwrite a higher count written by another process.  This helper
        resolves the conflict by keeping the higher observation count and more
        recent timestamp per key, while preferring the in-memory tool/reason
        strings (same key → same tool/reason, but ours reflects the live session).

        When *baseline* is provided, keys present in both ``disk_data`` and
        ``memory_data`` use ``disk_count + delta`` (where ``delta`` is the
        in-memory observations since the baseline was captured) instead of
        ``max(disk_count, memory_count)``.  This preserves per-process observation
        increments that ``max`` would lose when two processes load from the same
        baseline and both observe the same key (e.g. each adds +3, but max only
        captures the larger of the two totals).
        """
        merged = dict(disk_data)
        for k, v in memory_data.items():
            if k in merged:
                existing = merged[k]
                if baseline is not None and k in baseline:
                    # Delta approach: disk_count + (memory_count - baseline_count).
                    # This preserves observations from this process even when
                    # another process wrote a higher count to disk after our load.
                    disk_c = existing.get("count", 0)
                    mem_c = v.get("count", 0)
                    bl_c = baseline[k]
                    if mem_c < bl_c:
                        # Decay was applied in record() since baseline was
                        # captured.  We keep our decayed count (mem_c) AND
                        # add any increments written by other processes since
                        # baseline (disk_c - bl_c), so cross-process progress
                        # is never lost.  In a single-process scenario where
                        # disk_c == bl_c this reduces to just mem_c.
                        merged_count = mem_c + max(0, disk_c - bl_c)
                    else:
                        delta = max(0, mem_c - bl_c)
                        merged_count = disk_c + delta
                elif baseline is not None:
                    # Key NOT in baseline: this process hadn't seen it at last disk read,
                    # so all its current observations are new.  Summing (disk_c + mem_c)
                    # preserves observations from both this process and other processes
                    # that wrote to disk since the baseline was captured.  Using max()
                    # would silently lose one side when two processes both observed
                    # the same new key between flushes.
                    merged_count = existing.get("count", 0) + v.get("count", 0)
                else:
                    # No baseline at all (legacy / non-delta call site) — fall back to max.
                    merged_count = max(existing.get("count", 0), v.get("count", 0))
                # Finite fallback for first_seen: avoid float("inf") in output JSON.
                _fs_e = existing.get("first_seen")
                _fs_v = v.get("first_seen")
                _fs_candidates = [x for x in (_fs_e, _fs_v) if isinstance(x, (int, float))]
                _merged_first_seen = min(_fs_candidates) if _fs_candidates else v.get("last_seen", time.time())
                merged_entry = {
                    "count": merged_count,
                    "last_seen": max(existing.get("last_seen", 0), v.get("last_seen", 0)),
                    "first_seen": _merged_first_seen,
                    "tool": v.get("tool", existing.get("tool", "")),
                    "reason": v.get("reason", existing.get("reason", "")),
                }
                # Reconcile the per-path breakdown across processes (display-only
                # metadata; per-path max avoids double-counting on merge).
                merged_paths = _merge_paths(existing.get("paths"), v.get("paths"))
                if merged_paths:
                    merged_entry["paths"] = merged_paths
                merged[k] = merged_entry
            else:
                merged[k] = v
        return merged

    def top_patterns(self, limit: int = 10) -> list[dict]:
        """Inspect the highest-frequency patterns (debugging/metrics)."""
        if not self.enabled or not self._lock:
            return []
        with self._lock:
            data = self._load()
            ranked = self._ranked_keys(data)
            scored = []
            for key, eff in ranked:
                entry = data[key]
                scored.append({
                    "key": key,
                    "tool": entry.get("tool", ""),
                    "reason": entry.get("reason", ""),
                    "count": entry.get("count", 0),
                    "effective": round(eff, 2),
                    "last_seen": entry.get("last_seen", 0.0),
                })
            return scored[:limit]


# Module-level registry so callers can share one store per repo_root (avoids
# re-reading the JSON file on every failure in a long session).  LRU eviction
# keeps multi-repo long-running processes bounded.
_MAX_STORES = 100
_stores: "OrderedDict[str, FailurePatternStore]" = OrderedDict()
_stores_guard = threading.Lock()
_atexit_registered = False


def _flush_all_stores() -> None:
    """Flush all live stores on process exit — avoids losing the last batch."""
    for store in list(_stores.values()):
        try:
            store.flush()
        except Exception:
            pass


def get_store(repo_root: str | Path, *, enabled: bool = True) -> FailurePatternStore:
    """Return a process-wide FailurePatternStore for ``repo_root``.

    .. caution::

       The ``enabled`` parameter only takes effect when a store is first created
       for this ``repo_root``.  If a store already exists in the process-wide
       cache (e.g., created earlier with ``enabled=True``), subsequent calls with
       ``enabled=False`` still return the cached (enabled) store.  To truly
       disable an already-cached store, call ``clear()`` on it, or avoid
       creating the store until the ``enabled`` state is final.
    """
    root = str(Path(repo_root).resolve()) if repo_root else ""
    global _atexit_registered
    with _stores_guard:
        if not _atexit_registered:
            _atexit_registered = True
            atexit.register(_flush_all_stores)
        s = _stores.get(root)
        if s is None:
            if len(_stores) >= _MAX_STORES:
                _evict_key, _evicted = _stores.popitem(last=False)  # evict oldest
                try:
                    _evicted.flush()  # flush pending writes before discarding
                except Exception:
                    pass
            s = FailurePatternStore(root, enabled=enabled)
            _stores[root] = s
        else:
            _stores.move_to_end(root)  # mark as recently used
            # Keep the lock's LRU position in sync with the store's so that
            # ``_locks`` does not evict a lock while its store is still alive.
            # The lock is created in ``FailurePatternStore.__init__`` via
            # ``_get_lock(s._path)``; touching it here prevents eviction.
            with _locks_guard:
                if str(s._path) in _locks:
                    _locks.move_to_end(str(s._path))
        return s


# ── Unified recall hook (shared by both agent loops) ──────────────────────
#
# Both the webapp agent turn pipeline (agent_turn_pipeline.py) and the CLI
# design-chat loop (design_chat_loop.py) call this single function on every
# tool failure.  Centralising classify → dedup → record → recall here means
# the two surfaces can never drift: the repo-local store (shared between
# them) is always fed the *same* ``FailureClassifier`` reason vocabulary, so
# a recurring failure accumulates toward the recall threshold regardless of
# which surface observed it.  See failure_classifier.py for the reason set.
#
# In-run dedup: a (session_key, repo, tool, reason) is recorded at most once
# per *run* (webapp agent turn) / per *turn* (CLI design-chat respond) so
# consecutive retries within that scope do not inflate the cross-run count —
# replacing the caller-side ``fail_streak == 0`` gate.  *session_key* is a
# fresh per-run object id supplied by each caller (webapp: id(ctx.fail_streak);
# CLI: id(DesignChatResult)), so a NEW run both re-records AND re-surfaces the
# recall hint — matching the original gate which reset every run.  A bare
# process-global set would never reset under a long-lived server process, so
# recall would fire at most once per process and then be silenced forever
# (even for new conversations that need it most); the bounded LRU below keeps
# memory in check while preserving per-run scoping.
_CLASSIFIER: "object | None" = None  # FailureClassifier, lazily imported
_RECORDED_SESSION_KEYS: "OrderedDict[tuple[str, str, str, str], None]" = OrderedDict()
_RECORDED_SESSION_KEYS_LIMIT = 4096


def reset_recall_session() -> None:
    """Clear the in-run dedup map.

    Test-only: each run supplies its own ``session_key`` (a fresh object id),
    so production code never needs to reset — a new run simply uses a new key.
    """
    _RECORDED_SESSION_KEYS.clear()


# ── Recall efficacy counters (process-lifetime, rg-fallback style) ───────────
#
# Mirrors webapp/ui/ui_tools.py's _rg_fallback_counts: lightweight in-memory
# counters surfaced at /stats/recall so the recall mechanism — unlike before —
# produces observable evidence of how often it fires AND whether that is
# followed by recovery.  This is the efficacy measurement that was missing:
# previously a [RECALL] hint could fire indefinitely with no record of whether
# it actually changed behaviour.
#
#   fired   — a [RECALL] hint was returned to a caller
#   helped  — the NEXT tool result in the same run succeeded (the LLM changed
#             approach after the hint and recovered)
#   ignored — the next tool result in the same run failed again
#
# The link between a fired hint and its outcome is a per-run marker keyed by
# session_key (the same key the caller passes to recall_on_failure).  The caller
# settles the marker via record_recall_outcome() on the next tool result, BEFORE
# recall_on_failure may set a new marker for the current failure, so the two
# never tangle.  Markers are bounded LRU like the dedup map.  Counters are
# process-lifetime (not persisted) — same scope as rg-fallback counters.
_recall_lock = threading.Lock()
_recall_counts: dict[str, int] = {"fired": 0, "helped": 0, "ignored": 0}
_pending_recall: "OrderedDict[str, None]" = OrderedDict()
_PENDING_RECALL_LIMIT = 4096


def _record_recall_fired(session_key: str) -> None:
    """Increment fired and arm the per-run marker for outcome settling."""
    try:
        with _recall_lock:
            _recall_counts["fired"] += 1
            if session_key:
                _pending_recall[session_key] = None
                while len(_pending_recall) > _PENDING_RECALL_LIMIT:
                    _pending_recall.popitem(last=False)
    except Exception:  # counters must never break recall
        pass


def record_recall_outcome(*, ok: bool, session_key: str = "") -> None:
    """Settle the pending recall marker for *session_key*.

    Call this on EVERY tool result in a run (success or failure), BEFORE
    recall_on_failure may set a new marker for the current failure.  If a prior
    recall hint fired in the same run, this next result settles it: success →
    ``helped``, failure → ``ignored``.  No-op when no marker is pending (the
    common case — most runs never fire a recall).
    """
    try:
        if not session_key:
            return
        with _recall_lock:
            # Use membership + del (not pop's return): the marker is stored with
            # a None value, and OrderedDict.pop(key, None) returns None both when
            # the key is absent AND when it is present-with-None-value — which
            # would silently treat every fired recall as "no marker pending".
            if session_key not in _pending_recall:
                return
            del _pending_recall[session_key]
            _recall_counts["helped" if ok else "ignored"] += 1
    except Exception:
        pass


def get_recall_counts() -> dict[str, int]:
    """Snapshot of the process-lifetime recall efficacy counters (for /stats)."""
    with _recall_lock:
        return dict(_recall_counts)


def reset_recall_counts() -> dict[str, int]:
    """Zero the recall efficacy counters and return the pre-reset snapshot."""
    with _recall_lock:
        snap = dict(_recall_counts)
        for k in _recall_counts:
            _recall_counts[k] = 0
        _pending_recall.clear()
        return snap


def recall_on_failure(
    tool_name: str,
    args: dict,
    result,
    repo_root: str | Path | None,
    *,
    exc: BaseException | None = None,
    session_key: str = "",
) -> str:
    """Classify a tool failure, record it once per run, return a recall hint.

    Returns ``""`` when no recall nudge applies (first-ever occurrence, below
    threshold, non-recallable coarse reason, or no repo root).  Otherwise
    returns the ``[RECALL] ...`` hint string for the caller to surface to the
    LLM (as a message, or as a suffix on the tool result).

    Never raises — recall is advisory and must not break the tool loop.

    *result* is the ``ToolResult`` (may be ``None`` when ``dispatch`` itself
    raised); pass the raised exception via *exc* in that case so the
    classifier's type/code hierarchy still applies to dispatch errors.

    *session_key* scopes dedup to a single run (webapp) / turn (CLI): pass a
    fresh per-run object id (e.g. ``str(id(ctx.fail_streak))`` /
    ``str(id(result))``).  Retries within the same run are deduped (no
    double-counting), but a NEW run re-records and re-surfaces the hint once
    the threshold is met — matching the original per-run ``fail_streak == 0``
    gate.  Defaults to ``""`` (callers that omit it share one process-wide
    scope, appropriate only for one-shot use).
    """
    try:
        if not repo_root:
            return ""
        global _CLASSIFIER
        if _CLASSIFIER is None:
            from .failure_classifier import FailureClassifier
            _CLASSIFIER = FailureClassifier()
        # Adapt a raised exception into a result-shaped namespace so the
        # classifier's exception-type / errno hierarchy covers dispatch
        # errors too (a bare exception has no ``.error`` attribute).
        if result is not None:
            src = result
        elif exc is not None:
            import types as _types
            src = _types.SimpleNamespace(error=exc, ok=False)
        else:
            return ""
        classification = _CLASSIFIER.classify(tool_name, args, src)
        reason = getattr(classification, "reason", "") or ""
        action = getattr(classification, "action", None)
        if reason in _NON_RECALLABLE_REASONS:
            return ""
        repo_key = str(Path(repo_root).resolve())
        store = get_store(repo_root)
        dedup_key = (session_key, repo_key, tool_name, reason)
        # Record only the first observation of this (session, repo, tool,
        # reason) in the current run; surface the recall hint at that same
        # moment.  Subsequent retries within the SAME run are left to the
        # run-local strategy-warning machinery, not re-spammed here.  A NEW
        # run passes a new session_key, so it re-records (incrementing the
        # cross-run count toward threshold) and re-hints once threshold is met.
        if dedup_key not in _RECORDED_SESSION_KEYS:
            _RECORDED_SESSION_KEYS[dedup_key] = None
            # Bounded LRU: evict oldest so a long-lived process cannot grow
            # the dedup map without bound.
            while len(_RECORDED_SESSION_KEYS) > _RECORDED_SESSION_KEYS_LIMIT:
                _RECORDED_SESSION_KEYS.popitem(last=False)
            # Forward the classifier's action (shapes the advice) and the
            # call's target path (shapes the WHERE breakdown) so the recall
            # hint carries the concrete next step and location instead of a
            # generic one-size-fits-all nudge.
            store.record(tool_name, reason, path=_extract_target_path(args))
            hint = store.recall_for(tool_name, reason, action=action)
            if hint:
                _record_recall_fired(session_key)
            return hint
        return ""
    except Exception:  # recall must never break the caller
        return ""
