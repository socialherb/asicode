"""
Design-insights file management (parse / serialize / stats / nudge thresholds).

``.asicode/design_insights.md`` accumulates append-only entries written by
``design_chat_loop._save_insight_to_file`` (``### [category] timestamp`` blocks)
and is occasionally hand-edited into a prose preamble (a ``#``/``>`` header +
blockquote "원칙" block). Both shapes coexist in the wild, so the parser is
line-based and **round-trip lossless**: ``serialize(parse(x)) == x`` for any
file content. This is the one invariant the /insights commands and the LLM
compactor rely on — nothing is silently dropped or rewritten.

This module is deliberately pure (no LLM client, no I/O side effects beyond the
one explicit ``atomic_write_text``): the caller (asi) owns the LLM call and
the user-confirmation flow.
"""
from __future__ import annotations

import calendar
import datetime
import logging
import math
import os
import tempfile
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from typing import Any  # f821-protected

from ..common.file_lock import cross_process_flock

# Module-level lazy tokenizer singleton — identical pattern to
# ``rag_searcher._TOKENIZER``. Avoids re-constructing the CodeTokenizer
# (which compiles internal regex-alternative scanners) every turn even
# on analyzed-cache hits, and is shared across archive parsing and
# promotion scoring.
_TOKENIZER: Any = None  # CodeTokenizer — lazy init

__all__ = [
    "COMPACT_BUDGET_BYTES",
    "NUDGE_AGE_DAYS_THRESHOLD",
    "NUDGE_BYTES_THRESHOLD",
    "NUDGE_COUNT_THRESHOLD",
    "InsightEntry",
    "InsightsStats",
    "atomic_write_text",
    "build_compact_messages",
    "build_verify_messages",
    "compute_stats",
    "drop_entry",
    "entry_age_days",
    "insights_path",
    "insights_write_lock",
    "load_insights_file",
    "parse_insights",
    "parse_timestamp",
    "select_entries_older_than",
    "serialize_insights",
    "should_nudge",
    # Two-tier archive (hard budget without losing durable insights)
    "ARCHIVE_INDEX_MAX_ENTRIES",
    "PROMOTE_MAX_BYTES",
    "PROMOTE_MAX_ENTRIES",
    "PROMOTE_MIN_SCORE",
    "append_entries_to_archive",
    "build_archive_index",
    "enforce_budget_by_demotion",
    "insights_archive_path",
    "load_archive_file",
    "select_demotion_candidates",
    "select_promotable_entries",
]

_logger = logging.getLogger(__name__)

# ── Nudge thresholds (Tier 2 / Tier 3) ────────────────────────────────────────
# Tuned to stay silent during normal use and only surface when the file has
# genuinely accumulated. Count/bytes are independent triggers; age captures the
# "long-stale, never reviewed" case. These align with load_design_insights'
# no-silent-truncation philosophy: we nudge + ask, never auto-drop.
NUDGE_COUNT_THRESHOLD = 15
NUDGE_BYTES_THRESHOLD = 6000
NUDGE_AGE_DAYS_THRESHOLD = 21

# ── Compact budget (single source of truth, shared with the nudge) ────────────
# The size that triggers the accumulation nudge (NUDGE_BYTES_THRESHOLD) is ALSO
# the target budget /insights compact enforces. This couples the warning to its
# remedy: when the nudge fires "over budget", compact is instructed to MERGE +
# tighten down to this same size. One threshold, two consumers → no drift.
# Without this coupling compact is pure value-triage and no-ops on an
# all-durable file (the normal steady state), so the file grows unbounded while
# every turn re-injects the full thing into context (load_design_insights).
COMPACT_BUDGET_BYTES = NUDGE_BYTES_THRESHOLD


@dataclass
class InsightEntry:
    """One ``### ...`` block of the insights file.

    ``lines`` keeps the *original* lines (with ``\\n``) so that serialization is
    byte-exact — the parser never rewrites whitespace, quote markers, or
    trailing newlines. ``header_line`` / ``category`` are derived views for
    display and filtering only.
    """

    lines: list[str] = field(default_factory=list)
    header_line: str = ""
    category: str = ""

    @property
    def body(self) -> str:
        """Body text without the header line (for display / compactor)."""
        return "".join(self.lines[1:]) if len(self.lines) > 1 else ""

    @property
    def text(self) -> str:
        """Full block text including the header line."""
        return "".join(self.lines)


@dataclass
class InsightsStats:
    """Snapshot of the insights file used by the nudge logic."""

    exists: bool = False
    count: int = 0
    bytes_size: int = 0
    tokens: int = 0
    mtime: Optional[float] = None
    age_days: Optional[float] = None  # file mtime age (whole-file)
    oldest_age_days: Optional[float] = None  # age of the OLDEST entry (per-entry)


def insights_path(repo_root: str) -> str:
    """Return the canonical insights file path under ``repo_root``."""
    return os.path.join(repo_root, ".asicode", "design_insights.md")


def _load_file_safe(path: str) -> str:
    """Read a UTF-8 file, returning ``""`` if missing/unreadable.

    Shared by :func:`load_insights_file` and :func:`load_archive_file` — the
    two readers differ only in which path they resolve.
    """
    if not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        _logger.debug("failed to read %s", path, exc_info=True)
        return ""


def load_insights_file(repo_root: str) -> str:
    """Read the raw insights file. Returns ``""`` if missing/unreadable."""
    return _load_file_safe(insights_path(repo_root))


def _parse_category(header_line: str) -> str:
    """Extract a ``[category]`` token from a ``### [cat] ts`` header.

    Returns ``""`` when there is no bracketed tag (e.g. a bare ``### `` or a
    hand-written prose header). Pure string ops — no regex.
    """
    s = header_line.strip()
    if s.startswith("###"):
        s = s[3:].lstrip()
    if s.startswith("["):
        end = s.find("]")
        if end > 0:
            return s[1:end].strip()
    return ""


def parse_timestamp(header_line: str) -> Optional[float]:
    """Extract epoch-seconds from a ``### [cat] YYYY-MM-DD HH:MM [±HHMM]`` header.

    Two header generations coexist in insights files:

    * new (``design_chat_loop._save_insight_to_file`` since 2026-07): local time
      with an explicit UTC offset (``%z``, e.g. ``2026-07-10 14:32 +0900``) —
      parsed offset-aware, so the epoch is exact.
    * legacy: ``time.strftime("%Y-%m-%d %H:%M", time.gmtime())`` (UTC, no
      offset) — parsed back as UTC (``calendar.timegm``), not local time.

    Returns ``None`` for headers with no parseable timestamp (bare ``### `` or
    hand-written prose headers), which the age policy treats as "never
    age-prune" (conservative). Pure — no clock access.
    """
    s = header_line.strip()
    if s.startswith("###"):
        s = s[3:].lstrip()
    if s.startswith("["):  # strip optional [category]
        end = s.find("]")
        if end > 0:
            s = s[end + 1:].strip()
    # New format first: 'YYYY-MM-DD HH:MM ±HHMM' is exactly 22 chars. Without
    # this branch the offset suffix would be silently dropped and the LOCAL
    # wall-clock misread as UTC — skewing every age computation by the zone
    # offset (9h for KST).
    try:
        return datetime.datetime.strptime(s[:22], "%Y-%m-%d %H:%M %z").timestamp()
    except (ValueError, TypeError):
        pass
    candidate = s[:16]  # legacy 'YYYY-MM-DD HH:MM' is exactly 16 chars
    try:
        return calendar.timegm(time.strptime(candidate, "%Y-%m-%d %H:%M"))
    except (ValueError, TypeError):
        return None


def entry_age_days(entry: "InsightEntry", now: Optional[float] = None) -> Optional[float]:
    """Age of ``entry`` in days from its header timestamp, or ``None`` if the
    header has no parseable timestamp. ``now`` defaults to the current time;
    pass it explicitly for deterministic tests."""
    ts = parse_timestamp(entry.header_line)
    if ts is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, (now - ts) / 86400.0)


def select_entries_older_than(
    entries: list[InsightEntry], days: float, now: Optional[float] = None
) -> list[int]:
    """Return the 1-based indices of entries strictly older than ``days``.

    Entries whose header has no parseable timestamp are NEVER selected — a
    hand-written or preamble-derived entry must not be age-pruned. The 1-based
    indexing matches :func:`drop_entry` and the ``/insights`` list display.
    """
    now = time.time() if now is None else now
    out: list[int] = []
    for i, entry in enumerate(entries, 1):
        age = entry_age_days(entry, now=now)
        if age is not None and age > days:
            out.append(i)
    return out


def parse_insights(content: str) -> tuple[list[str], list[InsightEntry]]:
    """Split insights file content into ``(preamble_lines, entries)``.

    The **preamble** is everything before the first ``### `` line — typically the
    ``#`` title and the ``>`` "원칙" blockquote, but it gracefully absorbs any
    hand-edited prose. Each subsequent ``### `` line starts a new entry that
    extends to (but excludes) the next ``### `` line.

    Round-trip invariant: ``serialize_insights(*parse_insights(x)) == x`` holds
    for every ``x`` because every original line is preserved verbatim.
    """
    lines = content.splitlines(keepends=True)

    preamble: list[str] = []
    i = 0
    n = len(lines)
    while i < n and not lines[i].lstrip().startswith("### "):
        preamble.append(lines[i])
        i += 1

    entries: list[InsightEntry] = []
    while i < n:
        header_line = lines[i]
        block: list[str] = [header_line]
        i += 1
        while i < n and not lines[i].lstrip().startswith("### "):
            block.append(lines[i])
            i += 1
        entries.append(
            InsightEntry(
                lines=block,
                header_line=header_line.rstrip("\n"),
                category=_parse_category(header_line),
            )
        )
    return preamble, entries


def serialize_insights(preamble: list[str], entries: list[InsightEntry]) -> str:
    """Inverse of :func:`parse_insights` — rebuilds the exact file text.

    Entries are flattened line-by-line (each ``entry.lines`` is itself a
    ``List[str]``), so the result is the preamble followed by every entry's
    original lines in order.
    """
    parts: list[str] = list(preamble)
    for entry in entries:
        parts.extend(entry.lines)
    return "".join(parts)


def drop_entry(entries: list[InsightEntry], index_1based: int) -> list[InsightEntry]:
    """Return a copy of ``entries`` with the 1-based ``index`` removed.

    Out-of-range indices are a no-op (returns the list unchanged) so the CLI can
    report "no such entry" without juggling partial mutations.
    """
    if 1 <= index_1based <= len(entries):
        return entries[: index_1based - 1] + entries[index_1based:]
    return list(entries)


def compute_stats(repo_root: str) -> InsightsStats:
    """Gather count / size / age of the insights file for nudge decisions."""
    path = insights_path(repo_root)
    if not os.path.exists(path):
        return InsightsStats(exists=False)
    content = load_insights_file(repo_root)
    _, entries = parse_insights(content)
    try:
        st = os.stat(path)
    except OSError:
        return InsightsStats(exists=True, count=len(entries))
    now = time.time()
    age_days: Optional[float] = None
    if st.st_mtime > 0:
        age_days = (now - st.st_mtime) / 86400.0
    # Per-entry age: the oldest entry is a far better staleness signal than the
    # file mtime, which any compact/edit/drop resets — so the mtime-based nudge
    # almost never fires in practice. Entries without a timestamp are ignored.
    _ages = [a for a in (entry_age_days(e, now=now) for e in entries) if a is not None]
    oldest_age_days = max(_ages) if _ages else None
    return InsightsStats(
        exists=True,
        count=len(entries),
        bytes_size=st.st_size,
        tokens=max(1, int(len(content) / 3.5)),
        mtime=st.st_mtime,
        age_days=age_days,
        oldest_age_days=oldest_age_days,
    )


def should_nudge(stats: InsightsStats) -> tuple[bool, str]:
    """Decide whether to show the accumulation nudge (Tier 2 / Tier 3).

    Returns ``(True, message)`` when any threshold is crossed, else
    ``(False, "")``. The message lists every crossed threshold so the user sees
    *why* the nudge fired. Stays silent below all thresholds — normal use is
    noise-free.
    """
    if not stats.exists:
        return False, ""
    reasons: list[str] = []
    if stats.count >= NUDGE_COUNT_THRESHOLD:
        reasons.append(f"{stats.count} items")
    if stats.bytes_size >= NUDGE_BYTES_THRESHOLD:
        reasons.append(f"{stats.bytes_size:,} bytes (~{stats.tokens:,} tokens)")
    # Prefer per-entry age (oldest entry); fall back to file mtime when no
    # entry carries a parseable timestamp.
    _age = stats.oldest_age_days if stats.oldest_age_days is not None else stats.age_days
    if _age is not None and _age >= NUDGE_AGE_DAYS_THRESHOLD:
        reasons.append(f"oldest entry {int(_age)} days old")
    if not reasons:
        return False, ""
    return (
        True,
        "💡 design_insights: "
        + ", ".join(reasons)
        + " /insights {list|verify|compact|drop <n>}",
    )


# ── Cross-process / re-entrant lock for insights RMW ──────────────────────────
# The active file + archive form ONE resource group: enforce_budget_by_demotion
# rewrites the active file AND appends the archive in a single RMW cycle. Without
# serialization, a concurrent writer (a parallel design-chat session or a second
# process) that appends between a compactor's read and rewrite is silently
# dropped — violating the documented "0 durable loss" contract. The lock is
# re-entrant (threading.RLock) so enforce_budget_by_demotion →
# append_entries_to_archive nesting does not self-deadlock, and cross-process
# (fcntl/msvcrt via file_lock.py) so parallel sessions serialize. Best-effort:
# degrades to no-op if no lock backend is available.
# WeakValueDictionary so that a per-repo RLock is GC'd once no caller holds a
# strong reference to it (i.e., no thread is currently inside
# ``insights_write_lock`` for that repo). A plain dict would grow unboundedly in
# a long-lived server process visiting many repos — the same leak class fixed in
# ``orchestrator._file_locks`` (see its module-level comment). The get/create in
# ``insights_write_lock`` returns a strong local ref (``lock =``), so while the
# lock is in use the weak entry stays alive and is shared across threads; when
# idle it is GC'd. Re-entrant calls are safe: the outer ``with lock:`` frame
# holds a strong ref for the duration of the nested call.
_INSIGHTS_THREAD_LOCKS: "weakref.WeakValueDictionary[str, threading.RLock]" = weakref.WeakValueDictionary()
_INSIGHTS_LOCKS_GUARD = threading.Lock()
# Per-thread "which repo flocks does THIS thread already hold" set. fcntl.flock
# is per-open-file-description: a re-entrant call that opens a SECOND fd on the
# same lock file while the first is still held deadlocks the process against
# itself. RLock re-enters fine, but we must skip the flock on re-entry.
_flock_state = threading.local()


@contextmanager
def insights_write_lock(repo_root: str) -> Iterator[None]:
    """Re-entrant cross-process+thread lock guarding ALL insights mutations.

    Acquire around every read-modify-write of the active file or archive
    (save / delete / edit / compact / demote / append-to-archive). The active
    and archive files share ONE lock because demotion touches both atomically;
    per-file locks would still allow cross-file races.

    Re-entrant within the same thread (RLock), so a caller that invokes another
    locking helper internally (``enforce_budget_by_demotion`` →
    ``append_entries_to_archive``) does not self-deadlock. On the FIRST entry by
    a thread the cross-process flock is acquired; on a nested re-entry by the
    SAME thread the flock is skipped (POSIX ``flock`` is per-open-file-description,
    so a second fd on the same lock file would deadlock the process against
    itself). Across threads and processes the flock still serializes.
    """
    key = os.path.abspath(repo_root)
    held = _flock_state.__dict__.setdefault("_held_repos", set())
    with _INSIGHTS_LOCKS_GUARD:
        lock = _INSIGHTS_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _INSIGHTS_THREAD_LOCKS[key] = lock
    with lock:
        if key in held:
            # Same thread already holds the flock for this repo — re-enter
            # without re-acquiring it (a second fd would self-deadlock).
            yield
        else:
            held.add(key)
            try:
                lock_path = Path(insights_path(repo_root) + ".lock")
                with cross_process_flock(lock_path):
                    yield
            finally:
                held.discard(key)


def atomic_write_text(path: str, content: str) -> None:
    """Atomically replace ``path`` with ``content`` (UTF-8, whole-file).

    Mirrors :func:`external_llm.common.atomic_io.atomic_write_json`: sibling
    temp file → ``os.replace`` into place, so an interrupt (SIGKILL, disk full)
    never leaves the insights file truncated/partial. Temp file is cleaned up on
    any failure.
    """
    base_dir = os.path.dirname(path) or "."
    os.makedirs(base_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix=".atomic_", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())  # durability: ensure data is on disk before rename
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Archive tier (two-tier hard budget) ────────────────────────────────────────
# design_insights.md is the ALWAYS-injected "active" tier with a hard byte budget
# (COMPACT_BUDGET_BYTES). When the LLM compactor cannot tighten it under budget
# (the steady state where every entry is a durable invariant — nothing to DROP),
# entries are DEMOTED to design_insights_archive.md instead of deleted. Demotion
# is NOT deletion: archived entries cost 0 context tokens, stay retrievable
# (``/insights archive restore <n>``), and may be auto-promoted back into a turn's
# context by relevance (:func:`select_promotable_entries`). This is what turns the
# soft LLM budget into a HARD ceiling without ever losing a durable insight.
ARCHIVE_INDEX_MAX_ENTRIES = 15
PROMOTE_MAX_ENTRIES = 3
PROMOTE_MAX_BYTES = 2000
PROMOTE_MIN_SCORE = 0.5  # BM25-IDF score threshold (adaptive to corpus size)

_ARCHIVE_PREAMBLE = (
    "# Archived Design Insights\n\n"
    "Durable insights demoted out of the always-injected active tier to keep "
    "design_insights.md within its byte budget. These are NOT deleted — retrieve "
    "any with `/insights archive restore <n>`, or they may be auto-promoted back "
    "into a turn's context when relevant to the current task.\n\n"
)


def insights_archive_path(repo_root: str) -> str:
    """Return the canonical archive file path under ``repo_root``."""
    return os.path.join(repo_root, ".asicode", "design_insights_archive.md")


def load_archive_file(repo_root: str) -> str:
    """Read the archive file. Returns ``""`` if missing/unreadable."""
    return _load_file_safe(insights_archive_path(repo_root))


# ── Parsed-archive cache ──────────────────────────────────────────────────
# ``build_context_messages`` calls ``build_archive_index`` (Layer 2) AND
# ``select_promotable_entries`` (Layer 3) every turn, each of which previously
# re-read + re-parsed the archive, and Layer 3 re-tokenized EVERY entry. As the
# archive grows this is O(archive) per turn for nothing when the file is
# unchanged. These caches key on ``(path, mtime, size)`` so any
# demotion/restore/drop write — which rewrites the whole file via
# ``atomic_write_text`` — invalidates instantly. Per-entry token-sets are cached
# alongside so Layer 3 only re-tokenizes the (cheap) task query per turn.
_ARCHIVE_WRITE_VERSIONS: dict[str, int] = {}  # path → monotonic counter; each archive path tracks its own write generation
_ARCHIVE_PARSED_CACHE: dict[str, tuple[int, int, int, list[InsightEntry]]] = {}
_ARCHIVE_ANALYZED_CACHE: dict[
    str, tuple[int, int, int, list[InsightEntry], list[frozenset[str]], dict[str, int], float]
] = {}
# Bounded entry cap (P4): these path-keyed caches grew unboundedly in a long-
# lived REPL visiting many repos. FIFO eviction under the GIL is consistent with
# the lock-free, single-threaded design (the current repo is the newest entry).
_ARCHIVE_CACHE_MAX_ENTRIES: int = 8


def _archive_capped_put(cache: dict, key, value, sibling_versions: dict | None = None) -> None:
    """Set ``cache[key] = value`` then FIFO-evict the oldest entry if over cap.

    Pure-dict, GIL-atomic — no lock needed (matches the lock-free cache family).

    When ``sibling_versions`` is given, the per-path write-version counter for an
    evicted key is popped in LOCKSTEP with the content-cache entry. This keeps
    the version dict bounded alongside the content caches and preserves the
    invariant that a version entry only exists while a content-cache entry for
    that path exists — preventing a stale ``version==0`` reset (from evicting
    the version alone) from matching a surviving cache entry. Worst case after
    lockstep is a safe false-miss / re-read, never a stale hit. See
    :func:`_archive_signature` for why the version counter exists
    (belt-and-suspenders against same-mtime+same-size writes, fix #2).
    """
    cache[key] = value
    while len(cache) > _ARCHIVE_CACHE_MAX_ENTRIES:
        _oldest = next(iter(cache))
        cache.pop(_oldest, None)
        if sibling_versions is not None:
            sibling_versions.pop(_oldest, None)


def _archive_signature(path: str) -> tuple[int, int, int]:
    """Return ``(mtime_ns, size, write_version)`` for ``path``.

    Uses *nanosecond* mtime (``st_mtime_ns``) + a per-path monotonic version
    counter so the signature is unique even when a write completes in the same
    second and produces the same byte count as the previous version (fixes #2:
    ns + monotonic counter).  Every explicit archive-write path increments its
    own version via :func:`_archive_invalidate`; the per-path counter is the
    belt-AND-suspenders backup against a future writer forgetting to call
    ``_archive_invalidate``, and avoids cross-repo false-invalidation (e.g. a
    write to repo A's archive does not invalidate repo B's cache).
    """
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size, _ARCHIVE_WRITE_VERSIONS.get(path, 0)
    except OSError:
        return 0, 0, _ARCHIVE_WRITE_VERSIONS.get(path, 0)


def _parsed_archive_cached(repo_root: str) -> list[InsightEntry]:
    """Return parsed archive entries, cached on ``(path, mtime_ns, size, version)``.

    Used by :func:`build_archive_index` (entries only — no tokenizer needed).
    """
    path = insights_archive_path(repo_root)
    mtime_ns, size, version = _archive_signature(path)
    cached = _ARCHIVE_PARSED_CACHE.get(path)
    if cached is not None and cached[0] == mtime_ns and cached[1] == size and cached[2] == version:
        return cached[3]
    content = load_archive_file(repo_root)
    entries = parse_insights(content)[1] if content.strip() else []
    _archive_capped_put(_ARCHIVE_PARSED_CACHE, path, (mtime_ns, size, version, entries),
                        sibling_versions=_ARCHIVE_WRITE_VERSIONS)
    return entries


def _archive_analyzed_cached(
    repo_root: str, tok: Any
) -> tuple[list[InsightEntry], list[frozenset[str]], dict[str, int], float]:
    """Return ``(entries, per-entry-token-sets, df, avgdl)`` cached on ``(path, mtime_ns, size, version)``.

    Reuses parsed entries from :func:`_parsed_archive_cached` so that a
    cache-cold turn parses the archive only once (fixes #3: double-parse).

    ``entries`` and ``toksets`` are always length-aligned (computed together),
    so callers can ``zip`` them safely. The BM25 corpus stats (``df`` per token
    and ``avgdl``) are derived purely from ``toksets`` and cached alongside, so
    :func:`select_promotable_entries` (called every turn) does not recompute
    them. Used by Layer 3 to avoid re-tokenizing the whole archive each turn —
    only the task query is re-tokenized.
    """
    path = insights_archive_path(repo_root)
    mtime_ns, size, version = _archive_signature(path)
    cached = _ARCHIVE_ANALYZED_CACHE.get(path)
    if cached is not None and cached[0] == mtime_ns and cached[1] == size and cached[2] == version:
        return cached[3], cached[4], cached[5], cached[6]
    # Reuse parsed entries from the shared cache — the archive was already
    # read+parsed by ``_parsed_archive_cached`` (via ``build_archive_index``)
    # earlier in this same turn.  This eliminates the double-parse on cold
    # cache (#3).
    entries = _parsed_archive_cached(repo_root)
    toksets = [frozenset(tok.tokenize(f"{e.header_line}\n{e.body}")) for e in entries]
    # BM25 corpus stats: document frequency per token + average doc length.
    # Derived purely from toksets — cache them to skip the per-turn recompute.
    df: dict[str, int] = {}
    for etoks in toksets:
        for t in etoks:
            df[t] = df.get(t, 0) + 1
    avgdl = (sum(len(et) for et in toksets) / len(entries)) if entries else 0.0
    _archive_capped_put(
        _ARCHIVE_ANALYZED_CACHE, path,
        (mtime_ns, size, version, entries, toksets, df, avgdl),
        sibling_versions=_ARCHIVE_WRITE_VERSIONS,
    )
    return entries, toksets, df, avgdl


def _archive_invalidate(repo_root: str) -> None:
    """Drop the archive caches for ``repo_root`` (after a known write).

    Increments the per-path monotonic write-version counter so that even if a
    future writer forgets to call this function, a stale ``(mtime_ns, size)``
    signature will still mis-match on the version component (belt-and-suspenders
    for #2).  Each repo root tracks its own version independently, avoiding
    cross-repo false-invalidation in multi-repo orchestrator mode.
    """
    global _ARCHIVE_WRITE_VERSIONS  # per-path monotonic counter
    path = insights_archive_path(repo_root)
    _ARCHIVE_WRITE_VERSIONS[path] = _ARCHIVE_WRITE_VERSIONS.get(path, 0) + 1
    _ARCHIVE_PARSED_CACHE.pop(path, None)
    _ARCHIVE_ANALYZED_CACHE.pop(path, None)



# ── Active insights file content cache ─────────────────────────────
# ``design_chat_loop.load_design_insights`` reads the ACTIVE
# ``design_insights.md`` every turn (block 0c of the prompt prefix, which MUST
# stay byte-stable across turns). The archive family above caches the ARCHIVE
# file; this mirrors it for the ACTIVE file so the per-turn ``open()+read()`` is
# skipped when the file is unchanged — matching the mtime cache already used for
# ``project.md`` (``context_manager``). Same ``(mtime_ns, size, write_version)``
# signature scheme; every active-file writer calls :func:`_active_invalidate`,
# and the per-path write-version counter is the belt-and-suspenders against a
# future writer forgetting to (identical rationale to :func:`_archive_invalidate`).
_ACTIVE_WRITE_VERSIONS: dict[str, int] = {}
_ACTIVE_CONTENT_CACHE: dict[str, tuple[int, int, int, str]] = {}


def _active_signature(path: str) -> tuple[int, int, int]:
    """Return ``(mtime_ns, size, write_version)`` for the active insights file."""
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size, _ACTIVE_WRITE_VERSIONS.get(path, 0)
    except OSError:
        return 0, 0, _ACTIVE_WRITE_VERSIONS.get(path, 0)


def load_active_insights_cached(repo_root: str) -> str:
    """Return the ACTIVE ``design_insights.md`` content, cached on signature.

    Used by :func:`design_chat_loop.load_design_insights` (block 0c) to skip the
    per-turn ``open()+read()``. The signature ``(mtime_ns, size, version)`` is
    identical to the archive cache family, so any active-file write detected via
    :func:`_active_invalidate` — or by a plain mtime/size change — invalidates
    instantly. Returns ``""`` when the file is missing (mirrors
    :func:`load_insights_file`).
    """
    path = insights_path(repo_root)
    mtime_ns, size, version = _active_signature(path)
    cached = _ACTIVE_CONTENT_CACHE.get(path)
    if (
        cached is not None
        and cached[0] == mtime_ns
        and cached[1] == size
        and cached[2] == version
    ):
        return cached[3]
    content = _load_file_safe(path)
    _archive_capped_put(_ACTIVE_CONTENT_CACHE, path, (mtime_ns, size, version, content),
                        sibling_versions=_ACTIVE_WRITE_VERSIONS)
    return content


def _active_invalidate(repo_root: str) -> None:
    """Drop the active-file content cache for ``repo_root`` (after a known write).

    Increments the per-path write-version counter so that even a same-ns-same-size
    write still mis-matches on the version component. Mirrors
    :func:`_archive_invalidate`.
    """
    global _ACTIVE_WRITE_VERSIONS
    path = insights_path(repo_root)
    _ACTIVE_WRITE_VERSIONS[path] = _ACTIVE_WRITE_VERSIONS.get(path, 0) + 1
    _ACTIVE_CONTENT_CACHE.pop(path, None)


def _entry_bytes(entry: "InsightEntry") -> int:
    """UTF-8 byte size of an entry's serialized lines."""
    return len("".join(entry.lines).encode("utf-8"))


def select_demotion_candidates(
    entries: list[InsightEntry],
    budget_bytes: int,
    now: Optional[float] = None,
) -> list[int]:
    """Return 1-based indices of entries to DEMOTE so the remaining entries fit
    ``budget_bytes``, preferring the OLDEST timestamped entries first.

    ``budget_bytes`` is an ENTRIES-ONLY budget — it must NOT include any
    preamble/non-entry overhead; the caller (e.g.
    :func:`enforce_budget_by_demotion`) subtracts the preamble before calling so
    the demotion math stays consistent with the full-file budget it enforces.

    Entries whose header has no parseable timestamp (hand-written / preamble-
    derived) are NEVER selected — mirrors :func:`select_entries_older_than`: those
    are the sacrosanct principles. If even demoting every timestamped entry cannot
    reach the budget (e.g. the timestamp-less set alone exceeds it), every
    timestamped entry is returned and the residual over-budget is accepted rather
    than ever demoting a principle or deleting anything.

    Greedy oldest-first: returns the minimum set of oldest entries whose combined
    size brings the ENTRIES-ONLY total under ``budget_bytes``.
    """
    now = time.time() if now is None else now
    total = sum(_entry_bytes(e) for e in entries)
    if total <= budget_bytes:
        return []
    # (1-based idx, age, size) — timestamped entries only.
    timed: list[tuple[int, float, int]] = []
    for i, entry in enumerate(entries, 1):
        age = entry_age_days(entry, now=now)
        if age is not None:
            timed.append((i, age, _entry_bytes(entry)))
    if not timed:
        return []  # all timestamp-less — never demote principles
    timed.sort(key=lambda t: t[1], reverse=True)  # oldest (largest age) first
    target_savings = total - budget_bytes
    demoted: list[int] = []
    saved = 0
    for idx, _age, size in timed:
        if saved >= target_savings:
            break
        demoted.append(idx)
        saved += size
    return demoted


def append_entries_to_archive(
    repo_root: str, entries: list[InsightEntry]
) -> None:
    """Append ``entries`` to the archive file (creating it with a preamble if new).

    Round-trip safe: parses the existing archive (preamble + entries), extends,
    re-serializes. Demoted blocks keep their original ``### [cat] ts`` headers
    verbatim, so a later ``restore`` re-inserts them unchanged.

    Crash-idempotent (dedup): :func:`enforce_budget_by_demotion` appends to the
    archive BEFORE truncating the active file, so a crash between the two writes
    leaves the demoted entries in BOTH files. The next enforce run re-demotes and
    re-appends them; a blind ``extend`` would then accumulate duplicates in the
    archive. We dedup on ``(header_line, body)`` — stable across demote/restore
    because headers keep their original timestamp verbatim — so a re-append is
    absorbed. Genuine distinct entries differ in body, so this only ever drops
    exact crash-duplicates.

    Concurrency: the whole read-modify-write is serialized by
    :func:`insights_write_lock` (re-entrant — :func:`enforce_budget_by_demotion`
    may already hold it).
    """
    if not entries:
        return
    with insights_write_lock(repo_root):
        path = insights_archive_path(repo_root)
        existing = load_archive_file(repo_root) or _ARCHIVE_PREAMBLE
        preamble, arch_entries = parse_insights(existing)
        seen: set[tuple[str, str]] = {
            (e.header_line, e.body) for e in arch_entries
        }
        for entry in entries:
            key = (entry.header_line, entry.body)
            if key in seen:
                continue
            seen.add(key)
            arch_entries.append(entry)
        atomic_write_text(path, serialize_insights(preamble, arch_entries))
        _archive_invalidate(repo_root)  # proactive cache drop (mtime/size key also covers it)


def enforce_budget_by_demotion(
    repo_root: str, budget_bytes: int, now: Optional[float] = None
) -> tuple[int, int]:
    """HARD budget backstop — the single source of truth guaranteeing the active
    file is ≤ ``budget_bytes``.

    If the active file is over budget, demote the oldest timestamped entries to
    the archive until it fits. Returns ``(n_demoted, remaining_bytes)``. Never
    deletes anything; never demotes a timestamp-less/principle entry. Returns
    ``(0, size)`` when already within budget, or when nothing timestamped can be
    demoted (the conservative residual-over-budget case).

    Concurrency: the whole read-modify-write (active + archive) is serialized by
    :func:`insights_write_lock`, re-entrant so the internal call to
    :func:`append_entries_to_archive` does not self-deadlock. Without this lock a
    concurrent ``_save_insight_to_file`` (append) landing between this function's
    read and its rewrite would be silently lost — violating the "0 durable loss"
    contract that the durability ordering below exists to uphold.
    """
    with insights_write_lock(repo_root):
        content = load_insights_file(repo_root)
        if not content:
            return 0, 0
        preamble, entries = parse_insights(content)
        total = len(content.encode("utf-8"))
        if total <= budget_bytes:
            return 0, total
        # ``budget_bytes`` is a FULL-FILE budget (preamble + entries), but
        # :func:`select_demotion_candidates` reasons about ENTRIES-ONLY bytes.
        # Subtract the non-entry preamble so the band
        # (entries_only <= budget < full_file) actually demotes instead of leaving
        # the file over budget (violating the hard-budget contract).
        preamble_bytes = len("".join(preamble).encode("utf-8"))
        entry_budget = max(0, budget_bytes - preamble_bytes)
        demote_idx = select_demotion_candidates(entries, entry_budget, now=now)
        if not demote_idx:
            return 0, total
        dset = set(demote_idx)
        demoted = [e for i, e in enumerate(entries, 1) if i in dset]
        kept = [e for i, e in enumerate(entries, 1) if i not in dset]
        new_content = serialize_insights(preamble, kept)
        # DURABILITY: append to the archive BEFORE truncating the active file. A
        # crash between the two writes then leaves (at worst) active+archive
        # DUPLICATES of the demoted entries — never a loss. Recovery is truly
        # idempotent: the next enforce run re-demotes the still-present entries, and
        # append_entries_to_archive dedups the re-append on (header_line, body) so
        # the archive never accumulates crash-duplicates. The old active-first order
        # removed the entries from active and, if the process died before the archive
        # append, dropped them permanently ("0 durable loss" goal violated).
        append_entries_to_archive(repo_root, demoted)
        atomic_write_text(insights_path(repo_root), new_content)
        _active_invalidate(repo_root)  # drop 0c content cache — active file just changed
        return len(demoted), len(new_content.encode("utf-8"))


def build_archive_index(
    repo_root: str, max_entries: int = ARCHIVE_INDEX_MAX_ENTRIES
) -> str:
    """Compact header-only index of archived entries, for always-on visibility.

    Returns ``""`` when there is no archive. Capped at ``max_entries`` (newest
    archived first — the most recently demoted are likeliest to still matter),
    with a trailing "… and N more" note. ~tens of bytes per line, so the index
    itself is bounded and never re-introduces unbounded context cost.
    """
    entries = _parsed_archive_cached(repo_root)
    if not entries:
        return ""
    shown = list(reversed(entries[-max_entries:]))  # newest archived first
    lines = [
        "",
        "=== ARCHIVED INSIGHTS (not auto-injected; "
        "/insights archive list | restore <n>) ===",
    ]
    # BYTE-STABILITY: show the absolute creation DATE, NOT a relative age ("Nd").
    # This index is injected into the cached 0c prompt-prefix block; a relative
    # age would tick over at each UTC day boundary and silently invalidate the
    # compressed-summary + verbatim-turns cache even when nothing changed. The
    # header timestamp never changes, so the date is stable forever. Relative
    # age is still surfaced in the interactive ``/insights archive list`` view.
    #
    # INDEX-STABILITY: use file-order 1-based indices (A1 = oldest, A<n> = newest)
    # to match ``/insights archive list`` and ``restore/drop <n>`` CLI commands.
    # The display order is newest-first, but the label must reflect the archive
    # file's actual position so users can reliably restore/drop by index.
    n_total = len(entries)
    for disp_idx, entry in enumerate(shown):
        # file_idx: 1-based position in archive file (oldest=1, newest=n_total)
        file_idx = n_total - disp_idx
        ts = parse_timestamp(entry.header_line)
        date_s = time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts is not None else "—"
        first = (entry.body.strip().split("\n", 1)[0])[:70]
        cat = entry.category or "—"
        lines.append(f"  A{file_idx}. [{cat}] ({date_s}) {first}")
    extra = len(entries) - len(shown)
    if extra > 0:
        lines.append(f"  … and {extra} more (see /insights archive list)")
    return "\n".join(lines) + "\n"


def select_promotable_entries(
    repo_root: str,
    task_query: str,
    max_entries: int = PROMOTE_MAX_ENTRIES,
    max_bytes: int = PROMOTE_MAX_BYTES,
    min_score: float = PROMOTE_MIN_SCORE,
) -> list[InsightEntry]:
    """Relevance-rank archived entries against ``task_query``; return the top few
    to PROMOTE back into this turn's context (Layer 3).

    Cheap & local (no LLM): BM25-IDF weighted token overlap over the archive
    corpus.  Entries sharing tokens with the task query are scored using BM25's
    IDF formula, which down-weights tokens that appear in many archive entries
    (common/unspecific language) and up-weights rare, informative ones.  The
    top ``max_entries`` by score (within ``max_bytes``) are promoted.  Returns
    ``[]`` when there is no archive, no query, or no match — so turns where no
    archived insight is relevant pay zero extra cost.
    """
    if not task_query or not task_query.strip():
        return []
    # Module-level lazy tokenizer singleton (fixes #4: no per-turn init).
    global _TOKENIZER  # lazy sentinel
    try:
        if _TOKENIZER is None:
            from external_llm.agent.rag_configs import CodeTokenizer
            _TOKENIZER = CodeTokenizer()
        tok = _TOKENIZER
    except Exception:
        return []  # non-critical — never block injection
    qset = set(tok.tokenize(task_query))
    if not qset:
        return []
    # Parsed entries + per-entry token-sets + BM25 corpus stats (df, avgdl) come
    # from the (mtime, size)-keyed cache, so only the task query is tokenized
    # per turn — the whole archive is NOT re-tokenized (nor its corpus stats
    # recomputed) when it hasn't changed.
    entries, toksets, df, avgdl = _archive_analyzed_cached(repo_root, tok)
    if not entries:
        return []
    n_docs = len(entries)
    if avgdl == 0:
        return []
    _K1, _B = 1.5, 0.75  # BM25 tuning (matches rag_searcher)
    # IDF depends ONLY on the query token (+ corpus stats df/n_docs, which are
    # loop-invariant) — NOT on the entry — so precompute it once over qset
    # (turn 13112 perf #8, symmetric to the tf_norm hoist below). Previously
    # math.log was recomputed on every entry × every shared term.
    idf_map = {
        qt: math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
        for qt in qset
    }
    # ── Score each entry ───────────────────────────────────────────────
    scored: list[tuple[float, InsightEntry]] = []
    for entry, etoks in zip(entries, toksets):
        shared = qset & etoks
        if not shared:
            continue
        doc_len = len(etoks)
        # TF normalisation depends ONLY on doc_len (not the query term), so hoist
        # it out of the per-term loop — it was recomputed on every `qt` iteration
        # even though doc_len/K1/B/avgdl are loop-invariant (turn 13110 perf #7).
        tf_norm = 1.0 * (_K1 + 1) / (1.0 + _K1 * (1 - _B + _B * doc_len / avgdl))
        # IDF is query-token-only and precomputed in idf_map above; `shared` is a
        # subset of qset so every lookup hits. score = tf_norm * Σ IDF over shared.
        score = tf_norm * sum(idf_map[qt] for qt in shared)
        if score >= min_score:
            scored.append((score, entry))
    if not scored:
        return []
    # Highest score first; stable so ties keep archive order.
    scored.sort(key=lambda t: t[0], reverse=True)
    promoted: list[InsightEntry] = []
    total = 0
    for _score, entry in scored:
        if len(promoted) >= max_entries:
            break
        esz = _entry_bytes(entry)
        if total + esz > max_bytes:
            continue  # skip oversize, try smaller high-relevance ones
        promoted.append(entry)
        total += esz
    return promoted


def _age_reference_block(content: str, now: Optional[float] = None) -> str:
    """Reference list of each entry's header + age, appended to the triage prompt.

    Lets the compact/verify model weigh staleness. Explicitly marked DO-NOT-COPY
    so the ages never leak into the rewritten file (which must preserve the
    original ``### [cat] timestamp`` header format). Returns ``""`` when there
    are no entries.
    """
    _, entries = parse_insights(content)
    rows: list[str] = []
    for i, entry in enumerate(entries, 1):
        age = entry_age_days(entry, now=now)
        age_str = f"{int(age)}d old" if age is not None else "age unknown"
        rows.append(f"  {i}. {entry.header_line}  — {age_str}")
    if not rows:
        return ""
    return (
        "\n\nEntry ages (recency reference for triage — DO NOT copy these ages "
        "into the rewritten file; preserve each original header line verbatim):\n"
        + "\n".join(rows)
    )


def build_compact_messages(content: str, budget_bytes: Optional[int] = None) -> list[dict]:
    """Build the LLM chat messages that compact the insights file.

    The model is asked to return the **entire rewritten file** (preamble +
    compacted entries). Compaction is a *value triage*, not mechanical dedup:
    KEEP architectural knowledge worth carrying into future sessions, DROP
    ephemeral debug/verification workarounds, and merge superseded entries
    into their resolution. The caller posts these messages to the
    compress-LLM and writes the reply back atomically.

    ``budget_bytes`` is the size ceiling: when the input exceeds it, a
    MANDATORY reduction directive is appended so the model MERGES related
    entries + tightens prose to fit, rather than echoing an all-durable file
    unchanged (which leaves the size nudge's warned condition unaddressed).
    When ``None`` or not exceeded, behavior is the original value-triage.
    """
    system = (
        "You curate a project's design-insights file (.asicode/design_insights.md). "
        "It accumulates notes across coding sessions. Your job: compact it so it carries "
        "ONLY the knowledge a future session would have to re-derive expensively. "
        "This is value triage, not mechanical dedup.\n\n"
        "KEEP (long-term architectural value) — summarize down to the core principle:\n"
        "  - design constraints, single-source-of-truth mappings, extension principles\n"
        "  - cross-cutting decisions (e.g. 'data, not per-language branches')\n"
        "  - dependency directions, invariants, patterns, capability shims\n"
        "  For each KEEP entry, drop the implementation/debug narrative (step-by-step "
        "troubleshooting, commit-hash chasing, ad-hoc verification scripts, 'current guard "
        "status' snapshots) and retain only the durable constraint + a one-line rationale.\n\n"
        "DROP (ephemeral / superseded):\n"
        "  - workarounds and 'avoidance rules' whose root cause is now fixed\n"
        "  - 'current guard status' snapshots that go stale once code moves\n"
        "  - implementation-completion / verification notes whose purpose is now done\n"
        "  - implementation battle records (the file's 원칙 says these are NOT kept)\n\n"
        "SUPERSEDED handling: when a later entry resolves or corrects an earlier one on "
        "the same subject, DROP the earlier one entirely and keep only the resolution — "
        "never preserve both, never carry a contradiction forward.\n\n"
        "RECENCY: an 'Entry ages' reference is provided below. Older entries are likelier "
        "to be stale workarounds/snapshots — when an old entry is NOT a durable invariant, "
        "prefer DROP. But never drop a still-valid durable constraint merely for being old: "
        "old, stable invariants (single-source-of-truth maps, dependency directions) are "
        "often the MOST valuable. Age informs triage; it does not override value.\n\n"
        "Format:\n"
        "1. Keep the '#' title and '>' blockquote '원칙' (principle) preamble VERBATIM.\n"
        "2. Preserve each surviving entry's '### [category] timestamp' header line format; "
        "merge near-duplicate entries into one, unioning their durable points.\n"
        "3. Do NOT invent new insights or alter technical facts; only curate what exists.\n"
        "4. Output the FULL rewritten file and NOTHING else (no markdown fences, no commentary)."
    )

    cur_bytes = len(content.encode("utf-8"))
    if budget_bytes is not None and cur_bytes > budget_bytes:
        # The size nudge fires at this same threshold. When it does, compact MUST
        # actually shrink the file — otherwise the warned condition is never
        # remedied. Reduction is by MERGING + tightening, never by dropping valid
        # durable facts or fabricating summaries: every durable point survives,
        # only the word-count shrinks.
        system += (
            "\n\nBUDGET ENFORCEMENT (MANDATORY): the file is "
            f"{cur_bytes:,} bytes — OVER the {budget_bytes:,}-byte budget. You MUST reduce "
            f"the rewritten output to ≤ {budget_bytes:,} bytes. Achieve this by MERGING "
            "related/same-subject entries into one tighter entry (union their durable points) "
            "and TIGHTENING prose (drop the narrative, keep the invariant). Do NOT drop a "
            "still-valid durable fact merely to hit the budget, and do NOT invent or merge "
            "unrelated facts. If merging cannot reach the budget, drop the single lowest-value "
            "entry and continue. This overrides the default 'echo unchanged when already "
            "minimal' behavior — an all-durable file can still be reduced by tightening."
        )

    user = f"Compact this design-insights file:\n\n{content}" + _age_reference_block(content)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verify_messages(content: str) -> list[dict]:
    """Build the LLM chat messages that *verify* the insights file.

    Unlike :func:`build_compact_messages` (a blind value-triage that just
    summarizes), verification is meant to run inside a **design-chat tool loop**
    (``DesignChatLoop``) so the agent can confirm each entry against the *current*
    codebase with its read tools (read_file/grep/find_symbol/get_file_outline)
    before rewriting the file in place with edit_text/apply_patch.

    The caller posts these messages to the design-chat loop; the loop's tool use
    performs the actual verification + rewrite. This is why there is no separate
    atomic write here — the agent edits the file directly.
    """
    system = (
        "You verify and curate a project's design-insights file "
        "(.asicode/design_insights.md). Unlike compaction, you MUST verify each "
        "entry against the CURRENT codebase with your read tools "
        "(read_file, grep, find_symbol, get_file_outline) before deciding its fate.\n\n"
        "DROP an entry when it is any of:\n"
        "  - an implementation-completion or verification note whose purpose is now done\n"
        "  - a stale 'current status' snapshot (guards/code that have since moved)\n"
        "  - a WRONG fact: the referenced symbol/file/path does NOT exist, or the "
        "described mechanism does NOT match the actual code (read the code to confirm)\n"
        "  - superseded or contradicted by a later entry on the same subject\n\n"
        "KEEP an entry only when it is a durable LONG-TERM architectural constraint "
        "worth carrying into future sessions: single-source-of-truth mappings, "
        "extension principles, invariants, cross-cutting decisions, capability shims. "
        "For each KEEP entry, trim the debug/verification narrative down to the core "
        "constraint plus a one-line rationale.\n\n"
        "RECENCY: an 'Entry ages' reference is provided below. Treat old entries with "
        "extra suspicion — they are likelier to be stale snapshots or fixed workarounds — "
        "but age never overrides verification: an old entry that is still TRUE and durable "
        "(confirmed against the code) is kept; a recent entry that is wrong is dropped.\n\n"
        "PROCESS:\n"
        "1. Read the current .asicode/design_insights.md (it is also given below).\n"
        "2. For each '### [category] timestamp' entry, confirm its technical claims "
        "against the codebase with your read tools. A claim is WRONG if a referenced "
        "symbol/file/path is missing or the described mechanism differs from reality.\n"
        "3. Rewrite .asicode/design_insights.md IN PLACE with edit_text/apply_patch: "
        "keep the '#' title and '>' blockquote preamble VERBATIM, preserve each "
        "surviving entry's '### [category] timestamp' header line format, and merge "
        "near-duplicates. Do NOT invent new insights or alter technical facts.\n"
        "4. Finish with a one-line summary of how many entries you kept vs dropped "
        "and the single most important correction you made (e.g. a wrong symbol path).\n\n"
        "You edit the file directly; do not print the whole rewritten file as text."
    )
    user = (
        "Verify and curate this design-insights file against the current codebase. "
        "Read the actual code to confirm each claim, then edit the file in place to "
        "drop wrong/done/ephemeral entries and keep only durable long-term insights.\n\n"
        f"{content}"
        + _age_reference_block(content)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
