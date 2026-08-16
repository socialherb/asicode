"""Shared parse cache for analysis scanners.

Every scanner used to ``open().read()`` + ``ast.parse()`` each file
independently, so a pipeline running N scanners over the same file set paid
the read/parse cost N times.  This module memoises both, keyed by
``(path, mtime_ns, size)`` so edits made mid-pipeline (e.g. by the executor)
invalidate entries automatically — no explicit cache-clearing protocol needed.

The returned ``ast.Module`` objects are shared across callers; scanners only
walk them, never mutate.

Cache sizing
------------
The cross-scanner reuse above only materialises when the cache can hold every
file in the working set *at once*.  Scanners run one-at-a-time over the full
file list (scanner A over all N files, then scanner B over all N, ...), so with
a fixed cache smaller than N the early entries from scanner A are evicted
before scanner B reaches them — every later scanner re-parses from scratch and
the shared cache delivers nothing.  ``ensure_capacity(n)`` grows the cache (up
to ``_MAX_CACHE_SIZE`` entries) so a known file set fits.

Every consumer with a known working set sizes the cache BEFORE its first
parse (P2 policy, 2026-08-11):

* the scanner registry — once per ``run()`` with ``len(file_paths)`` (the
  CAPPED scan list, <= ``SCAN_FILE_CAP``);
* ``RepositoryGraph.build(collect_imported_names=True)`` — with the UNCAPPED
  walked py count, because the graph walk never truncates (unlike the scan
  lists) and its name pass parses every walked py file;
* ``compute_cross_file_referenced_names_light`` — with its py working set
  (the scan list unioned with the uncapped ``graph.py_files``), before the
  importer pass.

Disk-cache concurrency policy (B1 read + B2 write, 2026-08-16)
--------------------------------------------------------------
The per-scanner disk caches under ``<repo_root>/.cache/`` are advisory,
regenerable state shared by concurrent processes — parallel sessions'
pre-commit gates, manual scans, long-lived REPLs.  Their contract is
LOCK-FREE by design:

* **WRITE — atomic whole-file replace only**, via
  ``common.atomic_io.atomic_write_json`` (sibling temp + ``os.replace``).
  A reader never observes a partial file; a crash mid-save leaves the
  PREVIOUS cache intact, never a truncated one.  The one deliberate
  exception is ``graph/structural_cache.save``, which streams a ~57MB
  payload entry-by-entry through its own temp+replace (a single ``json.dump``
  would re-create the ~335MB transient that streaming removed) — same
  rename atomicity, different serialization body.
* **NO LOCKS** — a stale lock would block every future gate over a defect
  that already self-heals.  Correctness must never depend on mutual
  exclusion for advisory data.
* **LAST-WRITER-WINS** — two processes saving concurrently both succeed;
  the file ends up as ONE complete payload, never a byte-level mix.  Entries
  only the loser had are recomputed on a later fingerprint miss.
* **READ — fail-open** (B1): any ``OSError``/``ValueError``/version mismatch
  yields an empty cache and a full recompute, and fingerprints are captured
  together with the bytes they describe (``read_with_fingerprint``), so a
  torn stat/read pair cannot poison an entry.

``ensure_capacity`` only ever grows, so the largest set wins and later calls
are no-ops — sizing at each consumer composes safely.  Growing changes the
entry-count ceiling in place and NEVER drops populated entries (unlike the
lru_cache-wrapper rebuild this module used before, which discarded the
previous consumer's warm entries on every regrow).

The hard ceiling (``_MAX_CACHE_SIZE``) is deliberately >= ``SCAN_FILE_CAP`` +
headroom, so any CAPPED scanner list always fits (pinned by a test in
tests/unit/analysis/test_parse_cache.py — parse_cache itself stays
stdlib-only and never imports scan_walk).  The UNCAPPED graph/cross-ref sets
are best-effort above the ceiling: a memory guard (ASTs are heavy), and the
structural gate fail-closes above the cap anyway, so no gated scan ever runs
on a set the cache cannot hold.

Byte budget
-----------
Entry count alone is not a memory bound: one large generated ``.py`` can be
tens of MB, and its AST is ~15x the source (measured 377 MiB AST / 25.8 MiB
source ≈ 14.6x).  Sharing the cache with CallGraphIndexer.build() put the
whole repo's ASTs resident for the process lifetime (+421 MB on this repo
before the budget).  The cache therefore tracks a byte cost per entry —
``len(src)`` for a read-only entry, ``len(src) * (1 + _AST_BYTES_PER_SOURCE_BYTE)``
for a parsed one — and evicts least-recently-used entries while the total
exceeds ``_MAX_CACHE_BYTES`` (256 MiB).  A single entry larger than the whole
budget is not cached at all.
"""

from __future__ import annotations

import ast
import logging
import os
import threading
from collections import OrderedDict, namedtuple
from typing import Optional

logger = logging.getLogger(__name__)

# Default capacity for ad-hoc callers that never invoke ``ensure_capacity``.
_DEFAULT_CACHE_SIZE = 256
# Hard ceiling — ASTs are heavy; never let the cache grow without bound when a
# caller passes a huge (or attacker-controlled) file count.
_MAX_CACHE_SIZE = 4096
# Headroom so a few files touched outside the declared set don't evict members
# of the working set.
_CAPACITY_HEADROOM = 16

# ASTs weigh ~15x their source (measured 377 MiB AST / 25.8 MiB source).
_AST_BYTES_PER_SOURCE_BYTE = 15
# Total byte budget: entry cost, not entry count, is the real memory bound.
# 256 MiB keeps the shared cache under control while holding a large repo's
# parsed set (this repo: ~810 py files ≈ 420 MB un-budgeted -> ~256 MB here).
_MAX_CACHE_BYTES = 256 << 20

# Cache state.  Each entry is ``key -> (payload, cost)`` where payload is
# ``(src, tree_or_None)``; ``cost`` is the byte weight used by the budget.
_cache: OrderedDict[tuple, tuple[tuple[Optional[str], Optional[ast.Module]], int]] = OrderedDict()
_bytes = 0  # Sum of entry costs currently resident.
_max_entries = _DEFAULT_CACHE_SIZE  # Entry-count ceiling (ensure_capacity).
_hits = 0
_misses = 0

# Bookkeeping lock (C1/C2, 2026-08-12): the cache is shared across three lock
# domains — RepositoryGraph.build (facade RLock), CallGraphIndexer.build (CGI
# RLock) and the scanners (no lock) — so one turn's parallel batch can touch
# it concurrently.  The lock guards ONLY the bookkeeping (_get/_put/_evict_lru/
# ensure_capacity/clear/cache_info): file I/O (_read_impl) and ast.parse
# (_parse_src) stay OUTSIDE it — serializing them would defeat the cache's
# purpose (parallel scanners paying the read/parse cost one at a time).
# RLock because _put and ensure_capacity call _evict_lru re-entrantly.
_lock = threading.RLock()


def _stat_key(abs_path: str) -> Optional[tuple]:
    try:
        st = os.stat(abs_path)
    except OSError:
        logger.debug("parse_cache: stat failed for %s", abs_path, exc_info=True)
        return None
    return (abs_path, st.st_mtime_ns, st.st_size)


def _read_impl(abs_path: str, mtime_ns: int, size: int) -> Optional[str]:
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        logger.debug("parse_cache: read failed for %s", abs_path, exc_info=True)
        return None


def _parse_src(src: str, abs_path: str) -> Optional[ast.Module]:
    try:
        return ast.parse(src, filename=abs_path)
    except SyntaxError:
        logger.debug("parse_cache: syntax error in %s", abs_path, exc_info=True)
        return None


def _get(key: tuple) -> Optional[tuple[Optional[str], Optional[ast.Module]]]:
    """Look up *key*, refreshing LRU order.  Returns None on miss."""
    global _hits, _misses
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            _misses += 1
            return None
        _hits += 1
        _cache.move_to_end(key)
        return entry[0]


def _put(key: tuple, payload: tuple[Optional[str], Optional[ast.Module]], cost: int) -> None:
    """Insert *payload* under *key* with byte weight *cost*, evicting LRU-first.

    A single entry heavier than the whole budget is not cached at all — it
    would evict everything else for one file that callers may never reuse.
    """
    global _bytes
    with _lock:
        old = _cache.pop(key, None)
        if old is not None:
            _bytes -= old[1]
        if cost > _MAX_CACHE_BYTES:
            return
        while _bytes + cost > _MAX_CACHE_BYTES and _cache:
            _evict_lru()
        _cache[key] = (payload, cost)
        _bytes += cost
        while len(_cache) > _max_entries:
            _evict_lru()


def _evict_lru() -> None:
    global _bytes
    _key, (_payload, cost) = _cache.popitem(last=False)
    _bytes -= cost


def _read_ast(abs_path: str) -> Optional[tuple[Optional[str], Optional[ast.Module]]]:
    """Stat once, then return ``(src, tree)`` — None values mean the file is
    missing/unreadable (src) or unparseable (tree).  Populates the cache when
    the entry was absent or only held the source string."""
    key = _stat_key(abs_path)
    if key is None:
        return None
    entry = _get(key)
    if entry is not None:
        src, tree = entry
        if tree is not None:
            return src, tree
        # Source-only entry (read_source populated it): parse and upgrade.
        tree = _parse_src(src, key[0])
        _put(key, (src, tree), len(src) * (1 + _AST_BYTES_PER_SOURCE_BYTE))
        return src, tree
    src = _read_impl(*key)
    if src is None:
        return None
    tree = _parse_src(src, key[0])
    _put(key, (src, tree), len(src) * (1 + _AST_BYTES_PER_SOURCE_BYTE))
    return src, tree


# ── Fingerprint order contract (B1, 2026-08-16) ─────────────────────────────
# Every per-file cache admits entries as ``cache[(mtime_ns, size)] = analysis``.
# The invariant that keeps concurrent edits harmless: the stat that produces
# the fingerprint must be taken BEFORE (or with) the read that produces the
# content — the stamp may then describe a state OLDER than the content (a
# writer interleaved), which only makes the torn entry UNREACHABLE: the next
# run's stat sees the post-write stamp, mismatches, and recomputes.  The
# reverse order (read, then stat) pairs a POST-write stamp with PRE-write
# analysis — an entry that HITS on every future run of the new content and
# silently serves stale results.  All eight call sites (parse_cache itself,
# vulture scan + 3 pre-process passes, dead-block, container, unused-import,
# importer-export, graph stamps) follow the safe order; new caches must use
# :func:`read_with_fingerprint` (fused) or :func:`stat_fingerprint` (enum-time)
# instead of re-deriving it.


def stat_fingerprint(abs_path: str) -> Optional[tuple[int, int]]:
    """Canonical ``(st_mtime_ns, st_size)`` cache fingerprint, or None.

    One stat code path for every per-file disk cache (replaces the five
    copy-pasted ``_*_stat`` helpers).  Order contract: a fingerprint from
    here may only key content read at or after this stat — see
    :func:`read_with_fingerprint` for why the reverse silently serves
    stale entries.
    """
    key = _stat_key(abs_path)
    if key is None:
        return None
    return (key[1], key[2])


def read_with_fingerprint(
    abs_path: str,
) -> Optional[tuple[str, tuple[int, int]]]:
    """Fused read: ``(content, fingerprint)`` captured under ONE stat.

    The stat precedes the read, so the returned pair can never pair a
    post-write stamp with pre-write content — the torn entry a concurrent
    writer produces is keyed by the PRE-write stamp and becomes unreachable
    (next run re-stats, mismatches, recomputes) instead of wrong.  Sites that
    consume BOTH the content and a fingerprint for cache admission must use
    this helper rather than stat-then-read by hand; never read first and
    stat afterwards.
    """
    key = _stat_key(abs_path)
    if key is None:
        return None
    src = _read_impl(key[0], key[1], key[2])
    if src is None:
        return None
    return src, (key[1], key[2])


def ensure_capacity(n: int) -> None:
    """Grow the cache so a working set of *n* files fits at once.

    No-op when the current ceiling already covers *n* (+ headroom), so
    repeated calls over the same file set cost nothing.  Shrinking is never
    performed — a smaller follow-up set keeps the larger ceiling.  Growing
    only raises the entry-count ceiling in place; populated entries are
    preserved (the old lru_cache-wrapper rebuild dropped every entry, which
    discarded the previous consumer's warm set on any regrow).
    """
    global _max_entries
    with _lock:
        target = min(n + _CAPACITY_HEADROOM, _MAX_CACHE_SIZE)
        _max_entries = max(_max_entries, target)
        while len(_cache) > _max_entries:
            _evict_lru()


def read_source(abs_path: str) -> Optional[str]:
    """Cached file read.  Returns None when the file is missing/unreadable."""
    key = _stat_key(abs_path)
    if key is None:
        return None
    entry = _get(key)
    if entry is not None:
        return entry[0]
    src = _read_impl(*key)
    if src is None:
        return None
    _put(key, (src, None), len(src))
    return src


def parse_ast(abs_path: str) -> Optional[ast.Module]:
    """Cached ``ast.parse``.  Returns None on read failure or SyntaxError."""
    pair = _read_ast(abs_path)
    if pair is None:
        return None
    return pair[1]


def read_and_parse(abs_path: str) -> tuple[Optional[str], Optional[ast.Module]]:
    """Cached read + parse under a SINGLE stat.

    ``read_source`` and ``parse_ast`` each stat independently, so calling both
    back-to-back can observe two different file versions (the read winning the
    race).  This is the consistent pair: both values come from one stat key,
    and the tree is always parsed from the returned source string.
    """
    pair = _read_ast(abs_path)
    if pair is None:
        return None, None
    return pair[0], pair[1]


def clear() -> None:
    """Drop all cached entries (tests / long-lived processes)."""
    global _bytes, _hits, _misses
    with _lock:
        _cache.clear()
        _bytes = 0
        _hits = 0
        _misses = 0


# Same shape as functools.lru_cache's CacheInfo so existing callers/tests can
# read .hits/.misses/.maxsize/.currsize.
CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def cache_info() -> CacheInfo:
    """``(hits, misses, maxsize, currsize)`` — lru_cache.cache_info() shape."""
    with _lock:
        return CacheInfo(_hits, _misses, _max_entries, len(_cache))


# ── Disk-cache path guard ────────────────────────────────────────────────────
# The per-file disk caches (dead-block extraction, container reachability,
# unused-import analysis, vulture scan) all live in ``<repo_root>/.cache/`` as
# plain ``.json`` files.  The path must be validated fail-closed: a mistyped
# cache path is not a performance problem but a DATA-DESTRUCTION bug — one has
# already overwritten ``webapp/ui/ui_tools.py`` with a 4.9 MB JSON dump
# (2026-08-16, parallel-session cache-path slip).  Loaders must NOT swallow
# ``CachePathError`` in their fail-open handlers: an invalid path is a
# programming error that should fail the run loudly, while a corrupt or
# absent cache file still fails open to a full recomputation.


class CachePathError(ValueError):
    """A cache path would escape ``<repo_root>/.cache/`` — fail-closed."""


def cache_file_path(repo_root: str, filename: str) -> str:
    """``os.path.join(repo_root, ".cache", filename)`` with a fail-closed guard.

    Validates that *filename* is a bare ``.json`` name (no directory
    components, no traversal) and that the resolved path stays inside
    ``<repo_root>/.cache/`` (e.g. a ``.cache`` symlink pointing elsewhere is
    rejected).  Raises :class:`CachePathError` on any deviation instead of
    returning a path that could point at — and overwrite — a source file.
    """
    if not repo_root:
        raise CachePathError(f"empty repo_root for cache file {filename!r}")
    if not filename or os.path.basename(filename) != filename:
        raise CachePathError(f"cache filename must be a bare name, got {filename!r}")
    if not filename.endswith(".json"):
        raise CachePathError(f"cache filename must end in .json, got {filename!r}")
    root = os.path.realpath(repo_root)
    if not os.path.isdir(root):
        raise CachePathError(f"repo_root is not a directory: {repo_root!r}")
    cache_dir = os.path.join(root, ".cache")
    path = os.path.join(cache_dir, filename)
    if os.path.dirname(os.path.realpath(path)) != cache_dir:
        raise CachePathError(f"cache path escapes {cache_dir!r}: {path!r}")
    return path


# ── Partial-update persistence policy (round 32-F2) ─────────────────────────
# Disk caches whose payload serialises the WHOLE corpus on every save (vulture:
# ~22MB for 930 files ≈ 4.6s of json.dump) must not pay that cost to persist a
# tiny edit: a per-file gate run that rescans 1-3 files costs ~0.1s/file to
# RECOMPUTE but 4.6s to persist.  Policy: persist a partial update only when it
# is a meaningful fraction of the corpus, or the corpus itself is small (small
# corpora are cheap to serialise and usually test fixtures that MUST persist to
# observe cache behaviour across processes).
SAVE_SKIP_MAX_FRACTION = 0.05
SAVE_SKIP_MIN_ENTRIES = 50


def should_persist_partial_update(dirty: int, total_entries: int) -> bool:
    """Whether *dirty* changed entries of *total_entries* are worth saving now.

    Skipping never changes results (fail-open: the next process re-stats,
    misses, and recomputes the stale entries) — it only trades a bounded
    recompute for a full-corpus serialisation.  A corpus smaller than
    :data:`SAVE_SKIP_MIN_ENTRIES` always persists; otherwise persist only when
    the changed fraction exceeds :data:`SAVE_SKIP_MAX_FRACTION` (cold scans:
    100% dirty → persist; per-file gate on a warm cache: a few files → skip).
    """
    if dirty <= 0:
        return False
    if total_entries < SAVE_SKIP_MIN_ENTRIES:
        return True
    return dirty > total_entries * SAVE_SKIP_MAX_FRACTION
