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
    _cache.clear()
    _bytes = 0
    _hits = 0
    _misses = 0


# Same shape as functools.lru_cache's CacheInfo so existing callers/tests can
# read .hits/.misses/.maxsize/.currsize.
CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")


def cache_info() -> CacheInfo:
    """``(hits, misses, maxsize, currsize)`` — lru_cache.cache_info() shape."""
    return CacheInfo(_hits, _misses, _max_entries, len(_cache))
