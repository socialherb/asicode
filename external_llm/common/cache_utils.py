"""SSOT for the FIFO-bounded dict cache helper used across layers.

``_capped_put`` was historically duplicated in ``agent/_shared_utils.py``
(cap default ``_WALK_CACHE_MAX_ENTRIES``) and ``common/repo_files.py`` (cap
default ``_FILE_INDEX_CACHE_MAX``) — identical logic, two homes, two docstrings
that could drift. The ``common`` package is the bottom of the dependency graph
(``common`` must NOT import ``agent`` — design invariant), so the canonical
implementation lives HERE and both original modules re-export it; callers that
imported it from either module keep working.

Why it exists at all: a long-lived process (REPL/orchestrator) visits many
repos, and path-keyed caches (file lists, walk results, git snapshots) grow
unboundedly unless FIFO-bounded. Eviction order is ``dict`` insertion order
(3.7+): the oldest entry is ``next(iter(cache))``, and a re-inserted key is
popped first so it lands at the back (most-recently-used).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bounded entry cap: these path-keyed caches grew unboundedly in a long-lived
# REPL that visited many repos (each holding a full file list). FIFO eviction
# under the GIL stays consistent with the lock-free, single-threaded design;
# the current repo is the newest entry, stale repos are evicted first.
_WALK_CACHE_MAX_ENTRIES: int = 8
# Same discipline for the per-repo file-index cache (one full path list per
# repo, never evicted without a cap).
_FILE_INDEX_CACHE_MAX: int = 8


def _capped_put(cache: dict, key: Any, value: Any, cap: int = _WALK_CACHE_MAX_ENTRIES) -> None:
    """Set ``cache[key] = value`` then evict the least-recently-used entry if over *cap*.

    Note: ``iter(cache)`` / ``next(iter(cache))`` is *not* atomic under
    free-threaded CPython (PEP 703 or concurrent threads that insert/delete).
    ``next()`` can fail with ``RuntimeError`` (dict resized during iteration)
    or ``StopIteration`` (concurrent drain) — the eviction loop catches both
    and bails out, leaving the cache temporarily over cap (harmless).

    ``dict`` insertion order (3.7+) yields the oldest via ``next(iter(cache))``.
    Re-inserting an existing key does NOT refresh its position, so the key is
    popped first: the (re-)inserted entry lands at the back, making it the
    most-recently-used entry and stale repos the correct eviction candidates.
    """
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > cap:
        try:
            _oldest = next(iter(cache))
            cache.pop(_oldest, None)
        except (RuntimeError, StopIteration):
            logger.debug(
                "_capped_put: concurrent dict resize, stopping eviction (cap=%s, size=%s)",
                cap,
                len(cache),
            )
            break  # concurrent dict resize or empty — give up eviction
