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
file list (scanner A over all N files, then scanner B over all N, …), so with a
fixed cache smaller than N the early entries from scanner A are evicted before
scanner B reaches them — every later scanner re-parses from scratch and the
shared cache delivers nothing.  ``ensure_capacity(n)`` grows the cache (up to
``_MAX_CACHE_SIZE``) so a known file set fits; the scanner registry calls it
once per ``run()`` with ``len(file_paths)``.
"""

from __future__ import annotations

import ast
import os
from functools import lru_cache
from typing import Optional

# Default capacity for ad-hoc callers that never invoke ``ensure_capacity``.
_DEFAULT_CACHE_SIZE = 256
# Hard ceiling — ASTs are heavy; never let the cache grow without bound when a
# caller passes a huge (or attacker-controlled) file count.
_MAX_CACHE_SIZE = 4096
# Headroom so a few files touched outside the declared set don't evict members
# of the working set.
_CAPACITY_HEADROOM = 16


def _stat_key(abs_path: str) -> Optional[tuple]:
    try:
        st = os.stat(abs_path)
    except OSError:
        return None
    return (abs_path, st.st_mtime_ns, st.st_size)


def _read_impl(abs_path: str, mtime_ns: int, size: int) -> Optional[str]:
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _ast_impl(abs_path: str, mtime_ns: int, size: int) -> Optional[ast.Module]:
    src = _read_cached(abs_path, mtime_ns, size)
    if src is None:
        return None
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


# Module-global cache wrappers — rebuilt by ``ensure_capacity`` when the working
# set outgrows the current maxsize.  ``_ast_impl`` resolves ``_read_cached`` at
# call time, so it always uses the live wrapper after a rebuild.
_read_cached = lru_cache(maxsize=_DEFAULT_CACHE_SIZE)(_read_impl)
_ast_cached = lru_cache(maxsize=_DEFAULT_CACHE_SIZE)(_ast_impl)


def ensure_capacity(n: int) -> None:
    """Grow the cache so a working set of *n* files fits at once.

    No-op when the current cache already holds *n* (+ headroom) entries, so
    repeated calls across scanners over the same file set rebuild at most once
    and preserve the entries the first scanner populated.  Shrinking is never
    performed — a smaller follow-up set keeps the larger cache.
    """
    global _read_cached, _ast_cached
    target = min(n + _CAPACITY_HEADROOM, _MAX_CACHE_SIZE)
    current = _ast_cached.cache_info().maxsize or 0
    if target <= current:
        return
    _read_cached = lru_cache(maxsize=target)(_read_impl)
    _ast_cached = lru_cache(maxsize=target)(_ast_impl)


def read_source(abs_path: str) -> Optional[str]:
    """Cached file read.  Returns None when the file is missing/unreadable."""
    key = _stat_key(abs_path)
    if key is None:
        return None
    return _read_cached(*key)


def parse_ast(abs_path: str) -> Optional[ast.Module]:
    """Cached ``ast.parse``.  Returns None on read failure or SyntaxError."""
    key = _stat_key(abs_path)
    if key is None:
        return None
    return _ast_cached(*key)


def clear() -> None:
    """Drop all cached entries (tests / long-lived processes)."""
    _read_cached.cache_clear()
    _ast_cached.cache_clear()
