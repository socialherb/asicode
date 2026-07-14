"""
Tool result cache for safe reuse of read-only tool results.
"""
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _paths_overlap(a: str, b: str) -> bool:
    """True if path ``a`` and ``b`` are the same file, or one is a directory
    containing the other (prefix match on path components, not raw strings —
    ``/x/foo`` must not "overlap" ``/x/foobar``)."""
    if a == b:
        return True
    a_dir = a.rstrip(os.sep) + os.sep
    b_dir = b.rstrip(os.sep) + os.sep
    return b.startswith(a_dir) or a.startswith(b_dir)


@dataclass
class CachedResult:
    """A cached tool result with metadata."""
    result: dict[str, Any]  # Serializable ToolResult fields
    timestamp: float
    ttl: int
    # Absolute file/dir path(s) this result depends on, or None if the scope
    # is unknown/repo-wide (e.g. a search with no path filter). Entries with
    # unknown scope are conservatively dropped by invalidate_paths() since we
    # can't prove they don't depend on whatever was just written.
    paths: Optional[frozenset[str]] = None

class ToolResultCache:
    """TTL-based LRU cache for tool results.

    Safety guarantees:
    - Only read-only tools should be cached (caller responsibility)
    - Cache is fully invalidated on any write tool success
    - TTL ensures stale results are not reused indefinitely
    """

    def __init__(self, max_entries: int = 256, default_ttl: int = 120):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._cache = OrderedDict()  # key -> CachedResult
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Create a stable hash key for tool+args."""
        # Sort keys for consistent hashing. json.dumps never raises here (default
        # falls back to str()), but a non-JSON-native arg (e.g. an object with an
        # identity/address in its repr) makes the *content* unstable across
        # otherwise-identical calls — every such call silently misses the cache.
        # Flag it so that's debuggable instead of just "cache never hits".
        _unstable = False

        def _default(o: Any) -> str:
            nonlocal _unstable
            _unstable = True
            return str(o)

        stable_args = json.dumps(args, sort_keys=True, default=_default)
        if _unstable:
            logger.debug(
                "Tool result cache key for %r includes non-JSON-serializable "
                "arg(s); if its str() varies across calls (e.g. contains an "
                "object id), this call will never cache-hit: %r",
                tool_name, args,
            )
        key_str = f"{tool_name}:{stable_args}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, tool_name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Retrieve cached result if present and not expired."""
        key = self._make_key(tool_name, args)
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            cached = self._cache[key]
            if time.monotonic() - cached.timestamp > cached.ttl:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return cached.result

    def set(
        self, tool_name: str, args: dict[str, Any], result: dict[str, Any],
        ttl: Optional[int] = None, paths: Optional[frozenset[str]] = None,
    ):
        """Store a result in the cache.

        ``paths``: absolute file/dir path(s) this result depends on, if known
        (e.g. the file a ``read_file`` call read). Enables ``invalidate_paths()``
        to drop only overlapping entries instead of a full ``clear()``. Leave
        None when the scope can't be determined (repo-wide search, etc.) — such
        entries are always dropped by ``invalidate_paths()``, matching the
        previous (always-full-clear) behavior for them.
        """
        key = self._make_key(tool_name, args)
        with self._lock:
            if key in self._cache:
                # Update existing entry — move to end (most recently used)
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_entries:
                # Remove oldest entry (LRU)
                self._cache.popitem(last=False)

            self._cache[key] = CachedResult(
                result=result,
                timestamp=time.monotonic(),
                ttl=ttl if ttl is not None else self.default_ttl,
                paths=paths,
            )

    def clear(self):
        """Clear all cached results (call after any write tool success)."""
        with self._lock:
            self._cache.clear()

    def invalidate_paths(self, paths: frozenset[str]) -> int:
        """Drop only cache entries whose recorded scope overlaps ``paths``
        (the absolute file path(s) a write tool just touched).

        Entries with unknown scope (``paths=None`` — e.g. a repo-wide search)
        are also dropped, since we can't prove they don't depend on what was
        written. Falls back to a full ``clear()`` when ``paths`` is empty
        (caller couldn't determine the write's target — e.g. a mutating bash
        command), preserving the previous conservative behavior for that case.

        Returns the number of entries removed.
        """
        if not paths:
            # Atomic len+clear under the lock so the returned count can't be
            # skewed by a concurrent set()/invalidate. NOTE: cannot call
            # self.clear() here — threading.Lock is non-reentrant, and clear()
            # re-acquires the same lock → would deadlock.
            with self._lock:
                before = len(self._cache)
                self._cache.clear()
            return before
        with self._lock:
            stale_keys = [
                key for key, cached in self._cache.items()
                if cached.paths is None
                or any(_paths_overlap(p, wp) for p in cached.paths for wp in paths)
            ]
            for key in stale_keys:
                del self._cache[key]
            return len(stale_keys)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_entries": self.max_entries,
            }
