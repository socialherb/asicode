"""
Run-scoped cache for graph enrichment and advisor query results.

This cache lives for the duration of a single run/session and prevents
redundant graph computations. It is NOT persisted across sessions.

Design follows the existing ToolResultCache pattern (SHA256 keys, TTL, LRU).
"""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedGraphResult:
    """Wrapper for a cached graph computation result."""
    value: Any
    timestamp: float
    cache_key: str
    category: str  # "enrichment", "repair_hints", "safety_issues", etc.
    ttl: int  # seconds; 0 means never expire (legacy behavior opt-out)


class RunScopedGraphCache:
    """
    Run/session-scoped cache for graph-derived results.

    Prevents redundant graph enrichment and advisor queries within
    a single planning/execution run. Supports dirty-file invalidation.

    NOT persisted across sessions.
    """

    def __init__(self, max_entries: int = 256, default_ttl: int = 120):
        self._cache: OrderedDict[str, CachedGraphResult] = OrderedDict()
        self._max_entries = max_entries
        self.default_ttl = default_ttl  # seconds; 0 disables TTL-based expiry
        self._dirty_files: set[str] = set()
        self._generation: int = 0  # incremented on any invalidation
        self._lock = threading.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._expired = 0  # entries dropped by TTL in get()
        self._invalidations = 0
        self._evictions = 0

    # ── Key generation ──────────────────────────────────────────────────────

    @staticmethod
    def make_key(category: str, **kwargs) -> str:
        """Generate deterministic cache key from category + keyword args."""
        # Sort kwargs for determinism, convert to JSON-safe format
        key_parts: dict[str, Any] = {"category": category}
        for k, v in sorted(kwargs.items()):
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                key_parts[k] = json.dumps(sorted(str(x) for x in v))
            elif isinstance(v, dict):
                key_parts[k] = json.dumps(v, sort_keys=True, default=str)
            else:
                key_parts[k] = str(v)
        raw = json.dumps(key_parts, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ── Core operations ─────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached result, or None on miss.

        Honors TTL: an entry whose age exceeds its ttl is evicted and
        treated as a miss (mirrors ToolResultCache.get semantics).
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL expiry check (ttl == 0 means never expire)
            if entry.ttl > 0 and (time.monotonic() - entry.timestamp) > entry.ttl:
                del self._cache[key]
                self._expired += 1
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any, category: str = "unknown", ttl: Optional[int] = None) -> None:
        """Store result in cache.

        Args:
            ttl: override default_ttl for this entry. Pass 0 to disable expiry
                for a long-lived entry. None falls back to self.default_ttl.
        """
        effective_ttl = self.default_ttl if ttl is None else ttl
        with self._lock:
            is_new = key not in self._cache
            self._cache[key] = CachedGraphResult(
                value=value,
                timestamp=time.monotonic(),
                cache_key=key,
                category=category,
                ttl=effective_ttl,
            )
            # Move to end AFTER assignment so both new and updated keys land at
            # the MRU end (mirrors ToolResultCache.set / FileContentCache.set;
            # fixes the old code that moved-then-overwrote, making the move a
            # no-op for existing keys).
            self._cache.move_to_end(key)
            # Evict AFTER inserting: _evict_if_needed() uses a strict-greater
            # (`> max_entries`) guard, so it can only trip once the new entry
            # is already counted.
            if is_new:
                self._evict_if_needed()

    def has(self, key: str) -> bool:
        """Check if key exists in cache and has not expired (honors TTL).

        Mirrors get() TTL semantics: expired entries are evicted and counted
        as misses, preventing the inconsistency where has()→True but
        get()→None.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return False

            # TTL expiry check (ttl == 0 means never expire)
            if entry.ttl > 0 and (time.monotonic() - entry.timestamp) > entry.ttl:
                del self._cache[key]
                self._expired += 1
                self._misses += 1
                return False

            return True

    # ── Invalidation ────────────────────────────────────────────────────────

    def invalidate_for_files(self, file_paths: list[str]) -> int:
        """
        Invalidate cache entries related to the given files.

        Since we can't track exact file→key mappings efficiently,
        we use a generation-based approach: mark files as dirty
        and increment generation. Callers should include generation
        in their cache keys to get automatic invalidation.

        Returns number of directly evicted entries (if any match).
        """
        if not file_paths:
            return 0

        with self._lock:
            self._dirty_files.update(file_paths)
            self._generation += 1
            evicted = 0

            # Also scan and remove entries whose keys contain dirty file paths
            # (best-effort; generation-based invalidation is the primary mechanism)
            keys_to_remove = []
            for key, entry in self._cache.items():
                # Check if any dirty file appears in the cached value
                val = entry.value
                if isinstance(val, dict):
                    # Check common fields that contain file paths
                    for field_name in ("primary_files", "impact_files", "files", "target_files"):
                        cached_files = val.get(field_name, [])
                        if isinstance(cached_files, list):
                            for f in file_paths:
                                if f in cached_files:
                                    keys_to_remove.append(key)
                                    break
                        if key in keys_to_remove:
                            break

            for key in keys_to_remove:
                del self._cache[key]
                evicted += 1

            self._invalidations += len(file_paths)
            _gen_captured = self._generation

        if evicted > 0:
            logger.debug(
                "Graph cache: invalidated %d entries for %d dirty files (gen=%d)",
                evicted, len(file_paths), _gen_captured,
            )

        return evicted

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._dirty_files.clear()
            self._generation += 1

    @property
    def generation(self) -> int:
        """Current cache generation (incremented on invalidation)."""
        with self._lock:
            return self._generation

    @property
    def dirty_files(self) -> set[str]:
        """Currently known dirty files."""
        with self._lock:
            return self._dirty_files.copy()


    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "expired": self._expired,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "invalidations": self._invalidations,
                "evictions": self._evictions,
                "generation": self._generation,
                "dirty_file_count": len(self._dirty_files),
            }

    def get_debug_summary(self) -> dict[str, Any]:
        """Concise debug summary for metadata."""
        stats = self.get_stats()
        return {
            "cache_size": stats["size"],
            "hit_rate": round(stats["hit_rate"], 3),
            "generation": stats["generation"],
            "dirty_files": stats["dirty_file_count"],
        }


    # ── Internal ────────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
            self._evictions += 1


# ── Module-level singleton ───────────────────────────────────────────────────

_global_graph_cache: Optional[RunScopedGraphCache] = None
_graph_cache_init_lock = threading.Lock()


def get_global_graph_cache() -> RunScopedGraphCache:
    """Get or create the global run-scoped graph cache."""
    global _global_graph_cache
    if _global_graph_cache is None:
        with _graph_cache_init_lock:
            if _global_graph_cache is None:
                _global_graph_cache = RunScopedGraphCache()
    return _global_graph_cache


def reset_global_graph_cache() -> None:
    """Reset the global graph cache (e.g., at start of new run)."""
    global _global_graph_cache
    with _graph_cache_init_lock:
        if _global_graph_cache is not None:
            _global_graph_cache.clear()
        _global_graph_cache = RunScopedGraphCache()
