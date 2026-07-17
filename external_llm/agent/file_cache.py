"""
File content caching layer for asicode agent.

Caches file contents with modification time validation and LRU eviction.
Reduces filesystem I/O for file reads within the agent pipeline.
"""
import os
import threading
from collections import OrderedDict
from typing import Any, Optional


class FileContentCache:
    """File content caching layer"""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, tuple[str, int, Optional[int], Optional[str]]] = OrderedDict()  # key -> (content, mtime_ns, total_lines, showing)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Optional[str]:
        """Get file content from cache"""
        result = self.get_with_metadata(file_path, start_line, end_line)
        return result[0] if result else None

    def get_with_metadata(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Optional[tuple[str, int, str]]:
        """Get file content with metadata (content, total_lines, showing) from cache"""
        cache_key = self._make_key(file_path, start_line, end_line)

        with self._lock:
            if cache_key in self.cache:
                content, cached_mtime, total_lines, showing = self.cache[cache_key]

                try:
                    current_mtime = os.stat(file_path).st_mtime_ns
                except OSError:
                    # File no longer exists, invalidate cache
                    del self.cache[cache_key]
                    self.misses += 1
                    return None

                if current_mtime != cached_mtime:
                    del self.cache[cache_key]
                    self.misses += 1
                    return None

                self.hits += 1
                self.cache.move_to_end(cache_key)  # LRU: mark as recently used
                return (content, total_lines, showing)

            self.misses += 1
            return None

    def set(self, file_path: str, content: str, start_line: Optional[int] = None, end_line: Optional[int] = None,
            total_lines: Optional[int] = None, showing: Optional[str] = None):
        """Store file content in cache with metadata"""
        cache_key = self._make_key(file_path, start_line, end_line)

        try:
            mtime = os.stat(file_path).st_mtime_ns
        except OSError:
            # Can't cache if file doesn't exist
            return

        with self._lock:
            # LRU cache size management — O(1) eviction via OrderedDict.
            # Only evict when adding a NEW key; updating an existing key does not
            # grow the cache, and evicting in that case would discard a
            # recently-fetched entry for no reason.
            is_new = cache_key not in self.cache
            if is_new and len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # evict least-recently-used (front)

            self.cache[cache_key] = (content, mtime, total_lines, showing)
            if not is_new:
                # Refresh LRU position so an update counts as recent use.
                self.cache.move_to_end(cache_key)

    def invalidate(self, file_path: str):
        """Invalidate all cache entries for a file"""
        with self._lock:
            keys_to_remove = [key for key in list(self.cache.keys())
                              if key == file_path or key.startswith(file_path + ":")]
            for key in keys_to_remove:
                del self.cache[key]

    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

    def _make_key(self, file_path: str, start_line: Optional[int], end_line: Optional[int]) -> str:
        """Generate cache key"""
        key_parts = [file_path]
        if start_line is not None:
            key_parts.append(f"start:{start_line}")
        if end_line is not None:
            key_parts.append(f"end:{end_line}")
        return ":".join(str(p) for p in key_parts)


# Global cache instance
_global_file_cache: Optional[FileContentCache] = None
_file_cache_init_lock = threading.Lock()


def get_global_file_cache() -> FileContentCache:
    """Get or create global file content cache"""
    global _global_file_cache
    if _global_file_cache is None:
        with _file_cache_init_lock:
            if _global_file_cache is None:
                _global_file_cache = FileContentCache()
    return _global_file_cache


def reset_global_file_cache(max_size: int = 1000):
    """Reset global file content cache

    Acquires the init lock to stay consistent with the double-checked locking in
    get_global_file_cache(). Without it, a concurrent get_global_file_cache() on
    another thread could observe a half-constructed FileContentCache or witness
    the global being swapped mid-read.
    """
    global _global_file_cache
    with _file_cache_init_lock:
        _global_file_cache = FileContentCache(max_size)
