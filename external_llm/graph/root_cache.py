"""Root-partitioned process-wide cache with fair admission (2026-08-12).

Why: the agent's three process-wide per-file caches (RepositoryGraph
``_extract_cache`` / ``_names_cache``, CallGraphIndexer ``_file_cache``)
were flat dicts with a global first-N admission cap.  In the webapp ONE
long-lived process serves MANY repos: once the first repo(s) filled the
cap, every later repo was refused for the rest of the process lifetime —
each of its builds re-parsed everything and re-loaded its multi-MB
snapshot (multi-repo starvation, 2026-08-12).

The fix keeps the flat dict API (keys are ``(root, subkey)`` tuples, as
before) but stores entries nested per root so each root's count is derived
(``len`` of its inner dict).  Admission policy:

* quota = cap // active-roots — every active root's guaranteed share;
* while the cache has room (total < cap), any root may admit — overflow
  sharing lets one root exceed its quota while nobody else needs it;
* when the cache is full, a new entry claims a slot from the
  most-over-quota OTHER root (evicting that root's OLDEST entry): a root
  below quota is PROVABLY always able to claim (with total == cap and
  root_count < quota, some other root must exceed quota), while at/over
  quota roots are refused unless a hoarder exists;
* single-root processes degenerate to the old behavior exactly:
  quota == cap and no other root -> refuse beyond cap (no clear-thrash,
  A242 contract preserved).

Registry bound: at most ``max_roots`` roots are tracked.  When a NEW root
arrives beyond the bound, the coldest (least recently admitted) root is
dropped ENTIRELY — quotas stay monotonic for the active set instead of
decaying as repos come and go (a non-monotonic guarantee is why an
unbounded registry was rejected).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any


class RootCache:
    """Flat-API, root-partitioned admission-controlled cache.

    Keys are ``(root, subkey)`` tuples — ``root`` is the normalized repo
    root (payloads carry root-relative fields) and ``subkey`` the absolute
    file path.  The dict surface is FLATTENED: ``__len__`` is the total
    entry count and iteration/``keys``/``items``/``dict(cache)`` yield
    ``(root, subkey)`` keys exactly like the old flat dict, so existing
    direct-access tests keep working.
    """

    def __init__(self, cap: int, max_roots: int = 4) -> None:
        self.cap = cap
        self.max_roots = max_roots
        self._roots: dict[str, dict[Any, Any]] = {}
        self._last_seen: dict[str, int] = {}
        self._seq = 0

    # ── flat dict surface ─────────────────────────────────────────────

    def __getitem__(self, key: tuple[str, Any]) -> Any:
        return self._roots[key[0]][key[1]]

    def __setitem__(self, key: tuple[str, Any], value: Any) -> None:
        root, sub = key
        inner = self._ensure_root(root)
        inner[sub] = value
        self._touch(root)

    def __delitem__(self, key: tuple[str, Any]) -> None:
        root, sub = key
        inner = self._roots[root]
        del inner[sub]
        if not inner:
            del self._roots[root]
            self._last_seen.pop(root, None)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        root, sub = key
        inner = self._roots.get(root)
        return inner is not None and sub in inner

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for root, inner in self._roots.items():
            for sub in inner:
                yield (root, sub)

    def __len__(self) -> int:
        return sum(len(inner) for inner in self._roots.values())

    def get(self, key: tuple[str, Any], default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: tuple[str, Any], default: Any = None) -> Any:
        try:
            value = self[key]
        except KeyError:
            return default
        del self[key]
        return value

    def keys(self) -> list[tuple[str, Any]]:
        return list(self)

    def items(self) -> Iterator[tuple[tuple[str, Any], Any]]:
        for key in self:
            yield key, self[key]

    def clear(self) -> None:
        self._roots.clear()
        self._last_seen.clear()

    def update(self, other: Any) -> None:
        """Restore/merge a (flattened) mapping, like ``dict.update``."""
        for key, value in dict(other).items():
            self[key] = value

    # ── admission ──────────────────────────────────────────────────────

    def admit(self, key: tuple[str, Any], value: Any) -> bool:
        """Insert under fair admission; True when the entry is cached.

        See the module docstring for the policy.  A refused entry is simply
        not cached — callers fall back to per-build bookkeeping (e.g. the
        merge-preserving snapshot persists refused extractions).
        """
        root, sub = key
        inner = self._ensure_root(root)
        if sub in inner:
            inner[sub] = value
            self._touch(root)
            return True
        if len(self) >= self.cap:
            victim = self._most_over_quota(exclude=root)
            if victim is None:
                return False
            self._evict_one(victim)
        inner[sub] = value
        self._touch(root)
        return True

    def sweep_dead(self) -> int:
        """Delete entries whose source no longer exists.

        Entries are stored per root with the ABSOLUTE path as the inner key,
        so the sweep stats the inner key directly; roots left empty are
        dropped from the registry.  Returns the number of entries removed.
        """
        removed = 0
        for root in list(self._roots):
            inner = self._roots[root]
            for key in list(inner):
                try:
                    os.stat(key)
                except OSError:
                    del inner[key]
                    removed += 1
            if not inner:
                del self._roots[root]
                self._last_seen.pop(root, None)
        return removed

    def count(self, root: str) -> int:
        """Live entry count for one root."""
        inner = self._roots.get(root)
        return len(inner) if inner is not None else 0

    def quota(self) -> int:
        """The guaranteed per-root share: ``cap // active roots``.

        Root-independent (a global fair share) — callers pair it with
        :meth:`count` per root for logging.
        """
        return max(1, self.cap // max(1, len(self._roots)))

    # ── internals ──────────────────────────────────────────────────────

    def _ensure_root(self, root: str) -> dict:
        if root not in self._roots and len(self._roots) >= self.max_roots:
            self._drop_coldest(exclude=root)
        return self._roots.setdefault(root, {})

    def _touch(self, root: str) -> None:
        self._seq += 1
        self._last_seen[root] = self._seq

    def _drop_coldest(self, exclude: str) -> None:
        cold = min(
            (r for r in self._roots if r != exclude),
            key=lambda r: self._last_seen.get(r, 0),
            default=None,
        )
        if cold is not None:
            del self._roots[cold]
            self._last_seen.pop(cold, None)

    def _most_over_quota(self, exclude: str) -> str | None:
        """Root (≠ exclude) with the most entries above quota, or None."""
        quota = self.quota()
        best: str | None = None
        best_over = 0
        for root, inner in self._roots.items():
            if root == exclude:
                continue
            over = len(inner) - quota
            if over > best_over:
                best_over = over
                best = root
        return best

    def _evict_one(self, root: str) -> None:
        """Evict the OLDEST entry of one root (insertion order)."""
        inner = self._roots[root]
        inner.pop(next(iter(inner)))
        if not inner:
            del self._roots[root]
            self._last_seen.pop(root, None)
