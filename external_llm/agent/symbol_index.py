"""Repo-wide symbol index for import-vs-create resolution (import_vs_create_resolution).

Builds a lightweight index of top-level Python symbol definitions across
the repository. Used by import_vs_create_resolution to decide whether a missing symbol should
be imported from an existing file or created from scratch.

Includes mtime-based caching and cheap local resolution to avoid
unnecessary repo-wide scans.
"""
from __future__ import annotations

import ast
import builtins
import os
import time
from dataclasses import dataclass
from typing import Optional

from ..languages import LanguageId

# Within this many seconds of the last scan, the index is served from cache
# without touching the filesystem at all. Back-to-back agent turns rarely
# change the repo file set, so this avoids a ~25ms walk per call on the
# hot-path (analysis_tools, executor_verification). Set conservatively low so
# genuinely new files surface quickly.
_INDEX_TTL_SECONDS = 2.0

_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", "runs",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
    "debug_dump", ".asicode", "worktrees",
}


@dataclass(frozen=True)
class SymbolLocation:
    name: str
    file_path: str  # relative to repo_root
    kind: str       # "class" | "function" | "async_function"


# ── In-memory cache ──────────────────────────────────────────────────────
# Internally we store a *file-keyed* index: {rel_path: [SymbolLocation, ...]}.
# This makes per-file add/change/remove O(1) for incremental rebuilds — only
# the changed files are re-parsed instead of the whole repo. The name-keyed
# index that consumers expect ({symbol_name: [SymbolLocation, ...]}) is derived
# via _name_index_from_file_index and cached alongside the file index so the
# TTL fast-path (the agent-loop hot-path) returns it in O(1) instead of
# re-deriving it every turn.
#
# Key: repo_root → (file_index, name_index, {rel_path: mtime}, monotonic_timestamp)
_INDEX_CACHE: dict[str, tuple[dict[str, list[SymbolLocation]], dict[str, list[SymbolLocation]], dict[str, float], float]] = {}

# Bounded entry cap (same pattern as _shared_utils._capped_put /
# _WALK_CACHE_MAX_ENTRIES): this path-keyed cache grew unboundedly in a long-
# lived REPL visiting many repos (each holding a full per-file index). LRU
# eviction under the GIL stays consistent with the lock-free, single-threaded
# design; the most-recently-touched repo is never the eviction candidate.
_INDEX_CACHE_MAX_ENTRIES: int = 8


def _capped_index_put(cache: dict, key, value) -> None:
    """Set ``cache[key] = value`` (LRU) then evict the least-recently-used entry.

    Pure-dict, GIL-atomic — no lock needed (consistent with the lock-free cache
    family). ``pop``-then-``set`` makes this a TRUE LRU: re-inserting an
    existing key (e.g. a TTL refresh of the current repo) MOVES it to
    most-recent, so the actively-used repo is never the eviction candidate.
    A bare ``cache[key] = value`` would preserve the key's ORIGINAL insertion
    position in CPython 3.7+, so under FIFO the hot repo that was first visited
    would be evicted first — a silent hot-cache miss on every repo switch.
    """
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > _INDEX_CACHE_MAX_ENTRIES:
        _oldest = next(iter(cache))
        cache.pop(_oldest, None)


def _scan_file(abs_path: str, rel_path: str) -> list[SymbolLocation]:
    """Parse a single .py file and return its top-level symbols."""
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return []

    result: list[SymbolLocation] = []
    for node in tree.body:
        kind: Optional[str] = None
        name: Optional[str] = None
        if isinstance(node, ast.ClassDef):
            kind = "class"
            name = node.name
        elif isinstance(node, ast.FunctionDef):
            kind = "function"
            name = node.name
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
            name = node.name
        elif isinstance(node, ast.Assign):
            # Type aliases and module-level constant assignments: e.g. MyType = Tuple[...]
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    result.append(SymbolLocation(name=tgt.id, file_path=rel_path, kind="constant"))
            continue
        elif isinstance(node, ast.AnnAssign):
            # Annotated assignments: e.g. MyVar: Dict[...] = {...}
            if isinstance(node.target, ast.Name):
                result.append(SymbolLocation(name=node.target.id, file_path=rel_path, kind="constant"))
            continue
        if kind and name:
            result.append(SymbolLocation(name=name, file_path=rel_path, kind=kind))
    return result


def _collect_mtimes(repo_root: str) -> dict[str, float]:
    """Walk the repo once and return {rel_path: mtime} for all .py files.

    Determinism: directory traversal is sorted (see build_repo_symbol_index
    docstring) so the resulting dict is reproducible across machines.
    """
    mtimes: dict[str, float] = {}
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for fn in sorted(files):
            if LanguageId.from_path(fn) is not LanguageId.PYTHON:
                continue
            abs_path = os.path.join(root, fn)
            rel_path = os.path.relpath(abs_path, repo_root)
            try:
                mtimes[rel_path] = os.path.getmtime(abs_path)
            except OSError:
                continue
    return mtimes


def _rebuild_file_index(
    repo_root: str, mtimes: dict[str, float]
) -> dict[str, list[SymbolLocation]]:
    """Re-parse the given file set into a fresh *file-keyed* index.

    Takes the mtimes dict (already gathered by a single walk) so the caller
    doesn't pay for a second traversal. Re-reads mtimes defensively to handle
    files deleted between the walk and the parse.

    Returns {rel_path: [SymbolLocation, ...]}.
    """
    file_index: dict[str, list[SymbolLocation]] = {}
    for rel_path in mtimes:
        abs_path = os.path.join(repo_root, rel_path)
        # _scan_file already returns top-level symbols for this file; keep the
        # list as-is (file-index does not need the per-name sort, that happens
        # at name-index derivation time).
        file_index[rel_path] = _scan_file(abs_path, rel_path)
    return file_index


def _name_index_from_file_index(
    file_index: dict[str, list[SymbolLocation]],
) -> dict[str, list[SymbolLocation]]:
    """Derive {symbol_name: [SymbolLocation, ...]} from a file-keyed index.

    Determinism: each symbol's candidate locations are sorted by (file_path,
    kind) so consumers indexing candidates[0] get a stable choice regardless
    of file traversal order. (file_path, kind) is a total order —
    SymbolLocation has no line field.
    """
    name_index: dict[str, list[SymbolLocation]] = {}
    for locs in file_index.values():
        for loc in locs:
            name_index.setdefault(loc.name, []).append(loc)
    for _locs in name_index.values():
        _locs.sort(key=lambda _item_: (_item_.file_path, _item_.kind))
    return name_index


def _apply_incremental(
    repo_root: str,
    old_file_index: dict[str, list[SymbolLocation]],
    old_mtimes: dict[str, float],
    cur_mtimes: dict[str, float],
) -> tuple[dict[str, list[SymbolLocation]], dict[str, float]]:
    """Incrementally update a file-keyed index from the mtime diff.

    Only files whose mtime changed, were added, or were removed since the last
    scan are touched: added/changed files get a single ``_scan_file`` call,
    removed files drop their index entry. Untouched files keep their parsed
    symbols verbatim — no re-parse. This turns a ~2s whole-repo reparse into a
    ~1ms single-file re-scan on the common "edit one file" hot-path.

    Returns (new_file_index, cur_mtimes).
    """
    # Mutate a copy so the cached old_file_index alias is not poisoned if a
    # later caller bails out mid-update.
    new_file_index = dict(old_file_index)

    old_keys = set(old_mtimes)
    cur_keys = set(cur_mtimes)

    # Removed files: drop their entries.
    for rel_path in old_keys - cur_keys:
        new_file_index.pop(rel_path, None)

    # Added or changed files: re-scan that single file.
    for rel_path in cur_keys:
        if rel_path not in old_keys or old_mtimes[rel_path] != cur_mtimes[rel_path]:
            abs_path = os.path.join(repo_root, rel_path)
            new_file_index[rel_path] = _scan_file(abs_path, rel_path)

    return new_file_index, cur_mtimes


def build_repo_symbol_index(repo_root: str) -> dict[str, list[SymbolLocation]]:
    """Scan repo for top-level class/function definitions (cached).

    Three-layer cache to keep the agent hot-path cheap:
      1. **TTL layer** — within `_INDEX_TTL_SECONDS` of the last scan, return the
         cached index without *any* filesystem walk. The agent loop calls this
         on every turn (analysis_tools, executor_verification); a repo-wide
         walk costs ~25ms even on a cache hit and is pointless work when the
         filesystem hasn't had time to change between back-to-back turns.
      2. **mtime layer** — once the TTL elapses, do *one* walk that simultaneously
         (a) collects the current .py file set, (b) reads each mtime, and
         (c) compares against the cached file-set/mtimes. If nothing changed,
         refresh the TTL timestamp and reuse the index. Fusing stale-detection
         and rebuild into a single walk also eliminates the TOCTOU window where
         a file added between the "is it stale?" walk and the "rebuild" walk
         would be missed.
      3. **Incremental rebuild** — when something did change, re-parse *only*
         the changed/added files and drop removed ones, reusing every untouched
         file's parsed symbols. A single-file edit (the common case) costs one
         ``_scan_file`` call (~1ms) instead of a whole-repo reparse (~2s on this
         repo). A newly created .py file is absent from old_mtimes → caught by
         the set comparison (not mtime), so decide_import_vs_create won't create
         a duplicate definition.

    Internally caches a *file-keyed* index ({rel_path: [SymbolLocation, ...]})
    so per-file add/change/remove is O(1); the name-keyed index consumers
    expect ({symbol_name: [SymbolLocation, ...]}) is derived ONCE per file-index
    change and cached alongside it, so the TTL fast-path returns it in O(1)
    instead of re-deriving every turn.

    Returns {symbol_name: [SymbolLocation, ...]}.
    """
    cached = _INDEX_CACHE.get(repo_root)
    if cached is not None:
        old_file_index, old_name_index, old_mtimes, old_ts = cached
        # TTL fast-path: skip the walk entirely on back-to-back turns — and
        # return the CACHED derived name index (O(1)) rather than re-deriving it.
        if time.monotonic() - old_ts < _INDEX_TTL_SECONDS:
            return old_name_index

        # TTL expired: one fused walk to gather the file set + mtimes.
        cur_mtimes = _collect_mtimes(repo_root)

        # Stale iff the tracked file set differs OR any mtime changed.
        # A newly created .py file is absent from old_mtimes → caught by the
        # set comparison (not mtime), so decide_import_vs_create won't create a
        # duplicate definition.
        if cur_mtimes == old_mtimes:
            # Nothing changed: refresh the TTL timestamp and reuse BOTH indexes.
            _capped_index_put(_INDEX_CACHE, repo_root, (old_file_index, old_name_index, old_mtimes, time.monotonic()))
            return old_name_index

        # Something changed: incremental rebuild — only changed files re-parse.
        new_file_index, mtimes = _apply_incremental(
            repo_root, old_file_index, old_mtimes, cur_mtimes
        )
        new_name_index = _name_index_from_file_index(new_file_index)
        _capped_index_put(_INDEX_CACHE, repo_root, (new_file_index, new_name_index, mtimes, time.monotonic()))
        return new_name_index

    # Cold cache: full walk + parse.
    mtimes = _collect_mtimes(repo_root)
    file_index = _rebuild_file_index(repo_root, mtimes)
    name_index = _name_index_from_file_index(file_index)
    _capped_index_put(_INDEX_CACHE, repo_root, (file_index, name_index, mtimes, time.monotonic()))
    return name_index


# ── Cheap local resolution ───────────────────────────────────────────────

def resolve_symbol_locally(
    symbol_name: str,
    current_file_abs: str,
    candidate_files: list[str],
    repo_root: str,
) -> Optional[dict]:
    """Try to resolve a missing symbol from local/plan-touched files first.

    Returns a decision dict (same shape as decide_import_vs_create) if found,
    or None if not found locally.
    """
    current_rel = os.path.relpath(current_file_abs, repo_root) if os.path.isabs(current_file_abs) else current_file_abs

    for fp in candidate_files:
        abs_fp = os.path.join(repo_root, fp) if not os.path.isabs(fp) else fp
        rel_fp = os.path.relpath(abs_fp, repo_root) if os.path.isabs(abs_fp) else fp

        if LanguageId.from_path(abs_fp) is not LanguageId.PYTHON or not os.path.isfile(abs_fp):
            continue

        for loc in _scan_file(abs_fp, rel_fp):
            if loc.name == symbol_name:
                if rel_fp == current_rel:
                    return {
                        "action": "noop",
                        "symbol": symbol_name,
                        "target_file": current_rel,
                        "reason": "already_defined_here",
                    }
                else:
                    return {
                        "action": "import",
                        "symbol": symbol_name,
                        "target_file": current_rel,
                        "source_file": rel_fp,
                        "kind": loc.kind,
                        "reason": "definition_in_plan_file",
                    }

    return None  # Not found locally → caller should try repo-wide


def decide_import_vs_create(
    symbol_name: str,
    current_file: str,
    symbol_index: dict[str, list[SymbolLocation]],
) -> dict:
    """Decide whether to import or create a missing symbol.

    Returns dict with keys: action, symbol, target_file, source_file (if import),
    kind, reason.
    """
    # Python builtins (exceptions, types, functions) never need import
    if hasattr(builtins, symbol_name):
        return {
            "action": "noop",
            "symbol": symbol_name,
            "target_file": current_file,
            "reason": "python_builtin",
        }

    candidates = symbol_index.get(symbol_name, [])

    # Already defined in current file → nothing to do
    for c in candidates:
        if c.file_path == current_file:
            return {
                "action": "noop",
                "symbol": symbol_name,
                "target_file": current_file,
                "reason": "already_defined_here",
            }

    # Defined elsewhere → import
    if candidates:
        # Prefer non-test files over test files
        non_test = [c for c in candidates
                    if "/tests/" not in c.file_path
                    and not c.file_path.startswith("tests/")
                    and "/test_" not in c.file_path]
        best = non_test[0] if non_test else candidates[0]
        return {
            "action": "import",
            "symbol": symbol_name,
            "target_file": current_file,
            "source_file": best.file_path,
            "kind": best.kind,
            "reason": "definition_exists_elsewhere",
        }

    # Not found anywhere → create
    return {
        "action": "create",
        "symbol": symbol_name,
        "target_file": current_file,
        "reason": "definition_missing_repo_wide",
    }
