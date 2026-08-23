"""Shared SSOT for listing repo files via ``git ls-files`` (TTL-cached).

Every consumer that needs "the repo's file set" — ``write_tools._repo_file_index``
(did-you-mean suggestions + glob), ``symbol_index._collect_file_signatures`` (import-vs-
create resolution), the editor-lane runtime gate / planner / executor file scans —
must enumerate the *same* set. Sourcing from a single ``git ls-files -z`` call
guarantees three properties a separate ``os.walk`` + hardcoded skip-set cannot:

  * ``.gitignore`` is respected automatically — no skip-set drift (the legacy
    ``_SKIP_DIRS`` had ``.venv`` but not ``venv``/``vendor/``/``third_party/``);
  * non-ASCII (Korean/CJK) paths survive unmangled (``-z`` — porcelain output
    C-quotes them by default, which would then fail membership tests);
  * vendored / generated copies (``vendor/``, ``*_pb2.py``) do NOT leak into
    the symbol index — a leaked symbol would flip a correct ``"create"`` into
    a wrong ``"import"`` and risk a DUPLICATE DEFINITION.

``cached_repo_file_list`` wraps the raw listing in a per-repo TTL cache (60 s,
cap 8, generation counter) so hot paths (runtime gate after every operation,
the agent-loop quality gate, planner scans) stop paying a full ``os.walk``
(measured ~185 ms / 62k files in this repo) or even a ``git ls-files``
subprocess (~20-37 ms) on every call. Writes invalidate the cache through two
funnels: the write TOOLS call ``invalidate_repo_file_index`` via
``tool_registry._invalidate_cache_after_write``, and *every* other write lands
in ``common.atomic_io`` which calls :func:`invalidate_for_written_path` — so a
file written this turn is visible to all cached consumers immediately.

``symbol_index._collect_file_signatures`` deliberately uses the RAW
:func:`git_list_repo_files`: it is the symbol index's change-detection source,
and a TTL-stale listing would hide newly created files until expiry
(duplicate-definition risk). Callers that need freshness must not use the cache.

Constraint: ``common`` must NOT import ``agent`` (design-insight invariant).
This module is pure stdlib (``os`` + ``subprocess``), so it sits at the bottom
of the dependency graph and is safe for any layer to import.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from .cache_utils import _FILE_INDEX_CACHE_MAX, _capped_put
from .walk_policy import _walk_should_skip_dir

logger = logging.getLogger(__name__)


# ── TTL cache (single per-repo file-list cache for the whole codebase) ────
_FILE_INDEX_TTL = 60.0
# Bounded (cap 8) — same discipline as the sibling per-repo caches
# ``_PY_WALK_CACHE`` / ``_TS_WALK_CACHE``. Without it a long-lived orchestrator
# visiting many repos grew the dict unboundedly (one full path list per repo,
# never evicted).
_FILE_INDEX_CACHE: dict[str, tuple[float, list[str]]] = {}
_FILE_INDEX_GEN: int = 0  # incremented on invalidation; stale writes skip cache store
# os.walk fallback skip-set (no .gitignore awareness — used only for non-git
# trees, which have no vendored copies to exclude in the first place).
_FILE_INDEX_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".tox",
        ".eggs",
        ".cache",
        ".idea",
        ".vscode",
        "site-packages",
    }
)


def canonical_repo_key(repo_root: str) -> str:
    """Canonical cache key for *repo_root* — shared by the file-index cache and
    the git-snapshot cache (``agent_context_manager.get_git_snapshot``).

    Callers reach a cache by different spellings of the same directory —
    ``ToolRegistry.repo_root`` is resolved in ``__init__`` while
    ``_effective_repo_root`` may carry an unresolved staging override, and the
    git-snapshot cache receives raw request strings from ``service.py`` — and
    on macOS ``/var`` vs ``/private/var`` made one repo occupy two entries of
    an 8-entry cache. Worse, the invalidator could then clear a key the reader
    never used. One canonical key makes both agree.
    """
    try:
        return str(Path(repo_root).resolve())
    except OSError:
        return str(repo_root)


def git_list_repo_files(repo_root: str) -> list[str] | None:
    """Return sorted repo-relative paths via ``git ls-files`` (NUL-separated).

    Uses ``-z`` (REQUIRED — porcelain output C-quotes non-ASCII paths like
    Korean, which would then fail membership tests downstream) + ``--cached``
    (tracked) + ``--others --exclude-standard`` (untracked but NOT gitignored)
    = every file visible in the work tree, with ``.gitignore`` respected
    automatically — so a hardcoded skip-set is not needed on the git path.

    Returns ``None`` when git is unavailable, the path is not a git checkout,
    or the call fails — callers fall back to ``os.walk`` in that case. A
    ``None`` return (NOT an empty list) is what distinguishes "git unusable,
    please walk" from "git OK, repo has zero files".
    """
    if not os.path.exists(os.path.join(repo_root, ".git")):
        return None
    try:
        r = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if r.returncode != 0:
            return None
        out = r.stdout.decode("utf-8", "replace")
        paths = [_p for _p in out.split("\0") if _p]
        paths.sort()
    except (subprocess.TimeoutExpired, OSError):  # git unavailable/unreadable → fall back to walk
        logger.debug("git ls-files failed for %s — falling back to os.walk", repo_root)
        return None
    else:
        return paths


def invalidate_repo_file_index(repo_root: str) -> None:
    """Drop the cached file listing for *repo_root*.

    Called from the post-write invalidation paths (``tool_registry`` write
    tools): the index is otherwise TTL-only, so a file created this turn stayed
    invisible to cached consumers for up to ``_FILE_INDEX_TTL`` seconds — the
    "cannot find what it just wrote" symptom the walk and symbol caches were
    fixed for. The generation bump also makes an in-flight listing skip the
    cache store, so a stale result never wins a race with a fresh one.
    """
    global _FILE_INDEX_GEN
    _FILE_INDEX_CACHE.pop(canonical_repo_key(repo_root), None)
    _FILE_INDEX_GEN += 1


def invalidate_for_written_path(file_path: str) -> None:
    """Drop every cached listing whose repo root contains *file_path*.

    Called from the atomic-write funnel (``common.atomic_io``) so ANY write —
    agent write tools OR the editor-lane operation handlers, both of which land
    in ``atomic_write_text``/``atomic_write_json``/``atomic_write_jsonl`` —
    makes a just-written file visible to all cached consumers within the same
    turn, without each write site needing to know its repo root. Writes outside
    every cached repo are a no-op. The generation bump (only when something was
    actually popped) keeps mid-collection listings from being cached stale.
    """
    global _FILE_INDEX_GEN
    try:
        _resolved = str(Path(file_path).resolve())
    except OSError:
        logger.debug("invalidate_for_written_path: could not resolve %s", file_path)
        return
    _popped = False
    for _key in list(_FILE_INDEX_CACHE):
        if _resolved.startswith(_key.rstrip(os.sep) + os.sep):
            _FILE_INDEX_CACHE.pop(_key, None)
            _popped = True
    if _popped:
        _FILE_INDEX_GEN += 1


def cached_repo_file_list(repo_root: str) -> list[str]:
    """Return a sorted list of repo-relative file paths, cached per repo_root.

    Primary source: ``git ls-files -z --cached --others --exclude-standard`` —
    fast (reads the index, no recursive walk), .gitignore-aware, and NUL-
    separated so non-ASCII (Korean, CJK, …) paths survive unmangled.

    Falls back to os.walk when git is unavailable or the path isn't a git
    checkout; the walk delegates directory pruning to the shared
    ``walk_policy._walk_should_skip_dir`` (the same predicate every repo
    walker uses) — no .gitignore awareness, but vendored/cache/build dirs are
    excluded so a non-git tree with a vendored copy doesn't pollute the index.

    Rebuilt when older than ``_FILE_INDEX_TTL`` seconds so a stale index
    (files added/moved) self-heals without paying the listing cost on every
    call, and dropped outright by :func:`invalidate_repo_file_index` /
    :func:`invalidate_for_written_path` after a write so the TTL is a backstop
    rather than the only freshness mechanism. A git failure or a partial
    os.walk abort is NOT cached, so the next call retries a full listing
    instead of serving an incomplete index.
    """
    now = time.monotonic()
    key = canonical_repo_key(repo_root)
    cached = _FILE_INDEX_CACHE.get(key)
    if cached and (now - cached[0]) < _FILE_INDEX_TTL:
        return cached[1]
    # Read generation BEFORE the slow listing — if invalidation bumps it while
    # we collect, the result is stale and must NOT be cached.
    _gen_before = _FILE_INDEX_GEN

    # Primary: git ls-files (fast + .gitignore-aware + non-ASCII safe)
    git_paths = git_list_repo_files(str(repo_root))
    if git_paths is not None:
        if _gen_before != _FILE_INDEX_GEN:
            return git_paths  # invalidated mid-collection — return fresh, skip cache
        _capped_put(_FILE_INDEX_CACHE, key, (now, git_paths), cap=_FILE_INDEX_CACHE_MAX)
        return git_paths

    # Fallback: os.walk delegating to the shared walk-policy predicate (no
    # .gitignore awareness).  _FILE_INDEX_SKIP_DIRS is kept defined above as a
    # re-export (write_tools_core) but the decision uses walk_policy — a
    # strict superset that also excludes vendor/ venv* *.egg-info (F7).
    paths: list[str] = []
    root = str(repo_root)
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _walk_should_skip_dir(d)]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                paths.append(rel.replace(os.sep, "/"))
    except Exception:
        # os.walk may abort partway (PermissionError on a sub-tree, etc.).
        # Do NOT cache a partial index — it would yield incomplete results for
        # the full TTL. Return what we have for THIS call (best-effort) but let
        # the next call retry the full walk.
        return sorted(paths)
    paths.sort()
    if _gen_before != _FILE_INDEX_GEN:
        return paths  # invalidated mid-collection — return fresh, skip cache
    _capped_put(_FILE_INDEX_CACHE, key, (now, paths), cap=_FILE_INDEX_CACHE_MAX)
    return paths
