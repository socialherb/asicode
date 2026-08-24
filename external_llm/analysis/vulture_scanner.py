"""Vulture-based dead code scanner — Python unused code detection.

Registered as ``vulture_dead_code_scanner`` in the ``ScannerRegistry`` and
runnable via ``run_structural_scan`` / ``RUN_SCANNER``.

Design notes:
- ``scan_vulture_dead_code`` is the registry entry point. It accepts an
  optional ``repo_graph`` so it can call ``decide_vulture_scan_scope()`` to
  decide between full-project and file-only scavenge. Leaf-only targets skip
  the expensive full-project walk (the historical ~90s cost).
- Candidates are normalized to ``VultureCandidate`` (kept even when this entry
  point is absent — the executor still consumes it).
- Non-authoritative: results are supplementary dead-code evidence, not
  deterministic DELETE ops. ``public_dead_code_scanner`` remains the primary
  cross-file reachability signal.
- Division of labor: ``public_dead_code_scanner`` resolves cross-file references
  for module-level functions/classes (more accurate than vulture's per-file
  view), so vulture EXCLUDES those kinds by default (``exclude_kinds``).
  vulture's unique value is class-level / private-scope detection
  (``method``/``variable``/``attribute``/``property``) that
  ``public_dead_code_scanner`` deliberately does not scan.

Python API (``vulture.Vulture``) is preferred over subprocess for optional-
dependency handling and structured result access.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import logging
import os
import sys
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.analysis import parse_cache
from external_llm.analysis.unused_import_scanner import _has_noqa_comment
from external_llm.common.atomic_io import atomic_write_json
from external_llm.common.cache_utils import _capped_put

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEDUP_LINE_GAP_TOLERANCE = 3  # max line gap to consider two candidates "same location"

# Vulture typ -> asicode kind.
# NOTE: vulture distinguishes ``function`` (module-level ``def``) from
# ``method`` (class-level ``def``) — verified empirically. This distinction
# matters for the overlap filter below (only module-level defs are redundant).
_VULTURE_KIND_MAP: dict[str, str] = {
    "function": "function",
    "method": "method",
    "class": "class",
    "attribute": "attribute",
    "variable": "variable",
    "import": "import",
    "parameter": "parameter",
    "property": "property",
}

# asicode kind -> vulture CLI error code (vulture/core.py:32-38). vulture's
# own get_unused_code() applies NO comment-based suppression (that lives in its
# CLI), so the scanner implements the flake8-style V-code contract itself.
_VULTURE_CODE_BY_KIND: dict[str, str] = {
    "attribute": "V101",
    "class": "V102",
    "function": "V103",
    "import": "V104",
    "method": "V105",
    "property": "V106",
    "variable": "V107",
}

# Kinds that ``public_dead_code_scanner`` already covers — and covers BETTER,
# because it resolves cross-file references (vulture only sees per-file usage).
# vulture reports module-level functions/classes as exactly these two kinds;
# ``method``/``variable``/``attribute``/``property`` are class-level or
# private-scope, which public_dead_code_scanner deliberately does NOT scan (see
# its module docstring) — so vulture is the ONLY signal for those. Emitting
# function/class here is pure noise + false-positive risk → excluded by default.
# Override via ``exclude_kinds`` (pass an empty collection to keep everything).
_PUBLIC_DEAD_CODE_OVERLAP_KINDS: frozenset[str] = frozenset({"function", "class"})

# Bounded retries when vulture hits files that vanished between enumeration
# and read (TOCTOU under parallel sessions / editors / transient probes).
_VANISHED_RETRIES = 3

# Dunder / protocol names — never dead code (used via implicit protocol).
# Includes NON-dunder framework-protocol methods that are invoked by the
# framework with no static caller (vulture cannot see polymorphic dispatch):
#   _missing_         — Enum metaclass fallback (value lookup)
#   handle_*          — html.parser.HTMLParser streaming callbacks
_ALWAYS_LIVE: frozenset[str] = frozenset(
    {
        "__init__",
        "__new__",
        "__str__",
        "__repr__",
        "__call__",
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__post_init__",
        "__hash__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__gt__",
        # Non-dunder framework protocols (no static caller by design):
        "_missing_",
        "handle_starttag",
        "handle_endtag",
        "handle_data",
    }
)

# libcst/ast visitor base classes. Subclasses define per-node-type dispatch
# hooks — ``visit_<Node>``, ``leave_<Node>`` — and lifecycle methods
# (``on_visit``/``on_leave``/``generic_visit``) that the framework invokes via
# ``getattr`` with no static caller. Suppression is decided by base-class
# inheritance (see ``_collect_visitor_hook_linenos``), NOT by name alone, so a
# coincidentally named business method (e.g. ``visit_url`` in a non-visitor
# class) is never over-suppressed.
_VISITOR_BASE_NAMES: frozenset[str] = frozenset(
    {
        "CSTVisitor",
        "CSTTransformer",  # libcst
        "NodeVisitor",
        "NodeTransformer",  # ast
    }
)
_VISITOR_HOOK_PREFIXES: tuple[str, ...] = ("visit_", "leave_")
_VISITOR_HOOK_EXACT: frozenset[str] = frozenset({"on_visit", "on_leave", "generic_visit"})

# Framework base classes whose members are consumed by the framework with no
# static caller — vulture reports them as dead, they are live by contract.
# Detection is STRUCTURAL (inheritance evidence), never name-shape alone:
#   Enum/IntEnum/StrEnum/Flag        — enum members (``NAME = value`` in body)
#   BaseModel (+ pydantic validators) — pydantic fields / ``model_config`` /
#                                       ``@model_validator``-decorated methods
#   BaseHTTPRequestHandler           — ``do_VERB`` verbs, ``log_message``,
#                                      ``server_version`` / ``protocol_version``,
#                                      ``close_connection`` (http.server protocol)
# A coincidentally named business attribute in a non-framework class is never
# suppressed (see ``_framework_live_for_file``).
_ENUM_BASE_NAMES: frozenset[str] = frozenset(
    {
        "Enum",
        "IntEnum",
        "StrEnum",
        "Flag",
        "IntFlag",
    }
)
_PYDANTIC_BASE_NAMES: frozenset[str] = frozenset(
    {
        "BaseModel",
        "BaseSettings",
        "RootModel",
    }
)
_PYDANTIC_FIELD_DECORATORS: frozenset[str] = frozenset(
    {
        "model_validator",
        "field_validator",
        "computed_field",
        "serializer",
        "validator",
    }
)
_HTTP_BASE_NAMES: frozenset[str] = frozenset(
    {
        "BaseHTTPRequestHandler",
        "StreamingHTTPRequestHandler",
        "SimpleHTTPRequestHandler",
        # Local quiet-disconnect wrapper around BaseHTTPRequestHandler: the
        # protocol surface (do_VERB / server_version / close_connection) is
        # equally framework-live on its subclasses, but the inheritance chain
        # crosses a file boundary (mcp/_session_queue.py), which
        # _inherits_from (same-file only) cannot resolve.
        "QuietHttpHandler",
    }
)
_HTTP_PROTOCOL_ATTRS: frozenset[str] = frozenset(
    {
        "server_version",
        "protocol_version",
        "log_message",
        "close_connection",
    }
)


# ── noqa suppression ──────────────────────────────────────────────────────────

# Module-level cache keyed by (path → mtime, lines): the scanner may run many
# times in one long-lived process, and the scanned files are edited between
# runs — a path-only cache would serve stale lines and suppress/flag against
# code that no longer exists. mtime comparison invalidates per file. Entries
# are FIFO-bounded via _capped_put: a long-lived process (REPL/orchestrator,
# test runs) accumulates paths from every repo it ever scans, and each entry
# holds the FULL file content — an entry cap keeps that growth bounded while
# comfortably covering one repo's full-scan working set (this repo: ~209 py
# files, avg ~28KB → cap 256 ≈ ~7MB worst-typical).
_SOURCE_LINES_CACHE_MAX_ENTRIES: int = 256
_source_lines_cache: dict[str, tuple[float, list[str]]] = {}


def _source_line_has_noqa(abs_path: str, lineno: int, codes: set[str] | None = None) -> bool:
    """Check if the source line at *lineno* (1-indexed) has a # noqa comment."""
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        return False
    cached = _source_lines_cache.get(abs_path)
    if cached is None or cached[0] != mtime:
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                cached = (mtime, fh.read().splitlines())
        except OSError:
            cached = (mtime, [])
        _capped_put(_source_lines_cache, abs_path, cached, _SOURCE_LINES_CACHE_MAX_ENTRIES)
    lines = cached[1]
    if 1 <= lineno <= len(lines):
        return _has_noqa_comment(lines[lineno - 1], codes)
    return False


# ── Test-path & string-dispatch suppression ───────────────────────────────────


def _is_test_path(rel_file: str) -> bool:
    """True if *rel_file* is a test file (pytest fixtures/parametrize noise).

    Test files produce a large class of false positives (fixtures, parametrize
    ids, ``conftest`` plugins) that are referenced by the pytest runtime, not by
    static calls. They are still PARSED (for cross-file reachability of the
    production symbols they import) — only their own candidates are dropped.
    """
    norm = rel_file.replace("\\", "/")
    parts = norm.split("/")
    if any(seg == "tests" for seg in parts):
        return True
    base = parts[-1]
    return base == "conftest.py" or base.startswith("test_") or base.endswith("_test.py")


def _is_cancelled(cancel_event: Any) -> bool:
    """Cooperative cancel check — None-safe.

    Used by the vulture scanner's pre/post-processing loops so ESC / Ctrl-C
    (which sets the agent's ``cancel_event``) can interrupt the scan between
    files.  ``v.scavenge()`` itself is an opaque library call that cannot be
    interrupted mid-parse; the checkpoints bracket it so cancel is honored
    before scavenge starts and during result post-processing.
    """
    return cancel_event is not None and cancel_event.is_set()


def _scavenge_tolerant(
    v: Any,
    scan_paths: list[str],
    exclude_patterns: list[str],
) -> None:
    """Run ``v.scavenge()``, surviving files that vanish between enumeration
    and read.

    ``vulture.utils.get_modules`` resolves its whole path list BEFORE any
    parsing and ``sys.exit()``s — raising ``SystemExit`` (a ``BaseException``,
    invisible to ``except Exception``) — when a path in the list no longer
    exists.  Under parallel sessions / editors / transient test probes a file
    can disappear in that window (observed 2026-08-08: a sibling worker's
    transient probe killed the zero-tolerance structural gate mid-run).

    Because get_modules completes before parsing starts, the SystemExit
    carries zero partial state: dropping the vanished paths and retrying is
    safe.  If no listed path is missing, the SystemExit is re-raised
    (unexpected — never mask it).  Bounded by ``_VANISHED_RETRIES``.
    """
    pending = list(scan_paths)
    for _ in range(_VANISHED_RETRIES + 1):
        try:
            v.scavenge(pending, exclude=exclude_patterns)
        except SystemExit:
            kept = [p for p in pending if os.path.exists(p)]
            if len(kept) == len(pending):
                raise  # no listed path vanished — unexpected SystemExit
            logger.warning(
                "[VULTURE_SCANNER] %d path(s) vanished mid-scan — retrying without them",
                len(pending) - len(kept),
            )
            pending = kept
        else:
            return
    # Bounded retries exhausted with a live vanished path each time — the
    # environment is churning too hard to scan; fail loudly rather than
    # silently skipping candidates.
    raise SystemExit(f"vulture: {len(pending)} path(s) kept vanishing during scan")


def _scavenge_with_cancel(
    v: Any,
    scan_paths: list[str],
    exclude_patterns: list[str],
    cancel_event: Any = None,
) -> bool:
    """Run ``v.scavenge()`` honoring a cooperative cancel.

    ``v.scavenge()`` is an opaque library call (bulk ``ast.parse`` over every
    scan path) that cannot be interrupted mid-parse — historically the dominant
    cost of a vulture scan (up to several seconds on large projects) and the
    exact window during which ESC / Ctrl-C felt dead.  To restore
    responsiveness DURING scavenge we move it into a daemon thread and poll the
    cancel event from the (now free) main thread.  Freed from the C call, the
    main thread can also service ``KeyboardInterrupt`` immediately, fixing
    Ctrl-C for the in-process path (the same mechanism that handles ESC).

    Returns True if scavenge completed; False if it was cancelled (or never
    started).  On cancel the daemon thread is ABANDONED — it keeps parsing the
    stale file set until it finishes or the process exits (it is a daemon, so
    it never blocks shutdown).  Abandonment is safe: ``v`` is local to the
    caller and is never touched again once this returns False, so there is no
    shared mutable state to race on; only CPU is consumed transiently.

    When ``cancel_event`` is None (direct API callers, tests, the non-
    interactive CLI), scavenge runs inline with no thread overhead — the common
    path where cancellation is irrelevant.
    """
    if _is_cancelled(cancel_event):
        return False
    if cancel_event is None:
        _scavenge_tolerant(v, scan_paths, exclude_patterns)
        return True

    import threading

    done = threading.Event()
    err: list = []

    def _run() -> None:
        try:
            _scavenge_tolerant(v, scan_paths, exclude_patterns)
        except BaseException as exc:  # re-raise on the caller
            err.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=_run, name="vulture-scavenge", daemon=True)
    worker.start()
    # Poll at ~20 Hz: responsive to cancel without busy-waiting.  ``done.wait``
    # runs in pure Python (bytecode loop), so KeyboardInterrupt lands promptly.
    while not done.wait(timeout=0.05):
        if _is_cancelled(cancel_event):
            return False  # abandon worker thread
    if err:
        raise err[0]
    return True


# ── Per-file scan cache ───────────────────────────────────────────────────────
# vulture's ``scan()`` is per-file (``scan(code, filename=path)`` appends the
# file's defined items to ``v.defined_*`` / ``v.unreachable_code`` and its
# referenced names to the GLOBAL ``v.used_names``), and ``get_unused_code()``
# matches items against that global used-names set — cross-file references
# are the point: a use in b.py keeps a.py's definition alive.  Per-file scan
# results are therefore cacheable independently under the same
# ``(st_mtime_ns, st_size)`` fingerprint the in-memory caches use, and a
# cache-hit run re-parses only the changed files while reproducing the SAME
# ``defined_*`` / ``used_names`` / ``unreachable_code`` state as a full
# scavenge — hence the same ``get_unused_code()`` output.  Round 32-P-F:
# full-project scans went from ~940 ast.parse / ~15s on asicode to
# cache-hot ~0.5s (this repo's structural gate runs the scanner on every
# pre-commit / lint.yml step, so the cache pays off within a single push).
# vulture typ → Vulture attribute that collects that kind's items (the plural
# forms are irregular: class→classes, property→props, variable→vars — never
# derive them with a blind ``+ "s"``).
_VULTURE_TYP_TO_ATTR = {
    "attribute": "defined_attrs",
    "class": "defined_classes",
    "function": "defined_funcs",
    "import": "defined_imports",
    "method": "defined_methods",
    "property": "defined_props",
    "variable": "defined_vars",
}
_VULTURE_DEFINED_ATTRS = (*_VULTURE_TYP_TO_ATTR.values(), "unreachable_code")

# Bumped 1 → 2 (2026-08-16): the per-file "used" payload changed semantics
# from a scan-ORDER-DEPENDENT delta (``set(used_names) - used_before``, which
# silently dropped names first used by earlier-scanned files — a full-repo
# entry restored in per-file mode falsely flagged ``logger``/``ast``/etc. as
# unused) to the file's ABSOLUTE used-name set (fresh-instance scan, valid
# under any restore subset/order).
# Bumped 2 → 3 (round 32-F2): the per-item "filename" key was hoisted to a
# single entry-level "fn" — every item of an entry shares the entry file's
# path, so repeating it 109K times wasted ~4.4MB (19% of the 22MB payload).
_VULTURE_CACHE_VERSION = 3


def _vulture_cache_path(repo_root: str) -> str:
    from . import parse_cache as _pc

    return _pc.cache_file_path(repo_root, f"vulture_scan_v{_VULTURE_CACHE_VERSION}.json")


def _load_vulture_scan_cache(repo_root: str, vulture_version: str) -> dict[str, dict]:
    """Per-file scan results keyed by repo-relative path — fail-open.

    Any read error / format mismatch / vulture-version mismatch returns {} —
    a stale cache must never change scan results, only cost a full re-scan.
    An empty *repo_root* (unit-test convention) bypasses the cache entirely,
    mirroring the other scanners' disk caches.
    """
    if not repo_root:
        return {}
    cache_path = _vulture_cache_path(repo_root)  # outside try: CachePathError must propagate
    try:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    if (
        payload.get("format") != _VULTURE_CACHE_VERSION
        or payload.get("vulture") != vulture_version
        or not isinstance(payload.get("files"), dict)
    ):
        return {}
    return payload["files"]


def _save_vulture_scan_cache(repo_root: str, files: dict[str, dict], vulture_version: str) -> None:
    """Atomic replace via the canonical writer; failures logged (fail-open).

    Delegates to :func:`atomic_write_json` (B2) — same rename atomicity the
    hand-rolled pid-tmp+``os.replace`` here provided, plus fsync, failure-path
    temp cleanup and the stale-temp sweep.  Lock-free last-writer-wins; see
    the disk-cache concurrency policy in ``parse_cache``.
    """
    if not repo_root:
        return
    payload = {
        "format": _VULTURE_CACHE_VERSION,
        "vulture": vulture_version,
        "files": files,
    }
    path = _vulture_cache_path(repo_root)
    try:
        atomic_write_json(path, payload, indent=None, ensure_ascii=True)
    except (OSError, TypeError, ValueError):
        logger.debug("[VULTURE_SCANNER] scan cache write failed", exc_info=True)


def _serialize_vulture_item(item) -> dict:
    """One vulture Item → JSON-safe dict (fields of vulture.core.Item).

    The filename is NOT repeated per item: it is hoisted to the entry's
    "fn" key (all items of an entry belong to that one file — round 32-F2,
    ~19% payload reduction)."""
    return {
        "name": item.name,
        "typ": item.typ,
        "first_lineno": item.first_lineno,
        "last_lineno": item.last_lineno,
        "message": item.message,
        "confidence": item.confidence,
    }


def _restore_vulture_entry(v: Any, entry: dict) -> None:
    """Rehydrate one cached per-file scan into *v* (defined items + used names).

    The restored items carry a ``str`` filename (the cached absolute path),
    NOT a ``pathlib.Path``.  vulture's ``get_unused_code()`` builds a
    ``set(defined_*)`` whose ``Item.__hash__`` hashes ``(filename, lineno,
    name)`` and sorts by ``str(filename).lower()`` — Path hashing is
    measurably expensive on 111K-item corpora (gate profile: 0.37s in
    ``get_unused_code`` alone).  ``Item`` accepts any filename type (the
    ``__init__`` does no coercion; ``size``/``get_report``/``_tuple`` all use
    ``str()`` on it), and a cold-``scan`` file's Path filename never co-occurs
    with a restored file's str filename for the SAME file in one run, so the
    hash/eq surfaces never mix.
    """

    import vulture.core

    for d in entry.get("items", []):
        item = vulture.core.Item(
            name=d["name"],
            typ=d["typ"],
            filename=entry.get("fn") or "",
            first_lineno=d["first_lineno"],
            last_lineno=d["last_lineno"],
            message=d.get("message", ""),
            confidence=d.get("confidence", 0),
        )
        if d["typ"] == "unreachable_code":
            v.unreachable_code.append(item)
        else:
            getattr(v, _VULTURE_TYP_TO_ATTR[d["typ"]]).append(item)
    used = entry.get("used")
    if used:
        v.used_names.update(used)


def _prepare_vulture_exclude_patterns(exclude_patterns: list[str]) -> list[str]:
    """scavenge()'s exclude normalization: bare patterns become ``*pat*``."""

    def prepare_pattern(pattern: str) -> str:
        if not any(char in pattern for char in "*?["):
            return f"*{pattern}*"
        return pattern

    return [prepare_pattern(p) for p in (exclude_patterns or [])]


def _scan_vulture_files_with_cache(
    v: Any,
    scan_paths: list[str],
    exclude_patterns: list[str],
    repo_root: str,
    cancel_event: Any = None,
    files_cache: dict | None = None,
    save_state: dict | None = None,
) -> bool:
    """Per-file vulture scan with a (mtime_ns, size)-keyed disk cache.

    Returns True when the scan completed; False when cancelled (or never
    started).  Replaces the whole-set ``v.scavenge()`` with a per-file
    ``v.scan(code, filename=path)`` loop whose results are cached, so a
    cache-hot run re-parses only the changed files while producing the SAME
    ``defined_*`` / ``used_names`` / ``unreachable_code`` state — and thus the
    same ``get_unused_code()`` output (cross-file matching is preserved: used
    names are rehydrated into the global ``used_names`` set before matching).

    Persistence (round 32-F2): when *save_state* is None the function persists
    via ``parse_cache.should_persist_partial_update`` — a partial update of a
    large corpus is SKIPPED (serialising ~22MB to persist a 1-3-file edit
    costs 4.6s while re-scanning those files costs ~0.1s each).  When the
    caller supplies *save_state* (``{"dirty": int}``), counting is delegated:
    the caller decides once, after its own pre-processing sync, so a cold run
    writes the payload exactly once instead of twice.

    Files that vanish between enumeration and read are skipped with a debug
    log — per-file granularity makes the whole-set SystemExit retry loop
    (``_scavenge_tolerant``) unnecessary: one missing file no longer aborts
    the scan.

    Exclude patterns use scavenge semantics (``*pat*`` fnmatch wrapping,
    case-insensitive).  Cancellation is checked between files — per-file
    granularity makes the scan interruptible at file boundaries without the
    daemon-thread machinery ``_scavenge_with_cancel`` needs for the opaque
    whole-set call.
    """
    import vulture
    import vulture.core

    if _is_cancelled(cancel_event):
        return False

    patterns = _prepare_vulture_exclude_patterns(exclude_patterns)
    if files_cache is None:
        files_cache = _load_vulture_scan_cache(repo_root, vulture.__version__)
    dirty = 0
    for abs_path in scan_paths:
        if _is_cancelled(cancel_event):
            return False
        rel = os.path.relpath(abs_path, repo_root)
        if patterns and vulture.core._match(rel, patterns, case=False):
            continue
        fp = _stat_fingerprint(abs_path)
        if fp is None:
            continue  # vanished between enumeration and read
        entry = files_cache.get(rel)
        if entry is not None and tuple(entry.get("fp") or ()) == fp:
            _restore_vulture_entry(v, entry)
            continue
        pair = parse_cache.read_with_fingerprint(abs_path)
        if pair is None:
            logger.debug("[VULTURE_SCANNER] %s vanished before read", abs_path)
            continue
        # Fused pair: the entry is keyed by the stamp taken with THIS read,
        # never a post-write stamp with pre-write content (B1 order contract).
        code, fp = pair
        marks = {attr: len(getattr(v, attr)) for attr in _VULTURE_DEFINED_ATTRS}
        # Capture THIS file's used names in isolation: vulture's scan only
        # WRITES ``used_names`` (never reads it — verified against vulture
        # 2.16 core.py), so clearing the shared set before the scan and
        # merging the additions back afterwards yields the file's ABSOLUTE
        # used-name set.  The old delta (``set(v.used_names) - used_before``)
        # was order/scope-dependent: an entry written by a full-repo run and
        # restored in per-file mode lost every name first used by an
        # earlier-scanned file, falsely flagging ``logger``/``ast``/etc. as
        # unused (2026-08-16 container_reachability gate run).
        _saved_used = set(v.used_names)
        v.used_names.clear()
        try:
            v.scan(code, filename=abs_path)
        except Exception:
            v.used_names.clear()
            v.used_names.update(_saved_used)
            logger.debug("[VULTURE_SCANNER] scan failed for %s", abs_path, exc_info=True)
            continue
        file_used = set(v.used_names)
        v.used_names.clear()
        v.used_names.update(_saved_used)
        v.used_names.update(file_used)
        files_cache[rel] = {
            "fp": list(fp),
            "fn": abs_path,
            "items": [
                _serialize_vulture_item(item)
                for attr in _VULTURE_DEFINED_ATTRS
                for item in getattr(v, attr)[marks[attr] :]
            ],
            "used": sorted(file_used),
        }
        dirty += 1
    if save_state is not None:
        save_state["dirty"] = save_state.get("dirty", 0) + dirty
    elif dirty and parse_cache.should_persist_partial_update(dirty, len(files_cache)):
        _save_vulture_scan_cache(repo_root, files_cache, vulture.__version__)
    return True


# ── Pre-processing fingerprint caches ─────────────────────────────────────────
# The two pre-processing passes below (_dispatch_names_for_file /
# _visitor_hooks_for_file) are pure functions of file content, and the scanner
# runs repeatedly over the same tree in one long-lived process (exploration
# cycles), so unchanged files are reused instead of re-tokenized / re-parsed on
# every scan. Fingerprint = (st_mtime_ns, st_size) — the same invalidation
# contract as the shared RAG index (PERF-4): a path-only cache would serve
# stale results for files edited between runs. Bounded by LRU eviction.
_T = TypeVar("_T")


class _FingerprintCache(Generic[_T]):
    """Per-path result cache keyed by a (st_mtime_ns, st_size) fingerprint."""

    __slots__ = ("_data", "_maxsize")

    def __init__(self, maxsize: int = 4096) -> None:
        self._data: OrderedDict[str, tuple[tuple[int, int], _T]] = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: str, fingerprint: tuple[int, int]) -> _T | None:
        entry = self._data.get(path)
        if entry is None:
            return None
        if entry[0] != fingerprint:
            # Content changed since cached (file edited) — drop stale entry.
            del self._data[path]
            return None
        self._data.move_to_end(path)
        return entry[1]

    def put(self, path: str, fingerprint: tuple[int, int], result: _T) -> None:
        self._data[path] = (fingerprint, result)
        self._data.move_to_end(path)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)


def _stat_fingerprint(path: str) -> tuple[int, int] | None:
    """(st_mtime_ns, st_size) — delegates to the canonical parse_cache helper
    (single stat code path; order contract documented there, B1)."""
    return parse_cache.stat_fingerprint(path)


_dispatch_names_cache: _FingerprintCache[frozenset[str]] = _FingerprintCache()
_visitor_hooks_cache: _FingerprintCache[frozenset[tuple[str, int]]] = _FingerprintCache()
_framework_live_cache: _FingerprintCache[frozenset[tuple[int, str]]] = _FingerprintCache()


def _dispatch_names_for_file(path: str) -> frozenset[str]:
    """Identifier-shaped string literals in one file (fingerprint-cached)."""
    fp = _stat_fingerprint(path)
    if fp is None:
        return frozenset()
    cached = _dispatch_names_cache.get(path, fp)
    if cached is not None:
        return cached

    import ast
    import tokenize

    names: set[str] = set()
    # Raw-bytes stream on purpose: tokenize.tokenize honours encoding
    # cookies / BOM; re-reading as str would mojibake non-UTF-8 sources.  The
    # stat→read order above already satisfies the B1 contract.
    try:
        with open(path, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type != tokenize.STRING:
                    continue
                try:
                    val = ast.literal_eval(tok.string)
                except Exception:
                    val = None
                if isinstance(val, str) and val.isidentifier():
                    names.add(val)
    except (OSError, SyntaxError, tokenize.TokenError):
        # Unreadable / broken file → no names (recomputed once fixed).
        logger.debug("skipping unreadable/broken file %s", path)
    result = frozenset(names)
    _dispatch_names_cache.put(path, fp, result)
    return result


def _visitor_hooks_for_file(path: str) -> frozenset[tuple[str, int]]:
    """Visitor-protocol hook locations in one file (fingerprint-cached)."""
    fp = _stat_fingerprint(path)
    if fp is None:
        return frozenset()
    cached = _visitor_hooks_cache.get(path, fp)
    if cached is not None:
        return cached

    import ast

    hooks: set[tuple[str, int]] = set()
    pair = parse_cache.read_with_fingerprint(path)
    if pair is None:
        return frozenset()
    try:
        tree = ast.parse(pair[0])
    except SyntaxError:
        return frozenset()

    class_bases: dict[str, list[str]] = {}
    methods: list[tuple[int, str, str]] = []  # (lineno, name, enclosing_class)

    class _Mapper(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            class_bases[node.name] = [
                (b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else None)) or ""
                for b in node.bases
            ]
            self.generic_visit(node)
            self.stack.pop()

        def _record(self, node):
            if self.stack:
                methods.append((node.lineno, node.name, self.stack[-1]))
            self.generic_visit(node)

        visit_FunctionDef = _record
        visit_AsyncFunctionDef = _record

    _Mapper().visit(tree)
    if class_bases:

        def _is_visitor(cn: str, _seen: set[str] | None = None) -> bool:
            _seen = _seen if _seen is not None else set()
            if cn in _seen or cn not in class_bases:
                return False
            _seen.add(cn)
            return any(
                b in _VISITOR_BASE_NAMES or (b in class_bases and _is_visitor(b, _seen)) for b in class_bases[cn]
            )

        abs_path = os.path.abspath(path)
        for lineno, name, cn in methods:
            if (name in _VISITOR_HOOK_EXACT or name.startswith(_VISITOR_HOOK_PREFIXES)) and _is_visitor(cn):
                hooks.add((abs_path, lineno))
    result = frozenset(hooks)
    _visitor_hooks_cache.put(path, fp, result)
    return result


def _framework_live_for_file(path: str) -> frozenset[tuple[int, str]]:
    """``(lineno, name)`` pairs consumed by frameworks vulture cannot see.

    Vulture's per-file reachability has no view of framework dispatch, so
    these categories read as "unused" while being live by contract:

      - enum members (``Enum``/``IntEnum``/``StrEnum``/``Flag`` subclasses):
        ``NAME = value`` in the class body.
      - pydantic fields / settings / validators (``BaseModel`` subclasses):
        annotated class-body fields, ``model_config = ...``, and
        ``@model_validator``-/``@field_validator``-decorated methods.
      - dataclass fields: ``@dataclass`` class-body annotated assignments.
      - http.server protocol surface (``BaseHTTPRequestHandler`` subclasses):
        ``do_VERB`` handlers, ``log_message``, and the ``server_version`` /
        ``protocol_version`` / ``close_connection`` attributes.
      - foreign-object attribute assignment ANYWHERE: ``obj.attr = ...`` /
        ``obj.attr += ...`` where ``obj`` is not the bare ``self`` Name
        (e.g. ``_session.default_buffer.on_text_changed += cb``,
        ``conn.row_factory = sqlite3.Row``,
        ``self._options.include_partial_messages = True``).  A bare
        ``self.x = ...`` stays a vulture-checkable instance attribute —
        except ``self.close_connection`` inside an HTTP handler, which is
        protocol state the server reads back.

    Detection is STRUCTURAL (base-class inheritance / decorator evidence /
    AST target shape), never name-pattern matching — a business attribute
    coincidentally named ``do_something`` outside an HTTP handler is never
    suppressed.
    """
    fp = _stat_fingerprint(path)
    if fp is None:
        return frozenset()
    cached = _framework_live_cache.get(path, fp)
    if cached is not None:
        return cached

    import ast

    live: set[tuple[int, str]] = set()
    pair = parse_cache.read_with_fingerprint(path)
    if pair is None:
        return frozenset()
    try:
        tree = ast.parse(pair[0])
    except (OSError, SyntaxError):
        return frozenset()

    def _base_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _deco_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return _deco_name(node.func)
        return ""

    # Same-file class → direct base names, for inherited framework detection
    # (e.g. ``class ExternalLLMRequest(_BaseModel)`` where ``_BaseModel``
    # itself subclasses pydantic ``BaseModel``).  Collected in one pre-pass so
    # an ancestor chain anywhere in the file is resolvable.
    _class_bases: dict[str, set[str]] = {}
    for _node in ast.walk(tree):
        if isinstance(_node, ast.ClassDef):
            _class_bases[_node.name] = {_base_name(b) for b in _node.bases}

    def _inherits_from(cls_name: str, candidates: frozenset[str], _seen: set[str] | None = None) -> bool:
        """True when *cls_name* (or a same-file ancestor) inherits a candidate."""
        if _seen is None:
            _seen = set()
        if cls_name in _seen or cls_name not in _class_bases:
            return False
        _seen.add(cls_name)
        bases = _class_bases[cls_name]
        return bool(bases & candidates) or any(_inherits_from(b, candidates, _seen) for b in bases if b in _class_bases)

    class _Mapper(ast.NodeVisitor):
        def __init__(self) -> None:
            # Stack of enclosing-class "is HTTP handler" flags — foreign-attr
            # suppression inside an HTTP handler needs to know the context.
            self._http_stack: list[bool] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            base_names = {_base_name(b) for b in node.bases}
            deco_names = {_deco_name(d) for d in node.decorator_list}
            is_enum = bool(base_names & _ENUM_BASE_NAMES)
            is_pydantic = _inherits_from(node.name, _PYDANTIC_BASE_NAMES)
            is_dataclass = "dataclass" in deco_names
            is_http = _inherits_from(node.name, _HTTP_BASE_NAMES)
            is_visitor = _inherits_from(node.name, _VISITOR_BASE_NAMES)

            # Class-body members the framework consumes directly.
            for stmt in node.body:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    for t in targets:
                        if not isinstance(t, ast.Name):
                            continue
                        if (
                            (is_enum or is_pydantic or is_dataclass)
                            or (is_http and t.id in _HTTP_PROTOCOL_ATTRS)
                            or (is_visitor and (t.id.startswith("visit_") or t.id.startswith("leave_")))
                        ):
                            # Enum member / pydantic or dataclass field / HTTP
                            # protocol attr / visitor method alias — all
                            # consumed by the framework, never dead code.
                            live.add((stmt.lineno, t.id))
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_http and (stmt.name.startswith("do_") or stmt.name == "log_message"):
                        live.add((stmt.lineno, stmt.name))
                    if is_pydantic and any(_deco_name(d) in _PYDANTIC_FIELD_DECORATORS for d in stmt.decorator_list):
                        live.add((stmt.lineno, stmt.name))
                        # vulture reports the DECORATOR line as first_lineno
                        # for decorated defs, not the ``def`` line — record
                        # both so the suppression matches either reporting.
                        for d in stmt.decorator_list:
                            live.add((getattr(d, "lineno", 0), stmt.name))

            self._http_stack.append(is_http)
            self.generic_visit(node)
            self._http_stack.pop()

        def _mark_foreign_targets(self, node: ast.AST) -> None:
            targets = getattr(node, "targets", None)
            if targets is None:
                targets = [getattr(node, "target")]  # noqa: B009 — AST node union (Assign/AnnAssign); attribute is node-type-specific
            for t in targets:
                if not isinstance(t, ast.Attribute):
                    continue
                is_bare_self = isinstance(t.value, ast.Name) and t.value.id == "self"
                in_http = bool(self._http_stack) and self._http_stack[-1]
                if (not is_bare_self) or (in_http and t.attr in _HTTP_PROTOCOL_ATTRS):
                    live.add((getattr(node, "lineno"), t.attr))  # noqa: B009 — AST node union; lineno exists on all statement nodes

        visit_Assign = _mark_foreign_targets
        visit_AugAssign = _mark_foreign_targets
        visit_AnnAssign = _mark_foreign_targets

    _Mapper().visit(tree)
    result = frozenset(live)
    _framework_live_cache.put(path, fp, result)
    return result


# ── Pre-processing disk cache (dispatch / visitor-hook / framework-live) ────
# The three pre-processing passes above are PURE per-file functions, but their
# in-memory fingerprint caches die with the process — a gate run re-tokenized
# every file for dispatch names (~2.5s) and re-parsed every file for visitor
# hooks + framework-live (~4s) on top of the v.scan disk cache, which covers
# only the vulture scan itself.  The v.scan cache entry is therefore extended
# with a "pre" section holding the three results under the same fingerprint:
# ``_warm_preprocess_caches`` rehydrates the in-memory caches from disk before
# the collections run, and ``_sync_preprocess_cache`` writes any newly
# computed results back into the entry so the next save persists them.  A
# cache-hot run then pays ~0.5s total for all pre-processing (measured
# 2026-08-16: 6.8s → 0.8s for vulture_dead_code_scanner in the gate).
#
# The visitor-hook results store ABSOLUTE paths; the per-file cache key is the
# repo-relative path, so only the linenos are persisted and the path is
# re-derived from the key on warm-up (they are always the same file).


def _warm_preprocess_caches(files_cache: dict, repo_root: str) -> None:
    """Rehydrate the in-memory pre-processing caches from *files_cache*.

    Entries whose fingerprint is stale are ignored — the caller's
    ``_stat_fingerprint`` check in each ``*_for_file`` function then misses and
    recomputes (the disk fingerprint is deliberately NOT verified here: the
    function-level check is the single invalidation point).
    """
    for rel, entry in files_cache.items():
        fp = tuple(entry.get("fp") or ())
        pre = entry.get("pre")
        if not fp or not isinstance(pre, dict):
            continue
        # abspath, NOT normpath: the fresh-computation path
        # (``_visitor_hooks_for_file``) keys its VALUE tuples by
        # ``os.path.abspath(path)``, and the candidate check matches
        # ``(abspath, lineno)``.  With a RELATIVE repo_root, normpath keeps
        # these tuples relative and every rehydrated visitor hook silently
        # stops matching — visit_* methods leaked as false dead code
        # (round 32-F2 regression, caught by cold-vs-warm parity).
        abs_path = os.path.abspath(os.path.join(repo_root, rel))
        dispatch = pre.get("dispatch")
        if isinstance(dispatch, list):
            _dispatch_names_cache.put(abs_path, fp, frozenset(dispatch))
        vhooks = pre.get("vhooks")
        if isinstance(vhooks, list):
            _visitor_hooks_cache.put(abs_path, fp, frozenset((abs_path, ln) for ln in vhooks))
        framework = pre.get("framework")
        if isinstance(framework, list):
            _framework_live_cache.put(abs_path, fp, frozenset((ln, name) for ln, name in framework))


def _sync_preprocess_cache(files_cache: dict, repo_root: str) -> int:
    """Write newly computed pre-processing results into *files_cache*.

    Returns the number of changed (entry, key) pairs — 0 means nothing
    changed (caller need not persist).  Only entries that already exist
    with a matching fingerprint are updated — a file outside the scan set
    (or invalidated meanwhile) is left alone.
    """
    # The framework convert MUST reproduce the EXACT on-disk JSON shape
    # (``[[lineno, name], ...]`` list-of-lists): JSON loads lists, and a
    # tuple-vs-list comparison between the rehydrated ``pre`` and the
    # freshly computed value marks every framework cache warm entry "dirty"
    # on every run (measured: 192/2956 → 41MB full-payload rewrite ~10s in
    # the structural gate).  A convert that normalises to the disk shape
    # makes a warm run compare equal and skip persistence (dirty < 5%).
    dirty = 0
    for cache, key, convert in (
        (_dispatch_names_cache, "dispatch", lambda r: sorted(r)),
        (_visitor_hooks_cache, "vhooks", lambda r: sorted(ln for _, ln in r)),
        (_framework_live_cache, "framework", lambda r: [[ln, name] for ln, name in sorted(r)]),
    ):
        for abs_path, (fp, result) in list(cache._data.items()):
            try:
                rel = os.path.relpath(abs_path, repo_root)
            except ValueError:
                logger.debug("[VULTURE_SCANNER] preprocess sync: relpath failed for %s", abs_path)
                continue
            if rel.startswith("..") or os.path.isabs(rel):
                continue  # outside the repo — not part of the cache domain
            entry = files_cache.get(rel)
            if entry is None or tuple(entry.get("fp") or ()) != fp:
                continue
            pre = entry.setdefault("pre", {})
            val = convert(result)
            if pre.get(key) != val:
                pre[key] = val
                dirty += 1
    return dirty


def _collect_dispatch_live_names(scan_paths: list[str], cancel_event: Any = None) -> frozenset[str]:
    """Identifier-shaped string literals found across *scan_paths*.

    Vulture cannot see string-based dispatch — e.g. a handler map
    ``{"grep": "_tool_grep"}`` later resolved via ``getattr(self, name)``. A
    ``method``/``function`` candidate whose name appears as a quoted string
    literal is plausibly invoked through such dispatch → suppress it.

    Structural rather than prefix-coded: detects the *mechanism* (string → name)
    instead of hardcoding ``_tool_`` etc., so any registry/getattr pattern is
    covered without per-registry edits.

    Conservative on both sides: only ``str.isidentifier()``-shaped literals are
    collected (log/docstring prose won't match), and the resulting set is
    consulted only for ``method``/``function`` candidates (variables/attributes
    keep reporting).
    """
    seen: set[str] = set()
    for path in scan_paths:
        if _is_cancelled(cancel_event):
            break
        seen.update(_dispatch_names_for_file(path))
    return frozenset(seen)


def _collect_visitor_hook_linenos(scan_paths: list[str], cancel_event: Any = None) -> set[tuple[str, int]]:
    """``(abs_path, def_lineno)`` pairs of visitor-protocol methods in visitor subclasses.

    ``libcst`` (``CSTVisitor``/``CSTTransformer``) and ``ast``
    (``NodeVisitor``/``NodeTransformer``) dispatch per-node-type hooks —
    ``visit_<Node>``, ``leave_<Node>``, and the lifecycle methods
    ``on_visit``/``on_leave``/``generic_visit`` — via ``getattr`` with no
    static caller, so vulture reports them as dead.

    Detection is STRUCTURAL: the method's enclosing class must inherit
    (directly, or via a same-file ancestor chain) from a name in
    ``_VISITOR_BASE_NAMES``. This is the same "detect the mechanism, not the
    naming convention" discipline as ``_collect_dispatch_live_names`` — a
    coincidentally named business method (e.g. ``visit_url`` in a non-visitor
    class) is NOT collected, so real dead code there is still reported.
    """
    seen: set[tuple[str, int]] = set()
    for path in scan_paths:
        if _is_cancelled(cancel_event):
            break
        seen.update(_visitor_hooks_for_file(path))
    return seen


# ── Candidate model ────────────────────────────────────────────────────────────


@dataclass
class VultureCandidate:
    """One unused code item found by Vulture, normalized to asicode format."""

    file: str
    name: str
    kind: str  # "function" | "class" | "variable" | "import" | "attribute" | "parameter" | "property"
    lineno: int
    end_lineno: int
    vulture_confidence: int  # raw Vulture confidence 0-100
    message: str
    normalized_confidence: float = 0.0  # 0.0-1.0, asicode remapped
    evidence_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:  # noqa: V105 — serialization contract: ScannerRegistry.run() calls this via hasattr(c, "to_dict"); per-file scans (pre-commit) only see same-file calls, so vulture reports it dead while it is live by contract
        return {
            "file": self.file,
            "name": self.name,
            "kind": self.kind,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "vulture_confidence": self.vulture_confidence,
            "normalized_confidence": self.normalized_confidence,
            "message": self.message,
            "evidence_sources": self.evidence_sources,
        }


# ── Confidence normalization ───────────────────────────────────────────────────


def _compute_normalized_confidence(
    vulture_confidence: int,
    name: str,
    kind: str,
) -> float:
    """Remap Vulture raw confidence (0-100) to asicode normalized (0.0-1.0).

    Adjustments (name/kind heuristics only):

    * ``__dunder__`` protocol names → 0.0 (always-live, skip entirely upstream)
    * ``test_`` prefix → 0.85 cap (pytest fixture/parametrization false-positive risk)
    * ``parameter`` kind → 0.75 cap (unused arguments sometimes intentional, e.g. interface conformance)
    * default: raw / 100
    """
    raw = vulture_confidence / 100.0

    if name in _ALWAYS_LIVE:
        return 0.0

    if name.startswith("test_"):
        raw = min(raw, 0.85)

    if kind == "parameter":
        raw = min(raw, 0.75)

    return max(0.0, min(raw, 1.0))


# ── Internal dedup ─────────────────────────────────────────────────────────────


def _dedup_candidates(candidates: list[VultureCandidate]) -> list[VultureCandidate]:
    """Merge candidates that refer to the same (file, name, kind) with nearby lines.

    When Vulture emits multiple items for what is effectively the same symbol
    (e.g. a function reported as both "function" and "variable"), merge them
    into one candidate with accumulated evidence.
    """
    if len(candidates) <= 1:
        return candidates

    # Group by (file, name)
    groups: dict[tuple[str, str], list[VultureCandidate]] = {}
    for c in candidates:
        key = (c.file, c.name)
        groups.setdefault(key, []).append(c)

    merged: list[VultureCandidate] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort by lineno
        group.sort(key=lambda c: c.lineno)

        # Cluster by line proximity
        clusters: list[list[VultureCandidate]] = []
        current_cluster = [group[0]]
        for c in group[1:]:
            prev_end = max(m.end_lineno for m in current_cluster)
            if c.lineno - prev_end <= _DEDUP_LINE_GAP_TOLERANCE:
                current_cluster.append(c)
            else:
                clusters.append(current_cluster)
                current_cluster = [c]
        clusters.append(current_cluster)

        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
                continue

            # Merge: keep highest-confidence as base, combine evidence_sources
            base = max(cluster, key=lambda c: c.vulture_confidence)
            kinds = sorted({c.kind for c in cluster})
            sources = sorted({src for c in cluster for src in c.evidence_sources})
            min_lineno = min(c.lineno for c in cluster)
            max_end = max(c.end_lineno for c in cluster)

            merged.append(
                VultureCandidate(
                    file=base.file,
                    name=base.name,
                    kind="|".join(kinds),
                    lineno=min_lineno,
                    end_lineno=max_end,
                    vulture_confidence=base.vulture_confidence,
                    message=base.message,
                    normalized_confidence=max(c.normalized_confidence for c in cluster),
                    evidence_sources=sources,
                )
            )

    return merged


# ── Scan scope decision ───────────────────────────────────────────────────────


def decide_vulture_scan_scope(
    graph: Any,
    file_paths: list[str],
    threshold: int,
) -> str:
    """Return ``"full_project"`` or ``"file_paths_only"`` based on hub/leaf classification.

    A file is a hub if its inbound importer count meets or exceeds *threshold*.
    Any hub in *file_paths* forces ``"full_project"`` to preserve cross-file accuracy.
    Falls back to ``"full_project"`` when *graph* is unavailable or raises.

    Args:
        graph: Repository graph facade exposing ``get_importers(file_path) -> list``.
        file_paths: Target files to classify.
        threshold: Importer count at or above which a file is considered a hub.
    """
    if graph is None or not hasattr(graph, "get_importers"):
        return "full_project"
    for fp in file_paths:
        try:
            if len(graph.get_importers(fp) or []) >= threshold:
                logger.debug(
                    "[VULTURE_SCOPE] %s has >= %d importer(s) — full_project scan",
                    fp,
                    threshold,
                )
                return "full_project"
        except Exception:
            return "full_project"
    logger.debug("[VULTURE_SCOPE] all targets are leaf-like — file_paths_only scan")
    return "file_paths_only"


def _collect_project_py_files(repo_root: str) -> list[str]:
    """Return absolute paths of all project .py files under *repo_root*.

    Reuses the canonical, per-root-cached walker
    ``external_llm.agent._shared_utils._walk_py_files`` — which skips
    ``.venv`` / ``node_modules`` / dot-dirs / caches / build artifacts — so
    vulture never parses vendored packages. See ``scan_vulture_dead_code`` for
    why this matters: ``vulture.scavenge([repo_root])`` walks the tree with
    vulture's OWN exclude rules, which do not match our vendored dirs.
    """
    from pathlib import Path

    from external_llm.agent._shared_utils import _walk_py_files

    return [str(p) for p in _walk_py_files(Path(repo_root), max_files=4000)]


# ── Main scanner entry point ───────────────────────────────────────────────────


def scan_vulture_dead_code(
    *,
    repo_root: str,
    file_paths: list[str] | None = None,
    min_confidence: int = _cfg.counts.SCANNER_VULTURE_MIN_CONFIDENCE,
    exclude_patterns: list[str] | None = None,
    max_per_file: int = _cfg.counts.SCANNER_VULTURE_MAX,
    repo_graph: Any = None,
    exclude_kinds: Iterable[str] | None = None,
    cancel_event: Any = None,
    cross_file_referenced_names: set | None = None,
) -> list[VultureCandidate]:
    """Run Vulture and return normalized dead-code candidates.

    Args:
        repo_root: Project root directory for Vulture to scan.
        file_paths: Target paths (project-relative or absolute). When *repo_graph*
            classifies every target as a leaf (importers below
            ``VULTURE_HUB_IMPORTER_THRESHOLD``), Vulture scavenges ONLY these
            paths (fast). If any target is a hub — or the graph is unavailable —
            Vulture scavenges the whole project: the project .py file set is
            enumerated explicitly (vendored dirs never parsed), so cross-file
            reachability stays accurate without the ~15x cost of parsing
            ``.venv`` / ``node_modules``.
        min_confidence: Vulture minimum confidence (0-100).
        exclude_patterns: Glob patterns to exclude (e.g. ``["*test*", "*migrations*"]``).
        max_per_file: Max candidates emitted per file. This is a REPORTING cap,
            not a dead-code-detection threshold: vulture may emit more than this
            per file, but only the first ``max_per_file`` survive. Any aggregate
            count (e.g. "N candidates") produced by this scanner is therefore
            POST-CAP — cite it together with the cap value then in effect, never
            as a raw vulture output size.
        repo_graph: Optional repository graph facade used for hub/leaf scope
            decision (see ``decide_vulture_scan_scope``).
        cross_file_referenced_names: Set of names referenced from OTHER files
            (whole-repo fact, same contract as ``public_dead_code_scanner``).
            In ``file_paths_only`` scope, a candidate whose name is in this
            set is consumed cross-file (value/attribute references) and must
            not be reported — vulture's per-file scan cannot see those uses,
            and ``_caller_live`` only covers call edges, so module-level
            variables consumed by value (e.g. a derived alias like
            ``_TS_LANGUAGES = SCAN_LANGUAGES``) would otherwise false-positive.
            Ignored in ``full_project`` scope, where vulture scans every
            project file and sees the uses itself.
        exclude_kinds: Candidate kinds to drop from results. Defaults to
            ``_PUBLIC_DEAD_CODE_OVERLAP_KINDS`` ({"function", "class"}) because
            ``public_dead_code_scanner`` already resolves those module-level
            definitions with cross-file reachability — vulture's per-file view of
            them is strictly inferior (redundant + false-positive-prone). The
            class-level kinds vulture uniquely covers (``method``/``variable``/
            ``attribute``/``property``) are never in the default. Pass an empty
            collection (e.g. ``exclude_kinds=()``) to keep everything, or add
            kinds to suppress more.
    """
    try:
        import vulture.core
    except ImportError:
        logger.warning("[VULTURE_SCANNER] vulture package not installed — install with: pip install 'asicode[vulture]'")
        return []

    file_paths = file_paths or []
    scan_start = _dt.datetime.now(tz=_dt.timezone.utc)
    candidates: list[VultureCandidate] = []

    # Resolve excluded kinds. By default drop module-level function/class —
    # they overlap with (and are inferior to) public_dead_code_scanner's
    # cross-file reachability. exclude_kinds=() keeps everything.
    _skip_kinds = frozenset(exclude_kinds) if exclude_kinds is not None else _PUBLIC_DEAD_CODE_OVERLAP_KINDS

    # ── Decide scan scope: full_project (accurate) vs file-only (fast) ──────────
    # Leaf-only targets skip the project-wide enumeration. Either way vulture
    # receives an EXPLICIT file list — never ``[repo_root]``, which made
    # vulture walk the tree with its own (looser) exclude rules and parse
    # .venv/node_modules (16658 files vs 956 here) — 91% of run_structural_scan
    # wall time and ~20k vendored false positives. See _collect_project_py_files.
    scope = decide_vulture_scan_scope(
        repo_graph,
        file_paths,
        _cfg.counts.VULTURE_HUB_IMPORTER_THRESHOLD,
    )
    if scope == "file_paths_only" and file_paths:
        scan_paths = [p if os.path.isabs(p) else os.path.join(repo_root, p) for p in file_paths]
        # String-based dispatch (handler maps resolved via getattr) is a
        # CROSS-FILE mechanism: the map usually lives in a registry file
        # OUTSIDE the leaf-scanned targets. file_paths_only must therefore
        # still collect identifier-shaped string literals from the whole
        # project — otherwise every dynamically dispatched method in a
        # leaf-scanned file is a false positive (the registry's
        # ``"name": "_handler"`` literals are invisible to a target-only
        # tokenize pass). Mirrors ``cross_file_referenced_names``, which the
        # structural-scanner gate also computes repo-wide in per-file mode.
        dispatch_scan_paths = _collect_project_py_files(repo_root)
    else:
        # full_project: enumerate the project .py set explicitly so vendored
        # dirs are never parsed, while vulture still sees every project module
        # for cross-file reachability. The result whitelist (file_paths, below)
        # then restricts reported candidates to the requested targets.
        scan_paths = _collect_project_py_files(repo_root)
        dispatch_scan_paths = scan_paths

    # Names referenced as identifier-shaped string literals → dispatch-live
    # (handler maps resolved via getattr). Collected once; consulted per
    # candidate below to suppress string-dispatched callables.
    # Cooperative cancel: if already set before the (expensive) pre-processing
    # and scavenge, return empty immediately.
    if _is_cancelled(cancel_event):
        logger.debug("[VULTURE_SCANNER] cancelled before pre-processing")
        return []

    # Pre-processing disk cache (see the section above _collect_dispatch_live_
    # names): load the per-file cache ONCE and rehydrate the in-memory dispatch
    # / visitor-hook / framework-live caches from it, so a cache-hot run skips
    # the tokenize/parse passes entirely (measured ~6s in the structural gate).
    files_cache = _load_vulture_scan_cache(repo_root, vulture.__version__)
    _warm_preprocess_caches(files_cache, repo_root)
    # Single persistence decision point (round 32-F2): the scan and the
    # pre-processing sync both accumulate into save_state, and the finally
    # block applies should_persist_partial_update ONCE — a cold run writes
    # the ~22MB payload exactly once instead of twice.
    save_state = {"dirty": 0}

    _dispatch_live = _collect_dispatch_live_names(dispatch_scan_paths, cancel_event=cancel_event)

    # In ``file_paths_only`` scope vulture parses ONLY the target files, so its
    # global used-names basis lacks cross-file usage: methods called from other
    # modules and imports consumed elsewhere (e.g. TYPE_CHECKING imports used
    # only in string annotations) come back as false positives. The repo graph
    # is repo-wide — seed a live-name set from caller edges (the same signal
    # ``compute_cross_file_referenced_names_light`` seeds from) and suppress any
    # candidate whose name has ≥1 caller outside the scanned targets. This
    # mirrors vulture's full-project semantics, where those same names are
    # used somewhere in the scanned set and never reported. Two sources:
    #   1. definitions of the target files with ≥1 caller edge, and
    #   2. names the target files IMPORT whose name has callers elsewhere —
    #      the import itself may be consumed only via string annotations
    #      (``-> "ToolResult"``), which vulture cannot see either way.
    _caller_live: set[str] = set()
    if scope == "file_paths_only" and repo_graph is not None and hasattr(repo_graph, "get_callers"):
        try:
            for p in file_paths:
                abs_p = p if os.path.isabs(p) else os.path.join(repo_root, p)
                rel_p = os.path.relpath(abs_p, repo_root)
                if hasattr(repo_graph, "get_symbols_in_file"):
                    for sym in repo_graph.get_symbols_in_file(rel_p) or []:
                        name = sym.name if hasattr(sym, "name") else ""
                        if name and repo_graph.get_callers(name):
                            _caller_live.add(name)
                # Fused single-stat cached read+parse — the same path vulture's
                # own scan uses (see _scan_vulture_files_with_cache), so each
                # target file is parsed exactly once instead of re-reading it
                # here with a bare open()+ast.parse().
                _, _tree = parse_cache.read_and_parse(abs_p)
                if _tree is not None:
                    for node in ast.walk(_tree):
                        if isinstance(node, ast.Import):
                            for a in node.names:
                                _n = a.asname or a.name.partition(".")[0]
                                if _n and repo_graph.get_callers(_n):
                                    _caller_live.add(_n)
                        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                            for a in node.names:
                                if a.name != "*":
                                    _n = a.asname or a.name
                                    if _n and repo_graph.get_callers(_n):
                                        _caller_live.add(_n)
        except Exception:
            logger.debug("[VULTURE_SCANNER] caller-live lookup failed", exc_info=True)

    # Visitor-protocol methods (visit_<Node>/leave_<Node>/on_visit/...) in
    # libcst/ast visitor subclasses — framework-dispatched via getattr, no
    # static caller. Detected structurally (base-class inheritance), not by
    # name alone. Keyed by (abs_path, def_lineno) for precise matching.
    _visitor_hooks = _collect_visitor_hook_linenos(scan_paths, cancel_event=cancel_event)

    # Bump recursion limit — Vulture's scavenge can recurse deeply on large
    # projects and raise RecursionError mid-scan (see git d2582924).
    _prev_rec_limit = sys.getrecursionlimit()
    _raised_rec_limit = _prev_rec_limit < 5000
    if _raised_rec_limit:
        sys.setrecursionlimit(5000)

    try:
        v = vulture.core.Vulture(verbose=False)
        # Per-file scan with a fingerprint-keyed disk cache (see
        # _scan_vulture_files_with_cache): cache-hot runs re-parse only the
        # changed files while reproducing the same defined_*/used_names state
        # as a full scavenge — and therefore the same get_unused_code() output.
        # Returns False if cancelled — ESC / Ctrl-C is honored between files
        # instead of mid-opaque-call.  Returning inside this try lets the
        # ``finally`` restore the recursion limit, fixing a latent leak where
        # the old standalone pre-scavenge cancel-check returned before the
        # finally ran.
        if not _scan_vulture_files_with_cache(
            v,
            scan_paths,
            exclude_patterns or [],
            repo_root,
            cancel_event=cancel_event,
            files_cache=files_cache,
            save_state=save_state,
        ):
            logger.debug("[VULTURE_SCANNER] cancelled before/during scan")
            return []

        per_file_counts: dict[str, int] = {}

        for item in v.get_unused_code(
            min_confidence=min_confidence,
            sort_by_size=False,
        ):
            if _is_cancelled(cancel_event):
                logger.debug("[VULTURE_SCANNER] cancelled during result processing")
                break
            # Vulture Item fields: name, typ, filename, first_lineno, last_lineno, confidence, message, size
            file_path = getattr(item, "filename", "")
            name = getattr(item, "name", "")
            typ = getattr(item, "typ", "")
            first_lineno = getattr(item, "first_lineno", 0)
            last_lineno = getattr(item, "last_lineno", 0)
            confidence = getattr(item, "confidence", 0)
            message = getattr(item, "message", "")

            if not file_path or not name:
                continue

            # Resolve project-relative path
            abs_file = os.path.abspath(file_path)
            try:
                rel_file = os.path.relpath(abs_file, repo_root)
            except ValueError:
                rel_file = abs_file

            # Drop candidates from test files (pytest fixtures / parametrize ids
            # / conftest plugins are referenced by the pytest runtime, not by
            # static calls). Tests are still parsed for reachability; only their
            # own candidates are suppressed here.
            if _is_test_path(rel_file):
                continue

            # file_paths whitelist filter
            if file_paths and not any(rel_file == fp or abs_file == os.path.abspath(fp) for fp in file_paths):
                continue

            # Filter dunder protocol names (always-live, regardless of kind)
            if name in _ALWAYS_LIVE:
                continue

            kind = _VULTURE_KIND_MAP.get(typ, typ)

            # Drop kinds covered by public_dead_code_scanner (cross-file aware).
            # Done BEFORE the per-file cap so excluded kinds never consume it.
            if kind in _skip_kinds:
                continue

            # Suppress string-dispatched callables: a method/function whose name
            # appears as a quoted identifier (handler map / getattr lookup) is
            # plausibly invoked through dispatch vulture cannot track. Variables
            # and attributes are NOT suppressed — a string match there is weaker
            # evidence and would risk hiding real dead code.
            if kind in ("method", "function") and name in _dispatch_live:
                continue

            # Suppress names with ≥1 caller edge in the repo graph (leaf-scope
            # cross-file usage — see the _caller_live seed above).
            if name in _caller_live:
                continue

            # Suppress names referenced from other files (whole-repo
            # cross-file fact; the gate injects
            # ``compute_cross_file_referenced_names_light`` output).  Mirrors
            # public_dead_code_scanner's contract: a value/attribute reference
            # (module-level alias, README-literal dispatch, property read)
            # has no call edge, so _caller_live cannot see it, but the name is
            # still live.  Only applied in file_paths_only scope — full_project
            # scans every project file, so vulture's own used_names already
            # covers these.
            if scope == "file_paths_only" and cross_file_referenced_names and name in cross_file_referenced_names:
                continue

            # Suppress framework-dispatched visitor hooks (visit_<Node>/
            # leave_<Node>/on_visit/on_leave/generic_visit) in libcst/ast
            # visitor subclasses. The (abs_file, lineno) match is inherently
            # precise — it identifies a specific def confirmed to live in a
            # visitor subclass, so a coincidentally named non-visitor method is
            # never wrongly dropped.
            if (abs_file, first_lineno) in _visitor_hooks:
                continue

            # Suppress framework-consumed definitions (enum members, pydantic /
            # dataclass fields, http.server protocol surface, foreign-object
            # attribute assignments). Location-precise (file+lineno+name): only
            # the exact definition the framework consumes is dropped.
            if (first_lineno, name) in _framework_live_for_file(abs_file):
                continue

            # Per-file reporting cap. This bounds emitted candidates per file; the
            # raw vulture output may exceed it. Downstream aggregates are post-cap,
            # NOT raw counts — see the max_per_file docstring.
            count = per_file_counts.get(rel_file, 0)
            if count >= max_per_file:
                continue
            # an inline noqa comment on the flagged line suppresses the
            # candidate: legacy F841 (ruff's unused-variable code) plus the
            # vulture code for this kind (V101-V107). vulture's own
            # get_unused_code() applies no noqa filtering, so the flake8-style
            # contract is implemented here (see _VULTURE_CODE_BY_KIND).
            suppress_codes: set[str] = {"F841"}
            v_code = _VULTURE_CODE_BY_KIND.get(kind)
            if v_code:
                suppress_codes.add(v_code)
            if first_lineno and _source_line_has_noqa(abs_file, first_lineno, suppress_codes):
                continue

            per_file_counts[rel_file] = count + 1

            norm_conf = _compute_normalized_confidence(confidence, name, kind)

            candidates.append(
                VultureCandidate(
                    file=rel_file,
                    name=name,
                    kind=kind,
                    lineno=first_lineno,
                    end_lineno=max(first_lineno, last_lineno),
                    vulture_confidence=confidence,
                    message=message,
                    normalized_confidence=norm_conf,
                    evidence_sources=["vulture_dead_code_scanner"],
                )
            )

    except Exception:
        logger.exception("[VULTURE_SCANNER] scan failed")
        return []
    finally:
        # Restore recursion limit if we raised it
        if _raised_rec_limit:
            sys.setrecursionlimit(_prev_rec_limit)
        # Persist any newly computed pre-processing results (dispatch /
        # visitor-hook / framework-live) into the v.scan cache entry — runs on
        # every exit path so a cancelled/errored scan still saves the work it
        # did complete.  No-op when nothing changed.  Partial updates of a
        # large corpus are deliberately SKIPPED (round 32-F2): the next
        # process re-scans the stale files for far less than a full-payload
        # serialisation costs.
        save_state["dirty"] += _sync_preprocess_cache(files_cache, repo_root)
        if parse_cache.should_persist_partial_update(save_state["dirty"], len(files_cache)):
            _save_vulture_scan_cache(repo_root, files_cache, vulture.__version__)

    # Internal dedup
    merged = _dedup_candidates(candidates)

    # Sort by normalized confidence descending, then file, then line
    merged.sort(key=lambda c: (-c.normalized_confidence, c.file, c.lineno))

    scan_end = _dt.datetime.now(tz=_dt.timezone.utc)
    elapsed_ms = int((scan_end - scan_start).total_seconds() * 1000)

    if merged:
        logger.info(
            "[VULTURE_SCANNER] %d candidate(s) in %d ms (scope=%s, min_confidence=%d, dedup applied)",
            len(merged),
            elapsed_ms,
            scope,
            min_confidence,
        )
    else:
        logger.info(
            "[VULTURE_SCANNER] no candidates found (scope=%s, min_confidence=%d, %d ms)",
            scope,
            min_confidence,
            elapsed_ms,
        )

    return merged
