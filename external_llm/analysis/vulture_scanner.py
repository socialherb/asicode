"""Vulture-based dead code scanner — Python unused code detection.

Registered as ``vulture_dead_code_scanner`` in the ``ScannerRegistry`` and
runnable via ``run_structural_scan`` / ``RUN_SCANNER``.

Design notes:
- ``scan_vulture_dead_code`` is the registry entry point. It accepts an
  optional ``repo_graph`` so it can call ``decide_vulture_scan_scope()`` to
  decide between full-project and file-only scavenge. Leaf-only targets skip
  the expensive full-project walk (the historical ~90s cost).
- Candidates are normalized to ``VultureCandidate`` (kept even when this entry
  point is absent — ``StructuralWorkset.from_vulture_candidates`` and the
  executor still consume it).
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

import datetime as _dt
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.analysis.unused_import_scanner import _has_noqa_comment as _has_noqa_comment

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

# Kinds that ``public_dead_code_scanner`` already covers — and covers BETTER,
# because it resolves cross-file references (vulture only sees per-file usage).
# vulture reports module-level functions/classes as exactly these two kinds;
# ``method``/``variable``/``attribute``/``property`` are class-level or
# private-scope, which public_dead_code_scanner deliberately does NOT scan (see
# its module docstring) — so vulture is the ONLY signal for those. Emitting
# function/class here is pure noise + false-positive risk → excluded by default.
# Override via ``exclude_kinds`` (pass an empty collection to keep everything).
_PUBLIC_DEAD_CODE_OVERLAP_KINDS: frozenset[str] = frozenset({"function", "class"})

# Dunder / protocol names — never dead code (used via implicit protocol).
# Includes NON-dunder framework-protocol methods that are invoked by the
# framework with no static caller (vulture cannot see polymorphic dispatch):
#   _missing_         — Enum metaclass fallback (value lookup)
#   handle_*          — html.parser.HTMLParser streaming callbacks
_ALWAYS_LIVE: frozenset[str] = frozenset({
    "__init__", "__new__", "__str__", "__repr__", "__call__",
    "__enter__", "__exit__", "__iter__", "__next__", "__len__",
    "__getitem__", "__setitem__", "__contains__",
    "__post_init__", "__hash__", "__eq__", "__ne__", "__lt__", "__gt__",
    # Non-dunder framework protocols (no static caller by design):
    "_missing_",
    "handle_starttag", "handle_endtag", "handle_data",
})

# libcst/ast visitor base classes. Subclasses define per-node-type dispatch
# hooks — ``visit_<Node>``, ``leave_<Node>`` — and lifecycle methods
# (``on_visit``/``on_leave``/``generic_visit``) that the framework invokes via
# ``getattr`` with no static caller. Suppression is decided by base-class
# inheritance (see ``_collect_visitor_hook_linenos``), NOT by name alone, so a
# coincidentally named business method (e.g. ``visit_url`` in a non-visitor
# class) is never over-suppressed.
_VISITOR_BASE_NAMES: frozenset[str] = frozenset({
    "CSTVisitor", "CSTTransformer",     # libcst
    "NodeVisitor", "NodeTransformer",   # ast
})
_VISITOR_HOOK_PREFIXES: tuple[str, ...] = ("visit_", "leave_")
_VISITOR_HOOK_EXACT: frozenset[str] = frozenset({"on_visit", "on_leave", "generic_visit"})


# ── noqa suppression ──────────────────────────────────────────────────────────

# Module-level cache keyed by (path → mtime, lines): the scanner may run many
# times in one long-lived process, and the scanned files are edited between
# runs — a path-only cache would serve stale lines and suppress/flag against
# code that no longer exists. mtime comparison invalidates per file; storing
# one entry per path keeps the cache bounded by the file count.
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
            with open(abs_path) as fh:
                cached = (mtime, fh.read().splitlines())
        except OSError:
            cached = (mtime, [])
        _source_lines_cache[abs_path] = cached
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
    return (
        base == "conftest.py"
        or base.startswith("test_")
        or base.endswith("_test.py")
    )


def _is_cancelled(cancel_event: Any) -> bool:
    """Cooperative cancel check — None-safe.

    Used by the vulture scanner's pre/post-processing loops so ESC / Ctrl-C
    (which sets the agent's ``cancel_event``) can interrupt the scan between
    files.  ``v.scavenge()`` itself is an opaque library call that cannot be
    interrupted mid-parse; the checkpoints bracket it so cancel is honored
    before scavenge starts and during result post-processing.
    """
    return cancel_event is not None and cancel_event.is_set()


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
        v.scavenge(scan_paths, exclude=exclude_patterns)
        return True

    import threading

    done = threading.Event()
    err: list = []

    def _run() -> None:
        try:
            v.scavenge(scan_paths, exclude=exclude_patterns)
        except BaseException as exc:  # noqa: BLE001 - re-raise on the caller
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
    import ast
    import tokenize

    seen: set[str] = set()
    for path in scan_paths:
        if _is_cancelled(cancel_event):
            break
        try:
            with open(path, "rb") as fh:
                for tok in tokenize.tokenize(fh.readline):
                    if tok.type != tokenize.STRING:
                        continue
                    try:
                        val = ast.literal_eval(tok.string)
                    except Exception:
                        continue
                    if isinstance(val, str) and val.isidentifier():
                        seen.add(val)
        except (OSError, SyntaxError, tokenize.TokenError):
            continue
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
    import ast

    seen: set[tuple[str, int]] = set()
    for path in scan_paths:
        if _is_cancelled(cancel_event):
            break
        try:
            with open(path, encoding="utf-8") as _f:
                tree = ast.parse(_f.read())
        except (OSError, SyntaxError):
            continue

        class_bases: dict[str, list[str]] = {}
        methods: list[tuple[int, str, str]] = []  # (lineno, name, enclosing_class)

        class _Mapper(ast.NodeVisitor):
            def __init__(self):
                self.stack: list[str] = []

            def visit_ClassDef(self, node):
                self.stack.append(node.name)
                class_bases[node.name] = [
                    b.id if isinstance(b, ast.Name)
                    else (b.attr if isinstance(b, ast.Attribute) else None)
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
        if not class_bases:
            continue

        def _is_visitor(cn: str, _seen: set[str] | None = None) -> bool:
            _seen = _seen if _seen is not None else set()
            if cn in _seen or cn not in class_bases:
                return False
            _seen.add(cn)
            return any(
                b in _VISITOR_BASE_NAMES
                or (b in class_bases and _is_visitor(b, _seen))
                for b in class_bases[cn]
            )

        abs_path = os.path.abspath(path)
        for lineno, name, cn in methods:
            if (
                name in _VISITOR_HOOK_EXACT
                or name.startswith(_VISITOR_HOOK_PREFIXES)
            ) and _is_visitor(cn):
                seen.add((abs_path, lineno))
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
    vulture_confidence: int  # raw Vulture confidence 0–100
    message: str
    normalized_confidence: float = 0.0  # 0.0–1.0, asicode remapped
    evidence_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
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
    """Remap Vulture raw confidence (0–100) to asicode normalized (0.0–1.0).

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
            kinds = sorted(set(c.kind for c in cluster))
            sources = sorted(set(
                src for c in cluster for src in c.evidence_sources
            ))
            min_lineno = min(c.lineno for c in cluster)
            max_end = max(c.end_lineno for c in cluster)

            merged.append(VultureCandidate(
                file=base.file,
                name=base.name,
                kind="|".join(kinds),
                lineno=min_lineno,
                end_lineno=max_end,
                vulture_confidence=base.vulture_confidence,
                message=base.message,
                normalized_confidence=max(c.normalized_confidence for c in cluster),
                evidence_sources=sources,
            ))

    return merged


# ── Scan scope decision ───────────────────────────────────────────────────────

def decide_vulture_scan_scope(
    graph: object,
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
                    fp, threshold,
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
    file_paths: Optional[list[str]] = None,
    min_confidence: int = _cfg.counts.SCANNER_VULTURE_MIN_CONFIDENCE,
    exclude_patterns: Optional[list[str]] = None,
    max_per_file: int = _cfg.counts.SCANNER_VULTURE_MAX,
    repo_graph: object = None,
    exclude_kinds: Optional[Iterable[str]] = None,
    cancel_event: Any = None,
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
        min_confidence: Vulture minimum confidence (0–100).
        exclude_patterns: Glob patterns to exclude (e.g. ``["*test*", "*migrations*"]``).
        max_per_file: Max candidates emitted per file. This is a REPORTING cap,
            not a dead-code-detection threshold: vulture may emit more than this
            per file, but only the first ``max_per_file`` survive. Any aggregate
            count (e.g. "N candidates") produced by this scanner is therefore
            POST-CAP — cite it together with the cap value then in effect, never
            as a raw vulture output size.
        repo_graph: Optional repository graph facade used for hub/leaf scope
            decision (see ``decide_vulture_scan_scope``).
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
        logger.warning(
            "[VULTURE_SCANNER] vulture package not installed — "
            "install with: pip install 'asicode[vulture]'"
        )
        return []

    file_paths = file_paths or []
    scan_start = _dt.datetime.now(tz=_dt.timezone.utc)
    candidates: list[VultureCandidate] = []

    # Resolve excluded kinds. By default drop module-level function/class —
    # they overlap with (and are inferior to) public_dead_code_scanner's
    # cross-file reachability. exclude_kinds=() keeps everything.
    _skip_kinds = (
        frozenset(exclude_kinds) if exclude_kinds is not None
        else _PUBLIC_DEAD_CODE_OVERLAP_KINDS
    )

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
        scan_paths = [
            p if os.path.isabs(p) else os.path.join(repo_root, p)
            for p in file_paths
        ]
    else:
        # full_project: enumerate the project .py set explicitly so vendored
        # dirs are never parsed, while vulture still sees every project module
        # for cross-file reachability. The result whitelist (file_paths, below)
        # then restricts reported candidates to the requested targets.
        scan_paths = _collect_project_py_files(repo_root)

    # Names referenced as identifier-shaped string literals → dispatch-live
    # (handler maps resolved via getattr). Collected once; consulted per
    # candidate below to suppress string-dispatched callables.
    # Cooperative cancel: if already set before the (expensive) pre-processing
    # and scavenge, return empty immediately.
    if _is_cancelled(cancel_event):
        logger.debug("[VULTURE_SCANNER] cancelled before pre-processing")
        return []

    _dispatch_live = _collect_dispatch_live_names(scan_paths, cancel_event=cancel_event)

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
        # Run scavenge with cooperative cancel (see _scavenge_with_cancel).
        # Returns False if cancelled mid-parse — ESC / Ctrl-C during the opaque
        # ast.parse now aborts promptly instead of blocking until it completes.
        # Returning inside this try lets the ``finally`` restore the recursion
        # limit, fixing a latent leak where the old standalone pre-scavenge
        # cancel-check returned before the finally ran.
        if not _scavenge_with_cancel(
            v, scan_paths, exclude_patterns or [], cancel_event=cancel_event,
        ):
            logger.debug("[VULTURE_SCANNER] cancelled before/during scavenge")
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
            if file_paths:
                if not any(
                    rel_file == fp or abs_file == os.path.abspath(fp)
                    for fp in file_paths
                ):
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

            # Suppress framework-dispatched visitor hooks (visit_<Node>/
            # leave_<Node>/on_visit/on_leave/generic_visit) in libcst/ast
            # visitor subclasses. The (abs_file, lineno) match is inherently
            # precise — it identifies a specific def confirmed to live in a
            # visitor subclass, so a coincidentally named non-visitor method is
            # never wrongly dropped.
            if (abs_file, first_lineno) in _visitor_hooks:
                continue

            # Per-file reporting cap. This bounds emitted candidates per file; the
            # raw vulture output may exceed it. Downstream aggregates are post-cap,
            # NOT raw counts — see the max_per_file docstring.
            count = per_file_counts.get(rel_file, 0)
            if count >= max_per_file:
                continue
            # noqa(F841) on the flagged line suppresses the candidate
            if first_lineno and _source_line_has_noqa(abs_file, first_lineno, {"F841"}):
                continue

            per_file_counts[rel_file] = count + 1

            norm_conf = _compute_normalized_confidence(confidence, name, kind)

            candidates.append(VultureCandidate(
                file=rel_file,
                name=name,
                kind=kind,
                lineno=first_lineno,
                end_lineno=max(first_lineno, last_lineno),
                vulture_confidence=confidence,
                message=message,
                normalized_confidence=norm_conf,
                evidence_sources=["vulture_dead_code_scanner"],
            ))

    except Exception:
        logger.exception("[VULTURE_SCANNER] scan failed")
        return []
    finally:
        # Restore recursion limit if we raised it
        if _raised_rec_limit:
            sys.setrecursionlimit(_prev_rec_limit)

    # Internal dedup
    merged = _dedup_candidates(candidates)

    # Sort by normalized confidence descending, then file, then line
    merged.sort(key=lambda c: (-c.normalized_confidence, c.file, c.lineno))

    scan_end = _dt.datetime.now(tz=_dt.timezone.utc)
    elapsed_ms = int((scan_end - scan_start).total_seconds() * 1000)

    if merged:
        logger.info(
            "[VULTURE_SCANNER] %d candidate(s) in %d ms "
            "(scope=%s, min_confidence=%d, dedup applied)",
            len(merged), elapsed_ms, scope, min_confidence,
        )
    else:
        logger.info(
            "[VULTURE_SCANNER] no candidates found (scope=%s, min_confidence=%d, %d ms)",
            scope, min_confidence, elapsed_ms,
        )

    return merged
