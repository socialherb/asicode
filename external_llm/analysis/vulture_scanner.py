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
from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg

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

# Dunder / protocol names — never dead code (used via implicit protocol)
_ALWAYS_LIVE: frozenset[str] = frozenset({
    "__init__", "__new__", "__str__", "__repr__", "__call__",
    "__enter__", "__exit__", "__iter__", "__next__", "__len__",
    "__getitem__", "__setitem__", "__contains__",
    "__post_init__", "__hash__", "__eq__", "__ne__", "__lt__", "__gt__",
})


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
        max_per_file: Max candidates emitted per file.
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

    # Bump recursion limit — Vulture's scavenge can recurse deeply on large
    # projects and raise RecursionError mid-scan (see git d2582924).
    _prev_rec_limit = sys.getrecursionlimit()
    _raised_rec_limit = _prev_rec_limit < 5000
    if _raised_rec_limit:
        sys.setrecursionlimit(5000)

    try:
        v = vulture.core.Vulture(verbose=False)
        v.scavenge(scan_paths, exclude=exclude_patterns or [])

        per_file_counts: dict[str, int] = {}

        for item in v.get_unused_code(
            min_confidence=min_confidence,
            sort_by_size=False,
        ):
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

            # Per-file cap
            count = per_file_counts.get(rel_file, 0)
            if count >= max_per_file:
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
