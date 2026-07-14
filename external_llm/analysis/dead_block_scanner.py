"""Dead-block scanner — finds clusters of unused definitions.

Phase 3 detector #2.  Targets unused module-level AND class-level constants,
assignments, and helper functions.

Analysis scope:
  - Module-level definitions (functions, classes, assignments)
  - Private names (``_`` prefix) only — public names may be imported from
    other modules and single-file analysis can't see those edges
  - Class-level assignments are NOT scanned: cross-file instance-attribute
    access (e.g. ``result.metadata``) is undetectable via static analysis.
    Scanning class assignments produces unacceptable false positives.
  - Skips dunder (``__foo__``) names
  - Skips names registered in ``__all__``
  - ``self.attr`` / ``cls.attr`` references count as external references
    for class-level attributes
  - Definitions with no reference *outside their own body* are considered dead
    (self-recursion stays inside the definition's line range, so it does NOT
    falsely save a dead recursive helper)

Cross-file reachability (graph_facade-based) is the natural Phase 3.5 upgrade.
For now, conservative single-file scope avoids false positives at the cost of
some false negatives (publicly-named-but-actually-unused symbols).

Adjacent dead definitions (line gap ≤ ``CLUSTER_GAP_TOLERANCE``, default 5)
group into a ``DeadBlockCandidate`` covering the contiguous range.  Clusters
of ≥ 2 members are emitted at confidence 1.0; unclustered single dead symbols
are emitted as singletons at reduced confidence (0.65) so downstream can
require corroborating (e.g. cross-file) evidence before acting on them.
"""

from __future__ import annotations

from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg

from ._dead_block_shared import (
    CLUSTER_GAP_TOLERANCE,
    DeadBlockCandidate,
    DeadBlockMember,
    _is_dead_candidate,
    scan_dead_block_core,
)

# Re-exported for callers/tests that import these via this module.
__all__ = [
    "CLUSTER_GAP_TOLERANCE",
    "DeadBlockCandidate",
    "DeadBlockMember",
    "_is_dead_candidate",
    "scan_dead_blocks",
]


# ── Public scan API ───────────────────────────────────────────────────────────

def scan_dead_blocks(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_DEAD_BLOCK_MAX,
    cluster_gap_tolerance: Optional[int] = None,
    cross_file_referenced_names: Optional[set] = None,
) -> list[DeadBlockCandidate]:
    """Scan files for clusters of unused module-level private symbols.

    Returns one candidate per cluster (≥ 2 adjacent dead members).  Files that
    fail to parse are skipped — detection is supplementary signal and must
    never block the main pipeline.

    When ``cross_file_referenced_names`` is provided (e.g. from
    ``RepositoryGraph.get_callers()`` aggregated across the repo), names in
    that set are treated as externally referenced, allowing the scanner to
    flag symbols that are dead even when a graph layer has confirmed zero
    cross-file references.  Without this parameter, the scanner relies on
    same-file Load reference counting only (conservative, no false positives).
    """
    candidates, truncated = scan_dead_block_core(
        repo_root=repo_root,
        file_paths=file_paths,
        max_per_file=max_per_file,
        cluster_gap_tolerance=cluster_gap_tolerance,
        cross_file_referenced_names=cross_file_referenced_names,
        singleton_confidence=0.65,
        mark_public=False,
        log_tag="DEAD_BLOCK",
        # Documented contract: private names only.  The cross-file set is
        # suppression evidence here; public-symbol detection is
        # public_dead_code_scanner's job (keeps the two scanners distinct).
        include_public=False,
    )
    if truncated:
        # Function attribute consumed by ScannerRegistry.run() (which resets
        # it via `del` before each invocation).
        scan_dead_blocks._truncated = truncated
    return candidates
