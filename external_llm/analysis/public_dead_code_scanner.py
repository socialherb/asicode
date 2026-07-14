"""Public dead code scanner — finds unused symbols (public + private).

Extends the ``dead_block_scanner`` approach to cover public symbols (no ``_``
prefix) by using cross-file reference information from the repository graph.

Analysis strategy:
  - Module-level definitions that have zero Load references in the same file AND
    zero cross-file callers (when ``cross_file_referenced_names`` is provided)
    are dead code.
  - Class-level assignments are NOT scanned — they are API-contract definitions
    whose usage cross-file via instance attribute access is undetectable by
    static analysis (e.g. ``result.metadata``). Scanning them produces
    unacceptable false positives. See ``dead_block_scanner`` for class-level
    private-symbol detection only.
  - Private symbols (``_`` prefix) are always candidates — same as dead_block_scanner.
  - Public symbols are ONLY candidates when ``cross_file_referenced_names`` is
    provided AND the name is NOT in that set (no cross-file callers).
  - Clusters of adjacent dead symbols (gap <= 5 lines) are emitted as a single
    ``DeadBlockCandidate`` (reuses the same data model as dead_block_scanner).

False positive prevention:
  - Requires ``cross_file_referenced_names`` for public symbol detection.
    Without it, only private symbols are scanned (same as dead_block_scanner).
  - Uses the same cluster_gap_tolerance and max_per_file heuristics.
"""

from __future__ import annotations

from typing import Optional

from external_llm.agent.config.thresholds import config as _cfg

from ._dead_block_shared import (
    CLUSTER_GAP_TOLERANCE,
    DeadBlockCandidate,
    DeadBlockMember,
    _collect_all_defs,
    _collect_name_references,
    _extract_all_list,
    _has_framework_injection_decorator,
    _is_externally_referenced,
    _ts_collect_all_defs,
    _ts_collect_name_references,
    _ts_extract_all_list,
    scan_dead_block_core,
)

# Re-export shared symbols for backward compatibility (tests and callers
# may import these from public_dead_code_scanner directly).
__all__ = [
    "CLUSTER_GAP_TOLERANCE",
    "DeadBlockCandidate",
    "DeadBlockMember",
    "_collect_all_defs",
    "_collect_name_references",
    "_extract_all_list",
    "_has_framework_injection_decorator",
    "_is_externally_referenced",
    "_ts_collect_all_defs",
    "_ts_collect_name_references",
    "_ts_extract_all_list",
    "scan_public_dead_blocks",
]



def scan_public_dead_blocks(
    *,
    repo_root: str,
    file_paths: list[str],
    max_per_file: int = _cfg.counts.SCANNER_PUBLIC_DEAD_BLOCK_MAX,
    cluster_gap_tolerance: Optional[int] = None,
    cross_file_referenced_names: Optional[set] = None,
) -> list[DeadBlockCandidate]:
    """Scan files for unused module-level symbols (public + private).

    Args:
        repo_root: Repository root path.
        file_paths: File paths to scan.
        max_per_file: Max dead-block clusters to emit per file.
        cluster_gap_tolerance: Max gap between dead defs to consider them
            part of the same cluster. Defaults to CLUSTER_GAP_TOLERANCE (5).
        cross_file_referenced_names: Set of names that have cross-file callers.
            When provided, public names NOT in this set AND without same-file
            references are considered dead.  Without this parameter, only
            private (``_``-prefixed) names are scanned — same as
            dead_block_scanner.

    Returns:
        List of ``DeadBlockCandidate`` (same model as dead_block_scanner).
    """
    candidates, truncated = scan_dead_block_core(
        repo_root=repo_root,
        file_paths=file_paths,
        max_per_file=max_per_file,
        cluster_gap_tolerance=cluster_gap_tolerance,
        cross_file_referenced_names=cross_file_referenced_names,
        singleton_confidence=0.55,
        mark_public=True,
        log_tag="PUBLIC_DEAD_BLOCK",
    )
    if truncated:
        # Function attribute consumed by ScannerRegistry.run() (which resets
        # it via `del` before each invocation).
        scan_public_dead_blocks._truncated = truncated
    return candidates
