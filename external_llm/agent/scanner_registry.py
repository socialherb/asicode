"""Scanner registry — maps canonical scanner names to callable analysis functions.

Registration-based architecture:
  - New scanners call ``register()`` at module load time.
  - Planner looks up scanners by name or file-path match.
  - ``RUN_SCANNER`` handler dispatches through ``ScannerRegistry.run()``.

Available scanners (auto-registered):
  - ``dead_block_scanner`` → ``scan_dead_blocks()``
  - ``duplicate_definition_scanner`` → ``scan_duplicate_definitions()``
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from ..languages import LanguageId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScannerSpec:
    """Declarative metadata for a registered scanner."""

    name: str
    """Canonical name, e.g. ``"dead_block_scanner"``."""

    description: str
    """One-line description for planner prompt injection."""

    input_schema: dict[str, str] = field(default_factory=dict)
    """Parameter name → type hint string, e.g. ``{"max_per_file": "int"}``."""

    output_type_name: str = ""
    """Human-readable output type, e.g. ``"DeadBlockCandidate"``."""

    produces_workset_kinds: list[str] = field(default_factory=list)
    """Workset kind strings, e.g. ``["dead_block_cluster"]``."""

    file_filter: str = ".py"
    """File extension filter — only files matching this extension are scanned.

    Legacy single-extension filter retained for backward compatibility. New
    scanners should prefer ``supported_languages`` (multi-language aware).
    When both are set, ``supported_languages`` takes precedence at run() time.
    """

    supported_languages: Optional[set["LanguageId"]] = None
    """Languages this scanner can meaningfully analyze.

    ``None`` (default) = no language constraint — scan any code file the
    caller supplies (backward compatible). A concrete set restricts scanning
    to files whose ``LanguageId.from_path`` is a member; files of unsupported
    languages are filtered out before the scanner runs, so a Python-only AST
    scanner (e.g. ``contradictory_logic_scanner``) never receives Go source
    and thus cannot produce language-mismatch false positives. Takes
    precedence over ``file_filter`` when both are set.
    """

    requires_graph: bool = False
    """When True, the RUN_SCANNER handler injects the repository graph facade
    (``self._call_graph``) as the ``repo_graph`` kwarg so the scanner can make
    cross-file decisions (e.g. vulture's hub/leaf scope choice). Scanners that
    only need a precomputed name set should use ``cross_file_referenced_names``
    in ``input_schema`` instead — graph objects are not serializable and must
    not be advertised there."""

    skip_in_all_mode: bool = False
    """When True, this scanner is excluded from ``scanner="all"`` runs.

    Used when a scanner is fully superseded by another scanner (e.g.
    ``dead_block_scanner`` ⊆ ``public_dead_code_scanner``).  Explicit
    invocation by name still works — this only affects the "all" expansion.
    """

    graph_required_for_results: bool = False
    """When True alongside ``requires_graph``, the scanner produces NO meaningful
    results when the graph is unavailable (``repo_graph=None``). The SCAN handler
    then skips it with an explicit message instead of silently returning 0.

    Distinct from ``requires_graph`` alone: vulture declares
    ``requires_graph=True`` but degrades gracefully (hub/leaf scope fallback)
    when the graph is absent, so it must NOT set this flag. Only scanners that
    hard-require the graph (e.g. ``broken_contract_scanner``'s caller-asymmetry
    check) set it."""


@dataclass
class ScannerResult:
    """Structured result from a scanner invocation, stored in accumulated_context."""

    scanner_name: str
    scanner_description: str
    candidates_raw: list[dict[str, Any]]
    total_candidates: int
    affected_files: set[str]
    truncated_count: int = 0
    """Number of candidates truncated by max_per_file limit (0 = all returned)."""


def _scanner_accepts_cancel_event(fn: Callable[..., Any]) -> bool:
    """True if *fn*'s signature accepts a ``cancel_event`` keyword argument.

    Scanners that opt into cooperative cancellation declare ``cancel_event``
    as a parameter (or accept ``**kwargs``). ``run()`` forwards the event only
    to scanners that accept it, so scanners unaware of cancellation are never
    broken by an unexpected keyword.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    return (
        "cancel_event" in sig.parameters
        or any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
    )


class ScannerRegistry:
    """Registry of analysis scanners callable via RUN_SCANNER operations."""

    def __init__(self) -> None:
        self._scanners: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ScannerSpec] = {}
        # Per-scanner lock serializing the reset→invoke→read critical section
        # in run() (see comment there). Created in register() so it always
        # exists for any registered scanner.
        self._run_locks: dict[str, threading.Lock] = {}

    def register(self, spec: ScannerSpec, fn: Callable[..., Any]) -> None:
        """Register a scanner function under the given spec."""
        self._scanners[spec.name] = fn
        self._specs[spec.name] = spec
        self._run_locks[spec.name] = threading.Lock()
        # Cache whether fn accepts a cancel_event kwarg (cooperative cancel).
        # Inspected once at registration to avoid per-run signature overhead.
        fn._accepts_cancel_event = _scanner_accepts_cancel_event(fn)
        logger.info(
            "[SCANNER_REGISTRY] registered '%s' (%s)",
            spec.name, spec.description,
        )

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Return the scanner function for *name*, or None."""
        return self._scanners.get(name)

    def get_spec(self, name: str) -> Optional[ScannerSpec]:
        """Return the scanner spec for *name*, or None."""
        return self._specs.get(name)

    def list_scanners(self) -> list[ScannerSpec]:
        """Return all registered scanner specs."""
        return list(self._specs.values())

    def list_names(self) -> list[str]:
        """Return all registered scanner names."""
        return list(self._scanners.keys())

    def resident_entry_point_names(self) -> set[str]:
        """Return ``__name__`` of every registered scanner entry-point function.

        These names are alive by definition: ``ScannerRegistry`` holds the
        callable and dispatches to it via ``RUN_SCANNER``. Yet they have no
        static call edge and no ``from <module> import <fn>`` entry —
        ``register(name, fn)`` passes them as a callback argument, which
        call-graph and import analysis cannot see. Cross-file ref computation
        merges this set into ``cross_file_referenced_names`` so dead-code
        scanners do not falsely flag scanner entry points (e.g.
        ``scan_vulture_dead_code`` in ``vulture_scanner.py``) as dead.

        Callables without a ``__name__`` (partial wrappers) are skipped.
        """
        names: set[str] = set()
        for fn in self._scanners.values():
            n = getattr(fn, "__name__", None)
            if n:
                names.add(n)
        return names

    def is_scanner_implementation_file(self, file_path: str) -> bool:
        """True when the file is a registered scanner module — must not be a grounding target."""
        stem = file_path.rsplit("/", 1)[-1]
        if LanguageId.from_path(stem) is LanguageId.PYTHON:
            stem = stem[:-3]
        return stem in self._scanners

    def names_for_spec_target_files(self, target_files: list[str]) -> list[str]:
        """Return scanner names whose file stem matches an entry in *target_files*.

        Matching logic:
          1. Extract the basename stem (strip ``.py``) of each target file.
          2. If that stem matches a registered scanner name, include it.
          3. Also strip any leading path segments to find scanners that live in
             subdirectories like ``external_llm/analysis/dead_block_scanner.py``.

        This is how the planner detects that spec targets include scanner modules
        (e.g. ``external_llm/analysis/dead_block_scanner.py`` → stem ``dead_block_scanner``
        → registered scanner ``dead_block_scanner``).
        """
        matched: list[str] = []
        registered = set(self._scanners.keys())
        for tf in target_files:
            stem = tf.rsplit("/", 1)[-1]  # basename
            if LanguageId.from_path(stem) is LanguageId.PYTHON:
                stem = stem[:-3]
            if stem in registered:
                matched.append(stem)
        return matched

    def run(
        self,
        name: str,
        repo_root: str = "",
        file_paths: Optional[list[str]] = None,
        *,
        cancel_event: Optional[Any] = None,
        **kwargs: Any,
    ) -> ScannerResult:
        """Invoke a scanner and wrap the result.

        Args:
            name: Registered scanner name.
            repo_root: Repository root path.
            file_paths: File paths to scan.  If None, the scanner receives
                        an empty list (scanner-specific defaults apply).
            **kwargs: Additional keyword arguments forwarded to the scanner.

        Returns:
            ScannerResult with serialized candidates and metadata.

        Raises:
            ValueError: If *name* is not registered.
        """
        fn = self.get(name)
        spec = self.get_spec(name)
        if fn is None or spec is None:
            raise ValueError(f"Unknown scanner: {name!r}")

        # ── Pre-filter file_paths by language capability ─────────────────────
        # A scanner declares which languages it can analyze. Files whose
        # ``LanguageId`` is not supported are dropped BEFORE the scanner runs,
        # so a Python-only AST scanner never receives (and mis-parses) Go/TS
        # source — eliminating the language-mismatch false positives that
        # occur when ``scanner="all"`` runs over a non-Python repo.
        #
        # Precedence: ``supported_languages`` (set) wins over the legacy
        # ``file_filter`` (single extension). When a scanner sets a concrete
        # ``supported_languages`` set, the extension filter is ignored — the
        # language check is strictly more precise (it correctly admits .ts and
        # .go for a tree-sitter scanner that ``file_filter=".py"`` would
        # wrongly exclude, and correctly rejects .go for a Python-only scanner
        # even when ``file_filter=""``).
        if file_paths:
            if spec.supported_languages is not None:
                _supported = spec.supported_languages
                file_paths = [
                    p for p in file_paths
                    if LanguageId.from_path(p) in _supported
                ]
            elif spec.file_filter:
                _ext = spec.file_filter
                if _ext and not _ext.startswith("."):
                    _ext = "." + _ext
                file_paths = [
                    p for p in file_paths
                    if p.endswith(_ext)
                ]

        # Size the shared AST cache to this scanner's file set so the next
        # scanner over the same set hits the cache instead of re-parsing.
        # Grows at most once per working set (see parse_cache.ensure_capacity).
        if file_paths:
            try:
                from ..analysis import parse_cache
                parse_cache.ensure_capacity(len(file_paths))
            except Exception:  # pragma: no cover - cache sizing is best-effort
                pass

        # ── Critical section: truncation is out-of-band state on the shared ──
        # function object (scanners set ``fn._truncated`` on themselves, then
        # run() reads it back). The reset→invoke→read sequence must be atomic
        # w.r.t. other concurrent ``run()`` calls for the SAME scanner: without
        # serialization one run reads another's ``_truncated`` and
        # misattributes ``truncated_count``. This is reachable because
        # ``run_structural_scan`` is a read-only tool (``_READ_ONLY_TOOLS``)
        # that parallelizes in the agent read phase. A per-scanner lock
        # serializes same-scanner runs while leaving different scanners
        # concurrent (each has its own lock).
        with self._run_locks[name]:
            # Reset per-call truncation tracker (set by scanner function on self).
            try:
                del fn._truncated
            except AttributeError:
                pass

            # Forward cancel_event only to scanners that accept it, so
            # cooperative cancellation reaches opt-in scanners (e.g. vulture)
            # without breaking scanners that don't declare the parameter.
            if cancel_event is not None and getattr(fn, "_accepts_cancel_event", False):
                kwargs["cancel_event"] = cancel_event
            candidates = fn(
                repo_root=repo_root,
                file_paths=file_paths or [],
                **kwargs,
            )

            truncated_count = getattr(fn, "_truncated", 0)
            if not isinstance(truncated_count, int):
                truncated_count = 0

        raw: list[dict[str, Any]] = []
        for c in candidates:
            if hasattr(c, "to_dict"):
                raw.append(c.to_dict())
            elif isinstance(c, dict):
                raw.append(c)
            else:
                raw.append({"repr": repr(c)})

        affected: set[str] = set()
        for c in candidates:
            if hasattr(c, "file") and c.file:
                affected.add(c.file)
            elif isinstance(c, dict) and c.get("file"):
                affected.add(c["file"])

        return ScannerResult(
            scanner_name=name,
            scanner_description=spec.description,
            candidates_raw=raw,
            total_candidates=len(candidates),
            affected_files=affected,
            truncated_count=truncated_count,
        )


# ── Module-level singleton ───────────────────────────────────────────────────

# ── Language capability sets (single source of truth) ──────────────────────
# Tree-sitter-backed scanners share this set — it must stay in sync with the
# ``_LANG_DEF_NODES`` keys in ``analysis/_dead_block_shared.py``.
_TS_LANGUAGES: frozenset = frozenset({
    LanguageId.PYTHON, LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT,
    LanguageId.GO, LanguageId.JAVA, LanguageId.KOTLIN,
})
_PYTHON_ONLY: frozenset = frozenset({LanguageId.PYTHON})
_SCANNER_REGISTRY = ScannerRegistry()


def _auto_register() -> None:
    """Register built-in scanners at module load time."""
    try:
        from ..analysis.dead_block_scanner import scan_dead_blocks

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="dead_block_scanner",
                description="Find clusters of unused module-level private symbols (Python-only: dead-code reachability is unreliable for other languages without native semantic analysis)",
                input_schema={
                    "max_per_file": "int",
                    "cluster_gap_tolerance": "Optional[int]",
                    "cross_file_referenced_names": "Optional[set]",
                },
                output_type_name="DeadBlockCandidate",
                produces_workset_kinds=["dead_block_cluster"],
                    file_filter=".py",
                    supported_languages=set(_PYTHON_ONLY),
                skip_in_all_mode=True,  # superseded by public_dead_code_scanner (superset)
            ),
            scan_dead_blocks,
        )
    except ImportError:
        logger.debug("[SCANNER_REGISTRY] dead_block_scanner not available")

    try:
        from ..analysis.duplicate_definition_scanner import (
            scan_duplicate_definitions,
        )

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="duplicate_definition_scanner",
                description="Find top-level duplicate definitions (same name, same kind)",
                input_schema={"max_per_file": "int"},
                output_type_name="DuplicateDefinitionCandidate",
                produces_workset_kinds=["duplicate_definition"],
                    file_filter="",
                    supported_languages=set(_TS_LANGUAGES),
            ),
            scan_duplicate_definitions,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] duplicate_definition_scanner not available"
        )

    try:
        from ..analysis.unused_import_scanner import scan_unused_imports

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="unused_import_scanner",
                description="Find unused import statements via AST reference analysis",
                input_schema={"max_per_file": "int"},
                output_type_name="UnusedImportCandidate",
                produces_workset_kinds=["unused_import"],
                    file_filter=".py",
                    supported_languages=set(_PYTHON_ONLY),
            ),
            scan_unused_imports,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] unused_import_scanner not available"
        )

    try:
        from ..analysis.public_dead_code_scanner import scan_public_dead_blocks

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="public_dead_code_scanner",
                description="Find unused public and private module-level symbols (cross-file reachability, Python-only)",
                input_schema={
                    "max_per_file": "int",
                    "cluster_gap_tolerance": "Optional[int]",
                    "cross_file_referenced_names": "Optional[set]",
                },
                output_type_name="DeadBlockCandidate",
                produces_workset_kinds=["public_dead_block_cluster"],
                    file_filter=".py",
                    supported_languages=set(_PYTHON_ONLY),
            ),
            scan_public_dead_blocks,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] public_dead_code_scanner not available"
        )

    try:
        from ..analysis.contradictory_logic_scanner import scan_contradictory_logic

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="contradictory_logic_scanner",
                description="Find contradictory conditions, unreachable branches, always-false assertions",
                input_schema={"max_per_file": "int"},
                output_type_name="ContradictoryCandidate",
                produces_workset_kinds=["contradictory_logic"],
                file_filter=".py",
                supported_languages=set(_PYTHON_ONLY),
            ),
            scan_contradictory_logic,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] contradictory_logic_scanner not available"
        )

    try:
        from ..analysis.ast_similarity_scanner import scan_similarity_candidates

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="ast_similarity_scanner",
                description="Find structurally similar symbol pairs (near-duplicates, shared-scaffold)",
                input_schema={
                    "max_per_file": "int",
                    "min_similarity": "float",
                    "symbol_filter": "Optional[list]",
                },
                output_type_name="SimilarityCandidate",
                produces_workset_kinds=["shared_scaffold", "paired_local_patch", "structural_pair"],
                file_filter=".py",
                supported_languages=set(_PYTHON_ONLY),
            ),
            scan_similarity_candidates,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] ast_similarity_scanner not available"
        )

    try:
        from ..analysis.vulture_scanner import scan_vulture_dead_code

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="vulture_dead_code_scanner",
                description=(
                    "Find unused Python methods/variables/attributes/properties/imports "
                    "via the Vulture static analyzer (non-authoritative supplementary "
                    "signal). Module-level function/class are excluded by default — "
                    "public_dead_code_scanner covers those with cross-file reachability."
                ),
                input_schema={
                    "max_per_file": "int",
                    "min_confidence": "int",
                    "exclude_patterns": "Optional[list]",
                    "exclude_kinds": "Optional[Iterable[str]]",
                },
                output_type_name="VultureCandidate",
                produces_workset_kinds=["vulture_dead_code"],
                file_filter=".py",
                supported_languages=set(_PYTHON_ONLY),
                requires_graph=True,
            ),
            scan_vulture_dead_code,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] vulture_dead_code_scanner not available "
            "(install 'asicode[vulture]')"
        )

    try:
        from ..analysis.container_reachability_scanner import (
            scan_container_reachability,
        )

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="container_reachability_scanner",
                description=(
                    "Find structurally unreachable keys in class-level and module-level "
                    "dict literals via intra-class constant-domain propagation"
                ),
                input_schema={
                    "max_per_file": "int",
                    "min_unreachable_keys": "int",
                    "cross_file_referenced_names": "Optional[set]",
                },
                output_type_name="ContainerReachabilityCandidate",
                produces_workset_kinds=["container_dead_keys"],
                file_filter=".py",
                supported_languages=set(_PYTHON_ONLY),
            ),
            scan_container_reachability,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] container_reachability_scanner not available"
        )

    try:
        from ..analysis.broken_contract_scanner import scan_broken_contracts

        _SCANNER_REGISTRY.register(
            ScannerSpec(
                name="broken_contract_scanner",
                description=(
                    "Find writer/reader pairs split by migration — one half still "
                    "live while the other is unreachable (orphan reader/writer)"
                ),
                input_schema={"max_per_file": "int"},
                output_type_name="BrokenContractCandidate",
                produces_workset_kinds=["broken_contract"],
                file_filter=".py",
                supported_languages=set(_PYTHON_ONLY),
                requires_graph=True,
                graph_required_for_results=True,
            ),
            scan_broken_contracts,
        )
    except ImportError:
        logger.debug(
            "[SCANNER_REGISTRY] broken_contract_scanner not available"
        )


def _verify_workset_handler_coverage() -> None:
    """Startup integrity check: every deterministic workset kind needs a handler.

    Cross-checks the union of all ``produces_workset_kinds`` declared by
    registered scanners against ``_WORKSET_HANDLERS`` in scanner_to_ops. A kind
    that no handler consumes is flagged — either it is an intentional
    LLM-judgment kind (ANALYZE_FIRST / EXTRACT_FUNCTION / PAIRED_MODIFY /
    VULTURE_DEAD), or it is a genuine scanner→op pipeline break.

    This static check complements the runtime guard in
    ``build_delete_ops_from_structural_worksets``: the static check fires at
    import time so a regression is visible immediately (B1 was invisible
    because the old code silently ``continue``-d on missing handlers).
    """
    try:
        from ..editor._editor_core.lane.scanner_to_ops import _WORKSET_HANDLERS
    except ImportError:
        # scanner_to_ops not importable in this environment — skip gracefully.
        return

    # Map every produced kind back to the scanner that produces it, so the
    # warning message is actionable (points at the offending scanner).
    kind_to_scanner: dict[str, str] = {}
    for spec in _SCANNER_REGISTRY.list_scanners():
        for kind in spec.produces_workset_kinds:
            kind_to_scanner.setdefault(kind, spec.name)

    for kind, scanner_name in sorted(kind_to_scanner.items()):
        if kind in _WORKSET_HANDLERS:
            continue
        # We cannot read suggested_strategy from a static kind (the adapter sets
        # it at runtime), so we only log an informational note. The runtime
        # guard in build_delete_ops_* does the authoritative break/no-break
        # classification via _is_pipeline_break(ws).
        logger.debug(
            "[SCANNER_REGISTRY] workset kind %r (from %s) has no "
            "_WORKSET_HANDLERS entry — verify it is an intentional LLM-judgment kind, "
            "else add a handler in scanner_to_ops.py",
            kind, scanner_name,
        )


_auto_register()
# NOTE: _verify_workset_handler_coverage() is NOT called at import time.
# It imports `editor._editor_core.lane.scanner_to_ops`, which would couple
# every importer of scanner_registry (transitively: design_chat_loop via
# tool_schemas → tool_registry → agent_turn_pipeline) to the heavy lane
# package — breaking the lane-decoupling contract asserted by
# test_design_chat_loop_closure_has_no_lane_module. The check is an
# informational DEBUG log with an authoritative runtime counterpart in
# build_delete_ops_from_structural_worksets (_is_pipeline_break), so the
# import-time guarantee is not load-bearing. It remains available as a
# public function for explicit invocation in diagnostics/tests.


def get_registry() -> "ScannerRegistry":
    """Return the module-level ScannerRegistry singleton.

    All agent tool handlers should call this function instead of constructing
    a new ScannerRegistry, which would miss the auto-registered scanners.
    """
    return _SCANNER_REGISTRY
