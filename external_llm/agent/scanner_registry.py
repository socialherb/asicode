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

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..analysis.scan_walk import SCAN_LANGUAGES
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

    file_filter: str = ".py"
    """File extension filter — only files matching this extension are scanned.

    Legacy single-extension filter retained for backward compatibility. New
    scanners should prefer ``supported_languages`` (multi-language aware).
    When both are set, ``supported_languages`` takes precedence at run() time.
    """

    supported_languages: set[LanguageId] | None = None
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
    return "cancel_event" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _hash_source_file(path: str) -> str | None:
    """sha256 hex digest of *path*, or None when the file cannot be read.

    Used by the scanner source-freshness check: the digest IS the logical
    version of a module's code — it changes iff the code changes, with no
    manual version bookkeeping to drift out of sync.
    """
    import hashlib

    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        logger.debug("[SCANNER_REGISTRY] cannot hash source %s", path, exc_info=True)
        return None


class ScannerRegistry:
    """Registry of analysis scanners callable via RUN_SCANNER operations."""

    def __init__(self) -> None:
        self._scanners: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ScannerSpec] = {}
        # Per-scanner lock serializing the reset→invoke→read critical section
        # in run() (see comment there). Created in register() so it always
        # exists for any registered scanner.
        self._run_locks: dict[str, threading.Lock] = {}
        # Scanner source freshness (R12-2): module source path → sha256 of the
        # code actually loaded at registration time. A mismatch against the
        # on-disk file later means this process executes pre-edit code.
        self._loaded_fingerprints: dict[str, str] = {}
        # P3-1 opt-in auto-reload: when True, hosts (structural-scan tool
        # handler) reload stale scanner modules in place instead of only
        # warning "restart required". Enabled process-wide via
        # ``ASICODE_SCANNER_AUTO_RELOAD=1``; hosts may also flip the attribute
        # on the shared registry at any time. Default keeps the warning-only
        # design (reload mutates live code and must be explicitly opted into).
        self.auto_reload_stale = os.environ.get("ASICODE_SCANNER_AUTO_RELOAD", "0") in {"1", "true", "yes"}

    def register(self, spec: ScannerSpec, fn: Callable[..., Any]) -> None:
        """Register a scanner function under the given spec."""
        self._scanners[spec.name] = fn
        self._specs[spec.name] = spec
        self._run_locks[spec.name] = threading.Lock()
        # Cache whether fn accepts a cancel_event kwarg (cooperative cancel).
        # Inspected once at registration to avoid per-run signature overhead.
        fn._accepts_cancel_event = _scanner_accepts_cancel_event(fn)  # type: ignore[attr-defined]  # dynamic scanner attribute
        self._record_source_fingerprints(fn)
        logger.info(
            "[SCANNER_REGISTRY] registered '%s' (%s)",
            spec.name,
            spec.description,
        )

    def get(self, name: str) -> Callable[..., Any] | None:
        """Return the scanner function for *name*, or None."""
        return self._scanners.get(name)

    def get_spec(self, name: str) -> ScannerSpec | None:
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
        file_paths: list[str] | None = None,
        *,
        cancel_event: Any | None = None,
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
                file_paths = [p for p in file_paths if LanguageId.from_path(p) in _supported]
            elif spec.file_filter:
                _ext = spec.file_filter
                if _ext and not _ext.startswith("."):
                    _ext = "." + _ext
                file_paths = [p for p in file_paths if p.endswith(_ext)]

        # Size the shared AST cache to this scanner's file set so the next
        # scanner over the same set hits the cache instead of re-parsing.
        # Grows at most once per working set (see parse_cache.ensure_capacity).
        if file_paths:
            with contextlib.suppress(OSError, ValueError):  # pragma: no cover - cache sizing is best-effort
                from ..analysis import parse_cache

                parse_cache.ensure_capacity(len(file_paths))

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
            with contextlib.suppress(AttributeError):
                del fn._truncated  # type: ignore[attr-defined]  # dynamic scanner attribute

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

    # ── Scanner source freshness (R12-2) ─────────────────────────────────────
    # A long-lived server (MCP / REPL / webapp) imports scanner modules once
    # and keeps executing that in-memory code.  When a scanner source file
    # changes on disk afterwards (e.g. a bugfix commit), the server silently
    # keeps serving the OLD logic — scan results reflect pre-fix code with no
    # observable signal.  The methods below detect this by comparing the code
    # loaded at registration time against the current on-disk source, so
    # callers can surface a "restart required" notice.

    def _record_source_fingerprints(self, fn: Callable[..., Any]) -> None:
        """Snapshot the loaded source of *fn*'s module for staleness checks.

        Records (a) the entry module itself and (b) every scanner-implementation
        module already imported in this process — scanner logic often lives in
        shared siblings (``_dead_block_shared`` is a dependency of
        ``dead_block_scanner``, not an entry point), and a change there is
        invisible if only entry modules are fingerprinted.

        Best-effort diagnostics: never raises, never breaks registration.
        """
        try:
            # (a) The entry module. fn.__module__ may be a non-string (mocks,
            # partials, C extensions) — only real module names are recorded.
            mod_name = getattr(fn, "__module__", None)
            if isinstance(mod_name, str):
                self._record_module_source(mod_name)
            # (b) Sibling implementation modules already imported. The union
            # across registrations covers everything loaded during startup;
            # modules imported lazily AFTER registration (e.g. parse_cache in
            # run()) are not recorded — they were loaded fresh, so they cannot
            # be stale, and re-recording them later would mask staleness.
            for name in list(sys.modules):
                if name.startswith(_SCANNER_IMPL_PKG):
                    self._record_module_source(name)
        except Exception:  # pragma: no cover - defensive, diagnostics only
            logger.debug(
                "[SCANNER_REGISTRY] source fingerprint recording failed",
                exc_info=True,
            )

    @staticmethod
    def _normalize_source_path(src: str) -> str | None:
        """Map a module ``__file__`` to its on-disk ``.py`` path (or None).

        Non-editable installs expose ``.pyc`` — hash/reload the sibling ``.py``.
        """
        if src.endswith(".pyc"):
            src = src[:-1]
        return src if src.endswith(".py") else None

    def _record_module_source(self, module_name: str) -> None:
        """Fingerprint *module_name*'s source file, if resolvable."""
        mod = sys.modules.get(module_name)
        if mod is None:
            return
        src = getattr(mod, "__file__", None)
        if not isinstance(src, str):
            return
        src = self._normalize_source_path(src)
        if src is None:
            return
        digest = _hash_source_file(src)
        if digest is not None:
            self._loaded_fingerprints[src] = digest

    def _module_for_source(self, path: str) -> Any | None:
        """Return the loaded module whose source file is *path* (or None)."""
        for mod in list(sys.modules.values()):
            src = getattr(mod, "__file__", None)
            if isinstance(src, str):
                src = self._normalize_source_path(src)
                if src == path:
                    return mod
        return None

    def reload_stale_sources(self) -> list[str]:
        """Reload scanner modules whose on-disk source changed since load.

        Best-effort opt-in (``auto_reload_stale``): returns the paths that were
        successfully reloaded. Modules that fail to reload keep serving their
        old code and are reported stale again by the next
        ``verify_loaded_sources()``.

        Sibling implementation modules are reloaded BEFORE the scanner entry
        modules that import them: ``importlib.reload`` re-executes a module
        body, but the import statements inside it hit ``sys.modules`` and
        serve the OLD sibling objects, so a shared helper must be refreshed
        first. After reloading, every registered scanner is re-registered from
        the reloaded module so ``run()`` dispatches to the NEW function
        objects (reload replaces module globals while the registry still holds
        the old callables), and the freshness fingerprints are re-snapshotted.
        """
        stale = self.verify_loaded_sources()
        if not stale:
            return []
        entry_mods = {
            sys.modules[fn.__module__]
            for fn in self._scanners.values()
            if isinstance(getattr(fn, "__module__", None), str) and fn.__module__ in sys.modules
        }
        stale_mods: list[tuple[str, Any]] = []
        for path in stale:
            mod = self._module_for_source(path)
            if mod is not None:
                stale_mods.append((path, mod))
        siblings_first = [(p, m) for p, m in stale_mods if m not in entry_mods]
        entries_after = [(p, m) for p, m in stale_mods if m in entry_mods]
        reloaded: list[str] = []
        for path, mod in siblings_first + entries_after:
            try:
                self._reload_module(mod, path)
            except Exception:
                logger.warning(
                    "[SCANNER_REGISTRY] reload failed for %s (keeping old code)",
                    path,
                    exc_info=True,
                )
            else:
                reloaded.append(path)
        if reloaded:
            # Re-register from the reloaded modules. Runs outside any run()
            # critical section (the handler reloads before invoking scanners),
            # so replacing per-scanner run locks here is safe.
            self._re_register_all()
        return reloaded

    @staticmethod
    def _reload_module(mod: Any, path: str) -> None:
        """Reload *mod* from its source file, bypassing the bytecode cache.

        ``importlib.reload`` (and ``exec_module``, which routes through
        ``SourceFileLoader.get_code``) may serve a VALID-looking
        ``__pycache__`` entry when the source changed within the same second
        AND kept the same size (pyc validation compares (mtime, size)) —
        silently keeping the OLD code. Reading the source directly and
        compiling it always reflects the on-disk file, so a reload can never
        serve stale bytecode. On failure the previous module object is
        restored to ``sys.modules`` so callers keep a working (old-code)
        module.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(mod.__name__, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        new_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod.__name__] = new_mod
        try:
            with open(path, "rb") as fh:
                source = fh.read()
            code = compile(source, path, "exec")
            exec(code, new_mod.__dict__)
        except Exception:
            sys.modules[mod.__name__] = mod
            raise

    def _re_register_all(self) -> None:
        """Re-register every scanner from its (possibly reloaded) module."""
        for name, spec in list(self._specs.items()):
            old_fn = self._scanners.get(name)
            mod_name = getattr(old_fn, "__module__", None)
            fn_name = getattr(old_fn, "__name__", None)
            if not isinstance(mod_name, str) or not isinstance(fn_name, str):
                continue
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            new_fn = getattr(mod, fn_name, None)
            if new_fn is None or new_fn is old_fn:
                continue
            self.register(spec, new_fn)

    def verify_loaded_sources(self) -> list[str]:
        """Return source files whose on-disk content differs from what loaded.

        Empty list = every fingerprinted scanner module still matches the code
        this process imported.  Non-empty = this process executes pre-edit code
        for those modules; callers should surface a "restart the server" notice
        (the structural-scan tool handler does this per invocation, and the MCP
        server also logs at boot).
        """
        # None (deleted/unreadable) never equals a recorded digest → stale.
        stale: list[str] = [
            path
            for path in sorted(self._loaded_fingerprints)
            if _hash_source_file(path) != self._loaded_fingerprints[path]
        ]
        return stale

    def source_versions(self) -> dict[str, str]:
        """Loaded-code fingerprints: source path → short sha256 prefix.

        Diagnostic view of the logical version of every fingerprinted module.
        """
        return {path: digest[:8] for path, digest in sorted(self._loaded_fingerprints.items())}


# ── Module-level singleton ───────────────────────────────────────────────────

# ── Language capability sets (single source of truth) ──────────────────────
# The 6-language scan set is owned by external_llm/analysis/scan_walk.py:
# ``SCAN_LANGUAGES`` is DERIVED from ``SCAN_EXTS`` through ``_EXT_MAP``
# (languages/models.py — the package's canonical extension → language map,
# import-time fail-fast), and ``_TS_LANGUAGES`` here is an IDENTITY ALIAS
# of it — the registry cannot drift from the scan walk (pinned by
# test_scan_walk_constants_are_single_source_aliases).
#
# The per-language judge maps — ``_LANG_TOP_LEVEL_NODES`` / ``_LANG_KIND_MAP``
# in analysis/duplicate_definition_scanner.py and ``_LANG_DEF_NODES`` in
# analysis/_dead_block_shared.py — must keep exactly these six languages as
# keys (a missing key silently unjudges that language, and for the five
# non-Python languages duplicate_definition_scanner is the sole gate judge —
# see the language-coverage contract at the gate).  Pinned by
# test_duplicate_definition_lang_keys_match_registry.
_TS_LANGUAGES: frozenset = SCAN_LANGUAGES
_PYTHON_ONLY: frozenset = frozenset({LanguageId.PYTHON})
_SCANNER_REGISTRY = ScannerRegistry()

# Modules under this package prefix form the scanner implementation surface.
# The freshness check fingerprints them so a long-lived server that loaded
# pre-edit code can be detected (R12-2). Derived from __package__ so a package
# rename keeps the check functional.
_SCANNER_IMPL_PKG: str = f"{(__package__ or 'external_llm.agent').rsplit('.', 1)[0]}.analysis"


def _auto_register() -> None:
    """Register built-in scanners at module load time."""
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
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
            skip_in_all_mode=True,  # superseded by public_dead_code_scanner (superset)
        ),
        scan_dead_blocks,
    )

    from ..analysis.duplicate_definition_scanner import (
        scan_duplicate_definitions,
    )

    _SCANNER_REGISTRY.register(
        ScannerSpec(
            name="duplicate_definition_scanner",
            description="Find top-level duplicate definitions (same name, same kind)",
            input_schema={"max_per_file": "int"},
            file_filter="",
            supported_languages=set(_TS_LANGUAGES),
        ),
        scan_duplicate_definitions,
    )

    from ..analysis.unused_import_scanner import scan_unused_imports

    _SCANNER_REGISTRY.register(
        ScannerSpec(
            name="unused_import_scanner",
            description="Find unused import statements via AST reference analysis",
            input_schema={"max_per_file": "int"},
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
        ),
        scan_unused_imports,
    )

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
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
        ),
        scan_public_dead_blocks,
    )

    from ..analysis.contradictory_logic_scanner import scan_contradictory_logic

    _SCANNER_REGISTRY.register(
        ScannerSpec(
            name="contradictory_logic_scanner",
            description="Find contradictory conditions, unreachable branches, always-false assertions",
            input_schema={"max_per_file": "int"},
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
        ),
        scan_contradictory_logic,
    )

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
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
        ),
        scan_similarity_candidates,
    )

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
                "cross_file_referenced_names": "Optional[set]",
            },
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
            requires_graph=True,
        ),
        scan_vulture_dead_code,
    )

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
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
        ),
        scan_container_reachability,
    )

    from ..analysis.broken_contract_scanner import scan_broken_contracts

    _SCANNER_REGISTRY.register(
        ScannerSpec(
            name="broken_contract_scanner",
            description=(
                "Find writer/reader pairs split by migration — one half still "
                "live while the other is unreachable (orphan reader/writer)"
            ),
            input_schema={"max_per_file": "int"},
            file_filter=".py",
            supported_languages=set(_PYTHON_ONLY),
            requires_graph=True,
            graph_required_for_results=True,
        ),
        scan_broken_contracts,
    )


_auto_register()
# NOTE: the two scanner↔handler coverage checks that used to live here were
# removed with the PLANNER lane — both read `_WORKSET_HANDLERS` /
# `_SCANNER_ADAPTERS` out of `lane.scanner_to_ops` / `lane.structural_workset`,
# which no longer exist. The authoritative runtime counterpart survives in
# build_delete_ops_from_structural_worksets (_is_pipeline_break).


def get_registry() -> ScannerRegistry:
    """Return the module-level ScannerRegistry singleton.

    All agent tool handlers should call this function instead of constructing
    a new ScannerRegistry, which would miss the auto-registered scanners.
    """
    return _SCANNER_REGISTRY
