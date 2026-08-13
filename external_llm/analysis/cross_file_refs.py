"""Cross-file referenced-name computation for dead-code scanners.

Builds the ``cross_file_referenced_names`` set that gates public-symbol
dead-code detection (``dead_block_scanner`` / ``public_dead_code_scanner`` /
``container_reachability_scanner``).  A name lands in the set when there is
ANY evidence it is referenced outside its defining file:

  1. **Call edges** — the repository graph reports ≥1 caller.
  2. **Import-based exports** — a file the graph reports as an importer does
     ``from <candidate module> import <name>`` (call edges miss constants,
     classes used as types, etc.).  The ORIGINAL name is recorded even when
     the importer aliases it (``import X as Y`` references X, not Y).
     Module-precise: the import source must resolve to the candidate module.
  3. **Imported names (the seed)** — every name imported or module-attr-read
     by the files in the *seed input*: ``from m import X as Y`` → X and Y,
     ``import m; m.X`` → X.  This is the only channel covering
     ``module.attr`` reads (config-flag constants have no call edge and no
     ImportFrom entry), and it is coarse by design — any import of the same
     name anywhere in the seed input counts, which over-suppresses, the safe
     direction for dead-code judgement.

CALLER CONTRACT (F1, 2026-08-11): channel 3 only sees files it is given.
``compute_cross_file_referenced_names_light`` seeds from *imported_names*
when supplied, else from *candidate_files* — so a caller passing only its
(possibly capped) scan list gets module-attr coverage restricted to that
list, and a ``module.attr`` read living in a non-candidate file is invisible.
Callers must pass a repo-wide seed: the structural gate hands over
``graph.imported_names``, the tool path unions the scan list with the graph's
uncapped ``py_files``.  The removed full variant walked ``get_importers()``
for this instead (module-precise but O(candidates * importers)); it was
dropped for the O(n) seed, which is why the contract now lives in the caller.

PYTHON-ONLY by contract (2026-08-11): every consumer of this set is a
Python-only dead-code scanner — the sole non-Python gate scanner
(``duplicate_definition_scanner``) takes no reference set.  Non-Python files
are therefore excluded structurally: ``extract_imported_names_for_file``
returns an empty set for them without reading, and the candidate pass
filters to ``LanguageId.PYTHON`` before iterating.  The tree-sitter TS/JS
name extraction that previously fed this set was removed: a TS
``import { build }`` is zero evidence about an unrelated Python ``build``,
and the cross-language collisions it seeded could hide genuinely dead Python
symbols from the structural gate.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Optional

from ..graph.repository_graph import path_to_module
from ..languages import LanguageId
from . import parse_cache

logger = logging.getLogger(__name__)


def _scanner_resident_entry_points() -> set:
    """Names of scanner entry points resident in the ``ScannerRegistry``.

    These callables are alive by construction (the registry dispatches to them
    via ``RUN_SCANNER``) but invisible to call-graph/import analysis — they are
    passed to ``register(name, fn)`` as callback arguments rather than being
    called statically. Without this suppression, ``public_dead_code_scanner``
    falsely reports every scanner entry point (e.g.
    ``scan_vulture_dead_code``, ``scan_duplicate_definitions``) as dead.

    Imported lazily and failure-isolated: in any environment where the registry
    is not importable (stripped runtime, partial install) the scanner degrades
    to the prior conservative behaviour — no false suppression, just no extra
    liveness signal.
    """
    try:
        from ..agent.scanner_registry import get_registry
        return get_registry().resident_entry_point_names()
    except Exception:
        logger.debug("[CROSS_FILE_REFS] scanner registry unavailable", exc_info=True)
        return set()


def _dotted_chain(node) -> Optional[str]:
    """Rebuild ``a.b.c`` from a Name/Attribute chain, or None if dynamic."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _add_python_import_names(tree, names: set) -> set:
    """Add every name an AST's imports bind or reference; return the local bindings.

    For ``from m import X as Y`` both X (the name defined in m — what
    dead-code suppression must match) and Y (the local binding) are added.
    The returned set holds the file's local import bindings, used by the
    caller to find ``binding.attr`` module-attribute reads.
    """
    bindings: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                bindings.add(local)
                names.add(local)
                names.add(alias.name.split(".")[-1])
                # `import a.b.c` binds `a` but attribute reads use the full
                # dotted chain — record it so _dotted_chain matching works.
                if not alias.asname and "." in alias.name:
                    bindings.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name and alias.name != "*":
                    names.add(alias.name.split(".")[-1])
                local = alias.asname or alias.name.split(".")[-1]
                if local and local != "*":
                    bindings.add(local)
                    names.add(local)
    return bindings


def _add_module_attr_reads(tree, bindings: set, names: set) -> None:
    """Add ``binding.attr`` reads to *names* for import-introduced bindings.

    Catches the config-flag pattern: ``import config; config.FLAG`` — FLAG
    has no call edge and no ImportFrom entry, yet it is a live reference.
    Coarse by design (any attribute on any imported binding counts), which
    over-suppresses — the safe direction for dead-code judgement.
    """
    if not bindings:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted_chain(node.value)
            if chain and (chain in bindings or chain.split(".")[0] in bindings):
                names.add(node.attr)


def extract_imported_names_for_file(abs_f: str) -> set:
    """Names one file imports or module-attr-reads (empty set on parse failure).

    Python-only by contract (module docstring): non-Python files return an
    empty set WITHOUT being read — the ref set has no non-Python consumer,
    and extracting TS/JS names would pollute Python dead-code judgement with
    cross-language name collisions.  File-level split of
    :func:`compute_imported_names` — lets callers cache the per-file result
    and union it later (the structural-scanner gate caches these under the
    same mtime/size manifest as its graph, so an incremental run skips
    re-parsing unchanged files entirely).
    """
    if LanguageId.from_path(abs_f) is not LanguageId.PYTHON:
        return set()
    tree = parse_cache.parse_ast(abs_f)
    if tree is None:
        return set()
    names = set()
    bindings = _add_python_import_names(tree, names)
    _add_module_attr_reads(tree, bindings, names)
    return names


def compute_imported_names(repo_root: str, file_paths: list[str]) -> set:
    """One-pass set of every name imported or module-attr-read in *file_paths*.

    ``from m import X as Y`` → X and Y, ``import a.b as c`` → c, and
    ``import m; m.X`` → X.  Python-only by contract (module docstring):
    non-Python files contribute nothing.  A name in this set is referenced by
    *some* file, so dead-code scanners must never judge it dead — this catches
    cross-file imports that single-file Load counting cannot see.  This is the
    O(n) seed channel of ``compute_cross_file_referenced_names_light``: any
    import of the same name *within file_paths* counts, which over-suppresses
    — the safe direction for dead-code judgement.  Coverage is exactly
    *file_paths*, so a name referenced only from a file outside it is not
    seen (module docstring, CALLER CONTRACT).
    """
    names: set = set()
    rr = repo_root or ""
    for f in file_paths or []:
        abs_f = f if os.path.isabs(f) else os.path.join(rr, f)
        names |= extract_imported_names_for_file(abs_f)
    return names


def compute_cross_file_referenced_names_light(
    graph,
    repo_root: str,
    candidate_files: list[str],
    imported_names: Optional[set] = None,
) -> Optional[set]:
    """O(n) cross-file referenced-name computation — the single implementation,
    used by both the structural gate and the tool path.

    Combines repo-wide imported names (one parse pass, cached) with graph
    caller edges (in-memory lookups).  For each candidate file it also scans
    the importer files the graph reports (``get_importers``) for
    ``from <candidate module> import X`` patterns and records X — this is the
    signal call edges miss entirely (constants, classes-used-as-types,
    functions imported but never called by name in the importer).  Without
    this pass public symbols exported only via ``from m import X`` would be
    falsely reported as dead (e.g. ``detect_cloud_provider``).  The importer
    pass resolves relative imports against the importer's package.

    ``module.attr`` reads come only from the seed, whose scope is the
    caller's: *imported_names* when supplied, else *candidate_files*.
    Passing a partial file list therefore narrows that channel to those files
    (module docstring, CALLER CONTRACT) — both shipping callers pass a
    repo-wide seed, and a new caller must too.

    *imported_names* is an optional pre-computed repo-wide imported-name set
    (default: recompute here over *candidate_files*) — the structural-scanner
    gate caches per-file results and passes the union so unchanged files are
    never re-parsed.
    Both channels are Python-only (see module docstring): non-Python
    candidates are filtered out before the per-file passes, and the seed
    must itself be py-only (as the gate's ``build(collect_imported_names=
    True)`` produces since 2026-08-11).

    Scanner-registry-resident entry points (``scan_*`` callables passed to
    ``ScannerRegistry.register``) are merged up front — they are alive by
    construction but invisible to both caller edges and import analysis.

    Returns None when *graph* is unusable: without caller edges, ``module.attr``
    usage of public symbols is invisible, so unlocking public-symbol detection
    would be unsound.
    """
    if graph is None or not candidate_files:
        return None
    if not (hasattr(graph, "get_symbols_in_file") and hasattr(graph, "get_callers")):
        return None
    has_importers = hasattr(graph, "get_importers")
    # Python-only by contract (module docstring): non-Python candidates
    # contribute neither caller-edge names nor importer exports.
    py_files = [f for f in candidate_files if LanguageId.from_path(f) is LanguageId.PYTHON]
    # Size the shared parse cache to the py working set BEFORE parsing: the
    # importer pass re-parses importer files for every candidate, and a
    # default-sized cache thrashes on any repo bigger than it (measured: 60%
    # miss / ~30% slower on asicode with the default 256).  The set is the
    # scan list unioned with the uncapped graph.py_files — sizing here covers
    # both callers (gate + tool); ensure_capacity is monotonic, so the graph
    # build's earlier sizing and the registry's later one compose (P2 2026-08-11).
    parse_cache.ensure_capacity(len(py_files))
    try:
        refs = (
            set(imported_names)
            if imported_names is not None
            else compute_imported_names(repo_root, candidate_files)
        )
        # Scanner entry points resident in the registry (e.g.
        # ``scan_vulture_dead_code``) are alive by construction but have no
        # call edge and no ``from m import fn`` entry — they are passed to
        # ``register()`` as callbacks. Merge them up front so dead-code
        # scanners never flag a live scanner entry point.
        refs |= _scanner_resident_entry_points()
        rr = repo_root or ""
        for f in py_files:
            # 1. Caller edges (call graph) — catches function calls.
            for sym in (graph.get_symbols_in_file(f) or []):
                name = sym.name if hasattr(sym, "name") else (getattr(sym, "symbol_name", "") or "")
                if name and graph.get_callers(name):
                    refs.add(name)
            # 2. Importer files that do ``from <candidate module> import X``
            #    — catches classes/constants exported but never called.
            if has_importers:
                refs.update(_importer_exported_names(graph, rr, f))
        logger.debug(
            "[CROSS_FILE_REFS] light: %d referenced name(s) from %d Python file(s)",
            len(refs), len(py_files),
        )
    except Exception:
        logger.debug("[CROSS_FILE_REFS] light computation failed — staying conservative", exc_info=True)
        return None
    else:
        return refs


def _importer_exported_names(graph, repo_root: str, candidate_file: str) -> set:
    """Names other Python files import from *candidate_file* via ``from m import X``.

    Parses each importer the graph reports (``get_importers``), resolves
    relative imports against the importer's package, and records every name
    bound by a ``from <candidate module> import X [as Y]``.  Both X (the
    exported symbol) and Y (the local alias) are recorded so judging the
    original dead while an aliased import lives is impossible.

    Deliberately does NOT cover ``module.attr`` reads, and returns empty when
    the graph reports no importers — both are the seed channel's job
    (``compute_imported_names``), and the seed only covers the files its
    caller passes.  See the module docstring's CALLER CONTRACT: a caller that
    seeds from a partial file list loses that coverage here silently.
    """
    out: set = set()
    importers = graph.get_importers(candidate_file)
    if not importers:
        return out
    # Canonical path→module prefix (single source with the graph builder):
    # "pkg/__init__.py" → "pkg" so `from pkg import X` importer matches resolve
    # (B1, 2026-08-11 — the raw-path form produced "pkg.__init__", which no
    # import edge stores, silently dropping package re-export importers).
    module_prefix = path_to_module(candidate_file, repo_root)
    for importer in importers:
        abs_imp = importer if os.path.isabs(importer) else os.path.join(repo_root, importer)
        tree = parse_cache.parse_ast(abs_imp)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imp_src = node.module or ""
            level = node.level or 0
            if level > 0:
                imp_pkg = os.path.dirname(importer).replace(os.sep, ".").replace("/", ".")
                for _ in range(level - 1):
                    imp_pkg = imp_pkg.rsplit(".", 1)[0] if "." in imp_pkg else ""
                imp_src_abs = (imp_pkg + "." + imp_src).strip(".") if imp_src else imp_pkg
            else:
                imp_src_abs = imp_src
            if imp_src_abs == module_prefix or imp_src_abs.startswith(module_prefix + "."):
                for alias in node.names:
                    if alias.name and alias.name != "*":
                        out.add(alias.name)
                    if alias.asname:
                        out.add(alias.asname)
    return out
