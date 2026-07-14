"""Cross-file referenced-name computation for dead-code scanners.

Builds the ``cross_file_referenced_names`` set that gates public-symbol
dead-code detection (``dead_block_scanner`` / ``public_dead_code_scanner`` /
``container_reachability_scanner``).  A name lands in the set when there is
ANY evidence it is referenced outside its defining file:

  1. **Call edges** — the repository graph reports ≥1 caller.
  2. **Import-based exports** — another file does
     ``from <candidate module> import <name>`` (call edges miss constants,
     classes used as types, etc.).  The ORIGINAL name is recorded even when
     the importer aliases it (``import X as Y`` references X, not Y).  When
     the graph has no import edges for a file, same-package siblings are
     scanned directly as a fallback.
  3. **Module-attribute reads** — another file does ``import <candidate
     module>`` (or ``from <pkg> import <module>``) and reads
     ``module.<name>``.  Constants accessed this way (config flags, etc.)
     have no call edge and no ImportFrom entry, so they need their own pass.
  4. **Imported names** — names the candidate files themselves import; these
     are defined elsewhere and must never be judged dead here.

TypeScript/JavaScript candidates are covered via tree-sitter named-import
extraction (``import { X as Y } from './mod'`` → X and Y); Python via ast.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Optional

from ..languages import LanguageId
from ..languages.tree_sitter_utils import extract_import_names as _ts_extract_import_names
from . import parse_cache

logger = logging.getLogger(__name__)

_TS_LANGS = (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT)


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


def _add_ts_import_names(abs_path: str, lang: LanguageId, names: set) -> None:
    """Add every name a TS/JS file imports (original + alias) to *names*."""
    try:
        with open(abs_path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return
    for _module, name in _ts_extract_import_names(content, lang.value):
        if name:
            names.add(name)


def compute_imported_names(repo_root: str, file_paths: list[str]) -> set:
    """One-pass set of every name imported or module-attr-read in *file_paths*.

    ``from m import X as Y`` → X and Y, ``import a.b as c`` → c, and
    ``import m; m.X`` → X.  TS/JS files contribute their named imports via
    tree-sitter.  A name in this set is referenced by *some* file, so
    dead-code scanners must never judge it dead — this catches cross-file
    imports that single-file Load counting cannot see.  Coarser than the
    module-precise matching in ``compute_cross_file_referenced_names`` (any
    import of the same name anywhere counts), trading a little
    under-detection for O(n).
    """
    names: set = set()
    rr = repo_root or ""
    for f in file_paths or []:
        abs_f = f if os.path.isabs(f) else os.path.join(rr, f)
        lang = LanguageId.from_path(f)
        if lang in _TS_LANGS:
            _add_ts_import_names(abs_f, lang, names)
            continue
        tree = parse_cache.parse_ast(abs_f)
        if tree is None:
            continue
        bindings = _add_python_import_names(tree, names)
        _add_module_attr_reads(tree, bindings, names)
    return names


def compute_cross_file_referenced_names_light(
    graph,
    repo_root: str,
    candidate_files: list[str],
) -> Optional[set]:
    """O(n) variant of ``compute_cross_file_referenced_names`` for tool paths.

    Combines repo-wide imported names (one parse pass, cached) with graph
    caller edges (in-memory lookups).  For each candidate file it also scans
    the importer files the graph reports (``get_importers``) for
    ``from <candidate module> import X`` patterns and records X — this is the
    signal call edges miss entirely (constants, classes-used-as-types,
    functions imported but never called by name in the importer).  Without
    this pass public symbols exported only via ``from m import X`` would be
    falsely reported as dead (e.g. ``detect_cloud_provider``).  Relative
    imports and ``module.attr`` reads remain the full version's job.

    Scanner-registry-resident entry points (``scan_*`` callables passed to
    ``ScannerRegistry.register``) are merged up front — they are alive by
    construction but invisible to both caller edges and import analysis.

    Returns None when *graph* is unusable — same contract as the full
    version: without caller edges, ``module.attr`` usage of public symbols is
    invisible, so unlocking public-symbol detection would be unsound.
    """
    if graph is None or not candidate_files:
        return None
    if not (hasattr(graph, "get_symbols_in_file") and hasattr(graph, "get_callers")):
        return None
    has_importers = hasattr(graph, "get_importers")
    try:
        refs = compute_imported_names(repo_root, candidate_files)
        # Scanner entry points resident in the registry (e.g.
        # ``scan_vulture_dead_code``) are alive by construction but have no
        # call edge and no ``from m import fn`` entry — they are passed to
        # ``register()`` as callbacks. Merge them up front so dead-code
        # scanners never flag a live scanner entry point.
        refs |= _scanner_resident_entry_points()
        rr = repo_root or ""
        for f in candidate_files:
            # 1. Caller edges (call graph) — catches function calls.
            for sym in (graph.get_symbols_in_file(f) or []):
                name = sym.name if hasattr(sym, "name") else (getattr(sym, "symbol_name", "") or "")
                if name and graph.get_callers(name):
                    refs.add(name)
            # 2. Importer files that do ``from <candidate module> import X``
            #    — catches classes/constants exported but never called.
            if has_importers and LanguageId.from_path(f) is LanguageId.PYTHON:
                refs.update(_importer_exported_names(graph, rr, f))
        logger.debug(
            "[CROSS_FILE_REFS] light: %d referenced name(s) from %d file(s)",
            len(refs), len(candidate_files),
        )
        return refs
    except Exception:
        logger.debug("[CROSS_FILE_REFS] light computation failed — staying conservative", exc_info=True)
        return None


def _importer_exported_names(graph, repo_root: str, candidate_file: str) -> set:
    """Names other Python files import from *candidate_file* via ``from m import X``.

    A focused, O(importers) slice of the full version's step 2: it parses each
    importer the graph reports (``get_importers``), resolves relative imports
    against the importer's package, and records every name bound by a
    ``from <candidate module> import X [as Y]``.  Both X (the exported symbol)
    and Y (the local alias) are recorded so judging the original dead while an
    aliased import lives is impossible.  ``module.attr`` reads and the
    same-package sibling fallback stay in the full version.
    """
    out: set = set()
    importers = graph.get_importers(candidate_file)
    if not importers:
        return out
    module_prefix = candidate_file.replace("/", ".").replace("\\", ".")
    if LanguageId.from_path(module_prefix) is LanguageId.PYTHON:
        module_prefix = module_prefix[:-3]
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


def _collect_ts_candidate_refs(
    graph, rr: str, f: str, refs: set,
) -> None:
    """Add names other TS/JS files import from candidate *f* to *refs*."""
    cand_noext = os.path.splitext(f)[0]
    importers = graph.get_importers(f) if hasattr(graph, "get_importers") else []
    for importer in importers:
        imp_lang = LanguageId.from_path(importer)
        if imp_lang not in _TS_LANGS:
            continue
        abs_imp = importer if os.path.isabs(importer) else os.path.join(rr, importer)
        try:
            with open(abs_imp, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue
        imp_dir = os.path.dirname(importer)
        for module, name in _ts_extract_import_names(content, imp_lang.value):
            if not name:
                continue
            resolved = os.path.normpath(os.path.join(imp_dir, module)) if module.startswith(".") else os.path.normpath(module)
            if resolved == cand_noext:
                refs.add(name)


def compute_cross_file_referenced_names(
    graph,
    repo_root: str,
    candidate_files: list[str],
) -> Optional[set]:
    """Return the cross-file referenced-name set, or None when unavailable.

    Returning None keeps downstream scanners in conservative private-only
    mode, so callers can pass the result through unconditionally.  None is
    returned when *graph* is falsy, lacks the caller-query API (a partial
    graph would unlock public-symbol deletion without real reachability
    evidence), or the computation fails.

    Scanner-registry-resident entry points (``scan_*`` callables passed to
    ``ScannerRegistry.register``) are seeded up front — they are alive by
    construction but have no call edge and no ``from m import fn`` entry
    (``register`` takes them as callback arguments).
    """
    if graph is None or not candidate_files:
        return None
    if not (hasattr(graph, "get_symbols_in_file") and hasattr(graph, "get_callers")):
        logger.debug(
            "[CROSS_FILE_REFS] graph lacks symbol/caller API (%s) — staying conservative",
            type(graph).__name__,
        )
        return None

    try:
        refs: set = set()
        # Scanner entry points resident in the registry are alive by
        # construction (dispatched via RUN_SCANNER) yet invisible to the
        # call-graph and import passes below — ``register(name, fn)`` passes
        # them as callback arguments. Seed them here so neither pass can flag
        # a live scanner entry point (e.g. ``scan_dead_blocks``) as dead.
        refs |= _scanner_resident_entry_points()

        # 1. Names with ≥1 caller edge in the graph.
        for f in candidate_files:
            for sym in (graph.get_symbols_in_file(f) or []):
                name = sym.name if hasattr(sym, "name") else (getattr(sym, "symbol_name", "") or "")
                if name and graph.get_callers(name):
                    refs.add(name)

        # 2. Names exported from candidate files via `from <module> import X`
        #    (or `module.X` attribute reads) in other files.  get_callers()
        #    only finds function-call edges.
        rr = repo_root or ""
        for f in candidate_files:
            if LanguageId.from_path(f) in _TS_LANGS:
                _collect_ts_candidate_refs(graph, rr, f, refs)
                continue
            module_prefix = f.replace("/", ".").replace("\\", ".")
            if LanguageId.from_path(module_prefix) is LanguageId.PYTHON:
                module_prefix = module_prefix[:-3]
            # `from <parent_pkg> import <module>` binds the module object —
            # attribute reads on that binding reference candidate symbols.
            _parent_pkg, _, _module_base = module_prefix.rpartition(".")
            importers = graph.get_importers(f) if hasattr(graph, "get_importers") else []
            # Graph import edges may be empty — scan same-package siblings directly.
            if not importers and rr:
                pkg_dir = os.path.join(rr, os.path.dirname(f))
                try:
                    if os.path.isdir(pkg_dir):
                        importers = [
                            os.path.relpath(os.path.join(pkg_dir, fn), rr)
                            for fn in os.listdir(pkg_dir)
                            if LanguageId.from_path(fn) is LanguageId.PYTHON
                            and fn != os.path.basename(f)
                        ]
                except OSError:
                    pass
            for importer in importers:
                abs_imp = importer if os.path.isabs(importer) else os.path.join(rr, importer)
                tree = parse_cache.parse_ast(abs_imp)
                if tree is None:
                    continue
                module_bindings: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == module_prefix:
                                module_bindings.add(alias.asname or alias.name)
                        continue
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    imp_src = node.module or ""
                    level = node.level or 0
                    if level > 0:
                        # Relative import: rebuild the absolute module path
                        # from the importer's package directory.
                        imp_pkg = os.path.dirname(importer).replace(os.sep, ".").replace("/", ".")
                        for _ in range(level - 1):
                            imp_pkg = imp_pkg.rsplit(".", 1)[0] if "." in imp_pkg else ""
                        imp_src_abs = (imp_pkg + "." + imp_src).strip(".") if imp_src else imp_pkg
                    else:
                        imp_src_abs = imp_src
                    if imp_src_abs == module_prefix or imp_src_abs.startswith(module_prefix + "."):
                        for alias in node.names:
                            # The name defined in the candidate module is
                            # alias.name; alias.asname is only the
                            # importer-local binding.  Record both — judging
                            # the original dead while an alias import lives
                            # is the worse failure.
                            if alias.name and alias.name != "*":
                                refs.add(alias.name)
                            if alias.asname:
                                refs.add(alias.asname)
                    if _parent_pkg and imp_src_abs == _parent_pkg:
                        for alias in node.names:
                            if alias.name == _module_base:
                                module_bindings.add(alias.asname or alias.name)
                if module_bindings:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Attribute):
                            chain = _dotted_chain(node.value)
                            if chain and chain in module_bindings:
                                refs.add(node.attr)

        # 3. Names the candidate files import — defined elsewhere, never dead here.
        for f in candidate_files:
            abs_f = f if os.path.isabs(f) else os.path.join(rr, f)
            lang = LanguageId.from_path(f)
            if lang in _TS_LANGS:
                _add_ts_import_names(abs_f, lang, refs)
                continue
            tree = parse_cache.parse_ast(abs_f)
            if tree is None:
                continue
            _add_python_import_names(tree, refs)

        logger.debug(
            "[CROSS_FILE_REFS] %d referenced name(s) from %d candidate file(s)",
            len(refs), len(candidate_files),
        )
        return refs
    except Exception:
        logger.debug("[CROSS_FILE_REFS] computation failed — staying conservative", exc_info=True)
        return None
