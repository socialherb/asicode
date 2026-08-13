#!/usr/bin/env python3
"""Zero-tolerance gate for the deterministic structural scanners.

The scanner registry (external_llm/agent/scanner_registry.py) powers the MCP
``run_structural_scan`` tool.  Most of its scanners are ASSIST tools: they
emit candidates for human triage (ast_similarity near-duplicates, dead-block
clusters, vulture low-confidence hints) and are intentionally not gateable.
Seven scanners, however, are deterministic and currently at ZERO candidates
repo-wide, so a candidate from them is a regression by construction:

  - contradictory_logic_scanner  — contradictory conditions, unreachable
                                   branches, always-false assertions
  - duplicate_definition_scanner — same name+kind defined twice
  - unused_import_scanner        — import never referenced in its file
  - dead_block_scanner           — clusters of unused module-level private
                                   symbols (cross-file-reference aware)
  - public_dead_code_scanner     — unused public+private module-level symbols
                                   (cross-file reachability)
  - vulture_dead_code_scanner    — class-level dead code via Vulture
                                   (framework-dispatch false positives
                                   suppressed structurally)
  - broken_contract_scanner      — writer/reader pairs split by migration

These seven ARE the gate: any candidate fails the build.  No baseline — zero
is the floor (same contract as scripts/check_lint_full.py).  The remaining
scanners run in report-only mode (counts printed, never fail).

Why the four graph-dependent scanners were promoted (2026-08-08): the gate
script previously ran WITHOUT a call graph, so dead_block / public_dead_code
could not see cross-file references (43 false positives each, all referenced
from tests or sibling modules) and broken_contract always skipped
(graph_required_for_results=True).  RepositoryGraph.get_callers is now an
O(1) reverse index (fa857800), so building the graph once (~5s) and injecting
``cross_file_referenced_names`` + ``repo_graph`` makes those scanners
accurate enough to gate.  vulture's remaining 36 framework-dispatch false
positives (enum members, pydantic/dataclass fields, http.server protocol
methods, foreign-object attribute assignments) are suppressed structurally in
vulture_scanner._framework_live_for_file.

Usage:
    python scripts/check_structural_scanners.py
    python scripts/check_structural_scanners.py <file>.py ...  # check only given files
    python scripts/check_structural_scanners.py --gate-only   # full repo, gate scanners only

The ONLY flag is ``--gate-only``; any other ``--*`` argument fails with exit 1
(no silent ignore — a typo must not degrade into a ~35s full scan).

Explicit file args (pre-commit per-file mode) scan only those files and skip
report-only scanners (they are whole-repo signals).  No args scans the whole
repo including report-only counts — the full picture for manual/weekly drift
checks.  ``--gate-only`` runs the same full-repo scan but ONLY the seven
gate scanners, skipping the two remaining report-only ones (ast_similarity /
container_reachability — measured ~35s of wall time 2026-08-07).
Explicit paths that are ALL rejected (out-of-repo or non-scannable extension)
FAIL with exit 1 — they never silently fall back to a full-repo scan: that
fallback made a gate test pass /tmp files and burn ~35s per run (120s timeout
flakes under parallel-suite contention, 2026-08-07).
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from external_llm.analysis.scan_walk import (  # noqa: E402 — eager by design, see above
    SCAN_EXTS,
    SCAN_FILE_CAP,
    walk_scan_files,
)

# The on-disk graph cache format is owned by external_llm.graph.structural_cache
# (single source of truth): this gate is its only WRITER, while
# RepositoryGraph.build() READS the same JSON to warm its first build.  The
# shared module is imported LAZILY inside the cache functions (this script's
# established style — see _load_registry/_build_graph) to keep the module
# importable without external_llm and avoid E402.
#
# ONE deliberate exception: the scan-walk single source
# (external_llm/analysis/scan_walk.py) is imported EAGERLY below — it is
# stdlib-only, and the gate's _SCAN_* names must be identity aliases of its
# constants at module scope so tests can pin the unification.

# Deterministic scanners — a candidate here is a regression.  Keep this list
# minimal: every entry must be at zero candidates repo-wide at gate time.
# container_reachability_scanner is gated ONLY with graph injection: the
# scanner structurally skips containers whose name appears in
# cross_file_referenced_names (cross-file consumption is invisible to its
# single-file scope), so the gate passes the full cross-file ref set.
GATE_SCANNERS: tuple[str, ...] = (
    "contradictory_logic_scanner",
    "duplicate_definition_scanner",
    "unused_import_scanner",
    "dead_block_scanner",
    "public_dead_code_scanner",
    "vulture_dead_code_scanner",
    "broken_contract_scanner",
    "container_reachability_scanner",
)

# Non-deterministic / assist-only scanners — report-only.
# ast_similarity_scanner is a RANKING scanner: its count is capped by
# max_candidates (measured: 20 at the default cap, 100 at max_candidates=100),
# so "0 candidates" is not an enumeration contract — the top-N set churns with
# every code change.  Candidates are mostly test methods (idiomatic
# arrange/act/assert similarity) plus intentional mirror pairs
# (e.g. get_callers/get_callees), which are not regressions.  It stays
# report-only by design.
REPORT_ONLY_SCANNERS: tuple[str, ...] = ("ast_similarity_scanner",)

# The extension tuple / cap / walk are the scan-walk SINGLE SOURCE
# (external_llm/analysis/scan_walk.py): the _SCAN_* names below are identity
# aliases — no copies exist anywhere — and _walk_scan_files delegates to it
# (pinned by test_scan_walk_constants_are_single_source_aliases).  The skip
# set has no gate-side consumer (the walk lives in scan_walk.py), so it is
# NOT aliased here — the behavioral pin is test_gate_walk_matches_shared_walk.
#
# ── Language-coverage contract (pinned by test_every_scan_ext_is_covered_by_a_gate_scanner /
#    test_gate_language_coverage_map in tests/unit/test_check_structural_scanners.py) ──
# ``LanguageId.from_path`` maps these 8 extensions to 6 languages.  Coverage in
# the GATE is intentionally asymmetric:
#
#     .py            → PYTHON      → all 8 gate scanners (7 AST scanners + duplicate_definition)
#     .ts / .tsx     → TYPESCRIPT  → duplicate_definition only
#     .js / .jsx     → JAVASCRIPT  → duplicate_definition only
#     .go            → GO          → duplicate_definition only
#     .java          → JAVA        → duplicate_definition only
#     .kt            → KOTLIN      → duplicate_definition only
#
# Invariant: every extension here must map to a LanguageId covered by AT LEAST
# ONE gate scanner — there is no scanned-but-unjudged language.  The 7 other
# gate scanners are Python-only BY DESIGN: dead-code reachability, vulture and
# contract analysis have no reliable non-Python semantics, and the registry's
# ``supported_languages`` filter keeps their Python-only ASTs away from Go/TS
# source (the Go-repo false-positive regression, 2026-08-07).  A new
# non-Python scanner is therefore the ONLY way to widen non-Python coverage:
# adding an extension here without one makes the docstring a lie and fails
# test_every_scan_ext_is_covered_by_a_gate_scanner.
#
# Conditional-coverage caveat (non-Python only): the zero-candidate floor for
# TS/JS/Go/Java/Kotlin holds ONLY while the tree-sitter grammar for that
# language resolves (tree_sitter_utils._EXT_TO_GRAMMAR_KEY — DERIVED from
# _EXT_MAP — via tree_sitter_language_pack).  duplicate_definition_scanner degrades to ``[]``
# on a missing grammar (no AST fallback outside Python) — so main() probes
# the scanned non-Python languages with is_language_available() BEFORE
# scanning and FAILS the gate when one does not resolve (fail-closed,
# 2026-08-11: an unjudged language must not pass the zero-candidate floor;
# pinned by test_gate_fails_closed_on_missing_grammar).  The Python floor
# is absolute: the 7 AST scanners plus the ast-module fallback need no
# grammar at all.  (Grammars are a declared dependency; a grammar
# regression now fails this gate in addition to tests/unit/languages.)
#
# Sync obligations — the 6-language set is DERIVED, not hand-synced:
# SCAN_EXTS → SCAN_LANGUAGES in external_llm/analysis/scan_walk.py (the
# single scan-walk source; the gate's _SCAN_EXTS below and the registry's
# _TS_LANGUAGES are identity aliases of it).  The derivation resolves
# through _EXT_MAP (languages/models.py) with import-time fail-fast, and
# the no-scan boundary for its other judged-language extensions
# (_UNSCANNED_VARIANTS) is pinned by test_scan_ext_map_boundary_is_exhaustive.
# The extension → grammar-key map (tree_sitter_utils._EXT_TO_GRAMMAR_KEY) is
# DERIVED from the same _EXT_MAP too — domain = full-AST query languages, key
# = LanguageId value except the single .tsx override, import-time fail-fast
# against the query maps and _LANG_MODULE_MAP.  _LANG_MODULE_MAP itself is
# DERIVED as well (domain = full-AST keys | parse-only html/css, value =
# tree_sitter_<key> except the single tsx override) — see tree_sitter_utils.py.
# The remaining hand-maintained names that must agree (verified at runtime by
# test_gate_language_coverage_map and test_duplicate_definition_lang_keys_match_registry):
#   1. _LANG_TOP_LEVEL_NODES / _LANG_KIND_MAP — duplicate_definition_scanner.py
#      (import-time _validate_judge_maps: every emitted node type has a kind,
#      no stale kind entries, kind values in {function,class,assignment})
#   2. _LANG_DEF_NODES          — analysis/_dead_block_shared.py
#      (import-time _validate_lang_def_nodes: (kind, is_container,
#      skip_if_enclosed) shape; per-language superset of the duplicate judge's
#      node types — the only deliberate extras are go's package-level
#      const/var declarations, pinned in the gate test)
_SCAN_EXTS = SCAN_EXTS
# ── Cap semantics + scope difference vs the graph build (documented contract)
# Documented at external_llm/analysis/scan_walk.py (SCAN_FILE_CAP): this
# walk truncates at the cap in sorted order while RepositoryGraph.build()
# never does — the gate graph is always a superset of the scan list.  The
# one intentional skip-set divergence (env/ scanned but not graphed) is
# pinned by test_scan_walk_and_graph_skip_sets_diverge_on_env.  The
# cross-file-ref pass is fed the scan list UNION graph.py_files (the
# uncapped py list) in main(), so references beyond the cap still suppress
# dead-code candidates — the cap truncates only the SCANNED candidate set,
# never the ref set (soundness gap closed 2026-08-11).  Even so, candidates
# beyond the cap are never PRODUCED — dead code in an unscanned file would
# pass the zero-candidate floor silently.  main() therefore FAILS the gate
# when the walk truncates (fail-closed, 2026-08-11; _files_beyond_cap +
# test_main_fails_closed_when_scan_list_truncated): an unjudged file is the
# same soundness class as the missing-grammar guard below, and a ⚠-only
# warning is CI-green-by-default — the silent-pass mode this guard exists
# to kill.  The documented remedy is raising SCAN_FILE_CAP or one walk.
_SCAN_FILE_CAP = SCAN_FILE_CAP


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    """Normalize explicit file args to repo-relative paths.

    Returns None when no file args survive (or none given).  The caller
    distinguishes the two: no args → full-repo scan; args given but all
    rejected → fail-closed error (main()).  pre-commit passes absolute
    in-repo paths; lint.yml passes none — both normalize to the same
    repo-relative key space as the full scan.
    """
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(_SCAN_EXTS) and not rel.startswith(".."):
            out.append(rel)
    return out or None


def _walk_scan_files(root: Path) -> list[str]:
    """Collect scannable source files under *root* (repo-relative paths).

    Delegates to ``scan_walk.walk_scan_files`` — the single source of the
    scan-walk contract (extensions, skip set, cap, determinism; see
    external_llm/analysis/scan_walk.py).  The cap is read from
    ``_SCAN_FILE_CAP`` at call time so tests can pin truncation.
    """
    return walk_scan_files(root, cap=_SCAN_FILE_CAP)


def _files_beyond_cap(file_paths: list[str], root: Path) -> int:
    """Scannable files the capped walk dropped (0 = not truncated).

    ``walk_scan_files`` returns early the moment it holds ``cap`` files, so a
    list of exactly ``_SCAN_FILE_CAP`` entries is ambiguous: an exactly-cap
    repo looks identical to a truncated one.  This counts the TRUE total with
    an uncapped walk — paid ONLY when truncation is possible (``len == cap``,
    so below-cap repos cost nothing) — and returns the overshoot, 0 when the
    repo is at or below the cap.  The gate then fails on truncation: files
    beyond the cap are UNJUDGED, and an unjudged file must not pass the
    zero-candidate floor (see the cap-semantics comment at _SCAN_FILE_CAP).
    """
    if len(file_paths) < _SCAN_FILE_CAP:
        return 0
    total = len(walk_scan_files(root, cap=sys.maxsize))
    return max(0, total - len(file_paths))


def _unjudged_languages(file_paths: list[str]) -> list[tuple[str, list[str]]]:
    """Non-Python languages in *file_paths* whose tree-sitter grammar does not resolve.

    duplicate_definition_scanner judges every non-Python scanned language
    through tree-sitter ALONE and degrades to ``[]`` on a missing grammar (no
    AST fallback outside Python), so a missing grammar would let that language
    pass the gate UNJUDGED.  ``main()`` fails the gate when this returns
    non-empty (fail-closed, 2026-08-11).  Python is never probed — its floor
    is the seven AST scanners plus the ast-module fallback, which need no
    grammar — and UNKNOWN extensions are never scanned (_SCAN_EXTS bounds
    both the walk and per-file normalization).

    Language ids are derived EXACTLY as the scanner derives them
    (``LanguageId.from_path(...).value``), so this names the grammar the judge
    would parse with (.tsx → typescript, not the separate tsx grammar key).
    The only divergence from the derived ``_EXT_TO_GRAMMAR_KEY`` is that one
    (.tsx → "tsx"); both keys resolve from the same module, pinned by
    test_unjudged_language_probe_matches_grammar_key_modules.
    """
    from external_llm.languages import LanguageId
    from external_llm.languages.tree_sitter_utils import is_language_available

    present: dict[str, list[str]] = {}
    for rel in file_paths:
        lid = LanguageId.from_path(rel)
        if lid is LanguageId.PYTHON or lid is LanguageId.UNKNOWN:
            continue
        present.setdefault(lid.value, []).append(rel)
    return [
        (lang, files)
        for lang, files in sorted(present.items())
        if not is_language_available(lang)
    ]


def _run_scanner(registry, name: str, file_paths: list[str], **kwargs) -> int:
    """Run one scanner, print its outcome, return candidate count."""
    try:
        result = registry.run(name, repo_root=str(REPO), file_paths=file_paths, **kwargs)
    except Exception as exc:  # fail-closed: a broken scanner is not a pass
        print(f"❌ {name}: SCANNER ERROR — {exc!r}")
        return -1
    total = result.total_candidates
    print(f"  {name}: {total} candidate(s) across {len(result.affected_files)} file(s)")
    for c in result.candidates_raw[:10]:
        print(f"      {str(c)[:220]}")
    if total > 10:
        print(f"      … and {total - 10} more")
    return total


def _load_registry():
    """Load the scanner registry — fail-closed (None only on import error)."""
    try:
        from external_llm.agent.scanner_registry import get_registry

        return get_registry()
    except Exception as exc:  # fail-closed: registry must load
        print(f"❌ scanner registry failed to load — {exc!r}")
        return None


_CACHE_DIR = REPO / ".cache"
_CACHE_PATH = _CACHE_DIR / "structural_graph_v1.json"


def _build_graph():
    """Build the repository call graph (shared by all scanners).

    The four graph-dependent gate scanners (dead_block / public_dead_code /
    vulture / broken_contract) share one graph; building it once here keeps
    the gate's total cost at one full extraction + one cross-ref pass
    instead of four.

    Pipeline integration (2026-08-11): the whole build — per-file extraction,
    the in-process and on-disk cache tiers, and cache persistence — is
    delegated to ``RepositoryGraph.build(collect_imported_names=True)``.
    That mode ALSO computes the per-file cross-file imported-name sets
    (``extract_imported_names_for_file``) the dead-code scanners consume,
    and rewrites ``.cache/structural_graph_v1.json`` with the COMPLETE
    payload (files + manifest + imported_names) when any file was re-parsed,
    so the next run in any process reuses this build instead of re-parsing
    (a commit-touch build goes from ~4.6s + ~10s cross-ref parse to well
    under 1s).  Cache corruption/version mismatch fails open to a full
    build; non-Python files are always re-processed via ripgrep.  The
    graph's walk order is the single injection order, so cache-served
    builds are bit-for-bit identical to full builds by construction — the
    historical manifest-order re-injection loop that lived here moved into
    the graph and no longer needs to be kept in sync.
    """
    from external_llm.graph.repository_graph import RepositoryGraph

    graph = RepositoryGraph(str(REPO), cache_path=_CACHE_PATH)
    graph.build(collect_imported_names=True)
    return graph, {
        "hit": graph.cache_stats["hit"],
        "total": graph.cache_stats["total"],
        "changed": graph.cache_stats["changed"],
        "imported_names": graph.imported_names,
    }


def _compute_cross_refs(graph, file_paths, imported_names=None):
    """Cross-file referenced names for *file_paths* (None when graph unusable)."""
    from external_llm.analysis.cross_file_refs import (
        compute_cross_file_referenced_names_light,
    )

    return compute_cross_file_referenced_names_light(graph, str(REPO), file_paths, imported_names)


def main() -> int:
    argv = sys.argv[1:]
    known_flags = ("--gate-only",)
    # Fail-closed on unknown flags (2026-08-11): an unknown --flag was
    # previously ignored, so a typo like ``--gate-onyl`` silently ran the FULL
    # ~35s scan and dropped the gate-only intent — the report-only section was
    # skipped (gate-only mode) or paid for (full mode), and CI cost inflated
    # with no signal.  A flag is a caller contract; a typo must fail loudly.
    unknown = [a for a in argv if a.startswith("--") and a not in known_flags]
    if unknown:
        print("❌ unknown flag(s): " + ", ".join(unknown))
        print("   supported: " + ", ".join(known_flags) + "  (file paths are positional)")
        return 1
    gate_only = "--gate-only" in argv
    args = [a for a in argv if not a.startswith("--")]
    paths = _resolve_scan_paths(args)
    if args and paths is None:
        # Explicit file args were given but NONE survived normalization
        # (out-of-repo or non-scannable extension).  Fail closed — a silent
        # fallback to the full-repo scan would burn ~35s on a caller mistake
        # (and has: a gate test passed /tmp files and hit 120s timeout flakes
        # under parallel-suite contention).
        print("❌ no scannable files among the given paths: " + ", ".join(repr(a) for a in args))
        print("   (paths must resolve inside the repo and end in one of: " + ", ".join(_SCAN_EXTS) + ")")
        return 1
    full_scan = paths is None
    file_paths = paths if paths is not None else _walk_scan_files(REPO)
    if not file_paths:
        print("⚠ no scannable source files found")
        return 1

    # ── Cap-truncation guard (fail-closed, 2026-08-11) ─────────────────────
    # The walk truncates at _SCAN_FILE_CAP in sorted order, so on a repo above
    # the cap the gate would judge only its first-N files while still printing
    # "0 candidates" — unjudged files silently passing the zero-candidate
    # floor, the same soundness class as the missing-grammar guard below.  A
    # warning alone was rejected: CI greens with ⚠ lines are how the silent
    # unknown-flag bug (B2) went unnoticed.  Fail BEFORE any work; the
    # documented remedy is raising SCAN_FILE_CAP or sharing one uncapped walk.
    if full_scan:
        beyond = _files_beyond_cap(file_paths, REPO)
        if beyond:
            print(
                f"❌ scan list truncated at cap {_SCAN_FILE_CAP} — {beyond} scannable "
                "file(s) beyond the cap were NOT scanned"
            )
            print("   the zero-candidate floor is repo-wide; unjudged files must not pass the gate")
            print("   fix: raise SCAN_FILE_CAP in external_llm/analysis/scan_walk.py or share one uncapped walk")
            return 1

    registry = _load_registry()
    if registry is None:
        return 1

    failed = False
    mode = f"{'gate-only ' if gate_only else ''}{'full repo' if full_scan else f'{len(file_paths)} file(s)'}"
    print(f"Structural scanner gate — {mode} ({len(file_paths)} files scanned)")

    # ── Grammar availability (fail-closed) ──────────────────────────────────
    # The non-Python zero-candidate floor is conditional on the tree-sitter
    # grammar resolving: duplicate_definition_scanner degrades to ``[]`` on a
    # missing grammar (no AST fallback outside Python), so an unavailable
    # grammar would let that language pass the gate UNJUDGED.  Probe the
    # languages present in the scan list BEFORE scanning and fail the gate on
    # a gap — see the conditional-coverage caveat at _SCAN_EXTS.
    unjudged = _unjudged_languages(file_paths)
    if unjudged:
        for lang, files in unjudged:
            shown = ", ".join(files[:3]) + ("…" if len(files) > 3 else "")
            print(
                f"❌ tree-sitter grammar for '{lang}' does not resolve — {len(files)} "
                f"scanned file(s) ({shown}) would be UNJUDGED by "
                "duplicate_definition_scanner"
            )
        print("   install the grammar (e.g. pip install tree-sitter-<lang>) or remove those")
        print("   files from the scan; fail-closed: a missing grammar must not pass the gate.")
        failed = True
    else:
        from external_llm.languages import LanguageId

        non_py = sorted({LanguageId.from_path("f" + ext).value for ext in _SCAN_EXTS} - {"python"})
        print(f"  [grammar check] all non-Python grammars resolve ({', '.join(non_py)})")

    # ── Shared call graph + cross-file references ────────────────────────────
    # The four graph-dependent gate scanners (dead_block / public_dead_code /
    # vulture / broken_contract) need the call graph and cross-file reference
    # set to avoid the cross-file false positives that kept them out of the
    # gate when the scan ran standalone.  Built ONCE here, injected into every
    # scanner that declares the capability (mirrors analysis_tools.py's
    # RUN_SCANNER handler: cross_file_referenced_names via input_schema,
    # repo_graph via requires_graph).
    needs_graph = any(
        spec is not None
        and (getattr(spec, "requires_graph", False) or "cross_file_referenced_names" in (spec.input_schema or {}))
        for spec in (registry.get_spec(n) for n in GATE_SCANNERS)
    )
    graph = None
    cross_refs = None
    if needs_graph:
        try:
            graph, cache_stats = _build_graph()
            # Cross-file refs are a WHOLE-REPO fact, not a property of the
            # candidate set: a name referenced by ANY file (e.g. ``logger``
            # via ``bjm.logger`` attr read) is live everywhere, so the ref
            # set must be computed over the full file list even in per-file
            # (pre-commit) mode — computing it over only the changed files
            # makes public_dead_code false-positive on names whose only
            # cross-file reference lives in an unchanged file
            # (measured 2026-08-08: config.py logger, subprocess_utils
            # CANCEL_POLL_INTERVAL).
            #
            # The ref input is the scan list UNION the graph's uncapped py
            # list (graph.py_files): the walk truncates at SCAN_FILE_CAP
            # while the graph build never does, and a reference living only
            # in a file beyond the cap would otherwise false-positive the
            # dead-code scanners (soundness gap closed 2026-08-11).  In
            # full-repo mode the scan list is already in hand (file_paths),
            # so no second walk is paid — one walk per mode, as before.
            ref_input = file_paths if full_scan else _walk_scan_files(REPO)
            cross_refs = _compute_cross_refs(
                graph,
                sorted(set(ref_input) | set(graph.py_files)),
                cache_stats["imported_names"],
            )
            print(
                f"  [call graph] {len(graph.call_edges)} edges, "
                f"cross-file refs: {len(cross_refs) if cross_refs else 'unavailable'}"
            )
            if cache_stats["total"]:
                from external_llm.graph.structural_cache import CACHE_VERSION

                print(
                    f"  [graph cache] {cache_stats['hit']}/{cache_stats['total']} files "
                    f"reused ({cache_stats['changed']} re-parsed, format v{CACHE_VERSION})"
                )
        except Exception as exc:  # fail-closed: graph is required, not optional
            print(f"❌ call graph build failed — {exc!r}")
            return 1

    for name in GATE_SCANNERS:
        spec = registry.get_spec(name)
        if spec is None:
            print(f"❌ {name}: NOT REGISTERED — gate would silently pass")
            failed = True
            continue
        kwargs: dict = {}
        if cross_refs is not None and "cross_file_referenced_names" in (spec.input_schema or {}):
            kwargs["cross_file_referenced_names"] = cross_refs
        if getattr(spec, "requires_graph", False):
            kwargs["repo_graph"] = graph
        count = _run_scanner(registry, name, file_paths, **kwargs)
        if count > 0 or count < 0:
            failed = True

    if full_scan and not gate_only:
        print("Report-only scanners (not gated — see docstring):")
        for name in REPORT_ONLY_SCANNERS:
            if registry.get_spec(name) is None:
                print(f"  {name}: not available")
                continue
            _run_scanner(registry, name, file_paths)

    if failed:
        if unjudged:
            print(
                "\n❌ tree-sitter grammar gap — non-Python language(s) cannot be judged; "
                "install the grammar(s) listed above and re-run."
            )
        else:
            print(
                "\n❌ Deterministic structural scanner(s) found candidates — this is a "
                "regression. Fix the code; do NOT add a baseline."
            )
        return 1
    print("\n✅ deterministic structural scanners: 0 candidates (no baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
