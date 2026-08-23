#!/usr/bin/env python3
"""Zero-tolerance gate for the deterministic structural scanners.

The scanner registry (external_llm/agent/scanner_registry.py) powers the MCP
``run_structural_scan`` tool.  Most of its scanners are ASSIST tools: they
emit candidates for human triage (ast_similarity near-duplicates, dead-block
clusters, vulture low-confidence hints) and are intentionally not gateable.
Nine scanners, however, are deterministic and currently at ZERO candidates
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
  - container_reachability_scanner — containers whose name has no cross-file
                        reference (graph-injected)
  - ast_similarity_scanner       — exact duplicates ONLY: pairs with similarity
                        1.0 (identical normalised structure) in
                        non-test source

  These nine ARE the gate: any candidate fails the build.  The floor is zero —
  with ONE scoped exception (2026-08-19): the five REFERENCE-DEPENDENT
  scanners (see BASELINE_ALLOWED_SCANNERS) judge "is this symbol referenced?",
  a fact that depends on WHICH FILES EXIST in the scanned tree.  The public
  snapshot (scripts/export_public.py) ships a strict subset, so symbols whose
  only consumers live in excluded files (webapp/, tools/, tasks/,
  webapp-coupled tests) become "unreferenced" there while the full tree keeps
  them live.  An exported tree therefore carries
  scripts/structural_scanner_baseline.txt — machine-generated at export,
  every entry verified to be referenced from an excluded file — and those
  entries are suppressed, never hand-written.  The other four scanners judge
  single-file facts (contradictory_logic / duplicate_definition /
  unused_import) or intra-tree pair identity (ast_similarity exact-1.0) that
  file exclusion cannot fabricate: zero tolerance, no baseline, in every
  tree.  ast_similarity is gated at a narrower contract than the others (only
  exact-1.0 non-test pairs; see GATE_SCANNERS note) — the ranked
  near-duplicate set is not an enumeration contract and stays report-only.

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
    python scripts/check_structural_scanners.py --gate-only \
        --dump-candidates out.json  # write every gate-scanner candidate as JSON,
                                    # exit 0 on candidates (export-time baseline
                                    # generation — scripts/export_public.py)

    The flags are ``--gate-only`` and the value-flag ``--dump-candidates
    <path>``; any other ``--*`` argument fails with exit 1 (no silent ignore —
    a typo must not degrade into a ~35s full scan).  ``--dump-candidates``
    runs the identical pipeline but RECORDS the raw pre-baseline identity of
    every candidate instead of failing on them — judgment belongs to the
    caller (export_public.py baseline generation, tests); machinery failures
    still exit 1 because an incomplete dump must never be consumed.

Explicit file args (pre-commit per-file mode) scan only those files and skip
report-only scanners (they are whole-repo signals).  Since 2026-08-16 (P-2)
per-file mode ALSO unions the git-untracked scannable files: pre-commit passes
only STAGED paths, so a new file with a violation would otherwise pass the gate
until its first commit while the full-scan floor (CI) caught it — per-file mode
must judge the same zero-candidate floor as the full scan.  No args scans the
whole repo including report-only counts — the full picture for manual/weekly
drift checks.  ``--gate-only`` runs the same full-repo scan but skips the
report-only section entirely.  Wall time measured 2026-08-21 (P14-2,
commit 394164d0): ~50s COLD (fresh repo/CI checkout — no .cache/, graph
build + every scanner's per-file cache miss) vs ~15s WARM (local,
.cache/ present).  The parse_cache byte-budget fix (256→384MiB) removed
mid-pass LRU eviction that made every later scanner re-parse from
scratch — the win shows in fresh PROCESSES over a warm .cache/ (test
runs: full-scan gate test 48.66s → 10.27s); a truly cold checkout still
pays one graph build + one vulture scan.  lint.yml restores .cache/
via actions/cache (P8-1, cb0a7fe3), so
CI pays the cold price only on a cache miss; a source-touching push still
misses for the touched files (fingerprint (mtime_ns, size) mismatch) and
self-heals on the next run.
Explicit paths that are ALL rejected (out-of-repo or non-scannable extension)
FAIL with exit 1 — they never silently fall back to a full-repo scan: that
fallback made a gate test pass /tmp files and burn ~35s per run (120s timeout
flakes under parallel-suite contention, 2026-08-07).
"""

import os
import subprocess
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
#
# ast_similarity_scanner is gated at a NARROWER contract than the others:
# only similarity-1.0 pairs (identical normalised structure) in NON-TEST
# source fail the build.  The ranked near-duplicate set is not gateable (see
# REPORT_ONLY_SCANNERS note), but 1.0 is a deterministic statement — two
# symbols with identical normalised structure are duplicates by construction,
# and the ranked cap cannot hide them once max_candidates is raised.  Test
# files are excluded: test methods are idiomatically similar (arrange/act/
# assert), 994 exact pairs measured 2026-08-16, and they are not regressions.
GATE_SCANNERS: tuple[str, ...] = (
    "contradictory_logic_scanner",
    "duplicate_definition_scanner",
    "unused_import_scanner",
    "dead_block_scanner",
    "public_dead_code_scanner",
    "vulture_dead_code_scanner",
    "broken_contract_scanner",
    "container_reachability_scanner",
    "ast_similarity_scanner",
)

# Non-deterministic / assist-only scanners — report-only.  Currently EMPTY:
# ast_similarity_scanner was promoted to the gate (2026-08-16, P-3) at the
# similarity-1.0 non-test contract; container_reachability_scanner is already
# in GATE_SCANNERS.  The ranked ast_similarity near-duplicate set stays
# unjudged by design:
# ast_similarity_scanner is a RANKING scanner: its count is capped by
# max_candidates (measured: 20 at the default cap, 100 at max_candidates=100),
# so "0 candidates" is not an enumeration contract — the top-N set churns with
# every code change.  Candidates are mostly test methods (idiomatic
# arrange/act/assert similarity), which are not regressions.  That ranked set
# remains report-only; only the exact-1.0 subset is gated.
REPORT_ONLY_SCANNERS: tuple[str, ...] = ()

# ── Export-artifact baseline (reference-dependent scanners only) ───────────
# These five judge symbol LIVENESS — "does any file reference this name?" —
# which is a function of WHICH FILES EXIST in the scanned tree.  The public
# snapshot excludes 170+ tracked files (webapp/, tools/, tasks/ and
# webapp-coupled tests); a symbol whose only consumers live there is live in
# the full tree yet "unreferenced" in the shipped subset.  Measured v0.2.25:
# private 991 files → 0 candidates, exported 814 files → 24 (dead_block 1,
# public_dead_code 12, vulture 11), every single one referenced only from
# excluded files.  scripts/export_public.py regenerates
# scripts/structural_scanner_baseline.txt at every export after
# machine-verifying each entry that way; a baseline entry naming any OTHER
# gate scanner fails the gate (_load_baseline) — their facts are single-file
# or intra-tree and cannot be fabricated by exclusion, so baselining them
# could only mask real regressions.
BASELINE_ALLOWED_SCANNERS: tuple[str, ...] = (
    "dead_block_scanner",
    "public_dead_code_scanner",
    "vulture_dead_code_scanner",
    "broken_contract_scanner",
    "container_reachability_scanner",
)
_BASELINE_PATH = REPO / "scripts" / "structural_scanner_baseline.txt"

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


def _untracked_scan_files(root: Path) -> list[str]:
    """Scannable git-untracked files under *root* (repo-relative paths).

    pre-commit hands the per-file gate only STAGED paths, so a new (untracked)
    file with a violation passed the gate until its first commit while the
    full-scan floor (CI) caught it — main() unions this set so per-file mode
    and full-scan mode judge the same zero-candidate floor (P-2, 2026-08-16).

    Best-effort: any git failure (no git binary, *root* outside a worktree)
    returns [] — untracked files cannot be enumerated without git, and the
    only production caller (pre-commit) always runs inside a worktree, so the
    misscan window does not exist there.  ``--exclude-standard`` respects
    .gitignore, so ignored scratch files are never scanned.  ``*_probe.py`` is
    excluded: the gate-mechanics tests write intentional-violation probes
    inside the repo and unlink them in finally, and a sibling worker's
    in-flight probe must not trip the zero-candidate floor (mirrors
    test_full_scan_is_zero_on_gate_scanners).  Entries resolve through the
    worktree root (``git ls-files`` paths are worktree-root-relative, which
    may be an ancestor of *root*) and are kept only when inside *root*.
    """
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if top.returncode != 0 or not top.stdout.strip():
            return []
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if listing.returncode != 0:
        return []
    toplevel = Path(top.stdout.strip())
    out: list[str] = []
    for raw in listing.stdout.split("\0"):
        if not raw or not raw.endswith(_SCAN_EXTS) or raw.endswith("_probe.py"):
            continue
        rel = os.path.relpath((toplevel / raw).resolve(), root.resolve())
        if rel.startswith(".."):
            continue
        out.append(rel)
    return sorted(out)


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
    is the eight AST scanners plus the ast-module fallback, which need no
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
    return [(lang, files) for lang, files in sorted(present.items()) if not is_language_available(lang)]


def _candidate_file(cand: dict) -> str:
    """Repo-relative file of one candidate ('' when the shape lacks one)."""
    v = cand.get("file") or cand.get("path")
    return v if isinstance(v, str) else ""


def _candidate_names(cand: dict) -> list[str]:
    """Identity symbol names of one candidate (order-preserving, deduped).

    Covers every gate-scanner candidate shape: dead_block / public_dead_code /
    broken_contract carry ``members: [{name, ...}]``, vulture a top-level
    ``name``, container_reachability ``container_symbol``, broken_contract
    additionally ``core_name`` / ``orphan_name``.  A candidate whose shape
    yields NO name cannot be keyed into the baseline and therefore always
    fails the gate — suppression must never rest on a guess (fail-closed).
    """
    names: list[str] = []
    members = cand.get("members")
    if isinstance(members, list):
        for m in members:
            if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"]:
                names.append(m["name"])
    for key in ("name", "symbol", "symbol_name", "container_symbol", "core_name", "orphan_name"):
        v = cand.get(key)
        if isinstance(v, str) and v:
            names.append(v)
    return list(dict.fromkeys(names))


def _load_baseline() -> tuple[set[tuple[str, str, str]], list[str]] | None:
    """Parse scripts/structural_scanner_baseline.txt; None when absent.

    The private tree never carries the file (zero tolerance, no baseline);
    the public snapshot carries the export-generated one.  Any problem — a
    malformed line, an empty field, or an entry naming a scanner outside
    BASELINE_ALLOWED_SCANNERS — is returned as a human-readable problem for
    main() to FAIL on: a hand-edited drift into the zero-tolerance scanners
    must not quietly weaken them.
    """
    if not _BASELINE_PATH.is_file():
        return None
    entries: set[tuple[str, str, str]] = set()
    problems: list[str] = []
    for lineno, raw in enumerate(_BASELINE_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("::", 2)
        if len(parts) != 3 or not all(p.strip() for p in parts):
            problems.append(f"line {lineno}: malformed entry (expected <scanner>::<file>::<symbol>)")
            continue
        scanner, rel, symbol = (p.strip() for p in parts)
        if scanner not in BASELINE_ALLOWED_SCANNERS:
            problems.append(
                f"line {lineno}: {scanner} is zero-tolerance — baseline entries are only "
                f"allowed for: {', '.join(BASELINE_ALLOWED_SCANNERS)}"
            )
            continue
        entries.add((scanner, rel, symbol))
    return entries, problems


def _missing_scanner_deps() -> list[str]:
    """Optional third-party deps a gate scanner silently degrades without.

    vulture_dead_code_scanner returns [] when ``import vulture.core`` fails
    (optional extra ``asicode[vulture]``) — with the package absent it prints
    "0 candidates" WITHOUT having judged anything, the same soundness class
    as a missing tree-sitter grammar.  Live case: v0.2.21-v0.2.25 public CI
    installed only the base package before this step, so the vulture arm of
    this gate never actually judged there.  main() fails when this is
    non-empty (fail-closed), same contract as _unjudged_languages.
    """
    missing: list[str] = []
    try:
        import vulture.core  # noqa: F401
    except ImportError:
        missing.append("vulture_dead_code_scanner — install the optional extra: pip install 'asicode[vulture]'")
    return missing


def _run_scanner(registry, name: str, file_paths: list[str], baseline=None, collect=None, **kwargs):
    """Run one scanner, print its outcome, return ``(failing, suppressed)``.

    ``baseline`` (the entry set from _load_baseline) suppresses candidates of
    BASELINE_ALLOWED_SCANNERS whose identity — (scanner, file, EVERY name) —
    is baselined: an export artifact.  A candidate only PARTLY covered stays
    failing (a hand-pruned baseline must not half-hide a cluster), and a
    candidate whose shape yields no file/names can never be suppressed.

    ``collect`` (export-time --dump-candidates mode) receives the RAW
    pre-baseline identity of every candidate as ``(scanner, file, names)``
    tuples — the dump must describe the tree, not the baseline.
    """
    try:
        result = registry.run(name, repo_root=str(REPO), file_paths=file_paths, **kwargs)
    except Exception as exc:  # fail-closed: a broken scanner is not a pass
        print(f"❌ {name}: SCANNER ERROR — {exc!r}")
        return -1, 0
    raw = list(result.candidates_raw)
    if collect is not None:
        for c in raw:
            collect.append((name, _candidate_file(c), _candidate_names(c)))
    suppressed = 0
    failing = raw
    if baseline and name in BASELINE_ALLOWED_SCANNERS:

        def _is_artifact(cand: dict) -> bool:
            names = _candidate_names(cand)
            rel = _candidate_file(cand)
            return bool(names) and bool(rel) and all((name, rel, n) in baseline for n in names)

        suppressed = sum(1 for c in raw if _is_artifact(c))
        failing = [c for c in raw if not _is_artifact(c)]
    total = len(failing)
    files = {c.get("file") for c in failing if isinstance(c, dict) and c.get("file")}
    note = f" ({suppressed} baseline-suppressed export artifact(s))" if suppressed else ""
    print(f"  {name}: {total} candidate(s) across {len(files)} file(s){note}")
    for c in failing[:10]:
        print(f"      {str(c)[:220]}")
    if total > 10:
        print(f"      … and {total - 10} more")
    return total, suppressed


def _run_ast_similarity_gate(registry, file_paths: list[str]) -> int:
    """Run the ast_similarity exact-duplicate gate; return failing pair count.

    The gate contract is NARROWER than the other scanners: only pairs whose
    similarity is exactly 1.0 (identical normalised structure) in NON-TEST
    source fail.  The ranked near-duplicate set is not an enumeration contract
    (see REPORT_ONLY_SCANNERS note) and test methods are idiomatically similar
    by design — 994 exact pairs live in tests/ (measured 2026-08-16), so test
    files are excluded wholesale.

    max_candidates is raised far above the default (20) so the 1.0 subset is
    not truncated by the ranking cap: at max_candidates=10000 a pair hidden by
    the cap would silently pass the gate, which would make "0 candidates" a
    lie.  The 1.0 filter is applied AFTER the run because the scanner's
    ``min_similarity`` gate is a soft ranking cutoff — a pair at 0.9999 must
    not fail the build, only structural identity does.
    """
    src_paths = [p for p in file_paths if not p.startswith("tests/")]
    if not src_paths:
        print("  ast_similarity_scanner: 0 candidate(s) (no non-test source files)")
        return 0
    try:
        result = registry.run(
            "ast_similarity_scanner",
            repo_root=str(REPO),
            file_paths=src_paths,
            min_similarity=0.999,
            max_candidates=10000,
        )
    except Exception as exc:  # fail-closed: a broken scanner is not a pass
        print(f"❌ ast_similarity_scanner: SCANNER ERROR — {exc!r}")
        return -1
    exact = [c for c in result.candidates_raw if c.get("similarity", 0.0) >= 0.9999]
    print(
        f"  ast_similarity_scanner: {len(exact)} exact-duplicate pair(s) "
        f"(similarity 1.0, non-test source) across {len(result.affected_files)} file(s)"
    )
    for c in exact[:10]:
        print(f"      {c.get('symbol_a')} <-> {c.get('symbol_b')} (sim {c.get('similarity')})")
    if len(exact) > 10:
        print(f"      … and {len(exact) - 10} more")
    return len(exact)


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
    # --dump-candidates <path> is the one VALUE flag: it runs the identical
    # pipeline but records the raw pre-baseline identity of every candidate
    # and exits 0 on candidates — the judgment belongs to the caller
    # (scripts/export_public.py baseline generation; tests).  Machinery
    # failures still exit 1: an incomplete dump must never be consumed.
    dump_path: str | None = None
    if "--dump-candidates" in argv:
        i = argv.index("--dump-candidates")
        if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
            print("❌ --dump-candidates requires a path argument")
            print("   usage: [--gate-only] [--dump-candidates <path>] [file paths]")
            return 1
        dump_path = argv[i + 1]
        del argv[i : i + 2]
    # Fail-closed on unknown flags (2026-08-11): an unknown --flag was
    # previously ignored, so a typo like ``--gate-onyl`` silently ran the FULL
    # ~35s scan and dropped the gate-only intent — the report-only section was
    # skipped (gate-only mode) or paid for (full mode), and CI cost inflated
    # with no signal.  A flag is a caller contract; a typo must fail loudly.
    unknown = [a for a in argv if a.startswith("--") and a not in known_flags]
    if unknown:
        print("❌ unknown flag(s): " + ", ".join(unknown))
        print("   supported: " + ", ".join(known_flags) + ", --dump-candidates <path>  (file paths are positional)")
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
    if full_scan:
        file_paths = _walk_scan_files(REPO)
        # Full-scan mode excludes scratch probe files (*_probe.py) just like
        # _untracked_scan_files does for per-file mode (and the zero-scan test
        # does): a sibling worker's transient intentional-violation probe must
        # not trip the zero-candidate floor (probes are written inside the repo
        # by gate-mechanics tests and unlinked in finally; os.walk does NOT
        # read .gitignore, so the exclusion must be explicit here, not left to
        # ignore patterns — mirrors _untracked_scan_files' *_probe.py check).
        file_paths = [f for f in file_paths if not f.endswith("_probe.py")]
    else:
        # P-2 (2026-08-16): pre-commit passes only STAGED paths, so a new
        # (untracked) file with a violation passed the per-file gate until its
        # first commit while the full-scan floor (CI) caught it.  Union the
        # git-untracked scannable set: per-file mode and full-scan mode must
        # judge the same zero-candidate floor.
        untracked = _untracked_scan_files(REPO)
        added = sorted(set(untracked) - set(paths))
        if added:
            print(f"  [per-file] +{len(added)} untracked file(s) (git ls-files --others --exclude-standard)")
        file_paths = sorted(set(paths) | set(untracked))
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

    # ── Export-artifact baseline (reference-dependent scanners, fail-closed) ─
    # Loaded BEFORE the registry/graph: baseline problems are an input-contract
    # violation (like an unknown flag) — fail before any heavy work.
    baseline_state = _load_baseline()
    baseline_entries: set[tuple[str, str, str]] = set()
    if baseline_state is not None:
        baseline_entries, baseline_problems = baseline_state
        scanners = ", ".join(sorted({s for s, _, _ in baseline_entries})) or "none"
        print(f"  [baseline] {len(baseline_entries)} export-artifact entries ({scanners})")
        for p in baseline_problems:
            print(f"❌ baseline {p}")
        if baseline_problems:
            return 1

    registry = _load_registry()
    if registry is None:
        return 1

    failed = False
    mode = f"{'gate-only ' if gate_only else ''}{'full repo' if full_scan else f'{len(file_paths)} file(s)'}"
    print(f"Structural scanner gate — {mode} ({len(file_paths)} files scanned)")

    # ── Optional scanner dependencies (fail-closed) ────────────────────────
    # Same soundness class as the grammar gap below: a scanner that silently
    # degrades to [] without an optional dependency would pass the floor
    # UNJUDGED (live case: v0.2.21-v0.2.25 public CI never judged vulture).
    missing_deps = _missing_scanner_deps()
    if missing_deps:
        for m in missing_deps:
            print(f"❌ optional scanner dependency missing — {m}")
        print("   fail-closed: a scanner that cannot run must not pass the zero-candidate floor.")
        failed = True
    else:
        print("  [deps] optional scanner dependencies present (vulture)")

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

    dump_collect: list[tuple[str, str, list[str]]] = []
    suppressed_total = 0
    for name in GATE_SCANNERS:
        spec = registry.get_spec(name)
        if spec is None:
            print(f"❌ {name}: NOT REGISTERED — gate would silently pass")
            failed = True
            continue
        if name == "ast_similarity_scanner":
            # Whole-repo pair signal: per-file mode (pre-commit) cannot judge
            # similarity pairs from a single changed file, so it is skipped
            # here exactly as the report-only scanners used to be.  The
            # full-scan floor (CI lint.yml --gate-only, local no-flag runs)
            # applies the zero-exact-pair contract repo-wide.
            if not full_scan:
                print("  ast_similarity_scanner: skipped in per-file mode (whole-repo pair signal)")
                continue
            count = _run_ast_similarity_gate(registry, file_paths)
            if count > 0:
                if dump_path:
                    print("❌ ast_similarity_scanner: exact duplicate(s) present — never baselined;")
                    print("   the dump cannot cover them, so it fails instead of lying by omission")
                failed = True
            if count < 0:
                failed = True
            continue
        kwargs: dict = {}
        if cross_refs is not None and "cross_file_referenced_names" in (spec.input_schema or {}):
            kwargs["cross_file_referenced_names"] = cross_refs
        if getattr(spec, "requires_graph", False):
            kwargs["repo_graph"] = graph
        count, suppressed = _run_scanner(
            registry, name, file_paths, baseline=baseline_entries, collect=dump_collect, **kwargs
        )
        suppressed_total += suppressed
        if count > 0 and not dump_path:
            failed = True
        if count < 0:
            failed = True  # machinery failure — fails dump mode too (incomplete dump)

    if full_scan and not gate_only:
        print("Report-only scanners (currently none — ast_similarity is gated at the exact-1.0 contract):")
        for name in REPORT_ONLY_SCANNERS:
            if registry.get_spec(name) is None:
                print(f"  {name}: not available")
                continue
            _run_scanner(registry, name, file_paths)

    if dump_path is not None:
        import json

        payload = {
            "candidates": sorted(
                ({"scanner": s, "file": f, "names": sorted(ns)} for s, f, ns in dump_collect),
                key=lambda e: (e["scanner"], e["file"], tuple(e["names"])),
            )
        }
        Path(dump_path).write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        print(f"\n📦 dumped {len(payload['candidates'])} gate-scanner candidate(s) → {dump_path}")
        if failed:
            print("❌ dump incomplete — see the failures above (a partial dump must not be consumed)")
            return 1
        return 0

    if failed:
        if unjudged:
            print(
                "\n❌ tree-sitter grammar gap — non-Python language(s) cannot be judged; "
                "install the grammar(s) listed above and re-run."
            )
        else:
            print(
                "\n❌ Deterministic structural scanner(s) found candidates — this is a "
                "regression. Fix the code; the export-artifact baseline is machine-generated "
                "by scripts/export_public.py and only covers reference-dependent scanners."
            )
        return 1
    if baseline_state is not None:
        print(
            f"\n✅ deterministic structural scanners: 0 failing candidates "
            f"({suppressed_total} export-artifact baseline-suppressed)"
        )
    else:
        print("\n✅ deterministic structural scanners: 0 candidates (no baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
