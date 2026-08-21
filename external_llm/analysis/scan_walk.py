"""Single source of truth for the structural-scan file walk.

Both consumers of the scannable-file walk previously carried INDEPENDENT
copies of the extension tuple, the skip-dir set, the file cap and the walk
itself, joined only by "keep in sync" comments:

  - the structural gate — ``scripts/check_structural_scanners.py``
  - the agent structural-scan tool — ``AnalysisToolsMixin`` in
    ``external_llm/agent/tool_handlers/analysis_tools.py``

This module is that single source: the gate's ``_SCAN_*`` names are identity
ALIASES of the constants here (pinned by
test_scan_walk_constants_are_single_source_aliases), the mixin carries no
``_SCAN_*`` attributes anymore, and both ``_walk_scan_files`` methods
delegate to :func:`walk_scan_files` (pinned by
test_gate_walk_matches_shared_walk / test_mixin_walk_delegates_to_shared_source).

Also here, derived from ``SCAN_EXTS``: ``SCAN_LANGUAGES`` — the languages
the scanned extensions map onto, resolved DIRECTLY through ``_EXT_MAP``
(languages/models.py — the package's canonical extension → language map)
with import-time fail-fast, plus the no-scan boundary for that map's other
judged-language extensions (``_UNSCANNED_VARIANTS``).  What is NOT here is
the judge-assignment contract (which gate scanner judges which language) —
that lives at the gate, see ``scripts/check_structural_scanners.py`` near
the ``_SCAN_EXTS`` alias.
"""

from __future__ import annotations

import os

from ..common.walk_policy import _walk_should_skip_dir
from ..languages import LanguageId
from ..languages.models import _EXT_MAP

# ── The scan-walk contract ──────────────────────────────────────────────────
# 8 extensions → 6 languages via _EXT_MAP (.tsx → typescript, .js/.jsx →
# javascript; see SCAN_LANGUAGES below).  Every extension must be judged by
# at least one gate scanner — there is no scanned-but-unjudged language
# (pinned by test_every_scan_ext_is_covered_by_a_gate_scanner).
SCAN_EXTS: tuple[str, ...] = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt")

# The _EXT_MAP extensions of the scanned languages that are deliberately NOT
# scanned: syntax-aliases of the SCAN_EXTS forms (stubs, module/script
# variants) that resolve to the same language and grammar.  The boundary is
# exhaustive — every judged-language extension in _EXT_MAP must sit on one
# side or the other (pinned by test_scan_ext_map_boundary_is_exhaustive):
# a NEW extension for a scanned language (say ".pyx": "PYTHON") fails that
# test until a scan/no-scan decision is made.
_UNSCANNED_VARIANTS: frozenset[str] = frozenset({".pyi", ".mts", ".cts", ".mjs", ".cjs", ".kts"})

# The language set the scanned extensions map onto — DERIVED from SCAN_EXTS
# through _EXT_MAP (languages/models.py), the package's canonical
# extension → language map.  Deliberately NOT LanguageId.from_path: that API
# degrades to UNKNOWN for an unmapped extension (right for single-file
# tooling), while the scan walk is a contract — an extension here missing
# from _EXT_MAP, or a map value that stops being a LanguageId member, must
# fail at IMPORT time (fail-fast; see _derive_scan_languages) rather than
# silently unjudge the language in the gate.  Consumers bind by identity
# (registry ``_TS_LANGUAGES``; pinned by
# test_scan_walk_constants_are_single_source_aliases); the per-language
# judge maps (_LANG_TOP_LEVEL_NODES / _LANG_KIND_MAP, _LANG_DEF_NODES) are
# pinned to it by test_duplicate_definition_lang_keys_match_registry.


def _derive_scan_languages() -> frozenset[LanguageId]:
    """Resolve SCAN_EXTS → languages through _EXT_MAP, failing at import time."""
    out: set[LanguageId] = set()
    for ext in SCAN_EXTS:
        name = _EXT_MAP.get(ext)
        if name is None:
            raise ValueError(
                f"SCAN_EXTS extension {ext!r} is missing from _EXT_MAP "
                "(external_llm/languages/models.py) — add it there or drop "
                "it from SCAN_EXTS"
            )
        try:
            out.add(LanguageId[name])
        except KeyError:
            raise ValueError(
                f"_EXT_MAP[{ext!r}] = {name!r} is not a LanguageId member (external_llm/languages/models.py)"
            ) from None
    return frozenset(out)


SCAN_LANGUAGES: frozenset[LanguageId] = _derive_scan_languages()


def _scan_should_skip_dir(d: str) -> bool:
    """Prune predicate for the structural-scan walk (F7).

    Delegates to the shared ``walk_policy._walk_should_skip_dir`` — the same
    predicate every repo walker uses — EXCEPT it intentionally visits ``env/``,
    a real source directory in some repos.  That single exemption is the ONLY
    skip-set divergence between this walk and ``RepositoryGraph.build`` (pinned
    by test_scan_walk_and_graph_skip_sets_diverge_on_env).  Before F7 the
    private ``SCAN_SKIP_DIRS`` + ``startswith('.')`` also missed ``vendor/``,
    ``site-packages/``, ``venv*`` prefixes and ``*.egg-info`` dirs — letting
    vendored deps pollute the structural scanners with false positives.
    """
    return _walk_should_skip_dir(d) and d != "env"


# ── Cap semantics + scope difference vs the graph build (documented contract)
# The walk feeds the SCANNER file lists (registry file_paths and the
# cross-file-ref pass) and truncates at SCAN_FILE_CAP files in sorted
# traversal order — the FIRST-N-by-sorted-order, deterministic across
# processes (pinned by test_walk_scan_files_is_deterministic_and_sorted).
# RepositoryGraph.build() walks the SAME repo WITHOUT any cap, so in a repo
# above the cap the gate graph is strictly MORE complete than the scanner
# list (every source file graphed, only the first 4000 scanned).  The
# asymmetry is deliberate: the cap bounds scanner latency on huge repos (an
# ASSIST signal), while the graph stays complete because a cache-warm build
# costs ~1s and the four graph-dependent gate scanners must see the whole
# repo.  If the repo ever exceeds the cap, the cross-file-ref pass (capped
# list) could miss names referenced only beyond the cap → dead-code scanners
# would false-positive; the fix is raising the cap or sharing one walk.
# The gate (scripts/check_structural_scanners.py) FAILS when this walk
# truncates (fail-closed, 2026-08-11): files beyond the cap are unjudged,
# and an unjudged file must not pass the gate's zero-candidate floor.
SCAN_FILE_CAP: int = 4000


def walk_scan_files(
    root: str | os.PathLike[str],
    *,
    cap: int = SCAN_FILE_CAP,
    base: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Collect scannable source files under *root* (repo-relative paths).

    Deterministic: directories and files are traversed in sorted order, so
    the returned list — and its truncation point under *cap* — is identical
    across processes (readdir order is nondeterministic; BUG-6).  The cap is
    a LATENCY guard for the scanner file list only: ``RepositoryGraph.build``
    walks the same root without a cap, so the graph is always a superset of
    this list (see ``SCAN_FILE_CAP``).

    Returned paths are relative to *base* (default: *root*).  The agent tool
    passes its ``repo_root`` so subdir scans still yield repo-relative paths
    (scanners open ``repo_root + path``); the gate walks ``REPO`` itself,
    where the two bases coincide.

    Skip-set note: the pruning delegates to ``_scan_should_skip_dir``, which
    applies the shared ``walk_policy._walk_should_skip_dir`` (vendor/
    site-packages/ venv*/ *.egg-info/ hidden dirs) EXCEPT it intentionally
    visits ``env/``.  ``env/`` is the ONE divergence from the graph build
    (pinned by test_scan_walk_and_graph_skip_sets_diverge_on_env): this walk
    scans it, the graph skips it.
    """
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _scan_should_skip_dir(d))
        for fn in sorted(filenames):
            if not fn.endswith(SCAN_EXTS):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), base or root)
            out.append(rel)
            if len(out) >= cap:
                return out
    return out
