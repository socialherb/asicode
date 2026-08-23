"""walk_policy.py -- SSOT for repo-walk admission and descent ordering.

Every walker that decides *which directories/files a repo walk visits* must
consume THIS module -- never a private copy.  The pre-B2' duplicates
(RepositoryGraph's ``_SKIP_DIRS``/``_SKIP_FILE_SUFFIXES`` vs the shared set)
drifted: ``vendor/`` was RG-only, ``.egg-info`` shared-only, ``.min.js``
RG-only -- the graph answered a different universe than the agent walkers
(B2' parity contract, 2026-08-11).  The module lives in ``common/`` so both
the agent layer (call_graph, symbol_search, rag_searcher,
``_shared_utils._walk_repo_files``) and the graph layer (RepositoryGraph) can
import it without a layer inversion -- RepositoryGraph previously imported
walk policy from ``external_llm.agent._shared_utils`` (F5, 2026-08-12).

Names keep the historical underscore prefix: they are intra-package
conventions, not a public API.  ``_shared_utils`` re-exports them for
backward compatibility (same pattern as ``common/cache_utils``).
"""

from __future__ import annotations

# Basename suffixes every walker must skip -- minified bundles are parser
# noise with no structural value (RepositoryGraph, P4 2026-08-11).  The TS/JS
# walker keep-predicate, the RAG walker and RepositoryGraph all apply this so
# every walker admits the SAME file set (B2' parity contract, 2026-08-11).
# .min.mjs/.min.cjs/.min.css cover the non-.js minified bundles the language
# layer already recognises (.mjs/.cjs/.css) (F-RAG-2).
_WALK_SKIP_FILE_SUFFIXES: tuple = (".min.js", ".min.mjs", ".min.cjs", ".min.css")

# Directory basenames pruned from every repo walk (in-place ``dirnames``
# pruning so ``os.walk`` never descends into them).  The set is the union of
# the former call_graph / symbol_search / RepositoryGraph implementations --
# the stricter of each -- so every consumer also excludes
# venv/site-packages/*.egg-info/vendor dirs alike.
_WALK_SKIP_DIRS: frozenset = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "worktrees",
        "vendor",
    }
)

# Directories whose contents are least useful for code-navigation tools
# (find_symbol / find_relevant_files / analyze_change_impact). The walker
# visits these LAST, so when a file cap is reached, real source code fills it
# before tests / fixtures / generated output do. This is a typed NAME set (a
# policy), not a regex -- ``os.walk`` already prunes vendor dirs via
# :data:`_WALK_SKIP_DIRS`; this set only reorders the survivors by relevance.
# Case-insensitive match against the directory basename.
_WALK_DEPRIORITIZED_DIRS: frozenset = frozenset(
    {
        "tests",
        "test",
        "__tests__",
        "tst",
        "spec",
        "specs",
        "__specs__",
        "fixtures",
        "testdata",
        "test_data",
        "test-data",
        "mocks",
        "stubs",
        "snapshots",
        "__snapshots__",
        "fakes",
        "examples",
        "samples",
        "out",
        "target",
        "generated",
        "gen",
        "autogen",
    }
)


def _walk_dir_sort_key(d: str) -> tuple[int, str]:
    """Sort key for ``os.walk`` directory descent order.

    Lower tuple = visited first. Deprioritized dirs (tests/fixtures/generated)
    get tier 1 so source subtrees (tier 0) are enumerated -- and thus fill the
    file cap -- before them. Within a tier, plain alphabetical order keeps the
    walk deterministic across machines/clones (``os.walk`` otherwise returns
    filesystem-enumeration order, which is non-reproducible).
    """
    return (1 if d.lower() in _WALK_DEPRIORITIZED_DIRS else 0, d)


def _walk_should_skip_dir(d: str) -> bool:
    """True if directory name *d* must be excluded from repo walks.

    Single pruning predicate shared by the shared walkers (``_walk_py_files`` /
    ``_walk_ts_js_files``), the RAG walker and RepositoryGraph's ``build()`` so
    no two walkers can drift. They previously diverged: the TS/JS walker
    carried a redundant ``node_modules`` substring check (already in
    ``_WALK_SKIP_DIRS`` as an exact match) while *missing* ``venv*`` (e.g.
    ``venv310``, ``myvenv``) and ``site-packages`` dirs -- letting vendored
    JS/TS bundled inside a Python package pollute the index; RepositoryGraph
    had a private copy that drifted the other way (B2', 2026-08-11).
    """
    return d.startswith((".", "venv")) or d in _WALK_SKIP_DIRS or d.endswith(".egg-info") or "site-packages" in d


def _rel_under_skipped_dir(rel: str) -> bool:
    """True if any directory component of repo-relative *rel* is walk-pruned.

    Companion to :func:`_walk_should_skip_dir`: the walk prunes ``dirnames``
    in-place (a file under ``node_modules``/``.venv`` is never visited), but
    the incremental path receives a single changed file path and must re-apply
    the SAME pruning or it indexes files a fresh ``build()`` would drop
    (call_graph B1, 2026-08-11). The basename (``parts[-1]``) is never a
    directory and is not tested.
    """
    return any(_walk_should_skip_dir(part) for part in str(rel).replace("\\", "/").split("/")[:-1])


def _path_is_walk_admissible(rel: str) -> bool:
    """True iff a fresh build()'s os.walk would have visited *rel*.

    Directory pruning AND the basename suffix policy -- the walk applies both
    (``_walk_repo_files`` prunes ``dirnames`` in place and the keep-predicate
    drops suffix-matched basenames), so any incremental path that re-applies
    only half diverges from a rebuild: the graph would depend on the write
    history instead of the tree (call_graph F1, 2026-08-12).  Language routing
    (py vs ts/js) is intentionally NOT part of this admission -- callers that
    index by language apply their own routing on top.
    """
    if _rel_under_skipped_dir(rel):
        return False
    return not str(rel).replace("\\", "/").split("/")[-1].endswith(_WALK_SKIP_FILE_SUFFIXES)
