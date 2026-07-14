"""
Impact-based test selection — the glue that closes the verification loop.

Context
-------
``verification_set_builder`` and ``impact_propagation`` already implement
tiered, symbol-aware test selection — but they are **never called** from the
live quality gate. ``agent_loop`` runs ``dispatch("run_tests", {})`` which
executes the *full* suite every time. This is the classic closed-loop break:
the smart selection exists, the dumb execution ignores it.

This module is the minimal, self-contained glue the quality gate calls to turn
"which files just changed" into a scoped ``args`` list for ``run_tests``. It
uses two signals, unioned:

  1. **Naming convention** (always available): edits to ``pkg/foo.py`` map to
     ``test_foo.py`` anywhere under ``tests/``. This is the dominant signal —
     it is how this repo (and most Python projects) organize tests, and it
     needs no index.
  2. **Call graph** (optional): transitive callers of symbols *defined* in the
     touched files, intersected with test files. Adds cross-module coverage
     (e.g. editing a helper used by ``test_bar.py`` even when no
     ``test_<edited-file>.py`` exists).

Safety contract
---------------
``select_affected_tests`` returns ``[]`` when nothing matches, and the caller
must then fall back to the full suite. This makes scoped verification
*safe-by-construction*: it never runs *zero* tests when a run was requested,
and it never silently skips verification because a name didn't resolve.
"""
from __future__ import annotations

import ast
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:  # avoid hard import at module load
    from .call_graph import CallGraphIndexer
# ── index cache ─────────────────────────────────────────────────────────────
# The stem+import index is built by walking all tests/ files (~476).
# Cache it so repeated select_affected_tests calls within a short window
# (e.g. multiple quality-gate firings per agent run) don't re-parse
# every file each time.
_index_cache: dict[str, tuple[dict[str, list[str]], dict[str, list[str]], float]] = {}
_INDEX_CACHE_TTL = 600  # seconds — safe to be generous; invalidate_index() is event-driven


def invalidate_index(repo_root: str | Path) -> None:
    """Drop the cached index for *repo_root* (e.g. after a new test file is created)."""
    _index_cache.pop(str(Path(repo_root).resolve()), None)


def _build_or_get_index(repo_root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return cached or freshly-built stem+import index for *repo_root*."""
    key = str(repo_root)
    now = time.time()
    cached = _index_cache.get(key)
    if cached is not None and now - cached[2] < _INDEX_CACHE_TTL:
        return cached[0], cached[1]
    idx = _build_test_stem_index(repo_root)
    _index_cache[key] = (*idx, now)
    return idx

_MAX_CALLGRAPH_DEPTH = 2   # callers + callers-of-callers
_MAX_TEST_FILES = 60       # cap so a "fix everything" edit doesn't select the whole suite


# ── symbol extraction ────────────────────────────────────────────────────────

def _defined_names(path: Path) -> set[str]:
    """Top-level defs/classes and methods defined in ``path``.

    Returns both bare (``foo``) and qualified (``Class.foo``) forms, since the
    call graph indexes methods as ``Class.method`` but callers may reference the
    bare name. Tolerates syntax errors (returns whatever parsed) so a
    half-edited file never breaks selection.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return names
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(sub.name)
                    names.add(f"{node.name}.{sub.name}")
    return names


def is_test_file(rel_path: str) -> bool:
    """Classify a repo-relative path as a test file.

    Public so cross-module callers (``agent_loop``, ``agent_turn_pipeline``)
    import a stable public name instead of reaching into a private one.  A
    path is a test file when it is a ``.py`` file AND matches one of the
    conventional test-file patterns: contains a ``/test_`` segment, starts
    with ``test_``, or ends with ``_test.py``.
    """
    p = rel_path.replace("\\", "/")
    return ("/test_" in p or p.startswith("test_") or p.endswith("_test.py")) and p.endswith(".py")


# Backward-compat alias — older callers and tests imported the private name.
_is_test_file = is_test_file


def git_status_test_files(repo_root: str | Path) -> list[str]:
    """Return repo-relative test files that are new/modified per ``git status``.

    Complements ``modified_files_set`` (populated only by editor handlers) by
    catching test files created via bash/heredoc/cp.  Flags:

    * ``--untracked-files=all`` — list each untracked file individually;
      without it git folds a brand-new directory into a single ``?? dir/``
      entry and the files inside are invisible to ``is_test_file``.
    * ``-z`` — NUL-terminated records with raw UTF-8 paths (no C-quoting of
      non-ASCII or space-containing paths, which would break path matching).

    Porcelain v1 ``-z`` record layout: ``XY <path>\\0``, plus one extra
    NUL-terminated ORIGIN-path field when the entry is a rename or copy
    (``XY <new>\\0<old>\\0``).  The origin field is present whenever X *or* Y
    is ``R``/``C`` — including ``RM``, ``RD``, `` R``, ``C `` — not just
    ``R `` (empirically verified: ``RM new\\0old\\0D  deleted\\0``).

    Only paths that still exist are returned: deleted files and rename-origin
    paths would otherwise flow into pytest args and abort the whole run
    ("file or directory not found", exit code 4), turning the quality gate
    into a false failure.
    """
    root = Path(repo_root)
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-z",
             "--untracked-files=all"],
            capture_output=True, timeout=5,
        )
    except Exception:  # noqa: BLE001 — best-effort augmentation only
        return []
    if r.returncode != 0:
        return []
    out: list[str] = []
    parts = r.stdout.split(b"\0")
    i = 0
    while i < len(parts):
        rec = parts[i]
        if not rec:
            i += 1
            continue
        status = rec[:2]
        if b"R" in status or b"C" in status:
            i += 1  # consume the origin-path field (renamed-away; never exists)
        path = rec[3:].decode("utf-8", errors="replace")
        if path and is_test_file(path) and (root / path).is_file():
            out.append(path)
        i += 1
    return out


# ── import extraction (signal 3) ─────────────────────────────────────────────

def _extract_imports(path: Path) -> set[str]:
    """Return set of full dotted module names imported by ``path``.

    Returns ``{"os.path", "typing", "external_llm.agent"}`` for imports like
    ``import os.path``, ``from typing import Optional``, and
    ``from external_llm.agent import orchestrator``.  Tolerates syntax errors
    silently.  Unlike the earlier top-level-only version, this retains the full
    dotted path so that signal 3 can do prefix matching (``external_llm.agent``
    matches ``external_llm.agent.orchestrator``) instead of collapsing
    everything to a single top-level key.
    """
    imports: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return imports
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.add(alias.name)  # full dotted path
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)  # full dotted path
    return imports


def _is_first_party_module(mod: str, repo_root: Path) -> bool:
    """Check if *mod* resolves to a real package/file under *repo_root*.

    stdlib and third-party packages like ``os``, ``json``, ``typing`` will
    not exist as files under the repo root, so this acts as a first-party
    filter for signal 3a import matching (only imports that exist in the
    repo are meaningful for test selection).
    """
    as_path = mod.replace(".", "/")
    return (
        (repo_root / f"{as_path}.py").exists()
        or (repo_root / as_path / "__init__.py").exists()
    )


# ── signal 1: naming convention ──────────────────────────────────────────────

def _build_test_stem_index(repo_root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Walk tests/ once and build **two** indexes.

    Returns
    -------
    ``(stem_index, import_index)``

    * **stem_index**: ``{stem: [rel_path, ...]}`` where ``stem`` is the source-file
      stem without ``test_`` prefix (``test_foo.py`` → ``foo``).  This lets edits
      to ``pkg/foo.py`` → lookup ``foo`` → find all ``test_foo*`` files in a
      single O(tests/) scan rather than N individual ``rglob`` calls.

    * **import_index**: ``{dotted_module: [test_file_rel_path, ...]}`` built by
      scanning imports of every test file.  Uses full dotted paths so prefix
      matching (3a) and own-module matching (3b) can find related tests across
      package boundaries, not just top-level packages.
    """
    stem_idx: dict[str, list[str]] = {}
    import_idx: dict[str, list[str]] = {}
    tests_dir = repo_root / "tests"
    if not tests_dir.is_dir():
        return stem_idx, import_idx
    for py in tests_dir.rglob("*.py"):
        try:
            rel = py.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        stem = py.stem
        # test_foo.py → stem='foo'  (strip leading test_)
        if stem.startswith("test_"):
            src_stem = stem[5:]
        # foo_test.py → stem='foo'  (strip trailing _test)
        elif stem.endswith("_test"):
            src_stem = stem[:-5]
        else:
            src_stem = None
        if src_stem:
            stem_idx.setdefault(src_stem, []).append(rel)
        # Build import index for every test file (signal 3).
        for mod in _extract_imports(py):
            import_idx.setdefault(mod, []).append(rel)
        # No break: full walk is cheap (O(files) ~500) and critical for correct
        # test selection — an incomplete index silently drops tests, making
        # scoped verification skip affected suites without failing.
    return stem_idx, import_idx


def _tests_for_module(stem_index: dict[str, list[str]], module_stem: str) -> list[str]:
    """Look up ``module_stem`` in a pre-built stem index."""
    if not module_stem or module_stem.startswith("."):
        return []
    return stem_index.get(module_stem, [])[:_MAX_TEST_FILES]


# ── signal 2: call graph ─────────────────────────────────────────────────────

def _caller_files(call_graph: "CallGraphIndexer", symbol: str, depth: int) -> set[str]:
    """Transitive set of files that call ``symbol`` (BFS up the reverse edges)."""
    seen_files: set[str] = set()
    frontier = {symbol}
    visited: set[str] = set()
    for _ in range(max(1, depth)):
        next_frontier: set[str] = set()
        for sym in frontier:
            if sym in visited:
                continue
            visited.add(sym)
            try:
                edges = call_graph.get_callers(sym)
            except Exception:  # noqa: BLE001 — indexer must never break selection
                continue
            for edge in edges:
                f = getattr(edge, "caller_file", None)
                if f:
                    seen_files.add(str(f).replace("\\", "/"))
                # Walk one more hop via the caller's own symbol.
                csym = getattr(edge, "caller_symbol", None)
                if csym:
                    next_frontier.add(csym)
        frontier = next_frontier
    return seen_files


# ── public API ───────────────────────────────────────────────────────────────

def select_affected_tests(
    repo_root: str | Path,
    touched_paths: Iterable[str],
    *,
    call_graph: Optional["CallGraphIndexer"] = None,
    max_tests: int = _MAX_TEST_FILES,
) -> list[str]:
    """Return scoped pytest ``args`` (file paths) for tests affected by edits.

    Always returns a de-duplicated, capped list. Empty list ⟹ caller should
    run the full suite (do not run zero tests). Paths are repo-relative POSIX.
    """
    repo = Path(repo_root).resolve()
    selected: dict[str, int] = {}  # path -> priority (lower = higher prio)

    def _add(path: str, prio: int) -> None:
        if path and path not in selected:
            selected[path] = prio

    def _norm(p: str) -> str:
        p = p.strip()
        if not p:
            return ""
        # Three cases handled in order:
        #   1. Absolute path inside the repo → relative via relative_to.
        #   2. Absolute path outside the repo → check if lstrip("/") gives a
        #      valid repo-relative path.  If the stripped file actually exists
        #      in the repo, treat it as a slash-prefixed relative (common
        #      convention in modified_files_set).  Otherwise silently drop
        #      to avoid false test selections.
        #   3. Plain relative path → normalize separators.
        try:
            ap = Path(p)
            if ap.is_absolute():
                try:
                    return str(ap.resolve().relative_to(repo)).replace("\\", "/")
                except (ValueError, OSError):
                    # Absolute path outside repo — try as slash-prefixed relative.
                    stripped = p.lstrip("/")
                    if stripped and (repo / stripped).exists():
                        return stripped.replace("\\", "/")
                    return ""
        except (ValueError, OSError):
            return ""
        return p.lstrip("/").replace("\\", "/")

    touched = [_norm(p) for p in touched_paths]
    py_touched = [p for p in touched if p.endswith(".py")]

    # ── Signal 1: naming-convention mapping (build indexes once) ──
    stem_index, import_index = _build_or_get_index(repo)
    for rel in py_touched:
        stem = Path(rel).stem
        if is_test_file(rel):
            # Editing a test file → run it directly.
            _add(rel, 0)
            continue
        for t in _tests_for_module(stem_index, stem):
            _add(t, 1)

    # ── Signal 2: call-graph callers ──
    if call_graph is not None:
        caller_files: set[str] = set()
        for rel in py_touched:
            abs_path = repo / rel
            if not abs_path.exists():
                continue
            for name in _defined_names(abs_path):
                caller_files |= _caller_files(call_graph, name, _MAX_CALLGRAPH_DEPTH)
        for cf in caller_files:
            if is_test_file(cf):
                _add(cf, 2)

    # ── Signal 3: import-graph (tests that import modules touched by the edit) ──
    if import_index:
        for rel in py_touched:
            abs_path = repo / rel
            if not abs_path.exists():
                continue
            # 3a. Tests that import what the *edited file* imports (upstream deps).
            #     Only first-party imports (modules that exist as files in the repo)
            #     are considered — stdlib/third-party imports like ``os``, ``json``,
            #     ``typing`` are too widely used to be selective.
            #     Prefix expansion stops at i=1 (top-level) — a file deep in the
            #     package tree importing ``external_llm.agent.bar`` should match
            #     tests that import ``external_llm.agent.bar`` or
            #     ``external_llm.agent``, but NOT everything under ``external_llm``.
            for mod in _extract_imports(abs_path):
                if not _is_first_party_module(mod, repo):
                    continue
                parts = mod.split(".")
                for i in range(len(parts), 1, -1):  # skip top-level (i=1)
                    prefix = ".".join(parts[:i])
                    for tied in import_index.get(prefix, []):
                        _add(tied, 3)
            # 3b. Tests that import the *edited file's own module* (downstream
            #     consumers).  Derive the dotted module path from the file path
            #     so, e.g., ``external_llm/agent/orchestrator.py`` yields keys
            #     ``external_llm.agent.orchestrator``, ``external_llm.agent``.
            #     Same top-level skip as 3a: a test that only imports
            #     ``external_llm`` is too weak a signal to be meaningful.
            own_mod = rel.removesuffix(".py").replace("/", ".").replace("\\", ".")
            parts = own_mod.split(".")
            for i in range(len(parts), 1, -1):  # skip top-level (i=1)
                prefix = ".".join(parts[:i])
                for tied in import_index.get(prefix, []):
                    _add(tied, 3)

    if not selected:
        return []

    # Stable order: priority first, then path. Cap to max_tests.
    ordered = [p for p, _ in sorted(selected.items(), key=lambda kv: (kv[1], kv[0]))]
    return ordered[:max_tests]
