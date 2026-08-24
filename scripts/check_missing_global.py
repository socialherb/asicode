#!/usr/bin/env python3
"""Check no function assigns a module-level global without declaring it.

This is the other half of scripts/check_f823_none.py.  Both gates exist
because of the same incident, and neither alone covers it.

The two shapes of a missing ``global``
--------------------------------------
A function that assigns a name it has not declared ``global`` binds a fresh
LOCAL.  What that costs depends on whether the function also reads the name:

  read-then-write   ``if not _dirty: ... ; _dirty = False``
                    → UnboundLocalError at runtime.  Ruff sees this as F823,
                      and check_f823_none.py gates it at zero.

  write-only        ``_dirty = True``
                    → no error, no warning, no ruff diagnostic.  The
                      assignment simply evaporates and the module global keeps
                      its old value forever.  **Nothing gated this.**

Motivating incidents (two, in this repo)
----------------------------------------
1. ``_rg_dirty`` (webapp/ui/ui_tools.py).  Per the note now standing at its
   declaration: "both writers then omitted ``global``, making every
   persistence path either raise or silently no-op."  The raising half became
   a 500 on ``/stats/rg-fallback``; the silent half froze
   ``.asicode/rg_fallback_counts.json`` at all-zeros for a day — zeros that
   read exactly like "the fallback never fired", which is the opposite of what
   they meant.

2. ``_completer_provider`` / ``_completer_model`` (external_llm/repl/repl_impl.py).
   All six assignments in ``_dispatch_command`` were write-only, so the REPL's
   ``/think`` autocomplete kept offering the STARTING model's reasoning values
   for the whole session after any ``/model`` switch (deepseek's ``max``
   missing, openai's ``none``/``medium`` wrongly offered).  Fixed by routing
   every write through a single ``_set_completer_context()``.

Note that in incident 1 **no** function declared the name ``global``.  A rule
keyed on "some other scope declares it global" would have missed it entirely,
which is why the check below is the broad one: any local assignment to a name
the module assigns at module scope.

Why that broad rule is affordable
---------------------------------
Because the repo already holds the convention.  Measured at gate time
(2026-08-07): 7 candidates repo-wide, 6 of them incident 2 and the 7th a
benign ``router = TaskRouter(...)`` shadowing the module's FastAPI
``APIRouter`` in webapp/routes/agent_stream.py (renamed ``_task_router``).
So the floor is ZERO, and this gate has **no baseline** — same contract as
check_f823_none.py / check_open_encoding.py / check_lint_full.py.

The convention is deliberate, not accidental; ui_tools.py states it at the
declaration of the very name that caused incident 1: state is declared "beside
the state it guards, and BEFORE its first reader — so the 'is this a module
global or a local?' question is answerable at a glance."  A local that reuses
a module global's name is precisely what makes that question unanswerable.

Scope of the rule
-----------------
Only names the module binds by ASSIGNMENT at module scope (``x = ...``,
``x: T = ...``, ``x += ...``) count as module state.  Imports, ``def`` and
``class`` names are excluded: shadowing those locally is a different (and
common, mostly harmless) pattern, and including them would trade a measured
zero for noise.

Only ASSIGNMENT binds are flagged, not every local binding.  A parameter, a
``for`` target, a ``with ... as`` name or an ``except ... as`` name that
happens to reuse a module global's name is a readability smell but not the
silent-no-op bug this gate is about.

Two remedies, both fine
-----------------------
  * the function meant to mutate module state → add ``global <name>``
    (or ``nonlocal`` for a closure over an enclosing function's local); or
  * the function meant a scratch local → rename it.

Do NOT add a baseline for this.

Usage:
    python scripts/check_missing_global.py
    python scripts/check_missing_global.py <file>.py ...  # check only given files

Explicit file args (pre-commit per-file mode) scan only those files; the check
is entirely intra-file, so per-file mode loses no signal.  No args (lint.yml
CI) scans the whole repo.
"""

import ast
import json
import os
import symtable
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories that hold no first-party source.  tests/ IS scanned: a test that
# silently fails to set module state asserts nothing, and the measured baseline
# over tests/ is zero.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".asicode",
}

# Per-file analysis disk cache (P15, 2026-08-24): the full-repo scan re-parses
# every scoped .py on every run (ast.parse + symtable), but _violations_in is a
# pure function of file content, so a (st_mtime_ns, st_size) fingerprint cache
# reuses prior results — same invalidation contract as the other analysis gates
# (A307, 786ffcdc).  Fail-open: corruption/version-mismatch → full recompute.
# `--no-cache` bypasses the cache entirely (pre-commit per-file runs where the
# file set and content are mid-edit).
_CACHE_VERSION = 1


def _cache_path() -> Path:
    """Per-repo cache file (REPO is monkeypatch-able in tests, so derived dynamically)."""
    return REPO / ".cache" / f"missing_global_v{_CACHE_VERSION}.json"


def _stat_fingerprint(path: Path) -> tuple[int, int] | None:
    """(st_mtime_ns, st_size) for *path*, or None when it cannot be stat'd."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _load_cache() -> dict[str, tuple[tuple[int, int], list[tuple[int, str, str, str]]]]:
    """Load the violations cache; ``{rel: (fingerprint, violations)}``, empty on any fault."""
    try:
        with open(_cache_path(), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}
    if payload.get("version") != _CACHE_VERSION:
        return {}
    out: dict[str, tuple[tuple[int, int], list[tuple[int, str, str, str]]]] = {}
    for rel, entry in (payload.get("files") or {}).items():
        if not isinstance(entry, dict):
            continue
        fp = entry.get("fp")
        vio = entry.get("violations")
        if (
            isinstance(fp, list)
            and len(fp) == 2
            and all(isinstance(x, int) for x in fp)
            and isinstance(vio, list)
            and all(
                isinstance(v, list) and len(v) == 4 and isinstance(v[0], int) and all(isinstance(x, str) for x in v[1:])
                for v in vio
            )
        ):
            out[rel] = (tuple(fp), [tuple(v) for v in vio])
    return out


def _save_cache(cache: dict[str, tuple[tuple[int, int], list[tuple[int, str, str, str]]]]) -> None:
    """Persist *cache* to disk (best-effort; failure costs a re-analysis)."""
    try:
        target = _cache_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        files = {rel: {"fp": list(fp), "violations": [list(v) for v in vio]} for rel, (fp, vio) in cache.items()}
        payload = {"version": _CACHE_VERSION, "files": files}
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except (OSError, ValueError, TypeError):
        pass


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    """Normalize explicit file args to repo-relative ``*.py`` paths.

    Returns ``None`` when no file args survive (or none were given) — the
    caller then scans the whole repo, preserving the no-args (lint.yml CI)
    behaviour.  pre-commit passes absolute paths; lint.yml passes none.
    """
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


def _iter_py_files(paths=None):
    if paths is None:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            for name in files:
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(root, name), REPO)
                yield Path(rel), Path(root) / name
        return
    for rel in paths:
        p = REPO / rel
        if not p.is_file() or p.suffix != ".py":
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in Path(rel).parts):
            continue
        yield Path(rel), p


def _record_assignment_targets(node, record) -> None:
    """Feed every name *node* binds by assignment to ``record(name, lineno)``.

    Assignment only — a ``for`` target, a ``with ... as`` name and an
    ``except ... as`` name all bind too, but they are out of scope for this
    gate (see "Scope of the rule").
    """
    if isinstance(node, ast.Assign):
        for target in node.targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    record(sub.id, sub.lineno)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and isinstance(node.target, ast.Name):
        record(node.target.id, node.lineno)


def _module_assigned_names(tree: ast.Module) -> set[str]:
    """Names the module binds by assignment at module scope.

    Deliberately not every module-scope binding — see "Scope of the rule".
    """
    names: set[str] = set()
    for node in tree.body:
        _record_assignment_targets(node, lambda name, _lineno: names.add(name))
    return names


def _assignment_binds(fn) -> dict[str, int]:
    """``{name: first lineno}`` for names this function binds by ASSIGNMENT.

    Its own body only — nested ``def``/``class``/``lambda`` bodies are separate
    scopes and are visited on their own.
    """
    found: dict[str, int] = {}
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def record(name: str, lineno: int) -> None:
        if name not in found or lineno < found[name]:
            found[name] = lineno

    def walk(node) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nested):
                continue  # separate scope
            _record_assignment_targets(child, record)
            walk(child)

    body = fn.body if isinstance(fn.body, list) else [fn.body]
    for stmt in body:
        # The statement itself, then its children: iter_child_nodes only sees
        # the latter, so a top-level `x = 1` needs classifying on its own.
        _record_assignment_targets(stmt, record)
        walk(stmt)
    return found


def _reads_name(fn, name: str) -> bool:
    """True if this scope also LOADs *name* (→ ruff already sees it as F823)."""
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def walk(node) -> bool:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nested):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == name:
                return True
            if walk(child):
                return True
        return False

    body = fn.body if isinstance(fn.body, list) else [fn.body]
    return any((isinstance(s, ast.Name) and isinstance(s.ctx, ast.Load) and s.id == name) or walk(s) for s in body)


def _function_scopes(table, out):
    """Index every function symbol table by ``(name, lineno)``."""
    for child in table.get_children():
        if child.get_type() == "function":
            out[(child.get_name(), child.get_lineno())] = child
        _function_scopes(child, out)


def _violations_in(source: str, filename: str) -> list[tuple[int, str, str, str]]:
    """``(lineno, func, name, shape)`` for each undeclared module-global write."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []  # the syntax gates own that

    module_names = _module_assigned_names(tree)
    if not module_names:
        return []  # nothing can be shadowed; skip the symtable build (488/954 files)

    try:
        table = symtable.symtable(source, filename, "exec")
    except (SyntaxError, ValueError):
        return []

    scopes: dict[tuple[str, int], symtable.SymbolTable] = {}
    _function_scopes(table, scopes)

    out: list[tuple[int, str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        # symtable calls a lambda's scope "lambda"; ast.Lambda has no .name at
        # all. Spelling it "<lambda>" here made every lambda scope fail to match
        # and be skipped silently — a lambda cannot hold a statement, but it can
        # hold a walrus, which binds exactly the same discarded local.
        fname = getattr(node, "name", None) or "lambda"
        scope = scopes.get((fname, node.lineno))
        if scope is None:
            continue
        for name, lineno in sorted(_assignment_binds(node).items(), key=lambda kv: kv[1]):
            if name not in module_names:
                continue
            try:
                sym = scope.lookup(name)
            except KeyError:
                continue
            # Python's own resolver decides this, not a hand-rolled scope walk:
            # a declared global/nonlocal is not local, and a free variable is
            # already bound to an enclosing scope.
            if sym.is_declared_global() or sym.is_free() or not sym.is_local():
                continue
            shape = (
                "read-then-write — also an UnboundLocalError (ruff F823)"
                if _reads_name(node, name)
                else "write-only — silently discarded, invisible to ruff"
            )
            out.append((lineno, fname, name, shape))
    return sorted(out)


def main() -> int:
    findings: list[str] = []
    paths = _resolve_scan_paths([a for a in sys.argv[1:] if not a.startswith("--")])
    cache = {} if "--no-cache" in sys.argv else _load_cache()
    dirty = False
    for rel, path in sorted(_iter_py_files(paths)):
        rel = str(rel)  # cache keys / findings are repo-relative str (Path != str)
        fp = _stat_fingerprint(path)
        if fp is not None and rel in cache and cache[rel][0] == fp:
            violations = cache[rel][1]
        else:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            violations = _violations_in(source, rel)
            if fp is not None:
                cache[rel] = (fp, violations)
                dirty = True
        for lineno, func, name, shape in violations:
            findings.append(f"  {rel}:{lineno}  in {func}()  assigns {name!r}\n      {shape}")

    if dirty:
        _save_cache(cache)

    if not findings:
        print(
            "✅ No function assigns a module-level global without declaring it "
            "(0 tolerated — this gate has no baseline)"
        )
        return 0

    print(f"❌ {len(findings)} undeclared write(s) to a module-level global:\n")
    print("\n".join(findings))
    print(
        "\nAssigning a name inside a function binds a LOCAL unless the function"
        "\ndeclares `global <name>` (or `nonlocal` for an enclosing scope). The"
        "\nwrite-only shape raises nothing and ruff reports nothing — the module"
        "\nglobal simply keeps its old value forever. That silently froze"
        "\n.asicode/rg_fallback_counts.json at all-zeros (webapp _rg_dirty) and"
        "\npinned the REPL's /think autocomplete to the startup model"
        "\n(repl_impl _completer_provider)."
        "\n\nFix by declaring `global`/`nonlocal` if module state was meant, or by"
        "\nrenaming the local if it was meant as scratch."
        "\nDo NOT add a baseline for this."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
