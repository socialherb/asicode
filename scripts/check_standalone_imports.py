#!/usr/bin/env python3
"""Check every module imports STANDALONE (fresh interpreter) vs baseline.

A module that only imports successfully because something else was imported
FIRST in the same process carries a hidden import-order contract. The
documented failure is ``external_llm.repl.repl_impl``: it does ``import asi``
at module level, and ``asi.py`` re-exports 76 symbols FROM repl_impl at the
bottom of the file — so ``import asi`` first works (the cycle resolves one
way) but ``import external_llm.repl.repl_impl`` first raises ImportError.
Every new test/tool/subprocess entry point that imports repl_impl first
would crash at import time.

This gate imports every in-scope module in its OWN subprocess (isolation is
the point: a shared interpreter masks order dependencies) and fails on any
module that is not baselined.

Baseline key: module name (``external_llm.repl.repl_impl``) — drift-proof
(no line numbers).

Usage:
    python scripts/check_standalone_imports.py            # check (working tree, full)
    python scripts/check_standalone_imports.py --index-only
        # check the HEAD+index snapshot instead of the working tree (CI:
        # exactly what a commit would ship).
    python scripts/check_standalone_imports.py --write-baseline  # regen
    python scripts/check_standalone_imports.py <file>.py ...      # per-file
        # pre-commit incremental mode: import only the given changed files.
        # No-args (lint.yml CI) scans the full repo; keys are module names in
        # both modes.

Scope: production code (external_llm/, services/, webapp/, root asi.py).
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "standalone_import_baseline.txt"

_IMPORT_TIMEOUT = 120  # heavy deps (torch etc.) can take a while to import
_MAX_WORKERS = 8

_SKIP_DIRS = frozenset(
    {
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
        ".git",
    }
)
_SCAN_ROOTS = ("external_llm", "services", "webapp")


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _in_scope(rel: str) -> bool:
    p = Path(rel)
    return p.suffix == ".py" and not _should_skip(p) and (p == Path("asi.py") or p.parts[0] in _SCAN_ROOTS)


def _module_name(rel: str) -> str | None:
    """Repo-relative .py path -> importable module name (None if not)."""
    p = Path(rel)
    parts = list(p.parts[:-1]) if p.name == "__init__.py" else [*p.parts[:-1], p.stem]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _iter_modules_under(root: Path) -> list[str]:
    """In-scope module names for a tree rooted at *root* (repo or tmp copy)."""
    mods: list[str] = []
    for scan in _SCAN_ROOTS:
        d = root / scan
        if d.is_dir():
            for p in d.rglob("*.py"):
                if _should_skip(p):
                    continue
                name = _module_name(str(p.relative_to(root)))
                if name:
                    mods.append(name)
    if (root / "asi.py").exists():
        mods.append("asi")
    return sorted(set(mods))


def _iter_worktree_modules(paths: list[str] | None = None) -> list[str]:
    mods: list[str] = []
    if paths is not None:
        for rel in paths:
            if _in_scope(rel):
                name = _module_name(rel)
                if name:
                    mods.append(name)
        return mods
    return _iter_modules_under(REPO)


def _materialize_index(tmp: Path) -> list[str] | None:
    """Materialize the git INDEX (what a commit would ship) under *tmp* via a
    single ``git checkout-index`` pass; return in-scope module names. None on
    git failure (caller falls back to the working tree).

    The FULL tracked tree is copied, not just the scan roots: in-scope modules
    import root-level siblings (common.py, utils/, context_collector.py,
    path_security.py, ...) that sit OUTSIDE _SCAN_ROOTS. Copying only the
    scan roots phantom-failed those imports in --index-only mode — lint.yml
    has been red on exactly this since v0.2.21 (56 ``No module named
    'common'`` style failures that never reproduce in the working tree,
    where the siblings are present by construction). checkout-index is one
    subprocess for ~930 tracked .py files; the old per-file ``git show``
    loop would have been ~930 subprocesses.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "checkout-index", "-a", "-f", f"--prefix={tmp}/"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return _iter_modules_under(tmp)


def _tracked_first_party_tops() -> set[str]:
    """Top-level import names the repo itself ships (tracked .py files).

    ``common.py`` -> ``common``, ``utils/string_helper.py`` -> ``utils``.
    Used to tell a missing FIRST-PARTY module (real standalone failure)
    from a missing third-party dependency (environment artifact).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    tops: set[str] = set()
    for rel in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if not rel:
            continue
        stem = Path(rel).parts[0]
        stem = Path(stem).stem
        if stem.isidentifier():
            tops.add(stem)
    return tops


_MISSING_DEP_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def _third_party_only_missing(
    err_text: str,
    first_party: set[str],
) -> set[str] | None:
    """Missing THIRD-PARTY top-level names if the failure is only that; else None.

    The gate's contract is first-party import ORDER — whether a third-party
    dependency happens to be *installed* is environment policy (lint.yml's
    baseline-diff job installs only ruff, so ~50 modules fail on
    requests/rich there while passing everywhere the package is installed).
    A plain missing-dependency ModuleNotFoundError is therefore skipped,
    not failed and not baselined. Anything else — a cycle, cannot-import-name,
    a missing FIRST-PARTY module — is a real failure in every environment.
    """
    missing = _MISSING_DEP_RE.findall(err_text)
    if not missing:
        return None
    tops = {name.split(".")[0] for name in missing}
    if tops & first_party:
        return None  # a first-party module failed to import — real failure
    return tops


def _try_import(module: str, cwd: Path) -> str:
    """Import *module* in a fresh interpreter; return '' on success, else a
    short stderr excerpt."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=cwd,
            capture_output=True,
            timeout=_IMPORT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {_IMPORT_TIMEOUT}s"
    if proc.returncode == 0:
        return ""
    err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
    return "\n".join(err[-8:])


def _check(modules: list[str], cwd: Path) -> dict[str, str]:
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for module, err in zip(modules, ex.map(lambda m: _try_import(m, cwd), modules), strict=True):
            if err:
                failures[module] = err
    return failures


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    out: set[str] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def _write_baseline(modules: set[str]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# asicode standalone-import baseline — modules that are EXPECTED to\n"
        "# fail a fresh-interpreter `import <module>` (verified import-order\n"
        "# dependencies / missing optional deps). A module listed here is a\n"
        "# known, intentional exception; a NET-NEW failure fails the gate.\n"
        "#\n"
        "# Generated by `scripts/check_standalone_imports.py --write-baseline`.\n"
        "# Key: module name (drift-proof). Stale entries (listed but importing\n"
        "# fine) are reported but do not fail — re-run --write-baseline after\n"
        "# fixing a baselined module to drop its entry.\n"
        "#\n"
    )
    lines = sorted(modules)
    BASELINE.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_scan_paths(args: list[str]) -> list[str] | None:
    out: list[str] = []
    for a in args:
        rel = os.path.relpath(Path(a).resolve(), Path(REPO).resolve())
        if rel.endswith(".py") and not rel.startswith(".."):
            out.append(rel)
    return out or None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    index_only = "--index-only" in sys.argv
    write_baseline = "--write-baseline" in sys.argv

    if args:
        # pre-commit incremental mode: import only the changed files (worktree).
        modules = _iter_worktree_modules(_resolve_scan_paths(args))
        failures = _check(modules, REPO)
    elif index_only:
        with tempfile.TemporaryDirectory(prefix="asi-import-") as td:
            tmp = Path(td)
            modules = _materialize_index(tmp)
            if modules is None:
                modules = _iter_worktree_modules()
                failures = _check(modules, REPO)
            else:
                failures = _check(modules, tmp)
    else:
        modules = _iter_worktree_modules()
        failures = _check(modules, REPO)

    first_party = _tracked_first_party_tops()
    real: dict[str, str] = {}
    env_skipped: dict[str, set[str]] = {}
    for module, err in failures.items():
        dep_tops = _third_party_only_missing(err, first_party)
        if dep_tops is not None:
            env_skipped[module] = dep_tops
        else:
            real[module] = err

    if write_baseline:
        _write_baseline(set(real))
        src = "HEAD+index" if index_only else ("given files" if args else "working tree")
        print(f"✅ Baseline written: {BASELINE} ({len(real)} entries, from {src})")
        return 0

    if env_skipped:
        counts = Counter(dep for tops in env_skipped.values() for dep in sorted(tops))
        detail = ", ".join(f"{dep} x{n}" for dep, n in counts.most_common())
        print(
            f"⏭ {len(env_skipped)} module(s) skipped: third-party import(s) "
            f"missing in this environment ({detail}) — dependency "
            f"presence is the unit-test job's contract, not import order."
        )

    baseline = _load_baseline()
    new_failures = {m: e for m, e in real.items() if m not in baseline}
    stale = sorted(baseline - set(real))

    if stale:
        print(
            f"Note: {len(stale)} stale baseline entr{'y' if len(stale) == 1 else 'ies'} "
            f"(importing fine now): {', '.join(stale)} — re-run --write-baseline"
        )

    if not new_failures:
        skipped_note = f", {len(env_skipped)} env-skipped" if env_skipped else ""
        print(
            f"✅ All {len(modules)} modules import standalone "
            f"({len(real)} baselined failure{'s' if len(real) != 1 else ''}"
            f"{skipped_note})"
        )
        return 0

    print(f"❌ {len(new_failures)} module(s) FAIL standalone import (not in baseline):\n")
    for m in sorted(new_failures):
        print(f"  {m}\n{new_failures[m]}\n")
    print("A module that only imports after something else was imported first carries")
    print("a hidden import-order contract — every fresh entry point can hit it. Break")
    print("the cycle, or add the module to the baseline only if it is a genuine")
    print("environment-dependent exception (e.g. missing optional dependency).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
