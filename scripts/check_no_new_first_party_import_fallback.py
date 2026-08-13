#!/usr/bin/env python3
"""Gate: no NEW ``try: <first-party import> / except ImportError`` fallbacks.

A first-party module (declared in pyproject.toml ``py-modules`` or
``packages.find.include`` — plus any relative import) can never fail to
import at runtime in this repo.  An ``except ImportError`` around it is a
dead fallback whose except body is dead code that drifts silently
(P26-1: the content-ratio guard was wired only to a legacy branch nobody
reached, and the fallback tail rotted).  Third-party lazy imports (numpy,
MCP SDK, …) are legitimate and excluded.

This gate does NOT touch pre-existing fallbacks — they are frozen as a
baseline (``scripts/first_party_import_fallback_baseline.txt``).  It catches
only NET-NEW ones, mirroring the F401/F811/F821 baseline-diff hooks.

Key: ``<rel_path>::<scope_qualname>::<ordinal>``

    scope_qualname = dotted names of enclosing FunctionDef/AsyncFunctionDef/
                     ClassDef, or ``<module>`` at module level.
    ordinal        = 0-based index of this Try node among Try nodes in the
                     SAME nearest scope that pair a first-party import with
                     an ``except ImportError``, in source order.

Usage:
    python scripts/check_no_new_first_party_import_fallback.py            # check (worktree)
    python scripts/check_no_new_first_party_import_fallback.py --index-only
        # check the HEAD+index snapshot (git ls-files + `git show :<path>`)
        # instead of the working tree — the hook must see exactly what the
        # commit ships (stash-trap: pre-commit stashes unstaged files).
    python scripts/check_no_new_first_party_import_fallback.py --write-baseline
    python scripts/check_no_new_first_party_import_fallback.py <file>.py ... [--index-only]
        # scan only the given files (pre-commit per-file mode).

Scope: production code only (external_llm/, services/, webapp/, root asi.py).
Test files are intentionally excluded (scaffolding legitimately fakes
missing deps).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "first_party_import_fallback_baseline.txt"

_SCAN_ROOTS = ("external_llm", "services", "webapp")
_SKIP_DIRS = frozenset({
    "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules",
    ".venv", "venv", "env", ".tox", "dist", "build", ".eggs", ".git",
})


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIRS for part in path.parts)


def _in_scope(rel: str) -> bool:
    p = Path(rel)
    return p.suffix == ".py" and p.parts[0] in _SCAN_ROOTS and not _should_skip(p)


def _first_party_tops() -> frozenset[str]:
    """Top-level module names that are first-party in this repo."""
    tops: set[str] = set()
    try:
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return frozenset()
    setup = data.get("tool", {}).get("setuptools", {})
    for mod in setup.get("py-modules", []) or []:
        tops.add(str(mod).split(".")[0])
    for pat in (setup.get("packages", {}).get("find", {}) or {}).get("include", []) or []:
        tops.add(str(pat).rstrip("*").rstrip("."))
    return frozenset(t for t in tops if t)


_FIRST_PARTY = _first_party_tops()


def _catches_importerror(handler: ast.ExceptHandler) -> bool:
    """True if this except clause catches ImportError (alone or in a tuple)."""
    if handler.type is None:
        return False  # bare except — a different gate's concern
    return any(
        isinstance(n, ast.Name) and n.id == "ImportError"
        for n in ast.walk(handler.type)
    )


def _is_first_party_import(node: ast.AST) -> bool:
    """True if the import statement targets a first-party module."""
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:  # relative import — same package, always first-party
            return True
        top = (node.module or "").split(".")[0]
    elif isinstance(node, ast.Import):
        top = (node.names[0].name or "").split(".")[0]
    else:
        return False
    return top in _FIRST_PARTY


def _try_body_imports(body: list[ast.stmt]) -> list[ast.AST]:
    """Import nodes in the try body, excluding those inside nested Try nodes
    (the inner try owns its own except handling)."""
    found: list[ast.AST] = []
    for stmt in body:
        if isinstance(stmt, ast.Try):
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                found.append(sub)
    return found


def _scan_source(source: str) -> list[tuple[str, int]]:
    """Return (scope_qualname, ordinal) pairs for first-party ImportError
    fallbacks, in source order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str, int]] = []
    counters: dict[str, int] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_catches_importerror(h) for h in node.handlers):
            continue
        imports = _try_body_imports(node.body)
        if not any(_is_first_party_import(i) for i in imports):
            continue
        # nearest enclosing scope qualname
        scope: list[str] = []
        for parent in _ancestors(tree, node):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope.append(parent.name)
        qual = ".".join(reversed(scope)) or "<module>"
        ordinal = counters.get(qual, 0)
        counters[qual] = ordinal + 1
        out.append((qual, ordinal))
    return out


def _ancestors(tree: ast.AST, node: ast.AST) -> list[ast.AST]:
    """Parent chain from tree root to node (inclusive)."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is node:
                return [*_ancestors(tree, parent), parent]
    return []


def _read_sources(rel_paths: list[str], index_only: bool) -> dict[str, str]:
    """rel_path -> source text (working tree or HEAD+index snapshot)."""
    sources: dict[str, str] = {}
    for rel in rel_paths:
        try:
            if index_only:
                out = subprocess.run(
                    ["git", "show", f":{rel}"],
                    cwd=REPO, capture_output=True, text=True, timeout=30,
                    check=False,
                )
                if out.returncode == 0:
                    sources[rel] = out.stdout
            else:
                sources[rel] = (REPO / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return sources


def _git_list_scope_py() -> list[str] | None:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
        timeout=60, check=False,
    )
    if out.returncode != 0:
        return None
    return [r for r in out.stdout.splitlines() if _in_scope(r)]


def main() -> int:
    index_only = "--index-only" in sys.argv
    write_baseline = "--write-baseline" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]

    rels = [p for p in paths if _in_scope(p)] if paths else _git_list_scope_py() or []
    if not rels:
        print("no in-scope python files")
        return 0

    sources = _read_sources(rels, index_only)
    keys: list[str] = []
    for rel in sorted(sources):
        for qual, ordinal in _scan_source(sources[rel]):
            keys.append(f"{rel}::{qual}::{ordinal}")

    baseline = set()
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                baseline.add(line)

    if write_baseline:
        header = (
            "# First-party ImportError fallbacks (dead fallbacks: a first-party\n"
            "# module can never fail to import at runtime).  Regenerate with:\n"
            "#   scripts/check_no_new_first_party_import_fallback.py --write-baseline\n"
        )
        BASELINE.write_text(header + "\n".join(sorted(keys)) + ("\n" if keys else ""), encoding="utf-8")
        print(f"baseline written: {len(keys)} entries -> {BASELINE.name}")
        return 0

    new = [k for k in keys if k not in baseline]
    for k in new:
        print(f"NEW first-party ImportError fallback: {k}")
    if new:
        print("A first-party module cannot fail to import — the except body is dead code.")
        print("Either drop the try/except (import at module level) or re-baseline with")
        print("`--write-baseline` if this is an intentional fallback.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
