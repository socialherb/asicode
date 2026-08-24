#!/usr/bin/env python3
"""Check the installed tree-sitter CORE satisfies the declared segfault-safe bound.

Why this gate exists (root cause: 2026-08-24)
---------------------------------------------
The tree-sitter core was not declared in pyproject for a long time: only
``tree-sitter-language-pack`` was listed, and its metadata says a bare
``Requires: tree-sitter`` with no bound, so a fresh install resolved the core
to whatever was newest. On 0.26.0 the core segfaults during the tree-sitter
node walk (measured 16/40 crashes vs 0/40 on 0.25.2; see
tests/unit/agent/test_tree_sitter_core_version.py).

The version-guard unit test only runs where pytest is installed. On 2026-08-24
the dev venv had NO pytest, so that guard never executed there — while the
venv had been upgraded to the segfaulting 0.26.0 core and the whole gate
segfaulted. That is the same dev-masks-shipping shape the export check exists
to catch, and it shows the guard needs a gate surface that does not depend on
pytest being installed. This script is that surface: wired into pre-commit
(host python3) and into the CI steps whose environment actually installs the
package (lint.yml's structural-scan step, release.yml's test job).

Fail-closed semantics (an env problem must fail the gate, never pass it):
  - core not installed          -> FAIL (the tree-sitter grammar path cannot run)
  - core >= 0.26                -> FAIL (the measured segfault bound; raising it
                                  requires a fresh 20-trial zero, not release
                                  notes or one passing run)
  - installed core violates the declared pyproject constraint -> FAIL (the
    env-vs-declaration drift class that let this ship)

Positional file args (pre-commit passes changed files) are ignored: this is an
environment-wide check, not a per-file scan.

Usage:
    python scripts/check_tree_sitter_core.py
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

try:
    from packaging.requirements import Requirement
    from packaging.version import Version

    _PACKAGING_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # packaging is a CORE dependency; fail closed otherwise
    _PACKAGING_IMPORT_ERROR = exc

_REPO = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO / "pyproject.toml"
# The measured segfault bound — stated independently of pyproject's own text so
# that loosening the constraint cannot silently loosen its own guard.
_SEGFAULT_BOUND = Version("0.26")


def _declared_specifiers() -> list | None:
    """The pyproject tree-sitter requirement specifiers, as SpecifierSets.

    Returns ``None`` when pyproject.toml is absent or declares no tree-sitter
    requirement — the exact UNDECLARED-dependency state that caused the
    incident is itself a failure.
    """
    if not _PYPROJECT.is_file():
        return None
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    out = []
    for raw in data.get("project", {}).get("dependencies", []):
        req = Requirement(raw)
        if req.name.lower().replace("_", "-") == "tree-sitter":
            out.append(req.specifier)
    return out or None


def _core_problem(installed: Version | None, declared: list | None) -> str | None:
    """One human-readable problem string, or ``None`` when the env is healthy.

    ``installed`` is None when the core is not installed; ``declared`` is the
    list of SpecifierSets from _declared_specifiers (None = not declared).
    """
    if installed is None:
        return (
            "tree-sitter CORE is not installed — the tree-sitter grammar path "
            "cannot run. The core is a declared dependency; reinstall the "
            "package (or install it explicitly)."
        )
    if installed >= _SEGFAULT_BOUND:
        return (
            f"tree-sitter core {installed} is at/above the segfaulting 0.26 "
            "(measured 16/40 crashes vs 0/40 on 0.25.2 — see "
            "tests/unit/agent/test_tree_sitter_core_version.py). Do not raise "
            "the bound on release notes or a single passing run: re-run the "
            "20-trial matrix and require a full zero."
        )
    if declared is None:
        return (
            "pyproject.toml declares no tree-sitter CORE requirement — the "
            "core must be declared WITH an upper bound. Leaving it to "
            "tree-sitter-language-pack's unbounded `Requires: tree-sitter` is "
            "what let a fresh install resolve to the segfaulting 0.26.0."
        )
    for spec in declared:
        if not spec.contains(installed, prereleases=True):
            return (
                f"installed tree-sitter core {installed} violates the declared "
                f"constraint {spec} — the environment and the shipped metadata "
                "disagree about what is supported."
            )
    return None


def main() -> int:
    if _PACKAGING_IMPORT_ERROR is not None:
        print(
            f"❌ packaging import failed ({_PACKAGING_IMPORT_ERROR!r}) — failing "
            "closed; packaging is a core dependency",
            file=sys.stderr,
        )
        return 1
    try:
        installed = Version(version("tree-sitter"))
    except PackageNotFoundError:
        installed = None
    declared = _declared_specifiers()
    problem = _core_problem(installed, declared)
    if problem is None:
        specs = ", ".join(str(s) for s in declared) if declared else "?"
        print(f"✅ tree-sitter core {installed} satisfies the declared constraint ({specs})")
        return 0
    print(f"❌ {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
