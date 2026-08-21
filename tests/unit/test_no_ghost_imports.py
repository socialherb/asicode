"""Guard against "ghost imports": first-party imports of modules that do not exist.

Python resolves imports at call time, and this codebase imports lazily by
convention — usually inside a ``try: ... except ImportError/Exception:`` that
degrades gracefully. Combined, those two facts let an import path go stale
without anything failing: the module is gone or moved, the ImportError fires on
every call, the handler swallows it, and the feature is simply never there
again. Nothing in the test suite notices, because the fallback path is the one
being tested.

Four such imports were found in the library tree, each disabling a real feature
for as long as it had been broken:

* ``external_llm.exploration.fix_spec_learner`` — MOVED under lane/ by a6c2dc15
  ("planner subtree consolidation phase 2"); the caller kept the pre-move path,
  so the strategy-learning prompt hint was silently never added.
* ``external_llm.editor._editor_core.lane.graph.virtual_graph`` — the module
  lives at ``external_llm/graph/virtual_graph.py`` and never existed under
  lane/, so ``_extract_roles`` returned an empty set on every call and role
  diversity contributed nothing to candidate scoring.
* ``external_llm.section_patcher`` — never existed on any branch.
* ``external_llm.agent.cross_file_flow_resolver`` — deleted on purpose by
  2fe5abf0 ("semantic verification → dataflow verification"), but still
  imported.

The first two were repointed; the last two were genuinely gone, so their dead
branches were removed. This test keeps the class from coming back.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Packages whose modules must resolve to a real tracked file.
FIRST_PARTY_PREFIXES = ("external_llm.", "webapp.", "utils.")

# tools/ is developer scripting, excluded from the public snapshot, and
# currently carries 13 stale imports of its own (planner-era module names).
# Scoping the guard to the library keeps it enforceable today; widen it once
# tools/ is cleaned up.
SCANNED_ROOTS = ("external_llm/", "webapp/", "utils/", "scripts/")


def _tracked_py_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], cwd=REPO, capture_output=True, check=True
    ).stdout
    # -z + NUL split: git C-quotes non-ASCII paths in the default output.
    return [p.decode("utf-8") for p in out.split(b"\0") if p]


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    return set(_tracked_py_files())


def _module_exists(module: str, tracked: set[str]) -> bool:
    parts = module.split(".")
    return (
        "/".join(parts) + ".py" in tracked
        or "/".join(parts) + "/__init__.py" in tracked
    )


def _iter_first_party_imports(path: Path):
    """Yield (module, lineno) for every absolute first-party import, at any depth.

    ``ast.walk`` rather than ``iter_child_nodes``: the whole point is that these
    imports live inside functions, which a module-level-only scan cannot see.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — resolved against the package
                continue
            if node.module:
                yield node.module, node.lineno


@pytest.fixture(scope="module")
def imports_by_file(tracked):
    """{rel: [(module, lineno), ...]} for every scoped, tracked .py file.

    Parsed ONCE per test module rather than re-walking the whole tree once per
    test.  The three consumers each used to call _iter_first_party_imports over
    every scoped file independently (3x full-tree ast.walk, ~2s each); this
    fixture shares the single pass (P13, 2026-08-21).
    """
    out: dict[str, list[tuple[str, int]]] = {}
    for rel in sorted(tracked):
        if not rel.startswith(SCANNED_ROOTS):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        out[rel] = list(_iter_first_party_imports(path))
    return out


def test_every_first_party_import_resolves_to_a_real_module(tracked, imports_by_file):
    ghosts: list[str] = []
    for rel, imports in imports_by_file.items():
        for module, lineno in imports:
            if not module.startswith(FIRST_PARTY_PREFIXES):
                continue
            if not _module_exists(module, tracked):
                ghosts.append(f"{module}  (imported by {rel}:{lineno})")

    assert not ghosts, (
        "Ghost import(s) — the module does not exist, so this import raises on "
        "every call. If it sits in a try/except the feature behind it is simply "
        "off, silently:\n  " + "\n  ".join(ghosts)
    )


@pytest.mark.parametrize(
    "module",
    ["external_llm.section_patcher", "external_llm.agent.cross_file_flow_resolver"],
)
def test_removed_modules_are_not_imported_again(module, tracked, imports_by_file):
    """These two are genuinely gone. If one is ever written, drop it from this
    list and reinstate the call site deliberately — do not let an import of a
    non-existent module creep back in."""
    assert not _module_exists(module, tracked), (
        f"{module} now exists — remove it from this guard and restore its "
        f"call site on purpose rather than by accident"
    )
    offenders = [
        f"{rel}:{lineno}"
        for rel, imports in imports_by_file.items()
        for mod, lineno in imports
        if mod == module
    ]
    assert not offenders, f"{module} is imported again at: {offenders}"
