"""The installed tree-sitter CORE must satisfy what pyproject declares.

This exists because the core was not declared at all: only
``tree-sitter-language-pack`` was listed, and its own metadata says a bare
``Requires: tree-sitter`` with no bound, so a fresh install resolved the core
to whatever was newest. On 0.26.0 that segfaults.

Measured in a clean ``python:3.12-slim`` container, 20 trials per combination,
calling ``SymbolSearcher.get_file_outline()`` on one ordinary Python file:

===============  ===============  ==========
core             language-pack    crashed
===============  ===============  ==========
0.26.0           1.13.5           7/20
0.26.0           1.12.5           9/20
0.25.2           1.13.5           0/20
0.25.2           1.12.5           0/20
===============  ===============  ==========

The core decides; the pack is irrelevant. The crash is a SIGSEGV inside the
node walk (``get_node_text`` -> ``node.start_byte``), nondeterministic and
landing at a different site each run — GC, fork, ``realpath``. That
nondeterminism is why this is guarded by a version assertion rather than by a
behavioural test: at a ~35-45% rate, a passing run proves nothing.

Not test-only: ``get_file_outline`` is on the ``read_file`` path
(``_over_cap_guidance``, for files past the output cap), so an unpinned install
could segfault the CLI on an ordinary large-file read. The development
environment masked it by already having core 0.25.2 installed — the same
dev-masks-shipping shape that the export + clean-venv check exists to catch.

To raise the bound: re-run the trial matrix against the newer core and require
a full 20-trial zero. Do not raise it on the strength of release notes, and do
not raise it on a single passing run.
"""
from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is a declared dependency
    pytest.skip("packaging unavailable", allow_module_level=True)

# pyproject.toml is read from the source tree rather than from installed
# metadata: an editable install caches the metadata generated at install time,
# so a freshly EDITED constraint would not be visible until someone reinstalls.
# That lag is exactly the dev-vs-shipped drift this module is about, and it
# would have made these tests report on a stale declaration.
_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def _declared_requirements() -> list[str]:
    if not _PYPROJECT.is_file():
        pytest.skip(f"pyproject.toml not found at {_PYPROJECT}")
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("project", {}).get("dependencies", []))


def _installed(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - grammar stack is required
        pytest.skip(f"{name} is not installed in this environment")


def test_core_is_below_the_segfaulting_release():
    """The concrete bound, stated independently of pyproject's own text.

    Restated here rather than parsed from the declaration so that loosening the
    constraint cannot silently loosen its own guard — changing the bound has to
    be a deliberate edit in two places, one of which carries the measurement.
    """
    assert Version(_installed("tree-sitter")) < Version("0.26"), (
        "tree-sitter core >= 0.26 segfaults during the tree-sitter node walk "
        "(measured 16/40 crashes vs 0/40 on 0.25.2). See this module's docstring "
        "before raising the bound."
    )


def test_pyproject_actually_declares_the_core():
    """The root cause was an UNDECLARED dependency, not merely an unbounded one.

    Relying on language-pack to pull the core in left the resolver free to pick
    any version, so this asserts the declaration exists at all — the thing whose
    absence caused the incident.
    """
    declared = _declared_requirements()
    names = {Requirement(r).name.lower().replace("_", "-") for r in declared}
    assert "tree-sitter" in names, (
        "pyproject must declare the tree-sitter CORE explicitly; leaving it to "
        "tree-sitter-language-pack's unbounded `Requires: tree-sitter` is what "
        "let a fresh install resolve to the segfaulting 0.26.0"
    )


def test_the_installed_core_satisfies_the_declared_constraint():
    """Environment-vs-declaration drift check.

    The development environment happened to hold a working core while the
    declared constraint allowed a broken one, so `it works here` and `it works
    for a new install` had drifted apart with nothing reporting it.
    """
    declared = _declared_requirements()
    installed = Version(_installed("tree-sitter"))
    for raw in declared:
        req = Requirement(raw)
        if req.name.lower().replace("_", "-") != "tree-sitter":
            continue
        assert req.specifier.contains(installed, prereleases=True), (
            f"installed tree-sitter {installed} violates the declared "
            f"constraint {req.specifier} — the environment and the shipped "
            f"metadata disagree about what is supported"
        )
        return
    pytest.fail("no tree-sitter requirement found to check against")
