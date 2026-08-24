"""Tests for scripts/check_tree_sitter_core.py — the env-wide tree-sitter core guard.

Why a separate test file when the version assertion already lives in
tests/unit/agent/test_tree_sitter_core_version.py: that guard runs ONLY under
pytest, and pytest is absent in some dev environments (the 2026-08-24
incident: venv had no pytest while holding the segfaulting 0.26.0 core, so the
guard never executed and the gate segfaulted). This script is wired into
pre-commit and CI with host python3 — evaluation surfaces that do not require
pytest. These tests pin its decision logic.
"""

import importlib.util
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_tree_sitter_core.py"
_spec = importlib.util.spec_from_file_location("check_tree_sitter_core", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]

_LOWER_BOUND = SpecifierSet(">=0.25,<0.26")


def test_not_installed_fails_closed():
    problem = g._core_problem(installed=None, declared=[_LOWER_BOUND])
    assert problem and "not installed" in problem


def test_segfaulting_bound_fails_closed():
    problem = g._core_problem(installed=Version("0.26"), declared=[_LOWER_BOUND])
    assert problem and "0.26" in problem and "crashes" in problem


def test_above_bound_fails_closed():
    problem = g._core_problem(installed=Version("0.27.1"), declared=[_LOWER_BOUND])
    assert problem and "0.26" in problem


def test_undeclared_dependency_fails_closed():
    problem = g._core_problem(installed=Version("0.25.2"), declared=None)
    assert problem and "declares no tree-sitter" in problem


def test_healthy_env_passes():
    problem = g._core_problem(installed=Version("0.25.2"), declared=[_LOWER_BOUND])
    assert problem is None


def test_installed_violates_declared_constraint_fails():
    # installed satisfies the segfault bound but violates a declared constraint
    problem = g._core_problem(installed=Version("0.24.9"), declared=[_LOWER_BOUND])
    assert problem and "violates the declared constraint" in problem


def test_undeclared_in_pyproject_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_PYPROJECT", tmp_path / "missing.toml")
    declared = g._declared_specifiers()
    assert declared is None
    problem = g._core_problem(installed=Version("0.25.2"), declared=declared)
    assert problem and "declares no tree-sitter" in problem


def test_pyproject_without_tree_sitter_declares_none(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["requests>=2"]\n', encoding="utf-8")
    original = g._PYPROJECT
    try:
        g._PYPROJECT = pyproject
        assert g._declared_specifiers() is None
    finally:
        g._PYPROJECT = original
