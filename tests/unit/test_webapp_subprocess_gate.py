"""Gate: subprocess calls in webapp (UI server) production code must be timeout-bounded.

Same rules and shared detectors as test_external_llm_subprocess_gate.py
(see subprocess_gate_detectors.py): a hung subprocess in the UI server wedges
an HTTP request handler — the browser spinner never resolves and the agent
keeps waiting on a response that can never arrive.

Covered tree: ``webapp/`` — the FastAPI server for the UI, including
``webapp/ui/ui_tools.py``, ``webapp/run_helpers.py``, ``webapp/run_store.py``
and ``webapp/routes/*``.

A manual sweep found the current webapp code fully compliant:
``ui_tools.py`` passes ``timeout=10`` / ``timeout=_OSASCRIPT_PICKER_TIMEOUT_S``
on every run-family call and bounds its ``rg`` Popen with a deadline poll loop
plus ``proc.communicate(timeout=3)``; ``run_helpers.py`` bounds its git Popen
with ``select`` polling and ``proc.wait(timeout=...)``; ``run_store.py`` and
``edit_run.py`` pass ``timeout=10`` (the latter through a plain-assignment
alias ``_sp = subprocess``, which the shared detectors track to fixed point).

The gate is a regression guard: compliance parametrizations over the current
tree would pass vacuously if a detector broke, so ``detector_self_tests()``
re-pins every detector's sensitivity on synthetic violations and safe forms.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.subprocess_gate_detectors import (
    detector_self_tests,
    os_popen_system_calls,
    popen_unbounded,
    prod_py_files,
    run_family_without_timeout,
)

WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"

_PATHS = [str(p) for p in prod_py_files(WEBAPP_DIR)]
_IDS = [str(p.relative_to(WEBAPP_DIR)) for p in prod_py_files(WEBAPP_DIR)]


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_run_family_calls_have_timeout(path: str):
    """subprocess.run/check_output/check_call/call must pass timeout= (hung child = wedged request handler)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    missing = run_family_without_timeout(tree)
    assert not missing, (
        f"{Path(path).name}: {len(missing)} subprocess call(s) without timeout= at lines {[ln for _, ln in missing]}"
    )


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_no_os_popen_system(path: str):
    """os.popen/os.system have no timeout mechanism — banned in production code."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    hits = os_popen_system_calls(tree)
    assert not hits, (
        f"{Path(path).name}: {len(hits)} os.popen/os.system call(s) at "
        f"lines {[ln for _, ln in hits]} — convert to subprocess.run(timeout=...) "
        "or run_bounded_subprocess"
    )


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_popen_with_pipe_is_bounded(path: str):
    """Popen reading PIPE output must be bounded in the same function or handed to a bounded helper."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    hits = popen_unbounded(tree)
    assert not hits, (
        f"{Path(path).name}: {len(hits)} unbounded Popen call(s) at lines {[(ln, why) for _, ln, why in hits]}"
    )


def test_detectors_are_alive():
    """Synthetic violation/acceptance cases — prevents the gates from passing vacuously."""
    detector_self_tests()
