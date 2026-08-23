"""Gate: subprocess calls in external_llm production code must be timeout-bounded.

The 27th audit round found unbounded git subprocess calls in tools/ scripts
(see test_tools_git_timeout_gate.py) — including ``git checkout -- .`` without
``timeout=``, where a hung git (index.lock contention, network FS) would block a
bench run forever and leave the repo mid-restore. external_llm is the agent's
production surface: a hung subprocess there blocks the whole agent turn and the
user's REPL.

A manual sweep found the current external_llm code fully compliant (run-family
calls always pass ``timeout=``; the Popen sites use deadline poll loops,
``communicate(timeout=)``, or hand the proc to bounded helpers such as
``_capture_bounded``). This gate exists to keep it that way — it is a
regression guard, not a verifier of past work.

Detector implementations live in subprocess_gate_detectors.py and are shared
with the webapp gate (test_webapp_subprocess_gate.py) so both surfaces enforce
identical rules. Rules:

   1. every ``subprocess.run/check_output/check_call/call`` must pass ``timeout=``;
   2. ``os.popen``/``os.system`` are banned outright (no timeout mechanism);
   3. ``subprocess.Popen`` with a visible ``stdout=``/``stderr=subprocess.PIPE``
      must be bounded in the same function: ``proc.wait(timeout=...)`` /
      ``proc.communicate(timeout=...)``, or the proc handed to another call
      (delegated bound, e.g. ``_capture_bounded(proc, timeout, ...)``).
      Popen without PIPE (DEVNULL / log file) is fire-and-forget, and a Popen
      with a ``**kwargs`` splat cannot be inspected statically — both exempt.

``detector_self_tests()`` pins the detectors' sensitivity (mutation-style):
the compliance parametrization would pass vacuously if a detector broke, so
every gate file re-runs the synthetic violation/acceptance cases.
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

EXTERNAL_LLM_DIR = Path(__file__).resolve().parents[2] / "external_llm"

_PATHS = [str(p) for p in prod_py_files(EXTERNAL_LLM_DIR)]
_IDS = [str(p.relative_to(EXTERNAL_LLM_DIR)) for p in prod_py_files(EXTERNAL_LLM_DIR)]


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_run_family_calls_have_timeout(path: str):
    """subprocess.run/check_output/check_call/call must pass timeout= (hung child = wedged agent turn)."""
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
