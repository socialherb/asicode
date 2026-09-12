"""Pin the coverage-report contract that keeps a green suite from reddening CI.

Measured incident (lint runs 34687877882 and 34689020064): the pytest step was
green (16425 passed on the second) and the job still failed, because a measured
file was gone before `coverage report` ran — a test-spawned interpreter had
imported an `asi-policy-*` temp-tree copy of the package (the a1_coverage.pth
measures every process) and its TemporaryDirectory then deleted the tree. A
missing source ABORTS the report: rc=1, no table, no fail_under verdict —
`No source for code: '/tmp/asi-policy-.../external_llm/__init__.py'`.

The fix is `[tool.coverage.report] ignore_errors = true` in pyproject.toml, the
documented remedy for "files deleted between the run and the report". This
module reproduces the shape with a real coverage run (measure -> delete ->
report) and pins both halves: with the repo config the report tolerates the gap
and still shows the surviving files; without it the report dies as CI did.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _measure_then_delete(tmp_path: Path) -> dict[str, str]:
    """Run coverage over a package, then delete one of its files (CI's shape)."""
    pkg = tmp_path / "measured_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "keeper.py").write_text("KEEP = 2\n", encoding="utf-8")
    victim = pkg / "victim.py"
    victim.write_text("VALUE = 1\n", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("import measured_pkg.keeper\nimport measured_pkg.victim\n", encoding="utf-8")

    env = {**os.environ, "COVERAGE_FILE": str(tmp_path / "cov.db"), "PYTHONPATH": str(tmp_path)}
    # This test itself runs under scripts/cov.sh in CI, and coverage exports
    # COVERAGE_RCFILE (the repo pyproject) into every process it instruments.
    # If that leaks, the child reads the REPO config: `source=[...]` filters
    # this fixture's package out (0 files measured) and `parallel=true` writes
    # a SUFFIX file, so the report's combine folds it into an empty base and
    # prints "No data to report" instead of exercising the cwd-config contract
    # below (measured: the nested run under cov.sh). Only the cwd may pick the
    # config, so every steering variable the outer run could leave behind goes.
    for key in ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG", "COVERAGE_RCFILE", "COV_DIR"):
        env.pop(key, None)

    run = subprocess.run(
        [sys.executable, "-m", "coverage", "run", str(runner)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    victim.unlink()  # what TemporaryDirectory cleanup does to the CI temp tree
    return env


def _report(cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """`coverage report` with the ratchet disabled: this data is a slice."""
    return subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--fail-under=0"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_repo_config_tolerates_a_measured_file_that_vanished(tmp_path: Path) -> None:
    """With pyproject's [tool.coverage.report] the report survives the gap."""
    env = _measure_then_delete(tmp_path)
    proc = _report(REPO, env)  # cwd=REPO: this is where pyproject.toml lives
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "keeper.py" in proc.stdout, proc.stdout  # the survivors still report
    assert "No source for code" in proc.stdout + proc.stderr  # said out loud, not silent


def test_without_that_config_the_same_state_reds_the_report(tmp_path: Path) -> None:
    """The red half: this IS the CI signature (rc=1, table lost entirely)."""
    env = _measure_then_delete(tmp_path)
    proc = _report(tmp_path, env)  # no pyproject here: coverage defaults apply
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "TOTAL" not in proc.stdout
    assert "No source for code" in proc.stdout + proc.stderr
