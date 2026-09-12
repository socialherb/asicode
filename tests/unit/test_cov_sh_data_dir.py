"""Contract: scripts/cov.sh tells every coverage process where to write, absolutely.

Why this file exists (measured 2026-09-12, CI run 34680698084): ``COV_DIR=
coverage-data`` — the value lint.yml passes — reached ``COVERAGE_FILE`` as a
*relative* path. coverage resolves a relative data-file path against the cwd of
every process it instruments, so xdist workers and test-spawned subprocesses
(whose cwd is a pytest tmp dir) wrote their parallel data under
``<child-cwd>/coverage-data/``: one full ``tests/unit`` run left 99 data files /
5.3MB across 69 directories under pytest's tmp dirs, every one of them outside
the cwd ``coverage combine`` runs in, i.e. invisible to the gate. The stray
directory also landed in the cwd of the real-CLI subprocess in
tests/unit/test_export_cli_args.py::test_real_cli_help_is_instant_and_clean and
turned that test red in CI.

The shim below is a test double for ``$PYTHON`` (cov.sh's documented seam: it
echoes the invocation and honors ``PYTHON=...``). It records the argv and the
coverage-relevant environment cov.sh hands to each child, and reports a scratch
``site.getsitepackages()`` directory, so pinning cov.sh's own contract needs no
nested pytest+coverage run, touches no real site-packages, and stays
deterministic. It also models coverage's resolution rule twice per process —
from this process's own cwd, and from ``SHIM_CHILD_CWD`` standing in for a child
this process spawns — which is what makes "the write target no longer depends on
where a child runs" directly assertable. The end-to-end symptom stays covered by
the real-interpreter test named above: it fails whenever this contract regresses
(verified against the pre-fix script, which leaves the stray directory behind).

The module also pins cov.sh's EXIT contract (same wrapper, next measured
incident): every step's rc is captured and echoed in the closing line, any
failed step reds the run, and pytest's rc wins when several steps failed.
Before that accounting, ``set -e`` killed the script at the failing step:
lint run 34689020064 exited 1 from ``coverage report`` while the suite was
16425 passed / 0 failed, and no line named the step — so a red log could not
be triaged from its own output. ``SHIM_FAIL_ON`` / ``SHIM_FAIL_RC`` inject a
failure into one step, which makes all three exit branches deterministic.

``SHIM_*`` is an env channel rather than argv because cov.sh owns argv: it runs
``$PYTHON -c ...``, then ``$PYTHON -m pytest <user args>``, then ``-m coverage
combine``/``report``, with argument shapes the shim must not disturb.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COV_SH = REPO / "scripts" / "cov.sh"

# Test double for $PYTHON. See the module docstring for what it does and why.
_SHIM = '''\
#!/usr/bin/env python3
"""Record what cov.sh hands a coverage process; write nothing but the log."""
import json
import os
import sys
from pathlib import Path


def _target(data_file, cwd):
    """Where coverage would write *data_file* for a process running in *cwd*."""
    path = Path(data_file)
    return path if path.is_absolute() else Path(cwd) / path


data_file = os.environ.get("COVERAGE_FILE")
with open(os.environ["SHIM_LOG"], "a", encoding="utf-8") as log:
    print(json.dumps({
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "coverage_file": data_file,
        "coverage_process_start": os.environ.get("COVERAGE_PROCESS_START"),
        "child_target": None if data_file is None else str(_target(data_file, os.getcwd())),
        "spawned_target": None if data_file is None else str(_target(data_file, os.environ["SHIM_CHILD_CWD"])),
    }), file=log)

if len(sys.argv) > 2 and sys.argv[1] == "-c" and "site" in sys.argv[2]:
    print(os.environ["SHIM_SITE"])
# Failure injection for the exit-contract tests; inert when unset.
fail_on = os.environ.get("SHIM_FAIL_ON")
if fail_on and any(fail_on in arg for arg in sys.argv[1:]):
    sys.exit(int(os.environ.get("SHIM_FAIL_RC", "3")))
sys.exit(0)
'''


def _run_cov_sh(
    tmp_path: Path,
    *,
    cov_dir: str | None,
    cwd: Path,
    child_cwd: Path,
    tag: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[list[dict], subprocess.CompletedProcess]:
    """Run cov.sh with the shim as $PYTHON; return (shim records, completed run).

    ``child_cwd`` is not cov.sh's cwd — it stands in for the cwd of a process
    coverage instruments (xdist worker / test-spawned subprocess), which is what
    a relative COVERAGE_FILE would be resolved against.
    """
    case = tmp_path / tag
    site = case / "site-packages"
    site.mkdir(parents=True)
    # Pre-created so cov.sh keeps its hands off any real site-packages: the
    # install branch is for fresh environments, not for this contract test.
    (site / "a1_coverage.pth").write_text("# shim\n", encoding="utf-8")
    shim = case / "shim_python"
    shim.write_text(_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    log = case / "invocations.jsonl"

    env = {
        **os.environ,
        "PYTHON": str(shim),
        "SHIM_LOG": str(log),
        "SHIM_SITE": str(site),
        "SHIM_CHILD_CWD": str(child_cwd),
        "PWD": str(cwd),  # a shell that runs cov.sh has a truthful PWD
    }
    # Drop the OUTER run's coverage steering: this test itself runs under cov.sh
    # in CI, and an inherited relative COV_DIR/COVERAGE_FILE would otherwise make
    # the assertions depend on how the suite was launched. COVERAGE_RCFILE is the
    # same family: coverage's process_startup exports it (the repo pyproject)
    # into every instrumented process, and the nested `-m coverage` invocations
    # below must see this test's environment, not the outer run's config pointer.
    for key in ("COV_DIR", "COVERAGE_FILE", "COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG", "COVERAGE_RCFILE"):
        env.pop(key, None)
    if cov_dir is not None:
        env["COV_DIR"] = cov_dir
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(COV_SH), "tests/unit", "-q"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    assert records, f"cov.sh never invoked $PYTHON (stdout={proc.stdout!r} stderr={proc.stderr!r})"
    return records, proc


def _coverage_file(records: list[dict], module: str) -> str:
    """The COVERAGE_FILE handed to the ``-m <module>`` children (pytest/coverage).

    ``-m coverage`` runs twice (combine, then report); every child of the same
    module must have been given the same file, or the gate reads what the run
    did not write.
    """
    hits = [r for r in records if r["argv"][:2] == ["-m", module]]
    assert hits, f"no `-m {module}` child in {records}"
    values = {r["coverage_file"] for r in hits}
    assert len(values) == 1 and None not in values, f"`-m {module}` children disagree: {values}"
    return values.pop()


def _same(path, other) -> bool:
    """Path equality after symlink normalization (macOS /tmp -> /private/tmp)."""
    return Path(path).resolve() == Path(other).resolve()


def _coverage_children(records: list[dict]) -> list[dict]:
    """Records of the children that must all be pointed at the same data file.

    The ``-c 'import coverage'`` / ``-c 'import site'`` sanity probes run BEFORE
    cov.sh configures the data dir (its own order), so they carry no
    COVERAGE_FILE; ``-m pytest`` and the ``-m coverage combine``/``report``
    children are the ones the gate merges.
    """
    out = [r for r in records if r["coverage_file"]]
    assert len(out) >= 3, f"expected pytest + combine + report children: {records}"
    return out


def test_relative_cov_dir_reaches_children_absolute_and_repo_rooted(tmp_path):
    """lint.yml's ``COV_DIR=coverage-data`` must not reach children relative.

    A relative COV_DIR stays relative to the repo root (cov.sh runs from there
    after its own ``cd``), so the probe directory is created and removed here.
    """
    probe = "coverage-data-contract-probe"
    probe_dir = REPO / probe
    try:
        records, proc = _run_cov_sh(
            tmp_path,
            cov_dir=probe,
            cwd=tmp_path,  # invoked from elsewhere: resolution must not use this cwd
            child_cwd=tmp_path / "child",
            tag="ci-spelling",
        )
        assert proc.returncode == 0, proc.stderr
        assert probe_dir.is_dir(), "cov.sh must create a fresh COV_DIR itself (set -e safe)"
        expected = probe_dir / "coverage"
        for module in ("pytest", "coverage"):
            value = _coverage_file(records, module)
            assert Path(value).is_absolute(), f"`-m {module}` child got a relative COVERAGE_FILE: {value}"
            assert _same(value, expected), f"`-m {module}` child writes {value}, expected {expected}"
        for record in _coverage_children(records):
            # One data file for the whole run, whatever a child's cwd is.
            assert _same(record["child_target"], expected), record
            assert _same(record["spawned_target"], expected), record
        # The reported data dir is the one actually used (triage lines agree).
        assert str(Path(_coverage_file(records, "pytest")).parent) in proc.stdout, proc.stdout
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def test_write_target_does_not_depend_on_child_cwd(tmp_path):
    """The bug's signature: one COV_DIR, two child cwds, two different files.

    The probe lives outside the repo (``os.path.relpath``), so this case leaves
    no directory behind; the literal CI spelling is covered above.
    """
    probe = os.path.relpath(tmp_path / "cov", REPO)
    expected = tmp_path / "cov" / "coverage"
    records_a, proc_a = _run_cov_sh(tmp_path, cov_dir=probe, cwd=tmp_path, child_cwd=tmp_path / "child-a", tag="a")
    records_b, proc_b = _run_cov_sh(tmp_path, cov_dir=probe, cwd=REPO, child_cwd=tmp_path / "child-b", tag="b")
    assert proc_a.returncode == 0 and proc_b.returncode == 0, (proc_a.stderr, proc_b.stderr)
    target_a = _coverage_file(records_a, "pytest")
    target_b = _coverage_file(records_b, "pytest")
    assert Path(target_a).is_absolute() and Path(target_b).is_absolute()
    assert _same(target_a, expected), f"resolved against the wrong base: {target_a}"
    assert _same(target_a, target_b), f"the caller's cwd changed the write target: {target_a} vs {target_b}"
    # Every instrumented process — including a child spawned from a foreign cwd —
    # is told the same file, so `coverage combine` sees all of it.
    for records in (records_a, records_b):
        for record in _coverage_children(records):
            assert _same(record["spawned_target"], expected), record


def test_absolute_cov_dir_is_used_verbatim(tmp_path):
    """An absolute COV_DIR is passed through, not re-rooted anywhere."""
    probe = tmp_path / "cov-abs"
    records, proc = _run_cov_sh(tmp_path, cov_dir=str(probe), cwd=REPO, child_cwd=tmp_path / "child", tag="abs")
    assert proc.returncode == 0, proc.stderr
    value = _coverage_file(records, "pytest")
    assert Path(value).is_absolute()
    assert _same(value, probe / "coverage"), value


def test_default_cov_dir_is_a_private_absolute_dir(tmp_path):
    """No COV_DIR: a fresh private dir outside the repo, still absolute."""
    records, proc = _run_cov_sh(tmp_path, cov_dir=None, cwd=tmp_path, child_cwd=tmp_path / "child", tag="default")
    assert proc.returncode == 0, proc.stderr
    value = Path(_coverage_file(records, "pytest"))
    try:
        assert value.is_absolute()
        assert value.name == "coverage", value
        assert value.parent.is_dir(), value.parent
        assert value.parent.name.startswith("asicode-cov."), value.parent
        assert not _same(value.parent, REPO) and REPO.resolve() not in value.parent.resolve().parents, value.parent
        for record in _coverage_children(records):
            assert _same(record["spawned_target"], value), record
    finally:
        # Cleans up even when an assertion above fails; the prefix guard keeps
        # this from ever removing anything cov.sh did not mint for this run.
        if value.parent.name.startswith("asicode-cov."):
            shutil.rmtree(value.parent, ignore_errors=True)


@pytest.mark.parametrize(
    ("step", "fail_rc", "expected_rc"),
    [("pytest", "5", 5), ("combine", "1", 1), ("report", "2", 2)],
)
def test_exit_contract_names_the_failing_step(tmp_path, step, fail_rc, expected_rc):
    """Any failed step reds the run, and the closing line names it with its rc.

    The report/combine branches are the ones `set -e` used to swallow: the
    process died at the failing command, so the pytest rc and the footer were
    both lost (lint run 34689020064: 16425 passed / 0 failed, then rc=1 from
    report with nothing in the log saying so).
    """
    _, proc = _run_cov_sh(
        tmp_path,
        cov_dir=None,
        cwd=tmp_path,
        child_cwd=tmp_path / "child",
        tag=f"exit-{step}",
        extra_env={"SHIM_FAIL_ON": step, "SHIM_FAIL_RC": fail_rc},
    )
    assert proc.returncode == expected_rc, (proc.stdout, proc.stderr)
    assert f"{step} rc={fail_rc}" in proc.stdout, proc.stdout


def test_report_still_runs_after_a_failing_suite(tmp_path):
    """A red suite must still produce a report — that picture is the point."""
    _, proc = _run_cov_sh(
        tmp_path,
        cov_dir=None,
        cwd=tmp_path,
        child_cwd=tmp_path / "child",
        tag="report-after-red-suite",
        extra_env={"SHIM_FAIL_ON": "pytest", "SHIM_FAIL_RC": "1"},
    )
    assert proc.returncode == 1, proc.stderr
    assert "combine rc=0" in proc.stdout and "report rc=0" in proc.stdout, proc.stdout
