"""Contract: a pty child driver that starts its OWN coverage instance must write a
SUFFIXED data file — never the base name of ``$COVERAGE_FILE`` — and must take its
measurement surface from the repo config, not from the cwd it inherits.

Why this is a contract, not a style preference: ``coverage combine`` (run by
``scripts/cov.sh`` after every measured suite) merges parallel data files into the
base name and REPLACES an existing base file with its own merge output. Measured
2026-09-12 with coverage 7.14.3: one base-name write beside one parallel file =>
combine printed ``Combined 1 file`` and the base writer's lines were gone from the
report — silently, since every step still exited 0. The two child drivers are the
only place in the tree that constructs a Coverage instance (pty_driver strips
COVERAGE_PROCESS_START/CONFIG, so the a1_coverage.pth hook cannot do it for them),
which makes their constructor arguments the single point of failure for pty-session
coverage data.

The tests below run the LITERAL ``coverage.Coverage(...)`` call extracted from each
driver with ``ast.get_source_segment``: an edit to the driver is an edit to what
runs here, so no copy of the call site can drift away from the file it guards.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from coverage import CoverageData

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_DRIVERS = (
    Path(__file__).parent / "repl_stage2_child.py",
    Path(__file__).parent / "repl_stage3_child.py",
)

# The probe runs the extracted call verbatim. It defines everything the call site
# may reference (`data_file`, `coverage`, `pathlib`, `Path`, `__file__`) and
# re-points `__file__` at the REAL driver, so a `parents[N]` config lookup still
# resolves against the repo the driver lives in.
_PROBE = """\
import json, os, pathlib, sys
from pathlib import Path

import coverage

data_file = sys.argv[1]
__file__ = sys.argv[2]
report_path = sys.argv[3]

cov = {ctor}

cov.start()
{body}
cov.stop(); cov.save()
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(
        {{
            "data_file": cov._data.data_filename(),
            "source": list(cov.config.source or []),
            "parallel": bool(cov.config.parallel),
        }},
        handle,
    )
"""


def _coverage_ctor_source(child: Path) -> str:
    """The literal text of the ``coverage.Coverage(...)`` call in *child*."""
    src = child.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src, filename=str(child))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Coverage"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "coverage"
        ):
            segment = ast.get_source_segment(src, node)
            assert segment, f"no source segment for the Coverage(...) call in {child.name}"
            return segment
    raise AssertionError(f"{child.name} never calls coverage.Coverage(...)")


def _repo_source_declaration() -> list[str]:
    """``[tool.coverage.run] source`` as the repo declares it (derived, not copied)."""
    config = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(config["tool"]["coverage"]["run"]["source"])


def _probe_env(base: Path) -> dict[str, str]:
    """Environment for a probe process.

    This suite itself runs under ``scripts/cov.sh`` in CI, and coverage's
    ``process_startup()`` (the a1_coverage.pth hook) EXPORTS ``COVERAGE_RCFILE``
    into every process it instruments. An inherited pointer would hand the probe
    the OUTER run's data file and config, making the assertions depend on how the
    suite was launched — the whole COVERAGE_* family is cleared, then
    ``COVERAGE_FILE`` is set the way cov.sh sets it for its children.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("COVERAGE_") and k != "COV_DIR"}
    env["COVERAGE_FILE"] = str(base)
    return env


def _run_probe(tmp_path: Path, name: str, *, ctor: str, body: str, file_ref: Path, base: Path, cwd: Path) -> dict:
    probe = tmp_path / f"probe_{name}.py"
    report = tmp_path / f"probe_{name}.json"
    probe.write_text(_PROBE.format(ctor=ctor, body=body), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(probe), str(base), str(file_ref), str(report)],
        cwd=str(cwd),
        env=_probe_env(base),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"probe {name} failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.fixture()
def cov_case(tmp_path: Path) -> tuple[Path, Path]:
    """``(COV_DIR stand-in, probe cwd)``.

    The probe cwd is deliberately NOT the repo root: it carries no coverage config,
    so anything the child's instance gets must come from its own arguments — that
    is exactly the situation cwd discovery used to collapse to parallel=False.
    """
    data_dir = tmp_path / "cov-data"
    data_dir.mkdir()
    plain_cwd = tmp_path / "no-config-cwd"
    plain_cwd.mkdir()
    return data_dir, plain_cwd


@pytest.mark.parametrize("child", _CHILD_DRIVERS, ids=lambda p: p.name)
def test_child_coverage_ctor_writes_a_suffixed_file(child: Path, tmp_path: Path, cov_case: tuple[Path, Path]) -> None:
    """The child's own instance must not write the base name that combine replaces."""
    data_dir, plain_cwd = cov_case
    base = data_dir / "coverage"

    info = _run_probe(
        tmp_path, child.stem, ctor=_coverage_ctor_source(child), body="pass", file_ref=child, base=base, cwd=plain_cwd
    )

    written = Path(info["data_file"])
    assert written != base, (
        "the child wrote the BASE data file — `coverage combine` replaces that file with "
        "its own merge output, so these lines would vanish from the report"
    )
    assert written.parent == base.parent, f"child data must land beside COVERAGE_FILE, got {written}"
    assert written.name.startswith(base.name + "."), f"expected a parallel suffix, got {written.name}"
    assert written.exists() and written.stat().st_size > 0
    # Nothing under the base name: exactly the child's own file exists.
    assert [p.name for p in sorted(data_dir.iterdir())] == [written.name]

    # The surface comes from the pinned repo config even though the probe's cwd has
    # none: cwd discovery would leave `parallel` off and `source` unset.
    assert info["parallel"] is True
    assert info["source"] == _repo_source_declaration()


def test_child_coverage_data_survives_combine(tmp_path: Path, cov_case: tuple[Path, Path]) -> None:
    """End-to-end: cov.sh's combine must keep the child's lines.

    Three writers, exactly like a measured run: cov.sh's own process (a plain
    parallel instance whose config comes from the repo pyproject — the same file
    the .pth hook gets via COVERAGE_PROCESS_START) plus one probe per child driver
    running in a config-less cwd. `import asi` gives the child probes REAL content
    under the pinned `source=` filter, and asi.py is the marker the main writer
    never measures — so its absence after combine is precisely the data loss.
    """
    data_dir, plain_cwd = cov_case
    base = data_dir / "coverage"

    plain_ctor = f"coverage.Coverage(data_file=data_file, config_file={str(_REPO_ROOT / 'pyproject.toml')!r})"
    main = _run_probe(
        tmp_path,
        "main",
        ctor=plain_ctor,
        body="import external_llm.model_catalog",
        file_ref=Path(__file__),  # unused by the plain call: no __file__ lookup in it
        base=base,
        cwd=plain_cwd,
    )
    assert Path(main["data_file"]) != base, "the main writer must also be a parallel instance"

    child_files = []
    for child in _CHILD_DRIVERS:
        info = _run_probe(
            tmp_path,
            child.stem,
            ctor=_coverage_ctor_source(child),
            body="import asi",
            file_ref=child,
            base=base,
            cwd=plain_cwd,
        )
        child_files.append(Path(info["data_file"]))

    # No base file may exist before combine — that is the bug's fingerprint.
    before = sorted(p.name for p in data_dir.iterdir())
    assert base.name not in before, f"a base-name data file was written: {before}"
    assert len(before) == 1 + len(child_files), f"expected one file per writer, got {before}"

    proc = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=str(_REPO_ROOT),
        env=_probe_env(base),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"combine failed: {proc.stdout}\n{proc.stderr}"

    data = CoverageData(basename=str(base))
    data.read()
    measured = {Path(f).name for f in data.measured_files()}
    assert "model_catalog.py" in measured, f"non-vacuity: the main writer's data is missing: {measured}"
    assert "asi.py" in measured, f"the pty child's lines did not survive combine: {measured}"


def test_ci_cov_dir_is_gitignored() -> None:
    """``COV_DIR=coverage-data`` (lint.yml / release.yml) lands at the repo root."""
    # Trailing slash: `/coverage-data/` is a directory-only pattern, so git needs
    # the path to BE a directory spelling to apply it (the dir may not exist yet —
    # a fresh checkout before the first cov.sh run).
    ignored = subprocess.run(["git", "check-ignore", "-q", "coverage-data/"], cwd=str(_REPO_ROOT), check=False)
    assert ignored.returncode == 0, "coverage-data/ is not ignored — a CI/local cov.sh run dirties the tree"
    # Negative control: the command can fail, so the check above is not vacuous.
    not_ignored = subprocess.run(["git", "check-ignore", "-q", "asi.py"], cwd=str(_REPO_ROOT), check=False)
    assert not_ignored.returncode == 1
