"""Tests for the `ruff format` zero-tolerance gate in check_lint_full.py.

The live invariant ("no file may drift from ruff format") is vacuously
passable if the detector is weakened, so these precision tests pin the
detection capability directly: a deliberately unformatted file MUST be
reported, a formatted file MUST NOT.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_lint_full.py"
_spec = importlib.util.spec_from_file_location("check_lint_full", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


def _need_ruff() -> bool:
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


@pytest.mark.skipif(not _need_ruff(), reason="ruff not installed")
def test_unformatted_file_is_reported(tmp_path):
    """A file ruff would reformat MUST be listed by _get_format_errors."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f(x):\n    return x\n\nx=1\n")
    errors = g._get_format_errors([str(bad)])
    assert errors, "unformatted file was not reported"
    assert any("bad.py" in e for e in errors), errors


@pytest.mark.skipif(not _need_ruff(), reason="ruff not installed")
def test_formatted_file_not_reported(tmp_path):
    """A ruff-formatted file MUST NOT be reported (no false positive)."""
    good = tmp_path / "good.py"
    good.write_text("def f(x):\n    return x\n\n\nx = 1\n")
    errors = g._get_format_errors([str(good)])
    assert errors == [], errors


@pytest.mark.skipif(not _need_ruff(), reason="ruff not installed")
def test_clean_repo_not_reported():
    """The repo itself must currently be format-clean (gate self-check)."""
    errors = g._get_format_errors(None)
    assert errors == [], f"repo has format drift: {errors[:5]}"


@pytest.mark.skipif(not _need_ruff(), reason="ruff not installed")
def test_non_py_scan_args_filtered(tmp_path):
    """Non-.py args must be filtered out (yaml/json config files).

    ruff format handles them anyway, but _resolve_scan_paths is .py-only by
    contract (mirrors the check hooks); the format gate must never crash on
    config files passed by pre-commit.
    """
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text("key: value\n")
    paths = g._resolve_scan_paths([str(yaml)])
    assert paths is None or all(p.endswith(".py") for p in paths)
