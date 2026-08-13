"""Per-file mode of the ruff-based gate scripts (F821 as the representative).

The three baseline-diff ruff hooks (F821/F401/F811) plus zero-tolerance F823
share one structure: `_get_current_errors(paths)` runs ruff on the given
files (whole repo when None).  pre-commit passes absolute paths; lint.yml
passes none.  The per-file key format must stay repo-relative so baseline keys
match the full scan exactly.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_f821_no_new.py"
_spec = importlib.util.spec_from_file_location("check_f821_no_new", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


def test_resolve_scan_paths_normalizes_abs_and_relative():
    rel = "scripts/check_f821_no_new.py"
    assert g._resolve_scan_paths([str(_SCRIPT)]) == [rel]
    assert g._resolve_scan_paths([rel]) == [rel]
    assert g._resolve_scan_paths([]) is None  # no args → full-repo scan
    assert g._resolve_scan_paths(["--write-baseline"]) is None  # flags filtered by main


def test_resolve_scan_paths_rejects_out_of_repo_and_non_py():
    assert g._resolve_scan_paths(["/etc/passwd"]) is None  # '..' → rejected
    assert g._resolve_scan_paths(["README.md"]) is None     # not .py → full scan


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not on PATH")
def test_per_file_keys_are_repo_relative_and_subset_of_full():
    per = g._get_current_errors(["scripts/check_f821_no_new.py"])
    assert all(k.startswith("scripts/check_f821_no_new.py::") for k in per), per
    full = g._get_current_errors()
    assert per <= full  # per-file must never produce keys the full scan misses
