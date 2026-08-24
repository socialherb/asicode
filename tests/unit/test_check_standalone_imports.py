"""Tests for the standalone-imports baseline-diff gate (check_standalone_imports.py).

The live invariant (no new first-party import fallbacks beyond baseline) is
vacuously-passable if the scanner is weakened; the module-name tests guard
detection capability, and the scan-pruning test seals the os.walk
optimization (F-4).
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_standalone_imports.py"
_spec = importlib.util.spec_from_file_location("check_standalone_imports", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


# ── module-name derivation ──────────────────────────────────────────────────
def test_module_name_plain():
    assert g._module_name("ext/pkg/mod.py") == "ext.pkg.mod"


def test_module_name_init_is_package():
    assert g._module_name("ext/pkg/__init__.py") == "ext.pkg"


def test_module_name_asi_root():
    assert g._module_name("asi.py") == "asi"


def test_module_name_invalid_identifier():
    assert g._module_name("ext/pkg/my-mod.py") is None


# ── scan pruning: os.walk must not descend into skipped dirs (F-4) ──────────
def test_iter_modules_under_prunes_skipped_dirs_entirely(tmp_path, monkeypatch):
    """Full scan must skip SKIP_DIRS subtrees before traversal (rglob era
    walked everything and discarded afterwards; os.walk prunes in place).

    A planted skip dir with *.py files must never surface in the scan.
    """
    monkeypatch.setattr(g, "REPO", tmp_path)
    monkeypatch.setattr(g, "_SCAN_ROOTS", ("ext",))
    prod = tmp_path / "ext"
    prod.mkdir()
    (prod / "real.py").write_text("x = 1\n", encoding="utf-8")
    for skipped in (".venv", "__pycache__", "build", "node_modules", ".git"):
        d = prod / skipped
        d.mkdir(parents=True, exist_ok=True)
        (d / "mod.py").write_text("x = 1\n", encoding="utf-8")
    mods = g._iter_modules_under(tmp_path)
    assert mods == ["ext.real"], mods
