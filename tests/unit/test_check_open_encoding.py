"""Tests for the no-locale-open gate (check_open_encoding.py).

The live invariant (no new unencodinged open/read_text/write_text calls) is
baseline-diff based; the precision tests guard detection capability and the
scan-pruning test seals the os.walk optimization (F-4).
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_open_encoding.py"
_spec = importlib.util.spec_from_file_location("check_open_encoding", _SCRIPT)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]


# ── precision: unencodinged open() IS flagged ──────────────────────────────
def test_open_without_encoding_flagged():
    src = "def f():\n    fh = open('a.txt')\n    return fh\n"
    assert g._violations_in(src) == [2]


def test_open_with_encoding_passes():
    src = "def f():\n    fh = open('a.txt', encoding='utf-8')\n    return fh\n"
    assert g._violations_in(src) == []


def test_open_binary_mode_passes():
    src = "def f():\n    fh = open('a.bin', 'rb')\n    return fh\n"
    assert g._violations_in(src) == []


def test_open_encoding_kwarg_passes():
    src = "def f():\n    fh = open('a.txt', encoding='utf-8')\n    return fh\n"
    assert g._violations_in(src) == []


def test_path_read_text_without_encoding_flagged():
    src = "def f():\n    t = Path('a.txt').read_text()\n    return t\n"
    assert g._violations_in(src) == [2]


def test_path_read_text_with_encoding_passes():
    src = "def f():\n    t = Path('a.txt').read_text(encoding='utf-8')\n    return t\n"
    assert g._violations_in(src) == []


def test_path_write_text_without_encoding_flagged():
    src = "def f():\n    Path('a.txt').write_text('x')\n"
    assert g._violations_in(src) == [2]


def test_path_write_text_with_encoding_passes():
    src = "def f():\n    Path('a.txt').write_text('x', encoding='utf-8')\n"
    assert g._violations_in(src) == []


def test_path_read_text_positional_encoding_passes():
    src = "def f():\n    t = Path('a.txt').read_text('utf-8')\n    return t\n"
    assert g._violations_in(src) == []


def test_path_write_text_second_positional_encoding_passes():
    src = "def f():\n    Path('a.txt').write_text('x', 'utf-8')\n"
    assert g._violations_in(src) == []


# ── scan pruning: os.walk must not descend into skipped dirs (F-4) ──────────
def test_iter_py_files_prunes_skipped_dirs_entirely(tmp_path, monkeypatch):
    """Full scan must skip SKIP_DIRS subtrees before traversal (rglob era
    walked everything and discarded afterwards; os.walk prunes in place).

    A planted skip dir with *.py files must never surface in the scan.
    """
    monkeypatch.setattr(g, "REPO", tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    for skipped in (".venv", "__pycache__", "build", "node_modules", "dist", ".git"):
        d = tmp_path / skipped
        d.mkdir(parents=True, exist_ok=True)
        (d / "mod.py").write_text("x = 1\n", encoding="utf-8")
    scanned = list(g._iter_py_files())
    assert scanned == [(Path("real.py"), tmp_path / "real.py")], scanned
    # explicit-path scan must still respect the skip (parity with rglob era)
    assert list(g._iter_py_files([str(tmp_path / ".venv" / "mod.py")])) == []
