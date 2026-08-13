"""P26-3: file text reads must not silently delete invalid UTF-8 bytes.

errors='ignore' on an open()/read_text() call drops invalid bytes, which can
merge adjacent characters into a DIFFERENT string (b'pack\\xffage' reads as
'package') — corrupting LLM prompt context (P23-1: patch_synthesizer) and
phantom-matching guard regexes (P26-1: rewrite safety guard). errors='replace'
keeps the byte visible as U+FFFD instead, so the reader sees the file's true
character stream.

This gate scans every production .py file and fails any open()/read_text()
call with errors='ignore', so the class cannot regress. Byte-decode calls
(.decode(..., errors='ignore')) on subprocess output are deliberately NOT
gated — stderr/stdout rendering for humans is standard practice there
(patch_engine error surfaces), and read_bytes().decode() sites carry their
own documented rationale (patch_synth._read_text_lines).
"""
from __future__ import annotations

import ast
import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SKIP_DIRS = {"tests", ".venv", ".asicode", "node_modules", "__pycache__", ".git"}


def _scan_src(src: str, name: str) -> list[str]:
    """Return ['name:lineno'] for open()/read_text() calls with errors='ignore'."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_file_read = (isinstance(func, ast.Name) and func.id == "open") or (
            isinstance(func, ast.Attribute) and func.attr == "read_text"
        )
        if not is_file_read:
            continue
        for kw in node.keywords:
            if (
                kw.arg == "errors"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "ignore"
            ):
                out.append(f"{name}:{node.lineno}")
                break
    return out


def _production_py_files() -> list[pathlib.Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        files.extend(
            pathlib.Path(dirpath) / f for f in filenames if f.endswith(".py")
        )
    return files


def test_no_file_read_with_errors_ignore():
    offenders: list[str] = []
    for path in _production_py_files():
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}::{loc}"
            for loc in _scan_src(path.read_text(encoding="utf-8"), path.name)
        )
    assert not offenders, (
        "open()/read_text() with errors='ignore' silently deletes invalid "
        "UTF-8 bytes, merging adjacent characters (b'pack\\xffage' → "
        "'package') and corrupting prompt context / guard-regex verdicts. "
        "Use errors='replace' (P23-1/P26-1). Offenders:\n"
        + "\n".join(offenders)
    )


# --- precision: the scanner must flag/pass the right shapes -----------------

def test_flags_open_with_errors_ignore():
    assert _scan_src("x = open(p, errors='ignore', encoding='utf-8')\n", "f.py") == ["f.py:1"]


def test_flags_read_text_with_errors_ignore():
    assert _scan_src("s = p.read_text(encoding='utf-8', errors='ignore')\n", "f.py") == ["f.py:1"]


def test_passes_open_with_errors_replace():
    assert _scan_src("x = open(p, errors='replace', encoding='utf-8')\n", "f.py") == []


def test_passes_binary_open():
    assert _scan_src("with open(p, 'rb') as fh: pass\n", "f.py") == []


def test_passes_decode_errors_ignore():
    # subprocess stderr/stdout decode — deliberate, standard practice.
    assert _scan_src("err = result.stderr.decode('utf-8', errors='ignore')\n", "f.py") == []
