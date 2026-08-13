"""P24-D: every SequenceMatcher constructor call must pass autojunk=False.

autojunk (default True) treats any element appearing in >1% of a >=200-char
sequence as "junk" and removes it from matching. For character-level text
comparisons that means '\\n', space, 'e', '(', ... — ratio() collapses toward
0 for blocks that are 99.9% identical (P22-1, P23-2, P24-1/2). For line-level
comparisons, repeated lines (blank lines, common tokens) get purged, skewing
thresholds.

This gate scans every production .py file and fails any SequenceMatcher
constructor call whose keywords lack autojunk=False, so the pattern cannot
regress in new code. Mirrors the AST-gate pattern of
test_service_internal_call_contract.py.
"""
from __future__ import annotations

import ast
import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SKIP_DIRS = {"tests", ".venv", ".asicode", "node_modules", "__pycache__", ".git"}


def _scan_src(src: str, name: str) -> list[str]:
    """Return ['name:lineno'] for SequenceMatcher calls lacking autojunk."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_matcher = (
            isinstance(func, ast.Name) and func.id == "SequenceMatcher"
        ) or (
            isinstance(func, ast.Attribute)
            and func.attr == "SequenceMatcher"
            and isinstance(func.value, ast.Name)
            and func.value.id == "difflib"
        )
        if not is_matcher:
            continue
        if any(kw.arg == "autojunk" for kw in node.keywords if kw.arg is not None):
            continue
        out.append(f"{name}:{node.lineno}")
    return out


def _production_py_files() -> list[pathlib.Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        files.extend(
            pathlib.Path(dirpath) / f for f in filenames if f.endswith(".py")
        )
    return files


def test_no_sequence_matcher_without_autojunk():
    offenders: list[str] = []
    for path in _production_py_files():
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}::{loc}"
            for loc in _scan_src(path.read_text(encoding="utf-8"), path.name)
        )
    assert not offenders, (
        "SequenceMatcher calls without autojunk=False: autojunk purges "
        "chars/lines appearing in >1% of a >=200-char sequence, collapsing "
        "ratio() toward 0 for near-identical text. Offenders:\n"
        + "\n".join(offenders)
    )


# --- precision: the scanner must flag/pass the right shapes -----------------
def test_flags_difflib_attribute_call():
    src = "import difflib\nx = difflib.SequenceMatcher(None)\n"
    assert _scan_src(src, "m.py") == ["m.py:2"]


def test_flags_name_import_call():
    src = "from difflib import SequenceMatcher\nx = SequenceMatcher(None, a, b)\n"
    assert _scan_src(src, "m.py") == ["m.py:2"]


def test_passes_autojunk_false():
    src = "import difflib\nx = difflib.SequenceMatcher(a, b, autojunk=False)\n"
    assert _scan_src(src, "m.py") == []


def test_passes_autojunk_keyword_without_positional():
    src = "import difflib\nx = difflib.SequenceMatcher(None, autojunk=False)\n"
    assert _scan_src(src, "m.py") == []


def test_ignores_comments_and_strings():
    src = (
        "# difflib.SequenceMatcher(None) in a comment\n"
        's = "difflib.SequenceMatcher(None)"\n'
    )
    assert _scan_src(src, "m.py") == []


def test_passes_set_seqs_reuse_after_gated_constructor():
    # The P5 optimization reuses one matcher via set_seqs(); the constructor
    # is the single gate point.
    src = (
        "import difflib\n"
        "m = difflib.SequenceMatcher(None, autojunk=False)\n"
        "m.set_seqs(a, b)\n"
    )
    assert _scan_src(src, "m.py") == []
