"""Regression tests for PatchSynthesizer.

Locks in the Fallback 1 silent-no-op fix: when a file line carries trailing
whitespace that the BEFORE block lacks, the ASICODE_BLOCK synthesis must
still emit a real diff (previously it produced an empty diff, silently
losing the change).
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from external_llm.hybrid_parser import ParseResult
from external_llm.output_modes import OutputMode
from external_llm.patch_synthesizer import PatchSynthesizer


def _synth(tmpdir, content, before, after, target="m.py"):
    path = os.path.join(tmpdir, target)
    if content is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    pr = ParseResult(
        success=True,
        mode=OutputMode.ASICODE_BLOCK,
        blocks=[{"before": before, "after": after}],
    )
    return PatchSynthesizer(tmpdir).synthesize(pr, target)


def test_exact_match_uses_direct_replace():
    with tempfile.TemporaryDirectory() as td:
        diff = _synth(td, "def foo():\n    return 1\n", "def foo():\n    return 1\n", "def foo():\n    return 2\n")
        assert "return 2" in diff
        assert "-    return 1" in diff


def test_fallback1_trailing_whitespace_applies_change():
    """Trailing whitespace in file line (absent in BEFORE) must still apply."""
    with tempfile.TemporaryDirectory() as td:
        content = "def foo():\n    return 1   \n"  # trailing ws
        before = "def foo():\n    return 1\n"   # no trailing ws
        after = "def foo():\n    return 2\n"
        diff = _synth(td, content, before, after)
        # Regression: previously this returned "" (silent no-op).
        assert diff != "", "change was silently lost (empty diff)"
        assert "return 2" in diff
        assert "-    return 1" in diff


def test_fallback2_indent_normalization_applies_change():
    """Uniformly-shifted BEFORE (min indent > 0) matches via dedent fallback."""
    with tempfile.TemporaryDirectory() as td:
        # file at 8 cols; BEFORE written at 4 cols (min indent 4 -> dedent works)
        content = "        x = 1\n        y = 2\n"
        before = "    x = 1\n    y = 2\n"
        after = "    x = 9\n    y = 2\n"
        diff = _synth(td, content, before, after)
        assert "x = 9" in diff


def test_fallback2_reindents_after_to_window_base():
    """Fallback 2 must shift AFTER to the matched window's base indent (8 cols)."""
    with tempfile.TemporaryDirectory() as td:
        content = "        x = 1\n        y = 2\n"
        before = "    x = 1\n    y = 2\n"      # before base = 4
        after = "    x = 9\n    y = 2\n"       # after at 4 -> must shift to 8
        diff = _synth(td, content, before, after)
        # applied AFTER line should be at 8-col indent, not 4
        assert "+        x = 9" in diff


def test_no_match_emits_empty_diff_and_warns(caplog):
    with tempfile.TemporaryDirectory() as td:
        diff = _synth(td, "def foo():\n    return 1\n", "nope\n    nothing\n", "x\ny\n")
        assert diff == ""


def test_mode_none_raises():
    with tempfile.TemporaryDirectory() as td:
        pr = ParseResult(success=True, mode=None)
        with pytest.raises(ValueError):
            PatchSynthesizer(td).synthesize(pr, "m.py")


def test_full_file_mode_creates_diff():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("a = 1\n")
        pr = ParseResult(success=True, mode=OutputMode.FULL_FILE, content="a = 2\n")
        diff = PatchSynthesizer(td).synthesize(pr, "m.py")
        assert "-a = 1" in diff
        assert "+a = 2" in diff
