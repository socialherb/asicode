"""Regression tests for plan_compiler._apply_edit_blocks fuzzy-match context.

P23-2: the error-context SequenceMatcher ran without autojunk=False. For a
30-line block whose lines are short and repetitive, autojunk auto-junked
'\\n', ' ', '=', etc. (each > 1% of the compared window) and collapsed
ratio() to ~0.16 for a block that was 97% identical — the "file context near
best match" hint was silently omitted from the retry error, defeating the
LLM-retry loop for multi-line edit blocks. With autojunk disabled the ratio
is ~0.99.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plan_compiler import PlanCompileError, _apply_edit_blocks, _read_text_if_exists


def test_fuzzy_match_context_ratio_survives_autojunk():
    """A before block differing in one line still yields a file-context hint."""
    lines = [f"l{i} = {i}" for i in range(30)]
    content = "\n".join(lines) + "\n"
    before_lines = list(lines)
    before_lines[5] = "l5 = 99"  # one-line drift -> exact/tolerant/reindent all fail
    before = "\n".join(before_lines)
    after = "\n".join(before_lines) + "\n"
    with pytest.raises(PlanCompileError) as ei:
        _apply_edit_blocks(
            text=content,
            edits=[{"before": before, "after": after}],
            expect_unique=True,
            path="m.py",
        )
    details = ei.value.details
    assert details.get("best_match_ratio", 0) > 0.35, f"ratio collapsed (autojunk): {details.get('best_match_ratio')}"
    assert details.get("file_context_near_match") is not None


def test_read_text_if_exists_gates_oversized_files():
    """P24-2: plan compile reads target files in full for diff/context.

    A plan op touching a huge file used to read it entirely into memory and
    push an unbounded unified diff toward the LLM prompt. The gate mirrors
    patch_engine._MAX_FILE_CHARS (250_000) and fails loudly with
    PlanCompileError so the model can switch to a targeted tool instead of
    silently prompting with megabytes of file content.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        big = root / "big.py"
        big.write_text("x" * 250_001, encoding="utf-8")
        with pytest.raises(PlanCompileError) as ei:
            _read_text_if_exists(big)
        assert "file_too_large" in str(ei.value)
        assert ei.value.details["max_bytes"] == 250_000

        small = root / "small.py"
        small.write_text("ok", encoding="utf-8")
        assert _read_text_if_exists(small) == "ok"

        assert _read_text_if_exists(root / "missing.py") is None
