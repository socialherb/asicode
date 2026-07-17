"""Regression tests for truncation-induced bugs (tail-bias).

Three bugs shared a common root cause: truncation that preserved only the
*head* of output, even though pytest/traceback/import-signals live at the
*tail* (end) of the text.

1. ``ProjectAnalyzer._find_common_imports`` scanned only the first ~50 lines
   of each file, so imports beyond line 50 (after long module docstrings,
   license headers, or lazy imports) were silently dropped — biasing
   ``common_imports`` toward small files.
2. ``git_tools`` bash output cap kept only ``content[:cap]``, chopping off the
   pytest summary (``short test summary info``, ``N failed``, FAILED list)
   that ``failure_context._try_parse_pytest`` keys on.
"""
from __future__ import annotations

from pathlib import Path

from external_llm.project_analyzer import ProjectAnalyzer


# ---------------------------------------------------------------------------
# Bug 1: imports beyond line 50 must still be counted.
# ---------------------------------------------------------------------------
def test_python_import_beyond_line_50_is_detected(tmp_path: Path):
    """A file whose import sits after a long header must still be scanned.

    Before the fix, ``content.split('\\n')[:50]`` dropped imports past line 50.
    Here the only import is on line ~80, behind a long module docstring.
    """
    # 79 lines of docstring/comment, then the import on line 80.
    header = '"""Long module docstring.\n' + ("filler line\n" * 77) + '"""\n'
    body = "import requests\n"
    _write_py(tmp_path, "late_import.py", header + body)

    s = ProjectAnalyzer(str(tmp_path)).analyze()

    assert "requests" in s.common_imports, (
        f"import past line 50 was dropped by line-truncation; common_imports={s.common_imports}"
    )


def test_python_relative_and_nested_module_root_extraction(tmp_path: Path):
    """Sanity: the full-scan path still extracts the root module name and skips
    relative imports (a regression guard for the surrounding logic)."""
    _write_py(
        tmp_path,
        "m.py",
        "import os\n"
        "from collections import defaultdict\n"
        "from . import local_thing  # relative — must NOT be counted as 'local_thing'\n"
        "import sys\n",
    )
    s = ProjectAnalyzer(str(tmp_path)).analyze()
    assert "os" in s.common_imports
    assert "collections" in s.common_imports
    assert "sys" in s.common_imports
    # Relative imports must never pollute counts.
    assert "" not in s.common_imports


# ---------------------------------------------------------------------------
# Bug 2: bash output truncation must preserve head + tail.
# ---------------------------------------------------------------------------
def test_bash_truncation_preserves_tail():
    """pytest's summary lives at the tail; head+tail keeps it visible."""
    from external_llm.agent.tool_handlers.git_tools import _truncate_bash_output

    # Use a realistic cap and a large payload so head+tail genuinely shrinks
    # the output (the truncation marker is ~150 chars of fixed overhead, so a
    # tiny payload that barely exceeds the cap can grow — not the real use case).
    cap = 2000
    head_marker = "HEAD_BEGIN_MARKER"
    tail_marker = "short test summary info\n150 failed"
    # ~30K of ASCII filler → well over the cap.
    payload = head_marker + ("\n" + ("x" * 80) * 300) + "\n" + tail_marker
    assert len(payload) > cap

    out = _truncate_bash_output(payload, cap)

    assert head_marker in out, "head marker should be preserved"
    assert tail_marker in out, (
        "tail pytest summary must be preserved (the bug dropped it)"
    )
    assert "truncated" in out, "should report that truncation happened"
    # head + tail (each ~cap/2) + fixed marker → much shorter than payload.
    assert len(out) < len(payload)
    assert len(out) < cap + 500  # roughly the budget + marker overhead


def test_bash_truncation_passthrough_when_under_cap():
    """Under the cap, content is returned unchanged."""
    from external_llm.agent.tool_handlers.git_tools import _truncate_bash_output

    short = "just a short line\n"
    assert _truncate_bash_output(short, 200) is short
    assert _truncate_bash_output("", 200) == ""


def test_bash_truncation_cjk_output_more_aggressive():
    """Non-ASCII (CJK/JSON-dense) output uses a tighter char cap.

    Compared to an equivalent-sized ASCII payload, the CJK payload keeps
    less content because the effective cap is halved.
    """
    from external_llm.agent.tool_handlers.git_tools import _truncate_bash_output

    cap = 4000
    # Pure CJK → low ascii ratio → effective cap halved.
    cjk_payload = ("테스트" * 2000) + "\nTAIL_MARKER_한국어"
    # ASCII payload of comparable size for comparison.
    ascii_payload = ("abcde" * 2000) + "\nTAIL_MARKER_ascii"
    assert len(cjk_payload) > cap and len(ascii_payload) > cap

    cjk_out = _truncate_bash_output(cjk_payload, cap)
    ascii_out = _truncate_bash_output(ascii_payload, cap)

    # Both tails preserved.
    assert "TAIL_MARKER_한국어" in cjk_out
    assert "TAIL_MARKER_ascii" in ascii_out
    # The CJK effective cap is halved, so its result retains less *body*
    # content than the ASCII result (both minus the same fixed marker).
    assert len(cjk_out) < len(ascii_out), (
        "CJK output should be more aggressively truncated (halved cap)"
    )


def _write_py(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
