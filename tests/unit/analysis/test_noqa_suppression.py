"""
Unit tests for # noqa suppression in the dead-code scanners
(_has_noqa_comment / _import_block_has_noqa in unused_import_scanner.py,
_source_line_has_noqa in vulture_scanner.py).

Locks the flake8 semantics the P3-1/P3-2 implementation initially got wrong
(caught in verification, 2026-07-16):

  * bare ``# noqa`` suppresses EVERY code — the first implementation returned
    False for it when specific codes were requested;
  * code matching must parse only the comma-separated code list and stop at
    prose — the first implementation regex-scanned the whole comment, so
    "# noqa: E501 …mentions F401…" wrongly suppressed F401.
"""
import textwrap

from external_llm.analysis.unused_import_scanner import (
    _has_noqa_comment,
    _import_block_has_noqa,
    scan_unused_imports,
)

# ─── _has_noqa_comment: basic formats ────────────────────────────────────────

def test_no_comment_at_all():
    assert not _has_noqa_comment("import os")
    assert not _has_noqa_comment("import os", {"F401"})


def test_plain_comment_is_not_noqa():
    assert not _has_noqa_comment("import os  # keep for side effects", {"F401"})


def test_exact_code_match():
    assert _has_noqa_comment("import os  # noqa: F401", {"F401"})
    assert _has_noqa_comment("import os  # NOQA: F401", {"F401"})
    assert _has_noqa_comment("import os  # noqa:F401", {"F401"})


def test_code_list_match():
    assert _has_noqa_comment("x = 1  # noqa: F401, F841", {"F401"})
    assert _has_noqa_comment("x = 1  # noqa: F841, F401", {"F401"})


def test_wrong_code_does_not_match():
    assert not _has_noqa_comment("import os  # noqa: F841", {"F401"})
    assert not _has_noqa_comment("import os  # noqa: E501", {"F401"})


def test_codes_none_matches_any_noqa():
    assert _has_noqa_comment("import os  # noqa")
    assert _has_noqa_comment("import os  # noqa: F841")


# ─── the two verification-caught regressions ─────────────────────────────────

def test_bare_noqa_suppresses_every_code():
    # flake8 semantics: "# noqa" with no code list suppresses everything.
    assert _has_noqa_comment("import os  # noqa", {"F401"})
    assert _has_noqa_comment("import os  # noqa", {"F841"})


def test_code_mentioned_in_prose_does_not_match():
    # The code list ends at the first non-code token; free text after it must
    # never match, even when it happens to contain a code-shaped word.
    assert not _has_noqa_comment(
        "import shutil  # noqa: E501 hint text mentions F401 accidentally",
        {"F401"},
    )


def test_descriptive_suffix_after_matching_code_still_matches():
    assert _has_noqa_comment(
        "from mod import X  # noqa: F401 — barrel re-export", {"F401"})
    assert _has_noqa_comment(
        "from mod import X  # noqa: F401, F841 — barrel re-export", {"F841"})


# ─── _import_block_has_noqa: multi-line imports ──────────────────────────────

def test_block_noqa_on_open_paren_line():
    lines = [
        "from pkg.mod import (  # noqa: F401",
        "    name_a,",
        "    name_b,",
        ")",
    ]
    assert _import_block_has_noqa(lines, 1, 4, {"F401"})


def test_block_without_noqa():
    lines = [
        "from pkg.mod import (",
        "    name_a,",
        ")",
    ]
    assert not _import_block_has_noqa(lines, 1, 3, {"F401"})


# ─── scan_unused_imports end-to-end ──────────────────────────────────────────

def test_scan_respects_noqa_but_flags_genuine_unused(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(textwrap.dedent("""\
        import os
        import sys  # noqa: F401
        import json  # noqa
        from pkg.sub import (  # noqa: F401
            alpha,
            beta,
        )
        import shutil  # noqa: E501 hint text mentions F401 accidentally
    """))
    res = scan_unused_imports(
        repo_root=str(tmp_path), file_paths=[str(f)], max_per_file=100)
    flagged = sorted(c.symbol_name for c in res)
    # Suppressed: sys (exact code), json (bare noqa), alpha/beta (block noqa).
    # Flagged: os (no noqa), shutil (E501 only — prose F401 must not count).
    assert flagged == ["os", "shutil"]
