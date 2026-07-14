"""Tests for validate_fix_spec_claims narrative-claim handling.

Locks in three fixes to the validator's Step 2 (narrative/primary_issue scan):

1. ``narrative_warnings`` was a declared-but-never-populated field. Claims that
   matched a remove/orphan pattern but could not be validated (no line range,
   or no resolvable file path) were silently dropped. They are now recorded.

2. The previously dead bare expression ``primary_issue + "\n" + narrative``
   (computed then discarded) is gone.

3. Symbol-based file resolution: when a claim mentions a known target's symbol,
   the matched target's file is used. The old code extracted the symbol via
   ``_extract_symbol_name`` but discarded the result, blindly taking
   ``targets[0].file`` — wrong file for multi-target FixSpecs whose claim
   references a later target's symbol.
"""
from __future__ import annotations

import textwrap

from external_llm.agent.fix_spec_claim_validator import validate_fix_spec_claims


def _write(repo, rel, src):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src), encoding="utf-8")


def test_narrative_claim_without_line_range_is_warned(tmp_path):
    """A remove/orphan claim with no line range is recorded, not silently dropped."""
    summary = validate_fix_spec_claims(
        {
            "targets": [],
            "primary_issue": "",
            "analysis_narrative": "remove the orphaned block",
        },
        repo_root=str(tmp_path),
    )
    assert len(summary.narrative_warnings) == 1
    assert "line range" in summary.narrative_warnings[0]
    assert summary.has_hallucination is False


def test_narrative_claim_without_resolvable_file_is_warned(tmp_path):
    """A claim with a line range but no targets/files is recorded as unverifiable."""
    summary = validate_fix_spec_claims(
        {
            "targets": [],
            "primary_issue": "delete the orphaned method at lines 10-20",
            "analysis_narrative": "",
        },
        repo_root=str(tmp_path),
    )
    assert len(summary.narrative_warnings) == 1
    assert "file path" in summary.narrative_warnings[0]
    assert summary.has_hallucination is False


def test_clean_fixspec_produces_no_narrative_warnings(tmp_path):
    """Regression guard: a FixSpec without structural-remove claims stays clean."""
    summary = validate_fix_spec_claims(
        {
            "targets": [],
            "primary_issue": "refactor the data layer for clarity",
            "analysis_narrative": "just analysis, no deletions claimed",
        },
        repo_root=str(tmp_path),
    )
    assert summary.narrative_warnings == []
    assert summary.has_hallucination is False


def test_symbol_resolution_uses_matched_targets_file_not_first(tmp_path):
    """A claim mentioning a later target's symbol resolves to THAT target's file.

    Two targets with different files. The claim text mentions target[1]'s symbol
    (``beta``) and gives a line range but no file path. The old code discarded the
    extracted symbol and took ``targets[0].file`` blindly.

    To make the chosen file observable, the two files differ structurally at the
    claimed line range:
      * a.py  (target[0], symbol ``alpha``): lines 5-7 fall INSIDE method alpha
        -> a claim validated against a.py is flagged hallucinated.
      * b.py  (target[1], symbol ``beta``):  lines 5-7 are orphaned (no method)
        -> a claim validated against b.py is a VALID claim.

    Correct symbol resolution therefore yields exactly one valid claim and zero
    hallucinations; the old targets[0] behaviour yields one hallucination.
    """
    _write(tmp_path, "a.py", """\
        # line 1
        # line 2
        # line 3
        def alpha():
            x = 1
            y = 2
            return x
    """)
    _write(tmp_path, "b.py", """\
        # line 1
        # line 2
        # line 3
        # line 4
        x = 1
        y = 2
        z = 3
    """)
    summary = validate_fix_spec_claims(
        {
            "targets": [
                {"symbol": "alpha", "file": "a.py"},
                {"symbol": "beta", "file": "b.py"},
            ],
            "primary_issue": "",
            "analysis_narrative": "remove the orphaned beta block at lines 5-7",
        },
        repo_root=str(tmp_path),
    )
    # The claim was validated against b.py (matched symbol) -> valid orphan.
    assert len(summary.valid_claims) == 1
    assert summary.valid_claims[0].file_path == "b.py"
    assert summary.has_hallucination is False
    assert summary.narrative_warnings == []
