"""Behavioral tests for _validate_orphaned_block_claim range-overlap detection.

Locks in the fix for a hallucination-detection gap: a claimed orphaned block
whose line range OVERLAPS a method/class (but whose START line falls outside it)
must still be flagged as hallucinated. The previous containment check only
tested whether the start line was inside the definition, so a block starting
before a method and extending into it slipped through unflagged — contradicting
the function's own docstring contract ("range falls outside any method/class").
"""
from __future__ import annotations

import textwrap

from external_llm.agent.fix_spec_claim_validator import (
    _validate_orphaned_block_claim,
)

import pytest


@pytest.fixture
def repo_with_method(tmp_path):
    """A repo whose only file has a single method spanning known lines.

    After textwrap.dedent + write, the method ``foo`` occupies lines 1-3 of the
    function body region. We construct explicit line numbers below by padding.
    """
    # 10 leading comment/blank lines, then a function def, so the method body
    # lands at a known, non-zero line range we can reason about precisely.
    src = textwrap.dedent('''\
        # line 1
        # line 2
        # line 3
        # line 4
        # line 5
        # line 6
        # line 7
        # line 8
        # line 9
        def foo():
            # line 11
            x = 1
            return x
        # line 14
        # line 15
    ''')
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    return tmp_path, "mod.py"  # foo: def at line 10, body 10-13


def _validate(repo_dir, rel, start, end):
    return _validate_orphaned_block_claim(
        claim=f"orphaned block at lines {start}-{end}",
        file_path=rel,
        line_range=(start, end),
        repo_root=str(repo_dir),
    )


def test_claim_starting_before_method_extending_into_it_is_flagged(repo_with_method):
    """BUG CASE: claim [8, 12] overlaps method foo [10, 13] but start=8 < 10.

    Previously NOT flagged (start outside method); now correctly hallucinated.
    """
    repo, rel = repo_with_method
    res = _validate(repo, rel, 8, 12)
    assert res.hallucinated is True
    assert res.is_valid is False
    assert "foo" in res.reason


def test_claim_spanning_whole_method_is_flagged(repo_with_method):
    """Claim [5, 15] fully contains method foo [10, 13] — must be flagged."""
    repo, rel = repo_with_method
    res = _validate(repo, rel, 5, 15)
    assert res.hallucinated is True
    assert res.is_valid is False


def test_claim_fully_inside_method_is_flagged(repo_with_method):
    """Regression guard: claim [11, 12] inside method foo [10, 13]."""
    repo, rel = repo_with_method
    res = _validate(repo, rel, 11, 12)
    assert res.hallucinated is True
    assert res.is_valid is False


def test_claim_entirely_before_method_is_valid(repo_with_method):
    """Regression guard: claim [1, 5] before method foo [10, 13] — valid orphaned."""
    repo, rel = repo_with_method
    res = _validate(repo, rel, 1, 5)
    assert res.hallucinated is False
    assert res.is_valid is True


def test_claim_entirely_after_method_is_valid(repo_with_method):
    """Regression guard: claim [14, 15] after method foo [10, 13] — valid orphaned."""
    repo, rel = repo_with_method
    res = _validate(repo, rel, 14, 15)
    assert res.hallucinated is False
    assert res.is_valid is True


def test_adjacent_claim_not_overlapping_is_valid(repo_with_method):
    """Claim ending exactly at the method's def line-1 must not be flagged.

    Claim [8, 9], method foo [10, 13]: end=9 < mstart=10 → no overlap → valid.
    """
    repo, rel = repo_with_method
    res = _validate(repo, rel, 8, 9)
    assert res.hallucinated is False
    assert res.is_valid is True


@pytest.fixture
def repo_with_class(tmp_path):
    src = textwrap.dedent('''\
        # line 1
        # line 2
        # line 3
        # line 4
        class Bar:
            # line 6
            def baz(self):
                # line 8
                return 0
        # line 10
    ''')
    f = tmp_path / "cls.py"
    f.write_text(src, encoding="utf-8")
    return tmp_path, "cls.py"  # class Bar: 5-9


def test_class_overlap_from_before_is_flagged(repo_with_class):
    """BUG CASE (class): claim [4, 6] overlaps class Bar [5, 9] but NOT method
    baz [7, 9], start=4 < cstart=5. The method check must miss it (no method
    overlap) and the class check must catch the partial overlap."""
    repo, rel = repo_with_class
    res = _validate(repo, rel, 4, 6)
    assert res.hallucinated is True
    assert res.is_valid is False
    assert "Bar" in res.reason


def test_class_entirely_outside_is_valid(repo_with_class):
    """Regression guard: claim [1, 2] before class Bar [5, 9] — valid."""
    repo, rel = repo_with_class
    res = _validate(repo, rel, 1, 2)
    assert res.hallucinated is False
    assert res.is_valid is True
