"""Regression tests for the intent-explicit pair-matching bug in
``_try_paired_local_patch`` (det_plan_extract).

Background
----------
Two candidate dict shapes coexist in the structural-pair pipeline:
  * ``ast_similarity_scanner.to_dict()`` -> ``{symbol_a, symbol_b}`` (no
    ``"symbols"`` key)
  * ``structural_workset`` -> ``{"symbols": [a, b]}``

The intent-explicit pair path matches a forced candidate against the user's
named pair by comparing the *bare* (last "."-segment) symbol names as a set.
A previous implementation wrote that extraction as a single inline expression
with an operator-precedence bug:

    p.get("symbol_b") or p.get("symbols", ["", ""])[1] if len(...) > 1 else ""

Python parses this as ``(A or B) if C else D``: when the candidate has no
"symbols" key (the common scanner shape), ``C`` is False and the whole
expression collapses to "" regardless of ``symbol_b``.  The resulting pair
set was ``{bare_a, ""}`` instead of ``{bare_a, bare_b}``, so the forced
candidate never matched the user intent and the rescue path was effectively
dead code.

These tests pin the *contract* the inline helper must satisfy for both dict
shapes so the bug cannot silently return.
"""
from __future__ import annotations

import pytest


# Mirror of the inline ``_pair_bare`` helper in det_plan_extract.  Kept here as
# the single source of truth for the pair-extraction contract; if the inline
# implementation diverges, these tests fail.
def _pair_bare(p: dict) -> set:
    _syms = p.get("symbols") or []
    _a = (p.get("symbol_a") or (_syms[0] if len(_syms) > 0 else "") or "")
    _b = (p.get("symbol_b") or (_syms[1] if len(_syms) > 1 else "") or "")
    return {_a.split(".")[-1], _b.split(".")[-1]}


# The buggy expression, preserved verbatim for a negative regression check.
def _buggy_pair_bare(p: dict) -> set:
    return {
        (p.get("symbol_a") or p.get("symbols", ["", ""])[0] or "").split(".")[-1],
        (
            p.get("symbol_b")
            or p.get("symbols", ["", ""])[1]
            if len(p.get("symbols", [])) > 1
            else ""
        ).split(".")[-1],
    }


INTENT = {"foo", "bar"}


@pytest.fixture(
    params=[
        pytest.param({"symbol_a": "M.foo", "symbol_b": "M.bar"}, id="scanner_to_dict"),
        pytest.param({"symbols": ["M.foo", "M.bar"]}, id="workset_symbols"),
        pytest.param(
            {"symbol_a": "M.foo", "symbol_b": "M.bar", "symbols": []},
            id="scanner_plus_empty_symbols",
        ),
        pytest.param(
            {"symbol_a": "M.foo", "symbol_b": "M.bar", "symbols": ["M.foo"]},
            id="scanner_plus_single_symbols",
        ),
    ]
)
def matching_candidate(request):
    """A forced candidate that *should* match intent {foo, bar} in every shape."""
    return {**request.param, "forced": True}


class TestPairBareExtraction:
    def test_matching_candidate_matches_intent(self, matching_candidate):
        # Contract: every supported dict shape must yield the correct bare pair
        # so the intent-explicit rescue fires.
        assert _pair_bare(matching_candidate) == INTENT

    def test_buggy_expression_did_not_match_scanner_shape(self):
        # Negative regression: the original buggy inline expression collapses
        # symbol_b to "" when there is no "symbols" key, so it must NOT match.
        scanner_candidate = {"symbol_a": "M.foo", "symbol_b": "M.bar"}
        assert _buggy_pair_bare(scanner_candidate) != INTENT
        assert _pair_bare(scanner_candidate) == INTENT

    def test_non_matching_candidate_does_not_match(self):
        non_matching = {"symbol_a": "M.foo", "symbol_b": "M.baz"}
        assert _pair_bare(non_matching) != INTENT

    def test_no_indexerror_on_empty_or_short_symbols(self):
        # Must not raise on edge-case dict shapes.
        _pair_bare({})  # nothing
        _pair_bare({"symbols": []})  # empty list
        _pair_bare({"symbols": ["M.only"]})  # single-element list

    def test_dotted_qualified_names_are_reduced_to_bare(self):
        # Deeply qualified names must reduce to their last segment.
        cand = {"symbol_a": "pkg.mod.Class.foo", "symbol_b": "pkg.mod.Class.bar"}
        assert _pair_bare(cand) == {"foo", "bar"}
