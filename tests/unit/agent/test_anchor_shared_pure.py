"""Unit tests for the pure anchor-matching helpers in anchor_shared.

These functions are pure (list[str] in -> dict/int out) with no filesystem or
LLM dependency, yet were previously exercised ONLY through integration tests
(``sem_harness._tool_anchor_edit``) that end at the write-tool boundary. That
left several documented failure classes and disambiguation branches
unverified:

* ``resolve_multiline_anchor`` -- the ``anchor_multiline_pattern`` (empty),
  ``anchor_miss``, and EOF-extension branches of ``multiline_mismatch``.
* ``_find_anchor_line`` -- occurrence overflow fallback, ctx_before / ctx_after
  disambiguation, and the last-line ctx_after skip.

This file locks those branches directly so a regression in the resolution
logic is caught at the unit level instead of only as a downstream write-tool
failure.
"""
import logging

from external_llm.agent.anchor_shared import (
    _find_anchor_line,
    resolve_multiline_anchor,
)

# -- resolve_multiline_anchor -------------------------------------------------


class TestResolveMultilineAnchor:
    """Direct unit tests for the multiline block resolver.

    Contract (see docstring): returns ``{ok, anchor, end, count}`` on success or
    ``{ok: False, error, failure_class}`` on failure with one of three classes.
    """

    # -- happy path ------------------------------------------------------------
    def test_basic_two_line_block_resolves(self):
        lines = ["x = 1\n", "y = 2\n", "z = 3\n"]
        res = resolve_multiline_anchor(lines, "x = 1\ny = 2")
        assert res["ok"] is True
        assert res["anchor"] == 0
        assert res["end"] == 1
        assert res["count"] == 2

    def test_three_line_block_resolves(self):
        lines = ["a\n", "b\n", "c\n", "d\n"]
        res = resolve_multiline_anchor(lines, "a\nb\nc")
        assert res["ok"] is True
        assert (res["anchor"], res["end"]) == (0, 2)
        assert res["count"] == 3

    def test_block_in_the_middle_of_file(self):
        lines = ["header\n", "x = 1\n", "y = 2\n", "footer\n"]
        res = resolve_multiline_anchor(lines, "x = 1\ny = 2")
        assert res["ok"] is True
        assert (res["anchor"], res["end"]) == (1, 2)

    def test_indented_block_strip_matches(self):
        """Pattern lines need only strip-match (indent tolerant)."""
        lines = ["def foo():\n", "    return 1\n", "    return 2\n"]
        res = resolve_multiline_anchor(lines, "return 1\nreturn 2")
        assert res["ok"] is True
        assert (res["anchor"], res["end"]) == (1, 2)

    # -- failure class: anchor_multiline_pattern -------------------------------
    def test_empty_pattern_rejected(self):
        res = resolve_multiline_anchor(["x = 1\n"], "")
        assert res["ok"] is False
        assert res["failure_class"] == "anchor_multiline_pattern"

    def test_whitespace_only_pattern_rejected(self):
        """A pattern that is only blank/whitespace lines has no anchor."""
        res = resolve_multiline_anchor(["x = 1\n"], "   \n\n  \t ")
        assert res["ok"] is False
        assert res["failure_class"] == "anchor_multiline_pattern"

    # -- failure class: anchor_miss --------------------------------------------
    def test_first_line_not_found_rejected(self):
        res = resolve_multiline_anchor(["x = 1\n", "y = 2\n"], "nope = 0\ny = 2")
        assert res["ok"] is False
        assert res["failure_class"] == "anchor_miss"
        # Error must steer the caller toward the first line specifically.
        assert "first line" in res["error"].lower()

    def test_anchor_miss_takes_precedence_over_block_check(self):
        """If the first line is absent we never get to compare later lines."""
        res = resolve_multiline_anchor(["x = 1\n"], "WRONG\nALSO WRONG")
        assert res["failure_class"] == "anchor_miss"

    # -- failure class: multiline_mismatch (line content) ----------------------
    def test_second_line_mismatch_rejected(self):
        res = resolve_multiline_anchor(
            ["x = 1\n", "y = 2\n", "z = 3\n"], "x = 1\nWRONG"
        )
        assert res["ok"] is False
        assert res["failure_class"] == "multiline_mismatch"
        # Error names the offending pattern line for actionable feedback.
        assert "WRONG" in res["error"]

    def test_partial_substring_match_is_accepted(self):
        """A pattern line that is a substring of the file line still matches
        (strip-match uses ``in``), so no mismatch."""
        lines = ["    value = compute(x, y)\n", "    other = 1\n"]
        res = resolve_multiline_anchor(lines, "value = compute\nother")
        assert res["ok"] is True

    # -- failure class: multiline_mismatch (EOF extension) ---------------------
    def test_non_empty_pattern_line_past_eof_rejected(self):
        """A non-empty pattern line that runs past EOF is a mismatch."""
        lines = ["x = 1\n"]
        res = resolve_multiline_anchor(lines, "x = 1\ny = 2\nz = 3")
        assert res["ok"] is False
        assert res["failure_class"] == "multiline_mismatch"
        assert "end of file" in res["error"].lower()

    def test_trailing_blank_pattern_line_past_eof_tolerated(self):
        """A trailing WHITESPACE-only pattern line past EOF is tolerated,
        not treated as a mismatch (blank pattern lines are skipped)."""
        lines = ["x = 1\n", "y = 2\n"]
        res = resolve_multiline_anchor(lines, "x = 1\ny = 2\n   \n")
        assert res["ok"] is True
        # end clamps to last file line.
        assert res["end"] == 1
        # count is the number of NON-EMPTY pattern lines: blank lines are
        # filtered out of _pat_lines up front, so count == 2 here.
        assert res["count"] == 2

    # -- leading blank pattern lines skipped -----------------------------------
    def test_leading_blank_lines_skipped(self):
        """Blank lines at the START of the pattern are dropped; the first
        non-empty line becomes the anchor search target."""
        lines = ["x = 1\n", "y = 2\n"]
        res = resolve_multiline_anchor(lines, "\n\n\nx = 1\ny = 2")
        assert res["ok"] is True
        assert res["anchor"] == 0

    # -- occurrence & context forwarding ---------------------------------------
    def test_occurrence_selects_nth_block(self):
        lines = ["x = 1\n", "y = 2\n", "x = 1\n", "y = 2\n"]
        # First line 'x = 1' appears twice; occurrence=1 -> first block.
        res = resolve_multiline_anchor(lines, "x = 1\ny = 2", occurrence=1)
        assert res["ok"] is True
        assert res["anchor"] == 0

    def test_ctx_after_disambiguates_block(self):
        lines = ["a = 1\n", "b = 2\n", "a = 1\n", "b = 2\n", "STOP\n"]
        # Two 'a = 1' lines; ctx_after pins the first-line candidate. Neither
        # 'a = 1' is followed directly by 'STOP' (both are followed by 'b = 2'),
        # so ctx_after='STOP' finds nothing -> anchor_miss.
        res = resolve_multiline_anchor(lines, "a = 1\nb = 2", ctx_after="STOP")
        assert res["ok"] is False
        assert res["failure_class"] == "anchor_miss"


# -- _find_anchor_line --------------------------------------------------------


class TestFindAnchorLine:
    """Direct unit tests for occurrence + context disambiguation semantics.

    These branches are currently only reachable through the write-tool path.
    """

    # -- occurrence semantics --------------------------------------------------
    def test_default_occurrence_returns_last_match(self):
        lines = ["x = 1\n", "y = 2\n", "x = 1\n"]
        # occurrence defaults to -1 -> LAST match.
        assert _find_anchor_line(lines, "x = 1") == 2

    def test_occurrence_minus_one_explicit_last(self):
        lines = ["a\n", "b\n", "a\n", "a\n"]
        assert _find_anchor_line(lines, "a", occurrence=-1) == 3

    def test_occurrence_one_returns_first(self):
        lines = ["a\n", "b\n", "a\n"]
        assert _find_anchor_line(lines, "a", occurrence=1) == 0

    def test_occurrence_two_returns_second(self):
        lines = ["a\n", "a\n", "a\n"]
        assert _find_anchor_line(lines, "a", occurrence=2) == 1

    def test_not_found_returns_none(self):
        assert _find_anchor_line(["x = 1\n"], "missing") is None

    def test_empty_file_returns_none(self):
        assert _find_anchor_line([], "x = 1") is None

    # -- occurrence overflow fallback ------------------------------------------
    def test_occurrence_overflow_falls_back_to_last(self, caplog):
        """Requesting occurrence=5 with only 2 matches must fall back to the
        last match (with a warning), NOT return None."""
        lines = ["a\n", "b\n", "a\n"]
        with caplog.at_level(logging.WARNING, logger="asicode.anchor_shared"):
            result = _find_anchor_line(lines, "a", occurrence=5)
        assert result == 2  # last match, not None
        assert any(
            "ANCHOR_OCCURRENCE_FALLBACK" in r.getMessage()
            for r in caplog.records
        ), "overflow fallback must emit ANCHOR_OCCURRENCE_FALLBACK warning"

    # -- ctx_before disambiguation ---------------------------------------------
    def test_ctx_before_selects_matching_candidate(self):
        lines = ["header\n", "target\n", "target\n"]
        # Both lines 1 and 2 match 'target'; ctx_before='header' pins line 1.
        assert _find_anchor_line(lines, "target", ctx_before="header") == 1

    def test_ctx_before_filters_out_non_matching_candidate(self):
        lines = ["noise\n", "target\n", "target\n"]
        # ctx_before='noise' pins line 1 (the one preceded by 'noise').
        assert _find_anchor_line(lines, "target", ctx_before="noise") == 1

    def test_ctx_before_on_first_line_skipped(self):
        """The ``_i > 0`` guard: at line 0 ctx_before is not checked, so a match
        on the first line is always accepted even when ctx_before could not
        possibly match. Line 1 is filtered out (ctx_before='anything' != the
        line-0 content), leaving line 0 as the sole match."""
        lines = ["target\n", "target\n"]
        assert _find_anchor_line(lines, "target", ctx_before="anything") == 0

    # -- ctx_after disambiguation ----------------------------------------------
    def test_ctx_after_selects_matching_candidate(self):
        lines = ["target\n", "target\n", "footer\n"]
        # Both lines 0 and 1 match 'target'; ctx_after='footer' pins line 1.
        assert _find_anchor_line(lines, "target", ctx_after="footer") == 1

    def test_ctx_after_on_last_line_returns_none(self):
        """The ``_i >= len(lines) - 1`` guard: a match on the last line cannot
        verify ctx_after, so it is skipped. If that was the ONLY match -> None."""
        lines = ["noise\n", "target\n"]
        assert _find_anchor_line(lines, "target", ctx_after="missing") is None

    # -- literal vs regex matching ---------------------------------------------
    def test_regex_pattern_matches(self):
        """``_match_anchor`` falls back to re.search, enabling ^...$ anchors."""
        lines = ["    pass\n", "pass\n"]
        # '^pass$' matches only the unindented line (re.search anchored).
        assert _find_anchor_line(lines, "^pass$") == 1

    def test_invalid_regex_falls_back_to_no_match(self):
        """A syntactically invalid regex must not raise -- it just won't match."""
        lines = ["x = 1\n"]
        # '(' is an invalid regex; _match_anchor catches re.error -> no match.
        assert _find_anchor_line(lines, "(") is None
