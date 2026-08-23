"""Tests for read_file's line cap, output budget and over-cap guidance.

``read_file`` used to refuse any file over 200 lines with a bare line count,
while placing NO cap on an explicit ``start_line``/``end_line`` — so the safe
case (a 250-line module) cost an extra round-trip and the dangerous one
(``end_line=999999`` on a 6.4K-line module, ~130K tokens) sailed through, via
exactly the escape hatch the refusal message recommended.

Pinned here:
  1. files within the line cap still return full content,
  2. over the cap, the response carries the symbol outline — the model can name
     its next read instead of guessing a range,
  3. over the cap with no extractable symbols degrades to the old bare count,
     so this path is never worse than what it replaced,
  4. an explicit range is truncated at the char budget, on a line boundary,
     naming the line to resume from,
  5. a single line wider than the whole budget still advances the resume line
     (otherwise "continue from N" would loop forever),
  6. ``metadata`` is always a dict — a handler returning None crashes the
     result cache in ToolRegistry._dispatch_impl,
  7. every outline row carries the symbol's END line, not just its start, and
     the range it prints is one a follow-up read_file accepts verbatim.
"""

from __future__ import annotations

import re
from typing import ClassVar

from external_llm.agent.config.thresholds import config as _cfg
from external_llm.agent.symbol_search import SymbolSearcher
from external_llm.agent.tool_handlers.read_tools import _outline_extent
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry


def _reg(tmp_path):
    return ToolRegistry(str(tmp_path), AgentConfig())


def _write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestLineCap:
    def test_file_within_cap_returns_full_content(self, tmp_path):
        n = _cfg.lines.READ_FILE_FULL_LINES
        _write(tmp_path, "small.py", "\n".join(f"x{i} = {i}" for i in range(n)))
        res = _reg(tmp_path).dispatch("read_file", {"path": "small.py"})
        assert res.ok
        assert "x0 = 0" in res.content
        assert f"x{n - 1} = {n - 1}" in res.content
        assert not res.metadata.get("over_line_cap")

    def test_median_sized_file_no_longer_needs_a_second_call(self, tmp_path):
        """281 lines — the median module in this repo — used to be refused."""
        _write(tmp_path, "median.py", "\n".join(f"y{i} = {i}" for i in range(281)))
        res = _reg(tmp_path).dispatch("read_file", {"path": "median.py"})
        assert res.ok
        assert "y280 = 280" in res.content, "a 281-line file must come back whole"

    def test_over_cap_returns_outline_not_just_a_count(self, tmp_path):
        n = _cfg.lines.READ_FILE_FULL_LINES + 50
        body = ["def alpha():", "    return 1", "", "class Beta:", "    def gamma(self):", "        return 2", ""]
        body += [f"# filler {i}" for i in range(n)]
        _write(tmp_path, "big.py", "\n".join(body))

        res = _reg(tmp_path).dispatch("read_file", {"path": "big.py"})
        assert res.ok
        assert res.metadata["over_line_cap"] is True
        assert res.metadata["line_count"] > _cfg.lines.READ_FILE_FULL_LINES
        # The point of the change: a symbol map, not a bare number.
        assert "alpha" in res.content
        assert "Beta" in res.content
        assert "gamma" in res.content, "methods must be listed — read_symbol takes a name"
        assert "read_symbol" in res.content
        # …and the body itself is still withheld.
        assert "# filler 0" not in res.content

    def test_over_cap_without_symbols_degrades_to_the_bare_count(self, tmp_path):
        """A .txt file yields no outline; the response must still be useful."""
        n = _cfg.lines.READ_FILE_FULL_LINES + 10
        _write(tmp_path, "notes.txt", "\n".join(f"line {i}" for i in range(n)))
        res = _reg(tmp_path).dispatch("read_file", {"path": "notes.txt"})
        assert res.ok
        assert str(n) in res.content
        assert "start_line" in res.content
        assert "line 0" not in res.content


class TestOutlineExtent:
    """Outline rows must carry the END line, not just the start.

    The outline exists to make the FOLLOW-UP read_file exact. Printing only a
    start line left the model to invent ``end_line``, and inventing it is where
    malformed ranges come from — including inverted ones (``start_line=600,
    end_line=460``, observed against a symbol that really did start at 460),
    which fail the call outright and cost a turn.
    """

    # alpha spans 5-7, Beta spans 9-11, VERSION is one line (3).
    _HEAD: ClassVar[list] = [
        "import os",  # 1
        "",  # 2
        "VERSION = 3",  # 3
        "",  # 4
        "def alpha(a, b):",  # 5
        "    x = a + b",  # 6
        "    return x",  # 7
        "",  # 8
        "class Beta:",  # 9
        "    def gamma(self):",  # 10
        "        return 2",  # 11
        "",  # 12
    ]

    def _big_module(self, tmp_path):
        body = list(self._HEAD)
        body += [f"# filler {i}" for i in range(_cfg.lines.READ_FILE_FULL_LINES)]
        _write(tmp_path, "big.py", "\n".join(body))

    def test_outline_row_carries_the_full_extent(self, tmp_path):
        self._big_module(tmp_path)
        res = _reg(tmp_path).dispatch("read_file", {"path": "big.py"})
        assert res.metadata["over_line_cap"] is True
        assert "5–7" in res.content, "a multi-line function must show start–end"
        assert "9–11" in res.content, "a class must show start–end"

    def test_the_printed_range_is_one_read_file_accepts(self, tmp_path):
        """The round trip that matters: copy a range out of the outline, pass it
        straight back, and the symbol comes back whole — no arithmetic, no
        guess, nothing for the model to invert."""
        self._big_module(tmp_path)
        reg = _reg(tmp_path)
        outline = reg.dispatch("read_file", {"path": "big.py"}).content

        m = re.search(r"lines\s+(\d+)–(\d+)\s+\[function\] alpha", outline)
        assert m, f"no start–end row for alpha in:\n{outline[:400]}"
        start, end = int(m.group(1)), int(m.group(2))

        res = reg.dispatch("read_file", {"path": "big.py", "start_line": start, "end_line": end})
        assert res.ok, res.error
        assert "def alpha(a, b):" in res.content, "range must start at the def"
        assert "return x" in res.content, "range must reach the last body line"
        assert "class Beta" not in res.content, "range must stop at the symbol's end"

    def test_one_line_symbol_prints_a_bare_line_number(self, tmp_path):
        """ "3-3" reads like a mistake and says nothing "3" does not."""
        self._big_module(tmp_path)
        res = _reg(tmp_path).dispatch("read_file", {"path": "big.py"})
        assert re.search(r"lines\s+3\s+\[constant\] VERSION", res.content), res.content[:400]
        assert "3–3" not in res.content

    def test_missing_extent_degrades_instead_of_fabricating_one(self):
        """``_outline_ripgrep`` matches a declaration by regex and never sets
        ``end_line``. That path must print the start alone rather than invent an
        end — a wrong range is worse than a missing one."""

        class _NoExtent:
            line = 42
            end_line = None

        assert _outline_extent(_NoExtent()) == "42"

    def test_both_python_outline_paths_agree_on_the_extent(self, tmp_path, monkeypatch):
        """tree-sitter and the ast fallback are separate code; the extent they
        report must not depend on which one ran."""
        self._big_module(tmp_path)
        import external_llm.agent.symbol_search as _ss

        def _alpha(searcher):
            return next(s for s in searcher.get_file_outline("big.py") if s.name == "alpha")

        ts_alpha = _alpha(SymbolSearcher(str(tmp_path)))
        monkeypatch.setattr(_ss, "_HAS_TS", False)
        ast_alpha = _alpha(SymbolSearcher(str(tmp_path)))

        assert (ts_alpha.line, ts_alpha.end_line) == (5, 7)
        assert (ast_alpha.line, ast_alpha.end_line) == (5, 7)


class TestCharBudget:
    def _wide_file(self, tmp_path, lines: int, width: int):
        _write(tmp_path, "wide.py", "\n".join("w" * width for _ in range(lines)))

    def test_explicit_range_is_capped(self, tmp_path):
        """The regression: an explicit range used to have no ceiling at all."""
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        self._wide_file(tmp_path, lines=4000, width=200)  # ~800K chars raw

        res = _reg(tmp_path).dispatch("read_file", {"path": "wide.py", "start_line": 1, "end_line": 999999})
        assert res.ok
        assert len(res.content) < budget * 1.2, "explicit range must respect the output budget"
        assert res.metadata["truncated"] is True

    def test_truncation_names_the_resume_line(self, tmp_path):
        self._wide_file(tmp_path, lines=4000, width=200)
        res = _reg(tmp_path).dispatch("read_file", {"path": "wide.py", "start_line": 1, "end_line": 999999})
        resume = res.metadata["resume_line"]
        assert resume > 1
        assert f"start_line={resume}" in res.content
        # The named line must be exactly the first one NOT emitted: the line
        # before it is present, the line itself is absent. Numbered lines are
        # formatted "  NNN │I│ code", so match that shape rather than the bare
        # number (which also occurs in the truncation prose).
        emitted = res.content.split("```")[1]
        assert f"{resume - 1:6d} │" in emitted
        assert f"{resume:6d} │" not in emitted
        # The header's advertised range must agree with what was emitted.
        assert f"lines 1–{resume - 1}" in res.content

    def test_resume_line_advances_even_for_one_oversized_line(self, tmp_path):
        """A single line wider than the budget must not pin resume_line at itself."""
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        _write(tmp_path, "one.py", "z" * (budget * 2) + "\ntail = 1\n")
        res = _reg(tmp_path).dispatch("read_file", {"path": "one.py", "start_line": 1, "end_line": 999999})
        assert res.ok
        assert res.metadata["resume_line"] == 2, "must move past the oversized line"
        assert len(res.content) < budget * 1.2

    def test_small_range_is_untouched(self, tmp_path):
        _write(tmp_path, "s.py", "\n".join(f"a{i} = {i}" for i in range(100)))
        res = _reg(tmp_path).dispatch("read_file", {"path": "s.py", "start_line": 10, "end_line": 12})
        assert res.ok
        assert not res.metadata.get("truncated")
        assert "a9 = 9" in res.content and "a11 = 11" in res.content
        assert "a12 = 12" not in res.content


class TestResultContract:
    def test_metadata_is_always_a_dict(self, tmp_path):
        """ToolRegistry._dispatch_impl does dict(result.metadata) when caching a
        read-only tool; a None here kills the whole call, and only when the
        cache is enabled — invisible to any test that disables it."""
        _write(tmp_path, "a.py", "x = 1\n")
        n = _cfg.lines.READ_FILE_FULL_LINES + 5
        _write(tmp_path, "b.py", "\n".join(f"c{i} = {i}" for i in range(n)))
        reg = _reg(tmp_path)
        for args in (
            {"path": "a.py"},
            {"path": "b.py"},
            {"path": "b.py", "start_line": 1, "end_line": 3},
            {"path": "b.py", "start_line": 99999},
        ):
            res = reg.dispatch("read_file", args)
            assert isinstance(res.metadata, dict), args

    def test_repeated_dispatch_hits_the_cache_without_raising(self, tmp_path):
        _write(tmp_path, "a.py", "x = 1\n")
        reg = _reg(tmp_path)
        first = reg.dispatch("read_file", {"path": "a.py"})
        second = reg.dispatch("read_file", {"path": "a.py"})
        assert first.ok and second.ok
        assert first.content == second.content


def test_oversized_single_line_signals_partial_line(tmp_path):
    """A single line wider than the budget is returned only as a prefix; the
    response must flag that the line's tail was dropped mid-line and is NOT
    recoverable by re-reading (start_line is line-granular).

    Without this signal the caller mistakes the partial line for the full line
    and resumes at the next line, permanently losing the tail — a silent data
    loss that an edit could be built on top of (minified JS/CSS, single-line
    JSON, base64 blobs)."""
    budget = _cfg.lines.READ_FILE_MAX_CHARS
    _write(tmp_path, "one.py", "x" * (budget * 2) + "\ntail = 1\n")
    res = _reg(tmp_path).dispatch("read_file", {"path": "one.py", "start_line": 1, "end_line": 999999})
    assert res.ok
    # Still advances past the oversized line (pinned by the existing test).
    assert res.metadata["resume_line"] == 2
    # NEW: line 1 was emitted only as a prefix.
    assert res.metadata["partial_line"] == 1
    assert "REST OF THAT LINE was dropped" in res.content
    assert "NOT recoverable" in res.content


def test_multi_line_truncation_has_no_partial_line_flag(tmp_path):
    """Ordinary multi-line truncation (many small lines) must NOT set
    ``partial_line`` — only the single-oversized-line path does, so the flag's
    presence is a reliable signal of mid-line data loss."""
    _write(tmp_path, "many.py", "\n".join("a" * 200 for _ in range(4000)))
    res = _reg(tmp_path).dispatch("read_file", {"path": "many.py", "start_line": 1, "end_line": 999999})
    assert res.ok
    assert res.metadata["truncated"] is True
    assert "partial_line" not in res.metadata
    assert "REST OF THAT LINE was dropped" not in res.content


class TestMalformedRange:
    """A zero line number is malformed, and must not read as "whole file".

    ``int(end_line or len(lines))`` made ``end_line=0`` falsy-fall-back to the
    last line, so a malformed range was silently reinterpreted as the widest
    possible one: on a 10K-line file it returned the char budget's worth
    (~60 KB, ≈15K tokens) instead of the ~3 KB outline guidance a no-range read
    gets. The model never learned its argument was wrong.

    NOTE the deliberate contrast with :meth:`TestOutputBudget.
    test_explicit_range_is_capped`: a *well-formed* wide range (``end_line=
    999999``) legitimately bypasses the line cap and is bounded by the char
    budget instead. That is by design. Only a malformed bound errors.
    """

    def _long_file(self, tmp_path):
        n = _cfg.lines.READ_FILE_FULL_LINES * 3
        _write(tmp_path, "long.py", "\n".join(f"x = {i}" for i in range(n)))
        return n

    def test_end_line_zero_is_rejected_not_widened(self, tmp_path):
        n = self._long_file(tmp_path)
        res = _reg(tmp_path).dispatch("read_file", {"path": "long.py", "end_line": 0})
        # ok=False since the range errors were split by mistake: a malformed
        # bound is a failed call, not an answer (see the range-error tests
        # below). The invariant this test was written for is unchanged — the
        # response must stay small, i.e. NOT widened to the whole file.
        assert res.ok is False
        assert "1-based" in res.error
        assert len(res.content) < 500, (
            "end_line=0 returned a payload — it was widened to the whole file "
            f"instead of reported as malformed (file has {n} lines)"
        )
        assert not res.metadata.get("truncated")

    def test_string_zero_is_rejected_too(self, tmp_path):
        """Models quote numbers; the coercion must not reopen the hole."""
        self._long_file(tmp_path)
        res = _reg(tmp_path).dispatch("read_file", {"path": "long.py", "end_line": "0"})
        assert res.ok is False and "1-based" in res.error

    def test_well_formed_range_still_works(self, tmp_path):
        self._long_file(tmp_path)
        res = _reg(tmp_path).dispatch("read_file", {"path": "long.py", "start_line": 10, "end_line": 20})
        assert res.ok and "lines 10–20" in res.content


class TestReadSymbolCharBudget:
    """read_symbol shares read_file's char budget — and must share its
    *arithmetic*, not reimplement it.

    read_symbol originally had no budget at all (``read_symbol("WriteToolsMixin")``
    returned ~356K chars / ~89K tokens in one tool result, while ``read_file`` on
    the same file refused at its 800-line cap and steered the model here).
    The budget was then added as a COPY of read_file's loop, against a 0-based
    origin, which reintroduced both defects the tests above pin for read_file:
    a resume line naming an already-emitted line, and no partial-line flag on an
    over-wide line. Both derivations now live in ``_apply_char_budget``.
    """

    def _sym_file(self, tmp_path, body_lines: int, width: int):
        body = "\n".join("    # " + "w" * width for _ in range(body_lines))
        _write(tmp_path, "big.py", f"def target():\n{body}\n    return 1\n")

    def test_symbol_body_is_capped(self, tmp_path):
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        self._sym_file(tmp_path, body_lines=4000, width=200)  # ~800K chars raw
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert len(res.content) < budget * 1.2, "read_symbol must respect the output budget"
        assert "Truncated at the" in res.content

    def test_resume_line_is_the_first_line_not_emitted(self, tmp_path):
        """The regression: ``start + len(kept)`` (0-based origin) named the LAST
        emitted line, so continuing there re-read a line the caller already had."""
        self._sym_file(tmp_path, body_lines=4000, width=200)
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        import re as _re

        resume = int(_re.search(r"start_line=(\d+)", res.content).group(1))
        emitted = [int(m.group(1)) for m in _re.finditer(r"^\s*(\d+) │", res.content, _re.M)]
        assert emitted, res.content
        assert resume == emitted[-1] + 1, (
            f"resume must be the first line NOT emitted: emitted through {emitted[-1]}, resume={resume}"
        )
        assert f"Lines {resume}–" in res.content, "the prose range must start at the first un-emitted line"

    def test_oversized_single_line_advances_and_flags_partial(self, tmp_path):
        """An over-wide line must advance PAST itself and say the tail is gone.

        Regression: read_symbol named that same line as the resume point (so the
        retry returned the identical prefix) and omitted read_file's
        unrecoverable-tail warning, so the caller believed re-reading would
        recover the rest."""
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        _write(tmp_path, "one.py", "def target():\n    x = '" + "z" * (budget * 2) + "'\n    return x\n")
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert len(res.content) < budget * 1.2
        # Line 1 ("def target():") fits, line 2 is the over-wide one.
        assert "start_line=2" in res.content, res.content[-300:]

    def test_symbol_within_budget_is_untouched(self, tmp_path):
        _write(tmp_path, "s.py", "def target():\n    return 42\n")
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert "return 42" in res.content
        assert "Truncated at the" not in res.content

    def test_truncation_is_visible_in_metadata(self, tmp_path):
        """The prose notice is for the model; the agent loop and telemetry read
        ``metadata``. Emitting one without the other made read_symbol truncation
        the only dropped output no consumer could detect programmatically."""
        self._sym_file(tmp_path, body_lines=4000, width=200)
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert res.metadata["truncated"] is True
        # resume_line is a FILE line number — the resumption call is a read_file.
        assert res.metadata["resume_line"] > 1
        assert f"start_line={res.metadata['resume_line']}" in res.content
        assert res.metadata["line_count"] == 4002  # def + 4000 body + return
        assert "partial_line" not in res.metadata

    def test_partial_line_is_flagged_in_metadata(self, tmp_path):
        """``partial_line`` fires only when the window's FIRST line is itself
        over-budget — that is the sole case where output is dropped MID-line and
        so cannot be recovered by resuming at a line boundary. A later over-wide
        line (see ``test_oversized_single_line_advances_and_flags_partial``) is
        ordinary line-boundary truncation and must NOT set the flag, or it stops
        being a reliable signal of unrecoverable loss."""
        params = ", ".join(f"a{i}=1" for i in range(9000))  # def line > budget
        _write(tmp_path, "one.py", f"def target({params}):\n    return 1\n")
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert res.metadata["partial_line"] == 1
        assert res.metadata["resume_line"] == 2, "must advance past the over-wide line"
        assert "REST OF THAT LINE was dropped" in res.content

    def test_untruncated_symbol_has_no_truncation_metadata(self, tmp_path):
        _write(tmp_path, "s.py", "def target():\n    return 42\n")
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        assert res.ok, res.error
        assert res.metadata == {}


# ── Range errors must name the right mistake ─────────────────────────────────
# One message covered two unrelated errors and reported the CLAMPED end, so
# asking for 9999-10005 of a 200-line file came back as "Line range 9999-200 is
# out of range" — a range the caller never sent. And an INVERTED range whose
# bounds both sit inside the file was answered with "file has 200 lines", which
# tells the model nothing it can act on. Both returned ok=True, so the failure
# read as "asked and answered" and no retry followed (the reasoning
# _tool_read_symbol already records for its own missing-argument case).


def _range_err(tmp_path, **args):
    _write(tmp_path, "big.py", "".join(f"l{i}=1\n" for i in range(1, 201)))
    return _reg(tmp_path).dispatch("read_file", {"path": "big.py", **args})


def test_inverted_range_says_so(tmp_path):
    res = _range_err(tmp_path, start_line=50, end_line=10)
    assert res.ok is False, "an empty range is a failed call, not an answer"
    assert "50" in res.error and "10" in res.error
    assert "after end_line" in res.error


def test_past_end_of_file_says_so(tmp_path):
    res = _range_err(tmp_path, start_line=9999, end_line=10005)
    assert res.ok is False
    assert "past the end" in res.error
    # The clamped end must not be reported back as if the caller asked for it.
    assert "9999–200" not in res.error and "9999-200" not in res.error


def test_start_past_end_without_end_line_is_not_called_inverted(tmp_path):
    """With end_line omitted the default IS the last line, so blaming an
    "end_line" the caller never sent repeats the misdirection being fixed."""
    res = _range_err(tmp_path, start_line=201)
    assert res.ok is False
    assert "past the end" in res.error
    assert "end_line" not in res.error


def test_end_line_past_eof_still_reads_to_eof(tmp_path):
    """Clamping a too-large end_line is correct and must keep working — only a
    range that yields NO lines is an error."""
    res = _range_err(tmp_path, start_line=198, end_line=10005)
    assert res.ok is True
    assert "l198=1" in res.content and "l200=1" in res.content
