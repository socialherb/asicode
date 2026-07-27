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
     result cache in ToolRegistry._dispatch_impl.
"""
from __future__ import annotations

from external_llm.agent.config.thresholds import config as _cfg
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


class TestCharBudget:
    def _wide_file(self, tmp_path, lines: int, width: int):
        _write(tmp_path, "wide.py", "\n".join("w" * width for _ in range(lines)))

    def test_explicit_range_is_capped(self, tmp_path):
        """The regression: an explicit range used to have no ceiling at all."""
        budget = _cfg.lines.READ_FILE_MAX_CHARS
        self._wide_file(tmp_path, lines=4000, width=200)  # ~800K chars raw

        res = _reg(tmp_path).dispatch(
            "read_file", {"path": "wide.py", "start_line": 1, "end_line": 999999}
        )
        assert res.ok
        assert len(res.content) < budget * 1.2, "explicit range must respect the output budget"
        assert res.metadata["truncated"] is True

    def test_truncation_names_the_resume_line(self, tmp_path):
        self._wide_file(tmp_path, lines=4000, width=200)
        res = _reg(tmp_path).dispatch(
            "read_file", {"path": "wide.py", "start_line": 1, "end_line": 999999}
        )
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
        res = _reg(tmp_path).dispatch(
            "read_file", {"path": "one.py", "start_line": 1, "end_line": 999999}
        )
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
    res = _reg(tmp_path).dispatch(
        "read_file", {"path": "one.py", "start_line": 1, "end_line": 999999}
    )
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
    res = _reg(tmp_path).dispatch(
        "read_file", {"path": "many.py", "start_line": 1, "end_line": 999999}
    )
    assert res.ok
    assert res.metadata["truncated"] is True
    assert "partial_line" not in res.metadata
    assert "REST OF THAT LINE was dropped" not in res.content
