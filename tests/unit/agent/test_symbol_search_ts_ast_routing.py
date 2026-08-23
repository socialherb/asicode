"""Routing tests for the superlinear tree-sitter guard on Python files.

tree-sitter's Python grammar parses a long RUN of indented comment lines
inside a function body quadratically in the run length (measured: 4000
200-char comment lines → 7.9 s vs 3 ms for ast.parse; 1000→0.50 s,
2000→1.98 s, 3000→4.53 s, 4000→7.88 s — while module-level or class-body
comments and non-comment lines stay linear). The guard
``_python_ts_parse_too_costly`` routes such files to the AST path, which
extracts the same symbols; tree-sitter remains the last resort for
syntax-broken files, where ast.parse raises and the error-tolerant
tree-sitter parse still finds symbols.

These tests pin the three-way contract:
  * comment walls inside function bodies go to AST and are FAST,
  * normal files keep the tree-sitter path (identical results),
  * syntax-broken files still yield symbols via the tree-sitter
    last resort.
"""

from __future__ import annotations

import time

import pytest

from external_llm.agent.symbol_search import (
    _HAS_TS,
    SymbolSearcher,
    _python_ts_parse_too_costly,
)
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

pytestmark = pytest.mark.skipif(not _HAS_TS, reason="tree-sitter not installed")

# The guard's thresholds — pinned here so a threshold change is a visible
# decision, not a silent behaviour drift.
MIN_LINES = 300
COMMENT_RUN = 50


def _comment_wall(lines: int = 4000, width: int = 200, indent: str = "    ") -> str:
    return "\n".join(f"{indent}# " + "w" * width for _ in range(lines))


def _reg(tmp_path) -> ToolRegistry:
    return ToolRegistry(str(tmp_path), AgentConfig())


class TestPrefilterDecision:
    def test_comment_wall_inside_function_is_costly(self):
        src = f"def target():\n{_comment_wall()}\n    return 1\n"
        assert _python_ts_parse_too_costly(src)

    def test_module_level_comments_are_not_costly(self):
        # Column-0 comments at module level — the same lines tree-sitter
        # parses in 8 ms — must NOT trip the guard.
        assert not _python_ts_parse_too_costly(_comment_wall(indent=""))

    def test_short_runs_stay_on_tree_sitter(self):
        # The largest run in this repo is 32 — well under the 50-line trigger.
        src = "def target():\n" + _comment_wall(lines=COMMENT_RUN - 1) + "\n    return 1\n"
        assert not _python_ts_parse_too_costly(src)

    def test_small_files_keep_error_tolerance(self):
        # def(1) + 297 comments + return(1) = 299 lines — below the 300-line
        # floor, so tree-sitter stays even though the run is long.
        src = f"def target():\n{_comment_wall(lines=MIN_LINES - 3)}\n    return 1\n"
        assert not _python_ts_parse_too_costly(src)

    def test_non_comment_long_lines_are_not_costly(self):
        src = "def target():\n" + "\n".join("    x" + "w" * 196 for _ in range(4000)) + "\n    return 1\n"
        assert not _python_ts_parse_too_costly(src)


class TestCommentWallSpeed:
    """The measured regression: read_symbol/get_file_outline took 7.9 s on a
    4000x200-char comment-wall file; the guard must keep them in the
    tens-of-milliseconds range while returning identical results."""

    def _wall_file(self, tmp_path) -> None:
        (tmp_path / "big.py").write_text(f"def target():\n{_comment_wall()}\n    return 1\n", encoding="utf-8")

    def test_read_symbol_on_comment_wall_is_fast_and_correct(self, tmp_path):
        self._wall_file(tmp_path)
        t0 = time.perf_counter()
        res = _reg(tmp_path).dispatch("read_symbol", {"name": "target"})
        elapsed = time.perf_counter() - t0
        assert res.ok, res.error
        assert elapsed < 2.0, f"comment-wall read_symbol took {elapsed:.2f}s"
        assert res.metadata["line_count"] == 4002
        assert res.metadata["truncated"] is True

    def test_get_file_outline_on_comment_wall_is_fast(self, tmp_path):
        self._wall_file(tmp_path)
        searcher = SymbolSearcher(str(tmp_path))
        t0 = time.perf_counter()
        outline = searcher.get_file_outline("big.py")
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"comment-wall outline took {elapsed:.2f}s"
        assert [s.name for s in outline] == ["target"]
        assert outline[0].line == 1
        assert outline[0].end_line == 4002

    def test_find_symbol_on_comment_wall_reports_the_extent(self, tmp_path):
        self._wall_file(tmp_path)
        searcher = SymbolSearcher(str(tmp_path))
        t0 = time.perf_counter()
        defs = searcher.find_symbol("target", kind="any")
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"comment-wall find_symbol took {elapsed:.2f}s"
        assert len(defs) == 1
        assert defs[0].line == 1
        assert defs[0].end_line == 4002


class TestSyntaxBrokenLastResort:
    """ast.parse raises on broken files, but the error-tolerant tree-sitter
    parse still extracts symbols — the guard must keep that behaviour for
    ordinary broken files, and must NOT pay the superlinear cost for a
    broken comment wall whose symbols are untrustworthy anyway."""

    def test_broken_file_without_wall_still_extracts_symbols(self, tmp_path):
        p = tmp_path / "broken.py"
        p.write_text("def target(:\n    return 1\n", encoding="utf-8")
        searcher = SymbolSearcher(str(tmp_path))
        out = searcher._extract_all_python_symbols(p, "broken.py")
        assert "target" in out

    def test_broken_comment_wall_file_skips_the_slow_last_resort(self, tmp_path):
        p = tmp_path / "broken.py"
        p.write_text(f"def target(:\n{_comment_wall()}\n    return 1\n", encoding="utf-8")
        assert _python_ts_parse_too_costly(p.read_text(encoding="utf-8"))
        searcher = SymbolSearcher(str(tmp_path))
        t0 = time.perf_counter()
        out = searcher._extract_all_python_symbols(p, "broken.py")
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"broken comment wall cost {elapsed:.2f}s"
        assert "target" not in out
