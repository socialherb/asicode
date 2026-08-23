"""get_file_outline must reuse the shared per-file symbol map.

``find_symbol`` and ``get_file_outline`` on the same Python file used to parse
it independently (the outline ran its own tree-sitter walk).  Both now go
through ``_python_symbol_map`` — whichever runs first warms ``_py_file_cache``
and the other reuses it, so a find_symbol + outline pair parses the file
exactly once, in either order.

The TS/JS outline (``_outline_ts_js``) follows the same contract through
``_ts_module_map`` / ``_ts_file_cache`` — see TestTsOutlineSharesFindSymbolCache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from external_llm.agent import symbol_search as _ss
from external_llm.agent.symbol_search import SymbolSearcher

_SRC = (
    "import os\n"
    "@dataclass\n"
    "class Foo:\n"
    "    x: int = 1\n"
    "    def bar(self):\n"
    "        return self.x\n"
    "\n"
    "TOP_CONST = 42\n"
    "\n"
    "def top_fn(a, b=1):\n"
    "    return a + b\n"
    "\n"
    "class Outer:\n"
    "    class Inner:\n"
    "        pass\n"
    "    def meth(self):\n"
    "        return 1\n"
)


def _searcher(tmp_path) -> SymbolSearcher:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "widget.py").write_text(_SRC, encoding="utf-8")
    return SymbolSearcher(str(repo))


def _counting_searcher(tmp_path, monkeypatch):
    """Searcher whose extractor records every call (parse-count pin)."""
    searcher = _searcher(tmp_path)
    calls: list[str] = []
    _orig = searcher._extract_all_python_symbols

    def _counting(file_path, rel):
        calls.append(str(file_path))
        return _orig(file_path, rel)

    monkeypatch.setattr(searcher, "_extract_all_python_symbols", _counting)
    return searcher, calls


class TestOutlineSharesFindSymbolCache:
    def test_outline_after_find_symbol_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        assert searcher.find_symbol("Foo", kind="class")
        outline = searcher.get_file_outline("widget.py")
        names = [s.name for s in outline]
        assert "Foo" in names and "top_fn" in names
        # find_symbol parsed the file; the outline must reuse that parse.
        assert calls.count(str(searcher.repo_root / "widget.py")) == 1

    def test_outline_before_find_symbol_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        outline = searcher.get_file_outline("widget.py")
        assert [s.name for s in outline] == ["Foo", "TOP_CONST", "top_fn", "Outer"]
        assert searcher.find_symbol("top_fn", kind="function")
        # The outline warmed the cache; find_symbol must reuse it.
        assert calls.count(str(searcher.repo_root / "widget.py")) == 1

    def test_repeated_outline_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        searcher.get_file_outline("widget.py")
        searcher.get_file_outline("widget.py")
        searcher.get_file_outline("widget.py")
        assert len(calls) == 1

    def test_outline_is_top_level_only(self, tmp_path):
        searcher = _searcher(tmp_path)
        outline = searcher.get_file_outline("widget.py")
        by_name = {s.name: s for s in outline}
        assert set(by_name) == {"Foo", "TOP_CONST", "top_fn", "Outer"}
        # methods / nested classes / class-level constants stay out of the outline
        assert "bar" not in by_name and "meth" not in by_name and "Inner" not in by_name

    def test_outline_sorted_by_line_with_end_line(self, tmp_path):
        searcher = _searcher(tmp_path)
        outline = searcher.get_file_outline("widget.py")
        lines = [s.line for s in outline]
        assert lines == sorted(lines)
        foo = next(s for s in outline if s.name == "Foo")
        assert foo.end_line == 6  # class extent, not just the header line
        assert all(s.end_line for s in outline)

    def test_outline_class_carries_decorators(self, tmp_path):
        """Deliberate additive change vs the former dedicated walk: class
        outline entries now expose decorators, matching find_symbol."""
        searcher = _searcher(tmp_path)
        outline = searcher.get_file_outline("widget.py")
        foo = next(s for s in outline if s.name == "Foo")
        assert foo.decorators == ["dataclass"]
        outer = next(s for s in outline if s.name == "Outer")
        assert outer.decorators is None

    def test_outline_matches_find_symbol_fields(self, tmp_path):
        """The map is the single source of truth — outline entries are the
        same SymbolDef objects find_symbol returns for the same symbols."""
        searcher = _searcher(tmp_path)
        outline = {s.name: s for s in searcher.get_file_outline("widget.py")}
        foo = searcher.find_symbol("Foo", kind="class")[0]
        assert outline["Foo"] is foo
        top_fn = searcher.find_symbol("top_fn", kind="function")[0]
        assert outline["top_fn"].signature == top_fn.signature
        assert outline["top_fn"].end_line == top_fn.end_line


class TestOutlineCacheInvalidation:
    def test_size_change_reparses(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        searcher.get_file_outline("widget.py")
        p = searcher.repo_root / "widget.py"
        p.write_text(_SRC + "def added_fn():\n    pass\n", encoding="utf-8")
        outline = searcher.get_file_outline("widget.py")
        assert "added_fn" in [s.name for s in outline]
        assert len(calls) == 2  # signature mismatch → rebuild

    def test_invalidate_file_caches_drops(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        searcher.get_file_outline("widget.py")
        searcher.invalidate_file_caches(["widget.py"])
        outline = searcher.get_file_outline("widget.py")
        assert outline  # refills on demand
        assert len(calls) == 2

    def test_outline_syntax_broken_still_finds_symbols(self, tmp_path):
        """Syntax-broken file: ast yields nothing but the error-tolerant
        tree-sitter parse still extracts symbols (last resort, unchanged from
        the extractor's contract)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "broken.py").write_text("def ok_fn():\n    return 1\ndef bad_fn(:\n", encoding="utf-8")
        searcher = SymbolSearcher(str(repo))
        outline = searcher.get_file_outline("broken.py")
        assert any(s.name == "ok_fn" for s in outline)


class TestPerFileCacheLruBound:
    """_py_file_cache / _ts_file_cache must be LRU-capped (memory bound)."""

    def test_py_cache_caps_and_evicts_oldest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ss, "_PY_FILE_CACHE_MAX_ENTRIES", 2)
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        repo = Path(searcher.repo_root)
        for name in ("a", "b", "c"):
            (repo / f"{name}.py").write_text(f"def {name}():\n    pass\n", encoding="utf-8")
        for name in ("a", "b", "c"):
            assert searcher._python_symbol_map(repo / f"{name}.py")
        # cap enforced: only the 2 most-recently-used files remain
        keys = {Path(k).name for k in searcher._py_file_cache}
        assert keys == {"b.py", "c.py"}
        # evicted file re-parses on next access (count pin)
        before = len(calls)
        assert searcher._python_symbol_map(repo / "a.py")
        assert len(calls) == before + 1

    def test_py_cache_hit_refreshes_recency(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ss, "_PY_FILE_CACHE_MAX_ENTRIES", 2)
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        repo = Path(searcher.repo_root)
        for name in ("a", "b"):
            (repo / f"{name}.py").write_text(f"def {name}():\n    pass\n", encoding="utf-8")
        for name in ("a", "b"):
            assert searcher._python_symbol_map(repo / f"{name}.py")
        # hit on a moves it to MRU (order becomes b, a)
        assert searcher._python_symbol_map(repo / "a.py")
        (repo / "c.py").write_text("def c():\n    pass\n", encoding="utf-8")
        assert searcher._python_symbol_map(repo / "c.py")
        # LRU (b) evicted; a survived thanks to the recency refresh
        keys = {Path(k).name for k in searcher._py_file_cache}
        assert keys == {"a.py", "c.py"}
        before = len(calls)
        assert searcher._python_symbol_map(repo / "a.py")
        assert len(calls) == before  # still served from cache, no re-parse

    def test_ts_cache_capped_too(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ss, "_TS_FILE_CACHE_MAX_ENTRIES", 2)
        repo = tmp_path / "tsrepo"
        repo.mkdir()
        for name in ("a", "b", "c"):
            (repo / f"{name}.ts").write_text(f"export function {name}() {{ return 1; }}\n", encoding="utf-8")
        searcher = SymbolSearcher(str(repo))
        for name in ("a", "b", "c"):
            searcher._ts_module_map(repo / f"{name}.ts")
        keys = {Path(k).name for k in searcher._ts_file_cache}
        assert keys == {"b.ts", "c.ts"}


class TestRealpathMemoLruBound:
    """_realpath_memo must be LRU-capped (memory bound) like the file caches."""

    def test_realpath_memo_caps_and_evicts_oldest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ss, "_REALPATH_MEMO_MAX_ENTRIES", 2)
        searcher = SymbolSearcher(str(tmp_path))
        calls: list[str] = []
        _orig = _ss.os.path.realpath
        monkeypatch.setattr(_ss.os.path, "realpath", lambda p: (calls.append(p), _orig(p))[1])
        for name in ("a", "b", "c"):
            searcher._cache_key(tmp_path / f"{name}.py")
        # cap enforced: only the 2 most-recently-used keys remain
        assert set(searcher._realpath_memo) == {
            str(tmp_path / "b.py"),
            str(tmp_path / "c.py"),
        }
        # evicted key re-resolves on next access (realpath call count pin)
        before = len(calls)
        searcher._cache_key(tmp_path / "a.py")
        assert len(calls) == before + 1

    def test_realpath_memo_hit_refreshes_recency(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ss, "_REALPATH_MEMO_MAX_ENTRIES", 2)
        searcher = SymbolSearcher(str(tmp_path))
        calls: list[str] = []
        _orig = _ss.os.path.realpath
        monkeypatch.setattr(_ss.os.path, "realpath", lambda p: (calls.append(p), _orig(p))[1])
        searcher._cache_key(tmp_path / "a.py")
        searcher._cache_key(tmp_path / "b.py")
        # hit on a moves it to MRU (order becomes b, a)
        searcher._cache_key(tmp_path / "a.py")
        searcher._cache_key(tmp_path / "c.py")
        # LRU (b) evicted; a survived thanks to the recency refresh
        assert set(searcher._realpath_memo) == {
            str(tmp_path / "a.py"),
            str(tmp_path / "c.py"),
        }
        before = len(calls)
        searcher._cache_key(tmp_path / "a.py")
        assert len(calls) == before  # still served from memo, no re-resolve


# ── TS/JS parity: the outline shares _ts_module_map with find_symbol ──────────
#
# Mirrors the Python contract above: find_symbol and get_file_outline on the
# same TS/JS file used to parse it independently (the outline ran its own
# TSSemanticTracer walk). Both now derive from _ts_module_map, so whichever
# runs first warms _ts_file_cache and the other reuses it.

_TS_SRC = (
    "export interface Item {\n"
    "  id: number\n"
    "}\n"
    "\n"
    "export const LIMIT = 10\n"
    "\n"
    "export function top_fn(a: number, b = 1): number {\n"
    "  return a + b\n"
    "}\n"
    "\n"
    "export class Base {\n"
    "}\n"
    "\n"
    "export class Foo extends Base {\n"
    "  x: number = 1\n"
    "  bar(): number {\n"
    "    return this.x\n"
    "  }\n"
    "}\n"
)


def _ts_searcher(tmp_path) -> SymbolSearcher:
    repo = tmp_path / "tsrepo"
    repo.mkdir()
    (repo / "widget.ts").write_text(_TS_SRC, encoding="utf-8")
    return SymbolSearcher(str(repo))


def _ts_counting_searcher(tmp_path, monkeypatch):
    """Searcher whose TS extractor records every call (parse-count pin)."""
    searcher = _ts_searcher(tmp_path)
    calls: list[str] = []
    _orig = searcher._ts_extract_all

    def _counting(module, rel):
        calls.append(rel)
        return _orig(module, rel)

    monkeypatch.setattr(searcher, "_ts_extract_all", _counting)
    return searcher, calls


try:
    from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer  # noqa: F401

    _HAS_TS_TRACER = True
except ImportError:
    _HAS_TS_TRACER = False


@pytest.mark.skipif(not _HAS_TS_TRACER, reason="TSSemanticTracer unavailable")
class TestTsOutlineSharesFindSymbolCache:
    def test_ts_outline_after_find_symbol_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _ts_counting_searcher(tmp_path, monkeypatch)
        assert searcher.find_symbol("top_fn", kind="function")
        outline = searcher.get_file_outline("widget.ts")
        names = [s.name for s in outline]
        assert "top_fn" in names and "Foo" in names
        # find_symbol parsed the file; the outline must reuse that parse.
        assert len(calls) == 1

    def test_ts_outline_before_find_symbol_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _ts_counting_searcher(tmp_path, monkeypatch)
        outline = searcher.get_file_outline("widget.ts")
        assert [s.name for s in outline] == ["Item", "LIMIT", "top_fn", "Base", "Foo"]
        assert searcher.find_symbol("top_fn", kind="function")
        # The outline warmed the cache; find_symbol must reuse it.
        assert len(calls) == 1

    def test_ts_repeated_outline_parses_once(self, tmp_path, monkeypatch):
        searcher, calls = _ts_counting_searcher(tmp_path, monkeypatch)
        searcher.get_file_outline("widget.ts")
        searcher.get_file_outline("widget.ts")
        searcher.get_file_outline("widget.ts")
        assert len(calls) == 1

    def test_ts_outline_is_top_level_only(self, tmp_path):
        searcher = _ts_searcher(tmp_path)
        outline = searcher.get_file_outline("widget.ts")
        by_name = {s.name: s for s in outline}
        assert set(by_name) == {"Item", "LIMIT", "top_fn", "Base", "Foo"}
        # methods stay out of the outline
        assert "bar" not in by_name

    def test_ts_outline_sorted_by_line_with_end_line(self, tmp_path):
        searcher = _ts_searcher(tmp_path)
        outline = searcher.get_file_outline("widget.ts")
        lines = [s.line for s in outline]
        assert lines == sorted(lines)
        foo = next(s for s in outline if s.name == "Foo")
        assert foo.end_line is not None and foo.end_line > foo.line
        assert all(s.end_line for s in outline)

    def test_ts_outline_class_carries_bases(self, tmp_path):
        """Deliberate additive change vs the former dedicated walk: class
        outline entries now expose bases, matching find_symbol."""
        searcher = _ts_searcher(tmp_path)
        by_name = {s.name: s for s in searcher.get_file_outline("widget.ts")}
        assert by_name["Foo"].bases == ["Base"]
        assert by_name["Base"].bases is None

    def test_ts_outline_matches_find_symbol_fields(self, tmp_path):
        """The map is the single source of truth — outline entries are the
        same SymbolDef objects find_symbol returns for the same symbols."""
        searcher = _ts_searcher(tmp_path)
        outline = {s.name: s for s in searcher.get_file_outline("widget.ts")}
        foo = searcher.find_symbol("Foo", kind="class")[0]
        assert outline["Foo"] is foo
        top_fn = searcher.find_symbol("top_fn", kind="function")[0]
        assert outline["top_fn"].signature == top_fn.signature
        assert outline["top_fn"].end_line == top_fn.end_line
