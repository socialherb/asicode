"""@dataclass fallback must reuse the cached per-file symbol map.

``find_symbol("Foo.__init__")`` on a @dataclass class returns the class
definition when no explicit ``__init__`` exists.  The decorator check used
to re-read and re-parse the whole file (former ``_is_dataclass`` helper);
it now reads the class's ``SymbolDef.decorators`` from the same cached map
the lookup just built — so the parent file is parsed exactly once per
``find_symbol`` call.
"""
from __future__ import annotations

from external_llm.agent.symbol_search import SymbolSearcher

_DATACLASS_SRC = (
    "from dataclasses import dataclass\n"
    "@dataclass\n"
    "class Foo:\n"
    "    x: int = 1\n"
    "    def bar(self):\n"
    "        return self.x\n"
)


def _searcher(tmp_path) -> SymbolSearcher:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "widget.py").write_text(_DATACLASS_SRC)
    return SymbolSearcher(str(repo))


class TestDataclassFallbackUsesCache:
    def test_class_symbol_carries_decorators(self, tmp_path):
        searcher = _searcher(tmp_path)
        defs = searcher.find_symbol("Foo", kind="class")
        assert defs and defs[0].decorators == ["dataclass"]

    def test_dataclass_init_fallback_returns_class(self, tmp_path):
        searcher = _searcher(tmp_path)
        res = searcher.find_symbol("Foo.__init__")
        assert res and len(res) == 1
        assert res[0].kind == "class" and res[0].name == "Foo"

    def test_fallback_parses_parent_file_once(self, tmp_path, monkeypatch):
        searcher = _searcher(tmp_path)
        extracted: list[str] = []
        _orig = searcher._extract_all_python_symbols

        def _counting(file_path, rel):
            extracted.append(str(file_path))
            return _orig(file_path, rel)

        monkeypatch.setattr(searcher, "_extract_all_python_symbols", _counting)
        res = searcher.find_symbol("Foo.__init__")
        assert res and res[0].name == "Foo"
        # find_symbol("Foo") parses the file once; the __init__ lookup and
        # the @dataclass decorator check hit the cache — no second parse.
        assert extracted.count(str(searcher.repo_root / "widget.py")) == 1

    def test_plain_class_init_fallback_not_triggered(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "widget.py").write_text(
            "class Bar:\n    def method(self):\n        pass\n"
        )
        searcher = SymbolSearcher(str(repo))
        res = searcher.find_symbol("Bar.__init__")
        # No @dataclass → no fallback; __init__ does not exist.
        assert res == []

    def test_get_symbol_info_exposes_class_decorators(self, tmp_path):
        searcher = _searcher(tmp_path)
        info = searcher.get_symbol_info("Foo", file_path="widget.py")
        assert info is not None
        assert info.get("decorators") == ["dataclass"]
