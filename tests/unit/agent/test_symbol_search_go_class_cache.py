"""Go dotted-name resolution must reuse a per-file class→methods map.

``_find_in_go`` used to read the file and run a tree-sitter walk
(``find_class_methods``) on every dotted lookup such as
``find_symbol("TodoList.Add")``.  It now goes through
``_go_class_methods_map`` / ``_go_file_cache`` — the same signature-keyed
per-file pattern as ``_python_symbol_map`` and ``_ts_module_map`` — so N
lookups on one file parse it once, and ``invalidate_file_caches`` drops the
map after the agent's own writes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from external_llm.agent.symbol_search import SymbolSearcher
from external_llm.languages import tree_sitter_utils as _tsu
from external_llm.languages.go_provider import GoSyntaxProvider

_GO_SRC = textwrap.dedent("""\
    package main

    type TodoList struct {
        Items []string
    }

    func (l *TodoList) Add(item string) error {
        l.Items = append(l.Items, item)
        return nil
    }

    func (l *TodoList) Remove(i int) {
        _ = i
    }

    type Server struct {
        Port int
    }

    func (s *Server) Start() error {
        return nil
    }

    func NewServer(port int) *Server {
        return &Server{Port: port}
    }
""")


def _ts_grammar_available(lang: str) -> bool:
    """True when the tree-sitter binding for ``lang`` is installed."""
    try:
        from external_llm.languages.tree_sitter_utils import get_available_languages

        return lang in get_available_languages()
    except Exception:
        return False


def _go_repo(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.go").write_text(_GO_SRC, encoding="utf-8")
    return repo


def _counting_searcher(tmp_path, monkeypatch):
    """Searcher whose batch extractor records every call (parse-count pin)."""
    searcher = SymbolSearcher(str(_go_repo(tmp_path)))
    calls: list[int] = []
    _orig = _tsu.extract_all_class_methods

    def _counting(code, language):
        calls.append(1)
        return _orig(code, language)

    # The Go provider imports extract_all_class_methods at call time, so
    # patching the module attribute counts every provider-side batch parse.
    monkeypatch.setattr(_tsu, "extract_all_class_methods", _counting)
    return searcher, calls


@pytest.mark.skipif(not _ts_grammar_available("go"), reason="tree-sitter-go not installed")
class TestGoClassMethodsCache:
    def test_two_dotted_lookups_parse_file_once(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        f = searcher.repo_root / "server.go"
        add = searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        remove = searcher._find_in_go(f, "Remove", "any", parent_class="TodoList")
        assert [d.line for d in add] == [7]
        assert [d.line for d in remove] == [12]
        assert [d.name for d in add + remove] == ["Add", "Remove"]
        # The batch map was built once; the second lookup hit _go_file_cache.
        assert len(calls) == 1

    def test_find_symbol_dotted_go_uses_cache(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        r1 = searcher.find_symbol("TodoList.Add", kind="method", search_path="server.go")
        r2 = searcher.find_symbol("Server.Start", kind="method", search_path="server.go")
        assert [(d.name, d.parent_class, d.line) for d in r1] == [("Add", "TodoList", 7)]
        assert [(d.name, d.parent_class, d.line) for d in r2] == [("Start", "Server", 20)]
        assert r1[0].signature == "func (l *TodoList) Add(item string) error {"
        assert r1[0].end_line == 10
        # Two different structs, one file → one batch parse for both.
        assert len(calls) == 1

    def test_signature_line_read_is_windowed(self, tmp_path, monkeypatch):
        # P26-3: _find_in_go used to re-read the WHOLE file for one signature
        # line, so every dotted lookup after the class-methods cache warmed
        # still materialized the file — huge generated Go files (e.g.
        # *.pb.go) paid a full read per lookup. The signature line must come
        # from a line-window read: with the map cache warm, a lookup performs
        # ZERO Path.read_text calls.
        repo = _go_repo(tmp_path)
        searcher = SymbolSearcher(str(repo))
        f = repo / "server.go"

        reads: list[str] = []
        _real_read_text = Path.read_text

        def _counting(self, *args, **kwargs):
            reads.append(str(self))
            return _real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _counting)

        first = searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        assert first and first[0].signature == "func (l *TodoList) Add(item string) error {"
        reads.clear()
        second = searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        assert second and second[0].signature == "func (l *TodoList) Add(item string) error {"
        # Red before P26-3: the second lookup re-read the whole file.
        assert reads == []

    def test_edit_refreshes_signature(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        f = searcher.repo_root / "server.go"
        searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        # Content + size change → (mtime_ns, size) signature no longer matches.
        f.write_text(_GO_SRC + "\n\nfunc (l *TodoList) Clear() {}\n", encoding="utf-8")
        found = searcher._find_in_go(f, "Clear", "any", parent_class="TodoList")
        assert [d.line for d in found] == [29]
        assert len(calls) == 2

    def test_invalidate_file_caches_drops_go_entry(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        f = searcher.repo_root / "server.go"
        searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        # repo-relative spelling, exactly like the post-write invalidation path
        searcher.invalidate_file_caches(["server.go"])
        searcher._find_in_go(f, "Remove", "any", parent_class="TodoList")
        assert len(calls) == 2

    def test_invalidate_file_caches_none_clears_go(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        f = searcher.repo_root / "server.go"
        searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        searcher.invalidate_file_caches()  # unknown scope: full clear
        searcher._find_in_go(f, "Add", "any", parent_class="TodoList")
        assert len(calls) == 2

    def test_unknown_class_lookup_is_cheap_dict_get(self, tmp_path, monkeypatch):
        searcher, calls = _counting_searcher(tmp_path, monkeypatch)
        f = searcher.repo_root / "server.go"
        assert searcher._find_in_go(f, "Nope", "any", parent_class="Missing") == []
        assert searcher._find_in_go(f, "Nope", "any", parent_class="Missing") == []
        # A missing class is a dict miss on the cached map — no re-parse.
        assert len(calls) == 1

    def test_read_error_returns_empty(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        searcher = SymbolSearcher(str(repo))
        missing = repo / "nope.go"
        assert searcher._find_in_go(missing, "Add", "any", parent_class="TodoList") == []


class TestGoProviderBatchParity:
    def test_find_class_methods_delegates_to_batch(self):
        provider = GoSyntaxProvider()
        by_class = provider.find_all_class_methods(_GO_SRC)
        assert set(by_class) == {"TodoList", "Server"}
        assert provider.find_class_methods(_GO_SRC, "TodoList") == by_class["TodoList"]
        assert provider.find_class_methods(_GO_SRC, "Server") == by_class["Server"]
        assert provider.find_class_methods(_GO_SRC, "Missing") == []

    @pytest.mark.skipif(not _ts_grammar_available("go"), reason="tree-sitter-go not installed")
    def test_regex_fallback_matches_tree_sitter(self, monkeypatch):
        provider = GoSyntaxProvider()
        expected = provider.find_all_class_methods(_GO_SRC)  # tree-sitter path
        monkeypatch.setattr(_tsu, "is_available", lambda: False)
        fallback = provider.find_all_class_methods(_GO_SRC)
        assert fallback == expected

    @pytest.mark.skipif(not _ts_grammar_available("go"), reason="tree-sitter-go not installed")
    def test_extract_all_class_methods_matches_per_class_extract(self):
        grouped = _tsu.extract_all_class_methods(_GO_SRC, "go")
        assert grouped is not None
        for cls, methods in grouped.items():
            assert _tsu.extract_class_methods(_GO_SRC, cls, "go") == methods
