"""_index_via_treesitter_batch must stat-gate file sizes and cap cumulative bytes.

The batch walker reads + tree-sitter-parses every file ``rg --files`` returns
for a provider's globs.  Its siblings bound the indexable set
(``_NONPY_INPROC_MAX_FILES`` / ``_NONPY_INPROC_MAX_BYTES`` in
``_rg_token_in_nonpy_files``), but the batch loop had neither gate — a single
minified ``dist/*.js`` (tens of MB is common) was read and parsed in full.

P26-4 adds: a per-file stat gate (skip oversized), a cumulative byte budget
(stop when spent), and a file-count cap — all reusing the sibling constants.
"""

from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

import pytest

from external_llm.agent.symbol_search import (
    _NONPY_INPROC_MAX_BYTES,
    _NONPY_INPROC_MAX_FILES,
    SymbolSearcher,
)
from external_llm.languages import LanguageRegistry

_GO_SRC = textwrap.dedent("""\
    package main

    func (l *TodoList) Add(item string) error {
        return nil
    }
""")


def _ts_grammar_available(lang: str) -> bool:
    """True when the tree-sitter binding for ``lang`` is installed."""
    try:
        from external_llm.languages.tree_sitter_utils import get_available_languages

        return lang in get_available_languages()
    except Exception:
        return False


def _go_provider():
    provider = LanguageRegistry.instance().get("server.go")
    assert provider is not None
    return provider


def _fake_stat_size(monkeypatch, fake: dict[str, int]) -> None:
    """Path.stat with st_size overridden per file name (real fields otherwise)."""
    real_stat = Path.stat

    def _stat(self):
        st = real_stat(self)
        if self.name in fake:
            st = os.stat_result(
                (
                    st.st_mode,
                    st.st_ino,
                    st.st_dev,
                    st.st_nlink,
                    st.st_uid,
                    st.st_gid,
                    fake[self.name],
                    st.st_atime,
                    st.st_mtime,
                    st.st_ctime,
                )
            )
        return st

    monkeypatch.setattr(Path, "stat", _stat)


@pytest.mark.skipif(
    not _ts_grammar_available("go") or shutil.which("rg") is None,
    reason="tree-sitter-go grammar and/or rg not installed",
)
class TestTreesitterBatchBounds:
    def test_oversized_file_skipped_before_read(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "server.go").write_text(_GO_SRC, encoding="utf-8")
        (repo / "huge.go").write_text("package main\n", encoding="utf-8")
        searcher = SymbolSearcher(str(repo))
        _fake_stat_size(monkeypatch, {"huge.go": _NONPY_INPROC_MAX_BYTES + 1})

        reads: list[str] = []
        _real_read_text = Path.read_text

        def _counting(self, *args, **kwargs):
            reads.append(self.name)
            return _real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _counting)

        index: dict = {}
        seen: set = set()
        searcher._index_via_treesitter_batch([_go_provider()], repo, index, seen)
        # Red before P26-4: huge.go was read in full (no stat gate).
        assert reads == ["server.go"]

    def test_cumulative_byte_budget_stops_walk(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "server.go").write_text(_GO_SRC, encoding="utf-8")
        (repo / "extra.go").write_text("package main\n\nfunc Extra() int { return 1 }\n", encoding="utf-8")
        searcher = SymbolSearcher(str(repo))
        # Each file faked to ~4MiB+1: the per-file gate passes (4MiB+1 < 8MiB)
        # but the cumulative 8MiB budget breaks the walk after the first file.
        half = _NONPY_INPROC_MAX_BYTES // 2 + 1
        _fake_stat_size(monkeypatch, {"server.go": half, "extra.go": half})

        index: dict = {}
        seen: set = set()
        searcher._index_via_treesitter_batch([_go_provider()], repo, index, seen)
        files = {sym.file for syms in index.values() for sym in syms}
        # Red before P26-4: both files were read and indexed.
        assert len(files) == 1
        assert _NONPY_INPROC_MAX_FILES > 0  # keep the constant referenced
