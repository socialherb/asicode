"""Per-file symbol caches must not serve pre-edit line numbers.

``SymbolSearcher._py_file_cache`` / ``_ts_file_cache`` memoise the full symbol
map of one file. They used to key on a bare ``st_mtime`` and were absent from
the post-write invalidation path entirely, so a file rewritten without a
visible stat change kept answering from cache — ``find_symbol`` then reported
the symbol's OLD line, and a newly added symbol was invisible. Those line
numbers feed the edit tools.

The stat collision is simulated with ``os.utime`` rather than raced: a
coarse-mtime filesystem (container bind mount, NFS/SMB) or an mtime-preserving
restore (tar, ``rsync -t``, ``cp -p``) produces exactly this state, and pinning
it is the only way to test the axis deterministically on a ns-resolution FS.

Two independent layers, tested separately because they fail differently:

* the ``(st_mtime_ns, st_size)`` signature — catches any edit that changes size;
* :meth:`SymbolSearcher.invalidate_file_caches` — catches the rest, including a
  same-size edit, and needs no filesystem assumptions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from external_llm.agent.symbol_search import SymbolSearcher
from external_llm.agent.tool_registry import AgentConfig, ToolRegistry

PINNED = 1_700_000_000

# Size-CHANGING edit: `target` moves 5 -> 9 and `inserted` appears.
V1 = 'def alpha():\n    return 1\n\n\ndef target():\n    return "ORIGINAL"\n'
V2 = 'def alpha():\n    return 1\n\n\ndef inserted():\n    return 0\n\n\ndef target():\n    return "EDITED"\n'

# Same-SIZE edit: byte-identical length, `target` moves 1 -> 5. The signature
# cannot see this one at all; only explicit invalidation can.
S1 = "def target():\n    return 1\n\n\ndef zzzzzz():\n    pass\n"
S2 = "def zzzzzz():\n    pass\n\n\ndef target():\n    return 1\n"


def _pin(p: Path) -> None:
    os.utime(p, (PINNED, PINNED))


def _write_pinned(p: Path, text: str) -> None:
    p.write_text(text)
    _pin(p)


def _line_of(s: SymbolSearcher, f: Path, name: str = "target") -> int | None:
    defs = s._find_in_python_cached(f, name, "function")
    return defs[0].line if defs else None


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


class TestSignatureLayer:
    """(st_mtime_ns, st_size) — the belt."""

    def test_size_changing_edit_under_pinned_mtime_is_seen(self, repo: Path):
        f = repo / "m.py"
        _write_pinned(f, V1)
        s = SymbolSearcher(str(repo))
        assert _line_of(s, f) == 5

        _write_pinned(f, V2)  # identical st_mtime, different st_size
        assert f.stat().st_mtime == PINNED

        assert _line_of(s, f) == 9, "stale line number served from cache"
        assert s._find_in_python_cached(f, "inserted", "function"), "symbol added by the edit is invisible"

    def test_same_size_edit_defeats_the_signature(self, repo: Path):
        """Pins WHY invalidate_file_caches has to exist."""
        f = repo / "m.py"
        _write_pinned(f, S1)
        s = SymbolSearcher(str(repo))
        assert _line_of(s, f) == 1
        before = f.stat()

        _write_pinned(f, S2)
        after = f.stat()
        assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
        # Signature is powerless here — the cache is knowingly stale.
        assert _line_of(s, f) == 1


class TestInvalidateFileCaches:
    """Explicit post-write drop — the suspenders, no FS assumptions."""

    @pytest.fixture()
    def stale(self, repo: Path):
        f = repo / "m.py"
        _write_pinned(f, S1)
        s = SymbolSearcher(str(repo))
        assert _line_of(s, f) == 1  # warm
        _write_pinned(f, S2)
        return s, f

    def test_absolute_path(self, stale):
        """The _snapshot_target_files convention."""
        s, f = stale
        s.invalidate_file_caches([str(f)])
        assert _line_of(s, f) == 5

    def test_repo_relative_path(self, stale):
        """The patch-mixin touched/written convention."""
        s, f = stale
        s.invalidate_file_caches(["m.py"])
        assert _line_of(s, f) == 5

    def test_no_args_clears_wholesale(self, stale):
        """The unknown-scope (bash) path."""
        s, f = stale
        s.invalidate_file_caches()
        assert _line_of(s, f) == 5

    def test_unrelated_path_leaves_cache_intact(self, stale):
        """Scoping is real — an unrelated write must not evict."""
        s, f = stale
        s.invalidate_file_caches(["somewhere/else.py"])
        assert _line_of(s, f) == 1

    def test_empty_list_is_not_a_wholesale_clear(self, stale):
        """An empty list means 'nothing was touched', not 'drop everything'."""
        s, f = stale
        s.invalidate_file_caches([])
        assert _line_of(s, f) == 1


class TestPostWriteWiring:
    """The caches must actually be reached from the write path.

    The bug was not a wrong key so much as a missing registration: every other
    cache was dropped after a write and these two were not. A green unit test
    on invalidate_file_caches would still ship the bug if nobody called it.
    """

    def test_invalidate_cache_after_write_drops_per_file_maps(self, repo: Path):
        f = repo / "m.py"
        _write_pinned(f, S1)
        reg = ToolRegistry(str(repo), AgentConfig(rag_enabled=False))
        s = reg._symbol_searcher
        assert _line_of(s, f) == 1
        assert s._py_file_cache, "precondition: cache warmed"

        _write_pinned(f, S2)
        reg._invalidate_cache_after_write([str(f)])

        assert _line_of(s, f) == 5

    def test_unknown_scope_invalidation_drops_per_file_maps(self, repo: Path):
        """bash can write anything — the wholesale path must cover these too."""
        f = repo / "m.py"
        _write_pinned(f, S1)
        reg = ToolRegistry(str(repo), AgentConfig(rag_enabled=False))
        s = reg._symbol_searcher
        assert _line_of(s, f) == 1

        _write_pinned(f, S2)
        reg._invalidate_caches_unknown_scope()

        assert _line_of(s, f) == 5
