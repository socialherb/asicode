"""Tests for external_llm/analysis/parse_cache.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from external_llm.analysis import parse_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts from the default cache state."""
    parse_cache.ensure_capacity(0)  # no-op, just to touch the module
    parse_cache.clear()
    # Restore default sizing so a prior test's growth doesn't leak.
    parse_cache._read_cached = parse_cache.lru_cache(
        maxsize=parse_cache._DEFAULT_CACHE_SIZE
    )(parse_cache._read_impl)
    parse_cache._ast_cached = parse_cache.lru_cache(
        maxsize=parse_cache._DEFAULT_CACHE_SIZE
    )(parse_cache._ast_impl)
    yield
    parse_cache.clear()


def _make_py(tmpdir: str, name: str, src: str) -> str:
    p = Path(tmpdir) / name
    p.write_text(src, encoding="utf-8")
    return str(p)


def test_parse_ast_caches_module():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "a.py", "x = 1\n")
        first = parse_cache.parse_ast(path)
        second = parse_cache.parse_ast(path)
        assert first is not None
        # Same object returned from cache (not re-parsed).
        assert first is second


def test_parse_ast_invalidates_on_edit():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "a.py", "x = 1\n")
        first = parse_cache.parse_ast(path)
        # Rewrite with different size -> stat key changes -> re-parse.
        Path(path).write_text("x = 1\ny = 2\n", encoding="utf-8")
        second = parse_cache.parse_ast(path)
        assert first is not second


def test_parse_ast_returns_none_on_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        path = _make_py(d, "bad.py", "def (:\n")
        assert parse_cache.parse_ast(path) is None


def test_parse_ast_returns_none_for_missing_file():
    assert parse_cache.parse_ast("/no/such/file/at/all.py") is None


def test_ensure_capacity_grows_cache():
    parse_cache.ensure_capacity(1000)
    info = parse_cache._ast_cached.cache_info()
    # 1000 + headroom, capped at _MAX_CACHE_SIZE.
    assert info.maxsize == min(1000 + parse_cache._CAPACITY_HEADROOM,
                               parse_cache._MAX_CACHE_SIZE)


def test_ensure_capacity_is_capped():
    parse_cache.ensure_capacity(10_000_000)
    assert parse_cache._ast_cached.cache_info().maxsize == parse_cache._MAX_CACHE_SIZE


def test_ensure_capacity_never_shrinks():
    parse_cache.ensure_capacity(500)
    big = parse_cache._ast_cached.cache_info().maxsize
    parse_cache.ensure_capacity(10)  # smaller request
    assert parse_cache._ast_cached.cache_info().maxsize == big


def test_grown_cache_holds_more_than_default_file_set():
    """The bug: a working set larger than the default size must survive a full
    pass so a second scanner over the same set hits the cache."""
    n = parse_cache._DEFAULT_CACHE_SIZE + 50
    with tempfile.TemporaryDirectory() as d:
        paths = [_make_py(d, f"m{i}.py", f"v{i} = {i}\n") for i in range(n)]
        parse_cache.ensure_capacity(n)
        trees = {p: parse_cache.parse_ast(p) for p in paths}
        # Second pass (simulating the next scanner) must return the SAME objects
        # for every file — i.e. nothing was evicted mid-pass.
        for p in paths:
            assert parse_cache.parse_ast(p) is trees[p]
