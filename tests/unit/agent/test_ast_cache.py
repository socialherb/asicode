"""Unit tests for the shared AST parse cache.

Validates three properties:
  1. Correctness — cached parse is ast.equivalent to a fresh parse.
  2. Cache behavior — repeated parses of the same content hit the cache.
  3. Error path — SyntaxError propagates / optional variant returns None.
"""
import ast

import pytest

from external_llm.agent import ast_cache


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    ast_cache.clear_cache()
    yield
    ast_cache.clear_cache()


class TestParseCached:
    def test_returns_ast_module(self):
        tree = ast_cache.parse_cached("def f(): return 1")
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == "f"

    def test_repeat_parse_hits_cache(self):
        content = "x = 1\ny = 2\n"
        ast_cache.parse_cached(content)
        ast_cache.parse_cached(content)
        ast_cache.parse_cached(content)
        info = ast_cache.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 1

    def test_distinct_content_distinct_entries(self):
        ast_cache.parse_cached("a = 1")
        ast_cache.parse_cached("b = 2")
        info = ast_cache.cache_info()
        assert info["misses"] == 2
        assert info["hits"] == 0
        assert info["currsize"] == 2

    def test_returns_same_object_on_hit(self):
        """Same reference on cache hit — callers must treat as read-only."""
        content = "def g(): pass"
        t1 = ast_cache.parse_cached(content)
        t2 = ast_cache.parse_cached(content)
        assert t1 is t2

    def test_syntax_error_raised(self):
        with pytest.raises(SyntaxError):
            ast_cache.parse_cached("def f(: bad")

    def test_syntax_error_not_cached(self):
        """Failed parses shouldn't occupy a cache slot."""
        try:
            ast_cache.parse_cached("def f(: bad")
        except SyntaxError:
            pass
        try:
            ast_cache.parse_cached("def f(: bad")
        except SyntaxError:
            pass
        info = ast_cache.cache_info()
        # Both miss; no hit because lru_cache does not cache exceptions
        assert info["hits"] == 0
        assert info["currsize"] == 0


class TestParseCachedOptional:
    def test_returns_tree_on_success(self):
        tree = ast_cache.parse_cached_optional("x = 1")
        assert isinstance(tree, ast.Module)

    def test_returns_none_on_syntax_error(self):
        tree = ast_cache.parse_cached_optional("def f(: bad")
        assert tree is None

    def test_shares_cache_with_strict(self):
        content = "pass"
        ast_cache.parse_cached(content)
        ast_cache.parse_cached_optional(content)
        info = ast_cache.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 1


class TestParseExprCached:
    def test_optional_returns_none_on_syntax_error(self):
        assert ast_cache.parse_expr_cached_optional("not an expr )") is None

    def test_optional_returns_tree_on_success(self):
        tree = ast_cache.parse_expr_cached_optional("func(1, 2)")
        assert isinstance(tree, ast.Expression)
        assert isinstance(tree.body, ast.Call)

    def test_repeat_parse_hits_expr_cache(self):
        content = "a > 0"
        ast_cache.parse_expr_cached_optional(content)
        ast_cache.parse_expr_cached_optional(content)
        info = ast_cache.cache_info()
        assert info["expr"]["hits"] == 1
        assert info["expr"]["misses"] == 1

    def test_expr_cache_independent_of_module_cache(self):
        """Same string parsed in module and expr mode uses separate slots."""
        ast_cache.parse_cached("x = 1")
        ast_cache.parse_expr_cached_optional("x + 1")
        info = ast_cache.cache_info()
        # Each cache saw exactly one miss, no hits
        assert info["module"]["misses"] == 1
        assert info["expr"]["misses"] == 1
        assert info["module"]["hits"] == 0
        assert info["expr"]["hits"] == 0


class TestCacheControl:
    def test_clear_cache_resets_stats(self):
        ast_cache.parse_cached("x = 1")
        ast_cache.parse_cached("x = 1")
        ast_cache.parse_expr_cached_optional("x + 1")
        assert ast_cache.cache_info()["hits"] == 1
        ast_cache.clear_cache()
        info = ast_cache.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["currsize"] == 0
        # Sub-cache stats also reset
        assert info["module"]["hits"] == 0
        assert info["expr"]["hits"] == 0

    def test_cache_info_exposes_maxsize(self):
        info = ast_cache.cache_info()
        # Aggregate maxsize covers both caches
        assert info["maxsize"] == ast_cache._CACHE_MAX * 2
        # Individual caps remain _CACHE_MAX
        assert info["module"]["maxsize"] == ast_cache._CACHE_MAX
        assert info["expr"]["maxsize"] == ast_cache._CACHE_MAX

    def test_bounded_eviction(self):
        """When capacity is exceeded, oldest entries are evicted (LRU)."""
        maxsize = ast_cache._CACHE_MAX
        # Fill cache past capacity with distinct contents
        for i in range(maxsize + 3):
            ast_cache.parse_cached(f"x_{i} = {i}")
        info = ast_cache.cache_info()
        assert info["module"]["currsize"] == maxsize
        # Earliest entries evicted — re-parsing them registers as miss
        ast_cache.parse_cached("x_0 = 0")
        new_info = ast_cache.cache_info()
        assert new_info["module"]["misses"] == info["module"]["misses"] + 1
