"""The tree-sitter memo caches must be bounded by bytes, not just entry count.

``parse_to_tree`` and ``_encode_content`` key on the FULL source text, so an
lru_cache maxsize bounds how many sources are held, not how large they are — and
a parsed Tree costs roughly 19x its source. Scanning the 200 largest .py files
in this repo retained 89 MB under the old settings (64/128 entries, no size
gate) against 17 MB after, for the same workload.

The size gate trades memoisation of rare huge files for a predictable ceiling.
These tests pin the three things that trade must not break: the intra-pipeline
reuse the caches exist for, correctness above the gate, and the
``cache_clear()`` surface ``invalidate_caches()`` depends on for grammar
hot-reload.
"""

from __future__ import annotations

import pytest

from external_llm.languages import tree_sitter_utils as tsu
from external_llm.languages.tree_sitter_utils import (
    _MAX_CACHED_SOURCE_CHARS,
    _encode_content,
    is_available,
    parse_to_tree,
)

pytestmark = pytest.mark.skipif(not is_available(), reason="tree-sitter not installed")

_SMALL = "def f():\n    return 1\n"
# Comfortably over the gate without depending on its exact value.
_LARGE = "def f():\n    return 1\n\n\n" * ((_MAX_CACHED_SOURCE_CHARS // 24) + 64)


def _fresh() -> None:
    parse_to_tree.cache_clear()
    _encode_content.cache_clear()


class TestSizeGate:
    def test_large_source_is_over_the_gate(self):
        """Guard the fixture itself, so the tests below stay meaningful."""
        assert len(_SMALL) <= _MAX_CACHED_SOURCE_CHARS < len(_LARGE)

    def test_small_source_is_memoised(self):
        """The documented win: one source parsed by several helpers in a row."""
        _fresh()
        first = parse_to_tree(_SMALL, "python")
        second = parse_to_tree(_SMALL, "python")
        assert first is not None
        assert first is second
        assert parse_to_tree.cache_info().hits == 1

    def test_large_source_is_not_memoised(self):
        _fresh()
        parse_to_tree(_LARGE, "python")
        parse_to_tree(_LARGE, "python")
        info = parse_to_tree.cache_info()
        assert info.hits == 0
        assert info.currsize == 0, "an oversized source must not occupy a cache slot"

    def test_large_source_still_parses_correctly(self):
        """Bypassing the cache must change retention, never the result."""
        _fresh()
        first = parse_to_tree(_LARGE, "python")
        second = parse_to_tree(_LARGE, "python")
        assert first is not None and second is not None
        assert first is not second
        assert not first.root_node.has_error
        assert first.root_node.type == second.root_node.type
        assert first.root_node.end_byte == second.root_node.end_byte

    def test_a_large_source_does_not_evict_small_ones(self):
        _fresh()
        parse_to_tree(_SMALL, "python")
        parse_to_tree(_LARGE, "python")
        assert parse_to_tree(_SMALL, "python") is not None
        assert parse_to_tree.cache_info().hits == 1


class TestEncodeContent:
    @pytest.mark.parametrize("source", [_SMALL, _LARGE, "한글 주석 = 1\n"])
    def test_encoding_is_correct_on_both_sides_of_the_gate(self, source: str):
        assert _encode_content(source) == source.encode("utf-8")

    def test_small_encode_is_memoised(self):
        _fresh()
        _encode_content(_SMALL)
        _encode_content(_SMALL)
        assert _encode_content.cache_info().hits == 1

    def test_large_encode_is_not_memoised(self):
        _fresh()
        _encode_content(_LARGE)
        _encode_content(_LARGE)
        assert _encode_content.cache_info().currsize == 0


class TestCacheSurfaceContract:
    """invalidate_caches() calls parse_to_tree.cache_clear() so a late-installed
    grammar takes effect without a process restart. Wrapping parse_to_tree in a
    plain function would silently drop that attribute."""

    def test_public_names_keep_the_lru_cache_surface(self):
        for fn in (parse_to_tree, _encode_content):
            assert callable(getattr(fn, "cache_clear", None))
            assert callable(getattr(fn, "cache_info", None))

    def test_invalidate_caches_clears_the_parse_cache(self):
        _fresh()
        parse_to_tree(_SMALL, "python")
        assert parse_to_tree.cache_info().currsize == 1
        tsu.invalidate_caches()
        assert parse_to_tree.cache_info().currsize == 0
