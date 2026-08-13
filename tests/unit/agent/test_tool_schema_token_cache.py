"""Guard against id()-reuse poisoning of ``_tool_schema_token_cache``.

The cache was keyed on ``id(tool_schemas)``; when a freed small schema list's id
was reused by a freshly-allocated large schema list, the large list inherited
the small list's (10x+ smaller) token count -- an under-count that inflates
``context_message_cap`` and risks prompt overflow. The key is now a content
fingerprint ``(len, names)`` which is immune to GC address reuse.
"""
from __future__ import annotations

from external_llm.agent import _shared_utils as su


def _clear_cache():
    su._tool_schema_token_cache.clear()


def test_cache_key_is_content_fingerprint_not_id():
    """Mutation guard: revert the key to ``id(tool_schemas)`` -> the key becomes
    an int and this FAILS."""
    _clear_cache()
    su.estimate_tokens_from_tool_schemas([{"name": "x"}, {"name": "y"}])
    keys = list(su._tool_schema_token_cache.keys())
    assert len(keys) == 1
    fp = keys[0]
    assert isinstance(fp, tuple)
    assert fp[0] == 2  # length component
    assert fp[1] == ("x", "y")  # names tuple


def test_fingerprint_distinguishes_content():
    _clear_cache()
    small_fp = su._tool_schema_fingerprint([{"name": "a"}, {"name": "b"}])
    big_fp = su._tool_schema_fingerprint([{"name": f"t{i}"} for i in range(40)])
    assert small_fp != big_fp
    assert big_fp[0] == 40


def test_distinct_content_produces_distinct_estimates():
    _clear_cache()
    small = [{"name": "a"}, {"name": "b"}]
    big = [{"name": f"t{i}", "description": "z" * 200} for i in range(40)]
    assert su.estimate_tokens_from_tool_schemas(big) > su.estimate_tokens_from_tool_schemas(small) * 5


def test_same_content_shares_cache_entry():
    """Structurally-equal fresh lists share a cache entry (the perf reason the
    cache exists for the lang_filter path, where each call returns a new list)."""
    _clear_cache()
    su.estimate_tokens_from_tool_schemas([{"name": "x"}, {"name": "y"}])
    before = len(su._tool_schema_token_cache)
    su.estimate_tokens_from_tool_schemas([{"name": "x"}, {"name": "y"}])  # new object, same names
    assert len(su._tool_schema_token_cache) == before


def test_openai_style_function_name_wrappers():
    """OpenAI tool schemas wrap the name under ``function.name``; the fingerprint
    must still extract it so the cache hits."""
    _clear_cache()
    su.estimate_tokens_from_tool_schemas(
        [{"function": {"name": "wrapped"}}, {"name": "flat"}])
    fp = next(iter(su._tool_schema_token_cache.keys()))
    assert fp[1] == ("wrapped", "flat")


def test_capped_put_refreshes_insertion_order_lru():
    """_shared_utils._capped_put (walk caches + git snapshot cache) must treat
    a re-inserted key as recently-used; before the fix a hot repo refreshed
    past its TTL stayed at the front of the dict and was the first eviction
    candidate once the cap was exceeded."""
    cache: dict = {}
    for i in range(4):
        su._capped_put(cache, f"r{i}", i, cap=4)
    for _ in range(50):
        su._capped_put(cache, "r0", 0, cap=4)
    su._capped_put(cache, "new", 99, cap=4)
    assert "r0" in cache and len(cache) == 4
    assert "r1" not in cache


def test_cjk_description_not_undercounted():
    """Tool-schema descriptions must be counted by the CJK-aware byte
    estimator: the old chars/3 formula counted a 3-byte Korean char as ~1/3
    token — the exact under-count this subsystem exists to prevent."""
    _clear_cache()
    _text = "한국어 설명" * 10
    # Distinct names: the content fingerprint keys on names only, so same-name
    # schemas would share a cache entry and mask the estimator difference.
    cjk = su.estimate_tokens_from_tool_schemas([{"name": "t_cjk", "description": _text}])
    ascii_ = su.estimate_tokens_from_tool_schemas(
        [{"name": "t_ascii", "description": "x" * len(_text)}])
    assert cjk > ascii_, "CJK (3 bytes/char) must cost MORE than same-length ASCII"
