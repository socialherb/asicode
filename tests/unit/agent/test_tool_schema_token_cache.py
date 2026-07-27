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
    fp = list(su._tool_schema_token_cache.keys())[0]
    assert fp[1] == ("wrapped", "flat")
