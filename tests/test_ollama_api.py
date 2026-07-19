"""Tests for external_llm.ollama_api — query_ollama_num_ctx + TTL-bounded cache.

Covers:
  * TTL-bounded caching (the stale-cache fix): an expired entry is re-queried
    so a Modelfile num_ctx change is eventually reflected instead of being
    served stale until process restart.
  * Fresh-cache hits (within TTL) avoid re-querying the server.
  * Connection / timeout / HTTP failures never poison the cache.
  * Ollama-format model-name guard.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

import external_llm.ollama_api as ollama_api
from external_llm.ollama_api import query_ollama_num_ctx

_TEST_URL = "http://test-ollama:11434"
_MODEL = "llama3:8b"


def _ok_resp(payload: dict) -> MagicMock:
    """Build a mock /api/show response that raise_for_status() treats as 2xx."""
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = payload
    return r


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate each test from prior cache state (manual dict is module-global)."""
    ollama_api._num_ctx_cache.clear()
    yield
    ollama_api._num_ctx_cache.clear()


def _age_past_ttl() -> None:
    """Rewind the cached entry's timestamp so the next read treats it as expired.

    Deterministic (no sleep): subtracts (TTL + 1) seconds from the stored
    monotonic timestamp, guaranteeing the freshness check fails.
    """
    key = (_MODEL, _TEST_URL)
    val, ts = ollama_api._num_ctx_cache[key]
    ollama_api._num_ctx_cache[key] = (val, ts - ollama_api._NUM_CTX_CACHE_TTL_SECONDS - 1)


# ── TTL-bounded cache (the fix) ────────────────────────────────────────────

class TestTTLCache:
    @patch("external_llm.ollama_api.requests.post")
    def test_fresh_hit_within_ttl_avoids_requery(self, mock_post):
        """Within TTL, the second call serves the cache without hitting the server."""
        mock_post.return_value = _ok_resp({"parameters": {"num_ctx": 4096}})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 4096
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 4096
        assert mock_post.call_count == 1

    @patch("external_llm.ollama_api.requests.post")
    def test_none_result_cached_within_ttl(self, mock_post):
        """A successful-but-absent num_ctx (None) is cached within the TTL window."""
        mock_post.return_value = _ok_resp({"parameters": {}, "modelfile": ""})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        # Second call served from cache (no re-query)
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 1

    @patch("external_llm.ollama_api.requests.post")
    def test_expired_entry_is_requeried(self, mock_post):
        """Past TTL, the cached value is ignored and the server is re-queried."""
        mock_post.return_value = _ok_resp({"parameters": {"num_ctx": 4096}})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 4096
        _age_past_ttl()
        mock_post.return_value = _ok_resp({"parameters": {"num_ctx": 8192}})
        # Expired → re-query → new value reflected
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 8192
        assert mock_post.call_count == 2

    @patch("external_llm.ollama_api.requests.post")
    def test_expired_none_picks_up_modelfile_addition(self, mock_post):
        """Cached None expires; a later num_ctx addition is reflected."""
        mock_post.return_value = _ok_resp({"parameters": {}, "modelfile": ""})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        _age_past_ttl()
        mock_post.return_value = _ok_resp({"parameters": {"num_ctx": 32768}})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 32768
        assert mock_post.call_count == 2

    @patch("external_llm.ollama_api.requests.post")
    def test_modelfile_text_path_caches(self, mock_post):
        """Priority-2 (modelfile PARAMETER text) result is cached within TTL."""
        modelfile = "# Modelfile\nPARAMETER num_ctx 16384\n"
        mock_post.return_value = _ok_resp({"modelfile": modelfile})
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 16384
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) == 16384
        assert mock_post.call_count == 1

    @patch("external_llm.ollama_api.requests.post")
    def test_cache_key_includes_base_url(self, mock_post):
        """Different base_url → distinct cache entries → two queries."""
        mock_post.return_value = _ok_resp({"parameters": {"num_ctx": 4096}})
        query_ollama_num_ctx(_MODEL, base_url_hint="http://a:11434")
        query_ollama_num_ctx(_MODEL, base_url_hint="http://b:11434")
        assert mock_post.call_count == 2

    def test_default_ttl_is_300(self):
        """Sanity anchor: documented TTL window."""
        assert ollama_api._NUM_CTX_CACHE_TTL_SECONDS == 300


# ── Failure paths never poison the cache ───────────────────────────────────

class TestNoCachePoisoning:
    @patch("external_llm.ollama_api.requests.post")
    def test_connection_error_not_cached(self, mock_post):
        """ConnectionError returns None but is NOT cached → next call re-queries."""
        mock_post.side_effect = requests.ConnectionError("refused")
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 2

    @patch("external_llm.ollama_api.requests.post")
    def test_timeout_not_cached(self, mock_post):
        mock_post.side_effect = requests.Timeout("slow")
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 2

    @patch("external_llm.ollama_api.requests.post")
    def test_http_404_not_cached(self, mock_post):
        err = requests.HTTPError(response=MagicMock(status_code=404))
        mock_post.side_effect = err
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 2

    @patch("external_llm.ollama_api.requests.post")
    def test_generic_exception_not_cached(self, mock_post):
        """The broad `except Exception` still returns None without poisoning cache."""
        mock_post.side_effect = ValueError("boom")
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert query_ollama_num_ctx(_MODEL, base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 2


# ── Ollama-format model-name guard ─────────────────────────────────────────

class TestModelNameGuard:
    @patch("external_llm.ollama_api.requests.post")
    def test_no_colon_never_queries(self, mock_post):
        """A model name without a colon tag is not an Ollama model → no query."""
        assert query_ollama_num_ctx("gpt-4o", base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 0

    @patch("external_llm.ollama_api.requests.post")
    def test_slash_never_queries(self, mock_post):
        """OpenRouter-style 'org/model' is skipped even if a colon is present."""
        assert query_ollama_num_ctx("qwen/qwen3:8b", base_url_hint=_TEST_URL) is None
        assert mock_post.call_count == 0


# ── _num_ctx_for_model priority resolution (OllamaClient) ──────────────────

from external_llm.providers import OllamaClient


class TestNumCtxForModelFallback:
    """Pin _num_ctx_for_model's priority chain and the flat 8192 floor.

    Regression guard: the floor MUST be 8192 for every model — including large
    tags like 'qwen3:99b' — because asicode's system prefix (core_prompt +
    project.md + design_insights ≈ 5272 tokens, measured via _cjk_aware_tokens)
    overflows Ollama's 4096 default. A size-based '13B+ -> 4096' tier is NOT
    viable and must never be reintroduced: 4096 < 5272 → asicode 400s on its own
    system prompt before any user content.
    """

    def _client(self):
        return OllamaClient(api_key=None, base_url=_TEST_URL, timeout=10)

    @patch("external_llm.ollama_api.query_ollama_num_ctx", return_value=None)
    @patch("external_llm.model_registry.get_ollama_num_ctx", return_value=None)
    def test_flat_8192_floor_even_for_huge_model_tag(self, _reg, _api):
        """A '99b' tag (far beyond any size tier) still returns 8192, never 4096."""
        assert self._client()._num_ctx_for_model("qwen3:99b") == 8192

    @patch("external_llm.ollama_api.query_ollama_num_ctx", return_value=None)
    @patch("external_llm.model_registry.get_ollama_num_ctx", return_value=None)
    def test_floor_is_universal_not_size_based(self, _reg, _api):
        """Small tags also get 8192 — the floor applies to every model, not '<8B'."""
        assert self._client()._num_ctx_for_model("qwen3:1.7b") == 8192

    @patch("external_llm.ollama_api.query_ollama_num_ctx", return_value=32768)
    def test_priority0_modelfile_value_wins(self, _api):
        """Priority 0 (/api/show Modelfile value) overrides the 8192 floor."""
        assert self._client()._num_ctx_for_model("bonsai27b") == 32768

    @patch("external_llm.ollama_api.query_ollama_num_ctx", return_value=None)
    @patch("external_llm.model_registry.get_ollama_num_ctx", return_value=6144)
    def test_priority1_registry_wins_over_floor(self, _reg, _api):
        """Priority 1 (OLLAMA_NUM_CTX_OVERRIDES) beats the 8192 floor."""
        assert self._client()._num_ctx_for_model("gemma:e2b") == 6144
