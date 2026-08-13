"""Ollama model-list cache contract (``repl_impl._get_ollama_models``).

Pins the TTL-cache semantics fixed in the repl_impl deep audit:

- The cache starts as ``None``, NOT ``[]``: a pre-populated empty list would
  pass the ``is not None`` TTL guard and make the FIRST call return no
  models whenever it happens within 10s of the monotonic reference point.
  The None sentinel means the first call ALWAYS fetches.
- A successful fetch is reused within the 10s TTL (subprocess runs once).
- A failure (binary missing / timeout) is cached as ``[]`` for the TTL (no
  hammering), and a call after TTL expiry retries.
- ``_list_provider_model_choices`` reuses the shared cached fetcher for its
  ollama section (single parse + shared TTL — no duplicate subprocess).
"""
from __future__ import annotations

import subprocess

import pytest

from external_llm.repl import repl_impl


@pytest.fixture(autouse=True)
def _isolate_ollama_cache():
    """Reset the module-level cache state around each test."""
    old_cache, old_ts = repl_impl._ollama_cache, repl_impl._ollama_cache_ts
    repl_impl._ollama_cache = None
    repl_impl._ollama_cache_ts = 0.0
    yield
    repl_impl._ollama_cache = old_cache
    repl_impl._ollama_cache_ts = old_ts


def _fake_ollama_list(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["ollama", "list"], returncode=returncode, stdout=stdout, stderr=""
    )


_LLAMA_OUT = "NAME\tID\tSIZE\nqwen2.5-coder:3b\tabc\t1.9GB\n"


def test_first_call_always_fetches(monkeypatch):
    """Cache starts None — the very first call must hit ollama, not the cache."""
    calls: list = []

    def _run(*a, **k):
        calls.append(1)
        return _fake_ollama_list(_LLAMA_OUT)

    monkeypatch.setattr(subprocess, "run", _run)
    assert repl_impl._get_ollama_models(timeout=5) == ["qwen2.5-coder:3b"]
    assert len(calls) == 1
    assert repl_impl._ollama_cache == ["qwen2.5-coder:3b"]


def test_ttl_reuses_cache_without_resubprocess(monkeypatch):
    """A second call within the 10s TTL returns the cached list (run once)."""
    calls: list = []

    def _run(*a, **k):
        calls.append(1)
        return _fake_ollama_list(_LLAMA_OUT)

    monkeypatch.setattr(subprocess, "run", _run)
    assert repl_impl._get_ollama_models() == ["qwen2.5-coder:3b"]
    assert repl_impl._get_ollama_models() == ["qwen2.5-coder:3b"]
    assert len(calls) == 1


def test_failure_cached_empty_then_retried_after_ttl(monkeypatch):
    """Failures cache as [] for the TTL; expiry triggers a retry."""
    calls: list = []

    def _run(*a, **k):
        calls.append(1)
        raise FileNotFoundError("ollama not installed")

    monkeypatch.setattr(subprocess, "run", _run)
    assert repl_impl._get_ollama_models() == []
    assert repl_impl._get_ollama_models() == []  # cached empty — no hammering
    assert len(calls) == 1
    repl_impl._ollama_cache_ts = -1000.0  # force TTL expiry
    assert repl_impl._get_ollama_models() == []
    assert len(calls) == 2


def test_provider_choices_reuse_shared_fetcher(monkeypatch):
    """_list_provider_model_choices appends ollama via the cached fetcher."""
    monkeypatch.setattr(
        repl_impl, "_get_ollama_models", lambda timeout=5: ["qwen2.5-coder:3b"]
    )
    choices = repl_impl._list_provider_model_choices()
    assert ("ollama", "qwen2.5-coder:3b") in choices
    assert any(prov != "ollama" for prov, _ in choices)  # known models kept


def test_empty_ollama_output_yields_no_choices(monkeypatch):
    monkeypatch.setattr(repl_impl, "_get_ollama_models", lambda timeout=5: [])
    choices = repl_impl._list_provider_model_choices()
    assert all(prov != "ollama" for prov, _ in choices)
