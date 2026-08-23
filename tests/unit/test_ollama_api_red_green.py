"""RED→GREEN: external_llm/ollama_api.py — final missing branch.

query_ollama_capabilities must return None when the shared /api/show query
returns None (server unreachable / model not found / non-native tag).
"""

from __future__ import annotations

from external_llm import ollama_api


def test_capabilities_none_when_query_none(monkeypatch):
    monkeypatch.setattr(ollama_api, "_query_ollama_show", lambda *a, **k: None)
    assert ollama_api.query_ollama_capabilities("llama3:latest") is None


def test_capabilities_empty_list_returns_none(monkeypatch):
    monkeypatch.setattr(ollama_api, "_query_ollama_show", lambda *a, **k: {"capabilities": []})
    assert ollama_api.query_ollama_capabilities("llama3:latest") is None


def test_capabilities_tuple_extraction(monkeypatch):
    monkeypatch.setattr(
        ollama_api,
        "_query_ollama_show",
        lambda *a, **k: {"capabilities": ["completion", "tools", "vision"]},
    )
    assert ollama_api.query_ollama_capabilities("llama3:latest") == ("completion", "tools", "vision")


def test_capabilities_not_a_list(monkeypatch):
    monkeypatch.setattr(ollama_api, "_query_ollama_show", lambda *a, **k: {"capabilities": "tools"})
    assert ollama_api.query_ollama_capabilities("llama3:latest") is None


def test_show_cache_key_normalizes_model_name():
    assert ollama_api._show_cache_key("Llama3:Latest", "http://x") == ("llama3:latest", "http://x")
