"""The agent must be able to tell 'no relevant docs (complete index)' from
'index truncated — results may be missing'.

``_tool_find_relevant_files`` reads ``RAGSearcher.index_truncated`` after the
search and surfaces it in both the rendered content (a grep/glob fallback hint)
and the structured metadata. Without this, a query whose true match sits beyond
the RAG file cap returns an answer identical to one against a small,
fully-indexed repo — actively misleading the agent into concluding the target
does not exist.
"""
from __future__ import annotations

from external_llm.agent.rag_searcher import SearchResult


def test_empty_results_with_truncated_index_advises_fallback(tool_registry, monkeypatch):
    s = tool_registry._rag_searcher
    monkeypatch.setattr(s, "find_relevant_files", lambda *a, **k: [])
    monkeypatch.setattr(s, "_index_truncated", True)

    result = tool_registry.dispatch("find_relevant_files", {"query": "widget"})

    assert result.ok
    assert (result.metadata or {}).get("index_truncated") is True
    # "No relevant files found" alone reads as 'searched, absent'. The hint must
    # nudge toward a full-repo fallback the cap does not bind.
    low = result.content.lower()
    assert ("grep" in low or "glob" in low), result.content
    assert ("incomplete" in low or "cap" in low), result.content


def test_empty_results_without_truncation_is_a_clean_negative(tool_registry, monkeypatch):
    s = tool_registry._rag_searcher
    monkeypatch.setattr(s, "find_relevant_files", lambda *a, **k: [])
    monkeypatch.setattr(s, "_index_truncated", False)

    result = tool_registry.dispatch("find_relevant_files", {"query": "widget"})

    assert result.ok
    assert (result.metadata or {}).get("index_truncated") is False
    low = result.content.lower()
    # A complete index with no hits is a clean negative — no fallback hint.
    assert "grep" not in low
    assert "No relevant files found" in result.content


def test_nonempty_results_with_truncation_appends_a_note(tool_registry, monkeypatch):
    s = tool_registry._rag_searcher
    fake = [SearchResult(file="src/widget.py", score=3.0, snippet="class Widget", line=10)]
    monkeypatch.setattr(s, "find_relevant_files", lambda *a, **k: fake)
    monkeypatch.setattr(s, "_index_truncated", True)

    result = tool_registry.dispatch("find_relevant_files", {"query": "widget"})

    assert result.ok
    assert (result.metadata or {}).get("index_truncated") is True
    assert "src/widget.py" in result.content
    low = result.content.lower()
    assert ("incomplete" in low or "cap" in low), result.content


def test_nonempty_results_without_truncation_has_no_note(tool_registry, monkeypatch):
    s = tool_registry._rag_searcher
    fake = [SearchResult(file="src/widget.py", score=3.0, snippet="class Widget", line=10)]
    monkeypatch.setattr(s, "find_relevant_files", lambda *a, **k: fake)
    monkeypatch.setattr(s, "_index_truncated", False)

    result = tool_registry.dispatch("find_relevant_files", {"query": "widget"})

    assert result.ok
    assert (result.metadata or {}).get("index_truncated") is False
    assert "src/widget.py" in result.content
    assert "incomplete" not in result.content.lower()
