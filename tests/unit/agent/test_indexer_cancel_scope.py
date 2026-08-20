"""Both repo indexers' cancel channel must observe the per-call scope.

``call_graph.CallGraphIndexer`` and ``rag_searcher.RAGSearcher`` share a
copy-paste twin ``_get_cancel_event`` that read only ``config.cancel_event``
(whole-turn ESC). A first build — the lazy cost of every graph query,
``analyze_change_impact`` and ``find_relevant_files`` — abandoned by ITS
caller (MCP ``wait_for`` timeout, aborted ``dispatch_parallel`` batch) sets a
per-call scope the build checkpoints never saw, so the walk ran to completion
while the abandoned dispatch's pool slot stayed occupied.
"""
from __future__ import annotations

import threading
import types

from external_llm.agent.call_graph import CallGraphIndexer
from external_llm.agent.cancel_scope import call_cancel_scope


def test_call_graph_cancel_event_merges_caller_scope(tmp_path):
    """Without config/scope → None (inert no-op, legacy behavior); under a
    scope → the scope itself; unwound → None again."""
    idx = CallGraphIndexer(str(tmp_path))
    assert idx._get_cancel_event() is None
    scope_ev = threading.Event()
    with call_cancel_scope(scope_ev):
        assert idx._get_cancel_event() is scope_ev
    assert idx._get_cancel_event() is None


def test_call_graph_cancel_event_composites_scope_with_config(tmp_path):
    """Scope + config events OR through the composite: a whole-turn ESC trips
    it just like a per-call abandon does."""
    cfg = types.SimpleNamespace(cancel_event=threading.Event())
    idx = CallGraphIndexer(str(tmp_path), config=cfg)
    scope_ev = threading.Event()
    with call_cancel_scope(scope_ev):
        merged = idx._get_cancel_event()
        assert merged is not None and not merged.is_set()
        cfg.cancel_event.set()
        assert merged.is_set()


def test_call_graph_ctor_event_keeps_precedence(tmp_path):
    """An explicit ctor event (tests / direct callers) stays the single
    source — the documented precedence is not widened by the scope merge."""
    explicit = threading.Event()
    idx = CallGraphIndexer(str(tmp_path), cancel_event=explicit)
    scope_ev = threading.Event()
    with call_cancel_scope(scope_ev):
        assert idx._get_cancel_event() is explicit


def test_call_graph_build_short_circuits_under_pre_set_scope(tmp_path):
    """Scope-driven mirror of the ctor-event short-circuit: a build whose
    caller already abandoned it must discard any partial index and leave
    ``_built`` False — no file is worth walking for a caller that left."""
    for i in range(4):
        (tmp_path / f"m{i}.py").write_text(
            f"def f{i}():\n    pass\n", encoding="utf-8"
        )
    idx = CallGraphIndexer(str(tmp_path))
    scope_ev = threading.Event()
    scope_ev.set()
    with call_cancel_scope(scope_ev):
        idx.build()
    assert idx._built is False
    assert len(idx._nodes) == 0
    assert len(idx._forward) == 0
    assert len(idx._reverse) == 0


def test_rag_cancel_event_merges_caller_scope(tmp_path):
    """The rag_searcher twin gets the same merge (first index build is the
    lazy cost of every ``find_relevant_files`` call)."""
    from external_llm.agent.rag_searcher import RAGSearcher

    rs = RAGSearcher(str(tmp_path))
    assert rs._get_cancel_event() is None
    scope_ev = threading.Event()
    with call_cancel_scope(scope_ev):
        assert rs._get_cancel_event() is scope_ev
    assert rs._get_cancel_event() is None

    cfg = types.SimpleNamespace(cancel_event=threading.Event())
    rs2 = RAGSearcher(str(tmp_path), config=cfg)
    with call_cancel_scope(scope_ev):
        merged = rs2._get_cancel_event()
        assert merged is not None and not merged.is_set()
        scope_ev.set()
        assert merged.is_set()
    scope_ev.clear()
