"""Tests for SpecGraphEnricher."""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

from external_llm.agent.execution_spec import ResolvedExecutionSpec
from external_llm.graph.models import CallEdge, SymbolNode
from external_llm.graph.run_scoped_graph_cache import RunScopedGraphCache
from external_llm.graph.spec_graph_enricher import SpecGraphEnricher

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_spec(**kwargs) -> ResolvedExecutionSpec:
    """Create a minimal ResolvedExecutionSpec for testing."""
    defaults = {
        "original_request": "test request",
        "intent": "feature",
        "request_type": "feature",
        "target_symbols": [],
        "target_files": [],
        "metadata": {},
    }
    defaults.update(kwargs)
    return ResolvedExecutionSpec(**defaults)


def _make_symbol_node(name="foo", qualname=None, file_path="src/foo.py", kind="function"):
    return SymbolNode(
        name=name,
        qualname=qualname or name,
        module="src.foo",
        file_path=file_path,
        kind=kind,
        start_line=1,
        end_line=10,
    )


def _make_call_edge(caller_symbol="bar", caller_file="src/bar.py", callee_symbol="foo", callee_file="src/foo.py"):
    return CallEdge(
        caller_symbol=caller_symbol,
        caller_file=caller_file,
        caller_line=5,
        callee_symbol=callee_symbol,
        callee_display=callee_symbol,
        callee_file=callee_file,
        confidence=1.0,
    )


def _make_facade(symbols=None, callers=None, callees=None, related=None, file_symbols=None):
    """Create a mock RepositoryGraphFacade."""
    facade = MagicMock()
    # symbols: dict name -> SymbolNode (or None)
    facade.get_symbol = MagicMock(
        side_effect=lambda name, *a, **kw: (symbols or {}).get(name)
    )
    facade.get_callers = MagicMock(return_value=callers or [])
    facade.get_callees = MagicMock(return_value=callees or [])
    facade.get_related_symbols = MagicMock(return_value=related or [])
    facade.get_symbols_in_file = MagicMock(return_value=file_symbols or [])
    return facade


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSpecGraphEnricher:

    def test_enrich_with_resolved_symbols(self):
        """target_symbols that resolve via graph get enriched."""
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert len(gc["resolved_symbols"]) == 1
        assert gc["resolved_symbols"][0]["name"] == "foo"
        assert gc["resolved_symbols"][0]["file_path"] == "src/foo.py"
        assert gc["unresolved_symbols"] == []
        assert "src/foo.py" in gc["primary_files"]

    def test_enrich_with_unresolved_symbols(self):
        """Unresolved symbols are recorded, don't crash."""
        facade = _make_facade(symbols={})  # nothing resolves
        spec = _make_spec(target_symbols=["ghost"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert gc["resolved_symbols"] == []
        assert "ghost" in gc["unresolved_symbols"]
        assert gc["graph_confidence"] == 0.0

    def test_enrich_mixed_resolved_unresolved(self):
        """Some symbols resolve, some don't. graph_confidence is partial."""
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})  # "bar" doesn't resolve
        spec = _make_spec(target_symbols=["foo", "bar"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert len(gc["resolved_symbols"]) == 1
        assert "bar" in gc["unresolved_symbols"]
        assert gc["graph_confidence"] == 0.5  # 1/2

    def test_enrich_no_target_symbols_uses_target_files(self):
        """No target_symbols → try target_files → get_symbols_in_file."""
        node = _make_symbol_node("helper", file_path="src/helper.py")
        facade = _make_facade(file_symbols=[node])
        spec = _make_spec(target_symbols=[], target_files=["src/helper.py"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        facade.get_symbols_in_file.assert_called_once_with("src/helper.py")
        gc = result.metadata["graph_context"]
        assert any(s["name"] == "helper" for s in gc["resolved_symbols"])

    def test_enrich_callers_callees_populated(self):
        """Resolved symbols get their callers/callees recorded."""
        node = _make_symbol_node("foo")
        caller_edge = _make_call_edge(caller_symbol="bar", caller_file="src/bar.py", callee_symbol="foo")
        callee_edge = _make_call_edge(caller_symbol="foo", caller_file="src/foo.py", callee_symbol="qux", callee_file="src/qux.py")
        facade = _make_facade(
            symbols={"foo": node},
            callers=[caller_edge],
            callees=[callee_edge],
        )
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert "foo" in gc["callers"]
        assert len(gc["callers"]["foo"]) == 1
        assert gc["callers"]["foo"][0]["symbol"] == "bar"
        assert "foo" in gc["callees"]
        # callee edge: caller_symbol="foo", callee_symbol="qux" → stored as callee_symbol
        assert gc["callees"]["foo"][0]["symbol"] == "qux"

    def test_enrich_impact_files_from_callers(self):
        """impact_files includes files of callers/callees."""
        node = _make_symbol_node("foo", file_path="src/foo.py")
        caller_edge = _make_call_edge(caller_symbol="bar", caller_file="src/bar.py", callee_symbol="foo")
        facade = _make_facade(symbols={"foo": node}, callers=[caller_edge])
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert "src/bar.py" in gc["impact_files"]
        # primary_files should not overlap with impact_files
        assert "src/foo.py" not in gc["impact_files"]

    def test_enrich_primary_files_from_resolved(self):
        """primary_files comes from resolved symbol file_paths."""
        node = _make_symbol_node("foo", file_path="src/foo.py")
        facade = _make_facade(symbols={"foo": node})
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert "src/foo.py" in gc["primary_files"]

    def test_enrich_graph_confidence_all_resolved(self):
        """All symbols resolved → confidence = 1.0."""
        nodes = {"foo": _make_symbol_node("foo"), "bar": _make_symbol_node("bar", file_path="src/bar.py")}
        facade = _make_facade(symbols=nodes)
        spec = _make_spec(target_symbols=["foo", "bar"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert gc["graph_confidence"] == 1.0

    def test_enrich_graph_confidence_none_resolved(self):
        """No symbols resolved → confidence = 0.0."""
        facade = _make_facade(symbols={})
        spec = _make_spec(target_symbols=["ghost1", "ghost2"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert gc["graph_confidence"] == 0.0

    def test_enrich_facade_exception_graceful(self):
        """Facade throws → spec returned unchanged (minimal graph_context), no crash."""
        facade = MagicMock()
        facade.get_symbol.side_effect = RuntimeError("db offline")

        spec = _make_spec(target_symbols=["foo"])
        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        # Must not crash; graph_context should exist (minimal fallback)
        assert "graph_context" in result.metadata

    def test_enrich_already_enriched_skips(self):
        """If graph_context already in metadata, skip re-enrichment."""
        existing_gc = {"graph_confidence": 0.99, "resolved_symbols": [{"name": "existing"}]}
        facade = _make_facade(symbols={"foo": _make_symbol_node("foo")})
        spec = _make_spec(
            target_symbols=["foo"],
            metadata={"graph_context": existing_gc},
        )

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        # Should not call get_symbol at all
        facade.get_symbol.assert_not_called()
        # graph_context unchanged
        assert result.metadata["graph_context"] is existing_gc

    def test_enrich_deduplicates_files(self):
        """Duplicate files in primary/impact are deduplicated."""
        node1 = _make_symbol_node("foo", file_path="src/foo.py")
        node2 = _make_symbol_node("foo2", file_path="src/foo.py")  # same file
        facade = _make_facade(symbols={"foo": node1, "foo2": node2})
        spec = _make_spec(target_symbols=["foo", "foo2"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert gc["primary_files"].count("src/foo.py") == 1

    def test_enrich_limits_related_symbols(self):
        """related_symbols capped at 20."""
        node = _make_symbol_node("foo")
        # Return 30 related symbols
        related = [{"symbol": f"sym_{i}", "file": f"src/f{i}.py", "kind": "function"} for i in range(30)]
        facade = _make_facade(symbols={"foo": node}, related=related)
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert len(gc["related_symbols"]) <= 20

    def test_enrich_limits_callers_callees_per_symbol(self):
        """callers/callees capped at 10 per symbol."""
        node = _make_symbol_node("foo")
        # 15 caller edges
        callers = [
            _make_call_edge(caller_symbol=f"caller_{i}", caller_file=f"src/c{i}.py", callee_symbol="foo")
            for i in range(15)
        ]
        facade = _make_facade(symbols={"foo": node}, callers=callers)
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert len(gc["callers"].get("foo", [])) <= 10

    def test_enrich_design_originated_spec(self):
        """Design-originated spec with target_symbols gets enriched."""
        node = _make_symbol_node("MyClass", kind="class", file_path="src/models.py")
        facade = _make_facade(symbols={"MyClass": node})
        spec = _make_spec(
            target_symbols=["MyClass"],
            metadata={"source": "design_spec", "design_origin": True},
        )

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert len(gc["resolved_symbols"]) == 1
        assert gc["resolved_symbols"][0]["kind"] == "class"
        # Other design metadata preserved
        assert result.metadata["source"] == "design_spec"

    def test_enrich_returns_same_spec_object(self):
        """enrich() returns the same spec object (mutated in place)."""
        facade = _make_facade()
        spec = _make_spec()

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        assert result is spec

    def test_enrich_empty_target_symbols_and_files(self):
        """Spec with no target_symbols and no target_files → minimal graph_context, no crash."""
        facade = _make_facade()
        spec = _make_spec(target_symbols=[], target_files=[])

        enricher = SpecGraphEnricher(facade)
        result = enricher.enrich(spec)

        gc = result.metadata["graph_context"]
        assert gc["resolved_symbols"] == []
        assert gc["unresolved_symbols"] == []
        assert gc["primary_files"] == []
        assert gc["impact_files"] == []
        # confidence defaults when no target_symbols (no division by 0)
        assert gc["graph_confidence"] == 0.0

    def test_enrich_logging_on_entry_exit(self):
        """Enricher logs INFO on entry and exit."""
        import logging
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        spec = _make_spec(target_symbols=["foo"])

        enricher = SpecGraphEnricher(facade)
        with contextlib.nullcontext():
            # Capture log records at INFO level
            import logging
            logger = logging.getLogger("external_llm.graph.spec_graph_enricher")
            with __import__("unittest.mock", fromlist=["patch"]).patch.object(logger, "info") as mock_info:
                enricher.enrich(spec)
                # Should have called info at least twice (entry + exit)
                assert mock_info.call_count >= 2
                calls_str = " ".join(str(c) for c in mock_info.call_args_list)
                assert "Starting graph enrichment" in calls_str
                assert "Graph enrichment complete" in calls_str

    def test_enrich_skip_logging_on_already_enriched(self):
        """When spec already has graph_context, enricher skips and logs debug."""
        import logging
        from unittest.mock import patch
        existing_gc = {"graph_confidence": 0.99, "resolved_symbols": []}
        facade = _make_facade()
        spec = _make_spec(metadata={"graph_context": existing_gc})

        enricher = SpecGraphEnricher(facade)
        logger = logging.getLogger("external_llm.graph.spec_graph_enricher")
        with patch.object(logger, "debug") as mock_debug:
            enricher.enrich(spec)
            # Should log debug about skipping
            calls_str = " ".join(str(c) for c in mock_debug.call_args_list)
            assert "already enriched" in calls_str


class TestEnrichmentCaching:
    def test_enrich_with_cache_stores_result(self):
        """First enrichment stores result in cache."""
        cache = RunScopedGraphCache()
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)
        spec = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec, cache=cache)
        assert spec.metadata.get("graph_cache_hit") is False
        assert spec.metadata.get("graph_cache_key") is not None

    def test_enrich_with_cache_hit(self):
        """Second enrichment with same targets hits cache."""
        cache = RunScopedGraphCache()
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)

        # First call
        spec1 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec1, cache=cache)

        # Second call with same targets
        spec2 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec2, cache=cache)
        assert spec2.metadata.get("graph_cache_hit") is True

    def test_enrich_cache_miss_different_targets(self):
        """Different targets → cache miss."""
        cache = RunScopedGraphCache()
        node_foo = _make_symbol_node("foo")
        node_bar = _make_symbol_node("bar", file_path="src/bar.py")
        facade = _make_facade(symbols={"foo": node_foo, "bar": node_bar})
        enricher = SpecGraphEnricher(facade)

        spec1 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec1, cache=cache)

        spec2 = _make_spec(target_symbols=["bar"])
        enricher.enrich(spec2, cache=cache)
        assert spec2.metadata.get("graph_cache_hit") is False

    def test_enrich_without_cache_still_works(self):
        """Enrichment works without cache (backward compat)."""
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)
        spec = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec)  # no cache argument
        assert "graph_context" in spec.metadata
        assert "graph_cache_hit" not in spec.metadata

    def test_enrich_cache_invalidation_causes_miss(self):
        """After invalidation (new generation), cache key changes → miss."""
        cache = RunScopedGraphCache()
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)

        spec1 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec1, cache=cache)

        # Invalidate (generation increments → different key)
        cache.invalidate_for_files(["foo.py"])

        spec2 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec2, cache=cache)
        # Due to generation change in key, this should be a miss
        assert spec2.metadata.get("graph_cache_hit") is False

    def test_enrich_cache_second_hit_no_facade_calls(self):
        """On cache hit, facade methods are not called again."""
        cache = RunScopedGraphCache()
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)

        spec1 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec1, cache=cache)
        call_count_after_first = facade.get_symbol.call_count

        spec2 = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec2, cache=cache)
        # No additional facade calls on cache hit
        assert facade.get_symbol.call_count == call_count_after_first

    def test_enrich_cache_key_stored_in_metadata(self):
        """Cache key is stored in spec.metadata for debugging."""
        cache = RunScopedGraphCache()
        node = _make_symbol_node("foo")
        facade = _make_facade(symbols={"foo": node})
        enricher = SpecGraphEnricher(facade)
        spec = _make_spec(target_symbols=["foo"])
        enricher.enrich(spec, cache=cache)
        key = spec.metadata.get("graph_cache_key")
        assert key is not None
        assert isinstance(key, str)
        assert len(key) == 24  # sha256 hex truncated to 24 chars

    # ── AST-fallback unresolved-list integrity (regression) ──────────────────────

    def test_ast_fallback_does_not_duplicate_unresolved_symbol(self):
        """Regression: an orphaned ``unresolved_symbols.append(sym_name)``
        statement left inside ``if _ast_resolved:`` used to re-append the stale
        *last* symbol from the Step 1 loop whenever AST fallback resolved >=1
        symbol.  This duplicated the last still-unresolved symbol, inflated
        ``len(unresolved_symbols)``, and drove ``graph_confidence`` *negative*
        (``_resolved_target_count`` underflowed → ``_graph_resolved_count = -1``
        → ``_weighted_resolved = -0.3`` → ``graph_confidence = -0.15``).
        """
        # Graph resolves neither "foo" nor "bar" → both start unresolved.
        facade = _make_facade(symbols={})
        spec = _make_spec(target_symbols=["foo", "bar"], target_files=["src/foo.py"])
        enricher = SpecGraphEnricher(facade)
        # AST fallback resolves only "foo" (the real method reads source files;
        # mocking isolates the orphan-append bug from filesystem effects).
        enricher._resolve_unresolved_via_ast = lambda *a, **k: [
            {"name": "foo", "file_path": "src/foo.py", "source": "ast_fallback"},
        ]

        gc = enricher._build_graph_context(spec)

        # "foo" resolved via AST; "bar" remains genuinely unresolved — once.
        assert gc["unresolved_symbols"] == ["bar"]
        # ast-weight 0.7 / 2 targets == 0.35 (must be >= 0, never negative).
        assert gc["graph_confidence"] == 0.35

    def test_ast_fallback_does_not_repollute_already_resolved_symbol(self):
        """Regression (same root cause): when the LAST symbol of the Step 1 loop
        resolved via the graph but an *earlier* symbol was unresolved then
        AST-resolved, the orphaned append re-added the already-resolved last
        symbol back into ``unresolved_symbols`` — reporting a phantom unresolved
        symbol that was in fact found by the graph.
        """
        # Graph resolves "bar" (last in loop) but not "foo".
        node_bar = _make_symbol_node("bar")
        facade = _make_facade(symbols={"bar": node_bar})
        spec = _make_spec(target_symbols=["foo", "bar"], target_files=["src/foo.py"])
        enricher = SpecGraphEnricher(facade)
        # AST fallback resolves the only genuinely-unresolved symbol "foo".
        enricher._resolve_unresolved_via_ast = lambda *a, **k: [
            {"name": "foo", "file_path": "src/foo.py", "source": "ast_fallback"},
        ]

        gc = enricher._build_graph_context(spec)

        # Everything resolves; nothing should remain unresolved.
        assert gc["unresolved_symbols"] == []
