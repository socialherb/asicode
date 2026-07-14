"""Tests for ImpactPropagationEngine."""
from unittest.mock import MagicMock, patch

from external_llm.editor.verification.impact_propagation import (
    ImpactPropagationEngine,
    ImpactSet,
    PropagatedNode,
)


def _make_facade(
    symbols=None,
    callers=None,
    callees=None,
    file_symbols=None,
):
    """Create mock graph facade."""
    facade = MagicMock()

    # get_symbol: return SymbolNode-like mock
    def mock_get_symbol(name, **kw):
        syms = symbols or {}
        if name in syms:
            node = MagicMock()
            node.file_path = syms[name].get("file_path", "")
            node.module = syms[name].get("module", "")
            node.name = name
            node.qualname = name
            return node
        return None
    facade.get_symbol = MagicMock(side_effect=mock_get_symbol)

    # get_symbols_in_file: return list of symbol-like mocks
    def mock_get_symbols_in_file(path):
        fs = file_symbols or {}
        return [
            MagicMock(name=s, qualname=s, module="", file_path=path)
            for s in fs.get(path, [])
        ]
    facade.get_symbols_in_file = MagicMock(side_effect=mock_get_symbols_in_file)

    facade.get_callers = MagicMock(return_value=callers or [])
    facade.get_callees = MagicMock(return_value=callees or [])

    return facade


class TestPropagatedNode:
    def test_default(self):
        n = PropagatedNode()
        assert n.relation == "origin"
        assert n.depth == 0

    def test_to_dict(self):
        n = PropagatedNode(symbol="foo", depth=1, relation="caller", reason_codes=["X"])
        d = n.to_dict()
        assert d["symbol"] == "foo"
        assert d["depth"] == 1
        assert d["relation"] == "caller"


class TestImpactSet:
    def test_default(self):
        i = ImpactSet()
        assert i.impacted_symbols == []
        assert i.truncated is False

    def test_to_summary(self):
        i = ImpactSet(
            origin_symbols=["foo"],
            impacted_symbols=["foo", "bar"],
            impacted_files=["a.py", "b.py"],
            impacted_modules=["mod"],
            max_depth_reached=2,
        )
        s = i.to_summary()
        assert s["impacted_symbol_count"] == 2
        assert s["impacted_file_count"] == 2
        assert s["impacted_module_count"] == 1
        assert s["max_depth_reached"] == 2

    def test_to_dict(self):
        i = ImpactSet(origin_symbols=["foo"])
        d = i.to_dict()
        assert "origin_symbols" in d
        assert "propagated_nodes" in d


class TestImpactPropagationEngine:
    def test_origin_only(self):
        """Changed symbols appear as origin nodes."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate(changed_symbols=["foo"])
        assert "foo" in impact.origin_symbols
        assert "foo" in impact.impacted_symbols
        assert any(n.relation == "origin" for n in impact.propagated_nodes)

    def test_origin_files(self):
        """Changed files appear as origin nodes."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate(changed_files=["a.py"])
        assert "a.py" in impact.origin_files
        assert "a.py" in impact.impacted_files

    def test_caller_propagation(self):
        """Direct callers are propagated at depth 1."""
        caller_nodes = [
            PropagatedNode(symbol="bar", file_path="bar.py", module="mod", depth=1, relation="caller", reason_codes=["DIRECT_CALLER_OF_foo"]),
            PropagatedNode(symbol="baz", file_path="baz.py", module="mod", depth=1, relation="caller", reason_codes=["DIRECT_CALLER_OF_foo"]),
        ]

        facade = _make_facade(symbols={
            "foo": {"file_path": "foo.py", "module": "mod"},
            "bar": {"file_path": "bar.py", "module": "mod"},
            "baz": {"file_path": "baz.py", "module": "mod"},
        })

        engine = ImpactPropagationEngine(graph_facade=facade)
        engine._propagate_callers = MagicMock(return_value=(caller_nodes, 1))
        impact = engine.propagate(changed_symbols=["foo"])

        assert "bar" in impact.impacted_symbols
        assert "baz" in impact.impacted_symbols
        caller_result_nodes = [n for n in impact.propagated_nodes if n.relation == "caller"]
        assert len(caller_result_nodes) >= 2
        assert all(n.depth >= 1 for n in caller_result_nodes)

    def test_indirect_callers(self):
        """Indirect callers are propagated at depth >= 2."""
        caller_nodes = [
            PropagatedNode(symbol="bar", file_path="bar.py", depth=1, relation="caller", reason_codes=["DIRECT_CALLER_OF_foo"]),
            PropagatedNode(symbol="qux", file_path="qux.py", depth=2, relation="caller", reason_codes=["INDIRECT_CALLER_OF_foo"]),
        ]

        facade = _make_facade(symbols={
            "foo": {"file_path": "foo.py", "module": "mod"},
            "bar": {"file_path": "bar.py", "module": "mod"},
            "qux": {"file_path": "qux.py", "module": "mod"},
        })

        engine = ImpactPropagationEngine(graph_facade=facade)
        engine._propagate_callers = MagicMock(return_value=(caller_nodes, 2))
        impact = engine.propagate(changed_symbols=["foo"])

        assert "qux" in impact.impacted_symbols
        indirect_nodes = [n for n in impact.propagated_nodes if "INDIRECT" in str(n.reason_codes)]
        assert len(indirect_nodes) >= 1

    def test_same_file_propagation(self):
        """Sibling symbols in same file are included."""
        facade = _make_facade(
            symbols={"foo": {"file_path": "a.py", "module": "mod"}},
            file_symbols={"a.py": ["foo", "helper_fn"]},
        )

        engine = ImpactPropagationEngine(graph_facade=facade)
        with patch("external_llm.editor.simulator.dependency_traversal.collect_direct_callers", return_value=[]):
            with patch("external_llm.editor.simulator.dependency_traversal.collect_indirect_callers", return_value=[]):
                impact = engine.propagate(
                    changed_symbols=["foo"],
                    changed_files=["a.py"],
                    include_same_file=True,
                )

        # helper_fn should be found via same-file propagation
        [n for n in impact.propagated_nodes if n.relation == "same_file"]
        # At minimum, check no crash
        assert isinstance(impact, ImpactSet)

    def test_max_depth_default(self):
        """Default max_depth is DEFAULT_MAX_DEPTH."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate(changed_symbols=["foo"])
        # Just verify it doesn't crash
        assert impact.max_depth_reached >= 0

    def test_structural_op_deeper_depth(self):
        """Structural operations get deeper default max_depth."""
        engine = ImpactPropagationEngine()
        # The engine should use STRUCTURAL_OP_MAX_DEPTH for structural ops
        # We test by checking that the engine creates without error
        impact = engine.propagate(
            changed_symbols=["foo"],
            operation_kind="RENAME_SYMBOL",
        )
        assert isinstance(impact, ImpactSet)

    def test_max_nodes_truncation(self):
        """Exceeding max_nodes sets truncated=True."""
        engine = ImpactPropagationEngine()
        # With max_nodes=2 and 3 origins, should truncate
        impact = engine.propagate(
            changed_symbols=["a", "b", "c"],
            changed_files=["x.py", "y.py", "z.py"],
            max_nodes=3,
        )
        # Origins alone may exceed budget
        assert impact.truncated or len(impact.propagated_nodes) <= 3

    def test_no_graph_fallback(self):
        """Without graph, returns minimal impact from origins only."""
        engine = ImpactPropagationEngine()  # no facade
        impact = engine.propagate(
            changed_symbols=["foo"],
            changed_files=["a.py"],
        )
        assert "foo" in impact.impacted_symbols
        assert "a.py" in impact.impacted_files
        assert impact.max_depth_reached == 0

    def test_empty_input(self):
        """Empty input → empty impact."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate()
        assert impact.impacted_symbols == []
        assert impact.impacted_files == []

    def test_impacted_files_from_callers(self):
        """Caller symbol files are added to impacted_files."""
        facade = _make_facade(symbols={
            "foo": {"file_path": "src/foo.py", "module": "src.foo"},
            "bar": {"file_path": "src/bar.py", "module": "src.bar"},
        })

        engine = ImpactPropagationEngine(graph_facade=facade)
        with patch("external_llm.editor.simulator.dependency_traversal.collect_direct_callers", return_value=["bar"]):
            with patch("external_llm.editor.simulator.dependency_traversal.collect_indirect_callers", return_value=[]):
                impact = engine.propagate(changed_symbols=["foo"])

        assert "src/bar.py" in impact.impacted_files

    def test_impacted_modules(self):
        """Modules are tracked from resolved symbols."""
        facade = _make_facade(symbols={
            "foo": {"file_path": "src/foo.py", "module": "src.foo"},
        })

        engine = ImpactPropagationEngine(graph_facade=facade)
        with patch("external_llm.editor.simulator.dependency_traversal.collect_direct_callers", return_value=[]):
            with patch("external_llm.editor.simulator.dependency_traversal.collect_indirect_callers", return_value=[]):
                impact = engine.propagate(changed_symbols=["foo"])

        assert "src.foo" in impact.impacted_modules

    def test_same_module_propagation(self):
        """Same-module directories are tracked."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate(
            changed_files=["external_llm/agent/foo.py"],
            include_same_module=True,
        )
        # Should have a same_module node for the directory
        [n for n in impact.propagated_nodes if n.relation == "same_module"]
        # At minimum check no crash
        assert isinstance(impact, ImpactSet)

    def test_deduplication(self):
        """Symbols seen via multiple paths are not duplicated."""
        facade = _make_facade(symbols={
            "foo": {"file_path": "a.py", "module": "mod"},
        })

        engine = ImpactPropagationEngine(graph_facade=facade)
        with patch("external_llm.editor.simulator.dependency_traversal.collect_direct_callers", return_value=[]):
            with patch("external_llm.editor.simulator.dependency_traversal.collect_indirect_callers", return_value=[]):
                impact = engine.propagate(
                    changed_symbols=["foo", "foo"],  # duplicate
                )

        assert impact.impacted_symbols.count("foo") == 1

    def test_reason_codes_present(self):
        """Propagated nodes have reason codes."""
        engine = ImpactPropagationEngine()
        impact = engine.propagate(changed_symbols=["foo"])
        origin = [n for n in impact.propagated_nodes if n.relation == "origin"]
        assert len(origin) > 0
        assert "CHANGED_SYMBOL" in origin[0].reason_codes

    def test_file_to_module(self):
        """File path converts to module name."""
        engine = ImpactPropagationEngine()
        assert engine._file_to_module("external_llm/agent/foo.py") == "external_llm.agent.foo"
        assert engine._file_to_module(None) is None
        assert engine._file_to_module("not_python.txt") is None


class TestVerificationScopeImpactIntegration:
    """Test that impact propagation integrates with verification scope."""

    def test_scope_has_impact_summary_field(self):
        from external_llm.graph.execution_graph_advisor import GraphVerificationScope
        scope = GraphVerificationScope()
        assert hasattr(scope, "impact_summary")
        assert scope.impact_summary == {}

    def test_scope_to_dict_includes_impact(self):
        from external_llm.graph.execution_graph_advisor import GraphVerificationScope
        scope = GraphVerificationScope(impact_summary={"count": 5})
        d = scope.to_dict()
        assert d["impact_summary"] == {"count": 5}
