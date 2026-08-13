"""Tests for SymbolAwareTestFinder."""

import pytest

from external_llm.testing.symbol_aware_test_finder import (
    MATCH_SCORES,
    SCOPE_LIMITS,
    SymbolAwareTestFinder,
    SymbolAwareTestTarget,
)


@pytest.fixture
def test_repo(tmp_path):
    """Create a minimal test repo structure."""
    # Source files
    src = tmp_path / "external_llm" / "agent"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    (src / "planner_agent.py").write_text(
        "class PlannerAgent:\n    def create_plan(self): pass\n"
    )
    (src / "tool_registry.py").write_text(
        "from .planner_agent import PlannerAgent\n"
        "class ToolRegistry:\n    pass\n"
    )

    # Test files
    tests = tmp_path / "tests" / "unit" / "agent"
    tests.mkdir(parents=True)
    (tests / "__init__.py").write_text("")
    (tests / "test_planner_agent.py").write_text(
        "from external_llm.editor._editor_core.lane.planner_agent import PlannerAgent\n"
        "def test_create_plan():\n    p = PlannerAgent()\n"
    )
    (tests / "test_tool_registry.py").write_text(
        "from external_llm.agent.tool_registry import ToolRegistry\n"
        "def test_registry():\n    pass\n"
    )
    (tests / "test_unrelated.py").write_text(
        "def test_something():\n    pass\n"
    )

    # Another test dir
    graph_tests = tmp_path / "tests" / "unit" / "graph"
    graph_tests.mkdir(parents=True)
    (graph_tests / "__init__.py").write_text("")
    (graph_tests / "test_graph_facade.py").write_text(
        "def test_facade():\n    pass\n"
    )

    return str(tmp_path)


class TestSymbolAwareTestTarget:
    def test_default(self):
        t = SymbolAwareTestTarget(test_path="tests/test_foo.py")
        assert t.priority_score == 0.0
        assert t.match_type == "filename_fallback"

    def test_to_dict(self):
        t = SymbolAwareTestTarget(
            test_path="tests/test_foo.py",
            priority_score=0.9,
            reason_codes=["DIRECT_SYMBOL_MATCH"],
            matched_symbols=["Foo"],
            match_type="direct_symbol",
        )
        d = t.to_dict()
        assert d["test_path"] == "tests/test_foo.py"
        assert d["priority_score"] == 0.9
        assert "DIRECT_SYMBOL_MATCH" in d["reason_codes"]


class TestSymbolAwareTestFinder:
    def test_direct_symbol_match(self, test_repo):
        """Symbol name found in test file → direct_symbol match."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_symbols=["PlannerAgent"],
        )
        paths = [t.test_path for t in targets]
        assert any("test_planner_agent" in p for p in paths)
        # Direct match should have highest score
        top = targets[0]
        assert top.match_type == "direct_symbol"
        assert top.priority_score == MATCH_SCORES["direct_symbol"]

    def test_module_import_match(self, test_repo):
        """Test file imports target module → module_import match."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_files=["external_llm/agent/planner_agent.py"],
        )
        paths = [t.test_path for t in targets]
        # Should find test that imports planner_agent
        assert any("test_planner_agent" in p for p in paths)

    def test_same_module_match(self, test_repo):
        """Corresponding test file by naming convention."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_files=["external_llm/agent/tool_registry.py"],
        )
        paths = [t.test_path for t in targets]
        assert any("test_tool_registry" in p for p in paths)

    def test_impact_adjacency(self, test_repo):
        """Test files in impact set are included."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            impact_files=["tests/unit/graph/test_graph_facade.py"],
        )
        paths = [t.test_path for t in targets]
        assert any("test_graph_facade" in p for p in paths)

    def test_filename_fallback(self, test_repo):
        """When no symbol/module match, use filename heuristic."""
        finder = SymbolAwareTestFinder(test_repo)
        # Use a file that doesn't have direct imports in tests
        targets = finder.discover_test_targets(
            target_files=["external_llm/agent/planner_agent.py"],
        )
        assert len(targets) > 0  # should find something

    def test_empty_inputs(self, test_repo):
        """No inputs → empty targets."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets()
        assert targets == []

    def test_scope_narrow_limits(self, test_repo):
        """Narrow scope limits to 5 targets."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_symbols=["PlannerAgent"],
            scope_level="narrow",
        )
        assert len(targets) <= SCOPE_LIMITS["narrow"]

    def test_scope_broad_allows_more(self, test_repo):
        """Broad scope allows up to 20 targets."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_symbols=["PlannerAgent"],
            scope_level="broad",
        )
        # Should allow more (though may not have 20 test files in fixture)
        assert len(targets) <= SCOPE_LIMITS["broad"]

    def test_ranking_order(self, test_repo):
        """Targets are ranked by priority score descending."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_symbols=["PlannerAgent"],
            target_files=["external_llm/agent/planner_agent.py"],
        )
        if len(targets) >= 2:
            assert targets[0].priority_score >= targets[1].priority_score

    def test_deduplication(self, test_repo):
        """Same test file found by multiple methods → single entry with best score."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            target_symbols=["PlannerAgent"],
            target_files=["external_llm/agent/planner_agent.py"],
        )
        paths = [t.test_path for t in targets]
        assert len(paths) == len(set(paths))  # no duplicates

    def test_to_path_list(self, test_repo):
        """to_path_list degrades to string list."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(target_symbols=["PlannerAgent"])
        paths = finder.to_path_list(targets)
        assert all(isinstance(p, str) for p in paths)

    def test_build_summary(self, test_repo):
        """build_summary produces metadata dict."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(target_symbols=["PlannerAgent"])
        summary = finder.build_summary(targets)
        assert summary["symbol_aware_targeting_used"] is True
        assert "target_count" in summary
        assert "top_targets" in summary

    def test_graph_context_fallback(self, test_repo):
        """Uses graph_context to extract symbols if not provided directly."""
        finder = SymbolAwareTestFinder(test_repo)
        targets = finder.discover_test_targets(
            graph_context={
                "resolved_symbols": [{"name": "PlannerAgent"}],
                "impact_files": [],
                "primary_files": ["external_llm/agent/planner_agent.py"],
            },
        )
        assert len(targets) > 0

    def test_nonexistent_repo(self):
        """Non-existent repo → empty targets."""
        finder = SymbolAwareTestFinder("/nonexistent/path")
        targets = finder.discover_test_targets(target_symbols=["Foo"])
        assert targets == []
