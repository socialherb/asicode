"""Tests for VerificationSetBuilder."""
from external_llm.editor.verification.verification_set_builder import (
    SCOPE_LIMITS,
    VerificationGroup,
    VerificationSet,
    VerificationSetBuilder,
    VerificationTarget,
)


def _make_symbol_target(path, score=1.0, match_type="direct_symbol", reason="TEST"):
    """Create a mock SymbolAwareTestTarget."""
    from unittest.mock import MagicMock
    t = MagicMock()
    t.test_path = path
    t.priority_score = score
    t.match_type = match_type
    t.reason_codes = [reason]
    t.matched_symbols = ["sym"]
    return t


def _make_dep_candidate(path, score=0.8, relation="direct_import"):
    """Create a mock TestCoverageCandidate."""
    from unittest.mock import MagicMock
    c = MagicMock()
    c.test_path = path
    c.coverage_score = score
    c.relation_types = [relation]
    c.reason_codes = ["DEP_GRAPH"]
    c.matched_files = ["src.py"]
    c.matched_modules = ["mod"]
    return c


def _make_impact_set(impacted_files=None):
    from unittest.mock import MagicMock
    impact = MagicMock()
    impact.impacted_files = impacted_files or []
    impact.impacted_symbols = []
    impact.impacted_modules = []
    return impact


def _make_risk(level="medium", score=3):
    from unittest.mock import MagicMock
    risk = MagicMock()
    risk.level = level
    risk.score = score
    return risk


class TestVerificationTarget:
    def test_to_dict(self):
        t = VerificationTarget(test_path="t.py", priority_score=0.9, source="symbol_aware")
        d = t.to_dict()
        assert d["test_path"] == "t.py"
        assert d["priority_score"] == 0.9


class TestVerificationGroup:
    def test_paths(self):
        g = VerificationGroup(targets=[
            VerificationTarget(test_path="a.py"),
            VerificationTarget(test_path="b.py"),
        ])
        assert g.paths == ["a.py", "b.py"]

    def test_to_dict(self):
        g = VerificationGroup(name="primary", targets=[VerificationTarget(test_path="a.py")])
        d = g.to_dict()
        assert d["name"] == "primary"
        assert d["target_count"] == 1


class TestVerificationSet:
    def test_total_target_count(self):
        vs = VerificationSet()
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.secondary.targets = [VerificationTarget(test_path="b.py")]
        assert vs.total_target_count == 2

    def test_flatten_paths_all_groups(self):
        vs = VerificationSet(selected_scope_level="standard")
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.secondary.targets = [VerificationTarget(test_path="b.py")]
        vs.fallback.targets = [VerificationTarget(test_path="c.py")]
        paths = vs.flatten_paths()
        assert paths == ["a.py", "b.py", "c.py"]

    def test_flatten_paths_minimal(self):
        vs = VerificationSet(selected_scope_level="minimal")
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.secondary.targets = [VerificationTarget(test_path="b.py")]
        paths = vs.flatten_paths()
        assert paths == ["a.py"]  # minimal = primary only

    def test_flatten_paths_narrow(self):
        vs = VerificationSet(selected_scope_level="narrow")
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.secondary.targets = [VerificationTarget(test_path="b.py")]
        vs.fallback.targets = [VerificationTarget(test_path="c.py")]
        paths = vs.flatten_paths()
        assert "a.py" in paths
        assert "b.py" in paths
        assert "c.py" not in paths  # narrow excludes fallback

    def test_flatten_paths_deduplicates(self):
        vs = VerificationSet(selected_scope_level="standard")
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.secondary.targets = [VerificationTarget(test_path="a.py")]
        paths = vs.flatten_paths()
        assert paths == ["a.py"]

    def test_flatten_paths_custom_groups(self):
        vs = VerificationSet()
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        vs.fallback.targets = [VerificationTarget(test_path="c.py")]
        paths = vs.flatten_paths(include_groups=["primary", "fallback"])
        assert paths == ["a.py", "c.py"]

    def test_to_summary(self):
        vs = VerificationSet(selected_scope_level="standard", composite_risk_level="high")
        vs.primary.targets = [VerificationTarget(test_path="a.py")]
        s = vs.to_summary()
        assert s["scope_level"] == "standard"
        assert s["primary_count"] == 1
        assert s["composite_risk_level"] == "high"


class TestVerificationSetBuilder:
    def test_symbol_targets_go_to_primary(self):
        """High-score symbol targets → primary group."""
        builder = VerificationSetBuilder()
        targets = [
            _make_symbol_target("tests/test_a.py", score=1.0),
            _make_symbol_target("tests/test_b.py", score=0.8),
        ]
        vs = builder.build(symbol_targets=targets)
        assert len(vs.primary.targets) >= 2

    def test_low_score_targets_go_to_fallback(self):
        """Low-score targets → fallback."""
        builder = VerificationSetBuilder()
        targets = [
            _make_symbol_target("tests/test_a.py", score=0.1),
        ]
        vs = builder.build(symbol_targets=targets)
        assert len(vs.fallback.targets) >= 1

    def test_medium_score_to_secondary(self):
        """Medium-score targets → secondary."""
        builder = VerificationSetBuilder()
        targets = [
            _make_symbol_target("tests/test_a.py", score=0.5),
        ]
        vs = builder.build(symbol_targets=targets)
        assert len(vs.secondary.targets) >= 1

    def test_dep_graph_candidates_merged(self):
        """Dep graph candidates are merged with symbol targets."""
        builder = VerificationSetBuilder()
        sym = [_make_symbol_target("tests/test_a.py", score=1.0)]
        dep = [_make_dep_candidate("tests/test_b.py", score=0.8)]
        vs = builder.build(symbol_targets=sym, dep_graph_candidates=dep)
        all_paths = vs.flatten_paths()
        assert "tests/test_a.py" in all_paths
        assert "tests/test_b.py" in all_paths

    def test_deduplication(self):
        """Same test from multiple sources → single entry with best score."""
        builder = VerificationSetBuilder()
        sym = [_make_symbol_target("tests/test_a.py", score=1.0)]
        dep = [_make_dep_candidate("tests/test_a.py", score=0.8)]
        vs = builder.build(symbol_targets=sym, dep_graph_candidates=dep)
        paths = vs.flatten_paths()
        assert paths.count("tests/test_a.py") == 1

    def test_impact_fallback(self):
        """Impact set test files used as fallback."""
        builder = VerificationSetBuilder()
        impact = _make_impact_set(impacted_files=["src/foo.py", "tests/test_foo.py"])
        vs = builder.build(impact_set=impact)
        all_paths = vs.flatten_paths()
        assert "tests/test_foo.py" in all_paths

    def test_scope_minimal_limits(self):
        """Minimal scope: only primary, limited count."""
        builder = VerificationSetBuilder()
        targets = [_make_symbol_target(f"tests/test_{i}.py", score=1.0) for i in range(10)]
        vs = builder.build(symbol_targets=targets, scope_level_hint="minimal")
        assert len(vs.primary.targets) <= SCOPE_LIMITS["minimal"]["primary"]
        assert len(vs.secondary.targets) == 0

    def test_scope_broad_allows_more(self):
        """Broad scope: includes all groups, higher limits."""
        builder = VerificationSetBuilder()
        sym = [_make_symbol_target(f"t/test_{i}.py", score=1.0) for i in range(5)]
        dep = [_make_dep_candidate(f"t/dep_{i}.py", score=0.5) for i in range(5)]
        imp = _make_impact_set(impacted_files=[f"tests/test_imp_{i}.py" for i in range(5)])
        vs = builder.build(
            symbol_targets=sym, dep_graph_candidates=dep, impact_set=imp,
            scope_level_hint="broad",
        )
        assert vs.total_target_count > len(sym)

    def test_critical_risk_escalates_to_broad(self):
        """Critical risk escalates scope to broad."""
        builder = VerificationSetBuilder()
        risk = _make_risk(level="critical")
        vs = builder.build(composite_risk=risk, scope_level_hint="narrow")
        assert vs.selected_scope_level == "broad"

    def test_high_risk_escalates(self):
        """High risk escalates minimal/narrow to standard/broad."""
        builder = VerificationSetBuilder()
        risk = _make_risk(level="high")
        vs = builder.build(composite_risk=risk, scope_level_hint="narrow")
        assert vs.selected_scope_level == "standard"

    def test_low_risk_no_escalation(self):
        """Low risk does not change scope level."""
        builder = VerificationSetBuilder()
        risk = _make_risk(level="low")
        vs = builder.build(composite_risk=risk, scope_level_hint="narrow")
        assert vs.selected_scope_level == "narrow"

    def test_empty_inputs(self):
        """No inputs → empty but valid set."""
        builder = VerificationSetBuilder()
        vs = builder.build()
        assert vs.total_target_count == 0
        assert isinstance(vs.flatten_paths(), list)

    def test_builder_error_returns_empty(self):
        """Builder errors → empty VerificationSet."""
        builder = VerificationSetBuilder()
        # Pass something that would cause attribute errors
        vs = builder.build(symbol_targets="not_a_list")
        assert isinstance(vs, VerificationSet)

    def test_group_rationale_populated(self):
        """Groups have rationale strings."""
        builder = VerificationSetBuilder()
        targets = [_make_symbol_target("t.py", score=1.0)]
        vs = builder.build(symbol_targets=targets)
        assert len(vs.primary.rationale) > 0

    def test_deterministic_ordering(self):
        """Same inputs → same output ordering."""
        builder = VerificationSetBuilder()
        targets = [
            _make_symbol_target("tests/b.py", score=0.9),
            _make_symbol_target("tests/a.py", score=1.0),
        ]
        vs1 = builder.build(symbol_targets=targets)
        vs2 = builder.build(symbol_targets=targets)
        assert vs1.primary.paths == vs2.primary.paths


class TestVerificationScopeIntegration:
    """Test that verification set summary appears in GraphVerificationScope."""

    def test_scope_has_verification_set_summary_field(self):
        from external_llm.graph.execution_graph_advisor import GraphVerificationScope
        scope = GraphVerificationScope()
        assert hasattr(scope, "verification_set_summary")
        assert scope.verification_set_summary == {}

    def test_scope_to_dict_includes_verification_set(self):
        from external_llm.graph.execution_graph_advisor import GraphVerificationScope
        scope = GraphVerificationScope(verification_set_summary={"primary_count": 3})
        d = scope.to_dict()
        assert d["verification_set_summary"] == {"primary_count": 3}
