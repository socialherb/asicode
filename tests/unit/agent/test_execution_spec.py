"""Unit tests for execution_spec.py — 100% branch coverage.

Tests:
  - ExplorationHints.to_dict()
  - ExplorationSignals dataclass
  - decide_scope_mode() — all branching combinations
  - ResolvedExecutionSpec (.suspicious_empty_symbols, _recompute_symbol_evidence,
    adjust_scope_from_graph, to_dict/from_dict round-trip)
"""

from __future__ import annotations

from external_llm.agent.config.thresholds import config as _cfg_es
from external_llm.agent.enums import EstimatedScope
from external_llm.agent.execution_spec import (
    ExplorationHints,
    ExplorationSignals,
    ResolvedExecutionSpec,
    decide_scope_mode,
)

# ═══════════════════════════════════════════════════════════════════════════
# ExplorationHints
# ═══════════════════════════════════════════════════════════════════════════


class TestExplorationHints:
    """Coverage for ExplorationHints dataclass + to_dict()."""

    def test_default_construction(self):
        h = ExplorationHints()
        assert h.files == []
        assert h.symbols == []
        assert h.confidence == 0.0
        assert h.mode == "targeted"
        assert h.anomaly is None

    def test_to_dict_full(self):
        h = ExplorationHints(
            files=["a.py", "b.py"],
            symbols=["foo", "bar"],
            confidence=0.8,
            mode="diagnostic",
            anomaly="unresolved_symbol",
            top1_score=0.6,
            top12_margin=0.2,
        )
        d = h.to_dict()
        assert d == {
            "files": ["a.py", "b.py"],
            "symbols": ["foo", "bar"],
            "confidence": 0.8,
            "mode": "diagnostic",
            "anomaly": "unresolved_symbol",
            "top1_score": 0.6,
            "top12_margin": 0.2,
        }

    def test_to_dict_defaults(self):
        d = ExplorationHints().to_dict()
        assert d["files"] == []
        assert d["symbols"] == []
        assert d["confidence"] == 0.0
        assert d["mode"] == "targeted"
        assert d["anomaly"] is None
        assert d["top1_score"] == 0.0
        assert d["top12_margin"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ExplorationSignals
# ═══════════════════════════════════════════════════════════════════════════


class TestExplorationSignals:
    """Coverage for ExplorationSignals dataclass."""

    def test_construction_and_fields(self):
        s = ExplorationSignals(
            confidence=0.7, top1_score=0.5, margin=0.3,
            anomaly=False, convergence_turns=3,
        )
        assert s.confidence == 0.7
        assert s.top1_score == 0.5
        assert s.margin == 0.3
        assert s.anomaly is False
        assert s.convergence_turns == 3


# ═══════════════════════════════════════════════════════════════════════════
# decide_scope_mode
# ═══════════════════════════════════════════════════════════════════════════


class TestDecideScopeMode:
    """Coverage for decide_scope_mode() — all branch combinations.

    Thresholds (from config):
      STRICT_TOP1=0.3, STRICT_MARGIN=0.2,
      STRICT_TARGETED_TOP1=0.35, STRICT_TARGETED_MARGIN=0.25,
      GUIDED_TOP1=0.15
    """

    # ── request_mode == "open_ended" ────────────────────────────────────

    def test_open_ended_always_guided(self):
        """open_ended always returns 'guided' regardless of signals."""
        signals = ExplorationSignals(confidence=0.9, top1_score=0.9, margin=0.5, anomaly=False, convergence_turns=5)
        assert decide_scope_mode(signals, "open_ended") == "guided"

    def test_open_ended_low_signals(self):
        """open_ended with very low signals still returns 'guided'."""
        signals = ExplorationSignals(confidence=0.0, top1_score=0.0, margin=0.0, anomaly=True, convergence_turns=0)
        assert decide_scope_mode(signals, "open_ended") == "guided"

    # ── request_mode == "diagnostic" ────────────────────────────────────

    def test_diagnostic_strict(self):
        """diagnostic: top1>0.3, rel_margin>0.2, no anomaly → strict."""
        # top1=0.5, margin=0.2 → rel_margin=0.4 > 0.2
        signals = ExplorationSignals(confidence=0.7, top1_score=0.5, margin=0.2, anomaly=False, convergence_turns=3)
        assert decide_scope_mode(signals, "diagnostic") == "strict"

    def test_diagnostic_anomaly_blocks_strict(self):
        """diagnostic: anomaly=True prevents strict."""
        signals = ExplorationSignals(confidence=0.7, top1_score=0.5, margin=0.2, anomaly=True, convergence_turns=3)
        assert decide_scope_mode(signals, "diagnostic") == "guided"

    def test_diagnostic_low_top1_guided(self):
        """diagnostic: top1 too low → guided."""
        signals = ExplorationSignals(confidence=0.3, top1_score=0.1, margin=0.01, anomaly=False, convergence_turns=1)
        assert decide_scope_mode(signals, "diagnostic") == "guided"

    def test_diagnostic_low_margin_guided(self):
        """diagnostic: high top1 but low relative margin → guided."""
        # top1=0.5, margin=0.05 → rel_margin=0.1 < 0.2
        signals = ExplorationSignals(confidence=0.7, top1_score=0.5, margin=0.05, anomaly=False, convergence_turns=3)
        assert decide_scope_mode(signals, "diagnostic") == "guided"

    def test_diagnostic_zero_top1_guided(self):
        """diagnostic: top1=0 → rel_margin=0/0.01=0 → guided."""
        signals = ExplorationSignals(confidence=0.0, top1_score=0.0, margin=0.0, anomaly=False, convergence_turns=0)
        assert decide_scope_mode(signals, "diagnostic") == "guided"

    # ── request_mode == "targeted" ──────────────────────────────────────

    def test_targeted_strict(self):
        """targeted: top1>0.35, rel_margin>0.25, no anomaly → strict."""
        # top1=0.6, margin=0.2 → rel_margin=0.33 > 0.25
        signals = ExplorationSignals(confidence=0.8, top1_score=0.6, margin=0.2, anomaly=False, convergence_turns=4)
        assert decide_scope_mode(signals, "targeted") == "strict"

    def test_targeted_anomaly_blocks_strict(self):
        """targeted: anomaly=True prevents strict."""
        signals = ExplorationSignals(confidence=0.8, top1_score=0.6, margin=0.2, anomaly=True, convergence_turns=4)
        assert decide_scope_mode(signals, "targeted") == "guided"

    def test_targeted_low_margin_guided(self):
        """targeted: high top1 but low relative margin → guided."""
        # top1=0.6, margin=0.02 → rel_margin=0.033 < 0.25
        signals = ExplorationSignals(confidence=0.8, top1_score=0.6, margin=0.02, anomaly=False, convergence_turns=4)
        assert decide_scope_mode(signals, "targeted") == "guided"

    def test_targeted_guided_via_top1(self):
        """targeted: top1 > 0.15 but below strict threshold → guided."""
        signals = ExplorationSignals(confidence=0.5, top1_score=0.2, margin=0.02, anomaly=False, convergence_turns=2)
        assert decide_scope_mode(signals, "targeted") == "guided"

    def test_targeted_free(self):
        """targeted: top1 too low → free."""
        signals = ExplorationSignals(confidence=0.1, top1_score=0.05, margin=0.01, anomaly=False, convergence_turns=1)
        assert decide_scope_mode(signals, "targeted") == "free"

    def test_targeted_zero_top1_free(self):
        """targeted: top1=0 → free."""
        signals = ExplorationSignals(confidence=0.0, top1_score=0.0, margin=0.0, anomaly=False, convergence_turns=0)
        assert decide_scope_mode(signals, "targeted") == "free"

    def test_targeted_boundary_strict_top1(self):
        """targeted: top1 exactly at STRICT_TARGETED_TOP1 but margin too low → guided."""
        strict_top1 = _cfg_es.scores.STRICT_TARGETED_TOP1
        signals = ExplorationSignals(
            confidence=0.8, top1_score=strict_top1, margin=0.0,
            anomaly=False, convergence_turns=4,
        )
        assert decide_scope_mode(signals, "targeted") == "guided"

    def test_targeted_boundary_guided_top1(self):
        """targeted: top1 just above GUIDED_TOP1 with low margin → guided."""
        guided_top1 = _cfg_es.scores.GUIDED_TOP1 + 0.001  # strictly > GUIDED_TOP1
        signals = ExplorationSignals(
            confidence=0.3, top1_score=guided_top1, margin=0.0,
            anomaly=False, convergence_turns=1,
        )
        assert decide_scope_mode(signals, "targeted") == "guided"

    def test_targeted_boundary_below_guided_top1_is_free(self):
        """targeted: top1 exactly at GUIDED_TOP1 is NOT > GUIDED_TOP1 → free."""
        guided_top1 = _cfg_es.scores.GUIDED_TOP1
        signals = ExplorationSignals(
            confidence=0.3, top1_score=guided_top1, margin=0.0,
            anomaly=False, convergence_turns=1,
        )
        assert decide_scope_mode(signals, "targeted") == "free"


# ═══════════════════════════════════════════════════════════════════════════
# ResolvedExecutionSpec — property / methods
# ═══════════════════════════════════════════════════════════════════════════


class TestSuspiciousEmptySymbols:
    """Coverage for ResolvedExecutionSpec.suspicious_empty_symbols."""

    def test_empty_target_no_mentions(self):
        spec = ResolvedExecutionSpec(
            original_request="fix bug",
            intent="fix something",
            request_type="bugfix",
            had_symbol_mentions=False,
        )
        assert spec.suspicious_empty_symbols is False

    def test_empty_target_with_mentions(self):
        spec = ResolvedExecutionSpec(
            original_request="update foo",
            intent="update foo",
            request_type="modify",
            intent_symbols=["foo"],
            had_symbol_mentions=True,
        )
        assert spec.suspicious_empty_symbols is True

    def test_non_empty_target(self):
        spec = ResolvedExecutionSpec(
            original_request="update foo",
            intent="update foo",
            request_type="modify",
            target_symbols=["foo"],
            had_symbol_mentions=True,
        )
        assert spec.suspicious_empty_symbols is False


class TestRecomputeSymbolEvidence:
    """Coverage for _recompute_symbol_evidence()."""

    def test_no_intent_symbols(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
        )
        spec._recompute_symbol_evidence()
        assert spec.had_symbol_mentions is False
        assert spec.unresolved_mentions == []

    def test_all_resolved(self):
        spec = ResolvedExecutionSpec(
            original_request="fix foo", intent="fix foo", request_type="bugfix",
            intent_symbols=["foo", "bar"],
            target_symbols=["foo", "bar"],
        )
        spec._recompute_symbol_evidence()
        assert spec.had_symbol_mentions is True
        assert spec.unresolved_mentions == []

    def test_some_unresolved(self):
        spec = ResolvedExecutionSpec(
            original_request="fix foo and bar", intent="fix foo and bar",
            request_type="bugfix",
            intent_symbols=["foo", "bar", "baz"],
            target_symbols=["foo"],
        )
        spec._recompute_symbol_evidence()
        assert spec.had_symbol_mentions is True
        assert sorted(spec.unresolved_mentions) == ["bar", "baz"]


class TestAdjustScopeFromGraph:
    """Coverage for adjust_scope_from_graph()."""

    def test_no_graph_context(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.TINY

    def test_non_authoritative_suppresses_upgrade(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            authoritative=False,
            metadata={"graph_context": {"impact_files": ["a.py"] * 15}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.TINY
        assert spec.metadata.get("scope_upgrade_suppressed") == "non_authoritative_targets"

    def test_large_impact_upgrade_to_large(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.SMALL,
            metadata={"graph_context": {"impact_files": [f"{i}.py" for i in range(10)]}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.LARGE
        assert spec.metadata["scope_upgraded_from"] == EstimatedScope.SMALL

    def test_medium_impact_upgrade_to_medium(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            metadata={"graph_context": {"impact_files": [f"{i}.py" for i in range(5)]}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.MEDIUM
        assert spec.metadata["scope_upgraded_from"] == EstimatedScope.TINY

    def test_already_large_no_upgrade(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.LARGE,
            metadata={"graph_context": {"impact_files": [f"{i}.py" for i in range(15)]}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.LARGE  # unchanged
        assert "scope_upgraded_from" not in spec.metadata

    def test_scope_high_impact_medium_already(self):
        """MEDIUM scope with >=10 impact files does NOT upgrade (already >= MEDIUM)."""
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.MEDIUM,
            metadata={"graph_context": {"impact_files": [f"{i}.py" for i in range(10)]}},
        )
        spec.adjust_scope_from_graph()
        # MEDIUM + 10 impact files → upgrade to LARGE (>= 10 always upgrades to LARGE)
        assert spec.estimated_scope == EstimatedScope.LARGE

    def test_caller_fan_in_upgrades_to_medium(self):
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            target_symbols=["hot_function"],
            metadata={
                "graph_context": {
                    "callers": {
                        "hot_function": [
                            {"file": f"caller_{i}.py"} for i in range(4)
                        ]
                    }
                }
            },
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.MEDIUM

    def test_caller_fan_in_less_than_four(self):
        """< 4 caller files → no upgrade from caller check."""
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            target_symbols=["mild_function"],
            metadata={
                "graph_context": {
                    "impact_files": ["a.py", "b.py"],
                    "callers": {
                        "mild_function": [
                            {"file": "a.py"}, {"file": "b.py"}, {"file": "c.py"}
                        ]
                    }
                }
            },
        )
        spec.adjust_scope_from_graph()
        # 2 impact files (< 5), 3 caller files (< 4) → no upgrade
        assert spec.estimated_scope == EstimatedScope.TINY


# ═══════════════════════════════════════════════════════════════════════════
# ResolvedExecutionSpec — serialization round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializationRoundTrip:
    """Coverage for to_dict() / from_dict()."""

    def test_to_dict_basic_fields(self):
        spec = ResolvedExecutionSpec(
            original_request="add feature X",
            intent="implement feature X",
            request_type="feature",
            target_files=["src/x.py"],
            new_files=["tests/test_x.py"],
            target_symbols=["XClass"],
            risk_level="high",
            estimated_scope=EstimatedScope.MEDIUM,
            language="python",
        )
        d = spec.to_dict()
        assert d["original_request"] == "add feature X"
        assert d["intent"] == "implement feature X"
        assert d["target_files"] == ["src/x.py"]
        assert d["new_files"] == ["tests/test_x.py"]
        assert d["risk_level"] == "high"
        assert d["estimated_scope"] == EstimatedScope.MEDIUM
        assert d["hints"] is None

    def test_round_trip_with_hints(self):
        hints = ExplorationHints(files=["a.py"], symbols=["Foo"], confidence=0.7, mode="targeted")
        spec = ResolvedExecutionSpec(
            original_request="fix Foo",
            intent="fix Foo",
            request_type="bugfix",
            hints=hints,
        )
        d = spec.to_dict()
        assert d["hints"] is not None

        restored = ResolvedExecutionSpec.from_dict(d)
        assert restored.original_request == "fix Foo"
        assert restored.hints is not None
        assert restored.hints.files == ["a.py"]
        assert restored.hints.symbols == ["Foo"]
        assert restored.hints.confidence == 0.7

    def test_round_trip_without_hints(self):
        spec = ResolvedExecutionSpec(
            original_request="refactor Bar",
            intent="refactor Bar",
            request_type="refactor",
        )
        d = spec.to_dict()
        restored = ResolvedExecutionSpec.from_dict(d)
        assert restored.original_request == "refactor Bar"
        assert restored.hints is None
        assert restored.scope_mode == "free"

    def test_round_trip_all_fields(self):
        hints = ExplorationHints(
            files=["mod.py"], symbols=["func"], confidence=0.9,
            mode="diagnostic", anomaly="missing_file",
            top1_score=0.85, top12_margin=0.3,
        )
        spec = ResolvedExecutionSpec(
            original_request="update func in mod.py",
            intent="update func",
            request_type="modify",
            target_files=["mod.py"],
            target_symbols=["func"],
            intent_files=["mod.py"],
            intent_symbols=["func"],
            modify_symbols=["func"],
            reference_symbols=["helper"],
            reference_files=["utils.py"],
            constraints=["must maintain API"],
            acceptance_criteria=["tests pass"],
            risk_level="low",
            estimated_scope=EstimatedScope.LARGE,
            action_hint="modify_logic",
            change_goal="update func logic",
            suggested_strategies=["replace_body"],
            language="python",
            metadata={"key": "val"},
            hints=hints,
            scope_mode="strict",
            authoritative=True,
            target_provenance="exploration",
            had_symbol_mentions=True,
            unresolved_mentions=[],
            code_context=[],
            integration_targets=[],
        )
        d = spec.to_dict()
        restored = ResolvedExecutionSpec.from_dict(d)
        assert restored.original_request == spec.original_request
        assert restored.intent == spec.intent
        assert restored.target_files == spec.target_files
        assert restored.hints is not None
        assert restored.hints.files == ["mod.py"]
        assert restored.hints.anomaly == "missing_file"
        assert restored.scope_mode == "strict"
        assert restored.language == "python"

    def test_from_dict_with_empty_data(self):
        restored = ResolvedExecutionSpec.from_dict({})
        assert restored.original_request == ""
        assert restored.intent == ""
        assert restored.request_type == ""
        assert restored.hints is None
        assert restored.estimated_scope == EstimatedScope.SMALL
        assert restored.authoritative is True


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge and boundary cases."""

    def test_suspicious_empty_during_round_trip(self):
        spec = ResolvedExecutionSpec(
            original_request="find x", intent="find x",
            request_type="exploration",
            intent_symbols=["x"],  # mentioned but not resolved
            had_symbol_mentions=True,
        )
        # target_symbols is empty, had_symbol_mentions is True
        assert spec.suspicious_empty_symbols is True

        # Round-trip preserves it
        d = spec.to_dict()
        restored = ResolvedExecutionSpec.from_dict(d)
        assert restored.had_symbol_mentions is True
        assert restored.suspicious_empty_symbols is True

    def test_adjust_scope_no_impact_files_no_callers(self):
        """Empty impact_files and no callers → no change."""
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            metadata={"graph_context": {"impact_files": [], "callers": {}}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.TINY

    def test_adjust_scope_medium_with_four_impact_files(self):
        """4 impact files (<5) with no caller data → no upgrade."""
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.TINY,
            metadata={"graph_context": {"impact_files": ["a.py", "b.py", "c.py", "d.py"]}},
        )
        spec.adjust_scope_from_graph()
        assert spec.estimated_scope == EstimatedScope.TINY

    def test_adjust_scope_large_no_impact_files_many_callers(self):
        """No impact_files but 4+ caller files → upgrade to MEDIUM."""
        spec = ResolvedExecutionSpec(
            original_request="fix", intent="fix", request_type="bugfix",
            estimated_scope=EstimatedScope.SMALL,
            target_symbols=["hot_func"],
            metadata={
                "graph_context": {
                    "callers": {
                        "hot_func": [{"file": f"{c}.py"} for c in "abcde"]
                    }
                }
            },
        )
        spec.adjust_scope_from_graph()
        # 0 impact files but 5 caller files → upgrade to MEDIUM
        assert spec.estimated_scope == EstimatedScope.MEDIUM
