"""Tests for edit_localization.graph_propagator."""

from external_llm.edit_localization.graph_propagator import (
    expand_candidates,
    propagate_scores,
)


def _make_graph(callees_map, callers_map=None):
    """Helper to build a minimal graph_context dict."""
    return {
        "callees": callees_map,
        "callers": callers_map or {},
        "resolved_symbols": {},
    }


class TestPropagateScores:
    """Test cross-symbol score propagation."""

    def test_callee_boosts_caller(self):
        """parse_definitions (low) should be boosted by _walk_definitions (high)."""
        scores = {
            "parse_definitions": (0.31, "low_base"),
            "_walk_definitions": (0.62, "high_base"),
        }
        graph = _make_graph({
            "parse_definitions": [
                {"symbol": "_walk_definitions", "file": "code.py", "confidence": 0.9},
            ],
        })

        result = propagate_scores(scores, graph)

        parse = result["parse_definitions"]
        walk = result["_walk_definitions"]

        # parse should be boosted
        assert parse.propagated_score > parse.base_score
        assert parse.base_score == 0.31
        # boost = 0.62 * 0.35 = 0.217 → propagated ≈ 0.527
        assert parse.propagated_score > 0.5
        assert "_walk_definitions" in parse.chain

        # walk should be unchanged (no callees or callee score not higher)
        assert walk.propagated_score == walk.base_score

    def test_no_graph_context(self):
        """Without graph, scores should be unchanged."""
        scores = {"foo": (0.3, "base")}
        result = propagate_scores(scores, None)

        assert result["foo"].propagated_score == 0.3
        assert result["foo"].chain == []

    def test_empty_callees(self):
        """Symbol with no callees — no boost."""
        scores = {"foo": (0.3, "base")}
        graph = _make_graph({})

        result = propagate_scores(scores, graph)
        assert result["foo"].propagated_score == 0.3

    def test_chain_propagation_two_levels(self):
        """A → B → C: A should get boost from C if C scores high."""
        scores = {
            "A": (0.1, "delegation"),
            "B": (0.3, "intermediate"),
            "C": (0.8, "value_determiner"),
        }
        graph = _make_graph({
            "A": [{"symbol": "B", "file": "code.py", "confidence": 0.9}],
            "B": [{"symbol": "C", "file": "code.py", "confidence": 0.9}],
        })

        result = propagate_scores(scores, graph)

        # A should be boosted via B → C chain
        a_result = result["A"]
        assert a_result.propagated_score > a_result.base_score
        # boost comes from C (0.8) through 2-level chain
        assert a_result.propagated_score > 0.3

    def test_callee_score_on_demand(self):
        """Callee not in initial scores → scored on demand via callback."""
        scores = {
            "caller": (0.2, "delegation"),
        }
        graph = _make_graph({
            "caller": [{"symbol": "callee_fn", "file": "impl.py", "confidence": 0.9}],
        })

        def mock_score(symbol, path):
            if symbol == "callee_fn":
                return (0.7, "high_value")
            return (0.5, "default")

        result = propagate_scores(scores, graph, extract_and_score_fn=mock_score)

        caller = result["caller"]
        assert caller.propagated_score > 0.2
        # boost = 0.7 * 0.35 = 0.245 → propagated ≈ 0.445
        assert caller.propagated_score > 0.4

    def test_propagation_capped_at_1(self):
        """Propagated score should never exceed 1.0."""
        scores = {
            "sym": (0.9, "already_high"),
        }
        graph = _make_graph({
            "sym": [{"symbol": "callee", "file": "f.py", "confidence": 1.0}],
            "callee": [],
        })
        # callee not in scores and no callback → no boost possible
        result = propagate_scores(scores, graph)
        assert result["sym"].propagated_score <= 1.0

    def test_real_kp6_scenario(self):
        """KP6: parse_definitions delegates to _walk_definitions."""
        scores = {
            "parse_definitions": (0.315, "direct:0.00 | flow:0.10 | role:0.65 | sem:0.75"),
            "_walk_definitions": (0.619, "direct:0.10 | flow:0.55 | role:1.00 | sem:1.00"),
        }
        graph = _make_graph({
            "parse_definitions": [
                {"symbol": "_walk_definitions", "file": "code_structure_utils.py", "confidence": 0.95},
            ],
        })

        result = propagate_scores(scores, graph)

        parse = result["parse_definitions"]
        walk = result["_walk_definitions"]

        # parse boosted by walk
        assert parse.propagated_score > parse.base_score
        # walk still higher (it IS the edit target)
        assert walk.propagated_score >= parse.propagated_score
        # but gap narrows — that's OK, both should be considered
        assert walk.propagated_score > 0.6


class TestMutatingCalleeBoost:
    """is_mutating=True on a callee edge should produce a larger boost."""

    def test_mutating_callee_boosts_more_than_pure(self):
        """db.save() caller should score higher than pure-call equivalent."""
        scores_mutating = {
            "create_user": (0.2, "base"),
            "db_save": (0.6, "high"),
        }
        graph_mutating = _make_graph({
            "create_user": [
                {"symbol": "db_save", "file": "db.py", "confidence": 1.0, "is_mutating": True},
            ],
        })

        scores_pure = {
            "get_user": (0.2, "base"),
            "db_fetch": (0.6, "high"),
        }
        graph_pure = _make_graph({
            "get_user": [
                {"symbol": "db_fetch", "file": "db.py", "confidence": 1.0, "is_mutating": False},
            ],
        })

        result_mut = propagate_scores(scores_mutating, graph_mutating)
        result_pure = propagate_scores(scores_pure, graph_pure)

        # Both start at 0.2; mutating callee boost must exceed pure callee boost
        mut_score = result_mut["create_user"].propagated_score
        pure_score = result_pure["get_user"].propagated_score
        assert mut_score > pure_score, (
            f"mutating boost {mut_score:.3f} should exceed pure boost {pure_score:.3f}"
        )

    def test_mutating_boost_magnitude(self):
        """Mutating boost = 0.6 * 0.52 = 0.312 → propagated ≈ 0.512."""
        scores = {
            "save_order": (0.2, "base"),
            "db_execute": (0.6, "high"),
        }
        graph = _make_graph({
            "save_order": [
                {"symbol": "db_execute", "file": "db.py", "confidence": 1.0, "is_mutating": True},
            ],
        })

        result = propagate_scores(scores, graph)
        save = result["save_order"]

        # boost = 0.6 * 0.52 = 0.312
        expected = 0.2 + 0.6 * 0.52
        assert abs(save.propagated_score - expected) < 0.01

    def test_mutating_tag_in_reason(self):
        """[mut] tag should appear in reason when chain is mutating."""
        scores = {
            "write_record": (0.1, "base"),
            "session_commit": (0.5, "high"),
        }
        graph = _make_graph({
            "write_record": [
                {"symbol": "session_commit", "file": "db.py", "is_mutating": True},
            ],
        })

        result = propagate_scores(scores, graph)
        assert "[mut]" in result["write_record"].reason

    def test_non_mutating_no_tag(self):
        """Pure callee should NOT add [mut] tag to reason."""
        scores = {
            "get_user": (0.2, "base"),
            "db_query": (0.6, "high"),
        }
        graph = _make_graph({
            "get_user": [
                {"symbol": "db_query", "file": "db.py", "is_mutating": False},
            ],
        })

        result = propagate_scores(scores, graph)
        assert "[mut]" not in result["get_user"].reason

    def test_deep_chain_mutating_propagates(self):
        """A → B → C (mutating): A should get high boost via chain."""
        scores = {
            "handle_request": (0.1, "base"),
            "persist_data": (0.3, "mid"),
            "db_save": (0.8, "high"),
        }
        graph = _make_graph({
            "handle_request": [
                {"symbol": "persist_data", "file": "svc.py", "is_mutating": False},
            ],
            "persist_data": [
                {"symbol": "db_save", "file": "db.py", "is_mutating": True},
            ],
        })

        result = propagate_scores(scores, graph)
        h = result["handle_request"]
        # Should be significantly boosted via the chain
        assert h.propagated_score > h.base_score + 0.2


class TestExpandCandidates:
    """Test call chain candidate expansion."""

    def test_basic_expansion(self):
        """Should find callees not in current set."""
        current = {"parse_definitions"}
        graph = _make_graph({
            "parse_definitions": [
                {"symbol": "_walk_definitions", "file": "utils.py", "confidence": 0.9},
            ],
        })

        expanded = expand_candidates(current, graph)
        assert ("_walk_definitions", "utils.py") in expanded

    def test_no_duplicates(self):
        """Should not return symbols already in current set."""
        current = {"A", "B"}
        graph = _make_graph({
            "A": [{"symbol": "B", "file": "f.py", "confidence": 0.9}],
        })

        expanded = expand_candidates(current, graph)
        assert not any(sym == "B" for sym, _ in expanded)

    def test_depth_2_expansion(self):
        """Should follow 2 levels deep: A → B → C."""
        current = {"A"}
        graph = _make_graph({
            "A": [{"symbol": "B", "file": "f.py", "confidence": 0.9}],
            "B": [{"symbol": "C", "file": "f.py", "confidence": 0.9}],
        })

        expanded = expand_candidates(current, graph)
        syms = {s for s, _ in expanded}
        assert "B" in syms
        assert "C" in syms

    def test_file_filter(self):
        """Should only return candidates in target files."""
        current = {"A"}
        graph = _make_graph({
            "A": [
                {"symbol": "B", "file": "target.py", "confidence": 0.9},
                {"symbol": "C", "file": "other.py", "confidence": 0.9},
            ],
        })

        expanded = expand_candidates(current, graph, target_files=["target.py"])
        syms = {s for s, _ in expanded}
        assert "B" in syms
        assert "C" not in syms

    def test_no_graph(self):
        """No graph context → empty expansion."""
        expanded = expand_candidates({"A"}, None)
        assert expanded == []

    def test_cycle_safe(self):
        """Cycles in call graph should not cause infinite loop."""
        current = {"A"}
        graph = _make_graph({
            "A": [{"symbol": "B", "file": "f.py", "confidence": 0.9}],
            "B": [{"symbol": "A", "file": "f.py", "confidence": 0.9}],  # cycle!
        })

        expanded = expand_candidates(current, graph)
        # Should finish without hanging
        assert len(expanded) <= 2
