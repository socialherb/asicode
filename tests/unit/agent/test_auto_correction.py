"""Tests for auto_correction — focusing on SL14 regression fixes
and the new fuzzy-match nearest-symbol feature."""
import os
import tempfile
import textwrap

from external_llm.agent.auto_correction import (
    _SIM_AUTO_CORRECT,
    _SIM_HINT_ONLY,
    _comment_bridge_connects_semantic_block,
    auto_correct_operation,
    compute_symbol_similarity,
    resolve_operation_facts,
)
from external_llm.agent.operation_models import Operation, OperationKind

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_py_file(content: str) -> str:
    """Write *content* to a temp .py file; return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


def _make_op(
    kind: OperationKind,
    symbol: str,
    path: str,
    action_hint: str = "",
) -> Operation:
    meta = {}
    if action_hint:
        meta["action_hint"] = action_hint
    return Operation(
        id="t1",
        kind=kind,
        path=path,
        symbol=symbol,
        intent="test",
        metadata=meta,
    )


# ── compute_symbol_similarity ────────────────────────────────────────────────

class TestComputeSymbolSimilarity:
    def test_identical_symbols(self):
        assert compute_symbol_similarity("FOO_BAR", "FOO_BAR") == 1.0

    def test_empty_symbols(self):
        assert compute_symbol_similarity("", "FOO") == 0.0
        assert compute_symbol_similarity("FOO", "") == 0.0

    def test_completely_different(self):
        score = compute_symbol_similarity("lineage_ratio", "http_timeout")
        assert score < 0.25, f"Expected low score, got {score}"

    def test_case_variation_same_tokens(self):
        # UPPER_SNAKE vs lower_snake should score very high
        score = compute_symbol_similarity("THRESHOLD_ALPHA", "threshold_alpha")
        assert score >= 0.70, f"Case variation should score high, got {score}"

    def test_leading_underscore_ignored(self):
        score = compute_symbol_similarity("_process_data", "process_data")
        assert score >= 0.70, f"Leading _ should be ignored, got {score}"
        assert score >= 0.85, f"Leading _ should be ignored, got {score}"

    def test_prefix_token_bonus(self):
        # ADAPT is a prefix of ADAPTATION — should get a bonus
        score_with_prefix = compute_symbol_similarity("ADAPT_ALPHA", "ADAPTATION_ALPHA")
        score_no_prefix = compute_symbol_similarity("ADAPT_ALPHA", "UNRELATED_BETA")
        assert score_with_prefix > score_no_prefix

    def test_close_rename_target(self):
        # Threshold_ADAPT_ALPHA vs THRESHOLD_ADAPTATION_ALPHA — near match
        score = compute_symbol_similarity(
            "THRESHOLD_ADAPT_ALPHA", "THRESHOLD_ADAPTATION_ALPHA"
        )
        assert score >= 0.50, f"Close name should score ≥0.50, got {score}"

    def test_sl14_divergent_names(self):
        # THRESHOLD_ADAPT_ALPHA vs _ADAPTATION_ALPHA — genuinely different
        # Expect moderate similarity but below auto-correct threshold
        score = compute_symbol_similarity("THRESHOLD_ADAPT_ALPHA", "_ADAPTATION_ALPHA")
        # Should not auto-correct (too different)
        assert score < _SIM_AUTO_CORRECT, (
            f"Divergent names should be below auto-correct threshold ({_SIM_AUTO_CORRECT}), "
            f"got {score}"
        )


# ── resolve_operation_facts: nearest_symbols ────────────────────────────────

class TestNearestSymbols:
    def test_nearest_symbols_populated_when_missing(self):
        """When symbol missing, nearest_symbols should be populated."""
        src = "THRESHOLD_ADAPT_ALPHA = 0.10\nOTHER_CONST = 1\n"
        path = _make_py_file(src)
        try:
            op = _make_op(
                OperationKind.MODIFY_SYMBOL,
                "THRESHOLD_ADAPT_ALPA",  # typo: ALPA vs ALPHA
                path,
            )
            facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
            assert not facts.symbol_exists
            assert len(facts.nearest_symbols) > 0
            names = [n for n, _ in facts.nearest_symbols]
            assert "THRESHOLD_ADAPT_ALPHA" in names, (
                f"Should suggest THRESHOLD_ADAPT_ALPHA for typo ALPA, got {names}"
            )
        finally:
            os.unlink(path)

    def test_nearest_symbols_empty_when_symbol_exists(self):
        """nearest_symbols must be empty when symbol exists."""
        src = "THRESHOLD_ALPHA = 0.10\n"
        path = _make_py_file(src)
        try:
            op = _make_op(OperationKind.MODIFY_SYMBOL, "THRESHOLD_ALPHA", path)
            facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
            assert facts.symbol_exists
            assert facts.nearest_symbols == []
        finally:
            os.unlink(path)

    def test_nearest_symbols_sorted_by_score(self):
        """Candidates should be in descending order of similarity score."""
        src = (
            "THRESH_ALPHA = 0.1\n"
            "ALPHA_CONST = 0.2\n"
            "UNRELATED_VAL = 99\n"
        )
        path = _make_py_file(src)
        try:
            op = _make_op(OperationKind.MODIFY_SYMBOL, "THRESH_ALPA", path)
            facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
            scores = [s for _, s in facts.nearest_symbols]
            assert scores == sorted(scores, reverse=True), "Should be sorted descending"
        finally:
            os.unlink(path)


# ── Rule A: symbol exists → keep ─────────────────────────────────────────────

def test_rule_a_symbol_exists():
    src = "THRESHOLD_ALPHA = 0.10\n"
    path = _make_py_file(src)
    try:
        op = _make_op(OperationKind.MODIFY_SYMBOL, "THRESHOLD_ALPHA", path)
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert facts.symbol_exists
        decision = auto_correct_operation(op, facts)
        assert decision.action == "keep"
        assert decision.rationale == "symbol_exists"
    finally:
        os.unlink(path)


# ── Rule B.4 — default (no action_hint): missing symbol → skip + replan ───

def test_rule_b4_missing_symbol_no_hint():
    """B.4-G: non-refactor modify_symbol with missing symbol → skip + replan.

    The replan_hint should mention the available symbols so the planner
    does not re-hallucinate the same wrong name (hallucination loop fix).
    """
    src = "OTHER_CONST = 42\n"
    path = _make_py_file(src)
    try:
        op = _make_op(OperationKind.MODIFY_SYMBOL, "NONEXISTENT", path)
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        assert facts.creatable
        decision = auto_correct_operation(op, facts)
        assert decision.action == "skip"
        assert decision.rationale == "symbol_not_found_hallucinated"
        assert decision.should_replan
        # replan_hint must reference the file or symbol context
        assert op.path in decision.replan_hint or "not found" in decision.replan_hint
        # Since 'NONEXISTENT' has zero token overlap with 'OTHER_CONST',
        # nearest_symbols will be empty → hint should make that clear
        if not facts.nearest_symbols:
            assert "symbol" in decision.replan_hint.lower()
    finally:
        os.unlink(path)


def test_rule_b4_replan_hint_includes_available_symbols():
    """B.4-G: when nearest_symbols exist, replan_hint must list them.

    This prevents the planner from re-hallucinating the same symbol
    on the next replan cycle (hallucination loop fix).
    """
    # Use a close-but-not-exact name so nearest_symbols has an entry
    src = "FAST_PATH_ENABLED = True\nOTHER_CONST = 42\n"
    path = _make_py_file(src)
    try:
        op = _make_op(OperationKind.MODIFY_SYMBOL, "FAST_PATH_ENABLE", path)
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        assert facts.creatable
        assert facts.nearest_symbols, "Should have at least 'FAST_PATH_ENABLED' in nearest_symbols"
        decision = auto_correct_operation(op, facts)
        assert decision.action == "skip"
        assert decision.rationale == "symbol_not_found_hallucinated"
        assert decision.should_replan
        # The replan_hint must include the actual symbol name that exists
        assert "FAST_PATH_ENABLED" in decision.replan_hint, (
            f"replan_hint should list 'FAST_PATH_ENABLED', got: {decision.replan_hint}"
        )
    finally:
        os.unlink(path)


# ── SL14 regression: rename/refactor with missing symbol must NOT create ─────

def test_rule_b4_refactor_missing_symbol_no_insert():
    """Rename op targeting a non-existent symbol must skip+replan, not insert."""
    src = "_ADAPTATION_ALPHA = 0.10\n"
    path = _make_py_file(src)
    try:
        # User asked to rename THRESHOLD_ADAPT_ALPHA (doesn't exist) → wrong name
        # These two names are too different for auto-correct
        op = _make_op(
            OperationKind.MODIFY_SYMBOL,
            "THRESHOLD_ADAPT_ALPHA",
            path,
            action_hint="refactor",
        )
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        assert facts.creatable  # no parent → would normally trigger B.4

        decision = auto_correct_operation(op, facts)
        # Must NOT silently create a new symbol (too dissimilar for auto-correct)
        assert decision.action in ("skip", "rewrite"), (
            f"Expected skip or auto-correct rewrite, got {decision.action!r}"
        )
        if decision.action == "skip":
            assert decision.rationale == "refactor_symbol_not_found"
            assert decision.should_replan
        else:
            # auto-corrected → must be MODIFY_SYMBOL (not insert_after)
            assert decision.corrected_kind == OperationKind.MODIFY_SYMBOL.value
            assert decision.rationale == "refactor_nearest_symbol_autocorrect"
    finally:
        os.unlink(path)


def test_rule_b4_rename_missing_symbol_no_insert():
    """Rename hint is also covered."""
    src = "SOME_CONST = 1\n"
    path = _make_py_file(src)
    try:
        op = _make_op(
            OperationKind.MODIFY_SYMBOL,
            "MISSING_CONST",
            path,
            action_hint="rename",
        )
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        decision = auto_correct_operation(op, facts)
        assert decision.action in ("skip", "rewrite")
    finally:
        os.unlink(path)


# ── B.4-R: fuzzy match auto-correct for refactor ops ────────────────────────

def test_fuzzy_auto_correct_typo():
    """A one-character typo in constant name should auto-correct."""
    src = "THRESHOLD_ADAPT_ALPHA = 0.10\nOTHER = 1\n"
    path = _make_py_file(src)
    try:
        op = _make_op(
            OperationKind.MODIFY_SYMBOL,
            "THRESHOLD_ADAPT_ALPA",  # typo: missing H
            path,
            action_hint="refactor",
        )
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        # Should find THRESHOLD_ADAPT_ALPHA as a near match
        assert len(facts.nearest_symbols) > 0
        best_name, best_score = facts.nearest_symbols[0]
        assert best_name == "THRESHOLD_ADAPT_ALPHA"

        decision = auto_correct_operation(op, facts)
        if best_score >= _SIM_AUTO_CORRECT:
            assert decision.action == "rewrite"
            assert decision.corrected_kind == OperationKind.MODIFY_SYMBOL.value
            assert decision.corrected_symbol == "THRESHOLD_ADAPT_ALPHA"
            assert decision.rationale == "refactor_nearest_symbol_autocorrect"
        else:
            # Below threshold: skip with hint
            assert decision.action == "skip"
            assert "THRESHOLD_ADAPT_ALPHA" in decision.replan_hint
    finally:
        os.unlink(path)


def test_fuzzy_hint_only_moderate_match():
    """A moderately similar name should produce a hint but not auto-correct."""
    src = "ADAPT_ALPHA_SCORE = 0.10\nUNRELATED = 1\n"
    path = _make_py_file(src)
    try:
        op = _make_op(
            OperationKind.MODIFY_SYMBOL,
            "ADAPT_ALPHA_THRESHOLD",  # related but not the same
            path,
            action_hint="refactor",
        )
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        if not facts.symbol_exists and facts.nearest_symbols:
            best_name, best_score = facts.nearest_symbols[0]
            decision = auto_correct_operation(op, facts)
            if _SIM_HINT_ONLY <= best_score < _SIM_AUTO_CORRECT:
                assert decision.action == "skip"
                assert best_name in decision.replan_hint
            # Either auto-corrected or hint — both are valid
    finally:
        os.unlink(path)


def test_replan_hint_contains_available_symbols():
    """When no good match, replan_hint should list available symbols."""
    src = "UNRELATED_X = 1\nUNRELATED_Y = 2\n"
    path = _make_py_file(src)
    try:
        op = _make_op(
            OperationKind.MODIFY_SYMBOL,
            "COMPLETELY_DIFFERENT_ZZZZ",
            path,
            action_hint="refactor",
        )
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert not facts.symbol_exists
        decision = auto_correct_operation(op, facts)
        assert decision.action == "skip"
        assert decision.should_replan
        # Hint should mention the file or some context
        assert op.path in decision.replan_hint or "not found" in decision.replan_hint
    finally:
        os.unlink(path)


# ── _comment_bridge_connects_semantic_block ───────────────────────────────────

class TestCommentBridgeConnectsSemanticBlock:
    """Unit tests for the 5 safeguards in _comment_bridge_connects_semantic_block."""

    # ── happy path ──────────────────────────────────────────────────────────

    def test_basic_bridge_passes(self):
        """Code added right after a symbol via blank+comment bridge → True.

        Buffer note: range(start, end+4) means sym end=3 covers lines 1-6.
        Semantic addition must be at line 7+ to be out-of-range.
        """
        after_lines = [
            "def foo():\n",   # 1  sym start
            "    pass\n",     # 2
            "    return 1\n", # 3  sym end (buffer: range(1,7) → lines 1-6 in-range)
            "\n",             # 4  bridge (blank, in buffer)
            "# helper\n",    # 5  bridge (comment, in buffer)
            "\n",             # 6  bridge (blank, in buffer)
            "NEW_CODE = 1\n", # 7  semantic addition (out of range: 7 ∉ range(1,7))
        ]
        sym_ranges = {"foo": (1, 3)}
        added = [(4, "\n"), (5, "# helper\n"), (6, "\n"), (7, "NEW_CODE = 1\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is True

    def test_no_semantic_out_returns_false(self):
        """All additions inside symbol range → no out-of-range → False."""
        after_lines = ["def foo():\n", "    x = 1\n", "    return x\n"]
        sym_ranges = {"foo": (1, 3)}
        added = [(2, "    x = 1\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False

    # ── safeguard (1): imports break the bridge ──────────────────────────────

    def test_import_in_added_block_is_semantic(self):
        """Import added between symbol end and new code counts as semantic → False."""
        after_lines = [
            "def foo():\n",   # 1
            "    pass\n",     # 2
            "\n",             # 3  bridge blank
            "import os\n",   # 4  added import — counts as semantic out-of-range
            "NEW_CODE = 1\n", # 5
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "import os\n"), (5, "NEW_CODE = 1\n")]
        # import os at ln=4 is out-of-range and is semantic → breaks bridge
        result = _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added)
        assert result is False

    # ── safeguard (2): closest symbol is selected ────────────────────────────

    def test_far_symbol_cannot_validate_nearby_addition(self):
        """A distant symbol must not be used to validate an insertion near a closer one.

        Setup:
          sym_A ends at line 2 (close to addition at line 10, bridge=7)
          sym_B ends at line 8 (closest to addition at line 10, bridge=1)
          bridge from sym_B contains existing code → should fail.
        """
        after_lines = [
            "def sym_a():\n",    # 1
            "    pass\n",        # 2  sym_A end
            "\n",                # 3
            "\n",                # 4
            "\n",                # 5
            "\n",                # 6
            "def sym_b():\n",    # 7  sym_B start
            "    return 2\n",    # 8  sym_B end
            "existing_code()\n", # 9  NOT added — existing code in bridge
            "NEW_CODE = 1\n",    # 10 semantic addition
        ]
        sym_ranges = {"sym_a": (1, 2), "sym_b": (7, 8)}
        added = [(10, "NEW_CODE = 1\n")]
        # closest sym is sym_b (bridge len=1: line 9). Line 9 has existing code → False.
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False

    # ── safeguard (3): contiguity — existing code wedged between additions ───

    def test_existing_code_between_semantic_additions_fails(self):
        """Existing non-comment code between two out-of-range additions → False."""
        after_lines = [
            "def foo():\n",      # 1
            "    pass\n",        # 2  sym end
            "\n",                # 3  bridge blank
            "NEW_A = 1\n",       # 4  first semantic addition
            "existing_code()\n", # 5  NOT added — existing code wedged in
            "NEW_B = 2\n",       # 6  second semantic addition
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "NEW_A = 1\n"), (6, "NEW_B = 2\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False

    # ── safeguard (4): GAP_THRESHOLD — existing blank lines separate clusters ──
    #
    # Key: added blank lines RESET the counter; only existing (non-added) blanks
    # accumulate it.  Check is `> GAP_THRESHOLD` (=3), so 4+ existing blanks fire.
    # Buffer: sym end=2 → range(1,6) covers lines 1-5; semantics at 6+ are out-of-range.

    def test_gap_threshold_four_existing_blanks_separates_clusters(self):
        """4 existing blank lines between out-of-range semantics → second cluster rejected.

        With only 2 semantics the inner loop exits without firing; the post-loop
        `if _found_gap_break: return False` guard catches it.
        """
        after_lines = [
            "def foo():\n",  # 1  sym start
            "    pass\n",    # 2  sym end (buffer: lines 1-5)
            "\n",            # 3  bridge (added, in buffer)
            "\n",            # 4  bridge (added, in buffer)
            "\n",            # 5  bridge (added, in buffer)
            "NEW_A = 1\n",   # 6  first semantic (out of range)
            "\n",            # 7  existing gap 1 (NOT added)
            "\n",            # 8  existing gap 2 (NOT added)
            "\n",            # 9  existing gap 3 (NOT added)
            "\n",            # 10 existing gap 4 → _consecutive_gap=4 > 3 → break
            "NEW_B = 2\n",   # 11 second semantic (separate cluster → rejected)
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "\n"), (5, "\n"), (6, "NEW_A = 1\n"), (11, "NEW_B = 2\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False

    def test_gap_threshold_two_existing_blanks_is_ok(self):
        """2 existing blank lines between out-of-range semantics → same cluster, True."""
        after_lines = [
            "def foo():\n",  # 1  sym start
            "    pass\n",    # 2  sym end (buffer: lines 1-5)
            "\n",            # 3  bridge (added, in buffer)
            "\n",            # 4  bridge (added, in buffer)
            "\n",            # 5  bridge (added, in buffer)
            "NEW_A = 1\n",   # 6  first semantic (out of range)
            "\n",            # 7  existing gap 1 (NOT added)
            "\n",            # 8  existing gap 2 (NOT added; 2 ≤ threshold)
            "NEW_B = 2\n",   # 9  second semantic (same cluster → ok)
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "\n"), (5, "\n"), (6, "NEW_A = 1\n"), (9, "NEW_B = 2\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is True

    # ── safeguard (5): indentation guard for top-level symbol ────────────────

    def test_indented_code_after_toplevel_symbol_fails(self):
        """Indented addition after a top-level (col-0) function → False."""
        after_lines = [
            "def foo():\n",       # 1  top-level sym (col 0)
            "    pass\n",         # 2  sym end
            "\n",                 # 3  bridge
            "    indented = 1\n", # 4  addition: indented → misplaced body
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "    indented = 1\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False

    def test_toplevel_addition_after_toplevel_symbol_passes(self):
        """Col-0 addition after a top-level function → True.

        Buffer: sym end=2 → range(1,6) covers lines 1-5. Addition must be at 6+.
        """
        after_lines = [
            "def foo():\n",    # 1  top-level sym (col 0)
            "    pass\n",      # 2  sym end (buffer: lines 1-5)
            "\n",              # 3  bridge (added, in buffer)
            "\n",              # 4  bridge (added, in buffer)
            "\n",              # 5  bridge (added, in buffer)
            "NEW_CONST = 1\n", # 6  addition: col 0, out of range → valid
        ]
        sym_ranges = {"foo": (1, 2)}
        added = [(3, "\n"), (4, "\n"), (5, "\n"), (6, "NEW_CONST = 1\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is True

    def test_bridge_exceeds_max_bridge_fails(self):
        """Bridge longer than MAX_BRIDGE (20) lines → False."""
        # sym ends at line 1, first semantic at line 23 → bridge len = 21
        after_lines = ["def foo(): pass\n"] + ["\n"] * 21 + ["NEW = 1\n"]
        sym_ranges = {"foo": (1, 1)}
        added = [(23, "NEW = 1\n")]
        assert _comment_bridge_connects_semantic_block(after_lines, sym_ranges, added) is False


# ── Module-level constant rename (happy path) ────────────────────────────────

def test_rule_a_module_level_constant_exists():
    """AST should find a module-level constant assignment."""
    src = "THRESHOLD_ADAPT_ALPHA = 0.10\n"
    path = _make_py_file(src)
    try:
        op = _make_op(OperationKind.MODIFY_SYMBOL, "THRESHOLD_ADAPT_ALPHA", path)
        facts = resolve_operation_facts(op, repo_root=os.path.dirname(path))
        assert facts.symbol_exists, "Module-level constant should be found"
        decision = auto_correct_operation(op, facts)
        assert decision.action == "keep"
    finally:
        os.unlink(path)
