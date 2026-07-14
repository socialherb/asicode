"""Tests for external_llm/analysis/contradictory_logic_scanner.py."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from external_llm.analysis.contradictory_logic_scanner import (
    ContradictoryCandidate,
    _assignment_target_overlaps,
    _branch_body_end,
    _collect_if_elif_chain,
    _extract_condition_names,
    _get_enclosing_symbol_name,
    _has_name_mutation,
    _is_constant_false,
    _check_boolop_tautology,
    scan_contradictory_logic,
)


def _make_py_file(source: str) -> str:
    """Write source to a temp .py file and return its absolute path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    )
    tmp.write(textwrap.dedent(source))
    tmp.close()
    return tmp.name


# ── Constant condition detection ─────────────────────────────────────────


def test_constant_true_condition():
    """if True: is detected as constant_true_condition."""
    src = _make_py_file("""\
        if True:
            print("always")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "constant_true_condition" in kinds
    Path(src).unlink()


def test_constant_false_condition():
    """if False: is detected as constant_false_condition."""
    src = _make_py_file("""\
        if False:
            print("never")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "constant_false_condition" in kinds
    Path(src).unlink()


def test_while_false():
    """while False: is detected as constant_false_condition."""
    src = _make_py_file("""\
        while False:
            print("never")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "constant_false_condition" in kinds
    Path(src).unlink()


def test_constant_zero_condition():
    """if 0: and if 0.0: are detected as constant_zero_condition."""
    src = _make_py_file("""\
        if 0:
            print("never")
        if 0.0:
            print("also never")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    # Both constant_false_condition (0 and 0.0 are falsy)
    assert len(candidates) >= 2
    Path(src).unlink()


def test_constant_empty_string():
    """if '': and if b'': are detected."""
    src = _make_py_file("""\
        if "":
            print("never")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) >= 1
    Path(src).unlink()


def test_constant_one_is_true():
    """if 1: is detected as constant_true_condition."""
    src = _make_py_file("""\
        if 1:
            print("always")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "constant_true_condition" in kinds
    Path(src).unlink()


def test_non_constant_condition_not_flagged():
    """Normal conditions are not flagged."""
    src = _make_py_file("""\
        x = 42
        if x > 0:
            print("normal")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) == 0
    Path(src).unlink()


# ── Contradictory boolean detection ──────────────────────────────────────


def test_contradictory_and():
    """x and not x pattern."""
    src = _make_py_file("""\
        def check(x):
            if x and not x:
                print("never")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "contradictory_boolean" in kinds
    Path(src).unlink()


def test_always_true_or():
    """x or not x pattern."""
    src = _make_py_file("""\
        def check(x):
            if x or not x:
                print("always")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "always_true_boolean" in kinds
    Path(src).unlink()


def test_non_contradictory_and_not_flagged():
    """x and not y (different names) is NOT contradictory."""
    src = _make_py_file("""\
        def check(x, y):
            if x and not y:
                print("normal")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) == 0
    Path(src).unlink()


# ── Always-false assert ──────────────────────────────────────────────────


def test_assert_false_detected():
    """assert False is detected as always_false_assert."""
    src = _make_py_file("""\
        def validate():
            assert False, "not implemented"
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "always_false_assert" in kinds
    Path(src).unlink()


def test_assert_condition_not_flagged():
    """Normal asserts are not flagged."""
    src = _make_py_file("""\
        def validate(x):
            assert x > 0, "positive required"
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) == 0
    Path(src).unlink()


# ── Duplicate condition ──────────────────────────────────────────────────


def test_duplicate_condition():
    """Same condition checked twice is flagged as duplicate."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                print("positive")
            if x > 0:
                print("positive again")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    # When blocks are close together, scanner may flag as mergeable_condition
    # instead of duplicate_condition (same condition, adjacent blocks).
    assert "duplicate_condition" in kinds or "mergeable_condition" in kinds
    Path(src).unlink()


def test_different_conditions_not_duplicate():
    """Different conditions are not flagged as duplicates."""
    src = _make_py_file("""\
        def check(x, y):
            if x > 0:
                print("x")
            if y > 0:
                print("y")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "duplicate_condition" not in kinds
    Path(src).unlink()


# ── Pattern interaction: inside functions ────────────────────────────────


def test_patterns_inside_function():
    """All patterns detected inside function bodies."""
    src = _make_py_file("""\
        def check(x):
            if True:
                return 1
            if False:
                return 2
            if x and not x:
                return 3
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    # Need at least 2: constant_true + (constant_false or contradictory_boolean)
    assert len(candidates) >= 2
    Path(src).unlink()


def test_patterns_inside_class():
    """Patterns detected inside class methods."""
    src = _make_py_file("""\
        class Processor:
            def process(self, x):
                if False:
                    return None
                if x and not x:
                    return None
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) >= 1
    Path(src).unlink()


def test_patterns_in_elif():
    """Elif branches are also checked."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            elif True:
                pass
            else:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    assert "constant_true_condition" in kinds
    Path(src).unlink()


# ── Edge cases ───────────────────────────────────────────────────────────


def test_max_per_file_enforced():
    """Respect max_per_file limit."""
    src = _make_py_file("""\
        if True: pass
        if False: pass
        if 0: pass
        if 1: pass
        if "": pass
        assert False
        if 0: pass
        if False: pass
        if True: pass
        if 0: pass
        if 0: pass
        if 0: pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_per_file=3)
    assert len(candidates) == 3
    Path(src).unlink()


def test_non_py_file_skipped():
    """Non-.py files are skipped (pre-filtered by ScannerRegistry)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write("if True: pass\n")
    tmp.close()
    from external_llm.agent.scanner_registry import get_registry
    reg = get_registry()
    result = reg.run("contradictory_logic_scanner", file_paths=[tmp.name])
    assert not result.candidates_raw, f"Expected 0 candidates, got {len(result.candidates_raw)}"
    Path(tmp.name).unlink()


def test_missing_file_skipped():
    """Non-existent files are skipped without error."""
    candidates = scan_contradictory_logic(repo_root="", file_paths=["/nonexistent/file.py"])
    assert not candidates


def test_syntax_error_file_skipped():
    """Files with syntax errors are skipped."""
    src = _make_py_file("""\
        if True
            print("missing colon")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert not candidates
    Path(src).unlink()


# ── Confidence / to_dict ─────────────────────────────────────────────────


def test_constant_condition_confidence():
    """Constant conditions have confidence 1.0."""
    src = _make_py_file("if True: pass\n")
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) == 1
    assert candidates[0].confidence == 1.0
    Path(src).unlink()


def test_contradictory_boolean_confidence():
    """Contradictory booleans have confidence 0.9."""
    src = _make_py_file("def f(x):\n    if x and not x: pass\n")
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    assert len(candidates) >= 1
    contra = [c for c in candidates if c.contradiction_kind == "contradictory_boolean"]
    assert len(contra) >= 1
    assert all(c.confidence == 0.9 for c in contra)
    Path(src).unlink()


def test_candidate_to_dict():
    """ContradictoryCandidate.to_dict() returns correct fields."""
    cand = ContradictoryCandidate(
        file="test.py",
        symbol="my_func",
        contradiction_kind="constant_false_condition",
        lineno=5,
        end_lineno=8,
        detail="condition is always False",
        confidence=1.0,
    )
    d = cand.to_dict()
    assert d["file"] == "test.py"
    assert d["symbol"] == "my_func"
    assert d["contradiction_kind"] == "constant_false_condition"
    assert d["lineno"] == 5
    assert d["end_lineno"] == 8
    assert d["detail"] == "condition is always False"
    assert d["confidence"] == 1.0


# ── Comprehensive integration ────────────────────────────────────────────


def test_all_patterns_in_one_file():
    """All 7 patterns detected in a single file."""
    src = _make_py_file("""\
        from __future__ import annotations

        def check(x):
            if True:                    # constant_true_condition
                pass
            if False:                   # constant_false_condition
                pass
            if 0:                       # constant_false (zero)
                pass
            if "":                      # constant_false (empty)
                pass
            if x and not x:             # contradictory_boolean
                pass
            if x or not x:              # always_true_boolean
                pass
            assert False                # always_false_assert
            if x > 0:
                pass
            if x > 0:                   # duplicate_condition
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in candidates}
    expected = {
        "constant_true_condition",
        "constant_false_condition",
        "contradictory_boolean",
        "always_true_boolean",
        "always_false_assert",
    }
    # Adjacent same-condition blocks may be flagged as mergeable_condition
    # or duplicate_condition depending on distance.
    expected.add(
        "mergeable_condition" if "mergeable_condition" in kinds
        else "duplicate_condition"
    )
    missing = expected - kinds
    assert not missing, f"Missing patterns: {missing}"
    Path(src).unlink()


# ── Internal helpers ─────────────────────────────────────────────────────


def test_is_constant_false():
    """_is_constant_false correctly identifies falsy constants."""
    import ast
    assert _is_constant_false(ast.Constant(value=False)) is True
    assert _is_constant_false(ast.Constant(value=True)) is False
    assert _is_constant_false(ast.Constant(value=0)) is True
    assert _is_constant_false(ast.Constant(value=0.0)) is True
    assert _is_constant_false(ast.Constant(value=1)) is False
    assert _is_constant_false(ast.Constant(value="")) is True
    assert _is_constant_false(ast.Constant(value="hello")) is None
    assert _is_constant_false(ast.Name(id="x", ctx=ast.Load())) is None


def test_is_contradictory_and():
    """_check_boolop_tautology detects x and not x (And)."""
    import ast
    tree = ast.parse("x and not x")
    expr = tree.body[0].value  # type: ignore
    results = _check_boolop_tautology(expr)
    assert len(results) == 1
    assert results[0][0] == "contradictory_boolean"
    assert "contradictory" in results[0][1]


def test_is_contradictory_and_negative():
    """_check_boolop_tautology returns empty for non-contradictory."""
    import ast
    tree = ast.parse("x and not y")
    expr = tree.body[0].value  # type: ignore
    assert _check_boolop_tautology(expr) == []


def test_is_contradictory_and_not_x_and_compare():
    """not x and failed > 0 — only 'not x', no bare 'x' — must NOT be flagged."""
    import ast
    # Real-world pattern: `not _any_write_succeeded and failed > 0`
    tree = ast.parse("not x and failed > 0")
    expr = tree.body[0].value  # type: ignore
    assert _check_boolop_tautology(expr) == []


def test_contradictory_and_not_flagged_in_scan():
    """not x and y > 0 pattern in real code is NOT flagged as contradictory."""
    src = _make_py_file("""\
        def check(x, failed):
            if not x and failed > 0:
                print("real pattern")
            elif not x and failed == 0:
                print("another real branch")
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    contra = [c for c in candidates if c.contradiction_kind == "contradictory_boolean"]
    assert not contra, f"False positives: {[c.detail for c in contra]}"
    Path(src).unlink()


def test_is_always_true_or():
    """_check_boolop_tautology detects x or not x (Or)."""
    import ast
    tree = ast.parse("x or not x")
    expr = tree.body[0].value  # type: ignore
    results = _check_boolop_tautology(expr)
    assert len(results) == 1
    assert results[0][0] == "always_true_boolean"
    assert "redundant" in results[0][1]


def test_is_always_true_or_negative():
    """_check_boolop_tautology returns empty for non-always-true."""
    import ast
    tree = ast.parse("x or not y")
    expr = tree.body[0].value  # type: ignore
    assert _check_boolop_tautology(expr) == []


def test_get_enclosing_symbol_name():
    """_get_enclosing_symbol_name finds correct enclosing symbol.

    Note: ast.walk is preorder, so the outermost match wins.
    """
    import ast
    tree = ast.parse("""\
class MyClass:
    def method(self):
        pass
""")
    name = _get_enclosing_symbol_name(tree, 3)  # inside method body
    assert name == "MyClass"  # ast.walk visits class before function (preorder)
    name = _get_enclosing_symbol_name(tree, 1)  # class level
    assert name == "MyClass"
    name = _get_enclosing_symbol_name(tree, 999)  # no match
    assert name == "<module>"


# ── _collect_if_elif_chain ───────────────────────────────────────────────


def test_collect_if_elif_chain_simple_if():
    """Single if with no else returns one-element chain and empty else_body."""
    import ast
    tree = ast.parse("if x > 0:\n    pass\n")
    if_node = tree.body[0]
    chain, else_body = _collect_if_elif_chain(if_node)
    assert len(chain) == 1
    assert else_body == []


def test_collect_if_elif_chain_if_else():
    """if/else returns one-element chain and non-empty else_body."""
    import ast
    tree = ast.parse("if x > 0:\n    pass\nelse:\n    pass\n")
    if_node = tree.body[0]
    chain, else_body = _collect_if_elif_chain(if_node)
    assert len(chain) == 1
    assert len(else_body) > 0


def test_collect_if_elif_chain_elif():
    """if/elif/else flattens to two-element chain plus else_body."""
    import ast
    src = "if a:\n    pass\nelif b:\n    pass\nelse:\n    pass\n"
    tree = ast.parse(src)
    if_node = tree.body[0]
    chain, else_body = _collect_if_elif_chain(if_node)
    assert len(chain) == 2
    assert len(else_body) > 0


def test_collect_if_elif_chain_triple_elif():
    """Three-branch if/elif/elif chain flattens correctly."""
    import ast
    src = "if a:\n    pass\nelif b:\n    pass\nelif c:\n    pass\n"
    tree = ast.parse(src)
    if_node = tree.body[0]
    chain, else_body = _collect_if_elif_chain(if_node)
    assert len(chain) == 3
    assert else_body == []


# ── Scope isolation: conditions in branches must not cross-contaminate ────


def test_same_condition_in_both_branches_not_duplicate():
    """if A: if x>0: ... else: if x>0: ... — x>0 is in different branches, NOT duplicate."""
    src = _make_py_file("""\
        def check(a, x):
            if a:
                if x > 0:
                    pass
            else:
                if x > 0:
                    pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"False positive duplicates: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_same_condition_sequential_is_duplicate():
    """if x>0: ... if x>0: ... at same flat level IS a real duplicate."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Expected duplicate or mergeable condition to be detected"
    Path(src).unlink()


def test_elif_duplicate_condition_detected():
    """if x>0: ... elif x>0: ... — duplicate condition in elif chain IS flagged."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            elif x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Expected duplicate_condition in if/elif chain"
    Path(src).unlink()


def test_try_except_branch_isolation():
    """Conditions inside try and except are separate scopes — no cross-contamination."""
    src = _make_py_file("""\
        def check(x):
            try:
                if x > 0:
                    pass
            except ValueError:
                if x > 0:
                    pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"False positive across try/except: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_real_world_not_x_and_compare_elif_chain():
    """Real-world agent_loop pattern: not x and y > 0 / not x and y == 0 in elif chain."""
    src = _make_py_file("""\
        def execute(write_ok, failed):
            if write_ok:
                note = "ok"
            elif not write_ok and failed > 0:
                note = "failed"
            elif not write_ok and failed == 0:
                note = "noop"
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src])
    contra = [c for c in candidates if c.contradiction_kind == "contradictory_boolean"]
    assert not contra, f"False positive contradictory_boolean: {[c.detail for c in contra]}"
    Path(src).unlink()


# ── max_dup_distance: distance-based filtering ───────────────────────────


def test_duplicate_within_distance_flagged():
    """Duplicate conditions within max_dup_distance are flagged."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            # a few lines later
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=10)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Expected duplicate_condition within distance threshold"
    Path(src).unlink()


def test_duplicate_beyond_distance_not_flagged():
    """Duplicate conditions beyond max_dup_distance are suppressed (likely different sections)."""
    lines = ["def check(x):"]
    lines.append("    if x > 0:")
    lines.append("        pass")
    # add 200 filler lines
    for i in range(200):
        lines.append(f"    y = {i}")
    lines.append("    if x > 0:")   # same condition, 200+ lines later
    lines.append("        pass")
    src = _make_py_file("\n".join(lines) + "\n")
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"False positive beyond distance: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_nearest_prior_tracked_for_distance():
    """Distance is measured from the NEAREST prior occurrence, not the first."""
    # condition at line ~2, ~4, ~6 — distance between each pair is ~2 lines
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            if x > 0:
                pass
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=5)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    # Both the 2nd and 3rd occurrences should be flagged (each ≤5 lines from prior)
    assert len(dups) >= 2, f"Expected at least 2 duplicates, got {len(dups)}"
    Path(src).unlink()


# ── Mutation barrier helpers ─────────────────────────────────────────────────


def test_extract_condition_names_simple():
    """Simple Name in condition returns that name, no attr."""
    import ast
    cond = ast.parse("x > 0", mode="eval").body
    names, has_attr = _extract_condition_names(cond)
    assert "x" in names
    assert not has_attr


def test_extract_condition_names_attr():
    """Attribute access sets has_attr_access flag."""
    import ast
    cond = ast.parse("self.config.enabled", mode="eval").body
    names, has_attr = _extract_condition_names(cond)
    assert "self" in names
    assert has_attr


def test_extract_condition_names_complex():
    """Complex expression with multiple names."""
    import ast
    cond = ast.parse("is_small and is_local and not self.is_subagent", mode="eval").body
    names, has_attr = _extract_condition_names(cond)
    assert {"is_small", "is_local", "self"} <= names
    assert has_attr


def test_assignment_target_overlaps_name():
    import ast
    target = ast.parse("x = 1").body[0].targets[0]
    assert _assignment_target_overlaps(target, {"x"})
    assert not _assignment_target_overlaps(target, {"y"})


def test_assignment_target_overlaps_tuple():
    import ast
    stmt = ast.parse("a, b = 1, 2").body[0]
    target = stmt.targets[0]
    assert _assignment_target_overlaps(target, {"a"})
    assert _assignment_target_overlaps(target, {"b"})
    assert not _assignment_target_overlaps(target, {"c"})


def test_has_name_mutation_direct_assign():
    """Direct assignment to watched name is a barrier."""
    import ast
    stmts = ast.parse("x = 10").body
    assert _has_name_mutation(stmts, {"x"}, has_attr_access=False)
    assert not _has_name_mutation(stmts, {"y"}, has_attr_access=False)


def test_has_name_mutation_aug_assign():
    """Augmented assignment (x += 1) is a barrier."""
    import ast
    stmts = ast.parse("x += 1").body
    assert _has_name_mutation(stmts, {"x"}, has_attr_access=False)


def test_has_name_mutation_call_no_attr():
    """Function call is NOT a barrier when condition has no attribute access."""
    import ast
    stmts = ast.parse("some_func()").body
    assert not _has_name_mutation(stmts, {"x"}, has_attr_access=False)


def test_has_name_mutation_call_with_attr():
    """Function call IS a barrier when condition references an attribute."""
    import ast
    stmts = ast.parse("some_method()").body
    assert _has_name_mutation(stmts, {"self"}, has_attr_access=True)


def test_has_name_mutation_nested_assign():
    """Assignment inside nested if block is also detected."""
    import ast
    stmts = ast.parse("if cond:\n    x = 5").body
    assert _has_name_mutation(stmts, {"x"}, has_attr_access=False)


# ── Integration: mutation barrier in scan ───────────────────────────────────


def test_mutation_barrier_prevents_false_duplicate():
    """x reassigned between two checks: NOT a duplicate even if close."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            x = compute_new_value()
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"Should not flag: x was reassigned. Got: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_no_mutation_between_checks_is_duplicate():
    """Directly adjacent identical checks: IS a duplicate (merge candidate).

    Adjacency gate (2026-06-12): 두 if 사이에 문장이 하나라도 끼면 — 비변이
    문장이라도 — 의도적 phase boundary로 간주해 플래그하지 않는다. 따라서
    플래그 대상은 직접 인접한 `if x: A / if x: B` 쌍뿐이다.
    """
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Expected duplicate: adjacent identical checks"
    Path(src).unlink()


def test_intervening_statement_is_phase_boundary_not_duplicate():
    """비변이 문장이 사이에 끼면 adjacency gate가 플래그를 막는다."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                pass
            y = x + 1
            if x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"Adjacency gate should suppress non-adjacent dup. Got: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_attr_condition_with_method_call_barrier():
    """self.attr condition with a method call in between: NOT a duplicate."""
    src = _make_py_file("""\
        def check(self):
            if self.config.enabled:
                pass
            self.update()
            if self.config.enabled:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"Method call should be barrier for attr condition. Got: {[c.detail for c in dups]}"
    Path(src).unlink()


def test_same_chain_always_duplicate_regardless_of_mutation():
    """Same if/elif chain with identical condition: always flagged (unreachable elif)."""
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                x = -1
            elif x > 0:
                pass
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Same-chain duplicate must always be flagged"
    Path(src).unlink()


# ── node_kind and branch range ───────────────────────────────────────────


def test_branch_body_end_empty():
    assert _branch_body_end(5, []) == 5


def test_branch_body_end_with_stmts():
    import ast
    tree = ast.parse("if x:\n    a = 1\n    b = 2\n")
    body = tree.body[0].body
    assert _branch_body_end(1, body) == 3


def test_node_kind_if_from_scan():
    src = _make_py_file("def f(x):\n    if x and not x:\n        pass\n")
    results = scan_contradictory_logic(repo_root="", file_paths=[src])
    contra = [c for c in results if c.contradiction_kind == "contradictory_boolean"]
    assert contra, "expected contradictory_boolean"
    assert contra[0].node_kind == "If"
    Path(src).unlink()


def test_node_kind_while_from_scan():
    src = _make_py_file("while False:\n    pass\n")
    results = scan_contradictory_logic(repo_root="", file_paths=[src])
    kinds = {c.contradiction_kind for c in results}
    assert "constant_false_condition" in kinds
    while_c = [c for c in results if c.contradiction_kind == "constant_false_condition"]
    assert while_c[0].node_kind == "While"
    Path(src).unlink()


def test_node_kind_assert_from_scan():
    src = _make_py_file("def f():\n    assert False\n")
    results = scan_contradictory_logic(repo_root="", file_paths=[src])
    asserts = [c for c in results if c.contradiction_kind == "always_false_assert"]
    assert asserts
    assert asserts[0].node_kind == "Assert"
    Path(src).unlink()


def test_duplicate_node_kind_reflects_while_not_hardcoded_if():
    """A `while X:` duplicating an adjacent `if X:` must report node_kind=='While'.

    Regression: the duplicate emitter hardcoded node_kind='If' even though
    while-conditions also feed the duplicate detector, mislabelling the AST
    node a downstream consumer would branch on.
    """
    src = _make_py_file("""\
        def check(x):
            if x > 0:
                v = 1
            while x > 0:
                v = 2
    """)
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates
            if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups, "Expected the repeated `x > 0` guard to be flagged"
    # The second (current) occurrence is the while loop.
    assert dups[0].node_kind == "While"
    Path(src).unlink()


def test_end_lineno_covers_full_branch_body():
    """end_lineno should cover the last line of the branch body, not just the if line."""
    src = _make_py_file("""\
        def f(x):
            if x and not x:
                dead_1()
                dead_2()
            after()
    """)
    results = scan_contradictory_logic(repo_root="", file_paths=[src])
    contra = [c for c in results if c.contradiction_kind == "contradictory_boolean"]
    assert contra
    c = contra[0]
    assert c.end_lineno > c.lineno, f"end_lineno={c.end_lineno} should be > lineno={c.lineno}"
    Path(src).unlink()


def test_duplicate_condition_node_kind_is_if():
    src = _make_py_file("""\
        def f(x):
            if x > 0:
                pass
            if x > 0:
                pass
    """)
    results = scan_contradictory_logic(repo_root="", file_paths=[src])
    dups = [c for c in results if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert dups
    assert dups[0].node_kind == "If"
    Path(src).unlink()


def test_distant_attr_condition_blocked_by_call_barrier():
    """Attribute condition 2000+ lines apart with method calls: NOT flagged."""
    lines = ["def run(self):"]
    lines.append("    if self.config.is_subagent:")
    lines.append("        pass")
    # simulate 200 lines of code including method calls
    for i in range(100):
        lines.append(f"    result_{i} = self.step_{i}()")
    lines.append("    if self.config.is_subagent:")
    lines.append("        pass")
    src = _make_py_file("\n".join(lines) + "\n")
    candidates = scan_contradictory_logic(repo_root="", file_paths=[src], max_dup_distance=100)
    dups = [c for c in candidates if c.contradiction_kind in ("duplicate_condition", "mergeable_condition")]
    assert not dups, f"Call barrier should prevent false positive. Got: {[c.detail for c in dups]}"
    Path(src).unlink()
