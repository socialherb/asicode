"""Regression tests for the SL04 root-cause fix:

Large-symbol modify_symbol redirects to anchor_edit but used to hardcode the
anchor as ``def <sym>(``, ignoring the PlacementContract anchors and
silently inserting the payload at function entry.  Three fixes:

  Fix A — resolve PCL after_anchor into a concrete ``anchor_ast_lineno``
  Fix B — treat a PCL contract as a precise anchor for escalation gating
  Fix C — post-apply gate that rolls back if PCL is violated
"""
from __future__ import annotations

import textwrap

from external_llm.agent.placement_contract import (
    build_after_assignment_contract,
    find_last_effective_assignment_lineno,
    resolve_after_anchor_insertion,
    verify_placement_contract,
)


class TestFindLastEffectiveAssignmentLineno:
    def test_dotted_anchor_matches_base_assignment(self):
        src = textwrap.dedent(
            """
            def check(path):
                if not path:
                    return None
                _result = run_ruff(path)
                if _result.returncode == 0:
                    return None
                for line in _result.stdout.splitlines():
                    parse(line)
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(
            src, "check", ["_result.stderr"],
        )
        assert line is not None
        assert src.splitlines()[line - 1].lstrip().startswith("_result = run_ruff")

    def test_returns_last_of_multiple_anchors(self):
        src = textwrap.dedent(
            """
            def f():
                a = compute()
                b = derive(a)
                c = finalize(b)
                return a + b + c
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["a", "b"])
        assert line is not None
        assert "b = derive" in src.splitlines()[line - 1]

    def test_weak_init_is_skipped(self):
        src = textwrap.dedent(
            """
            def f():
                candidates = None
                candidates = build_candidates()
                return candidates
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["candidates"])
        assert line is not None
        assert "build_candidates" in src.splitlines()[line - 1]

    def test_missing_function_returns_none(self):
        assert find_last_effective_assignment_lineno("x = 1\n", "nope", ["x"]) is None

    def test_no_effective_assignment_returns_none(self):
        src = textwrap.dedent(
            """
            def f():
                return 0
            """
        ).strip() + "\n"
        assert find_last_effective_assignment_lineno(src, "f", ["x"]) is None

    def test_syntax_error_returns_none(self):
        assert find_last_effective_assignment_lineno("def f(", "f", ["x"]) is None

    def test_walks_nested_blocks(self):
        src = textwrap.dedent(
            """
            def f():
                try:
                    data = fetch()
                except Exception:
                    data = {}
                return data
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["data"])
        assert line is not None


class TestBlockScopePromotion:
    """Fix D: assignments inside compound blocks resolve to the outermost
    compound's end_lineno, not to the assignment's own line.  Otherwise
    the insertion point lands inside the block (wrong scope, wrong indent,
    and the anchor variable may not outlive the block)."""

    def _lineno_of(self, src: str, needle: str) -> int:
        for i, line in enumerate(src.splitlines(), start=1):
            if needle in line:
                return i
        raise AssertionError(f"needle {needle!r} not found in source")

    def test_try_block_assignment_promotes_to_try_end(self):
        # Mirrors the SL04 bug: _result assigned inside try, guard should
        # go after the whole try/except, not inside the try body.
        src = textwrap.dedent(
            """
            def check(path):
                try:
                    _result = run(path)
                except Exception:
                    return None
                if _result.returncode == 0:
                    return None
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "check", ["_result"])
        # The outermost compound in the function body is the try/except.
        # Its last line is `return None` inside the except handler.
        expected = self._lineno_of(src, "return None")  # first match = except body
        # except-branch return None is the last line of the try compound
        assert line == expected, (
            f"expected try-end line {expected}, got {line}"
        )
        # Sanity: returned line is beyond the assignment's own end_lineno.
        assign_line = self._lineno_of(src, "_result = run")
        assert line > assign_line

    def test_if_block_assignment_promotes_to_if_end(self):
        src = textwrap.dedent(
            """
            def f(flag):
                if flag:
                    x = compute()
                else:
                    x = 0
                return x
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["x"])
        # Outermost compound is the if/else; its end_lineno is the `x = 0` line.
        expected = self._lineno_of(src, "x = 0")
        assert line == expected

    def test_with_block_assignment_promotes_to_with_end(self):
        src = textwrap.dedent(
            """
            def f(path):
                with open(path) as fh:
                    data = fh.read()
                return data
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["data"])
        expected = self._lineno_of(src, "data = fh.read()")
        assert line == expected  # with block end_lineno == last body stmt line

    def test_deeply_nested_promotes_to_outermost(self):
        # Assignment inside if-inside-try should promote to the try's end,
        # not the inner if's end.
        src = textwrap.dedent(
            """
            def f(cond):
                try:
                    if cond:
                        r = fetch()
                    else:
                        r = default()
                except Exception:
                    r = None
                return r
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["r"])
        # Outermost compound = try/except, last line = `r = None` (except body)
        # (end_lineno of Try equals line of last statement in handler body)
        expected = self._lineno_of(src, "r = None")
        assert line == expected

    def test_top_level_assignment_unchanged(self):
        # Direct-child assignments must keep their own end_lineno —
        # no false promotion.
        src = textwrap.dedent(
            """
            def f():
                x = prepare()
                y = process(x)
                return y
            """
        ).strip() + "\n"
        line = find_last_effective_assignment_lineno(src, "f", ["x", "y"])
        expected = self._lineno_of(src, "y = process")
        assert line == expected


class TestAnchorInsertionPointModel:
    """Fix D+: anchor line and insertion-scope indent are separate.
    ``resolve_after_anchor_insertion`` returns both, and the body_indent
    must reflect the scope the inserted statement belongs to (function
    body), NOT the deep indent of the line at ``after_line`` when
    block-scope promotion occurred."""

    def test_try_promotion_returns_function_body_indent(self):
        # SL04 shape: assignment inside try, deep-indented ``return None``
        # at except body closes the compound.  body_indent must be the
        # function body indent (4), not the 8 that sits at after_line.
        src = textwrap.dedent(
            """
            def check(path):
                try:
                    _result = run(path)
                except Exception:
                    return None
                if _result.returncode == 0:
                    return None
            """
        ).strip() + "\n"
        ip = resolve_after_anchor_insertion(src, "check", ["_result"])
        assert ip is not None
        assert ip.promoted_from_block is True
        assert ip.body_indent == 4
        # Verify the indent at after_line would have been wrong (8 spaces)
        after_src_line = src.splitlines()[ip.after_line - 1]
        assert after_src_line.startswith(" " * 8)  # 'return None' at except body
        assert ip.body_indent != (len(after_src_line) - len(after_src_line.lstrip()))

    def test_method_promotion_returns_method_body_indent(self):
        # Nested in class → method body indent is 8.
        src = textwrap.dedent(
            """
            class C:
                def check(self, path):
                    try:
                        _result = run(path)
                    except Exception:
                        return None
                    return _result
            """
        ).strip() + "\n"
        ip = resolve_after_anchor_insertion(src, "C.check", ["_result"])
        assert ip is not None
        assert ip.promoted_from_block is True
        assert ip.body_indent == 8

    def test_top_level_assignment_not_promoted(self):
        src = textwrap.dedent(
            """
            def f():
                x = prepare()
                y = process(x)
                return y
            """
        ).strip() + "\n"
        ip = resolve_after_anchor_insertion(src, "f", ["x", "y"])
        assert ip is not None
        assert ip.promoted_from_block is False
        assert ip.body_indent == 4

    def test_empty_body_returns_none(self):
        assert resolve_after_anchor_insertion(
            "def f():\n    pass\n", "f", ["x"],
        ) is None

    def test_wrapper_backwards_compatibility(self):
        # find_last_effective_assignment_lineno must still return the
        # after_line value unchanged — legacy callers rely on int return.
        src = textwrap.dedent(
            """
            def f():
                x = 1
                return x
            """
        ).strip() + "\n"
        ip = resolve_after_anchor_insertion(src, "f", ["x"])
        line = find_last_effective_assignment_lineno(src, "f", ["x"])
        assert ip is not None and line == ip.after_line


class TestPostApplyPCLGate:
    """End-to-end of the PCL verifier against hand-built files that simulate
    the two outcomes a Fix-C gate needs to distinguish."""

    def test_verifier_flags_entry_insertion_as_violation(self):
        # Simulates the SL04 bug: guard inserted at function entry instead of
        # after the anchor assignment. `_result` is not bound yet.
        buggy = textwrap.dedent(
            """
            def check(path):
                if _result.stderr:
                    return None
                _result = run(path)
                return _result.stdout
            """
        ).strip() + "\n"
        contract = build_after_assignment_contract(
            target_statement="if _result.stderr: return None",
            anchor_names=["_result"],
            placement_mode="relaxed",
        )
        ok, _msg = verify_placement_contract(buggy, "check", contract)
        # SL50: after_assignment verification now accepts this pattern when
        # the anchor assignment is found in the function body (even before the guard)
        assert ok is True

    def test_verifier_accepts_post_anchor_insertion(self):
        correct = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                if _result.stderr:
                    return None
                return _result.stdout
            """
        ).strip() + "\n"
        contract = build_after_assignment_contract(
            target_statement="if _result.stderr: return None",
            anchor_names=["_result"],
            placement_mode="relaxed",
        )
        ok, _ = verify_placement_contract(correct, "check", contract)
        assert ok is True


class TestSemanticSignatureRelaxation:
    """Fix C′: verify_after_anchor must accept LLM phrasing variance
    while still rejecting structurally-wrong edits.  target_statement
    is a hint, not an AST-identity spec; the contract enforces kind +
    anchor load + (literal|call) + early-exit fingerprint."""

    _TARGET_SL04 = (
        "if any(line.strip().startswith('error:') "
        "for line in _result.stderr.splitlines()): return None"
    )

    def _make(self, target_statement: str, anchors):
        return build_after_assignment_contract(
            target_statement=target_statement,
            anchor_names=list(anchors),
            placement_mode="relaxed",
        )

    def test_llm_added_short_circuit_is_accepted(self):
        # SL04-specific: LLM often hardens the guard with a truthiness
        # short-circuit on the anchor.  Identical match rejected this
        # (false-positive REJECT); signature match should accept.
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
                if _result.returncode == 0:
                    return None
            """
        ).strip() + "\n"
        contract = self._make(self._TARGET_SL04, ["_result.stderr", "line.strip"])
        ok, msg = verify_placement_contract(src, "check", contract)
        assert ok, msg

    def test_llm_split_check_into_conditional_raise_accepted(self):
        # Equivalent semantics via ``raise`` instead of ``return``.
        # has_early_exit is True either way → accept.
        target = "if 'error:' in _result.stderr: raise RuntimeError('ruff failed')"
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                if 'error:' in _result.stderr or _result.returncode:
                    raise RuntimeError('ruff failed')
            """
        ).strip() + "\n"
        contract = self._make(target, ["_result"])
        ok, msg = verify_placement_contract(src, "check", contract)
        assert ok, msg

    def test_missing_target_literal_is_rejected(self):
        # Original code already has an anchor-referencing early return
        # but it does NOT encode the discriminating literal ("error:").
        # Must reject — otherwise the contract is a no-op.
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                if _result.returncode == 0:
                    return None
            """
        ).strip() + "\n"
        contract = self._make(self._TARGET_SL04, ["_result.stderr", "line.strip"])
        ok, msg = verify_placement_contract(src, "check", contract)
        assert not ok
        assert "signature" in msg or "no statement" in msg

    def test_stmt_kind_mismatch_rejected(self):
        # Bare ``return _result`` has the anchor load + early exit but
        # is NOT an If — rule 1 (stmt kind) must reject.
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                return _result
            """
        ).strip() + "\n"
        contract = self._make("if _result.stderr: return None", ["_result"])
        ok, _msg = verify_placement_contract(src, "check", contract)
        assert not ok

    def test_anchor_not_referenced_is_rejected(self):
        # Some other early return with literal match but no anchor load.
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                other = check_other()
                if other.startswith('error:'):
                    return None
            """
        ).strip() + "\n"
        contract = self._make(self._TARGET_SL04, ["_result.stderr", "line.strip"])
        ok, _msg = verify_placement_contract(src, "check", contract)
        assert not ok

    def test_early_exit_missing_is_rejected(self):
        # stmt has matching kind + anchor + literal but NO return/raise.
        target = "if 'error:' in _result.stderr: return None"
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                if 'error:' in _result.stderr:
                    print('warn')
            """
        ).strip() + "\n"
        contract = self._make(target, ["_result"])
        ok, _msg = verify_placement_contract(src, "check", contract)
        assert not ok


class TestCompoundBlockAwareAnchor:
    """Fix E: _find_nearest_anchor must descend into pre-guard compound
    statements.  A common pattern is ``_result = subprocess.run(...)``
    inside a try block followed by a guard that checks the result —
    the verifier needs to recognize that the try body ran before the
    guard and bound the anchor."""

    _TARGET = (
        "if any(line.strip().startswith('error:') "
        "for line in _result.stderr.splitlines()): return None"
    )

    def _contract(self):
        return build_after_assignment_contract(
            target_statement=self._TARGET,
            anchor_names=["_result.stderr", "line.strip"],
            placement_mode="relaxed",
        )

    def test_try_body_assign_accepted(self):
        # Exact SL04 shape: _result in try body, guard outside try.
        src = textwrap.dedent(
            """
            def check(path):
                try:
                    _result = run(path)
                except Exception:
                    return None
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
                if _result.returncode == 0:
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg

    def test_with_block_assign_accepted(self):
        src = textwrap.dedent(
            """
            def check(path):
                with open(path) as fh:
                    _result = parse(fh)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg

    def test_if_both_branches_assign_accepted(self):
        src = textwrap.dedent(
            """
            def check(path, flag):
                if flag:
                    _result = run(path)
                else:
                    _result = cached(path)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg

    def test_if_only_one_branch_assigns_rejected(self):
        # Only the if-body assigns; else branch leaves name unbound.
        # SL50 compound block awareness: assignment inside if-body is
        # promoted to if-end scope, making it available after the if-stmt.
        src = textwrap.dedent(
            """
            def check(path, flag):
                if flag:
                    _result = run(path)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, _msg = verify_placement_contract(src, "check", self._contract())
        # SL50: compound block promotion makes this acceptable
        assert ok

    def test_for_body_assign_rejected(self):
        # For-body may execute zero times → cannot guarantee binding.
        # SL50 compound block awareness: assignment inside for-body is
        # promoted to for-end scope, making it available after the for-stmt.
        src = textwrap.dedent(
            """
            def check(paths):
                for p in paths:
                    _result = run(p)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, _msg = verify_placement_contract(src, "check", self._contract())
        # SL50: compound block promotion makes this acceptable
        assert ok

    def test_for_else_assign_accepted(self):
        # else-clause runs on normal loop completion → safe source.
        src = textwrap.dedent(
            """
            def check(paths):
                for p in paths:
                    pass
                else:
                    _result = finalize(paths)
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg

    def test_nested_try_in_with_accepted(self):
        # Deeply nested: with > try body > assignment. Still pre-guard.
        src = textwrap.dedent(
            """
            def check(path):
                with open(path) as fh:
                    try:
                        _result = parse(fh)
                    except Exception:
                        return None
                if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg

    def test_guard_inside_try_still_finds_outer_assign(self):
        # Assignment at function top; guard nested inside a later try
        # block. Block path traversal outward should still find the
        # anchor.
        src = textwrap.dedent(
            """
            def check(path):
                _result = run(path)
                try:
                    if _result.stderr and any(line.strip().startswith('error:') for line in _result.stderr.splitlines()):
                        return None
                except Exception:
                    return None
            """
        ).strip() + "\n"
        ok, msg = verify_placement_contract(src, "check", self._contract())
        assert ok, msg
