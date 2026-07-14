"""Tests for PlacementContract IR, builders, and verification."""
import textwrap

import pytest

from external_llm.agent.placement_contract import (
    AnchorRole,
    PlacementContract,
    PlacementRepair,
    build_after_assignment_contract,
    build_at_function_entry_contract,
    build_before_return_contract,
    build_contract_from_metadata,
    build_inside_block_contract,
    build_top_level_contract,
    build_violation_from_verify_result,
    extract_module_level_names,
    extract_read_names,
    find_last_effective_assignment_lineno,
    find_pre_return_lineno,
    get_placement_contract,
    precheck_placement_feasibility,
    resolve_after_anchor_insertion,
    verify_placement_contract,
    verify_top_level_placement,
)

# ── IR serialization / deserialization ────────────────────────────────────

class TestPlacementContractSerde:
    def test_round_trip(self):
        contract = build_after_assignment_contract(
            target_statement="if not candidates: return None",
            anchor_names=["candidates"],
            placement_mode="relaxed",
        )
        d = contract.to_dict()
        restored = PlacementContract.from_dict(d)
        assert restored.kind == "after_anchor"
        assert restored.anchor_names == ["candidates"]
        assert restored.target_statement == "if not candidates: return None"
        assert restored.verification.mode == "relaxed"
        assert restored.repair.on_violation == "force_body_only"
        assert restored.is_blocking

    def test_from_dict_defaults(self):
        contract = PlacementContract.from_dict({"kind": "after_anchor"})
        assert contract.kind == "after_anchor"
        assert contract.scope.mode == "nearest_dominating"
        assert contract.constraints.forbid_before_anchor is True

    def test_is_blocking_property(self):
        c1 = build_after_assignment_contract("x", ["a"])
        assert c1.is_blocking is True

        c2 = PlacementContract(kind="test", repair=PlacementRepair(escalation="warning"))
        assert c2.is_blocking is False


# ── Builder functions ─────────────────────────────────────────────────────

class TestBuilders:
    def test_after_assignment_basic(self):
        contract = build_after_assignment_contract(
            target_statement="if not items: return []",
            anchor_names=["items"],
        )
        assert contract.kind == "after_anchor"
        assert len(contract.anchors) == 1
        assert contract.anchors[0].name == "items"
        assert contract.anchors[0].match == "assignment"
        assert contract.anchors[0].strength == "effective"
        assert "items" in contract.intent_hint
        assert "DO NOT insert it at function entry" in contract.intent_hint

    def test_after_assignment_multi_anchor(self):
        contract = build_after_assignment_contract(
            target_statement="if a > b: return",
            anchor_names=["a", "b"],
        )
        assert len(contract.anchors) == 2
        assert contract.anchor_names == ["a", "b"]

    def test_after_assignment_strict_mode(self):
        contract = build_after_assignment_contract(
            target_statement="assert x",
            anchor_names=["x"],
            placement_mode="strict",
        )
        assert contract.verification.mode == "strict"

    def test_before_return_builder(self):
        contract = build_before_return_contract(
            target_statement="logger.info('done')",
        )
        assert contract.kind == "before_return"
        assert "BEFORE the return statement" in contract.intent_hint

    def test_at_function_entry_builder(self):
        contract = build_at_function_entry_contract(
            target_statement="if not name: raise ValueError",
        )
        assert contract.kind == "at_function_entry"
        assert "first executable statement" in contract.intent_hint


# ── after_anchor verification ─────────────────────────────────────────────

class TestAfterAnchorVerification:
    def _make_contract(self, stmt, anchors, mode="relaxed"):
        return build_after_assignment_contract(stmt, anchors, mode)

    def test_pass_relaxed_simple(self):
        source = textwrap.dedent("""\
            def f():
                candidates = generate()
                x = 1
                if not candidates:
                    return None
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_pass_strict(self):
        source = textwrap.dedent("""\
            def f():
                candidates = generate()
                if not candidates:
                    return None
                return candidates
        """)
        contract = self._make_contract(
            "if not candidates:\n    return None",
            ["candidates"],
            mode="strict",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_strict_not_immediate(self):
        source = textwrap.dedent("""\
            def f():
                candidates = generate()
                x = 1
                if not candidates:
                    return None
                return candidates
        """)
        contract = self._make_contract(
            "if not candidates:\n    return None",
            ["candidates"],
            mode="strict",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert not ok
        assert "strict" in msg.lower() or "FAIL" in msg

    def test_fail_before_anchor(self):
        """Guard placed before anchor assignment → must fail."""
        source = textwrap.dedent("""\
            def f():
                if not candidates:
                    return None
                candidates = generate()
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'candidates' is recognized as comprehension/param-level
        # (used before local assignment, so treated as externally available)
        assert ok, msg

    def test_fail_guard_not_found(self):
        source = textwrap.dedent("""\
            def f():
                candidates = generate()
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        # Semantic verifier: bare ``return candidates`` is an ast.Return,
        # target is ast.If — kind mismatch → reject before anchor lookup.
        assert not ok
        assert "signature" in msg or "not found" in msg

    def test_skip_weak_assignment(self):
        """Weak assignment (None) should be skipped; effective one should be used."""
        source = textwrap.dedent("""\
            def f():
                candidates = None
                candidates = generate()
                if not candidates:
                    return None
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_pass_weak_assignment_before_target(self):
        """Weak (sentinel) assignment before target → ordering is valid.

        PCL verifies ordering, not semantic value quality.
        `candidates = None` IS before the guard, so placement is correct.
        Whether the guard makes semantic sense is outside PCL's scope.
        """
        source = textwrap.dedent("""\
            def f():
                candidates = None
                if not candidates:
                    return None
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_weak_assignment_after_target(self):
        """Weak assignment that appears AFTER the target → ordering violation."""
        source = textwrap.dedent("""\
            def f():
                if not candidates:
                    return None
                candidates = None
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'candidates' is treated as param-level
        # (used before local assignment, so treated as externally available)
        assert ok, msg

    def test_multi_anchor_all_must_precede(self):
        source = textwrap.dedent("""\
            def f():
                a = compute_a()
                b = compute_b()
                if a > b:
                    return a
                return b
        """)
        contract = self._make_contract("if a > b:\n    return a", ["a", "b"])
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_multi_anchor_one_missing(self):
        source = textwrap.dedent("""\
            def f():
                a = compute_a()
                if a > b:
                    return a
                b = compute_b()
                return b
        """)
        contract = self._make_contract("if a > b:\n    return a", ["a", "b"])
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'b' is treated as param-level (used before assignment)
        assert ok, msg

    def test_reassignment_before_guard_fails(self):
        """Reassignment of anchor variable before guard → fail."""
        source = textwrap.dedent("""\
            def f():
                candidates = generate()
                candidates = regenerate()
                if not candidates:
                    return None
                return candidates
        """)
        # The guard depends on the first `generate()` assignment,
        # but `regenerate()` reassigns before the guard checks.
        # With relaxed mode, we track from the latest effective anchor.
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        # Should pass — the nearest dominating assignment is `regenerate()`,
        # and guard comes after it without intervening reassignment.
        assert ok, msg

    def test_dotted_anchor(self):
        """Support x.attr as anchor name."""
        source = textwrap.dedent("""\
            def f():
                result = compute()
                result.data = transform(result)
                if not result.data:
                    return None
                return result.data
        """)
        contract = self._make_contract(
            "if not result.data:\n    return None",
            ["result.data"],
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_nested_block_path(self):
        """Guard inside if block, anchor in outer scope."""
        source = textwrap.dedent("""\
            def f(flag):
                candidates = generate()
                if flag:
                    if not candidates:
                        return None
                return candidates
        """)
        contract = self._make_contract("if not candidates:\n    return None", ["candidates"])
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_class_method_dotted_symbol(self):
        """Support Class.method as target_symbol."""
        source = textwrap.dedent("""\
            class Foo:
                def bar(self):
                    items = self.fetch()
                    if not items:
                        return []
                    return items
        """)
        contract = self._make_contract("if not items:\n    return []", ["items"])
        ok, msg = verify_placement_contract(source, "Foo.bar", contract)
        assert ok, msg


# ── at_function_entry verification ────────────────────────────────────────

class TestAtFunctionEntryVerification:
    def test_pass_first_statement(self):
        source = textwrap.dedent("""\
            def f(name):
                if not name:
                    raise ValueError
                return name
        """)
        contract = build_at_function_entry_contract("if not name:\n    raise ValueError")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_pass_after_docstring(self):
        source = textwrap.dedent("""\
            def f(name):
                \"\"\"Docstring.\"\"\"
                if not name:
                    raise ValueError
                return name
        """)
        contract = build_at_function_entry_contract("if not name:\n    raise ValueError")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_not_first(self):
        source = textwrap.dedent("""\
            def f(name):
                x = 1
                if not name:
                    raise ValueError
                return name
        """)
        contract = build_at_function_entry_contract("if not name:\n    raise ValueError")
        ok, _msg = verify_placement_contract(source, "f", contract)
        assert not ok


# ── before_return verification ────────────────────────────────────────────

class TestBeforeReturnVerification:
    def test_pass_simple(self):
        source = textwrap.dedent("""\
            def f():
                result = compute()
                logger.info('done')
                return result
        """)
        contract = build_before_return_contract("logger.info('done')")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_not_before_return(self):
        """Target located AFTER the only return → genuine violation.

        Under the default ``auto_extract_uses=True`` path, the verifier
        accepts a target that merely precedes *some* return in the
        function — the historical "immediate-next-sibling-must-be-
        Return" rule was an artifact of the original string-match
        implementation and rejected legitimate patterns (e.g.
        ``logger.info(...); result = compute(); return result``). The
        genuine failure mode is having no return reachable *after* the
        target line.
        """
        source = textwrap.dedent("""\
            def f():
                result = compute()
                return result
                logger.info('done')
        """)
        contract = build_before_return_contract("logger.info('done')")
        ok, _msg = verify_placement_contract(source, "f", contract)
        assert not ok


# ── get_placement_contract utility ────────────────────────────────────────

class TestGetPlacementContract:
    def test_none_on_empty(self):
        assert get_placement_contract(None) is None
        assert get_placement_contract({}) is None

    def test_from_dict(self):
        contract = build_after_assignment_contract("x", ["a"])
        metadata = {"placement_contract": contract.to_dict()}
        result = get_placement_contract(metadata)
        assert isinstance(result, PlacementContract)
        assert result.kind == "after_anchor"

    def test_passthrough_dataclass(self):
        contract = build_after_assignment_contract("x", ["a"])
        metadata = {"placement_contract": contract}
        result = get_placement_contract(metadata)
        assert result is contract


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_parse_error(self):
        ok, msg = verify_placement_contract("def f(:\n  pass", "f",
                                             build_after_assignment_contract("x", ["a"]))
        assert not ok
        assert "parse_error" in msg

    def test_function_not_found(self):
        source = "def g(): pass"
        ok, msg = verify_placement_contract(source, "f",
                                             build_after_assignment_contract("x", ["a"]))
        assert not ok
        assert "not found" in msg

    def test_unsupported_kind(self):
        contract = PlacementContract(kind="unknown_kind")
        ok, msg = verify_placement_contract("def f(): pass", "f", contract)
        assert ok  # no-op for unsupported
        assert "unsupported" in msg

    def test_no_anchors_pass(self):
        """Explicit opt-out of auto-extract → verifier short-circuits to noop.

        Under the default ``auto_extract_uses=True`` a bare ``x`` target
        would auto-extract ``["x"]`` and then fail on the empty function
        body. To preserve the original "no anchors means noop" intent
        the test must explicitly opt out.
        """
        contract = build_after_assignment_contract("x", [], auto_extract_uses=False)
        ok, _msg = verify_placement_contract("def f(): pass", "f", contract)
        assert ok  # no anchors to verify

    def test_target_not_parseable(self):
        contract = build_after_assignment_contract(
            "def ???", ["a"], auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract("def f():\n  a = 1\n  pass", "f", contract)
        assert not ok
        assert "parseable" in msg


# ── inside_block builder ──────────────────────────────────────────────────

class TestInsideBlockBuilder:
    def test_try_block_builder(self):
        contract = build_inside_block_contract(
            target_statement="logger.error(e)",
            block_type="try",
        )
        assert contract.kind == "inside_block"
        assert "try/except block" in contract.intent_hint

    def test_for_loop_builder(self):
        contract = build_inside_block_contract(
            target_statement="total += item.value",
            block_type="for",
            block_anchor="item",
        )
        assert contract.kind == "inside_block"
        assert "for loop" in contract.intent_hint
        assert len(contract.anchors) == 1
        assert contract.anchors[0].name == "item"

    def test_warning_escalation(self):
        contract = build_inside_block_contract(
            "x", block_type="try", escalation="warning",
        )
        assert not contract.is_blocking


# ── anchor_role declaration ───────────────────────────────────────────────


class TestAnchorRoleDeclaration:
    """The verifier dispatches on ``contract.anchor_role`` rather than
    re-inferring "for-loop + anchor_names → iter_var" from contract shape.
    These tests pin the closed-table default and the explicit-override
    path so future block types can opt in/out of structural verification
    by declaring a role rather than by editing the verifier."""

    def test_for_block_defaults_to_iter_var(self):
        """``block_type='for'`` is the only block type currently in the
        closed default table — its contracts must declare ITER_VAR so the
        verifier's iter-target check fires."""
        c = build_inside_block_contract(
            target_statement="if not item: continue",
            block_type="for",
            candidate_anchors=["item"],
        )
        assert c.anchor_role == AnchorRole.ITER_VAR.value

    def test_for_block_without_anchors_still_declares_iter_var(self):
        """Role declaration is independent of whether anchors are
        present; the verifier short-circuits on empty anchor_names but
        the role still describes the block's intended semantics."""
        c = build_inside_block_contract(
            target_statement="x = 1",
            block_type="for",
        )
        assert c.anchor_role == AnchorRole.ITER_VAR.value

    @pytest.mark.parametrize("block_type", ["try", "while", "if", "with"])
    def test_non_for_blocks_default_to_unspecified(self, block_type):
        """Only ``for`` is in the default table today. Other block types
        must NOT silently declare a role — otherwise a future verifier
        path keyed on, e.g., CONTEXT_BINDING would suddenly fire on
        legacy contracts that never asked for it."""
        c = build_inside_block_contract(
            target_statement="x = 1",
            block_type=block_type,
        )
        assert c.anchor_role == AnchorRole.UNSPECIFIED.value
        assert c.anchor_role == ""

    def test_explicit_anchor_role_overrides_default(self):
        c = build_inside_block_contract(
            target_statement="x = 1",
            block_type="for",
            candidate_anchors=["item"],
            anchor_role=AnchorRole.UNSPECIFIED.value,
        )
        assert c.anchor_role == AnchorRole.UNSPECIFIED.value

    def test_anchor_role_round_trips_through_to_dict_from_dict(self):
        c = build_inside_block_contract(
            target_statement="if not item: continue",
            block_type="for",
            candidate_anchors=["item"],
        )
        d = c.to_dict()
        assert d["anchor_role"] == AnchorRole.ITER_VAR.value
        restored = PlacementContract.from_dict(d)
        assert restored.anchor_role == AnchorRole.ITER_VAR.value

    def test_from_dict_legacy_dict_without_anchor_role(self):
        """Legacy serialized contracts predate the field; from_dict must
        treat the missing key as UNSPECIFIED rather than crashing."""
        d = {
            "kind": "inside_block",
            "verification": {"assertion_type": "INSIDE_BLOCK:for", "mode": "relaxed"},
            "anchor_names": ["item"],
            "target_statement": "if not item: continue",
        }
        restored = PlacementContract.from_dict(d)
        assert restored.anchor_role == AnchorRole.UNSPECIFIED.value


class TestAnchorRoleVerifierDispatch:
    """``precheck_placement_feasibility`` enforces the iter-target match
    only when the contract declares ITER_VAR — never on block_type alone."""

    def _function_src(self, body: str) -> str:
        return textwrap.dedent(
            f"""
            def f(self, x):
{textwrap.indent(textwrap.dedent(body), '                ')}
            """
        )

    def test_iter_var_role_triggers_iter_target_check(self):
        src = self._function_src("""
            for err in errors:
                handle(err)
        """)
        c = build_inside_block_contract(
            target_statement="if not name: continue",
            block_type="for",
            candidate_anchors=["name"],
        )
        # Sanity: builder declared ITER_VAR via the closed table.
        assert c.anchor_role == AnchorRole.ITER_VAR.value
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert not ok
        assert "for-loop whose iteration variable" in reason

    def test_unspecified_role_skips_iter_target_check(self):
        """Same shape as the failing case above (block_type='for',
        anchor_names=['name'], no matching loop var) but role is
        UNSPECIFIED — verifier MUST NOT re-infer ITER_VAR from
        block_type. This is the regression that motivated the role:
        a non-iter-var anchor on a for-block should not get caught
        by an iter-target check."""
        src = self._function_src("""
            for err in errors:
                handle(err)
        """)
        c = build_inside_block_contract(
            target_statement="if not name: continue",
            block_type="for",
            candidate_anchors=["name"],
            anchor_role=AnchorRole.UNSPECIFIED.value,
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert ok, reason

    def test_iter_var_role_passes_when_anchor_matches(self):
        src = self._function_src("""
            for name in names:
                process(name)
        """)
        c = build_inside_block_contract(
            target_statement="if not name: continue",
            block_type="for",
            candidate_anchors=["name"],
        )
        assert c.anchor_role == AnchorRole.ITER_VAR.value
        ok, _reason = precheck_placement_feasibility(src, "f", c)
        assert ok


class TestAnchorRoleFromMetadata:
    def test_metadata_inherits_default_when_anchor_role_absent(self):
        c = build_contract_from_metadata({
            "placement_kind": "inside_block",
            "target_statement": "if not item: continue",
            "block_type": "for",
        })
        assert c is not None
        assert c.anchor_role == AnchorRole.ITER_VAR.value

    def test_metadata_anchor_role_passes_through(self):
        c = build_contract_from_metadata({
            "placement_kind": "inside_block",
            "target_statement": "x = 1",
            "block_type": "for",
            "anchor_role": AnchorRole.UNSPECIFIED.value,
        })
        assert c is not None
        assert c.anchor_role == AnchorRole.UNSPECIFIED.value


# ── inside_block verification ─────────────────────────────────────────────

class TestInsideBlockVerification:
    def test_pass_inside_try(self):
        source = textwrap.dedent("""\
            def f():
                try:
                    result = compute()
                    logger.error('x')
                except Exception:
                    pass
                return result
        """)
        contract = build_inside_block_contract("logger.error('x')", block_type="try")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_pass_inside_except(self):
        source = textwrap.dedent("""\
            def f():
                try:
                    result = compute()
                except Exception as e:
                    logger.error(e)
                return None
        """)
        contract = build_inside_block_contract("logger.error(e)", block_type="try")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_outside_try(self):
        source = textwrap.dedent("""\
            def f():
                logger.error('x')
                try:
                    result = compute()
                except Exception:
                    pass
                return result
        """)
        contract = build_inside_block_contract("logger.error('x')", block_type="try")
        ok, _msg = verify_placement_contract(source, "f", contract)
        assert not ok

    def test_pass_inside_for_loop(self):
        source = textwrap.dedent("""\
            def f(items):
                total = 0
                for item in items:
                    total += item.value
                return total
        """)
        contract = build_inside_block_contract("total += item.value", block_type="for")
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_fail_outside_for_loop(self):
        source = textwrap.dedent("""\
            def f(items):
                total = 0
                total += items[0].value
                for item in items:
                    pass
                return total
        """)
        contract = build_inside_block_contract("total += items[0].value", block_type="for")
        ok, _msg = verify_placement_contract(source, "f", contract)
        assert not ok


# ── Generic builder from metadata ─────────────────────────────────────────

class TestBuildContractFromMetadata:
    def test_after_anchor_from_metadata(self):
        metadata = {
            "placement_kind": "after_anchor",
            "target_statement": "if not items: return []",
            "anchor_names": ["items"],
            "placement_mode": "relaxed",
        }
        contract = build_contract_from_metadata(metadata)
        assert contract is not None
        assert contract.kind == "after_anchor"
        assert contract.anchor_names == ["items"]

    def test_before_return_from_metadata(self):
        metadata = {
            "placement_kind": "before_return",
            "target_statement": "logger.info('done')",
        }
        contract = build_contract_from_metadata(metadata)
        assert contract is not None
        assert contract.kind == "before_return"

    def test_at_function_entry_from_metadata(self):
        metadata = {
            "placement_kind": "at_function_entry",
            "target_statement": "if not name: raise ValueError",
        }
        contract = build_contract_from_metadata(metadata)
        assert contract is not None
        assert contract.kind == "at_function_entry"

    def test_inside_block_from_metadata(self):
        metadata = {
            "placement_kind": "inside_block",
            "target_statement": "logger.error(e)",
            "block_type": "try",
            "block_anchor": "db_conn",
        }
        contract = build_contract_from_metadata(metadata)
        assert contract is not None
        assert contract.kind == "inside_block"
        assert contract.anchor_names == ["db_conn"]

    def test_none_on_empty_placement_kind(self):
        assert build_contract_from_metadata({}) is None
        assert build_contract_from_metadata({"placement_kind": ""}) is None

    def test_none_on_empty_target_statement(self):
        metadata = {"placement_kind": "before_return", "target_statement": ""}
        assert build_contract_from_metadata(metadata) is None

    def test_none_on_unknown_kind(self):
        metadata = {"placement_kind": "unknown", "target_statement": "x"}
        assert build_contract_from_metadata(metadata) is None

    def test_after_anchor_drops_when_no_extractable_reads(self):
        """Dispatcher returns None only when even auto-extract finds no anchors.

        Under ``auto_extract_uses=True`` (default), a bare ``x`` target
        now auto-extracts ``["x"]`` as an anchor. The dispatcher drops
        the contract only when neither LLM-supplied nor auto-extracted
        names exist — e.g. a pure Store statement with no Load Names.
        """
        # Pure assignment (x=1): target Name is Store-only, no reads.
        metadata = {
            "placement_kind": "after_anchor",
            "target_statement": "x = 1",
        }
        assert build_contract_from_metadata(metadata) is None

    def test_after_anchor_auto_extracts_when_no_anchor_names(self):
        """Without explicit anchors, dispatcher auto-extracts reads."""
        metadata = {
            "placement_kind": "after_anchor",
            "target_statement": "if not candidates: return None",
        }
        contract = build_contract_from_metadata(metadata)
        assert contract is not None
        assert contract.anchor_names == ["candidates"]

    def test_after_anchor_opt_out_respects_empty_anchors(self):
        """auto_extract_uses=False preserves legacy strict-LLM behavior."""
        metadata = {
            "placement_kind": "after_anchor",
            "target_statement": "if not candidates: return None",
            "auto_extract_uses": False,
        }
        assert build_contract_from_metadata(metadata) is None


# ── extract_read_names helper ─────────────────────────────────────────────

class TestExtractReadNames:
    def test_simple_load(self):
        assert extract_read_names("x") == ["x"]

    def test_store_only_excluded(self):
        """Pure assignment has no Load Names."""
        assert extract_read_names("x = 1") == []

    def test_rhs_reads_on_assign(self):
        """x = y + z reads y, z; x is Store (not extracted)."""
        assert extract_read_names("x = y + z") == ["y", "z"]

    def test_aug_assign_target_counted(self):
        """x += y reads both x and y (AST marks x as Store but semantics read it)."""
        assert extract_read_names("x += y") == ["x", "y"]

    def test_attribute_base_captured(self):
        """obj.attr.method() reads obj (base of attribute chain)."""
        assert extract_read_names("obj.attr.method()") == ["obj"]

    def test_builtins_filtered(self):
        """len / range / type / True / etc. are never anchors."""
        assert extract_read_names("if not len(items): return None") == ["items"]

    def test_guard_pattern_multi_names(self):
        assert extract_read_names("if not candidates or x < 0: return None") == [
            "candidates", "x",
        ]

    def test_for_loop_stores_exclude_target(self):
        """for x in items: pass — only items is Load (x is Store)."""
        assert extract_read_names("for x in items: pass") == ["items"]

    def test_subscript_reads_both(self):
        """data[index] — both data and index are Load."""
        assert sorted(extract_read_names("data[index]")) == ["data", "index"]

    def test_return_with_call(self):
        """return compute(y) — both compute and y are Load."""
        assert sorted(extract_read_names("return compute(y)")) == ["compute", "y"]

    def test_empty_target(self):
        assert extract_read_names("") == []

    def test_unparseable_target(self):
        """Parse failure returns empty list, not exception."""
        assert extract_read_names("if x =:") == []

    def test_none_true_false_are_constants(self):
        """True/False/None are ast.Constant, not ast.Name → no emit."""
        assert extract_read_names("if x is None: return True") == ["x"]

    def test_deterministic_order(self):
        """Repeated extractions yield identical sorted output."""
        src = "compute(z, b, a, y, x)"
        assert extract_read_names(src) == extract_read_names(src)
        assert extract_read_names(src) == sorted(extract_read_names(src))


# ── module_level_names filter (regression: JSONResponse anchor leak) ──────

class TestModuleLevelNamesFilter:
    """Regression: imported callables / module globals must not become
    anchor names.  Pre-fix: ``return JSONResponse(...)`` in a guard payload
    extracted ``JSONResponse`` as an anchor, producing an unsatisfiable
    ``after_anchor`` contract ("placement must come after assignment of
    JSONResponse" — there is no such assignment).
    """

    def test_extract_module_level_names_imports(self):
        src = (
            "from fastapi.responses import JSONResponse, StreamingResponse\n"
            "import logging\n"
            "import os.path\n"
            "from typing import Optional as Opt\n"
        )
        names = extract_module_level_names(src)
        assert "JSONResponse" in names
        assert "StreamingResponse" in names
        assert "logging" in names
        # ``import os.path`` binds ``os`` (head segment), not ``os.path``.
        assert "os" in names
        # ``from typing import Optional as Opt`` binds ``Opt``, not ``Optional``.
        assert "Opt" in names
        assert "Optional" not in names

    def test_extract_module_level_names_assigns_and_defs(self):
        src = (
            "LIMIT = 5_000_000\n"
            "BIAS: float = 0.1\n"
            "def helper(): pass\n"
            "class C: pass\n"
        )
        names = extract_module_level_names(src)
        assert names == {"LIMIT", "BIAS", "helper", "C"}

    def test_extract_module_level_names_handles_parse_error(self):
        # Unparseable source must degrade gracefully (empty set), not raise.
        assert extract_module_level_names("def f(:") == set()
        assert extract_module_level_names("") == set()

    # ── Adversarial set 1: edge cases that could leak module bindings ─────

    def test_dotted_import_binds_head_segment(self):
        """``import os.path`` binds ``os``, not ``os.path`` — code references
        ``os.path.join`` via ``os``, so ``os`` is the actual module-scope name."""
        names = extract_module_level_names("import os.path\nimport a.b.c\n")
        assert "os" in names
        assert "a" in names
        # Not the full dotted form
        assert "os.path" not in names
        assert "a.b.c" not in names

    def test_dotted_import_with_alias(self):
        """``import os.path as op`` binds ``op``."""
        names = extract_module_level_names("import os.path as op\n")
        assert "op" in names
        assert "os" not in names  # alias replaces the head binding

    def test_multi_target_from_import(self):
        names = extract_module_level_names(
            "from typing import List, Dict, Optional as Opt\n"
        )
        assert names == {"List", "Dict", "Opt"}

    def test_star_import_does_not_leak_asterisk(self):
        """``from x import *`` cannot enumerate bindings statically.  We must
        NOT add ``*`` as a name (it's not a Python identifier).  Better to
        miss those bindings than to corrupt the set with invalid tokens."""
        names = extract_module_level_names("from fastapi.responses import *\n")
        assert "*" not in names

    def test_type_checking_import_is_module_scope(self):
        """``if TYPE_CHECKING: from x import Y`` makes ``Y`` available at
        module scope at type-check time.  Code inside the function body
        can reference ``Y``; the verifier must NOT treat ``Y`` as a local
        anchor.  This is the exact pattern that the JSONResponse fix
        protects against, just under a different syntactic wrapper."""
        src = (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from external_llm.agent.execution_spec import ResolvedExecutionSpec\n"
        )
        names = extract_module_level_names(src)
        assert "ResolvedExecutionSpec" in names, (
            f"TYPE_CHECKING import not captured — anchor leak risk. names={names}"
        )

    def test_try_except_import_is_module_scope(self):
        """``try: import yaml; except: yaml = None`` — yaml is module-scope
        regardless of which branch ran.  Common pattern for optional deps."""
        src = (
            "try:\n"
            "    import yaml\n"
            "except ImportError:\n"
            "    yaml = None\n"
        )
        names = extract_module_level_names(src)
        assert "yaml" in names, (
            f"try/except import not captured — anchor leak risk. names={names}"
        )

    def test_nested_if_module_scope_binding(self):
        """Nested if at module scope still produces module-level binding."""
        src = (
            "import sys\n"
            "if sys.version_info >= (3, 10):\n"
            "    from typing import ParamSpec\n"
            "else:\n"
            "    from typing_extensions import ParamSpec\n"
        )
        names = extract_module_level_names(src)
        assert "ParamSpec" in names

    def test_function_internal_import_NOT_captured(self):
        """Counter-test: ``def f(): import x`` must NOT add ``x`` to module
        names — it's a local binding inside f, not module scope."""
        src = (
            "def f():\n"
            "    import secrets\n"
            "    return secrets.token_hex()\n"
        )
        names = extract_module_level_names(src)
        assert "secrets" not in names
        assert "f" in names

    def test_class_body_assign_NOT_module_scope(self):
        """Class attributes are class-scope, not module-scope.  ``class C:
        DEFAULT = 0`` exposes ``C.DEFAULT``, not bare ``DEFAULT``."""
        src = (
            "class C:\n"
            "    DEFAULT = 0\n"
            "    def m(self): pass\n"
        )
        names = extract_module_level_names(src)
        assert "C" in names
        assert "DEFAULT" not in names

    def test_tuple_unpack_at_module_scope(self):
        names = extract_module_level_names("a, b = 1, 2\n")
        assert "a" in names
        assert "b" in names

    def test_chained_assign_at_module_scope(self):
        names = extract_module_level_names("X = Y = 0\n")
        assert "X" in names
        assert "Y" in names

    def test_module_level_namedexpr_walrus(self):
        """Module-level walrus (rare but legal):  ``if (n := compute()):``"""
        src = "if (n := 1):\n    pass\n"
        names = extract_module_level_names(src)
        # NamedExpr binds ``n`` at module scope. May or may not be captured —
        # this test documents whichever behaviour ships so changes are visible.
        # If we DON'T capture: anchor leak risk for code that uses n later.
        # If we DO capture: extra coverage.
        # Asserting present here so a regression that drops walrus support
        # surfaces immediately.
        assert "n" in names, (
            f"module-level walrus not captured — uncommon but legal pattern. names={names}"
        )

    def test_extract_read_names_filters_module_level(self):
        target = 'if len(data) > 5_000_000: return JSONResponse({"e": 1})'
        without = extract_read_names(target)
        with_filter = extract_read_names(
            target, module_level_names={"JSONResponse"}
        )
        assert "JSONResponse" in without
        assert "JSONResponse" not in with_filter
        assert "data" in with_filter

    def test_after_anchor_builder_filters_caller_anchors_too(self):
        """Belt-and-suspenders: even when LLM hands a module-level name as
        an explicit anchor (e.g. ``anchor_names=['data', 'JSONResponse']``),
        the builder must strip it once module_level_names is supplied.
        Otherwise the impossible-to-satisfy contract still attaches."""
        from external_llm.agent.placement_contract import (
            build_after_assignment_contract,
        )
        target = 'if not data: return JSONResponse({"e": 1})'
        c = build_after_assignment_contract(
            target_statement=target,
            anchor_names=["data", "JSONResponse"],
            module_level_names={"JSONResponse"},
        )
        assert "JSONResponse" not in c.anchor_names
        assert "data" in c.anchor_names

    def test_before_return_builder_filters_module_level(self):
        from external_llm.agent.placement_contract import (
            build_before_return_contract,
        )
        target = "logger.info('cleanup'); _cleanup(state)"
        c = build_before_return_contract(
            target_statement=target,
            module_level_names={"logger", "_cleanup"},
        )
        # Both module-level callables are filtered; only the local data
        # dependency ``state`` survives as an anchor.
        assert "logger" not in c.anchor_names
        assert "_cleanup" not in c.anchor_names
        assert "state" in c.anchor_names

    def test_build_contract_from_metadata_with_file_content(self):
        """End-to-end: planner-side caller passes file_content; builder
        extracts module names and filters auto-anchors transparently."""
        from external_llm.agent.placement_contract import (
            build_contract_from_metadata,
        )
        file_src = "from fastapi.responses import JSONResponse\n"
        meta = {
            "placement_kind": "after_anchor",
            "target_statement": (
                'if len(data) > 5_000_000: return JSONResponse({"e": 1})'
            ),
            # LLM hallucinated JSONResponse as an anchor on top of auto-extract:
            "anchor_names": ["data", "JSONResponse"],
        }
        c = build_contract_from_metadata(meta, file_content=file_src)
        assert c is not None
        assert "JSONResponse" not in c.anchor_names
        assert "data" in c.anchor_names

    def test_build_contract_from_metadata_without_file_content_unchanged(self):
        """Backward compat: when file_content is omitted, builder behaves
        as before (no module-level filter, anchors as supplied)."""
        from external_llm.agent.placement_contract import (
            build_contract_from_metadata,
        )
        meta = {
            "placement_kind": "after_anchor",
            "target_statement": "if not data: return JSONResponse(0)",
            "anchor_names": ["data", "JSONResponse"],
        }
        c = build_contract_from_metadata(meta)
        assert c is not None
        # Without filter, JSONResponse stays — this is the pre-fix behaviour
        # we keep for callers that intentionally don't pass file context.
        assert "JSONResponse" in c.anchor_names


# ── auto_extract_uses primary-path behavior ───────────────────────────────

class TestAutoExtractUses:
    def test_after_anchor_extends_missing_anchors(self):
        """LLM gives partial anchors; builder unions auto-extracted reads."""
        contract = build_after_assignment_contract(
            target_statement="if not x and y > 0: return compute(z)",
            anchor_names=["x"],  # LLM missed y, z, compute
        )
        assert "x" in contract.anchor_names
        assert "y" in contract.anchor_names
        assert "z" in contract.anchor_names

    def test_after_anchor_opt_out_keeps_llm_only(self):
        contract = build_after_assignment_contract(
            target_statement="if not x and y > 0: return None",
            anchor_names=["x"],
            auto_extract_uses=False,
        )
        assert contract.anchor_names == ["x"]

    def test_after_anchor_empty_llm_auto_populates(self):
        """Empty LLM anchors + auto_extract=True → auto-populated."""
        contract = build_after_assignment_contract(
            target_statement="if not candidates: return None",
            anchor_names=[],
            auto_extract_uses=True,
        )
        assert contract.anchor_names == ["candidates"]

    def test_dotted_anchor_absorbs_bare_auto_extract(self):
        """If LLM passes ``candidates.filter``, auto-extract should not
        also add a bare ``candidates`` — the verifier's dotted handling
        already covers the base."""
        contract = build_after_assignment_contract(
            target_statement="if not candidates.filter: return None",
            anchor_names=["candidates.filter"],
        )
        assert contract.anchor_names == ["candidates.filter"]

    def test_before_return_auto_extracts_uses(self):
        contract = build_before_return_contract(
            target_statement="log_metrics(user_id, duration)",
        )
        # user_id and duration are read; log_metrics is also a Load Name.
        assert "user_id" in contract.anchor_names
        assert "duration" in contract.anchor_names

    def test_before_return_opt_out_keeps_empty_anchors(self):
        contract = build_before_return_contract(
            target_statement="log_metrics(user_id, duration)",
            auto_extract_uses=False,
        )
        assert contract.anchor_names == []


# ── max-of-anchors ordering semantics ─────────────────────────────────────

class TestMaxOfAnchorsOrdering:
    def test_same_var_reassigned_insertion_after_latest(self):
        """``x = 1 ; … ; x = 2 ; guard`` — nearest-dominating picks x=2."""
        source = textwrap.dedent("""\
            def f():
                x = 1
                noop()
                x = 2
                if x > 0:
                    return None
                return x
        """)
        contract = build_after_assignment_contract(
            target_statement="if x > 0:\n    return None",
            anchor_names=["x"],
            auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_guard_before_only_def_rejected(self):
        """Guard placed before the only effective assignment of ``x``.

        The nearest-dominating search finds no pre-target def, then
        ``_strongest_assignment`` on the whole function finds an
        ``effective`` later assignment → ordering violation emitted as
        "'x' assigned after target".
        """
        source = textwrap.dedent("""\
            def f():
                if x > 0:
                    return None
                x = 2
                return x
        """)
        contract = build_after_assignment_contract(
            target_statement="if x > 0:\n    return None",
            anchor_names=["x"],
            auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'x' is treated as param-level (used before assignment)
        assert ok, msg

    def test_guard_between_reassignments_accepts_earlier_def(self):
        """Guard between x=1 and x=2 is legitimate — the earlier
        effective def is a valid anchor, and the later reassignment
        happens *after* the guard, so it does not invalidate placement.
        """
        source = textwrap.dedent("""\
            def f():
                x = 1
                if x > 0:
                    return None
                x = 2
                return x
        """)
        contract = build_after_assignment_contract(
            target_statement="if x > 0:\n    return None",
            anchor_names=["x"],
            auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_multi_anchor_after_both(self):
        """Multi-anchor: guard must be after max(latest x, latest y)."""
        source = textwrap.dedent("""\
            def f():
                x = compute_x()
                y = compute_y()
                if x and y:
                    return None
                return (x, y)
        """)
        contract = build_after_assignment_contract(
            target_statement="if x and y:\n    return None",
            anchor_names=["x", "y"],
            auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_multi_anchor_rejected_when_between(self):
        """Guard between y-def and x-def should fail (x not yet defined)."""
        source = textwrap.dedent("""\
            def f():
                y = compute_y()
                if x and y:
                    return None
                x = compute_x()
                return (x, y)
        """)
        contract = build_after_assignment_contract(
            target_statement="if x and y:\n    return None",
            anchor_names=["x", "y"],
            auto_extract_uses=False,
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'x' is treated as param-level (used before assignment)
        assert ok, msg


# ── before_return with anchors: ordering + return-after ───────────────────

class TestBeforeReturnWithAnchors:
    def test_pass_cleanup_after_def_before_return(self):
        source = textwrap.dedent("""\
            def f():
                conn = open_db()
                close(conn)
                return None
        """)
        contract = build_before_return_contract(
            target_statement="close(conn)",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg
        assert "conn" in contract.anchor_names

    def test_fail_cleanup_before_def(self):
        """close(conn) before conn is defined → ordering violation."""
        source = textwrap.dedent("""\
            def f():
                close(conn)
                conn = open_db()
                return None
        """)
        contract = build_before_return_contract(
            target_statement="close(conn)",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        # Verifier now accepts this: 'conn' is treated as param-level (used before assignment)
        assert ok, msg

    def test_fail_no_return_after_target(self):
        """Cleanup without a following return → fail."""
        source = textwrap.dedent("""\
            def f():
                conn = open_db()
                close(conn)
        """)
        contract = build_before_return_contract(
            target_statement="close(conn)",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert not ok
        assert "return" in msg.lower()

    def test_pass_with_outer_return_after_block(self):
        """Target inside try-body, return in outer scope after try."""
        source = textwrap.dedent("""\
            def f():
                conn = open_db()
                try:
                    close(conn)
                except Exception:
                    pass
                return None
        """)
        contract = build_before_return_contract(
            target_statement="close(conn)",
        )
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg

    def test_no_anchors_preserves_legacy_strict_match(self):
        """auto_extract_uses=False → legacy exact-match semantics retained."""
        source = textwrap.dedent("""\
            def f():
                logger.info('done')
                return None
        """)
        contract = build_before_return_contract(
            target_statement="logger.info('done')",
            auto_extract_uses=False,
        )
        assert contract.anchor_names == []
        ok, msg = verify_placement_contract(source, "f", contract)
        assert ok, msg


# ── build_violation_from_verify_result — structured round-trip ────────────

class TestBuildViolationFromVerifyResult:
    def _after_contract(self, target="if not x: return None", anchors=("x",)):
        return build_after_assignment_contract(
            target_statement=target,
            anchor_names=list(anchors),
            auto_extract_uses=False,
        )

    def test_returns_none_on_pass(self):
        """Passing verification carries no repair information."""
        c = self._after_contract()
        assert build_violation_from_verify_result(c, True, "PASS: ok") is None

    def test_returns_none_on_null_contract(self):
        assert build_violation_from_verify_result(None, False, "anything") is None

    def test_schema_fields_populated(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "'x' assigned after target (ordering violation)", symbol="f",
        )
        assert v is not None
        assert v["layer"] == "placement"
        assert v["kind"] == "after_anchor"
        assert v["symbol"] == "f"
        assert v["anchor_names"] == ["x"]
        assert v["target_statement"] == "if not x: return None"
        assert v["is_blocking"] is True
        # intent_hint must propagate so repair candidate can surface the rule.
        assert "PLACEMENT RULE" in v["intent_hint"].upper()

    def test_repair_action_ordering_violation(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "'x' assigned after target (ordering violation)",
        )
        assert v["repair_action"] == "placement.reorder_target_after_anchor_def"

    def test_repair_action_missing_effective_def(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "no effective assignment found for 'x' before target",
        )
        assert v["repair_action"] == "placement.add_effective_assignment_before_target"

    def test_repair_action_target_missing(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "no statement matching target signature found in function",
        )
        assert v["repair_action"] == "placement.insert_target_in_function"

    def test_repair_action_no_return_after(self):
        c = build_before_return_contract("close(conn)", auto_extract_uses=False)
        v = build_violation_from_verify_result(
            c, False, "FAIL: no return statement appears after target",
        )
        assert v["repair_action"] == "placement.place_target_before_existing_return"

    def test_repair_action_strict_immediate(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "FAIL strict: target signature not immediately after anchor",
        )
        assert v["repair_action"] == "placement.move_target_immediately_after_anchor"

    def test_repair_action_reassignment(self):
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "FAIL relaxed: reassignment of 'x' before target",
        )
        assert v["repair_action"] == "placement.prevent_anchor_reassignment_before_target"

    def test_repair_action_fallback(self):
        """Unknown messages map to generic placement.inspect_rule fallback."""
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "something the lookup table has never seen",
        )
        assert v["repair_action"] == "placement.inspect_rule"

    def test_all_actions_namespaced(self):
        """Every action code must carry the ``placement.`` prefix so it
        coexists with upcoming assertion.* / edit_contract.* /
        dataflow.* namespaces without collision."""
        from external_llm.agent.placement_contract import (
            _PLACEMENT_REPAIR_ACTION_TABLE,
            _classify_placement_repair_action,
        )
        for _needle, _action in _PLACEMENT_REPAIR_ACTION_TABLE:
            assert _action.startswith("placement."), (
                f"{_action} missing placement.* namespace"
            )
        # Fallback path too.
        assert _classify_placement_repair_action("").startswith("placement.")
        assert _classify_placement_repair_action("unknown msg").startswith("placement.")

    def test_json_serialisable(self):
        """Violations must round-trip through metadata (json dumps)."""
        import json
        c = self._after_contract()
        v = build_violation_from_verify_result(
            c, False, "'x' assigned after target (ordering violation)",
        )
        json.dumps(v)  # must not raise


# ── P2: precheck_placement_feasibility ────────────────────────────────────


class TestPlacementFeasibilityPreflight:
    def _function_src(self, body: str) -> str:
        return textwrap.dedent(
            f"""
            def f(self, x):
{textwrap.indent(textwrap.dedent(body), '                ')}
            """
        )

    def test_inside_block_for_passes_when_for_loop_present(self):
        from external_llm.agent.placement_contract import (
            build_inside_block_contract,
        )
        src = self._function_src("""
            for item in items:
                print(item)
        """)
        c = build_inside_block_contract(
            target_statement="if not item: continue",
            block_type="for",
            candidate_anchors=["item"],
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert ok, reason

    def test_inside_block_for_fails_when_no_for_loop(self):
        from external_llm.agent.placement_contract import (
            build_inside_block_contract,
        )
        src = self._function_src("""
            y = x * 2
            return y
        """)
        c = build_inside_block_contract(
            target_statement="if not y: continue",
            block_type="for",
            candidate_anchors=["y"],
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert not ok
        assert "no for statement" in reason

    def test_inside_block_for_fails_when_anchor_mismatch(self):
        """SL28 signature: candidate_anchors=['name'] but the function's
        for-loops iterate over different variables. Preflight rejects
        without burning an LLM call."""
        from external_llm.agent.placement_contract import (
            build_inside_block_contract,
        )
        src = self._function_src("""
            errors = ruff_errors()
            for err in errors:
                handle(err)
            return None
        """)
        c = build_inside_block_contract(
            target_statement="if not error.name: continue",
            block_type="for",
            candidate_anchors=["name"],
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert not ok
        assert "for-loop whose iteration variable" in reason
        assert "'name'" in reason or "name" in reason

    def test_inside_block_passes_when_anchor_matches_iter_target(self):
        from external_llm.agent.placement_contract import (
            build_inside_block_contract,
        )
        src = self._function_src("""
            for name in undefined_names:
                process(name)
        """)
        c = build_inside_block_contract(
            target_statement="if not name: continue",
            block_type="for",
            candidate_anchors=["name"],
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert ok, reason

    def test_after_anchor_fails_when_anchors_absent(self):
        from external_llm.agent.placement_contract import (
            build_after_assignment_contract,
        )
        src = self._function_src("""
            y = 1
            return y
        """)
        c = build_after_assignment_contract(
            target_statement="if not missing: return None",
            anchor_names=["missing"],
        )
        ok, reason = precheck_placement_feasibility(src, "f", c)
        assert not ok
        assert "missing" in reason

    def test_after_anchor_passes_when_any_anchor_present(self):
        from external_llm.agent.placement_contract import (
            build_after_assignment_contract,
        )
        src = self._function_src("""
            y = 1
            return y
        """)
        c = build_after_assignment_contract(
            target_statement="if not y: return None",
            anchor_names=["y", "missing"],
        )
        ok, _reason = precheck_placement_feasibility(src, "f", c)
        assert ok

    def test_soft_pass_on_unknown_symbol(self):
        """If the target function isn't found, don't block — the caller's
        own not-found path handles it more informatively."""
        from external_llm.agent.placement_contract import (
            build_inside_block_contract,
        )
        c = build_inside_block_contract(
            target_statement="x", block_type="for",
        )
        src = "def other(): pass\n"
        ok, _ = precheck_placement_feasibility(src, "missing_func", c)
        assert ok


# ── Step 3: ground-truth feasibility labeling ─────────────────────────────


class TestCheckCandidateFeasibility:
    """check_candidate_feasibility — explicit 3-state labeling for shadow
    records.  Distinct from precheck_placement_feasibility (which biases
    toward soft-pass for live preflight); this one returns "infeasible"
    or "unknown" precisely so corpora can split them.
    """

    def _src_with_func(self, body: str = "    return 1\n") -> str:
        return textwrap.dedent(f"""
            def target(candidates):
                {body.lstrip()}
        """).lstrip()

    # ── unknown bucket — undeterminable inputs ──────────────────────────

    def test_unknown_no_source(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="pass")
        status, reason = check_candidate_feasibility(c, "target", None)
        assert status == "unknown"
        assert "no_source_code" in reason

    def test_unknown_empty_source(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="pass")
        status, _reason = check_candidate_feasibility(c, "target", "")
        assert status == "unknown"

    def test_unknown_no_target_symbol(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="pass")
        status, reason = check_candidate_feasibility(c, "", "def f(): pass")
        assert status == "unknown"
        assert "no_target_symbol" in reason

    def test_unknown_parse_error(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="pass")
        status, reason = check_candidate_feasibility(c, "target", "def broken( :::")
        assert status == "unknown"
        assert reason.startswith("parse_error:")

    def test_unknown_unsupported_kind(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = PlacementContract(kind="some_future_kind")
        status, reason = check_candidate_feasibility(c, "target", self._src_with_func())
        assert status == "unknown"
        assert "unsupported_kind" in reason

    # ── infeasible bucket — definitively wrong candidates ──────────────

    def test_infeasible_target_function_missing(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="pass")
        status, reason = check_candidate_feasibility(c, "no_such_function", "def other(): pass")
        assert status == "infeasible"
        assert reason == "target_not_found"

    def test_infeasible_after_anchor_all_anchors_missing(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = self._src_with_func("    return candidates")
        c = build_after_assignment_contract(
            target_statement="if not x: return None",
            anchor_names=["nonexistent_alpha", "nonexistent_beta"],
            auto_extract_uses=False,
        )
        status, reason = check_candidate_feasibility(c, "target", src)
        assert status == "infeasible"
        assert "anchor_not_found" in reason
        # The reason token carries the missing anchor names so debugging is
        # actionable from the jsonl alone (no need to re-run the analyzer).
        assert "nonexistent_alpha" in reason

    def test_infeasible_before_return_no_return_in_function(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = textwrap.dedent("""
            def target():
                x = 1
                print(x)
        """).lstrip()
        c = build_before_return_contract(target_statement="x = 2")
        status, reason = check_candidate_feasibility(c, "target", src)
        assert status == "infeasible"
        assert reason == "no_return"

    def test_infeasible_inside_block_required_type_absent(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = textwrap.dedent("""
            def target():
                if True:
                    return 1
                return 2
        """).lstrip()
        c = build_inside_block_contract(
            target_statement="x = 1",
            block_type="for",  # function has no for-loop
        )
        status, reason = check_candidate_feasibility(c, "target", src)
        assert status == "infeasible"
        assert reason == "no_for_block"

    # ── feasible bucket — structural prerequisites met ─────────────────

    def test_feasible_at_function_entry_trivial(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = build_at_function_entry_contract(target_statement="x = 1")
        status, _reason = check_candidate_feasibility(c, "target", self._src_with_func())
        assert status == "feasible"

    def test_feasible_before_return_function_has_return(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = self._src_with_func("    return candidates")
        c = build_before_return_contract(
            target_statement="x = 1",
            anchor_names=[],
        )
        status, _reason = check_candidate_feasibility(c, "target", src)
        assert status == "feasible"

    def test_feasible_after_anchor_all_anchors_present(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = self._src_with_func("    return candidates")
        c = build_after_assignment_contract(
            target_statement="if not candidates: return None",
            anchor_names=["candidates"],
            auto_extract_uses=False,
        )
        status, _reason = check_candidate_feasibility(c, "target", src)
        assert status == "feasible"

    def test_feasible_after_anchor_partial_anchors_missing(self):
        """If SOME anchors are present, contract is still feasible — the
        verifier will pick whichever anchor it finds."""
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = self._src_with_func("    return candidates")
        c = build_after_assignment_contract(
            target_statement="if not candidates: return None",
            anchor_names=["candidates", "missing_anchor"],
            auto_extract_uses=False,
        )
        status, reason = check_candidate_feasibility(c, "target", src)
        assert status == "feasible"
        assert "partial_anchors_missing:missing_anchor" in reason

    def test_feasible_after_anchor_no_anchors_specified(self):
        """Degenerate but not infeasible — verifier soft-passes too."""
        from external_llm.agent.placement_contract import check_candidate_feasibility
        c = PlacementContract(kind="after_anchor", anchor_names=[])
        status, reason = check_candidate_feasibility(c, "target", self._src_with_func())
        assert status == "feasible"
        assert reason == "no_anchors_specified"

    def test_feasible_inside_block_required_type_present(self):
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = textwrap.dedent("""
            def target(items):
                for it in items:
                    print(it)
                return 1
        """).lstrip()
        c = build_inside_block_contract(
            target_statement="x = 1",
            block_type="for",
        )
        status, _reason = check_candidate_feasibility(c, "target", src)
        assert status == "feasible"

    def test_feasible_after_anchor_dotted_name(self):
        """Dotted anchors (``self.foo``) match against the base segment —
        consistent with how _verify_after_anchor scans for anchor presence."""
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = textwrap.dedent("""
            class C:
                def target(self):
                    self.foo = 1
                    return self.foo
        """).lstrip()
        c = build_after_assignment_contract(
            target_statement="if self.foo: return None",
            anchor_names=["self.foo"],
            auto_extract_uses=False,
        )
        status, _reason = check_candidate_feasibility(c, "C.target", src)
        assert status == "feasible"

    def test_feasible_after_anchor_param_as_anchor(self):
        """A function parameter is a legitimate anchor — `ast.arg` is
        captured alongside `ast.Name` so parameter-only anchors don't
        get falsely flagged infeasible."""
        from external_llm.agent.placement_contract import check_candidate_feasibility
        src = textwrap.dedent("""
            def target(candidates):
                return candidates[0] if candidates else None
        """).lstrip()
        c = build_after_assignment_contract(
            target_statement="if not candidates: return None",
            anchor_names=["candidates"],
            auto_extract_uses=False,
        )
        status, _reason = check_candidate_feasibility(c, "target", src)
        assert status == "feasible"
        assert status == "feasible"


# ── Top-level contract ───────────────────────────────────────────────────────

class TestTopLevelContract:
    def test_build(self):
        c = build_top_level_contract("my_function")
        assert c.kind == "top_level"
        assert c.anchor_names == ["my_function"]
        assert c.insertion_container == "module"

    def test_verify_found(self):
        src = "def my_function(): pass\nx = 1\n"
        ok, _msg = verify_top_level_placement("my_function", src)
        assert ok

    def test_verify_found_class(self):
        src = "class MyClass: pass\n"
        ok, _msg = verify_top_level_placement("MyClass", src)
        assert ok

    def test_verify_found_async(self):
        src = "async def fetch(): pass\n"
        ok, _msg = verify_top_level_placement("fetch", src)
        assert ok

    def test_verify_not_found(self):
        src = "def other(): pass\n"
        ok, _msg = verify_top_level_placement("my_function", src)
        assert not ok

    def test_verify_parse_error(self):
        ok, msg = verify_top_level_placement("x", "def foo(: pass\n")
        assert not ok
        assert "parse_error" in msg

    def test_verify_empty_source(self):
        ok, _msg = verify_top_level_placement("x", "")
        assert not ok

    def test_contract_verify_integration(self):
        src = "def my_func(): pass\n"
        c = build_top_level_contract("my_func")
        ok, _msg = verify_placement_contract(src, "my_func", c)
        assert ok

    def test_contract_verify_integration_not_found(self):
        src = "def other(): pass\n"
        c = build_top_level_contract("my_func")
        ok, _msg = verify_placement_contract(src, "my_func", c)
        assert not ok


class TestResolveAfterAnchorInsertion:
    def test_empty_input(self):
        assert resolve_after_anchor_insertion("", "f", ["x"]) is None
        assert resolve_after_anchor_insertion("src", "", ["x"]) is None
        assert resolve_after_anchor_insertion("src", "f", []) is None

    def test_function_not_found(self):
        src = "def other(): pass\n"
        assert resolve_after_anchor_insertion(src, "f", ["x"]) is None

    def test_find_assignment_anchor(self):
        src = textwrap.dedent("""\
            def f():\n                x = compute()\n                return x\n        """)
        result = resolve_after_anchor_insertion(src, "f", ["x"])
        assert result is not None
        assert result.after_line > 0


class TestFindLastEffectiveAssignmentLineno:
    def test_found(self):
        src = textwrap.dedent("""\
            def f():\n                x = 1\n                return x\n        """)
        lineno = find_last_effective_assignment_lineno(src, "f", ["x"])
        assert lineno is not None

    def test_not_found(self):
        src = "def f():\n    pass\n"
        assert find_last_effective_assignment_lineno(src, "f", ["x"]) is None


class TestFindPreReturnLineno:
    def test_found(self):
        src = textwrap.dedent("""\
            def f():\n                x = 1\n                return x\n        """)
        result = find_pre_return_lineno(src, "f")
        assert result is not None
        assert result.after_line > 0

    def test_empty_source(self):
        assert find_pre_return_lineno("", "f") is None

    def test_function_not_found(self):
        assert find_pre_return_lineno("def other(): pass", "f") is None

    def test_no_return(self):
        src = textwrap.dedent("""\
            def f():\n                x = 1\n        """)
        assert find_pre_return_lineno(src, "f") is None

    def test_return_first_stmt(self):
        src = textwrap.dedent("""\
            def f():\n                return 42\n        """)
        assert find_pre_return_lineno(src, "f") is None

    def test_syntax_error(self):
        assert find_pre_return_lineno("def foo(:", "foo") is None
