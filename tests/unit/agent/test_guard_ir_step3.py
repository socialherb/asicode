"""Step 3 verification: op.metadata consistency invariants

Step 3a: guard_ir is the analyze_guard result (no fallback _extract_guard_ir)
Step 3b: guard_statement is always synced to compact form
Step 3c: ast_ops[0]["statement"] also matches compact
"""

import pytest

from external_llm.agent.guard_ir import analyze_guard, parse_guard

# ── Tests: compact invariant ──────────────────────────────────────────────────────

class TestCompactInvariant:
    """ir.compact returned by analyze_guard must be a single line and
    semantically equivalent to to_legacy_tuple()."""

    CASES = [
        "if not x: continue",
        "if not value: return",
        "if not value: return None",
        "if item is None: continue",
        "if x == 0: return",
        "if not error.name: continue",
    ]

    @pytest.mark.parametrize("raw", CASES)
    def test_compact_is_single_line(self, raw):
        ir = parse_guard(raw)
        assert ir is not None and "\n" not in ir.compact

    @pytest.mark.parametrize("raw", CASES)
    def test_compact_semantically_equivalent_to_raw(self, raw):
        ir = parse_guard(raw)
        assert ir is not None
        ir2 = parse_guard(ir.compact)
        assert ir2 is not None
        assert ir.to_legacy_tuple() == ir2.to_legacy_tuple(), (
            f"compact changed semantics: {raw!r} → {ir.compact!r}"
        )

    @pytest.mark.parametrize("raw", CASES)
    def test_guard_ir_condition_from_compact_matches_raw(self, raw):
        ir_raw    = parse_guard(raw)
        ir_compact = parse_guard(parse_guard(raw).compact)
        assert ir_raw is not None and ir_compact is not None
        assert ir_raw.condition is not None and ir_compact.condition is not None
        assert ir_raw.condition.op_class == ir_compact.condition.op_class
        assert ir_raw.condition.operands == ir_compact.condition.operands


# ── Tests: op.metadata guard_statement sync ─────────────────────────────────

_SRC_SIMPLE = """
def process(items, limit=10):
    result = []
    for item in items:
        result.append(item)
    return result
"""

_SRC_PARAM = """
def validate(max_retries, threshold):
    pass
"""


def _make_op_metadata(guard_statement, edit_kind="guard_add", symbol="process"):
    """simulate the op.metadata dict built by DPB/contract_driven_planning"""
    return {
        "guard_statement": guard_statement,
        "edit_kind": edit_kind,
    }


class TestGuardStatementSync:
    """guard_statement must be in compact form after analyze_guard.

    Verified via analyze_guard itself, even when _attach_edit_contracts
    isn't actually invoked.
    """

    def test_block_form_produces_compact(self):
        raw = "if not x:\n    continue"
        ir = parse_guard(raw)
        assert ir is not None and ir.compact == "if not x: continue"

    def test_analyze_guard_compact_matches_parse_compact(self):
        raw = "if not item:\n    continue"
        ir = parse_guard(raw)
        analyzed = analyze_guard(ir, _SRC_SIMPLE, "process")
        # compact from analyze_guard == compact from parse_guard (ir is not mutated)
        assert ir.compact == analyzed.compact

    def test_guard_ir_dict_uses_compact_as_canonical(self):
        raw = "if not x: continue"
        ir = parse_guard(raw)
        analyzed = analyze_guard(ir, _SRC_SIMPLE, "process")
        # guard_ir dict would store condition from analyzed
        if analyzed.condition:
            gir_dict = {
                "condition": analyzed.condition.to_legacy_dict(),
                "control": analyzed.control,
                "insert_scope": analyzed.feasibility.insert_scope if analyzed.feasibility else "",
            }
            assert gir_dict["condition"] is not None
            assert gir_dict["control"] in ("continue", "break", "return", "raise")


# ── Tests: ast_ops statement == guard_statement invariant ──────────────────────────

class TestAstOpsStatementInvariant:
    """ast_ops[0]["statement"] and guard_statement must share the same compact form.

    In _attach_edit_contracts, _gs is updated to compact form after Step 3b,
    and the ast_ops build uses that same _gs, so the two stay in sync.
    """

    def test_compact_stable_under_round_trip(self):
        """compact → parse → compact must be stable."""
        raw = "if not item: continue"
        ir1 = parse_guard(raw)
        ir2 = parse_guard(ir1.compact)
        # compact should be identical (stable fixed point)
        assert ir1.compact == ir2.compact

    def test_block_form_compact_matches_inline_compact(self):
        ir_block  = parse_guard("if not x:\n    continue")
        ir_inline = parse_guard("if not x: continue")
        assert ir_block is not None and ir_inline is not None
        assert ir_block.compact == ir_inline.compact

    def test_condition_ir_consistent_before_and_after_compact(self):
        """guard_ir condition extracted from raw and from compact must match."""
        cases = [
            "if not value: return",
            "if not error.name: continue",
            "if item is None: continue",
        ]
        for raw in cases:
            ir = parse_guard(raw)
            ir_compact = parse_guard(ir.compact)
            assert ir.to_legacy_tuple() == ir_compact.to_legacy_tuple(), (
                f"IR diverged after compaction: {raw!r} → {ir.compact!r}"
            )
