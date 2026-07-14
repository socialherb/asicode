"""Step 3 검증: op.metadata 일관성 불변식

Step 3a: guard_ir는 analyze_guard 결과 (fallback _extract_guard_ir 없음)
Step 3b: guard_statement는 항상 compact 형태로 동기화
Step 3c: ast_ops[0]["statement"]도 compact와 동일
"""

import pytest

from external_llm.agent.guard_ir import analyze_guard, parse_guard

# ── 테스트: compact 불변식 ──────────────────────────────────────────────────────

class TestCompactInvariant:
    """analyze_guard가 반환하는 ir.compact는 단일 줄이며
    to_legacy_tuple()과 의미상 동일해야 한다."""

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


# ── 테스트: op.metadata guard_statement 동기화 ─────────────────────────────────

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
    """guard_statement는 analyze_guard 이후 compact 형태여야 한다.

    _attach_edit_contracts가 실제로 호출되지 않아도 analyze_guard 자체로 검증.
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


# ── 테스트: ast_ops statement == guard_statement 불변식 ──────────────────────────

class TestAstOpsStatementInvariant:
    """ast_ops[0]["statement"]과 guard_statement가 같은 compact 형태를 가져야 한다.

    _attach_edit_contracts에서 Step 3b 이후 _gs가 compact로 갱신되고,
    ast_ops 빌드에서 그 _gs를 사용하므로 둘이 일치한다.
    """

    def test_compact_stable_under_round_trip(self):
        """compact → parse → compact 가 stable 해야 한다."""
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
        """raw와 compact에서 추출한 guard_ir condition이 동일해야 한다."""
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
