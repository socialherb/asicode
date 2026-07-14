"""Step 1 verification: parse_guard() GuardIR invariants

Verifies that parse_guard() produces the correct GuardIR for a variety of
inputs. _extract_guard_ir was removed in Step 8, so this verifies the
invariants of parse_guard itself.
"""

import pytest

from external_llm.agent.guard_ir import parse_guard

# ──────────────────────────────────────────────────────────────────────────────
# parse_guard fixtures — reuse the existing _extract_guard_ir cases as-is
# ──────────────────────────────────────────────────────────────────────────────

# (description, raw_guard_string, expected_condition_op, expected_control)
# expected_condition_op=None / expected_control="" → parse failure or no control flow
GUARD_CASES = [
    ("simple not + continue",    "if not x: continue",                    "Not",     "continue"),
    ("not name continue",        "if not name: continue",                  "Not",     "continue"),
    ("attribute access continue","if not error.name: continue",            "Not",     "continue"),
    ("param guard raise",        "if not value: raise ValueError('missing')", "Not",  "raise"),
    ("param guard return",       "if not value: return",                   "Not",     "return"),
    ("param guard return None",  "if not value: return None",              "Not",     "return"),
    ("break guard",              "if done: break",                         "Name",    "break"),
    ("and condition continue",   "if not a and not b: continue",           "And",     "continue"),
    ("compare eq guard",         "if x == 0: return",                     "Eq",      "return"),
    ("compare notin guard",      "if item not in seen: continue",          "NotIn",   "continue"),
    ("is None guard",            "if x is None: return None",             "Is",      "return"),
    ("self attr guard",          "if not self.enabled: return",            "Not",     "return"),
    ("nested attribute",         "if not obj.sub.attr: continue",          "Not",     "continue"),
    ("raise valueerror",         "if x < 0: raise ValueError(x)",         "Lt",      "raise"),
    # Block form → same semantics as inline
    ("block form continue",      "if not x:\n    continue",               "Not",     "continue"),
    ("block form return",        "if not y:\n    return None",            "Not",     "return"),
    # Edge: bare if without control flow
    ("no control flow",          "if x: pass",                            None,      ""),
    # Edge: syntax error
    ("syntax error",             "if not :",                              None,      ""),
]

NULL_CASES = [
    ("empty string",    ""),
    ("whitespace only", "   "),
]


@pytest.mark.parametrize("desc,raw,exp_op,exp_ctrl", GUARD_CASES)
def test_parse_guard_condition_and_control(desc: str, raw: str, exp_op, exp_ctrl) -> None:
    """parse_guard produces correct condition.op_class and control."""
    ir = parse_guard(raw)
    assert ir is not None, f"[{desc}] parse_guard returned None for {raw!r}"

    if exp_op is None:
        assert ir.condition is None, f"[{desc}] expected no condition, got {ir.condition}"
        assert ir.control == exp_ctrl, f"[{desc}] expected control={exp_ctrl!r}, got {ir.control!r}"
    else:
        assert ir.condition is not None, f"[{desc}] expected condition with op={exp_op!r}, got None"
        assert ir.condition.op_class == exp_op, (
            f"[{desc}] op_class: expected {exp_op!r}, got {ir.condition.op_class!r}"
        )
        assert ir.control == exp_ctrl, (
            f"[{desc}] control: expected {exp_ctrl!r}, got {ir.control!r}"
        )


@pytest.mark.parametrize("desc,raw", NULL_CASES)
def test_parse_guard_empty_returns_none(desc: str, raw: str) -> None:
    assert parse_guard(raw) is None, f"[{desc}] expected None for {raw!r}"


# ──────────────────────────────────────────────────────────────────────────────
# Basic GuardIR field invariants
# ──────────────────────────────────────────────────────────────────────────────

class TestGuardIRFields:
    def test_canonical_not_empty_for_valid_guard(self) -> None:
        ir = parse_guard("if not x: continue")
        assert ir is not None
        assert ir.canonical != ""

    def test_compact_is_single_line(self) -> None:
        ir = parse_guard("if not x:\n    continue")
        assert ir is not None
        assert "\n" not in ir.compact, f"compact should be single-line: {ir.compact!r}"

    def test_compact_matches_inline_guard(self) -> None:
        ir = parse_guard("if not x: continue")
        assert ir is not None
        assert ir.compact == "if not x: continue"

    def test_raw_preserved(self) -> None:
        raw = "if not   x :  continue"
        ir = parse_guard(raw)
        assert ir is not None
        assert ir.raw == raw

    def test_control_field_correct(self) -> None:
        cases = [
            ("if not x: continue", "continue"),
            ("if not x: break",    "break"),
            ("if not x: return",   "return"),
            ("if not x: raise ValueError()", "raise"),
        ]
        for raw, expected_ctrl in cases:
            ir = parse_guard(raw)
            assert ir is not None and ir.control == expected_ctrl, (
                f"{raw!r}: expected control={expected_ctrl!r}, got {ir.control!r}"
            )

    def test_no_control_flow_gives_none_condition(self) -> None:
        ir = parse_guard("if x: pass")
        assert ir is not None
        assert ir.condition is None
        assert ir.control == ""

    def test_syntax_error_gives_empty_canonical(self) -> None:
        ir = parse_guard("if not :")
        assert ir is not None
        assert ir.canonical == ""
        assert ir.condition is None

    def test_empty_returns_none(self) -> None:
        assert parse_guard("") is None
        assert parse_guard("   ") is None

    def test_attribute_pairs_captured(self) -> None:
        ir = parse_guard("if not error.name: continue")
        assert ir is not None and ir.condition is not None
        assert ("error", "name") in ir.condition.attribute_pairs

    def test_is_parsed_property(self) -> None:
        assert parse_guard("if not x: continue").is_parsed is True
        assert parse_guard("if not :").is_parsed is False


# ──────────────────────────────────────────────────────────────────────────────
# canonical / compact form verification
# ──────────────────────────────────────────────────────────────────────────────

class TestCanonicalCompact:
    def test_block_and_inline_give_same_canonical(self) -> None:
        ir_block  = parse_guard("if not x:\n    continue")
        ir_inline = parse_guard("if not x: continue")
        assert ir_block is not None and ir_inline is not None
        assert ir_block.canonical == ir_inline.canonical

    def test_compact_collapsed_from_block(self) -> None:
        ir = parse_guard("if not x:\n    continue")
        assert ir is not None
        assert "\n" not in ir.compact

    def test_compact_semantically_equivalent(self) -> None:
        ir = parse_guard("if not value: return None")
        assert ir is not None and ir.compact
        ir2 = parse_guard(ir.compact)
        assert ir2 is not None
        assert ir2.to_legacy_tuple() == ir.to_legacy_tuple()
