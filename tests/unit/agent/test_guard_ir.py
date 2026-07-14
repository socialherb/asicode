"""Tests for guard_ir — guard statement parsing and analysis.

Covers GuardIR data classes, parse_guard, _compute_op_class, _extract_control,
_extract_condition, _make_compact, _safe_unparse, and _parse_guard_ir_fast.
"""

import ast

from external_llm.agent.guard_ir import (
    GuardCondition,
    GuardIR,
    GuardPlacement,
    _compute_op_class,
    _extract_condition,
    _extract_control,
    _make_compact,
    _parse_guard_ir_fast,
    _safe_unparse,
    parse_guard,
)

# ══════════════════════════════════════════════════════════════════════════
# GuardCondition
# ══════════════════════════════════════════════════════════════════════════

class TestGuardCondition:
    def test_to_legacy_dict_basic(self):
        gc = GuardCondition(op_class="NotEq", operands=["x", "0"],
                            attribute_pairs=[])
        d = gc.to_legacy_dict()
        assert d["op"] == "NotEq"
        assert d["operands"] == ["x", "0"]

    def test_to_legacy_dict_with_attribute_pairs(self):
        gc = GuardCondition(op_class="Is", operands=["x"],
                            attribute_pairs=[("obj", "attr")])
        d = gc.to_legacy_dict()
        assert "attribute_pairs" in d
        assert d["attribute_pairs"] == [("obj", "attr")]


# ══════════════════════════════════════════════════════════════════════════
# GuardIR
# ══════════════════════════════════════════════════════════════════════════

class TestGuardIR:
    def test_to_legacy_tuple_with_condition(self):
        gc = GuardCondition(op_class="Gt", operands=["x", "0"],
                            attribute_pairs=[])
        ir = GuardIR(raw="if x > 0: return", canonical="if x > 0: return",
                     compact="if x > 0: return", condition=gc, control="return")
        cond, ctrl = ir.to_legacy_tuple()
        assert cond is not None
        assert cond["op"] == "Gt"
        assert ctrl == "return"

    def test_to_legacy_tuple_no_condition(self):
        ir = GuardIR(raw="invalid", canonical="", compact="",
                     condition=None, control="")
        cond, ctrl = ir.to_legacy_tuple()
        assert cond is None
        assert ctrl is None

    def test_is_parsed_true(self):
        ir = GuardIR(raw="if x: break", canonical="if x: break",
                     compact="if x: break", condition=None, control="break")
        assert ir.is_parsed is True

    def test_is_parsed_false(self):
        ir = GuardIR(raw="", canonical="", compact="",
                     condition=None, control="")
        assert ir.is_parsed is False

    def test_is_template_placeholder(self):
        gc = GuardCondition(op_class="Name", operands=["condition"], attribute_pairs=[])
        ir = GuardIR(raw="if condition: continue", canonical="if condition: continue",
                     compact="if condition: continue", condition=gc, control="continue")
        assert ir.is_template_placeholder is True

    def test_is_template_placeholder_with_placement(self):
        gc = GuardCondition(op_class="Name", operands=["condition"], attribute_pairs=[])
        pl = GuardPlacement(anchors=[], had_unresolved=False,
                            hallucinated_bases=frozenset(),
                            host_function_flavor="plain", loop_candidates=[])
        ir = GuardIR(raw="if condition: continue", canonical="if condition: continue",
                     compact="if condition: continue", condition=gc, control="continue",
                     placement=pl)
        assert ir.is_template_placeholder is False

    def test_is_template_placeholder_not_name(self):
        gc = GuardCondition(op_class="Gt", operands=["x", "0"], attribute_pairs=[])
        ir = GuardIR(raw="if x > 0: return", canonical="if x > 0: return",
                     compact="if x > 0: return", condition=gc, control="return")
        assert ir.is_template_placeholder is False

    def test_is_template_placeholder_no_condition(self):
        ir = GuardIR(raw="", canonical="", compact="",
                     condition=None, control="")
        assert ir.is_template_placeholder is False


# ══════════════════════════════════════════════════════════════════════════
# GuardIR.parse_guard
# ══════════════════════════════════════════════════════════════════════════

class TestParseGuard:
    def test_parse_guard_empty(self):
        assert parse_guard("") is None
        assert parse_guard("  ") is None

    def test_parse_guard_valid_return(self):
        ir = parse_guard("if x > 0: return")
        assert ir is not None
        assert ir.control == "return"
        assert ir.condition is not None
        assert ir.condition.op_class == "Gt"

    def test_parse_guard_valid_break(self):
        ir = parse_guard("if error: break")
        assert ir is not None
        assert ir.control == "break"
        assert ir.condition is not None

    def test_parse_guard_valid_continue(self):
        ir = parse_guard("if idx >= len(data): continue")
        assert ir is not None
        assert ir.control == "continue"

    def test_parse_guard_valid_raise(self):
        ir = parse_guard("if not ok: raise ValueError")
        assert ir is not None
        assert ir.control == "raise"

    def test_parse_guard_with_pass_fallback(self):
        """Invalid guard syntax with pass fallback."""
        src = """
if x > 0 return
"""
        ir = parse_guard(src)
        # With pass fallback, it should be parsed if the original fails
        assert ir is not None

    def test_parse_guard_non_guard(self):
        """Non-if code returns GuardIR with condition=None."""
        ir = parse_guard("x = 1")
        assert ir is not None
        assert ir.condition is None

    def test_parse_guard_no_control(self):
        """Guard without control keyword returns condition=None."""
        ir = parse_guard("if x > 0: pass")
        assert ir is not None
        assert ir.condition is None  # no return/break/continue/raise

    def test_parse_guard_attribute_condition(self):
        ir = parse_guard("if obj.is_valid: return")
        assert ir is not None
        assert ir.condition is not None
        assert ir.condition.attribute_pairs == [("obj", "is_valid")]


# ══════════════════════════════════════════════════════════════════════════
# _compute_op_class
# ══════════════════════════════════════════════════════════════════════════

class TestComputeOpClass:
    def test_unary_op(self):
        expr = ast.parse("not x", mode="eval").body
        assert _compute_op_class(expr) == "Not"

    def test_bool_op(self):
        expr = ast.parse("x and y", mode="eval").body
        assert _compute_op_class(expr) == "And"

    def test_compare(self):
        expr = ast.parse("x > 0", mode="eval").body
        assert _compute_op_class(expr) == "Gt"

    def test_name(self):
        expr = ast.parse("x", mode="eval").body
        assert _compute_op_class(expr) == "Name"

    def test_constant(self):
        expr = ast.parse("True", mode="eval").body
        assert _compute_op_class(expr) == "Constant"


# ══════════════════════════════════════════════════════════════════════════
# _extract_control
# ══════════════════════════════════════════════════════════════════════════

class TestExtractControl:
    def test_continue(self):
        tree = ast.parse("if x: continue")
        assert _extract_control(tree.body[0]) == "continue"

    def test_break(self):
        tree = ast.parse("if x: break")
        assert _extract_control(tree.body[0]) == "break"

    def test_return_value(self):
        tree = ast.parse("if x: return 42")
        assert _extract_control(tree.body[0]) == "return"

    def test_raise(self):
        tree = ast.parse("if x: raise ValueError('bad')")
        assert _extract_control(tree.body[0]) == "raise"

    def test_no_control(self):
        tree = ast.parse("if x: pass")
        assert _extract_control(tree.body[0]) == ""


# ══════════════════════════════════════════════════════════════════════════
# _extract_condition
# ══════════════════════════════════════════════════════════════════════════

class TestExtractCondition:
    def test_simple_compare(self):
        tree = ast.parse("if x > 0: break")
        cond = _extract_condition(tree.body[0])
        assert cond.op_class == "Gt"
        assert "x" in cond.operands

    def test_bool_op(self):
        tree = ast.parse("if x and y: break")
        cond = _extract_condition(tree.body[0])
        assert cond.op_class == "And"

    def test_attribute_access(self):
        tree = ast.parse("if obj.is_valid: break")
        cond = _extract_condition(tree.body[0])
        assert cond.attribute_pairs == [("obj", "is_valid")]


# ══════════════════════════════════════════════════════════════════════════
# _make_compact
# ══════════════════════════════════════════════════════════════════════════

class TestMakeCompact:
    def test_single_line(self):
        assert _make_compact("if x: return") == "if x: return"

    def test_two_lines_if_else(self):
        result = _make_compact("if x:\n    return")
        assert result == "if x: return"

    def test_empty(self):
        assert _make_compact("") == ""

    def test_multiline_no_colon(self):
        result = _make_compact("line1\nline2\nline3")
        assert result == "line1 line2 line3"


# ══════════════════════════════════════════════════════════════════════════
# _safe_unparse
# ══════════════════════════════════════════════════════════════════════════

class TestSafeUnparse:
    def test_unparse_expr(self):
        node = ast.parse("x + 1", mode="eval").body
        result = _safe_unparse(node)
        assert "x + 1" in result

    def test_unparse_failure(self):
        """Some ast nodes may not be unparsable."""
        node = ast.Module(body=[ast.Pass()], type_ignores=[])
        # This should not raise
        result = _safe_unparse(node)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════
# _parse_guard_ir_fast
# ══════════════════════════════════════════════════════════════════════════

class TestParseGuardIrFast:
    def test_valid_return(self):
        ir, ctrl = _parse_guard_ir_fast("if x > 0: return")
        assert ir is not None
        assert ir["op"] == "Gt"
        assert ctrl == "return"

    def test_valid_break(self):
        ir, ctrl = _parse_guard_ir_fast("if error: break")
        assert ir is not None
        assert ctrl == "break"

    def test_syntax_error(self):
        ir, ctrl = _parse_guard_ir_fast("if x")
        assert ir is None
        assert ctrl is None

    def test_no_control(self):
        ir, ctrl = _parse_guard_ir_fast("if x: pass")
        assert ir is None
        assert ctrl is None

    def test_not_if_stmt(self):
        ir, ctrl = _parse_guard_ir_fast("x = 1")
        assert ir is None
        assert ctrl is None

    def test_attribute_operand(self):
        ir, _ctrl = _parse_guard_ir_fast("if obj.is_active: return")
        assert ir is not None
        assert "attribute_pairs" in ir
        assert ("obj", "is_active") in ir["attribute_pairs"]
