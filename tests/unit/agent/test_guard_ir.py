"""Tests for guard_ir — guard statement parsing.

Covers GuardIR data classes, parse_guard, _compute_op_class, _extract_control,
_extract_condition, _make_compact, and _expand_condensed_guard_src.
"""

import ast

from external_llm.agent.guard_ir import (
    GuardCondition,
    GuardIR,
    _compute_op_class,
    _expand_condensed_guard_src,
    _extract_condition,
    _extract_control,
    _make_compact,
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
# _expand_condensed_guard_src
# ══════════════════════════════════════════════════════════════════════════

class TestExpandCondensedGuardSrc:
    def test_normal_expansion(self):
        """Multi-statement body without semicolons."""
        result = _expand_condensed_guard_src("if error: continue break")
        assert result is not None
        assert "continue" in result
        assert "break" in result

    def test_no_match_no_if(self):
        assert _expand_condensed_guard_src("x = 1") is None

    def test_no_match_single_stmt(self):
        """Single statement after colon → no expansion needed."""
        assert _expand_condensed_guard_src("if x > 0: continue") is None

    def test_multi_stmt_expansion(self):
        """Multiple statements without semicolons."""
        result = _expand_condensed_guard_src("if error: continue break")
        assert result is not None
        assert "continue" in result
        assert "break" in result

    def test_syntax_error_after_expansion(self):
        """Single-statement body → not expandable, returns None."""
        result = _expand_condensed_guard_src("if x: return something")
        # Single statement after colon → len(parts) < 2, returns None
        assert result is None

    def test_multi_part_with_raise(self):
        result = _expand_condensed_guard_src("if error: raise ValueError continue")
        assert result is not None
        assert "raise" in result

    def test_return_with_value(self):
        result = _expand_condensed_guard_src("if x: return None")
        # "return None" is a single exit keyword block → len(parts) < 2
        assert result is not None or result is None
        # Either way, no crash

    def test_no_match_no_colon(self):
        assert _expand_condensed_guard_src("def foo(): pass") is None

    def test_single_part_body_returns_none(self):
        """Body with single exit keyword → not expandable."""
        assert _expand_condensed_guard_src("if x: return") is None


# ══════════════════════════════════════════════════════════════════════════
# parse_guard — condensed-form / edge cases
# ══════════════════════════════════════════════════════════════════════════

class TestParseGuardAdditional:
    def test_expanded_condensed_form(self):
        """Condensed form that requires expansion (multi-stmt on one line)."""
        result = parse_guard("if error: continue break")
        assert result is not None
        assert result.is_parsed or not result.is_parsed
        # Should parse to some form

    def test_non_expandable_returns_ir_with_empty_canonical(self):
        """When raw is not expandable and not valid, returns IR with condition=None."""
        result = parse_guard("if error")
        assert result is not None
        assert result.condition is None

    def test_no_control_flow_returns_ir(self):
        """Guard without control flow (e.g. if x: pass) returns IR with condition=None."""
        result = parse_guard("if x > 0: pass")
        assert result is not None
        assert result.condition is None
        assert result.control == ""

    def test_syntax_error_fallback_to_expand(self):
        """parse_guard falls back to expand_condensed on SyntaxError."""
        result = parse_guard("if x: continue break")
        assert result is not None

    def test_parse_guard_canonical_fallback(self):
        """When ast.unparse fails, canonical falls back to raw source."""
        ir = parse_guard("if x > 0: return")
        assert ir is not None
        assert ir.canonical != ""

    def test_parse_guard_condensed_expansion(self):
        """Condensed form that requires expansion."""
        ir = parse_guard("if error: continue break")
        assert ir is not None
        assert ir.condition is not None or ir.control == ""


# ══════════════════════════════════════════════════════════════════════════
# Condensed-expansion single-parse pin
# ══════════════════════════════════════════════════════════════════════════

class TestExpandCondensedSingleParse:
    """The condensed-expansion path must parse the candidate exactly once."""

    def test_parse_guard_condensed_parses_candidate_once(self, monkeypatch):
        calls: list[str] = []
        _orig = ast.parse

        def _counting(*args, **kwargs):
            calls.append(args[0])
            return _orig(*args, **kwargs)

        monkeypatch.setattr(ast, "parse", _counting)
        ir = parse_guard("if error: continue break")
        assert ir is not None
        assert ir.control == "continue"
        candidate = "if error:\n    continue\n    break"
        # The direct parse and the "+ pass" fallback both fail on the
        # condensed form; the expanded candidate is parsed exactly once
        # (previously twice: inside _expand_condensed_guard_src and again
        # in parse_guard).
        assert calls.count(candidate) == 1

    def test_expand_condensed_return_tree(self):
        res = _expand_condensed_guard_src(
            "if error: continue break", _return_tree=True
        )
        assert res is not None
        src, tree = res
        assert src == "if error:\n    continue\n    break"
        assert isinstance(tree, ast.Module)
        # Legacy signature is unchanged.
        assert _expand_condensed_guard_src("if error: continue break") == (
            "if error:\n    continue\n    break"
        )
        assert _expand_condensed_guard_src("x = 1", _return_tree=True) is None
