"""Unit tests for repair_strategies — covers the language-specific repair
functions (Python/Java/Kotlin/Go) and the shared helpers.

These functions are pure transforms over (code, error, classification); they
return a list of PrimitiveOp (raw-replacement or insert-import) or None.  The
shared missing-return path is exercised in test_repair_missing_return.py; here
we cover every other strategy + the Go-specific helpers (_go_zero_value, type
mismatch, unused import) + the registry/dispatch surface.
"""

from __future__ import annotations

import pytest

from external_llm.editor._editor_core.vm.classification import (
    Classification,
    EvidenceSource,
    FailureType,
)
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.repair_strategies import (
    _get_indent,
    _go_zero_value,
    _make_raw_replacement,
    _trim_call_arguments,
    get_strategies,
    go_repair_argument_mismatch,
    go_repair_missing_return,
    go_repair_syntax_error,
    go_repair_type_mismatch,
    go_repair_unknown_symbol,
    go_repair_unused_import,
    java_repair_argument_mismatch,
    java_repair_duplicate_identifier,
    java_repair_syntax_error,
    java_repair_unknown_symbol,
    kotlin_repair_argument_mismatch,
    kotlin_repair_duplicate_identifier,
    kotlin_repair_missing_return,
    kotlin_repair_syntax_error,
    kotlin_repair_unknown_symbol,
    py_repair_argument_mismatch,
    py_repair_duplicate_identifier,
    py_repair_missing_variable,
    py_repair_syntax_error,
    repair_unknown_symbol,
)
from external_llm.editor.primitives.models import PrimitiveKind

# ── fixtures & helpers ──────────────────────────────────────────────────────


def _cls(symbol=None, ftype=FailureType.UNKNOWN_SYMBOL):
    return Classification(type=ftype, source=EvidenceSource.NONE, symbol=symbol)


def _err(message, line=None):
    return VerifyError(message=message, line=line)


def _raw(ops):
    """Extract the replaced source from a raw-replacement op list."""
    assert ops is not None and len(ops) == 1
    assert ops[0].kind == PrimitiveKind.INSERT_STATEMENT
    return ops[0].payload["__raw_code__"]


def _import(ops):
    """Extract the import statement from an insert-import op list."""
    assert ops is not None and len(ops) == 1
    assert ops[0].kind == PrimitiveKind.INSERT_IMPORT
    return ops[0].payload["statement"]


# ── low-level helpers ───────────────────────────────────────────────────────


class TestLowLevelHelpers:
    def test_make_raw_replacement(self):
        ops = _make_raw_replacement("x = 1")
        assert _raw(ops) == "x = 1"

    @pytest.mark.parametrize(
        "line, indent",
        [
            ("    x", "    "),
            ("\tx", "\t"),
            ("x", ""),
            ("  \t y", "  \t "),
        ],
    )
    def test_get_indent(self, line, indent):
        assert _get_indent(line) == indent

    @pytest.mark.parametrize(
        "t, zero",
        [
            ("int", "0"),
            ("int64", "0"),
            ("uint", "0"),
            ("uint32", "0"),
            ("float32", "0.0"),
            ("float64", "0.0"),
            ("bool", "false"),
            ("boolean", "false"),
            ("string", '""'),
            ("error", "nil"),
            ("[]int", "nil"),
            ("[]string", "nil"),
            ("map[string]int", "nil"),
            ("*int", "nil"),
            ("*Foo", "nil"),
            ("func()", "nil"),
            ("chan int", "nil"),
            ("interface{}", "nil"),
            ("Foo", "Foo{}"),
            ("time.Time", "time.Time{}"),
        ],
    )
    def test_go_zero_value(self, t, zero):
        assert _go_zero_value(t) == zero

    def test_trim_call_arguments_drops_last(self):
        lines = ["foo(a, b, c)\n"]
        assert _trim_call_arguments(lines, 0) == "foo(a, b)\n"

    def test_trim_call_arguments_keep_first(self):
        lines = ["foo(a, b, c)\n"]
        assert _trim_call_arguments(lines, 0, keep_first=True) == "foo(a)\n"

    def test_trim_call_arguments_single_arg_returns_none(self):
        lines = ["foo(a)\n"]
        assert _trim_call_arguments(lines, 0) is None
        assert _trim_call_arguments(lines, 0, keep_first=True) is None

    def test_trim_call_arguments_no_parens_returns_none(self):
        lines = ["foo\n"]
        assert _trim_call_arguments(lines, 0) is None


# ── Python ──────────────────────────────────────────────────────────────────


class TestPyRepairMissingVariable:
    def test_known_typing_name_from_import(self):
        ops = py_repair_missing_variable("x = List()\n", _err("x", 1), _cls("List"))
        assert _import(ops) == "from typing import List"

    def test_known_module_bare_import(self):
        ops = py_repair_missing_variable("os.getcwd()\n", _err("x", 1), _cls("os"))
        assert _import(ops) == "import os"

    def test_unknown_symbol_returns_none(self):
        assert py_repair_missing_variable("x\n", _err("x", 1), _cls("NotARealSym")) is None

    def test_no_symbol_returns_none(self):
        assert py_repair_missing_variable("x\n", _err("x", 1), _cls(None)) is None


class TestPyRepairSyntaxError:
    def test_missing_colon_after_def(self):
        code = "def f()\n    pass\n"
        assert _raw(py_repair_syntax_error(code, _err("expected ':'", 1), None)) == "def f():\n    pass\n"

    def test_missing_colon_after_if(self):
        code = "if x\n    pass\n"
        assert _raw(py_repair_syntax_error(code, _err("expected ':'", 1), None)) == "if x:\n    pass\n"

    def test_non_keyword_line_returns_none(self):
        # Line lacks a compound-statement keyword; colon insertion would be wrong.
        code = "x = 1\n    pass\n"
        assert py_repair_syntax_error(code, _err("expected ':'", 1), None) is None

    def test_already_has_colon_returns_none(self):
        code = "def f():\n    pass\n"
        assert py_repair_syntax_error(code, _err("expected ':'", 1), None) is None

    def test_other_message_returns_none(self):
        assert py_repair_syntax_error("def f():\n    pass\n", _err("indent", 1), None) is None

    def test_no_line_returns_none(self):
        assert py_repair_syntax_error("x\n", _err("indent"), None) is None


class TestPyRepairArgumentMismatch:
    def test_too_many_args_keeps_first_only(self):
        code = "foo(a, b, c)\n"
        assert (
            _raw(py_repair_argument_mismatch(code, _err("takes 1 positional argument but 3 were given", 1), None))
            == "foo(a)\n"
        )

    def test_missing_required_returns_none(self):
        assert py_repair_argument_mismatch("foo()\n", _err("missing 1 required positional argument", 1), None) is None

    def test_no_paren_returns_none(self):
        assert (
            py_repair_argument_mismatch("foo\n", _err("takes 1 positional argument but 3 were given", 1), None) is None
        )

    def test_empty_args_returns_none(self):
        assert (
            py_repair_argument_mismatch("foo()\n", _err("takes 1 positional argument but 3 were given", 1), None)
            is None
        )

    def test_no_line_returns_none(self):
        assert py_repair_argument_mismatch("foo(a)\n", _err("x"), None) is None


class TestPyRepairDuplicateIdentifier:
    def test_def_gets_dup_suffix(self):
        code = "def foo():\n    pass\n"
        assert (
            _raw(py_repair_duplicate_identifier(code, _err("name 'foo' redefined", 1), None))
            == "def foo_dup():\n    pass\n"
        )

    def test_class_gets_Dup_suffix(self):
        code = "class Foo:\n    pass\n"
        assert (
            _raw(py_repair_duplicate_identifier(code, _err("duplicate class Foo", 1), None))
            == "class FooDup:\n    pass\n"
        )

    def test_no_marker_returns_none(self):
        assert py_repair_duplicate_identifier("def foo():\n", _err("syntax error", 1), None) is None

    def test_no_line_returns_none(self):
        assert py_repair_duplicate_identifier("def foo():\n", _err("redefined"), None) is None


# ── Java ────────────────────────────────────────────────────────────────────


class TestJavaRepairUnknownSymbol:
    def test_known_type_adds_import_with_semicolon(self):
        ops = java_repair_unknown_symbol("List x;\n", _err("x", 1), _cls("List"))
        assert _import(ops) == "import java.util.List;"

    def test_already_imported_returns_none(self):
        code = "import java.util.List;\nList x;\n"
        assert java_repair_unknown_symbol(code, _err("x", 2), _cls("List")) is None

    def test_unknown_symbol_returns_none(self):
        assert java_repair_unknown_symbol("X x;\n", _err("x", 1), _cls("NotAType")) is None

    def test_no_symbol_returns_none(self):
        assert java_repair_unknown_symbol("x;\n", _err("x", 1), _cls(None)) is None


class TestJavaRepairSyntaxError:
    def test_missing_semicolon_appended(self):
        code = "x = 1\n"
        assert _raw(java_repair_syntax_error(code, _err("';' expected", 1), None)) == "x = 1;\n"

    def test_alternate_marker(self):
        code = "x = 1\n"
        assert java_repair_syntax_error(code, _err("expected ';'", 1), None) is not None

    def test_already_semicolon_returns_none(self):
        code = "x = 1;\n"
        assert java_repair_syntax_error(code, _err("';' expected", 1), None) is None

    def test_no_marker_returns_none(self):
        assert java_repair_syntax_error("x = 1;\n", _err("brace", 1), None) is None


class TestJavaRepairArgumentMismatch:
    def test_too_many_drops_last(self):
        code = "foo(a, b, c);\n"
        assert (
            _raw(
                java_repair_argument_mismatch(code, _err("actual and formal argument lists differ in length", 1), None)
            )
            == "foo(a, b);\n"
        )

    def test_single_arg_returns_none(self):
        code = "foo(a);\n"
        assert (
            java_repair_argument_mismatch(code, _err("actual and formal argument lists differ in length", 1), None)
            is None
        )

    def test_empty_args_returns_none(self):
        code = "foo();\n"
        assert (
            java_repair_argument_mismatch(code, _err("actual and formal argument lists differ in length", 1), None)
            is None
        )

    def test_no_marker_returns_none(self):
        assert java_repair_argument_mismatch("foo(a, b);\n", _err("type", 1), None) is None

    def test_no_line_returns_none(self):
        assert java_repair_argument_mismatch("foo(a, b);\n", _err("differ"), None) is None


class TestJavaRepairDuplicateIdentifier:
    def test_class_dup_suffix(self):
        code = "class Foo {\n}\n"
        assert (
            _raw(java_repair_duplicate_identifier(code, _err("duplicate class Foo", 1), None)) == "class FooDup {\n}\n"
        )

    def test_no_marker_returns_none(self):
        assert java_repair_duplicate_identifier("class Foo {}\n", _err("syntax", 1), None) is None


# ── Kotlin ──────────────────────────────────────────────────────────────────


class TestKotlinRepairUnknownSymbol:
    def test_known_type_bare_import(self):
        ops = kotlin_repair_unknown_symbol("val x: List<Int>\n", _err("x", 1), _cls("List"))
        assert _import(ops) == "import kotlin.collections.List"

    def test_no_semicolon(self):
        # Kotlin imports must NOT carry a trailing semicolon.
        assert not _import(kotlin_repair_unknown_symbol("val x: List<Int>\n", _err("x", 1), _cls("List"))).endswith(";")

    def test_already_imported_returns_none(self):
        code = "import kotlin.collections.List\nval x: List<Int>\n"
        assert kotlin_repair_unknown_symbol(code, _err("x", 2), _cls("List")) is None


class TestKotlinRepairSyntaxError:
    def test_expecting_marker(self):
        code = "val x = 1\n"
        assert _raw(kotlin_repair_syntax_error(code, _err("expecting ';'", 1), None)) == "val x = 1;\n"

    def test_no_marker_returns_none(self):
        assert kotlin_repair_syntax_error("val x = 1\n", _err("brace", 1), None) is None


class TestKotlinRepairMissingReturn:
    def test_typed_function_inserts_return_null(self):
        code = "fun compute(): Int {\n    val x = 1\n}\n"
        assert _raw(kotlin_repair_missing_return(code, _err("missing return", 3), None)) == (
            "fun compute(): Int {\n    val x = 1\n        return null\n}\n"
        )

    def test_unit_body_function_inserts_bare_return(self):
        code = "fun compute() {\n    println()\n}\n"
        assert _raw(kotlin_repair_missing_return(code, _err("missing return", 3), None)) == (
            "fun compute() {\n    println()\n        return\n}\n"
        )

    def test_no_fun_header_returns_none(self):
        assert kotlin_repair_missing_return("val x = 1\n", _err("missing return", 1), None) is None

    def test_no_line_returns_none(self):
        assert kotlin_repair_missing_return("fun f() {\n}\n", _err("missing return"), None) is None


class TestKotlinRepairArgumentMismatch:
    def test_too_many_drops_last(self):
        code = "foo(a, b, c)\n"
        assert _raw(kotlin_repair_argument_mismatch(code, _err("too many arguments", 1), None)) == "foo(a, b)\n"

    def test_required_drops_last(self):
        code = "foo(a, b, c)\n"
        assert _raw(kotlin_repair_argument_mismatch(code, _err("required: 2, found: 3", 1), None)) == "foo(a, b)\n"

    def test_single_arg_returns_none(self):
        assert kotlin_repair_argument_mismatch("foo(a)\n", _err("too many", 1), None) is None

    def test_no_marker_returns_none(self):
        assert kotlin_repair_argument_mismatch("foo(a, b)\n", _err("type", 1), None) is None

    def test_no_line_returns_none(self):
        assert kotlin_repair_argument_mismatch("foo(a, b)\n", _err("too many"), None) is None


class TestKotlinRepairDuplicateIdentifier:
    def test_fun_dup_suffix(self):
        code = "fun foo() {\n}\n"
        assert (
            _raw(kotlin_repair_duplicate_identifier(code, _err("duplicate declaration", 1), None))
            == "fun fooDup() {\n}\n"
        )

    def test_class_dup_suffix(self):
        code = "class Foo {\n}\n"
        assert (
            _raw(kotlin_repair_duplicate_identifier(code, _err("duplicate declaration", 1), None))
            == "class FooDup {\n}\n"
        )

    def test_no_marker_returns_none(self):
        assert kotlin_repair_duplicate_identifier("fun foo() {}\n", _err("syntax", 1), None) is None


# ── Go ──────────────────────────────────────────────────────────────────────


class TestGoRepairUnknownSymbol:
    def test_known_pkg_adds_import(self):
        ops = go_repair_unknown_symbol("fmt.Println()\n", _err("undefined: fmt", 1), _cls("fmt"))
        assert _import(ops) == 'import "fmt"'

    def test_already_imported_returns_none(self):
        code = 'import "fmt"\nfmt.Println()\n'
        assert go_repair_unknown_symbol(code, _err("undefined: fmt", 2), _cls("fmt")) is None

    def test_case_correction_lowercase_symbol(self):
        # Symbol is lowercase but the codebase has the Capitalized variant.
        code = "type Myvar struct{}\nfunc main() {\n    myvar()\n}\n"
        ops = go_repair_unknown_symbol(code, _err("undefined: myvar", 3), _cls("myvar"))
        assert _raw(ops) == "type Myvar struct{}\nfunc main() {\n    Myvar()\n}\n"

    def test_case_correction_uppercase_symbol(self):
        # Capitalized symbol but lowercase variant exists in code.
        code = "var myvar = 1\nfunc main() {\n    Myvar()\n}\n"
        ops = go_repair_unknown_symbol(code, _err("undefined: Myvar", 3), _cls("Myvar"))
        assert _raw(ops) == "var myvar = 1\nfunc main() {\n    myvar()\n}\n"

    def test_no_candidate_present_returns_none(self):
        code = "func main() {\n    myvar()\n}\n"
        assert go_repair_unknown_symbol(code, _err("undefined: myvar", 2), _cls("myvar")) is None

    def test_no_symbol_returns_none(self):
        assert go_repair_unknown_symbol("x\n", _err("x", 1), _cls(None)) is None


class TestGoRepairUnusedImport:
    def test_remove_from_import_block(self):
        code = 'import (\n    "fmt"\n    "os"\n)\n'
        assert _raw(go_repair_unused_import(code, _err("imported and not used: fmt", 1), _cls("fmt"))) == (
            'import (\n    "os"\n)\n'
        )

    def test_remove_single_import(self):
        code = 'import "fmt"\n'
        # The lone import line is dropped; the remaining empty line list joins to "".
        assert _raw(go_repair_unused_import(code, _err("imported and not used: fmt", 1), _cls("fmt"))) == ""

    def test_not_found_returns_none(self):
        code = 'import "os"\n'
        assert go_repair_unused_import(code, _err("imported and not used: fmt", 1), _cls("fmt")) is None

    def test_no_symbol_returns_none(self):
        assert go_repair_unused_import('import "fmt"\n', _err("unused", 1), _cls(None)) is None


class TestGoRepairSyntaxError:
    def test_missing_semicolon(self):
        code = "x := 1\n"
        assert _raw(go_repair_syntax_error(code, _err("expected ';'", 1), None)) == "x := 1;\n"

    def test_expected_newline(self):
        code = "x := 1\n"
        assert go_repair_syntax_error(code, _err("expected newline", 1), None) is not None

    def test_line_ends_brace_no_semicolon(self):
        # A line ending in "{" must not get a semicolon.
        code = "if x {\n"
        assert go_repair_syntax_error(code, _err("expected ';'", 1), None) is None

    def test_missing_open_brace(self):
        code = "func foo()\n"
        assert _raw(go_repair_syntax_error(code, _err("expected '{'", 1), None)) == "func foo() {\n"

    def test_no_marker_returns_none(self):
        assert go_repair_syntax_error("x := 1\n", _err("type", 1), None) is None

    def test_no_line_returns_none(self):
        assert go_repair_syntax_error("x := 1\n", _err("expected ';'"), None) is None


class TestGoRepairArgumentMismatch:
    def test_too_many_drops_last(self):
        code = "foo(a, b, c)\n"
        assert _raw(go_repair_argument_mismatch(code, _err("too many arguments", 1), None)) == "foo(a, b)\n"

    def test_too_many_single_arg_returns_none(self):
        assert go_repair_argument_mismatch("foo(a)\n", _err("too many arguments", 1), None) is None

    def test_not_enough_fills_zero_values_from_want(self):
        code = "f()\n"
        ops = go_repair_argument_mismatch(
            code,
            _err("not enough arguments in call to f\n    have ()\n    want (int, string)", 1),
            None,
        )
        assert _raw(ops) == 'f(0, "")\n'

    def test_not_enough_partial_existing(self):
        code = "f(a)\n"
        ops = go_repair_argument_mismatch(
            code,
            _err("not enough arguments\n    have (string)\n    want (string, int)", 1),
            None,
        )
        assert _raw(ops) == "f(a, 0)\n"

    def test_not_enough_no_want_fallback_nil(self):
        code = "f(a)\n"
        ops = go_repair_argument_mismatch(code, _err("not enough arguments", 1), None)
        assert _raw(ops) == "f(a, nil)\n"

    def test_not_enough_empty_args_nil(self):
        code = "f()\n"
        ops = go_repair_argument_mismatch(code, _err("not enough arguments", 1), None)
        assert _raw(ops) == "f(nil)\n"

    def test_no_line_returns_none(self):
        assert go_repair_argument_mismatch("foo(a, b)\n", _err("too many"), None) is None


class TestGoRepairMissingReturn:
    def test_brace_same_line_return_nil(self):
        # Idiomatic Go: "{" on the header line → ret_type detection disabled → nil.
        code = "func compute() int {\n    x := 1\n}\n"
        assert _raw(go_repair_missing_return(code, _err("missing return at end of function", 3), None)) == (
            "func compute() int {\n    x := 1\n        return nil\n}\n"
        )

    def test_typed_function_zero_value(self):
        # Header without inline "{" lets the return type be parsed; body indented.
        code = "func compute() int\n    x := 1\n"
        assert _raw(go_repair_missing_return(code, _err("missing return at end of function", 2), None)) == (
            "func compute() int\n    x := 1\n        return 0\n"
        )

    def test_no_missing_return_marker(self):
        assert go_repair_missing_return("func f() {\n}\n", _err("syntax", 2), None) is None

    def test_no_line_returns_none(self):
        assert go_repair_missing_return("func f() {\n}\n", _err("missing return"), None) is None


class TestGoRepairTypeMismatch:
    # Real compiler output (Go 1.26):
    #   current (>=1.21): "cannot use x (variable of type int) as float64 value in ..."
    #                     "cannot use t (variable of struct type time.Time) as string value in ..."
    #                     "cannot use nil as time.Time value in ..."
    #   legacy  (<=1.20): "cannot use x (type int) as type float64 in ..."
    #                     "cannot use nil as type time.Time in ..."

    def test_nil_to_value_type_replaced_with_zero_value(self):
        # Modern format: the nil message has NO "(type nil)" fragment — the
        # expression itself is "nil". Casing preserved (original-message search),
        # so the qualified type survives verbatim into the zero-value literal.
        code = "x := nil\n"
        ops = go_repair_type_mismatch(code, _err("cannot use nil as time.Time value in assignment", 1), None)
        assert _raw(ops) == "x := time.Time{}\n"

    def test_nil_to_value_type_legacy_format(self):
        ops = go_repair_type_mismatch("x := nil\n", _err("cannot use nil as type time.Time in assignment", 1), None)
        assert _raw(ops) == "x := time.Time{}\n"

    def test_nil_replaces_only_first_nil(self):
        code = "var p *Foo = nil\n"
        ops = go_repair_type_mismatch(code, _err("cannot use nil as *Foo value in assignment", 1), None)
        # *Foo is a pointer type → zero value is nil, so the line is unchanged.
        assert _raw(ops) == "var p *Foo = nil\n"

    def test_numeric_cast_wraps_rhs(self):
        code = "y := x\n"
        ops = go_repair_type_mismatch(
            code, _err("cannot use x (variable of type int) as float64 value in assignment", 1), None
        )
        assert _raw(ops) == "y := float64(x)\n"

    def test_numeric_cast_legacy_format(self):
        ops = go_repair_type_mismatch(
            "y := x\n", _err("cannot use x (type int) as type float64 in assignment", 1), None
        )
        assert _raw(ops) == "y := float64(x)\n"

    def test_struct_type_message_matches_but_not_repairable(self):
        # "variable of struct type time.Time" — parsed fine (casing preserved),
        # but time.Time is not a numeric type → no repair.
        assert (
            go_repair_type_mismatch(
                "var s string = t\n",
                _err("cannot use t (variable of struct type time.Time) as string value in assignment", 1),
                None,
            )
            is None
        )

    def test_non_numeric_types_returns_none(self):
        assert (
            go_repair_type_mismatch(
                "x := a\n", _err("cannot use a (variable of type Foo) as Bar value in assignment", 1), None
            )
            is None
        )

    def test_no_match_returns_none(self):
        assert go_repair_type_mismatch("x := 1\n", _err("type error", 1), None) is None

    def test_no_line_returns_none(self):
        assert go_repair_type_mismatch("x := nil\n", _err("cannot use nil"), None) is None


# ── registry / dispatch ─────────────────────────────────────────────────────


class TestRegistry:
    def test_get_strategies_python(self):
        s = get_strategies("python")
        assert FailureType.MISSING_VARIABLE in s
        assert FailureType.DUPLICATE_IDENTIFIER in s

    def test_get_strategies_go_has_type_mismatch(self):
        assert FailureType.TYPE_MISMATCH in get_strategies("go")
        assert FailureType.UNUSED_IMPORT in get_strategies("go")

    @pytest.mark.parametrize("lang", ["python", "java", "kotlin", "go"])
    def test_each_language_has_strategies(self, lang):
        assert get_strategies(lang)  # non-empty

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="No repair strategies"):
            get_strategies("rust")

    def test_repair_unknown_symbol_fallback_returns_none(self):
        # The fallback shim returns None (registry overrides it).
        assert repair_unknown_symbol("x\n", _err("x", 1), _cls("x")) is None
