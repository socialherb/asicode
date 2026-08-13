"""Tests for the shared missing-return repair (py/java family).

Covers both wrappers around ``_repair_missing_return`` so the language-specific
parameters (header predicate, comment skip, body-end marker, inserted stmt)
stay pinned to their original behavior.
"""
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.repair_strategies import (
    java_repair_missing_return,
    py_repair_missing_return,
)


def _raw_code(ops):
    """Extract the replaced source from a raw-replacement op list."""
    assert ops is not None and len(ops) == 1
    return ops[0].payload["__raw_code__"]


class TestPyRepairMissingReturn:
    def test_inserts_return_none_at_body_end(self):
        code = "def compute():\n    x = 1\n"
        error = VerifyError(message="missing return", line=2)
        out = _raw_code(py_repair_missing_return(code, error, None))
        assert out == "def compute():\n    x = 1\n    return None\n"

    def test_nested_method_finds_inner_def(self):
        code = (
            "class A:\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            pass\n"
        )
        error = VerifyError(message="missing return", line=4)
        out = _raw_code(py_repair_missing_return(code, error, None))
        assert out == (
            "class A:\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            pass\n"
            "            return None\n"
        )

    def test_comment_lines_do_not_end_body_scan(self):
        # Comment lines are transparent to the scan: they neither end the body
        # nor become the insertion anchor, so the return lands before them.
        code = "def f():\n    x = 1\n    # trailing comment\n"
        error = VerifyError(message="missing return", line=3)
        out = _raw_code(py_repair_missing_return(code, error, None))
        assert out == (
            "def f():\n    x = 1\n    return None\n    # trailing comment\n"
        )

    def test_dedent_ends_body_scan(self):
        # The scan must stop at the next top-level def, not insert into it.
        code = "def f():\n    x = 1\ndef g():\n    y = 2\n"
        error = VerifyError(message="missing return", line=2)
        out = _raw_code(py_repair_missing_return(code, error, None))
        assert out == "def f():\n    x = 1\n    return None\ndef g():\n    y = 2\n"

    def test_no_line_returns_none(self):
        error = VerifyError(message="missing return", line=None)
        assert py_repair_missing_return("def f(): pass\n", error, None) is None


class TestJavaRepairMissingReturn:
    def test_inserts_return_null_before_close_brace(self):
        code = (
            "class A {\n"
            "    public int compute() {\n"
            "        return 1;\n"
            "    }\n"
            "}\n"
        )
        error = VerifyError(message="missing return statement", line=3)
        out = _raw_code(java_repair_missing_return(code, error, None))
        assert out == (
            "class A {\n"
            "    public int compute() {\n"
            "        return 1;\n"
            "        return null;\n"
            "    }\n"
            "}\n"
        )

    def test_private_method(self):
        code = (
            "class A {\n"
            "    private int f() {\n"
            "        int x = 1;\n"
            "    }\n"
            "}\n"
        )
        error = VerifyError(message="missing return statement", line=3)
        out = _raw_code(java_repair_missing_return(code, error, None))
        assert out == (
            "class A {\n"
            "    private int f() {\n"
            "        int x = 1;\n"
            "        return null;\n"
            "    }\n"
            "}\n"
        )

    def test_no_line_returns_none(self):
        error = VerifyError(message="missing return statement", line=None)
        assert java_repair_missing_return("class A {}", error, None) is None
