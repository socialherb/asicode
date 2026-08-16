"""Regression tests for CodeAnalyzer (external_llm/code_analyzer.py).

Primary target: CA-B2 — ``_extract_function_info`` iterated only
``node.args.args`` (positional-or-keyword), silently dropping positional-only
(before ``/``), keyword-only (after ``*``), ``*vararg`` and ``**kwarg`` from
the rendered signature. The lossy signature was fed to the LLM via
``super_context_builder``'s "Key Functions" block, misinforming the model
about call arity / keyword-only constraints. Coverage was 0/62 branches
before this file.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest import mock

from external_llm.code_analyzer import CodeAnalyzer


def _analyze(src: str, tmp_path: Path):
    p = tmp_path / "sample.py"
    p.write_text(textwrap.dedent(src))
    return CodeAnalyzer().analyze_file(p)


def _sig(src: str, tmp_path: Path) -> str:
    analysis = _analyze(src, tmp_path)
    return CodeAnalyzer().format_function_signature(analysis.functions[0])


# --- CA-B2: signature completeness across all parameter kinds ---

def test_posonly_kwonly_kwarg_all_preserved(tmp_path):
    """`def f(a, /, b, *, c=2, **kw)` — was rendered as `def f(b):`."""
    sig = _sig(
        """
        def f(a, /, b, *, c=2, **kw):
            return a + b + c
        """,
        tmp_path,
    )
    assert "a" in sig          # positional-only  (was DROPPED)
    assert "/" in sig          # positional-only separator
    assert "b" in sig
    assert "c" in sig          # keyword-only     (was DROPPED)
    assert "**kw" in sig       # kwargs           (was DROPPED)


def test_posonly_not_dropped(tmp_path):
    analysis = _analyze("def f(a, /, b):\n    return b\n", tmp_path)
    func = analysis.functions[0]
    assert "a" in func.args
    assert "b" in func.args
    assert CodeAnalyzer().format_function_signature(func).startswith("def f(a, /, b")


def test_vararg_kwonly_chain(tmp_path):
    sig = _sig("def f(a, *args, b, **kw):\n    pass\n", tmp_path)
    assert "*args" in sig
    assert "b" in sig
    assert "**kw" in sig


def test_kwonly_without_vararg_has_bare_star(tmp_path):
    sig = _sig("def f(a, *, b):\n    pass\n", tmp_path)
    # a bare '*' separator must precede `b` so it stays keyword-only
    assert "*, b" in sig.replace(" ", "") or "* b" in sig or "*,b" in sig.replace(" ", "")
    assert "b" in sig


def test_type_hints_kept_across_kinds(tmp_path):
    analysis = _analyze(
        "def f(a: int, /, b: str, *, c: float, **kw: bool):\n    pass\n", tmp_path
    )
    func = analysis.functions[0]
    assert func.type_hints.get("a") == "int"
    assert func.type_hints.get("b") == "str"
    assert func.type_hints.get("c") == "float"
    sig = CodeAnalyzer().format_function_signature(func)
    assert "a: int" in sig
    assert "c: float" in sig
    assert "**kw: bool" in sig


def test_vararg_annotation(tmp_path):
    sig = _sig("def f(*args: int):\n    pass\n", tmp_path)
    assert "*args: int" in sig


# --- backward-compat / guardrails ---

def test_plain_args_backward_compat(tmp_path):
    assert _sig("def f(x, y):\n    return x + y\n", tmp_path).startswith("def f(x, y):")


def test_no_args(tmp_path):
    assert _sig("def f():\n    pass\n", tmp_path).startswith("def f():")


def test_async_function(tmp_path):
    analysis = _analyze("async def f(a, *, b):\n    return b\n", tmp_path)
    func = analysis.functions[0]
    assert func.is_async is True
    sig = CodeAnalyzer().format_function_signature(func)
    assert sig.startswith("async def f(")
    assert "b" in sig


# --- CA-B3: AnnAssign (annotated assignments) captured into global_vars ---
#
# Before this fix only ``ast.Assign`` (``X = 5``) populated ``global_vars``;
# ``ast.AnnAssign`` (``X: int = 5``) was silently dropped, so typed constants
# and type aliases never reached the LLM-facing "Type aliases/Constants" block
# in ``super_context_builder._extract_type_info``.

def test_annassign_typed_constant_captured(tmp_path):
    """``MAX_RETRIES: int = 5`` was dropped; only plain ``X = 5`` was handled."""
    a = _analyze("MAX_RETRIES: int = 5\n", tmp_path)
    assert a.global_vars["MAX_RETRIES"] == "5"


def test_annassign_complex_value(tmp_path):
    a = _analyze("DATA: list[int] = [1, 2, 3]\n", tmp_path)
    assert a.global_vars["DATA"] == "[1, 2, 3]"


def test_annassign_annotation_only_skipped(tmp_path):
    """Annotation-only ``x: int`` (no value) has nothing to record — not stored."""
    a = _analyze("DECLARED: int\nlogger: Logger\n", tmp_path)
    assert "DECLARED" not in a.global_vars
    assert "logger" not in a.global_vars


def test_annassign_coexists_with_plain_assign(tmp_path):
    a = _analyze("A = 1\nB: int = 2\n", tmp_path)
    assert a.global_vars == {"A": "1", "B": "2"}


def test_annassign_nested_not_global(tmp_path):
    """AnnAssign inside a function body must NOT leak to module-level globals."""
    a = _analyze(
        """
        def f():
            LOCAL: int = 9
            return LOCAL
        """,
        tmp_path,
    )
    assert "LOCAL" not in a.global_vars
    assert len(a.functions) == 1


def test_annassign_in_class_body_not_global(tmp_path):
    """AnnAssign inside a class body is an attribute, not a module global."""
    a = _analyze(
        """
        class C:
            ATTR: int = 1
        """,
        tmp_path,
    )
    assert "ATTR" not in a.global_vars
    assert len(a.classes) == 1


# --- CA-P1: top-level detection via precomputed id-set (was O(n*m)) ---
#
# ``_is_top_level`` re-scanned ``tree.body`` (m items) on every qualifying node
# during ``ast.walk`` (n nodes). It was replaced by a single ``{id(item) ...}``
# set with O(1) lookup. Correctness is identical (AST nodes compare by
# identity), so the regression guards below assert nesting exclusion holds.

def test_toplevel_filter_excludes_nested_defs(tmp_path):
    a = _analyze(
        """
        def top_func():
            def nested_func():
                pass

        class TopClass:
            def method(self):
                pass

            class NestedClass:
                pass
        """,
        tmp_path,
    )
    assert [f.name for f in a.functions] == ["top_func"]
    assert [c.name for c in a.classes] == ["TopClass"]


def test_toplevel_assign_excludes_nested(tmp_path):
    """Module-level Assign/AnnAssign captured; nested ones excluded."""
    a = _analyze(
        """
        GLOBAL = 1
        TYPED: int = 2
        def f():
            INNER = 3
            INNER_TYPED: str = "x"
        """,
        tmp_path,
    )
    assert a.global_vars == {"GLOBAL": "1", "TYPED": "2"}


def test_toplevel_detection_many_nodes_stable(tmp_path):
    """Regression guard for the O(n*m)->O(n) refactor: many nodes must still be
    classified correctly (no mis-attribution to top level)."""
    body = "\n".join(f"def f{i}():\n    x = {i}\n" for i in range(50))
    a = _analyze(body + "\n", tmp_path)
    assert len(a.functions) == 50
    assert {f.name for f in a.functions} == {f"f{i}" for i in range(50)}


# --- analyze_file error handling (L91-93): parse failure / missing file ---

def test_analyze_file_syntax_error_returns_none(tmp_path):
    """Unparseable source → analyze_file swallows and returns None."""
    p = tmp_path / "bad.py"
    p.write_text("def (\n")
    assert CodeAnalyzer().analyze_file(p) is None


def test_analyze_file_missing_file_returns_none(tmp_path):
    """Non-existent file → read_text raises → caught → None."""
    assert CodeAnalyzer().analyze_file(tmp_path / "ghost.py") is None


# --- ast.Import (L126-127) ---

def test_plain_import_captured(tmp_path):
    """``import os`` / ``import sys as system`` → ImportInfo with module+alias."""
    a = _analyze("import os\nimport sys as system\n", tmp_path)
    pairs = {(i.module, i.alias) for i in a.imports}
    assert ("os", None) in pairs
    assert ("sys", "system") in pairs


# --- ast.ImportFrom (L133-149): absolute + relative level encoding ---

def test_from_import_absolute(tmp_path):
    """``from a.b import c, d`` → module='a.b', names=['c','d']."""
    a = _analyze("from a.b import c, d\n", tmp_path)
    imp = a.imports[0]
    assert imp.module == "a.b"
    assert imp.names == ["c", "d"]


def test_from_import_relative_level_encoded(tmp_path):
    """``from ..pkg import y`` (level 2) → module='..pkg' (dots from node.level)."""
    a = _analyze("from ..pkg import y\n", tmp_path)
    assert a.imports[0].module == "..pkg"
    assert a.imports[0].names == ["y"]


def test_from_import_no_module_relative(tmp_path):
    """``from . import x`` (level 1, module None) → module='.'."""
    a = _analyze("from . import x\n", tmp_path)
    assert a.imports[0].module == "."
    assert a.imports[0].names == ["x"]


# --- module-level Call nodes (L153-155) ---

def test_module_level_calls_collected(tmp_path):
    """Top-level Call nodes populate ``analysis.calls`` (Name + Attribute)."""
    a = _analyze("print(value)\nobj.method()\n", tmp_path)
    assert "print" in a.calls
    assert "method" in a.calls


def test_call_with_non_name_func_skipped(tmp_path):
    """``(lambda: 1)()`` — func is Lambda → _get_call_name returns None, no crash."""
    a = _analyze("(lambda: 1)()\n", tmp_path)
    assert a.calls == set()


# --- return types (L238 extraction, L342 signature) ---

def test_return_type_captured(tmp_path):
    a = _analyze("def f() -> int:\n    return 1\n", tmp_path)
    assert a.functions[0].return_type == "int"
    sig = CodeAnalyzer().format_function_signature(a.functions[0])
    assert "-> int" in sig


def test_return_type_absent(tmp_path):
    a = _analyze("def f():\n    pass\n", tmp_path)
    assert a.functions[0].return_type is None


# --- _collect_calls: per-function calls + nested-scope exclusion (L268-271, DG-B1) ---

def test_function_calls_collected_per_scope(tmp_path):
    a = _analyze(
        """
        def outer():
            helper()
            obj.do_thing()
        """,
        tmp_path,
    )
    func = a.functions[0]
    assert "helper" in func.calls
    assert "do_thing" in func.calls


def test_function_calls_exclude_nested_scope(tmp_path):
    """Calls inside a nested def belong to THAT scope, not the enclosing fn."""
    a = _analyze(
        """
        def outer():
            outer_call()
            def inner():
                inner_call()
        """,
        tmp_path,
    )
    func = a.functions[0]  # outer is the only top-level function
    assert "outer_call" in func.calls
    assert "inner_call" not in func.calls


# --- docstring in function signature (L349-350) ---

def test_docstring_in_function_signature(tmp_path):
    a = _analyze(
        '''
        def f():
            """First line of doc.

            More detail.
            """
            pass
        ''',
        tmp_path,
    )
    sig = CodeAnalyzer().format_function_signature(a.functions[0])
    assert '"""First line of doc."""' in sig
    assert "More detail" not in sig  # only the first line is rendered


def test_function_signature_no_docstring(tmp_path):
    a = _analyze("def f():\n    pass\n", tmp_path)
    sig = CodeAnalyzer().format_function_signature(a.functions[0])
    assert '"""' not in sig


# --- format_class_signature (L358-381) ---

def test_format_class_with_bases_docstring_methods(tmp_path):
    a = _analyze(
        '''
        class Dog(Animal):
            """A dog subclass."""

            def bark(self):
                """Make noise."""
                return "woof"

            def sit(self):
                pass
        ''',
        tmp_path,
    )
    out = CodeAnalyzer().format_class_signature(a.classes[0])
    assert "class Dog(Animal):" in out
    assert '"""A dog subclass."""' in out
    assert "    def bark(self):" in out  # indented method signature


def test_format_class_no_bases_no_methods(tmp_path):
    a = _analyze("class Empty:\n    pass\n", tmp_path)
    out = CodeAnalyzer().format_class_signature(a.classes[0])
    assert out.startswith("class Empty:")
    assert "(" not in out


def test_format_class_with_decorator(tmp_path):
    a = _analyze(
        '''
        @dec
        class C:
            """doc"""
        ''',
        tmp_path,
    )
    out = CodeAnalyzer().format_class_signature(a.classes[0])
    assert out.startswith("@dec")
    assert "class C:" in out


# --- _node_to_string fallback (L312-320): ast.unparse failure on older Python ---

def test_node_to_string_fallback_name():
    """When ast.unparse raises, a Name node falls back to node.id."""
    analyzer = CodeAnalyzer()
    node = ast.parse("x", mode="eval").body
    with mock.patch("external_llm.code_analyzer.ast.unparse", side_effect=RuntimeError):
        assert analyzer._node_to_string(node) == "x"


def test_node_to_string_fallback_constant():
    analyzer = CodeAnalyzer()
    node = ast.parse("5", mode="eval").body
    with mock.patch("external_llm.code_analyzer.ast.unparse", side_effect=RuntimeError):
        assert analyzer._node_to_string(node) == repr(5)


def test_node_to_string_fallback_attribute():
    analyzer = CodeAnalyzer()
    node = ast.parse("a.b", mode="eval").body
    with mock.patch("external_llm.code_analyzer.ast.unparse", side_effect=RuntimeError):
        assert analyzer._node_to_string(node) == "a.b"


def test_node_to_string_fallback_other_node():
    """Non Name/Constant/Attribute (e.g. BinOp) → type name fallback."""
    analyzer = CodeAnalyzer()
    node = ast.parse("x + y", mode="eval").body
    with mock.patch("external_llm.code_analyzer.ast.unparse", side_effect=RuntimeError):
        assert analyzer._node_to_string(node) == "BinOp"


# --- remaining branch coverage: non-Name assign targets + unnamed inner calls ---

def test_assign_with_non_name_target_skipped(tmp_path):
    """Tuple-unpacking target ``a, b = 1, 2`` is a Tuple, not a Name → skipped."""
    a = _analyze("a, b = 1, 2\nREAL = 5\n", tmp_path)
    # neither a nor b recorded (Tuple target is not a Name); REAL is.
    assert "a" not in a.global_vars
    assert "b" not in a.global_vars
    assert a.global_vars["REAL"] == "5"


def test_function_unnamed_inner_call_skipped(tmp_path):
    """A call with non-Name/Attribute func inside a fn body is skipped gracefully."""
    a = _analyze(
        """
        def f():
            (lambda: 1)()
            real_call()
        """,
        tmp_path,
    )
    func = a.functions[0]
    assert "real_call" in func.calls
    # the lambda call contributes nothing (no resolvable name)
