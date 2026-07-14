"""Adversarial tests for code_structure_utils.py.

Set 5: AST walker wrapper-miss family — same root cause as placement_shadow's
module-level filter (Set 1).  ``parse_definitions`` / ``_walk_definitions`` /
``find_import_boundary_ast`` / ``symbol_exists_in_module`` all use
``ast.iter_child_nodes`` which only sees direct children of the module, so
definitions wrapped in ``if TYPE_CHECKING:`` / ``try-except`` / ``with`` /
``for`` / ``while`` are silently invisible.

Per CLAUDE.md, code_structure_utils is the canonical structure-detection
oracle for the patch engine — its blind spots propagate into patch strategy
selection.  The whole 348-line file ships without unit tests; this file
opens that coverage gap.
"""
from __future__ import annotations

import textwrap

from external_llm.code_structure_utils import (
    NON_SCOPE_COMPOUND_STMTS,
    collect_defined_names,
    extract_function_signature,
    extract_symbol_name,
    find_definition_at_line,
    find_import_boundary_ast,
    is_class_def,
    is_decorator,
    is_function_def,
    iter_module_scope_nodes,
    parse_definitions,
    symbol_defined_anywhere,
    symbol_exists_in_module,
)

# ── Line-level heuristics ──────────────────────────────────────────────────


class TestLineLevelHeuristics:
    def test_is_function_def_basic(self):
        assert is_function_def("def foo():")
        assert is_function_def("async def foo():")
        assert is_function_def("    def nested():")
        assert not is_function_def("class Foo:")
        assert not is_function_def("# def foo")
        assert not is_function_def("")

    def test_is_function_def_dict_key_with_def_value(self):
        # ``"key": def_value`` is not a function — but starts with ``"`` so OK.
        assert not is_function_def('"key": def value')

    def test_is_function_def_label_named_def(self):
        # Counter: ``def: stmt`` (statement label syntax — actually invalid
        # Python but the heuristic shouldn't claim it as a function).
        assert not is_function_def("def: pass")

    def test_is_class_def_basic(self):
        assert is_class_def("class Foo:")
        assert is_class_def("class Foo(Base):")
        assert is_class_def("    class Nested:")
        assert not is_class_def("def Foo():")

    def test_is_class_def_classmethod_decorator_above(self):
        # The line itself is just the decorator — not a class def.
        assert not is_class_def("@dataclass")

    def test_is_decorator_basic(self):
        assert is_decorator("@cached")
        assert is_decorator("    @property")
        assert is_decorator("@functools.wraps(func)")
        assert not is_decorator("# @comment")
        assert not is_decorator("def foo():")

    def test_extract_symbol_name_function(self):
        name, kind = extract_symbol_name("def foo(x):")
        assert (name, kind) == ("foo", "function")

    def test_extract_symbol_name_async_function(self):
        name, kind = extract_symbol_name("async def bar():")
        assert (name, kind) == ("bar", "function")

    def test_extract_symbol_name_class(self):
        name, kind = extract_symbol_name("class MyClass(Base):")
        assert (name, kind) == ("MyClass", "class")

    def test_extract_symbol_name_nested_indented(self):
        name, kind = extract_symbol_name("    def inner(self):")
        assert (name, kind) == ("inner", "function")

    def test_extract_symbol_name_class_with_metaclass(self):
        name, kind = extract_symbol_name("class C(Base, metaclass=Meta):")
        assert (name, kind) == ("C", "class")

    def test_extract_symbol_name_assignment_returns_none(self):
        name, kind = extract_symbol_name("x = 42")
        assert (name, kind) == (None, "unknown")

    def test_extract_symbol_name_pep695_generic_class(self):
        # Python 3.12+ generic class syntax: ``class C[T]:``
        # The current regex uses ``[\\(\\[:]`` so this should match — pin it.
        name, kind = extract_symbol_name("class Container[T]:")
        # Without a fix this might emit "Container[T" or similar — assert
        # the right shape so the failure mode is obvious.
        assert kind == "class"
        assert name == "Container"


# ── parse_definitions / _walk_definitions ──────────────────────────────────


class TestParseDefinitions:
    """Set 5 main hunting ground: every wrapper-miss case the placement
    shadow filter fixed must also be fixed here, OR the function must
    document why it deliberately ignores wrapped defs."""

    def test_top_level_function(self):
        src = "def foo():\n    return 1\n"
        defs = parse_definitions(src)
        assert len(defs) == 1
        assert defs[0].name == "foo"
        assert defs[0].kind == "function"

    def test_top_level_class_with_methods(self):
        src = (
            "class C:\n"
            "    def m1(self): pass\n"
            "    async def m2(self): pass\n"
        )
        defs = parse_definitions(src)
        names = {d.name for d in defs}
        assert names == {"C", "m1", "m2"}
        # Methods carry parent_class
        for d in defs:
            if d.name in ("m1", "m2"):
                assert d.parent_class == "C"

    def test_async_function_kind(self):
        src = "async def f():\n    return 1\n"
        defs = parse_definitions(src)
        assert len(defs) == 1
        assert defs[0].kind == "async_function"

    def test_decorator_stack_uses_first_decorator_lineno(self):
        src = (
            "@dec1\n"
            "@dec2\n"
            "def f():\n"
            "    pass\n"
        )
        defs = parse_definitions(src)
        assert len(defs) == 1
        # start_line is the FIRST decorator (line 1), not the def line
        assert defs[0].start_line == 1
        assert defs[0].decorators == ["dec1", "dec2"]

    def test_syntax_error_returns_empty(self):
        defs = parse_definitions("def f(:")
        assert defs == []

    # ── Adversarial: wrapper-stmt invisibility ────────────────────────

    def test_function_inside_type_checking_block(self):
        """``if TYPE_CHECKING: def helper(): ...`` — helper is a
        module-level binding at runtime under TYPE_CHECKING semantics
        (and a real def under most Python tools).  AST walker that uses
        only iter_child_nodes will MISS this def — same root cause as
        the JSONResponse anchor leak in placement_shadow."""
        src = textwrap.dedent("""
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                def stub_helper() -> int:
                    return 1
        """).lstrip()
        defs = parse_definitions(src)
        names = {d.name for d in defs}
        assert "stub_helper" in names, (
            f"TYPE_CHECKING-wrapped def invisible — same wrapper-miss "
            f"as placement Set 1. names={names}"
        )

    def test_function_inside_try_except(self):
        """Common pattern for optional features: ``try: def fast_impl():
        ... ; except ImportError: def fast_impl(): ...`` — both branches
        define a module-level function and the AST walker must see them."""
        src = textwrap.dedent("""
            try:
                def fast_impl():
                    return "fast"
            except ImportError:
                def fast_impl():
                    return "slow"
        """).lstrip()
        defs = parse_definitions(src)
        names = [d.name for d in defs]
        assert names.count("fast_impl") >= 1, (
            f"try/except def invisible. defs={names}"
        )

    def test_class_inside_if_version_check(self):
        """``if sys.version_info >= (3, 11): class C(...): else: class C(...)``
        — both branches define module-level class."""
        src = textwrap.dedent("""
            import sys
            if sys.version_info >= (3, 11):
                class FastPath:
                    pass
            else:
                class FastPath:
                    pass
        """).lstrip()
        defs = parse_definitions(src)
        names = [d.name for d in defs]
        assert "FastPath" in names, (
            f"version-gated class invisible. defs={names}"
        )

    def test_function_inside_with_block(self):
        """``with contextlib.suppress(Exception): def helper(): ...`` —
        legal Python; helper is module-level."""
        src = textwrap.dedent("""
            import contextlib
            with contextlib.suppress(Exception):
                def helper():
                    return None
        """).lstrip()
        defs = parse_definitions(src)
        names = {d.name for d in defs}
        assert "helper" in names, (
            f"with-wrapped def invisible. names={names}"
        )

    def test_nested_if_class_inside_class(self):
        """if -> if -> def: nested wrapper, all module-scope."""
        src = textwrap.dedent("""
            FLAG_A = True
            FLAG_B = True
            if FLAG_A:
                if FLAG_B:
                    def deeply_wrapped():
                        return 0
        """).lstrip()
        defs = parse_definitions(src)
        names = {d.name for d in defs}
        assert "deeply_wrapped" in names

    def test_function_inside_function_NOT_promoted_to_module(self):
        """Counter-test: nested function INSIDE another function is NOT
        module-level — but parse_definitions DOES still record it (this
        function deliberately walks into function bodies via the
        recursive _walk_definitions call).  Pin that behaviour so a
        refactor that drops nested-def visibility surfaces."""
        src = textwrap.dedent("""
            def outer():
                def inner():
                    return 1
                return inner
        """).lstrip()
        defs = parse_definitions(src)
        names = {d.name for d in defs}
        assert "outer" in names
        assert "inner" in names  # nested-def visibility preserved


# ── find_definition_at_line ────────────────────────────────────────────────


class TestFindDefinitionAtLine:
    def test_finds_def_when_line_inside_body(self):
        src = "def foo():\n    return 1\n"
        d = find_definition_at_line(src, 2)
        assert d is not None and d.name == "foo"

    def test_returns_none_for_line_outside_any_def(self):
        src = "import os\n\ndef foo():\n    return 1\n"
        d = find_definition_at_line(src, 1)
        assert d is None

    def test_innermost_match_wins_for_method(self):
        src = textwrap.dedent("""
            class C:
                def m(self):
                    return 1
        """).lstrip()
        d = find_definition_at_line(src, 3)
        assert d is not None
        # innermost: method m, not the enclosing class C
        assert d.name == "m"

    def test_finds_wrapped_def_when_walker_handles_wrappers(self):
        """Same wrapper-miss pattern: if parse_definitions misses a
        wrapped def, find_definition_at_line(line in the wrapped body)
        returns None and the patch engine has no idea what symbol it's
        looking at."""
        src = textwrap.dedent("""
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                def wrapped_def():
                    return "in body"
        """).lstrip()
        d = find_definition_at_line(src, 4)  # line of "return"
        assert d is not None and d.name == "wrapped_def", (
            f"wrapped def invisible to find_definition_at_line. got={d}"
        )


# ── find_import_boundary_ast ───────────────────────────────────────────────


class TestFindImportBoundary:
    def test_basic_imports_then_def(self):
        src = "import os\nimport sys\n\ndef f(): pass\n"
        boundary = find_import_boundary_ast(src)
        # Last import end is line 2; first non-import (def) is line 4.
        assert boundary == 3 or boundary == 4

    def test_empty_source_returns_one(self):
        assert find_import_boundary_ast("") == 1

    def test_module_docstring_above_imports_handled(self):
        src = '"""docstring"""\nimport os\n\ndef f(): pass\n'
        boundary = find_import_boundary_ast(src)
        # Must come after both docstring (line 1) and import (line 2).
        assert boundary >= 3

    def test_only_imports(self):
        src = "import os\nimport sys\n"
        boundary = find_import_boundary_ast(src)
        assert boundary >= 3

    def test_try_except_import_NOT_lost_at_boundary(self):
        """Optional-dep pattern: ``try: import yaml; except: yaml = None``
        — the yaml import is module-scope.  The boundary AST scan must
        recognise the try as containing an import (not as the first
        non-import statement).  Without this fix, the boundary lands at
        the try line, and any later regular ``import`` is now AFTER the
        boundary — completely wrong."""
        src = textwrap.dedent("""
            import os
            try:
                import yaml
            except ImportError:
                yaml = None
            import sys

            def f():
                pass
        """).lstrip()
        boundary = find_import_boundary_ast(src)
        # The real boundary is the line of `def f` (line 8 in this src).
        # If boundary lands BEFORE the import sys line, we've lost
        # imports below it.
        # Find the line number of "def f" and "import sys" for assert.
        lines = src.splitlines()
        def_line = next(i + 1 for i, ln in enumerate(lines) if ln.strip().startswith("def f"))
        import_sys_line = next(
            i + 1 for i, ln in enumerate(lines) if "import sys" in ln
        )
        assert boundary > import_sys_line, (
            f"boundary {boundary} fell before 'import sys' line {import_sys_line} "
            f"— try-except import truncated the import region. def line={def_line}"
        )


# ── symbol_exists_in_module ────────────────────────────────────────────────


class TestSymbolExistsInModule:
    def test_basic_def(self):
        src = "def foo(): pass\n"
        assert symbol_exists_in_module(src, "foo")

    def test_basic_class(self):
        src = "class C: pass\n"
        assert symbol_exists_in_module(src, "C")

    def test_module_level_assign(self):
        src = "X = 42\n"
        assert symbol_exists_in_module(src, "X")

    def test_annotated_assign(self):
        src = "X: int = 42\n"
        assert symbol_exists_in_module(src, "X")

    def test_returns_false_for_nonexistent(self):
        src = "def foo(): pass\n"
        assert not symbol_exists_in_module(src, "bar")

    def test_returns_false_on_syntax_error(self):
        assert not symbol_exists_in_module("def f(:", "f")

    def test_class_method_NOT_module_level(self):
        """Counter: class method is class-scope, not module-scope."""
        src = "class C:\n    def method(self): pass\n"
        assert not symbol_exists_in_module(src, "method")
        assert symbol_exists_in_module(src, "C")

    # ── Adversarial: wrapped module-level definitions ─────────────────

    def test_type_checking_def_is_module_level(self):
        """if TYPE_CHECKING: def helper() — same wrapper-miss family."""
        src = textwrap.dedent("""
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                def stub():
                    return None
        """).lstrip()
        assert symbol_exists_in_module(src, "stub"), (
            "TYPE_CHECKING-wrapped def not detected as module-level"
        )

    def test_try_except_assign_is_module_level(self):
        """Optional-dep pattern: try: SECRET = lookup() ; except: SECRET = ''
        — SECRET is bound at module scope regardless of branch."""
        src = textwrap.dedent("""
            try:
                SECRET = lookup_env()
            except KeyError:
                SECRET = ''
        """).lstrip()
        assert symbol_exists_in_module(src, "SECRET"), (
            "try-except assign not detected as module-level"
        )

    def test_version_gated_class_is_module_level(self):
        src = textwrap.dedent("""
            import sys
            if sys.version_info >= (3, 11):
                class FastPath: pass
            else:
                class FastPath: pass
        """).lstrip()
        assert symbol_exists_in_module(src, "FastPath")


# ── symbol_defined_anywhere (uses ast.walk; should be more permissive) ────


class TestSymbolDefinedAnywhere:
    def test_top_level_function(self):
        src = "def foo(): pass\n"
        assert symbol_defined_anywhere(src, "foo")

    def test_class_method(self):
        src = "class C:\n    def method(self): pass\n"
        # ast.walk reaches into class body
        assert symbol_defined_anywhere(src, "method")

    def test_nested_function(self):
        src = textwrap.dedent("""
            def outer():
                def inner():
                    pass
        """).lstrip()
        assert symbol_defined_anywhere(src, "inner")

    def test_wrapper_stmts_handled_by_walk(self):
        """ast.walk DOES descend into all nodes, so wrapper-miss should
        not affect this function — pin that behaviour to detect a
        regression that switches it back to iter_child_nodes."""
        src = textwrap.dedent("""
            from typing import TYPE_CHECKING
            if TYPE_CHECKING:
                def wrapped(): pass
        """).lstrip()
        assert symbol_defined_anywhere(src, "wrapped")


# ── collect_defined_names ──────────────────────────────────────────────────


class TestCollectDefinedNames:
    def test_basic_collection(self):
        src = "def f(): pass\nclass C: pass\nX = 1\n"
        names = collect_defined_names(src)
        assert {"f", "C", "X"}.issubset(names)


# ── TS/JS/Go symbol detection (tree-sitter-first, regex fallback) ─────────
# These pin the fix for the comment false-positive bug: a symbol mentioned
# only inside a comment must NOT be reported as defined.  Before the
# tree-sitter migration, the regex-based detectors matched symbols inside
# comments, producing false-positives that could let a broken edit pass
# verification (e.g. a deleted symbol whose name lingered in a comment).


class TestTSSymbolDefinedAnywhere:
    """symbol_defined_anywhere with file_path for TS/JS — comment-safe."""

    def test_comment_mentioned_symbol_is_not_defined(self):
        src = (
            "// const fakeSymbol = 1;\n"
            "// function fakeFunc() {}\n"
            "export function realFunc() { return 1; }\n"
        )
        assert not symbol_defined_anywhere(src, "fakeSymbol", "app.ts")
        assert not symbol_defined_anywhere(src, "fakeFunc", "app.ts")
        assert symbol_defined_anywhere(src, "realFunc", "app.ts")

    def test_class_and_method(self):
        src = (
            "export class Foo {\n"
            "  getBar(): string { return ''; }\n"
            "}\n"
        )
        assert symbol_defined_anywhere(src, "Foo", "app.ts")
        assert symbol_defined_anywhere(src, "getBar", "app.ts")
        assert symbol_defined_anywhere(src, "Foo.getBar", "app.ts")

    def test_javascript_path(self):
        src = "export function jsFunc() {}\n"
        assert symbol_defined_anywhere(src, "jsFunc", "app.js")
        assert not symbol_defined_anywhere(src, "missing", "app.js")

    def test_collect_excludes_comments(self):
        src = (
            "// function fakeFunc() {}\n"
            "export function realFunc() {}\n"
        )
        names = collect_defined_names(src, "app.ts")
        assert "realFunc" in names
        assert "fakeFunc" not in names


class TestGoSymbolDefinedAnywhere:
    """symbol_defined_anywhere with file_path for Go — comment-safe."""

    def test_comment_mentioned_symbol_is_not_defined(self):
        src = (
            "package main\n"
            "// func fakeFunc() {}\n"
            "// type fakeType struct{}\n"
            "func realFunc() {}\n"
            "type realType struct{}\n"
        )
        assert not symbol_defined_anywhere(src, "fakeFunc", "app.go")
        assert not symbol_defined_anywhere(src, "fakeType", "app.go")
        assert symbol_defined_anywhere(src, "realFunc", "app.go")
        assert symbol_defined_anywhere(src, "realType", "app.go")

    def test_method_bare_and_dotted(self):
        src = (
            "package main\n"
            "type Foo struct{}\n"
            "func (f *Foo) GetBar() string { return \"\" }\n"
        )
        # bare method name
        assert symbol_defined_anywhere(src, "GetBar", "app.go")
        # dotted (regex fallback — Go methods aren't nested in type range)
        assert symbol_defined_anywhere(src, "Foo.GetBar", "app.go")
        assert symbol_defined_anywhere(src, "Foo", "app.go")

    def test_collect_excludes_comments(self):
        src = (
            "package main\n"
            "// func fakeFunc() {}\n"
            "func realFunc() {}\n"
        )
        names = collect_defined_names(src, "app.go")
        assert "realFunc" in names
        assert "fakeFunc" not in names


class TestJvmSymbolDefinedAnywhere:
    """symbol_defined_anywhere with file_path for Java/Kotlin.

    Java exercises the tree-sitter-first path (grammar installed → accurate,
    comment-safe). Kotlin exercises the regex fallback (grammar may be absent),
    so only positive findings are asserted for Kotlin (regex cannot prove
    comment-safety). Both paths must agree on real-symbol presence.
    """

    # ── Java (tree-sitter-first) ──────────────────────────────────────────
    def test_java_comment_mentioned_symbol_is_not_defined(self):
        src = (
            "package com.example;\n"
            "// public class FakeClass {}\n"
            "// void fakeMethod() {}\n"
            "public class Calculator {\n"
            "    public int add(int a, int b) { return a + b; }\n"
            "}\n"
        )
        assert not symbol_defined_anywhere(src, "FakeClass", "App.java")
        assert not symbol_defined_anywhere(src, "fakeMethod", "App.java")
        assert symbol_defined_anywhere(src, "Calculator", "App.java")
        assert symbol_defined_anywhere(src, "add", "App.java")

    def test_java_method_bare_and_dotted(self):
        src = (
            "package com.example;\n"
            "public class Calculator {\n"
            "    public int add(int a, int b) { return a + b; }\n"
            "    private void log(String msg) {}\n"
            "}\n"
        )
        # bare method name (nested member)
        assert symbol_defined_anywhere(src, "add", "App.java")
        assert symbol_defined_anywhere(src, "log", "App.java")
        # dotted — containment-confirmed (Java nests members, unlike Go)
        assert symbol_defined_anywhere(src, "Calculator.add", "App.java")
        # dotted referencing a member that does NOT exist → False (definitive)
        assert not symbol_defined_anywhere(src, "Calculator.missing", "App.java")
        # dotted referencing a non-existent class → False
        assert not symbol_defined_anywhere(src, "NoClass.add", "App.java")

    def test_java_interface_and_enum(self):
        src = (
            "package com.example;\n"
            "interface Runnable { void run(); }\n"
            "enum Color { RED, GREEN, BLUE }\n"
        )
        assert symbol_defined_anywhere(src, "Runnable", "App.java")
        assert symbol_defined_anywhere(src, "run", "App.java")
        assert symbol_defined_anywhere(src, "Color", "App.java")

    def test_java_collect_excludes_comments(self):
        src = (
            "package com.example;\n"
            "// public class FakeClass {}\n"
            "public class RealClass {}\n"
        )
        names = collect_defined_names(src, "App.java")
        assert "RealClass" in names
        assert "FakeClass" not in names

    # ── Kotlin (regex fallback when grammar absent) ───────────────────────
    def test_kotlin_class_fun_object(self):
        src = (
            "package com.example\n"
            "class Calculator(val x: Int) {\n"
            "    fun add(b: Int): Int = x + b\n"
            "}\n"
            "object Singleton { val name = \"x\" }\n"
        )
        assert symbol_defined_anywhere(src, "Calculator", "App.kt")
        assert symbol_defined_anywhere(src, "add", "App.kt")
        assert symbol_defined_anywhere(src, "Calculator.add", "App.kt")
        assert symbol_defined_anywhere(src, "Singleton", "App.kt")
        assert not symbol_defined_anywhere(src, "missing", "App.kt")

    def test_kotlin_collect(self):
        src = (
            "package com.example\n"
            "class Calculator {\n"
            "    fun add(b: Int): Int = x + b\n"
            "}\n"
        )
        names = collect_defined_names(src, "App.kt")
        assert "Calculator" in names
        assert "add" in names


class TestSymbolExistsInModuleNonPython:
    """symbol_exists_in_module module-level detection for TS/Go."""

    def test_ts_module_level_excludes_nested_methods(self):
        src = (
            "export class Foo {\n"
            "  method() {}\n"
            "}\n"
            "export function topLevel() {}\n"
        )
        assert symbol_exists_in_module(src, "topLevel", "app.ts")
        assert symbol_exists_in_module(src, "Foo", "app.ts")
        # method is nested inside Foo → NOT module-level
        assert not symbol_exists_in_module(src, "method", "app.ts")
        # dotted member → never module-level
        assert not symbol_exists_in_module(src, "Foo.method", "app.ts")

    def test_go_module_level_excludes_receiver_methods(self):
        src = (
            "package main\n"
            "type Foo struct{}\n"
            "func (f *Foo) GetBar() string { return \"\" }\n"
            "func TopLevel() {}\n"
            "var GlobalVar = 42\n"
            "const Pi = 3.14\n"
        )
        assert symbol_exists_in_module(src, "Foo", "app.go")
        assert symbol_exists_in_module(src, "TopLevel", "app.go")
        assert symbol_exists_in_module(src, "GlobalVar", "app.go")
        assert symbol_exists_in_module(src, "Pi", "app.go")
        # receiver method → NOT module-level (Go-specific correctness)
        assert not symbol_exists_in_module(src, "GetBar", "app.go")

    def test_backward_compat_no_filepath(self):
        """Without file_path, symbol_exists_in_module keeps Python behaviour."""
        src = "def foo(): pass\nclass C: pass\n"
        assert symbol_exists_in_module(src, "foo")
        assert symbol_exists_in_module(src, "C")
        assert not symbol_exists_in_module(src, "missing")


# ── extract_function_signature ────────────────────────────────────────────


class TestExtractFunctionSignature:
    """``extract_function_signature`` deliberately EXCLUDES the function
    name — its purpose is comparing signatures across refactor renames,
    so the canonical form is ``(args)->return``."""

    def test_basic(self):
        src = "def foo(a, b=1): return a + b\n"
        sig = extract_function_signature(src, "foo")
        assert sig is not None
        # Signature only — no function name.
        assert "a" in sig and "b" in sig
        assert sig.startswith("(") and ")->" in sig

    def test_async_function(self):
        src = "async def bar(x: int) -> str: return str(x)\n"
        sig = extract_function_signature(src, "bar")
        assert sig is not None
        # Annotations preserved.
        assert "x:int" in sig
        assert sig.endswith("->str")

    def test_missing_symbol_returns_none(self):
        sig = extract_function_signature("def other(): pass\n", "missing")
        assert sig is None


# ── iter_module_scope_nodes: canonical wrapper-aware walker ───────────────


class TestIterModuleScopeNodes:
    """The single source of truth helper that placement_contract,
    intent_verifier, and code_structure_utils itself all import.  Pin
    the contract so a refactor can't silently re-introduce wrapper-miss
    by changing the descent rules."""

    def _names_yielded(self, src: str):
        import ast as _ast
        tree = _ast.parse(src)
        return [
            getattr(n, "name", None) or getattr(n, "id", None)
            for n in iter_module_scope_nodes(tree)
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
        ]

    def test_root_tree_not_yielded(self):
        """Caller pre-refactor used iter_child_nodes(tree); the new helper
        must yield exactly the same starting set (children of the module)
        and NOT yield the Module node itself."""
        import ast as _ast
        tree = _ast.parse("def f(): pass\n")
        nodes = list(iter_module_scope_nodes(tree))
        assert tree not in nodes
        assert any(isinstance(n, _ast.FunctionDef) and n.name == "f" for n in nodes)

    def test_wrapper_descent_if(self):
        names = self._names_yielded(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    def helper(): pass\n"
        )
        assert "helper" in names

    def test_wrapper_descent_try(self):
        names = self._names_yielded(
            "try:\n"
            "    def fast(): pass\n"
            "except ImportError:\n"
            "    def fast(): pass\n"
        )
        assert names.count("fast") >= 1

    def test_wrapper_descent_with_async_for_while(self):
        for stmt, body in [
            ("with open('x') as f:", "    def helper(): pass"),
            ("for _ in range(1):", "    def helper(): pass"),
            ("while True:", "    def helper(): pass\n    break"),
        ]:
            src = f"{stmt}\n{body}\n"
            names = self._names_yielded(src)
            assert "helper" in names, f"wrapper descent failed for: {stmt}"

    def test_function_body_NOT_entered(self):
        """``def outer(): def inner(): ...`` — inner is local to outer,
        NOT module scope.  The walker must yield outer (module-scope
        binding) and stop — inner must not appear as if it were
        module-level."""
        import ast as _ast
        tree = _ast.parse(
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
        )
        names = [
            n.name for n in iter_module_scope_nodes(tree)
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
        ]
        assert names == ["outer"], (
            f"walker entered function body — inner leaked to module scope. names={names}"
        )

    def test_class_body_NOT_entered(self):
        """``class C: def m(self): ...`` — m is class-scope, not module."""
        import ast as _ast
        tree = _ast.parse(
            "class C:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        names = [
            n.name for n in iter_module_scope_nodes(tree)
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef))
        ]
        assert names == ["C"]

    def test_expression_descent_for_walrus(self):
        """Module-level ``if (n := compute()):`` — NamedExpr is nested
        inside If.test (not a stmt-level child).  The walker must
        descend into expression nodes too so callers like
        placement_contract.extract_module_level_names can find walrus
        bindings.  Pin this; if a future refactor stops at stmt level,
        the walrus capture in placement_contract regresses silently."""
        import ast as _ast
        tree = _ast.parse("if (n := 1):\n    pass\n")
        named_exprs = [
            n for n in iter_module_scope_nodes(tree)
            if isinstance(n, _ast.NamedExpr)
        ]
        assert len(named_exprs) == 1
        assert isinstance(named_exprs[0].target, _ast.Name)
        assert named_exprs[0].target.id == "n"

    def test_deterministic_source_order(self):
        """Two consecutive runs over the same source yield identical
        sequences (no nondeterministic ordering in the DFS)."""
        import ast as _ast
        src = (
            "def a(): pass\n"
            "if True:\n"
            "    def b(): pass\n"
            "def c(): pass\n"
        )
        tree = _ast.parse(src)
        run1 = [
            n.name for n in iter_module_scope_nodes(tree)
            if isinstance(n, _ast.FunctionDef)
        ]
        run2 = [
            n.name for n in iter_module_scope_nodes(tree)
            if isinstance(n, _ast.FunctionDef)
        ]
        assert run1 == run2
        # Source order: a, b, c
        assert run1 == ["a", "b", "c"]

    def test_non_scope_compound_stmts_set_completeness(self):
        """The set must cover every non-scope compound; if Python adds a
        new one (e.g. PEP 654 ExceptionGroup may have already pulled
        ast.TryStar in), this test forces the maintainer to update the
        canonical set rather than drift silently."""
        import ast as _ast
        expected = {
            _ast.If, _ast.Try,
            _ast.With, _ast.AsyncWith,
            _ast.For, _ast.AsyncFor,
            _ast.While,
        }
        assert set(NON_SCOPE_COMPOUND_STMTS) == expected, (
            f"NON_SCOPE_COMPOUND_STMTS drift detected. "
            f"current={set(NON_SCOPE_COMPOUND_STMTS)}, expected={expected}"
        )


class TestExtractFunctionSignatureExtras:
    """Adversarial cases on top of the basic TestExtractFunctionSignature."""

    def test_kwargs_and_vararg(self):
        src = "def f(*args, **kwargs): pass\n"
        sig = extract_function_signature(src, "f")
        assert sig is not None
        assert "*args" in sig
        assert "**kwargs" in sig

    def test_signature_stable_under_rename(self):
        """Refactor rename should produce identical signature."""
        sig_a = extract_function_signature(
            "def old_name(x: int) -> int: return x\n", "old_name"
        )
        sig_b = extract_function_signature(
            "def new_name(x: int) -> int: return x\n", "new_name"
        )
        assert sig_a == sig_b
        assert sig_a is not None


class TestExtractFunctionSignatureDetailed:
    """``extract_function_signature_detailed`` returns separated param_sig/return_type."""

    def test_separated_fields(self):
        from external_llm.code_structure_utils import FunctionSignature, extract_function_signature_detailed
        src = "def foo(a: int, b: str) -> bool: return True\n"
        sig = extract_function_signature_detailed(src, "foo")
        assert sig is not None
        assert isinstance(sig, FunctionSignature)
        assert "a:int" in sig.param_sig
        assert "b:str" in sig.param_sig
        assert sig.return_type == "bool"

    def test_canonical_equals_legacy(self):
        from external_llm.code_structure_utils import extract_function_signature, extract_function_signature_detailed
        src = "def bar(x: int = 5) -> str: return str(x)\n"
        detailed = extract_function_signature_detailed(src, "bar")
        legacy = extract_function_signature(src, "bar")
        assert detailed is not None
        assert legacy is not None
        assert detailed.canonical == legacy

    def test_return_type_only_change(self):
        from external_llm.code_structure_utils import extract_function_signature_detailed
        before = "def f(a: int) -> str: return ''\n"
        after = "def f(a: int) -> int: return 0\n"
        pre = extract_function_signature_detailed(before, "f")
        post = extract_function_signature_detailed(after, "f")
        assert pre is not None and post is not None
        # param_sig unchanged, return_type changed
        assert pre.param_sig == post.param_sig
        assert pre.return_type != post.return_type
        assert pre.return_type == "str"
        assert post.return_type == "int"

    def test_no_return_annotation(self):
        from external_llm.code_structure_utils import extract_function_signature_detailed
        src = "def g(a): pass\n"
        sig = extract_function_signature_detailed(src, "g")
        assert sig is not None
        assert "a:" in sig.param_sig
        assert sig.return_type == ""

    def test_async_function(self):
        from external_llm.code_structure_utils import extract_function_signature_detailed
        src = "async def h(x: float) -> None: return\n"
        sig = extract_function_signature_detailed(src, "h")
        assert sig is not None
        assert "x:float" in sig.param_sig
        assert sig.return_type == "None"

    def test_missing_symbol_returns_none(self):
        from external_llm.code_structure_utils import extract_function_signature_detailed
        sig = extract_function_signature_detailed("def other(): pass\n", "missing")
        assert sig is None

    def test_kwargs_and_vararg(self):
        from external_llm.code_structure_utils import extract_function_signature_detailed
        src = "def f(*args, **kwargs): pass\n"
        sig = extract_function_signature_detailed(src, "f")
        assert sig is not None
        assert "*args:" in sig.param_sig
        assert "**kwargs:" in sig.param_sig
