"""Behavioral tests for fix_spec_claim_validator._get_method_ranges.

The helper classifies every function definition as either a class method
(direct child of a ClassDef body) or a module-level/nested function. It was
recently rewritten from an O(N²) nested ``ast.walk`` to a single-pass O(N)
identity-set approach; these tests pin the classification behavior so the
performance refactor cannot regress it.
"""

from __future__ import annotations

import ast

from external_llm.agent.fix_spec_claim_validator import _get_method_ranges


def _ranges(src: str):
    return _get_method_ranges(ast.parse(src))


def _by_name(src: str):
    return {name: flag for name, _start, _end, flag in _ranges(src)}


def test_module_level_function_is_not_a_method():
    src = "def top_level():\n    return 1\n"
    assert _by_name(src) == {"top_level": 0}


def test_direct_class_method_is_a_method():
    src = """
class Foo:
    def bar(self):
        return 1
"""
    assert _by_name(src) == {"bar": 1}


def test_async_class_method_is_a_method():
    src = """
class Foo:
    async def bar(self):
        return 1
"""
    assert _by_name(src) == {"bar": 1}


def test_nested_function_inside_method_is_not_a_method():
    # A function defined inside a method body is NOT a direct child of the
    # ClassDef, so it must not be flagged as a method.
    src = """
class Foo:
    def bar(self):
        def inner():
            return 1
        return inner
"""
    assert _by_name(src) == {"bar": 1, "inner": 0}


def test_method_in_nested_class_is_a_method():
    src = """
class Outer:
    class Inner:
        def deep(self):
            return 1
"""
    assert _by_name(src) == {"deep": 1}


def test_multiple_classes_each_own_their_methods():
    src = """
class A:
    def a_method(self):
        return 1

class B:
    def b_method(self):
        return 1

def standalone():
    return 1
"""
    assert _by_name(src) == {"a_method": 1, "b_method": 1, "standalone": 0}


def test_static_and_class_methods_are_methods():
    src = """
class Foo:
    @staticmethod
    def s():
        return 1
    @classmethod
    def c(cls):
        return 1
"""
    assert _by_name(src) == {"s": 1, "c": 1}


def test_line_ranges_are_recorded():
    src = """
class Foo:
    def bar(self):
        x = 1
        return x
"""
    ranges = _ranges(src)
    (name, start, end, flag), = ranges
    assert name == "bar"
    assert flag == 1
    # bar spans from its def line (line 3) through the return (line 5)
    assert start == 3
    assert end == 5


def test_empty_module_has_no_ranges():
    assert _ranges("") == []
    assert _ranges("# just a comment\n") == []


def test_ordering_is_deterministic_and_stable():
    # ast.walk uses BFS (it yields a node before recursing into its children),
    # so module-level defs appear before methods nested inside a class body.
    # The refactor must preserve this exact order — both old and new impls walk
    # the same tree, so running twice must give identical results.
    src = """
def first():
    pass
class C:
    def second(self):
        pass
def third():
    pass
"""
    first_pass = [name for name, *_ in _ranges(src)]
    second_pass = [name for name, *_ in _ranges(src)]
    assert first_pass == second_pass  # deterministic
    assert set(first_pass) == {"first", "second", "third"}
    # module-level defs come before the method nested in the class body
    assert first_pass.index("first") < first_pass.index("second")
    assert first_pass.index("third") < first_pass.index("second")
