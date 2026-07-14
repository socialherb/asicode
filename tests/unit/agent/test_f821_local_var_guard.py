"""Tests for F821 auto-repair local variable guard.

Core guarantees:
1. Names assigned later within a function (forward reference) are excluded from import-insertion targets
2. Truly undefined names (never assigned anywhere in the function) are handled normally
3. Various assignment forms are recognized: walrus operator / for target / with...as target
"""
from __future__ import annotations

import ast
import textwrap


def _collect_local_assigned_names(source: str) -> set:
    """Extract the local-variable-guard logic from repair_engine for isolated testing."""
    local_assigned_names: set = set()
    try:
        tree = ast.parse(source)
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Function parameters are local names
            fn_args = fn_node.args
            for arg in fn_args.args + fn_args.posonlyargs + fn_args.kwonlyargs:
                local_assigned_names.add(arg.arg)
            if fn_args.vararg:
                local_assigned_names.add(fn_args.vararg.arg)
            if fn_args.kwarg:
                local_assigned_names.add(fn_args.kwarg.arg)
            for child in ast.walk(fn_node):
                if isinstance(child, ast.Assign):
                    for t in child.targets:
                        if isinstance(t, ast.Name):
                            local_assigned_names.add(t.id)
                elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                    at = getattr(child, 'target', None)
                    if isinstance(at, ast.Name):
                        local_assigned_names.add(at.id)
                elif isinstance(child, ast.NamedExpr):
                    if isinstance(child.target, ast.Name):
                        local_assigned_names.add(child.target.id)
                elif isinstance(child, ast.For):
                    if isinstance(child.target, ast.Name):
                        local_assigned_names.add(child.target.id)
                elif isinstance(child, ast.With):
                    for wi in child.items:
                        wv = getattr(wi, 'optional_vars', None)
                        if isinstance(wv, ast.Name):
                            local_assigned_names.add(wv.id)
    except Exception:
        pass
    return local_assigned_names


class TestLocalAssignedNames:
    """Verify the _local_assigned_names collection logic."""

    def test_simple_assignment(self):
        src = textwrap.dedent("""
        def foo():
            content = f.read()
            return content
        """)
        names = _collect_local_assigned_names(src)
        assert "content" in names

    def test_for_loop_target(self):
        src = textwrap.dedent("""
        def foo():
            for item in lst:
                pass
        """)
        names = _collect_local_assigned_names(src)
        assert "item" in names

    def test_with_as_target(self):
        src = textwrap.dedent("""
        def foo():
            with open("x") as f:
                data = f.read()
        """)
        names = _collect_local_assigned_names(src)
        assert "f" in names
        assert "data" in names

    def test_aug_assign(self):
        src = textwrap.dedent("""
        def foo():
            count = 0
            count += 1
        """)
        names = _collect_local_assigned_names(src)
        assert "count" in names

    def test_function_parameters(self):
        """Function parameters are also recognized as local names."""
        src = textwrap.dedent("""
        def __init__(self, available_strategies=None, timeout=30, **kwargs):
            self._available_strategies = available_strategies
        """)
        names = _collect_local_assigned_names(src)
        assert "self" in names
        assert "available_strategies" in names
        assert "timeout" in names
        assert "kwargs" in names

    def test_walrus_operator(self):
        src = textwrap.dedent("""
        def foo(lst):
            if (n := len(lst)) > 0:
                return n
        """)
        names = _collect_local_assigned_names(src)
        assert "n" in names

    def test_module_level_name_not_included(self):
        """Module-scope variables should not be included."""
        src = textwrap.dedent("""
        MODULE_VAR = "hello"

        def foo():
            pass
        """)
        names = _collect_local_assigned_names(src)
        assert "MODULE_VAR" not in names

    def test_truly_undefined_not_in_local_names(self):
        """A name that was truly never assigned should not be in local_names."""
        src = textwrap.dedent("""
        def foo():
            print(undefined_name)
        """)
        names = _collect_local_assigned_names(src)
        assert "undefined_name" not in names

    def test_multiple_functions(self):
        """Collect local variables from multiple functions."""
        src = textwrap.dedent("""
        def foo():
            result = 1

        def bar():
            value = 2
        """)
        names = _collect_local_assigned_names(src)
        assert "result" in names
        assert "value" in names

    def test_nested_function_assignment(self):
        """Detect assignments within nested functions too."""
        src = textwrap.dedent("""
        def outer():
            def inner():
                inner_var = 42
            return inner
        """)
        names = _collect_local_assigned_names(src)
        assert "inner_var" in names


class TestF821GuardScenario:
    """Reproduce real case 1 failure pattern: prevent import insertion when content is a forward reference."""

    def test_forward_reference_content_detected_as_local(self):
        """Patch pattern generated by DeepSeek: content is used before it is assigned."""
        src = textwrap.dedent("""
        def _collect_prior_created_files(self, state, max_lines=200):
            for abs_path in paths:
                if not _os.path.isfile(abs_path):
                    continue
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpf:
                        tmpf.write(content)   # content used before assignment
                        tmp_path = tmpf.name
                    with open(abs_path, 'r') as f:
                        content = f.read()   # content assigned here
                    lines = content.splitlines()
                except Exception:
                    pass
        """)
        names = _collect_local_assigned_names(src)
        # content IS a local variable (assigned via `content = f.read()`)
        # → guard must prevent import insertion
        assert "content" in names

    def test_truly_missing_import_still_flagged(self):
        """A truly missing name should not be in local_names (repair is allowed to proceed)."""
        src = textwrap.dedent("""
        def foo():
            result = some_undefined_function()
        """)
        names = _collect_local_assigned_names(src)
        # some_undefined_function is not assigned, so not in locals
        assert "some_undefined_function" not in names
        # result IS assigned
        assert "result" in names
