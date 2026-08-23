"""Regression tests for ASTRewriter (external_llm/ast_rewrite.py).

AR-B1: ``_replace_node`` sliced at ``node.lineno - 1``. ``node.lineno`` points at
the ``def``/``class`` line, so decorators (which live ABOVE it) were left in
``lines[:start]``. Since ``new_code`` carries a complete symbol (decorators
included), the original decorators were re-introduced and applied twice.

This is the live patch-apply path (patch_engine.py primary + fallback rewrite,
service.py repair fallback, VM repair strategies) — duplication silently
corrupts files for side-effecting decorators like @property / @lru_cache /
@app.route. Coverage was 0/N branches on the decorator boundary before this file.

Same harness pattern as test_semantic_patch.py (SP-B1); the only divergence is
that ASTRewriter raises ValueError on not-found instead of returning None.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from external_llm.ast_rewrite import ASTRewriter


@pytest.fixture
def rw(tmp_path):
    return ASTRewriter(str(tmp_path))


def _write(tmp_path, name: str, source: str) -> str:
    (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return name


# --------------------------------------------------------------------------- #
# AR-B1: decorator boundary — no duplication
# --------------------------------------------------------------------------- #
class TestDecoratorBoundary:
    """The original decorator must be replaced, not layered beneath new_code's."""

    def test_decorated_function_no_duplicate(self, rw, tmp_path):
        original = """
        @decorator_a
        @decorator_b
        def old(a, b):
            return a + b
        """
        new_code = textwrap.dedent("""
        @decorator_a
        @decorator_b
        def old(a, b):
            return a - b
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_function(name, "old", new_code)
        assert result.new_text.count("@decorator") == 2  # was 4 (AR-B1)
        # body changed, decorators preserved exactly once
        tree = ast.parse(result.new_text)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        assert len(fn.decorator_list) == 2
        assert "a - b" in result.new_text

    def test_start_line_includes_decorators(self, rw, tmp_path):
        # No leading newline so the decorator is on physical line 0 (0-indexed).
        source = "@deco\ndef f():\n    return 1\n"
        (tmp_path / "sample.py").write_text(source, encoding="utf-8")
        new_code = "@deco\ndef f():\n    return 2\n"
        result = rw.replace_function("sample.py", "f", new_code)
        # start_line (0-indexed) must point at the decorator, not the `def` line.
        assert result.start_line == 0
        # The replaced span begins with the decorator — no duplication, no drift.
        assert result.new_text.count("@deco") == 1

    def test_decorated_class_no_duplicate(self, rw, tmp_path):
        original = """
        @register
        class Old:
            x = 1
        """
        new_code = textwrap.dedent("""
        @register
        class Old:
            x = 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_class(name, "Old", new_code)
        assert result.new_text.count("@register") == 1  # was 2 (AR-B1)
        assert "x = 2" in result.new_text

    def test_async_decorated_function_no_duplicate(self, rw, tmp_path):
        original = """
        @asyncio.coroutine
        async def old():
            return 1
        """
        new_code = textwrap.dedent("""
        @asyncio.coroutine
        async def old():
            return 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_function(name, "old", new_code)
        assert result.new_text.count("@asyncio.coroutine") == 1
        tree = ast.parse(result.new_text)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
        assert len(fn.decorator_list) == 1

    def test_topmost_decorator_is_start(self, rw, tmp_path):
        # With three stacked decorators, start must be the TOPMOST lineno-1,
        # not the bottom decorator or the def line.
        original = "@a\n@b\n@c\ndef f():\n    pass\n"
        (tmp_path / "sample.py").write_text(original, encoding="utf-8")
        new_code = "@a\n@b\n@c\ndef f():\n    return True\n"
        result = rw.replace_function("sample.py", "f", new_code)
        assert result.new_text.count("@") == 3  # exactly three decorators
        assert result.start_line == 0  # topmost @a is line 0

    def test_decorated_method_no_duplicate(self, rw, tmp_path):
        original = """
        class Foo:
            @property
            def value(self):
                return self._v
        """
        new_code = textwrap.dedent("""
            @property
            def value(self):
                return self._v * 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_method(name, "Foo", "value", new_code)
        assert result.new_text.count("@property") == 1  # was 2 (AR-B1)
        assert "_v * 2" in result.new_text

    def test_decorator_with_arguments_preserved_once(self, rw, tmp_path):
        original = """
        @functools.lru_cache(maxsize=128)
        def compute(x):
            return x * 1
        """
        new_code = textwrap.dedent("""
        @functools.lru_cache(maxsize=128)
        def compute(x):
            return x * 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_function(name, "compute", new_code)
        assert result.new_text.count("@functools.lru_cache") == 1
        assert "x * 2" in result.new_text

    def test_full_file_remains_syntactically_valid(self, rw, tmp_path):
        # Replacing a decorated symbol must not corrupt surrounding code.
        original = """import functools


@functools.lru_cache(maxsize=8)
def cached(x):
    return x


def plain():
    return 0
"""
        (tmp_path / "sample.py").write_text(original, encoding="utf-8")
        new_code = "@functools.lru_cache(maxsize=8)\ndef cached(x):\n    return x + 1\n"
        result = rw.replace_function("sample.py", "cached", new_code)
        # The whole file must still parse and keep both functions.
        tree = ast.parse(result.new_text)
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert names == {"cached", "plain"}
        assert result.new_text.endswith("\n")


# --------------------------------------------------------------------------- #
# Plain symbols — regression guards (no decorators)
# --------------------------------------------------------------------------- #
class TestPlainReplacement:
    def test_plain_function_replaced(self, rw, tmp_path):
        original = """
        def add(a, b):
            return a + b
        """
        new_code = textwrap.dedent("""
        def add(a, b):
            return a - b
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_function(name, "add", new_code)
        assert "a - b" in result.new_text
        assert "a + b" not in result.new_text

    def test_plain_class_replaced(self, rw, tmp_path):
        original = """
        class Model:
            version = 1
        """
        new_code = textwrap.dedent("""
        class Model:
            version = 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_class(name, "Model", new_code)
        assert "version = 2" in result.new_text

    def test_plain_method_replaced(self, rw, tmp_path):
        original = """
        class Calc:
            def compute(self, x):
                return x
        """
        new_code = textwrap.dedent("""
            def compute(self, x):
                return x + 10
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_method(name, "Calc", "compute", new_code)
        assert "x + 10" in result.new_text

    def test_nested_class_chain_method(self, rw, tmp_path):
        original = """
        class Outer:
            class Inner:
                def deep(self):
                    return None
        """
        new_code = textwrap.dedent("""
                def deep(self):
                    return True
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_method(name, "Outer.Inner", "deep", new_code)
        assert "return True" in result.new_text


# --------------------------------------------------------------------------- #
# Guardrails — not-found raises ValueError
# --------------------------------------------------------------------------- #
class TestGuardrails:
    def test_function_not_found_raises(self, rw, tmp_path):
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        with pytest.raises(ValueError, match="Function not found: missing"):
            rw.replace_function("sample.py", "missing", "def missing():\n    pass\n")

    def test_class_not_found_raises(self, rw, tmp_path):
        _write(tmp_path, "sample.py", "class Foo:\n    pass\n")
        with pytest.raises(ValueError, match="Class not found: Missing"):
            rw.replace_class("sample.py", "Missing", "class Missing:\n    pass\n")

    def test_method_not_found_raises(self, rw, tmp_path):
        _write(tmp_path, "sample.py", "class Foo:\n    pass\n")
        with pytest.raises(ValueError, match=r"Method not found: Foo\.absent"):
            rw.replace_method("sample.py", "Foo", "absent", "def absent(self):\n    pass\n")

    def test_method_class_chain_not_found_raises(self, rw, tmp_path):
        _write(tmp_path, "sample.py", "class Foo:\n    pass\n")
        with pytest.raises(ValueError, match=r"Class not found: Ghost"):
            rw.replace_method("sample.py", "Ghost.Sub", "m", "def m(self):\n    pass\n")

    def test_oversized_file_refused(self, rw, tmp_path):
        big = tmp_path / "big.py"
        with open(big, "wb") as f:
            f.truncate(64 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match="too large"):
            rw.replace_function("big.py", "f", "def f():\n    pass\n")


# --------------------------------------------------------------------------- #
# generate_patch — git-header + hunk
# --------------------------------------------------------------------------- #
class TestGeneratePatch:
    def test_patch_has_git_header_and_hunk(self, rw, tmp_path):
        original = """
        def f():
            return 1
        """
        new_code = textwrap.dedent("""
        def f():
            return 2
        """)
        name = _write(tmp_path, "sample.py", original)
        result = rw.replace_function(name, "f", new_code)
        patch = rw.generate_patch(name, result)
        assert patch.startswith("diff --git a/sample.py b/sample.py\n")
        assert "@@" in patch
        assert "-    return 1" in patch
        assert "+    return 2" in patch
