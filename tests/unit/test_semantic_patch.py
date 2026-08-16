"""Regression tests for SemanticPatchEngine (external_llm/semantic_patch.py).

Primary target: SP-B1 — decorated-symbol replacement used `node.lineno - 1`
which points at the `def`/`class` line, leaving the original decorators in
`lines[:start]` while `new_code` re-introduced them, producing a doubled
decorator list (silently wrong: `@property`/`@lru_cache`/side-effecting
decorators applied twice). Coverage was 0/20 branches before this file.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from external_llm.semantic_patch import SemanticPatchEngine


@pytest.fixture
def engine(tmp_path):
    return SemanticPatchEngine(str(tmp_path))


def _write(tmp_path, name: str, source: str) -> str:
    (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return name


def _apply(engine, tmp_path, name: str, original: str, new_code: str):
    fp = _write(tmp_path, name, original)
    return engine.apply_semantic_patch(file_path=fp, new_code=textwrap.dedent(new_code))


# ---------------------------------------------------------------------------
# SP-B1: decorated-symbol replacement must not duplicate decorators
# ---------------------------------------------------------------------------

class TestDecoratorBoundary:
    def test_decorated_function_no_duplicate(self, engine, tmp_path):
        original = """
        @decorator_a
        @decorator_b
        def foo(x):
            return x + 1
        """
        new_code = """
        @decorator_a
        @decorator_b
        def foo(x):
            return x * 100
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.new_text.count("@decorator") == 2  # was 4 (SP-B1)

    def test_start_line_includes_decorators(self, engine, tmp_path):
        # Write source directly (no leading newline) so the decorator sits on
        # line index 0 and the start_line invariant is exact.
        source = "@cache\ndef foo(x):\n    return x\n"
        (tmp_path / "sample.py").write_text(source, encoding="utf-8")
        new_code = "@cache\ndef foo(x):\n    return x + 1\n"
        result = engine.apply_semantic_patch(file_path="sample.py", new_code=new_code)
        assert result is not None
        # start_line (0-indexed) must point at the decorator, not the `def` line.
        assert result.start_line == 0
        assert result.end_line == 3  # `    return x` (last body line)
        # The replaced span begins with the decorator — no duplication, no drift.
        assert result.new_text.splitlines()[0].strip() == "@cache"
        assert result.new_text.count("@cache") == 1

    def test_decorated_class_no_duplicate(self, engine, tmp_path):
        original = """
        @dataclass_like
        class Foo:
            x: int = 1
        """
        new_code = """
        @dataclass_like
        class Foo:
            x: int = 2
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.new_text.count("@dataclass_like") == 1  # was 2

    def test_async_decorated_function_no_duplicate(self, engine, tmp_path):
        original = """
        @retry
        async def foo(x):
            return x
        """
        new_code = """
        @retry
        async def foo(x):
            return x + 1
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.new_text.count("@retry") == 1  # was 2

    def test_original_decorator_dropped_when_new_code_omits_it(self, engine, tmp_path):
        # new_code deliberately has no decorator -> the rewrite intent is to
        # remove it. Decorators must NOT survive from the original.
        original = """
        @cache
        def foo(x):
            return x
        """
        new_code = """
        def foo(x):
            return x + 1
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert "@cache" not in result.new_text

    def test_new_decorator_added_when_original_had_none(self, engine, tmp_path):
        original = """
        def foo(x):
            return x
        """
        new_code = """
        @cache
        def foo(x):
            return x + 1
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.new_text.count("@cache") == 1

    def test_decorator_with_arguments_preserved_once(self, engine, tmp_path):
        original = """
        @app.route("/foo")
        def foo():
            return "foo"
        """
        new_code = """
        @app.route("/foo")
        def foo():
            return "bar"
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.new_text.count('@app.route("/foo")') == 1  # was 2

    def test_full_file_remains_syntactically_valid(self, engine, tmp_path):
        original = """
        const = 1

        @cache
        def foo(x):
            return x

        OTHER = 2
        """
        new_code = """
        @cache
        def foo(x):
            return x + 1
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        # Surrounding top-level symbols must be preserved verbatim.
        assert "const = 1" in result.new_text
        assert "OTHER = 2" in result.new_text
        assert result.new_text.count("@cache") == 1
        # And the whole file must still parse.
        ast.parse(result.new_text)


# ---------------------------------------------------------------------------
# Regression guards: plain (undecorated) behavior unchanged
# ---------------------------------------------------------------------------

class TestPlainSymbolReplacement:
    def test_plain_function_replaced(self, engine, tmp_path):
        original = """
        def foo(x):
            return x + 1
        """
        new_code = """
        def foo(x):
            return x * 100
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert "return x * 100" in result.new_text
        assert "return x + 1" not in result.new_text

    def test_plain_class_replaced(self, engine, tmp_path):
        original = """
        class Foo:
            x = 1
        """
        new_code = """
        class Foo:
            x = 2
            y = 3
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert "y = 3" in result.new_text
        assert result.kind == "class"

    def test_async_function_kind(self, engine, tmp_path):
        original = """
        async def foo(x):
            return x
        """
        new_code = """
        async def foo(x):
            return x + 1
        """
        result = _apply(engine, tmp_path, "sample.py", original, new_code)
        assert result is not None
        assert result.kind == "async_function"


# ---------------------------------------------------------------------------
# apply_semantic_patch guard rails (branch coverage)
# ---------------------------------------------------------------------------

class TestApplyGuardRails:
    def test_empty_new_code_returns_none(self, engine, tmp_path):
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        assert engine.apply_semantic_patch(file_path="sample.py", new_code="") is None

    def test_whitespace_only_new_code_returns_none(self, engine, tmp_path):
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        assert engine.apply_semantic_patch(file_path="sample.py", new_code="   \n  ") is None

    def test_non_def_class_body_returns_none(self, engine, tmp_path):
        # body[0] is an assignment, not def/class/async def -> fall-through None
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        result = engine.apply_semantic_patch(
            file_path="sample.py", new_code="X = 1\n"
        )
        assert result is None

    def test_syntax_error_new_code_returns_none(self, engine, tmp_path):
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        # unparseable -> ast.parse raises inside try -> None
        assert engine.apply_semantic_patch(
            file_path="sample.py", new_code="def (:\n"
        ) is None

    def test_symbol_not_found_returns_none(self, engine, tmp_path):
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        result = engine.apply_semantic_patch(
            file_path="sample.py", new_code="def bar():\n    pass\n"
        )
        assert result is None

    def test_class_not_found_returns_none(self, engine, tmp_path):
        _write(tmp_path, "sample.py", "class Foo:\n    pass\n")
        result = engine.apply_semantic_patch(
            file_path="sample.py", new_code="class Bar:\n    pass\n"
        )
        assert result is None

    def test_comment_only_new_code_returns_none(self, engine, tmp_path):
        # Non-empty text that parses to an empty module body -> guard at line 66.
        _write(tmp_path, "sample.py", "def foo():\n    pass\n")
        result = engine.apply_semantic_patch(
            file_path="sample.py", new_code="# just a comment\n"
        )
        assert result is None

    def test_missing_target_file_returns_none(self, engine, tmp_path):
        # _load_ast raises (FileNotFoundError) inside try -> None
        assert engine.apply_semantic_patch(
            file_path="does_not_exist.py", new_code="def foo():\n    pass\n"
        ) is None


# ---------------------------------------------------------------------------
# generate_patch
# ---------------------------------------------------------------------------

class TestGeneratePatch:
    def test_patch_has_git_header_and_hunk(self, engine, tmp_path):
        original = """
        def foo():
            return 1
        """
        new_code = """
        def foo():
            return 2
        """
        result = _apply(engine, tmp_path, "mod.py", original, new_code)
        patch = engine.generate_patch("mod.py", result)
        assert "diff --git a/mod.py b/mod.py" in patch
        assert "--- a/mod.py" in patch
        assert "+++ b/mod.py" in patch
        assert "@@" in patch

    def test_no_change_yields_header_only(self, engine, tmp_path):
        # Source on disk and new_code must be byte-identical for a true no-op;
        # avoid textwrap.dedent's leading-newline artifact.
        body = "def foo():\n    return 1\n"
        (tmp_path / "mod.py").write_text(body, encoding="utf-8")
        result = engine.apply_semantic_patch(file_path="mod.py", new_code=body)
        patch = engine.generate_patch("mod.py", result)
        # No content diff -> difflib emits nothing -> only the git header line
        assert patch.startswith("diff --git")
        assert "@@" not in patch
