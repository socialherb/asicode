"""Tests for the standalone modify_symbol tool."""
import ast
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from external_llm.agent.symbol_modify_tool import (
    _apply_ast_precise,
    _apply_surgical_edit,
    _correct_full_block_body_drift,
    _correct_indent_drift,
    _find_symbol_ast_node,
    _find_symbol_line_range,
    _find_symbol_range_via_treesitter,
    _looks_like_full_symbol_block,
    _realign_dedented_leading_lines,
    modify_symbol,
)
from external_llm.common.atomic_io import atomic_write_text
from external_llm.common.indent_utils import min_indent


def _ts_grammar_available(lang: str) -> bool:
    """True when the tree-sitter binding for ``lang`` is installed."""
    try:
        from external_llm.languages.tree_sitter_utils import get_available_languages
        return lang in get_available_languages()
    except Exception:
        return False


# ── Helpers ─────────────────────────────────────────────────────────────────

SAMPLE_SOURCE = textwrap.dedent("""\
    import os
    import sys

    class Greeter:
        \"\"\"A simple greeter.\"\"\"
        def __init__(self, name: str):
            self.name = name

        def greet(self) -> str:
            \"\"\"Return a greeting.\"\"\"
            return f"Hello, {self.name}!"

        def farewell(self) -> str:
            return f"Goodbye, {self.name}!"
""")


def _write_temp_file(content: str, suffix: str = ".py") -> str:
    """Write content to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


# ── Tests for _looks_like_full_symbol_block ─────────────────────────────────

class TestLooksLikeFullSymbolBlock:
    def test_full_function_def(self):
        assert _looks_like_full_symbol_block("def foo():\n    pass\n")

    def test_full_class_def(self):
        assert _looks_like_full_symbol_block("class Foo:\n    pass\n")

    def test_body_only(self):
        assert not _looks_like_full_symbol_block("    pass\n")

    def test_decorator_then_def(self):
        assert _looks_like_full_symbol_block("@property\ndef foo(self):\n    return 42\n")

    def test_empty_string(self):
        assert not _looks_like_full_symbol_block("")

    def test_blank_lines_first(self):
        assert _looks_like_full_symbol_block("\n\n\ndef foo():\n    pass\n")


# ── Tests for _realign_dedented_leading_lines ───────────────────────────────

class TestRealignDedentedLeadingLines:
    def test_dedented_decorator_realigned_to_def(self):
        code = "@staticmethod\n    def foo(self):\n        return 1\n"
        fixed = _realign_dedented_leading_lines(code)
        assert fixed == "    @staticmethod\n    def foo(self):\n        return 1\n"

    def test_dedented_comment_realigned_to_def(self):
        code = "# note\n    def foo(self):\n        return 1\n"
        fixed = _realign_dedented_leading_lines(code)
        assert fixed.startswith("    # note\n    def foo")

    def test_consistent_block_untouched(self):
        code = "    @staticmethod\n    def foo(self):\n        return 1\n"
        assert _realign_dedented_leading_lines(code) == code

    def test_def_first_line_untouched(self):
        code = "def foo():\n    return 1\n"
        assert _realign_dedented_leading_lines(code) == code

    def test_deeper_leading_lines_untouched(self):
        # Multi-line decorator continuation deeper than the def — leave as is.
        code = "    @retry(\n        max=3,\n    )\n    def foo(self):\n        return 1\n"
        assert _realign_dedented_leading_lines(code) == code


# ── Tests for _find_symbol_ast_node ─────────────────────────────────────────

class TestFindSymbolAstNode:
    def test_find_top_level_function(self):
        source = "def foo(): pass\ndef bar(): pass\n"
        node = _find_symbol_ast_node(source, "foo")
        assert node is not None
        assert node.name == "foo"

    def test_find_class_method(self):
        source = "class X:\n    def method(self): pass\n"
        node = _find_symbol_ast_node(source, "X.method")
        assert node is not None
        assert node.name == "method"

    def test_symbol_not_found(self):
        node = _find_symbol_ast_node("x = 1\n", "nonexistent")
        assert node is None

    def test_invalid_syntax_returns_none(self):
        node = _find_symbol_ast_node("def foo(:\n", "foo")
        assert node is None


# ── Tests for _apply_ast_precise ────────────────────────────────────────────

class TestApplyAstPrecise:
    def test_full_block_replacement(self):
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = "    def foo(self):\n        return 42\n"
        diff, mode = _apply_ast_precise(source, "test.py", "X.foo", new_code)
        assert diff is not None
        assert "python_full_block" in mode
        assert "42" in diff

    def test_body_only_replacement(self):
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = "        return 42\n"
        diff, mode = _apply_ast_precise(source, "test.py", "X.foo", new_code)
        assert diff is not None
        assert "python_body_only" in mode

    def test_no_change_returns_none(self):
        source = "def foo(): return 1\n"
        diff, mode = _apply_ast_precise(source, "test.py", "foo", "def foo(): return 1\n")
        assert diff is None
        assert mode == "no_change"

    def test_symbol_not_found(self):
        diff, mode = _apply_ast_precise("x = 1\n", "test.py", "nonexistent", "def x(): pass\n")
        assert diff is None
        assert "skipped_symbol_not_found" in mode

    def test_invalid_syntax_source(self):
        diff, mode = _apply_ast_precise("def foo(:\n", "test.py", "foo", "def foo(): pass\n")
        assert diff is None
        assert "skipped_ast_error" in mode


# ── Tests for modify_symbol (end-to-end) ────────────────────────────────────

class TestModifySymbol:
    def test_full_block(self):
        path = _write_temp_file(SAMPLE_SOURCE)
        try:
            new_greet = '''    def greet(self) -> str:
        \"\"\"Modified greeting.\"\"\"
        return f"Hi, {self.name}!"
'''
            success, diff, new_content = modify_symbol(path, "Greeter.greet", new_greet)
            assert success, f"Failed: {diff}"
            assert "Hi, " in new_content
            assert "Hello" not in new_content
            assert "def greet" in new_content
        finally:
            os.unlink(path)

    def test_body_only(self):
        path = _write_temp_file(SAMPLE_SOURCE)
        try:
            new_greet_body = '''        \"\"\"Modified greeting.\"\"\"
        return f"Hey, {self.name}!"
'''
            success, diff, new_content = modify_symbol(path, "Greeter.greet", new_greet_body)
            assert success, f"Failed: {diff}"
            assert "Hey, " in new_content
            assert "Hello" not in new_content
            assert "def greet" in new_content  # signature preserved
        finally:
            os.unlink(path)

    def test_dedented_decorator_first_line(self):
        """Regression: a full block whose first line (decorator) lost its
        indentation while the rest kept the original method depth.

        Every strategy anchors the block on its first line, so without
        normalization the def lands one level deeper than its decorator
        (IndentationError: unexpected indent) and all three strategies fail.
        """
        source = textwrap.dedent('''\
            class Client:
                @staticmethod
                def mark(messages):
                    """Old docstring."""
                    if not messages:
                        return
                    messages[-1]["cached"] = True
        ''')
        path = _write_temp_file(source)
        try:
            # First line at col 0, remaining lines at original class-method depth.
            new_code = (
                '@staticmethod\n'
                '    def mark(messages):\n'
                '        """New docstring."""\n'
                '        if not messages:\n'
                '            return None\n'
                '        messages[-1]["cached"] = "ephemeral"\n'
            )
            success, diff, new_content = modify_symbol(path, "Client.mark", new_code)
            assert success, f"Failed: {diff}"
            ast.parse(new_content)
            assert "    @staticmethod\n" in new_content
            assert "    def mark(messages):\n" in new_content
            assert '"ephemeral"' in new_content
        finally:
            os.unlink(path)

    def test_context_lines_not_corrupted_by_diff_roundtrip(self):
        """Regression: unchanged context lines must keep their exact indentation.

        modify_symbol builds a unified diff then re-applies it via
        _apply_diff_to_source. A bug there appended the diff's leading-space
        context marker to every unchanged line, corrupting indentation and
        producing "unindent does not match any outer indentation level".
        """
        path = _write_temp_file(SAMPLE_SOURCE)
        try:
            new_greet = '''    def greet(self) -> str:
        return f"Hi, {self.name}!"
'''
            success, diff, new_content = modify_symbol(path, "Greeter.greet", new_greet)
            assert success, f"Failed: {diff}"
            # Result must still parse — no spurious indentation drift.
            ast.parse(new_content)
            # Context lines surrounding the edit must be byte-identical (not
            # shifted right by a stray leading space).
            assert "import os\n" in new_content
            assert "class Greeter:\n" in new_content
            assert "    def __init__(self, name: str):\n" in new_content
            assert "        self.name = name\n" in new_content
            assert "    def farewell(self) -> str:\n" in new_content
            assert "        return f\"Goodbye, {self.name}!\"\n" in new_content
            # And the file on disk matches what was returned.
            assert Path(path).read_text() == new_content
        finally:
            os.unlink(path)

    def test_broken_indentation_does_not_write_invalid_python(self):
        """Regression: when no strategy can produce valid Python, the file must be
        left untouched and a clear error returned (not written then rolled back)."""
        source = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def handler(self, x):
                        if x:
                            return 1
                        return 0
        """)
        path = _write_temp_file(source)
        try:
            original = Path(path).read_text()
            # Internally-inconsistent indentation that cannot be re-indented to
            # valid Python by any strategy.
            bad_code = "return 1\n   return 2\n  return 3"
            success, msg, _ = modify_symbol(path, "Inner.handler", bad_code)
            assert not success
            assert "apply_patch" in msg
            # File untouched and still valid.
            assert Path(path).read_text() == original
            ast.parse(Path(path).read_text())
        finally:
            os.unlink(path)

    def test_indentation_auto_correction(self):
        source = textwrap.dedent("""\
            class X:
                def foo(self):
                    return 1
        """)
        path = _write_temp_file(source)
        try:
            # Intentionally wrong indentation (6 spaces instead of 8)
            new_code = "      def foo(self):\n          return 42\n"
            success, diff, new_content = modify_symbol(path, "X.foo", new_code)
            assert success, f"Failed: {diff}"
            # Check indentation was auto-corrected
            for line in new_content.splitlines():
                if "return 42" in line:
                    assert line.startswith("        "), f"Wrong indent: {line[:20]!r}"
                    break
        finally:
            os.unlink(path)

    def test_indent_unit_normalized_to_file_2space(self):
        """LLM emits 2-space body into a 4-space file -> body normalized to 4-space.

        compile() passes either way, so this is the silent inconsistency the
        auto-correction's char-count shift used to leave behind.
        """
        source = textwrap.dedent("""\
            class X:
                def foo(self, n=0):
                    a = 1
                    if n:
                        b = 2
                    c = 3

                def bar(self):
                    pass
        """)
        path = _write_temp_file(source)
        try:
            # Full block emitted with a 2-space indent unit.
            new_code = (
                "def foo(self, n=0):\n"
                "  a = 100\n"
                "  if n:\n"
                "    b = 2\n"
                "  c = 3\n"
            )
            success, diff, new_content = modify_symbol(path, "X.foo", new_code)
            assert success, f"Failed: {diff}"
            lines = {ln.strip(): ln for ln in new_content.splitlines() if ln.strip()}
            assert lines["a = 100"].startswith("        a"), repr(lines["a = 100"])
            assert lines["b = 2"].startswith("            b"), repr(lines["b = 2"])
            assert lines["c = 3"].startswith("        c"), repr(lines["c = 3"])
            # Sibling method untouched -> file stays internally consistent.
            assert "        pass" in new_content
        finally:
            os.unlink(path)

    def test_indent_unit_normalized_to_file_tabs(self):
        """LLM emits tab-indented body into a 4-space file -> normalized to spaces."""
        source = textwrap.dedent("""\
            class X:
                def foo(self, n=0):
                    a = 1
                    c = 3
        """)
        path = _write_temp_file(source)
        try:
            new_code = "def foo(self, n=0):\n\ta = 100\n\tif n:\n\t\tb = 2\n\tc = 3\n"
            success, diff, new_content = modify_symbol(path, "X.foo", new_code)
            assert success, f"Failed: {diff}"
            assert "\t" not in new_content, "tabs should be normalized to file's spaces"
            lines = {ln.strip(): ln for ln in new_content.splitlines() if ln.strip()}
            assert lines["a = 100"].startswith("        a"), repr(lines["a = 100"])
            assert lines["b = 2"].startswith("            b"), repr(lines["b = 2"])
        finally:
            os.unlink(path)

    def test_alignment_continuation_no_indent_explosion(self):
        """Paren-aligned continuation lines must not poison unit detection.

        Regression: a body-only block whose multi-line call aligns arguments to
        the open paren (e.g. column 27) made ``model_unit`` collapse to 1 via the
        leading-run GCD. The body was then re-indented by ``file_unit /
        model_unit`` ≈ ×4, exploding every line (116-space indents, 24/40 where
        12/16 were expected). Logical lines must remap cleanly while the aligned
        continuations shift with their owner, preserving the alignment.
        """
        source = textwrap.dedent("""\
            class Foo:
                def bar(self, args):
                    \"\"\"Doc.\"\"\"
                    old = 1
                    return old
        """)
        path = _write_temp_file(source)
        try:
            new_code = (
                '"""Doc."""\n'
                'result = self._make_result(ok=False,\n'
                '                           content="",\n'
                '                           error="pattern required")\n'
                'if not args:\n'
                '    return result\n'
                'for x in args:\n'
                '    if x > 0:\n'
                '        total += x\n'
                'return total\n'
            )
            success, diff, new_content = modify_symbol(path, "Foo.bar", new_code)
            assert success, f"Failed: {diff}"
            import ast as _ast
            _ast.parse(new_content)  # must be valid Python
            lines = {ln.strip(): ln for ln in new_content.splitlines() if ln.strip()}
            # Logical body lines normalized to the method body indent (8) / nesting.
            assert lines['if not args:'].startswith("        if"), repr(lines['if not args:'])
            assert lines['return result'].startswith("            return"), repr(lines['return result'])
            assert lines['total += x'].startswith("                total"), repr(lines['total += x'])
            # Aligned continuation tracks the open paren of its owner line (col 35),
            # never blown up to a 100+ space indent.
            content_line = lines['content="",']
            assert content_line.startswith(" " * 35 + 'content'), repr(content_line)
            assert (len(content_line) - len(content_line.lstrip())) < 40, repr(content_line)
        finally:
            os.unlink(path)

    def test_matching_unit_preserves_hanging_indent(self):
        """4-space LLM code into a 4-space file keeps PEP8 hanging-indent verbatim."""
        source = textwrap.dedent("""\
            class X:
                def foo(
                        self,
                        a: int = 0,
                ) -> None:
                    self.a = a
        """)
        path = _write_temp_file(source)
        try:
            new_code = (
                "def foo(\n"
                "        self,\n"
                "        a: int = 0,\n"
                "        b: int = 0,\n"
                ") -> None:\n"
                "    self.a = a\n"
                "    self.b = b\n"
            )
            success, diff, new_content = modify_symbol(path, "X.foo", new_code)
            assert success, f"Failed: {diff}"
            lines = {ln.strip(): ln for ln in new_content.splitlines() if ln.strip()}
            # def at 4, hanging args at 12 (8 relative), body at 8 — all preserved.
            assert lines["def foo("].startswith("    def"), repr(lines["def foo("])
            assert lines["b: int = 0,"].startswith("            b"), repr(lines["b: int = 0,"])
            assert lines["self.b = b"].startswith("        self.b"), repr(lines["self.b = b"])
        finally:
            os.unlink(path)

    def test_redundant_import_stripping(self):
        source = textwrap.dedent("""\
            import os
            def foo():
                return 1
        """)
        new_code = textwrap.dedent("""\
            import os
            def foo():
                return 42
        """)
        path = _write_temp_file(source)
        try:
            success, diff, new_content = modify_symbol(path, "foo", new_code)
            assert success, f"Failed: {diff}"
            # The inline import 'import os' in new_code should be stripped
            # since it's already at module level
            assert "return 42" in new_content
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        success, diff, _ = modify_symbol("/nonexistent/file.py", "foo", "def foo(): pass\n")
        assert not success
        assert "not found" in diff.lower()

    def test_repo_root_path_resolution(self):
        path = _write_temp_file(SAMPLE_SOURCE)
        try:
            rel_path = os.path.basename(path)
            repo_root = os.path.dirname(path)
            success, diff, _new_content = modify_symbol(
                rel_path, "Greeter.greet",
                "    def greet(self):\n        return \"ok\"\n",
                repo_root=repo_root
            )
            assert success, f"Failed: {diff}"
        finally:
            os.unlink(path)

    def test_non_python_file(self):
        source = textwrap.dedent("""\
            function greet(name) {
                return "Hello, " + name + "!";
            }
            function farewell(name) {
                return "Goodbye, " + name + "!";
            }
        """)
        path = _write_temp_file(source, suffix=".js")
        try:
            success, diff, new_content = modify_symbol(path, "greet",
                "function greet(name) {\n    return \"Hi, \" + name + \"!\";\n}\n")
            assert success, f"Failed: {diff}"
            assert "Hi, " in new_content
        finally:
            os.unlink(path)

    def test_top_level_function_replacement(self):
        source = textwrap.dedent("""\
            import sys

            def helper():
                return 1

            def main():
                return helper()
        """)
        path = _write_temp_file(source)
        try:
            new_helper = "def helper():\n    return 42\n"
            success, diff, new_content = modify_symbol(path, "helper", new_helper)
            assert success, f"Failed: {diff}"
            assert "return 42" in new_content
            assert "return 1" not in new_content
        finally:
            os.unlink(path)


# ── Tests for _find_symbol_line_range ──────────────────────────────────────

class TestFindSymbolLineRange:
    def test_python_function(self):
        source = "def foo():\n    pass\n"
        r = _find_symbol_line_range(source, "foo", "test.py")
        assert r is not None
        assert r[0] == 0  # 0-indexed start
        assert r[1] == 2  # exclusive end (pass at line 1, end at line 2)

    def test_non_python_heuristic(self):
        source = "function foo() {\n  return 1;\n}\n"
        r = _find_symbol_line_range(source, "foo", "test.js")
        assert r is not None
        assert r[0] == 0
        # The heuristic detects symbol end by indentation change.
        # '}' at line 2 has less indentation than '  return 1;' so it may vary.
        assert r[1] in (2, 3)

    def test_nonexistent_symbol(self):
        r = _find_symbol_line_range("x = 1\n", "nonexistent", "test.py")
        assert r is None


# ── Provider-based symbol location (modifiers / annotations) ────────────────
# Regression: _find_symbol_line_range previously used a hardcoded prefix list
# ("fun ", "func ", "def ", ...) and required the stripped line to *start*
# with one. This silently failed for any declaration carrying a leading
# modifier keyword (private/override/suspend/internal/public/...), which is
# the common case in Kotlin/Java/Go/Rust/etc. The fix routes non-Python
# lookup through the per-language provider patterns (typed policy) instead.

class TestFindSymbolProviderModifiers:
    """Symbol location must find declarations with leading modifiers."""

    # (source, symbol, file, expected_start_line_index, desc)
    CASES = [
        # Kotlin — every common modifier form
        ("    private fun allocateUniqueNames(): List<String> {\n        x()\n    }\n",
         "allocateUniqueNames", "Engine.kt", 0, "private fun"),
        ("    override fun isRecordingNow(): Boolean {\n        return true\n    }\n",
         "isRecordingNow", "Engine.kt", 0, "override fun"),
        ("    suspend fun fetchAsync(): String {\n        return \"\"\n    }\n",
         "fetchAsync", "Engine.kt", 0, "suspend fun"),
        ("    internal fun helper(): Int {\n        return 1\n    }\n",
         "helper", "Engine.kt", 0, "internal fun"),
        # Bare fun must still work (no regression)
        ("    fun startRecording() {\n        doWork()\n    }\n",
         "startRecording", "Engine.kt", 0, "bare fun"),
        # Java — visibility + override modifiers
        ("    private void doWork() {\n        run();\n    }\n",
         "doWork", "Foo.java", 0, "private void"),
        ("    public final String getName() {\n        return name;\n    }\n",
         "getName", "Foo.java", 0, "public final method"),
        # Go — method on receiver type
        ("func (s *Server) Start() {\n    s.listen()\n}\n",
         "Start", "server.go", 0, "Go receiver method"),
        # Rust — pub/visibility modifiers
        ("pub fn compute(x: i32) -> i32 {\n    x + 1\n}\n",
         "compute", "lib.rs", 0, "pub fn"),
        ("    pub(crate) fn helper() {\n        todo!()\n    }\n",
         "helper", "lib.rs", 0, "pub(crate) fn"),
    ]

    @pytest.mark.parametrize("source,symbol,fname,expect_start,desc", CASES,
                             ids=[c[4] for c in CASES])
    def test_modifier_declarations_found(self, source, symbol, fname, expect_start, desc):
        r = _find_symbol_line_range(source, symbol, fname)
        assert r is not None, f"FAILED to locate '{symbol}' ({desc}) — modifier-form regression"
        assert r[0] == expect_start

    def test_kotlin_class_with_data_modifier(self):
        # Kotlin data class — 'data' modifier before 'class'
        source = "data class Point(val x: Int, val y: Int)\n"
        r = _find_symbol_line_range(source, "Point", "Point.kt")
        assert r is not None, "data class must be locatable via provider pattern"

    def test_call_site_does_not_match(self):
        # A call site must NOT be mistaken for a definition — provider
        # regexes anchor on the declaration keyword.
        source = "fun caller() {\n    allocateUniqueNames()\n}\n"
        r = _find_symbol_line_range(source, "allocateUniqueNames", "Engine.kt")
        assert r is None, "call site should not match a declaration pattern"


class TestFindSymbolRangeViaTreeSitter:
    """The AST path yields an exact (start, end) straight from the parse — no
    brace-balancing. Runs only when the relevant tree-sitter grammar is
    installed; when it is not, _find_symbol_line_range transparently falls back
    to the regex path (covered by TestFindSymbolProviderModifiers above).
    """

    def test_go_method_range_from_ast(self):
        if not _ts_grammar_available("go"):
            pytest.skip("tree-sitter-go not installed")
        src = textwrap.dedent("""\
            package main

            func (s *Server) Start() error {
                return nil
            }
        """)
        # Start is on line 3 (1-indexed); closing brace on line 5.
        # AST contract: 0-indexed start, exclusive end.
        r = _find_symbol_range_via_treesitter(src, "Start", "server.go")
        assert r == (2, 5)

    def test_go_function_range_includes_full_body(self):
        if not _ts_grammar_available("go"):
            pytest.skip("tree-sitter-go not installed")
        src = textwrap.dedent("""\
            package main

            func NewServer(port int) *Server {
                return &Server{Port: port}
            }
        """)
        r = _find_symbol_range_via_treesitter(src, "NewServer", "server.go")
        assert r is not None
        lines = src.splitlines()
        # The slice must be the complete function: signature line through brace.
        assert lines[r[0]].lstrip().startswith("func NewServer")
        assert lines[r[1] - 1].strip() == "}"

    def test_kotlin_returns_none_when_grammar_missing(self):
        # Kotlin grammar is not shipped by default; the helper must return None
        # (NOT raise) so _find_symbol_line_range falls back to the regex path.
        if _ts_grammar_available("kotlin"):
            pytest.skip("tree-sitter-kotlin IS installed — fallback path N/A")
        src = 'class Engine {\n    fun go(): String = "x"\n}\n'
        assert _find_symbol_range_via_treesitter(src, "go", "Engine.kt") is None

    def test_python_skipped(self):
        # Python is handled by the AST path, never by the tree-sitter helper.
        src = "def go():\n    return 1\n"
        assert _find_symbol_range_via_treesitter(src, "go", "m.py") is None


class TestModifySymbolNonPythonEndToEnd:
    """Full modify_symbol write path for modifier-bearing declarations."""

    def _run(self, src: str, symbol: str, fname: str, new_body: str):
        d = tempfile.mkdtemp()
        p = os.path.join(d, fname)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            return modify_symbol(p, symbol, new_body, repo_root=d)
        finally:
            os.unlink(p)
            os.rmdir(d)

    def test_kotlin_private_fun_modified(self):
        src = textwrap.dedent("""\
            class Engine {
                private fun allocateUniqueNames(p: String): List<String> {
                    return emptyList()
                }
            }""")
        new = textwrap.dedent("""\
            private fun allocateUniqueNames(p: String): List<String> {
                return mutableListOf()
            }""")
        ok, diff, nc = self._run(src, "allocateUniqueNames", "Engine.kt", new)
        assert ok, f"modify_symbol failed: {diff}"
        assert "mutableListOf" in nc
        assert "emptyList" not in nc

    def test_java_override_method_modified(self):
        src = textwrap.dedent("""\
            class Foo extends Base {
                @Override
                public String toString() {
                    return "old";
                }
            }""")
        new = textwrap.dedent("""\
            public String toString() {
                return "new";
            }""")
        ok, diff, nc = self._run(src, "toString", "Foo.java", new)
        assert ok, f"modify_symbol failed: {diff}"
        assert '"new"' in nc

    def test_go_method_full_block_modified_via_ast(self):
        """Go uses the tree-sitter range (grammar installed) — full-block edit."""
        if not _ts_grammar_available("go"):
            pytest.skip("tree-sitter-go not installed")
        src = textwrap.dedent("""\
            package main

            type Server struct {
                Port int
            }

            func (s *Server) Start() error {
                return nil
            }
        """)
        new = textwrap.dedent("""\
            func (s *Server) Start() error {
                return errors.New("started")
            }
        """)
        ok, diff, nc = self._run(src, "Start", "server.go", new)
        assert ok, f"modify_symbol failed: {diff}"
        assert 'errors.New("started")' in nc
        assert "return nil" not in nc


# ── Defense-1: body-only indent-drift correction ────────────────────────────

class TestMinIndent:
    def test_uniform_indent(self):
        assert min_indent(["    a", "    b", "    c"]) == 4

    def test_mixed_indent_takes_min(self):
        assert min_indent(["    a", "        b", "    c"]) == 4

    def test_blank_lines_ignored(self):
        assert min_indent(["", "    a", "", "    b"]) == 4

    def test_empty(self):
        assert min_indent([]) == 0

    def test_all_blank(self):
        assert min_indent(["", "  "]) == 0


class TestCorrectIndentDrift:
    def test_no_drift_is_noop(self):
        lines = ["    x = 1", "    if True:", "        y = 2", "    return x"]
        assert _correct_indent_drift(lines, "    ", "foo") == lines

    def test_one_level_drift_corrected(self):
        drifted = ["        x = 1", "        if True:", "            y = 2", "        return x"]
        out = _correct_indent_drift(drifted, "    ", "foo")
        assert out == ["    x = 1", "    if True:", "        y = 2", "    return x"]

    def test_blank_lines_preserved(self):
        drifted = ["        x = 1", "", "        return x"]
        out = _correct_indent_drift(drifted, "    ", "foo")
        assert out == ["    x = 1", "", "    return x"]

    def test_under_indent_not_corrected(self):
        # under-indent is a parse error caught by the downstream compile() guard;
        # _correct_indent_drift must not mask it.
        lines = ["  x = 1", "  return x"]
        assert _correct_indent_drift(lines, "    ", "foo") == lines

    def test_module_level_target_zero_noop(self):
        lines = ["x = 1"]
        assert _correct_indent_drift(lines, "", "mod") == lines

    def test_tab_drift_corrected(self):
        drifted = ["\t\t\tx = 1", "\t\t\treturn x"]
        out = _correct_indent_drift(drifted, "\t", "foo")
        assert out == ["\tx = 1", "\treturn x"]

    def test_nested_depth_preserved(self):
        # Critical: a drift correction must flatten nesting levels — nested
        # statements keep their RELATIVE depth, only the base shifts.
        drifted = ["        x = 1", "        for i in r:", "            j = i + 1", "        return x"]
        out = _correct_indent_drift(drifted, "    ", "foo")
        assert out == ["    x = 1", "    for i in r:", "        j = i + 1", "    return x"]

    def test_shallow_outlier_does_not_mask_drift(self):
        # Regression: a body where the MAJORITY of lines drifted one level deep
        # but a single line (e.g. a docstring) sits at the target depth. The
        # min-indent diagnostic used to read the shallow outlier, report "no
        # drift", and leave the bulk of the body over-indented — relying on the
        # downstream compile() to reject the whole edit. The mode-based
        # diagnostic must fire and correct the drifted majority while leaving
        # the shallow outlier untouched (not push it into under-indent).
        drifted = [
            '        """Read a symbol."""',           # 8-space (target) — shallow outlier
            '            name = args.get("name")',    # 12-space (drift)
            '            return self._make_result(name)',  # 12-space (drift)
        ]
        out = _correct_indent_drift(drifted, "        ", "Foo.bar")
        assert out == [
            '        """Read a symbol."""',           # outlier untouched (still 8)
            '        name = args.get("name")',        # corrected 12 -> 8
            '        return self._make_result(name)',  # corrected 12 -> 8
        ]

    def test_under_indent_outlier_not_overcorrected(self):
        # When the mode is at target but a single line is shallower, the
        # correction must NOT fire (mode == target → noop). Guards against the
        # inverse mistake: shifting an already-correct body because of a deep
        # outlier would under-indent the majority.
        lines = [
            "    x = 1",         # 4 (target)
            "    return x",      # 4 (target)
        ]
        out = _correct_indent_drift(lines, "    ", "foo")
        assert out == lines  # mode == target → noop

    def test_nested_function_body_not_flattened(self):
        # REGRESSION (2026-06-22): a body whose MAJORITY of lines sit inside a
        # NESTED def/for/if has a deep *mode* but a shallow *min*. The old
        # mode-based diagnostic mis-read the deep mode as drift and flattened
        # the nested body into a parse error. min == target (the outer
        # statements) and the block parses cleanly, so no correction must fire.
        nested = [
            "    def inner():",        # 4 (target) — outer logical stmt
            "        a = 1",           # 8 — body of inner (majority)
            "        b = 2",           # 8
            "        return result",   # 8
            "    return inner",        # 4 (target) — outer logical stmt
        ]
        out = _correct_indent_drift(nested, "    ", "foo")
        assert out == nested, f"nested body flattened: {out}"

    def test_deep_nested_control_flow_not_flattened(self):
        # Same regression class, exercised with control-flow headers instead
        # of a nested def. The majority of lines sit inside the for/if block.
        nested = [
            "    for item in items:",  # 4 (target)
            "        if item.ok:",     # 8
            "            process(item)",  # 12
            "            log(item)",     # 12
            "    return done",         # 4 (target)
        ]
        out = _correct_indent_drift(nested, "    ", "foo")
        assert out == nested, f"nested control-flow flattened: {out}"


class TestApplyAstPreciseDriftDefense:
    """End-to-end: body-only edits land at the symbol's base indent regardless
    of the model's indentation, and a drift warning is observable when the
    re-anchor under-shots.

    NOTE: ``_reindent_relative`` already re-anchors to ``body_indent`` in the
    common path (``base_prefix + extra`` where ``extra = rel`` is anchor-relative,
    so the model's absolute indent is factored out). Defense-1
    (``_correct_indent_drift``) is the SAFETY NET for edge cases where the
    re-anchor mis-fires. These integration tests therefore assert the OUTPUT
    contract (correct indent) directly; the drift-correction logic itself is
    unit-tested in ``TestCorrectIndentDrift`` above."""

    def test_drifted_body_only_lands_at_base_indent(self):
        source = "class Foo:\n    def bar(self):\n        x = 1\n        return x\n"
        # Model sent the body one level too deep (12 spaces instead of 8).
        drifted_body = "            x = 999\n            return x"
        diff, mode = _apply_ast_precise(source, "test.py", "Foo.bar", drifted_body)
        assert mode == "python_body_only"
        assert diff is not None
        added = _added_lines(diff)
        # The spliced line must sit at 8 spaces (the method's body indent),
        # NOT 12 (the drifted depth the model sent).
        assert any(_indent_of(ln) == 8 and "x = 999" in ln for ln in added), \
            f"drift not corrected, added lines: {added}"

    def test_correct_body_only_lands_at_base_indent(self):
        source = "class Foo:\n    def bar(self):\n        x = 1\n        return x\n"
        correct_body = "        x = 999\n        return x"
        diff, mode = _apply_ast_precise(source, "test.py", "Foo.bar", correct_body)
        assert mode == "python_body_only"
        added = _added_lines(diff)
        assert any(_indent_of(ln) == 8 and "x = 999" in ln for ln in added)

    def test_module_level_body_lands_at_base_indent(self):
        source = "def foo():\n    x = 1\n    return x\n"
        # Model sent the body dedented to col 0; file uses 4-space.
        drifted_body = "x = 999\nreturn x"
        diff, mode = _apply_ast_precise(source, "test.py", "foo", drifted_body)
        assert mode == "python_body_only"
        added = _added_lines(diff)
        assert any(_indent_of(ln) == 4 and "x = 999" in ln for ln in added), \
            f"normalization failed, added: {added}"

    def test_indent_drift_warning_is_observable(self, caplog):
        # The defense logs a WARNING whenever it applies a corrective shift.
        # _correct_indent_drift is exercised directly (the common path already
        # produces correct indent, so we drive the helper to assert the signal).
        from external_llm.agent.symbol_modify_tool import _correct_indent_drift
        drifted = ["            x = 1", "            return x"]
        with caplog.at_level("WARNING", logger="asicode.modify_symbol_tool"):
            _correct_indent_drift(drifted, "        ", "Foo.bar")
        assert any("indent drift" in r.message for r in caplog.records)

    def test_drifted_body_with_shallow_docstring_lands_at_base_indent(self):
        # Regression for the mode-vs-min bug: the model sent a body whose first
        # physical line (a docstring) sits at the correct base indent but whose
        # remaining statements drifted one level deep. The min-indent diagnostic
        # read the docstring's depth and reported "no drift", so the spliced
        # body had mixed 8/12-space indentation and the downstream compile()
        # rejected the WHOLE edit. The mode-based diagnostic must correct the
        # drifted majority and let the edit succeed.
        source = "class Foo:\n    def bar(self):\n        x = 1\n        return x\n"
        drifted_body = '        """New docstring."""\n            x = 999\n            return x'
        diff, mode = _apply_ast_precise(source, "test.py", "Foo.bar", drifted_body)
        assert mode == "python_body_only"
        assert diff is not None, "edit must succeed (previously failed via compile)"
        added = _added_lines(diff)
        # Both the docstring and the statements must land at 8-space indent.
        assert any(_indent_of(ln) == 8 and "docstring" in ln for ln in added), \
            f"docstring not at 8-space: {added}"
        assert any(_indent_of(ln) == 8 and "x = 999" in ln for ln in added), \
            f"drifted statement not corrected to 8-space: {added}"

    def test_nested_function_body_not_flattened(self):
        # REGRESSION (2026-06-22): a body whose MAJORITY of lines sit inside a
        # NESTED def has a deep *mode* but a shallow *min*. The old mode-based
        # diagnostic mis-read the deep mode as drift and flattened the nested
        # body into a parse error. End-to-end: a correctly-indented body-only
        # replacement containing a nested def must round-trip with nesting
        # preserved (12-space), NOT flattened to 8-space.
        source = (
            "class Foo:\n"
            "    def compute(self):\n"
            "        acc = []\n"
            "        def inner(v):\n"
            "            return v * 2\n"
            "        acc.append(inner(3))\n"
            "        return sum(acc)\n"
        )
        # Body-only replacement: does NOT start with a def/class (so it stays
        # in body-only mode), but contains a nested def whose body is the
        # majority of lines.
        new_body = (
            "        acc = []\n"
            "        def inner(v):\n"
            "            return v * 5\n"
            "        acc.append(inner(3))\n"
            "        return sum(acc)\n"
        )
        diff, mode = _apply_ast_precise(source, "test.py", "Foo.compute", new_body)
        assert mode == "python_body_only", f"unexpected mode: {mode}"
        assert diff is not None, "edit must succeed"
        added = _added_lines(diff)
        # The nested body line must land at 12-space (preserved nesting),
        # NOT 8-space (which would be the flattened regression).
        assert any(_indent_of(ln) == 12 and "v * 5" in ln for ln in added), \
            f"nested body flattened to 8-space: {added}"


# ── Defense-2: mode misclassification warning ───────────────────────────────

class TestModeMisclassificationWarning:
    """When a def/class statement reaches the body-only path (because the
    prefix heuristic failed to classify it), a warning is emitted so the
    mis-routing is observable rather than silently splicing a nested def."""

    def test_tab_variant_def_triggers_warning(self, caplog):
        source = "class Foo:\n    def bar(self):\n        x = 1\n"
        # 'def\tbar():' — _looks_like_full_symbol_block returns False (prefix
        # requires 'def ' with a space), so this enters body-only and starts
        # with a def statement: the misclassification defense fires.
        weird_def = "def\tbar():\n        pass\n"
        with caplog.at_level("WARNING", logger="asicode.modify_symbol_tool"):
            _apply_ast_precise(source, "test.py", "Foo.bar", weird_def)
        assert any("misclassification" in r.message for r in caplog.records)

    def test_normal_body_only_no_warning(self, caplog):
        source = "class Foo:\n    def bar(self):\n        x = 1\n"
        with caplog.at_level("WARNING", logger="asicode.modify_symbol_tool"):
            _apply_ast_precise(source, "test.py", "Foo.bar", "        x = 999\n")
        assert not any("misclassification" in r.message for r in caplog.records)


# ── Full-block body indent drift (silent over-indent regression) ─────────────
# Regression: the full-block path of _apply_ast_precise re-anchors the def line
# but, unlike the body-only path, did NOT validate the body against the symbol's
# original body indent. A body the model emitted one full indent unit deeper
# than its own def line was faithfully reproduced one level too deep. Python
# parses a uniformly-over-indented body as valid, so compile() let it through
# and the caller saw success with a silent one-level body drift.

class TestFullBlockBodyIndentDrift:
    def _body_indent(self, content: str, name: str) -> int:
        """col_offset of the first statement in `name`'s body."""
        tree = ast.parse(content)
        def walk(node):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name == name:
                    return ch.body[0].col_offset
                r = walk(ch)
                if r is not None:
                    return r
            return None
        return walk(tree)

    def _run(self, source: str, symbol: str, new_code: str) -> str:
        """Apply a full-block replacement to a temp file and return the result."""
        path = _write_temp_file(source)
        try:
            success, diff, new_content = modify_symbol(path, symbol, new_code)
            assert success, f"modify_symbol failed: {diff}"
            return new_content
        finally:
            os.unlink(path)

    def test_class_method_body_one_level_deep_is_corrected(self):
        # def is at the file's 4-space (matches original), but the body is
        # 12-space (one unit deeper than the expected 8).
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = "    def foo(self):\n            return 42\n"
        new_content = self._run(source, "X.foo", new_code)
        assert self._body_indent(new_content, "foo") == 8  # not 12

    def test_full_block_body_drift_two_units_corrected(self):
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = "    def foo(self):\n                return 42\n"  # 16-space body
        new_content = self._run(source, "X.foo", new_code)
        assert self._body_indent(new_content, "foo") == 8

    def test_full_block_correct_body_unchanged(self):
        # Body already at the right indent (8) must NOT be shifted.
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = "    def foo(self):\n        return 42\n"
        new_content = self._run(source, "X.foo", new_code)
        assert self._body_indent(new_content, "foo") == 8

    def test_full_block_nested_structure_preserved_after_correction(self):
        # Body one level deep AND containing a nested if: the relative profile
        # (the if-body being one level deeper than the return) must survive the
        # corrective shift — only the whole-body offset is reduced by one unit.
        source = "class X:\n    def foo(self):\n        return 1\n"
        new_code = (
            "    def foo(self):\n"
            "            if True:\n"
            "                x = 10\n"
            "            return x\n"
        )
        new_content = self._run(source, "X.foo", new_code)
        lines = {ln.strip(): ln for ln in new_content.splitlines() if ln.strip()}
        # Top-level body statements land at 8 (one unit below the def's 4).
        assert _indent_of(lines["return x"]) == 8
        assert _indent_of(lines["if True:"]) == 8
        # The nested if-body stays one level deeper than the if (12, not 16).
        assert _indent_of(lines["x = 10"]) == 12

    def test_module_level_body_drift_corrected(self):
        # def at col 0; model body over-indented to 8 instead of 4.
        source = "def foo():\n    return 1\n"
        new_code = "def foo():\n        return 42\n"
        new_content = self._run(source, "foo", new_code)
        assert self._body_indent(new_content, "foo") == 4

    def test_multiline_signature_not_split(self):
        # The def statement owns its multi-line signature continuation rows;
        # the correction must split at the BODY, not inside the signature.
        source = (
            "class X:\n"
            "    def foo(self,\n"
            "             a):\n"
            "        return 1\n"
        )
        new_code = (
            "    def foo(self,\n"
            "             a):\n"
            "            return 42\n"  # body drifted one level (12 vs 8)
        )
        new_content = self._run(source, "X.foo", new_code)
        assert self._body_indent(new_content, "foo") == 8

    def test_helper_no_drift_is_noop(self):
        # Direct unit test of the helper: a correctly-indented block is returned
        # unchanged (correction must not fire).
        block = ["    def foo(self):", "        return 1", "        return 2"]
        assert _correct_full_block_body_drift(block, " ", 4, 4, "foo") == block

    def test_helper_body_only_drift_corrected(self):
        # Direct unit test: body one unit deep is corrected, def line untouched.
        block = ["    def foo(self):", "            return 1"]
        out = _correct_full_block_body_drift(block, " ", 4, 4, "foo")
        assert out[0] == "    def foo(self):"  # def line preserved
        assert _indent_of(out[1]) == 8  # body corrected 12 -> 8


class TestFullBlockDataclassDecoratorPreserved:
    """BUG: ``_strip_redundant_dataclass_decorator`` ran UNCONDITIONALLY before
    the full-block vs body-only decision. In the full-block path the model's
    block (decorators included) REPLACES the original symbol region, so
    stripping ``@dataclass`` from new_body silently deleted the decorator: the
    original decorator lines were overwritten, not preserved as a header. The
    net effect was a ``@dataclass`` class losing its decorator on a full-block
    ``modify_symbol`` — which in turn made ``dataclass`` an unused import
    (pyflakes F401), exactly the symptom observed in production. The strip
    belongs ONLY in the body-only path, where ``header_lines`` keeps the
    original decorator and a misclassified full block in new_body would
    otherwise duplicate it as ``@dataclass\\n@dataclass\\nclass X:``.
    """

    def _run(self, source: str, symbol: str, new_code: str) -> str:
        """Apply a modify_symbol edit to a temp file and return the result."""
        path = _write_temp_file(source)
        try:
            success, diff, new_content = modify_symbol(path, symbol, new_code)
            assert success, f"modify_symbol failed: {diff}"
            return new_content
        finally:
            os.unlink(path)

    def test_full_block_with_dataclass_keeps_decorator(self):
        # The model re-declares @dataclass in a FULL block AND adds field z.
        # Before the fix, strip deleted the decorator (count 0). After the fix
        # it must survive exactly once.
        source = textwrap.dedent('''\
            from dataclasses import dataclass


            @dataclass
            class Foo:
                x: int
                y: int
        ''')
        new_code = textwrap.dedent('''\
            @dataclass
            class Foo:
                x: int
                y: int
                z: int
        ''')
        new_content = self._run(source, "Foo", new_code)
        # The decorator MUST survive exactly once (the bug dropped it to 0).
        assert new_content.count("@dataclass") == 1
        # The new field was applied.
        assert "z: int" in new_content
        # `dataclass` is still used → not an unused import (F401 guard).
        assert "from dataclasses import dataclass" in new_content

    def test_body_only_edit_preserves_decorator(self):
        # Body-only path: header_lines preserves the original @dataclass even
        # though new_body carries no class line. The strip is a no-op here
        # (a bare body can't parse to a ClassDef), so the decorator survives
        # via the preserved header.
        source = textwrap.dedent('''\
            from dataclasses import dataclass


            @dataclass
            class Foo:
                x: int
                y: int
        ''')
        new_body = textwrap.dedent('''\
            x: int
            y: int
            z: int
        ''')
        new_content = self._run(source, "Foo", new_body)
        assert new_content.count("@dataclass") == 1
        assert "z: int" in new_content


class TestSurgicalEditMultilineSignatureBodyStart:
    """BUG-3: body_start must skip a multi-line signature, not the first
    parameter row. A body-only edit must preserve the signature verbatim."""

    def test_multiline_signature_params_preserved(self):
        source = textwrap.dedent('''\
            def foo(
                a: int,
                b: str,
            ) -> None:
                x = a + len(b)
                return x
        ''')
        # body-only replacement (no def line) — new body
        new_body = textwrap.dedent('''\
            y = a * 2
            return y
        ''')
        diff = _apply_surgical_edit(source, "m.py", "foo", new_body, 0, 6)
        assert diff is not None
        added = _added_lines(diff)
        removed = _removed_lines(diff)
        # Signature rows (def foo( / a: int, / b: str, / ) -> None:) must be
        # neither added nor removed — they are preserved verbatim. The buggy
        # body_start picked 'a: int,' as the body, so the diff DELETED the
        # signature continuation while keeping the orphaned 'def foo(' line.
        added_text = "\n".join(added)
        removed_text = "\n".join(removed)
        assert "a: int" not in added_text, "parameter row leaked into added body"
        assert "b: str" not in added_text
        assert "a: int" not in removed_text, "parameter row was DELETED — signature corrupted"
        assert "b: str" not in removed_text
        assert ") -> None:" not in removed_text, "header close was DELETED — signature corrupted"
        assert "y = a * 2" in added_text
        assert "return y" in added_text

    def test_singleline_signature_body_start(self):
        source = textwrap.dedent('''\
            def foo(a: int) -> None:
                x = a + 1
                return x
        ''')
        new_body = textwrap.dedent('''\
            y = a * 2
            return y
        ''')
        diff = _apply_surgical_edit(source, "m.py", "foo", new_body, 0, 3)
        assert diff is not None
        added = _added_lines(diff)
        removed = _removed_lines(diff)
        added_text = "\n".join(added)
        removed_text = "\n".join(removed)
        assert "def foo" not in added_text
        assert "def foo" not in removed_text, "header was DELETED — signature corrupted"
        assert "y = a * 2" in added_text

    def test_decorator_plus_singleline_signature(self):
        source = textwrap.dedent('''\
            @deco
            def foo(a: int) -> None:
                x = a + 1
                return x
        ''')
        new_body = textwrap.dedent('''\
            y = a * 2
            return y
        ''')
        diff = _apply_surgical_edit(source, "m.py", "foo", new_body, 0, 4)
        assert diff is not None
        added = _added_lines(diff)
        removed = _removed_lines(diff)
        added_text = "\n".join(added)
        removed_text = "\n".join(removed)
        assert "@deco" not in added_text
        assert "def foo" not in added_text
        assert "def foo" not in removed_text
        assert "@deco" not in removed_text
        assert "y = a * 2" in added_text


class TestFindSymbolAstNodeTreeReuse:
    """IMP-A: _find_symbol_ast_node accepts a pre-parsed tree to avoid a
    redundant second ast.parse of the same source."""

    def test_tree_kwarg_reuses_parsed_tree(self):
        source = "def foo():\n    return 1\n"
        tree = ast.parse(source)
        node = _find_symbol_ast_node(source, "foo", tree=tree)
        assert node is not None
        assert node.name == "foo"

    def test_no_tree_falls_back_to_parse(self):
        node = _find_symbol_ast_node("def foo():\n    return 1\n", "foo")
        assert node is not None
        assert node.name == "foo"


# ── diff application helpers ─────────────────────────────────────────────────

def _added_lines(diff: str) -> list:
    """Return the content (no leading '+') of every added line in a unified
    diff, in order. Sufficient for indent-presence assertions."""
    out = []
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or line.startswith(("---", "+++")):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return out


def _removed_lines(diff: str) -> list:
    """Return the content (no leading '-') of every removed line in a unified
    diff, in order. Used to assert that a body-only edit does NOT delete the
    symbol's signature (the BUG-3 regression)."""
    out = []
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            out.append(line[1:])
    return out


def _indent_of(line: str) -> int:
    """Leading-whitespace width of a line (tabs count as 1 each; assertions
    here are on space-indented files so this is unambiguous)."""
    return len(line) - len(line.lstrip())


# ── Atomic write: crash-safety + permission preservation ───────────────────


class TestModifySymbolAtomicWrite:
    def test_preserves_executable_bit(self, tmp_path):
        """modify_symbol now writes via atomic_write_text (mkstemp + os.replace).
        mkstemp creates the temp as mode 0600 and os.replace keeps the temp's
        mode, so the exec bit would be stripped unless explicitly preserved.
        Guard that an executable script stays executable after a symbol edit."""
        src = tmp_path / "script.py"
        src.write_text("def main():\n    return 1\n")
        os.chmod(src, 0o755)
        assert os.stat(src).st_mode & 0o111  # sanity: started executable

        ok, _diff, _new = modify_symbol(str(src), "main", "def main():\n    return 2\n")
        assert ok, f"modify failed: {_diff}"

        assert os.stat(src).st_mode & 0o111, "execute bit stripped by atomic write"
        assert "return 2" in src.read_text()

    def test_no_leftover_temp_file(self, tmp_path):
        """After a successful modify_symbol the target dir must contain only the
        edited file — no orphaned ``.atomic_`` temp (os.replace renamed it in)."""
        src = tmp_path / "m.py"
        src.write_text("def f():\n    return 1\n")
        ok, _d, _n = modify_symbol(str(src), "f", "def f():\n    return 2\n")
        assert ok
        leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".atomic_")]
        assert leftovers == [], f"orphaned temp files: {leftovers}"


class TestAtomicWriteText:
    def test_writes_content_and_preserves_mode(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("old")
        os.chmod(p, 0o755)
        atomic_write_text(str(p), "new content")
        assert p.read_text() == "new content"
        assert os.stat(p).st_mode & 0o111, "exec bit lost"

    def test_temp_cleaned_up_and_original_intact_on_rename_failure(self, tmp_path, monkeypatch):
        """The corruption window the fix closes: if the rename fails (simulating a
        crash/SIGKILL/disk-full at os.replace), the temp is unlinked and the
        original file is NOT truncated (open(path,'w') would have zeroed it)."""
        import external_llm.common.atomic_io as aio
        p = tmp_path / "f.txt"
        p.write_text("original")

        def boom(src, dst):
            raise OSError("simulated rename failure")
        monkeypatch.setattr(aio.os, "replace", boom)

        with pytest.raises(OSError):
            atomic_write_text(str(p), "new")

        leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".atomic_")]
        assert leftovers == [], f"temp not cleaned up after failure: {leftovers}"
        assert p.read_text() == "original", "original corrupted by failed write"

    def test_new_file_gets_umask_aware_mode_not_0600(self, tmp_path):
        """A NEWLY CREATED file must mirror open(path,"w") = 0o666 & ~umask, NOT
        the mkstemp default 0600 — otherwise write_plan's created source files
        would silently lose group/world read perms (a regression vs the old
        truncating write). Asserted by parity with a sibling open('w') file, so
        the test is correct under any process umask."""
        new_atomic = tmp_path / "via_atomic.txt"
        new_plain = tmp_path / "via_open.txt"
        atomic_write_text(str(new_atomic), "x")
        with open(new_plain, "w") as fh:  # baseline: 0o666 & ~umask
            fh.write("x")
        assert new_atomic.read_text() == "x"
        assert (os.stat(new_atomic).st_mode & 0o777) == (os.stat(new_plain).st_mode & 0o777), (
            f"new-file mode {oct(os.stat(new_atomic).st_mode & 0o777)} != open('w') "
            f"{oct(os.stat(new_plain).st_mode & 0o777)} (mkstemp 0600 leaked through?)"
        )

    def test_new_file_explicit_mode_param_honored(self, tmp_path):
        """Caller-supplied ``mode`` overrides the umask default for new files
        (e.g. 0o600 for secrets). Uses 0o640 — distinct from the mkstemp default
        0600 — so the test fails if ``mode`` is silently ignored."""
        p = tmp_path / "secret.txt"
        atomic_write_text(str(p), "shh", mode=0o640)
        assert p.read_text() == "shh"
        assert (os.stat(p).st_mode & 0o777) == 0o640

    def test_explicit_mode_ignored_for_existing_file(self, tmp_path):
        """For an existing target the preserved mode wins even if ``mode`` is
        passed — an executable script must not be stripped to 0600."""
        p = tmp_path / "script.py"
        p.write_text("old")
        os.chmod(p, 0o755)
        atomic_write_text(str(p), "new", mode=0o600)  # mode must be ignored
        assert p.read_text() == "new"
        assert os.stat(p).st_mode & 0o111, "exec bit stripped despite preserved mode"

    def test_creates_missing_parent_directory(self, tmp_path):
        """A path whose parent dir does not yet exist is written successfully
        (mirrors atomic_write_json, which always created the parent)."""
        p = tmp_path / "nested" / "deep" / "out.txt"
        atomic_write_text(str(p), "payload")
        assert p.read_text() == "payload"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
