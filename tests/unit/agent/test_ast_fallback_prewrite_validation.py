"""Tests for AST fallback pre-write syntax validation + indentation normalization.

Covers the logic added to _try_ast_rewrite_fallback that:
  1. Validates the spliced file content with ast.parse() BEFORE writing to disk
  2. On IndentationError: tries ast.parse + ast.unparse on _fb_code alone to
     recover canonical indentation (may lose LLM-added comments, but produces
     valid Python)
  3. On unrecoverable syntax error: returns None without writing (forces strategy
     ladder to choose a different approach instead of quality-gate rollback)
"""
import ast
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Helper: reproduce the normalization logic from _try_ast_rewrite_fallback
# ---------------------------------------------------------------------------

def _normalize_fb_code_via_unparse(fb_code: str, orig_indent: str) -> str | None:
    """Mirror of the normalization path in _try_ast_rewrite_fallback."""
    try:
        tree = ast.parse(fb_code)
        norm_raw = ast.unparse(tree)
        if orig_indent:
            fn_lines = norm_raw.splitlines()
            fn_min = min(
                (len(_item_) - len(_item_.lstrip()) for _item_ in fn_lines if _item_.strip()),
                default=0,
            )
            norm_raw = "\n".join(
                (orig_indent + _item_[fn_min:]) if _item_.strip() else ""
                for _item_ in fn_lines
            )
        return norm_raw
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# Tests: normalization helper
# ---------------------------------------------------------------------------

class TestNormalizeViaUnparse:
    """Unit tests for the ast.parse + ast.unparse normalization helper."""

    def test_valid_function_preserved(self):
        code = textwrap.dedent("""\
            def foo(x):
                if x:
                    return x
                return None
        """)
        result = _normalize_fb_code_via_unparse(code, "")
        assert result is not None
        # Must be valid Python after normalization
        ast.parse(result)

    def test_top_level_function_reindented_for_class_method(self):
        # LLM generates function at 0-indent; class method needs 4-space indent
        code = textwrap.dedent("""\
            def check(self, name):
                if not name:
                    return
                do_something(name)
        """)
        orig_indent = "    "
        result = _normalize_fb_code_via_unparse(code, orig_indent)
        assert result is not None
        # All non-blank lines should start with at least 4 spaces
        for line in result.splitlines():
            if line.strip():
                assert line.startswith("    "), f"line not reindented: {line!r}"

    def test_indentation_error_in_fb_code_returns_none(self):
        # Code with broken internal indentation → ast.parse fails → should return None
        broken = (
            "def foo(x):\n"
            "    if x:\n"
            "        return x\n"
            "       return None\n"   # ← 7 spaces instead of 8 or 4
        )
        result = _normalize_fb_code_via_unparse(broken, "")
        assert result is None

    def test_continue_guard_preserved_after_normalize(self):
        # The specific SL28 guard that must survive normalization
        code = textwrap.dedent("""\
            def _check(self, errors):
                for name in errors:
                    if not name:
                        continue
                    handle(name)
        """)
        result = _normalize_fb_code_via_unparse(code, "")
        assert result is not None
        assert "continue" in result
        ast.parse(result)

    def test_deeply_nested_normalised(self):
        # 3-level deep nesting (similar to SL28's real structure)
        code = textwrap.dedent("""\
            def f(items):
                for item in items:
                    if item:
                        for sub in item:
                            if not sub:
                                continue
                            process(sub)
        """)
        result = _normalize_fb_code_via_unparse(code, "")
        assert result is not None
        ast.parse(result)
        assert "continue" in result


# ---------------------------------------------------------------------------
# Tests: pre-write validation gate behaviour
# ---------------------------------------------------------------------------

class TestPreWriteValidationGate:
    """Behaviour tests for the pre-write validation gate in _try_ast_rewrite_fallback.

    We test the LOGIC (parse → check → recover-or-skip) without invoking the
    full _try_ast_rewrite_fallback stack (which requires disk access, LLM, etc.).
    """

    def _simulate_validation_gate(self, new_text: str, fb_code: str, orig_indent: str = ""):
        """Simulate the pre-write validation gate logic.

        Returns:
            ("write", write_text)  — safe to write
            ("skip", reason)       — skip write (normalization failed)
        """
        try:
            ast.parse(new_text)
            return ("write", new_text)
        except SyntaxError as syn:
            is_indent = isinstance(syn, IndentationError) or "indent" in str(syn).lower()
            if not is_indent:
                return ("skip", f"syntax:{syn}")
            norm_fb = _normalize_fb_code_via_unparse(fb_code, orig_indent)
            if norm_fb is None:
                return ("skip", "normalization_failed")
            # Re-splice: replace the function portion with the normalised version
            # (In _try_ast_rewrite_fallback this re-runs ASTRewriter; here we
            # just verify the normalised function compiles standalone.)
            try:
                ast.parse(norm_fb)
                return ("write", norm_fb)  # simplified: use normalised func alone
            except SyntaxError as e2:
                return ("skip", f"norm_still_invalid:{e2}")

    def test_valid_file_passes_through(self):
        valid = "def foo():\n    return 1\n"
        action, content = self._simulate_validation_gate(valid, valid)
        assert action == "write"
        assert content == valid

    def test_indent_error_triggers_normalization(self):
        # fb_code has correct internal indent; splice introduced an outer issue
        fb_code = textwrap.dedent("""\
            def foo():
                for x in items:
                    if not x:
                        continue
                    process(x)
        """)
        # Simulate a file where the function was spliced with wrong outer indent
        broken_file = (
            "class Foo:\n"
            "  def foo(self):\n"     # 2-space (non-standard)
            "      for x in items:\n"  # 6-space
            "          if not x:\n"    # 10-space
            "            continue\n"   # 12-space
            "          process(x)\n"   # 10-space
        )
        action, _ = self._simulate_validation_gate(broken_file, fb_code)
        # fb_code itself parses fine → normalization should succeed
        assert action == "write"

    def test_broken_fb_code_returns_skip(self):
        # fb_code itself has inconsistent indentation in the SAME block
        # → ast.parse fails → normalization returns None → skip
        broken_fb = (
            "def foo():\n"
            "    for x in items:\n"
            "        do_something()\n"   # 8-space
            "       continue\n"          # 7-space — inconsistent with 8-space block
        )
        # Verify the test input is actually broken Python
        import ast as _ast_check
        with pytest.raises(IndentationError):
            _ast_check.parse(broken_fb)
        broken_file = "class C:\n" + broken_fb
        action, _reason = self._simulate_validation_gate(broken_file, broken_fb)
        assert action == "skip"

    def test_non_indent_syntax_error_returns_skip(self):
        # Syntax error that is NOT an indentation issue → skip immediately
        invalid_file = "def foo(:\n    pass\n"  # invalid syntax
        fb_code = "def foo():\n    pass\n"
        action, reason = self._simulate_validation_gate(invalid_file, fb_code)
        assert action == "skip"
        assert "syntax:" in reason

    def test_continue_guard_survives_normalization(self):
        # The SL28 target: after normalization, 'continue' guard must remain
        fb_code = textwrap.dedent("""\
            def _check_f821_and_auto_repair(self, errors):
                for name in errors:
                    if not name:
                        continue
                    module = self._resolve(name)
        """)
        broken_file = (
            "class RepairEngine:\n"
            "     def _check_f821_and_auto_repair(self, errors):\n"  # 5-space indent bug
            "         for name in errors:\n"
            "           if not name:\n"     # 11-space instead of 12
            "                 continue\n"  # 17-space — mismatch
        )
        action, write_text = self._simulate_validation_gate(broken_file, fb_code)
        assert action == "write"
        assert "continue" in write_text
