"""Unit tests for SyntaxValidator — the language-agnostic syntax/symbol facade.

Covers both dispatch branches of every static method: the provider path
(real PythonSyntaxProvider) and the tree-sitter fallback path (LanguageId.UNKNOWN
or a monkeypatched dispatch), plus find_symbol_in_file's file-read / OSError /
kind-resolution branches and _ts_find_symbol_in_file.
"""
from __future__ import annotations

from external_llm.languages import tree_sitter_utils as ts_utils
from external_llm.languages.models import LanguageId
from external_llm.languages.syntax_validator import SyntaxValidator

PY = LanguageId.PYTHON
UNK = LanguageId.UNKNOWN


# ── validate_syntax ─────────────────────────────────────────────────────────

class TestValidateSyntax:
    def test_python_valid(self):
        r = SyntaxValidator.validate_syntax("x = 1\n", PY)
        assert r.ok is True

    def test_python_invalid(self):
        r = SyntaxValidator.validate_syntax("def broken(:\n", PY)
        assert r.ok is False

    def test_fallback_unknown_lang(self):
        # No provider for UNKNOWN → tree-sitter fallback (graceful ok=True)
        r = SyntaxValidator.validate_syntax("x = 1", UNK)
        assert r.ok is True


# ── find_symbol_range ───────────────────────────────────────────────────────

class TestFindSymbolRange:
    def test_python_found(self):
        rng = SyntaxValidator.find_symbol_range("def foo():\n    pass\n", "foo", PY)
        assert rng == (1, 2)

    def test_python_missing(self):
        assert SyntaxValidator.find_symbol_range("def foo():\n    pass\n", "nope", PY) is None

    def test_fallback_unknown_lang(self):
        assert SyntaxValidator.find_symbol_range("def foo(): pass", "foo", UNK) is None


# ── find_symbols ────────────────────────────────────────────────────────────

class TestFindSymbols:
    def test_python(self):
        syms = SyntaxValidator.find_symbols("def foo():\n    pass\n", PY)
        assert syms == [("foo", "function", 1, 2)]

    def test_fallback_unknown_lang(self):
        assert SyntaxValidator.find_symbols("def foo(): pass", UNK) == []


# ── extract_symbol_body ─────────────────────────────────────────────────────

class TestExtractSymbolBody:
    def test_python(self):
        body = SyntaxValidator.extract_symbol_body("def foo():\n    x = 1\n    return x\n", "foo", PY)
        assert body == (2, 3)

    def test_fallback_unknown_lang(self):
        assert SyntaxValidator.extract_symbol_body("def foo():\n    x=1\n", "foo", UNK) is None


# ── is_dead_code_introduced ─────────────────────────────────────────────────

class TestIsDeadCodeIntroduced:
    def test_python_valid_new(self):
        assert SyntaxValidator.is_dead_code_introduced("x = 1", "x = 2", PY) is False

    def test_python_invalid_new(self):
        assert SyntaxValidator.is_dead_code_introduced("x = 1", "def broken(:\n", PY) is True

    def test_fallback_ok(self):
        # UNKNOWN → fallback validate_syntax always ok → dead code False
        assert SyntaxValidator.is_dead_code_introduced("a", "b", UNK) is False

    def test_fallback_failing(self, monkeypatch):
        import external_llm.languages.base as base_mod
        from external_llm.languages.models import SyntaxValidationResult
        monkeypatch.setattr(
            base_mod, "tree_sitter_syntax_fallback",
            lambda content, lang, file_path="": SyntaxValidationResult(ok=False, errors=["boom"]),
        )
        assert SyntaxValidator.is_dead_code_introduced("a", "b", UNK) is True


# ── find_symbol_in_file ─────────────────────────────────────────────────────

class TestFindSymbolInFile:
    def test_unknown_extension(self):
        assert SyntaxValidator.find_symbol_in_file("file.xyz", "anything") is None

    def test_python_with_content_found(self):
        info = SyntaxValidator.find_symbol_in_file(
            "t.py", "foo", content="def foo():\n    pass\n")
        assert info["file"] == "t.py"
        assert info["line"] == 1
        assert info["end_line"] == 2
        assert info["kind"] == "function"
        assert info["name"] == "foo"

    def test_python_with_content_missing(self):
        assert SyntaxValidator.find_symbol_in_file(
            "t.py", "nope", content="def foo():\n    pass\n") is None

    def test_python_reads_file_when_content_none(self, tmp_path):
        p = tmp_path / "t.py"
        p.write_text("def foo():\n    pass\n", encoding="utf-8")
        info = SyntaxValidator.find_symbol_in_file(str(p), "foo")
        assert info["line"] == 1
        assert info["kind"] == "function"

    def test_python_missing_file_oserror(self):
        assert SyntaxValidator.find_symbol_in_file(
            "/nonexistent_dir_xyz/t.py", "foo") is None

    def test_kind_stays_symbol_when_enumeration_mismatches(self, monkeypatch):
        class _FakeProvider:
            def find_symbol_in_file(self, file_path, symbol_name, content):
                return (3, 5)
            def find_symbols(self, content):
                # Same name but different start line → kind resolution must miss
                return [("foo", "function", 1, 2)]
        import external_llm.languages.syntax_validator as sv
        monkeypatch.setattr(sv, "_get_provider", lambda lang: _FakeProvider())
        info = SyntaxValidator.find_symbol_in_file("t.py", "foo", content="x")
        assert info["kind"] == "symbol"

    def test_fallback_ts_when_no_provider(self, monkeypatch):
        # Dispatch returns None for a real language → falls through to
        # _ts_find_symbol_in_file (tree-sitter direct lookup).
        import external_llm.languages.syntax_validator as sv
        monkeypatch.setattr(sv, "_get_provider", lambda lang: None)
        info = SyntaxValidator.find_symbol_in_file(
            "t.py", "foo", content="def foo():\n    pass\n")
        assert info["line"] == 1
        assert info["end_line"] == 2
        assert info["kind"] == "function"


# ── _ts_find_symbol_in_file ─────────────────────────────────────────────────

class TestTsFindSymbolInFile:
    def test_no_range(self, monkeypatch):
        monkeypatch.setattr(ts_utils, "find_symbol_range", lambda *a: None)
        assert SyntaxValidator._ts_find_symbol_in_file("t.go", "foo", "x", LanguageId.GO) is None

    def test_range_with_kind_match(self, monkeypatch):
        monkeypatch.setattr(ts_utils, "find_symbol_range", lambda *a: (2, 4))
        monkeypatch.setattr(
            ts_utils, "find_all_symbols",
            lambda *a: [("foo", "function", 2, 4), ("bar", "variable", 5, 5)],
        )
        info = SyntaxValidator._ts_find_symbol_in_file("t.go", "foo", "x", LanguageId.GO)
        assert info["file"] == "t.go"
        assert info["line"] == 2
        assert info["end_line"] == 4
        assert info["kind"] == "function"
        assert info["name"] == "foo"

    def test_range_without_kind_match(self, monkeypatch):
        monkeypatch.setattr(ts_utils, "find_symbol_range", lambda *a: (2, 4))
        monkeypatch.setattr(ts_utils, "find_all_symbols", lambda *a: [("bar", "variable", 5, 5)])
        info = SyntaxValidator._ts_find_symbol_in_file("t.go", "foo", "x", LanguageId.GO)
        assert info["kind"] == "symbol"
