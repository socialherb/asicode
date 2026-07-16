"""Regression tests for the C / C++ syntax providers.

Covers the three contracts that make the gcc/g++ integration safe:

1. **Resolution safety** — an isolated temp-file ``-fsyntax-only`` compile
   cannot resolve project headers, so both gcc's ``No such file or directory``
   and clang's ``file not found`` (clang is the default ``gcc`` on macOS) must
   be filtered out and NOT roll back a valid edit. Only genuine syntax errors
   gate the write.
2. **Graceful degrade** — when no compiler is on ``$PATH`` the provider
   returns ``ok=True`` (the validator's tree-sitter fallback then applies),
   never blocking edits.
3. **Compiler-agnostic detection** — gcc and clang phrasings are both covered.

Mirrors test_isolated_compile_resolution.py. subprocess.run / shutil.which are
mocked so these tests need no C toolchain installed.
"""
from types import SimpleNamespace
from unittest.mock import patch

from external_llm.languages.base import (
    _filter_genuine_syntax_errors,
    _is_resolution_error,
)
from external_llm.languages.c_provider import CppSyntaxProvider, CSyntaxProvider
from external_llm.languages.models import LanguageId, SyntaxError_


def _fake_proc(returncode, stdout):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


# ── C: resolution safety ────────────────────────────────────────────────────

class TestCResolutionSafety:
    def _validate(self, stdout, returncode=1):
        p = CSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/gcc"), \
             patch("external_llm.languages.c_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("main.c", "irrelevant — subprocess mocked")

    def test_clang_missing_header_phrasing_is_filtered(self):
        # clang (the default gcc on macOS): fatal error: 'foo.h' file not found
        out = "main.c:1:10: fatal error: 'foo.h' file not found\n"
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_gcc_missing_header_phrasing_is_filtered(self):
        # real gcc: fatal error: foo.h: No such file or directory
        out = "main.c:1:10: fatal error: foo.h: No such file or directory\n"
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_genuine_syntax_error_still_fails(self):
        out = "main.c:5:2: error: expected ';' after expression\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert r.errors[0].line == 5

    def test_mixed_keeps_only_syntax_error(self):
        out = (
            "main.c:1:10: fatal error: foo.h: No such file or directory\n"
            "main.c:5:2: error: expected ';' after expression\n"
        )
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "expected ';'" in r.errors[0].message

    def test_clean_compile_is_ok(self):
        r = self._validate("", returncode=0)
        assert r.ok is True

    def test_warnings_do_not_fail(self):
        # Warnings (returncode 0, or non-matching severity) never block.
        out = "main.c:3:9: warning: unused variable 'x'\n"
        r = self._validate(out, returncode=0)
        assert r.ok is True


# ── C++: resolution safety ──────────────────────────────────────────────────

class TestCppResolutionSafety:
    def _validate(self, stdout, returncode=1):
        p = CppSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/g++"), \
             patch("external_llm.languages.c_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("main.cpp", "irrelevant — subprocess mocked")

    def test_missing_header_is_filtered(self):
        out = "main.cpp:1:10: fatal error: 'foo.h' file not found\n"
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_undeclared_symbol_without_include_error_is_genuine(self):
        # Without a failed include, "'db' was not declared in this scope" is
        # a genuine typo (e.g. ``total += valeu;``), not a cascade from a
        # missing header.  It must gate the edit.
        out = "main.cpp:5:5: error: 'db' was not declared in this scope\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "not declared" in r.errors[0].message

    def test_genuine_syntax_error_still_fails(self):
        out = "main.cpp:5:2: error: expected expression\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1


# ── Graceful degrade (no compiler on $PATH) ─────────────────────────────────

class TestCompilerAbsentDegrade:
    # ── When no compiler is on $PATH, tree-sitter provides a zero-toolchain
    #    syntax check. Valid code passes; syntax errors are caught. ──────────

    def test_c_valid_passes_tree_sitter_fallback(self):
        """Valid C code passes tree-sitter fallback when no compiler available."""
        with patch("external_llm.languages.c_provider.shutil.which", return_value=None):
            r = CSyntaxProvider().validate_syntax("main.c", "int main(void){return 0;}")
        assert r.ok is True
        assert r.language is LanguageId.C

    def test_cpp_valid_passes_tree_sitter_fallback(self):
        """Valid C++ code passes tree-sitter fallback when no compiler available."""
        with patch("external_llm.languages.c_provider.shutil.which", return_value=None):
            r = CppSyntaxProvider().validate_syntax("main.cpp", "int main(){return 0;}")
        assert r.ok is True
        assert r.language is LanguageId.CPP

    def test_c_syntax_error_caught_by_tree_sitter(self):
        """Tree-sitter fallback catches real C syntax errors (missing semicolon)."""
        with patch("external_llm.languages.c_provider.shutil.which", return_value=None):
            r = CSyntaxProvider().validate_syntax("main.c", "int main(void){return 0}")
        assert r.ok is False
        assert r.language is LanguageId.C
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message

    def test_cpp_syntax_error_caught_by_tree_sitter(self):
        """Tree-sitter fallback catches real C++ syntax errors (stray asterisk)."""
        with patch("external_llm.languages.c_provider.shutil.which", return_value=None):
            r = CppSyntaxProvider().validate_syntax("main.cpp", "int main(){return * }")
        assert r.ok is False
        assert r.language is LanguageId.CPP
        assert len(r.errors) >= 1
        assert "tree-sitter" in r.errors[0].message

    def test_semantics_degrades_when_no_compiler(self):
        import os
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tf:
            tf.write("int main(void){return 0;}")
            path = tf.name
        try:
            with patch("external_llm.languages.c_provider.shutil.which", return_value=None):
                r = CSyntaxProvider().validate_semantics(path)
            assert r.ok is True
        finally:
            os.unlink(path)


# ── Resolution classifier phrases (base.py) ─────────────────────────────────

class TestCResolutionClassifier:
    def test_c_phrases_cover_both_compilers(self):
        # gcc phrasing
        assert _is_resolution_error("foo.h: No such file or directory", LanguageId.C)
        # clang phrasing (macOS default gcc)
        assert _is_resolution_error("'foo.h' file not found", LanguageId.C)
        assert _is_resolution_error("implicit declaration of function 'f'", LanguageId.C)
        # genuine syntax error is NOT a resolution error
        assert not _is_resolution_error("expected ';' after expression", LanguageId.C)

    def test_cpp_phrases(self):
        assert _is_resolution_error("foo.h: No such file or directory", LanguageId.CPP)
        assert _is_resolution_error("'db' was not declared in this scope", LanguageId.CPP)
        assert not _is_resolution_error("expected expression", LanguageId.CPP)

    def test_filter_drops_only_resolution_errors(self):
        errs = [
            SyntaxError_(file="f.c", line=1, col=10,
                         message="fatal error: foo.h: No such file or directory"),
            SyntaxError_(file="f.c", line=5, col=2, message="expected ';' after expression"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.C)
        assert len(kept) == 1
        assert "expected ';'" in kept[0].message


class TestContextDependentFiltering:
    """Context-dependent resolution phrases are only dropped when a failed include
    is also present.  Without an include failure they are genuine typos that
    MUST gate the edit — otherwise ``total += valeu;`` would pass the syntax
    gate under GNU g++ ("was not declared in this scope").
    """

    # ── C: "implicit declaration of" ────────────────────────────────────────

    def test_c_implicit_decl_kept_when_no_include_error(self):
        errs = [
            SyntaxError_(file="f.c", line=3, col=5,
                         message="warning: implicit declaration of function 'printf'"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.C)
        # "implicit declaration of" is context-dependent; without an include
        # failure it's a genuine error (function never declared at all, not
        # just a missing header).
        assert len(kept) == 1
        assert "implicit declaration" in kept[0].message

    def test_c_implicit_decl_filtered_when_include_fails(self):
        errs = [
            SyntaxError_(file="f.c", line=1, col=10,
                         message="fatal error: stdio.h: No such file or directory"),
            SyntaxError_(file="f.c", line=3, col=5,
                         message="warning: implicit declaration of function 'printf'"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.C)
        # Include failure present → implicit declaration is a cascade, not a typo.
        assert len(kept) == 0

    # ── CPP: "was not declared in this scope" ───────────────────────────────

    def test_cpp_undeclared_kept_when_no_include_error(self):
        """GNU g++ emits 'was not declared in this scope' for a local typo
        like ``total += valeu;``.  Without a failed include this is a real error
        that must gate the edit."""
        errs = [
            SyntaxError_(file="f.cpp", line=5, col=12,
                         message="'valeu' was not declared in this scope"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.CPP)
        assert len(kept) == 1
        assert "not declared" in kept[0].message

    def test_cpp_undeclared_filtered_when_header_missing(self):
        """'was not declared in this scope' that cascades from a missing header
        must still be filtered so the valid edit is not rolled back."""
        errs = [
            SyntaxError_(file="f.cpp", line=1, col=10,
                         message="fatal error: vector: No such file or directory"),
            SyntaxError_(file="f.cpp", line=7, col=5,
                         message="'std' was not declared in this scope"),
            SyntaxError_(file="f.cpp", line=8, col=5,
                         message="'vector' has not been declared"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.CPP)
        assert len(kept) == 0

    # ── Include failures are always filtered (unconditional) ─────────────────

    def test_include_error_always_filtered(self):
        errs = [
            SyntaxError_(file="f.cpp", line=1, col=10,
                         message="fatal error: boost/preprocessor.hpp: No such file or directory"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.CPP)
        assert len(kept) == 0

    # ── Genuine syntax errors are always kept ────────────────────────────────

    def test_genuine_syntax_error_kept_alongside_context_dependent(self):
        """A genuine syntax error must gate the edit even when a context-dependent
        typo is also present."""
        errs = [
            SyntaxError_(file="f.cpp", line=5, col=12,
                         message="'valeu' was not declared in this scope"),
            SyntaxError_(file="f.cpp", line=10, col=1,
                         message="expected ';' after expression"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.CPP)
        assert len(kept) == 2  # both are genuine (no include error)


# ── Registry / wiring ───────────────────────────────────────────────────────

class TestRegistryWiring:
    def test_c_and_cpp_providers_registered(self):
        from external_llm.languages.registry import LanguageRegistry
        r = LanguageRegistry.instance()
        assert r.get("foo.c").__class__.__name__ == "CSyntaxProvider"
        assert r.get("bar.cpp").__class__.__name__ == "CppSyntaxProvider"
        assert r.get("header.h").language_id() is LanguageId.C
        assert r.get("header.hpp").language_id() is LanguageId.CPP

    def test_file_globs(self):
        assert "*.c" in CSyntaxProvider().get_file_globs()
        assert "*.h" in CSyntaxProvider().get_file_globs()
        assert "*.cpp" in CppSyntaxProvider().get_file_globs()
        assert "*.hpp" in CppSyntaxProvider().get_file_globs()

    def test_capabilities_advertise_syntax_and_semantics(self):
        caps = CSyntaxProvider().capabilities()
        assert caps.has_syntax_validator
        assert caps.has_semantic_validator
        assert caps.has_symbol_search

    def test_tool_map_lists_c_and_cpp(self):
        from external_llm.languages.dependency_checker import _LANGUAGE_TOOL_MAP
        assert [t.cmd for t in _LANGUAGE_TOOL_MAP[LanguageId.C]] == ["gcc"]
        assert [t.cmd for t in _LANGUAGE_TOOL_MAP[LanguageId.CPP]] == ["g++"]


# ── Symbol search (tree-sitter / regex fallback) ────────────────────────────

class TestSymbolSearch:
    SRC = (
        "#include <stdio.h>\n"
        "static int helper(int x){ return x+1; }\n"
        "struct Point { int x; int y; };\n"
        "int main(void){ printf(\"%d\", helper(5)); return 0; }\n"
    )

    def test_find_function(self):
        p = CSyntaxProvider()
        result = p.find_symbol_in_file("t.c", "helper", self.SRC)
        assert result is not None
        assert result[0] == 2  # helper is on line 2

    def test_find_top_level_definitions(self):
        p = CSyntaxProvider()
        defs = p.find_top_level_definitions(self.SRC)
        names = {name for (name, _kind, _s, _e) in defs}
        assert "helper" in names
        assert "main" in names


# ── .h header union validation (C / C++) ──────────────────────────────────────
# Regression: a valid C++ header (namespace/templates) routes to the C provider
# (``_EXT_MAP[".h"] = C``). gcc-C reports a GENUINE C syntax error
# ("unknown type name 'namespace'") that is NOT a resolution error → would roll
# back a valid edit. The union retry (gcc-C fails → g++-CPP) must accept it.
#
# Non-vacuous by construction: revert the union branch in ``_validate_syntax_impl``
# / ``validate_semantics`` and every ``ok is True`` assertion below turns red,
# because the primary C compile genuinely fails.

class TestHeaderUnionValidation:
    @staticmethod
    def _run_factory():
        """Distinguish C vs C++ compile by the resolved compiler name (cmd[0])."""
        def _run(cmd, **kwargs):
            cc = cmd[0]
            if cc in ("gcc", "clang"):
                # gcc-C genuinely rejects ``namespace`` — not a resolution error.
                return _fake_proc(1, "t.h:1:1: error: unknown type name 'namespace'\n")
            # g++-CPP accepts valid C++.
            return _fake_proc(0, "")
        return _run

    def test_h_cpp_header_passes_syntax_union(self):
        """A ``.h`` C++ header must NOT be rolled back (gcc-C fails, g++ ok)."""
        p = CSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/gcc"), \
             patch("external_llm.languages.c_provider.subprocess.run",
                   side_effect=self._run_factory()):
            r = p.validate_syntax("types.h", "namespace geo { struct Point { int x; }; }")
        assert r.ok is True
        assert r.language is LanguageId.C

    def test_h_genuinely_broken_still_fails(self):
        """A ``.h`` broken in BOTH C and C++ must still fail (union is a superset)."""
        p = CSyntaxProvider()

        def _run(cmd, **kwargs):
            # Both gcc and g++ reject "int x = ;".
            return _fake_proc(1, "t.h:1:9: error: expected expression\n")

        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/gcc"), \
             patch("external_llm.languages.c_provider.subprocess.run", side_effect=_run):
            r = p.validate_syntax("types.h", "int x = ;")
        assert r.ok is False

    def test_c_file_cpp_content_does_not_get_union(self):
        """A ``.c`` file is unambiguously C — C++ content must fail, no retry."""
        p = CSyntaxProvider()
        calls = []

        def _run(cmd, **kwargs):
            calls.append(cmd[0])
            return _fake_proc(1, "t.c:1:1: error: unknown type name 'namespace'\n")

        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/gcc"), \
             patch("external_llm.languages.c_provider.subprocess.run", side_effect=_run):
            r = p.validate_syntax("main.c", "namespace geo {}")
        assert r.ok is False
        # No C++ retry for .c — only the C compile ran.
        assert all(c in ("gcc", "clang") for c in calls)

    def test_h_cpp_header_passes_semantics_union(self, tmp_path):
        """On-disk semantics: gcc-C fails on a C++ ``.h`` → g++-CPP retry accepts."""
        h_file = tmp_path / "types.h"
        h_file.write_text("namespace geo { struct Point { int x; }; }\n")
        p = CSyntaxProvider()
        compilers_seen = []

        def _run(cmd, **kwargs):
            cc = cmd[0]
            compilers_seen.append(cc)
            if cc in ("gcc", "clang"):
                return _fake_proc(1, "types.h:1:1: error: unknown type name 'namespace'\n")
            return _fake_proc(0, "")

        with patch("external_llm.languages.c_provider.shutil.which", return_value="/usr/bin/g++"), \
             patch("external_llm.languages.c_provider.subprocess.run", side_effect=_run):
            r = p.validate_semantics(str(h_file))
        assert r.ok is True
        # The C++ retry actually fired.
        assert any(c in ("g++", "clang++") for c in compilers_seen)


# ── _find_block_end: string / char / comment skipping ─────────────────────────
# Regression: the brace counter ignored literals and comments, so an unbalanced
# brace inside them (``char *a = "{";``) corrupted depth and returned the wrong
# end line. Each case below returns the WRONG line on the pre-fix code.

class TestFindBlockEndLiterals:
    @staticmethod
    def _end(src):
        offset = src.index("void")
        return CSyntaxProvider._find_block_end(src, offset)

    def test_unbalanced_brace_in_string(self):
        # ONE extra '{' inside a string literal; body really closes on line 3.
        src = (
            "void f() {\n"        # line 1: depth 1
            '    char *a = "{";\n'  # line 2: '{' inside string (must be skipped)
            "}\n"                 # line 3: depth 0 → return 3
        )
        # Pre-fix: '{' in string → depth 2, then '}' line 3 → depth 1 ≠ 0 →
        # runs off end → returns fallback line 1.
        assert self._end(src) == 3

    def test_closing_brace_in_block_comment(self):
        src = (
            "void f() {\n"     # line 1: depth 1
            "    /* } */\n"    # line 2: '}' inside comment (must be skipped)
            "}\n"              # line 3: depth 0 → return 3
        )
        # Pre-fix: '}' in comment → depth 0 → returns 2 (wrong).
        assert self._end(src) == 3

    def test_closing_brace_in_char_literal(self):
        src = (
            "void f() {\n"      # line 1: depth 1
            "    char c = '}';\n"  # line 2: '}' inside char literal
            "}\n"               # line 3: depth 0 → return 3
        )
        # Pre-fix: '}' in char → depth 0 → returns 2 (wrong).
        assert self._end(src) == 3

    def test_line_comment_with_brace(self):
        src = (
            "void f() {\n"   # line 1: depth 1
            "    // {\n"     # line 2: '{' in line comment
            "}\n"            # line 3: depth 0 → return 3
        )
        # Pre-fix: '{' in comment → depth 2; '}' line 3 → depth 1 → fallback 1.
        assert self._end(src) == 3

    def test_escaped_quote_in_string(self):
        # Ensure `\"` escape does not terminate the string scan early.
        src = (
            'void f() {\n'           # line 1: depth 1
            '    char *s = "\\"}";\n'  # line 2: escaped quote, then '}' in string
            "}\n"                    # line 3: depth 0 → return 3
        )
        assert self._end(src) == 3


# ── _resolve_compilers: per-instance caching (test isolation) ─────────────────
# Cache must (a) de-dupe `shutil.which` across calls within one instance, and
# (b) NOT leak across instances (fresh provider → mock fires). The singleton
# registry provider keeps the cache across the agent loop.

class TestCompilerResolutionCache:
    def test_cache_dedupes_within_instance(self):
        from external_llm.languages.c_provider import _C_COMPILERS
        p = CSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which",
                   return_value="/usr/bin/gcc") as m:
            r1 = p._resolve_compilers(_C_COMPILERS)
            r2 = p._resolve_compilers(_C_COMPILERS)  # cached hit
            assert m.call_count == 1  # second call served from cache
        assert r1 == r2 == "gcc"

    def test_fresh_instance_re_resolves(self):
        """Test isolation: a new instance has an empty cache → mock fires."""
        from external_llm.languages.c_provider import _C_COMPILERS
        p2 = CSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which",
                   return_value=None) as m:
            r = p2._resolve_compilers(_C_COMPILERS)
            # gcc→None then clang→None: both candidates probed (2 which() calls),
            # proving the fresh instance did NOT inherit a cached value.
            assert m.call_count == 2
        assert r is None

    def test_cache_keyed_by_candidates(self):
        """C and C++ candidate tuples are independent cache keys."""
        from external_llm.languages.c_provider import _C_COMPILERS, _CPP_COMPILERS
        p = CSyntaxProvider()
        with patch("external_llm.languages.c_provider.shutil.which",
                   return_value="/usr/bin/x"):
            c_res = p._resolve_compilers(_C_COMPILERS)
            cpp_res = p._resolve_compilers(_CPP_COMPILERS)
        assert c_res == "gcc"
        assert cpp_res == "g++"
