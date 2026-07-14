"""Regression tests: Go/Java/Kotlin validate_syntax must not roll back valid edits
when an isolated temp-file compile cannot resolve project imports.

Mirrors test_typescript_syntax_only.py. The pre-write ``validate_syntax`` gate
compiles a single temp file WITHOUT go.mod / sourcepath / classpath, so the
backing compiler emits symbol/module/package RESOLUTION failures
("no required module provides package", "cannot find symbol",
"unresolved reference") for code that is perfectly valid in its real project.
Those must NOT gate the edit — only genuine SYNTAX errors should. The on-disk
``validate_semantics`` pass (full project context, run after the write) is the
authoritative resolution check.

Surfaced by run_20260610: editing internal/ui/model/ui.go (Go, imports
charm.land/bubbles/v2/help) was rolled back by apply_patch/anchor_edit with
"no required module provides package … go.mod file not found" even though the
edit was valid (edit_text, whose gate is Python-only, applied the same edit
fine). subprocess.run is mocked so these tests need no toolchain.
"""
from types import SimpleNamespace
from unittest.mock import patch

from external_llm.languages.base import (
    _filter_genuine_syntax_errors,
    _is_resolution_error,
)
from external_llm.languages.go_provider import GoSyntaxProvider
from external_llm.languages.java_provider import JavaSyntaxProvider
from external_llm.languages.kotlin_provider import KotlinSyntaxProvider
from external_llm.languages.models import LanguageId, SyntaxError_


def _fake_proc(returncode, stdout):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


# ── Go ──────────────────────────────────────────────────────────────────────

class TestGoResolutionSafety:
    def _validate(self, stdout, returncode=1):
        p = GoSyntaxProvider()
        with patch("external_llm.languages.go_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("main.go", "irrelevant — subprocess mocked")

    def test_module_resolution_error_is_not_a_syntax_error(self):
        # The EXACT failure from run_20260610: editing a file that imports a
        # non-stdlib package, compiled without module context.
        out = (
            "/var/folders/x/tmp.go:22:2: no required module provides package "
            "charm.land/bubbles/v2/help: go.mod file not found in current "
            "directory or any parent directory; see 'go help modules'\n"
        )
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_genuine_syntax_error_still_fails(self):
        out = "main.go:5:1: expected ';', found 'EOF'\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1

    def test_mixed_keeps_only_syntax_error(self):
        out = (
            "main.go:3:2: no required module provides package foo/bar\n"
            "main.go:5:1: expected ';', found 'EOF'\n"
        )
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "expected ';'" in r.errors[0].message

    def test_clean_compile_is_ok(self):
        r = self._validate("", returncode=0)
        assert r.ok is True


# ── Java ────────────────────────────────────────────────────────────────────

class TestJavaResolutionSafety:
    def _validate(self, stdout, returncode=1):
        p = JavaSyntaxProvider()
        with patch("external_llm.languages.java_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("Server.java", "irrelevant — subprocess mocked")

    def test_resolution_errors_are_ignored(self):
        out = (
            "Server.java:7: error: package org.springframework does not exist\n"
            "Server.java:12: error: cannot find symbol\n"
        )
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_genuine_syntax_error_still_fails(self):
        out = "Server.java:5: error: ';' expected\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1


# ── Kotlin ──────────────────────────────────────────────────────────────────

class TestKotlinResolutionSafety:
    def _validate(self, stdout, returncode=1):
        p = KotlinSyntaxProvider()
        with patch("external_llm.languages.kotlin_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("Server.kt", "irrelevant — subprocess mocked")

    def test_resolution_errors_are_ignored(self):
        out = "Server.kt:7:5: error: unresolved reference 'launch'\n"
        r = self._validate(out)
        assert r.ok is True
        assert not r.errors

    def test_genuine_syntax_error_still_fails(self):
        out = "Server.kt:5:1: error: expecting member declaration\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1


# ── Shared classifier (base.py) ─────────────────────────────────────────────

class TestResolutionClassifier:
    def test_go_phrases(self):
        assert _is_resolution_error("no required module provides package x", LanguageId.GO)
        assert _is_resolution_error("go.mod file not found in ...", LanguageId.GO)
        assert not _is_resolution_error("expected ';', found 'EOF'", LanguageId.GO)

    def test_java_phrases(self):
        assert _is_resolution_error("package foo does not exist", LanguageId.JAVA)
        assert _is_resolution_error("cannot find symbol", LanguageId.JAVA)
        assert not _is_resolution_error("';' expected", LanguageId.JAVA)

    def test_kotlin_phrases(self):
        assert _is_resolution_error("unresolved reference: foo", LanguageId.KOTLIN)
        assert not _is_resolution_error("expecting member declaration", LanguageId.KOTLIN)

    def test_typescript_and_javascript_have_no_phrases(self):
        # TS uses is_genuine_syntax_error (TS1xxx); JS uses node --check (pure
        # parse). Neither needs resolution classification — confirms they are
        # intentionally NOT wired into this mechanism.
        assert not _is_resolution_error("cannot find name 'process'", LanguageId.TYPESCRIPT)
        assert not _is_resolution_error("Cannot find module 'x'", LanguageId.JAVASCRIPT)

    def test_filter_drops_only_resolution_errors(self):
        errs = [
            SyntaxError_(file="f.go", line=3, col=2, message="no required module provides package x"),
            SyntaxError_(file="f.go", line=5, col=1, message="expected ';', found 'EOF'"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.GO)
        assert len(kept) == 1
        assert "expected ';'" in kept[0].message

    def test_filter_all_resolution_returns_empty(self):
        errs = [
            SyntaxError_(file="f.java", line=7, col=0, message="cannot find symbol"),
            SyntaxError_(file="f.java", line=8, col=0, message="package foo does not exist"),
        ]
        assert _filter_genuine_syntax_errors(errs, LanguageId.JAVA) == []
