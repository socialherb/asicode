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

    def test_lone_cannot_find_symbol_is_a_genuine_typo(self):
        # A local typo (`total += valeu;`) with NO failed import: javac emits
        # ONLY "cannot find symbol". Without the "package … does not exist"
        # context this is a real error that must gate the edit — otherwise the
        # typo silently passes (the exact C/C++ gate-bypass class, now Java).
        out = (
            "Server.java:12: error: cannot find symbol\n"
            "  symbol:   variable valeu\n"
        )
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "cannot find symbol" in r.errors[0].message

    def test_public_class_filename_artifact_does_not_mask_typo(self):
        # The isolated temp file is randomly named, so a public class ALWAYS
        # draws "class X is public, should be declared in a file named X.java".
        # That artifact is unconditionally dropped but must NOT count as
        # resolution context — the co-occurring "cannot find symbol" typo must
        # still gate. (Empirically confirmed with javac 17: public-class typos
        # emit both lines with no "does not exist".)
        out = (
            "Server.java:1: error: class Server is public, should be declared "
            "in a file named Server.java\n"
            "Server.java:12: error: cannot find symbol\n"
        )
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "cannot find symbol" in r.errors[0].message

    def test_syntax_gate_writes_to_a_real_output_dir_not_devnull(self):
        # Regression: ``javac -d /dev/null`` aborts with "not a directory"
        # BEFORE compiling (rc=2, no file:line: diagnostic the regex matches),
        # so the gate returned ok=True for EVERY source — a silent fail-open
        # that disabled Java syntax validation entirely. The -d target must be
        # a real (temp) directory.
        import os
        captured = {}

        def _capture(cmd, *a, **k):
            d_target = cmd[cmd.index("-d") + 1]
            captured["d_target"] = d_target
            # Must be a real directory *at compile time* (temp dir is cleaned up
            # in the finally block, so check it here while it still exists).
            captured["is_dir"] = os.path.isdir(d_target)
            return _fake_proc(0, "")

        p = JavaSyntaxProvider()
        with patch("external_llm.languages.java_provider.subprocess.run",
                   side_effect=_capture):
            p.validate_syntax("Server.java", "public class Server {}")
        assert captured["d_target"] != os.devnull
        assert captured["is_dir"] is True


# ── Kotlin ──────────────────────────────────────────────────────────────────

class TestKotlinResolutionSafety:
    def _validate(self, stdout, returncode=1, content="irrelevant — subprocess mocked"):
        p = KotlinSyntaxProvider()
        with patch("external_llm.languages.kotlin_provider.subprocess.run",
                   return_value=_fake_proc(returncode, stdout)):
            return p.validate_syntax("Server.kt", content)

    def test_resolution_errors_from_failed_import_are_ignored(self):
        """A coroutine import that can't resolve in the classpath-less temp-file
        compile is environmental noise.  Real kotlinc reports an unresolved
        reference ON THE IMPORT LINE (the unresolved package segment) plus the
        cascading usage error — both must be dropped so the valid edit is not
        rolled back."""
        content = (
            "import kotlinx.coroutines.GlobalScope\n"
            "import kotlinx.coroutines.launch\n"
            "\n"
            "fun main() {\n"
            "    GlobalScope.launch { }\n"
            "}\n"
        )
        out = (
            "Server.kt:1:8: error: unresolved reference 'kotlinx'\n"
            "Server.kt:2:8: error: unresolved reference 'kotlinx'\n"
            "Server.kt:5:5: error: unresolved reference 'GlobalScope'\n"
        )
        r = self._validate(out, content=content)
        assert r.ok is True
        assert not r.errors

    def test_bare_unresolved_reference_is_genuine_error(self):
        """A bare ``unresolved reference`` with NO import failure is a genuine
        local typo (``total += valeu``), not a cascade — it must gate the edit.
        Previously this was unconditionally filtered, silently disabling the
        syntax gate (same defect class as the Java 'cannot find symbol' /
        g++ 'was not declared in this scope' fix)."""
        content = (
            "fun main() {\n"
            "    var total = 0\n"
            "    total += valeu\n"
            "    println(total)\n"
            "}\n"
        )
        out = "Server.kt:3:14: error: unresolved reference 'valeu'\n"
        r = self._validate(out, content=content)
        assert r.ok is False
        assert len(r.errors) == 1
        assert "unresolved reference" in r.errors[0].message

    def test_genuine_syntax_error_still_fails(self):
        out = "Server.kt:5:1: error: expecting member declaration\n"
        r = self._validate(out)
        assert r.ok is False
        assert len(r.errors) == 1

    def test_syntax_gate_uses_real_output_dir_not_script_mode(self):
        # Regression: ``kotlinc -script`` only works for ``.kts`` files — for a
        # ``.kt`` file kotlinc aborts with "unrecognized script type" (rc=1, no
        # file:line: diagnostic the regex matches), so the gate returned ok=True
        # for EVERY source — a silent fail-open that disabled Kotlin syntax
        # validation entirely (same defect class as the Java ``-d /dev/null``
        # bug). The command must NOT pass ``-script`` and must point ``-d`` at a
        # real (temp) directory.
        import os
        captured = {}

        def _capture(cmd, *a, **k):
            captured["has_script"] = "-script" in cmd
            if "-d" in cmd:
                d_target = cmd[cmd.index("-d") + 1]
                captured["d_target"] = d_target
                captured["is_dir"] = os.path.isdir(d_target)
            return _fake_proc(0, "")

        p = KotlinSyntaxProvider()
        with patch("external_llm.languages.kotlin_provider.subprocess.run",
                   side_effect=_capture):
            p.validate_syntax("Server.kt", "fun main() {}")
        assert captured["has_script"] is False
        assert captured["d_target"] != os.devnull
        assert captured["is_dir"] is True


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
        # "cannot find symbol" IS a cascade here: "package foo does not exist"
        # supplies the resolution context, so both drop.
        errs = [
            SyntaxError_(file="f.java", line=7, col=0, message="cannot find symbol"),
            SyntaxError_(file="f.java", line=8, col=0, message="package foo does not exist"),
        ]
        assert _filter_genuine_syntax_errors(errs, LanguageId.JAVA) == []

    def test_java_lone_cannot_find_symbol_is_kept(self):
        # No "does not exist" context ⇒ genuine typo, must survive the filter.
        errs = [
            SyntaxError_(file="f.java", line=12, col=0, message="cannot find symbol"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.JAVA)
        assert len(kept) == 1
        assert "cannot find symbol" in kept[0].message

    def test_java_public_class_artifact_is_not_resolution_context(self):
        # "is public, should be declared" is dropped (isolated-compile artifact)
        # but does NOT grant resolution context, so the typo is kept.
        errs = [
            SyntaxError_(file="f.java", line=1, col=0,
                         message="class Foo is public, should be declared in a file named Foo.java"),
            SyntaxError_(file="f.java", line=12, col=0, message="cannot find symbol"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.JAVA)
        assert len(kept) == 1
        assert "cannot find symbol" in kept[0].message

    # ── Kotlin: line-based context (no phrase-level disambiguator) ──────────

    def test_kotlin_bare_unresolved_reference_is_kept(self):
        # No resolution context ⇒ genuine typo, must survive the filter. Kotlin
        # has no _RESOLUTION_CONTEXT_PHRASES, so the default (None) context is
        # always False — the provider supplies line-based context instead.
        errs = [
            SyntaxError_(file="f.kt", line=3, col=14,
                         message="unresolved reference 'valeu'"),
        ]
        kept = _filter_genuine_syntax_errors(errs, LanguageId.KOTLIN)
        assert len(kept) == 1
        assert "unresolved reference" in kept[0].message

    def test_kotlin_unresolved_reference_dropped_with_import_context(self):
        # Provider detected an import-line failure → has_resolution_context=True
        # → the co-occurring usage error is cascade noise, dropped.
        errs = [
            SyntaxError_(file="f.kt", line=1, col=8,
                         message="unresolved reference 'kotlinx'"),
            SyntaxError_(file="f.kt", line=5, col=5,
                         message="unresolved reference 'GlobalScope'"),
        ]
        kept = _filter_genuine_syntax_errors(
            errs, LanguageId.KOTLIN, has_resolution_context=True,
        )
        assert kept == []

    def test_kotlin_genuine_syntax_error_kept_with_import_context(self):
        # Even when an import failed, a genuine SYNTAX error must still gate.
        errs = [
            SyntaxError_(file="f.kt", line=1, col=8,
                         message="unresolved reference 'kotlinx'"),
            SyntaxError_(file="f.kt", line=5, col=1,
                         message="expecting member declaration"),
        ]
        kept = _filter_genuine_syntax_errors(
            errs, LanguageId.KOTLIN, has_resolution_context=True,
        )
        assert len(kept) == 1
        assert "expecting member declaration" in kept[0].message
