"""Regression tests for typed failure classifier (Phase 1-4).

These tests verify that the classifier correctly handles:
1. Java -XDrawDiagnostics output format (Phase 1 fix)
2. Python pyright --outputjson rule codes (Phase 1 completion)
3. Verifier output parsing (Phase 1)
4. Kotlin kotlinc output parsing (locale-independent verification)
"""
import os
import shutil
import tempfile

import pytest
from external_llm.editor._editor_core.vm.failure_classifier import (
    JavaFailureClassifier,
    PythonFailureClassifier,
    create_failure_classifier,
)
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.classification import FailureType, EvidenceSource
from external_llm.editor._editor_core.vm.verifier import JavaVerifier, KotlinVerifier


class TestJavaFailureClassifierRegression:
    """Test Java classifier handles -XDrawDiagnostics format correctly."""

    def test_cant_resolve_location(self):
        """compiler.err.cant.resolve.location → UNKNOWN_SYMBOL (regression test)."""
        classifier = JavaFailureClassifier()
        error = VerifyError(
            message="variable: x",
            line=5,
            column=9,
            code="compiler.err.cant.resolve.location",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.UNKNOWN_SYMBOL
        assert result.source == EvidenceSource.ERROR_CODE

    def test_doesnt_exist(self):
        """compiler.err.doesnt.exist → MISSING_IMPORT."""
        classifier = JavaFailureClassifier()
        error = VerifyError(
            message="package foo.bar",
            line=1,
            column=1,
            code="compiler.err.doesnt.exist",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.MISSING_IMPORT
        assert result.source == EvidenceSource.ERROR_CODE

    def test_expected(self):
        """compiler.err.expected → SYNTAX_ERROR."""
        classifier = JavaFailureClassifier()
        error = VerifyError(
            message="';'",
            line=3,
            column=10,
            code="compiler.err.expected",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.SYNTAX_ERROR
        assert result.source == EvidenceSource.ERROR_CODE

    def test_missing_return_stmt(self):
        """compiler.err.missing.ret.stmt → MISSING_RETURN."""
        classifier = JavaFailureClassifier()
        error = VerifyError(
            message="",
            line=10,
            column=1,
            code="compiler.err.missing.ret.stmt",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.MISSING_RETURN
        assert result.source == EvidenceSource.ERROR_CODE

    def test_legacy_format_fallback(self):
        """Legacy format (without -XDrawDiagnostics) still works via keyword matching."""
        classifier = JavaFailureClassifier()
        error = VerifyError(
            message="cannot find symbol\n  symbol: variable x",
            line=5,
            column=9,
            code="ERROR",  # Generic code for legacy format
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.UNKNOWN_SYMBOL
        assert result.source == EvidenceSource.MESSAGE_FALLBACK


class TestJavaVerifierParser:
    """Test _parse_javac_output handles actual javac output formats."""

    def test_diagnostic_with_arguments(self):
        """Parse diagnostic key with arguments (trailing colon + message)."""
        verifier = JavaVerifier()
        output = "Check.java:5:9: compiler.err.cant.resolve.location: variable: x"
        errors = verifier._parse_javac_output(output)
        assert len(errors) == 1
        assert errors[0].code == "compiler.err.cant.resolve.location"
        assert errors[0].line == 5
        assert errors[0].column == 9
        assert "variable: x" in errors[0].message

    def test_diagnostic_without_arguments(self):
        """Parse diagnostic key without arguments (no trailing colon)."""
        verifier = JavaVerifier()
        output = "Check.java:3:5: compiler.err.missing.ret.stmt"
        errors = verifier._parse_javac_output(output)
        assert len(errors) == 1
        assert errors[0].code == "compiler.err.missing.ret.stmt"
        assert errors[0].line == 3
        assert errors[0].column == 5
        # When no arguments, message should be the key itself
        assert errors[0].message == "compiler.err.missing.ret.stmt"

    def test_diagnostic_unreachable_stmt(self):
        """Parse compiler.err.unreachable.stmt (no arguments)."""
        verifier = JavaVerifier()
        output = "Check2.java:4:13: compiler.err.unreachable.stmt"
        errors = verifier._parse_javac_output(output)
        assert len(errors) == 1
        assert errors[0].code == "compiler.err.unreachable.stmt"
        assert errors[0].line == 4
        assert errors[0].column == 13


class TestPythonFailureClassifierRegression:
    """Test Python classifier handles pyright rule codes correctly."""

    def test_pyright_undefined_variable(self):
        """reportUndefinedVariable → MISSING_VARIABLE (Layer B)."""
        classifier = PythonFailureClassifier()
        error = VerifyError(
            message="'x' is not defined",
            line=4,
            column=1,
            code="reportUndefinedVariable",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.MISSING_VARIABLE
        assert result.source == EvidenceSource.ERROR_CODE

    def test_pyright_missing_imports(self):
        """reportMissingImports → MISSING_IMPORT (Layer B)."""
        classifier = PythonFailureClassifier()
        error = VerifyError(
            message="Cannot find module 'foo'",
            line=1,
            column=1,
            code="reportMissingImports",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.MISSING_IMPORT
        assert result.source == EvidenceSource.ERROR_CODE

    def test_pyright_invalid_syntax(self):
        """reportInvalidSyntax → SYNTAX_ERROR (Layer B)."""
        classifier = PythonFailureClassifier()
        error = VerifyError(
            message="Invalid syntax",
            line=2,
            column=5,
            code="reportInvalidSyntax",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.SYNTAX_ERROR
        assert result.source == EvidenceSource.ERROR_CODE

    def test_compile_error_fallback(self):
        """compile() error codes still work (backward compatibility)."""
        classifier = PythonFailureClassifier()
        error = VerifyError(
            message="undefined name 'x'",
            line=3,
            column=1,
            code="E0602",
        )
        result = classifier.classify_typed([error])
        assert result.type == FailureType.MISSING_VARIABLE
        assert result.source == EvidenceSource.ERROR_CODE


class TestClassifierFactory:
    """Test factory function returns correct classifier."""

    def test_java_classifier(self):
        classifier = create_failure_classifier("java")
        assert isinstance(classifier, JavaFailureClassifier)

    def test_python_classifier(self):
        classifier = create_failure_classifier("python")
        assert isinstance(classifier, PythonFailureClassifier)

    def test_unsupported_language(self):
        with pytest.raises(ValueError, match="No failure classifier"):
            create_failure_classifier("rust")


class TestKotlinVerifierParser:
    """Test _parse_kotlinc_output handles actual kotlinc output formats.

    kotlinc is locale-independent (always outputs English diagnostics),
    so these tests verify the parser works regardless of system locale.
    """

    def test_parse_unresolved_reference(self):
        """Parse kotlinc error for undefined symbol (real kotlinc output)."""
        verifier = KotlinVerifier()
        output = "Bad.kt:1:22: error: unresolved reference 'undefinedSymbol'."
        errors = verifier._parse_kotlinc_output(output)
        assert len(errors) == 1
        assert errors[0].line == 1
        assert errors[0].column == 22
        assert "unresolved reference" in errors[0].message
        assert errors[0].code == "ERROR"

    def test_parse_type_mismatch(self):
        """Parse kotlinc error for type mismatch."""
        verifier = KotlinVerifier()
        output = "Main.kt:5:10: error: type mismatch: inferred type is String but Int was expected."
        errors = verifier._parse_kotlinc_output(output)
        assert len(errors) == 1
        assert errors[0].line == 5
        assert errors[0].column == 10
        assert "type mismatch" in errors[0].message

    def test_parse_multiple_errors(self):
        """Parse multiple kotlinc errors from single output."""
        verifier = KotlinVerifier()
        output = """Bad.kt:1:22: error: unresolved reference 'foo'.
Bad.kt:2:10: error: unresolved reference 'bar'.
fun main() { val x = foo + bar }
                     ^^^    ^^^"""
        errors = verifier._parse_kotlinc_output(output)
        assert len(errors) == 2
        assert errors[0].line == 1
        assert errors[1].line == 2


def _kotlinc_available() -> bool:
    """Check if kotlinc is available on the system."""
    return shutil.which("kotlinc") is not None


@pytest.mark.skipif(not _kotlinc_available(), reason="kotlinc not installed")
class TestKotlinVerifierLive:
    """Live tests that invoke real kotlinc.

    These verify that:
    1. The -J-Duser.language=en flag doesn't break invocation
    2. kotlinc output is locale-independent (always English)
    3. The parser correctly handles real kotlinc stderr
    """

    def test_kotlin_verify_valid_code(self):
        """Valid Kotlin code must pass verification (rc=0)."""
        verifier = KotlinVerifier()
        ok, errors = verifier.verify("fun main() { val x = 1 }\n")
        assert ok is True
        assert errors == []

    def test_kotlin_verify_catches_undefined_symbol(self):
        """Undefined symbol must fail verification with parsed errors."""
        verifier = KotlinVerifier()
        ok, errors = verifier.verify("fun main() { val x = undefinedSymbol }\n")
        assert ok is False
        assert len(errors) >= 1
        # Verify the error was parsed correctly (not swallowed by fallback)
        assert any("unresolved reference" in e.message for e in errors)

    def test_kotlin_verify_locale_independent(self):
        """Verify that kotlinc output is the same across locales.

        This test runs kotlinc with different locale settings and verifies
        the output is identical (kotlinc is locale-independent).
        """
        import subprocess

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = os.path.join(tmp_dir, "Bad.kt")
            with open(bad_file, "w") as f:
                f.write("fun main() { val x = undefinedSymbol }\n")

            # Run with C locale
            env_c = os.environ.copy()
            env_c["LC_ALL"] = "C"
            env_c["LANG"] = "C"
            proc_c = subprocess.run(
                ["kotlinc", bad_file],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir, env=env_c,
            )

            # Run with Korean locale
            env_ko = os.environ.copy()
            env_ko["LC_ALL"] = "ko_KR.UTF-8"
            env_ko["LANG"] = "ko_KR.UTF-8"
            proc_ko = subprocess.run(
                ["kotlinc", bad_file],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir, env=env_ko,
            )

            # Run with -J-Duser.language=en flag
            proc_j = subprocess.run(
                ["kotlinc", "-J-Duser.language=en", bad_file],
                capture_output=True, text=True, timeout=30,
                cwd=tmp_dir,
            )

            # All three should produce the same error line
            def extract_error_line(output: str) -> str:
                for line in output.split("\n"):
                    if "error:" in line and "Bad.kt:" in line:
                        return line.strip()
                return ""

            error_c = extract_error_line(proc_c.stderr)
            error_ko = extract_error_line(proc_ko.stderr)
            error_j = extract_error_line(proc_j.stderr)

            # kotlinc is locale-independent: all outputs should be identical
            assert error_c == error_ko == error_j
            assert "error:" in error_c  # English, not "오류:"
