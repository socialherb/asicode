"""Regression tests for typed failure classifier (Phase 1-4).

These tests verify that the classifier correctly handles:
1. Java -XDrawDiagnostics output format (Phase 1 fix)
2. Python pyright --outputjson rule codes (Phase 1 completion)
"""

import pytest

from external_llm.editor._editor_core.vm.classification import EvidenceSource, FailureType
from external_llm.editor._editor_core.vm.failure_classifier import (
    JavaFailureClassifier,
    PythonFailureClassifier,
    create_failure_classifier,
)
from external_llm.editor._editor_core.vm.models import VerifyError


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

