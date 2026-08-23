"""RED→GREEN: close remaining coverage gaps in external_llm/editor/_editor_core/vm/.

Branches the earlier vm test files left open:

- failure_classifier.py: empty-error lists (classify/classify_typed), the
  tree-sitter Layer A path (ERROR node, MISSING node with FixHint), regex-only
  matches in both the typed and non-typed paths, UNKNOWN fallbacks, and
  position-based symbol extraction.
- repair_registry.py: register()/get() on the REAL class (tool_registry tests
  substitute a fake RepairRegistry, so these methods were never exercised).
- repair_strategies.py: out-of-range error.line guards and paren-mismatch
  guards in the shared repair helpers, plus the Go argument-mismatch
  zero-value fill and the Go case-correction path guards.
"""

from __future__ import annotations

from external_llm.editor._editor_core.vm.classification import (
    Classification,
    EvidenceSource,
    FailureType,
)
from external_llm.editor._editor_core.vm.failure_classifier import (
    PythonFailureClassifier,
)
from external_llm.editor._editor_core.vm.models import VerifyError
from external_llm.editor._editor_core.vm.repair_registry import RepairRegistry
from external_llm.editor._editor_core.vm.repair_strategies import (
    _trim_call_arguments,
    go_repair_argument_mismatch,
    go_repair_syntax_error,
    go_repair_type_mismatch,
    go_repair_unknown_symbol,
    java_repair_duplicate_identifier,
    java_repair_syntax_error,
    py_repair_argument_mismatch,
    py_repair_missing_return,
    py_repair_syntax_error,
)
from external_llm.languages.tree_sitter_utils import SyntaxErrorNode

# ── helpers (mirror tests/unit/vm/test_repair_strategies.py) ───────────────


def _cls(symbol=None, ftype=FailureType.UNKNOWN_SYMBOL):
    return Classification(type=ftype, source=EvidenceSource.NONE, symbol=symbol)


def _err(message, line=None):
    return VerifyError(message=message, line=line)


# ── failure_classifier.py ──────────────────────────────────────────────────


class TestClassifierEmptyAndFallbacks:
    """Empty-error lists + non-typed and typed fallback paths."""

    def test_classify_empty_errors_unknown(self):
        assert PythonFailureClassifier().classify([]) is FailureType.UNKNOWN

    def test_classify_typed_empty_errors_none_evidence(self):
        result = PythonFailureClassifier().classify_typed([])
        assert result.type is FailureType.UNKNOWN
        assert result.source is EvidenceSource.NONE

    def test_classify_non_typed_error_code_map(self):
        # Non-typed classify() reaches the error-code map branch (L79).
        cls = PythonFailureClassifier()
        assert cls.classify([VerifyError(message="", code="E0602")]) is FailureType.MISSING_VARIABLE

    def test_classify_non_typed_regex_match(self):
        # Message that hits ONLY a regex pattern (no keyword): "takes N positional".
        cls = PythonFailureClassifier()
        err = VerifyError(message="f() takes 2 positional arguments but 1 was given")
        assert cls.classify([err]) is FailureType.ARGUMENT_MISMATCH

    def test_classify_non_typed_no_match_unknown(self):
        cls = PythonFailureClassifier()
        assert cls.classify([VerifyError(message="garbage diagnostic")]) is FailureType.UNKNOWN

    def test_classify_typed_regex_match(self):
        # Typed path: keyword loop misses, regex loop matches (L125-128).
        cls = PythonFailureClassifier()
        err = VerifyError(message="f() takes 2 positional arguments but 1 was given", line=1, column=1)
        result = cls.classify_typed([err])
        assert result.type is FailureType.ARGUMENT_MISMATCH
        assert result.source is EvidenceSource.MESSAGE_FALLBACK

    def test_classify_typed_no_match_unknown(self):
        cls = PythonFailureClassifier()
        result = cls.classify_typed([VerifyError(message="garbage diagnostic")])
        assert result.type is FailureType.UNKNOWN
        assert result.source is EvidenceSource.NONE


class TestClassifierTreeSitterLayerA:
    """Layer A: structural syntax-error detection via find_error_nodes."""

    def test_layer_a_error_node_no_fix_hint(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(
            tsu,
            "find_error_nodes",
            lambda code, lang: [SyntaxErrorNode(kind="ERROR", missing_token="", line=0, column=0)],
        )
        cls = PythonFailureClassifier()
        result = cls.classify_typed(
            [VerifyError(message="x = ", line=1, column=1)],
            code="x = \n",
            language="python",
        )
        assert result.type is FailureType.SYNTAX_ERROR
        assert result.source is EvidenceSource.TREE_SITTER
        assert result.fix_hint is None

    def test_layer_a_missing_node_fix_hint(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(
            tsu,
            "find_error_nodes",
            lambda code, lang: [SyntaxErrorNode(kind="MISSING", missing_token=";", line=1, column=1)],
        )
        cls = PythonFailureClassifier()
        result = cls.classify_typed(
            [VerifyError(message="x = 1", line=2, column=2)],
            code="x = 1",
            language="python",
        )
        assert result.type is FailureType.SYNTAX_ERROR
        assert result.source is EvidenceSource.TREE_SITTER
        assert result.fix_hint is not None
        assert result.fix_hint.kind == "insert_token"
        assert result.fix_hint.token == ";"
        # tree-sitter points are 0-based → FixHint line/column are 1-based
        assert result.fix_hint.line == 2
        assert result.fix_hint.column == 2

    def test_position_based_symbol_extraction(self, monkeypatch):
        import external_llm.languages.tree_sitter_utils as tsu

        monkeypatch.setattr(tsu, "find_error_nodes", lambda code, lang: [])
        monkeypatch.setattr(
            tsu,
            "extract_symbol_at_position",
            lambda code, lang, line, col: "foo",
        )
        cls = PythonFailureClassifier()
        err = VerifyError(message="'foo' is not defined", line=3, column=7)
        result = cls.classify_typed([err], code="x = foo\n", language="python")
        # keyword match → MESSAGE_FALLBACK; symbol comes from the position lookup
        assert result.type is FailureType.MISSING_VARIABLE
        assert result.source is EvidenceSource.MESSAGE_FALLBACK
        assert result.symbol == "foo"


# ── repair_registry.py ─────────────────────────────────────────────────────


class TestRepairRegistryRealClass:
    """register()/get() on the real registry (tool_registry tests fake it)."""

    def test_register_and_get(self):
        reg = RepairRegistry("python")

        def _custom_strategy(code, error, classification):
            return None

        reg.register(FailureType.UNUSED_IMPORT, _custom_strategy)
        assert reg.get(FailureType.UNUSED_IMPORT) is _custom_strategy

    def test_get_missing_returns_none(self):
        reg = RepairRegistry("python")
        assert reg.get(FailureType.UNUSED_IMPORT) is None


# ── repair_strategies.py ───────────────────────────────────────────────────


class TestSharedGuardBranches:
    """Out-of-range error lines and paren-mismatch guards in shared helpers."""

    def test_trim_call_arguments_unclosed_paren(self):
        # "(" present but no ")" on the line → cannot trim.
        assert _trim_call_arguments(["foo(a\n"], 0) is None

    def test_argument_mismatch_line_out_of_range(self):
        assert (
            py_repair_argument_mismatch(
                "f(a)\n",
                _err("takes 1 positional argument but 2 were given", 5),
                None,
            )
            is None
        )

    def test_py_syntax_error_line_out_of_range(self):
        assert py_repair_syntax_error("x = 1\n", _err("expected ':'", 5), None) is None

    def test_missing_return_header_is_last_line(self):
        # Header at EOF → no body to insert into.
        assert py_repair_missing_return("def f():", _err("missing return", 1), None) is None

    def test_missing_return_body_not_indented(self):
        # Body line at column 0 → not a real body → cannot determine indent.
        assert (
            py_repair_missing_return(
                "def f():\nx = 1",
                _err("missing return", 2),
                None,
            )
            is None
        )

    def test_missing_semicolon_line_out_of_range(self):
        assert java_repair_syntax_error("x;\n", _err("';' expected", 5), None) is None

    def test_duplicate_identifier_line_out_of_range(self):
        assert (
            java_repair_duplicate_identifier(
                "class Foo {}\n",
                _err("duplicate class", 5),
                None,
            )
            is None
        )

    def test_duplicate_identifier_no_pattern_match(self):
        # Marker present, line in range, but no "class X" pattern on the line.
        assert (
            java_repair_duplicate_identifier(
                "Foo x;\n",
                _err("duplicate class", 1),
                None,
            )
            is None
        )


class TestGoRepairGuardBranches:
    """Go-specific guard branches left open by the earlier tests."""

    def test_unknown_symbol_no_line_after_import_lookup(self):
        # symbol not in the import map and error carries no line → Path 2 guard.
        assert (
            go_repair_unknown_symbol(
                "x := 1\n",
                _err("undefined: myvar"),
                _cls("myvar"),
            )
            is None
        )

    def test_unknown_symbol_line_out_of_range(self):
        assert (
            go_repair_unknown_symbol(
                "x := 1\n",
                _err("undefined: myvar", 5),
                _cls("myvar"),
            )
            is None
        )

    def test_syntax_error_line_out_of_range(self):
        # Semicolon helper returns None (L357), then the '{' branch hits the
        # same out-of-range guard (L594).
        assert (
            go_repair_syntax_error(
                "x := 1\n",
                _err("expected newline", 5),
                None,
            )
            is None
        )

    def test_argument_mismatch_line_out_of_range(self):
        assert (
            go_repair_argument_mismatch(
                "foo(a)\n",
                _err("too many arguments", 5),
                None,
            )
            is None
        )

    def test_argument_mismatch_not_enough_no_parens(self):
        assert (
            go_repair_argument_mismatch(
                "foo\n",
                _err("not enough arguments in call to foo", 1),
                None,
            )
            is None
        )

    def test_argument_mismatch_not_enough_unclosed_paren(self):
        assert (
            go_repair_argument_mismatch(
                "foo(a\n",
                _err("not enough arguments in call to foo", 1),
                None,
            )
            is None
        )

    def test_type_mismatch_line_out_of_range(self):
        assert (
            go_repair_type_mismatch(
                "x := 1\n",
                _err("cannot use t as string value in assignment", 5),
                None,
            )
            is None
        )
