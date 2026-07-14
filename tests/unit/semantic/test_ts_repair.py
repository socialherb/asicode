"""Tests for TS/JS Repair System (P7)."""
from __future__ import annotations

import pytest

from external_llm.editor._editor_core.ts_vm.execution_vm.models import VerifyError
from external_llm.editor._editor_core.ts_vm.repair.failure_classifier import (
    FailureType,
    TSFailureClassifier,
)
from external_llm.editor._editor_core.ts_vm.repair.repair_planner import TSRepairPlanner
from external_llm.editor._editor_core.ts_vm.repair.repair_strategies import (
    repair_argument_mismatch,
    repair_missing_return,
    repair_syntax_error,
    repair_unknown_symbol,
)
from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer
from external_llm.languages.tree_sitter_utils import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="tree-sitter not installed"
)


@pytest.fixture
def classifier():
    return TSFailureClassifier()


@pytest.fixture
def planner():
    return TSRepairPlanner()


@pytest.fixture
def tracer():
    return TSSemanticTracer()


# ══════════════════════════════════════════════════════════════════════════════
#  FAILURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════


def test_classify_cannot_find_name(classifier):
    errors = [VerifyError(message="Cannot find name 'useState'")]
    assert classifier.classify(errors) == FailureType.UNKNOWN_SYMBOL


def test_classify_cannot_find_name_tsc_code(classifier):
    errors = [VerifyError(message="Cannot find name 'foo'", code="TS2304")]
    assert classifier.classify(errors) == FailureType.UNKNOWN_SYMBOL


def test_classify_type_mismatch(classifier):
    errors = [VerifyError(
        message="Type 'string' is not assignable to type 'number'",
        code="TS2322",
    )]
    assert classifier.classify(errors) == FailureType.TYPE_MISMATCH


def test_classify_type_mismatch_pattern(classifier):
    errors = [VerifyError(message="type X is not assignable to type Y")]
    assert classifier.classify(errors) == FailureType.TYPE_MISMATCH


def test_classify_argument_mismatch(classifier):
    errors = [VerifyError(
        message="Expected 2 arguments, but got 3",
        code="TS2554",
    )]
    assert classifier.classify(errors) == FailureType.ARGUMENT_MISMATCH


def test_classify_argument_mismatch_pattern(classifier):
    errors = [VerifyError(message="Expected 1 argument, got 0")]
    assert classifier.classify(errors) == FailureType.ARGUMENT_MISMATCH


def test_classify_missing_return(classifier):
    errors = [VerifyError(
        message="A function whose declared type is neither 'void' nor 'any' must return a value",
        code="TS2355",
    )]
    assert classifier.classify(errors) == FailureType.MISSING_RETURN


def test_classify_duplicate_identifier(classifier):
    errors = [VerifyError(message="Duplicate identifier 'foo'", code="TS2300")]
    assert classifier.classify(errors) == FailureType.DUPLICATE_IDENTIFIER


def test_classify_property_not_exist(classifier):
    errors = [VerifyError(
        message="Property 'bar' does not exist on type 'Foo'",
        code="TS2339",
    )]
    assert classifier.classify(errors) == FailureType.PROPERTY_NOT_EXIST


def test_classify_syntax_error(classifier):
    errors = [VerifyError(message="Parse error near: '{{{broken'")]
    assert classifier.classify(errors) == FailureType.SYNTAX_ERROR


def test_classify_expected_semicolon(classifier):
    errors = [VerifyError(message="Expected ';' at end of statement")]
    assert classifier.classify(errors) == FailureType.SYNTAX_ERROR


def test_classify_cannot_find_module(classifier):
    errors = [VerifyError(
        message="Cannot find module './missing'",
        code="TS2307",
    )]
    assert classifier.classify(errors) == FailureType.MISSING_IMPORT


def test_classify_unknown(classifier):
    errors = [VerifyError(message="Something completely unexpected")]
    assert classifier.classify(errors) == FailureType.UNKNOWN


def test_classify_empty(classifier):
    assert classifier.classify([]) == FailureType.UNKNOWN


def test_classify_all(classifier):
    errors = [
        VerifyError(message="Cannot find name 'foo'"),
        VerifyError(message="Type 'X' is not assignable to 'Y'"),
    ]
    types = classifier.classify_all(errors)
    assert types == [FailureType.UNKNOWN_SYMBOL, FailureType.TYPE_MISMATCH]


# ── extract helpers ──────────────────────────────────────────────────────────


def test_extract_symbol(classifier):
    error = VerifyError(message="Cannot find name 'useState'")
    assert classifier.extract_symbol(error) == "useState"


def test_extract_symbol_with_quotes(classifier):
    error = VerifyError(message="Cannot find name 'React'")
    assert classifier.extract_symbol(error) == "React"


def test_extract_symbol_none(classifier):
    error = VerifyError(message="Something else entirely")
    assert classifier.extract_symbol(error) is None


def test_extract_expected_args(classifier):
    error = VerifyError(message="Expected 2 arguments, but got 3")
    assert classifier.extract_expected_args(error) == 2


def test_extract_expected_args_none(classifier):
    error = VerifyError(message="Type mismatch")
    assert classifier.extract_expected_args(error) is None


# ══════════════════════════════════════════════════════════════════════════════
#  REPAIR STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════


# ── unknown symbol → import ──────────────────────────────────────────────────


def test_strategy_unknown_symbol_usestate(tracer):
    code = "function App() { useState(0) }"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Cannot find name 'useState'")

    ops = repair_unknown_symbol(code, error, module)
    assert ops is not None
    assert len(ops) == 1
    assert ops[0].kind.value == "INSERT_IMPORT"
    assert "useState" in ops[0].payload["statement"]
    assert "react" in ops[0].payload["statement"]


def test_strategy_unknown_symbol_react_default(tracer):
    code = "const el = React.createElement('div')"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Cannot find name 'React'")

    ops = repair_unknown_symbol(code, error, module)
    assert ops is not None
    assert "import React from 'react'" in ops[0].payload["statement"]


def test_strategy_unknown_symbol_express(tracer):
    code = "const app = express()"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Cannot find name 'express'")

    ops = repair_unknown_symbol(code, error, module)
    assert ops is not None
    assert "import express from 'express'" in ops[0].payload["statement"]


def test_strategy_unknown_symbol_unmapped(tracer):
    code = "const x = unknownLib()"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Cannot find name 'unknownLib'")

    ops = repair_unknown_symbol(code, error, module)
    assert ops is None  # no mapping → can't repair


def test_strategy_unknown_symbol_already_imported(tracer):
    code = """\
import { useState } from 'react'
function App() { useState(0) }
"""
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Cannot find name 'useState'")

    ops = repair_unknown_symbol(code, error, module)
    assert ops is None  # already imported


# ── syntax error ─────────────────────────────────────────────────────────────


def test_strategy_syntax_semicolon(tracer):
    code = "const x = 1\nconst y = 2"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Expected ';' at end", line=1)

    ops = repair_syntax_error(code, error, module)
    assert ops is not None
    # Should produce raw code with semicolon added
    raw = ops[0].payload.get("__raw_code__")
    assert raw is not None
    assert "const x = 1;" in raw


def test_strategy_syntax_unclosed_brace(tracer):
    code = "function f() {\n  return 1\n"
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(message="Expected '}'", line=2)

    ops = repair_syntax_error(code, error, module)
    assert ops is not None
    raw = ops[0].payload.get("__raw_code__")
    assert raw is not None
    assert "}" in raw


# ── missing return ───────────────────────────────────────────────────────────


def test_strategy_missing_return(tracer):
    code = """\
function compute(): number {
  const x = 42
}
"""
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(
        message="must return a value", line=2, code="TS2355")

    ops = repair_missing_return(code, error, module)
    assert ops is not None
    assert ops[0].kind.value == "INSERT_STATEMENT"
    assert "return" in ops[0].payload["statement"]
    assert ops[0].payload["anchor"] == "compute"
    assert ops[0].payload["position"] == "end"


# ── argument mismatch ────────────────────────────────────────────────────────


def test_strategy_argument_mismatch_zero_args(tracer):
    code = """\
function run() {
  init(1, 2, 3)
}
"""
    module = tracer.analyze_core(code, "a.ts")
    error = VerifyError(
        message="Expected 0 arguments, but got 3", line=2)

    ops = repair_argument_mismatch(code, error, module)
    assert ops is not None
    assert ops[0].kind.value == "UPDATE_CALL"
    assert ops[0].payload["callee"] == "init"
    assert ops[0].payload["new_args"] == ""


# ══════════════════════════════════════════════════════════════════════════════
#  REPAIR PLANNER
# ══════════════════════════════════════════════════════════════════════════════


def test_planner_unknown_symbol(planner):
    code = "function App() { useState(0) }"
    errors = [VerifyError(message="Cannot find name 'useState'")]

    plan = planner.plan(code, errors)
    assert plan is not None
    assert plan.failure_type == FailureType.UNKNOWN_SYMBOL
    assert len(plan.ops) == 1
    assert not plan.is_raw


def test_planner_syntax_error(planner):
    code = "const x = 1\nconst y = 2"
    errors = [VerifyError(message="Expected ';'", line=1)]

    plan = planner.plan(code, errors)
    assert plan is not None
    assert plan.failure_type == FailureType.SYNTAX_ERROR
    assert plan.is_raw
    assert ";" in plan.raw_code


def test_planner_no_strategy(planner):
    errors = [VerifyError(message="Something we can't handle")]
    plan = planner.plan("const x = 1", errors)
    assert plan is None


def test_planner_empty_errors(planner):
    plan = planner.plan("const x = 1", [])
    assert plan is None


# ══════════════════════════════════════════════════════════════════════════════
#  VM INTEGRATION (repair loop)
# ══════════════════════════════════════════════════════════════════════════════


from external_llm.editor._editor_core.ts_vm.execution_vm.vm import TSExecutionVM
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveKind, PrimitiveOp


@pytest.fixture
def vm():
    return TSExecutionVM(use_tsc=False, use_eslint=False)


def test_vm_still_works_after_p7_upgrade(vm):
    """P6 tests should still pass with new repair planner."""
    code = "function foo() { return 1 }"
    result = vm.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return 2"},
        ),
    ])
    assert result.success
    assert "return 2" in result.code


def test_vm_rollback_still_works(vm):
    code = "function foo() { return 1 }"
    result = vm.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "nonexistent", "body": "x"},
        ),
    ])
    assert not result.success
    assert result.rolled_back
    assert "return 1" in result.code


def test_vm_complex_refactor_still_works(vm):
    code = """\
function fetchData() {
  return fetch('/api')
}

function unused() {
  return null
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { cache } from './cache'"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "fetchData",
                "body": "return cache.get('/api') || fetch('/api')",
            },
        ),
        PrimitiveOp(
            kind=PrimitiveKind.DELETE_NODE,
            payload={"name": "unused"},
        ),
    ]
    result = vm.execute(code, "service.ts", ops)
    assert result.success
    assert "import { cache }" in result.code
    assert "unused" not in result.code
