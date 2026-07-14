"""Tests for TS/JS Execution VM (P6)."""
from __future__ import annotations

import pytest

from external_llm.editor._editor_core.ts_vm.execution_vm.ast_rewriter import ASTRewriter
from external_llm.editor._editor_core.ts_vm.execution_vm.models import VMResult
from external_llm.editor._editor_core.ts_vm.execution_vm.rollback_manager import RollbackManager
from external_llm.editor._editor_core.ts_vm.execution_vm.verifier import TSVerifier
from external_llm.editor._editor_core.ts_vm.execution_vm.vm import TSExecutionVM
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveKind, PrimitiveOp
from external_llm.languages.tree_sitter_utils import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="tree-sitter not installed"
)


@pytest.fixture
def vm():
    # Disable tsc/eslint for unit tests (no external deps)
    return TSExecutionVM(use_tsc=False, use_eslint=False)


@pytest.fixture
def rewriter():
    return ASTRewriter()


@pytest.fixture
def verifier():
    return TSVerifier(use_tsc=False, use_eslint=False)


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLBACK MANAGER
# ══════════════════════════════════════════════════════════════════════════════


def test_rollback_basic():
    rm = RollbackManager("original")
    rm.push("v1")
    rm.push("v2")

    assert rm.last() == "v2"
    assert rm.depth == 3
    assert rm.rollback() == "original"


def test_rollback_one():
    rm = RollbackManager("original")
    rm.push("v1")
    rm.push("v2")

    assert rm.rollback_one() == "v1"
    assert rm.last() == "v1"


def test_rollback_one_at_original():
    rm = RollbackManager("original")
    assert rm.rollback_one() == "original"


# ══════════════════════════════════════════════════════════════════════════════
#  AST REWRITER
# ══════════════════════════════════════════════════════════════════════════════


def test_ast_rewriter_valid_mutation(rewriter):
    code = "function foo() { return 1 }"
    result = rewriter.apply(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return 2"},
        ),
    ])

    assert result.success
    assert "return 2" in result.code


def test_ast_rewriter_rejects_broken_ast(rewriter):
    """If a primitive produces unparseable code, the rewriter rejects it."""
    code = "function foo() { return 1 }"
    # Inject intentionally broken body
    result = rewriter.apply(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return {{{"},
        ),
    ])

    # The rewriter should reject this
    assert not result.success
    assert "return 1" in result.code  # original preserved


def test_ast_rewriter_multi_op(rewriter):
    code = """\
function greet() {
  return 'hello'
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { log } from './log'"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "greet", "body": "log('hi')\nreturn 'hi'"},
        ),
    ]

    result = rewriter.apply(code, "a.ts", ops)
    assert result.success
    assert "import { log }" in result.code
    assert "log('hi')" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFIER (parse level only, no tsc/eslint)
# ══════════════════════════════════════════════════════════════════════════════


def test_verifier_valid_code(verifier):
    code = "function foo() { return 1 }"
    ok, errors = verifier.verify(code, "a.ts")
    assert ok
    assert errors == []


def test_verifier_broken_code(verifier):
    code = "function foo() { return {{{"
    ok, errors = verifier.verify(code, "a.ts")
    assert not ok
    assert len(errors) > 0
    assert errors[0].line is not None


def test_verifier_empty_code(verifier):
    ok, _errors = verifier.verify("", "a.ts")
    assert ok


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTION VM — FULL CYCLE
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════


def test_vm_success(vm):
    code = "function foo() { return 1 }"
    result = vm.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return 2"},
        ),
    ])

    assert result.success
    assert "return 2" in result.code
    assert not result.rolled_back


def test_vm_rollback_on_apply_failure(vm):
    code = "function foo() { return 1 }"
    result = vm.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "nonexistent", "body": "x"},
        ),
    ])

    assert not result.success
    assert result.rolled_back
    assert "return 1" in result.code  # original code


def test_vm_rollback_on_parse_failure(vm):
    """Code that breaks AST should be rolled back."""
    code = "function foo() { return 1 }"
    result = vm.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return {{{"},
        ),
    ])

    assert not result.success
    assert result.rolled_back
    assert "return 1" in result.code


def test_vm_multi_op_success(vm):
    code = """\
import { old } from './old'

function process() {
  old()
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "process", "body": "console.log('new')"},
        ),
    ]

    result = vm.execute(code, "a.ts", ops)
    assert result.success
    assert "console.log('new')" in result.code


def test_vm_empty_ops(vm):
    code = "const x = 1"
    result = vm.execute(code, "a.ts", [])

    # Empty ops → apply succeeds, verify passes
    assert result.success
    assert result.code == code


# ── Multi-op sequence ────────────────────────────────────────────────────────


def test_vm_complex_refactor(vm):
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
    assert "cache.get" in result.code
    assert "unused" not in result.code


# ── Partial failure ──────────────────────────────────────────────────────────


def test_vm_partial_failure_rollback(vm):
    """If second op fails, entire execution rolls back."""
    code = """\
function a() { return 1 }
function b() { return 2 }
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "a", "body": "return 10"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "nonexistent", "body": "fail"},
        ),
    ]

    result = vm.execute(code, "a.ts", ops)
    assert not result.success
    assert result.rolled_back
    # Should be rolled back to original
    assert "return 1" in result.code


# ── E2E realistic ────────────────────────────────────────────────────────────


def test_vm_e2e_api_migration(vm):
    """Realistic scenario: migrate API calls with safety."""
    code = """\
import { http } from './http'

function getUsers() {
  return http.get('/users')
}

function getUser(id) {
  return http.get('/users/' + id)
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { api } from './api-client'"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "getUsers",
                "body": "return api.fetchAll('users')",
            },
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "getUser",
                "body": "return api.fetchOne('users', id)",
            },
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REMOVE_IMPORT,
            payload={"source": "./http"},
        ),
    ]

    result = vm.execute(code, "users.ts", ops)

    assert result.success
    assert "import { api } from './api-client'" in result.code
    assert "api.fetchAll('users')" in result.code
    assert "api.fetchOne('users', id)" in result.code
    assert "./http" not in result.code
    assert not result.rolled_back


# ══════════════════════════════════════════════════════════════════════════════
#  VMResult MODEL
# ══════════════════════════════════════════════════════════════════════════════


def test_vm_result_model():
    r = VMResult(success=True, code="x", message="ok")
    assert r.repair_attempts == 0
    assert r.verify_errors == []
    assert not r.rolled_back
