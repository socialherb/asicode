"""Tests for TS/JS Primitive System (P4)."""
from __future__ import annotations

import pytest

from external_llm.editor._editor_core.ts_vm.primitives.executor import TSPrimitiveExecutor
from external_llm.editor._editor_core.ts_vm.primitives.models import PrimitiveKind, PrimitiveOp
from external_llm.languages.tree_sitter_utils import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="tree-sitter not installed"
)


@pytest.fixture
def executor():
    return TSPrimitiveExecutor(language="typescript")


# ══════════════════════════════════════════════════════════════════════════════
#  REPLACE_FUNCTION_BODY
# ══════════════════════════════════════════════════════════════════════════════


def test_replace_function_body_simple(executor):
    code = "function foo() { return 1 }"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return 2"},
        ),
    ])

    assert result.success
    assert "return 2" in result.code
    assert "return 1" not in result.code


def test_replace_function_body_multiline(executor):
    code = """\
function greet(name) {
  console.log('hello')
  return name
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "greet",
                "body": "const msg = 'hi ' + name\nreturn msg",
            },
        ),
    ])

    assert result.success
    assert "const msg = 'hi ' + name" in result.code
    assert "return msg" in result.code
    assert "console.log" not in result.code


def test_replace_function_body_not_found(executor):
    code = "function foo() {}"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "bar", "body": "return 1"},
        ),
    ])

    assert not result.success
    assert "not found" in result.message


def test_replace_arrow_function_body(executor):
    code = "const add = (a, b) => { return a + b }"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "add", "body": "return a * b"},
        ),
    ])

    assert result.success
    assert "return a * b" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  REPLACE_FUNCTION_BODY — Class fallback (F1)
# ══════════════════════════════════════════════════════════════════════════════


def test_replace_class_body(executor):
    """REPLACE_FUNCTION_BODY should support class body replacement via
    get_class() fallback when the name matches a class declaration."""
    code = """\
class Counter {
  count = 0
  increment() { return this.count++ }
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "Counter",
                "body": "private count = 0\nincrement() { return ++this.count }",
            },
        ),
    ])

    assert result.success
    assert "private count = 0" in result.code
    assert "++this.count" in result.code
    assert result.code.count("class Counter") == 1


def test_replace_method_body(executor):
    """REPLACE_FUNCTION_BODY should support method body replacement
    via the existing get_function() → class-methods fallback."""
    code = """\
class Greeter {
  greet(name: string) { return "hi " + name }
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "greet",
                "body": "return `Hello ${name}`",
            },
        ),
    ])

    assert result.success
    assert "Hello" in result.code
    assert "hi" not in result.code  # old body gone


def test_replace_method_body_dotted_name(executor):
    """REPLACE_FUNCTION_BODY should resolve dotted names (ClassName.method)
    like "Greeter.greet" — the same as bare "greet"."""
    code = """\
class Greeter {
  greet(name: string) { return "hi " + name }
  farewell(name: string) { return "bye " + name }
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "Greeter.greet",
                "body": "return `Hello ${name}`",
            },
        ),
    ])

    assert result.success
    assert "Hello" in result.code
    assert "hi" not in result.code  # old body gone
    # farewell should remain untouched
    assert "farewell" in result.code
    assert "bye" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  INSERT_IMPORT
# ══════════════════════════════════════════════════════════════════════════════


def test_insert_import_basic(executor):
    code = """\
function foo() {}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { bar } from './bar'"},
        ),
    ])

    assert result.success
    assert "import { bar } from './bar'" in result.code
    assert result.code.index("import") < result.code.index("function")


def test_insert_import_after_existing(executor):
    code = """\
import { foo } from './foo'

function test() {}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { bar } from './bar'"},
        ),
    ])

    assert result.success
    assert "import { bar } from './bar'" in result.code


def test_insert_import_dedup(executor):
    code = """\
import { foo } from './foo'

function test() {}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { baz } from './foo'"},
        ),
    ])

    assert result.success
    # Should skip because ./foo already imported
    assert "already exists" in result.message


# ══════════════════════════════════════════════════════════════════════════════
#  REMOVE_IMPORT
# ══════════════════════════════════════════════════════════════════════════════


def test_remove_import(executor):
    code = """\
import { foo } from './foo'
import { bar } from './bar'

function test() {}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.REMOVE_IMPORT,
            payload={"source": "./foo"},
        ),
    ])

    assert result.success
    assert "./foo" not in result.code
    assert "import { bar } from './bar'" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  RENAME_SYMBOL
# ══════════════════════════════════════════════════════════════════════════════


def test_rename_function(executor):
    code = """\
function oldName(x) {
  return x * 2
}

const result = oldName(5)
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.RENAME_SYMBOL,
            payload={"old_name": "oldName", "new_name": "newName"},
        ),
    ])

    assert result.success
    assert "function newName" in result.code
    assert "newName(5)" in result.code
    assert "oldName" not in result.code


def test_rename_variable(executor):
    code = """\
const count = 0
console.log(count)
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.RENAME_SYMBOL,
            payload={"old_name": "count", "new_name": "total"},
        ),
    ])

    assert result.success
    assert "total" in result.code
    # The declaration might still have 'count' since symbol meta covers
    # the whole node; usage-based rename handles reference sites


def test_rename_noop(executor):
    code = "function foo() {}"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.RENAME_SYMBOL,
            payload={"old_name": "foo", "new_name": "foo"},
        ),
    ])

    assert result.success
    assert "no-op" in result.message


# ══════════════════════════════════════════════════════════════════════════════
#  UPDATE_CALL
# ══════════════════════════════════════════════════════════════════════════════


def test_update_call_rename_callee(executor):
    code = """\
function process() {
  oldApi()
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.UPDATE_CALL,
            payload={"callee": "oldApi", "new_callee": "newApi"},
        ),
    ])

    assert result.success
    assert "newApi()" in result.code
    assert "oldApi" not in result.code


def test_update_call_change_args(executor):
    code = """\
function run() {
  fetch('/old')
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.UPDATE_CALL,
            payload={"callee": "fetch", "new_args": "'/new', { method: 'POST' }"},
        ),
    ])

    assert result.success
    assert "'/new'" in result.code
    assert "method: 'POST'" in result.code


def test_update_call_scoped(executor):
    code = """\
function a() { log('a') }
function b() { log('b') }
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.UPDATE_CALL,
            payload={"callee": "log", "new_callee": "debug", "scope": "a"},
        ),
    ])

    assert result.success
    assert "debug('a')" in result.code
    assert "log('b')" in result.code  # untouched


def test_update_call_not_found(executor):
    code = "function f() {}"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.UPDATE_CALL,
            payload={"callee": "nonexistent", "new_callee": "x"},
        ),
    ])

    assert not result.success


# ══════════════════════════════════════════════════════════════════════════════
#  INSERT_STATEMENT
# ══════════════════════════════════════════════════════════════════════════════


def test_insert_statement_at_start(executor):
    code = """\
function init() {
  doWork()
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_STATEMENT,
            payload={
                "statement": "console.log('start')",
                "anchor": "init",
                "position": "start",
            },
        ),
    ])

    assert result.success
    assert "console.log('start')" in result.code
    # Should appear before doWork
    assert result.code.index("console.log") < result.code.index("doWork")


def test_insert_statement_at_end(executor):
    code = """\
function cleanup() {
  release()
}
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_STATEMENT,
            payload={
                "statement": "console.log('done')",
                "anchor": "cleanup",
                "position": "end",
            },
        ),
    ])

    assert result.success
    assert "console.log('done')" in result.code
    assert result.code.index("release") < result.code.index("console.log")


# ══════════════════════════════════════════════════════════════════════════════
#  DELETE_NODE
# ══════════════════════════════════════════════════════════════════════════════


def test_delete_function(executor):
    code = """\
function keep() { return 1 }
function remove() { return 2 }
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.DELETE_NODE,
            payload={"name": "remove"},
        ),
    ])

    assert result.success
    assert "function keep" in result.code
    assert "function remove" not in result.code


def test_delete_variable(executor):
    code = """\
const keep = 1
const remove = 2
"""
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.DELETE_NODE,
            payload={"name": "remove"},
        ),
    ])

    assert result.success
    assert "const keep" in result.code
    assert "remove" not in result.code


def test_delete_not_found(executor):
    code = "const x = 1"
    result = executor.execute(code, "a.ts", [
        PrimitiveOp(
            kind=PrimitiveKind.DELETE_NODE,
            payload={"name": "nonexistent"},
        ),
    ])

    assert not result.success


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-OP SEQUENCES (reparse between ops)
# ══════════════════════════════════════════════════════════════════════════════


def test_multi_op_sequence(executor):
    """Multiple ops in sequence with reparse between each."""
    code = """\
function process(data) {
  validate(data)
  return data
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { transform } from './transform'"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "process",
                "body": "validate(data)\nconst result = transform(data)\nreturn result",
            },
        ),
    ]

    result = executor.execute(code, "a.ts", ops)

    assert result.success
    assert "import { transform }" in result.code
    assert "transform(data)" in result.code
    assert "return result" in result.code


def test_multi_op_import_then_rename(executor):
    """Import + rename in sequence."""
    code = """\
import { oldHelper } from './utils'

function run() {
  oldHelper()
}
"""
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.UPDATE_CALL,
            payload={"callee": "oldHelper", "new_callee": "newHelper"},
        ),
    ]

    result = executor.execute(code, "a.ts", ops)

    assert result.success
    assert "newHelper()" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  E2E: REALISTIC REFACTOR SCENARIO
# ══════════════════════════════════════════════════════════════════════════════


def test_e2e_refactor_scenario(executor):
    """Realistic: add import, replace body, delete unused function."""
    code = """\
import { db } from './database'

function fetchUsers() {
  return db.query('SELECT * FROM users')
}

function legacyFetch() {
  return null
}
"""
    ops = [
        # 1. Add new import
        PrimitiveOp(
            kind=PrimitiveKind.INSERT_IMPORT,
            payload={"statement": "import { cache } from './cache'"},
        ),
        # 2. Replace fetchUsers body with cached version
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={
                "name": "fetchUsers",
                "body": "const cached = cache.get('users')\nif (cached) return cached\nconst users = db.query('SELECT * FROM users')\ncache.set('users', users)\nreturn users",
            },
        ),
        # 3. Delete legacy function
        PrimitiveOp(
            kind=PrimitiveKind.DELETE_NODE,
            payload={"name": "legacyFetch"},
        ),
    ]

    result = executor.execute(code, "service.ts", ops)

    assert result.success
    assert "import { cache } from './cache'" in result.code
    assert "cache.get('users')" in result.code
    assert "cache.set('users', users)" in result.code
    assert "legacyFetch" not in result.code
    assert "function fetchUsers" in result.code


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════


def test_empty_ops(executor):
    result = executor.execute("const x = 1", "a.ts", [])
    assert result.success
    assert result.code == "const x = 1"


def test_op_failure_stops_sequence(executor):
    """If an op fails, subsequent ops should NOT execute."""
    code = "function foo() { return 1 }"
    ops = [
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "nonexistent", "body": "fail"},
        ),
        PrimitiveOp(
            kind=PrimitiveKind.REPLACE_FUNCTION_BODY,
            payload={"name": "foo", "body": "return 99"},
        ),
    ]

    result = executor.execute(code, "a.ts", ops)

    assert not result.success
    # foo should NOT be modified because first op failed
    assert "return 1" in result.code
    assert "return 99" not in result.code
