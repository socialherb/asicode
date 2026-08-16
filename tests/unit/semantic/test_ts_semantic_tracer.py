"""Tests for TS/JS Semantic Tracer — Core IR."""
from __future__ import annotations

import textwrap

import pytest

from external_llm.editor.semantic import ts_semantic_tracer as ts_mod
from external_llm.editor.semantic.ts_semantic_tracer import (
    TSSemanticTracer,
    _build_jsx_parser,
    _build_tsx_parser,
)
from external_llm.languages.tree_sitter_utils import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="tree-sitter not installed"
)

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ts_tracer():
    return TSSemanticTracer(language="typescript")


@pytest.fixture
def js_tracer():
    return TSSemanticTracer(language="javascript")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE IR TESTS  (analyze_core)
# ══════════════════════════════════════════════════════════════════════════════


# ── imports ──────────────────────────────────────────────────────────────────

IMPORT_CODE = """\
import express from 'express'
import { Router, Request, Response } from 'express'
import * as path from 'path'
"""


def test_core_imports(ts_tracer):
    m = ts_tracer.analyze_core(IMPORT_CODE, "server.ts")

    assert len(m.imports) == 3

    # Default import
    imp0 = m.imports[0]
    assert imp0.source == "express"
    assert imp0.default_name == "express"

    # Named imports
    imp1 = m.imports[1]
    assert imp1.source == "express"
    assert set(imp1.specifiers) == {"Router", "Request", "Response"}

    # Namespace import
    imp2 = m.imports[2]
    assert imp2.source == "path"
    assert imp2.namespace_name == "path"

    assert m.import_sources == {"express", "path"}


# ── functions ────────────────────────────────────────────────────────────────

FUNC_CODE = """\
export function greet(name: string): string {
  return 'Hello ' + name
}

export const fetchData = async (url: string) => {
  const res = await fetch(url)
  return res.json()
}

function helper(x: number) {
  return x * 2
}
"""


def test_core_functions(ts_tracer):
    m = ts_tracer.analyze_core(FUNC_CODE, "utils.ts")

    assert len(m.functions) == 3
    names = [f.name for f in m.functions]
    assert "greet" in names
    assert "fetchData" in names
    assert "helper" in names

    greet = m.get_function("greet")
    assert greet is not None
    assert greet.is_exported is True
    assert greet.is_async is False
    assert len(greet.params) >= 1
    assert greet.params[0].name == "name"

    fetch_fn = m.get_function("fetchData")
    assert fetch_fn is not None
    assert fetch_fn.is_async is True
    assert fetch_fn.is_exported is True

    helper = m.get_function("helper")
    assert helper is not None
    assert helper.is_exported is False


# ── exports ──────────────────────────────────────────────────────────────────

EXPORT_CODE = """\
export function foo() {}
export const bar = 42
export default function main() {}
"""


def test_core_exports(ts_tracer):
    m = ts_tracer.analyze_core(EXPORT_CODE, "mod.ts")

    export_names = m.exported_symbols
    assert "foo" in export_names
    assert "bar" in export_names
    assert "main" in export_names


# ── variables ────────────────────────────────────────────────────────────────

VAR_CODE = """\
const API_URL = 'https://api.example.com'
let count = 0
const items = [1, 2, 3]
const config = { debug: true }
const instance = new Database()
"""


def test_core_variables(ts_tracer):
    m = ts_tracer.analyze_core(VAR_CODE, "config.ts")

    var_names = [v.name for v in m.variables]
    assert "API_URL" in var_names
    assert "count" in var_names
    assert "items" in var_names
    assert "config" in var_names
    assert "instance" in var_names

    api = next(v for v in m.variables if v.name == "API_URL")
    assert api.decl_kind == "const"
    assert api.initializer_type == "literal"

    items = next(v for v in m.variables if v.name == "items")
    assert items.initializer_type == "array"

    inst = next(v for v in m.variables if v.name == "instance")
    assert inst.initializer_type == "new"


# ── call graph ───────────────────────────────────────────────────────────────

CALL_GRAPH_CODE = """\
import { validate } from './validator'
import { save } from './db'

export function createUser(data: any) {
  validate(data)
  const user = transform(data)
  save(user)
  console.log('done')
  return user
}

function transform(raw: any) {
  return normalize(raw)
}
"""


def test_core_call_graph(ts_tracer):
    m = ts_tracer.analyze_core(CALL_GRAPH_CODE, "user.ts")

    assert len(m.functions) == 2

    # createUser calls: validate, transform, save, log (method)
    cu_callees = m.callees_of("createUser")
    assert "validate" in cu_callees
    assert "transform" in cu_callees
    assert "save" in cu_callees
    assert "log" in cu_callees  # console.log → callee = "log"

    # transform calls: normalize
    t_callees = m.callees_of("transform")
    assert "normalize" in t_callees

    # Reverse: who calls validate?
    callers = m.callers_of("validate")
    assert "createUser" in callers

    # Method call detection
    log_site = next(
        cs for cs in m.call_sites if cs.callee == "log")
    assert log_site.is_method_call is True
    assert log_site.receiver == "console"


# ── class ────────────────────────────────────────────────────────────────────

CLASS_CODE = """\
export class UserService {
  private db: Database

  constructor(db: Database) {
    this.db = db
  }

  async findById(id: string) {
    return this.db.query(id)
  }

  static create() {
    return new UserService(new Database())
  }
}
"""


def test_core_class(ts_tracer):
    m = ts_tracer.analyze_core(CLASS_CODE, "service.ts")

    assert len(m.classes) == 1
    cls = m.get_class("UserService")
    assert cls is not None
    assert cls.is_exported is True

    method_names = [met.name for met in cls.methods]
    assert "constructor" in method_names
    assert "findById" in method_names
    assert "create" in method_names

    find = next(met for met in cls.methods if met.name == "findById")
    assert find.is_async is True

    create = next(met for met in cls.methods if met.name == "create")
    assert create.is_static is True


# ── class inheritance ────────────────────────────────────────────────────────

INHERITANCE_CODE = """\
class Animal {
  name: string
  speak() {}
}

class Dog extends Animal {
  bark() {}
}
"""


def test_core_class_inheritance(ts_tracer):
    m = ts_tracer.analyze_core(INHERITANCE_CODE, "animals.ts")

    assert len(m.classes) == 2
    dog = m.get_class("Dog")
    assert dog is not None
    assert dog.extends == "Animal"


# ── interface ────────────────────────────────────────────────────────────────

INTERFACE_CODE = """\
export interface UserDTO {
  id: string
  name: string
  email: string
}

interface Searchable {
  search(query: string): Promise<any[]>
}
"""


def test_core_interface(ts_tracer):
    m = ts_tracer.analyze_core(INTERFACE_CODE, "types.ts")

    assert len(m.interfaces) == 2

    user_dto = next(i for i in m.interfaces if i.name == "UserDTO")
    assert user_dto.is_exported is True
    assert any(p.name == "id" for p in user_dto.properties)
    assert any(p.name == "name" for p in user_dto.properties)
    assert any(p.name == "email" for p in user_dto.properties)

    searchable = next(i for i in m.interfaces if i.name == "Searchable")
    assert searchable.is_exported is False
    assert "search" in searchable.methods


# ── type alias + enum ────────────────────────────────────────────────────────

TYPE_ENUM_CODE = """\
export type Status = 'active' | 'inactive'

export enum Role {
  ADMIN,
  USER,
  GUEST,
}
"""


def test_core_type_alias_and_enum(ts_tracer):
    m = ts_tracer.analyze_core(TYPE_ENUM_CODE, "constants.ts")

    assert len(m.type_aliases) == 1
    assert m.type_aliases[0].name == "Status"
    assert m.type_aliases[0].is_exported is True

    assert len(m.enums) == 1
    assert m.enums[0].name == "Role"
    assert m.enums[0].is_exported is True


# ── all_symbols ──────────────────────────────────────────────────────────────

MIXED_CODE = """\
import { db } from './db'

export function handler() {
  db.query()
}

export class Service {}

export interface Config {
  port: number
}

export type Mode = 'dev' | 'prod'

const VERSION = '1.0'
"""


def test_core_all_symbols(ts_tracer):
    m = ts_tracer.analyze_core(MIXED_CODE, "app.ts")

    syms = m.all_symbols
    assert "handler" in syms
    assert "Service" in syms
    assert "Config" in syms
    assert "Mode" in syms
    assert "VERSION" in syms


# ── Node.js backend pattern ──────────────────────────────────────────────────

NODE_CODE = """\
import express from 'express'
import { UserService } from './services/user'

const app = express()
const userService = new UserService()

app.get('/users', async (req, res) => {
  const users = await userService.findAll()
  res.json(users)
})

app.post('/users', async (req, res) => {
  const user = await userService.create(req.body)
  res.status(201).json(user)
})

app.listen(3000)
"""


def test_core_node_pattern(ts_tracer):
    m = ts_tracer.analyze_core(NODE_CODE, "server.ts")

    # Imports
    assert len(m.imports) == 2
    assert m.imports[0].default_name == "express"

    # Variables: app, userService
    var_names = [v.name for v in m.variables]
    assert "app" in var_names
    assert "userService" in var_names

    # Call graph: top-level calls
    all_callees = [cs.callee for cs in m.call_sites]
    assert "express" in all_callees  # const app = express()
    assert "listen" in all_callees  # app.listen(3000)
    assert "get" in all_callees  # app.get(...)
    assert "post" in all_callees  # app.post(...)


# ── empty ────────────────────────────────────────────────────────────────────

def test_core_empty(ts_tracer):
    m = ts_tracer.analyze_core("", "empty.ts")
    assert m.functions == []
    assert m.classes == []
    assert m.imports == []
    assert m.call_sites == []
    assert m.all_symbols == []


# ── JavaScript ───────────────────────────────────────────────────────────────

JS_CODE = """\
const http = require('http')

function handleRequest(req, res) {
  process(req)
  res.end('ok')
}

http.createServer(handleRequest).listen(8080)
"""


def test_core_javascript(js_tracer):
    m = js_tracer.analyze_core(JS_CODE, "server.js")

    func_names = [f.name for f in m.functions]
    assert "handleRequest" in func_names

    callees = m.callees_of("handleRequest")
    assert "process" in callees
    assert "end" in callees


# ══════════════════════════════════════════════════════════════════════════════
#  P2.5 EXECUTION IR TESTS
# ══════════════════════════════════════════════════════════════════════════════


# ── IRNodeMeta on all nodes ──────────────────────────────────────────────────

def test_execution_ir_function_meta(ts_tracer):
    """Every function should have IRNodeMeta with stable identity."""
    code = "function foo(x) { bar(x) }"
    m = ts_tracer.analyze_core(code, "a.ts")

    f = m.get_function("foo")
    assert f is not None
    assert f.meta is not None
    assert f.meta.start_line == 1
    assert f.meta.end_line == 1
    assert f.meta.start_byte == 0
    assert f.meta.end_byte == len(code)
    assert len(f.meta.node_id) == 12  # md5[:12]


def test_execution_ir_class_meta(ts_tracer):
    code = """\
class Svc {
  run() {}
}
"""
    m = ts_tracer.analyze_core(code, "svc.ts")

    cls = m.get_class("Svc")
    assert cls is not None
    assert cls.meta is not None
    assert cls.meta.start_line == 1

    # Method should also have meta
    assert len(cls.methods) == 1
    assert cls.methods[0].meta is not None
    assert cls.methods[0].meta.start_line == 2


def test_execution_ir_import_meta(ts_tracer):
    code = "import { foo } from './lib'"
    m = ts_tracer.analyze_core(code, "x.ts")

    assert len(m.imports) == 1
    assert m.imports[0].meta is not None
    assert m.imports[0].meta.start_byte == 0


def test_execution_ir_variable_meta(ts_tracer):
    code = "const x = 42"
    m = ts_tracer.analyze_core(code, "v.ts")

    assert len(m.variables) == 1
    assert m.variables[0].meta is not None


def test_execution_ir_callsite_meta(ts_tracer):
    code = "function f() { g() }"
    m = ts_tracer.analyze_core(code, "c.ts")

    assert len(m.call_sites) >= 1
    cs = next(s for s in m.call_sites if s.callee == "g")
    assert cs.meta is not None
    assert cs.meta.start_line == 1


# ── symbol table ─────────────────────────────────────────────────────────────

def test_symbol_table_function(ts_tracer):
    code = "function greet(name) { console.log(name) }"
    m = ts_tracer.analyze_core(code, "s.ts")

    # Function symbol
    sym = m.get_symbol("greet")
    assert sym is not None
    assert sym.kind.value == "function"
    assert sym.scope == "<module>"

    # Param symbol
    param_sym = m.get_symbol("name")
    assert param_sym is not None
    assert param_sym.kind.value == "param"
    assert param_sym.scope == "greet"


def test_symbol_table_variable(ts_tracer):
    code = "const x = 1\nlet y = 2"
    m = ts_tracer.analyze_core(code, "v.ts")

    x_sym = m.get_symbol("x")
    assert x_sym is not None
    assert x_sym.kind.value == "variable"

    y_sym = m.get_symbol("y")
    assert y_sym is not None


def test_symbol_table_class(ts_tracer):
    code = """\
class Dog {
  bark() {}
}
"""
    m = ts_tracer.analyze_core(code, "c.ts")

    cls_sym = m.get_symbol("Dog")
    assert cls_sym is not None
    assert cls_sym.kind.value == "class"

    method_sym = m.get_symbol("bark")
    assert method_sym is not None
    assert method_sym.kind.value == "method"
    assert method_sym.scope == "Dog"


def test_symbol_table_interface_enum(ts_tracer):
    code = """\
interface Runnable { run(): void }
enum Color { RED, GREEN }
type ID = string
"""
    m = ts_tracer.analyze_core(code, "t.ts")

    assert m.get_symbol("Runnable") is not None
    assert m.get_symbol("Runnable").kind.value == "interface"
    assert m.get_symbol("Color") is not None
    assert m.get_symbol("Color").kind.value == "enum"
    assert m.get_symbol("ID") is not None
    assert m.get_symbol("ID").kind.value == "type_alias"


def test_symbols_in_scope(ts_tracer):
    code = """\
const a = 1
function foo(x) {
  const b = 2
}
"""
    m = ts_tracer.analyze_core(code, "sc.ts")

    module_syms = m.symbols_in_scope("<module>")
    module_names = [s.name for s in module_syms]
    assert "a" in module_names
    assert "foo" in module_names

    foo_syms = m.symbols_in_scope("foo")
    foo_names = [s.name for s in foo_syms]
    assert "x" in foo_names


# ── usage graph ──────────────────────────────────────────────────────────────

def test_usage_graph(ts_tracer):
    code = """\
import { validate } from './v'

function process(data) {
  validate(data)
  return data
}
"""
    m = ts_tracer.analyze_core(code, "u.ts")

    # 'validate' should be used inside 'process'
    val_usages = m.usages_of("validate")
    assert len(val_usages) >= 1
    assert any(u.scope == "process" for u in val_usages)

    # 'data' should be used inside 'process'
    data_usages = m.usages_of("data")
    assert len(data_usages) >= 1
    assert all(u.scope == "process" for u in data_usages)


def test_usage_meta(ts_tracer):
    code = "function f() { g() }"
    m = ts_tracer.analyze_core(code, "um.ts")

    g_usages = m.usages_of("g")
    assert len(g_usages) >= 1
    assert g_usages[0].meta is not None


# ── assignment / data flow ───────────────────────────────────────────────────

def test_assignment_from_call(ts_tracer):
    code = """\
function process() {
  const result = compute()
  return result
}
"""
    m = ts_tracer.analyze_core(code, "a.ts")

    result_assigns = m.assignments_to("result")
    assert len(result_assigns) >= 1
    a = result_assigns[0]
    assert a.source == "compute"
    assert a.source_type == "call"
    assert a.scope == "process"


def test_assignment_from_variable(ts_tracer):
    code = """\
function copy() {
  const original = items
}
"""
    m = ts_tracer.analyze_core(code, "a.ts")

    assigns = m.assignments_to("original")
    assert len(assigns) >= 1
    assert assigns[0].source == "items"
    assert assigns[0].source_type == "variable"


def test_assignment_from_new(ts_tracer):
    code = "const svc = new Service()"
    m = ts_tracer.analyze_core(code, "n.ts")

    assigns = m.assignments_to("svc")
    assert len(assigns) >= 1
    assert assigns[0].source == "Service"
    assert assigns[0].source_type == "new"


def test_assignment_from_literal(ts_tracer):
    code = 'const name = "hello"'
    m = ts_tracer.analyze_core(code, "l.ts")

    assigns = m.assignments_to("name")
    assert len(assigns) >= 1
    assert assigns[0].source_type == "literal"


def test_data_sources_of(ts_tracer):
    code = """\
function load() {
  const raw = fetchData()
  const parsed = transform(raw)
  return parsed
}
"""
    m = ts_tracer.analyze_core(code, "ds.ts")

    # parsed ← transform
    sources = m.data_sources_of("parsed")
    assert "transform" in sources

    # raw ← fetchData
    sources2 = m.data_sources_of("raw")
    assert "fetchData" in sources2


def test_assignment_meta(ts_tracer):
    code = "const x = foo()"
    m = ts_tracer.analyze_core(code, "am.ts")

    assigns = m.assignments_to("x")
    assert len(assigns) >= 1
    assert assigns[0].meta is not None


# ── combined: end-to-end execution IR ────────────────────────────────────────

E2E_CODE = """\
import { db } from './database'

export function createUser(input: any) {
  const validated = validate(input)
  const user = db.insert(validated)
  notify(user)
  return user
}

function validate(data: any) {
  if (!data.name) throw new Error('missing name')
  return data
}
"""


def test_execution_ir_e2e(ts_tracer):
    """Full execution IR: meta + symbols + usages + assignments + call graph."""
    m = ts_tracer.analyze_core(E2E_CODE, "user_service.ts")

    # 1. All nodes have meta
    for f in m.functions:
        assert f.meta is not None, f"function {f.name} missing meta"
    for imp in m.imports:
        assert imp.meta is not None
    for cs in m.call_sites:
        assert cs.meta is not None

    # 2. Symbol table complete
    sym_names = [s.name for s in m.symbols]
    assert "createUser" in sym_names
    assert "validate" in sym_names
    assert "input" in sym_names  # param

    # 3. Call graph
    cu_callees = m.callees_of("createUser")
    assert "validate" in cu_callees
    assert "insert" in cu_callees
    assert "notify" in cu_callees

    # 4. Usages
    validate_usages = m.usages_of("validate")
    assert any(u.scope == "createUser" for u in validate_usages)

    # 5. Assignments / data flow
    validated_assigns = m.assignments_to("validated")
    assert len(validated_assigns) >= 1
    assert validated_assigns[0].source == "validate"
    assert validated_assigns[0].source_type == "call"

    user_assigns = m.assignments_to("user")
    assert len(user_assigns) >= 1
    assert user_assigns[0].source == "insert"

    # 6. Data flow queries
    assert "validate" in m.data_sources_of("validated")
    assert "insert" in m.data_sources_of("user")


# ══════════════════════════════════════════════════════════════════════════════
#  IR convenience lookups  (RED→GREEN: get_function/get_class/get_symbol)
# ══════════════════════════════════════════════════════════════════════════════


def test_ir_module_lookup_conveniences():
    """TSModule lookup helpers resolve dotted/bare names, classes, symbols.

    Covers the dotted-name path (L347-353), bare-name class-method path
    (L361-365), get_class miss (L371) and get_symbol miss (L386).
    """
    from external_llm.editor.semantic.ts_ir_models import (
        IRClass,
        IRFunction,
        IRMethod,
        IRSymbol,
        SymbolKind,
        TSModule,
    )

    mod = TSModule(
        file_path="game.ts",
        functions=[IRFunction(name="greet")],
        classes=[
            IRClass(
                name="Game",
                methods=[
                    IRMethod(name="lockPiece"),
                    IRMethod(name="move"),
                ],
            )
        ],
        symbols=[
            IRSymbol(name="score", kind=SymbolKind.VARIABLE),
            IRSymbol(name="state", kind=SymbolKind.VARIABLE, scope="Game"),
        ],
    )

    # Dotted name: ClassName.methodName (hit + miss)
    assert mod.get_function("Game.lockPiece").name == "lockPiece"
    assert mod.get_function("Game.nope") is None
    assert mod.get_function("Missing.method") is None
    # Bare name: top-level function, then class method, then miss
    assert mod.get_function("greet").name == "greet"
    assert mod.get_function("move").name == "move"
    assert mod.get_function("absent") is None
    # get_class hit + miss
    assert mod.get_class("Game").name == "Game"
    assert mod.get_class("Nope") is None
    # get_symbol hit + miss
    assert mod.get_symbol("score").kind == SymbolKind.VARIABLE
    assert mod.get_symbol("absent") is None


# ══════════════════════════════════════════════════════════════════════════════
#  RED→GREEN: uncovered branches
# ══════════════════════════════════════════════════════════════════════════════


def test_parser_degradation_paths(monkeypatch):
    """Parser unavailable/failure paths degrade to None/empty (L83-99,
    L151-163, L202)."""
    # Grammar loader failure → builders return None.
    def _boom(*_a, **_k):
        raise ImportError("grammar missing")

    monkeypatch.setattr("external_llm.languages.tree_sitter_utils.get_parser", _boom)
    assert _build_tsx_parser() is None
    assert _build_jsx_parser() is None

    # tree-sitter not available → analyze_core returns an empty module.
    tracer = TSSemanticTracer(language="typescript")
    monkeypatch.setattr(ts_mod, "is_available", lambda: False)
    m = tracer.analyze_core("const x = 1;", "x.ts")
    assert m.imports == [] and m.functions == []

    # Parser resolved but None → _parse returns None.
    tracer2 = TSSemanticTracer(language="javascript")
    monkeypatch.setattr(ts_mod, "is_available", lambda: True)
    monkeypatch.setattr(tracer2, "_get_parser", lambda: None)
    assert tracer2._parse("const x = 1;", "x.ts") is None

    # Parser raises → _parse returns None.
    class _BoomParser:
        def parse(self, code):
            raise RuntimeError("parse boom")

    monkeypatch.setattr(tracer2, "_get_parser", lambda: _BoomParser())
    assert tracer2._parse("const x = 1;", "x.ts") is None


def test_export_clause_and_default(ts_tracer):
    """export { X, Y as Z } and export default expr (L246-247, L339-354)."""
    m = ts_tracer.analyze_core(
        "export { foo, bar as baz };\nexport default myThing;\n", "mod.ts"
    )
    names = {(e.name, e.kind.value) for e in m.exports}
    assert ("foo", "named") in names
    assert ("baz", "named") in names
    assert ("myThing", "default") in names


def test_import_type_only(ts_tracer):
    """import type marks is_type_only (L311)."""
    m = ts_tracer.analyze_core('import type { Foo } from "m";\n', "t.ts")
    assert len(m.imports) == 1
    assert m.imports[0].is_type_only is True


def test_variable_initializer_classification(ts_tracer):
    """Initializer classification: identifier → variable, unknown node
    (ternary) → None (L481, L523)."""
    m = ts_tracer.analyze_core(
        "const a = b;\nconst f = () => 1;\nconst t = a ? b : c;\n", "v.ts"
    )
    assert {"a", "t"} <= {v.name for v in m.variables}
    assert "f" in {fn.name for fn in m.functions}
    # The ternary initializer is an unclassified node → source_type None.
    ta = next(ass for ass in m.assignments if ass.target == "t")
    assert ta.source_type is None


def test_classify_arrow(ts_tracer):
    """_classify_initializer maps function-like nodes to 'arrow' (L514)."""
    root = ts_tracer._parse("const f = () => 1;", "a.ts")
    assert root is not None
    arrow = None

    def find(node):
        nonlocal arrow
        if arrow is not None:
            return
        if node.type == "arrow_function":
            arrow = node
            return
        for c in node.children:
            find(c)

    find(root)
    assert arrow is not None
    assert ts_tracer._classify_initializer(arrow) == "arrow"


def test_class_implements_and_accessors(ts_tracer):
    """implements clause + get/set accessors (L546-547, L594-596)."""
    code = textwrap.dedent("""\
        class A implements B, C {
            get value() { return this._v; }
            set value(v: number) { this._v = v; }
        }
    """)
    m = ts_tracer.analyze_core(code, "svc.ts")
    assert m.classes and m.classes[0].name == "A"
    assert m.classes[0].implements == ["B", "C"]
    getters = [mm for mm in m.classes[0].methods if mm.is_getter]
    setters = [mm for mm in m.classes[0].methods if mm.is_setter]
    assert len(getters) == 1
    assert len(setters) == 1


def test_interface_extends(ts_tracer):
    """interface extends clause (L631)."""
    m = ts_tracer.analyze_core("interface A extends B {}\n", "t.ts")
    assert m.interfaces and m.interfaces[0].extends == ["B"]


def test_assignment_new_expression(ts_tracer):
    """new-expression initializer records its class as source inside a
    function body (L806-809)."""
    m = ts_tracer.analyze_core(
        "const f = () => { const a = new Foo(); };\n", "n.ts"
    )
    assert any(ass.target == "a" and ass.source == "Foo" for ass in m.assignments)


def test_usage_skip_import_specifier_and_member_property(ts_tracer):
    """Usage collection skips import specifiers and member properties
    (L767, L772)."""
    m = ts_tracer.analyze_core(
        'import { helper } from "m";\nfunction f() { obj.prop; helper(); }\n', "u.ts"
    )
    assert all(u.symbol != "prop" for u in m.usages)
    assert any(u.symbol == "obj" for u in m.usages)


def test_assignment_skips_without_value_and_array_pattern(ts_tracer):
    """Assignments skip value-less declarators (L792) and array patterns
    (L795); top-level destructuring registers variables (L410-432)."""
    m = ts_tracer.analyze_core(
        "const [a, b] = pair;\nconst f = () => {\n  const x;\n  const [c, d] = pair2;\n};\n",
        "as.ts",
    )
    assert {"a", "b"} <= {v.name for v in m.variables}
    assert not any(ass.target == "x" for ass in m.assignments)


def test_function_symbol_loop_skips_unrelated(ts_tracer):
    """The function-symbol loop skips unrelated symbols (L449)."""
    m = ts_tracer.analyze_core("const other = 1;\nconst f = () => 1;\n", "p.ts")
    assert "f" in {fn.name for fn in m.functions}


def test_params_forms_js(js_tracer):
    """JS param forms: rest_pattern, object_pattern, array_pattern directly
    under formal_parameters (L857, L860, L862)."""
    m = js_tracer.analyze_core(
        "function f(...args) {}\nfunction g({x}, [y]) {}\n", "f.js"
    )
    by_name = {fn.name: fn for fn in m.functions}
    assert by_name["f"].params[0].is_rest is True
    assert by_name["g"].params[0].name == "{...}"
    assert by_name["g"].params[1].name == "[...]"


def test_typed_params_ts(ts_tracer):
    """TS typed/optional params still extract plain names (L845)."""
    m = ts_tracer.analyze_core(
        "function typed(a: number, b?: string) {}\n", "p.ts"
    )
    params = m.functions[0].params
    assert [p.name for p in params] == ["a", "b"]


def test_callee_name_and_walk_leaf(ts_tracer):
    """_walk's leaf exit path (L893)."""
    root = ts_tracer._parse("", "empty.ts")
    assert root is not None
    assert list(ts_tracer._walk(root)) == [root]


def test_syntax_error_tolerance(ts_tracer):
    """Broken constructs degrade without crashing (L301, L406, L707, L826,
    L832, L839)."""
    m = ts_tracer.analyze_core(
        "import {\nconst = 5;\nfunction nope\nfoo(\nconst z = obj.;\n", "bad.ts"
    )
    assert isinstance(m, ts_mod.TSModule)
