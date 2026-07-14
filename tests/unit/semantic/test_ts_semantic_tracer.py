"""Tests for TS/JS Semantic Tracer — Core IR + Profile (React)."""
from __future__ import annotations

import pytest

from external_llm.editor.semantic.ts_semantic_tracer import TSSemanticTracer
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
#  PROFILE (REACT) TESTS  (analyze)
# ══════════════════════════════════════════════════════════════════════════════


COUNTER_TSX = """\
import React, { useState } from 'react'

export function Counter() {
  const [count, setCount] = useState(0)

  return <button onClick={() => setCount(count + 1)}>{count}</button>
}
"""


def test_simple_component(ts_tracer):
    result = ts_tracer.analyze(COUNTER_TSX, "Counter.tsx")

    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.name == "Counter"
    assert comp.is_exported is True

    hook_names = [h.name for h in comp.hooks]
    assert "useState" in hook_names

    assert len(comp.state_vars) >= 1
    sv = comp.state_vars[0]
    assert sv.name == "count"
    assert sv.setter == "setCount"
    assert sv.initial_value == "0"

    event_names = [e.event_name for e in comp.events]
    assert "onClick" in event_names
    assert comp.jsx_root == "button"


ARROW_COMPONENT = """\
import { useEffect, useState } from 'react'

export const TodoList = () => {
  const [items, setItems] = useState([])

  useEffect(() => {
    fetch('/api/todos').then(r => r.json()).then(setItems)
  }, [])

  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={() => console.log(item)}>
          {item.text}
        </li>
      ))}
    </ul>
  )
}
"""


def test_arrow_component(ts_tracer):
    result = ts_tracer.analyze(ARROW_COMPONENT, "TodoList.tsx")

    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.name == "TodoList"

    hook_names = [h.name for h in comp.hooks]
    assert "useState" in hook_names
    assert "useEffect" in hook_names

    effect_hook = next(h for h in comp.hooks if h.name == "useEffect")
    assert effect_hook.deps is not None
    assert effect_hook.deps == []

    assert len(comp.state_vars) >= 1
    assert comp.state_vars[0].name == "items"

    event_names = [e.event_name for e in comp.events]
    assert "onClick" in event_names
    assert comp.jsx_root == "ul"


IMPORTS_CODE = """\
import React from 'react'
import { useState, useEffect } from 'react'
import axios from 'axios'

function App() {
  return <div />
}
"""


def test_imports(ts_tracer):
    result = ts_tracer.analyze(IMPORTS_CODE, "App.tsx")

    assert len(result.imports) == 3

    react_imp = next(
        (i for i in result.imports
         if i.source == "react" and i.default_import), None)
    assert react_imp is not None
    assert react_imp.default_import == "React"

    hooks_imp = next(
        (i for i in result.imports
         if i.source == "react" and "useState" in i.specifiers), None)
    assert hooks_imp is not None
    assert "useEffect" in hooks_imp.specifiers


UTIL_CODE = """\
export function formatDate(date: Date): string {
  return date.toISOString()
}

export const fetchData = async (url: string) => {
  const res = await fetch(url)
  return res.json()
}

function helperInternal(x: number) {
  return x * 2
}
"""


def test_regular_functions(ts_tracer):
    result = ts_tracer.analyze(UTIL_CODE, "utils.ts")

    assert len(result.components) == 0

    func_names = [f.name for f in result.functions]
    assert "formatDate" in func_names
    assert "fetchData" in func_names
    assert "helperInternal" in func_names

    format_fn = next(f for f in result.functions if f.name == "formatDate")
    assert format_fn.is_exported is True

    helper_fn = next(
        f for f in result.functions if f.name == "helperInternal")
    assert helper_fn.is_exported is False


MULTI_HOOKS = """\
import { useState, useRef, useCallback, useMemo } from 'react'

function SearchForm() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const inputRef = useRef(null)

  const handleSearch = useCallback(() => {
    fetch('/search?q=' + query).then(r => r.json()).then(setResults)
  }, [query])

  const count = useMemo(() => results.length, [results])

  return (
    <form onSubmit={(e) => { e.preventDefault(); handleSearch() }}>
      <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)} />
      <span>{count} results</span>
    </form>
  )
}
"""


def test_multiple_hooks(ts_tracer):
    result = ts_tracer.analyze(MULTI_HOOKS, "SearchForm.tsx")

    assert len(result.components) == 1
    comp = result.components[0]

    hook_names = [h.name for h in comp.hooks]
    assert "useState" in hook_names
    assert "useRef" in hook_names
    assert "useCallback" in hook_names
    assert "useMemo" in hook_names

    assert len(comp.state_vars) == 2
    state_names = {sv.name for sv in comp.state_vars}
    assert state_names == {"query", "results"}

    event_names = [e.event_name for e in comp.events]
    assert "onSubmit" in event_names
    assert "onChange" in event_names


PROPS_COMPONENT = """\
function UserCard({ name, age, onDelete }) {
  return (
    <div className="card">
      <h2>{name}</h2>
      <p>Age: {age}</p>
      <button onClick={onDelete}>Delete</button>
    </div>
  )
}
"""


def test_component_props(ts_tracer):
    result = ts_tracer.analyze(PROPS_COMPONENT, "UserCard.tsx")

    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.name == "UserCard"

    prop_names = [p.name for p in comp.props]
    assert "name" in prop_names
    assert "age" in prop_names
    assert "onDelete" in prop_names
    assert comp.jsx_root == "div"

    event_names = [e.event_name for e in comp.events]
    assert "onClick" in event_names


FRAGMENT_CODE = """\
function Layout() {
  return (
    <>
      <header>Top</header>
      <main>Content</main>
    </>
  )
}
"""


def test_jsx_fragment(ts_tracer):
    result = ts_tracer.analyze(FRAGMENT_CODE, "Layout.tsx")

    assert len(result.components) == 1
    assert result.components[0].jsx_root == "Fragment"


CUSTOM_HOOK = """\
function Dashboard() {
  const data = useCustomData()
  const theme = useTheme()

  return <div>{data}</div>
}
"""


def test_custom_hooks(ts_tracer):
    result = ts_tracer.analyze(CUSTOM_HOOK, "Dashboard.tsx")

    assert len(result.components) == 1
    hook_names = [h.name for h in result.components[0].hooks]
    assert "useCustomData" in hook_names
    assert "useTheme" in hook_names


def test_module_level_properties(ts_tracer):
    result = ts_tracer.analyze(MULTI_HOOKS, "SearchForm.tsx")

    assert "useState" in result.all_hooks
    assert "useRef" in result.all_hooks
    assert len(result.all_state_vars) == 2
    assert "onSubmit" in result.all_events or "onChange" in result.all_events


def test_empty_code(ts_tracer):
    result = ts_tracer.analyze("", "empty.tsx")
    assert result.components == []
    assert result.functions == []
    assert result.imports == []


COUNTER_JSX = """\
import React, { useState } from 'react'

function Counter() {
  const [count, setCount] = useState(0)
  return <button onClick={() => setCount(count + 1)}>{count}</button>
}
"""


def test_javascript_component(js_tracer):
    result = js_tracer.analyze(COUNTER_JSX, "Counter.jsx")

    assert len(result.components) == 1
    comp = result.components[0]
    assert comp.name == "Counter"
    assert "useState" in [h.name for h in comp.hooks]
    assert "onClick" in [e.event_name for e in comp.events]


# ══════════════════════════════════════════════════════════════════════════════
#  DUAL-LAYER CONSISTENCY TEST
# ══════════════════════════════════════════════════════════════════════════════

DUAL_CODE = """\
import { useState } from 'react'
import { api } from './api'

export function UserList() {
  const [users, setUsers] = useState([])

  const loadUsers = async () => {
    const data = await api.fetchAll()
    setUsers(data)
  }

  return (
    <div>
      <button onClick={loadUsers}>Load</button>
      <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>
    </div>
  )
}

export function formatName(user: any): string {
  return user.first + ' ' + user.last
}
"""


def test_dual_layer_consistency(ts_tracer):
    """Both layers should see the same imports and function-level structure."""
    core = ts_tracer.analyze_core(DUAL_CODE, "dual.tsx")
    profile = ts_tracer.analyze(DUAL_CODE, "dual.tsx")

    # Both see 2 imports
    assert len(core.imports) == 2
    assert len(profile.imports) == 2

    # Core sees all as functions (no component distinction)
    core_names = [f.name for f in core.functions]
    assert "UserList" in core_names
    assert "formatName" in core_names

    # Profile separates component from function
    assert len(profile.components) == 1
    assert profile.components[0].name == "UserList"
    assert len(profile.functions) == 1
    assert profile.functions[0].name == "formatName"

    # Core has call graph
    ul_callees = core.callees_of("UserList")
    assert "useState" in ul_callees


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
