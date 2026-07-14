"""Tests for Python Semantic Tracer — AST-based behavioral trace extraction."""
from __future__ import annotations

import os
import tempfile
import textwrap

from external_llm.editor.semantic.semantic_tracer import (
    SemanticTrace,
    extract_trace_from_files,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _trace_code(source: str) -> SemanticTrace:
    """Write source to a temp .py file and extract trace."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(textwrap.dedent(source))
        path = f.name
    try:
        return extract_trace_from_files([path])
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Basic function — calls, instantiations, persist, return names
# ══════════════════════════════════════════════════════════════════════════════


BASIC_CODE = """\
class Message:
    pass

def create_message(content: str, sender_id: int):
    validate(content)
    msg = Message(content=content, sender_id=sender_id)
    db.add(msg)
    db.commit()
    return msg
"""


def test_basic_calls():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert "validate" in ft.calls
    assert "Message" in ft.calls
    assert "add" in ft.calls
    assert "commit" in ft.calls


def test_basic_instantiations():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert "Message" in ft.instantiations


def test_basic_persist_calls():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert "db.add" in ft.persist_calls
    assert "db.commit" in ft.persist_calls


def test_basic_return_names():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert "msg" in ft.return_names


def test_basic_return_has_entity_ref():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert ft.return_has_entity_ref is True


def test_basic_param_names():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    assert ft.param_names == ["content", "sender_id"]


def test_basic_call_order():
    trace = _trace_code(BASIC_CODE)
    ft = trace.function_traces["create_message"]
    # validate should come before Message
    order = ft.call_order
    assert order.index("validate") < order.index("Message")
    assert order.index("Message") < order.index("add")


def test_aggregate_sets():
    trace = _trace_code(BASIC_CODE)
    assert "Message" in trace.all_instantiations
    assert "Message" in trace.all_classes
    assert "create_message" in trace.all_functions
    assert "db.add" in trace.all_persist_calls
    assert "db.commit" in trace.all_persist_calls


def test_semantic_trace_properties():
    trace = _trace_code(BASIC_CODE)
    assert "Message" in trace.created_entities
    assert "db.add" in trace.persisted_entities
    assert "msg" in trace.return_vars


# ══════════════════════════════════════════════════════════════════════════════
#  2. Entity binding detection
# ══════════════════════════════════════════════════════════════════════════════


BINDING_CODE = """\
class User:
    pass

def register(name, email):
    user = User(name=name, email=email)
    return user
"""


def test_entity_bindings_keyword():
    trace = _trace_code(BINDING_CODE)
    ft = trace.function_traces["register"]
    bindings = ft.entity_bindings
    assert ("name", "name") in bindings
    assert ("email", "email") in bindings


BINDING_POSITIONAL_CODE = """\
class Point:
    pass

def make_point(x, y):
    p = Point(x, y)
    return p
"""


def test_entity_bindings_positional():
    trace = _trace_code(BINDING_POSITIONAL_CODE)
    ft = trace.function_traces["make_point"]
    positional = [b for b in ft.entity_bindings if b[0] == "_positional"]
    sources = {b[1] for b in positional}
    assert "x" in sources
    assert "y" in sources


# ══════════════════════════════════════════════════════════════════════════════
#  3. Error branch detection
# ══════════════════════════════════════════════════════════════════════════════


ERROR_BRANCH_CODE = """\
def get_item(item_id):
    item = lookup(item_id)
    if not item:
        raise ValueError("not found")
    return item
"""


def test_error_branch_detected():
    trace = _trace_code(ERROR_BRANCH_CODE)
    ft = trace.function_traces["get_item"]
    assert ft.has_error_branch is True


def test_error_before_success():
    trace = _trace_code(ERROR_BRANCH_CODE)
    ft = trace.function_traces["get_item"]
    # The raise is inside an if before the return → error_before_success
    assert ft.error_before_success is True


ERROR_AFTER_RETURN_CODE = """\
def process(data):
    result = transform(data)
    return result
    if not result:
        raise RuntimeError("fail")
"""


def test_error_after_return():
    trace = _trace_code(ERROR_AFTER_RETURN_CODE)
    ft = trace.function_traces["process"]
    assert ft.has_error_branch is True
    # The raise comes after a return statement, so error_before_success should be False
    assert ft.error_before_success is False


NO_ERROR_CODE = """\
def simple(x):
    return x + 1
"""


def test_no_error_branch():
    trace = _trace_code(NO_ERROR_CODE)
    ft = trace.function_traces["simple"]
    assert ft.has_error_branch is False
    assert ft.error_before_success is False


BARE_RAISE_CODE = """\
def strict(x):
    raise NotImplementedError("todo")
"""


def test_bare_raise_detected():
    trace = _trace_code(BARE_RAISE_CODE)
    ft = trace.function_traces["strict"]
    assert ft.has_error_branch is True
    assert ft.error_before_success is True


# ══════════════════════════════════════════════════════════════════════════════
#  4. Return reference tracking
# ══════════════════════════════════════════════════════════════════════════════


RETURN_DICT_CODE = """\
class Order:
    pass

def create_order(price):
    order = Order(price=price)
    return {"id": order.id, "price": order.price}
"""


def test_return_dict_entity_ref():
    trace = _trace_code(RETURN_DICT_CODE)
    ft = trace.function_traces["create_order"]
    assert ft.return_has_entity_ref is True
    assert "order" in ft.return_names


RETURN_CALL_CODE = """\
def fetch():
    return serialize(data)
"""


def test_return_call_name():
    trace = _trace_code(RETURN_CALL_CODE)
    ft = trace.function_traces["fetch"]
    assert "serialize" in ft.return_names


RETURN_ATTR_CODE = """\
class Foo:
    pass

def get_foo_name():
    foo = Foo()
    return foo.name
"""


def test_return_attribute_entity_ref():
    trace = _trace_code(RETURN_ATTR_CODE)
    ft = trace.function_traces["get_foo_name"]
    assert "foo" in ft.return_names
    assert ft.return_has_entity_ref is True


RETURN_NO_ENTITY_CODE = """\
def add(a, b):
    result = a + b
    return result
"""


def test_return_no_entity_ref():
    trace = _trace_code(RETURN_NO_ENTITY_CODE)
    ft = trace.function_traces["add"]
    assert "result" in ft.return_names
    assert ft.return_has_entity_ref is False


# ══════════════════════════════════════════════════════════════════════════════
#  5. extract_trace_from_files with temp files
# ══════════════════════════════════════════════════════════════════════════════


def test_extract_from_real_file():
    source = textwrap.dedent("""\
    class Item:
        pass

    def add_item(name):
        item = Item(name=name)
        db.add(item)
        return item
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(source)
        path = f.name
    try:
        trace = extract_trace_from_files([path])
        assert "add_item" in trace.function_traces
        ft = trace.function_traces["add_item"]
        assert ft.file_path == path
        assert "Item" in ft.instantiations
        assert "db.add" in ft.persist_calls
    finally:
        os.unlink(path)


def test_extract_from_multiple_files():
    source_a = textwrap.dedent("""\
    class Cat:
        pass

    def make_cat(name):
        return Cat(name=name)
    """)
    source_b = textwrap.dedent("""\
    def feed(cat):
        db.save(cat)
    """)
    files = []
    try:
        for src in [source_a, source_b]:
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            )
            f.write(src)
            f.close()
            files.append(f.name)

        trace = extract_trace_from_files(files)
        assert "make_cat" in trace.function_traces
        assert "feed" in trace.function_traces
        assert "Cat" in trace.all_classes
        assert "Cat" in trace.all_instantiations
        assert "db.save" in trace.all_persist_calls
    finally:
        for p in files:
            os.unlink(p)


def test_extract_skips_non_python():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    ) as f:
        f.write("not python")
        path = f.name
    try:
        trace = extract_trace_from_files([path])
        assert len(trace.function_traces) == 0
    finally:
        os.unlink(path)


def test_extract_skips_syntax_error():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write("def broken(\n  return ???")
        path = f.name
    try:
        trace = extract_trace_from_files([path])
        assert len(trace.function_traces) == 0
    finally:
        os.unlink(path)


def test_extract_skips_missing_file():
    trace = extract_trace_from_files(["/nonexistent/path/foo.py"])
    assert len(trace.function_traces) == 0


def test_extract_with_repo_root():
    source = textwrap.dedent("""\
    def hello():
        print("hi")
    """)
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "mod.py")
    with open(filepath, "w") as f:
        f.write(source)
    try:
        # Pass relative path + repo_root
        trace = extract_trace_from_files(["mod.py"], repo_root=tmpdir)
        assert "hello" in trace.function_traces
    finally:
        os.unlink(filepath)
        os.rmdir(tmpdir)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Edge cases
# ══════════════════════════════════════════════════════════════════════════════


EMPTY_FUNCTION_CODE = """\
def noop():
    pass
"""


def test_empty_function():
    trace = _trace_code(EMPTY_FUNCTION_CODE)
    ft = trace.function_traces["noop"]
    assert ft.calls == set()
    assert ft.instantiations == set()
    assert ft.persist_calls == set()
    assert ft.return_names == set()
    assert ft.return_has_entity_ref is False
    assert ft.has_error_branch is False
    assert ft.entity_bindings == []
    assert ft.call_order == []


NO_RETURN_CODE = """\
def side_effect(data):
    db.insert(data)
"""


def test_function_no_return():
    trace = _trace_code(NO_RETURN_CODE)
    ft = trace.function_traces["side_effect"]
    assert ft.return_names == set()
    assert ft.return_has_entity_ref is False
    assert "db.insert" in ft.persist_calls


NESTED_CALLS_CODE = """\
def process(items):
    result = transform(validate(parse(items)))
    return result
"""


def test_nested_calls():
    trace = _trace_code(NESTED_CALLS_CODE)
    ft = trace.function_traces["process"]
    assert "transform" in ft.calls
    assert "validate" in ft.calls
    assert "parse" in ft.calls


ASYNC_FUNCTION_CODE = """\
class Record:
    pass

async def async_create(name):
    record = Record(name=name)
    await db.add(record)
    return record
"""


def test_async_function():
    trace = _trace_code(ASYNC_FUNCTION_CODE)
    ft = trace.function_traces["async_create"]
    assert "Record" in ft.instantiations
    assert "db.add" in ft.persist_calls
    assert "record" in ft.return_names
    assert ft.return_has_entity_ref is True


CLASS_METHOD_CODE = """\
class Service:
    def process(self, data):
        result = compute(data)
        return result
"""


def test_class_method_self_excluded():
    trace = _trace_code(CLASS_METHOD_CODE)
    ft = trace.function_traces["process"]
    assert "self" not in ft.param_names
    assert "data" in ft.param_names


STORE_SUBSCRIPT_CODE = """\
def cache_item(key, value):
    store[key] = value
    db_map[key] = value
"""


def test_subscript_persist_pattern():
    trace = _trace_code(STORE_SUBSCRIPT_CODE)
    ft = trace.function_traces["cache_item"]
    assert "store[]=" in ft.persist_calls


PERSIST_METHODS_CODE = """\
def save_all(item):
    session.add(item)
    repo.insert(item)
    collection.put(item)
    item.save()
"""


def test_various_persist_patterns():
    trace = _trace_code(PERSIST_METHODS_CODE)
    ft = trace.function_traces["save_all"]
    assert "session.add" in ft.persist_calls
    assert "repo.insert" in ft.persist_calls
    assert "collection.put" in ft.persist_calls
    # item.save() — .save() triggers persist even without known obj
    assert "item.save" in ft.persist_calls


UPPERCASE_NOT_IN_CLASSES_CODE = """\
def use_config():
    cfg = Config(debug=True)
    return cfg
"""


def test_uppercase_heuristic_instantiation():
    """Uppercase first letter is treated as instantiation even without class def."""
    trace = _trace_code(UPPERCASE_NOT_IN_CLASSES_CODE)
    ft = trace.function_traces["use_config"]
    assert "Config" in ft.instantiations
    assert ft.return_has_entity_ref is True


RETURN_DICT_WITH_NAME_CODE = """\
class Ticket:
    pass

def create_ticket(title):
    t = Ticket(title=title)
    return {"ticket": t, "status": "ok"}
"""


def test_return_dict_with_entity_name():
    trace = _trace_code(RETURN_DICT_WITH_NAME_CODE)
    ft = trace.function_traces["create_ticket"]
    assert "t" in ft.return_names
    assert ft.return_has_entity_ref is True


MULTIPLE_FUNCTIONS_CODE = """\
class Widget:
    pass

def create_widget(name):
    w = Widget(name=name)
    db.add(w)
    return w

def delete_widget(widget_id):
    widget = lookup(widget_id)
    if not widget:
        raise ValueError("missing")
    db.delete(widget)
"""


def test_multiple_functions_traced():
    trace = _trace_code(MULTIPLE_FUNCTIONS_CODE)
    assert "create_widget" in trace.function_traces
    assert "delete_widget" in trace.function_traces
    ft_del = trace.function_traces["delete_widget"]
    assert ft_del.has_error_branch is True
    assert "lookup" in ft_del.calls


NESTED_IF_RAISE_CODE = """\
def validate(x):
    if x > 0:
        if x > 100:
            raise ValueError("too big")
    return x
"""


def test_nested_if_raise():
    trace = _trace_code(NESTED_IF_RAISE_CODE)
    ft = trace.function_traces["validate"]
    assert ft.has_error_branch is True
