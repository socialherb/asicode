"""Unit tests for external_llm.graph.models canonical model types."""

from external_llm.graph.models import CallEdge, EdgeKind, ImportEdge, SymbolId, SymbolKind, SymbolNode


def test_symbol_kind_values():
    assert SymbolKind.MODULE == "module"
    assert SymbolKind.CLASS == "class"
    assert SymbolKind.FUNCTION == "function"
    assert SymbolKind.METHOD == "method"


def test_edge_kind_values():
    assert EdgeKind.CALLS == "calls"
    assert EdgeKind.IMPORTS == "imports"
    assert EdgeKind.DEFINES == "defines"
    assert EdgeKind.INHERITS == "inherits"
    assert EdgeKind.CONTAINS == "contains"


def test_symbol_id_creation():
    sid = SymbolId(
        module="mymodule",
        qualname="MyClass.method",
        file_path="mymodule.py",
        kind=SymbolKind.METHOD,
    )
    assert sid.module == "mymodule"
    assert sid.qualname == "MyClass.method"
    assert sid.file_path == "mymodule.py"
    assert sid.kind == SymbolKind.METHOD


def test_symbol_id_equality():
    sid1 = SymbolId(module="m", qualname="foo", file_path="m.py", kind=SymbolKind.FUNCTION)
    sid2 = SymbolId(module="m", qualname="foo", file_path="m.py", kind=SymbolKind.FUNCTION)
    assert sid1 == sid2


def test_symbol_id_inequality_different_file():
    sid1 = SymbolId(module="m", qualname="foo", file_path="a.py", kind=SymbolKind.FUNCTION)
    sid2 = SymbolId(module="m", qualname="foo", file_path="b.py", kind=SymbolKind.FUNCTION)
    assert sid1 != sid2


def test_symbol_id_inequality_different_qualname():
    sid1 = SymbolId(module="m", qualname="foo", file_path="a.py", kind=SymbolKind.FUNCTION)
    sid2 = SymbolId(module="m", qualname="bar", file_path="a.py", kind=SymbolKind.FUNCTION)
    assert sid1 != sid2


def test_symbol_id_hashable():
    sid = SymbolId(module="m", qualname="foo", file_path="m.py", kind=SymbolKind.FUNCTION)
    s = {sid}
    assert sid in s


def test_symbol_id_same_name_different_file():
    """Same qualname in different files should produce distinct SymbolIds."""
    sid_a = SymbolId(module="pkg.a", qualname="helper", file_path="pkg/a.py", kind=SymbolKind.FUNCTION)
    sid_b = SymbolId(module="pkg.b", qualname="helper", file_path="pkg/b.py", kind=SymbolKind.FUNCTION)
    assert sid_a != sid_b
    assert sid_a.qualname == sid_b.qualname  # same name
    assert sid_a.file_path != sid_b.file_path  # different file


def test_symbol_node_symbol_id_function():
    node = SymbolNode(
        name="foo",
        qualname="foo",
        module="mymod",
        file_path="mymod.py",
        kind="function",
        start_line=1,
        end_line=5,
    )
    sid = node.symbol_id
    assert isinstance(sid, SymbolId)
    assert sid.kind == SymbolKind.FUNCTION
    assert sid.qualname == "foo"
    assert sid.module == "mymod"
    assert sid.file_path == "mymod.py"


def test_symbol_node_symbol_id_method():
    node = SymbolNode(
        name="bar",
        qualname="MyClass.bar",
        module="mymod",
        file_path="mymod.py",
        kind="method",
        start_line=10,
        end_line=20,
    )
    sid = node.symbol_id
    assert sid.kind == SymbolKind.METHOD
    assert sid.qualname == "MyClass.bar"


def test_symbol_node_symbol_id_class():
    node = SymbolNode(
        name="MyClass",
        qualname="MyClass",
        module="mymod",
        file_path="mymod.py",
        kind="class",
        start_line=1,
        end_line=30,
    )
    sid = node.symbol_id
    assert sid.kind == SymbolKind.CLASS


def test_symbol_node_symbol_id_unknown_kind_defaults_to_function():
    node = SymbolNode(
        name="x",
        qualname="x",
        module="m",
        file_path="m.py",
        kind="unknown_kind",
        start_line=1,
        end_line=1,
    )
    sid = node.symbol_id
    assert sid.kind == SymbolKind.FUNCTION


def test_call_edge_defaults():
    edge = CallEdge(
        caller_symbol="a",
        caller_file="a.py",
        caller_line=1,
        callee_symbol="b",
        callee_display="b",
    )
    assert edge.callee_file is None
    assert edge.callee_line is None
    assert edge.confidence == 1.0
    assert edge.edge_kind == EdgeKind.CALLS


def test_import_edge():
    edge = ImportEdge(importer="a.py", imported="b", import_type="import")
    assert edge.alias is None
    edge2 = ImportEdge(importer="a.py", imported="b.C", import_type="from", alias="C2")
    assert edge2.alias == "C2"
