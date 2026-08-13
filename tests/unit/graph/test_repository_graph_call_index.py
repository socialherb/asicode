"""Tests for the lazily-built reverse call-edge indexes in RepositoryGraph.

``get_callers`` / ``get_callees`` previously scanned ``call_edges`` linearly
(O(N x E)); the index makes queries O(1) + O(result) while preserving the
legacy semantics exactly: exact match first, bare-name suffix fallback,
dedup by (caller, callee, file_path, line).  These tests pin the equivalence
against a reference implementation of the legacy scan and verify index
invalidation on mutation.
"""
import shutil
import tempfile
from pathlib import Path

from external_llm.graph.repository_graph import RepositoryGraph

SRC = '''\
def helper():
    pass

def top():
    helper()

class A:
    def foo(self):
        helper()

    def bar(self):
        self.foo()

class B:
    def foo(self):
        helper()

    def baz(self):
        helper()
        helper()  # same callee, consecutive lines -> both kept

    def one_line(self):
        helper(); helper()  # same line, same callee -> dedup target

def main():
    a = A()
    a.foo(); a.foo()  # same line -> identical (caller, callee, file, line)
    a.bar()
    B().foo()
'''


def _build_graph(source: str):
    d = tempfile.mkdtemp(prefix="test_rci_")
    fp = Path(d) / "mod.py"
    fp.write_text(source)
    try:
        g = RepositoryGraph(str(d))
        g.build()
        return g
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _legacy_edges(graph, field, symbol_name):
    """Reference implementation of the pre-index linear scan."""
    exact = [e for e in graph.call_edges if getattr(e, field) == symbol_name]
    if exact:
        return exact
    parts = symbol_name.split(".")
    method = parts[-1] if parts else symbol_name
    result = []
    seen = set()
    for edge in graph.call_edges:
        val = getattr(edge, field)
        if not val:
            continue
        if val.endswith(f".{method}") or val == method:
            key = (edge.caller, edge.callee, edge.file_path, edge.line)
            if key not in seen:
                seen.add(key)
                result.append(edge)
    return result


def _key(edges):
    return {(e.caller, e.callee, e.file_path, e.line) for e in edges}


def test_index_matches_legacy_linear_scan():
    """Indexed queries must return exactly what the legacy scan returned."""
    g = _build_graph(SRC)
    queries = [
        "helper", "top", "main",
        "A.foo", "A.bar", "B.foo", "B.baz", "B.one_line",
        "foo", "bar", "baz", "one_line",
        "nonexistent", "A.nonexistent", "",
        "A.foo.bar", "self.foo",
    ]
    for q in queries:
        assert _key(g.get_callers(q)) == _key(_legacy_edges(g, "callee", q)), q
        assert _key(g.get_callees(q)) == _key(_legacy_edges(g, "caller", q)), q


def test_exact_match_takes_priority_over_suffix():
    """A qualified query must return only exact matches, not the suffix union."""
    g = _build_graph(SRC)
    # main -> 'a.foo' (receiver literal, class-unresolved) has an exact
    # bucket; the suffix bucket 'foo' would also match A.bar (self.foo) and
    # B().foo — exact-first must exclude them.
    callers = g.get_callers("a.foo")
    assert {e.caller for e in callers} == {"main"}, callers
    # get_callees matches the caller field: exact bucket 'A.bar' is the
    # self.foo() edge — receiver-unresolved, so callee stays bare 'foo'.
    callees = g.get_callees("A.bar")
    assert {e.callee for e in callees} == {"foo"}, callees


def test_suffix_fallback_bare_name():
    """A bare-name query with no exact match resolves via suffix across classes."""
    g = _build_graph(SRC)
    # 'bar' has no exact callee bucket (self.bar() is never called) -> suffix.
    callers = g.get_callers("bar")
    assert {e.caller for e in callers} == {"main"}, callers


def test_exact_path_keeps_duplicates_legacy_parity():
    """The exact path never deduped — two same-line calls yield two edges."""
    g = _build_graph(SRC)
    helper_edges = [e for e in g.get_callers("helper") if e.caller == "B.one_line"]
    assert len(helper_edges) == 2, helper_edges


def test_suffix_path_dedups():
    """The suffix path dedups by (caller, callee, file_path, line)."""
    g = _build_graph(SRC)
    # 'Y.foo' has no exact bucket -> suffix 'foo'; main's duplicate a.foo
    # edges (same line, identical 4-tuple) must collapse to one.
    callers = g.get_callers("Y.foo")
    assert {e.caller for e in callers} == {"A.bar", "main"}, callers
    foo_edges = [e for e in callers if e.callee == "a.foo"]
    assert len(foo_edges) == 1, foo_edges


def test_reparse_file_invalidates_index():
    """Mutation via reparse_file must be reflected by the next query."""
    d = tempfile.mkdtemp(prefix="test_rci_")
    fp = Path(d) / "mod.py"
    try:
        fp.write_text("def a():\n    b()\n\ndef b():\n    pass\n")
        g = RepositoryGraph(str(d))
        g.build()
        assert {e.caller for e in g.get_callers("b")} == {"a"}

        # Call removed -> index must drop the stale edge.
        fp.write_text("def a():\n    pass\n\ndef b():\n    pass\n")
        g.reparse_file(str(fp))
        assert g.get_callers("b") == []

        # New calls added -> index must include them.
        fp.write_text(
            "def a():\n    b()\n    c()\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
        )
        g.reparse_file(str(fp))
        assert {e.callee for e in g.get_callees("a")} == {"b", "c"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_remove_file_invalidates_index():
    """remove_file must drop the indexed edges for that file."""
    d = tempfile.mkdtemp(prefix="test_rci_")
    fp = Path(d) / "mod.py"
    try:
        fp.write_text("def a():\n    b()\n\ndef b():\n    pass\n")
        g = RepositoryGraph(str(d))
        g.build()
        assert g.get_callers("b")
        g.remove_file("mod.py")
        assert g.get_callers("b") == []
        assert g.get_callees("a") == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_returned_lists_are_copies():
    """Mutating a returned list must not corrupt the index."""
    g = _build_graph(SRC)
    first = g.get_callers("helper")
    first.clear()
    second = g.get_callers("helper")
    assert second, "mutating a returned list must not corrupt the index"
