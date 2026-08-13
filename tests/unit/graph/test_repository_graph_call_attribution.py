"""P3 Stage 1: RG call-edge canonical attribution (CGI convention parity).

RepositoryGraph's Python call extraction now emits, per call edge, the SAME
canonical attribution CallGraphIndexer produces — ``callee_symbol`` (qualified
for ``self.``/``cls.`` receivers), dotted ``callee_display``, and the
0.9/0.85/0.5 call-form confidence — while PRESERVING the legacy bare ``callee``
field that the analysis scanners (vulture/cross_file_refs/broken_contract/
dead_block) query against.  This is barrier (1) of the CGI/RG dual-build merge:
once RG's snapshot carries these fields, CGI can consume it as SSOT.
"""
import ast
import shutil
import tempfile
import textwrap
from pathlib import Path

from external_llm.agent.call_graph import CallGraphIndexer
from external_llm.graph.repository_graph import GraphVisitor, RepositoryGraph
from external_llm.graph.structural_cache import data_from_json, data_to_json


def _visit(source: str, rel: str = "mod.py") -> GraphVisitor:
    visitor = GraphVisitor(rel, "/repo")
    visitor.visit(ast.parse(textwrap.dedent(source)))
    return visitor


def _build_graph(source: str):
    d = tempfile.mkdtemp(prefix="test_rgca_")
    fp = Path(d) / "mod.py"
    fp.write_text(textwrap.dedent(source))
    try:
        g = RepositoryGraph(str(d))
        g.build()
        return g
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _edge(visitor: GraphVisitor, callee: str) -> list:
    return [c for c in visitor.calls if c.callee == callee]


# ── Canonical attribution spec (mirrors CallGraphIndexer._parse_call) ─────────

def test_plain_call_canonical_fields():
    v = _visit("""\
        def top():
            helper()
    """)
    (e,) = _edge(v, "helper")
    assert e.callee_symbol == "helper"
    assert e.callee_display == "helper"
    assert e.confidence == 0.9


def test_self_call_qualified_canonical_and_bare_legacy():
    v = _visit("""\
        class A:
            def foo(self):
                pass

            def bar(self):
                self.foo()
    """)
    (e,) = _edge(v, "foo")  # legacy callee stays BARE (scanner contract)
    assert e.callee == "foo"
    assert e.callee_symbol == "A.foo"      # CGI convention: qualified
    assert e.callee_display == "self.foo"  # CGI convention: dotted display
    assert e.confidence == 0.85


def test_cls_call_matches_cgi_attr_only():
    """CGI parity pin: CallGraphIndexer._parse_call special-cases ONLY ``self``
    (not ``cls.``), so ``cls.build()`` falls to the generic-attribute branch —
    attr-only callee_symbol, 0.5.  RG's canonical fields mirror that exactly
    (the legacy ``callee`` still strips the receiver, per RG's own contract)."""
    v = _visit("""\
        class A:
            @classmethod
            def make(cls):
                return cls.build()

            @classmethod
            def build(cls):
                pass
    """)
    (e,) = _edge(v, "build")
    assert e.callee == "build"            # legacy bare
    assert e.callee_symbol == "build"     # CGI: attr-only for cls.
    assert e.callee_display == "cls.build"
    assert e.confidence == 0.5


def test_obj_call_attr_only_canonical():
    v = _visit("""\
        def main():
            a = A()
            a.foo()
            mod.helper()
    """)
    (e,) = _edge(v, "a.foo")              # legacy callee keeps the dotted receiver
    assert e.callee == "a.foo"
    assert e.callee_symbol == "foo"       # canonical drops the receiver (CGI rule)
    assert e.callee_display == "a.foo"
    assert e.confidence == 0.5
    (e2,) = _edge(v, "mod.helper")
    assert e2.callee_symbol == "helper"
    assert e2.callee_display == "mod.helper"
    assert e2.confidence == 0.5


def test_nested_function_self_call_low_confidence():
    """CGI parity: a nested function is NOT a direct method, so its ``self.``
    call falls to the generic-attribute branch (attr-only, 0.5) — even though
    it is lexically inside a class."""
    v = _visit("""\
        class A:
            def outer(self):
                def inner(self):
                    self.foo()
                inner(self)
    """)
    (e,) = [c for c in v.calls if c.callee_display == "self.foo"]
    assert e.callee == "foo"
    assert e.callee_symbol == "foo"       # attr-only, NOT "A.foo"
    assert e.confidence == 0.5


def test_module_level_calls_skipped():
    """Both workers skip calls outside any function body."""
    v = _visit("""\
        import os
        os.getcwd()
    """)
    assert v.calls == []


def test_nested_class_qualifies_with_bare_immediate_name():
    """P3 Stage 1 follow-up (item 2): a ``self.`` call inside a NESTED class
    qualifies with the bare immediate class NAME (``Inner.bar``), NOT the full
    qualname (``Outer.Inner.bar``).  CGI's ``class_names`` maps a method to its
    enclosing ``ClassDef.name`` — this test pins that RG derives the same bare
    name so the two payloads stay bit-for-bit equal (the SSOT-merge gate)."""
    v = _visit("""\
        class Outer:
            class Inner:
                def foo(self):
                    self.bar()

                def bar(self):
                    self.helper()

                def helper(self):
                    pass

            def top_method(self):
                self.helper()
    """)
    triples = {(e.callee_symbol, e.callee_display, e.confidence)
               for e in v.calls if e.callee_display.startswith("self.")}
    assert triples == {
        ("Inner.bar", "self.bar", 0.85),        # nested class → bare "Inner"
        ("Inner.helper", "self.helper", 0.85),  # nested class → bare "Inner"
        ("Outer.helper", "self.helper", 0.85),  # top-level class → bare "Outer"
    }


def test_nested_class_cross_worker_parity():
    """The nested-class fixture through both workers yields identical
    (callee_symbol, callee_display, confidence) triples — the regression guard
    for the qualname-vs-bare-name divergence (item 2)."""
    fixture = """\
        class Outer:
            class Inner:
                def foo(self):
                    self.bar()

                def bar(self):
                    pass

            def top_method(self):
                self.helper()

            def helper(self):
                pass
    """
    tree = ast.parse(textwrap.dedent(fixture))

    # CGI side: real _extract_file class_names BFS (bare ClassDef.name)
    class_names: dict[int, str] = {}
    func_nodes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_names[id(child)] = node.name
    cgi = CallGraphIndexer(".")._collect_calls(func_nodes, "mod.py", class_names)
    cgi_triples = {(e.callee_symbol, e.callee_display, round(e.confidence, 2))
                   for e in cgi if e.callee_display.startswith("self.")}

    # RG side
    rg = _visit(fixture)
    rg_triples = {(e.callee_symbol, e.callee_display, round(e.confidence, 2))
                  for e in rg.calls if e.callee_display.startswith("self.")}

    assert rg_triples == cgi_triples == {
        ("Inner.bar", "self.bar", 0.85),
        ("Outer.helper", "self.helper", 0.85),
    }
# ── Legacy behavior pinned ────────────────────────────────────────────────────

def test_legacy_queries_unchanged():
    """get_callees/get_callers still resolve through the bare ``callee`` field."""
    g = _build_graph("""\
        class A:
            def foo(self):
                pass

            def bar(self):
                self.foo()
    """)
    callees = g.get_callees("bar")
    assert [e.callee for e in callees] == ["foo"]
    callers = g.get_callers("foo")
    assert [e.caller for e in callers] == ["A.bar"]


# ── Snapshot round-trip ───────────────────────────────────────────────────────

def test_snapshot_roundtrip_keeps_canonical_fields():
    v = _visit("""\
        class A:
            def foo(self):
                pass

            def bar(self):
                self.foo()
    """)
    data = {"symbols": v.symbols, "calls": v.calls, "imports": v.imports}
    back = data_from_json(data_to_json(data))
    (e,) = [c for c in back["calls"] if c.callee == "foo"]
    assert e.callee_symbol == "A.foo"
    assert e.callee_display == "self.foo"
    assert e.confidence == 0.85


def test_old_format_snapshot_loads_with_defaults():
    """A pre-P3 snapshot (no canonical keys) must still reconstruct — the new
    fields are additive with defaults."""
    old = [{"caller": "A.bar", "callee": "foo", "file_path": "mod.py", "line": 4}]
    back = data_from_json({"symbols": [], "calls": old, "imports": []})
    (e,) = back["calls"]
    assert e.callee_symbol is None
    assert e.callee_display == ""
    assert e.confidence == 1.0


# ── Cross-worker parity: RG canonical fields == CGI _parse_call ───────────────

_FIXTURE = """\
    def helper():
        pass

    class A:
        def foo(self):
            helper()

        def bar(self):
            self.foo()

    def main():
        a = A()
        a.foo()
        mod.helper()
"""


def test_cgi_parity_canonical_attribution():
    """The SAME fixture through CallGraphIndexer._collect_calls (pure) and
    GraphVisitor must yield identical (callee_symbol, callee_display,
    confidence) triples."""
    tree = ast.parse(textwrap.dedent(_FIXTURE))

    # CGI side: replicate _extract_file's class_names BFS + _collect_calls
    # (pure extraction — __init__ does no walking/parsing, so a bare instance
    # is enough to run the collector).
    func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    class_names: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_names[id(child)] = node.name
    cgi = CallGraphIndexer(".")._collect_calls(func_nodes, "mod.py", class_names)
    cgi_triples = {(e.callee_symbol, e.callee_display, round(e.confidence, 2)) for e in cgi}

    # RG side: GraphVisitor canonical fields.
    rg = _visit(_FIXTURE)
    rg_triples = {(e.callee_symbol, e.callee_display, round(e.confidence, 2)) for e in rg.calls}

    assert rg_triples == cgi_triples
    assert cgi_triples == {
        ("helper", "helper", 0.9),
        ("A", "A", 0.9),          # the `a = A()` constructor call
        ("A.foo", "self.foo", 0.85),
        ("foo", "a.foo", 0.5),
        ("helper", "mod.helper", 0.5),
    }


# ── Unsupported-form fallback (P2, 2026-08-12) ─────────────────────────────────

def test_unsupported_form_fallback_low_confidence_and_marked():
    """Chained/dynamic call forms must NOT outrank resolved calls: the legacy
    fallback edge carries confidence 0.2 (was the 1.0 sentinel, which ranked
    ABOVE fully-resolved foo()=0.9) plus an explicit ``resolution="fallback"``
    marker — the SSOT conversion drops by the marker, never by confidence
    value (P2, 2026-08-12)."""
    v = _visit("""\
        def run():
            return make()()
    """)
    # the chained form produces TWO edges: the inner make() (resolved) and the
    # outer make()() (unsupported form) — pick the fallback by its marker.
    (fallback,) = [e for e in v.calls if e.resolution == "fallback"]
    assert fallback.callee == "make"  # legacy bare callee still resolved
    assert fallback.confidence == 0.2
    assert fallback.resolution == "fallback"
    # the inner call stays a fully-resolved edge
    assert any(e.resolution == "" and e.confidence == 0.9 for e in v.calls)


def test_attribute_chain_dynamic_receiver_fallback_marked():
    """obj.attr.m() — attribute chain rooted in an expression (not a Name) is
    unsupported → same low-confidence fallback contract."""
    v = _visit("""\
        def run(obj):
            return obj().make()
    """)
    (fallback,) = [e for e in v.calls if e.resolution == "fallback"]
    assert fallback.confidence == 0.2
    assert fallback.resolution == "fallback"


def test_snapshot_roundtrip_keeps_resolution_marker():
    """The resolution marker must survive the snapshot round-trip so the SSOT
    conversion sees it on cache-served payloads."""
    v = _visit("""\
        def run():
            return make()()
    """)
    data = {"symbols": v.symbols, "calls": v.calls, "imports": v.imports}
    back = data_from_json(data_to_json(data))
    (e,) = [c for c in back["calls"] if c.resolution == "fallback"]
    assert e.resolution == "fallback"
    assert e.confidence == 0.2
