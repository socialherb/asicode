"""Regression tests for nested-scope qualname uniqueness in RepositoryGraph.

These guard against a data-loss defect where nested functions/classes sharing
a bare name (e.g. two ``def wrapper`` closures in sibling decorators) collided
on ``unique_id = f"{path}:{qualname}"``: the first definition was silently
overwritten in ``RepositoryGraph.symbols`` and ``file_symbols`` accumulated
duplicate entries.  Nested symbols must now carry a fully-qualified qualname
that encodes their scope path (``deco_a.wrapper``), mirroring Python's own
``__qualname__`` nesting semantics.
"""
import shutil
import tempfile
from pathlib import Path

from external_llm.graph.repository_graph import RepositoryGraph


def _build_graph(source: str):
    d = tempfile.mkdtemp(prefix="test_rq_")
    fp = Path(d) / "mod.py"
    fp.write_text(source)
    try:
        g = RepositoryGraph(str(d))
        g.build()
        return g
    finally:
        shutil.rmtree(d, ignore_errors=True)


TWO_WRAPPERS = """\
def deco_a(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def deco_b(func):
    def wrapper(*args, **kwargs):
        return helper()
        return func(*args, **kwargs)
    return wrapper
"""


def test_sibling_nested_functions_no_collision():
    """Two ``def wrapper`` in sibling scopes must both survive with distinct
    qualnames — no silent overwrite of the first definition."""
    g = _build_graph(TWO_WRAPPERS)
    wrappers = [s for s in g.symbols.values() if s.name == "wrapper"]
    qualnames = {s.qualname for s in wrappers}
    assert qualnames == {"deco_a.wrapper", "deco_b.wrapper"}, qualnames
    # Both line ranges must be present (start=2 for deco_a's, start=7 for deco_b's).
    starts = sorted(s.start_line for s in wrappers)
    assert starts == [2, 7], starts


def test_file_symbols_has_no_duplicate_ids():
    """file_symbols must not contain duplicate unique-ids after the collision."""
    g = _build_graph(TWO_WRAPPERS)
    ids = g.file_symbols["mod.py"]
    assert len(ids) == len(set(ids)), f"duplicate ids in file_symbols: {ids}"


def test_get_symbols_in_file_returns_unique_symbols():
    """get_symbols_in_file must not return the surviving symbol twice."""
    g = _build_graph(TWO_WRAPPERS)
    syms = g.get_symbols_in_file("mod.py")
    qualnames = [s.qualname for s in syms]
    assert len(qualnames) == len(set(qualnames)), f"duplicate symbols: {qualnames}"
    assert "deco_a.wrapper" in qualnames
    assert "deco_b.wrapper" in qualnames


def test_symbol_locations_disambiguates_nested():
    """_symbol_locations must resolve to the qualname-unique entry."""
    g = _build_graph(TWO_WRAPPERS)
    # Both nested wrappers are now distinct; the (qualname, file) key resolves.
    a = g._symbol_locations.get(("deco_a.wrapper", "mod.py"))
    b = g._symbol_locations.get(("deco_b.wrapper", "mod.py"))
    assert a is not None and b is not None
    assert a != b


def test_get_symbol_qualified_nested_qualname():
    g = _build_graph(TWO_WRAPPERS)
    s = g.get_symbol("deco_a.wrapper")
    assert s is not None
    assert s.qualname == "deco_a.wrapper"
    assert s.start_line == 2


def test_call_attribution_to_correct_wrapper():
    """A call inside ``deco_b.wrapper`` must be attributed to
    ``deco_b.wrapper``, not lumped onto the (formerly colliding) bare
    ``wrapper`` owner."""
    g = _build_graph(TWO_WRAPPERS)
    callees = {e.callee for e in g.get_callees("deco_b.wrapper")}
    assert "helper" in callees, callees
    # deco_a.wrapper must NOT own deco_b's helper call.
    a_callees = {e.callee for e in g.get_callees("deco_a.wrapper")}
    assert "helper" not in a_callees, a_callees


def test_get_callees_bare_suffix_still_matches():
    """Bare-name query (no scope qualifier) must still resolve via suffix
    matching against the now-qualified qualnames."""
    g = _build_graph(TWO_WRAPPERS)
    union = {e.callee for e in g.get_callees("wrapper")}
    assert "helper" in union or "func" in union, union


CLASS_IN_FUNCTION = """\
def factory():
    class Local:
        def m(self):
            return run()
    return Local
"""


def test_local_class_inside_function_qualified():
    """A class defined inside a function must get a qualified qualname so two
    functions defining a local class of the same name don't collide."""
    g = _build_graph(CLASS_IN_FUNCTION)
    assert g.get_symbol("factory.Local") is not None
    m = g.get_symbol("factory.Local.m")
    assert m is not None
    callees = {e.callee for e in g.get_callees("factory.Local.m")}
    assert "run" in callees, callees


TOPLEVEL_CLASS = """\
class MyClass:
    def method(self):
        return self.helper()

    def helper(self):
        return 1
"""


def test_toplevel_class_qualname_unchanged():
    """Top-level class qualnames must remain bare (``MyClass``) and methods
    stay ``MyClass.method`` — the nesting fix must not regress top-level
    class semantics exercised by downstream get_symbol('MyClass.method')."""
    g = _build_graph(TOPLEVEL_CLASS)
    assert g.get_symbol("MyClass") is not None
    m = g.get_symbol("MyClass.method")
    assert m is not None and m.qualname == "MyClass.method"
    callees = {e.callee for e in g.get_callees("MyClass.method")}
    assert "helper" in callees, callees


NESTED_IN_METHOD = """\
class Cache:
    def get(self, k):
        def miss():
            return load(k)
        return miss()
"""


def test_nested_function_inside_method_qualified():
    """A function nested inside a method must carry the class.method.fn path
    and correctly own its inner calls."""
    g = _build_graph(NESTED_IN_METHOD)
    miss = g.get_symbol("Cache.get.miss")
    assert miss is not None, "nested fn in method must be qualified"
    callees = {e.callee for e in g.get_callees("Cache.get.miss")}
    assert "load" in callees, callees
    # The method itself owns the call to miss().
    method_callees = {e.callee for e in g.get_callees("Cache.get")}
    assert "miss" in method_callees, method_callees
