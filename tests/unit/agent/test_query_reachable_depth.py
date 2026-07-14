"""Regression tests for _query_reachable BFS depth bounding.

Covers an off-by-one defect: _query_reachable recorded neighbors at DISCOVERY
time (a node at ``depth`` appends children at ``depth + 1``) but guarded the
queue at POP time with ``depth > max_depth``. A node already AT ``max_depth``
was therefore still expanded, recording children at ``max_depth + 1`` — one
level past the bound the result header promises ("depth ≤max_depth"). The fix
stops expansion at ``depth >= max_depth``.

The companion mode _query_transitive_importers records at POP time (after its
own ``> max_depth`` guard) and was already correct; a parity test pins both to
the same bound semantics so they cannot drift apart again.
"""
from types import SimpleNamespace

from external_llm.agent.tool_handlers.analysis_tools import AnalysisToolsMixin


class _FakeResult:
    def __init__(self, ok, content, metadata=None, error=""):
        self.ok = ok
        self.content = content
        self.metadata = metadata or {}
        self.error = error


def _make_host(callee_chain=None, importer_chain=None):
    """Build an AnalysisToolsMixin host backed by a linear fake call graph.

    callee_chain / importer_chain map a node to its direct successors so a
    straight line A->B->C->... can be walked one hop per depth level.
    """
    _callees = callee_chain or {}
    _importers = importer_chain or {}

    class _FakeGraph:
        def get_callees(self, sym, file_path=None):
            return [
                SimpleNamespace(callee_symbol=n, callee_file="f.py",
                                caller_file="f.py", callee_line=1)
                for n in _callees.get(sym, [])
            ]

        def get_callers(self, sym, file_path=None):
            return []

        def get_importers(self, f):
            return _importers.get(f, [])

    class _Host(AnalysisToolsMixin):
        repo_root = "/tmp"
        _call_graph = _FakeGraph()

        def _make_result(self, ok, content, metadata=None, error=""):
            return _FakeResult(ok, content, metadata, error)

    return _Host()


_CHAIN = {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["E"], "E": []}


def test_reachable_respects_max_depth_bound():
    """Recorded depths must never exceed max_depth (was leaking max_depth+1)."""
    host = _make_host(callee_chain=_CHAIN)
    for md in (1, 2, 3):
        res = host._query_reachable("A", "downstream", max_depth=md, limit=50)
        depths = {x["symbol"]: x["depth"] for x in res.metadata["reachable"]}
        assert depths, f"max_depth={md}: nothing reachable"
        assert max(depths.values()) <= md, (
            f"max_depth={md} leaked beyond bound: {depths}")


def test_reachable_max_depth_one_stops_at_one_hop():
    """max_depth=1 must yield only the direct neighbor B, not C (the 2-hop)."""
    host = _make_host(callee_chain=_CHAIN)
    res = host._query_reachable("A", "downstream", max_depth=1, limit=50)
    syms = {x["symbol"] for x in res.metadata["reachable"]}
    assert syms == {"B"}, f"expected only B at depth 1, got {syms}"


def test_reachable_max_depth_zero_yields_nothing():
    """0 hops from the source reaches no other symbol (was leaking B)."""
    host = _make_host(callee_chain=_CHAIN)
    res = host._query_reachable("A", "downstream", max_depth=0, limit=50)
    assert not res.metadata.get("reachable"), (
        f"depth 0 should reach nothing, got {res.metadata.get('reachable')}")


def test_reachable_importers_parity_same_bound():
    """reachable and transitive-importers must report the SAME node count for an
    equivalent linear chain at each max_depth — guarding against the two BFS
    bound semantics drifting apart again.
    """
    callee = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
    importer = {"a.py": ["b.py"], "b.py": ["c.py"], "c.py": ["d.py"], "d.py": []}
    host = _make_host(callee_chain=callee, importer_chain=importer)
    for md in (1, 2, 3):
        r_reach = host._query_reachable("A", "downstream", max_depth=md, limit=50)
        r_imp = host._query_transitive_importers("a.py", max_depth=md, limit=50)
        n_reach = len(r_reach.metadata.get("reachable", []))
        n_imp = len(r_imp.metadata.get("importers", []))
        assert n_reach == n_imp == md, (
            f"max_depth={md}: reachable={n_reach}, importers={n_imp} "
            f"(both should equal {md})")
