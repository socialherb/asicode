"""P3 Stage 2: RepositoryGraph snapshot as the CallGraphIndexer SSOT.

The release gate's ``.cache/structural_graph_v1.json`` becomes the single
source of truth for the agent call graph: when its manifest stamp matches a
file's current stat, the build serves RG's per-file extraction CONVERTED to
the CGI payload shape (``_rg_payload_to_cgi``) instead of re-parsing.  RG's
extraction carries CGI-convention fields (``cgi_symbol``/``caller_symbol``/
``caller_def_line``/``is_async``/``ast_depth``) and RG's ``generic_visit``
mirrors CGI's LIFO call traversal, so the converted graph must be
bit-for-bit identical to a cold parse of the same tree.  These tests pin:

* a fresh RG snapshot serves 100% of files with ZERO CGI parses, and the
  resulting graph equals a full-parse build (nodes, forward edges, order);
* decorator calls attribute to the decorated function (not the enclosing
  function), matching CGI;
* nested-function / same-qualname-redefinition cases stay distinct;
* a stale RG snapshot (file changed after it was written) falls back to
  CGI's own parse for that file only;
* an absent/corrupt RG snapshot fails open to a fresh parse.
"""
import textwrap
from pathlib import Path
from unittest import mock

import pytest

import external_llm.agent.call_graph as cg_module
from external_llm.agent.call_graph import CallGraphIndexer, _file_cache
from external_llm.graph.repository_graph import RepositoryGraph


@pytest.fixture(autouse=True)
def isolated_file_cache():
    """Save/restore the process-wide caches and gc rate-limit around every test."""
    saved = dict(_file_cache)
    saved_deficit = _file_cache._gc_deficit
    _file_cache.clear()
    _file_cache._gc_deficit = 0
    yield
    _file_cache.clear()
    _file_cache.update(saved)
    _file_cache._gc_deficit = saved_deficit


def _make_repo(tmp_path, files: dict) -> str:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(tmp_path)


def _count_computes(idx, counter):
    """Wrap an indexer's _extract_file to count fresh parses (returns wrapper)."""
    orig = idx._extract_file

    def _wrap(path):
        counter["n"] += 1
        return orig(path)

    return _wrap


def _edge_key(e):
    return (
        e.caller_symbol, e.caller_file, e.caller_line,
        e.callee_symbol, e.callee_display, e.confidence,
        tuple(e.call_args), e.is_mutating,
    )


def _graph_snapshot(idx):
    """(nodes, forward) tuples — order-sensitive, so parity is exact."""
    nodes = tuple(sorted((s, n.file, n.line, n.kind) for s, n in idx._nodes.items()))
    fwd = tuple(
        sorted((k, tuple(_edge_key(e) for e in v)) for k, v in idx._forward.items())
    )
    return nodes, fwd


SRC = {
    "app.py": """
        import logging
        from rich.markup import escape as _escape

        logger = logging.getLogger(__name__)

        class _Margin:
            def __init__(self, width):
                self._w = width
                self._setup()

            def _setup(self):
                logger.debug("setup %s", self._w)

        def _helper(x):
            if isinstance(x, str):
                return _escape(x)
            return x

        def main():
            m = _Margin(4)
            _helper(m)
            logger.warning("done")

        main()
    """,
    "mod2.py": """
        import os

        def _spin():
            import time
            while not _stop.wait(0.1):
                time.sleep(0.01)
            return os.path.exists("/tmp/x")

        def outer():
            def _spin():
                return 1
            return _spin()
    """,
}


@pytest.fixture()
def rg_indexer(tmp_path):
    """Repo with an RG snapshot already built; returns (repo, CallGraphIndexer)."""
    repo = _make_repo(tmp_path, SRC)
    rg = RepositoryGraph(repo)
    rg.build(collect_imported_names=True)  # writes .cache/structural_graph_v1.json
    idx = CallGraphIndexer(repo)
    return repo, idx


def test_fresh_rg_snapshot_serves_zero_parse_and_parity(rg_indexer):
    repo, idx = rg_indexer
    counter = {"n": 0}
    idx._extract_file = _count_computes(idx, counter)
    idx.build()

    assert counter["n"] == 0, "fresh RG snapshot must serve every py file"
    assert idx.cache_stats["changed"] == 0
    assert idx.cache_stats["total"] == 2

    # parity: full-parse build (RG tier disabled) == RG-served build
    idx2 = CallGraphIndexer(repo)
    with mock.patch.object(CallGraphIndexer, "_rg_file_data", return_value=None):
        idx2.build()
    assert _graph_snapshot(idx) == _graph_snapshot(idx2)


def test_decorator_call_attributed_to_decorated_function(rg_indexer):
    repo, idx = rg_indexer
    # add a decorated function to the repo, rebuild RG snapshot, rebuild CGI
    p = Path(repo) / "app.py"
    p.write_text(
        p.read_text(encoding="utf-8") + "\n" + textwrap.dedent("""
            def mount(app):
                @app.get("/")
                def ui_root():
                    return 1
        """),
        encoding="utf-8",
    )
    rg = RepositoryGraph(repo)
    rg.build(collect_imported_names=True)
    idx = CallGraphIndexer(repo)
    idx.build()
    callers = sorted(e.caller_symbol for e in idx._forward.get("ui_root", []))
    assert callers == ["ui_root"], f"decorator call must attribute to ui_root: {callers}"


def test_stale_rg_snapshot_falls_back_for_changed_file(rg_indexer):
    repo, idx = rg_indexer
    # touch mod2.py AFTER the RG snapshot was written
    p = Path(repo) / "mod2.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    counter = {"n": 0}
    idx._extract_file = _count_computes(idx, counter)
    idx.build()
    # only the changed file re-parses; the rest serve from RG snapshot
    assert counter["n"] == 1
    assert idx.cache_stats["changed"] == 1
    assert idx.cache_stats["total"] == 2


def test_absent_rg_snapshot_falls_back_to_legacy_tiers(tmp_path):
    repo = _make_repo(tmp_path, SRC)
    idx = CallGraphIndexer(repo)  # no RG build -> no structural snapshot
    counter = {"n": 0}
    idx._extract_file = _count_computes(idx, counter)
    idx.build()
    assert counter["n"] == 2, "absent RG snapshot -> full CGI parse"
    assert idx.cache_stats["changed"] == 2


def test_corrupt_rg_snapshot_fails_open(tmp_path):
    repo = _make_repo(tmp_path, SRC)
    snap = Path(repo) / ".cache" / "structural_graph_v1.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text("{not json", encoding="utf-8")
    idx = CallGraphIndexer(repo)
    counter = {"n": 0}
    idx._extract_file = _count_computes(idx, counter)
    idx.build()
    assert counter["n"] == 2, "corrupt RG snapshot must fail open to CGI parse"


def test_invalid_rg_snapshot_loads_once_not_per_file(tmp_path, monkeypatch):
    """A corrupt / version-mismatched RG snapshot must load ONCE per build.

    Regression (2026-08-12): the CGI ``_rg_cache`` tier (P3 Stage 2) copied
    the pre-F9 pattern — a failed load pinned the mtime marker to 0, so the
    per-file RG tier (``_rg_file_data``) re-read + re-parsed the whole
    snapshot JSON for EVERY file.  On asicode (818 py files, ~39MB
    snapshot) a schema-version bump from new CallEdge fields would turn
    agent builds into the same 300s+ hang F9 fixed on the RG side.  The
    marker must pin the CURRENT mtime: only a REWRITTEN snapshot (new
    mtime) or a fresh build (marker reset in build()) retries the load.
    """
    repo = _make_repo(tmp_path, SRC)
    snap = Path(repo) / ".cache" / "structural_graph_v1.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text("{not json", encoding="utf-8")

    loads = {"n": 0}
    orig = cg_module._load_rg_snapshot

    def counting(path):
        loads["n"] += 1
        return orig(path)

    monkeypatch.setattr(cg_module, "_load_rg_snapshot", counting)

    idx = CallGraphIndexer(repo)
    counter = {"n": 0}
    idx._extract_file = _count_computes(idx, counter)
    idx.build()
    assert loads["n"] == 1, "corrupt RG snapshot must load ONCE per build"
    assert counter["n"] == 2, "every file must still fail open to a fresh parse"


def test_nested_same_qualname_defs_stay_distinct(rg_indexer):
    _repo, idx = rg_indexer
    idx.build()
    # two _spin functions (module-level and nested) — both must keep edges
    spins = [e for k, v in idx._forward.items() if k == "_spin" for e in v]
    assert len(spins) >= 2, "module _spin and nested _spin must both keep edges"
    lines = sorted(e.caller_line for e in spins)
    assert len(set(lines)) == len(lines), "each _spin's edges must stay distinct"


def test_rg_conversion_drops_unsupported_fallback_edges():
    from external_llm.agent.call_graph import _rg_payload_to_cgi
    payload = {
        "symbols": [
            {"name": "f", "qualname": "f", "kind": "function", "start_line": 1,
             "cgi_symbol": "f", "is_async": False, "ast_depth": 1},
        ],
        "calls": [
            # legacy fallback for chained call obj.m()() — CGI emits nothing
            # (P2 shape: explicit resolution marker, low confidence)
            {"caller": "f", "caller_symbol": "f", "caller_def_line": 1,
             "file_path": "app.py", "line": 3, "callee_symbol": "obj.m",
             "callee_display": "obj.m", "confidence": 0.2,
             "resolution": "fallback",
             "call_args": [], "is_mutating": False},
            # P2 (2026-08-12): confidence==1.0 WITHOUT a resolution marker is
            # now a REAL edge — the old drop clause is gone (it matched zero
            # edges and CallEdge.confidence defaults to 1.0, so it would
            # silently swallow any future edge that omits the field).
            {"caller": "f", "caller_symbol": "f", "caller_def_line": 1,
             "file_path": "app.py", "line": 5, "callee_symbol": "obj.n",
             "callee_display": "obj.n", "confidence": 1.0,
             "call_args": [], "is_mutating": False},
            # normal call
            {"caller": "f", "caller_symbol": "f", "caller_def_line": 1,
             "file_path": "app.py", "line": 4, "callee_symbol": "g",
             "callee_display": "g", "confidence": 0.9,
             "call_args": [], "is_mutating": False},
        ],
    }
    out = _rg_payload_to_cgi(payload)
    assert out["defs"] == [["f", 1, "function"]]
    # fallback dropped; the confidence==1.0 edge (no resolution marker) kept
    assert [c["callee_display"] for c in out["calls"]] == ["obj.n", "g"]


def test_rg_conversion_async_kind_and_method_kind():
    from external_llm.agent.call_graph import _rg_payload_to_cgi
    payload = {
        "symbols": [
            {"name": "go", "qualname": "C.go", "kind": "method", "start_line": 2,
             "cgi_symbol": "C.go", "is_async": False, "ast_depth": 2},
            {"name": "run", "qualname": "run", "kind": "function", "start_line": 1,
             "cgi_symbol": "run", "is_async": True, "ast_depth": 1},
        ],
        "calls": [],
    }
    out = _rg_payload_to_cgi(payload)
    # BFS order: (ast_depth 1) run first, then (2) C.go
    assert out["defs"] == [["run", 1, "async_function"], ["C.go", 2, "method"]]


def test_missing_rg_snapshot_self_heals_via_graph_builder(tmp_path, monkeypatch):
    """P1 (2026-08-12): a gate-less build (0 RG-served, no snapshot file)
    triggers ONE GraphBuilder build so the SSOT snapshot self-heals."""
    from external_llm.graph import graph_builder as gb_module

    repo = _make_repo(tmp_path, {
        "a.py": "def fa():\n    return 1\n",
        "b.py": "def fb():\n    fa()\n",
    })
    calls = {"n": 0}
    orig = gb_module.GraphBuilder.build_repo_graph

    def counting(self, repo_root=None):
        calls["n"] += 1
        return orig(self, repo_root)

    monkeypatch.setattr(gb_module.GraphBuilder, "build_repo_graph", counting)

    idx = CallGraphIndexer(repo)
    idx.build()
    assert calls["n"] == 1, "missing RG snapshot must self-heal exactly once"
    assert idx.cache_stats["rg_served"] == 0
    assert idx.cache_stats["rg_self_healed"] == 1
    snapshot = tmp_path / ".cache" / "structural_graph_v1.json"
    assert snapshot.exists(), "self-heal must create the RG snapshot"

    # A fresh process now serves every file from the healed snapshot.
    _file_cache.clear()
    idx2 = CallGraphIndexer(repo)
    idx2.build()
    assert idx2.cache_stats["rg_served"] == 2
    assert idx2.cache_stats["rg_self_healed"] == 0
    assert idx2.cache_stats["hit"] == 2


def test_present_rg_snapshot_no_self_heal(tmp_path, monkeypatch):
    """P1: a usable snapshot (files served) must NOT trigger self-heal."""
    from external_llm.graph import graph_builder as gb_module

    repo = _make_repo(tmp_path, {"a.py": "def fa():\n    pass\n"})
    RepositoryGraph(repo).build(collect_imported_names=True)  # write snapshot
    calls = {"n": 0}
    orig = gb_module.GraphBuilder.build_repo_graph

    def counting(self, repo_root=None):
        calls["n"] += 1
        return orig(self, repo_root)

    monkeypatch.setattr(gb_module.GraphBuilder, "build_repo_graph", counting)

    _file_cache.clear()
    idx = CallGraphIndexer(repo)
    idx.build()
    assert calls["n"] == 0
    assert idx.cache_stats["rg_served"] == 1
    assert idx.cache_stats["rg_self_healed"] == 0


def test_confidence_1_0_edge_without_fallback_marker_is_kept():
    """P2 (2026-08-12): the dead ``confidence == 1.0`` drop clause is gone.

    It matched zero edges (every legacy fallback carries
    ``resolution == "fallback"``) but ``CallEdge.confidence`` defaults to
    1.0, so any future edge that omits it would have been silently
    swallowed from the call graph."""
    from external_llm.agent.call_graph import _rg_payload_to_cgi

    payload = {
        "symbols": [{
            "name": "f", "qualname": "m.f", "kind": "function",
            "start_line": 1, "ast_depth": 0, "cgi_symbol": "f", "is_async": False,
        }],
        "calls": [{
            "caller": "m.f", "caller_symbol": "m.f", "caller_def_line": 1,
            "file_path": "m.py", "line": 3, "callee_symbol": None,
            "callee_display": "g", "confidence": 1.0,  # no resolution key
        }],
        "imports": [],
    }
    cgi = _rg_payload_to_cgi(payload)
    assert len(cgi["calls"]) == 1
    assert cgi["calls"][0]["callee_display"] == "g"
