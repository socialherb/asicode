"""CallGraphIndexer cache-tier invariants after the P3 Stage 3 SSOT merge.

CGI's own disk snapshot (``.cache/call_graph_v1.json``) is GONE:
RepositoryGraph's ``.cache/structural_graph_v1.json`` is the single disk tier
(served via ``_rg_file_data`` — see test_call_graph_rg_ssot.py for the SSOT
behavior).  These tests pin the remaining invariants:

* a build NEVER writes ``.cache/call_graph_v1.json`` — no orphaned dual cache
  on any path (cold, warm, changed, deleted);
* warm rebuilds are served by the process-wide ``_file_cache`` with zero
  parses;
* cached payloads are never polluted by in-place callee resolution;
* invalidate_files() works on top of a cached build;
* a cancelled build leaves no index and no snapshot;
* the per-file size gate still excludes giant files from the index.
"""
import textwrap

import pytest

import external_llm.agent.call_graph as cg_module
from external_llm.agent.call_graph import CallGraphIndexer, _file_cache


@pytest.fixture(autouse=True)
def isolated_file_cache():
    """Save/restore the process-wide caches and gc rate-limit around every test."""
    saved = dict(_file_cache)
    saved_deficit = cg_module._file_cache_gc_deficit
    _file_cache.clear()
    cg_module._file_cache_gc_deficit = 0
    yield
    _file_cache.clear()
    _file_cache.update(saved)
    cg_module._file_cache_gc_deficit = saved_deficit


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

    idx._extract_file = _wrap
    return orig


def _graph_snapshot(idx):
    """JSON-izable snapshot of the full graph state for parity comparison."""
    return {
        "nodes": {s: (n.file, n.line, n.kind) for s, n in sorted(idx._nodes.items())},
        "forward": {
            s: sorted(
                (e.caller_symbol, e.caller_file, e.caller_line, e.callee_symbol,
                 e.callee_display, e.callee_file, e.callee_line, e.confidence)
                for e in es
            )
            for s, es in sorted(idx._forward.items())
        },
        "reverse": {
            s: sorted(
                (e.caller_symbol, e.caller_file, e.caller_line, e.callee_symbol,
                 e.callee_display, e.callee_file, e.callee_line, e.confidence)
                for e in es
            )
            for s, es in sorted(idx._reverse.items())
        },
        "file_nodes": {r: sorted(v) for r, v in sorted(idx._file_nodes.items())},
        "file_edges": {
            r: sorted(
                (e.caller_symbol, e.caller_file, e.caller_line, e.callee_symbol,
                 e.callee_display, e.callee_file, e.callee_line, e.confidence)
                for e in es
            )
            for r, es in sorted(idx._file_edges.items())
        },
        "def_sources": {
            s: sorted((r, line, kind) for r, (line, kind) in sorted(srcs.items()))
            for s, srcs in sorted(idx._def_sources.items())
        },
        "file_defs": {r: sorted(v) for r, v in sorted(idx._file_defs.items())},
    }


_FILES = {
    "mod.py": """
        def helper():
            return 1

        def caller():
            return helper()

        class Service:
            def run(self):
                return self._work()

            def _work(self):
                return helper()
    """,
    "other.py": """
        from mod import caller

        def top():
            return caller()
    """,
}


# ─── SSOT invariant: no CGI disk snapshot ever ────────────────────────────────

def test_build_never_writes_call_graph_v1_json(tmp_path):
    """The removed CGI disk tier must stay gone: no build path (cold, warm,
    changed, deleted) may ever create ``.cache/call_graph_v1.json`` (P3 Stage 3)."""
    repo = _make_repo(tmp_path, _FILES)
    idx = CallGraphIndexer(repo)
    idx.build()
    assert not (tmp_path / ".cache" / "call_graph_v1.json").exists()

    idx.invalidate()
    idx.build()
    assert not (tmp_path / ".cache" / "call_graph_v1.json").exists()

    (tmp_path / "mod.py").write_text(
        textwrap.dedent("""
            def helper():
                return 2

            def brand_new():
                return helper()
        """),
        encoding="utf-8",
    )
    _file_cache.clear()
    idx2 = CallGraphIndexer(repo)
    idx2.build()
    assert "brand_new" in idx2._nodes
    assert not (tmp_path / ".cache" / "call_graph_v1.json").exists()

    (tmp_path / "other.py").unlink()
    _file_cache.clear()
    idx3 = CallGraphIndexer(repo)
    idx3.build()
    assert "top" not in idx3._nodes
    assert not (tmp_path / ".cache" / "call_graph_v1.json").exists()


# ─── warm in-process tier ─────────────────────────────────────────────────────

def test_warm_rebuild_served_by_in_process_tier(tmp_path):
    repo = _make_repo(tmp_path, _FILES)
    idx = CallGraphIndexer(repo)
    idx.build()
    idx.invalidate()  # full reset — the rebuild must not re-parse
    counter = {"n": 0}
    _count_computes(idx, counter)
    idx.build()
    assert counter["n"] == 0
    assert idx.cache_stats["hit"] == 2
    assert idx.cache_stats["changed"] == 0
    assert idx.cache_stats["total"] == 2


def test_cached_payload_not_polluted_by_callee_resolution(tmp_path):
    repo = _make_repo(tmp_path, _FILES)
    idx1 = CallGraphIndexer(repo)
    idx1.build()
    # The graph's edges were resolved in place (callee_file filled)…
    assert all(e.callee_file == "mod.py" for es in idx1._forward.values() for e in es)
    # …but the cached payloads must still be pristine pre-resolution dicts.
    for key, (_mtime_ns, _size, payload) in _file_cache.items():
        for d in payload["calls"]:
            assert d["callee_file"] is None, f"cache polluted by resolution: {key} {d}"
            assert d["callee_line"] is None
    # A later rebuild from the cache still resolves identically.
    idx1.invalidate()
    counter = {"n": 0}
    _count_computes(idx1, counter)
    idx1.build()
    assert counter["n"] == 0
    assert all(e.callee_file == "mod.py" for es in idx1._forward.values() for e in es)


# ─── incremental interplay ────────────────────────────────────────────────────

def test_invalidate_files_on_top_of_cached_build(tmp_path):
    repo = _make_repo(tmp_path, _FILES)
    idx1 = CallGraphIndexer(repo)
    idx1.build()
    _file_cache.clear()
    idx2 = CallGraphIndexer(repo)
    idx2.build()  # RG-tier-served or computed — either way a built index
    (tmp_path / "other.py").write_text(
        textwrap.dedent("""
            def top():
                return 42
        """),
        encoding="utf-8",
    )
    idx2.invalidate_files(["other.py"])
    assert idx2._built is True
    assert "helper" not in idx2._file_defs.get("other.py", ())
    assert "top" in idx2._nodes
    # Full rebuild afterwards reflects the new file (no ghost of the old).
    idx2.invalidate()
    idx2.build()
    assert "caller" not in idx2._file_defs.get("other.py", ())
    assert idx2.get_callees("top") == []


# ─── cancel / gates ───────────────────────────────────────────────────────────

def test_cancelled_build_leaves_no_index(tmp_path):
    import threading

    repo = _make_repo(tmp_path, _FILES)
    ev = threading.Event()
    ev.set()
    idx = CallGraphIndexer(repo, cancel_event=ev)
    idx.build()
    assert idx._built is False
    assert idx._nodes == {}
    assert not (tmp_path / ".cache" / "call_graph_v1.json").exists()


def test_size_gate_files_not_indexed(tmp_path):
    big = tmp_path / "huge.py"
    big.write_text("x = 1\n" + "y = 1\n" * 200_000, encoding="utf-8")
    _make_repo(tmp_path, _FILES)
    idx = CallGraphIndexer(str(tmp_path))
    idx.build()
    assert "x" not in idx._nodes
