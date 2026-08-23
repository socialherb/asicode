"""Cross-consumer shared parse cache (P1, 2026-08-11).

RepositoryGraph.extract_file and CallGraphIndexer._index_file both parse the
SAME Python files in the same turn. Before P1 each did its own ``read_text`` +
``ast.parse``; after P1 both route through ``external_llm.analysis.parse_cache``
so one cached parse serves every consumer (and the structural scanners, which
already used it). The two workers walk the AST read-only — GraphVisitor is an
``ast.NodeVisitor`` and _index_file keeps its ``class_names`` map in a side
table keyed by ``id(node)`` — so sharing one tree object is safe.
"""

import ast
import sys
import threading
import time

import pytest

from external_llm.agent.call_graph import CallGraphIndexer
from external_llm.analysis import parse_cache
from external_llm.graph import repository_graph as rg_module
from external_llm.graph.repository_graph import RepositoryGraph, _extract_cache


@pytest.fixture(autouse=True)
def _isolated_caches():
    """Reset both process-wide caches around every test."""
    parse_cache.clear()
    _extract_cache.clear()
    _extract_cache._gc_deficit = 0
    yield
    parse_cache.clear()
    _extract_cache.clear()


def test_extract_file_and_index_file_share_one_parse(tmp_path, monkeypatch):
    """The same file is ast.parsed ONCE across both workers (P1)."""
    f = tmp_path / "m.py"
    f.write_text("def foo():\n    return bar()\n\n\ndef bar():\n    pass\n", encoding="utf-8")

    real_parse = ast.parse
    parses: list[int] = []

    def counting_parse(src, *args, **kwargs):
        parses.append(1)
        return real_parse(src, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", counting_parse)

    rg = RepositoryGraph(str(tmp_path))
    payload = rg.extract_file(str(f))
    assert payload is not None and parses, "extract_file must parse the file"
    after_rg = len(parses)

    # Second consumer: same file, same stat → served from parse_cache, no parse.
    idx = CallGraphIndexer(str(tmp_path))
    idx._index_file(f)
    after_cgi = len(parses)

    assert after_rg == 1, f"extract_file parsed {after_rg} times"
    assert after_cgi == 1, f"CGI re-parsed ({after_cgi} total) instead of sharing parse_cache"


def test_extract_file_results_unchanged_by_routing(tmp_path):
    """Routing through parse_cache must not alter the extracted payload."""
    src = "class C:\n    def m(self):\n        return self.n()\n\n    def n(self):\n        return 1\n"
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")

    rg = RepositoryGraph(str(tmp_path))
    payload = rg.extract_file(str(f))
    assert payload is not None
    names = sorted(s.name for s in payload["symbols"])
    callees = sorted((c.caller, c.callee) for c in payload["calls"])
    # Symbol names are unqualified (C.m stored as "m"); call attribution
    # qualifies the caller ("C.m"). This is the pre-P1 contract — unchanged.
    assert names == ["C", "m", "n"], names
    assert ("C.m", "n") in callees, callees


def test_index_file_skips_on_unparseable_source(tmp_path):
    """parse_ast returns None on SyntaxError → _index_file skips (build parity)."""
    f = tmp_path / "broken.py"
    f.write_text("def (\n", encoding="utf-8")

    idx = CallGraphIndexer(str(tmp_path))
    idx._index_file(f)  # must not raise
    assert not idx._nodes, "unparseable file must contribute no nodes"


def test_non_utf8_py_decodes_lossily_and_produces_symbols(tmp_path):
    """F4 (2026-08-12): the decode policy is utf-8/replace, NOT strict.

    A latin-1 .py used to be silently skipped (read_text strict → UnicodeError
    → None); since P1 it decodes lossily and yields symbols.  That result is
    persisted into _extract_cache and the disk snapshot, so the contract must
    be pinned: both workers agree on the lossy decode.  (The non-UTF-8 byte
    must sit inside a string literal — a name/identifier containing an
    invalid byte becomes U+FFFD and is a SyntaxError, which still skips.)
    """
    f = tmp_path / "latin1.py"
    f.write_bytes(b"s = '\xe9'\n\ndef f():\n    return s\n")

    # parse_cache is the shared decoder — it must not raise and must parse.
    tree = parse_cache.parse_ast(str(f))
    assert tree is not None, "lossy decode must still parse"
    names = sorted(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    assert names == ["f"], names

    # RepositoryGraph.extract_file agrees (its payload is what gets persisted).
    rg = RepositoryGraph(str(tmp_path))
    payload = rg.extract_file(str(f))
    assert payload is not None
    sym_names = sorted(s.name for s in payload["symbols"])
    assert sym_names == ["f", "s"], sym_names

    # CallGraphIndexer agrees too — one shared policy across all consumers.
    idx = CallGraphIndexer(str(tmp_path))
    idx._index_file(f)
    assert idx._nodes, "lossy-decoded file must contribute nodes"


def test_extract_file_skips_giant_py_like_cgi(tmp_path, monkeypatch):
    """F5 (2026-08-12): extract_file applies the same 1 MiB per-file gate as
    CallGraphIndexer._index_file, so RG doesn't parse giant generated .py
    files into the SHARED parse_cache where CGI never reuses them."""
    f = tmp_path / "giant.py"
    f.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(rg_module, "_MAX_PY_BYTES", 4)  # smaller than the file

    rg = RepositoryGraph(str(tmp_path))
    assert rg.extract_file(str(f)) is None, "oversized file must be skipped"
    # And it must not have polluted the shared parse cache.
    assert parse_cache.cache_info().currsize == 0


def test_py_size_gate_single_source_of_truth():
    """P2 (2026-08-12): RG's per-file Python size gate is IMPORTED from the
    agent thresholds (agent/config/thresholds.py), not mirrored.

    A parallel hardcode drifted apart silently breaks CGI/RG parity on giant
    generated .py files: RG would parse what CGI skips, polluting the SHARED
    parse_cache with entries CGI never reuses (pure memory loss).  This test
    pins the binding so the drift class can't come back.
    """
    from external_llm.agent import call_graph as cg_module
    from external_llm.agent.config.thresholds import config

    assert rg_module._MAX_PY_BYTES == cg_module._MAX_PY_BYTES
    assert cg_module._MAX_PY_BYTES == config.lines.CALLGRAPH_PY_MAX_BYTES


# ── thread safety (C1/C2, 2026-08-12) ─────────────────────────────────────


def test_concurrent_parse_no_races_no_byte_drift(tmp_path, monkeypatch):
    """C2: concurrent parse_ast/read_source across threads must not raise and
    must never desync the byte accounting.

    Pre-fix the same harness raised ``KeyError: 'dictionary is empty'``
    (``_evict_lru``'s popitem racing another thread's eviction) and drifted
    ``_bytes`` to ~31x the real resident cost (lost updates on the global
    read-modify-write) — which silently collapsed the hit rate toward 0% for
    the rest of the process lifetime (``clear()`` was the only reset).

    The byte invariant is the regression core: ``_bytes`` (budget driver)
    must equal the sum of resident entry costs.  Shrinking
    ``_MAX_CACHE_BYTES`` forces the eviction path under contention.
    """
    files = []
    for i in range(40):
        p = tmp_path / f"m{i}.py"
        p.write_text(f"def f{i}():\n    return {i}\n", encoding="utf-8")
        files.append(str(p))
    monkeypatch.setattr(parse_cache, "_MAX_CACHE_BYTES", 10_000)  # force evictions

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        errors: list[str] = []
        stop = threading.Event()

        def worker(n: int) -> None:
            try:
                i = 0
                while not stop.is_set():
                    p = files[i % len(files)]
                    parse_cache.parse_ast(p)
                    parse_cache.read_source(p)
                    i += 1
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(f"worker{n}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop.set()
        for t in threads:
            t.join(timeout=15)
        assert not errors, f"concurrent parse_cache access raised: {errors[:3]}"
        with parse_cache._lock:
            tracked = parse_cache._bytes
            real = sum(cost for _, (_, cost) in parse_cache._cache.items())
        assert tracked == real, f"byte accounting drift: tracked {tracked} != real {real}"
        assert parse_cache.cache_info().currsize > 0, "cache must still hold entries"
    finally:
        sys.setswitchinterval(old)
