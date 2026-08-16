"""Tests for the non-Python extraction cache (P2, 2026-08-11).

``_process_file_ripgrep`` mirrors ``_process_file_cached``'s two-tier order
(in-process ``_extract_cache`` → parse via ``_extract_non_python``) with the SAME
staleness contract and admission control.  These tests pin the contract:

* a non-Python file parsed once is served bit-for-bit identically on every
  rebuild without re-reading or re-parsing;
* the process-wide ``_extract_cache`` carries non-Python payloads alongside
  Python ones (language-agnostic key + shape); and
* edits (mtime/size change) force a re-parse while unchanged files do not.

The tests stub tree-sitter_utils so they need no installed grammar, exactly
like the P1 end-to-end tests in ``test_symbol_range_index.py``.
"""
import os
import time

import pytest

from external_llm.graph import repository_graph as rg_module
from external_llm.graph.repository_graph import (
    RepositoryGraph,
    _extract_cache,
    _extract_cache_key,
)


@pytest.fixture(autouse=True)
def isolated_extract_cache():
    """Save/restore the process-wide cache and gc rate-limit around every test."""
    saved = dict(_extract_cache)
    saved_deficit = _extract_cache._gc_deficit
    _extract_cache.clear()
    _extract_cache._gc_deficit = 0
    yield
    _extract_cache.clear()
    _extract_cache.update(saved)
    _extract_cache._gc_deficit = saved_deficit


def _stub_tree_sitter(monkeypatch, symbols, calls, imports=None):
    """Stub tree_sitter_utils so the unit test needs no installed grammar."""
    monkeypatch.setattr(
        "external_llm.languages.tree_sitter_utils.is_available", lambda: True
    )
    # tree= kwarg: _extract_non_python parses once and shares the tree (P5).
    monkeypatch.setattr(
        "external_llm.languages.tree_sitter_utils.find_all_symbols",
        lambda content, lang, tree=None: symbols,
    )
    monkeypatch.setattr(
        "external_llm.languages.tree_sitter_utils.extract_calls",
        lambda content, lang, tree=None: calls,
    )
    monkeypatch.setattr(
        "external_llm.languages.tree_sitter_utils.extract_imports",
        lambda content, lang, tree=None: imports or [],
    )


def _snapshot(graph):
    """A hashable summary of a graph for bit-for-bit comparison across builds."""
    return (
        frozenset((uid, node.name, node.qualname, node.kind, node.start_line, node.end_line)
                  for uid, node in graph.symbols.items()),
        tuple((e.caller, e.callee, e.file_path, e.line) for e in graph.call_edges),
        tuple((e.importer, e.imported, e.import_type) for e in graph.import_edges),
    )


def test_nonpython_file_is_cached_and_served_identically(tmp_path, monkeypatch):
    """build() #2 must serve the .ts file from cache with a byte-identical graph."""
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("alpha", "function", 1, 5), ("beta", "function", 7, 9)],
        calls=[("helper", 2), ("helper", 8)],
        imports=[],
    )
    (tmp_path / "mod.ts").write_text(
        "export function alpha() { helper(); }\nexport function beta() { helper(); }\n",
        encoding="utf-8",
    )

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)  # track=True → counters accumulate
    before = _snapshot(g)
    assert len(g.symbols) == 2
    assert len(g.call_edges) == 2
    # First build with collect_imported_names=True → track=True → first build
    # is all misses.
    assert g.cache_stats["hit"] == 0
    # The file was admitted into the process-wide cache.
    key = _extract_cache_key(str(tmp_path), str(tmp_path / "mod.ts"))
    assert key in _extract_cache

    # Second build — nothing changed → served from cache, identical graph.
    g.build(collect_imported_names=True)
    after = _snapshot(g)
    assert after == before, "cached non-Python payload must inject bit-for-bit identically"
    assert g.cache_stats["hit"] == 1, "the .ts file must count as a cache hit on build #2"


def test_nonpython_file_edit_forces_reparse(tmp_path, monkeypatch):
    """An mtime change must invalidate the cached payload and re-parse."""
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("alpha", "function", 1, 5)],
        calls=[("helper", 2)],
        imports=[],
    )
    f = tmp_path / "mod.ts"
    f.write_text("export function alpha() { helper(); }\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)
    assert g.cache_stats["hit"] == 0
    # First build parses+admits the .ts file.  Non-Python re-parses land in
    # _fresh_parsed_nonpy (the persisted snapshot is py-only, so _fresh_parsed
    # stays the py-only rewrite-hint list; P1, 2026-08-11).
    assert len(g._fresh_parsed_nonpy) == 1
    assert g._fresh_parsed == []

    # Edit the file — new mtime → cache miss → re-parse.
    f.write_text("export function alpha() { helper(); other(); }\n", encoding="utf-8")
    # Force a distinct mtime so the staleness check fires reliably across
    # filesystems with coarse mtime granularity.
    ts = time.time() + 5
    os.utime(f, (ts, ts))

    g.build(collect_imported_names=True)
    # Re-parsed (admitted) → counted as changed, not a hit.
    assert g.cache_stats["hit"] == 0
    assert len(g._fresh_parsed_nonpy) == 1
    assert g.cache_stats["changed"] == 1


def test_nonpython_and_python_share_one_cache(tmp_path, monkeypatch):
    """Both languages populate the same _extract_cache under lang-agnostic keys."""
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("tsfn", "function", 1, 3)],
        calls=[],
        imports=[],
    )
    (tmp_path / "mod.py").write_text("def pyfn():\n    pass\n", encoding="utf-8")
    (tmp_path / "mod.ts").write_text("export function tsfn() {}\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)

    py_key = _extract_cache_key(str(tmp_path), str(tmp_path / "mod.py"))
    ts_key = _extract_cache_key(str(tmp_path), str(tmp_path / "mod.ts"))
    assert py_key in _extract_cache, "Python file cached"
    assert ts_key in _extract_cache, "non-Python file cached in the SAME cache"
    assert {s.name for s in g.symbols.values()} == {"pyfn", "tsfn"}

    # Second build serves BOTH from cache.
    g.build(collect_imported_names=True)
    assert g.cache_stats["hit"] == 2


def test_nonpython_intra_file_symbol_dedup_preserved(tmp_path, monkeypatch):
    """Duplicate qualnames within one file keep only the first (legacy dedup)."""
    _stub_tree_sitter(
        monkeypatch,
        symbols=[
            ("dup", "function", 1, 2),
            ("dup", "function", 4, 5),  # same qualname → dropped
            ("unique", "function", 7, 8),
        ],
        calls=[],
        imports=[],
    )
    (tmp_path / "mod.ts").write_text(
        "export function dup() {}\nexport function dup() {}\nexport function unique() {}\n",
        encoding="utf-8",
    )
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert sorted(s.name for s in g.symbols.values()) == ["dup", "unique"]


def test_nonpython_reparse_file_refreshes_cache(tmp_path, monkeypatch):
    """reparse_file() routes non-Python files through the cached processor (P2)."""
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("alpha", "function", 1, 3)],
        calls=[],
        imports=[],
    )
    f = tmp_path / "mod.ts"
    f.write_text("export function alpha() {}\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert len(g.symbols) == 1

    # Simulate an external edit, then incremental reparse via the facade path.
    f.write_text("export function alpha() { renamed(); }\n", encoding="utf-8")
    ts = time.time() + 5
    os.utime(f, (ts, ts))

    g.reparse_file(str(f))
    # The cache now reflects the re-parsed file — a subsequent build serves it
    # from cache (hit), not by re-parsing.
    key = _extract_cache_key(str(tmp_path), str(f))
    assert key in _extract_cache
    assert len(g.symbols) == 1


# ══════════════════════════════════════════════════════════════════════════════
# P1 (2026-08-11): language-agnostic cache_stats + admitted-only "changed"
# ══════════════════════════════════════════════════════════════════════════════

def test_cache_stats_count_py_and_nonpy_walks(tmp_path, monkeypatch):
    """total covers both languages; hit+changed never exceeds total.

    Regression: before P1, ``total`` counted only .py files while ``hit`` and
    ``changed`` counted non-Python too — a warm mixed build reported hit(847)
    > total(806), a 105% hit rate.  The invariant ``hit + changed <= total``
    now holds for every build.
    """
    _stub_tree_sitter(monkeypatch, symbols=[("tsfn", "function", 1, 3)], calls=[], imports=[])
    (tmp_path / "mod.py").write_text("def pyfn():\n    pass\n", encoding="utf-8")
    (tmp_path / "mod.ts").write_text("export function tsfn() {}\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)
    assert g.cache_stats["total"] == 2
    assert g.cache_stats["hit"] == 0
    assert g.cache_stats["changed"] == 2  # py + non-py admitted re-parses
    assert len(g._py_stamps) == 1
    assert len(g._nonpy_stamps) == 1
    assert g.cache_stats["hit"] + g.cache_stats["changed"] <= g.cache_stats["total"]

    g.build(collect_imported_names=True)
    assert g.cache_stats["total"] == 2
    assert g.cache_stats["hit"] == 2
    assert g.cache_stats["changed"] == 0
    assert g.cache_stats["hit"] + g.cache_stats["changed"] <= g.cache_stats["total"]


def test_nonpython_cap_overflow_reparse_not_counted_as_changed(tmp_path, monkeypatch):
    """Cap-overflow non-Python re-parses must not count as "changed" (P1).

    Mirrors the Python path's P0 contract: a file beyond the admission cap is
    re-parsed every build but never persisted, so counting it would skew
    cache_stats["changed"] and fire the snapshot rewrite hint forever.  The
    pre-P1 code appended non-Python re-parses unconditionally (ignoring
    ``admitted``), so an N>cap repo reported changed=N every build.
    """
    _stub_tree_sitter(monkeypatch, symbols=[("fn", "function", 1, 3)], calls=[], imports=[])
    for i in range(6):
        (tmp_path / f"mod_{i}.ts").write_text(f"export function fn{i}() {{}}\n", encoding="utf-8")
    monkeypatch.setattr(rg_module, "_EXTRACT_CACHE_MAX_ENTRIES", 4)
    monkeypatch.setattr(rg_module._extract_cache, "cap", 4)

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)
    assert g.cache_stats["total"] == 6
    assert g.cache_stats["hit"] == 0
    assert g.cache_stats["changed"] == 4  # only the 4 ADMITTED re-parses
    assert g.cache_stats["hit"] + g.cache_stats["changed"] <= g.cache_stats["total"]

    g.build(collect_imported_names=True)
    assert g.cache_stats["hit"] == 4  # admitted subset served from cache
    assert g.cache_stats["changed"] == 0  # overflow re-parses are NOT changes
    assert g.cache_stats["hit"] + g.cache_stats["changed"] <= g.cache_stats["total"]


# ══════════════════════════════════════════════════════════════════════════════
# P2 (2026-08-11): stat → cache lookup → READ ONLY ON MISS
# ══════════════════════════════════════════════════════════════════════════════

def test_nonpython_cache_hit_does_not_reparse(tmp_path, monkeypatch):
    """A cache-hit build must not re-extract the file (and so not re-read it).

    Regression: the pre-P2 order read the file BEFORE the cache lookup, so a
    warm build re-read every non-Python file from disk (1.25MB per rebuild on
    asicode; multi-hundred-MB on TS/JS-heavy repos) even though the cached
    payload was served.  The reorder (stat → lookup → read on miss) makes the
    warm path a pure stat + dict lookup.
    """
    _stub_tree_sitter(monkeypatch, symbols=[("alpha", "function", 1, 3)], calls=[], imports=[])
    (tmp_path / "mod.ts").write_text("export function alpha() {}\n", encoding="utf-8")

    extract_calls = {"n": 0}
    orig = RepositoryGraph._extract_non_python

    def counting(self, file_path, content):
        extract_calls["n"] += 1
        return orig(self, file_path, content)

    monkeypatch.setattr(RepositoryGraph, "_extract_non_python", counting)

    g = RepositoryGraph(str(tmp_path))
    g.build(collect_imported_names=True)
    assert extract_calls["n"] == 1  # cold build parses once
    g.build(collect_imported_names=True)
    assert extract_calls["n"] == 1  # warm build: served from cache, NO re-parse
    assert g.cache_stats["hit"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# P3 (2026-08-11): dedup'd same-name functions still capture calls in their span
# ══════════════════════════════════════════════════════════════════════════════

def test_duplicate_function_name_still_attributes_calls(tmp_path, monkeypatch):
    """A dedup'd (same-name) function must still capture calls in ITS span.

    Regression: dedup skipped ``_sym_ranges.append`` for the second same-named
    function, so a call inside it was attributed to an OUTER function (or "")
    — the tightest-enclosing index (P1, aada701a) was exact, but its input was
    already missing the duplicate's range.  The symbol itself stays dedup'd
    (first qualname wins, legacy contract); only the attribution range is kept
    for every function.
    """
    _stub_tree_sitter(
        monkeypatch,
        symbols=[
            ("outer", "function", 1, 10),
            ("dup", "function", 2, 4),
            ("dup", "function", 6, 8),  # same qualname → symbol dropped
        ],
        calls=[("helper", 7)],  # inside the SECOND dup
        imports=[],
    )
    (tmp_path / "mod.ts").write_text(
        "export function outer() {\n"
        "  function dup() {}\n"
        "  function dup() { helper(); }\n"
        "}\n",
        encoding="utf-8",
    )

    g = RepositoryGraph(str(tmp_path))
    g.build()
    # Only one symbol per qualname (legacy dedup contract preserved).
    assert sorted(s.name for s in g.symbols.values()) == ["dup", "outer"]
    # But the call on line 7 attributes to the second dup, not to outer.
    assert [(e.caller, e.callee, e.line) for e in g.call_edges] == [("dup", "helper", 7)]
