"""Tests for _SymbolRangeIndex — bisect-based non-Python caller attribution.

``_process_file_ripgrep`` previously scanned every function range per call
edge (O(calls x functions)).  The index sorts ranges by start line once and
answers "smallest-span function containing this line" in O(log n + nesting
depth).  These tests pin the exact pre-P1 semantics against a reference
implementation of the old linear scan.
"""

import textwrap

from external_llm.graph.repository_graph import RepositoryGraph, _SymbolRangeIndex


def _linear_find(line, sym_ranges):
    """Reference: the pre-P1 linear scan (smallest span wins, first tie)."""
    best = None
    for sym_name, start, end in sym_ranges:
        if start <= line <= end:
            span = end - start
            if best is None or span < best[1]:
                best = (sym_name, span)
    return best[0] if best else ""


RANGES = [
    ("f0", 1, 5),
    ("f1", 7, 12),
    ("f2", 8, 9),  # nested in f1
    ("f3", 20, 40),
    ("f4", 22, 25),  # nested in f3; same start as f5, smaller span
    ("f5", 22, 30),  # nested in f3; same start as f4, larger span
    ("f6", 50, 50),  # single-line function
    ("f7", 55, 60),
    ("f8", 70, 100),
]


def test_parity_with_linear_reference():
    """Index must return exactly what the old linear scan returned."""
    idx = _SymbolRangeIndex(RANGES)
    for line in range(0, 110):
        assert idx.find(line) == _linear_find(line, RANGES), line


def test_input_order_independent():
    """Same ranges in different orders must give identical answers."""
    permutations = [list(reversed(RANGES)), RANGES[4:] + RANGES[:4]]
    for order in permutations:
        idx = _SymbolRangeIndex(order)
        for line in range(0, 110):
            assert idx.find(line) == _linear_find(line, order), (line, order)


def test_nested_functions_innermost_wins():
    idx = _SymbolRangeIndex([("outer", 1, 10), ("inner", 3, 6)])
    assert idx.find(2) == "outer"  # outer body, outside inner
    assert idx.find(4) == "inner"  # inner body
    assert idx.find(6) == "inner"  # inner end line (inclusive)
    assert idx.find(8) == "outer"
    assert idx.find(1) == "outer"  # outer start line (inclusive)
    assert idx.find(10) == "outer"  # outer end line (inclusive)


def test_sibling_functions_do_not_bleed_into_gaps():
    idx = _SymbolRangeIndex([("a", 1, 3), ("b", 5, 7)])
    assert idx.find(1) == "a"
    assert idx.find(4) == ""  # gap between siblings
    assert idx.find(5) == "b"
    assert idx.find(8) == ""


def test_boundary_lines_are_inclusive():
    idx = _SymbolRangeIndex([("f", 3, 7)])
    assert idx.find(3) == "f"
    assert idx.find(7) == "f"
    assert idx.find(2) == ""
    assert idx.find(8) == ""


def test_single_line_function():
    idx = _SymbolRangeIndex([("one", 10, 10)])
    assert idx.find(10) == "one"
    assert idx.find(9) == ""
    assert idx.find(11) == ""


def test_empty_index_returns_empty():
    assert _SymbolRangeIndex([]).find(1) == ""


def test_unsorted_input_is_sorted_internally():
    idx = _SymbolRangeIndex([("b", 5, 7), ("a", 1, 3)])
    assert idx.find(2) == "a"
    assert idx.find(6) == "b"


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end: _process_file_ripgrep caller attribution uses the index
# ══════════════════════════════════════════════════════════════════════════════


def _stub_tree_sitter(monkeypatch, symbols, calls):
    """Stub tree_sitter_utils so the unit test needs no installed grammar."""
    monkeypatch.setattr("external_llm.languages.tree_sitter_utils.is_available", lambda: True)
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
        lambda content, lang, tree=None: [],
    )


def test_ripgrep_path_attributes_calls_to_innermost_function(tmp_path, monkeypatch):
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("outer", "function", 1, 10), ("inner", "function", 3, 6)],
        calls=[("helper", 4), ("helper", 2)],
    )
    (tmp_path / "mod.ts").write_text(
        textwrap.dedent("""\
        export function outer() {
          doWork();
          export function inner() {
            doWork();
          }
        }
        """),
        encoding="utf-8",
    )
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert {s.name for s in g.symbols.values()} == {"outer", "inner"}
    edges = {(e.caller, e.callee, e.line) for e in g.call_edges}
    assert edges == {("inner", "helper", 4), ("outer", "helper", 2)}, edges


def test_ripgrep_path_zero_calls_is_safe(tmp_path, monkeypatch):
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("outer", "function", 1, 10)],
        calls=[],
    )
    (tmp_path / "mod.ts").write_text("export function outer() {}\n", encoding="utf-8")
    g = RepositoryGraph(str(tmp_path))
    g.build()
    assert {s.name for s in g.symbols.values()} == {"outer"}
    assert g.call_edges == []


def test_ripgrep_path_no_function_symbols_is_safe(tmp_path, monkeypatch):
    _stub_tree_sitter(
        monkeypatch,
        symbols=[("Thing", "class", 1, 10)],  # no function-kind ranges
        calls=[("helper", 4)],
    )
    (tmp_path / "mod.ts").write_text("export class Thing {}\n", encoding="utf-8")
    g = RepositoryGraph(str(tmp_path))
    g.build()
    # No enclosing function -> caller stays empty (legacy behavior)
    assert [(e.caller, e.callee) for e in g.call_edges] == [("", "helper")]
