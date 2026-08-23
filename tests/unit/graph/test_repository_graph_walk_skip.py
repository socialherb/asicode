"""Tests for RepositoryGraph.build()'s walk skip policy (P4, 2026-08-11).

The graph walk skips ``vendor/`` directories and ``*.min.js`` bundles —
parser noise with no structural value (on asicode they were 33% of non-Python
bytes and 32% of non-Python symbols: n/t/e/i obfuscated names from
chart.umd.min.js, highlight.min.js, marked.min.js, purify.min.js).  The scan
walk (external_llm/analysis/scan_walk.py) and the other workers already
skipped ``vendor`` — this aligns the graph with the rest of the repo's
walkers.
"""

from external_llm.graph.repository_graph import RepositoryGraph


def test_build_skips_vendor_dir_and_minified_bundles(tmp_path, monkeypatch):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "dep.js").write_text("export function dep() {}\n", encoding="utf-8")
    (tmp_path / "bundle.min.js").write_text("export function minfn() {}\n", encoding="utf-8")
    (tmp_path / "app.js").write_text("export function app() {}\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build()

    assert {s.name for s in g.symbols.values()} == {"app"}
    assert {rel for rel, _p, _st in g._nonpy_stamps} == {"app.js"}
    assert g.cache_stats["total"] == 1


def test_vendor_dir_skipped_for_python_too(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "vendored.py").write_text("def dead():\n    pass\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("def real():\n    pass\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build()

    assert g.py_files == ["real.py"]
    assert {s.name for s in g.symbols.values()} == {"real"}


def test_reparse_file_honors_walk_skip(tmp_path):
    """reparse_file must not inject symbols from walk-pruned paths (B1).

    build() prunes node_modules/ and *.min.js during its os.walk. The
    incremental reparse path -- fed repo-relative paths by write tools --
    must honor the SAME admission or the graph's contents depend on the
    write history rather than the tree: reparse injects, full build drops.
    """
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    pkg = tmp_path / "node_modules" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "m.py").write_text("def m():\n    return 2\n", encoding="utf-8")
    (tmp_path / "lib.min.js").write_text("function f() {}\n", encoding="utf-8")

    g = RepositoryGraph(str(tmp_path))
    g.build()
    build_symbols = {s.name for s in g.symbols.values()}
    assert build_symbols == {"app"}  # walk pruned the other two

    # Write tool reports the pruned paths as touched -> reparse must skip them.
    g.reparse_file(str(pkg / "m.py"))
    g.reparse_file(str(tmp_path / "lib.min.js"))
    assert {s.name for s in g.symbols.values()} == build_symbols, "incremental path diverged from build"

    # A full rebuild must be a no-op against the already-correct state.
    g.build()
    assert {s.name for s in g.symbols.values()} == build_symbols
