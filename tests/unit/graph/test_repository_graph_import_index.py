"""Tests for the lazily-built reverse import-edge index in RepositoryGraph.

``get_importers`` previously scanned ``import_edges`` linearly (O(N x E) over
all callers); the index makes a query one dict lookup per match branch while
preserving the legacy semantics EXACTLY: edge order, first-wins importer
dedup, dotted-prefix match for Python modules, the same-directory constraint
on relative-form matches, and the path-resolved match for non-Python files.
These tests pin the equivalence against a reference implementation of the
legacy scan and verify index invalidation on mutation (via the public
reparse_file / remove_file paths, not the private invalidators).
"""
import os
import random
import shutil
import tempfile
from pathlib import Path

from external_llm.graph.models import ImportEdge
from external_llm.graph.repository_graph import RepositoryGraph, path_to_module
from external_llm.languages import LanguageId


def _write(d, rel, source):
    p = Path(d) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source, encoding="utf-8")


def _build_repo(tmp):
    """A repo exercising every legacy match branch.

    - Python dotted-prefix matches: ``import utils`` (edge "utils"),
      ``import pkg.a`` (edge "pkg.a"), ``from pkg import REEXPORTED``
      (edge "pkg.REEXPORTED", package-init query), ``from .operation_models
      import ModelX`` (edge "operation_models.ModelX", relative form).
    - Same-basename decoy in another directory (must NOT match).
    - Non-Python path-resolved matches (synthetic edges; deterministic
      regardless of tree-sitter availability).
    """
    _write(tmp, "utils.py", "X = 1\n")
    _write(tmp, "mod.py", "import utils\nfrom pkg import REEXPORTED\n")
    _write(tmp, "pkg/__init__.py", "REEXPORTED = 42\n")
    _write(tmp, "pkg/operation_models.py", "ModelX = 1\n")
    _write(tmp, "pkg/a.py", "from .operation_models import ModelX\n")
    _write(tmp, "pkg/sub/b.py", "import pkg.a\n")
    _write(tmp, "other/operation_models.py", "OTHER = 1\n")
    _write(tmp, "other/c.py", "from .operation_models import OTHER\n")
    _write(tmp, "main.ts", "import { f } from './utils';\n")
    _write(tmp, "__tests__/t.ts", "import { f } from '../utils';\n")
    _write(tmp, "web/use.ts", "import { g } from './helper';\n")
    _write(tmp, "web/helper.ts", "export const g = 1;\n")


def _synthetic_edges():
    """Weird-but-legal edge shapes the extractors never emit."""
    return [
        ImportEdge("pkg/a.py", "pkg.operation_models.ModelX", "from"),  # dotted from-import
        ImportEdge("pkg/a.py", "operation_models", "from"),          # bare relative form
        ImportEdge("other/c.py", "operation_models.ModelX", "from"), # same basename, wrong dir
        ImportEdge("main.py", "utils.helper", "from"),               # ancestor prefix match
        ImportEdge("x.py", "", "import"),                            # empty imported: skipped
        ImportEdge("", "utils", "import"),                           # empty importer: dedupable
        ImportEdge("web/use.ts", "./helper", "js_require"),          # non-py, same-dir dot import
        ImportEdge("web/use.ts", "../utils", "js_require"),          # non-py, parent-dir dot import
        ImportEdge("deep/a/b/c/d.py", "very.deep.module.path", "import"),  # depth 4 dotted
        ImportEdge("main.py", "utils", "import"),                    # dup importer in bucket
        ImportEdge("main.py", "utils.x", "from"),                    # second edge, same importer
    ]


def _legacy_get_importers(graph, file_path):
    """Reference implementation of the pre-index linear scan (verbatim logic)."""
    if not file_path:
        return []
    if LanguageId.from_path(file_path) is not LanguageId.PYTHON:
        _cand_noext = os.path.splitext(file_path)[0]
        _importers = []
        _seen = set()
        for edge in graph.import_edges:
            _imp = edge.imported or ""
            if not _imp or edge.importer in _seen:
                continue
            if _imp.startswith("."):
                _resolved = os.path.normpath(os.path.join(os.path.dirname(edge.importer or ""), _imp))
            else:
                _resolved = os.path.normpath(_imp)
            if _resolved in (_cand_noext, file_path):
                _seen.add(edge.importer)
                _importers.append(edge.importer)
        return _importers
    _module_prefix = path_to_module(file_path, graph.repo_root)
    _module_basename = _module_prefix.rsplit(".", 1)[-1]
    _basename_differs = _module_basename != _module_prefix
    _importers = []
    _seen = set()
    for edge in graph.import_edges:
        _imp = edge.imported or ""
        if _imp == _module_prefix or _imp.startswith(_module_prefix + "."):
            if edge.importer not in _seen:
                _seen.add(edge.importer)
                _importers.append(edge.importer)
        elif _basename_differs and (_imp == _module_basename or _imp.startswith(_module_basename + ".")):
            _cand_dir = os.path.dirname(file_path)
            _imp_dir = os.path.dirname(edge.importer or "")
            if _cand_dir == _imp_dir and edge.importer not in _seen:
                _seen.add(edge.importer)
                _importers.append(edge.importer)
    return _importers


_QUERIES = [
    # every real file in the fixture
    "utils.py", "mod.py", "pkg/__init__.py", "pkg/operation_models.py",
    "pkg/a.py", "pkg/sub/b.py", "other/operation_models.py", "other/c.py",
    "main.ts", "__tests__/t.ts", "web/use.ts", "web/helper.ts",
    # module/dir forms and extensionless candidates
    "utils", "pkg", "pkg/", "pkg/operation_models", "pkg/sub/b", "other/operation_models",
    "web/helper", "web/use", "main", "operation_models.py", "operation_models",
    # edge cases
    "", "nonexistent.py", "nonexistent", "deep/a/b/c/d.py", "x.y.z", "a/b/c/d/e/f.py",
]


def _assert_parity(graph, queries):
    for q in queries:
        got = graph.get_importers(q)
        want = _legacy_get_importers(graph, q)
        # Exact list equality — order is part of the legacy contract.
        assert got == want, f"query {q!r}: index={got!r} legacy={want!r}"


def test_index_matches_legacy_scan():
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _build_repo(d)
        g = RepositoryGraph(d)
        g.build()
        _assert_parity(g, _QUERIES)
        assert g.get_importers("pkg/operation_models.py") == ["pkg/a.py"], g.get_importers("pkg/operation_models.py")
        assert g.get_importers("utils.py") == ["mod.py"], g.get_importers("utils.py")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_index_matches_legacy_with_synthetic_edges():
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _build_repo(d)
        g = RepositoryGraph(d)
        g.build()
        g.import_edges.extend(_synthetic_edges())
        _assert_parity(g, _QUERIES)
        # The same-directory constraint must exclude the decoy importer.
        importers = g.get_importers("pkg/operation_models.py")
        assert "other/c.py" not in importers, importers
        # ... but the decoy DOES import its own same-named module.
        assert "other/c.py" in g.get_importers("other/operation_models.py"), g.get_importers("other/operation_models.py")
        # Non-Python path-resolved matching fires only for NON-Python queries
        # (the Python branch never sees "../utils" — legacy semantics).
        importers = g.get_importers("utils.ts")
        assert "main.ts" in importers and "web/use.ts" in importers, importers
        assert "__tests__/t.ts" in g.get_importers("utils.ts"), g.get_importers("utils.ts")
        assert "web/use.ts" in g.get_importers("web/helper.ts"), g.get_importers("web/helper.ts")
        assert "web/use.ts" in g.get_importers("web/helper"), g.get_importers("web/helper")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_random_edges_parity_with_legacy():
    """Seeded fuzz: random importers/values must never diverge from the scan."""
    rng = random.Random(20260811)
    files = [
        "a.py", "pkg/b.py", "pkg/sub/c.py", "x/__init__.py", "y/z.py",
        "main.ts", "__tests__/t.ts", "web/helper.ts", "",
    ]
    values = [
        "utils", "utils.x", "pkg.b", "pkg.b.y", "pkg.sub.c", "x.y",
        "a.b.c.d.e", ".", "..", "../utils", "./helper", "helper",
        "operation_models", "operation_models.X", "", "web/helper",
        "web/helper.ts", "x.y.z.w.v.u",
    ]
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        g = RepositoryGraph(d)
        g.build()  # empty repo — all edges below are synthetic
        for _ in range(400):
            g.import_edges.append(ImportEdge(rng.choice(files), rng.choice(values), "import"))
        _assert_parity(g, _QUERIES + files)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reparse_file_invalidates_import_index():
    """External edits via reparse_file must be reflected by the next query."""
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _write(d, "a.py", "x = 1\n")
        _write(d, "main.py", "import a\n")
        g = RepositoryGraph(d)
        g.build()
        assert g.get_importers("a.py") == ["main.py"]  # index built here

        # External edit: importer drops the import.
        _write(d, "main.py", "y = 2\n")
        g.reparse_file(str(Path(d) / "main.py"))
        assert g.get_importers("a.py") == []

        # External edit: a new file starts importing.
        _write(d, "b.py", "import a\n")
        g.reparse_file(str(Path(d) / "b.py"))
        assert g.get_importers("a.py") == ["b.py"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_remove_file_invalidates_import_index():
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _write(d, "a.py", "x = 1\n")
        _write(d, "main.py", "import a\n")
        g = RepositoryGraph(d)
        g.build()
        assert g.get_importers("a.py") == ["main.py"]
        g.remove_file("main.py")
        assert g.get_importers("a.py") == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_returned_lists_are_copies():
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _write(d, "a.py", "x = 1\n")
        _write(d, "main.py", "import a\n")
        g = RepositoryGraph(d)
        g.build()
        first = g.get_importers("a.py")
        first.clear()
        assert g.get_importers("a.py") == ["main.py"], "mutating a returned list must not corrupt the index"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_index_is_rebuilt_after_second_build():
    """A second build() RESETS edges (P1, 2026-08-11): re-extraction must not
    double-append, and the rebuilt index must track exactly the fresh edges."""
    d = tempfile.mkdtemp(prefix="test_rii_")
    try:
        _write(d, "a.py", "x = 1\n")
        _write(d, "main.py", "import a\n")
        g = RepositoryGraph(d)
        g.build()
        assert g.get_importers("a.py") == ["main.py"]
        _write(d, "b.py", "import a\n")
        g.build()  # reset + re-extract: exactly one edge per importer, in
        # walk order (a.py, b.py, main.py) — identical to a fresh build
        assert g.get_importers("a.py") == ["b.py", "main.py"]
        assert len(g.import_edges) == 2, g.import_edges
        # Idempotent: an unchanged third build must not accumulate edges.
        g.build()
        assert len(g.import_edges) == 2, g.import_edges
        assert g.get_importers("a.py") == ["b.py", "main.py"]
    finally:
        shutil.rmtree(d, ignore_errors=True)
