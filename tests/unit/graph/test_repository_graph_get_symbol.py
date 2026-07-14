"""Regression tests for ``RepositoryGraph.get_symbol`` cross-file disambiguation.

Guards against a defect where the qualname branch (``'.' in name``) ignored
``file_path``/``prefer_files`` and returned whichever symbol happened to be
first in dict iteration order.  Two files commonly define the same qualname
(e.g. ``MyClass.helper`` in a/v1.py and b/v2.py, or test stubs mirroring
production classes); the lookup must disambiguate by file exactly like the
bare-name branch, and return ``None`` under strict scoping when no candidate
resides in the requested file.
"""
import shutil
import tempfile
from pathlib import Path

from external_llm.graph.repository_graph import RepositoryGraph


def _build_multifile(files: dict):
    """Build a graph from ``{relpath: source}`` and return (graph, tmpdir)."""
    d = tempfile.mkdtemp(prefix="test_gs_")
    for rel, src in files.items():
        fp = Path(d) / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(src)
    g = RepositoryGraph(d)
    g.build()
    return g, d


SAME_QUAL_TWO_FILES = {
    "a.py": "class MyClass:\n    def helper(self):\n        return 1\n",
    "b.py": "class MyClass:\n    def helper(self):\n        return 2\n",
}

SAME_QUAL_NESTED_PATHS = {
    "pkg/mod_a/svc.py": "class MyClass:\n    def helper(self):\n        return 1\n",
    "pkg/mod_b/svc.py": "class MyClass:\n    def helper(self):\n        return 2\n",
}


def _teardown(d):
    shutil.rmtree(d, ignore_errors=True)


# ── qualname + file_path (exact) ─────────────────────────────────────────


def test_qualname_file_path_disambiguates_exact_a():
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        node = g.get_symbol("MyClass.helper", file_path="a.py")
        assert node is not None
        assert node.file_path == "a.py"
    finally:
        _teardown(d)


def test_qualname_file_path_disambiguates_exact_b():
    """The previously-wrong case: asking for b.py must NOT return a.py."""
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        node = g.get_symbol("MyClass.helper", file_path="b.py")
        assert node is not None
        assert node.file_path == "b.py", f"expected b.py, got {node.file_path}"
    finally:
        _teardown(d)


def test_qualname_file_path_strict_miss_returns_none():
    """file_path pointing at a file with no such qualname -> None (strict)."""
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        assert g.get_symbol("MyClass.helper", file_path="nonexistent.py") is None
    finally:
        _teardown(d)


# ── qualname + file_path (suffix match) ──────────────────────────────────


def test_qualname_file_path_suffix_match_short_input():
    """Short input (basename-ish) must match a longer stored path, mirroring
    the bare-name suffix semantics (e.g. 'mod_b/svc.py' -> 'pkg/mod_b/svc.py')."""
    g, d = _build_multifile(SAME_QUAL_NESTED_PATHS)
    try:
        node = g.get_symbol("MyClass.helper", file_path="mod_b/svc.py")
        assert node is not None
        assert node.file_path == "pkg/mod_b/svc.py", node.file_path
        node = g.get_symbol("MyClass.helper", file_path="mod_a/svc.py")
        assert node.file_path == "pkg/mod_a/svc.py", node.file_path
    finally:
        _teardown(d)


# ── qualname + prefer_files ──────────────────────────────────────────────


def test_qualname_prefer_files_picks_preferred():
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        node = g.get_symbol("MyClass.helper", prefer_files=["b.py"])
        assert node is not None
        assert node.file_path == "b.py", node.file_path
    finally:
        _teardown(d)


def test_qualname_unscoped_returns_without_crash():
    """Unscoped qualname lookup with multiple matches must not crash; it
    returns a deterministic candidate (dict-first). No file_path scoping."""
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        node = g.get_symbol("MyClass.helper")
        assert node is not None
        assert node.file_path in ("a.py", "b.py")
    finally:
        _teardown(d)


def test_qualname_not_found_returns_none():
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        assert g.get_symbol("Nope.nada") is None
    finally:
        _teardown(d)


# ── bare-name regressions (unchanged behavior) ───────────────────────────


def test_bare_name_file_path_still_works():
    g, d = _build_multifile(SAME_QUAL_TWO_FILES)
    try:
        node = g.get_symbol("MyClass", file_path="a.py")
        assert node is not None and node.file_path == "a.py"
        node = g.get_symbol("MyClass", file_path="b.py")
        assert node is not None and node.file_path == "b.py"
    finally:
        _teardown(d)


def test_bare_name_single_match_unscoped():
    g, d = _build_multifile({"a.py": SAME_QUAL_TWO_FILES["a.py"]})
    try:
        node = g.get_symbol("helper")
        assert node is not None and node.file_path == "a.py"
    finally:
        _teardown(d)


def test_bare_name_suffix_match_regression():
    """Bare-name suffix matching (short input -> long stored) must be intact."""
    g, d = _build_multifile(SAME_QUAL_NESTED_PATHS)
    try:
        node = g.get_symbol("MyClass", file_path="mod_a/svc.py")
        assert node.file_path == "pkg/mod_a/svc.py", node.file_path
    finally:
        _teardown(d)
