"""Unit tests for RepositoryGraph.build()'s first-build DISK cache tier.

The structural gate (scripts/check_structural_scanners.py) persists per-file
extraction results in ``.cache/structural_graph_v1.json``.  A fresh process
(empty ``_extract_cache``) reuses that JSON so the first build does not
re-parse the whole repo.  These tests pin the two contracts:

* a disk-warm first build is bit-for-bit identical to a cold build, and
* reuse happens exactly when (and only when) the manifest stamp matches the
  current stat — stale/missing/corrupt entries fail open to extract_file.
"""
import json
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest

from external_llm.graph.repository_graph import RepositoryGraph, _extract_cache
from external_llm.graph.structural_cache import (
    data_to_json,
    default_cache_path,
    save,
)


@pytest.fixture(autouse=True)
def isolated_extract_cache():
    """Save/restore the process-wide cache around every test."""
    saved = dict(_extract_cache)
    _extract_cache.clear()
    yield
    _extract_cache.clear()
    _extract_cache.update(saved)


def _make_repo(files: dict) -> str:
    d = tempfile.mkdtemp(prefix="test_rg_disk_")
    for rel_path, source in files.items():
        full = Path(d) / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(textwrap.dedent(source))
    return d


def _write_gate_cache(repo: str) -> None:
    """Produce a gate-format cache for *repo* exactly as _build_graph does."""
    manifest: dict[str, list[int]] = {}
    files: dict[str, dict] = {}
    graph = RepositoryGraph(repo)
    for dirpath, _dirs, filenames in os.walk(repo):
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            full = Path(dirpath) / fn
            rel = os.path.relpath(full, repo)
            st = full.stat()
            manifest[rel] = [st.st_mtime_ns, st.st_size]
            files[rel] = data_to_json(graph.extract_file(str(full)))
    save(default_cache_path(repo), manifest, files, {})


def _graph_snapshot(graph):
    """Field-level snapshot — SymbolNode is a plain class (no __eq__)."""
    symbols = {}
    for uid, s in sorted(graph.symbols.items()):
        symbols[uid] = (
            s.name, s.qualname, s.module, s.file_path, s.kind,
            s.start_line, s.end_line, s.language, s.signature_hash,
            s.docstring, s.signature, tuple(s.bases or ()),
        )
    edges = [(e.caller, e.callee, e.file_path, e.line) for e in graph.call_edges]
    imports = [(e.importer, e.imported, e.import_type) for e in graph.import_edges]
    return symbols, edges, imports


def _count_extract_calls(monkeypatch):
    """Return a counter object: counts RepositoryGraph.extract_file calls."""
    calls = {"n": 0}
    orig = RepositoryGraph.extract_file

    def counting(self, path):
        calls["n"] += 1
        return orig(self, path)

    monkeypatch.setattr(RepositoryGraph, "extract_file", counting)
    return calls


_SRC = {
    "a.py": "def fa():\n    return 1\n",
    "b.py": "import a\n\n\ndef fb():\n    return a.fa()\n",
}


# ── Warm first build ─────────────────────────────────────────────────────────

def test_first_build_serves_unchanged_files_from_gate_cache(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()  # fresh process state: empty _extract_cache
        assert calls["n"] == 0  # every file served from the disk tier
        assert "a.py:fa" in g.symbols
        assert "b.py:fb" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_disk_warm_first_build_bit_for_bit_identical_to_cold():
    repo = _make_repo(_SRC)
    try:
        cold = RepositoryGraph(repo)
        cold.build()
        _write_gate_cache(repo)
        warm = RepositoryGraph(repo)  # NEW instance — in-process cache empty
        warm.build()
        assert _graph_snapshot(warm) == _graph_snapshot(cold)
        # Edge ORDER identical too — injection follows the same sorted walk.
        assert [e.callee for e in warm.call_edges] == [e.callee for e in cold.call_edges]
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_disk_warm_build_populates_process_cache(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        RepositoryGraph(repo).build()  # disk-warm; fills _extract_cache
        calls = _count_extract_calls(monkeypatch)
        g2 = RepositoryGraph(repo)
        g2.build()  # second build — in-process cache, no disk tier needed
        assert calls["n"] == 0
        assert "a.py:fa" in g2.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_nested_rel_keys_served_from_disk(monkeypatch):
    repo = _make_repo({
        "pkg/__init__.py": "",
        "pkg/mod.py": "def helper(x):\n    return x\n",
    })
    try:
        _write_gate_cache(repo)
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 0
        assert "pkg/mod.py:helper" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── Staleness: reuse exactly on (and only on) stamp match ────────────────────

def test_stale_stamp_reextracts(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        a = Path(repo) / "a.py"
        st = a.stat()
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 1  # only the touched file re-parsed
        assert "a.py:fa" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_missing_files_entry_reextracts(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        path = default_cache_path(repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["files"]["b.py"]
        path.write_text(json.dumps(data))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 1  # b.py has no payload → parsed; a.py from disk
        assert "b.py:fb" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_corrupt_payload_falls_back_to_parse(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        path = default_cache_path(repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["files"]["a.py"] = {"symbols": "junk", "calls": [], "imports": []}
        path.write_text(json.dumps(data))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 1  # corrupt entry re-parsed, NOT skipped
        assert "a.py:fa" in g.symbols  # file present in the graph
        assert "b.py:fb" in g.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── Fail-open: any cache problem degrades to a full build ────────────────────

def test_corrupt_json_falls_back_to_full_parse(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        default_cache_path(repo).write_text("{not json!!")
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 2
        assert len(g.symbols) == 2
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_version_mismatch_falls_back_to_full_parse(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        path = default_cache_path(repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = 999
        path.write_text(json.dumps(data))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 2
        assert len(g.symbols) == 2
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_missing_imported_names_section_falls_back(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        path = default_cache_path(repo)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["imported_names"]
        path.write_text(json.dumps(data))
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 2  # mandatory section missing → corrupt
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_no_cache_file_falls_back_to_full_parse(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        calls = _count_extract_calls(monkeypatch)
        g = RepositoryGraph(repo)
        g.build()
        assert calls["n"] == 2
        assert len(g.symbols) == 2
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── Reload on gate rewrite ───────────────────────────────────────────────────

def test_reload_after_gate_rewrite_serves_fresh_payload(monkeypatch):
    repo = _make_repo(_SRC)
    try:
        _write_gate_cache(repo)
        g = RepositoryGraph(repo)
        g.build()
        # The agent edits a.py; a gate run (mid-session) rewrites the cache
        # with the new payload+stamp.  The next build must RELOAD the JSON
        # (mtime changed) and serve the fresh extraction — not the stale
        # in-process entry.
        (Path(repo) / "a.py").write_text("def fa():\n    return 1\n\ndef extra():\n    return 2\n")
        _write_gate_cache(repo)
        calls = _count_extract_calls(monkeypatch)
        g2 = RepositoryGraph(repo)
        g2.build()
        assert calls["n"] == 0
        assert "a.py:extra" in g2.symbols  # fresh payload from the reloaded cache
        assert "b.py:fb" in g2.symbols
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ── Nested roots / cwd independence ─────────────────────────────────────────

def test_nested_roots_do_not_share_extract_cache_payloads():
    """Regression: _extract_cache was keyed by abs path alone, but payloads
    carry root-relative fields (SymbolNode.file_path / .module, CallEdge
    .file_path). A graph on repo/pkg served repo's cached extraction of
    pkg/m.py — file_path "pkg/m.py" instead of "m.py", module "pkg.m"
    instead of "m" — and handed back the very SAME SymbolNode objects for
    graphs that must disagree on them."""
    repo = _make_repo({
        "pkg/__init__.py": "",
        "pkg/m.py": "def helper(x):\n    return x\n",
    })
    try:
        g1 = RepositoryGraph(repo)
        g1.build()
        sym1 = g1.symbols["pkg/m.py:helper"]
        assert sym1.file_path == "pkg/m.py"
        assert sym1.module == "pkg.m"

        # Nested root — same process, so the shared module cache is warm.
        g2 = RepositoryGraph(os.path.join(repo, "pkg"))
        g2.build()
        sym2 = g2.symbols["m.py:helper"]
        assert sym2.file_path == "m.py", (
            f"nested root must get pkg-relative file_path, got {sym2.file_path!r}"
        )
        assert sym2.module == "m", (
            f"nested root must get pkg-relative module, got {sym2.module!r}"
        )
        # Distinct objects — no cross-root aliasing through the shared cache.
        assert sym2 is not sym1
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_module_names_independent_of_cwd(monkeypatch, tmp_path):
    """Regression: _path_to_module ran os.path.relpath on an already-relative
    path — resolved against the CWD, so a build from cwd != repo_root produced
    module names like "....var.folders...pkg.m" (and, mixed with disk-cache
    payloads written from the repo root, a per-file module convention that
    disagreed within one graph)."""
    repo = _make_repo({
        "pkg/__init__.py": "",
        "pkg/m.py": "def helper(x):\n    return x\n",
    })
    try:
        monkeypatch.chdir(tmp_path)  # cwd is NOT the repo root
        g = RepositoryGraph(repo)
        g.build()
        sym = g.symbols["pkg/m.py:helper"]
        assert sym.module == "pkg.m", f"got {sym.module!r}"
        assert sym.file_path == "pkg/m.py"
    finally:
        shutil.rmtree(repo, ignore_errors=True)
