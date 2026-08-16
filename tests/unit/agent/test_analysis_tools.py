"""Coverage-completion tests for AnalysisToolsMixin (RED→GREEN: 35% → 100%).

Complements test_structural_scan_language_filter.py (scan-language gating) and
test_query_reachable_depth.py (BFS bound semantics) by exercising the handlers
and branches those files leave uncovered:

  - _tool_get_project_info (every structure section + failure fallback)
  - _tool_analyze_change_impact (edge dedup, direction gating, enrichment,
    path normalization, best-effort failures)
  - _walk_scan_files file-cap warning
  - _tool_run_structural_scan edge branches (scan_path variants, mid-run cancel
    checkpoint, graph-required skip, repo_graph injection, scanner failures,
    candidate rendering variants, test-file candidate split)
  - _tool_query_dependency_graph mode dispatch (all 4 modes + errors)
  - _query_subgraph / _query_transitive_importers / _query_reachable /
    _query_symbol_path (success, failure, and boundary branches)
"""

from __future__ import annotations

import logging
import os
import threading
from types import SimpleNamespace

from external_llm.agent.tool_handlers.analysis_tools import AnalysisToolsMixin

# ── Shared fakes ──────────────────────────────────────────────────────────────


class _Host(AnalysisToolsMixin):
    """Minimal mixin host: dict-returning _make_result, optional fake graph."""

    def __init__(self, repo_root: str = "/tmp", graph: object = None):
        self.repo_root = repo_root
        self._call_graph = graph
        self.config = None

    def _make_result(self, ok=True, content="", error=None, metadata=None, retryable=False):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}, "retryable": retryable}


class _FakeScanTools(AnalysisToolsMixin):
    """Host with _walk_scan_files pinned to a fixed list (no real fs walk)."""

    def __init__(self, repo_root: str, files: list[str], graph: object = None):
        self.repo_root = repo_root
        self._files = files
        self._call_graph = graph
        self.config = None

    def _walk_scan_files(self, root: str) -> list:
        return list(self._files)

    def _make_result(self, ok=True, content="", error=None, metadata=None, retryable=False):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}, "retryable": retryable}


def _edge(**kw):
    base = {
        "caller_symbol": "",
        "caller_file": "",
        "caller_line": 0,
        "callee_symbol": "",
        "callee_file": "",
        "callee_line": 0,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeGraph:
    """Configurable call-graph stub recording every call made against it."""

    def __init__(self, *, callers=None, callees=None, importers=None, symbols=None, deps=None, symbol_file=None):
        self._callers = callers or {}
        self._callees = callees or {}
        self._importers = importers or {}
        self._symbols = symbols or {}
        self._deps = deps or {}
        self._symbol_file = symbol_file or {}
        self.calls: list[tuple] = []

    def get_callers(self, sym, file_path=None):
        self.calls.append(("get_callers", sym, file_path))
        return self._callers.get(sym, [])

    def get_callees(self, sym, file_path=None):
        self.calls.append(("get_callees", sym, file_path))
        return self._callees.get(sym, [])

    def get_importers(self, f):
        self.calls.append(("get_importers", f))
        return self._importers.get(f, [])

    def get_symbols_in_file(self, f):
        return self._symbols.get(f, [])

    def get_file_dependencies(self, f):
        return self._deps.get(f, [])

    def get_symbol_file(self, sym):
        return self._symbol_file.get(sym)


class _SelectiveRaisingGraph(_FakeGraph):
    """_FakeGraph whose chosen methods raise — best-effort except-path coverage."""

    def __init__(
        self,
        *,
        raise_callers=False,
        raise_callees=False,
        raise_importers=False,
        raise_symbols=False,
        raise_deps=False,
        **kw,
    ):
        super().__init__(**kw)
        self._r = {
            "callers": raise_callers,
            "callees": raise_callees,
            "importers": raise_importers,
            "symbols": raise_symbols,
            "deps": raise_deps,
        }

    def get_callers(self, sym, file_path=None):
        if self._r["callers"]:
            raise RuntimeError("callers boom")
        return super().get_callers(sym, file_path)

    def get_callees(self, sym, file_path=None):
        if self._r["callees"]:
            raise RuntimeError("callees boom")
        return super().get_callees(sym, file_path)

    def get_importers(self, f):
        if self._r["importers"]:
            raise RuntimeError("importers boom")
        return super().get_importers(f)

    def get_symbols_in_file(self, f):
        if self._r["symbols"]:
            raise RuntimeError("symbols boom")
        return super().get_symbols_in_file(f)

    def get_file_dependencies(self, f):
        if self._r["deps"]:
            raise RuntimeError("deps boom")
        return super().get_file_dependencies(f)


class _GraphWithoutSymbolFile:
    """Graph lacking get_symbol_file — the hasattr() guard's negative branch."""

    def __init__(self):
        self._callers = {}
        self._callees = {}

    def get_callers(self, sym, file_path=None):
        return []

    def get_callees(self, sym, file_path=None):
        return []


def _canned_result(candidates=(), affected=(), description="desc", total=0):
    return SimpleNamespace(
        candidates_raw=list(candidates),
        affected_files=list(affected),
        scanner_description=description,
        total_candidates=total,
    )


def _boom(*a, **k):
    raise RuntimeError("boom")


def _registry(monkeypatch):
    """Real ScannerRegistry singleton + hermetic cross-refs (no real files)."""
    from external_llm.agent import scanner_registry as sr_mod

    reg = sr_mod.get_registry()
    monkeypatch.setattr(
        "external_llm.analysis.cross_file_refs.compute_cross_file_referenced_names_light",
        lambda *a, **k: set(),
    )
    return reg


# ── _tool_get_project_info ────────────────────────────────────────────────────


def _fake_analyzer(monkeypatch, structure_factory):
    captured: dict = {}

    class _FakeProjectAnalyzer:
        def __init__(self, root):
            captured["root"] = root

        def analyze(self):
            return structure_factory()

    monkeypatch.setattr("external_llm.project_analyzer.ProjectAnalyzer", _FakeProjectAnalyzer)
    return captured


def test_get_project_info_full_structure(monkeypatch):
    """Every populated structure field must appear in content + metadata."""
    captured = _fake_analyzer(
        monkeypatch,
        lambda: SimpleNamespace(
            languages=["Python", "Go"],
            frameworks=["Django"],
            framework="",
            project_types=["app"],
            entry_points=["main.py", "setup.py"],
            test_dir="tests",
            naming_style="snake_case",
            common_imports=["os", "sys"],
            directories={
                "src": ["s1.py", "s2.py", "s3.py", "s4.py"],
                "other": [f"o{i}.py" for i in range(10)],
            },
            primary_language="Python",
        ),
    )
    tools = _Host(repo_root="/tmp")
    res = tools._tool_get_project_info({})

    assert res["ok"] is True
    content = res["content"]
    assert "Languages: Python, Go" in content
    assert "Frameworks: Django" in content
    assert "Project types: app" in content
    assert "Entry points: main.py, setup.py" in content
    assert "Test directory: tests" in content
    assert "Naming style: snake_case" in content
    assert "Common imports: os, sys" in content
    assert "Directories:" in content
    # Non-'other' buckets cap at 3 entries; the 'other' bucket at 8.
    assert "  src: s1.py, s2.py, s3.py" in content
    assert "  other: o0.py, o1.py, o2.py, o3.py, o4.py, o5.py, o6.py, o7.py" in content
    assert "o8.py" not in content
    md = res["metadata"]
    assert md["languages"] == ["Python", "Go"]
    assert md["primary_language"] == "Python"
    assert md["frameworks"] == ["Django"]
    assert captured["root"] == "/tmp"


def test_get_project_info_framework_fallback(monkeypatch):
    """Empty frameworks list falls back to the single framework field."""
    _fake_analyzer(
        monkeypatch,
        lambda: SimpleNamespace(
            languages=["Python"],
            frameworks=[],
            framework="Flask",
            project_types=[],
            entry_points=[],
            test_dir="",
            naming_style="",
            common_imports=[],
            directories={},
            primary_language="Python",
        ),
    )
    res = _Host()._tool_get_project_info({})
    assert res["ok"] is True
    assert "Framework: Flask" in res["content"]
    assert res["metadata"]["frameworks"] == ["Flask"]


def test_get_project_info_empty_structure_fallback(monkeypatch):
    """All-empty structure → explicit 'Unable to determine' message."""
    _fake_analyzer(
        monkeypatch,
        lambda: SimpleNamespace(
            languages=[],
            frameworks=[],
            framework="",
            project_types=[],
            entry_points=[],
            test_dir="",
            naming_style="",
            common_imports=[],
            directories={},
            primary_language="",
        ),
    )
    res = _Host()._tool_get_project_info({})
    assert res["ok"] is True
    assert res["content"] == "Unable to determine project structure"


def test_get_project_info_analyzer_failure(monkeypatch):
    """analyze() raising must yield a graceful 'unavailable' result."""
    _fake_analyzer(monkeypatch, _boom)
    res = _Host()._tool_get_project_info({})
    assert res["ok"] is True
    assert "Project info unavailable: boom" in res["content"]


# ── _tool_analyze_change_impact ───────────────────────────────────────────────


def test_analyze_change_impact_symbol_required():
    res = _Host()._tool_analyze_change_impact({})
    assert res["ok"] is False
    assert res["error"] == "'symbol' is required"


def test_analyze_change_impact_full_report_dedups_edges():
    """Duplicate (caller_symbol, caller_file, caller_line) edges collapse."""
    caller = _edge(caller_symbol="c1", caller_file="x.py", caller_line=1)
    graph = _FakeGraph(
        callers={"foo": [caller, caller, _edge(caller_symbol="c2", caller_file="y.py", caller_line=2)]},
        callees={
            "foo": [
                _edge(callee_symbol="d1", callee_file="z.py", callee_line=5),
                _edge(callee_symbol="d1", callee_file="z.py", callee_line=5),
            ]
        },
        importers={"src/a.py": ["m.py", "n.py"]},
        deps={"src/a.py": [SimpleNamespace(imported="os", import_type="import")]},
        symbol_file={"foo": "src/a.py"},
    )
    tools = _Host(graph=graph)
    res = tools._tool_analyze_change_impact({"symbol": "foo", "file_path": "src/a.py"})

    assert res["ok"] is True
    content = res["content"]
    assert "## Impact analysis for `foo`" in content
    assert "### Callers (2)" in content  # 3 edges, 1 dup → 2 unique
    assert "### Callees (1)" in content
    assert "### Importers (2)" in content
    assert "### File dependencies (1)" in content
    assert "`os` (import)" in content
    assert "**Summary**: 2 callers, 1 callees, ~5 affected files" in content
    md = res["metadata"]
    assert md["caller_count"] == 2
    assert md["callee_count"] == 1
    assert md["caller_files"] == ["x.py", "y.py"]
    assert md["importer_count"] == 2
    assert md["depth"] == 2
    assert md["direction"] == "both"


def test_analyze_change_impact_direction_gates_opposite_traversal():
    """direction=downstream skips caller lookup; upstream skips callee lookup."""
    graph = _FakeGraph(
        callers={"foo": [_edge(caller_symbol="c1", caller_file="x.py", caller_line=1)]},
        callees={"foo": [_edge(callee_symbol="d1", callee_file="z.py", callee_line=5)]},
        symbol_file={"foo": "src/a.py"},
    )
    tools = _Host(graph=graph)

    res = tools._tool_analyze_change_impact({"symbol": "foo", "file_path": "src/a.py", "direction": "downstream"})
    assert "### Callers (none found)" in res["content"]
    assert "### Callees (1)" in res["content"]
    assert ("get_callers", "foo", "src/a.py") not in graph.calls
    graph.calls.clear()

    res = tools._tool_analyze_change_impact({"symbol": "foo", "file_path": "src/a.py", "direction": "upstream"})
    assert "### Callers (1)" in res["content"]
    assert "### Callees (none found)" in res["content"]
    assert ("get_callees", "foo", "src/a.py") not in graph.calls


def test_analyze_change_impact_include_importers_false():
    """include_importers=False suppresses both importers and file deps."""
    graph = _FakeGraph(
        callers={"foo": [_edge(caller_symbol="c1", caller_file="x.py", caller_line=1)]},
        importers={"src/a.py": ["m.py"]},
        deps={"src/a.py": [SimpleNamespace(imported="os", import_type="import")]},
        symbol_file={"foo": "src/a.py"},
    )
    res = _Host(graph=graph)._tool_analyze_change_impact(
        {"symbol": "foo", "file_path": "src/a.py", "include_importers": False}
    )
    assert "### Importers" not in res["content"]
    assert "### File dependencies" not in res["content"]


def test_analyze_change_impact_enrichment_failures_best_effort(caplog):
    """get_importers / get_file_dependencies raising must not fail the tool."""
    graph = _SelectiveRaisingGraph(
        raise_importers=True,
        raise_deps=True,
        symbol_file={"foo": "src/a.py"},
    )
    with caplog.at_level(logging.DEBUG):
        res = _Host(graph=graph)._tool_analyze_change_impact({"symbol": "foo", "file_path": "src/a.py"})
    assert res["ok"] is True
    assert "get_importers failed" in caplog.text
    assert "get_file_dependencies failed" in caplog.text


def test_analyze_change_impact_graph_without_symbol_file_skips_enrichment():
    """No get_symbol_file + no file_path → importers/deps sections omitted."""
    res = _Host(graph=_GraphWithoutSymbolFile())._tool_analyze_change_impact({"symbol": "foo"})
    assert res["ok"] is True
    assert "### Importers" not in res["content"]
    assert "### File dependencies" not in res["content"]
    assert "### Callers (none found)" in res["content"]


def test_analyze_change_impact_file_path_fallback_as_symbol_file():
    """file_path is used as sym_file when the graph has no symbol_file entry."""
    graph = _FakeGraph(importers={"src/a.py": ["m.py"]})
    res = _Host(graph=graph)._tool_analyze_change_impact({"symbol": "foo", "file_path": "src/a.py"})
    assert "### Importers (1)" in res["content"]
    assert ("get_importers", "src/a.py") in graph.calls


def test_analyze_change_impact_absolute_path_normalized():
    """Absolute file_path inside repo_root is normalized to repo-relative."""
    graph = _FakeGraph(importers={"src/a.py": ["m.py"]})
    res = _Host(repo_root="/tmp", graph=graph)._tool_analyze_change_impact(
        {"symbol": "foo", "file_path": "/tmp/src/a.py"}
    )
    assert res["ok"] is True
    assert ("get_importers", "src/a.py") in graph.calls


def test_analyze_change_impact_absolute_path_outside_root_tolerated():
    """relative_to ValueError is suppressed — path passed through unchanged."""
    graph = _FakeGraph(importers={})
    res = _Host(repo_root="/tmp", graph=graph)._tool_analyze_change_impact(
        {"symbol": "foo", "file_path": "/outside/x.py"}
    )
    assert res["ok"] is True
    assert ("get_importers", "/outside/x.py") in graph.calls


def test_analyze_change_impact_graph_failure_returns_error():
    res = _Host(graph=_SelectiveRaisingGraph(raise_callers=True))._tool_analyze_change_impact({"symbol": "foo"})
    assert res["ok"] is False
    assert res["error"].startswith("analyze_change_impact error")


# ── _walk_scan_files ──────────────────────────────────────────────────────────


def test_walk_scan_files_cap_warning(tmp_path, monkeypatch, caplog):
    """Crossing SCAN_FILE_CAP emits a truncation warning (cap patched small).

    Uses the REAL walk (host without an override) — the cap check lives in the
    mixin wrapper, not in the shared scan_walk.
    """
    monkeypatch.setattr("external_llm.agent.tool_handlers.analysis_tools.SCAN_FILE_CAP", 1)
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    tools = _Host(repo_root=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        out = tools._walk_scan_files(str(tmp_path))
    assert out == ["a.py", "b.py"]
    assert "file cap 1 reached" in caplog.text


# ── _tool_run_structural_scan edge branches ───────────────────────────────────


def test_structural_scan_scanner_required():
    res = _FakeScanTools("/tmp", ["a.py"])._tool_run_structural_scan({})
    assert res["ok"] is False
    assert res["error"] == "'scanner' is required"


def test_structural_scan_registry_load_failure(monkeypatch):
    monkeypatch.setattr("external_llm.agent.scanner_registry.get_registry", _boom)
    res = _FakeScanTools("/tmp", ["a.py"])._tool_run_structural_scan({"scanner": "all"})
    assert res["ok"] is False
    assert "Failed to load scanner registry" in res["error"]


def test_structural_scan_freshness_check_unavailable(monkeypatch, caplog):
    """verify_loaded_sources raising → debug note, scan still proceeds."""
    reg = _registry(monkeypatch)
    monkeypatch.setattr(reg, "verify_loaded_sources", _boom)
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    with caplog.at_level(logging.DEBUG):
        res = _FakeScanTools("/tmp", ["a.py"])._tool_run_structural_scan({"scanner": "unused_import_scanner"})
    assert res["ok"] is True
    assert "scanner freshness check unavailable" in caplog.text


def test_structural_scan_auto_reload_reload_failure_keeps_banner(tmp_path, monkeypatch):
    """reload_stale_sources raising → _reloaded=[] → restart banner kept."""
    reg = _registry(monkeypatch)
    stale = os.path.join(str(tmp_path), "external_llm/analysis/_dead_block_shared.py")
    monkeypatch.setattr(reg, "auto_reload_stale", True)
    monkeypatch.setattr(reg, "verify_loaded_sources", lambda: [stale])
    monkeypatch.setattr(reg, "reload_stale_sources", _boom)
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan({"scanner": "all"})
    assert res["ok"] is True
    assert "STALE SCANNER CODE DETECTED" in res["content"]


def test_structural_scan_auto_reload_post_reverify_failure_clears_stale(tmp_path, monkeypatch):
    """Second verify raising → _stale_modules reset → no banner."""
    reg = _registry(monkeypatch)
    stale = os.path.join(str(tmp_path), "external_llm/analysis/_dead_block_shared.py")
    verifies = {"n": 0}

    def _verify():
        verifies["n"] += 1
        if verifies["n"] == 1:
            return [stale]
        raise RuntimeError("boom")

    monkeypatch.setattr(reg, "auto_reload_stale", True)
    monkeypatch.setattr(reg, "verify_loaded_sources", _verify)
    monkeypatch.setattr(reg, "reload_stale_sources", lambda: [stale])
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan({"scanner": "all"})
    assert res["ok"] is True
    assert "STALE SCANNER CODE DETECTED" not in res["content"]


def test_structural_scan_unknown_scanner(tmp_path):
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan({"scanner": "no_such_scanner"})
    assert res["ok"] is False
    assert "Unknown scanner: 'no_such_scanner'" in res["error"]
    assert "Available:" in res["error"]


def test_structural_scan_scan_path_is_file(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x = 1\n")
    reg = _registry(monkeypatch)
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan(
        {"scanner": "unused_import_scanner", "path": "a.py"}
    )
    assert res["ok"] is True
    assert "on a.py" in res["content"]  # header path suffix
    assert "Scanned 1 file(s)." in res["content"]


def test_structural_scan_scan_path_is_dir(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    reg = _registry(monkeypatch)
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    res = _FakeScanTools(str(tmp_path), ["pkg/a.py"])._tool_run_structural_scan(
        {"scanner": "unused_import_scanner", "path": "pkg"}
    )
    assert res["ok"] is True
    assert "on pkg" in res["content"]
    assert "Scanned 1 file(s)." in res["content"]


def test_structural_scan_scan_path_not_found(tmp_path):
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan(
        {"scanner": "unused_import_scanner", "path": "missing.py"}
    )
    assert res["ok"] is False
    assert res["error"] == "Path not found: missing.py"


def test_structural_scan_no_scannable_files(tmp_path):
    res = _FakeScanTools(str(tmp_path), [])._tool_run_structural_scan({"scanner": "all"})
    assert res["ok"] is True
    assert res["content"] == "No scannable source files found."


def test_structural_scan_cross_refs_failure_conservative(monkeypatch, caplog):
    """compute_cross_file_referenced_names_light raising → conservative mode."""
    reg = _registry(monkeypatch)
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result())
    monkeypatch.setattr(
        "external_llm.analysis.cross_file_refs.compute_cross_file_referenced_names_light",
        _boom,
    )
    with caplog.at_level(logging.DEBUG):
        res = _FakeScanTools("/tmp", ["a.py"])._tool_run_structural_scan({"scanner": "unused_import_scanner"})
    assert res["ok"] is True
    assert "cross-file refs unavailable" in caplog.text


def test_structural_scan_cancel_checkpoint_preserves_partial(tmp_path, monkeypatch):
    """ESC between scanners → partial results + '(cancelled after N/M)' line."""
    reg = _registry(monkeypatch)
    ev = threading.Event()
    calls = {"n": 0}

    def _run(name, **kw):
        calls["n"] += 1
        ev.set()  # cancel after the first scanner
        return _canned_result(candidates=[{"file": "a.py", "name": "first", "description": ""}])

    monkeypatch.setattr(reg, "run", _run)
    tools = _FakeScanTools(str(tmp_path), ["a.py"])
    tools.config = SimpleNamespace(cancel_event=ev)
    res = tools._tool_run_structural_scan({"scanner": "all"})

    assert res["ok"] is True
    assert calls["n"] == 1
    assert "(cancelled after 1/" in res["content"]
    assert "first" in res["content"]  # partial results preserved


def test_structural_scan_graph_required_scanner_skipped(tmp_path, monkeypatch):
    """broken_contract (graph_required_for_results) without graph → explicit skip."""
    reg = _registry(monkeypatch)
    run_calls = []
    monkeypatch.setattr(reg, "run", lambda *a, **k: run_calls.append(1) or _canned_result())
    tools = _FakeScanTools(str(tmp_path), ["a.py"], graph=None)
    res = tools._tool_run_structural_scan({"scanner": "broken_contract_scanner"})

    assert res["ok"] is True
    assert "Skipped: scanner requires the call graph" in res["content"]
    assert run_calls == []
    assert res["metadata"]["per_scanner"][0]["skipped_requires_graph"] is True


def test_structural_scan_requires_graph_scanner_receives_repo_graph(tmp_path, monkeypatch):
    """vulture (requires_graph, graceful fallback) → repo_graph kwarg injected."""
    reg = _registry(monkeypatch)
    captured: dict = {}
    graph = object()

    def _run(name, **kw):
        captured.update(kw)
        return _canned_result()

    monkeypatch.setattr(reg, "run", _run)
    tools = _FakeScanTools(str(tmp_path), ["a.py"], graph=graph)
    res = tools._tool_run_structural_scan({"scanner": "vulture_dead_code_scanner"})

    assert res["ok"] is True
    assert captured["repo_graph"] is graph


def test_structural_scan_scanner_run_failure(tmp_path, monkeypatch):
    """registry.run raising → per-scanner ERROR line, overall result still ok."""
    reg = _registry(monkeypatch)
    monkeypatch.setattr(reg, "run", _boom)
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan({"scanner": "unused_import_scanner"})
    assert res["ok"] is True
    assert "ERROR" in res["content"]
    assert "boom" in res["content"]


def test_structural_scan_candidate_rendering_variants(tmp_path, monkeypatch):
    """All candidate field layouts render (occurrences/members/pairs/plain)."""
    reg = _registry(monkeypatch)
    candidates = [
        {"file": "a.py", "occurrences": [[3, 10]], "name": "dup_a", "description": "found it"},
        {"file": "b.py", "occurrences": [], "name": "occ_empty"},
        {"file": "c.py", "lineno": 7, "symbol": "sym_c"},
        {"file": "d.py", "members": [{"name": "m_d"}], "reason": "because"},
        {"file": "e.py", "symbol_a": "s1", "symbol_b": "s2", "detail": "pair"},
        {"file": "f.py", "start_line": 9, "symbol_name": "sn_f", "message": "msg f"},
        {"file": "g.py", "cluster_start": 11, "suggested_action": "fix it"},
        {"file": "h.py", "line": 12, "import_line_text": "import x", "is_test_file": True},
    ]
    monkeypatch.setattr(reg, "run", lambda *a, **k: _canned_result(candidates=candidates, total=8))
    res = _FakeScanTools(str(tmp_path), ["a.py"])._tool_run_structural_scan({"scanner": "unused_import_scanner"})

    content = res["content"]
    assert "  - a.py:3 dup_a — found it" in content
    assert "  - b.py:? occ_empty — " in content
    assert "  - c.py:7 sym_c — " in content
    assert "  - d.py:? m_d — because" in content
    assert "  - e.py:? s1 ↔ s2 — pair" in content
    assert "  - f.py:9 sn_f — msg f" in content
    assert "  - g.py:11  — fix it" in content
    assert "  - h.py:12  — import x" in content
    assert "(7 in production, 1 in test files" in content
    ps = res["metadata"]["per_scanner"][0]
    assert ps["reported"] == 8
    assert ps["test_file_candidates"] == 1
    assert res["metadata"]["total_candidates"] == 8


# ── _tool_query_dependency_graph dispatch ─────────────────────────────────────


def test_query_dependency_graph_unknown_mode():
    res = _Host()._tool_query_dependency_graph({"mode": "bogus"})
    assert res["ok"] is False
    assert "Unknown mode: bogus" in res["error"]


def test_query_dependency_graph_subgraph_requires_source():
    res = _Host()._tool_query_dependency_graph({"mode": "subgraph"})
    assert res["ok"] is False
    assert "'source' (file path) is required for subgraph mode" in res["error"]


def test_query_dependency_graph_subgraph_success():
    graph = _FakeGraph(
        symbols={
            "a.py": [SimpleNamespace(name="f", kind="function", signature="", start_line=1, end_line=2)],
        }
    )
    res = _Host(graph=graph)._tool_query_dependency_graph({"mode": "subgraph", "source": "a.py"})
    assert res["ok"] is True
    assert "## Subgraph for `a.py`" in res["content"]


def test_query_dependency_graph_importers_requires_source():
    res = _Host()._tool_query_dependency_graph({"mode": "importers"})
    assert res["ok"] is False
    assert "'source' (file path) is required for importers mode" in res["error"]


def test_query_dependency_graph_importers_success():
    graph = _FakeGraph(importers={"a.py": ["b.py"]})
    res = _Host(graph=graph)._tool_query_dependency_graph({"mode": "importers", "source": "a.py"})
    assert res["ok"] is True
    assert "## Transitive importers for `a.py`" in res["content"]


def test_query_dependency_graph_reachable_requires_source():
    res = _Host()._tool_query_dependency_graph({"mode": "reachable"})
    assert res["ok"] is False
    assert "'source' (symbol name) is required for reachable mode" in res["error"]


def test_query_dependency_graph_reachable_success():
    graph = _FakeGraph(callees={"A": [_edge(callee_symbol="B", callee_file="f.py")]})
    res = _Host(graph=graph)._tool_query_dependency_graph(
        {"mode": "reachable", "source": "A", "direction": "downstream"}
    )
    assert res["ok"] is True
    assert "## Reachable symbols from `A`" in res["content"]


def test_query_dependency_graph_path_requires_source_and_target():
    res = _Host()._tool_query_dependency_graph({"mode": "path", "source": "A"})
    assert res["ok"] is False
    assert "Both 'source' and 'target'" in res["error"]


def test_query_dependency_graph_path_success():
    graph = _FakeGraph(callees={"A": [_edge(callee_symbol="B", callee_file="f.py")]})
    res = _Host(graph=graph)._tool_query_dependency_graph(
        {"mode": "path", "source": "A", "target": "B", "direction": "downstream"}
    )
    assert res["ok"] is True
    assert "Path found" in res["content"]


# ── _query_subgraph ───────────────────────────────────────────────────────────


def test_query_subgraph_no_symbols():
    res = _Host(graph=_FakeGraph())._query_subgraph("a.py", limit=50)
    assert res["ok"] is True
    assert "No symbols found in this file via graph." in res["content"]


def test_query_subgraph_symbols_edges_imports():
    sym_f = SimpleNamespace(name="f", kind="function", signature="def f(x)", start_line=1, end_line=5)
    sym_g = SimpleNamespace(name="g", kind="function", signature="", start_line=10, end_line=12)
    graph = _FakeGraph(
        symbols={"a.py": [sym_f, sym_g]},
        callees={
            "f": [_edge(callee_symbol="g", callee_file="a.py", callee_line=3)],
            "g": [],
        },
        deps={"a.py": [SimpleNamespace(imported="os", import_type="import")]},
    )
    res = _Host(graph=graph)._query_subgraph("a.py", limit=50)

    assert res["ok"] is True
    content = res["content"]
    assert "**Symbols** (2):" in content
    assert "- function `f` (1-5) — `def f(x)`" in content
    assert "- function `g` (10-12)" in content
    assert "**Internal edges** (1):" in content
    assert "`f` → `g` (line 3)" in content
    assert "**Imports** (1):" in content
    assert "`os` (import)" in content
    md = res["metadata"]
    assert md["symbol_count"] == 2
    assert md["symbols"][0]["name"] == "f"
    assert md["internal_edges"] == ["  `f` → `g` (line 3)"]
    assert md["imports"] == [{"imported": "os", "type": "import"}]


def test_query_subgraph_absolute_path_normalized():
    """Absolute path is normalized before graph lookup (metadata keeps input)."""
    graph = _FakeGraph(
        symbols={
            "a.py": [SimpleNamespace(name="f", kind="function", signature="", start_line=1, end_line=2)],
        }
    )
    res = _Host(repo_root="/tmp", graph=graph)._query_subgraph("/tmp/a.py", limit=50)
    assert res["ok"] is True
    # Symbols keyed by the RELATIVE path — found only if lookup was normalized.
    assert "**Symbols** (1):" in res["content"]


def test_query_subgraph_symbols_failure_yields_none_found():
    res = _Host(graph=_SelectiveRaisingGraph(raise_symbols=True))._query_subgraph("a.py", limit=50)
    assert res["ok"] is True
    assert "No symbols found in this file via graph." in res["content"]


def test_query_subgraph_callees_failure_skips_internal_edges(caplog):
    graph = _SelectiveRaisingGraph(
        raise_callees=True,
        symbols={"a.py": [SimpleNamespace(name="f", kind="function", signature="", start_line=1, end_line=2)]},
    )
    with caplog.at_level(logging.DEBUG):
        res = _Host(graph=graph)._query_subgraph("a.py", limit=50)
    assert res["ok"] is True
    assert "**Internal edges**" not in res["content"]
    assert "subgraph: get_callees failed" in caplog.text


def test_query_subgraph_deps_failure_skips_imports(caplog):
    graph = _SelectiveRaisingGraph(
        raise_deps=True,
        symbols={"a.py": [SimpleNamespace(name="f", kind="function", signature="", start_line=1, end_line=2)]},
    )
    with caplog.at_level(logging.DEBUG):
        res = _Host(graph=graph)._query_subgraph("a.py", limit=50)
    assert res["ok"] is True
    assert "**Imports**" not in res["content"]
    assert "subgraph: get_file_dependencies failed" in caplog.text


# ── _query_transitive_importers ───────────────────────────────────────────────


def test_query_transitive_importers_chain():
    graph = _FakeGraph(importers={"a.py": ["b.py"], "b.py": ["c.py"]})
    res = _Host(graph=graph)._query_transitive_importers("a.py", max_depth=2, limit=50)

    assert res["ok"] is True
    content = res["content"]
    assert "Found 2 transitive importers (depth ≤2):" in content
    assert "└─ b.py" in content
    assert "  └─ c.py" in content
    md = res["metadata"]
    assert md["importers"] == ["b.py", "c.py"]
    assert md["importer_count"] == 2


def test_query_transitive_importers_none():
    res = _Host(graph=_FakeGraph())._query_transitive_importers("a.py", max_depth=5, limit=50)
    assert res["ok"] is True
    assert "No importers found." in res["content"]


def test_query_transitive_importers_failure_best_effort():
    res = _Host(graph=_SelectiveRaisingGraph(raise_importers=True))._query_transitive_importers(
        "a.py", max_depth=5, limit=50
    )
    assert res["ok"] is True
    assert "No importers found." in res["content"]


def test_query_transitive_importers_absolute_path_normalized():
    """Absolute path is normalized before BFS (metadata keeps the input)."""
    graph = _FakeGraph(importers={"a.py": ["b.py"]})
    res = _Host(repo_root="/tmp", graph=graph)._query_transitive_importers("/tmp/a.py", max_depth=5, limit=50)
    assert res["ok"] is True
    # Importers keyed by the RELATIVE path — found only if BFS used normalized.
    assert "Found 1 transitive importers" in res["content"]


# ── _query_reachable ──────────────────────────────────────────────────────────


def test_query_reachable_upstream_uses_callers():
    graph = _FakeGraph(callers={"A": [_edge(caller_symbol="X", caller_file="x.py", caller_line=1)]})
    res = _Host(graph=graph)._query_reachable("A", "upstream", max_depth=5, limit=50)

    assert res["ok"] is True
    assert "(upstream (callers))" in res["content"]
    assert "Found 1 reachable symbols (depth ≤5):" in res["content"]
    assert res["metadata"]["reachable"][0] == {
        "symbol": "X",
        "depth": 1,
        "via": "A",
        "file": "x.py",
    }


def test_query_reachable_edges_failure():
    res = _Host(graph=_SelectiveRaisingGraph(raise_callees=True))._query_reachable(
        "A", "downstream", max_depth=5, limit=50
    )
    assert res["ok"] is True
    assert "No downstream (callees) found." in res["content"]


def test_query_reachable_cycle_skips_visited():
    graph = _FakeGraph(
        callees={
            "A": [_edge(callee_symbol="B", callee_file="f.py")],
            "B": [_edge(callee_symbol="A", callee_file="f.py")],
        }
    )
    res = _Host(graph=graph)._query_reachable("A", "downstream", max_depth=5, limit=50)
    assert res["ok"] is True
    syms = {r["symbol"] for r in res["metadata"]["reachable"]}
    assert syms == {"B"}  # A already visited → skipped


# ── _query_symbol_path ────────────────────────────────────────────────────────


def test_query_symbol_path_downstream_found():
    graph = _FakeGraph(
        callees={
            "A": [_edge(callee_symbol="B", callee_file="f.py")],
            "B": [_edge(callee_symbol="C", callee_file="f.py")],
        }
    )
    res = _Host(graph=graph)._query_symbol_path("A", "C", "downstream", max_depth=3, limit=50)

    assert res["ok"] is True
    md = res["metadata"]
    assert md["path_found"] is True
    assert md["path"] == ["A", "B", "C"]
    assert md["path_length"] == 2
    assert md["direction"] == "downstream"
    assert "Path found (depth=2 via callees):" in res["content"]
    assert "→ `B`" in res["content"]


def test_query_symbol_path_upstream_found():
    graph = _FakeGraph(
        callers={
            "A": [_edge(caller_symbol="B", caller_file="f.py")],
            "B": [_edge(caller_symbol="C", caller_file="f.py")],
        }
    )
    res = _Host(graph=graph)._query_symbol_path("A", "C", "upstream", max_depth=3, limit=50)

    assert res["ok"] is True
    assert res["metadata"]["direction"] == "upstream"
    assert res["metadata"]["path"] == ["A", "B", "C"]
    assert "Path found (depth=2 via callers):" in res["content"]
    assert "← `B`" in res["content"]


def test_query_symbol_path_both_downstream_wins():
    graph = _FakeGraph(callees={"A": [_edge(callee_symbol="B", callee_file="f.py")]})
    res = _Host(graph=graph)._query_symbol_path("A", "B", "both", max_depth=3, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["direction"] == "downstream"


def test_query_symbol_path_both_upstream_fallback():
    """Downstream misses (no callees) → upstream leg still finds the path."""
    graph = _FakeGraph(callers={"A": [_edge(caller_symbol="B", caller_file="f.py")]})
    res = _Host(graph=graph)._query_symbol_path("A", "B", "both", max_depth=3, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["direction"] == "upstream"
    assert res["metadata"]["path"] == ["A", "B"]


def test_query_symbol_path_not_found():
    res = _Host(graph=_FakeGraph())._query_symbol_path("A", "B", "both", max_depth=5, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["path_found"] is False
    assert "No path found (depth ≤5)" in res["content"]


def test_query_symbol_path_edges_failure():
    res = _Host(graph=_SelectiveRaisingGraph(raise_callees=True, raise_callers=True))._query_symbol_path(
        "A", "B", "both", max_depth=5, limit=50
    )
    assert res["ok"] is True
    assert res["metadata"]["path_found"] is False


def test_query_symbol_path_negative_max_depth_guard():
    """The len(path) > max_depth+1 guard fires for degenerate max_depth=-1."""
    graph = _FakeGraph(callees={"A": [_edge(callee_symbol="B", callee_file="f.py")]})
    res = _Host(graph=graph)._query_symbol_path("A", "B", "both", max_depth=-1, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["path_found"] is False


def test_query_symbol_path_max_depth_one_stops_expansion():
    """max_depth=1 must not enqueue B's children (len(path) < max_depth guard)."""
    graph = _FakeGraph(
        callees={
            "A": [_edge(callee_symbol="B", callee_file="f.py")],
            "B": [_edge(callee_symbol="C", callee_file="f.py")],
        }
    )
    res = _Host(graph=graph)._query_symbol_path("A", "C", "downstream", max_depth=1, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["path_found"] is False


def test_query_symbol_path_upstream_max_depth_one_stops_expansion():
    graph = _FakeGraph(
        callers={
            "A": [_edge(caller_symbol="B", caller_file="f.py")],
            "B": [_edge(caller_symbol="C", caller_file="f.py")],
        }
    )
    res = _Host(graph=graph)._query_symbol_path("A", "C", "upstream", max_depth=1, limit=50)
    assert res["ok"] is True
    assert res["metadata"]["path_found"] is False
