"""Contract tests for language-aware scanner filtering in run_structural_scan.

Guards the Go-repo false-positive regression: when ``scanner="all"`` (or any
Python-only scanner) runs over a non-Python repo, language-mismatched scanners
must be skipped with an explicit ``skipped_language_mismatch`` notice rather
than mis-parsing foreign source through a Python AST.

The filtering has two layers, both exercised here:
  1. ``ScannerRegistry.run()`` drops unsupported-language files before the
     scanner runs (unit-tested in test_scanner_registry.py).
  2. ``AnalysisToolsMixin._tool_run_structural_scan`` short-circuits a scanner
     when NO scanned file matches its supported_languages, emitting a visible
     ``Skipped:`` line (this file).
"""

from __future__ import annotations

import os
from typing import Any

from external_llm.agent.tool_handlers.analysis_tools import AnalysisToolsMixin


class _FakeAnalysisTools(AnalysisToolsMixin):
    """Minimal concrete host for the mixin — only the attributes the scan
    handler reads, with _walk_scan_files overridden to a fixed file list so no
    real filesystem is required."""

    def __init__(self, repo_root: str, files: list[str]):
        self.repo_root = repo_root
        self._call_graph = None
        self._files = files

    def _walk_scan_files(self, root: str) -> list:
        return list(self._files)

    def _make_result(
        self,
        ok: bool = True,
        content: str = "",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        retryable: bool = False,
    ):
        return {"ok": ok, "content": content, "error": error, "metadata": metadata or {}}


class TestStructuralScanLanguageFilter:
    """scanner='all' over a pure-Go repo must skip Python-only scanners."""

    def test_all_on_go_repo_skips_python_only_scanners(self, tmp_path):
        """Go-only file set: the Python-only scanners emit a Skipped line."""
        repo = str(tmp_path)
        files = ["main.go", "internal/backend/backend.go", "internal/ui/chat/agent.go"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})
        content = result["content"]

        # Each Python-only scanner must be reported as skipped.
        py_only = [
            "unused_import_scanner",
            "contradictory_logic_scanner",
            "ast_similarity_scanner",
            "vulture_dead_code_scanner",
            "container_reachability_scanner",
            "broken_contract_scanner",
            # Dead-code scanners are Python-only (cross-reference reachability
            # is unreliable for other languages without native analysis).
            # dead_block_scanner is excluded from "all" mode — it is fully
            # superseded by public_dead_code_scanner.
            "public_dead_code_scanner",
        ]
        for name in py_only:
            assert f"## {name}" in content, f"{name} missing from scan output:\n{content}"
            assert "Skipped:" in content.split(f"## {name}")[1].split("##")[0], f"{name} not marked Skipped:\n{content}"

        # The metadata must record the skip reason for each.
        skipped = [e for e in result["metadata"].get("per_scanner", []) if e.get("skipped_language_mismatch")]
        skipped_names = {e["scanner"] for e in skipped}
        assert set(py_only) <= skipped_names, f"expected all Python-only scanners skipped, got {skipped_names}"

    def test_all_on_go_repo_runs_tree_sitter_scanners(self, tmp_path):
        """duplicate_definition_scanner is NOT skipped on a Go repo — Go is in
        its supported_languages set. (dead_block/public_dead are Python-only now.)"""
        repo = str(tmp_path)
        files = ["main.go", "internal/backend/backend.go"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})
        content = result["content"]

        ts_names = [
            "duplicate_definition_scanner",
        ]
        for name in ts_names:
            assert f"## {name}" in content
            # They must NOT be marked skipped.
            block = content.split(f"## {name}")[1].split("##")[0]
            assert "Skipped:" not in block, f"{name} wrongly skipped on Go repo:\n{block}"

    def test_single_python_scanner_on_go_repo_reports_skipped(self, tmp_path):
        """Explicitly invoking a Python-only scanner on Go files: the scanner
        is skipped (visible notice) rather than running ast.parse on Go source."""
        repo = str(tmp_path)
        files = ["main.go", "server.go"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan({"scanner": "contradictory_logic_scanner", "path": ""})
        content = result["content"]
        assert "Skipped:" in content
        assert "go" in content  # present language reported

    def test_skip_notice_lists_present_languages(self, tmp_path):
        """The Skipped line must name both supported and present languages so
        the user understands WHY the scanner was skipped."""
        repo = str(tmp_path)
        files = ["a.go", "b.go", "c.ts"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan({"scanner": "vulture_dead_code_scanner", "path": ""})
        content = result["content"]
        # present languages include go + typescript; python absent
        assert "go" in content
        assert "typescript" in content
        assert "python" in content  # the scanner's supported language is listed

    def test_python_scanner_on_python_repo_not_skipped(self, tmp_path):
        """Regression guard: a Python-only scanner on a Python file set must
        NOT be skipped — the gate must only fire on genuine mismatches."""
        repo = str(tmp_path)
        files = ["main.py", "lib/utils.py"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan({"scanner": "unused_import_scanner", "path": ""})
        content = result["content"]
        assert "## unused_import_scanner" in content
        assert "Skipped:" not in content


# ── Cooperative cancel pre-check ──────────────────────────────────────────


class TestStructuralScanCancelPreCheck:
    """The scan handler short-circuits to ok=False when config.cancel_event is
    already set on entry — ESC / Ctrl-C before the scan starts returns
    immediately instead of entering the (potentially minutes-long) loop."""

    def test_pre_set_cancel_event_returns_false_immediately(self, tmp_path):
        import threading
        from types import SimpleNamespace

        repo = str(tmp_path)
        # Go-only file set is irrelevant — the pre-check fires before any walk.
        tools = _FakeAnalysisTools(repo, ["a.py"])
        ev = threading.Event()
        ev.set()
        tools.config = SimpleNamespace(cancel_event=ev)

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})

        assert result["ok"] is False
        assert "cancel" in (result["error"] or "").lower()

    def test_unset_cancel_event_does_not_short_circuit(self, tmp_path):
        """A non-set cancel_event must NOT trigger the pre-check — the scan
        proceeds and produces output. Guards against an over-eager gate."""
        import threading
        from types import SimpleNamespace

        repo = str(tmp_path)
        tools = _FakeAnalysisTools(repo, ["main.go"])
        ev = threading.Event()  # NOT set
        tools.config = SimpleNamespace(cancel_event=ev)

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})

        assert result["ok"] is True
        assert result["content"]  # produced output, not the cancel short-circuit

    def test_no_config_does_not_short_circuit(self, tmp_path):
        """No ``config`` attribute at all (e.g. direct API/test use) must be
        None-safe — the pre-check is skipped and the scan runs normally."""
        repo = str(tmp_path)
        tools = _FakeAnalysisTools(repo, ["main.go"])
        # tools.config intentionally absent

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})

        assert result["ok"] is True
        assert result["content"]


# ── Scanner source-freshness banner (R12-2) ──────────────────────────────────


class TestStructuralScanFreshnessBanner:
    """A scanner module changed on disk after load must be surfaced in the tool
    result, not silently scanned with pre-fix in-memory code."""

    def test_stale_scanner_modules_emit_banner(self, tmp_path, monkeypatch):
        from external_llm.agent import scanner_registry as sr_mod

        repo = str(tmp_path)
        tools = _FakeAnalysisTools(repo, ["main.go"])
        stale_file = os.path.join(
            repo,
            "external_llm/analysis/_dead_block_shared.py",
        )
        # Patch the singleton INSTANCE, not the class: the handler resolves
        # ``registry.verify_loaded_sources`` on the instance, and other tests
        # (in this file and test_analysis_tools.py) instance-patch the same
        # name. pytest 9 restores instance patches by setattr-back (not
        # delattr), so the instance permanently carries the attribute — a later
        # CLASS-level patch is then SHADOWED by that instance attribute and the
        # banner silently disappears (flaky under xdist when this test lands on
        # a worker that already ran one of those tests). Instance patching is
        # immune to that ordering.
        reg = sr_mod.get_registry()
        monkeypatch.setattr(reg, "verify_loaded_sources", lambda: [stale_file])

        result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})

        assert result["ok"] is True
        assert "STALE SCANNER CODE DETECTED" in result["content"]


class _RealWalkHost(AnalysisToolsMixin):
    """Host using the REAL _walk_scan_files (filesystem-backed), unlike
    _FakeAnalysisTools which overrides it with a fixed list."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self._call_graph = None


def test_walk_scan_files_is_deterministic_and_sorted(tmp_path):
    """BUG-6: _walk_scan_files must traverse directories in sorted order so the
    file set under _SCAN_FILE_CAP is identical across processes (readdir order
    is nondeterministic)."""
    (tmp_path / "z_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "z_dir" / "z.py").write_text("")
    (tmp_path / "z_dir" / "a.py").write_text("")
    (tmp_path / "a_dir" / "m.py").write_text("")
    (tmp_path / "root_zz.py").write_text("")
    (tmp_path / "root_aa.py").write_text("")

    tools = _RealWalkHost(str(tmp_path))
    first = tools._walk_scan_files(str(tmp_path))
    second = tools._walk_scan_files(str(tmp_path))

    assert first == second, "same tree must produce the same scan file set"
    assert first == [
        "root_aa.py",
        "root_zz.py",
        "a_dir/m.py",
        "z_dir/a.py",
        "z_dir/z.py",
    ]


def test_mixin_walk_delegates_to_shared_single_source(tmp_path):
    """The mixin no longer carries its own _SCAN_EXTS/_SCAN_SKIP_DIRS/
    _SCAN_FILE_CAP mirror — it delegates to
    external_llm/analysis/scan_walk.py, the single scan-walk source shared
    with the structural gate (scripts/check_structural_scanners.py)."""
    from external_llm.analysis.scan_walk import walk_scan_files

    assert not hasattr(AnalysisToolsMixin, "_SCAN_EXTS")
    assert not hasattr(AnalysisToolsMixin, "_SCAN_SKIP_DIRS")
    assert not hasattr(AnalysisToolsMixin, "_SCAN_FILE_CAP")

    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.go").write_text("")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "c.py").write_text("")
    tools = _RealWalkHost(str(tmp_path))
    assert tools._walk_scan_files(str(tmp_path)) == walk_scan_files(str(tmp_path))
    assert tools._walk_scan_files(str(tmp_path)) == ["a.py", "b.go"]


def test_mixin_subdir_scan_yields_repo_relative_paths(tmp_path):
    """A subdir *root* must still yield repo-relative paths (scanners open
    ``repo_root + path``): the shared walk's ``base=repo_root`` preserves the
    pre-unification semantics."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "top.py").write_text("")
    tools = _RealWalkHost(str(tmp_path))
    assert tools._walk_scan_files(os.path.join(str(tmp_path), "pkg")) == ["pkg/a.py"]


class TestStructuralScanAutoReload:
    """P3-1 opt-in: with registry.auto_reload_stale, the handler reloads stale
    scanner modules in place instead of only emitting the restart banner."""

    def _run(self, tmp_path, monkeypatch, stale_file, *, auto_reload, verify, reload_fn):
        # The handler imports get_registry() inside the function, so the
        # singleton it returns is patched directly (monkeypatch restores it).
        from external_llm.agent import scanner_registry as sr_mod

        reg = sr_mod.get_registry()
        monkeypatch.setattr(reg, "auto_reload_stale", auto_reload)
        monkeypatch.setattr(reg, "verify_loaded_sources", verify)
        monkeypatch.setattr(reg, "reload_stale_sources", reload_fn)
        repo = str(tmp_path)
        tools = _FakeAnalysisTools(repo, ["main.go"])
        return tools._tool_run_structural_scan({"scanner": "all", "path": ""})

    def test_auto_reload_replaces_banner_when_reload_succeeds(self, tmp_path, monkeypatch):
        stale_file = os.path.join(
            str(tmp_path),
            "external_llm/analysis/_dead_block_shared.py",
        )
        verifies = {"n": 0}

        def fake_verify():
            verifies["n"] += 1
            return [stale_file] if verifies["n"] == 1 else []

        result = self._run(
            tmp_path,
            monkeypatch,
            stale_file,
            auto_reload=True,
            verify=fake_verify,
            reload_fn=lambda: [stale_file],
        )

        assert verifies["n"] == 2  # stale check + post-reload re-check
        assert result["ok"] is True
        assert "STALE SCANNER CODE DETECTED" not in result["content"]

    def test_auto_reload_keeps_banner_for_unreloadable_remainder(self, tmp_path, monkeypatch):
        stale_file = os.path.join(
            str(tmp_path),
            "external_llm/analysis/_dead_block_shared.py",
        )
        result = self._run(
            tmp_path,
            monkeypatch,
            stale_file,
            auto_reload=True,
            verify=lambda: [stale_file],
            reload_fn=lambda: [],  # reload failed for all
        )

        assert result["ok"] is True
        assert "STALE SCANNER CODE DETECTED" in result["content"]

    def test_warning_only_when_flag_off(self, tmp_path, monkeypatch):
        """Default (flag off) keeps the restart notice — no reload attempted."""
        stale_file = os.path.join(
            str(tmp_path),
            "external_llm/analysis/_dead_block_shared.py",
        )
        reload_calls = []
        result = self._run(
            tmp_path,
            monkeypatch,
            stale_file,
            auto_reload=False,
            verify=lambda: [stale_file],
            reload_fn=lambda: reload_calls.append(1) or [],
        )

        assert reload_calls == []
        assert "STALE SCANNER CODE DETECTED" in result["content"]


class TestStructuralScanCrossRefsUnion:
    """The tool's cross-file-ref input unions the graph's UNCAPPED py list.

    Mirrors the structural gate's contract (scripts/check_structural_scanners
    .py, 2026-08-11): the scan walk truncates at SCAN_FILE_CAP while the
    graph build never does, so a name referenced only from a file beyond the
    cap would otherwise be judged dead.  Capture the candidate list the
    handler hands to ``compute_cross_file_referenced_names_light``: it must
    be the scan list UNION graph.py_files, never the capped scan list alone.
    """

    def test_ref_input_unions_facade_py_files(self, tmp_path, monkeypatch):
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text("def f():\n    return 1\n")

        class _FakeGraph:
            # c.py is beyond the (capped) scan walk — only the graph knows it.
            py_files: list = ["a.py", "b.py", "c.py"]  # noqa: RUF012

        captured: dict = {}

        def _fake_light(graph, repo_root, candidate_files, imported_names=None):
            captured["files"] = candidate_files
            return {"x"}

        monkeypatch.setattr(
            "external_llm.analysis.cross_file_refs.compute_cross_file_referenced_names_light",
            _fake_light,
        )
        tools = _FakeAnalysisTools(str(tmp_path), ["a.py", "b.py"])
        tools._call_graph = _FakeGraph()

        result = tools._tool_run_structural_scan({"scanner": "public_dead_code_scanner", "path": ""})

        assert result["ok"] is True
        assert captured["files"] == ["a.py", "b.py", "c.py"]  # union, not capped

    def test_standalone_without_graph_keeps_plain_scan_list(self, tmp_path, monkeypatch):
        """No graph (or graph without py_files) → the scan list is passed
        unchanged — the standalone conservative mode keeps its contract."""
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text("def f():\n    return 1\n")

        captured: dict = {}

        def _fake_light(graph, repo_root, candidate_files, imported_names=None):
            captured["files"] = candidate_files
            return {"x"}

        monkeypatch.setattr(
            "external_llm.analysis.cross_file_refs.compute_cross_file_referenced_names_light",
            _fake_light,
        )
        tools = _FakeAnalysisTools(str(tmp_path), ["a.py", "b.py"])
        # _FakeAnalysisTools.__init__ leaves _call_graph = None

        result = tools._tool_run_structural_scan({"scanner": "public_dead_code_scanner", "path": ""})

        assert result["ok"] is True
        assert captured["files"] == ["a.py", "b.py"]


class TestStructuralScanScopeCancel:
    """The scan handler's cooperative-cancel pre-check must observe the
    per-call scope (executor-side abandonment: MCP timeout, aborted parallel
    batch) in addition to the agent-loop ESC event — a call abandoned while
    queued returns immediately instead of entering the minutes-long loop."""

    def test_scope_set_pre_check_returns_cancelled(self, tmp_path):
        import threading

        from external_llm.agent.cancel_scope import call_cancel_scope

        tools = _FakeAnalysisTools(str(tmp_path), ["a.py"])
        ev = threading.Event()
        ev.set()
        with call_cancel_scope(ev):
            result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})
        assert result["ok"] is False
        assert result["error"] == "Operation cancelled before structural scan"

    def test_unset_scope_scan_runs(self, tmp_path):
        """An installed-but-unset scope must not change scan behavior."""
        import threading

        from external_llm.agent.cancel_scope import call_cancel_scope

        tools = _FakeAnalysisTools(str(tmp_path), ["main.go"])
        with call_cancel_scope(threading.Event()):
            result = tools._tool_run_structural_scan({"scanner": "all", "path": ""})
        assert result["ok"] is True
        assert "Scanned 1 file(s)." in result["content"]
