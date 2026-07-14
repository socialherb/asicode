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
        self, ok: bool = True, content: str = "", error: str | None = None,
        metadata: dict[str, Any] | None = None,
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
            assert f"## {name}" in content, (
                f"{name} missing from scan output:\n{content}"
            )
            assert "Skipped:" in content.split(f"## {name}")[1].split("##")[0], (
                f"{name} not marked Skipped:\n{content}"
            )

        # The metadata must record the skip reason for each.
        skipped = [
            e for e in result["metadata"].get("per_scanner", [])
            if e.get("skipped_language_mismatch")
        ]
        skipped_names = {e["scanner"] for e in skipped}
        assert set(py_only) <= skipped_names, (
            f"expected all Python-only scanners skipped, got {skipped_names}"
        )

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
            assert "Skipped:" not in block, (
                f"{name} wrongly skipped on Go repo:\n{block}"
            )

    def test_single_python_scanner_on_go_repo_reports_skipped(self, tmp_path):
        """Explicitly invoking a Python-only scanner on Go files: the scanner
        is skipped (visible notice) rather than running ast.parse on Go source."""
        repo = str(tmp_path)
        files = ["main.go", "server.go"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan(
            {"scanner": "contradictory_logic_scanner", "path": ""}
        )
        content = result["content"]
        assert "Skipped:" in content
        assert "go" in content  # present language reported

    def test_skip_notice_lists_present_languages(self, tmp_path):
        """The Skipped line must name both supported and present languages so
        the user understands WHY the scanner was skipped."""
        repo = str(tmp_path)
        files = ["a.go", "b.go", "c.ts"]
        tools = _FakeAnalysisTools(repo, files)

        result = tools._tool_run_structural_scan(
            {"scanner": "vulture_dead_code_scanner", "path": ""}
        )
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

        result = tools._tool_run_structural_scan(
            {"scanner": "unused_import_scanner", "path": ""}
        )
        content = result["content"]
        assert "## unused_import_scanner" in content
        assert "Skipped:" not in content
