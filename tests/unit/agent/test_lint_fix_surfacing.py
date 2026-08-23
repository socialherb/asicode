"""Lint fix surfacing: LintIssue.fix rendered inline and in metadata.

Covers:
- LintResult.fixable_count property
- run_ruff parsing ruff's fix info into LintIssue.fix and the summary
  "(N auto-fixable)" suffix
- _tool_run_lint rendering the fix hint inline and exposing fixable
  issues in metadata
"""

from __future__ import annotations

import json
import types
from unittest import mock

from external_llm.agent.lint_runner import LintIssue, LintResult, LintRunner
from external_llm.agent.tool_handlers.test_tools import TestToolsMixin
from external_llm.agent.tool_registry import ToolResult

# ── LintResult.fixable_count ──────────────────────────────────────────────


def test_fixable_count_counts_only_issues_with_fix():
    result = LintResult(
        ok=False,
        issues=[
            LintIssue(file="a.py", line=1, col=1, code="F401", message="unused", fix="Remove unused import"),
            LintIssue(file="a.py", line=2, col=1, code="E501", message="long"),
            LintIssue(file="a.py", line=3, col=1, code="F841", message="assigned", fix="Remove assignment"),
        ],
    )
    assert result.fixable_count == 2


def test_fixable_count_zero_when_no_issues():
    assert LintResult(ok=True, issues=[]).fixable_count == 0


# ── run_ruff parses fix info ──────────────────────────────────────────────

_RAW = [
    {
        "filename": "a.py",
        "location": {"row": 1, "column": 1},
        "code": "F401",
        "message": "`os` imported but unused",
        "severity": "warning",
        "fix": {"message": "Remove unused import: `os`"},
    },
    {
        "filename": "a.py",
        "location": {"row": 5, "column": 8},
        "code": "E501",
        "message": "Line too long (100 > 88)",
        "severity": "error",
    },
]


def _mock_ruff_run(stdout: str, returncode: int = 1):
    return mock.patch(
        "external_llm.agent.lint_runner.subprocess.run",
        return_value=types.SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr="",
        ),
    )


def test_run_ruff_parses_fix_message_into_issue(tmp_path):
    (tmp_path / "a.py").write_text("import os\n")
    with _mock_ruff_run(json.dumps(_RAW)):
        result = LintRunner(str(tmp_path)).run_ruff("")
    assert result.ok is False
    assert len(result.issues) == 2
    assert result.issues[0].fix == "Remove unused import: `os`"
    assert result.issues[1].fix is None
    assert result.fixable_count == 1
    assert result.summary == "2 lint issue(s) found (1 auto-fixable)"


def test_run_ruff_summary_omits_suffix_when_none_fixable(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    with _mock_ruff_run(json.dumps([_RAW[1]])):
        result = LintRunner(str(tmp_path)).run_ruff("")
    assert result.summary == "1 lint issue(s) found"


# ── _tool_run_lint rendering / metadata ───────────────────────────────────


class _FakeConfig:
    max_lint_issues = 50


class _FakeLintRunner:
    def __init__(self, result: LintResult):
        self._result = result

    def run_lint(self, path: str, max_issues: int = 50) -> LintResult:
        return self._result


class _Handler(TestToolsMixin):
    """Minimal registry-shaped object exposing _tool_run_lint."""

    def __init__(self, result: LintResult):
        self.config = _FakeConfig()
        self._lint_runner = _FakeLintRunner(result)
        self.repo_root = "/tmp"

    def _make_result(self, **kwargs) -> ToolResult:
        return ToolResult(**kwargs)


def test_tool_run_lint_renders_fix_inline():
    result = LintResult(
        ok=False,
        summary="2 lint issue(s) found (1 auto-fixable)",
        issues=[
            LintIssue(
                file="a.py",
                line=1,
                col=1,
                code="F401",
                message="`os` imported but unused",
                fix="Remove unused import: `os`",
            ),
            LintIssue(file="a.py", line=5, col=8, code="E501", message="Line too long (100 > 88)"),
        ],
    )
    tr = _Handler(result)._tool_run_lint({"path": "a.py"})
    assert tr.ok is True  # issue detection is not a tool error
    lines = tr.content.splitlines()
    assert lines[0] == "2 lint issue(s) found (1 auto-fixable)"
    assert lines[1].endswith("[fix: Remove unused import: `os`]")
    assert "[fix:" not in lines[2]


def test_tool_run_lint_metadata_fixable():
    result = LintResult(
        ok=False,
        issues=[
            LintIssue(file="a.py", line=1, col=1, code="F401", message="unused", fix="Remove unused import"),
            LintIssue(file="a.py", line=2, col=1, code="E501", message="long"),
        ],
    )
    tr = _Handler(result)._tool_run_lint({"path": "a.py"})
    assert tr.metadata["issue_count"] == 2
    assert tr.metadata["fixable_count"] == 1
    assert tr.metadata["fixable_issues"] == [
        {"file": "a.py", "line": 1, "col": 1, "code": "F401", "fix": "Remove unused import"},
    ]


def test_tool_run_lint_metadata_omits_fixable_keys_when_none():
    result = LintResult(
        ok=False,
        issues=[LintIssue(file="a.py", line=2, col=1, code="E501", message="long")],
    )
    tr = _Handler(result)._tool_run_lint({"path": "a.py"})
    assert tr.metadata == {"issue_count": 1}


def test_tool_run_lint_collapses_newlines_in_fix_hint():
    result = LintResult(
        ok=False,
        issues=[
            LintIssue(file="a.py", line=1, col=1, code="F401", message="unused", fix="Remove unused import\n\n  `os`"),
        ],
    )
    tr = _Handler(result)._tool_run_lint({"path": "a.py"})
    assert "[fix: Remove unused import `os`]" in tr.content
