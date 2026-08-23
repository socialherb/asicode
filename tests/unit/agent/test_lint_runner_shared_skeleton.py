"""Shared lint-command skeleton contract (R24 single-source refactor).

``LintRunner._run_lint_command`` and ``_resolve_path_or_error`` are the single
subprocess / path-resolution skeletons shared by all four lint runners (ruff,
generic, eslint, gofmt+golangci-lint).  ``_lint_summary`` is the single source
for the result summary line (``"N lint issue(s) found"`` / ``"no lint issues"``).  These tests lock the failure mapping —
tool missing → graceful skip, timeout → tool-named error, other failure →
error, go passes → soft-fail — so a future edit cannot silently change the
contract for one runner only.
"""

from __future__ import annotations

import json
import logging
import subprocess
import types
from unittest import mock

from external_llm.agent.lint_runner import (
    LintIssue,
    LintRunner,
    _lint_summary,
    _parse_lint_json,
)


def _fake_proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _run_patch(return_value=None, side_effect=None):
    return mock.patch(
        "external_llm.agent.lint_runner.subprocess.run",
        return_value=return_value,
        side_effect=side_effect,
    )


def test_success_returns_proc_with_no_error(tmp_path):
    with _run_patch(return_value=_fake_proc()):
        proc, err = LintRunner(str(tmp_path))._run_lint_command(["tool"], "tool")
    assert err is None
    assert proc is not None
    assert proc.returncode == 0


def test_skeleton_passes_cwd_capture_text_timeout(tmp_path):
    with _run_patch(return_value=_fake_proc()) as m:
        LintRunner(str(tmp_path))._run_lint_command(["tool", "x"], "tool", timeout=60)
    assert m.call_args.kwargs["cwd"] == str(tmp_path)
    assert m.call_args.kwargs["capture_output"] is True
    assert m.call_args.kwargs["text"] is True
    assert m.call_args.kwargs["timeout"] == 60


def test_missing_tool_maps_to_graceful_skip(tmp_path):
    with _run_patch(side_effect=FileNotFoundError("tool")):
        proc, err = LintRunner(str(tmp_path))._run_lint_command(["tool"], "tool")
    assert proc is None
    assert err is not None
    assert err.ok is True
    assert err.skipped is True
    assert err.summary == "tool not installed; lint skipped"


def test_timeout_maps_to_tool_named_error(tmp_path):
    with _run_patch(side_effect=subprocess.TimeoutExpired("tool", timeout=30)):
        proc, err = LintRunner(str(tmp_path))._run_lint_command(["tool"], "tool")
    assert proc is None
    assert err is not None
    assert err.ok is False
    assert err.summary == "tool timed out"
    assert err.error == "tool timed out after 30s"


def test_unexpected_failure_maps_to_error(tmp_path):
    with _run_patch(side_effect=RuntimeError("boom")):
        proc, err = LintRunner(str(tmp_path))._run_lint_command(["tool"], "tool")
    assert proc is None
    assert err is not None
    assert err.ok is False
    assert err.summary == "boom"
    assert err.error == "boom"


def test_ruff_timeout_surfaces_through_run_ruff(tmp_path):
    # end-to-end wiring: the skeleton feeds run_ruff's error path
    (tmp_path / "a.py").write_text("x = 1\n")
    with _run_patch(side_effect=subprocess.TimeoutExpired("ruff", timeout=30)):
        result = LintRunner(str(tmp_path)).run_ruff("")
    assert result.ok is False
    assert result.error == "ruff timed out after 30s"


def test_generic_timeout_names_the_actual_tool(tmp_path):
    # R24: generic runner's timeout message names cmd[0] (was hardcoded "lint")
    (tmp_path / "a.txt").write_text("x\n")
    runner = LintRunner(str(tmp_path))
    with _run_patch(side_effect=subprocess.TimeoutExpired("mytool", timeout=30)):
        result = runner._run_generic_lint(["mytool", "check"], "a.txt")
    assert result.ok is False
    assert result.error == "mytool timed out after 30s"


def test_resolve_path_or_error_returns_path_when_valid(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    abs_path, err = LintRunner(str(tmp_path))._resolve_path_or_error("a.py")
    assert err is None
    assert abs_path is not None
    assert abs_path == (tmp_path / "a.py").resolve()


def test_resolve_path_or_error_maps_missing_to_invalid_result(tmp_path):
    abs_path, err = LintRunner(str(tmp_path))._resolve_path_or_error("nope.py")
    assert abs_path is None
    assert err is not None
    assert err.ok is False
    assert err.error == "Path not found or outside repo: 'nope.py'"
    assert err.summary == "invalid path: 'nope.py'"


def test_go_lint_soft_fails_when_gofmt_missing(tmp_path):
    # gofmt absent is non-fatal: pass 2 is skipped, result stays ok (no issues)
    (tmp_path / "main.go").write_text("package main\n")

    def _missing(*args, **kwargs):
        raise FileNotFoundError("gofmt")

    with mock.patch("external_llm.agent.lint_runner.subprocess.run", _missing):
        result = LintRunner(str(tmp_path))._run_go_lint("main.go")
    assert result.ok is True
    assert result.skipped is False
    assert result.summary == "no lint issues"


def test_lint_summary_reports_issue_count():
    one = [LintIssue(file="a.py", line=1, col=1, code="E501", message="long line")]
    assert _lint_summary(one) == "1 lint issue(s) found"
    assert _lint_summary(one * 3) == "3 lint issue(s) found"


def test_lint_summary_empty_reports_clean():
    assert _lint_summary([]) == "no lint issues"


def test_generic_summary_flows_through_shared_helper(tmp_path):
    # Wiring: _run_generic_lint builds its summary via _lint_summary, so a
    # findings run reads "2 lint issue(s) found" — not a hand-rolled string.
    (tmp_path / "a.txt").write_text("x\n")
    with _run_patch(return_value=_fake_proc(returncode=1, stdout="boom 1\nboom 2\n")):
        result = LintRunner(str(tmp_path))._run_generic_lint(["mytool"], "a.txt")
    assert result.ok is False
    assert result.summary == "2 lint issue(s) found"


def test_parse_lint_json_malformed_output_returns_empty_and_warns(caplog):
    # A broken tool response must never fail the gate — decode errors degrade to [].
    with caplog.at_level(logging.WARNING):
        issues = _parse_lint_json("{not json", "ruff", lambda raw: [raw])
    assert issues == []
    assert "Failed to parse ruff output" in caplog.text


def test_parse_lint_json_empty_stdout_skips_parse():
    # Empty/whitespace stdout must not even attempt a parse (ruff rc=0 emits none).
    called = False

    def _parse(raw):
        nonlocal called
        called = True
        return []

    assert _parse_lint_json("  ", "ruff", _parse) == []
    assert called is False


def test_parse_lint_json_truncates_to_max_issues():
    issues = [LintIssue(file=f"a{i}.py", line=1, col=1, code="E1", message="m") for i in range(3)]
    out = _parse_lint_json("[1,2,3]", "ruff", lambda raw: issues, max_issues=2)
    assert out == issues[:2]


def test_parse_lint_json_extracts_parse_results():
    out = _parse_lint_json(
        "[{}]", "eslint", lambda raw: [LintIssue(file="a.py", line=1, col=1, code="E1", message="m") for _ in raw]
    )
    assert len(out) == 1
    assert out[0].code == "E1"


def test_ruff_json_issues_flow_through_parse_helper(tmp_path):
    # Wiring: run_ruff's issue extraction goes through _parse_lint_json, so the
    # severity filter + truncation semantics are the helper's.
    (tmp_path / "a.py").write_text("x = 1\n")
    payload = json.dumps(
        [
            {
                "filename": str(tmp_path / "a.py"),
                "location": {"row": 1, "column": 1},
                "code": "F401",
                "message": "unused import",
                "severity": "error",
                "fix": {"message": "remove"},
            }
        ]
    )
    with _run_patch(return_value=_fake_proc(returncode=1, stdout=payload)):
        result = LintRunner(str(tmp_path)).run_ruff("a.py")
    assert result.ok is False
    assert len(result.issues) == 1
    assert result.issues[0].code == "F401"
    assert result.issues[0].fix == "remove"
    assert result.summary == "1 lint issue(s) found (1 auto-fixable)"
