"""eslint runner returncode contract (mirror of run_ruff).

Covers _run_eslint error surfacing — regression guard for the silent-pass
bug where fatal eslint runs (rc != 0/1, or rc==1 with no JSON stdout) were
reported as "no lint issues":

- rc 0            → clean run (ok=True)
- rc 1 + JSON     → findings parsed into LintIssue list
- rc 2            → fatal (config error / unresolvable file) → ok=False +
                    error with exit code + stderr snippet
- rc 1 + empty stdout → npx could not resolve eslint (stderr-only) →
                    ok=False, not "no lint issues"
- stderr snippet truncated to 200 chars; full stderr still attached
- npx missing (FileNotFoundError) → graceful skip preserved
"""
from __future__ import annotations

import json
import types
from unittest import mock

import pytest

from external_llm.agent.lint_runner import LintRunner


def _eslint_run(stdout: str, returncode: int = 0, stderr: str = ""):
    return mock.patch(
        "external_llm.agent.lint_runner.subprocess.run",
        return_value=types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr,
        ),
    )


@pytest.fixture
def ts_file(tmp_path):
    p = tmp_path / "sample.ts"
    p.write_text("const x: number = 1;\n")
    return p


def test_eslint_clean_run(tmp_path, ts_file):
    with _eslint_run(stdout="[]", returncode=0):
        result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.ok is True
    assert result.summary == "no lint issues"
    assert result.error is None


def test_eslint_findings_parsed_from_json(tmp_path, ts_file):
    raw = json.dumps([{
        "filePath": str(ts_file),
        "messages": [{
            "ruleId": "no-unused-vars", "severity": 2, "line": 1, "column": 5,
            "message": "'x' is assigned a value but never used",
        }],
    }])
    with _eslint_run(stdout=raw, returncode=1):
        result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.ok is False
    assert len(result.issues) == 1
    assert result.issues[0].code == "no-unused-vars"
    assert result.issues[0].severity == "error"
    assert result.summary == "1 lint issue(s) found"
    assert result.error is None


def test_eslint_fatal_exit_code_surfaces_error(tmp_path, ts_file):
    # rc=2 = fatal (invalid config / file not found) → previously silent pass
    with _eslint_run(stdout="", returncode=2,
                     stderr="Oops! Something went wrong! ... ConfigError"):
        result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.ok is False
    assert result.error is not None
    assert "exit code 2" in result.error
    assert "ConfigError" in result.error
    assert result.summary == result.error
    assert result.stderr == "Oops! Something went wrong! ... ConfigError"


def test_eslint_npx_resolution_failure_surfaces_error(tmp_path, ts_file):
    # npx cannot resolve the eslint package: rc=1, JSON absent, stderr only.
    # A findings run always emits JSON, so this must NOT read as "no lint issues".
    with _eslint_run(stdout="", returncode=1,
                     stderr="npm ERR! Could not find eslint"):
        result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.ok is False
    assert result.error is not None
    assert "exit code 1" in result.error
    assert "npm ERR" in result.error
    assert result.issues == []


def test_eslint_error_stderr_truncated_to_200(tmp_path, ts_file):
    long_stderr = "E" * 500
    with _eslint_run(stdout="", returncode=2, stderr=long_stderr):
        result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.error == f"eslint failed with exit code 2: {'E' * 200}"
    # full stderr still attached for debugging
    assert result.stderr == long_stderr


def test_eslint_not_installed_skips_gracefully(tmp_path, ts_file, monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("npx")
    monkeypatch.setattr(
        "external_llm.agent.lint_runner.subprocess.run", _raise)
    result = LintRunner(str(tmp_path)).run_lint(str(ts_file))
    assert result.ok is True
    assert result.skipped is True
    assert "not installed" in result.summary
