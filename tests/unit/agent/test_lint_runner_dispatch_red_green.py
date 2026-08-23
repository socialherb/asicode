"""RED→GREEN: LintRunner 디스패치/gofmt/golangci/경로 실패 브랜치.

run_lint의 언어 디스패치(ruff/eslint/go/generic/none), gofmt -d diff 파싱,
golangci-lint 이슈 수집, _resolve_path/_normalize_file_path 예외 경로를 고정.
"""

from __future__ import annotations

import json
import types

from external_llm.agent.lint_runner import LintIssue, LintResult, LintRunner
from external_llm.languages import LanguageRegistry


def _proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunRuffErrors:
    def test_invalid_path_returns_error_result(self, tmp_path):
        r = LintRunner(str(tmp_path))
        out = r.run_ruff(str(tmp_path / "missing.py"))
        assert out.ok is False
        assert "invalid path" in out.summary

    def test_ruff_nonzero_rc_surfaces_stderr(self, tmp_path, monkeypatch):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        monkeypatch.setattr(
            "external_llm.agent.lint_runner.subprocess.run", lambda *a, **k: _proc(returncode=2, stderr="syntax error")
        )
        out = LintRunner(str(tmp_path)).run_ruff(str(f))
        assert out.ok is False
        assert "exit code 2" in out.summary
        assert "syntax error" in out.summary
        assert out.stderr == "syntax error"

    def test_summary_marks_truncation_when_at_cap(self, tmp_path, monkeypatch):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        issues = [
            {
                "filename": "a.py",
                "location": {"row": i, "column": 1},
                "code": "F401",
                "message": f"unused {i}",
                "severity": "warning",
                "fix": None,
            }
            for i in range(1, 3)
        ]
        monkeypatch.setattr(
            "external_llm.agent.lint_runner.subprocess.run",
            lambda *a, **k: _proc(returncode=1, stdout=json.dumps(issues)),
        )
        out = LintRunner(str(tmp_path)).run_ruff(str(f), max_issues=2)
        assert out.ok is False
        assert "(truncated to 2)" in out.summary

    def test_fixable_count_annotated(self, tmp_path, monkeypatch):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        issues = [
            {
                "filename": "a.py",
                "location": {"row": 1, "column": 1},
                "code": "F401",
                "message": "unused",
                "severity": "warning",
                "fix": {"message": "remove"},
            },
        ]
        monkeypatch.setattr(
            "external_llm.agent.lint_runner.subprocess.run",
            lambda *a, **k: _proc(returncode=1, stdout=json.dumps(issues)),
        )
        out = LintRunner(str(tmp_path)).run_ruff(str(f))
        assert "(1 auto-fixable)" in out.summary


class TestRunLintDispatch:
    def test_python_dispatches_to_ruff(self, tmp_path, monkeypatch):
        r = LintRunner(str(tmp_path))
        sentinel = LintResult(ok=True, summary="s")
        monkeypatch.setattr(r, "run_ruff", lambda path, max_issues=50: sentinel)
        assert r.run_lint("a.py") is sentinel

    def test_go_dispatches_to_go_lint(self, tmp_path, monkeypatch):
        r = LintRunner(str(tmp_path))
        sentinel = LintResult(ok=True, summary="s")
        monkeypatch.setattr(r, "_run_go_lint", lambda path, max_issues=50: sentinel)
        assert r.run_lint("a.go") is sentinel

    def test_ts_dispatches_to_eslint(self, tmp_path, monkeypatch):
        r = LintRunner(str(tmp_path))
        sentinel = LintResult(ok=True, summary="s")
        monkeypatch.setattr(r, "_run_eslint", lambda path, max_issues=50: sentinel)
        assert r.run_lint("a.ts") is sentinel

    def test_provider_with_lint_command_uses_generic(self, tmp_path, monkeypatch):
        r = LintRunner(str(tmp_path))
        sentinel = LintResult(ok=True, summary="generic")
        provider = types.SimpleNamespace(get_lint_command=lambda p: ["my-linter", p])
        reg = types.SimpleNamespace(get=lambda p: provider)
        monkeypatch.setattr(LanguageRegistry, "instance", staticmethod(lambda: reg))
        monkeypatch.setattr(r, "_run_generic_lint", lambda cmd, path, max_issues=50: sentinel)
        assert r.run_lint("a.rb") is sentinel

    def test_no_linter_registered_skipped(self, tmp_path, monkeypatch):
        r = LintRunner(str(tmp_path))
        reg = types.SimpleNamespace(get=lambda p: None)
        monkeypatch.setattr(LanguageRegistry, "instance", staticmethod(lambda: reg))
        out = r.run_lint("a.xyz")
        assert out.ok is True and out.skipped is True
        assert "no linter for unknown" in out.summary


class TestGenericLint:
    def test_invalid_path_returns_error(self, tmp_path):
        r = LintRunner(str(tmp_path))
        out = r._run_generic_lint(["x"], str(tmp_path / "nope.rb"))
        assert out.ok is False
        assert "invalid path" in out.summary

    def test_rc_zero_ok(self, tmp_path, monkeypatch):
        (tmp_path / "a.rb").write_text("x = 1")
        monkeypatch.setattr("external_llm.agent.lint_runner.subprocess.run", lambda *a, **k: _proc(returncode=0))
        out = LintRunner(str(tmp_path))._run_generic_lint(["tool"], "a.rb")
        assert out.ok is True
        assert "no lint issues" in out.summary

    def test_blank_lines_skipped_and_max_issues_break(self, tmp_path, monkeypatch):
        (tmp_path / "a.rb").write_text("x = 1")
        monkeypatch.setattr(
            "external_llm.agent.lint_runner.subprocess.run",
            lambda *a, **k: _proc(returncode=1, stdout="w1\n\nw2\nw3\n"),
        )
        out = LintRunner(str(tmp_path))._run_generic_lint(["tool"], "a.rb", max_issues=2)
        assert [i.message for i in out.issues] == ["w1", "w2"]  # 빈 줄 스킵 + cap
        assert out.ok is False


class TestGoLint:
    def test_invalid_path_returns_error(self, tmp_path):
        r = LintRunner(str(tmp_path))
        out = r._run_go_lint(str(tmp_path / "nope.go"))
        assert out.ok is False

    def test_gofmt_failure_soft_logs_and_continues(self, tmp_path, monkeypatch, caplog):
        import logging

        (tmp_path / "a.go").write_text("package main\n")
        r = LintRunner(str(tmp_path))
        err = LintResult(ok=False, summary="gofmt broke", error="gofmt broke")
        monkeypatch.setattr(r, "_run_lint_command", lambda cmd, tool, timeout=30: (None, err))
        with caplog.at_level(logging.WARNING):
            out = r._run_go_lint(str(tmp_path / "a.go"))
        assert out.ok is True  # 소프트 실패 — 이슈 없이 통과
        assert "gofmt check failed" in caplog.text

    def test_gofmt_diff_parsed_into_issues(self, tmp_path, monkeypatch):
        (tmp_path / "a.go").write_text("package main\n")
        r = LintRunner(str(tmp_path))
        diff = "@@ -10,6 +10,8 @@\n context\n+    bad indent\n+more bad\n"
        calls = {"n": 0}

        def fake(cmd, tool, timeout=30):
            calls["n"] += 1
            if calls["n"] == 1:  # gofmt
                return _proc(returncode=0, stdout=diff), None
            return _proc(returncode=0), None  # golangci-lint

        monkeypatch.setattr(r, "_run_lint_command", fake)
        out = r._run_go_lint(str(tmp_path / "a.go"))
        assert out.ok is False
        codes = [i.code for i in out.issues]
        assert codes == ["gofmt", "gofmt"]
        assert out.issues[0].line == 10  # hunk 헤더에서 추출한 라인
        assert "bad indent" in out.issues[0].message

    def test_golangci_issues_collected(self, tmp_path, monkeypatch):
        (tmp_path / "a.go").write_text("package main\n")
        r = LintRunner(str(tmp_path))
        calls = {"n": 0}

        def fake(cmd, tool, timeout=30):
            calls["n"] += 1
            if calls["n"] == 1:  # gofmt clean
                return _proc(returncode=0, stdout=""), None
            return _proc(returncode=1, stdout="", stderr="error line 1\n\n\nerror line 2\n"), None

        monkeypatch.setattr(r, "_run_lint_command", fake)
        out = r._run_go_lint(str(tmp_path / "a.go"))
        assert out.ok is False
        assert [i.code for i in out.issues] == ["golangci-lint", "golangci-lint"]

    def test_golangci_soft_failure_logged(self, tmp_path, monkeypatch, caplog):
        import logging

        (tmp_path / "a.go").write_text("package main\n")
        r = LintRunner(str(tmp_path))
        err = LintResult(ok=False, summary="golangci broke", error="x")
        calls = {"n": 0}

        def fake(cmd, tool, timeout=30):
            calls["n"] += 1
            if calls["n"] == 1:
                return _proc(returncode=0), None
            return None, err

        monkeypatch.setattr(r, "_run_lint_command", fake)
        with caplog.at_level(logging.WARNING):
            out = r._run_go_lint(str(tmp_path / "a.go"))
        assert out.ok is True
        assert "golangci-lint check failed" in caplog.text

    def test_issue_cap_slices_go_issues(self, tmp_path, monkeypatch):
        (tmp_path / "a.go").write_text("package main\n")
        r = LintRunner(str(tmp_path))
        diff = "@@ -1,1 +1,3 @@\n+" + "\n+".join(f"bad {i}" for i in range(5)) + "\n"

        def fake(cmd, tool, timeout=30):
            return _proc(returncode=0, stdout=diff), None

        monkeypatch.setattr(r, "_run_lint_command", fake)
        out = r._run_go_lint(str(tmp_path / "a.go"), max_issues=2)
        assert len(out.issues) == 2


class TestPathResolutionErrors:
    def test_resolve_path_embedded_null_byte_returns_none(self, tmp_path):
        r = LintRunner(str(tmp_path))
        assert r._resolve_path("\x00") is None  # Path.resolve() ValueError → None

    def test_normalize_absolute_outside_repo_kept_as_is(self, tmp_path):
        r = LintRunner(str(tmp_path))
        assert r._normalize_file_path("/elsewhere/x.py") == "/elsewhere/x.py"

    def test_eslint_invalid_path_returns_error(self, tmp_path):
        r = LintRunner(str(tmp_path))
        out = r._run_eslint(str(tmp_path / "missing.ts"))
        assert out.ok is False
        assert "invalid path" in out.summary


class TestPathBoundaryAndFixableCount:
    def test_resolve_path_escaping_repo_rejected(self, tmp_path):
        r = LintRunner(str(tmp_path))
        assert r._resolve_path("../outside.py") is None

    def test_fixable_count_property(self, tmp_path):
        from external_llm.agent.lint_runner import LintResult

        r = LintResult(
            ok=False,
            issues=[
                LintIssue(file="a", line=1, col=0, code="F401", message="m", fix="remove"),
                LintIssue(file="a", line=2, col=0, code="E501", message="m", fix=None),
            ],
        )
        assert r.fixable_count == 1
