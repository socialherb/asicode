"""
Lint Runner for asicode Agent

Runs language-aware lint checks (ruff for Python, eslint for TS/JS).
Gracefully skips if the linter is not installed.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from ..languages import LanguageId, LanguageRegistry

logger = logging.getLogger(__name__)


@dataclass
class LintIssue:
    file: str
    line: int
    col: int
    code: str
    message: str
    severity: str = "error"
    # ruff's fix information (if available)
    fix: Optional[str] = None


@dataclass
class LintResult:
    ok: bool
    issues: list[LintIssue] = field(default_factory=list)
    summary: str = ""
    skipped: bool = False
    error: Optional[str] = None
    stderr: Optional[str] = None  # ruff's stderr output (for debugging)


class LintRunner:
    """
    Runs ruff lint check on a file or path.
    Gracefully skips if ruff is not installed.
    """
    DEFAULT_MAX_ISSUES: int = 50  #default maximum issue can/number

    def __init__(self, repo_root: str):
        self.repo_root = str(Path(repo_root).resolve())

    def run_ruff(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
        severity_filter: Optional[Literal["error", "warning", "info"]] = None
    ) -> LintResult:
        """
        Run ruff check on the given path (file or directory).
        Returns LintResult with structured issues.
        If ruff is not installed, returns LintResult(ok=True, skipped=True) with graceful skip.

        Args:
            path: File or directory path to lint.
                  Empty string or "." means the entire repository.
            max_issues: Maximum number of issues to return. Defaults to 50.
            severity_filter: Filter issues by severity (error, warning, info).
                             If None, all issues are returned.
        """
        abs_path = self._resolve_path(path)
        if abs_path is None:
            return LintResult(
                ok=False,
                summary=f"invalid path: {path!r}",
                error=f"Path not found or outside repo: {path!r}",
            )

        #ruff execution
        try:
            proc = subprocess.run(
                ["ruff", "check", "--output-format=json", str(abs_path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            stderr_output = proc.stderr
        except FileNotFoundError:
            logger.debug("ruff not installed; skipping lint")
            return LintResult(ok=True, skipped=True, summary="ruff not installed; lint skipped")
        except subprocess.TimeoutExpired:
            return LintResult(ok=False, summary="ruff timed out", error="ruff timed out after 30s")
        except Exception as e:
            return LintResult(ok=False, summary=str(e), error=str(e))

        # When ruff returned an error (e.g., syntax error, file not found)
        if proc.returncode != 0 and proc.returncode != 1:
            # ruff's common exit codes:
            # 0: success (no issues found)
            # 1: Issues found
            # 2: Error occurred (e.g., syntax error, file not found)
            error_msg = f"ruff failed with exit code {proc.returncode}"
            if stderr_output:
                error_msg += f": {stderr_output[:200]}"
            return LintResult(
                ok=False, summary=error_msg, error=error_msg, stderr=stderr_output)

        issues: list[LintIssue] = []
        if proc.stdout.strip():
            try:
                raw = json.loads(proc.stdout)
                for item in raw:
                    loc = item.get("location", {})
                    fix_info = item.get("fix", {})
                    severity = item.get("severity", "error")

                    # Apply severity filter
                    if severity_filter is None or severity == severity_filter:
                        issues.append(LintIssue(
                            file=self._normalize_file_path(item.get("filename", path)),
                            line=loc.get("row", 0),
                            col=loc.get("column", 0),
                            code=item.get("code", ""),
                            message=item.get("message", ""),
                            severity=severity,
                            fix=fix_info.get("message") if fix_info else None))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse ruff output")

        # Limit issues to max_issues if specified
        if max_issues > 0 and len(issues) > max_issues:
            issues = issues[:max_issues]

        ok = len(issues) == 0
        if issues:
            summary = f"{len(issues)} lint issue(s) found"
            if max_issues > 0 and len(issues) == max_issues:
                summary += f" (truncated to {max_issues})"
        else:
            summary = "no lint issues"

        return LintResult(ok=ok, issues=issues, summary=summary)

    def run_lint(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
    ) -> LintResult:
        """Language-aware lint: dispatches to ruff (Python), eslint (TS/JS),
        gofmt+golangci-lint (Go), or provider dispatch.

        For unsupported languages, returns ``LintResult(ok=True, skipped=True)``.
        """
        lang = LanguageId.from_path(path)
        if lang == LanguageId.PYTHON:
            return self.run_ruff(path, max_issues=max_issues)
        if lang in (LanguageId.TYPESCRIPT, LanguageId.JAVASCRIPT):
            return self._run_eslint(path, max_issues=max_issues)
        if lang == LanguageId.GO:
            return self._run_go_lint(path, max_issues=max_issues)
        # Generic dispatch via provider
        provider = LanguageRegistry.instance().get(path)
        if provider:
            cmd = provider.get_lint_command(path)
            if cmd:
                return self._run_generic_lint(cmd, path, max_issues=max_issues)
        # No linter registered
        return LintResult(ok=True, skipped=True, summary=f"no linter for {lang.value}")

    def _run_generic_lint(
        self,
        cmd: list[str],
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
    ) -> LintResult:
        """Run a generic lint command. Gracefully skips if the tool is not installed."""
        abs_path = self._resolve_path(path)
        if abs_path is None:
            return LintResult(
                ok=False,
                summary=f"invalid path: {path!r}",
                error=f"Path not found or outside repo: {path!r}",
            )

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            tool = cmd[0] if cmd else "linter"
            logger.debug("%s not installed; skipping lint", tool)
            return LintResult(ok=True, skipped=True, summary=f"{tool} not installed; lint skipped")
        except subprocess.TimeoutExpired:
            return LintResult(ok=False, summary="lint timed out", error="lint timed out after 30s")
        except Exception as e:
            return LintResult(ok=False, summary=str(e), error=str(e))

        if proc.returncode == 0:
            return LintResult(ok=True, summary="no lint issues")

        # Parse generic output: each non-empty line is an issue
        issues: list[LintIssue] = []
        for line in (proc.stdout + proc.stderr).splitlines():
            line = line.strip()
            if not line:
                continue
            issues.append(LintIssue(
                file=path, line=0, col=0, code="", message=line[:200],
            ))
            if max_issues > 0 and len(issues) >= max_issues:
                break

        ok = len(issues) == 0
        summary = f"{len(issues)} lint issue(s) found" if issues else "no lint issues"
        return LintResult(ok=ok, issues=issues, summary=summary)

    def _run_eslint(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
    ) -> LintResult:
        """Run eslint on a TS/JS file. Gracefully skips if eslint is not installed."""
        abs_path = self._resolve_path(path)
        if abs_path is None:
            return LintResult(
                ok=False,
                summary=f"invalid path: {path!r}",
                error=f"Path not found or outside repo: {path!r}",
            )

        try:
            proc = subprocess.run(
                ["npx", "eslint", "--format=json", str(abs_path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            logger.debug("npx/eslint not installed; skipping lint")
            return LintResult(ok=True, skipped=True, summary="eslint not installed; lint skipped")
        except subprocess.TimeoutExpired:
            return LintResult(ok=False, summary="eslint timed out", error="eslint timed out after 30s")
        except Exception as e:
            return LintResult(ok=False, summary=str(e), error=str(e))

        issues: list[LintIssue] = []
        if proc.stdout.strip():
            try:
                raw = json.loads(proc.stdout)
                for file_entry in raw:
                    for msg in file_entry.get("messages", []):
                        severity = "error" if msg.get("severity", 0) >= 2 else "warning"
                        issues.append(LintIssue(
                            file=self._normalize_file_path(
                                file_entry.get("filePath", path)),
                            line=msg.get("line", 0),
                            col=msg.get("column", 0),
                            code=msg.get("ruleId", "") or "",
                            message=msg.get("message", ""),
                            severity=severity,
                        ))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse eslint output")

        if max_issues > 0 and len(issues) > max_issues:
            issues = issues[:max_issues]

        ok = len(issues) == 0
        summary = f"{len(issues)} lint issue(s) found" if issues else "no lint issues"
        return LintResult(ok=ok, issues=issues, summary=summary)

    def _run_go_lint(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
    ) -> LintResult:
        """Run gofmt -d (format check) then golangci-lint run on a Go file.

        ``gofmt -d`` prints a unified diff when the file is not gofmt-compliant;
        ``golangci-lint`` catches deeper lint issues.  Two-pass ensures that
        formatting regressions (e.g. LLM-emitted space indentation in a
        tab-indented file) are not silently accepted.
        """
        abs_path = self._resolve_path(path)
        if abs_path is None:
            return LintResult(
                ok=False,
                summary=f"invalid path: {path!r}",
                error=f"Path not found or outside repo: {path!r}",
            )

        all_issues: list[LintIssue] = []
        gofmt_available = True

        # ── Pass 1: gofmt -d (formatting check) ──────────────────────────
        try:
            proc = subprocess.run(
                ["gofmt", "-d", str(abs_path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            gofmt_available = False
            logger.debug("gofmt not installed; skipping gofmt check")
        except subprocess.TimeoutExpired:
            logger.warning("gofmt timed out for %s", path)
        except Exception as e:
            logger.debug("gofmt error: %s", e)
        else:
            # gofmt -d prints diff to stdout when file is unformatted.
            # Extract line numbers from diff hunk headers (e.g. @@ -10,6 +10,8 @@)
            if proc.stdout.strip():
                fmt_line = 0
                for line in proc.stdout.splitlines():
                    m = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                    if m:
                        fmt_line = int(m.group(1))
                    elif line.startswith("+") and not line.startswith("+++"):
                        all_issues.append(LintIssue(
                            file=path,
                            line=fmt_line,
                            col=0,
                            code="gofmt",
                            message=f"format: {line[1:80].rstrip()}",
                            severity="error",
                        ))

        # ── Pass 2: golangci-lint ─────────────────────────────────────────
        if gofmt_available:
            try:
                proc = subprocess.run(
                    ["golangci-lint", "run", str(abs_path)],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except FileNotFoundError:
                logger.debug("golangci-lint not installed; skipping")
            except subprocess.TimeoutExpired:
                logger.warning("golangci-lint timed out for %s", path)
            except Exception as e:
                logger.debug("golangci-lint error: %s", e)
            else:
                if proc.returncode != 0:
                    for line in (proc.stdout + proc.stderr).splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        all_issues.append(LintIssue(
                            file=path, line=0, col=0,
                            code="golangci-lint", message=stripped[:200],
                        ))

        if max_issues > 0 and len(all_issues) > max_issues:
            all_issues = all_issues[:max_issues]

        ok = len(all_issues) == 0
        summary = f"{len(all_issues)} lint issue(s) found" if all_issues else "no lint issues"
        return LintResult(ok=ok, issues=all_issues, summary=summary)

    def _resolve_path(self, path: str) -> Optional[Path]:
        """Resolve path within repo_root, return None if invalid."""
        try:
            repo = Path(self.repo_root)
            p = (repo / path).resolve()
            if not str(p).startswith(str(repo)):
                return None
            if not p.exists():
                return None
            return p
        except Exception:
            return None  # non-critical — never block execution

    def _normalize_file_path(self, file_path: str) -> str:
        """
        Normalize file path to be relative to repo_root.
        If the path is already relative or cannot be made relative, return as is.
        """
        try:
            # Try to make it relative to repo_root
            abs_path = Path(file_path)
            if abs_path.is_absolute():
                return str(abs_path.relative_to(self.repo_root))
        except (ValueError, TypeError):
            pass
        return file_path

