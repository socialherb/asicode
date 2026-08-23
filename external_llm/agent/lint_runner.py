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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    fix: str | None = None


@dataclass
class LintResult:
    ok: bool
    issues: list[LintIssue] = field(default_factory=list)
    summary: str = ""
    skipped: bool = False
    error: str | None = None
    stderr: str | None = None  # ruff's stderr output (for debugging)

    @property
    def fixable_count(self) -> int:
        """Number of issues carrying an auto-fix suggestion (``fix`` set)."""
        return sum(1 for i in self.issues if i.fix)


def _lint_summary(issues: list[LintIssue]) -> str:
    """Shared summary line: ``"N lint issue(s) found"`` or ``"no lint issues"``.

    Single source for the summary string emitted by all four lint runners
    (ruff, generic, eslint, gofmt+golangci-lint).  ``run_ruff`` appends the
    ``auto-fixable`` / ``truncated`` suffixes on top of the base string.
    """
    return f"{len(issues)} lint issue(s) found" if issues else "no lint issues"


def _parse_lint_json(
    stdout: str,
    tool: str,
    parse: Callable[[list], list[LintIssue]],
    max_issues: int = 0,
) -> list[LintIssue]:
    """Parse *tool*'s JSON lint stdout into ``LintIssue``s.

    Single source for the JSON-parse skeleton shared by the ruff and eslint
    runners: the ``stdout.strip()`` guard, the decode tolerance (a malformed
    tool response logs a warning and yields no issues — it must never fail the
    lint gate), and the ``max_issues`` truncation.

    *parse* converts the decoded JSON array into issues; the two runners'
    schemas differ (ruff: flat issue objects; eslint: per-file ``messages``).
    """
    issues: list[LintIssue] = []
    if stdout.strip():
        try:
            raw = json.loads(stdout)
            issues.extend(parse(raw))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse %s output", tool)
    if max_issues > 0 and len(issues) > max_issues:
        issues = issues[:max_issues]
    return issues


class LintRunner:
    """
    Runs ruff lint check on a file or path.
    Gracefully skips if ruff is not installed.
    """

    DEFAULT_MAX_ISSUES: int = 50  # default maximum issue can/number

    def __init__(self, repo_root: str):
        self.repo_root = str(Path(repo_root).resolve())

    def run_ruff(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
        severity_filter: Literal["error", "warning", "info"] | None = None,
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
        abs_path, err = self._resolve_path_or_error(path)
        if err is not None:
            return err

        proc, err = self._run_lint_command(["ruff", "check", "--output-format=json", str(abs_path)], "ruff")
        if err is not None:
            return err
        assert proc is not None  # (proc, None) arm per _run_lint_command contract
        stderr_output = proc.stderr

        # When ruff returned an error (e.g., syntax error, file not found)
        if proc.returncode not in {0, 1}:
            # ruff's common exit codes:
            # 0: success (no issues found)
            # 1: Issues found
            # 2: Error occurred (e.g., syntax error, file not found)
            error_msg = f"ruff failed with exit code {proc.returncode}"
            if stderr_output:
                error_msg += f": {stderr_output[:200]}"
            return LintResult(ok=False, summary=error_msg, error=error_msg, stderr=stderr_output)

        issues = _parse_lint_json(
            proc.stdout,
            "ruff",
            lambda raw: [
                LintIssue(
                    file=self._normalize_file_path(item.get("filename", path)),
                    line=item.get("location", {}).get("row", 0),
                    col=item.get("location", {}).get("column", 0),
                    code=item.get("code", ""),
                    message=item.get("message", ""),
                    severity=item.get("severity", "error"),
                    fix=(item.get("fix") or {}).get("message"),
                )
                for item in raw
                if severity_filter is None or item.get("severity", "error") == severity_filter
            ],
            max_issues,
        )

        ok = len(issues) == 0
        summary = _lint_summary(issues)
        if issues:
            fixable = sum(1 for i in issues if i.fix)
            if fixable:
                summary += f" ({fixable} auto-fixable)"
            if max_issues > 0 and len(issues) == max_issues:
                summary += f" (truncated to {max_issues})"

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
        _abs_path, err = self._resolve_path_or_error(path)
        if err is not None:
            return err

        proc, err = self._run_lint_command(cmd, cmd[0] if cmd else "linter")
        if err is not None:
            return err
        assert proc is not None  # (proc, None) arm per _run_lint_command contract

        if proc.returncode == 0:
            return LintResult(ok=True, summary=_lint_summary([]))

        # Parse generic output: each non-empty line is an issue
        issues: list[LintIssue] = []
        for line in (proc.stdout + proc.stderr).splitlines():
            line = line.strip()
            if not line:
                continue
            issues.append(
                LintIssue(
                    file=path,
                    line=0,
                    col=0,
                    code="",
                    message=line[:200],
                )
            )
            if max_issues > 0 and len(issues) >= max_issues:
                break

        ok = len(issues) == 0
        summary = _lint_summary(issues)
        return LintResult(ok=ok, issues=issues, summary=summary)

    def _run_eslint(
        self,
        path: str,
        max_issues: int = DEFAULT_MAX_ISSUES,
    ) -> LintResult:
        """Run eslint on a TS/JS file. Gracefully skips if eslint is not installed."""
        abs_path, err = self._resolve_path_or_error(path)
        if err is not None:
            return err

        proc, err = self._run_lint_command(["npx", "eslint", "--format=json", str(abs_path)], "eslint")
        if err is not None:
            return err
        assert proc is not None  # (proc, None) arm per _run_lint_command contract

        # ESLint exit codes: 0 = clean, 1 = findings (JSON always emitted to
        # stdout). Anything else is a fatal run failure (invalid config,
        # unresolvable file) and must surface as an error — mirror run_ruff.
        # rc==1 with EMPTY stdout is also a failure (e.g. npx could not
        # resolve the eslint package and printed only to stderr): a findings
        # run always emits JSON, so rc != 0 with no stdout can never mean
        # "no lint issues".
        if proc.returncode not in (0, 1) or (proc.returncode == 1 and not proc.stdout.strip()):
            error_msg = f"eslint failed with exit code {proc.returncode}"
            if proc.stderr:
                error_msg += f": {proc.stderr[:200]}"
            return LintResult(ok=False, summary=error_msg, error=error_msg, stderr=proc.stderr)

        issues = _parse_lint_json(
            proc.stdout,
            "eslint",
            lambda raw: [
                LintIssue(
                    file=self._normalize_file_path(fe.get("filePath", path)),
                    line=msg.get("line", 0),
                    col=msg.get("column", 0),
                    code=msg.get("ruleId", "") or "",
                    message=msg.get("message", ""),
                    severity="error" if msg.get("severity", 0) >= 2 else "warning",
                )
                for fe in raw
                for msg in fe.get("messages", [])
            ],
            max_issues,
        )

        ok = len(issues) == 0
        summary = _lint_summary(issues)
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
        abs_path, err = self._resolve_path_or_error(path)
        if err is not None:
            return err

        all_issues: list[LintIssue] = []
        gofmt_available = True

        # ── Pass 1: gofmt -d (formatting check) ──────────────────────────
        proc, err = self._run_lint_command(["gofmt", "-d", str(abs_path)], "gofmt")
        if err is not None:
            # Soft failure — log and continue (gofmt missing also skips pass 2).
            gofmt_available = not err.skipped
            if not err.skipped:
                logger.warning("gofmt check failed for %s: %s", path, err.summary)
        else:
            assert proc is not None  # (proc, None) arm per _run_lint_command contract
            # gofmt -d prints diff to stdout when file is unformatted.
            # Extract line numbers from diff hunk headers (e.g. @@ -10,6 +10,8 @@)
            if proc.stdout.strip():
                fmt_line = 0
                for line in proc.stdout.splitlines():
                    m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                    if m:
                        fmt_line = int(m.group(1))
                    elif line.startswith("+") and not line.startswith("+++"):
                        all_issues.append(
                            LintIssue(
                                file=path,
                                line=fmt_line,
                                col=0,
                                code="gofmt",
                                message=f"format: {line[1:80].rstrip()}",
                                severity="error",
                            )
                        )

        # ── Pass 2: golangci-lint ─────────────────────────────────────────
        if gofmt_available:
            proc, err = self._run_lint_command(["golangci-lint", "run", str(abs_path)], "golangci-lint", timeout=60)
            if err is not None:
                if not err.skipped:
                    logger.warning("golangci-lint check failed for %s: %s", path, err.summary)
                # Tool missing: helper already logged at debug — continue.
            else:
                assert proc is not None  # (proc, None) arm per _run_lint_command contract
                if proc.returncode != 0:
                    for line in (proc.stdout + proc.stderr).splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        all_issues.append(
                            LintIssue(
                                file=path,
                                line=0,
                                col=0,
                                code="golangci-lint",
                                message=stripped[:200],
                            )
                        )

        if max_issues > 0 and len(all_issues) > max_issues:
            all_issues = all_issues[:max_issues]

        ok = len(all_issues) == 0
        summary = _lint_summary(all_issues)
        return LintResult(ok=ok, issues=all_issues, summary=summary)

    def _resolve_path_or_error(self, path: str) -> tuple[Path | None, LintResult | None]:
        """Resolve *path* for linting: ``(abs_path, None)`` or ``(None, error)``.
        Thin wrapper over :meth:`_resolve_path` that maps a ``None`` (not found /
        outside repo) to the standard invalid-path ``LintResult`` — the shared
        guard for all four lint runners.
        """
        abs_path = self._resolve_path(path)
        if abs_path is None:
            return None, LintResult(
                ok=False,
                summary=f"invalid path: {path!r}",
                error=f"Path not found or outside repo: {path!r}",
            )
        return abs_path, None

    def _run_lint_command(
        self,
        cmd: list[str],
        tool: str,
        timeout: int = 30,
    ) -> tuple[subprocess.CompletedProcess | None, LintResult | None]:
        """Run *cmd* under the shared subprocess skeleton (single source).
        Returns ``(proc, None)`` on completion, or ``(None, result)`` when the
        command could not run — tool missing (→ ``skipped``), timeout (→ error),
        or unexpected failure (→ error).  The gofmt/golangci-lint passes treat
        the ``(None, result)`` arm as non-fatal (log and continue).
        """
        try:
            return subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            ), None
        except FileNotFoundError:
            logger.debug("%s not installed; skipping lint", tool)
            return None, LintResult(ok=True, skipped=True, summary=f"{tool} not installed; lint skipped")
        except subprocess.TimeoutExpired:
            return None, LintResult(ok=False, summary=f"{tool} timed out", error=f"{tool} timed out after {timeout}s")
        except Exception as e:
            return None, LintResult(ok=False, summary=str(e), error=str(e))

    def _resolve_path(self, path: str) -> Path | None:
        """Resolve *path* within repo_root, return None if it escapes or is absent.

        Containment uses ``relative_to``, not ``str.startswith``: the latter
        accepted any sibling sharing the root's textual prefix, so
        ``run_lint("../repo-evil/secret.py")`` from ``/a/repo`` resolved to
        ``/a/repo-evil/secret.py`` and was linted. Same boundary bug (and same
        fix) as ``path_security._repo_within_allowlist`` documents.

        ``path_security.resolve_inside_repo`` is the SSOT for repo-relative
        paths but is not a drop-in here: ``run_ruff`` accepts ``""``/``"."`` to
        mean "the whole repository", which that function rejects as invalid.
        """
        try:
            repo = Path(self.repo_root).resolve()
            p = (repo / path).resolve()
            if p != repo and not p.is_relative_to(repo):
                return None
            if not p.exists():
                return None
        except Exception:
            logger.debug("path resolution failed", exc_info=True)  # non-critical — never block execution
            return None
        return p

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
            logger.debug("path normalization failed", exc_info=True)
        return file_path
