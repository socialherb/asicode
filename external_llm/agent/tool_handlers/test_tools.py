"""Test and lint tool handlers for ToolRegistry."""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any

from external_llm.languages.registry import LanguageRegistry
from external_llm.testing.symbol_aware_test_finder import SymbolAwareTestFinder

if TYPE_CHECKING:
    from ..tool_registry import ToolResult
class TestToolsMixin:
    """Mixin providing test and lint tool implementations for ToolRegistry."""

    @staticmethod
    def _detect_provider_from_args(extra_args):
        """Detect test language provider from file args."""
        registry = LanguageRegistry.instance()
        for arg in extra_args:
            ext = os.path.splitext(arg)[1].lower()
            if ext in ('.ts', '.tsx', '.js', '.jsx', '.go', '.java', '.kt'):
                provider = registry.get(arg)
                if provider and provider.capabilities().has_test_runner:
                    return provider
        return None

    def _tool_run_tests(self, args: dict[str, Any]) -> "ToolResult":
        if self.config.cancel_event and self.config.cancel_event.is_set():
            return self._make_result(
                ok=False,
                content="",
                error="Operation cancelled before test execution",
                execution_time=0.0,
                retryable=False,
            )

        extra_args = args.get("args") or []
        # Normalise: LLM sometimes passes the full command in args
        # (e.g. ["python3", "-m", "pytest", "tests/..."]) — strip the prefix
        # so the final command doesn't double up as "python3 -m pytest python3 -m pytest ...".
        if isinstance(extra_args, str):
            # String instead of list — split on whitespace as best-effort recovery
            extra_args = extra_args.split()
        else:
            extra_args = list(extra_args)
        # Strip leading "python3 -m pytest" / "pytest" / "python -m pytest" tokens
        _PYTEST_PREFIX_TOKENS = {"python3", "python", "-m", "pytest"}
        while extra_args and extra_args[0] in _PYTEST_PREFIX_TOKENS:
            extra_args.pop(0)

        try:
            from ..test_runner import TestRunner
        except ImportError:
            return self._make_result(ok=False, content="", error="TestRunner not available")

        # Try provider-based test runner for non-Python languages
        provider = self._detect_provider_from_args(extra_args)
        if provider and provider.capabilities().has_test_runner:
            runner = TestRunner.from_provider(self.repo_root, provider, test_args=extra_args)
            result = runner.run(timeout_sec=120)
        else:
            runner = TestRunner(self.repo_root)
            pytest_cmd = [runner.python_executable, "-m", "pytest", *extra_args]
            result = runner.run_pytest(args=pytest_cmd, timeout_sec=120)

        metadata = {
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "failing_tests": result.failing_tests or [],
            "passed": result.passed_count,
            "failed": result.failed_count,
            "errors": result.error_count,
            "skipped": result.skipped_count,
            "xpassed": result.xpassed_count,
            "xfailed": result.xfailed_count,
            "failed_test_details": result.failed_test_details or [],
            "error_test_details": result.error_test_details or [],
        }

        content_parts = []

        total_tests = result.passed_count + result.failed_count + result.error_count + \
                     result.skipped_count + result.xpassed_count + result.xfailed_count
        if total_tests > 0 or result.summary_line:
            if result.summary_line:
                content_parts.append(f"## Test Summary — {result.summary_line}")
            else:
                summary_parts = []
                if result.passed_count > 0:
                    summary_parts.append(f"Passed: {result.passed_count}")
                if result.failed_count > 0:
                    summary_parts.append(f"Failed: {result.failed_count}")
                if result.error_count > 0:
                    summary_parts.append(f"Errors: {result.error_count}")
                if result.skipped_count > 0:
                    summary_parts.append(f"Skipped: {result.skipped_count}")
                if result.xpassed_count > 0:
                    summary_parts.append(f"XPassed: {result.xpassed_count}")
                if result.xfailed_count > 0:
                    summary_parts.append(f"XFailed: {result.xfailed_count}")
                counts_str = ", ".join(summary_parts) if summary_parts else ""
                content_parts.append(f"## Test Summary ({counts_str})" if counts_str else "## Test Summary")

        if result.failed_test_details:
            content_parts.append("\n## Failed Tests")
            for test in result.failed_test_details[:10]:
                content_parts.append(f"### {test.get('name', 'Unknown test')}")
                error_type = test.get('error_type', '')
                message = test.get('message', '')
                if error_type or message:
                    error_desc = error_type
                    if message:
                        error_desc += f": {message}"
                    content_parts.append(error_desc)

                file_path = test.get('file', '')
                line_num = test.get('line', 0)
                if file_path:
                    line_info = f"Line: {line_num}" if line_num > 0 else ""
                    content_parts.append(f"File: {file_path}" + (f", {line_info}" if line_info else ""))

                traceback = test.get('traceback', '')
                if traceback:
                    traceback_lines = traceback.split('\n')[:5]
                    content_parts.append("Traceback (first lines):")
                    for line in traceback_lines:
                        content_parts.append(f"  {line}")
                content_parts.append("")

        if result.error_test_details:
            content_parts.append("\n## Error Tests")
            for test in result.error_test_details[:10]:
                content_parts.append(f"### {test.get('name', 'Unknown test')}")
                error_type = test.get('error_type', '')
                message = test.get('message', '')
                if error_type or message:
                    error_desc = error_type
                    if message:
                        error_desc += f": {message}"
                    content_parts.append(error_desc)

                file_path = test.get('file', '')
                line_num = test.get('line', 0)
                if file_path:
                    line_info = f"Line: {line_num}" if line_num > 0 else ""
                    content_parts.append(f"File: {file_path}" + (f", {line_info}" if line_info else ""))

                traceback = test.get('traceback', '')
                if traceback:
                    traceback_lines = traceback.split('\n')[:5]
                    content_parts.append("Traceback (first lines):")
                    for line in traceback_lines:
                        content_parts.append(f"  {line}")
                content_parts.append("")

        if not content_parts:
            if result.summary_line:
                content_parts.append(result.summary_line)
            if result.failing_tests:
                content_parts.append("Failing tests:")
                content_parts.extend(f"  - {t}" for t in result.failing_tests[:20])
            if result.first_traceback:
                tb = result.first_traceback[:1000]
                content_parts.append(f"\nFirst traceback:\n{tb}")
            if not content_parts:
                content_parts.append("Tests passed" if result.ok else "Tests failed (no details)")

        content = "\n".join(content_parts)

        # ── Proactive test result notification ───────────────────────────────
        # Forward pass/fail to TriggerEngine so the proactive system can push
        # analysis or recovery suggestions to the browser.
        try:
            from external_llm.editor.agent.autonomous.proactive_runner import _runners, _runners_lock
            _repo = getattr(self, "repo_root", None)
            if _repo:
                with _runners_lock:
                    _runner = _runners.get(_repo)
                if _runner:
                    _runner.notify_test_result(result.ok, {
                        "ok": result.ok,
                        "summary_line": result.summary_line or "",
                        "failing_tests": result.failing_tests or [],
                        "first_traceback": result.first_traceback or "",
                        "failed_count": result.failed_count,
                        "passed_count": result.passed_count,
                    })
        except Exception:
            pass  # Never break test tool behavior

        return self._make_result(
            ok=result.ok,
            content=content,
            metadata=metadata,
        )

    def _tool_run_lint(self, args: dict[str, Any]) -> "ToolResult":
        path = args.get("path", ".")
        result = self._lint_runner.run_lint(path, max_issues=self.config.max_lint_issues)

        if result.skipped:
            return self._make_result(ok=True, content=result.summary, metadata={"skipped": True})

        if not result.ok:
            parts = [result.summary]
            for issue in result.issues[:20]:
                parts.append(f"  {issue.file}:{issue.line}:{issue.col} [{issue.code}] {issue.message}")
            # Lint finding issues is a successful tool execution (should show ✓ not ✗)
            # If result.error is None, it's issue detection, not a real error (timeout/crash)
            is_real_error = result.error is not None and not result.issues
            return self._make_result(
                ok=not is_real_error,
                content="\n".join(parts),
                metadata={"issue_count": len(result.issues)},
            )

        return self._make_result(ok=True, content=result.summary)

    def _tool_find_tests_for_symbol(self, args: dict[str, Any]) -> "ToolResult":
        """Find test files related to a symbol or file path. Supports Python (pytest), TS/JS (jest/vitest), and Go."""
        symbol = args.get("symbol")
        file_path = args.get("file_path")
        finder = SymbolAwareTestFinder(self.repo_root)
        test_files = finder.find_tests_for_symbol(symbol=symbol, file_path=file_path)
        if test_files:
            content = f"Found {len(test_files)} related test file(s):\n" + "\n".join(test_files)
        else:
            content = "No related tests found."
        return self._make_result(ok=True, content=content)
