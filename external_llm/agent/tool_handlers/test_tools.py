"""Test and lint tool handlers for ToolRegistry."""
from __future__ import annotations

import os
from contextlib import suppress
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
        # String instead of list — split on whitespace as best-effort recovery
        extra_args = extra_args.split() if isinstance(extra_args, str) else list(extra_args)
        # Strip leading "python3 -m pytest" / "pytest" / "python -m pytest" tokens
        _PYTEST_PREFIX_TOKENS = {"python3", "python", "-m", "pytest"}
        while extra_args and extra_args[0] in _PYTEST_PREFIX_TOKENS:
            extra_args.pop(0)

        # test_runner is first-party — import cannot fail.
        from ..test_runner import TestRunner

        # Timeout budget comes from config (default 300 s, not the 120 s this
        # hardcoded before): with an empty test_paths the TDD gate runs the
        # FULL suite, and 120 s was smaller than it — the gate timed out on
        # every green run and surfaced as a failure.
        _timeout = getattr(self.config, "test_timeout_sec", 300)

        # Live cancel check, not a dispatch-entry snapshot: the design-chat
        # REPL swaps config.cancel_event per turn, so reading it inside the
        # runner's poll loop is what lets a cancel reach a run that is ALREADY
        # executing (bash/grep do the same).
        def _cancel_requested() -> bool:
            _ev = getattr(self.config, "cancel_event", None)
            return _ev is not None and _ev.is_set()

        # Try provider-based test runner for non-Python languages
        provider = self._detect_provider_from_args(extra_args)
        if provider and provider.capabilities().has_test_runner:
            runner = TestRunner.from_provider(self.repo_root, provider, test_args=extra_args)
            result = runner.run(timeout_sec=_timeout, cancel_check=_cancel_requested)
        else:
            runner = TestRunner(self.repo_root)
            pytest_cmd = [runner.python_executable, "-m", "pytest", *extra_args]
            result = runner.run_pytest(args=pytest_cmd, timeout_sec=_timeout, cancel_check=_cancel_requested)

        metadata = {
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "cancelled": result.cancelled,
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

        # First line, before any counts: a killed run's counts are whatever it
        # happened to print before the kill, and read as a finished run without
        # this. The TDD cycle turns ok=False into "[TDD] Tests failed", so a
        # timeout otherwise arrives as a real failure with no way to tell.
        if result.cancelled:
            content_parts.append(
                "## 🛑 Test run CANCELLED — killed by cancel request.\n"
                "Counts below are partial and NOT a test failure."
            )
        elif result.timed_out:
            content_parts.append(
                f"## ⏱ Test run TIMED OUT after {result.duration_ms // 1000}s "
                "— killed before it finished.\n"
                "Counts below are partial and NOT a test failure. Re-run a "
                "narrower scope (specific test paths / -k) rather than debugging "
                "these as failures."
            )

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
                    content_parts.extend(f"  {line}" for line in traceback_lines)
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
                    content_parts.extend(f"  {line}" for line in traceback_lines)
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
        with suppress(OSError, RuntimeError, ValueError, TypeError):  # Never break test tool behavior
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

        if result.cancelled:
            return self._make_result(
                ok=False,
                content=content,
                error="Operation cancelled during test execution",
                retryable=False,
                metadata={**metadata, "cancelled": True},
            )

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
                line = f"  {issue.file}:{issue.line}:{issue.col} [{issue.code}] {issue.message}"
                if issue.fix:
                    # Surface ruff's auto-fix hint inline so the model can act on
                    # it without re-running the linter. Newlines are collapsed to
                    # keep the per-issue single-line format intact.
                    fix_hint = " ".join(issue.fix.split())
                    line += f"  [fix: {fix_hint}]"
                parts.append(line)
            # Lint finding issues is a successful tool execution (should show ✓ not ✗)
            # If result.error is None, it's issue detection, not a real error (timeout/crash)
            is_real_error = result.error is not None and not result.issues
            metadata: dict[str, Any] = {"issue_count": len(result.issues)}
            if result.fixable_count:
                metadata["fixable_count"] = result.fixable_count
                metadata["fixable_issues"] = [
                    {"file": i.file, "line": i.line, "col": i.col,
                     "code": i.code, "fix": i.fix}
                    for i in result.issues if i.fix
                ][:20]
            return self._make_result(
                ok=not is_real_error,
                content="\n".join(parts),
                metadata=metadata,
            )

        return self._make_result(ok=True, content=result.summary)

    def _tool_find_tests_for_symbol(self, args: dict[str, Any]) -> "ToolResult":
        """Find test files covering a symbol or file. Python, TS/JS, and Go.

        Reports each hit with WHY it matched, not just its path. The finder
        ranks by match type — a test that references the symbol by name
        (``direct_symbol``) is evidence of a different order than one that
        merely imports the module (``module_import``) or happens to sit in the
        matching directory (``same_module``) — and every candidate below the
        top tier is a guess the model should be able to see as one. Collapsing
        that to a bare path list, as this used to, made a 0.2-score filename
        guess indistinguishable from a direct hit.
        """
        symbol = args.get("symbol") or args.get("name")
        file_path = args.get("file_path") or args.get("path")
        if not symbol and not file_path:
            return self._make_result(
                ok=False,
                content="",
                error=(
                    "find_tests_for_symbol needs `symbol` (a function/class name) "
                    "or `file_path` (a file whose tests you want). "
                    "Both empty matches nothing."
                ),
                retryable=True,
            )
        finder = SymbolAwareTestFinder(self.repo_root)
        targets = finder.discover_test_targets(
            target_symbols=[symbol] if symbol else None,
            target_files=[file_path] if file_path else None,
        )
        if not targets:
            _asked = symbol or file_path
            return self._make_result(
                ok=True,
                content=(
                    f"No test file references {_asked!r}. "
                    "This means no match was found, NOT that the symbol is "
                    "untested by some other name — check a caller, or the "
                    "module's own test file, before concluding it has no cover."
                ),
                metadata={"match_count": 0},
            )
        lines = [f"{len(targets)} test file(s), strongest match first:"]
        for t in targets:
            _why = t.match_type
            if t.matched_symbols:
                _why += f" ({', '.join(t.matched_symbols[:3])})"
            lines.append(f"  {t.test_path}  [{_why}] [scope={t.scope_level_hint}]")
        return self._make_result(
            ok=True,
            content="\n".join(lines),
            metadata={
                "match_count": len(targets),
                "top_match_type": targets[0].match_type,
            },
        )
