"""Extra branch coverage for test_tools.py (TestToolsMixin).

Covers the render paths and provider/lint/finder branches that the main
run_tests contract file does not reach: provider-based runner dispatch,
count-only summaries, failed/error detail rendering, empty-content fallback,
proactive test notification, lint skipped/ok, and find_tests_for_symbol.
"""
from __future__ import annotations

import sys
import threading
import types
from typing import ClassVar

import pytest

from external_llm.agent.tool_handlers.test_tools import TestToolsMixin
from external_llm.agent.tool_registry import AgentConfig
from external_llm.testing.symbol_aware_test_finder import SymbolAwareTestTarget


class _Host(TestToolsMixin):
    """Minimal host exposing only what these handlers touch."""

    def __init__(self, config=None, repo_root="."):
        self.config = config or AgentConfig()
        self.repo_root = repo_root
        self._lint_runner = _FakeLintRunner()

    def _make_result(self, **kwargs):
        fields = {
            "ok": False,
            "content": "",
            "error": None,
            "metadata": {},
            "execution_time": 0.0,
            "partial_failure": False,
            "retryable": True,
            "retry_count": 0,
        }
        fields.update(kwargs)
        return types.SimpleNamespace(**fields)


def _fake_result(**overrides):
    base = {
        "ok": True,
        "exit_code": 0,
        "duration_ms": 100,
        "timed_out": False,
        "cancelled": False,
        "failing_tests": [],
        "passed_count": 1,
        "failed_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "xpassed_count": 0,
        "xfailed_count": 0,
        "failed_test_details": [],
        "error_test_details": [],
        "summary_line": "1 passed",
        "first_traceback": None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _FakeRunner:
    calls: ClassVar[list[dict]] = []
    result = None
    python_executable = sys.executable

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_provider(cls, *args, **kwargs):
        return cls()

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def run_pytest(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeProvider:
    def __init__(self, has_runner=True):
        self._has_runner = has_runner

    def capabilities(self):
        return types.SimpleNamespace(has_test_runner=self._has_runner)


class _FakeLintRunner:
    result = None

    def run_lint(self, path, max_issues=50):
        return self.result


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr("external_llm.agent.test_runner.TestRunner", _FakeRunner)
    _FakeRunner.calls = []
    _FakeLintRunner.result = None
    return _Host(AgentConfig())


def _result_for(host, **overrides):
    _FakeRunner.result = _fake_result(**overrides)
    return host._tool_run_tests({"args": ["tests/unit/test_x.py"]})


# ---------------------------------------------------------------------------
# cancel / prefix / provider dispatch
# ---------------------------------------------------------------------------


def test_cancelled_before_execution(host):
    host.config.cancel_event = threading.Event()
    host.config.cancel_event.set()
    res = host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert res.ok is False
    assert "Operation cancelled before test execution" in res.error
    assert res.execution_time == 0.0
    assert res.retryable is False


def test_pytest_prefix_tokens_stripped(host):
    _FakeRunner.result = _fake_result()
    res = host._tool_run_tests({"args": ["python3", "-m", "pytest", "tests/unit/test_x.py"]})
    assert res.ok is True
    pytest_cmd = _FakeRunner.calls[0]["args"]
    assert pytest_cmd[0] == sys.executable
    assert pytest_cmd[1:] == ["-m", "pytest", "tests/unit/test_x.py"]


def test_string_args_split_on_whitespace(host):
    _FakeRunner.result = _fake_result()
    res = host._tool_run_tests({"args": "tests/unit/test_x.py -k foo"})
    assert res.ok is True
    pytest_cmd = _FakeRunner.calls[0]["args"]
    assert pytest_cmd[-3:] == ["tests/unit/test_x.py", "-k", "foo"]


def test_provider_based_runner_used(host, monkeypatch):
    """Non-Python file arg with a provider that has a test runner."""
    provider = _FakeProvider(has_runner=True)

    class _Registry:
        def get(self, arg):
            return provider if arg.endswith(".go") else None

    monkeypatch.setattr(
        "external_llm.languages.registry.LanguageRegistry.instance",
        staticmethod(lambda: _Registry()),
    )
    _FakeRunner.result = _fake_result()
    res = host._tool_run_tests({"args": ["main_test.go"]})
    assert res.ok is True
    assert _FakeRunner.calls and "timeout_sec" in _FakeRunner.calls[0]
    assert "cancel_check" in _FakeRunner.calls[0]


def test_provider_detected_for_extensions(host, monkeypatch):
    """_detect_provider_from_args returns provider for known exts; None else."""
    provider = _FakeProvider(has_runner=True)

    class _Registry:
        def get(self, arg):
            return provider if arg.endswith((".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt")) else None

    monkeypatch.setattr(
        "external_llm.languages.registry.LanguageRegistry.instance",
        staticmethod(lambda: _Registry()),
    )
    assert host._detect_provider_from_args(["a.ts"]) is provider
    assert host._detect_provider_from_args(["a.go"]) is provider
    assert host._detect_provider_from_args(["a.py"]) is None
    assert host._detect_provider_from_args([]) is None


def test_provider_without_runner_falls_back_to_pytest(host, monkeypatch):
    provider = _FakeProvider(has_runner=False)

    class _Registry:
        def get(self, arg):
            return provider

    monkeypatch.setattr(
        "external_llm.languages.registry.LanguageRegistry.instance",
        staticmethod(lambda: _Registry()),
    )
    _FakeRunner.result = _fake_result()
    res = host._tool_run_tests({"args": ["main.go"]})
    assert res.ok is True
    assert _FakeRunner.calls[0]["args"][0] == sys.executable  # pytest path


# ---------------------------------------------------------------------------
# summary rendering branches
# ---------------------------------------------------------------------------


def test_summary_without_line_renders_counts(host):
    res = _result_for(
        host,
        summary_line=None,
        passed_count=3,
        failed_count=1,
        error_count=2,
        skipped_count=4,
        xpassed_count=5,
        xfailed_count=6,
    )
    assert "## Test Summary" in res.content
    assert "Passed: 3" in res.content
    assert "Failed: 1" in res.content
    assert "Errors: 2" in res.content
    assert "Skipped: 4" in res.content
    assert "XPassed: 5" in res.content
    assert "XFailed: 6" in res.content


def test_summary_without_line_zero_counts(host):
    # total_tests == 0 and no summary_line -> nothing to render -> fallback
    res = _result_for(host, summary_line=None, passed_count=0, failed_count=0)
    assert "Tests passed" in res.content


def test_failed_test_details_rendered(host):
    res = _result_for(
        host,
        ok=False,
        failed_test_details=[
            {
                "name": "test_boom",
                "error_type": "AssertionError",
                "message": "x != y",
                "file": "tests/test_x.py",
                "line": 42,
                "traceback": "line1\nline2\nline3\nline4\nline5\nline6",
            }
        ],
    )
    assert "## Failed Tests" in res.content
    assert "### test_boom" in res.content
    assert "AssertionError: x != y" in res.content
    assert "File: tests/test_x.py, Line: 42" in res.content
    assert "Traceback (first lines):" in res.content
    assert "  line1" in res.content
    assert "  line5" in res.content
    assert "line6" not in res.content  # capped at 5


def test_error_test_details_rendered(host):
    res = _result_for(
        host,
        ok=False,
        error_test_details=[
            {
                "name": "test_err",
                "error_type": "TypeError",
                "file": "tests/test_y.py",
                "line": 0,
                "traceback": "tb1\ntb2",
            }
        ],
    )
    assert "## Error Tests" in res.content
    assert "### test_err" in res.content
    assert "TypeError" in res.content
    assert "File: tests/test_y.py" in res.content


def test_empty_content_fallback_summary_line(host):
    res = _result_for(host, summary_line="5 passed", passed_count=0, failed_count=0)
    assert res.content == "## Test Summary — 5 passed"


def test_empty_content_fallback_failing_tests_and_traceback(host):
    res = _result_for(
        host,
        summary_line=None,
        passed_count=0,
        failed_count=0,
        failing_tests=["tests/test_a.py::t1", "tests/test_b.py::t2"],
        first_traceback="traceback line",
    )
    assert "Failing tests:" in res.content
    assert "  - tests/test_a.py::t1" in res.content
    assert "First traceback:" in res.content


def test_empty_content_fallback_tests_passed(host):
    res = _result_for(host, summary_line=None, passed_count=0, failed_count=0)
    assert "Tests passed" in res.content


def test_notify_proactive_runner(host, monkeypatch):
    """A registered proactive runner for the repo receives test results."""
    notified = {}

    class _Runner:
        def notify_test_result(self, ok, data):
            notified["ok"] = ok
            notified["data"] = data

    import threading

    import external_llm.editor.agent.autonomous.proactive_runner as pr

    monkeypatch.setattr(pr, "_runners", {"repo": _Runner()})
    # with-statement looks up __enter__ on the TYPE, not the instance, so a
    # SimpleNamespace stand-in would raise TypeError inside the suppress and
    # silently skip the notification. Use a real lock.
    monkeypatch.setattr(pr, "_runners_lock", threading.Lock())
    host.repo_root = "repo"
    _result_for(host, ok=False, failed_count=2)
    assert notified.get("ok") is False
    assert notified["data"]["failed_count"] == 2


# ---------------------------------------------------------------------------
# _tool_run_lint
# ---------------------------------------------------------------------------


def _lint_result(**overrides):
    from external_llm.agent.lint_runner import LintResult

    base = {
        "ok": True,
        "issues": [],
        "summary": "no lint issues",
        "skipped": False,
        "error": None,
    }
    base.update(overrides)
    return LintResult(**base)


def test_lint_skipped(host):
    _FakeLintRunner.result = _lint_result(skipped=True, summary="skipped: no tool")
    res = host._tool_run_lint({"path": "."})
    assert res.ok is True
    assert res.metadata["skipped"] is True


def test_lint_ok(host):
    _FakeLintRunner.result = _lint_result(summary="no lint issues")
    res = host._tool_run_lint({"path": "."})
    assert res.ok is True
    assert res.content == "no lint issues"


def test_lint_issues_with_fix_hints(host):
    from external_llm.agent.lint_runner import LintIssue

    _FakeLintRunner.result = _lint_result(
        ok=False,
        issues=[
            LintIssue(file="a.py", line=1, col=2, code="E001", message="bad", fix="a = 1"),
            LintIssue(file="b.py", line=3, col=4, code="E002", message="worse", fix=None),
        ],
        summary="2 lint issue(s) found",
    )
    res = host._tool_run_lint({"path": "."})
    assert res.ok is True  # issue detection is not a real error
    assert res.metadata["issue_count"] == 2
    assert res.metadata["fixable_count"] == 1
    assert "a.py:1:2 [E001] bad  [fix: a = 1]" in res.content
    assert "b.py:3:4 [E002] worse" in res.content


def test_lint_real_error_not_ok(host):
    from external_llm.agent.lint_runner import LintResult

    _FakeLintRunner.result = LintResult(
        ok=False, issues=[], summary="lint failed", error="timeout"
    )
    res = host._tool_run_lint({"path": "."})
    assert res.ok is False  # real error (no issues, error set)


# ---------------------------------------------------------------------------
# _tool_find_tests_for_symbol
# ---------------------------------------------------------------------------


def test_find_tests_requires_symbol_or_file(host):
    res = host._tool_find_tests_for_symbol({})
    assert res.ok is False
    assert "needs `symbol`" in res.error
    assert res.retryable is True


def test_find_tests_no_targets(host, monkeypatch):
    class _Finder:
        def __init__(self, root):
            self.root = root

        def discover_test_targets(self, target_symbols=None, target_files=None):
            return []

    monkeypatch.setattr(
        "external_llm.agent.tool_handlers.test_tools.SymbolAwareTestFinder", _Finder
    )
    res = host._tool_find_tests_for_symbol({"symbol": "foo"})
    assert res.ok is True
    assert "No test file references 'foo'" in res.content
    assert res.metadata["match_count"] == 0


def test_find_tests_with_targets(host, monkeypatch):
    targets = [
        SymbolAwareTestTarget(
            test_path="tests/test_foo.py",
            match_type="direct_symbol",
            matched_symbols=["foo"],
            scope_level_hint="narrow",
        ),
        SymbolAwareTestTarget(
            test_path="tests/test_module.py",
            match_type="module_import",
            scope_level_hint="standard",
        ),
    ]

    class _Finder:
        def __init__(self, root):
            self.root = root

        def discover_test_targets(self, target_symbols=None, target_files=None):
            return targets

    monkeypatch.setattr(
        "external_llm.agent.tool_handlers.test_tools.SymbolAwareTestFinder", _Finder
    )
    res = host._tool_find_tests_for_symbol({"symbol": "foo"})
    assert res.ok is True
    assert "2 test file(s), strongest match first:" in res.content
    assert "tests/test_foo.py  [direct_symbol (foo)] [scope=narrow]" in res.content
    assert "tests/test_module.py  [module_import] [scope=standard]" in res.content
    assert res.metadata["match_count"] == 2
    assert res.metadata["top_match_type"] == "direct_symbol"


def test_find_tests_uses_file_path_fallback(host, monkeypatch):
    seen = {}

    class _Finder:
        def __init__(self, root):
            self.root = root

        def discover_test_targets(self, target_symbols=None, target_files=None):
            seen["symbols"] = target_symbols
            seen["files"] = target_files
            return [SymbolAwareTestTarget(test_path="tests/test_x.py")]

    monkeypatch.setattr(
        "external_llm.agent.tool_handlers.test_tools.SymbolAwareTestFinder", _Finder
    )
    res = host._tool_find_tests_for_symbol({"file_path": "src/x.py"})
    assert res.ok is True
    assert seen["symbols"] is None
    assert seen["files"] == ["src/x.py"]


def test_error_test_details_with_message_combined(host):
    """error_type + message -> 'type: message' combined line."""
    res = _result_for(
        host,
        ok=False,
        error_test_details=[
            {
                "name": "test_e",
                "error_type": "ValueError",
                "message": "bad value",
                "file": "tests/test_z.py",
                "line": 7,
                "traceback": "",
            }
        ],
    )
    assert "ValueError: bad value" in res.content
    assert "File: tests/test_z.py, Line: 7" in res.content


def test_empty_content_fallback_plain_summary_line(host):
    """content_parts empty + summary_line set -> bare summary line appended.

    Reaches the L186 branch: total_tests == 0 while summary_line is truthy
    (e.g. a run that printed a summary but reported no counted tests).
    """
    _FakeRunner.result = _fake_result(
        ok=True,
        summary_line="no tests collected",
        passed_count=0,
        failed_count=0,
        error_count=0,
        skipped_count=0,
        xpassed_count=0,
        xfailed_count=0,
        failing_tests=[],
        first_traceback=None,
    )
    res = host._tool_run_tests({"args": ["tests/unit/test_x.py"]})
    assert "no tests collected" in res.content
